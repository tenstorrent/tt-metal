# SPDX-License-Identifier: Apache-2.0
"""Exact recurrent-prefix snapshots in vLLM-owned physical block coordinates.

Each live request has independent conv/recurrent buffers. The scheduler owns pool
block identities and lifetime; a saved row is immutable until that block is reused.
No saved row aliases live request state, and restore never mutates the pool.
"""
import math
import operator
from pathlib import Path

import torch
import ttnn
from .optimized_decoder import LinearState

_KERNEL = str(Path(__file__).with_name('kernels') / 'serving_state_copy.cpp')


class ServingStatePool:
    def __init__(self, model, active_states, num_blocks_by_layer):
        self.mesh = model.mesh
        self.active = {}
        self.pools = {}
        self.num_blocks = {}
        self.prepared = False
        self.counters = dict(saves=0, restores=0, metadata_uploads=0)
        if len(active_states) != len(model.layer_indices):
            raise ValueError('State list must follow selected model layer order')
        for original_idx, layer, active in zip(model.layer_indices, model.layers, active_states):
            if layer.layer_type != 'linear_attention':
                continue
            count = operator.index(num_blocks_by_layer[original_idx])
            if count < 1:
                raise ValueError('Every linear layer needs a positive scheduler block count')
            if tuple(active.conv.shape)[1:] != (3,2560) or active.conv.dtype != ttnn.bfloat16 or active.conv.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError('Expected BF16 row-major convolution state [B,3,2560]')
            if tuple(active.recurrent.shape)[1:] != (12,128,128) or active.recurrent.dtype != ttnn.float32 or active.recurrent.layout != ttnn.TILE_LAYOUT:
                raise ValueError('Expected FP32 tiled recurrence [B,12,128,128]')
            if active.conv.shape[0] != active.recurrent.shape[0]:
                raise ValueError('Convolution and recurrent slot counts differ')
            self.active[original_idx] = active
            self.num_blocks[original_idx] = count
            def pool_tensor(source):
                return ttnn.zeros([count,*tuple(source.shape)[1:]],dtype=source.dtype,
                    layout=source.layout,device=self.mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.pools[original_idx] = LinearState(pool_tensor(active.conv),pool_tensor(active.recurrent))
        if not self.active:
            raise ValueError('A recurrent state pool needs at least one linear layer')
        self.layer_indices = tuple(self.active)
        self.batch_size = int(next(iter(self.active.values())).conv.shape[0])
        self.host_ids = torch.zeros(len(self.active),8,dtype=torch.int32)
        self.ids = ttnn.from_torch(self.host_ids,dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        self.allocated_bytes_per_device = sum(
            self.num_blocks[i] * (3*2560*2 + 12*128*128*4) for i in self.layer_indices)

    def _copy_tensor(self, source, destination, layer_row):
        if source.buffer_address() == destination.buffer_address():
            raise ValueError('Snapshot copies require distinct live and pool allocations')
        if source.is_sharded() or destination.is_sharded():
            raise ValueError('Snapshot storage must be interleaved DRAM')
        if source.layout == ttnn.ROW_MAJOR_LAYOUT:
            page_bytes = int(source.shape[-1])*2
            pages_per_row = math.prod(tuple(source.shape)[1:-1])
        else:
            page_bytes = 4096  # exact FP32 tile, including every mantissa bit
            pages_per_row = math.prod(tuple(source.padded_shape)[1:])//1024
        count = min(8,pages_per_row)
        grid = self.mesh.compute_with_storage_grid_size()
        if count > grid.x*grid.y:
            raise ValueError('Insufficient worker cores for state copy')
        coords = [ttnn.CoreCoord(i//grid.y,i%grid.y) for i in range(count)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core,core) for core in coords])
        compile_args = [page_bytes,pages_per_row,count,int(source.shape[0]),int(destination.shape[0])]
        for tensor in (source,destination,self.ids):
            compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        runtime = ttnn.RuntimeArgs()
        for ordinal,core in enumerate(coords):
            runtime[core.x][core.y] = [source.buffer_address(),destination.buffer_address(),
                self.ids.buffer_address(),layer_row,ordinal]
        descriptor = ttnn.ProgramDescriptor(kernels=[ttnn.KernelDescriptor(
            kernel_source=_KERNEL,core_ranges=cores,compile_time_args=compile_args,
            runtime_args=runtime,config=ttnn.ReaderConfigDescriptor())],cbs=[
                ttnn.CBDescriptor(total_size=page_bytes,core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0,data_format=source.dtype,page_size=page_bytes)]),
                ttnn.CBDescriptor(total_size=32,core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1,data_format=ttnn.uint32,page_size=32)])
            ],semaphores=[])
        ttnn.generic_op([source,destination,self.ids],descriptor)

    def _transfer(self, slot, block_ids, restore):
        slot = operator.index(slot)
        if not 0 <= slot < self.batch_size:
            raise ValueError('Live slot is outside the configured batch')
        if set(block_ids) != set(self.layer_indices):
            raise ValueError('Supply the scheduler block ID for every selected linear layer')
        ids = {i:operator.index(block_ids[i]) for i in self.layer_indices}
        if any(not 0 <= block < self.num_blocks[i] for i,block in ids.items()):
            raise ValueError('Snapshot block ID is outside the scheduler pool')
        for row,i in enumerate(self.layer_indices):
            self.host_ids[row,0] = ids[i] if restore else slot
            self.host_ids[row,1] = slot if restore else ids[i]
        host = ttnn.from_torch(self.host_ids,dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        ttnn.copy_host_to_device_tensor(host,self.ids)
        self.counters['metadata_uploads']+=1
        for row,i in enumerate(self.layer_indices):
            source,destination=(self.pools[i],self.active[i]) if restore else (self.active[i],self.pools[i])
            self._copy_tensor(source.conv,destination.conv,row)
            self._copy_tensor(source.recurrent,destination.recurrent,row)

    def warmup(self):
        """Compile both directions before capture, preserving the live slot exactly.

        Newly allocated pool row0 becomes a valid copy of live slot0. The scheduler
        has not assigned request ownership yet, so this is only warmup storage.
        """
        if self.prepared:
            return
        blocks={i:0 for i in self.layer_indices}
        self._transfer(0,blocks,False)
        self._transfer(0,blocks,True)
        self.prepared=True

    def save(self, slot, block_ids):
        if not self.prepared:
            raise RuntimeError('Prepare snapshot copies before model trace capture')
        self._transfer(slot,block_ids,False)
        self.counters['saves']+=1

    def restore(self, slot, block_ids):
        if not self.prepared:
            raise RuntimeError('Prepare snapshot copies before model trace capture')
        self._transfer(slot,block_ids,True)
        self.counters['restores']+=1
