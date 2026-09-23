# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental device layer loop; bind all KV addresses before trace capture."""

from pathlib import Path
import ttnn
from .mlp import _grid


class DecoderLoop:
    def __init__(self, body, caches, *, first_layer=0, num_layers=None, embedding_weight=None, head=None):
        if body.preparation is None:
            raise ValueError("Layer loop requires the complete decoder body")
        self.body = body
        self.head = head
        self.embedding_weight = embedding_weight
        if embedding_weight is not None and (
            embedding_weight.dtype != ttnn.bfloat16
            or embedding_weight.layout != ttnn.ROW_MAJOR_LAYOUT
            or embedding_weight.shape[-1] != 1024
            or first_layer != 0
        ):
            raise ValueError("Embedding fusion requires the original TP-local BF16 row-major weight")
        self.caches = tuple(caches)
        self.first = first_layer
        self.count = len(caches) - first_layer if num_layers is None else num_layers
        if len(caches) != len(body.layers) or not 0 <= first_layer < first_layer + self.count <= len(body.layers):
            raise ValueError("Provide each layer's cache and a valid layer interval")
        reference = caches[0]
        for pair in caches:
            if len(pair) != 2 or any(
                (tuple(t.shape), t.dtype, t.layout, t.memory_config())
                != (tuple(r.shape), r.dtype, r.layout, r.memory_config())
                for t, r in zip(pair, reference)
            ):
                raise ValueError("All layer caches must share the same specification")
        addresses = body.address_rows.clone()
        for index, pair in enumerate(caches):
            addresses[index, 4] = pair[0].buffer_address()
            addresses[index, 5] = pair[1].buffer_address()
        # This constructor must precede every warmup/capture of the body.
        body.address_table = ttnn.from_torch(
            addresses,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=body.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(body.mesh),
        )
        body.address_rows = addresses
        self.cores = (
            body.projection_cores
            + body.sfpu_cores
            + body.communication_cores
            + body.norm_cores
            + body.attention_stage.cores
            + body.preparation.cores
        )
        if len({(core.x, core.y) for core in self.cores}) != len(self.cores):
            raise ValueError("Loop workers must be disjoint")
        self.grid = _grid(self.cores)
        self.state = ttnn.zeros(
            (1, 1, len(self.cores) * 32, 32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=body.mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        self.barriers = [
            ttnn.create_global_semaphore(body.mesh, self.grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(3)
        ]
        self.barrier_addresses = [ttnn.get_global_semaphore_address(s) for s in self.barriers]
        self.head_ready = (
            ttnn.create_global_semaphore(body.mesh, body.reduction.grid, 0, ttnn.BufferType.L1_SMALL)
            if head is not None
            else None
        )
        if head is not None and any(c in self.cores for c in (*head.cores, *head.norm.cores)):
            raise ValueError("Terminal head and norm workers must be disjoint from the layer loop")
        self.head_prefetch = None
        if body.tuning.prefetch_head_workers:
            if head is None:
                raise ValueError("Head-worker staging requires the terminal head in the resident program")
            from .head_prefetch import HeadPrefetch
            self.head_prefetch = HeadPrefetch(self)
        ttnn.synchronize_device(body.mesh)

    def append(self, program, tokens=None):
        if self.body.tuning.prefetch_gu_blocks or self.body.tuning.prefetch_down_blocks:
            from .prefetch import append_prefetch
            program = append_prefetch(self.body, program)
        if self.head_prefetch is not None:
            program = self.head_prefetch.append_consumers(program)
        if self.embedding_weight is not None:
            from math import prod

            if (
                tokens is None
                or tokens.dtype != ttnn.uint32
                or tokens.layout != ttnn.ROW_MAJOR_LAYOUT
                or tokens.is_sharded()
                or prod(tuple(tokens.shape)) != 32
            ):
                raise ValueError("Embedding fusion requires the native interleaved row-major uint32 token row")
            kernels = list(program.kernels)
            for kernel in kernels:
                if Path(kernel.kernel_source).name != "reduce_reader.cpp":
                    continue
                rt = kernel.runtime_args
                offsets = set()
                for c in self.body.communication_cores:
                    args = list(rt[c.x][c.y])
                    offsets.add(len(args))
                    rt[c.x][c.y] = [*args, self.embedding_weight.buffer_address(), tokens.buffer_address(), 1]
                if len(offsets) != 1:
                    raise ValueError("Unexpected communication-reader runtime layout")
                kernel.runtime_args = rt
                kernel.defines = [
                    *kernel.defines,
                    ("FUSE_EMBEDDING", "1"),
                    ("EMBED_RT_OFFSET", str(offsets.pop())),
                    ("EMBED_CT_OFFSET", str(len(kernel.compile_time_args))),
                ]
                kernel.compile_time_args = [
                    *kernel.compile_time_args,
                    *ttnn.TensorAccessorArgs(self.embedding_weight).get_compile_time_args(),
                    *ttnn.TensorAccessorArgs(tokens).get_compile_time_args(),
                ]
            program.kernels = kernels
            program.cbs = [
                *program.cbs,
                ttnn.CBDescriptor(
                    total_size=4096,
                    core_ranges=_grid(self.body.communication_cores),
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=4096)
                    ],
                ),
            ]
        if self.head is not None:
            kernels = list(program.kernels)
            coordinator = self.body.mesh.worker_core_from_logical_core(self.body.communication_cores[0])
            physical = [self.body.mesh.worker_core_from_logical_core(c) for c in self.head.norm.cores]
            for kernel in kernels:
                if Path(kernel.kernel_source).name != "reduce_writer.cpp":
                    continue
                rt = kernel.runtime_args
                offsets = set()
                for c in self.body.communication_cores:
                    args = list(rt[c.x][c.y])
                    offsets.add(len(args))
                    rt[c.x][c.y] = [
                        *args,
                        self.body.gather_output.buffer_address(),
                        ttnn.get_global_semaphore_address(self.head_ready),
                        coordinator.x,
                        coordinator.y,
                        *[v for xy in physical for v in (xy.x, xy.y)],
                        0,
                        self.count,
                    ]
                if len(offsets) != 1:
                    raise ValueError("Unexpected terminal gather runtime layout")
                kernel.runtime_args = rt
                kernel.defines = [
                    *kernel.defines,
                    ("TAIL_RT_OFFSET", str(offsets.pop())),
                    ("FUSED_SUFFIX_HEADER", '"models/demos/llama31_8b_qb2/tt/megakernel/kernels/all_gather_tail.hpp"'),
                ]
            program.kernels = kernels
        masks = {(c.x, c.y): [0, 0, 0] for c in self.cores}
        for cb in program.cbs:
            for c in ttnn.corerange_to_cores(cb.core_ranges, row_wise=True):
                for fmt in cb.format_descriptors:
                    index = fmt.buffer_index
                    masks[c.x, c.y][index // 32] |= 1 << (index % 32)
        for sem in program.semaphores:
            for c in ttnn.corerange_to_cores(sem.core_ranges, row_wise=True):
                masks[c.x, c.y][2] |= 1 << sem.id
        physical = [self.body.mesh.worker_core_from_logical_core(c) for c in self.cores]
        coordinates = [v for c in physical for v in (c.x, c.y)]
        indices = {(c.x, c.y): i for i, c in enumerate(self.cores)}
        kernels = list(program.kernels)
        for kernel in kernels:
            original = kernel.kernel_source
            name = Path(original).name
            patch = {
                "mlp.cpp": 1,
                "qkv.cpp": 2,
                "cache_dataflow.cpp": 3,
                "attention_reader.cpp": 4,
                "reduce_reader.cpp": 5,
                "reduce_writer.cpp": 6,
            }.get(name, 0)
            defines = dict(kernel.defines)
            rt = kernel.runtime_args
            cores = ttnn.corerange_to_cores(kernel.core_ranges, row_wise=True)
            original_args = {}
            for c in cores:
                try:
                    original_args[c.x, c.y] = list(rt[c.x][c.y])
                except IndexError:
                    original_args[c.x, c.y] = []
            offset = max(len(args) for args in original_args.values())
            for c in cores:
                args = original_args[c.x, c.y]
                args += [0] * (offset - len(args))
                rt[c.x][c.y] = [
                    *args,
                    self.state.buffer_address(),
                    *masks[c.x, c.y],
                    self.body.address_table.buffer_address(),
                    self.first,
                    self.count,
                    *self.barrier_addresses,
                    physical[0].x,
                    physical[0].y,
                    len(self.cores),
                    indices[c.x, c.y],
                    *coordinates,
                ]
            kernel.runtime_args = rt
            kernel.defines = [
                *kernel.defines,
                ("LOOP_SOURCE", '"' + original + '"'),
                ("LOOP_PATCH", str(patch)),
                ("BOUNDED_LAYER_BARRIER", str(int(self.body.tuning.bounded_barrier))),
                ("LOOP_RT_OFFSET", str(offset)),
                ("LOOP_CT_OFFSET", str(len(kernel.compile_time_args))),
                ("LOOP_CACHE_COLUMN", "5" if "VALUE" in defines else "4"),
            ]
            if "models/demos/llama31_8b_qb2/tt/megakernel/kernels" not in original:
                kernel.defines = [*kernel.defines, ("LOOP_NATIVE", "1")]
            kernel.compile_time_args = [
                *kernel.compile_time_args,
                *ttnn.TensorAccessorArgs(self.body.address_table).get_compile_time_args(),
            ]
            kernel.kernel_source = str(Path(__file__).with_name("kernels") / "layer_loop.cpp")
        program.kernels = kernels
        if self.head is not None:
            program = self.head.append(program, self.body.gather_output, wait_for_gather=True)
            if self.head_prefetch is not None:
                program = self.head_prefetch.append_helpers(program)
        return program

    def tensors(self):
        return (
            [self.state, *[tensor for pair in self.caches for tensor in pair]]
            + ([self.embedding_weight] if self.embedding_weight is not None else [])
            + (self.head.tensors() if self.head is not None else [])
        )

    def __call__(self, residual, position, page_table, rotary_position=None, tokens=None):
        # Both residual phases share one TensorAccessor specification. Match
        # the native layer's input conversion before substituting the L1
        # reduction output on the second phase and subsequent layers.
        residual = ttnn.to_memory_config(residual, self.body.layers[0].local_residual_memcfg)
        return self.body(
            residual,
            self.first,
            residual=residual,
            layer_loop=self,
            embedding_tokens=tokens,
            cache_inputs=(
                *self.caches[self.first],
                position,
                page_table,
                position if rotary_position is None else rotary_position,
            ),
        )
