# SPDX-License-Identifier: Apache-2.0
"""Bounded TP4 scratch for serialized, complete out/down decoder invocations.

The two roles must alternate on every rank. Completion of the intervening
collective proves that all ranks finished the previous use of the other slot.
One slot alone is unsafe: arrival readiness is not remote-consumer completion.
Keep this owner alive until all referencing traces retire and queued work ends.
Concurrent CQs/subdevices require distinct owners or external serialization.
"""
import ttnn
from models.common.modules.tt_ccl import TT_CCL


class DecodeAllReduceWorkspace:
    def __init__(self, mesh_device, dtype):
        if list(mesh_device.shape) != [1, 4] or dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            raise ValueError('Workspace requires TP4 and BF16 or BFP8 payloads')
        self.mesh_device, self.dtype = mesh_device, dtype
        # One mesh-wide semaphore pool, shared by all decoder views/layers.
        # Per-layer copies waste L1 even when their collectives are serialized.
        self.ccl=TT_CCL(mesh_device)
        self.native_rs_buffers = {}
        self.prefill_rs_buffers = {}
        self.prefill_expected_role = None
        self.expected_role = None
        self.valid = True
        grid = mesh_device.compute_with_storage_grid_size()
        semaphore_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(grid.x-1,grid.y-1))])
        scratch_memory = ttnn.create_sharded_memory_config([32,2560],ttnn.CoreGrid(x=8,y=1),
            ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
        self.slots = {role:(
            ttnn.empty([1,1,32,20480],device=mesh_device,dtype=dtype,layout=ttnn.TILE_LAYOUT,memory_config=scratch_memory),
            ttnn.create_global_semaphore(mesh_device,semaphore_grid,0)) for role in ('out','down')}
        self.scratch_bytes_per_device = 2 * (32*20480//1024) * (2048 if dtype==ttnn.bfloat16 else 1088)

    def prepare_native_rs(self, batches):
        """Setup only: all supported logical batch shapes precede trace capture."""
        for batch in batches:
            if batch in self.native_rs_buffers:
                continue
            if not 1 <= batch <= 32:
                raise ValueError('Native RS workspace supports one tile of logical users')
            self.native_rs_buffers[batch] = {
                role: tuple(ttnn.zeros([1,1,batch,width],device=self.mesh_device,
                    dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=memory)
                    for width,memory in ((5120,ttnn.DRAM_MEMORY_CONFIG),(1280,ttnn.L1_MEMORY_CONFIG)))
                for role in ('out','down')}

    def prepare_prefill_rs(self):
        """One shared 4096-row signature, allocated before any trace capture."""
        if self.prefill_rs_buffers:
            return
        self.prefill_rs_buffers = {
            role: tuple(ttnn.empty([1,1,4096,width],device=self.mesh_device,
                dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for width in (5120,1280)) for role in ('out','down')}

    def begin_prefill(self):
        if not self.valid or self.expected_role is not None:
            raise RuntimeError('Workspace requires serialized complete prefill invocations')
        self.expected_role, self.prefill_expected_role = 'prefill', 'out'

    def require_prefill_role(self, role):
        if not self.valid or self.expected_role!='prefill' or self.prefill_expected_role!=role:
            raise RuntimeError('Prefill scratch requires complete ordered out/down projections')

    def complete_prefill_role(self, role):
        self.require_prefill_role(role)
        self.prefill_expected_role = 'down' if role=='out' else 'out'

    def borrow_prefill_rs(self, role):
        self.require_prefill_role(role)
        return self.prefill_rs_buffers[role]

    def finish_prefill(self):
        if self.expected_role!='prefill' or self.prefill_expected_role!='out':
            self.abort()
            raise RuntimeError('Unbalanced prefill projection sequence; workspace invalidated')
        self.expected_role, self.prefill_expected_role = None, None

    def borrow_native_rs(self, role, batch):
        self.require_role(role)
        buffers=self.native_rs_buffers[batch][role]
        self.expected_role = 'down' if role=='out' else 'complete'
        return buffers

    def validate_owner(self, mesh_device, dtype):
        if mesh_device is not self.mesh_device or dtype != self.dtype or not self.valid:
            raise ValueError('Workspace mesh, dtype, or lifetime does not match this decoder')

    def begin(self):
        if not self.valid or self.expected_role is not None:
            raise RuntimeError('Workspace requires serialized complete decoder invocations')
        self.expected_role = 'out'

    def require_role(self, role):
        if not self.valid or role != self.expected_role:
            raise RuntimeError('Shared scratch requires a complete out/down decode_forward; partial projection use is forbidden')

    def borrow(self, role):
        self.require_role(role)
        self.expected_role = 'down' if role=='out' else 'complete'
        return self.slots[role]

    def finish(self):
        if self.expected_role != 'complete':
            self.abort()
            raise RuntimeError('Unbalanced all-reduce sequence; workspace invalidated')
        self.expected_role = None

    def abort(self):
        # A partial failed enqueue cannot be repaired by resetting a Python counter.
        self.valid = False
        self.expected_role = None
        self.prefill_expected_role = None
