# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-place KV-cache ops must leave the cache's own TensorTopology untouched.

``ttnn.update_cache`` / ``ttnn.fill_cache`` and the experimental
``paged_update_cache`` / ``paged_fill_cache`` / ``paged_fused_update_cache`` write
into a caller-owned cache and return that same tensor. ``device_operation::launch``
relabels every returned tensor; an op without a ``compute_output_topologies`` hook
gets the union of all input labels, which describes a freshly allocated output, not
a buffer whose distribution was fixed by whoever allocated it.

Negative control (fails before the hook was added): allocate the cache the way the
shipped Llama models do, ``ReplicateTensorToMesh`` (collapsed 1-D label
``{2},[Replicate]``), and feed an update mapped with
``ShardTensor2dMesh(mesh_shape=(1, 2), dims=(None, 1))`` (2-D label
``{1,2},[Replicate, Shard(1)]``). The union ignores a fully replicated input, takes
the max distribution rank (2) from the update and drops the rank-1 cache label
outright, so the cache came back labelled ``{1,2},[Replicate, Shard(1)]`` -- the
update's label -- on the caller's handle as well, because Tensor copies share
``tensor_attributes``. The ``tensor_topology() == before`` assertions below are
exactly what failed, and ``!= update label`` pins the value the union produced.

The program-cache tests confirm the fix is invisible to program-cache keys: the op
hashes never included topology, so a replicated update followed by a mesh-sharded
update of the same per-device shape must hit the cached program.
"""

import pytest
import torch

import ttnn

MESH_SHAPE = (1, 2)
NUM_DEVICES = MESH_SHAPE[0] * MESH_SHAPE[1]
# Decode-time update tensors carry the head (update_cache) or user (paged_*) axis
# padded to a full tile, exactly as the models hand them to the kernels.
PADDED_ROWS = 32


def _to_mesh(mesh_device, tensor, *, sharded, shard_dim, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config):
    """Replicate ``tensor`` to every device, or split ``shard_dim`` across the mesh columns."""
    if sharded:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, shard_dim))
    else:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )


def _replicated(mesh_device, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return _to_mesh(
        mesh_device,
        tensor,
        sharded=False,
        shard_dim=0,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _height_sharded_l1(shard_grid, shard_shape):
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _core_row(row, num_cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(num_cores - 1, row))])


def _assert_collapsed_replicate(topology):
    """The cache label the models produce: one Replicate placement over the flattened mesh."""
    placements = topology.placements()
    assert len(placements) == 1
    assert isinstance(placements[0], ttnn.PlacementReplicate)


class _Case:
    """One in-place cache op: how to build its tensors, run it, and compute the torch reference.

    ``cache_local`` shapes are per-device (the cache is replicated, so host shape == device
    shape). ``update_shard_dim`` / ``update_local_extent`` describe how a mesh-sharded update
    splits across the two devices, and ``concat_dim`` is the cache axis along which the
    per-device caches are concatenated to form the global reference.
    """

    num_caches = 1
    update_shard_dim = 1
    concat_dim = 1

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

    def new_caches(self):
        """Returns [(cache_torch_local, cache_tt)] with every cache replicated."""
        raise NotImplementedError

    def new_updates(self, sharded):
        """Returns [(update_torch, update_tt)]; the torch tensor is global when sharded, local otherwise."""
        raise NotImplementedError

    def run(self, caches_tt, updates_tt):
        """Returns the list of handles the op returned, one per cache."""
        raise NotImplementedError

    def apply(self, cache, update):
        """In-place torch reference of one device's op on its local cache/update slices."""
        raise NotImplementedError

    def local_update(self, update, sharded, device_idx):
        if not sharded:
            return update
        return update.narrow(self.update_shard_dim, device_idx * self.update_local_extent, self.update_local_extent)

    def expected_global(self, cache_local, steps):
        """Replays ``steps`` = [(update_torch, sharded)] on each device's cache and concatenates."""
        per_device = []
        for device_idx in range(NUM_DEVICES):
            cache = cache_local.clone()
            for update, sharded in steps:
                self.apply(cache, self.local_update(update, sharded, device_idx))
            per_device.append(cache)
        return torch.cat(per_device, dim=self.concat_dim)

    def read_global(self, cache_tt):
        return ttnn.to_torch(cache_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=self.concat_dim))


class _UpdateCacheCase(_Case):
    """ttnn.update_cache: cache [users, heads, seq, head_dim], update [1, heads, padded users, head_dim]."""

    num_users = 4
    local_heads = 2
    seq_len = 128
    head_dim = 64
    update_idx = 7
    update_local_extent = local_heads

    def new_caches(self):
        cache = torch.randn(self.num_users, self.local_heads, self.seq_len, self.head_dim, dtype=torch.bfloat16)
        return [(cache, _replicated(self.mesh_device, cache))]

    def new_updates(self, sharded):
        heads = self.local_heads * NUM_DEVICES if sharded else self.local_heads
        update = torch.randn(1, heads, PADDED_ROWS, self.head_dim, dtype=torch.bfloat16)
        update_tt = _to_mesh(
            self.mesh_device,
            update,
            sharded=sharded,
            shard_dim=self.update_shard_dim,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return [(update, update_tt)]

    def run(self, caches_tt, updates_tt):
        return [ttnn.update_cache(caches_tt[0], updates_tt[0], self.update_idx)]

    def apply(self, cache, update):
        for user in range(self.num_users):
            cache[user, :, self.update_idx, :] = update[0, :, user, :]


class _FillCacheCase(_Case):
    """ttnn.fill_cache: cache [users, heads, seq, head_dim], update [1, heads, fill_len, head_dim]."""

    num_users = 2
    local_heads = 2
    seq_len = 128
    head_dim = 64
    fill_len = 64
    batch_idx = 1
    update_idx = 32
    update_local_extent = local_heads

    def new_caches(self):
        cache = torch.randn(self.num_users, self.local_heads, self.seq_len, self.head_dim, dtype=torch.bfloat16)
        return [(cache, _replicated(self.mesh_device, cache))]

    def new_updates(self, sharded):
        heads = self.local_heads * NUM_DEVICES if sharded else self.local_heads
        update = torch.randn(1, heads, self.fill_len, self.head_dim, dtype=torch.bfloat16)
        update_tt = _to_mesh(
            self.mesh_device,
            update,
            sharded=sharded,
            shard_dim=self.update_shard_dim,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return [(update, update_tt)]

    def run(self, caches_tt, updates_tt):
        return [ttnn.fill_cache(caches_tt[0], updates_tt[0], self.batch_idx, update_idx=self.update_idx)]

    def apply(self, cache, update):
        cache[self.batch_idx, :, self.update_idx : self.update_idx + self.fill_len, :] = update[0]


class _PagedUpdateCacheCase(_Case):
    """paged_update_cache (non-paged decode path): cache [users, heads, seq, head_dim],
    update [1, users, padded heads, head_dim] height-sharded in L1 with one core per user."""

    local_users = 8
    num_heads = 1
    seq_len = 128
    head_dim = 128
    positions = [5, 17, 40, 77, 2, 99, 64, 31]
    update_local_extent = local_users
    concat_dim = 0
    input_core_row = 0

    def new_caches(self):
        cache = torch.randn(self.local_users, self.num_heads, self.seq_len, self.head_dim, dtype=torch.bfloat16)
        return [(cache, _replicated(self.mesh_device, cache))]

    def _new_update(self, sharded, core_row):
        users = self.local_users * NUM_DEVICES if sharded else self.local_users
        update = torch.randn(1, users, PADDED_ROWS, self.head_dim, dtype=torch.bfloat16)
        memory_config = _height_sharded_l1(_core_row(core_row, self.local_users), [PADDED_ROWS, self.head_dim])
        update_tt = _to_mesh(
            self.mesh_device, update, sharded=sharded, shard_dim=self.update_shard_dim, memory_config=memory_config
        )
        return update, update_tt

    def new_updates(self, sharded):
        return [self._new_update(sharded, self.input_core_row)]

    def positions_tt(self):
        return _replicated(
            self.mesh_device,
            torch.tensor(self.positions, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def run(self, caches_tt, updates_tt):
        positions_tt = self.positions_tt()
        return [ttnn.experimental.paged_update_cache(caches_tt[0], updates_tt[0], update_idxs_tensor=positions_tt)]

    def apply(self, cache, update):
        for user, position in enumerate(self.positions):
            cache[user, : self.num_heads, position, :] = update[0, user, : self.num_heads, :]


class _PagedFusedUpdateCacheCase(_PagedUpdateCacheCase):
    """paged_fused_update_cache: two caches, two updates on disjoint core rows, one returned tuple."""

    num_caches = 2
    positions = [3, 9, 33, 100, 0, 127, 50, 66]

    def new_caches(self):
        # zero-argument super() is not available inside a comprehension's scope
        new_single_cache = super().new_caches
        return [new_single_cache()[0] for _ in range(self.num_caches)]

    def new_updates(self, sharded):
        return [self._new_update(sharded, core_row) for core_row in range(self.num_caches)]

    def run(self, caches_tt, updates_tt):
        out1, out2 = ttnn.experimental.paged_fused_update_cache(
            caches_tt[0], updates_tt[0], caches_tt[1], updates_tt[1], update_idxs_tensor=self.positions_tt()
        )
        return [out1, out2]


class _PagedFillCacheCase(_Case):
    """paged_fill_cache: cache [blocks, heads, block_size, head_dim], update [1, heads, fill_len, head_dim]."""

    num_blocks = 8
    local_heads = 2
    block_size = 32
    head_dim = 64
    fill_len = 64
    page_table = [[3, 0], [5, 6]]
    batch_idx = 1
    update_local_extent = local_heads

    def new_caches(self):
        cache = torch.randn(self.num_blocks, self.local_heads, self.block_size, self.head_dim, dtype=torch.bfloat16)
        return [(cache, _replicated(self.mesh_device, cache))]

    def new_updates(self, sharded):
        heads = self.local_heads * NUM_DEVICES if sharded else self.local_heads
        update = torch.randn(1, heads, self.fill_len, self.head_dim, dtype=torch.bfloat16)
        update_tt = _to_mesh(
            self.mesh_device,
            update,
            sharded=sharded,
            shard_dim=self.update_shard_dim,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return [(update, update_tt)]

    def run(self, caches_tt, updates_tt):
        page_table_tt = _replicated(
            self.mesh_device,
            torch.tensor(self.page_table, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache_tt, update_tt = caches_tt[0], updates_tt[0]
        return [ttnn.experimental.paged_fill_cache(cache_tt, update_tt, page_table_tt, batch_idx=self.batch_idx)]

    def apply(self, cache, update):
        for virtual_block in range(self.fill_len // self.block_size):
            physical_block = self.page_table[self.batch_idx][virtual_block]
            start = virtual_block * self.block_size
            cache[physical_block] = update[0, :, start : start + self.block_size, :]


CASES = [
    pytest.param(_UpdateCacheCase, id="update_cache"),
    pytest.param(_FillCacheCase, id="fill_cache"),
    pytest.param(_PagedUpdateCacheCase, id="paged_update_cache"),
    pytest.param(_PagedFillCacheCase, id="paged_fill_cache"),
    pytest.param(_PagedFusedUpdateCacheCase, id="paged_fused_update_cache"),
]


def _assert_labels_kept(caches_tt, outputs_tt, updates, before):
    for cache_tt, out_tt, (_, update_tt), topology in zip(caches_tt, outputs_tt, updates, before):
        assert cache_tt.tensor_topology() == topology, "caller's cache handle was relabelled"
        assert out_tt.tensor_topology() == topology, "returned cache handle was relabelled"
        # Before the hook, the union handed the cache exactly the update's label.
        assert out_tt.tensor_topology() != update_tt.tensor_topology()


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
@pytest.mark.parametrize("case_type", CASES)
def test_inplace_cache_op_keeps_cache_topology(mesh_device, case_type):
    torch.manual_seed(11)
    case = case_type(mesh_device)
    caches = case.new_caches()
    updates = case.new_updates(sharded=True)
    caches_tt = [cache_tt for _, cache_tt in caches]
    before = [cache_tt.tensor_topology() for cache_tt in caches_tt]

    # Guard the configuration: a collapsed 1-D Replicate cache against a 2-D Shard update is
    # the rank-mismatch case the union mislabels; if either label changes shape the test is moot.
    for topology, (_, update_tt) in zip(before, updates):
        _assert_collapsed_replicate(topology)
        assert len(update_tt.tensor_topology().placements()) == 2
        assert update_tt.tensor_topology() != topology

    outputs_tt = case.run(caches_tt, [update_tt for _, update_tt in updates])

    _assert_labels_kept(caches_tt, outputs_tt, updates, before)
    for (cache_local, _), out_tt, (update, _) in zip(caches, outputs_tt, updates):
        assert torch.equal(case.read_global(out_tt), case.expected_global(cache_local, [(update, True)]))


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
@pytest.mark.parametrize("case_type", CASES)
def test_inplace_cache_op_program_cache_is_topology_blind(mesh_device, case_type):
    torch.manual_seed(13)
    case = case_type(mesh_device)
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()

    try:
        caches = case.new_caches()
        caches_tt = [cache_tt for _, cache_tt in caches]
        before = [cache_tt.tensor_topology() for cache_tt in caches_tt]

        replicated_updates = case.new_updates(sharded=False)
        outputs_tt = case.run(caches_tt, [update_tt for _, update_tt in replicated_updates])
        cache_entries = mesh_device.num_program_cache_entries()
        assert cache_entries > 0
        for cache_tt, out_tt, topology in zip(caches_tt, outputs_tt, before):
            assert cache_tt.tensor_topology() == topology
            assert out_tt.tensor_topology() == topology

        # Same per-device shapes and memory configs, only the mesh label differs: must be a cache hit.
        sharded_updates = case.new_updates(sharded=True)
        outputs_tt = case.run(caches_tt, [update_tt for _, update_tt in sharded_updates])
        assert mesh_device.num_program_cache_entries() == cache_entries

        _assert_labels_kept(caches_tt, outputs_tt, sharded_updates, before)
        for (cache_local, _), out_tt, (replicated, _), (sharded, _) in zip(
            caches, outputs_tt, replicated_updates, sharded_updates
        ):
            expected = case.expected_global(cache_local, [(replicated, False), (sharded, True)])
            assert torch.equal(case.read_global(out_tt), expected)
    finally:
        mesh_device.disable_and_clear_program_cache()
