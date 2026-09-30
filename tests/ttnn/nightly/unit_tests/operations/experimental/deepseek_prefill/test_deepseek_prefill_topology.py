# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TensorTopology contract of the in-place deepseek_prefill ops: the caller-owned output keeps the
caller's declared distribution.

``ttnn.experimental.deepseek_prefill.insert`` and ``moe_padding_config`` write into a tensor the
caller allocated and hand back a handle to that same tensor. Before these ops defined
``compute_output_topologies``, the framework relabelled that handle with the UNION of every input's
placements. The union only differs from the output's own label when an *unrelated* input carries a
higher-rank (N-D) shard label, so each test here feeds exactly that configuration:

  * output labelled by a collapsed 1-D mapper (``ReplicateTensorToMesh`` / ``ShardTensorToMesh``),
  * one other input labelled by ``ShardTensor2dMesh`` (an N-D label).

Negative control (pre-fix behaviour, reproducible by deleting the op's ``compute_output_topologies``):
the returned handle -- and, because tensor handles share their attributes, the caller's original
handle too -- came back with the 2-D union label, i.e. ``distribution_shape == (1, 2)`` and a
``PlacementShard`` copied from the unrelated input. The ``!=`` assertions against that unrelated
input's topology below are exactly what failed before the fix.

Because ``Tensor.tensor_topology()`` returns a reference into the tensor, a pre-op snapshot must be
taken as a value (``repr``) or compared against an independent tensor built with the same mapper;
both are done here.

The program cache is topology-blind (neither op hashes a topology), so a second call with the
differently-labelled input must hit the cached program.
"""

import pytest
import torch

import ttnn

TILE = 32


def _replicated(mesh_device, torch_tensor, *, dtype, layout):
    return ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _sharded_2d_along_cols(mesh_device, torch_tensor, *, dtype, layout):
    """N-D label: replicate over mesh rows, shard tensor dim 0 over mesh columns (the 2 devices of a 1x2)."""
    return ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
    )


def _index_tensor(mesh_device, values):
    return _replicated(
        mesh_device, torch.tensor(values, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def _per_device(tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


# ---------------------------------------------------------------------------
# prefill_insert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_prefill_insert_keeps_caller_topology(mesh_device):
    """global_tensor is 1-D Replicate; local_tensor is a 2-D shard whose two halves hold identical data
    (so the Replicate label stays true to the data). The returned handle must keep the 1-D Replicate
    label, the numeric result must be the in-place slice copy, and the second call must be a
    program-cache hit."""
    torch.manual_seed(0)
    global_rows, local_rows, hidden_dim = 128, 64, 64
    starts, counts, expert_id = [0, 32, 64, 96], [32, 32, 32, 32], 0
    rows = counts[expert_id]
    start = starts[expert_id]

    global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    local_a = torch.randn(local_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    local_b = torch.randn(local_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)

    g = _replicated(mesh_device, global_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    # Independent tensor with the same mapper: a reference label that no op can relabel by aliasing.
    g_ref = _replicated(mesh_device, torch.zeros_like(global_torch), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    topology_before = repr(g.tensor_topology())
    assert g.tensor_topology() == g_ref.tensor_topology()
    assert list(g.tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]

    # Production-like labelling: everything replicated (union == own even before the fix).
    l_a = _replicated(mesh_device, local_a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    # Breaking labelling: 2-D shard of two identical halves -> N-D label, replicated data.
    l_b = _sharded_2d_along_cols(
        mesh_device, torch.cat([local_b, local_b], dim=0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
    )
    assert list(l_b.tensor_topology().distribution_shape()) == list(mesh_device.shape)
    assert l_b.tensor_topology() != g_ref.tensor_topology()

    s = _index_tensor(mesh_device, starts)
    c = _index_tensor(mesh_device, counts)
    idx_table = _index_tensor(mesh_device, list(range(len(starts))))

    # bfp8_b-quantised snapshots for bit-exact expectations (the kernel is a tile-level byte copy).
    initial_g_q = _per_device(g)[0].clone()
    local_a_q = _per_device(l_a)[0]
    local_b_q = _per_device(l_b)[0]

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        out = ttnn.experimental.deepseek_prefill.insert(g, l_a, s, c, idx_table, local_expert_id=expert_id)
        ttnn.synchronize_device(mesh_device)
        entries = mesh_device.num_program_cache_entries()
        assert entries > 0

        expected = initial_g_q.clone()
        expected[start : start + rows, :] = local_a_q[:rows, :]
        for shard in _per_device(out):
            assert torch.equal(shard.float(), expected.float())
        assert repr(out.tensor_topology()) == topology_before

        # Same caller-owned global_tensor, unrelated input now carries the 2-D shard label.
        out = ttnn.experimental.deepseek_prefill.insert(g, l_b, s, c, idx_table, local_expert_id=expert_id)
        ttnn.synchronize_device(mesh_device)

        # Topology-blind program cache: the relabelled input must not miss.
        assert mesh_device.num_program_cache_entries() == entries

        # The contract under test. Pre-fix: both handles carried the union label == l_b's label.
        assert repr(out.tensor_topology()) == topology_before, out.tensor_topology()
        assert repr(g.tensor_topology()) == topology_before, g.tensor_topology()
        assert out.tensor_topology() == g_ref.tensor_topology()
        assert out.tensor_topology() != l_b.tensor_topology()
        assert list(out.tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]

        expected[start : start + rows, :] = local_b_q[:rows, :]
        for shard in _per_device(out):
            assert torch.equal(shard.float(), expected.float())
    finally:
        mesh_device.disable_and_clear_program_cache()


# ---------------------------------------------------------------------------
# moe_padding_config
# ---------------------------------------------------------------------------


def _sequential_counts(actual_isl, sp_factor, tokens_per_chip):
    """Host reference for actual_start == 0: the rotation is the identity, so chip c carries global rows
    [c*tokens_per_chip, (c+1)*tokens_per_chip) and its real rows are the prefix that lies below actual_isl."""
    return [max(0, min(actual_isl - c * tokens_per_chip, tokens_per_chip)) for c in range(sp_factor)]


def _meta_replicated(mesh_device, value):
    """The production form: 1-element uint32 ROW_MAJOR DRAM tensor, replicated (collapsed 1-D label)."""
    return _replicated(
        mesh_device,
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _meta_sharded_2d(mesh_device, value, num_shards):
    """Same per-device 1-element tensor, but labelled by a 2-D shard of `num_shards` identical values."""
    return _sharded_2d_along_cols(
        mesh_device,
        torch.full((num_shards, 1, 1, 1), value, dtype=torch.int64),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _config_tensor(mesh_device, config_mapper, sp_factor):
    """Per-device [1, 2] uint32 ROW_MAJOR DRAM config row under a collapsed 1-D label."""
    if config_mapper == "replicate_1d":
        return _replicated(
            mesh_device, torch.zeros((1, 2), dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    assert config_mapper == "shard_1d"
    # The production labelling: one row per chip along the SP axis.
    return ttnn.from_torch(
        torch.zeros((sp_factor, 2), dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _read_rows(config):
    rows = [shard.to(torch.int64).reshape(-1).tolist() for shard in _per_device(config)]
    return [row[0] for row in rows], [row[1] for row in rows]


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
@pytest.mark.parametrize("config_mapper", ["replicate_1d", "shard_1d"])
def test_moe_padding_config_keeps_caller_topology(mesh_device, config_mapper):
    """config carries a collapsed 1-D label (the Shard(0) variant is the production labelling); the
    actual_start/actual_end metadata tensors carry a 2-D shard label. The config handle must keep its
    own label, the per-chip rows must match the host reference, and the second call must be a
    program-cache hit. The SP axis is mesh axis 1 (the 2-device axis of a 1x2 mesh)."""
    cluster_axis = 1
    sp_factor = int(mesh_device.shape[cluster_axis])
    tokens_per_chip = 2 * TILE
    pad_side = 0

    config = _config_tensor(mesh_device, config_mapper, sp_factor)
    config_ref = _config_tensor(mesh_device, config_mapper, sp_factor)
    topology_before = repr(config.tensor_topology())
    assert config.tensor_topology() == config_ref.tensor_topology()
    assert list(config.tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        # Production labelling first: replicated metadata (union == own even before the fix).
        isl_1 = tokens_per_chip + 16
        ttnn.experimental.deepseek_prefill.moe_padding_config(
            config,
            _meta_replicated(mesh_device, 0),
            _meta_replicated(mesh_device, isl_1),
            tokens_per_chip=tokens_per_chip,
            pad_side=pad_side,
            cluster_axis=cluster_axis,
        )
        ttnn.synchronize_device(mesh_device)
        entries = mesh_device.num_program_cache_entries()
        assert entries > 0
        counts, sides = _read_rows(config)
        assert counts == _sequential_counts(isl_1, sp_factor, tokens_per_chip)
        assert sides == [pad_side] * sp_factor
        assert repr(config.tensor_topology()) == topology_before

        # Breaking labelling: the unrelated metadata inputs now carry a 2-D shard label.
        isl_2 = 40
        start_t = _meta_sharded_2d(mesh_device, 0, sp_factor)
        end_t = _meta_sharded_2d(mesh_device, isl_2, sp_factor)
        assert list(start_t.tensor_topology().distribution_shape()) == list(mesh_device.shape)
        out = ttnn.experimental.deepseek_prefill.moe_padding_config(
            config,
            start_t,
            end_t,
            tokens_per_chip=tokens_per_chip,
            pad_side=pad_side,
            cluster_axis=cluster_axis,
        )
        ttnn.synchronize_device(mesh_device)

        # Topology-blind program cache: same structural hash, so a hit.
        assert mesh_device.num_program_cache_entries() == entries

        # The contract under test. Pre-fix: both handles carried the 2-D union label.
        assert repr(out.tensor_topology()) == topology_before, out.tensor_topology()
        assert repr(config.tensor_topology()) == topology_before, config.tensor_topology()
        assert out.tensor_topology() == config_ref.tensor_topology()
        assert out.tensor_topology() != start_t.tensor_topology()
        assert list(out.tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]

        counts, sides = _read_rows(out)
        assert counts == _sequential_counts(isl_2, sp_factor, tokens_per_chip)
        assert sides == [pad_side] * sp_factor
    finally:
        mesh_device.disable_and_clear_program_cache()
