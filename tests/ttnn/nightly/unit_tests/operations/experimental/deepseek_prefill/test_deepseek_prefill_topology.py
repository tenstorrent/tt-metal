# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-topology labels of the in-place deepseek_prefill ops.

``ttnn.experimental.deepseek_prefill.insert`` and ``moe_padding_config`` write into the caller's tensor and return
it. It keeps its label unless another input is sharded along a mesh axis on which that label is replicated; every
device then holds a different result, and the output takes the union of the labels, on the caller's handle too.

  * kept: output ``{2},[Shard(0)]`` (``moe_padding_config``'s config row in production, on a 1x2 mesh), the other
    input ``{1,2},[Replicate, Shard(0)]`` along the same mesh axis. The output keeps its collapsed label, where the
    union is the other input's N-D label (the ``!=`` assertions).
  * follows the data: output ``{2},[Replicate]``, the other input N-D-sharded with a different value on every
    device. Both handles carry that input's label, and each device's data is checked against its own input.

The label before the op is kept as a ``repr`` (readable failures) and as an independent tensor built with the same
mapper. The second call, with differently-labelled inputs, must hit the program cache.

CI pruning: ``insert`` has no production counterpart, so its tests carry the same ``no_production_counterpart`` rule
as ``test_deepseek_prefill_insert.py``; ``moe_padding_config`` is production (``TtMoEGatePrefill``) and stays
collected.
"""

import pytest
import torch

import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import (
    ci_pruning,
)

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


def _sharded_1d(mesh_device, torch_tensor, *, dim, dtype, layout):
    """Collapsed 1-D label ``{2},[Shard(dim)]``: the default mapper's labelling of a sharded allocation."""
    return ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )


def _index_tensor(mesh_device, values):
    return _replicated(
        mesh_device,
        torch.tensor(values, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _per_device(tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


# ---------------------------------------------------------------------------
# prefill_insert
# ---------------------------------------------------------------------------


def _insert_case(mesh_device, *, global_label):
    """global_tensor replicated (``{2},[Replicate]``) or collapsed-sharded on dim 0 (``{2},[Shard(0)]``); a replicated
    local_a (production labelling) and an N-D-sharded local_b whose two halves differ.
    """
    torch.manual_seed(0)
    global_rows, local_rows, hidden_dim = 128, 64, 64
    starts, counts, expert_id = [0, 32, 64, 96], [32, 32, 32, 32], 0
    num_devices = mesh_device.get_num_devices()

    local_a = torch.randn(local_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    local_b = torch.randn(local_rows * num_devices, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
    kwargs = dict(dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    if global_label == "replicate":
        global_torch = torch.randn(global_rows, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
        g = _replicated(mesh_device, global_torch, **kwargs)
        g_ref = _replicated(mesh_device, torch.zeros_like(global_torch), **kwargs)
    else:
        global_torch = torch.randn(global_rows * num_devices, hidden_dim, dtype=torch.float32).to(torch.bfloat16)
        g = _sharded_1d(mesh_device, global_torch, dim=0, **kwargs)
        g_ref = _sharded_1d(mesh_device, torch.zeros_like(global_torch), dim=0, **kwargs)
    assert g.tensor_topology() == g_ref.tensor_topology()
    assert list(g.tensor_topology().distribution_shape()) == [num_devices]

    l_a = _replicated(mesh_device, local_a, **kwargs)
    l_b = _sharded_2d_along_cols(mesh_device, local_b, **kwargs)
    assert list(l_b.tensor_topology().distribution_shape()) == list(mesh_device.shape)
    assert l_b.tensor_topology() != g_ref.tensor_topology()
    l_b_q = _per_device(l_b)
    assert not torch.equal(l_b_q[0], l_b_q[1]), "precondition: local_b must differ per device"

    index = (
        _index_tensor(mesh_device, starts),
        _index_tensor(mesh_device, counts),
        _index_tensor(mesh_device, list(range(len(starts)))),
    )
    return dict(
        g=g,
        g_ref=g_ref,
        l_a=l_a,
        l_b=l_b,
        index=index,
        expert_id=expert_id,
        start=starts[expert_id],
        rows=counts[expert_id],
        # bfp8_b-quantised snapshots for bit-exact expectations (the kernel is a tile-level byte copy).
        initial_g_q=[t.clone() for t in _per_device(g)],
        local_a_q=_per_device(l_a)[0],
        local_b_q=l_b_q,
    )


def _run_insert(mesh_device, c, local):
    out = ttnn.experimental.deepseek_prefill.insert(c["g"], local, *c["index"], local_expert_id=c["expert_id"])
    ttnn.synchronize_device(mesh_device)
    return out


def _assert_insert_rows(c, out, local_q_per_device):
    """Device d's global holds its own initial rows with [start:start+rows] replaced by its own local rows."""
    for dev, shard in enumerate(_per_device(out)):
        expected = c["initial_g_q"][dev].clone()
        expected[c["start"] : c["start"] + c["rows"], :] = local_q_per_device[dev][: c["rows"], :]
        assert torch.equal(shard.float(), expected.float()), f"device {dev}"


@pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)
@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_prefill_insert_sharded_global_keeps_own_label(mesh_device):
    """global_tensor is 1-D ``Shard(0)``; local_tensor carries the 2-D ``[Replicate, Shard(0)]`` label along the
    same mesh axis, so it is sharded only where global_tensor already is: the returned handle keeps the 1-D label
    (the union would be local_tensor's 2-D one), each device's rows come from its own local slice, and the second
    call is a program-cache hit."""
    c = _insert_case(mesh_device, global_label="shard")
    topology_before = repr(c["g"].tensor_topology())

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        out = _run_insert(mesh_device, c, c["l_a"])  # production labelling first: union == own either way
        entries = mesh_device.num_program_cache_entries()
        assert entries > 0
        _assert_insert_rows(c, out, [c["local_a_q"]] * mesh_device.get_num_devices())
        assert repr(out.tensor_topology()) == topology_before

        out = _run_insert(mesh_device, c, c["l_b"])
        assert mesh_device.num_program_cache_entries() == entries  # topology-blind program cache

        assert repr(out.tensor_topology()) == topology_before, out.tensor_topology()
        assert repr(c["g"].tensor_topology()) == topology_before, c["g"].tensor_topology()
        assert out.tensor_topology() == c["g_ref"].tensor_topology()
        assert out.tensor_topology() != c["l_b"].tensor_topology()  # not the union
        assert list(out.tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]
        _assert_insert_rows(c, out, c["local_b_q"])
    finally:
        mesh_device.disable_and_clear_program_cache()


@pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)
@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_prefill_insert_label_follows_the_data(mesh_device):
    """global_tensor is 1-D ``Replicate``; local_tensor is 2-D-sharded with a different half per device, so after
    the insert every device's global_tensor differs: both handles carry local_tensor's label (the union). A kept
    ``Replicate`` would make the serialiser save one device's rows as everyone's. A replicated local_tensor (the
    production labelling) keeps the label, and the relabelled call is still a program-cache hit.
    """
    c = _insert_case(mesh_device, global_label="replicate")
    topology_before = repr(c["g"].tensor_topology())

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        out = _run_insert(mesh_device, c, c["l_a"])
        entries = mesh_device.num_program_cache_entries()
        assert entries > 0
        _assert_insert_rows(c, out, [c["local_a_q"]] * mesh_device.get_num_devices())
        assert repr(out.tensor_topology()) == topology_before

        out = _run_insert(mesh_device, c, c["l_b"])
        assert mesh_device.num_program_cache_entries() == entries

        assert out.tensor_topology() == c["l_b"].tensor_topology(), out.tensor_topology()
        assert c["g"].tensor_topology() == c["l_b"].tensor_topology(), c["g"].tensor_topology()
        assert repr(out.tensor_topology()) != topology_before
        _assert_insert_rows(c, out, c["local_b_q"])
    finally:
        mesh_device.disable_and_clear_program_cache()


# ---------------------------------------------------------------------------
# moe_padding_config
# ---------------------------------------------------------------------------


def _sequential_counts(actual_isl, sp_factor, tokens_per_chip):
    """Host reference for actual_start == 0: the rotation is the identity, so chip c carries global rows
    [c*tokens_per_chip, (c+1)*tokens_per_chip) and its real rows are the prefix that lies below actual_isl.
    """
    return [max(0, min(actual_isl - c * tokens_per_chip, tokens_per_chip)) for c in range(sp_factor)]


def _meta_replicated(mesh_device, value):
    """The production form: 1-element uint32 ROW_MAJOR DRAM tensor, replicated (collapsed 1-D label)."""
    return _replicated(
        mesh_device,
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _meta_sharded_2d(mesh_device, values):
    """Per-device 1-element tensors under a 2-D shard label: device c holds ``values[c]``."""
    return _sharded_2d_along_cols(
        mesh_device,
        torch.tensor(values, dtype=torch.int64).reshape(len(values), 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _config_tensor(mesh_device, config_mapper, sp_factor):
    """Per-device [1, 2] uint32 ROW_MAJOR DRAM config row under a collapsed 1-D label."""
    if config_mapper == "replicate_1d":
        return _replicated(
            mesh_device,
            torch.zeros((1, 2), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    assert config_mapper == "shard_1d"
    # One row per chip along the SP axis (the production labelling, collapsed to 1-D on a 1x2 mesh).
    return _sharded_1d(
        mesh_device,
        torch.zeros((sp_factor, 2), dtype=torch.int32),
        dim=0,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _read_rows(config):
    rows = [shard.to(torch.int64).reshape(-1).tolist() for shard in _per_device(config)]
    return [row[0] for row in rows], [row[1] for row in rows]


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
@pytest.mark.parametrize("config_mapper", ["replicate_1d", "shard_1d"])
def test_moe_padding_config_topology(mesh_device, config_mapper):
    """config carries a collapsed 1-D label; the first call uses replicated metadata (production), the second
    2-D-sharded metadata with a different ``actual_end`` per device. ``shard_1d`` (the production labelling): the
    metadata is sharded only where the config already is, so the config keeps its 1-D label although the union
    would be the 2-D one. ``replicate_1d``: every device computes a different row, so the config takes the
    metadata's 2-D label on both handles. The per-chip rows match the host reference for each device's own
    ``actual_end`` either way, and the second call is a program-cache hit. The SP axis is mesh axis 1.
    """
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
        # Production labelling first: replicated metadata, whose union with the config is the config's label.
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
        # A replicated config keeps {2},[Replicate] here, although each chip writes its own row from its coordinate
        # along cluster_axis: the op has no compute_output_topologies, and the framework default checks the config
        # against the metadata only, which are replicated. The label is checked for a sharded config only.
        if config_mapper != "replicate_1d":
            assert repr(config.tensor_topology()) == topology_before

        # The metadata now carries a 2-D shard label, with a different actual_end on every device.
        isl_per_device = [40, 72][:sp_factor]
        expected = [
            _sequential_counts(isl, sp_factor, tokens_per_chip)[chip] for chip, isl in enumerate(isl_per_device)
        ]
        single_value_rows = [_sequential_counts(isl, sp_factor, tokens_per_chip) for isl in isl_per_device + [isl_1]]
        assert expected not in single_value_rows, "precondition: no single actual_end gives the per-device rows"
        start_t = _meta_sharded_2d(mesh_device, [0] * sp_factor)
        end_t = _meta_sharded_2d(mesh_device, isl_per_device)
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

        if config_mapper == "shard_1d":
            assert repr(out.tensor_topology()) == topology_before, out.tensor_topology()
            assert repr(config.tensor_topology()) == topology_before, config.tensor_topology()
            assert out.tensor_topology() == config_ref.tensor_topology()
            assert out.tensor_topology() != start_t.tensor_topology()  # not the union
            assert list(out.tensor_topology().distribution_shape()) == [mesh_device.get_num_devices()]
        else:
            assert out.tensor_topology() == start_t.tensor_topology(), out.tensor_topology()
            assert config.tensor_topology() == start_t.tensor_topology(), config.tensor_topology()
            assert repr(out.tensor_topology()) != topology_before

        counts, sides = _read_rows(out)
        assert counts == expected
        assert sides == [pad_side] * sp_factor
    finally:
        mesh_device.disable_and_clear_program_cache()
