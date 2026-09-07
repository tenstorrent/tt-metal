# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the fused Blaze-compatible CSA compressor, plus the host-side slab-alignment
contract that decides the shapes those tests (and prefill) run at."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.mla.compressor import (
    CSA_STATE_ROWS,
    csa_slab_align,
    csa_state_predecessor_ca,
    csa_state_rows,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

_BATCH = 1
_COMPRESS_RATE = 4
_HEAD_DIM = 512
_INDEX_HEAD_DIM = 128
_STATE_ROWS = CSA_STATE_ROWS
_LOCAL_SEQ_LEN = 128
_PCC = 0.999


def _update_state(kv_state, score_state, kv, gate, position_bias, start_position, head_dim):
    kv_state = kv_state.clone()
    score_state = score_state.clone()
    for local_position in range(kv.shape[2]):
        position = start_position + local_position
        slot = position % _COMPRESS_RATE
        ca_row, cb_row = csa_state_rows(position, _COMPRESS_RATE)
        biased_gate = gate[:, :, local_position] + position_bias[:, :, slot]
        kv_state[:, :, ca_row] = kv[:, :, local_position, :head_dim]
        kv_state[:, :, cb_row] = kv[:, :, local_position, head_dim:]
        score_state[:, :, ca_row] = biased_gate[..., :head_dim]
        score_state[:, :, cb_row] = biased_gate[..., head_dim:]
    return kv_state, score_state


def _compress_local(kv, gate, position_bias, predecessor_kv, predecessor_score, start_position, head_dim):
    n_windows = kv.shape[2] // _COMPRESS_RATE
    pooled = torch.zeros(_BATCH, 1, n_windows, head_dim, dtype=torch.bfloat16)
    for window in range(n_windows):
        current_start = window * _COMPRESS_RATE
        current_end = current_start + _COMPRESS_RATE
        absolute_start = start_position + current_start
        current_kv = kv[:, :, current_start:current_end]
        current_gate = gate[:, :, current_start:current_end] + position_bias

        if window == 0:
            state_start = csa_state_predecessor_ca(absolute_start, _COMPRESS_RATE)
            previous_ca_kv = predecessor_kv[:, :, state_start : state_start + _COMPRESS_RATE]
            previous_ca_score = predecessor_score[:, :, state_start : state_start + _COMPRESS_RATE]
        else:
            previous_start = current_start - _COMPRESS_RATE
            previous_end = current_start
            previous_ca_kv = kv[:, :, previous_start:previous_end, :head_dim]
            previous_ca_score = gate[:, :, previous_start:previous_end, :head_dim] + position_bias[..., :head_dim]

        overlap_kv = torch.cat([previous_ca_kv, current_kv[..., head_dim:]], dim=2)
        overlap_score = torch.cat([previous_ca_score, current_gate[..., head_dim:]], dim=2)
        weights = overlap_score.softmax(dim=2, dtype=torch.float32).to(overlap_kv.dtype)
        pooled[:, :, window] = (overlap_kv * weights).sum(dim=2)
    return pooled


def _torch_csa_compressor(
    kv,
    gate,
    position_bias,
    initial_kv_state,
    initial_score_state,
    sp_factor,
    seq_len_actual,
    first_token_position,
    head_dim,
):
    local_seq_len = kv.shape[2] // sp_factor
    pooled_outputs = []
    kv_states = []
    score_states = []
    predecessor_kv = initial_kv_state
    predecessor_score = initial_score_state

    for rank in range(sp_factor):
        local_start = rank * local_seq_len
        local_end = local_start + local_seq_len
        local_valid = max(0, min(local_seq_len, seq_len_actual - local_start))
        local_position = first_token_position + local_start
        local_kv = kv[:, :, local_start:local_end]
        local_gate = gate[:, :, local_start:local_end]

        complete_tokens = local_valid // _COMPRESS_RATE * _COMPRESS_RATE
        local_pooled = _compress_local(
            local_kv[:, :, :complete_tokens],
            local_gate[:, :, :complete_tokens],
            position_bias,
            predecessor_kv,
            predecessor_score,
            local_position,
            head_dim,
        )
        padded_pooled = torch.zeros(_BATCH, 1, local_seq_len // _COMPRESS_RATE, head_dim, dtype=torch.bfloat16)
        padded_pooled[:, :, : local_pooled.shape[2]] = local_pooled
        pooled_outputs.append(padded_pooled)

        local_kv_state, local_score_state = _update_state(
            predecessor_kv,
            predecessor_score,
            local_kv[:, :, :local_valid],
            local_gate[:, :, :local_valid],
            position_bias,
            local_position,
            head_dim,
        )
        kv_states.append(local_kv_state)
        score_states.append(local_score_state)
        predecessor_kv = local_kv_state
        predecessor_score = local_score_state

    return (
        torch.cat(pooled_outputs, dim=2),
        torch.cat(kv_states, dim=2),
        torch.cat(score_states, dim=2),
    )


def _make_inputs(sp_factor, local_seq_len, remainder, first_token_position, head_dim, empty_ranks=0):
    """``empty_ranks`` trailing SP ranks are left with no valid tokens at all, which is what a chunk
    shorter than its padded slab does to the tail of the mesh. Those ranks must still emit a state -- the
    exchange chains it along the axis, so the last rank's state is the one the next chunk starts from
    whether or not it saw a token."""
    torch.manual_seed(42)
    padded_seq_len = local_seq_len * sp_factor
    seq_len_actual = local_seq_len * (sp_factor - empty_ranks) - _COMPRESS_RATE + remainder
    kv = torch.randn(_BATCH, 1, padded_seq_len, 2 * head_dim, dtype=torch.bfloat16)
    gate = torch.randn_like(kv)
    position_bias = torch.randn(1, 1, _COMPRESS_RATE, 2 * head_dim, dtype=torch.bfloat16)
    initial_kv_state = torch.randn(_BATCH, 1, _STATE_ROWS, head_dim, dtype=torch.bfloat16)
    initial_score_state = torch.randn_like(initial_kv_state)
    expected = _torch_csa_compressor(
        kv,
        gate,
        position_bias,
        initial_kv_state,
        initial_score_state,
        sp_factor,
        seq_len_actual,
        first_token_position,
        head_dim,
    )
    return kv, gate, position_bias, initial_kv_state, initial_score_state, seq_len_actual, expected


def _run_csa_compressor(mesh_device, local_seq_len, remainder, first_token_position, head_dim, empty_ranks=0):
    mesh_shape = tuple(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape
    kv, gate, bias, initial_kv, initial_score, seq_len_actual, expected = _make_inputs(
        sp_factor, local_seq_len, remainder, first_token_position, head_dim, empty_ranks
    )
    sp_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(2, None))
    replicated_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(tensor, mapper):
        return ttnn.from_torch(
            tensor,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    pooled, kv_state, score_state = ttnn.experimental.deepseek_prefill.csa_compressor(
        to_device(kv, sp_mapper),
        to_device(gate, sp_mapper),
        to_device(bias, replicated_mapper),
        to_device(initial_kv.repeat(1, 1, sp_factor, 1), sp_mapper),
        to_device(initial_score.repeat(1, 1, sp_factor, 1), sp_mapper),
        seq_len_actual=seq_len_actual,
        first_token_position=first_token_position,
        cluster_axis=0,
        topology=ttnn.Topology.Linear,
    )

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(2, 1))
    actual = [ttnn.to_torch(tensor, mesh_composer=composer) for tensor in (pooled, kv_state, score_state)]
    expected_pooled, expected_kv, expected_score = expected
    for tp_rank in range(tp_factor):
        actual_pooled = actual[0][:, tp_rank : tp_rank + 1]
        passed, message = assert_with_pcc(expected_pooled.float(), actual_pooled.float(), pcc=_PCC)
        assert passed, f"CSA compressor PCC failed: {message}"
        assert torch.equal(actual[1][:, tp_rank : tp_rank + 1], expected_kv)
        assert torch.equal(actual[2][:, tp_rank : tp_rank + 1], expected_score)


@pytest.mark.parametrize("head_dim", [_HEAD_DIM, _INDEX_HEAD_DIM], ids=["head512", "head128"])
@pytest.mark.parametrize("first_token_position", [0, _COMPRESS_RATE])
@pytest.mark.parametrize("remainder", range(_COMPRESS_RATE))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 1),
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="mesh-1x1"),
            id="1x1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_csa_compressor_single_device(mesh_device, device_params, remainder, first_token_position, head_dim):
    _run_csa_compressor(mesh_device, _LOCAL_SEQ_LEN, remainder, first_token_position, head_dim)


@pytest.mark.parametrize("first_token_position", [0, _COMPRESS_RATE])
@pytest.mark.parametrize("remainder", range(_COMPRESS_RATE))
@pytest.mark.parametrize(
    "mesh_device, device_params, local_seq_len, head_dim",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            _LOCAL_SEQ_LEN,
            _HEAD_DIM,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric1d-2x2",
        ),
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            _LOCAL_SEQ_LEN,
            _INDEX_HEAD_DIM,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric1d-2x2-head128",
        ),
        pytest.param(
            (2, 2),
            fabric2d_device_params(),
            16,
            _HEAD_DIM,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric2d-2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_csa_compressor_mesh(mesh_device, device_params, local_seq_len, head_dim, remainder, first_token_position):
    _run_csa_compressor(mesh_device, local_seq_len, remainder, first_token_position, head_dim)


@pytest.mark.parametrize("first_token_position", [0, _COMPRESS_RATE])
@pytest.mark.parametrize("remainder", range(_COMPRESS_RATE))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric1d-2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_csa_compressor_mesh_empty_tail(mesh_device, device_params, remainder, first_token_position):
    """A chunk whose real length stops short of the padded slab leaves the trailing SP ranks with no
    valid tokens. Chunked prefill hits this whenever a non-final chunk is narrower than chunk_tokens, and
    the state those ranks emit is what the NEXT chunk starts from, so it has to be their predecessor's
    rather than the base state they were handed."""
    _run_csa_compressor(mesh_device, _LOCAL_SEQ_LEN, remainder, first_token_position, _HEAD_DIM, empty_ranks=1)


# Every (sp, tp) the V4 tests and the demo run at. tp < compress_rate is the interesting half: that is
# where the per-chip entry-tiling term binds, and where an 8x4-only reading of the alignment is wrong.
@pytest.mark.parametrize("sp_factor, tp_factor", [(1, 1), (2, 1), (2, 2), (4, 2), (2, 4), (8, 4)])
def test_csa_slab_align_tile_aligns_each_chip_share_of_the_entries(sp_factor, tp_factor):
    """A slab has to leave every chip a whole number of entry TILES, not just whole entries.

    The block-cyclic indexer score op is handed the local slab in tokens as block_cyclic_chunk_local,
    divides it by key_compression_ratio to get its per-shard chunk in compressed rows, and TT_FATALs
    unless that is tile-aligned. Missing this term costs nothing on an 8x4 mesh -- where the TP-gather
    term happens to be exactly equal -- and fails at the shortest legal prompt on 2x2 and 4x2."""
    align = csa_slab_align(_COMPRESS_RATE, sp_factor, tp_factor)

    entries_per_chip = align // sp_factor // _COMPRESS_RATE
    assert align % (sp_factor * _COMPRESS_RATE) == 0, (
        f"a {align}-token slab does not split into whole compression windows per chip "
        f"(sp={sp_factor}, rate={_COMPRESS_RATE})"
    )
    assert entries_per_chip > 0 and entries_per_chip % ttnn.TILE_SIZE == 0, (
        f"a {align}-token slab leaves each chip {entries_per_chip} compressed rows, which the "
        f"block-cyclic score op rejects as not tile-aligned (sp={sp_factor}, tp={tp_factor})"
    )
    assert align // sp_factor // tp_factor % ttnn.TILE_SIZE == 0, (
        f"a {align}-token slab leaves each chip a non-tile-aligned token share "
        f"(sp={sp_factor}, tp={tp_factor}), which the indexer's TP gathers need"
    )
