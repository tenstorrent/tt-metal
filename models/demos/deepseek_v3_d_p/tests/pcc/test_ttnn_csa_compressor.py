# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC coverage for the overlap-aware TtCSACompressor."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_csa_compressor import _torch_csa_compressor
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_hca import _MESH_CONFIGS, _MODEL_CONFIGS, _SEED, _config
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCSACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc

_SHAPES = [32, 34]
_PCC = 0.999


def _golden(reference, hidden_states, seq_len_actual, compress_rate, sp_factor, initial_kv, initial_score):
    batch, seq_len, _ = hidden_states.shape
    kv = reference.kv_proj(hidden_states).unsqueeze(1).to(torch.bfloat16)
    gate = reference.gate_proj(hidden_states).unsqueeze(1).to(torch.bfloat16)
    position_bias = reference.position_bias.reshape(1, 1, compress_rate, -1).to(torch.bfloat16)
    pooled, kv_state, score_state = _torch_csa_compressor(
        kv,
        gate,
        position_bias,
        initial_kv,
        initial_score,
        sp_factor,
        seq_len_actual,
        0,
    )
    valid_entries = seq_len_actual // compress_rate
    compressed = reference.kv_norm(pooled[:, 0, :valid_entries].to(hidden_states.dtype))
    positions = torch.arange(valid_entries, device=compressed.device) * compress_rate
    positions = positions.unsqueeze(0).expand(batch, -1)
    cos, sin = reference.rotary_emb(compressed, position_ids=positions, layer_type="compress")
    compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin)
    return compressed, kv_state, score_state


@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
def test_csa_compressor_mesh(mesh_device, device_params, topology, seq_len, model_config):
    torch.manual_seed(_SEED)

    config = _config(model_config)
    reference = DeepseekV4CSACompressor(config).eval()
    with torch.no_grad():
        reference.position_bias.normal_(0.0, 0.02)
        reference.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(1, seq_len, config.hidden_size)
    compress_rate = config.compress_rates["compressed_sparse_attention"]
    hidden_padded, seq_len_actual = TtCSACompressor.prepare_input(hidden, mesh_device.shape[0], compress_rate)
    head_dim = config.head_dim
    initial_kv = torch.zeros(1, 1, 64, head_dim, dtype=torch.bfloat16)
    initial_score = torch.full_like(initial_kv, float("-inf"))
    with torch.no_grad():
        expected, expected_kv_state, expected_score_state = _golden(
            reference,
            hidden_padded,
            seq_len_actual,
            compress_rate,
            mesh_device.shape[0],
            initial_kv,
            initial_score,
        )

    tt_model = TtCSACompressor.from_reference(
        mesh_device,
        reference,
        config,
        sp_axis=0,
        tp_axis=1,
        topology=topology,
    )
    tt_input = ttnn.from_torch(
        hidden_padded.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=(2, 3),
        ),
    )
    state_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=tuple(mesh_device.shape),
        dims=(2, None),
    )
    repeated_initial_kv = initial_kv.repeat(1, 1, mesh_device.shape[0], 1)
    repeated_initial_score = initial_score.repeat(1, 1, mesh_device.shape[0], 1)
    tt_initial_kv = ttnn.from_torch(
        repeated_initial_kv,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=state_mapper,
    )
    tt_initial_score = ttnn.from_torch(
        repeated_initial_score,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=state_mapper,
    )

    tt_model.alloc_tables(hidden_padded.shape[1], hidden_padded.shape[1])
    compressed_kv, block_bias, kv_state, score_state = tt_model(
        tt_input,
        tt_initial_kv,
        tt_initial_score,
        seq_len_actual=seq_len_actual,
    )
    assert block_bias is None

    actual = ttnn.to_torch(
        compressed_kv,
        mesh_composer=ttnn.create_mesh_composer(
            mesh_device,
            ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1)),
        ),
    )
    valid_entries = seq_len_actual // compress_rate
    assert actual.shape[2] == hidden_padded.shape[1] // compress_rate
    actual = actual[:, :, :valid_entries]

    assert actual.shape == expected.shape
    passed, message = assert_with_pcc(expected.float(), actual.float(), pcc=_PCC)
    assert passed, f"CSA compressor PCC failed: {message}"

    state_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device,
        mesh_shape=tuple(mesh_device.shape),
        dims=(2, 1),
    )
    actual_kv_state = ttnn.to_torch(kv_state, mesh_composer=state_composer)[:, :1]
    actual_score_state = ttnn.to_torch(score_state, mesh_composer=state_composer)[:, :1]
    assert actual_kv_state.shape == expected_kv_state.shape
    assert actual_score_state.shape == expected_score_state.shape
    passed, message = assert_with_pcc(expected_kv_state.float(), actual_kv_state.float(), pcc=_PCC)
    assert passed, f"CSA KV state PCC failed: {message}"
    finite = torch.isfinite(expected_score_state)
    assert torch.equal(torch.isfinite(actual_score_state), finite)
    passed, message = assert_with_pcc(
        expected_score_state[finite].float(),
        actual_score_state[finite].float(),
        pcc=_PCC,
    )
    assert passed, f"CSA score state PCC failed: {message}"
