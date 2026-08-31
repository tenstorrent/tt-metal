# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Temporary PCC coverage for the Cb-only TtCSACompressor skeleton."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_hca import _MESH_CONFIGS, _MODEL_CONFIGS, _SEED, _config
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCSACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc

_SHAPES = [32, 34]
_PCC = 0.999


def _cb_only_golden(reference, hidden_states, compress_rate):
    """Match the temporary TT behavior; replace with the full CSA reference when overlap lands."""
    batch, seq_len, _ = hidden_states.shape
    head_dim = reference.head_dim
    usable = seq_len // compress_rate * compress_rate

    kv = reference.kv_proj(hidden_states[:, :usable])
    gate = reference.gate_proj(hidden_states[:, :usable])
    n_windows = usable // compress_rate
    kv = kv.view(batch, n_windows, compress_rate, 2 * head_dim)
    gate = gate.view(batch, n_windows, compress_rate, 2 * head_dim) + reference.position_bias

    kv_cb = kv[..., head_dim:]
    gate_cb = gate[..., head_dim:]
    weights = gate_cb.softmax(dim=2, dtype=torch.float32).to(kv_cb.dtype)
    compressed = reference.kv_norm((kv_cb * weights).sum(dim=2))

    positions = torch.arange(n_windows, device=compressed.device) * compress_rate
    positions = positions.unsqueeze(0).expand(batch, -1)
    cos, sin = reference.rotary_emb(compressed, position_ids=positions, layer_type="compress")
    return apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)


@pytest.mark.parametrize("seq_len", _SHAPES, ids=[f"seq{s}" for s in _SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
def test_csa_compressor_cb_only_mesh(mesh_device, device_params, topology, seq_len, model_config):
    torch.manual_seed(_SEED)

    config = _config(model_config)
    reference = DeepseekV4CSACompressor(config).eval()
    with torch.no_grad():
        reference.position_bias.normal_(0.0, 0.02)
        reference.kv_norm.weight.uniform_(0.5, 1.5)

    hidden = torch.randn(1, seq_len, config.hidden_size)
    compress_rate = config.compress_rates["compressed_sparse_attention"]
    with torch.no_grad():
        expected = _cb_only_golden(reference, hidden, compress_rate).unsqueeze(1)

    tt_model = TtCSACompressor.from_reference(
        mesh_device,
        reference,
        config,
        sp_axis=0,
        tp_axis=1,
        topology=topology,
    )
    hidden_padded, seq_len_actual = TtCSACompressor.prepare_input(hidden, mesh_device.shape[0], compress_rate)
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

    tt_model.alloc_tables(hidden_padded.shape[1], hidden_padded.shape[1])
    compressed_kv, block_bias = tt_model(tt_input, seq_len_actual=seq_len_actual)
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
    assert passed, f"CSA Cb-only compressor PCC failed: {message}"
