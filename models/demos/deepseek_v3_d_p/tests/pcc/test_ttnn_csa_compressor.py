# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC coverage for the overlap-aware TtCSACompressor.

Scoped deliberately narrow. What is unique here is the RMSNorm + indexed-RoPE wrapper the compressor
puts around ``ttnn.experimental.deepseek_prefill.csa_compressor`` -- the op's own pooling and state
output are checked bit-exactly against a torch reference in
``tests/op_unit_tests/test_csa_compressor.py``. Here the compressed entries are checked against the
reference ``DeepseekV4CSACompressor``; the outgoing state has no reference counterpart, so it is
checked against the op's torch model.

That wrapper does not vary with the prompt length or the model variant, so this runs one shape on
flash only, once per mesh, both aligned and ragged. Aligned is what prefill produces; ragged leaves a
partial compression window, so the trim to ``valid_entries`` has something to trim.

The length is given per chip and scaled by the mesh's SP factor, so every mesh runs the same local
shape and a profile taken on one box describes the others."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACompressor
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_csa_compressor import _torch_csa_compressor
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import V4_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tests.pcc.v4_test_utils import V4_SEED, v4_reference_config
from models.demos.deepseek_v3_d_p.tt.mla.compressor import CSA_STATE_ROWS, TtCSACompressor
from tests.ttnn.utils_for_testing import assert_with_pcc

# PER-CHIP padded prompt length, not global.
_LOCAL_SHAPES = [640]
# How far short of a whole slab the real prompt stops. prepare_input pads it straight back up, so the
# per-chip shape is exactly _LOCAL_SHAPES either way; a nonzero tail leaves the last rank mid-window.
_TAILS = [0, 2]
_TAIL_IDS = ["aligned", "ragged"]
_PCC = 0.999


def _golden_state(
    reference, hidden_states, seq_len_actual, compress_rate, sp_factor, initial_kv, initial_score, head_dim
):
    kv = reference.kv_proj(hidden_states).unsqueeze(1).to(torch.bfloat16)
    gate = reference.gate_proj(hidden_states).unsqueeze(1).to(torch.bfloat16)
    position_bias = reference.position_bias.reshape(1, 1, compress_rate, -1).to(torch.bfloat16)
    _, kv_state, score_state = _torch_csa_compressor(
        kv,
        gate,
        position_bias,
        initial_kv,
        initial_score,
        sp_factor,
        seq_len_actual,
        0,
        head_dim,
    )
    return kv_state, score_state


@pytest.mark.parametrize("tail", _TAILS, ids=_TAIL_IDS)
@pytest.mark.parametrize("local_seq_len", _LOCAL_SHAPES, ids=[f"local{s}" for s in _LOCAL_SHAPES])
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    V4_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_csa_compressor_mesh(mesh_device, device_params, topology, local_seq_len, tail):
    torch.manual_seed(V4_SEED)

    config = v4_reference_config(DeepSeekV4FlashConfig)
    reference = DeepseekV4CSACompressor(config).eval()
    with torch.no_grad():
        reference.position_bias.normal_(0.0, 0.02)
        reference.kv_norm.weight.uniform_(0.5, 1.5)
        reference.indexer.position_bias.normal_(0.0, 0.02)

    seq_len = local_seq_len * mesh_device.shape[0] - tail
    hidden = torch.randn(1, seq_len, config.hidden_size)
    compress_rate = config.compress_rates["compressed_sparse_attention"]
    hidden_padded, seq_len_actual = TtCSACompressor.prepare_input(hidden, mesh_device.shape[0], compress_rate)
    head_dim = config.head_dim
    initial_kv = torch.zeros(1, 1, CSA_STATE_ROWS, head_dim, dtype=torch.bfloat16)
    initial_score = torch.full_like(initial_kv, float("-inf"))
    with torch.no_grad():
        # Unpadded prompt: the reference drops the partial window itself, leaving seq_len_actual // m entries.
        expected, _ = reference(
            hidden,
            torch.zeros(1, seq_len_actual, config.q_lora_rank),
            torch.arange(seq_len_actual).unsqueeze(0),
            past_key_values=None,
            layer_idx=0,
        )
        expected_kv_state, expected_score_state = _golden_state(
            reference,
            hidden_padded,
            seq_len_actual,
            compress_rate,
            mesh_device.shape[0],
            initial_kv,
            initial_score,
            head_dim,
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
    compressed_kv, kv_state, score_state = tt_model(
        tt_input,
        tt_initial_kv,
        tt_initial_score,
        seq_len_actual=seq_len_actual,
    )

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
