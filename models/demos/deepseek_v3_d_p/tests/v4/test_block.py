# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for TtV4Block: hyper-connection residual -> norm -> V4 attention -> norm -> MoE.

The reference is composed from the CPU modules (reference.deepseek_v4.block) rather than run through
a whole HF model: there is no V4 checkpoint, so the weights are random either way, and a whole-model
module would only add the MoE's runtime.

MoE width is narrowed from the shipped 384 x 3072. This test grades COMPOSITION -- norm placement,
the residual, the attention state across the chunk, the sharding contract at the attention/FFN seam
-- and each submodule's arithmetic has its own PCC test at full width. Full width is 50 GB of host
weights for no extra signal here.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.block import (
    build_v4_block_reference,
    v4_block_forward,
    v4_block_state_dict,
    v4_mhc_weights,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtHCACompressor
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Block
from tests.ttnn.utils_for_testing import assert_with_pcc

BLOCK_OUTPUT_PCC = 0.98
_SEED = 42
_TEST_EXPERTS = 64
_TEST_MOE_INTERMEDIATE = 512
_TEST_VOCAB = 4096


def _pack_streams(x, tp_factor):
    """[1, S, n, D] -> [1, 1, S, n*D], ordered for a TP-sharded mesh.

    The device packs the streams on the last dim, and TP shards that dim contiguously -- so a plain
    reshape would hand chip 0 the whole of stream 0 instead of its own D-slice of every stream. The
    columns are reordered chip-major-then-stream so the contiguous split lands right; inside a chip
    the order is stream-major again, which is what the hyper-connection slices.
    """
    b, seq, n, d = x.shape
    return x.reshape(b, seq, n, tp_factor, d // tp_factor).permute(0, 1, 3, 2, 4).reshape(b, 1, seq, n * d)


def _unpack_streams(t, n, tp_factor):
    """The inverse of _pack_streams: [1, 1, S, n*D] -> [1, S, n, D]."""
    b, _, seq, width = t.shape
    d = width // n
    return t.reshape(b, seq, tp_factor, n, d // tp_factor).permute(0, 1, 3, 2, 4).reshape(b, seq, n, d)


def _test_config(model_config, num_hidden_layers):
    """(hf_config, model_cfg) for the block test, narrowed consistently on both sides.

    q_lora_rank and o_groups are explicit because DeepseekV4Config's defaults are Flash's: a Pro run
    that left them out would build Pro widths with Flash's latent and grouping, and the reference
    would agree with it, so PCC would pass on the wrong model.
    """
    m = model_config
    cfg = DeepseekV4Config(
        hidden_size=m.EMB_SIZE,
        head_dim=m.HEAD_DIM,
        num_attention_heads=m.NUM_ATTENTION_HEADS,
        q_lora_rank=m.Q_LORA_RANK,
        o_groups=m.O_GROUPS,
        num_hidden_layers=num_hidden_layers,
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=m.COMPRESS_ROPE_THETA,
        rms_norm_eps=m.RMS_NORM_EPS,
        swiglu_limit=m.SWIGLU_LIMIT,
        n_routed_experts=_TEST_EXPERTS,
        num_experts_per_tok=m.NUM_EXPERTS_PER_TOKEN,
        intermediate_size=_TEST_MOE_INTERMEDIATE,
        routed_scaling_factor=m.ROUTE_SCALE,
        vocab_size=_TEST_VOCAB,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks

    class _Narrowed(m):
        NUM_ROUTED_EXPERTS = _TEST_EXPERTS
        MOE_INTERMEDIATE_SIZE = _TEST_MOE_INTERMEDIATE
        VOCAB_SIZE = _TEST_VOCAB

    return cfg, _Narrowed


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE,
                worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [2048], ids=["seq2048"])
# Layer 0 is hash_moe (tid2eid routing), layer 4 the plain top-k MoE -- both heavily_compressed
# attention. Two layers rather than one because the router is the only thing that differs between
# them, so a failure on one and not the other points straight at the gate.
@pytest.mark.parametrize("layer_idx", [0, 4], ids=["layer0_hash", "layer4_topk"])
# The residual streams entering a layer. "identical" is what the model itself feeds layer 0, straight
# out of mhc_expand()-ing the embedding -- faithful, but blind: with equal streams comb reduces to
# exactly the identity (its columns sum to 1, so sum_i comb[i,j]*X = X whatever its values) and pre
# only matters through its sum. 20 of the projection's 24 outputs are unobservable there. "distinct"
# is what every later layer sees, and it is the case that can actually see them.
@pytest.mark.parametrize("streams", ["identical", "distinct"], ids=["identical", "distinct"])
@pytest.mark.parametrize("model_config", [DeepSeekV4ProConfig], ids=["pro"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_block(mesh_device, device_params, num_links, seq_len, layer_idx, streams, model_config, tmp_path):
    """One DeepSeek-V4 decoder block vs the composed CPU reference, unchunked."""
    topology = per_axis_topology(device_params["fabric_config"])
    config, model_cfg = _test_config(model_config, num_hidden_layers=layer_idx + 1)
    config.max_seq_len = seq_len
    attn_kind = config.layer_types[layer_idx]
    mlp_kind = config.mlp_layer_types[layer_idx]
    # The gate family follows the layer's mlp type: a hash layer reads tid2eid, a moe layer runs the
    # single-group top-k. Crossing them routes to different experts with no error.
    gate_mode = GateComputeMode.HASH_DEVICE if mlp_kind == "hash_moe" else GateComputeMode.DEVICE_FP32
    logger.info(f"[v4 block] layer {layer_idx}: {attn_kind} / {mlp_kind} / {gate_mode.value}")

    sp_factor, tp_factor = mesh_device.shape[0], mesh_device.shape[1]
    ms = tuple(mesh_device.shape)

    ref = build_v4_block_reference(config, layer_idx, seed=_SEED)
    hidden = torch.randn(1, seq_len, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    # Pad to the attention's granularity, the same rule alloc_state asserts.
    if attn_kind == "sliding_attention":
        align = config.sliding_window * sp_factor
        padded_len = -(-seq_len // align) * align
        hidden_padded = torch.zeros(1, padded_len, config.hidden_size, dtype=hidden.dtype)
        hidden_padded[:, :seq_len] = hidden
        seq_len_actual = seq_len
    else:
        compress_rate = config.compress_rates[attn_kind]
        hidden_padded, seq_len_actual = TtHCACompressor.prepare_input(hidden, sp_factor, compress_rate)
    padded_len = hidden_padded.shape[1]
    logger.info(f"[v4 block] S_real={seq_len_actual} S_pad={padded_len} mesh={ms}")

    # V4's residual is hc_mult streams, fp32 (the mHC parametrization op is fp32-only).
    n = config.hc_mult
    if streams == "identical":
        ref_pad = hidden_padded.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
    else:
        ref_pad = torch.randn(1, padded_len, n, config.hidden_size)
        ref_pad[:, seq_len_actual:] = 0  # the padded tail no real query reads
    ref_in = ref_pad[:, :seq_len_actual]

    out_ref = v4_block_forward(ref, config, ref_in, input_ids)

    block = TtV4Block(
        mesh_device=mesh_device,
        config=config,
        model_cfg=model_cfg,
        state_dict=v4_block_state_dict(ref, config),
        layer_idx=layer_idx,
        seq_len=padded_len,
        attn_reference=ref["attn"],
        mhc_weights=v4_mhc_weights(ref),
        num_links=num_links,
        topology=topology,
        sp_axis=0,
        tp_axis=1,
        gate_fallback_mode=gate_mode,
        weight_cache_path=tmp_path,
    )

    tt_input = ttnn.from_torch(
        _pack_streams(ref_pad, tp_factor),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),  # seq @ SP, hidden @ TP
    )

    tt_out = block(tt_input, actual_isl=seq_len_actual, input_ids=input_ids)

    full = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3)))
    out = _unpack_streams(full, n, tp_factor)[:, :seq_len_actual]

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"
    _, pcc_msg = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), BLOCK_OUTPUT_PCC)
    logger.info(f"[v4 block L{layer_idx} {attn_kind} / {mlp_kind} / {streams}] PCC: {pcc_msg}")
