# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for TtV4Block: hyper-connection residual -> norm -> V4 attention -> norm -> MoE.

Random weights on both sides, and the reference is composed from the CPU modules
(reference.deepseek_v4.block) rather than run through a whole HF model. What this grades is
COMPOSITION: norm placement, the residual, the attention state across the chunk, and the sharding
contract at the attention/FFN seam.
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
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Block
from tests.ttnn.utils_for_testing import assert_with_pcc

# The checkpoint's compress_ratios, truncated to the layers the rows below use.
_COMPRESS_RATIOS = {
    DeepSeekV4ProConfig: (128, 128, 4, 128),
    DeepSeekV4FlashConfig: (0, 0, 4, 128),
}
# One row per (variant, floor, layer, attention, MoE) the device can build. No CSA row: layers 2, 4,
# 6 ... are CSA in both models and it has no device implementation yet.
_CASES = [
    pytest.param(DeepSeekV4ProConfig, 0.98, 0, "heavily_compressed_attention", "hash_moe", id="pro-L0-hca-hash"),
    # TODO this row does not reach 0.98: 0.9597 at full MoE width, reproducible to 0.0018 over seeds
    # 0, 42 and 1234, where the hash row next to it is unaffected at 0.9893. Moving the whole gate to
    # the host (HOST_ALL) does not recover it, so it is not where the top-k runs. Needs a debug pass,
    # and the next step is the teacher-forced chunked block on a real trace: the router sees random
    # activations here, and L3 is HCA + top-k there too, so that run says whether the error survives
    # real activations or is an artifact of this test's input.
    pytest.param(DeepSeekV4ProConfig, 0.98, 3, "heavily_compressed_attention", "moe", id="pro-L3-hca-topk"),
    pytest.param(DeepSeekV4FlashConfig, 0.988, 0, "sliding_attention", "hash_moe", id="flash-L0-swa-hash"),
    pytest.param(DeepSeekV4FlashConfig, 0.988, 3, "heavily_compressed_attention", "moe", id="flash-L3-hca-topk"),
]
_SEED = 42


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


def _test_config(model_config, layer_idx):
    """(hf_config, model_cfg) for the block test, narrowed consistently on both sides.

    ``compress_ratios`` is the variant's own, so ``layer_types[layer_idx]`` is the attention kind that
    layer has in the model; the router follows ``num_hash_layers``, 3 in both checkpoints.

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
        num_hidden_layers=layer_idx + 1,
        compress_ratios=list(_COMPRESS_RATIOS[model_config]),
        compress_rates=dict(m.COMPRESS_RATES),
        compress_rope_theta=m.COMPRESS_ROPE_THETA,
        rms_norm_eps=m.RMS_NORM_EPS,
        swiglu_limit=m.SWIGLU_LIMIT,
        n_routed_experts=m.NUM_ROUTED_EXPERTS,
        num_experts_per_tok=m.NUM_EXPERTS_PER_TOKEN,
        intermediate_size=m.MOE_INTERMEDIATE_SIZE,
        routed_scaling_factor=m.ROUTE_SCALE,
        vocab_size=m.VOCAB_SIZE,
    )
    cfg._attn_implementation = "eager"  # V4 is eager-only: the sdpa interface silently drops the sinks
    return cfg, m


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
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
# The residual streams entering a layer. "identical" is what the model itself feeds layer 0, straight
# out of mhc_expand()-ing the embedding -- faithful, but blind: with equal streams comb reduces to
# exactly the identity (its columns sum to 1, so sum_i comb[i,j]*X = X whatever its values) and pre
# only matters through its sum. 20 of the projection's 24 outputs are unobservable there. "distinct"
# is what every later layer sees, and it is the case that can actually see them.
@pytest.mark.parametrize("streams", ["identical", "distinct"], ids=["identical", "distinct"])
@pytest.mark.parametrize("model_config, block_pcc, layer_idx, attn_kind, mlp_kind", _CASES)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_block(
    mesh_device,
    device_params,
    num_links,
    seq_len,
    streams,
    model_config,
    block_pcc,
    layer_idx,
    attn_kind,
    mlp_kind,
):
    """One DeepSeek-V4 decoder block vs the composed CPU reference, unchunked."""
    topology = per_axis_topology(device_params["fabric_config"])
    config, model_cfg = _test_config(model_config, layer_idx)
    config.max_seq_len = seq_len
    # The row names the layer and its pair; this is where the config has to agree.
    assert (config.layer_types[layer_idx], config.mlp_layer_types[layer_idx]) == (attn_kind, mlp_kind), (
        f"case asks for layer {layer_idx} to be {attn_kind}/{mlp_kind}, the config gives "
        f"{config.layer_types[layer_idx]}/{config.mlp_layer_types[layer_idx]}"
    )
    # The gate family follows the layer's mlp type: a hash layer reads tid2eid, a moe layer runs the
    # single-group top-k. Crossing them routes to different experts with no error.
    gate_mode = GateComputeMode.HASH_DEVICE if mlp_kind == "hash_moe" else GateComputeMode.DEVICE_FP32
    logger.info(f"[v4 block] layer {layer_idx}: {attn_kind} / {mlp_kind} / {gate_mode.value}")

    tp_factor = mesh_device.shape[1]
    ms = tuple(mesh_device.shape)

    ref = build_v4_block_reference(config, layer_idx, seed=_SEED)
    hidden = torch.randn(1, seq_len, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    # V4's residual is hc_mult streams, fp32 (the mHC parametrization op is fp32-only).
    n = config.hc_mult
    if streams == "identical":
        ref_in = hidden.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
    else:
        ref_in = torch.randn(1, seq_len, n, config.hidden_size)

    out_ref = v4_block_forward(ref, config, ref_in, input_ids)

    block = TtV4Block(
        mesh_device=mesh_device,
        config=config,
        model_cfg=model_cfg,
        state_dict=v4_block_state_dict(ref, config),
        layer_idx=layer_idx,
        seq_len=seq_len,
        attn_reference=ref["attn"],
        mhc_weights=v4_mhc_weights(ref),
        num_links=num_links,
        topology=topology,
        sp_axis=0,
        tp_axis=1,
        gate_fallback_mode=gate_mode,
        weight_cache_path=None,  # random weights: nothing worth caching
    )

    tt_input = ttnn.from_torch(
        _pack_streams(ref_in, tp_factor),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),  # seq @ SP, hidden @ TP
    )

    tt_out = block(tt_input, actual_isl=seq_len, input_ids=input_ids)

    full = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3)))
    out = _unpack_streams(full, n, tp_factor)

    assert out.shape == out_ref.shape, f"shape mismatch: tt {tuple(out.shape)} vs ref {tuple(out_ref.shape)}"
    _, pcc_msg = assert_with_pcc(out_ref.to(torch.float32), out.to(torch.float32), block_pcc)
    logger.info(f"[v4 block L{layer_idx} {attn_kind} / {mlp_kind} / {streams}] PCC: {pcc_msg}")
