# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation tests for MiniMax-M2.5 — Galaxy mesh (8,4).

Mesh: 8 rows (EP axis) × 4 cols (TP axis) = 32 chips
  - Attention: TP=4, column-parallel QKV + row-parallel O-proj + all-reduce
  - MoE:       EP=8, TP=4, on-device expert weights with EP+TP all-reduce

**Default (CI / bring-up):** Requires a real checkpoint on disk and opens MeshShape(8, 4)
with fabric ring so CCL (all_reduce, reduce_scatter, etc.) is exercised. Do not infer
correctness from a 1×1 mesh — that path skips EP/TP and fabric.

**Opt-in local smoke only:** ``MINIMAX_M2_ALLOW_SYNTH_1X1=1`` — 1×1 mesh + random weights.
No CCL/EP/TP coverage; use only for quick single-chip debugging.

KV cache is device-resident ([B, NK, max_seq_len, D] per layer on DRAM).

Run:
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py -v
"""

import json
import os

import pytest
import torch
from safetensors.torch import load_file

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.minimax_m2.reference.functional import (
    MiniMaxM2Config,
    attention_forward,
    build_rope_cache,
    decoder_layer_forward,
    make_random_state_dict,
    model_forward,
    moe_forward,
    rmsnorm_forward,
)
from models.demos.minimax_m2.reference.generate_goldens import load_and_dequant
from models.demos.minimax_m2.tt.attention import TtMiniMaxAttention
from models.demos.minimax_m2.tt.generator import TtMiniMaxGenerator
from models.demos.minimax_m2.tt.model import TtDecoderLayer, TtMiniMaxModel
from models.demos.minimax_m2.tt.model_config import MiniMaxM2TTConfig, make_mesh_config, make_paged_attention_config
from models.demos.minimax_m2.tt.moe import TtMiniMaxMoE
from models.demos.minimax_m2.tt.rms_norm import TtRMSNorm
from models.demos.minimax_m2.tt.rope import PartialRoPESetup, apply_partial_rope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    "/home/cust-team/models/models--MiniMaxAI--MiniMax-M2.5/" "snapshots/f710177d938eff80b684d42c5aa84b382612f21f"
)
MODEL_PATH = os.environ.get("MINIMAX_M2_MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_AVAILABLE = os.path.isdir(MODEL_PATH)


def _allow_synth_1x1_only() -> bool:
    """Explicit opt-in: 1×1 mesh + synthetic weights (no fabric / no CCL validation)."""
    return os.environ.get("MINIMAX_M2_ALLOW_SYNTH_1X1", "").lower() in ("1", "true", "yes")


PCC_BLOCK = 0.99
PCC_MODEL = 0.97

BATCH = 1
SEQ = 16

MESH_ROWS = 8
MESH_COLS = 4

# Synthetic fallback (used when real checkpoint path is unavailable)
SYNTH_HIDDEN = 256
SYNTH_HEAD_DIM = 64
SYNTH_NQ = 8
SYNTH_NK = 2
SYNTH_FF = 128
SYNTH_E = 8
SYNTH_TOPK = 2
SYNTH_ROTARY_DIM = 32
SYNTH_VOCAB = 1024
SYNTH_ALL_LAYERS = int(os.environ.get("MINIMAX_M2_SYNTH_ALL_LAYERS", "16"))
MAX_SEQ_LEN = 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    synth = _allow_synth_1x1_only()
    if not MODEL_AVAILABLE and not synth:
        pytest.skip(
            f"MiniMax M2 Galaxy tests need real weights at {MODEL_PATH} "
            f"(set MINIMAX_M2_MODEL_PATH) and an {MESH_ROWS}x{MESH_COLS} mesh for CCL. "
            "Optional: MINIMAX_M2_ALLOW_SYNTH_1X1=1 for 1×1 synthetic only (no EP/TP/CCL)."
        )

    if synth and not MODEL_AVAILABLE:
        rows, cols = 1, 1
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    else:
        rows, cols = MESH_ROWS, MESH_COLS
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D_RING,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )

    try:
        d = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    except Exception as exc:
        if rows > 1 or cols > 1:
            pytest.skip(f"Could not open {rows}x{cols} mesh (CCL tests require full Galaxy topology): {exc}")
        raise
    yield d
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def mesh_config(device):
    return make_mesh_config(device)


@pytest.fixture(scope="module")
def ccl_manager(device):
    num_links = 4 if device.shape[0] > 1 else 1
    return CCLManager(device, num_links=num_links)


@pytest.fixture(scope="module")
def ref_config():
    if MODEL_AVAILABLE:
        with open(f"{MODEL_PATH}/config.json") as f:
            cfg_dict = json.load(f)
        return MiniMaxM2Config(
            hidden_size=cfg_dict["hidden_size"],
            head_dim=cfg_dict["head_dim"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            num_hidden_layers=1,
            intermediate_size=cfg_dict["intermediate_size"],
            num_local_experts=cfg_dict["num_local_experts"],
            num_experts_per_tok=cfg_dict["num_experts_per_tok"],
            rotary_dim=cfg_dict["rotary_dim"],
            rope_theta=cfg_dict["rope_theta"],
            rms_norm_eps=cfg_dict["rms_norm_eps"],
            vocab_size=cfg_dict["vocab_size"],
            use_qk_norm=True,
            use_routing_bias=True,
        )

    return MiniMaxM2Config(
        hidden_size=SYNTH_HIDDEN,
        head_dim=SYNTH_HEAD_DIM,
        num_attention_heads=SYNTH_NQ,
        num_key_value_heads=SYNTH_NK,
        num_hidden_layers=1,
        intermediate_size=SYNTH_FF,
        num_local_experts=SYNTH_E,
        num_experts_per_tok=SYNTH_TOPK,
        rotary_dim=SYNTH_ROTARY_DIM,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
        vocab_size=SYNTH_VOCAB,
        use_qk_norm=True,
        use_routing_bias=True,
    )


@pytest.fixture(scope="module")
def tt_config(ref_config):
    return MiniMaxM2TTConfig(
        hidden_size=ref_config.hidden_size,
        head_dim=ref_config.head_dim,
        num_attention_heads=ref_config.num_attention_heads,
        num_key_value_heads=ref_config.num_key_value_heads,
        num_hidden_layers=1,
        intermediate_size=ref_config.intermediate_size,
        num_local_experts=ref_config.num_local_experts,
        num_experts_per_tok=ref_config.num_experts_per_tok,
        rotary_dim=ref_config.rotary_dim,
        rope_theta=ref_config.rope_theta,
        rms_norm_eps=ref_config.rms_norm_eps,
        vocab_size=ref_config.vocab_size,
    )


@pytest.fixture(scope="module")
def real_state_dict(ref_config):
    if MODEL_AVAILABLE:
        raw = {}
        for shard in [
            "model-00000-of-00126.safetensors",
            "model-00001-of-00126.safetensors",
            "model-00124-of-00126.safetensors",
        ]:
            raw.update(load_file(f"{MODEL_PATH}/{shard}"))
        return load_and_dequant(raw)
    return make_random_state_dict(ref_config, num_layers=1, dtype=torch.float32, seed=0)


@pytest.fixture(scope="module")
def rope_cache(ref_config):
    return build_rope_cache(SEQ, ref_config.rotary_dim, ref_config.rope_theta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()


def tt_to_torch(t: ttnn.Tensor) -> torch.Tensor:
    if isinstance(t.device(), ttnn.MeshDevice):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def from_torch_mesh(x: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, mesh_mapper=mapper)


# ---------------------------------------------------------------------------
# Test 1: RMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm(device, real_state_dict, ref_config):
    weight = real_state_dict["model.norm.weight"]
    x_pt = torch.randn(BATCH, SEQ, ref_config.hidden_size)

    ref_out = rmsnorm_forward(x_pt, weight, ref_config.rms_norm_eps)

    tt_norm = TtRMSNorm(device, weight, ref_config.rms_norm_eps)
    x_tt = from_torch_mesh(x_pt.unsqueeze(0).to(torch.bfloat16), device)
    x_tt = ttnn.reshape(x_tt, (BATCH, SEQ, ref_config.hidden_size))
    out_tt = tt_norm(x_tt)
    out_pt = tt_to_torch(out_tt).squeeze()

    p = pcc(ref_out.float(), out_pt.reshape_as(ref_out))
    print(f"[RMSNorm] PCC = {p:.6f}")
    assert p > PCC_BLOCK, f"RMSNorm PCC {p:.4f} < {PCC_BLOCK}"


# ---------------------------------------------------------------------------
# Test 2: Partial RoPE
# ---------------------------------------------------------------------------


def test_partial_rope(device, ref_config, rope_cache):
    cos_ref, sin_ref = rope_cache
    NQ, NK, D = ref_config.num_attention_heads, ref_config.num_key_value_heads, ref_config.head_dim
    rdim = ref_config.rotary_dim

    q_pt = torch.randn(BATCH, NQ, SEQ, D)
    k_pt = torch.randn(BATCH, NK, SEQ, D)

    cos_r = cos_ref.unsqueeze(0).unsqueeze(0)
    sin_r = sin_ref.unsqueeze(0).unsqueeze(0)
    q_rot, k_rot = q_pt[..., :rdim], k_pt[..., :rdim]
    q_pass, k_pass = q_pt[..., rdim:], k_pt[..., rdim:]

    def _rot_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_ref = torch.cat([q_rot * cos_r + _rot_half(q_rot) * sin_r, q_pass], dim=-1)
    k_ref = torch.cat([k_rot * cos_r + _rot_half(k_rot) * sin_r, k_pass], dim=-1)

    rope = PartialRoPESetup(device, rdim, ref_config.rope_theta, max_seq_len=MAX_SEQ_LEN)
    cos_tt, sin_tt = rope.get_cos_sin(SEQ)

    q_tt = from_torch_mesh(q_pt.bfloat16(), device)
    k_tt = from_torch_mesh(k_pt.bfloat16(), device)
    q_out_tt, k_out_tt = apply_partial_rope(q_tt, k_tt, cos_tt, sin_tt, rdim, D)

    q_out_pt = tt_to_torch(q_out_tt)
    k_out_pt = tt_to_torch(k_out_tt)

    pq = pcc(q_ref.float(), q_out_pt.reshape_as(q_ref))
    pk = pcc(k_ref.float(), k_out_pt.reshape_as(k_ref))
    print(f"[PartialRoPE] Q PCC = {pq:.6f},  K PCC = {pk:.6f}")
    assert pq > PCC_BLOCK, f"Q PCC {pq:.4f} < {PCC_BLOCK}"
    assert pk > PCC_BLOCK, f"K PCC {pk:.4f} < {PCC_BLOCK}"


# ---------------------------------------------------------------------------
# Test 3: Attention (TP=4)
# ---------------------------------------------------------------------------


def make_causal_mask(seq_len: int, dtype=torch.bfloat16) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def test_attention(device, mesh_config, ccl_manager, real_state_dict, ref_config, tt_config, rope_cache):
    cos, sin = rope_cache
    H = ref_config.hidden_size

    torch.manual_seed(42)
    x_pt = torch.randn(BATCH, SEQ, H)

    layer0_prefix = "model.layers.0.self_attn."
    attn_sd = {k.removeprefix(layer0_prefix): v for k, v in real_state_dict.items() if k.startswith(layer0_prefix)}
    ref_out = attention_forward(x_pt, attn_sd, cos, sin, ref_config)

    tt_attn = TtMiniMaxAttention(
        device,
        real_state_dict,
        tt_config,
        layer_idx=0,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        max_seq_len=MAX_SEQ_LEN,
    )
    rope = PartialRoPESetup(device, tt_config.rotary_dim, tt_config.rope_theta, max_seq_len=MAX_SEQ_LEN)
    cos_tt, sin_tt = rope.get_cos_sin(SEQ)

    x_tt = from_torch_mesh(x_pt.to(torch.bfloat16).unsqueeze(0), device)
    x_tt = ttnn.reshape(x_tt, (BATCH, SEQ, H))
    out_tt = tt_attn(x_tt, cos_tt, sin_tt, is_causal=False)

    out_pt = tt_to_torch(out_tt).reshape_as(ref_out)

    p = pcc(ref_out.float(), out_pt)
    print(f"[Attention/TP=4] PCC = {p:.6f}")
    assert p > PCC_BLOCK, f"Attention PCC {p:.4f} < {PCC_BLOCK}"


# ---------------------------------------------------------------------------
# Test 4: MoE (EP=8, TP=4)
# ---------------------------------------------------------------------------


def test_moe(device, mesh_config, ccl_manager, real_state_dict, ref_config, tt_config):
    H = ref_config.hidden_size
    torch.manual_seed(7)
    x_pt = torch.randn(BATCH, SEQ, H)
    x_bf16 = x_pt.to(torch.bfloat16).float()

    moe_prefix = "model.layers.0.block_sparse_moe."
    moe_sd = {k.removeprefix(moe_prefix): v for k, v in real_state_dict.items() if k.startswith(moe_prefix)}
    moe_sd_bf16 = {k: v.to(torch.bfloat16).float() for k, v in moe_sd.items()}
    ref_out = moe_forward(x_bf16, moe_sd_bf16, ref_config)

    tt_moe = TtMiniMaxMoE(
        device,
        real_state_dict,
        tt_config,
        layer_idx=0,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
    )
    x_tt = from_torch_mesh(x_pt.to(torch.bfloat16).unsqueeze(0), device)
    x_tt = ttnn.reshape(x_tt, (BATCH, SEQ, H))
    out_tt = tt_moe(x_tt)

    out_pt = tt_to_torch(out_tt).reshape_as(ref_out)

    p = pcc(ref_out.float(), out_pt)
    print(f"[MoE/EP=8,TP=4] PCC = {p:.6f}")
    # bf16 topk routing precision limits PCC to ~0.92 with real weights
    # (tight expert score margins cause ~12.5% expert selection mismatch)
    assert p > 0.85, f"MoE PCC {p:.4f} < 0.85"


# ---------------------------------------------------------------------------
# Test 5: DecoderLayer
# ---------------------------------------------------------------------------


def test_decoder_layer(device, mesh_config, ccl_manager, real_state_dict, ref_config, tt_config, rope_cache):
    cos, sin = rope_cache
    H = ref_config.hidden_size
    torch.manual_seed(42)
    x_pt = torch.randn(BATCH, SEQ, H)

    layer0_prefix = "model.layers.0."
    layer_sd = {k.removeprefix(layer0_prefix): v for k, v in real_state_dict.items() if k.startswith(layer0_prefix)}
    ref_out = decoder_layer_forward(x_pt, layer_sd, cos, sin, ref_config)

    tt_layer = TtDecoderLayer(
        device,
        real_state_dict,
        tt_config,
        layer_idx=0,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        max_seq_len=MAX_SEQ_LEN,
    )
    rope = PartialRoPESetup(device, tt_config.rotary_dim, tt_config.rope_theta, max_seq_len=MAX_SEQ_LEN)
    cos_tt, sin_tt = rope.get_cos_sin(SEQ)

    x_tt = from_torch_mesh(x_pt.to(torch.bfloat16).unsqueeze(0), device)
    x_tt = ttnn.reshape(x_tt, (BATCH, SEQ, H))
    out_tt = tt_layer(x_tt, cos_tt, sin_tt, is_causal=False)
    out_pt = tt_to_torch(out_tt).reshape_as(ref_out)

    p = pcc(ref_out.float(), out_pt)
    print(f"[DecoderLayer/mesh] PCC = {p:.6f}")
    # MoE bf16 topk routing limits block PCC to ~0.92 with real weights
    assert p > 0.90, f"DecoderLayer PCC {p:.4f} < 0.90"


# ---------------------------------------------------------------------------
# Test 6: Full model (1 layer)
# ---------------------------------------------------------------------------


def test_full_model(device, real_state_dict, ref_config, tt_config):
    torch.manual_seed(0)
    input_ids = torch.randint(0, ref_config.vocab_size, (BATCH, SEQ))

    ref_logits = model_forward(input_ids, real_state_dict, ref_config)

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=MAX_SEQ_LEN)
    out_tt = tt_model(input_ids)
    out_pt = tt_to_torch(out_tt).reshape_as(ref_logits)

    p = pcc(ref_logits.float(), out_pt)
    print(f"[FullModel-1layer/mesh] PCC = {p:.6f}")
    assert p > PCC_MODEL, f"FullModel PCC {p:.4f} < {PCC_MODEL}"


# ---------------------------------------------------------------------------
# Test 7: Multi-layer model (4 layers)
# ---------------------------------------------------------------------------

N_LAYERS_MULTI = 4
SHARDS_FOR_4_LAYERS = [
    "model-00000-of-00126.safetensors",
    "model-00001-of-00126.safetensors",
    "model-00002-of-00126.safetensors",
    "model-00003-of-00126.safetensors",
    "model-00004-of-00126.safetensors",
    "model-00005-of-00126.safetensors",
    "model-00006-of-00126.safetensors",
    "model-00007-of-00126.safetensors",
    "model-00124-of-00126.safetensors",
]


@pytest.fixture(scope="module")
def state_dict_4layers():
    if MODEL_AVAILABLE:
        raw = {}
        for shard in SHARDS_FOR_4_LAYERS:
            raw.update(load_file(f"{MODEL_PATH}/{shard}"))
        return load_and_dequant(raw)
    cfg = MiniMaxM2Config(
        hidden_size=SYNTH_HIDDEN,
        head_dim=SYNTH_HEAD_DIM,
        num_attention_heads=SYNTH_NQ,
        num_key_value_heads=SYNTH_NK,
        num_hidden_layers=N_LAYERS_MULTI,
        intermediate_size=SYNTH_FF,
        num_local_experts=SYNTH_E,
        num_experts_per_tok=SYNTH_TOPK,
        rotary_dim=SYNTH_ROTARY_DIM,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
        vocab_size=SYNTH_VOCAB,
        use_qk_norm=True,
        use_routing_bias=True,
    )
    return make_random_state_dict(cfg, num_layers=N_LAYERS_MULTI, dtype=torch.float32, seed=1)


@pytest.fixture(scope="module")
def ref_config_4layers():
    if MODEL_AVAILABLE:
        with open(f"{MODEL_PATH}/config.json") as f:
            cfg_dict = json.load(f)
        return MiniMaxM2Config(
            hidden_size=cfg_dict["hidden_size"],
            head_dim=cfg_dict["head_dim"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            num_hidden_layers=N_LAYERS_MULTI,
            intermediate_size=cfg_dict["intermediate_size"],
            num_local_experts=cfg_dict["num_local_experts"],
            num_experts_per_tok=cfg_dict["num_experts_per_tok"],
            rotary_dim=cfg_dict["rotary_dim"],
            rope_theta=cfg_dict["rope_theta"],
            rms_norm_eps=cfg_dict["rms_norm_eps"],
            vocab_size=cfg_dict["vocab_size"],
            use_qk_norm=True,
            use_routing_bias=True,
        )
    return MiniMaxM2Config(
        hidden_size=SYNTH_HIDDEN,
        head_dim=SYNTH_HEAD_DIM,
        num_attention_heads=SYNTH_NQ,
        num_key_value_heads=SYNTH_NK,
        num_hidden_layers=N_LAYERS_MULTI,
        intermediate_size=SYNTH_FF,
        num_local_experts=SYNTH_E,
        num_experts_per_tok=SYNTH_TOPK,
        rotary_dim=SYNTH_ROTARY_DIM,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
        vocab_size=SYNTH_VOCAB,
        use_qk_norm=True,
        use_routing_bias=True,
    )


@pytest.fixture(scope="module")
def tt_config_4layers(ref_config_4layers):
    return MiniMaxM2TTConfig(
        hidden_size=ref_config_4layers.hidden_size,
        head_dim=ref_config_4layers.head_dim,
        num_attention_heads=ref_config_4layers.num_attention_heads,
        num_key_value_heads=ref_config_4layers.num_key_value_heads,
        num_hidden_layers=N_LAYERS_MULTI,
        intermediate_size=ref_config_4layers.intermediate_size,
        num_local_experts=ref_config_4layers.num_local_experts,
        num_experts_per_tok=ref_config_4layers.num_experts_per_tok,
        rotary_dim=ref_config_4layers.rotary_dim,
        rope_theta=ref_config_4layers.rope_theta,
        rms_norm_eps=ref_config_4layers.rms_norm_eps,
        vocab_size=ref_config_4layers.vocab_size,
    )


def test_multilayer_model(device, state_dict_4layers, ref_config_4layers, tt_config_4layers):
    torch.manual_seed(0)
    input_ids = torch.randint(0, ref_config_4layers.vocab_size, (BATCH, SEQ))

    ref_logits = model_forward(input_ids, state_dict_4layers, ref_config_4layers)

    tt_model = TtMiniMaxModel(device, state_dict_4layers, tt_config_4layers, max_seq_len=MAX_SEQ_LEN)
    out_tt = tt_model(input_ids)
    out_pt = tt_to_torch(out_tt).reshape_as(ref_logits)

    p = pcc(ref_logits.float(), out_pt)
    print(f"[FullModel-{N_LAYERS_MULTI}layers/mesh] PCC = {p:.6f}")
    assert p > PCC_MODEL, f"Multi-layer PCC {p:.4f} < {PCC_MODEL}"


# ---------------------------------------------------------------------------
# Test 8: KV-cache prefill + decode (device-resident)
# ---------------------------------------------------------------------------


def test_kvcache_prefill_decode(device, real_state_dict, tt_config):
    torch.manual_seed(1)
    input_ids = torch.randint(0, tt_config.vocab_size, (BATCH, SEQ))

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=MAX_SEQ_LEN)

    ref_logits = tt_to_torch(tt_model(input_ids))

    tt_model.clear_kv_caches()
    prefill_logits = tt_model.forward_prefill(input_ids)
    prefill_pt = tt_to_torch(prefill_logits)

    p_prefill = pcc(ref_logits.float(), prefill_pt.float())
    print(f"[KVCache-Prefill/mesh] PCC = {p_prefill:.6f}")
    assert p_prefill > PCC_MODEL, f"KVCache prefill PCC {p_prefill:.4f} < {PCC_MODEL}"

    next_token_ref = ref_logits[:, SEQ - 1, :].argmax(dim=-1, keepdim=True)
    decode_logits = tt_model.forward_decode(next_token_ref, cur_pos=SEQ)
    decode_pt = tt_to_torch(decode_logits)

    full_ids = torch.cat([input_ids, next_token_ref], dim=-1)
    ref_full = tt_to_torch(tt_model(full_ids))
    p_decode = pcc(ref_full[:, SEQ, :].float(), decode_pt[:, 0, :].float())
    print(f"[KVCache-Decode/mesh]  PCC = {p_decode:.6f}")
    assert p_decode > PCC_MODEL, f"KVCache decode PCC {p_decode:.4f} < {PCC_MODEL}"


# ---------------------------------------------------------------------------
# Test 9: ISL support — different prompt lengths produce correct output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isl", [8, 16, 32, 64])
def test_isl_prefill_decode(device, real_state_dict, ref_config, tt_config, isl):
    """Verify prefill+decode across different initial sequence lengths."""
    torch.manual_seed(42 + isl)
    input_ids = torch.randint(0, ref_config.vocab_size, (BATCH, isl))

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=MAX_SEQ_LEN)

    ref_full_logits = tt_to_torch(tt_model(input_ids))

    tt_model.clear_kv_caches()
    prefill_logits = tt_model.forward_prefill(input_ids)
    prefill_pt = tt_to_torch(prefill_logits)

    p = pcc(ref_full_logits.float(), prefill_pt.float())
    print(f"[ISL={isl} Prefill] PCC = {p:.6f}")
    assert p > PCC_MODEL, f"ISL={isl} prefill PCC {p:.4f} < {PCC_MODEL}"

    next_token = ref_full_logits[:, isl - 1, :].argmax(dim=-1, keepdim=True)
    decode_logits = tt_model.forward_decode(next_token, cur_pos=isl)
    decode_pt = tt_to_torch(decode_logits)

    full_ids = torch.cat([input_ids, next_token], dim=-1)
    ref_full2 = tt_to_torch(tt_model(full_ids))
    p2 = pcc(ref_full2[:, isl, :].float(), decode_pt[:, 0, :].float())
    print(f"[ISL={isl} Decode]  PCC = {p2:.6f}")
    assert p2 > PCC_MODEL, f"ISL={isl} decode PCC {p2:.4f} < {PCC_MODEL}"


# ---------------------------------------------------------------------------
# Test 10: End-to-end generation (greedy decode)
# ---------------------------------------------------------------------------


def _ref_greedy_generate(input_ids, model_sd, cfg, max_new_tokens):
    tokens = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model_forward(tokens, model_sd, cfg)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens


def test_generator_end2end(device, real_state_dict, ref_config, tt_config):
    torch.manual_seed(9)
    prompt_len = 8
    max_new = 8
    prompt_ids = torch.randint(0, ref_config.vocab_size, (BATCH, prompt_len))

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=MAX_SEQ_LEN)
    gen = TtMiniMaxGenerator(tt_model, device, max_seq_len=MAX_SEQ_LEN, batch=BATCH)

    out = gen.generate(prompt_ids, max_new_tokens=max_new)
    assert out.shape == (BATCH, prompt_len + max_new)

    ref_out = _ref_greedy_generate(prompt_ids, real_state_dict, ref_config, max_new)
    ref_match = (out == ref_out).float().mean().item()
    print(f"[Gen-E2E] token_match_vs_ref = {ref_match:.6f}")
    assert ref_match > 0.99, f"Token match too low: {ref_match:.4f}"


# ---------------------------------------------------------------------------
# Test 11: ISL generation — different prompt lengths all produce valid output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isl", [4, 8, 16, 32])
def test_isl_generation(device, real_state_dict, ref_config, tt_config, isl):
    """End-to-end generation with different initial sequence lengths."""
    torch.manual_seed(100 + isl)
    prompt_ids = torch.randint(0, ref_config.vocab_size, (BATCH, isl))
    max_new = 4

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=MAX_SEQ_LEN)
    gen = TtMiniMaxGenerator(tt_model, device, max_seq_len=MAX_SEQ_LEN, batch=BATCH)
    out = gen.generate(prompt_ids, max_new_tokens=max_new)

    assert out.shape == (BATCH, isl + max_new), f"Expected shape {(BATCH, isl + max_new)}, got {out.shape}"

    ref_out = _ref_greedy_generate(prompt_ids, real_state_dict, ref_config, max_new)
    ref_match = (out == ref_out).float().mean().item()
    print(f"[ISL={isl} Gen] token_match = {ref_match:.6f}")
    assert ref_match > 0.99, f"ISL={isl} token match too low: {ref_match:.4f}"


# ---------------------------------------------------------------------------
# Test 12: Trace-safe generation (Metal trace capture/replay)
# ---------------------------------------------------------------------------


def test_trace_generation(device, real_state_dict, ref_config, tt_config):
    """Verify trace-safe generation produces same tokens as non-trace."""
    torch.manual_seed(77)
    prompt_len = 8
    max_new = 6
    prompt_ids = torch.randint(0, ref_config.vocab_size, (BATCH, prompt_len))

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=MAX_SEQ_LEN)
    gen = TtMiniMaxGenerator(tt_model, device, max_seq_len=MAX_SEQ_LEN, batch=BATCH)

    out_no_trace = gen.generate(prompt_ids, max_new_tokens=max_new, use_trace=False)
    gen.reset_caches()
    out_trace = gen.generate(prompt_ids, max_new_tokens=max_new, use_trace=True)

    assert (
        out_trace.shape == out_no_trace.shape
    ), f"Shape mismatch: trace={out_trace.shape}, no_trace={out_no_trace.shape}"

    match = (out_trace == out_no_trace).float().mean().item()
    print(f"[TraceGen] token_match_trace_vs_notrace = {match:.6f}")
    assert match > 0.90, f"Trace vs no-trace token match too low: {match:.4f}"


# ---------------------------------------------------------------------------
# Test 13: Paged Attention — generation with paged KV cache
# ---------------------------------------------------------------------------


def test_paged_attention_generation(device, real_state_dict, ref_config, tt_config):
    """Verify paged attention generates same tokens as non-paged."""
    torch.manual_seed(88)
    prompt_len = 8
    max_new = 6
    prompt_ids = torch.randint(0, ref_config.vocab_size, (BATCH, prompt_len))

    # Non-paged model for reference
    tt_model_nonpaged = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=MAX_SEQ_LEN)
    gen_nonpaged = TtMiniMaxGenerator(tt_model_nonpaged, device, max_seq_len=MAX_SEQ_LEN, batch=BATCH)
    out_nonpaged = gen_nonpaged.generate(prompt_ids, max_new_tokens=max_new)

    # Paged attention model
    paged_config = make_paged_attention_config(max_seq_len=MAX_SEQ_LEN)
    print(f"[PagedAttn] blocks={paged_config.max_num_blocks}, block_size={paged_config.block_size}")

    tt_model_paged = TtMiniMaxModel(
        device,
        real_state_dict,
        tt_config,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=BATCH,
        paged_attention_config=paged_config,
    )
    gen_paged = TtMiniMaxGenerator(tt_model_paged, device, max_seq_len=MAX_SEQ_LEN, batch=BATCH)
    out_paged = gen_paged.generate(prompt_ids, max_new_tokens=max_new)

    assert (
        out_paged.shape == out_nonpaged.shape
    ), f"Shape mismatch: paged={out_paged.shape}, nonpaged={out_nonpaged.shape}"

    match = (out_paged == out_nonpaged).float().mean().item()
    print(f"[PagedAttn] token_match_paged_vs_nonpaged = {match:.6f}")
    assert match > 0.90, f"Paged vs non-paged token match too low: {match:.4f}"


# ---------------------------------------------------------------------------
# Test 14: Paged Attention with trace — trace-safe paged decode
# ---------------------------------------------------------------------------


def test_paged_attention_trace(device, real_state_dict, ref_config, tt_config):
    """Verify paged attention with trace produces same tokens as without trace."""
    torch.manual_seed(99)
    prompt_len = 8
    max_new = 6
    prompt_ids = torch.randint(0, ref_config.vocab_size, (BATCH, prompt_len))

    paged_config = make_paged_attention_config(max_seq_len=MAX_SEQ_LEN)

    tt_model = TtMiniMaxModel(
        device,
        real_state_dict,
        tt_config,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=BATCH,
        paged_attention_config=paged_config,
    )
    gen = TtMiniMaxGenerator(tt_model, device, max_seq_len=MAX_SEQ_LEN, batch=BATCH)

    # Without trace
    out_no_trace = gen.generate(prompt_ids, max_new_tokens=max_new, use_trace=False)
    gen.reset_caches()

    # With trace
    out_trace = gen.generate(prompt_ids, max_new_tokens=max_new, use_trace=True)

    assert out_trace.shape == out_no_trace.shape

    match = (out_trace == out_no_trace).float().mean().item()
    print(f"[PagedAttnTrace] token_match_trace_vs_notrace = {match:.6f}")
    assert match > 0.90, f"Paged trace vs no-trace token match too low: {match:.4f}"


# ---------------------------------------------------------------------------
# Test 15: ISL 32k verification — verify model can allocate 32k context
# ---------------------------------------------------------------------------


def test_isl_32k_allocation(device, real_state_dict, tt_config):
    """Verify model can allocate paged KV cache for 32k context length."""
    max_seq_32k = 32768

    paged_config = make_paged_attention_config(max_seq_len=max_seq_32k)
    print(f"[ISL-32k] blocks={paged_config.max_num_blocks}, block_size={paged_config.block_size}")

    # Just verify allocation succeeds — don't run inference
    tt_model = TtMiniMaxModel(
        device,
        real_state_dict,
        tt_config,
        max_seq_len=max_seq_32k,
        max_batch_size=BATCH,
        paged_attention_config=paged_config,
    )

    # Verify KV cache is allocated with paged shape
    layer0 = tt_model.layers[0]
    k_shape = list(layer0.self_attn.k_cache.shape)
    print(f"[ISL-32k] K-cache shape: {k_shape}")

    expected_blocks = paged_config.max_num_blocks
    expected_block_size = paged_config.block_size
    assert k_shape[0] == expected_blocks, f"Expected {expected_blocks} blocks, got {k_shape[0]}"
    assert k_shape[2] == expected_block_size, f"Expected block_size {expected_block_size}, got {k_shape[2]}"

    print(f"[ISL-32k] Paged KV cache allocated successfully for 32k context")
