# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation tests for MiniMax-M2.5 — Galaxy mesh (8,4).

Mesh: 8 rows (EP axis) × 4 cols (TP axis) = 32 chips
  - Attention: TP=4, column-parallel QKV + row-parallel O-proj + all-reduce
  - MoE:       EP=8, TP=4, on-device expert weights with EP+TP all-reduce

Thresholds:
  - Individual blocks (RMSNorm, Attention, MoE, DecoderLayer): PCC > 0.99
  - Full model (1 layer): PCC > 0.97

Run:
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    pytest models/demos/minimax_m2/tests/test_minimax_m2_tt.py -v
"""

import json

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
    model_forward,
    moe_forward,
    rmsnorm_forward,
)
from models.demos.minimax_m2.reference.generate_goldens import load_and_dequant
from models.demos.minimax_m2.tt.attention import TtMiniMaxAttention
from models.demos.minimax_m2.tt.model import TtDecoderLayer, TtMiniMaxModel
from models.demos.minimax_m2.tt.model_config import MiniMaxM2TTConfig, make_mesh_config
from models.demos.minimax_m2.tt.moe import TtMiniMaxMoE
from models.demos.minimax_m2.tt.rms_norm import TtRMSNorm
from models.demos.minimax_m2.tt.rope import PartialRoPESetup, apply_partial_rope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "/home/tt-admin/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/snapshots/f710177d938eff80b684d42c5aa84b382612f21f"

PCC_BLOCK = 0.99  # per-block threshold
PCC_MODEL = 0.97  # full-model threshold (1 layer)

BATCH = 1
SEQ = 16  # short sequence for fast tests

# Galaxy mesh shape
MESH_ROWS = 8  # EP axis
MESH_COLS = 4  # TP axis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    """Open Galaxy mesh device (8, 4) = 32 chips with FABRIC_1D_RING for CCL all-reduce."""
    # Enable Ethernet fabric BEFORE opening the mesh — required for ttnn.all_reduce / CCL
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    d = ttnn.open_mesh_device(ttnn.MeshShape(MESH_ROWS, MESH_COLS))
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


@pytest.fixture(scope="module")
def tt_config():
    return MiniMaxM2TTConfig(num_hidden_layers=1)


@pytest.fixture(scope="module")
def real_state_dict():
    """Load + dequantize FP8 weights from shards 0, 1, 124 (layer 0 + embed + norm)."""
    raw = {}
    for shard in [
        "model-00000-of-00126.safetensors",
        "model-00001-of-00126.safetensors",
        "model-00124-of-00126.safetensors",
    ]:
        raw.update(load_file(f"{MODEL_PATH}/{shard}"))
    return load_and_dequant(raw)


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
    """Convert TTNN tensor to torch, handling MeshDevice by reading from first device."""
    if isinstance(t.device(), ttnn.MeshDevice):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def from_torch_mesh(x: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    """Create TTNN tensor replicated across all mesh devices."""
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

    # Reference
    cos_r = cos_ref.unsqueeze(0).unsqueeze(0)
    sin_r = sin_ref.unsqueeze(0).unsqueeze(0)
    q_rot, k_rot = q_pt[..., :rdim], k_pt[..., :rdim]
    q_pass, k_pass = q_pt[..., rdim:], k_pt[..., rdim:]

    def _rot_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_ref = torch.cat([q_rot * cos_r + _rot_half(q_rot) * sin_r, q_pass], dim=-1)
    k_ref = torch.cat([k_rot * cos_r + _rot_half(k_rot) * sin_r, k_pass], dim=-1)

    # TTNN — cos/sin are replicated across mesh
    rope = PartialRoPESetup(device, rdim, ref_config.rope_theta, max_seq_len=128)
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
    """Additive causal mask: 0 on/below diagonal, -inf above. [1, 1, S, S]"""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]


def test_attention(device, mesh_config, ccl_manager, real_state_dict, ref_config, tt_config, rope_cache):
    cos, sin = rope_cache
    H = ref_config.hidden_size

    torch.manual_seed(42)
    x_pt = torch.randn(BATCH, SEQ, H)

    # Reference forward (layer 0)
    layer0_prefix = "model.layers.0.self_attn."
    attn_sd = {k.removeprefix(layer0_prefix): v for k, v in real_state_dict.items() if k.startswith(layer0_prefix)}
    ref_out = attention_forward(x_pt, attn_sd, cos, sin, ref_config)

    # TTNN forward with TP=4
    tt_attn = TtMiniMaxAttention(
        device,
        real_state_dict,
        tt_config,
        layer_idx=0,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
    )
    rope = PartialRoPESetup(device, tt_config.rotary_dim, tt_config.rope_theta, max_seq_len=128)
    cos_tt, sin_tt = rope.get_cos_sin(SEQ)

    x_tt = from_torch_mesh(x_pt.to(torch.bfloat16).unsqueeze(0), device)
    x_tt = ttnn.reshape(x_tt, (BATCH, SEQ, H))
    out_tt = tt_attn(x_tt, cos_tt, sin_tt, is_causal=False)

    # After all-reduce, all devices hold the same result; read from device[0]
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

    # Reference forward (layer 0)
    moe_prefix = "model.layers.0.block_sparse_moe."
    moe_sd = {k.removeprefix(moe_prefix): v for k, v in real_state_dict.items() if k.startswith(moe_prefix)}
    ref_out = moe_forward(x_pt, moe_sd, ref_config)

    # TTNN forward with EP=8, TP=4
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
    assert p > PCC_BLOCK, f"MoE PCC {p:.4f} < {PCC_BLOCK}"


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
    )
    rope = PartialRoPESetup(device, tt_config.rotary_dim, tt_config.rope_theta, max_seq_len=128)
    cos_tt, sin_tt = rope.get_cos_sin(SEQ)

    x_tt = from_torch_mesh(x_pt.to(torch.bfloat16).unsqueeze(0), device)
    x_tt = ttnn.reshape(x_tt, (BATCH, SEQ, H))
    out_tt = tt_layer(x_tt, cos_tt, sin_tt, is_causal=False)
    out_pt = tt_to_torch(out_tt).reshape_as(ref_out)

    p = pcc(ref_out.float(), out_pt)
    print(f"[DecoderLayer/mesh] PCC = {p:.6f}")
    assert p > PCC_BLOCK, f"DecoderLayer PCC {p:.4f} < {PCC_BLOCK}"


# ---------------------------------------------------------------------------
# Test 6: Full model (1 layer)
# ---------------------------------------------------------------------------


def test_full_model(device, real_state_dict, ref_config, tt_config):
    torch.manual_seed(0)
    input_ids = torch.randint(0, ref_config.vocab_size, (BATCH, SEQ))

    ref_logits = model_forward(input_ids, real_state_dict, ref_config)

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=128)
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
    raw = {}
    for shard in SHARDS_FOR_4_LAYERS:
        raw.update(load_file(f"{MODEL_PATH}/{shard}"))
    return load_and_dequant(raw)


@pytest.fixture(scope="module")
def ref_config_4layers():
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


@pytest.fixture(scope="module")
def tt_config_4layers():
    return MiniMaxM2TTConfig(num_hidden_layers=N_LAYERS_MULTI)


def test_multilayer_model(device, state_dict_4layers, ref_config_4layers, tt_config_4layers):
    """4-layer end-to-end PCC test."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, ref_config_4layers.vocab_size, (BATCH, SEQ))

    ref_logits = model_forward(input_ids, state_dict_4layers, ref_config_4layers)

    tt_model = TtMiniMaxModel(device, state_dict_4layers, tt_config_4layers, max_seq_len=128)
    out_tt = tt_model(input_ids)
    out_pt = tt_to_torch(out_tt).reshape_as(ref_logits)

    p = pcc(ref_logits.float(), out_pt)
    print(f"[FullModel-{N_LAYERS_MULTI}layers/mesh] PCC = {p:.6f}")
    assert p > PCC_MODEL, f"Multi-layer PCC {p:.4f} < {PCC_MODEL}"


# ---------------------------------------------------------------------------
# Test 8: KV-cache prefill + decode
# ---------------------------------------------------------------------------


def test_kvcache_prefill_decode(device, real_state_dict, tt_config):
    torch.manual_seed(1)
    input_ids = torch.randint(0, 200064, (BATCH, SEQ))

    tt_model = TtMiniMaxModel(device, real_state_dict, tt_config, max_seq_len=128)

    ref_logits = tt_to_torch(tt_model(input_ids))  # [B, SEQ, V]

    kv_caches = tt_model.allocate_kv_cache(batch=BATCH)
    prefill_logits, kv_caches = tt_model.forward_prefill(input_ids, kv_caches)
    prefill_pt = tt_to_torch(prefill_logits)

    p_prefill = pcc(ref_logits.float(), prefill_pt.float())
    print(f"[KVCache-Prefill/mesh] PCC = {p_prefill:.6f}")
    assert p_prefill > PCC_MODEL, f"KVCache prefill PCC {p_prefill:.4f} < {PCC_MODEL}"

    next_token_ref = ref_logits[:, SEQ - 1, :].argmax(dim=-1, keepdim=True)
    decode_logits, _ = tt_model.forward_decode(next_token_ref, kv_caches, cur_pos=SEQ)
    decode_pt = tt_to_torch(decode_logits)

    full_ids = torch.cat([input_ids, next_token_ref], dim=-1)
    ref_full = tt_to_torch(tt_model(full_ids))
    p_decode = pcc(ref_full[:, SEQ, :].float(), decode_pt[:, 0, :].float())
    print(f"[KVCache-Decode/mesh]  PCC = {p_decode:.6f}")
    assert p_decode > PCC_MODEL, f"KVCache decode PCC {p_decode:.4f} < {PCC_MODEL}"
