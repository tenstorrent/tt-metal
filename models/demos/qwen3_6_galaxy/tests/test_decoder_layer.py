# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for models/demos/qwen3_6_galaxy/tt/llama_decoder.py — Task 7.

Tests are written RED-first (before implementation) and then run GREEN after
TtQwen36DecoderLayer is implemented.

Hardware required: 32-chip BH GLX 8×4 mesh.

Run all tests:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_decoder_layer.py -x -s -v
"""

import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_B = 1
_T_PREFILL = 32  # prefill seq-len (≥ 32, divisible by 32)
_T_DECODE = 1
_H = 5120
_PCC_THRESH = 0.99


# ---------------------------------------------------------------------------
# Fabric mesh fixture (same as all prior tasks)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the full 8×4 fabric mesh with FABRIC_1D_RING topology."""
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _load_layer_weights(layer_idx: int) -> dict:
    """Load all weights for a single decoder layer from safetensors.

    Returns a flat dict with keys in the form expected by TtQwen36DecoderLayer
    constructor:
      input_layernorm.weight       [H]
      post_attention_layernorm.weight [H]
      mlp.gate_proj.weight         [intermediate, H]
      mlp.up_proj.weight           [intermediate, H]
      mlp.down_proj.weight         [H, intermediate]
      self_attn.*  or  linear_attn.*  depending on layer_type

    We load the raw safetensors keys (model.language_model.layers.N.*) and
    strip the prefix so the constructor can use them directly.
    """
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    pfx = f"model.language_model.layers.{layer_idx}"
    keys_needed = [k for k in weight_map if k.startswith(pfx + ".")]

    # Find which shard files we need
    files_needed = sorted({weight_map[k] for k in keys_needed})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in keys_needed:
            if k in shard:
                raw[k] = shard[k].float()

    # Strip prefix and return short-key dict
    result = {}
    for k, v in raw.items():
        short = k[len(pfx) + 1 :]  # strip "model.language_model.layers.N."
        result[short] = v
    return result


# ---------------------------------------------------------------------------
# CPU reference helpers
# ---------------------------------------------------------------------------


def _load_config() -> "Qwen36Config":
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        d = json.load(f)
    return Qwen36Config(d)


def _build_ref_decoder_layer(config, layer_idx: int, weights: dict):
    """Instantiate a reference HybridDecoderLayer and load weights.

    Weight dict is the flat short-key dict returned by _load_layer_weights().
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer

    layer = HybridDecoderLayer(config, layer_idx)
    layer.eval()

    with torch.no_grad():
        layer.input_layernorm.weight.data.copy_(weights["input_layernorm.weight"])
        layer.post_attention_layernorm.weight.data.copy_(weights["post_attention_layernorm.weight"])
        layer.mlp.gate_proj.weight.data.copy_(weights["mlp.gate_proj.weight"])
        layer.mlp.up_proj.weight.data.copy_(weights["mlp.up_proj.weight"])
        layer.mlp.down_proj.weight.data.copy_(weights["mlp.down_proj.weight"])

        lt = config.layer_types[layer_idx]
        if lt == "full_attention":
            attn = layer.attention
            attn.q_proj.weight.data.copy_(weights["self_attn.q_proj.weight"])
            attn.k_proj.weight.data.copy_(weights["self_attn.k_proj.weight"])
            attn.v_proj.weight.data.copy_(weights["self_attn.v_proj.weight"])
            attn.o_proj.weight.data.copy_(weights["self_attn.o_proj.weight"])
            attn.q_norm.weight.data.copy_(weights["self_attn.q_norm.weight"])
            attn.k_norm.weight.data.copy_(weights["self_attn.k_norm.weight"])
        else:
            dn = layer.attention
            dn.in_proj_qkv.weight.data.copy_(weights["linear_attn.in_proj_qkv.weight"])
            dn.in_proj_z.weight.data.copy_(weights["linear_attn.in_proj_z.weight"])
            dn.in_proj_a.weight.data.copy_(weights["linear_attn.in_proj_a.weight"])
            dn.in_proj_b.weight.data.copy_(weights["linear_attn.in_proj_b.weight"])
            dn.conv1d.weight.data.copy_(weights["linear_attn.conv1d.weight"])
            dn.A_log.data.copy_(weights["linear_attn.A_log"])
            dn.dt_bias.data.copy_(weights["linear_attn.dt_bias"])
            dn.norm.weight.data.copy_(weights["linear_attn.norm.weight"])
            dn.out_proj.weight.data.copy_(weights["linear_attn.out_proj.weight"])

    return layer


def _build_rope_cos_sin(T: int):
    """Build MRoPE cos/sin for text positions [0, T)."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    return cos, sin  # [1, T, 64]


# ---------------------------------------------------------------------------
# PCC helper
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    cc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return cc


# ---------------------------------------------------------------------------
# TTNN helpers
# ---------------------------------------------------------------------------


def _send_to_device(t: torch.Tensor, mesh_device, dtype=None):
    """Replicate tensor to all 32 devices."""
    import ttnn

    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _gather_from_device(tt_tensor, mesh_device, T: int = None):
    """Gather replicated TTNN tensor back to CPU."""
    import ttnn

    all_dev = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    result = all_dev[0:1]  # first device
    if T is not None:
        result = result[:, :T, :]
    return result


def _build_tt_decoder_layer(mesh_device, args, weights: dict, layer_idx: int):
    """Instantiate TtQwen36DecoderLayer with real weights."""
    from models.demos.qwen3_6_galaxy.tt.llama_decoder import TtQwen36DecoderLayer

    return TtQwen36DecoderLayer(
        mesh_device=mesh_device,
        args=args,
        state_dict=weights,
        layer_idx=layer_idx,
    )


# ---------------------------------------------------------------------------
# Test 1: linear_attention layer 0, prefill PCC > 0.99
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_decoder_layer_0_linear_attention_pcc_on_8x4(mesh_8x4):
    """TtQwen36DecoderLayer layer_idx=0 (linear_attention): prefill PCC > 0.99.

    Verifies that the hybrid decoder correctly dispatches to DeltaNet for
    layer_types[0]='linear_attention', computes norms + deltanet + MLP, and
    matches the CPU reference HybridDecoderLayer with PCC > 0.99.
    """
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    layer_idx = 0
    weights = _load_layer_weights(layer_idx)
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    assert (
        config.layer_types[layer_idx] == "linear_attention"
    ), f"Expected layer_types[{layer_idx}]='linear_attention', got '{config.layer_types[layer_idx]}'"

    # Random hidden state
    torch.manual_seed(42)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    # --- CPU reference ---
    ref_layer = _build_ref_decoder_layer(config, layer_idx, weights)
    cos_ref, sin_ref = _build_rope_cos_sin(_T_PREFILL)
    with torch.no_grad():
        causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
        )
        out_ref, _, _, _ = ref_layer(x_cpu.float(), cos_ref, sin_ref, attention_mask=causal_mask)
    print(f"\n[Layer0] CPU ref shape: {out_ref.shape}")

    # --- TTNN ---
    tt_layer = _build_tt_decoder_layer(mesh_8x4, args, weights, layer_idx)

    x_tt = _send_to_device(x_cpu, mesh_8x4)
    cos_tt = _send_to_device(cos_ref.unsqueeze(0), mesh_8x4)  # [1, 1, T, 64]
    sin_tt = _send_to_device(sin_ref.unsqueeze(0), mesh_8x4)

    with torch.no_grad():
        out_tt, new_dn_state, new_conv_state = tt_layer.forward(
            x_tt,
            current_pos=0,
            rot_mats=(cos_tt, sin_tt),
            attention_mask=None,
            mode="prefill",
        )

    out_tt_cpu = _gather_from_device(out_tt, mesh_8x4, T=_T_PREFILL)
    print(f"[Layer0] TT out shape: {out_tt_cpu.shape}")

    pcc = _pcc(out_tt_cpu, out_ref[:, :_T_PREFILL, :].float())
    print(f"[Layer0] PCC = {pcc:.6f}  (thresh={_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"Layer0 PCC {pcc:.4f} < {_PCC_THRESH}"
    print("[Layer0] PASSED")


# ---------------------------------------------------------------------------
# Test 2: full_attention layer 3, prefill PCC > 0.99
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_decoder_layer_3_full_attention_pcc_on_8x4(mesh_8x4):
    """TtQwen36DecoderLayer layer_idx=3 (full_attention): prefill PCC > 0.99.

    Verifies that the hybrid decoder correctly dispatches to GatedAttention for
    layer_types[3]='full_attention', applies norms + gated attention + MLP, and
    matches the CPU reference with PCC > 0.99.
    """
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    layer_idx = 3
    weights = _load_layer_weights(layer_idx)
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    assert (
        config.layer_types[layer_idx] == "full_attention"
    ), f"Expected layer_types[{layer_idx}]='full_attention', got '{config.layer_types[layer_idx]}'"

    # Random hidden state
    torch.manual_seed(43)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    # --- CPU reference ---
    ref_layer = _build_ref_decoder_layer(config, layer_idx, weights)
    cos_ref, sin_ref = _build_rope_cos_sin(_T_PREFILL)
    with torch.no_grad():
        causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
        )
        out_ref, _, _, _ = ref_layer(x_cpu.float(), cos_ref, sin_ref, attention_mask=causal_mask)
    print(f"\n[Layer3] CPU ref shape: {out_ref.shape}")

    # --- TTNN ---
    tt_layer = _build_tt_decoder_layer(mesh_8x4, args, weights, layer_idx)

    x_tt = _send_to_device(x_cpu, mesh_8x4)
    cos_tt = _send_to_device(cos_ref.unsqueeze(0), mesh_8x4)
    sin_tt = _send_to_device(sin_ref.unsqueeze(0), mesh_8x4)

    with torch.no_grad():
        out_tt, new_dn_state, new_conv_state = tt_layer.forward(
            x_tt,
            current_pos=0,
            rot_mats=(cos_tt, sin_tt),
            attention_mask=None,
            mode="prefill",
        )

    out_tt_cpu = _gather_from_device(out_tt, mesh_8x4, T=_T_PREFILL)
    print(f"[Layer3] TT out shape: {out_tt_cpu.shape}")

    pcc = _pcc(out_tt_cpu, out_ref[:, :_T_PREFILL, :].float())
    print(f"[Layer3] PCC = {pcc:.6f}  (thresh={_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"Layer3 PCC {pcc:.4f} < {_PCC_THRESH}"
    print("[Layer3] PASSED")


# ---------------------------------------------------------------------------
# Test 3: 4-layer hybrid slice prefill PCC > 0.99
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_4layer_hybrid_slice_prefill_pcc_on_8x4(mesh_8x4):
    """Stack 4 decoder layers (idx 0-3 = lin,lin,lin,full) and run prefill.

    Verifies that state threading across the 4 layers works correctly and the
    composed output matches the CPU reference with PCC > 0.99.
    """
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 4
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    # Load all 4 layers' weights
    weights_list = [_load_layer_weights(i) for i in range(N_LAYERS)]

    # Random hidden state
    torch.manual_seed(44)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    cos_ref, sin_ref = _build_rope_cos_sin(_T_PREFILL)
    causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
    )

    # --- CPU reference: stack 4 layers ---
    ref_layers = [_build_ref_decoder_layer(config, i, weights_list[i]) for i in range(N_LAYERS)]
    x_ref = x_cpu.float()
    conv_state, rec_state = None, None
    kv_caches = [None] * N_LAYERS
    with torch.no_grad():
        for i, rl in enumerate(ref_layers):
            x_ref, kv_caches[i], conv_state, rec_state = rl(
                x_ref,
                cos_ref,
                sin_ref,
                attention_mask=causal_mask,
                kv_cache=kv_caches[i],
                conv_state=conv_state,
                recurrent_state=rec_state,
            )
    out_ref = x_ref
    print(f"\n[4Layer Prefill] CPU ref output shape: {out_ref.shape}")

    # --- TTNN: stack 4 layers ---

    tt_layers = [_build_tt_decoder_layer(mesh_8x4, args, weights_list[i], i) for i in range(N_LAYERS)]

    x_tt = _send_to_device(x_cpu, mesh_8x4)
    cos_tt = _send_to_device(cos_ref.unsqueeze(0), mesh_8x4)
    sin_tt = _send_to_device(sin_ref.unsqueeze(0), mesh_8x4)

    dn_state, conv_state_tt = None, None
    with torch.no_grad():
        for tt_layer in tt_layers:
            x_tt, dn_state, conv_state_tt = tt_layer.forward(
                x_tt,
                current_pos=0,
                rot_mats=(cos_tt, sin_tt),
                attention_mask=None,
                mode="prefill",
                deltanet_state=dn_state,
                deltanet_conv_state=conv_state_tt,
            )

    out_tt_cpu = _gather_from_device(x_tt, mesh_8x4, T=_T_PREFILL)
    print(f"[4Layer Prefill] TT output shape: {out_tt_cpu.shape}")

    pcc = _pcc(out_tt_cpu, out_ref[:, :_T_PREFILL, :].float())
    print(f"[4Layer Prefill] PCC = {pcc:.6f}  (thresh={_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"4-layer prefill PCC {pcc:.4f} < {_PCC_THRESH}"
    print("[4Layer Prefill] PASSED")


# ---------------------------------------------------------------------------
# Test 4: 4-layer hybrid slice decode-step PCC > 0.99
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_4layer_hybrid_slice_decode_step_pcc_on_8x4(mesh_8x4):
    """Stack 4 decoder layers, pre-prefill T=16, then run a single decode step.

    Verifies that the decode step (T=1) correctly uses KV cache (full_attention)
    and DeltaNet recurrent state (linear_attention) populated during prefill.
    PCC threshold: > 0.99 against CPU reference.
    """
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 4
    T_PRE = 16  # prefill tokens (padded to 32 for TTNN tile alignment)
    T_PRE_PAD = 32  # tile-aligned prefill length

    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)
    weights_list = [_load_layer_weights(i) for i in range(N_LAYERS)]

    torch.manual_seed(45)
    x_pre_cpu = torch.randn(_B, T_PRE_PAD, _H, dtype=torch.bfloat16)
    x_dec_cpu = torch.randn(_B, 1, _H, dtype=torch.bfloat16)

    cos_pre, sin_pre = _build_rope_cos_sin(T_PRE_PAD)
    cos_dec, sin_dec = _build_rope_cos_sin(1)

    # ---------------------------------------------------------------------------
    # CPU reference: prefill then decode
    # ---------------------------------------------------------------------------
    ref_layers = [_build_ref_decoder_layer(config, i, weights_list[i]) for i in range(N_LAYERS)]

    causal_mask_pre = torch.zeros(1, 1, T_PRE_PAD, T_PRE_PAD)
    causal_mask_pre = causal_mask_pre.masked_fill(
        torch.triu(torch.ones(T_PRE_PAD, T_PRE_PAD), diagonal=1).bool(), float("-inf")
    )

    # Prefill pass
    x_ref = x_pre_cpu.float()
    cs_ref, rs_ref = None, None
    kv_caches_ref = [None] * N_LAYERS
    with torch.no_grad():
        for i, rl in enumerate(ref_layers):
            x_ref, kv_caches_ref[i], cs_ref, rs_ref = rl(
                x_ref,
                cos_pre,
                sin_pre,
                attention_mask=causal_mask_pre,
                kv_cache=kv_caches_ref[i],
                conv_state=cs_ref,
                recurrent_state=rs_ref,
            )

    # Decode pass: take last real token position
    x_ref = x_dec_cpu.float()
    cs_ref_dec, rs_ref_dec = cs_ref, rs_ref
    kv_caches_ref_dec = list(kv_caches_ref)  # carry KV cache forward
    with torch.no_grad():
        for i, rl in enumerate(ref_layers):
            x_ref, kv_caches_ref_dec[i], cs_ref_dec, rs_ref_dec = rl(
                x_ref,
                cos_dec,
                sin_dec,
                attention_mask=None,
                kv_cache=kv_caches_ref_dec[i],
                conv_state=cs_ref_dec,
                recurrent_state=rs_ref_dec,
            )
    out_ref_dec = x_ref
    print(f"\n[4Layer Decode] CPU ref decode output shape: {out_ref_dec.shape}")

    # ---------------------------------------------------------------------------
    # TTNN: prefill then decode
    # ---------------------------------------------------------------------------

    tt_layers = [_build_tt_decoder_layer(mesh_8x4, args, weights_list[i], i) for i in range(N_LAYERS)]

    # Prefill
    x_tt = _send_to_device(x_pre_cpu, mesh_8x4)
    cos_tt_pre = _send_to_device(cos_pre.unsqueeze(0), mesh_8x4)
    sin_tt_pre = _send_to_device(sin_pre.unsqueeze(0), mesh_8x4)

    dn_state_tt, conv_state_tt = None, None
    with torch.no_grad():
        for tt_layer in tt_layers:
            x_tt, dn_state_tt, conv_state_tt = tt_layer.forward(
                x_tt,
                current_pos=0,
                rot_mats=(cos_tt_pre, sin_tt_pre),
                attention_mask=None,
                mode="prefill",
                deltanet_state=dn_state_tt,
                deltanet_conv_state=conv_state_tt,
            )

    # Decode step
    x_tt_dec = _send_to_device(x_dec_cpu, mesh_8x4)
    cos_tt_dec = _send_to_device(cos_dec.unsqueeze(0), mesh_8x4)
    sin_tt_dec = _send_to_device(sin_dec.unsqueeze(0), mesh_8x4)

    with torch.no_grad():
        for tt_layer in tt_layers:
            x_tt_dec, dn_state_tt, conv_state_tt = tt_layer.forward(
                x_tt_dec,
                current_pos=T_PRE_PAD,
                rot_mats=(cos_tt_dec, sin_tt_dec),
                attention_mask=None,
                mode="decode",
                deltanet_state=dn_state_tt,
                deltanet_conv_state=conv_state_tt,
            )

    out_tt_dec_cpu = _gather_from_device(x_tt_dec, mesh_8x4, T=1)
    print(f"[4Layer Decode] TT decode output shape: {out_tt_dec_cpu.shape}")

    pcc = _pcc(out_tt_dec_cpu, out_ref_dec[:, :1, :].float())
    print(f"[4Layer Decode] PCC = {pcc:.6f}  (thresh={_PCC_THRESH})")
    assert pcc > _PCC_THRESH, f"4-layer decode PCC {pcc:.4f} < {_PCC_THRESH}"
    print("[4Layer Decode] PASSED")
