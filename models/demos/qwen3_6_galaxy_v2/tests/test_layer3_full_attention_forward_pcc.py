# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-7 — Layer-3 full_attention prefill PCC test on BH GLX 8×4.

Builds a 1-layer TtTransformer with the hybrid pattern forced to
``["full_attention"]`` (which covers QKVG + per-head QK-norm + partial RoPE
+ sigmoid output gate), forwards a T-token random hidden state through
just the attention block, and asserts the output matches the CPU
reference ``GatedAttention`` from
``models/demos/qwen3_6_galaxy/reference/qwen36.py`` with PCC > 0.99.

Block-level pattern (mirrors v1 ``test_qwen36_deltanet.py`` and
``test_llama_attention.py``): the input is a full-H replicated random
hidden state (pre-norm). We relabel layer 3 → layer 0 in the HF
state-dict so the TtTransformer (which iterates ``range(n_layers=1)``)
picks up the layer 3 weights at slot 0.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_B = 1
_T_PREFILL = 128  # TT_CCL prefill ``support_seqlens`` minimum
_LAYER_IDX = 3
_H = 5120
_PCC_THRESH = 0.99


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bh_glx_mesh():
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
# Weight loading + relabel
# ---------------------------------------------------------------------------


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        f"model.language_model.layers.{layer_idx}.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _relabel_layer_idx(state_dict: dict, src_idx: int, dst_idx: int) -> dict:
    src_pfx = f"model.language_model.layers.{src_idx}."
    dst_pfx = f"model.language_model.layers.{dst_idx}."
    out = {}
    for k, v in state_dict.items():
        if k.startswith(src_pfx):
            out[dst_pfx + k[len(src_pfx) :]] = v
        else:
            out[k] = v
    return out


def _self_attn_short_weights(state_dict: dict, layer_idx: int) -> dict:
    """Extract self_attn.* short keys for the CPU reference."""
    pfx = f"model.language_model.layers.{layer_idx}.self_attn."
    return {k[len(pfx) :]: v.float() for k, v in state_dict.items() if k.startswith(pfx)}


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def _build_partial_rope_cos_sin(T: int):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    return build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )


def _cpu_reference_gated_attn(sd_short: dict, x: torch.Tensor, T: int) -> torch.Tensor:
    """Run the validated GatedAttention CPU reference (block alone, no norm/MLP)."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    model = GatedAttention(config).eval()
    with torch.no_grad():
        model.q_proj.weight.data.copy_(sd_short["q_proj.weight"])
        model.k_proj.weight.data.copy_(sd_short["k_proj.weight"])
        model.v_proj.weight.data.copy_(sd_short["v_proj.weight"])
        model.o_proj.weight.data.copy_(sd_short["o_proj.weight"])
        model.q_norm.weight.data.copy_(sd_short["q_norm.weight"])
        model.k_norm.weight.data.copy_(sd_short["k_norm.weight"])

        cos, sin = _build_partial_rope_cos_sin(T)
        causal_mask = torch.zeros(1, 1, T, T)
        causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))
        out, _ = model(x.float(), cos, sin, kv_cache=None, attention_mask=causal_mask)
    return out  # [B, T, H]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _send_replicated(t: torch.Tensor, mesh, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _gather_replicated_first_dev(tt_tensor, mesh, T: int = None):
    out = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    out = out[0:1]
    if T is not None:
        while out.dim() > 3 and out.shape[0] == 1:
            out = out.squeeze(0)
        if out.dim() == 3:
            out = out[:, :T, :]
        else:
            out = out[..., :T, :]
    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_qwen36_layer3_full_attention_prefill_pcc(bh_glx_mesh):
    """Layer-3 (full_attention) prefill: PCC > 0.99 vs CPU GatedAttention reference."""
    state_dict_orig = _load_state_dict_for_layer(_SNAPSHOT, _LAYER_IDX)
    print(f"[Layer3] loaded {len(state_dict_orig)} weights")

    # Sanity: this layer is full_attention.
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    assert config.layer_types[_LAYER_IDX] == "full_attention"

    # Relabel layer 3 -> layer 0 so the 1-layer TtTransformer picks up the right weights.
    state_dict_for_tt = _relabel_layer_idx(state_dict_orig, src_idx=_LAYER_IDX, dst_idx=0)

    # Build TT model.
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(bh_glx_mesh)
    args.n_layers = 1
    args.linear_attention_pattern = ["full_attention"]
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=bh_glx_mesh,
        state_dict=state_dict_for_tt,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    assert getattr(model.layers[0], "is_linear_attention_layer", True) is False
    print("[Layer3] TT 1-layer full_attention built")

    # Random hidden state.
    torch.manual_seed(43)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    # CPU reference (just the attention block).
    sd_short = _self_attn_short_weights(state_dict_orig, _LAYER_IDX)
    out_ref = _cpu_reference_gated_attn(sd_short, x_cpu, _T_PREFILL)
    print(f"[Layer3] CPU ref shape: {out_ref.shape}")

    # TT forward — feed the attention block directly with full-H replicated input
    # and partial-RoPE cos/sin (built at rope_dim=64, the qwen3.6 oracle).
    x_tt = _send_replicated(x_cpu, bh_glx_mesh, dtype=ttnn.bfloat16)
    cos_ref, sin_ref = _build_partial_rope_cos_sin(_T_PREFILL)
    cos_tt = _send_replicated(cos_ref.unsqueeze(0), bh_glx_mesh)
    sin_tt = _send_replicated(sin_ref.unsqueeze(0), bh_glx_mesh)

    tt_out = model.layers[0].attention.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="prefill",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=0,
        chunk_start_idx_tensor=None,
        kv_cache=None,
        batch_size=1,
    )
    tt_out_cpu = _gather_replicated_first_dev(tt_out, bh_glx_mesh, T=_T_PREFILL)
    tt_out_cpu = tt_out_cpu.reshape(_B, _T_PREFILL, _H).float()
    print(f"[Layer3] TT out shape: {tt_out_cpu.shape}")

    pcc = _pcc(tt_out_cpu, out_ref[:, :_T_PREFILL, :])
    p99 = torch.quantile((tt_out_cpu.float() - out_ref[:, :_T_PREFILL, :].float()).abs().flatten(), 0.99).item()
    print(f"[Layer3] PCC = {pcc:.6f} (thresh={_PCC_THRESH})  |  p99 abs-diff = {p99:.4f}")
    assert pcc > _PCC_THRESH, f"Layer3 full_attention PCC {pcc:.4f} < {_PCC_THRESH} (p99={p99:.4f})"
    print("[Layer3] PASSED")
