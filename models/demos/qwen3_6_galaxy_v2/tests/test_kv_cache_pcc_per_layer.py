# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-DEC-3 — KV cache contents PCC per full-attention layer (post-prefill).

Diagnostic to verify the hypothesis that prefill's compounding drift corrupts
the KV cache contents at later layers, contributing to the 64L decode PCC
degradation observed even after teacher forcing the residual stream every 16L.

Approach:
  1. Build 64L TT model, run prefill T=128.
  2. For each full_attention layer (3, 19, 35, 51, 63):
     - Read TT's KV cache (gather from mesh to torch).
     - Run HF CPU reference forward through all layers, capturing kv_cache_new
       at each full_attn layer.
     - Compute PCC(TT_K, HF_K), PCC(TT_V, HF_V).
  3. If PCC degrades with depth → confirms hypothesis.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && pytest models/demos/qwen3_6_galaxy_v2/tests/test_kv_cache_pcc_per_layer.py -v -s
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
_T_PREFILL = 128
_N_LAYERS = 64
_FULL_ATTN_LAYERS = [3, 19, 35, 51, 63]  # one per 16L segment


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


def _load_state_dict_for_layers(snapshot_dir: pathlib.Path, layer_indices: list[int]) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = ["model.language_model.embed_tokens.", "model.language_model.norm.", "lm_head."] + [
        f"model.language_model.layers.{i}." for i in layer_indices
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


def _build_tt_model(mesh, state_dict, n_layers):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


def _send_col_sharded(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, 1, T, H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _build_partial_rope(mesh, positions: torch.Tensor):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    cos_tt = ttnn.from_torch(
        cos_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() < 2:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _gather_kv_cache_from_mesh(kv_tt, mesh, args):
    """Gather TT KV cache to torch.

    V2-TP layout: per-chip [B=1, n_kv_local=1, max_S, hd] with KV heads on
    rows (8-way), cols replicate.  ConcatMesh2dToTensor(dims=(0, 1)) gives
    [rows*B=8, cols*n_kv_local=4, max_S, hd] — the 8 rows are the 8 padded
    KV heads stacked on dim 0; the 4 cols on dim 1 are all identical
    (replicated).
    Returns [n_kv=4, max_S, hd] (de-padded to match HF's 4 heads).
    """
    out = ttnn.to_torch(
        kv_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=args.cluster_shape),
    )
    # Drop replicated cols: keep col 0 only.
    out = out[:, 0]  # [8, max_S, hd]
    # De-pad: heads are k0,k0,k1,k1,k2,k2,k3,k3 → take every other → k0..k3.
    out = out[::2]  # [4, max_S, hd]
    return out


def _cpu_reference_capture_kv(state_dict_hf, layer_indices_full: list[int], n_layers: int, x_full: torch.Tensor):
    """Run full HF reference for `n_layers` and capture kv_cache_new at each full_attn layer.

    Returns dict {layer_idx: (K, V)} where K, V are [B, n_kv, T, head_dim] each.
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config, build_mrope_cos_sin

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    T = x_full.shape[1]
    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    causal_mask = torch.zeros(1, 1, T, T)
    causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))

    hidden = x_full.float()
    captured_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in range(n_layers):
        layer = HybridDecoderLayer(config, layer_idx).eval()
        pfx = f"model.language_model.layers.{layer_idx}."
        layer_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict_hf.items():
            if k.startswith(pfx):
                short = k[len(pfx) :]
                if short.startswith("self_attn."):
                    layer_sd["attention." + short[len("self_attn.") :]] = v.float()
                elif short.startswith("linear_attn."):
                    layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
                else:
                    layer_sd[short] = v.float()
        layer.load_state_dict(layer_sd, strict=False)
        with torch.no_grad():
            hidden, kv_cache_new, _, _ = layer(hidden, cos, sin, attention_mask=causal_mask)
        if layer_idx in layer_indices_full:
            # kv_cache_new is tuple (K, V) for full_attn layers
            if kv_cache_new is not None:
                K_hf, V_hf = kv_cache_new
                captured_kv[layer_idx] = (K_hf.detach().clone(), V_hf.detach().clone())
                print(f"  [HF L{layer_idx}] captured K {tuple(K_hf.shape)} V {tuple(V_hf.shape)}")
        del layer
    return captured_kv


@pytest.mark.hardware
def test_kv_cache_pcc_per_full_attn_layer(bh_glx_mesh):
    """Confirm KV cache degradation hypothesis: PCC drops with layer depth."""
    print("\n=== KV cache PCC diagnostic — 5 full_attn layers across 64L prefill ===")
    layer_indices = list(range(_N_LAYERS))
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"loaded {len(state_dict)} weights")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, _N_LAYERS)
    print(f"TT 64L model built")

    torch.manual_seed(44)
    x_full = torch.randn(1, _T_PREFILL, args.dim, dtype=torch.bfloat16)

    # TT prefill T=128 (populates KV caches at every full_attn layer).
    x_tt = _send_col_sharded(x_full, bh_glx_mesh, args)
    cos_tt, sin_tt = _build_partial_rope(bh_glx_mesh, torch.arange(_T_PREFILL, dtype=torch.long))
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    _ = model.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="prefill",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    print(f"TT prefill T={_T_PREFILL} done")

    # CPU reference forward, capturing kv_cache_new at each target layer.
    print(f"running HF CPU reference forward (slow, ~5min)...")
    hf_kv = _cpu_reference_capture_kv(state_dict, _FULL_ATTN_LAYERS, _N_LAYERS, x_full)
    print(f"HF KV captured for {len(hf_kv)} layers: {sorted(hf_kv.keys())}")

    # Read TT's KV cache for each target layer and compare.
    print("\n=== TT vs HF KV cache PCC per full_attn layer ===")
    print(f"{'layer':>6}  {'K PCC':>10}  {'V PCC':>10}  {'TT K shape':>22}  {'HF K shape':>22}")
    for layer_idx in _FULL_ATTN_LAYERS:
        if not getattr(model.layers[layer_idx], "is_linear_attention_layer", False) is False:
            print(f"  layer {layer_idx}: SKIP (not full_attn)")
            continue
        attn = model.layers[layer_idx].attention
        K_tt = _gather_kv_cache_from_mesh(attn.layer_past[0], bh_glx_mesh, args)
        V_tt = _gather_kv_cache_from_mesh(attn.layer_past[1], bh_glx_mesh, args)
        K_hf, V_hf = hf_kv[layer_idx]

        # TT cache: [n_kv, max_S, head_dim] (after gather + squeeze).
        # HF cache: [B, n_kv, T_prefilled, head_dim]
        # Slice TT to prefilled positions only.
        if K_tt.dim() == 3:
            K_tt = K_tt[:, :_T_PREFILL, :].unsqueeze(0)  # [1, n_kv, T, hd]
            V_tt = V_tt[:, :_T_PREFILL, :].unsqueeze(0)
        elif K_tt.dim() == 4:
            K_tt = K_tt[:, :, :_T_PREFILL, :]
            V_tt = V_tt[:, :, :_T_PREFILL, :]
        # HF: ensure same shape
        K_hf_clipped = K_hf[:, :, :_T_PREFILL, :] if K_hf.dim() == 4 else K_hf

        K_pcc = _pcc(K_tt, K_hf_clipped)
        V_pcc = _pcc(V_tt, V_hf if V_hf.dim() == K_hf_clipped.dim() else V_hf)
        print(
            f"  {layer_idx:>4}  {K_pcc:>10.6f}  {V_pcc:>10.6f}  {str(tuple(K_tt.shape)):>22}  {str(tuple(K_hf_clipped.shape)):>22}"
        )

    print("\n=== DONE ===")
