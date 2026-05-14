# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-decode-debug — per-layer-count decode PCC sweep.

Runs the same prefill T=128 + 1 decode step at different `n_layers` values so
we can localise the layer at which the 64L PCC cliff occurs.

Each parametrize case runs in its own fixture instantiation so DeltaNet
persistent buffers + KV cache are fresh.  Failure assertion is RELAXED — we
log PCC and never abort — so all rows in the sweep run.
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


@pytest.fixture(scope="function")
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
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
    ] + [f"model.language_model.layers.{i}." for i in layer_indices]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_tt_model(mesh, state_dict, n_layers: int):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    # Truncate pattern to first n_layers entries (matches canonical order).
    args.linear_attention_pattern = args.linear_attention_pattern[:n_layers]
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


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
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


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    t_4d = t.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _cpu_reference_decode_only(state_dict_hf: dict, layer_indices: list[int], x_full: torch.Tensor):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import (
        HybridDecoderLayer,
        Qwen36Config,
        RMSNorm,
        build_mrope_cos_sin,
    )

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
    for layer_idx in layer_indices:
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
            hidden, _, _, _ = layer(hidden, cos, sin, attention_mask=causal_mask)

    final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
    final_norm.weight.data.copy_(state_dict_hf["model.language_model.norm.weight"].float())
    lm_head_w = state_dict_hf["lm_head.weight"].float()
    with torch.no_grad():
        normed = final_norm(hidden)
        logits = normed @ lm_head_w.t()
    return hidden, logits


@pytest.mark.hardware
@pytest.mark.parametrize("n_layers", [4, 8, 16, 24, 32, 48, 64])
def test_qwen36_per_layer_decode_pcc(bh_glx_mesh, n_layers):
    """Run prefill T=128 + 1 decode at the given n_layers; log PCC, never fail."""
    print(f"\n=== [decode-{n_layers}L] starting ===")
    layer_indices = list(range(n_layers))
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"[decode-{n_layers}L] loaded {len(state_dict)} weights")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, n_layers)
    print(f"[decode-{n_layers}L] TT model built")

    torch.manual_seed(44)
    T_full = _T_PREFILL + 1
    x_full = torch.randn(1, T_full, args.dim, dtype=torch.bfloat16)

    hidden_ref_full, logits_ref_full = _cpu_reference_decode_only(state_dict, layer_indices, x_full)
    decode_pos = _T_PREFILL
    logits_ref_decode = logits_ref_full[:, decode_pos : decode_pos + 1, :]

    # TT prefill
    x_prefill = x_full[:, :_T_PREFILL, :]
    x_tt = _send_col_sharded_hidden(x_prefill, bh_glx_mesh, args)
    cos_tt_prefill, sin_tt_prefill = _build_partial_rope_cos_sin_tt(
        bh_glx_mesh, torch.arange(_T_PREFILL, dtype=torch.long)
    )
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
        rot_mats=(cos_tt_prefill, sin_tt_prefill),
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

    # TT decode
    x_decode = x_full[:, _T_PREFILL : _T_PREFILL + 1, :]
    x_decode_tt = _send_col_sharded_hidden(x_decode, bh_glx_mesh, args)
    cos_tt_decode, sin_tt_decode = _build_partial_rope_cos_sin_tt(
        bh_glx_mesh, torch.tensor([_T_PREFILL], dtype=torch.long)
    )
    tt_out = model.forward(
        x_decode_tt,
        current_pos=_T_PREFILL,
        rot_mats=(cos_tt_decode, sin_tt_decode),
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )

    out0 = tt_out[0]
    logits_torch = ttnn.to_torch(
        out0,
        mesh_composer=ttnn.ConcatMesh2dToTensor(bh_glx_mesh, dims=(3, 0), mesh_shape=args.cluster_shape),
    )
    n_cols = args.cluster_shape[1]
    logits_torch = logits_torch[: logits_torch.shape[0] // n_cols]
    while logits_torch.dim() > 3 and logits_torch.shape[0] == 1:
        logits_torch = logits_torch.squeeze(0)
    logits_decode_tt = logits_torch[:, 0:1, : args.vocab_size]

    ref_logits_flat = logits_ref_decode[0, 0, :].float()
    tt_logits_flat = logits_decode_tt.reshape(-1)[: args.vocab_size].float()
    pcc_l = _pcc(tt_logits_flat, ref_logits_flat)
    p99_l = torch.quantile((tt_logits_flat - ref_logits_flat).abs().flatten(), 0.99).item()
    pred_tt = int(tt_logits_flat.argmax().item())
    pred_ref = int(ref_logits_flat.argmax().item())
    print(
        f"[decode-{n_layers}L] LOGITS PCC = {pcc_l:.6f}  |  p99 abs-diff = {p99_l:.4f}  "
        f"|  pred TT={pred_tt} ref={pred_ref} match={pred_tt == pred_ref}"
    )
    # log only — no assertion so the parametrize sweep can continue.
    assert pcc_l > -2.0, "sanity bound"
