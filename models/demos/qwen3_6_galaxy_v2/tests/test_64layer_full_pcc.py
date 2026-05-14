# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-7b — 64-layer full Qwen3.6-27B TtTransformer prefill PCC test on BH GLX 8x4.

Builds the full 64-layer TtTransformer with HF weights for the canonical
[lin, lin, lin, full] × 16 hybrid pattern, runs prefill T=128 of the
"The capital of France is" prompt (padded with zeros), and asserts:

- hidden-state PCC > 0.99 vs CPU reference Qwen36TextModel (all 64 layers)
- logits PCC > 0.99 (CPU norm + lm_head applied to TT hidden state)
- predicted next-token id at position 4 (last prompt token) decodes to a
  string containing "Paris" (matching v1 demo behaviour)

Memory: 27B params × 2 bytes (bf16) ≈ 54 GB total weights, sharded across
8×4 mesh → ≈ 6.75 GB per row-axis chip. Fits comfortably in 32 GB DRAM
per chip.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_64layer_full_pcc.py \\
            -v -s
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
_PCC_THRESH_HIDDEN = 0.99
_PCC_THRESH_LOGITS = 0.99


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


def _load_state_dict_all_layers(snapshot_dir: pathlib.Path) -> dict:
    """Load HF weights for embedding + all 64 layers + final norm + lm_head."""
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        "model.language_model.layers.",
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


def _cpu_reference_64_layers(state_dict_hf: dict, x: torch.Tensor):
    """Run the full 64-layer reference at float32, return (hidden, logits)."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import (
        HybridDecoderLayer,
        Qwen36Config,
        RMSNorm,
        build_mrope_cos_sin,
    )

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    T = x.shape[1]
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

    hidden = x.float()
    for layer_idx in range(_N_LAYERS):
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
        del layer  # free memory after each layer

    final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
    final_norm.weight.data.copy_(state_dict_hf["model.language_model.norm.weight"].float())
    lm_head_w = state_dict_hf["lm_head.weight"].float()
    with torch.no_grad():
        normed = final_norm(hidden)
        logits = normed @ lm_head_w.t()
    return hidden, logits


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_tt_model(mesh, state_dict, pattern: list[str], n_layers: int):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    args.linear_attention_pattern = pattern
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


def _build_partial_rope_cos_sin_tt(mesh, T: int):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
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


def _gather_col_sharded_to_full(tt_tensor, mesh, args, T: int):
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


@pytest.mark.hardware
def test_qwen36_64_layer_full_prefill_pcc(bh_glx_mesh):
    """Full 64-layer hybrid prefill — hidden PCC + logits PCC + Paris token check."""
    print("[64L] loading HF state_dict for all 64 layers + embedding/norm/lm_head ...")
    state_dict = _load_state_dict_all_layers(_SNAPSHOT)
    print(f"[64L] loaded {len(state_dict)} weight tensors")

    # Build the canonical [lin, lin, lin, full] × 16 pattern from HF config.
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    pattern = list(config.layer_types)
    assert len(pattern) == _N_LAYERS

    print(f"[64L] building TT 64-layer TtTransformer (pattern = [lin × 3, full] × 16) ...")
    model, args = _build_tt_model(bh_glx_mesh, state_dict, pattern, _N_LAYERS)
    print(f"[64L] TT 64-layer model built")

    # --- Paris prompt setup ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]
    print(f"[64L] prompt = {prompt!r}, T_prompt = {T_prompt}, ids = {input_ids[0].tolist()}")
    # Pad to T=128.
    input_ids_padded = torch.zeros(1, _T_PREFILL, dtype=input_ids.dtype)
    input_ids_padded[0, :T_prompt] = input_ids[0]

    # CPU reference: build the embedded hidden state from the same tokens, then
    # run all 64 layers + final norm + lm_head at float32.
    embed_w = state_dict["model.language_model.embed_tokens.weight"].float()
    x_cpu_torch = embed_w[input_ids_padded[0]].unsqueeze(0)  # [1, T, H]
    print(f"[64L] embedded input shape: {x_cpu_torch.shape}")
    print(f"[64L] running CPU reference for 64 layers (may take ~5–10 min) ...")
    hidden_ref, logits_ref = _cpu_reference_64_layers(state_dict, x_cpu_torch)
    print(f"[64L] CPU ref hidden shape: {hidden_ref.shape}, logits shape: {logits_ref.shape}")

    # TT forward.
    x_tt = _send_col_sharded_hidden(x_cpu_torch.to(torch.bfloat16), bh_glx_mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    print(f"[64L] running TT 64-layer prefill ...")
    tt_hidden = model.forward(
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
    tt_hidden_cpu = _gather_col_sharded_to_full(tt_hidden, bh_glx_mesh, args, T=_T_PREFILL)
    tt_hidden_cpu = tt_hidden_cpu.reshape(1, _T_PREFILL, -1).float()
    print(f"[64L] TT hidden shape: {tt_hidden_cpu.shape}")

    pcc_h = _pcc(tt_hidden_cpu, hidden_ref[:, :_T_PREFILL, :])
    p99_h = torch.quantile((tt_hidden_cpu.float() - hidden_ref[:, :_T_PREFILL, :].float()).abs().flatten(), 0.99).item()
    print(f"[64L] HIDDEN PCC = {pcc_h:.6f} (thresh={_PCC_THRESH_HIDDEN})  |  p99 abs-diff = {p99_h:.4f}")

    # CPU norm + lm_head applied to TT hidden state (V2-7b: TT's own lm_head
    # path is exercised in the 4L test; for 64L we focus on the layer-stack
    # propagation since the LM head is a single linear layer with no
    # quantisation surprise beyond what the 4L logits PCC already verified).
    from models.demos.qwen3_6_galaxy.reference.qwen36 import RMSNorm as RefRMSNorm

    final_norm = RefRMSNorm(config.hidden_size, eps=config.rms_norm_eps, zero_centered=True)
    final_norm.weight.data.copy_(state_dict["model.language_model.norm.weight"].float())
    lm_head_w = state_dict["lm_head.weight"].float()
    with torch.no_grad():
        tt_logits_full = final_norm(tt_hidden_cpu.float()) @ lm_head_w.t()
    last_idx = T_prompt - 1  # token 4 ("is" → next is " Paris")
    tt_logits_last = tt_logits_full[0, last_idx, :].float()
    ref_logits_last = logits_ref[0, last_idx, :].float()
    pcc_l = _pcc(tt_logits_last, ref_logits_last)
    p99_l = torch.quantile((tt_logits_last - ref_logits_last).abs().flatten(), 0.99).item()
    print(f"[64L] LOGITS PCC (at position {last_idx}) = {pcc_l:.6f} (thresh={_PCC_THRESH_LOGITS}) | p99 = {p99_l:.4f}")

    pred_tok_tt = int(tt_logits_last.argmax().item())
    pred_tok_ref = int(ref_logits_last.argmax().item())
    pred_text_tt = tokenizer.decode([pred_tok_tt])
    pred_text_ref = tokenizer.decode([pred_tok_ref])
    print(f"[64L] predicted next-token: TT id={pred_tok_tt!r} text={pred_text_tt!r}")
    print(f"[64L] predicted next-token: REF id={pred_tok_ref!r} text={pred_text_ref!r}")

    # Per-layer hidden divergence diagnostics in case 64L fails — pull the first
    # 32 features at last token for both TT and ref to look at compounding error.
    diff = (tt_hidden_cpu.float() - hidden_ref[:, :_T_PREFILL, :].float()).abs()
    print(
        f"[64L] hidden abs-diff stats: max={diff.max().item():.4f}, mean={diff.mean().item():.4f}, "
        f"99th-pct-by-token={torch.quantile(diff[0, last_idx, :], 0.99).item():.4f}"
    )

    assert pcc_h > _PCC_THRESH_HIDDEN, f"64L hidden PCC {pcc_h:.4f} < {_PCC_THRESH_HIDDEN} (p99={p99_h:.4f})"
    assert pcc_l > _PCC_THRESH_LOGITS, f"64L logits PCC {pcc_l:.4f} < {_PCC_THRESH_LOGITS} (p99={p99_l:.4f})"
    assert (
        pred_tok_tt == pred_tok_ref or "Paris" in pred_text_tt
    ), f"64L Paris token mismatch: TT={pred_tok_tt!r}({pred_text_tt!r}) vs REF={pred_tok_ref!r}({pred_text_ref!r})"
    print("[64L] PASSED")
