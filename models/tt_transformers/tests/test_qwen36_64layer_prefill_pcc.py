# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""64-layer ISL=T prefill PCC — Hybrid TT attention + CPU norms/MLP.

Runs all 64 HybridDecoderLayer layers sequentially:
  * CPU path   : pure float32 reference (gold standard)
  * Hybrid path: TT attention blocks (GDN or full) + CPU norms/MLP

This measures how much attention-error accumulates across the full 64-layer
stack at a given ISL without loading the full 27B checkpoint into RAM —
weights are streamed one layer at a time.

Environment variables:
  QWEN36_SNAPSHOT   : path to Qwen3.5-27B (or Qwen3.6-27B) HF snapshot dir
  QWEN36_64L_T      : prefill sequence length (default 128; set to 8192 for
                      the full ISL-8k test — slow, ~20 min on CPU norms/MLP)
  MESH_DEVICE       : e.g. P150x4 for 4-chip 1D-TP (default P150x4)
  QWEN36_64L_LAYERS : comma-separated subset of layers (e.g. "0,1,2,3" for
                      a quick 4-layer sanity run; default = all 64)

Run (4-chip, quick 128-token sanity check)::

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && MESH_DEVICE=P150x4 \\
           QWEN36_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654 \\
           HF_MODEL=Qwen/Qwen3.5-27B \\
           pytest models/tt_transformers/tests/test_qwen36_64layer_prefill_pcc.py -v -s

Run (4-chip, full ISL-8k)::

    QWEN36_64L_T=8192 pytest ... -v -s
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

# ---------------------------------------------------------------------------
# Constants / env-var config
# ---------------------------------------------------------------------------

_SNAPSHOT = pathlib.Path(
    os.environ.get(
        "QWEN36_SNAPSHOT",
        "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B"
        "/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654",
    )
)
_B = 1
_T = int(os.environ.get("QWEN36_64L_T", "128"))
_H = 5120
# 64-layer composite PCC with BFP8 GDN + BF16 FA. Individual blocks each pass
# 0.99 (GDN: 0.993, FA: 0.9997) but the delta-net SSD scan recurrence means
# per-block errors accumulate over 64 layers. Observed composite at ISL=128:
# ~0.9875. Threshold set to 0.98 to catch regressions while accepting the
# BFP8 GDN floor.
_PCC_THRESH = 0.98

# Optional: test only a subset of layers (comma-separated indices).
_LAYER_SUBSET_ENV = os.environ.get("QWEN36_64L_LAYERS", "")

_MESH_DEVICE_PARAM = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), (1, 4))

_framework_mesh = pytest.mark.parametrize("mesh_device", [_MESH_DEVICE_PARAM], indirect=True)
_framework_device_params = pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)


# ---------------------------------------------------------------------------
# Safetensors helpers
# ---------------------------------------------------------------------------


def _weight_map(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    return idx["weight_map"]


def _load_keys(snapshot_dir: pathlib.Path, wmap: dict, keys: list[str]) -> dict[str, torch.Tensor]:
    files = sorted({wmap[k] for k in keys if k in wmap})
    out = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in keys:
            if k in shard:
                out[k] = shard[k]
    return out


def _load_full_layer(snapshot_dir: pathlib.Path, wmap: dict, layer_idx: int) -> dict[str, torch.Tensor]:
    """Load all weights for layer ``layer_idx`` (attention + MLP + norms)."""
    prefix = f"model.language_model.layers.{layer_idx}."
    keys = [k for k in wmap if k.startswith(prefix)]
    return _load_keys(snapshot_dir, wmap, keys)


def _load_embed(snapshot_dir: pathlib.Path, wmap: dict) -> dict[str, torch.Tensor]:
    """Load embedding weights."""
    keys = [k for k in wmap if "embed_tokens" in k]
    return _load_keys(snapshot_dir, wmap, keys)


# ---------------------------------------------------------------------------
# CPU reference weight loading
# ---------------------------------------------------------------------------


def _ref_state_dict(layer_sd_hf: dict, layer_idx: int, layer_type: str) -> dict:
    """Strip HF prefix from ``layer_sd_hf`` and remap attention prefix for the
    reference ``HybridDecoderLayer.load_state_dict``.

    HF keys: ``model.language_model.layers.N.{input_layernorm,mlp,self_attn,linear_attn}.*``
    Ref keys: ``{input_layernorm,mlp,attention}.*``
    """
    hf_prefix = f"model.language_model.layers.{layer_idx}."
    attn_prefix = "linear_attn." if layer_type == "linear_attention" else "self_attn."

    out = {}
    for k, v in layer_sd_hf.items():
        if not k.startswith(hf_prefix):
            continue
        bare = k[len(hf_prefix) :]
        # Remap attention sub-module prefix to match reference `self.attention.*`
        if bare.startswith(attn_prefix):
            bare = "attention." + bare[len(attn_prefix) :]
        out[bare] = v.float()
    return out


# ---------------------------------------------------------------------------
# TT upload / download
# ---------------------------------------------------------------------------


def _send_hidden(t: torch.Tensor, mesh) -> ttnn.Tensor:
    """Upload [B, T, H] (or [T, H] with unsqueeze) → replicated 4D TT tensor."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, B * T, H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _gather_output(tt_tensor: ttnn.Tensor, mesh, T: int) -> torch.Tensor:
    """Reassemble fractured reduce-scatter output → CPU [T, H]."""
    out = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
    out = out[..., :_H].reshape(-1, _H)[:T]
    return out.float()


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def _rot_mats_for_full_attn(mesh, T: int, rope_theta: float) -> tuple:
    """Build (cos_tt, sin_tt) [1,1,T,64] for full-attention partial RoPE."""
    from models.tt_transformers.tt.rope import get_rot_mats_hf

    return tuple(
        get_rot_mats_hf(
            head_dim=64,  # rope_dim = 256 * 0.25
            device=mesh,
            seq_len=T,
            theta=rope_theta,
            rope_scaling=None,
            datatype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    )


def _mrope_cos_sin_cpu(T: int, config) -> tuple:
    """Build CPU float32 (cos, sin) for the reference GatedAttention."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions_3d = torch.stack([torch.arange(T)] * 3)  # [3, T] — text-only: all same
    return build_mrope_cos_sin(
        positions_3d,
        config.head_dim,
        config.partial_rotary_factor,
        config.mrope_section,
        config.rope_theta,
    )  # (cos, sin) each [1, T, rotary_dim]


def _causal_mask(T: int) -> torch.Tensor:
    """[1, 1, T, T] additive float mask with -inf in upper triangle."""
    m = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# PCC helpers
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return corr


def _abs_err(tt: torch.Tensor, ref: torch.Tensor) -> tuple:
    d = (tt.float() - ref.float()).abs()
    return d.max().item(), d.mean().item()


# ---------------------------------------------------------------------------
# ModelArgs builder (same as single-layer test)
# ---------------------------------------------------------------------------


def _build_args(mesh):
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(mesh, max_batch_size=_B, max_seq_len=max(_T, 2048))
    assert getattr(args, "is_qwen36", False), "ModelArgs did not detect qwen3.6"
    return args


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


@pytest.mark.hardware
@_framework_device_params
@_framework_mesh
def test_qwen36_64layer_prefill_pcc(mesh_device):
    """64-layer hybrid TT (attention on device, norms+MLP on CPU) vs float32 reference."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.qwen36_full_attention import TtQwen36FullAttention
    from models.tt_transformers.tt.qwen36_gdn_attention import TtQwen36GDNAttention

    # ------------------------------------------------------------------ setup
    with open(_SNAPSHOT / "config.json") as f:
        config = Qwen36Config(json.load(f))

    wmap = _weight_map(_SNAPSHOT)
    cos_cpu, sin_cpu = _mrope_cos_sin_cpu(_T, config)
    cmask = _causal_mask(_T)
    rot_mats_tt = _rot_mats_for_full_attn(mesh_device, _T, config.rope_theta)

    args = _build_args(mesh_device)
    tt_ccl = TT_CCL(mesh_device)

    # Random input tokens → initial hidden state
    torch.manual_seed(42)
    embed_raw = _load_embed(_SNAPSHOT, wmap)
    embed_w = embed_raw["model.language_model.embed_tokens.weight"].float()
    input_ids = torch.randint(0, config.vocab_size, (_B, _T))
    x_ref = torch.nn.functional.embedding(input_ids, embed_w)  # [B, T, H]
    x_hybrid = x_ref.clone()

    # Determine which layers to run (default: all)
    if _LAYER_SUBSET_ENV:
        layer_indices = [int(s.strip()) for s in _LAYER_SUBSET_ENV.split(",")]
    else:
        layer_indices = list(range(config.num_hidden_layers))

    print(
        f"\n[64L prefill] T={_T}  layers={len(layer_indices)}  "
        f"mesh={list(mesh_device.shape)}  tp={_MESH_DEVICE_PARAM[-1]}"
    )

    per_layer = []  # (layer_idx, layer_type, pcc)

    for layer_idx in layer_indices:
        layer_type = config.layer_types[layer_idx]
        layer_sd_hf = _load_full_layer(_SNAPSHOT, wmap, layer_idx)

        # ------------------------------------------------ CPU reference path
        ref_sd = _ref_state_dict(layer_sd_hf, layer_idx, layer_type)
        ref_layer = HybridDecoderLayer(config, layer_idx).eval()
        missing, unexpected = ref_layer.load_state_dict(ref_sd, strict=False)
        assert not [
            m for m in missing if "conv1d.bias" not in m
        ], f"Layer {layer_idx} ref missing weights: {[m for m in missing if 'conv1d.bias' not in m]}"

        with torch.no_grad():
            x_ref, _kv, _cv, _rv = ref_layer(
                x_ref,
                cos=cos_cpu,
                sin=sin_cpu,
                attention_mask=(cmask if layer_type == "full_attention" else None),
            )

        # ------------------------------------------------ Hybrid TT path
        # Apply CPU input_layernorm (same weights as reference → exact norm)
        with torch.no_grad():
            x_norm = ref_layer.input_layernorm(x_hybrid)  # [B, T, H]

        # Build TT attention block for this layer
        if layer_type == "linear_attention":
            tt_attn = TtQwen36GDNAttention(
                mesh_device=mesh_device,
                args=args,
                layer_num=layer_idx,
                dtype=ttnn.bfloat8_b,
                state_dict=layer_sd_hf,
                tt_ccl=tt_ccl,
            )
            x_norm_tt = _send_hidden(x_norm[0], mesh_device)  # [T, H] → TT
            tt_out = tt_attn.forward(x_norm_tt, mode="prefill")
        else:
            tt_attn = TtQwen36FullAttention(
                mesh_device=mesh_device,
                args=args,
                layer_num=layer_idx,
                state_dict=layer_sd_hf,
                tt_ccl=tt_ccl,
            )
            x_norm_tt = _send_hidden(x_norm[0], mesh_device)  # [T, H] → TT
            tt_out = tt_attn.forward(x_norm_tt, rot_mats=rot_mats_tt, mode="prefill")

        attn_cpu = _gather_output(tt_out, mesh_device, _T)  # [T, H]
        tt_out.deallocate(True)
        x_norm_tt.deallocate(True)
        del tt_attn  # free device buffers

        # Residual add
        x_hybrid = x_hybrid + attn_cpu.unsqueeze(0)  # [B, T, H]

        # CPU post_attention_layernorm + MLP
        with torch.no_grad():
            x_post = ref_layer.post_attention_layernorm(x_hybrid)
            x_hybrid = x_hybrid + ref_layer.mlp(x_post)

        # PCC at this layer boundary
        pcc = _pcc(x_hybrid.reshape(-1), x_ref.reshape(-1))
        max_err, mae = _abs_err(x_hybrid, x_ref)
        per_layer.append((layer_idx, layer_type, pcc))
        tag = "GDN" if layer_type == "linear_attention" else " FA"
        print(f"  layer {layer_idx:2d} [{tag}] PCC={pcc:.6f}  max_abs={max_err:.4f}  mae={mae:.4f}")

        # Discard reference layer to free RAM before loading next
        del ref_layer, ref_sd, layer_sd_hf

    # ---------------------------------------------------------- final summary
    final_pcc = per_layer[-1][2]
    gdn_pccs = [p for _, t, p in per_layer if t == "linear_attention"]
    fa_pccs = [p for _, t, p in per_layer if t == "full_attention"]

    print(f"\n[64L prefill @ T={_T}]  final hidden-state PCC = {final_pcc:.6f}")
    if gdn_pccs:
        print(f"  GDN layers: min={min(gdn_pccs):.6f}  mean={sum(gdn_pccs)/len(gdn_pccs):.6f}  max={max(gdn_pccs):.6f}")
    if fa_pccs:
        print(f"  FA  layers: min={min(fa_pccs):.6f}  mean={sum(fa_pccs)/len(fa_pccs):.6f}  max={max(fa_pccs):.6f}")

    # Report the 5 worst layers to guide debugging
    worst = sorted(per_layer, key=lambda x: x[2])[:5]
    print("  worst 5 layers:")
    for li, lt, p in worst:
        print(f"    layer {li:2d} [{lt[:3]}]  PCC={p:.6f}")

    assert final_pcc > _PCC_THRESH, f"64-layer prefill PCC {final_pcc:.4f} < {_PCC_THRESH} at T={_T}"
