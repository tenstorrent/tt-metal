#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Generate golden tensor for Mamba2Layer and verify PCC against HF model.

Usage:
    source python_env/bin/activate && export PYTHONPATH=$(pwd)
    python models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/reference/generate_mamba2_golden.py
"""

import importlib
import importlib.util
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

SNAP = (
    "/home/ttuser/.cache/huggingface/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/"
    "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(GOLDEN_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Inject mock mamba_ssm so modeling_nemotron_h.py can be imported without the
# actual CUDA kernels.  We only need rmsnorm_fn which is called from the
# MambaRMSNormGated.forward().  We implement it in pure PyTorch.
# ---------------------------------------------------------------------------


def _rmsnorm_fn_mock(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=True):
    """Pure-PyTorch drop-in for mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn.

    NemotronH calls this with norm_before_gate=False (gate x first, then norm).
    """
    B_S = x.shape[:-1]
    D = x.shape[-1]

    if group_size is None:
        group_size = D

    x_f = x.float()
    if z is not None and not norm_before_gate:
        # gate first: x = x * silu(z)
        x_f = x_f * F.silu(z.float())

    # per-group RMSNorm
    x_grouped = x_f.view(*B_S, -1, group_size)
    var = x_grouped.pow(2).mean(-1, keepdim=True)
    x_normed = x_grouped * torch.rsqrt(var + eps)
    x_normed = x_normed.view(*B_S, D)

    if z is not None and norm_before_gate:
        # gate after norm
        x_normed = x_normed * F.silu(z.float())

    out = weight.float() * x_normed
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


# Build fake mamba_ssm package hierarchy
def _make_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


_mamba_ssm = _make_module("mamba_ssm")
_mamba_ssm_ops = _make_module("mamba_ssm.ops")
_mamba_ssm_ops_triton = _make_module("mamba_ssm.ops.triton")
_mamba_ssm_ops_triton_layernorm = _make_module("mamba_ssm.ops.triton.layernorm_gated")
_mamba_ssm_ops_triton_layernorm.rmsnorm_fn = _rmsnorm_fn_mock
_mamba_ssm_ops_triton_ssd = _make_module("mamba_ssm.ops.triton.ssd_combined")
_mamba_ssm_ops_triton_ssd.mamba_chunk_scan_combined = None
_mamba_ssm_ops_triton_ssd.mamba_split_conv1d_scan_combined = None
_mamba_ssm_ops_triton_selective = _make_module("mamba_ssm.ops.triton.selective_state_update")
_mamba_ssm_ops_triton_selective.selective_state_update = None

# causal_conv1d mock (not needed for torch_forward path)
_causal_conv1d = _make_module("causal_conv1d")
_causal_conv1d.causal_conv1d_fn = None
_causal_conv1d.causal_conv1d_update = None

# ---------------------------------------------------------------------------
# Load HF configuration + modeling modules via importlib to handle relative
# imports inside the snapshot directory.
# ---------------------------------------------------------------------------
snap_pkg = "nemotron_h_snap"
pkg = types.ModuleType(snap_pkg)
pkg.__path__ = [SNAP]
pkg.__package__ = snap_pkg
sys.modules[snap_pkg] = pkg


def _load_snap_module(module_name, filename):
    """Load a module from the snapshot directory as part of snap_pkg."""
    full_name = f"{snap_pkg}.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, os.path.join(SNAP, filename))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = snap_pkg
    sys.modules[full_name] = mod
    # Also register under bare module name so cross-module imports work
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mod_cfg = _load_snap_module("configuration_nemotron_h", "configuration_nemotron_h.py")
mod_mod = _load_snap_module("modeling_nemotron_h", "modeling_nemotron_h.py")

NemotronHMamba2Mixer = mod_mod.NemotronHMamba2Mixer
NemotronHConfig = mod_cfg.NemotronHConfig

# ---------------------------------------------------------------------------
# Import our reference function
# ---------------------------------------------------------------------------
_ref_dir = os.path.dirname(os.path.abspath(__file__))
_demo_dir = os.path.dirname(_ref_dir)
if _demo_dir not in sys.path:
    sys.path.insert(0, _demo_dir)
from reference.functional import layer_norm, mamba2_layer

# ---------------------------------------------------------------------------
# Load weights from shard (only layer 0 mixer + pre-norm)
# ---------------------------------------------------------------------------
from safetensors.torch import load_file

shard = load_file(f"{SNAP}/model-00001-of-00013.safetensors")

norm_weight = shard["backbone.layers.0.norm.weight"]
in_proj_weight = shard["backbone.layers.0.mixer.in_proj.weight"]
conv1d_weight = shard["backbone.layers.0.mixer.conv1d.weight"]
conv1d_bias_w = shard["backbone.layers.0.mixer.conv1d.bias"]
dt_bias = shard["backbone.layers.0.mixer.dt_bias"]
A_log = shard["backbone.layers.0.mixer.A_log"]
norm_mixer_weight = shard["backbone.layers.0.mixer.norm.weight"]
D_param = shard["backbone.layers.0.mixer.D"]
out_proj_weight = shard["backbone.layers.0.mixer.out_proj.weight"]

# ---------------------------------------------------------------------------
# Generate a small deterministic random input
# ---------------------------------------------------------------------------
torch.manual_seed(42)
x = torch.randn(1, 32, 2688, dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# Run our reference function
# ---------------------------------------------------------------------------
with torch.no_grad():
    y_ref = mamba2_layer(
        hidden_states=x,
        norm_weight=norm_weight,
        in_proj_weight=in_proj_weight,
        conv1d_weight=conv1d_weight,
        conv1d_bias=conv1d_bias_w,
        dt_bias=dt_bias,
        A_log=A_log,
        norm_mixer_weight=norm_mixer_weight,
        D=D_param,
        out_proj_weight=out_proj_weight,
        norm_eps=1e-5,
        num_heads=64,
        head_dim=64,
        n_groups=8,
        ssm_state_size=128,
        chunk_size=128,
    )

# ---------------------------------------------------------------------------
# Save golden tensor
# ---------------------------------------------------------------------------
golden_path = os.path.join(GOLDEN_DIR, "Mamba2Layer.pt")
torch.save({"input": x, "output": y_ref}, golden_path)

# ---------------------------------------------------------------------------
# PCC verification against HF torch_forward
# ---------------------------------------------------------------------------
cfg = NemotronHConfig.from_pretrained(SNAP, trust_remote_code=True)
mixer = NemotronHMamba2Mixer(cfg, layer_idx=0)

# Load state dict: rename backbone.layers.0.mixer.* -> bare key
state = {k.replace("backbone.layers.0.mixer.", ""): v for k, v in shard.items() if "backbone.layers.0.mixer." in k}
mixer.load_state_dict(state, strict=True)
mixer.eval()

# Apply pre-norm, run HF torch_forward, add residual
with torch.no_grad():
    normed_hf = layer_norm(x, norm_weight, eps=1e-5)
    mixer_out_hf = mixer.torch_forward(
        normed_hf.float(),
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    )
    y_hf = x + mixer_out_hf.to(x.dtype)


# ---------------------------------------------------------------------------
# Compute PCC
# ---------------------------------------------------------------------------
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a_m = a.mean()
    b_m = b.mean()
    num = ((a - a_m) * (b - b_m)).sum()
    den = torch.sqrt(((a - a_m) ** 2).sum() * ((b - b_m) ** 2).sum())
    if den.item() == 0.0:
        return 1.0 if num.item() == 0.0 else 0.0
    return (num / den).item()


pcc_val = pcc(y_ref, y_hf)
status = "ok" if pcc_val >= 0.99 else "failing"

result = {
    "block": "Mamba2Layer",
    "status": status,
    "pcc": round(pcc_val, 6),
    "notes": (
        f"Reference vs HF torch_forward (pre-norm+mixer+residual). "
        f"Input shape: {list(x.shape)}, dtype: {x.dtype}. "
        f"Shard: model-00001-of-00013.safetensors layer 0."
    ),
}

print(json.dumps(result))
