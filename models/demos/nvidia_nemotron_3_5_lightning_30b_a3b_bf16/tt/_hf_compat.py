# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-loading compatibility shims for NemotronH on a CUDA-less CPU/TT host.

`modeling_nemotron_h.py` hard-imports `mamba_ssm` (a CUDA/Triton package) at
module load and wraps each block in `torch.cuda.stream(...)`. Both raise on
this host. We inject a pure-torch `mamba_ssm` shim and neutralize the cuda
stream calls so `AutoModelForCausalLM.from_pretrained(...)` and the reference
forward run on CPU. Mirrors tests/pcc/conftest.py so the e2e package can load
HF outside pytest collection.
"""
from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import types

import torch
import torch.nn.functional as F


def _neutralize_cuda_stream():
    def _null_stream(*args, **kwargs):
        return contextlib.nullcontext()

    def _null_default_stream(*args, **kwargs):
        return None

    torch.cuda.stream = _null_stream
    torch.cuda.default_stream = _null_default_stream


def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    """Pure-torch equivalent of mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn."""
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1.0 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd) * weight
    else:
        last = x.shape[-1]
        g = last // group_size
        x_group = x.reshape(*x.shape[:-1], g, group_size)
        rstd = 1.0 / torch.sqrt(x_group.square().mean(dim=-1, keepdim=True) + eps)
        out = (x_group * rstd).reshape(*x.shape[:-1], last) * weight
    if bias is not None:
        out = out + bias
    if z is not None and norm_before_gate:
        out = out * F.silu(z)
    return out.to(dtype)


def _raise_fast_path(*args, **kwargs):
    raise RuntimeError(
        "mamba-ssm fast-path kernel invoked under the CPU shim; the pure-torch "
        "path should have been taken (is_fast_path_available must be False)."
    )


def _install_mamba_ssm_shim():
    if "mamba_ssm" in sys.modules and getattr(sys.modules["mamba_ssm"], "_tt_shim", False):
        return

    def _mk(name):
        m = types.ModuleType(name)
        m._tt_shim = True
        m.__path__ = []
        # A real ModuleSpec so transformers' _is_package_available -> find_spec
        # returns a spec (not None, not ValueError); importlib.metadata.version
        # then raises PackageNotFoundError -> is_*_available() == False, so the
        # modeling file takes its pure-torch fallback (fast path stays disabled).
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return m

    mamba_ssm = _mk("mamba_ssm")
    ops = _mk("mamba_ssm.ops")
    triton = _mk("mamba_ssm.ops.triton")

    layernorm_gated = _mk("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_gated.rmsnorm_fn = _rmsnorm_fn

    class _RMSNorm(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, x, *a, **k):
            return x

    layernorm_gated.RMSNorm = _RMSNorm

    selective_state_update = _mk("mamba_ssm.ops.triton.selective_state_update")
    selective_state_update.selective_state_update = _raise_fast_path

    ssd_combined = _mk("mamba_ssm.ops.triton.ssd_combined")
    ssd_combined.mamba_chunk_scan_combined = _raise_fast_path
    ssd_combined.mamba_split_conv1d_scan_combined = _raise_fast_path

    sys.modules["mamba_ssm"] = mamba_ssm
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.triton"] = triton
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"] = layernorm_gated
    sys.modules["mamba_ssm.ops.triton.selective_state_update"] = selective_state_update
    sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd_combined


def _install_causal_conv1d_shim():
    """`modeling_nemotron_h.py` statically imports `causal_conv1d`, so HF's
    check_imports rejects the model when the package is absent even though the
    runtime guards it behind `is_causal_conv1d_available()`. Inject a pure shim
    so check_imports passes; availability still reports False (no metadata), so
    the pure-torch causal-conv path is taken."""
    if "causal_conv1d" in sys.modules and getattr(sys.modules["causal_conv1d"], "_tt_shim", False):
        return
    m = types.ModuleType("causal_conv1d")
    m._tt_shim = True
    m.__path__ = []
    m.__spec__ = importlib.machinery.ModuleSpec("causal_conv1d", loader=None)
    m.causal_conv1d_fn = _raise_fast_path
    m.causal_conv1d_update = _raise_fast_path
    sys.modules["causal_conv1d"] = m


def install_hf_compat():
    """Install all shims. Idempotent. Call before loading the HF model."""
    _install_mamba_ssm_shim()
    _install_causal_conv1d_shim()
    _neutralize_cuda_stream()
