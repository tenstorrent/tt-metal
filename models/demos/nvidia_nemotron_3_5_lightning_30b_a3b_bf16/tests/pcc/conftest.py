# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Conftest for Nemotron tests — provides a mamba-ssm shim for HF model loading.

`modeling_nemotron_h.py` has a module-level hard gate (~line 64):

    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    except ImportError: raise ImportError("mamba-ssm is required ...")

mamba-ssm is a CUDA/Triton package that cannot be installed in this CPU TT
env, so the dynamic import raises before any submodule resolves. We inject a
fake `mamba_ssm` package tree into `sys.modules` AT CONFTEST-IMPORT TIME
(before test collection / model load).

Key correctness point (2026-06-11): `MambaRMSNormGated.forward` calls
`rmsnorm_fn` UNCONDITIONALLY (there is no torch fallback in that module). So
`rmsnorm_fn` must be a real pure-torch impl equivalent to mamba_ssm's
`rms_norm_ref`. The other fast-path fns (selective_state_update,
mamba_chunk_scan_combined, ...) are gated behind `is_fast_path_available`
(never hit on CPU), so they can stay as inert placeholders. We do NOT install
package metadata, so `is_mamba_2_ssm_available()` stays False and the pure
torch path is always taken.
"""

import contextlib
import sys
import types

import torch
import torch.nn.functional as F


def _neutralize_cuda_stream():
    """Make `torch.cuda.stream(...)` / `torch.cuda.default_stream(...)` no-ops.

    `NemotronHBlock.forward` (modeling_nemotron_h.py:769) wraps its entire body
    in `with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):`
    to avoid multi-GPU NaN issues. On this CUDA-less TT host
    `torch.cuda.default_stream(...)` raises, so building the torch reference for
    the per-component PCC test throws → the test SKIPs → it is conflated with a
    real failure and scored OTHER, and the block can never graduate. We replace
    both calls with CPU-safe shims: `default_stream` returns None and `stream`
    returns a null context manager, so the `with` block runs inline on CPU.
    """

    def _null_stream(*args, **kwargs):
        return contextlib.nullcontext()

    def _null_default_stream(*args, **kwargs):
        return None

    torch.cuda.stream = _null_stream
    torch.cuda.default_stream = _null_default_stream


def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    """Pure-torch equivalent of mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn.

    Mirrors `rms_norm_ref`: optional gate (silu) applied before/after norm,
    grouped RMS over `group_size`-wide chunks of the last dim, weight scale.
    """
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
        "mamba-ssm fast-path kernel invoked under the CPU test shim; the pure-torch "
        "path should have been taken (is_fast_path_available must be False)."
    )


def _install_mamba_ssm_shim():
    if "mamba_ssm" in sys.modules and getattr(sys.modules["mamba_ssm"], "_tt_shim", False):
        return

    def _mk(name):
        m = types.ModuleType(name)
        m._tt_shim = True
        m.__path__ = []  # mark as a package so submodule imports resolve
        return m

    mamba_ssm = _mk("mamba_ssm")
    ops = _mk("mamba_ssm.ops")
    triton = _mk("mamba_ssm.ops.triton")

    layernorm_gated = _mk("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_gated.rmsnorm_fn = _rmsnorm_fn

    class _RMSNorm(torch.nn.Module):  # not used on CPU path, present for parity
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


# Install at conftest-import time, before test collection / model load.
_install_mamba_ssm_shim()
_neutralize_cuda_stream()
