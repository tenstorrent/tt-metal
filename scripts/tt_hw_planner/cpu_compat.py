"""Pure-CPU stand-ins for accelerator-only model dependencies.

Many Hugging Face models hard-import a GPU-only package (e.g. NVIDIA's
``mamba-ssm`` / ``causal-conv1d``) at module-load time, so the model
cannot even be *constructed* on a non-GPU box -- the import raises before
any forward runs. This blocks the bring-up tool from opening the model to
inspect its structure, even though the model itself ships a pure-PyTorch
fallback path that would run fine on CPU.

This module supplies minimal pure-PyTorch implementations of the helpers
those packages expose, and registers them into ``sys.modules`` so the
import succeeds and the model's own CPU path takes over.

Design rules:
  * Vendor-agnostic: the trigger is "an accelerator package is missing",
    not "is this NVIDIA". The fix always retreats to plain CPU/torch.
  * Never shadow a real install: a stand-in is installed only when the
    genuine package is absent.
  * Correctness matters: implementations used on the CPU reference path
    (e.g. ``rmsnorm_fn``) are real math, not no-ops. Fast-path-only
    helpers that the CPU path never calls raise a clear error if invoked.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from typing import List

_MARKER = "_TT_CPU_COMPAT"

_ACCEL_PACKAGES = frozenset(
    {
        "mamba_ssm",
        "causal_conv1d",
        "flash_attn",
        "flash_attn_2",
        "triton",
        "apex",
        "xformers",
        "vllm",
        "flashinfer",
        "deepspeed",
        "mambapy",
    }
)


def _genuinely_importable(name: str) -> bool:
    """True only if ``name`` is a real install (not one of our stand-ins)."""
    mod = sys.modules.get(name)
    if mod is not None:
        return not getattr(mod, _MARKER, False)
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _new_package(fullname: str) -> types.ModuleType:
    mod = types.ModuleType(fullname)
    mod.__path__ = []
    setattr(mod, _MARKER, True)
    return mod


def _new_module(fullname: str) -> types.ModuleType:
    mod = types.ModuleType(fullname)
    setattr(mod, _MARKER, True)
    return mod


def _register(fullname: str, mod: types.ModuleType) -> None:
    sys.modules[fullname] = mod
    if "." in fullname:
        parent_name, _, leaf = fullname.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, mod)


def _rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    import torch
    import torch.nn.functional as F

    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        z = z.float() if z is not None else None
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        out = x * rstd * weight
    else:
        x_g = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
        rstd = torch.rsqrt(x_g.pow(2).mean(-1, keepdim=True) + eps)
        out = (x_g * rstd).reshape_as(x) * weight
    if bias is not None:
        out = out + bias
    if z is not None and norm_before_gate:
        out = out * F.silu(z)
    return out.to(dtype)


def _fast_path_only(name: str):
    def _stub(*args, **kwargs):
        raise RuntimeError(
            f"{name} is a GPU fast-path kernel with no CPU stand-in; the model "
            f"should take its pure-PyTorch path on CPU and never call this. If "
            f"you see this, the model's CPU fallback is not wired as expected."
        )

    return _stub


def _install_mamba_ssm() -> List[str]:
    import torch.nn as nn

    pkg = _new_package("mamba_ssm")
    ops = _new_package("mamba_ssm.ops")
    triton = _new_package("mamba_ssm.ops.triton")
    _register("mamba_ssm", pkg)
    _register("mamba_ssm.ops", ops)
    _register("mamba_ssm.ops.triton", triton)

    layernorm_gated = _new_module("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_gated.rmsnorm_fn = _rmsnorm_fn

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6, group_size=None, norm_before_gate=True, **kw):
            super().__init__()
            import torch

            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            self.group_size = group_size
            self.norm_before_gate = norm_before_gate

        def forward(self, x, z=None):
            return _rmsnorm_fn(x, self.weight, None, z, self.eps, self.group_size, self.norm_before_gate)

    layernorm_gated.RMSNorm = RMSNorm
    _register("mamba_ssm.ops.triton.layernorm_gated", layernorm_gated)

    sel = _new_module("mamba_ssm.ops.triton.selective_state_update")
    sel.selective_state_update = _fast_path_only("selective_state_update")
    _register("mamba_ssm.ops.triton.selective_state_update", sel)

    ssd = _new_module("mamba_ssm.ops.triton.ssd_combined")
    ssd.mamba_chunk_scan_combined = _fast_path_only("mamba_chunk_scan_combined")
    ssd.mamba_split_conv1d_scan_combined = _fast_path_only("mamba_split_conv1d_scan_combined")
    _register("mamba_ssm.ops.triton.ssd_combined", ssd)

    return ["mamba_ssm"]


_PROVIDERS = {
    "mamba_ssm": _install_mamba_ssm,
}


class _HollowCallable:
    def __init__(self, qualname: str) -> None:
        object.__setattr__(self, "_qualname", qualname)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _HollowCallable(f"{object.__getattribute__(self, '_qualname')}.{name}")

    def __call__(self, *args, **kwargs):
        q = object.__getattribute__(self, "_qualname")
        raise RuntimeError(
            f"'{q}' is a hollow CPU stand-in for a missing accelerator package "
            f"and was actually called; this code path needs a real CPU "
            f"implementation of '{q}'."
        )


class _HollowModule(types.ModuleType):
    def __init__(self, fullname: str) -> None:
        super().__init__(fullname)
        self.__path__ = []
        setattr(self, _MARKER, True)

    def __getattr__(self, name: str):
        if name == "__version__":
            return "0.0.0+hollow"
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _HollowCallable(f"{self.__name__}.{name}")


class _AccelHollowFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in _ACCEL_PACKAGES or top in _PROVIDERS:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _HollowModule(spec.name)

    def exec_module(self, module):
        pass


_ACCEL_HOLLOW_FINDER = _AccelHollowFinder()


def install_cpu_compat() -> List[str]:
    """Install pure-CPU stand-ins for any known accelerator package that is
    missing. Returns the list of package names a stand-in was installed for
    (empty if every known package is genuinely present or already stubbed)."""
    installed: List[str] = []
    for name, provider in _PROVIDERS.items():
        if _genuinely_importable(name):
            continue
        if name in sys.modules and getattr(sys.modules[name], _MARKER, False):
            continue
        try:
            installed.extend(provider())
        except Exception as exc:
            print(f"  [cpu-compat] failed to install stand-in for {name!r}: {type(exc).__name__}: {exc}")
    if _ACCEL_HOLLOW_FINDER not in sys.meta_path:
        sys.meta_path.append(_ACCEL_HOLLOW_FINDER)
    return installed
