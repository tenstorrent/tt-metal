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


def _neutralize_cuda_stream_apis() -> bool:
    """When torch has no CUDA, make ``torch.cuda.stream`` / ``default_stream`` /
    ``current_stream`` no-ops so models that unconditionally wrap their forward
    in ``with torch.cuda.stream(torch.cuda.default_stream(x.device)):`` (a
    common HF Mamba pattern) run on CPU instead of raising "Torch not compiled
    with CUDA enabled". No-op on a real CUDA host, and idempotent."""
    try:
        import torch
    except Exception:
        return False
    try:
        if torch.cuda.is_available():
            return False
    except Exception:
        pass
    import contextlib

    def _noop_stream(*_a, **_k):
        return contextlib.nullcontext()

    def _noop_default_stream(*_a, **_k):
        return None

    patched = False
    for _name, _fn in (
        ("stream", _noop_stream),
        ("default_stream", _noop_default_stream),
        ("current_stream", _noop_default_stream),
    ):
        try:
            _cur = getattr(torch.cuda, _name, None)
            if _cur is not None and getattr(_cur, _MARKER, False):
                continue
            setattr(_fn, _MARKER, True)
            setattr(torch.cuda, _name, _fn)
            patched = True
        except Exception:
            pass
    return patched


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
    _neutralize_cuda_stream_apis()
    return installed


class UnrepairableCPUError(RuntimeError):
    """Raised when a callable crashes with a CPU incompatibility that no
    registered remedy can match. The message names the exact signature so a new
    remedy is a one-line registry addition (never a silent mask)."""


def _cuda_absent() -> bool:
    try:
        import torch

        return not torch.cuda.is_available()
    except Exception:
        return True


def _missing_top_module(exc, tb_text=""):
    import re

    nm = getattr(exc, "name", None)
    if nm:
        return nm.split(".")[0]
    s = (str(exc) if exc is not None else "") + " " + (tb_text or "")
    m = re.search(r"['\"]?([a-zA-Z_][\w.\-]+)['\"]?\s+(?:is required|cannot be imported|is needed)", s)
    if not m:
        m = re.search(r"No module named ['\"]([\w.]+)['\"]", s)
    return m.group(1).split(".")[0].replace("-", "_") if m else None


def _looks_like_accel(name) -> bool:
    import re

    if not name:
        return False
    if name in _ACCEL_PACKAGES or name in _PROVIDERS:
        return True
    return bool(
        re.search(
            r"(flash|triton|cuda|mamba|conv1d|xformer|vllm|flashinfer|deepspeed|apex|cutlass|nccl|bitsandbytes|selective_scan|natten|causal|fused)",
            name,
            re.I,
        )
    )


def _remedy_accel_import_match(exc, tb):
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return _looks_like_accel(_missing_top_module(exc, tb))
    s = (tb or "").lower() + " " + str(exc or "").lower()
    if any(k in s for k in ("importerror", "modulenotfound", "no module named", "cannot be imported", "is required")):
        return _looks_like_accel(_missing_top_module(exc, tb))
    return False


def _remedy_accel_import_apply(exc, tb):
    install_cpu_compat()
    return True


def _remedy_cuda_stream_match(exc, tb):
    s = (str(exc or "") + " " + (tb or "")).lower()
    if "torch not compiled with cuda" in s:
        return True
    return "cuda" in s and any(
        k in (tb or "")
        for k in ("torch.cuda.stream", "default_stream", "current_stream", "cuda.synchronize", "cuda.Event")
    )


def _remedy_cuda_stream_apply(exc, tb):
    return bool(_neutralize_cuda_stream_apis())


def _remedy_tensor_cuda_match(exc, tb):
    if not _cuda_absent():
        return False
    s = (str(exc or "") + " " + (tb or "")).lower()
    return ("torch not compiled with cuda" in s or "no cuda" in s) and (
        ".cuda(" in (tb or "")
        or "to('cuda" in (tb or "")
        or 'to("cuda' in (tb or "")
        or "device='cuda" in (tb or "")
        or 'device="cuda' in (tb or "")
    )


def _remedy_tensor_cuda_apply(exc, tb):
    if not _cuda_absent():
        return False
    try:
        import torch
    except Exception:
        return False
    if getattr(torch.Tensor.cuda, _MARKER, False):
        return False
    _orig_to = torch.Tensor.to

    def _cuda_noop(self, *a, **k):
        return self

    def _to_cpu(self, *a, **k):
        a = tuple("cpu" if (isinstance(x, str) and "cuda" in x) else x for x in a)
        if isinstance(k.get("device"), str) and "cuda" in k["device"]:
            k["device"] = "cpu"
        return _orig_to(self, *a, **k)

    setattr(_cuda_noop, _MARKER, True)
    torch.Tensor.cuda = _cuda_noop
    torch.Tensor.to = _to_cpu
    return True


def _remedy_cuda_autocast_match(exc, tb):
    s = (str(exc or "") + " " + (tb or "")).lower()
    return "autocast" in s and ("cuda" in s or "torch not compiled with cuda" in s)


def _remedy_cuda_autocast_apply(exc, tb):
    if not _cuda_absent():
        return False
    import contextlib

    try:
        import torch
    except Exception:
        return False
    if getattr(getattr(torch.cuda.amp, "autocast", None), _MARKER, False):
        return False

    def _nullac(*a, **k):
        return contextlib.nullcontext()

    setattr(_nullac, _MARKER, True)
    try:
        torch.cuda.amp.autocast = _nullac
    except Exception:
        return False
    return True


_CPU_REPAIR_REMEDIES = [
    ("accel-import", _remedy_accel_import_match, _remedy_accel_import_apply),
    ("cuda-stream", _remedy_cuda_stream_match, _remedy_cuda_stream_apply),
    ("tensor-cuda", _remedy_tensor_cuda_match, _remedy_tensor_cuda_apply),
    ("cuda-autocast", _remedy_cuda_autocast_match, _remedy_cuda_autocast_apply),
]


def cpu_repair_remedy_for(exc, tb_text=""):
    """Return the name of the first remedy matching (exc, tb_text), else None.
    Inspection only -- applies nothing. Lets callers that catch their own
    exceptions (e.g. capture's per-driver try/except that swallows into a
    forward-errors string) decide whether to remedy-and-retry."""
    for name, match, _apply in _CPU_REPAIR_REMEDIES:
        try:
            if match(exc, tb_text):
                return name
        except Exception:
            continue
    return None


def apply_cpu_repair(name, exc=None, tb_text="") -> bool:
    """Apply the named remedy. Returns True if it changed anything."""
    for _name, _match, apply in _CPU_REPAIR_REMEDIES:
        if _name == name:
            try:
                return bool(apply(exc, tb_text))
            except Exception:
                return False
    return False


_LLM_CPU_REMEDY_PROMPT = """A HuggingFace model failed while running on a CPU-only host (no CUDA / no GPU). \
The cause is almost always that the model code assumes a GPU: it calls a CUDA-only torch API \
(torch.cuda.stream, torch.cuda.default_stream, torch.cuda.synchronize, tensor.cuda(), \
torch.autocast('cuda'), device='cuda'), or imports a GPU-only accelerator package whose name \
clearly signals GPU/accelerator use.

Diagnose THIS specific failure and propose a MINIMAL Python remedy that monkeypatches / neutralizes \
the offending GPU-only infrastructure so the SAME call succeeds on CPU.

STRICT RULES:
- Only neutralize GPU/accelerator INFRASTRUCTURE. On CPU these are meaningless, so a no-op / \
contextlib.nullcontext / cpu-redirect is correct.
- NEVER change model math. NEVER mask a genuine missing non-GPU dependency or a real logic bug. \
If this error is NOT clearly a GPU/CPU incompatibility, your function MUST raise \
NotImplementedError("not a CPU/GPU compatibility issue").
- Only patch when CUDA is genuinely absent (guard with `not torch.cuda.is_available()`). Idempotent.

Return ONLY one Python code block defining exactly this function (imports INSIDE the function):

```python
def apply_remedy() -> bool:
    ...
    return True
```

--- THE FAILURE ---
{sig}

--- TRACEBACK (tail) ---
{tb}
"""


def _llm_available() -> bool:
    import shutil

    return shutil.which("claude") is not None


def llm_propose_cpu_remedy(signature_text, tb_text="", model="sonnet", timeout=220):
    """LAST-RESORT intelligence: when no registry remedy matches, ask an LLM to
    diagnose the CPU/GPU incompatibility and return Python code defining
    ``apply_remedy() -> bool``. Returns the code string, or None if the LLM is
    unavailable or gave nothing usable. The LLM is instructed to RAISE
    NotImplementedError for non-GPU issues so genuine bugs are never masked."""
    import re
    import subprocess

    if not _llm_available():
        return None
    prompt = _LLM_CPU_REMEDY_PROMPT.format(sig=str(signature_text)[:400], tb=(tb_text or "")[-3000:])
    try:
        r = subprocess.run(
            ["claude", "-p", "--model", model], input=prompt, capture_output=True, text=True, timeout=timeout
        )
    except Exception:
        return None
    out = r.stdout or ""
    m = re.search(r"```(?:python)?\s*(.*?)```", out, re.S)
    code = (m.group(1) if m else out).strip()
    return code if "def apply_remedy" in code else None


def apply_llm_cpu_remedy(code) -> bool:
    """Exec an LLM-proposed remedy and run its ``apply_remedy()``. Returns
    whether it changed anything; False if the LLM refused (NotImplementedError)
    or the code errored. Exec of LLM-proposed code matches the tool's existing
    agentic trust model (it already runs claude with file write/edit under
    ``--dangerously-skip-permissions``) and is always verified by re-running the
    caller's operation -- a wrong remedy simply crashes again and is discarded."""
    if not code:
        return False
    ns: dict = {}
    try:
        exec(code, ns)
        fn = ns.get("apply_remedy")
        if fn is None:
            return False
        return bool(fn())
    except NotImplementedError:
        return False
    except Exception:
        return False


def run_with_cpu_repair(fn, max_repairs=8, verbose=False, use_llm=False, llm_model="sonnet", max_llm=3):
    """Run ``fn()``; on a CPU-incompatibility crash auto-apply the matching
    remedy and retry (bounded). Returns ``(result, applied_remedy_names)``.

    With ``use_llm=True``, when NO registry remedy matches, an LLM is asked to
    diagnose-and-propose a remedy (intelligence beyond the registry); it is
    applied and verified by re-running ``fn`` (a wrong proposal just crashes
    again). The LLM path is gated on the ``claude`` CLI being available and
    bounded by ``max_llm``. Re-raises :class:`UnrepairableCPUError` -- naming the
    exact signature -- only when BOTH the registry and (optionally) the LLM are
    exhausted, so genuine missing deps / real bugs are surfaced, never masked.
    CPU-only-gated remedies are no-ops on a real CUDA host."""
    import traceback

    applied = []
    while True:
        try:
            return fn(), applied
        except Exception as exc:
            tb = traceback.format_exc()
            name = None
            for _name, match, _apply in _CPU_REPAIR_REMEDIES:
                if _name in applied:
                    continue
                try:
                    if match(exc, tb):
                        name = _name
                        break
                except Exception:
                    continue
            if name is None:
                if use_llm and applied.count("llm") < max_llm:
                    _code = llm_propose_cpu_remedy(f"{type(exc).__name__}: {str(exc)[:300]}", tb, model=llm_model)
                    if _code and apply_llm_cpu_remedy(_code):
                        applied.append("llm")
                        if verbose:
                            print("  [cpu-repair] no registry match -> LLM-proposed remedy applied; retrying ...")
                        if len(applied) > max_repairs + max_llm:
                            raise UnrepairableCPUError(f"exceeded repair budget; applied={applied}") from exc
                        continue
                raise UnrepairableCPUError(
                    f"no CPU-compat remedy matched; add one for signature -> " f"{type(exc).__name__}: {str(exc)[:160]}"
                ) from exc
            changed = apply_cpu_repair(name, exc, tb)
            applied.append(name)
            if verbose:
                print(f"  [cpu-repair] applied {name!r} (changed={changed}); retrying ...")
            if len(applied) > max_repairs + max_llm:
                raise UnrepairableCPUError(f"exceeded repair budget; applied={applied}") from exc
