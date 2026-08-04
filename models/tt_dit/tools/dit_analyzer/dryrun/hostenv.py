# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Making a laptop interpreter able to import ``models.tt_dit``.

Nothing here touches the graph. It arranges for the host-side imports tt_dit
performs at module level to succeed, prefers real torch on ``device='meta'`` for
weights, and reports what it had to substitute so a run is never quietly less
faithful than it looks.
"""

from __future__ import annotations

import os
import sys
import types
from typing import List, Optional, Sequence

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))

#: What `ensure_host_env` had to stand in for, reported by `ditcheck dryrun`.
SUBSTITUTIONS: List[str] = []

_TORCH: Optional[types.ModuleType] = None


def ensure_host_env() -> types.ModuleType:
    """Import-path prerequisites, in order of preference: real, then metadata-only.

    Deliberately does *not* touch anything under ``models.``: importing tt_dit's own
    modules must wait until the shim is installed as ``ttnn`` -- see
    :func:`ensure_model_env`.
    """
    global _TORCH
    if _TORCH is not None:
        return _TORCH

    if sys.version_info < (3, 10):
        _install_annotation_hook()
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    _TORCH = _ensure_torch()
    _ensure_module("loguru", _make_loguru)
    _ensure_module("safetensors", _make_safetensors)
    _ensure_module("typing_extensions", _make_typing_extensions)
    return _TORCH


def ensure_model_env() -> None:
    """Prerequisites that import `models.*`, run **after** the shim is installed.

    `models.common.utility_functions` imports `ttnn` at module level, so probing it
    too early would pull *real* ttnn into ``sys.modules`` on any machine that has it
    — and then `install()` would refuse to run at all. Probed here, it resolves
    against the shim, and only falls back to a stub if it cannot be imported (it
    also wants numpy and pytest, which tt_dit does not need for `is_blackhole`).
    """
    _ensure_module("models.common.utility_functions", _make_utility_functions)


def have_real_torch() -> bool:
    return bool(_TORCH) and getattr(_TORCH, "__file__", None) is not None


def host_tensor(shape: Sequence[int], dtype=None):
    """A shape without bytes: a torch meta tensor, or the metadata-only stand-in."""
    torch = ensure_host_env()
    if have_real_torch():
        return torch.empty(tuple(int(d) for d in shape), dtype=dtype or torch.bfloat16, device="meta")
    return torch.empty(list(shape), dtype=dtype or torch.bfloat16)


#: ttnn dtype name -> torch dtype name. Block-float formats have no torch
#: equivalent; real code hands ttnn a bf16/fp32 tensor and lets it convert.
_TORCH_DTYPES = {
    "bfloat16": "bfloat16",
    "bfloat8_b": "bfloat16",
    "bfloat4_b": "bfloat16",
    "float32": "float32",
    "uint16": "int32",
    "uint32": "int32",
    "int32": "int32",
}


def torch_dtype_for(ttnn_dtype):
    torch = ensure_host_env()
    name = _TORCH_DTYPES.get(getattr(ttnn_dtype, "name", "bfloat16"), "bfloat16")
    return getattr(torch, name)


# -----------------------------------------------------------------------------
# substitutions
# -----------------------------------------------------------------------------
def _ensure_torch() -> types.ModuleType:
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError:
        from . import hostfakes  # noqa: PLC0415

        SUBSTITUTIONS.append("torch: metadata-only stand-in (no torch on this interpreter)")
        return hostfakes.install()


def _ensure_module(name: str, make) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    make()


def _make_loguru() -> None:
    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    loguru = types.ModuleType("loguru")
    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru
    SUBSTITUTIONS.append("loguru: silent logger")


def _make_safetensors() -> None:
    st = types.ModuleType("safetensors")
    st.safe_open = lambda *a, **k: None
    st_torch = types.ModuleType("safetensors.torch")
    st_torch.load_file = lambda *a, **k: {}
    st.torch = st_torch
    sys.modules["safetensors"] = st
    sys.modules["safetensors.torch"] = st_torch
    SUBSTITUTIONS.append("safetensors: no checkpoint reader (a dry run loads no weights)")


def _make_typing_extensions() -> None:
    te = types.ModuleType("typing_extensions")
    te.deprecated = lambda *a, **k: (lambda obj: obj)
    te.Self = object
    sys.modules["typing_extensions"] = te
    SUBSTITUTIONS.append("typing_extensions: deprecated/Self only")


def _make_utility_functions() -> None:
    """`models.common.utility_functions` pulls in numpy *and pytest*; tt_dit wants
    `is_blackhole`. Worth splitting upstream (spike finding)."""
    for name in ("models", "models.common"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(REPO, *name.split("."))]
            sys.modules[name] = mod
    uf = types.ModuleType("models.common.utility_functions")
    uf.is_blackhole = lambda *a, **k: _arch_is("blackhole")
    uf.is_wormhole_b0 = lambda *a, **k: _arch_is("wormhole_b0")
    uf.nearest_32 = lambda x: -(-int(x) // 32) * 32
    uf.nearest_y = lambda x, y: -(-int(x) // y) * y
    sys.modules["models.common.utility_functions"] = uf
    SUBSTITUTIONS.append("models.common.utility_functions: arch predicates read the dry-run mesh")


def _arch_is(arch: str) -> bool:
    """Arch predicates answer from the mesh under test, not from real hardware."""
    from .context import CTX

    return CTX.arch == arch


def _install_annotation_hook() -> None:
    """Compile tt_dit sources with ``from __future__ import annotations``.

    tt_dit targets Python >= 3.10 and uses PEP 604 unions (``ttnn.Tensor | None``)
    in *evaluated* annotation positions. Stringifying annotations at compile time
    lets an older interpreter import it; a run on the repo's own interpreter never
    reaches this.
    """
    import __future__  # noqa: PLC0415
    import importlib.abc  # noqa: PLC0415
    import importlib.machinery  # noqa: PLC0415

    flag = __future__.annotations.compiler_flag
    sys.dont_write_bytecode = True  # and never read a .pyc compiled without the flag

    if not hasattr(types, "NoneType"):  # 3.10+; utils/tracing.py imports it
        types.NoneType = type(None)

    class _Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = self.get_filename(fullname)
            return compile(self.get_data(path), path, "exec", dont_inherit=True, flags=flag)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("models.tt_dit"):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                return spec
            spec.loader = _Loader(spec.loader.name, spec.loader.path)
            return spec

    sys.meta_path.insert(0, _Finder())
    SUBSTITUTIONS.append(
        "python %d.%d: tt_dit compiled with postponed annotations (repo targets 3.10+)"
        % (sys.version_info[0], sys.version_info[1])
    )
