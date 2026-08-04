# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Installing the shim as ``ttnn`` by import shadowing.

Model code is not edited and does not know it is being analysed: it imports
``ttnn`` and gets this. The shadowing is deliberately loud -- it refuses to
displace a real ttnn, marks the module it installs, and :func:`assert_installed`
lets anything that is about to emit a graph check that it is talking to the shim.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, Sequence

from . import ops, stubs
from .context import CTX
from .tensor import Shape, Tensor

_MODULES = ("ttnn", "ttnn.experimental", "ttnn.transformer", "ttnn.device", "ttnn.operations", "ttnn.distributed")
_SAVED: Dict[str, Any] = {}


class ShimModule(types.ModuleType):
    """Explicit ops where there are semantics; recorded misses everywhere else."""

    def __init__(self, name: str, members: Dict[str, Any]):
        super().__init__(name)
        self.__file__ = __file__
        self.__dryrun__ = True
        for key, value in members.items():
            setattr(self, key, value)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in ops.NOOPS:
            return lambda *a, **k: None
        if name[:1].isupper():  # a config object or enum the model builds and passes on
            return stubs.Stub(self.__name__ + "." + name)

        call = self.__name__ + "." + name

        def maybe_op(*a, **k):
            args = list(a) + list(k.values())
            if _carries_tensor(args):
                return recorder_unregistered(call, args)
            # No tensor in or out: device bookkeeping, not dataflow.
            return stubs.Stub(call)

        return maybe_op


def _carries_tensor(args: Sequence[Any]) -> bool:
    for x in args:
        if isinstance(x, Tensor):
            return True
        if isinstance(x, (list, tuple)) and any(isinstance(y, Tensor) for y in x):
            return True
    return False


def recorder_unregistered(call: str, args: Sequence[Any]):
    from . import recorder

    return recorder.unregistered(call, args)


# -----------------------------------------------------------------------------
# module assembly
# -----------------------------------------------------------------------------
def _ttnn_namespace() -> Dict[str, Any]:
    members: Dict[str, Any] = dict(ops.TENSOR_OPS)
    for name in dir(stubs):
        if name.startswith("_") or name == "CTX":
            continue
        value = getattr(stubs, name)
        # Everything stubs.py defines itself, and nothing it merely imported.
        if getattr(value, "__module__", None) in (stubs.__name__, None):
            members[name] = value
    members.update(
        {
            "Tensor": Tensor,
            "Shape": Shape,
            "Device": stubs.MeshDevice,
            "MeshDevice": stubs.MeshDevice,
            "TILE_SIZE": stubs.TILE_SIZE,
            # Real ttnn spells these both ways; `Parameter` defaults use the
            # namespaced form and `_check_data` compares by value.
            "Layout": stubs.Layout,
            "DataType": stubs.DataType,
            "MemoryConfig": stubs.Stub("MemoryConfig"),
            "ShardSpec": stubs.Stub("ShardSpec"),
            "CoreRange": stubs.Stub("CoreRange"),
            "CoreRangeSet": stubs.Stub("CoreRangeSet"),
            "SubDevice": stubs.Stub("SubDevice"),
            "get_memory_config": lambda *a, **k: stubs.DRAM_MEMORY_CONFIG,
            # `models.common.utility_functions.is_blackhole` reads this, and the
            # model keys chunk sizes and program configs off it. Left to the generic
            # stub it would answer "not blackhole" for every mesh, silently.
            "get_arch_name": lambda *a, **k: CTX.arch,
            "distributed_context_get_rank": lambda *a, **k: 0,
            "using_distributed_env": lambda *a, **k: False,
            "get_num_devices": lambda *a, **k: CTX.mesh_device.get_num_devices() if CTX.mesh_device else 1,
        }
    )
    return members


def _device_module() -> ShimModule:
    mod = ShimModule(
        "ttnn.device",
        {
            "Arch": stubs.Namespace(
                "ttnn.device.Arch", BLACKHOLE=stubs.Enum("BLACKHOLE"), WORMHOLE_B0=stubs.Enum("WORMHOLE_B0")
            ),
            "is_blackhole": lambda *a, **k: _arch_is("blackhole"),
            "is_wormhole_b0": lambda *a, **k: _arch_is("wormhole_b0"),
        },
    )
    return mod


def _arch_is(arch: str) -> bool:
    return CTX.arch == arch


def build_ttnn() -> ShimModule:
    """The shim module tree, not yet installed."""
    ttnn = ShimModule("ttnn", _ttnn_namespace())
    ttnn.experimental = ShimModule("ttnn.experimental", ops.EXPERIMENTAL_OPS)
    ttnn.transformer = ShimModule("ttnn.transformer", ops.TRANSFORMER_OPS)
    ttnn.device = _device_module()
    ttnn.operations = ShimModule(
        "ttnn.operations",
        {
            # Group-norm core-grid helpers: they size program configs and the
            # (opaque) norm weight/mask tensors, not any activation the analyzer
            # reasons about, so a benign 1 keeps the forward running. Exact values
            # are a conformance concern, not a redundancy one.
            "normalization": ShimModule(
                "ttnn.operations.normalization",
                {
                    "dram_group_norm_virtual_columns": lambda *a, **k: 1,
                    "find_max_tile_span": lambda *a, **k: 1,
                },
            ),
        },
    )
    ttnn.distributed = ShimModule("ttnn.distributed", {})
    return ttnn


def install(mesh_shape: Sequence[int] = (4, 8), arch: str = "blackhole", grid=(13, 10), force: bool = False):
    """Shadow ``ttnn`` with the shim and return a metadata mesh device."""
    from .hostenv import ensure_host_env, ensure_model_env

    real = sys.modules.get("ttnn")
    if real is not None and not getattr(real, "__dryrun__", False) and not force:
        raise RuntimeError(
            "real ttnn is already imported (%s); a dry run must not shadow it mid-process. "
            "Run `ditcheck dryrun` in its own process." % getattr(real, "__file__", "?")
        )

    ensure_host_env()
    CTX.arch_name = arch
    ttnn = build_ttnn()
    for name in _MODULES:
        if name not in _SAVED:
            _SAVED[name] = sys.modules.get(name)
    for name in _MODULES:
        target = ttnn
        for part in name.split(".")[1:]:
            target = getattr(target, part)
        sys.modules[name] = target

    CTX.installed = True
    # Only now is it safe to import anything under `models.`: tt_dit imports ttnn at
    # module level, and it must find the shim.
    ensure_model_env()
    return stubs.MeshDevice(mesh_shape, arch, grid)


def uninstall() -> None:
    """Put ``sys.modules`` back, so a dry run cannot leak into another test."""
    for name, previous in _SAVED.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    _SAVED.clear()
    CTX.installed = False


def assert_installed() -> None:
    """Fail loudly if the graph is about to be built against something else."""
    ttnn = sys.modules.get("ttnn")
    if ttnn is None or not getattr(ttnn, "__dryrun__", False):
        raise RuntimeError(
            "the dry-run shim is not installed as `ttnn` (found %s); the graph would be "
            "recorded from the wrong module" % (getattr(ttnn, "__file__", None) or ttnn)
        )
