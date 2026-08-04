# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Run real ``models/tt_dit`` code against a metadata-only ttnn and record the graph.

    from dit_analyzer.dryrun import install, start, load_meta_weights

    device = install(mesh_shape=(4, 8), arch="blackhole")   # shadows `ttnn`
    graph = start(device, axis_names=("tp", "sp"), calls=48)
    ...                                                     # build and call the model
    report = analyze_graph(graph)

No device, no checkpoint, no trace capture: ``LTXTransformerBlock.forward`` is
ordinary Python, and if every ttnn call it makes returns metadata instead of
doing work, the forward runs on a laptop and emits the graph as a side effect.

The named targets in :mod:`.targets` are the supported way in; ``ditcheck
dryrun`` drives them. Everything here has to be installed in its own process --
``install()`` shadows ``ttnn`` in ``sys.modules`` and refuses to displace a real
one.
"""

from __future__ import annotations

from typing import Any, Dict

from ..ir import Graph
from . import ops, recorder
from .context import CTX
from .hostenv import SUBSTITUTIONS, ensure_host_env, have_real_torch
from .install import assert_installed, build_ttnn, install, uninstall
from .recorder import missing_ops
from .weights import load_meta_weights

__all__ = [
    "CTX",
    "SUBSTITUTIONS",
    "assert_installed",
    "build_ttnn",
    "ensure_host_env",
    "have_real_torch",
    "install",
    "load_meta_weights",
    "missing_ops",
    "start",
    "uninstall",
]


def start(mesh_device, axis_names=("axis0", "axis1"), name="dryrun", steps=1, calls=1, **meta) -> Graph:
    """Begin recording a graph. Checks it is the shim that will be recorded."""
    assert_installed()
    graph = recorder.start(mesh_device, axis_names=axis_names, name=name, steps=steps, calls=calls, **meta)
    ops.reset_caches()
    return graph


def provenance() -> Dict[str, Any]:
    """What this run was, honestly: how faithful the host environment was."""
    return {
        "torch": "real (device='meta')" if have_real_torch() else "metadata-only stand-in",
        "substitutions": list(SUBSTITUTIONS),
        "unregistered_ops": missing_ops(),
    }
