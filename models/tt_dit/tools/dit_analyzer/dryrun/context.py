# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The state of one dry run.

Kept in its own module with no intra-package imports so ``tensor``, ``stubs``,
``recorder`` and ``ops`` can all reach the live mesh without importing each
other in a cycle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..ir import ACT, Graph, Mesh

TILE = 32
#: Only frames under this path are recorded as source locations.
MODEL_MARKER = "models/tt_dit"
#: Model code vs shared library code, for `Node.attribution` (roadmap blocker 44).
MODEL_CODE = "models/tt_dit/models/"
#: How many tt_dit frames to keep per node.
STACK_DEPTH = 3
#: Frames that only dispatch (``Module.__call__`` -> ``forward``) and would push a
#: real call site out of the window.
DISPATCH_FILES = ("models/tt_dit/layers/module.py",)


class Context:
    """Everything a running dry run needs; one instance, reset by ``start()``."""

    def __init__(self) -> None:
        self.graph: Optional[Graph] = None
        self.mesh: Optional[Mesh] = None
        self.mesh_device: Any = None
        #: Set by `install()`, before any mesh exists, so arch queries during model
        #: construction have an answer.
        self.arch_name = "blackhole"
        self.counter = 0
        #: `Node.calls` stamped on every node -- how often the traced region runs
        #: per forward (48 layers). Derived from the run in phase 9.
        self.calls = 1
        #: ttnn call name -> [count, example source location, tensor arity].
        self.unregistered: Dict[str, List[Any]] = {}
        self.installed = False
        #: Set while a module's weights are being loaded, so `from_torch` can tag
        #: the symbols it mints as parameters (constant across denoise steps).
        self.loading_weights = False
        #: Symbol kind / name hint for the next `from_torch`, used by weight loading
        #: so a finding names `video_attn.to_qkv.weight` instead of `const_57`.
        self.entry_kind = ACT
        self.entry_base: Optional[str] = None

    def reset(self) -> None:
        self.graph = None
        self.mesh = None
        self.counter = 0
        self.calls = 1
        self.unregistered = {}
        self.loading_weights = False
        self.entry_kind = ACT
        self.entry_base = None

    @property
    def mesh_shape(self):
        return self.mesh.shape

    @property
    def arch(self) -> str:
        """The arch under test. Answers `ttnn.get_arch_name()` and `is_blackhole()`,
        which key program configs and chunk-size tables in the model code, so a wrong
        answer here is a silently different run."""
        return self.mesh.arch if self.mesh is not None else self.arch_name

    def require_graph(self) -> Graph:
        if self.graph is None:
            raise RuntimeError("no dry run in progress: call dryrun.start(mesh_device) first")
        return self.graph


CTX = Context()
