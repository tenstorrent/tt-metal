# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
High-Level Fusion API: Sequential and Parallel.

Provides the user-facing API for composing operations into fused kernels.
Sequential chains ops linearly; Parallel runs ops on disjoint core subsets.

Usage (linear chain):
    >>> fused = Sequential(op0, op1, op2).build()
    >>> composite.launch([fused])

Usage (branching tree):
    >>> fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    >>> composite.launch([fused])
"""

import os
from typing import Any, List, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.common import _get_risc_type


class FusedOp:
    """Result of fusing ops via Sequential/Parallel.

    Wraps an ``OpDescriptor`` and adds ``semaphores`` refs that prevent
    GC of GlobalSemaphores whose L1 addresses are baked into runtime args.

    Properties ``descriptor``, ``input_tensors``, and ``output_tensors``
    forward to the underlying ``OpDescriptor``, so ``FusedOp`` is
    duck-type compatible with ``OpDescriptor`` (e.g. for
    ``composite.launch()``).

    Cannot be nested in Sequential/Parallel -- ``_resolve()`` rejects it
    with a TypeError.
    """

    __slots__ = ("op", "semaphores", "kernel_labels")

    def __init__(
        self,
        op: OpDescriptor,
        semaphores: Tuple[Any, ...] = (),
        kernel_labels: Tuple[str, ...] = (),
    ):
        self.op = op
        self.semaphores = semaphores
        self.kernel_labels = kernel_labels

    @property
    def descriptor(self):
        return self.op.descriptor

    @property
    def input_tensors(self):
        return self.op.input_tensors

    @property
    def output_tensors(self):
        return self.op.output_tensors

    def dump_kernel_sources(self, output_dir: str) -> None:
        """Write fused kernel sources to reader.cpp, writer.cpp, compute.cpp.

        If multiple kernels share the same RISC type (e.g. two readers for
        different core groups), they are written as reader_0.cpp, reader_1.cpp, etc.

        Args:
            output_dir: Directory to write files into (created if needed).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Group kernels by RISC type
        by_type: dict[str, list] = {}
        for kernel in self.op.descriptor.kernels:
            risc = _get_risc_type(kernel)
            # Map internal names to friendly filenames
            if risc == "riscv_0":
                name = "reader"
            elif risc == "riscv_1":
                name = "writer"
            elif risc == "compute":
                name = "compute"
            else:
                name = "unknown"
            by_type.setdefault(name, []).append(kernel)

        for name, kernels in by_type.items():
            for i, kernel in enumerate(kernels):
                if kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE:
                    source = kernel.kernel_source
                else:
                    path = kernel.kernel_source
                    source = ""
                    for base in [os.environ.get("TT_METAL_HOME", ""), ""]:
                        full = os.path.join(base, path) if base else path
                        if os.path.exists(full):
                            with open(full) as f:
                                source = f.read()
                            break
                    if not source:
                        source = f"// Could not read file: {path}\n"

                filename = f"{name}.cpp" if len(kernels) == 1 else f"{name}_{i}.cpp"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w") as f:
                    f.write(source)

    def _apply_kernel_dir(self, kernel_dir: str) -> None:
        """Switch kernel sources to file-based, writing files only if they don't exist.

        For single-kernel roles, files are named ``reader.cpp``, ``writer.cpp``,
        ``compute.cpp``.  For multi-kernel roles (parallel branches), filenames
        include op names and core ranges, e.g.
        ``reader_rms_norm_matmul_cores_0x0-3x3.cpp``.
        """
        os.makedirs(kernel_dir, exist_ok=True)

        # Group kernels by RISC type: name -> [(global_idx, kernel)]
        by_type: dict[str, list] = {}
        for idx, kernel in enumerate(self.op.descriptor.kernels):
            risc = _get_risc_type(kernel)
            name = {"riscv_0": "reader", "riscv_1": "writer", "compute": "compute"}.get(risc, "unknown")
            by_type.setdefault(name, []).append((idx, kernel))

        for name, entries in by_type.items():
            for idx, kernel in entries:
                if len(entries) == 1:
                    filename = f"{name}.cpp"
                else:
                    label = self.kernel_labels[idx] if idx < len(self.kernel_labels) else ""
                    core_tag = _core_range_tag(kernel.core_ranges)
                    tag = f"{label}_{core_tag}" if label else core_tag
                    filename = f"{name}_{tag}.cpp"
                filepath = os.path.join(kernel_dir, filename)
                abspath = os.path.abspath(filepath)

                if not os.path.exists(abspath):
                    with open(abspath, "w") as f:
                        f.write(kernel.kernel_source)

                kernel.kernel_source = abspath
                kernel.source_type = ttnn.KernelDescriptor.SourceType.FILE

    def __repr__(self):
        n_kernels = len(self.op.descriptor.kernels) if hasattr(self.op.descriptor, "kernels") else "?"
        return (
            f"FusedOp(kernels={n_kernels}, "
            f"inputs={len(self.op.input_tensors)}, "
            f"outputs={len(self.op.output_tensors)})"
        )


class Sequential:
    """A sequence of ops to fuse into a single dispatch.

    Items can be ``OpDescriptor``, ``Sequential``, or ``Parallel`` objects.
    Nested ``Sequential`` items are automatically flattened.

    Usage::

        # Inline
        fused = Sequential(op0, op1, op2).build()

        # Incremental
        s = Sequential(op0)
        s.add(op1).add(op2)
        fused = s.build()

        # Composition
        stem = Sequential(op0, op1)
        full = Sequential(stem, op2).build()  # flattened
    """

    def __init__(self, *items):
        if not items:
            raise ValueError("Sequential() requires at least 1 item")
        self._items = list(items)

    def add(self, item):
        """Append an item.  Returns self for chaining."""
        self._items.append(item)
        return self

    def build(self, device=None, kernel_dir: str = None) -> FusedOp:
        """Build the fused op.  Device is auto-extracted from tensors if not provided.

        Args:
            device: Target device.  Auto-extracted from tensors if None.
            kernel_dir: Optional directory for file-based kernel sources.
                When set, kernel sources are written to files and the JIT
                compiles from disk instead of in-memory strings.  Existing
                files are NOT overwritten — delete them to force regeneration.
        """
        r = self._build_internal(device)
        fused = FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
            kernel_labels=r.kernel_labels,
        )
        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def _build_internal(self, device=None):
        """Internal build returning intermediate _BuildResult."""
        from models.experimental.ops.descriptors.fusion.graph import OpGraphBuilder

        if device is None:
            device = _extract_device(self._items)
        nodes = _resolve(self)
        if len(nodes) != 1:
            raise ValueError("Sequential must resolve to a single root node")
        return OpGraphBuilder(nodes[0])._build_internal(device)


class Parallel:
    """Items that run in parallel on disjoint core subsets.

    Each item runs independently on its own cores.  Items can be
    ``OpDescriptor``, ``Sequential``, or ``Parallel`` objects.

    Usage::

        # Inline
        fused = Parallel(op_a, op_b).build()
        composite.launch([fused])

        # As part of a Sequential
        fused = Sequential(stem, Parallel(branch_a, branch_b)).build()
    """

    def __init__(self, *items):
        if len(items) < 2:
            raise ValueError("Parallel() requires at least 2 items")
        self._items = list(items)

    def add(self, item):
        """Add a branch.  Returns self for chaining."""
        self._items.append(item)
        return self

    def build(self, device=None, kernel_dir: str = None) -> FusedOp:
        """Build each item independently and merge into one FusedOp.

        Args:
            device: Target device.  Auto-extracted from tensors if None.
            kernel_dir: Optional directory for file-based kernel sources.
                When set, kernel sources are written to files and the JIT
                compiles from disk instead of in-memory strings.  Existing
                files are NOT overwritten — delete them to force regeneration.
        """
        r = self._build_internal(device)
        fused = FusedOp(
            op=OpDescriptor(r.descriptor, r.input_tensors, r.output_tensors),
            semaphores=r.semaphores,
            kernel_labels=r.kernel_labels,
        )
        if kernel_dir is not None:
            fused._apply_kernel_dir(kernel_dir)
        return fused

    def _build_internal(self, device=None):
        """Internal build returning intermediate _BuildResult."""
        from models.experimental.ops.descriptors.fusion.graph import _merge_build_results

        if device is None:
            device = _extract_device(self._items)
        built = [_build_item(item, device) for item in self._items]
        return _merge_build_results(built)


# =============================================================================
# Internal Helpers
# =============================================================================


def _resolve(item) -> List:
    """Convert a user-facing item into a list of OpNode trees.

    Handles all three types uniformly:
    - ``OpDescriptor`` -> ``[OpNode(op)]``
    - ``Parallel`` -> flat list of children (one OpNode per branch)
    - ``Sequential`` -> single-element list containing a chain of OpNodes
    """
    from models.experimental.ops.descriptors.fusion.graph import OpNode

    if isinstance(item, OpDescriptor):
        return [OpNode(item)]

    if isinstance(item, Parallel):
        return [node for child in item._items for node in _resolve(child)]

    if isinstance(item, Sequential):
        # Flatten nested Sequential items
        flat: List = []
        for sub in item._items:
            if isinstance(sub, Sequential):
                flat.extend(sub._items)
            else:
                flat.append(sub)

        if not flat:
            raise ValueError("Sequential() has no items after flattening")

        # Process right-to-left: resolve the last item to get tail nodes,
        # then walk remaining items in reverse, each becoming parent of tail.
        tail = _resolve(flat[-1])

        for sub_item in reversed(flat[:-1]):
            parents = _resolve(sub_item)
            if len(parents) != 1:
                raise ValueError(
                    "Items before a Parallel in a Sequential are not allowed — "
                    "the tree would diverge and can't rejoin.  Place trailing "
                    "items inside each branch instead."
                )
            _set_leaf_children(parents[0], tail)
            tail = parents

        return tail

    if isinstance(item, FusedOp):
        raise TypeError(
            "FusedOp cannot be nested in Sequential/Parallel — "
            "it is the result of build() and has already been fused."
        )

    raise TypeError(f"Unsupported item type: {type(item).__name__}")


def _extract_device(items):
    """Walk item tree, return device from first tensor found."""
    for item in items:
        if isinstance(item, OpDescriptor):
            for t in item.input_tensors:
                return t.device()
            for t in item.output_tensors:
                return t.device()
        elif isinstance(item, (Sequential, Parallel)):
            dev = _extract_device(item._items)
            if dev is not None:
                return dev
    raise ValueError(
        "Cannot auto-extract device: no items contain device-backed tensors. "
        "Pass device explicitly to build(device=...)."
    )


def _set_leaf_children(node, children: List):
    """Attach children to the deepest single-child leaf of a node subtree."""
    current = node
    while current.children:
        if len(current.children) != 1:
            raise ValueError("Cannot attach children to a node that already branches")
        current = current.children[0]
    current.children = list(children)


def _build_item(item, device):
    """Build a single item into a _BuildResult."""
    from models.experimental.ops.descriptors.fusion.common import _BuildResult

    if isinstance(item, OpDescriptor):
        return _BuildResult(
            descriptor=item.descriptor,
            input_tensors=item.input_tensors,
            output_tensors=item.output_tensors,
        )
    if isinstance(item, (Sequential, Parallel)):
        return item._build_internal(device)
    raise TypeError(f"Unsupported item type: {type(item).__name__}")


def _core_range_tag(core_ranges) -> str:
    """Create a filename-safe tag from a CoreRangeSet, e.g. ``'cores_0x0-3x3'``."""
    parts = []
    for cr in core_ranges.ranges():
        parts.append(f"{cr.start.x}x{cr.start.y}-{cr.end.x}x{cr.end.y}")
    return "cores_" + "_".join(parts)


__all__ = [
    "Sequential",
    "Parallel",
    "FusedOp",
]
