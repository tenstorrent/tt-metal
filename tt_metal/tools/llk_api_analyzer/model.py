# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data model for the LLK API analysis results.

The model is a small set of plain dataclasses with ``to_dict`` methods so the
analysis can be serialized to JSON without coupling the extraction logic to any
particular output format.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from .descriptors import KernelDescriptors


class ComputeThread(str, Enum):
    """The three Tensix compute RISC threads a compute kernel compiles into."""

    UNPACK = "unpack"  # TRISC0
    MATH = "math"  # TRISC1
    PACK = "pack"  # TRISC2

    @classmethod
    def from_trisc(cls, trisc_id: int) -> "ComputeThread":
        return {0: cls.UNPACK, 1: cls.MATH, 2: cls.PACK}[trisc_id]


class ApiLayer(str, Enum):
    """Which layer of the LLK stack a function belongs to."""

    LLK_CORE = "llk_core"  # _llk_* in tt-llk/.../llk_lib
    LLK_API = "llk_api"  # llk_* wrappers in hw/ckernels/.../llk_api
    COMPUTE_API = "compute_api"  # user-facing ckernel::* in hw/inc/api/compute
    OTHER = "other"


# Ordering used when sorting calls: by API layer (top of the stack first), then
# by compute thread (unpack -> math -> pack), then by function name.
_LAYER_ORDER = {
    ApiLayer.COMPUTE_API: 0,
    ApiLayer.LLK_API: 1,
    ApiLayer.LLK_CORE: 2,
    ApiLayer.OTHER: 3,
}
_THREAD_ORDER = {
    ComputeThread.UNPACK: 0,
    ComputeThread.MATH: 1,
    ComputeThread.PACK: 2,
}


@dataclass(frozen=True)
class TemplateArg:
    """A compile-time template argument of an API call."""

    name: str
    value: int | bool | str
    type_name: str | None = None
    enum_name: str | None = None

    @property
    def display_value(self) -> str:
        if self.enum_name is not None:
            return self.enum_name
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        return str(self.value)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type_name,
            "enum": self.enum_name,
            "display": self.display_value,
        }


@dataclass(frozen=True)
class RuntimeArg:
    """A runtime argument of an API call.

    ``static_values`` holds the compile-time-constant value(s) the optimizer
    propagated (recovered from ``DW_AT_const_value`` or constant location
    expressions). It usually has a single value, but holds several when an
    unrolled loop calls one inlined API over a set of constant indices (e.g.
    ``dst_index`` taking ``0..7``). When the value lives in a register/stack at
    runtime it is not statically recoverable and ``is_static`` is ``False``.
    """

    name: str
    static_values: tuple[int, ...] = ()
    is_static: bool = False

    @property
    def display(self) -> str:
        if not self.is_static:
            return self.name
        if len(self.static_values) == 1:
            return f"{self.name}={self.static_values[0]}"
        return f"{self.name}={{{_format_int_set(self.static_values)}}}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "static_values": list(self.static_values),
            "is_static": self.is_static,
        }


def format_function_display(name: str, template_args: list[TemplateArg] | None = None) -> str:
    """Render a function name (with optional template args) in backticks."""
    if not template_args:
        return f"`{name}`"
    templates = ", ".join(f"{t.name}={t.display_value}" for t in template_args)
    return f"`{name}<{templates}>`"


def _format_int_set(values: tuple[int, ...]) -> str:
    """Render a set of ints compactly, collapsing contiguous runs to ``a..b``."""
    if not values:
        return ""
    ordered = sorted(set(values))
    parts: list[str] = []
    run_start = run_prev = ordered[0]
    for value in ordered[1:]:
        if value == run_prev + 1:
            run_prev = value
            continue
        parts.append(str(run_start) if run_start == run_prev else f"{run_start}..{run_prev}")
        run_start = run_prev = value
    parts.append(str(run_start) if run_start == run_prev else f"{run_start}..{run_prev}")
    return ",".join(parts)


@dataclass
class ApiCall:
    """A single LLK API invocation found inlined into a compute thread."""

    name: str
    qualified_name: str
    layer: ApiLayer
    thread: ComputeThread
    template_args: list[TemplateArg] = field(default_factory=list)
    runtime_args: list[RuntimeArg] = field(default_factory=list)
    operation: str | None = None  # enclosing compute-API op, if any
    source_file: str | None = None  # where the API is *declared*
    call_file: str | None = None  # where the API is *called from* (call site)
    call_line: int | None = None
    call_column: int | None = None

    @property
    def display_header(self) -> str:
        """The ``name<param=value, ...>`` header (enum/bool values resolved)."""
        return format_function_display(self.name, self.template_args)

    @property
    def call_site(self) -> str | None:
        """``file:line:col`` of the call site (basename), or ``None``."""
        if self.call_file is None and self.call_line is None:
            return None
        parts = [os.path.basename(self.call_file) if self.call_file else "?"]
        if self.call_line is not None:
            parts.append(str(self.call_line))
            if self.call_column:
                parts.append(str(self.call_column))
        return ":".join(parts)

    @property
    def sort_key(self) -> tuple:
        """Order by API layer, then thread (unpack/math/pack), then name/site."""
        return (
            _LAYER_ORDER.get(self.layer, 99),
            _THREAD_ORDER.get(self.thread, 99),
            self.name,
            self.display_header,
            self.call_line or 0,
            self.call_column or 0,
        )

    @property
    def group_key(self) -> tuple:
        """Groups occurrences of one API config invoked from one call site."""
        return (self.layer, self.thread, self.display_header, self.call_site or "")

    @property
    def static_arg_combo(self) -> tuple:
        """The combination of constant runtime-argument values for this call."""
        return tuple((a.name, a.static_values) for a in self.runtime_args if a.is_static)

    @property
    def dynamic_arg_names(self) -> list[str]:
        return [a.name for a in self.runtime_args if not a.is_static]

    @property
    def signature_key(self) -> tuple:
        """Identity used to aggregate calls with identical configuration.

        Based on the API and its (compile-time) template arguments only, so the
        cross-run aggregation collapses identical configurations regardless of
        per-call-site runtime values.
        """
        return (
            self.name,
            self.layer,
            tuple((t.name, t.display_value) for t in self.template_args),
        )

    @property
    def instance_key(self) -> tuple:
        """Identity that also distinguishes constant runtime-argument values.

        Used for per-kernel deduplication so that, e.g., ``pop_tiles(operand=0)``
        and ``pop_tiles(operand=1)`` are reported separately.
        """
        return self.signature_key + (tuple((a.name, a.static_values) for a in self.runtime_args if a.is_static),)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "layer": self.layer.value,
            "thread": self.thread.value,
            "operation": self.operation,
            "template_args": [t.to_dict() for t in self.template_args],
            "runtime_args": [r.to_dict() for r in self.runtime_args],
            "source_file": self.source_file,
            "call_file": self.call_file,
            "call_line": self.call_line,
            "call_column": self.call_column,
            "call_site": self.call_site,
        }


@dataclass
class KernelAnalysis:
    """The result of analyzing one compute kernel (its three TRISC ELFs)."""

    name: str
    path: str
    descriptors: KernelDescriptors | None = None
    api_calls: list[ApiCall] = field(default_factory=list)
    threads_analyzed: list[ComputeThread] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def unique_calls(self) -> list[ApiCall]:
        seen: dict[tuple, ApiCall] = {}
        for call in self.api_calls:
            seen.setdefault(call.instance_key + (call.thread,), call)
        return sorted(seen.values(), key=lambda c: c.sort_key)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "descriptors": self.descriptors.to_dict() if self.descriptors else None,
            "threads_analyzed": [t.value for t in self.threads_analyzed],
            "api_calls": [c.to_dict() for c in self.unique_calls()],
            "errors": self.errors,
        }


@dataclass
class AggregatedApi:
    """A unique (API, configuration) tuple aggregated across a whole run."""

    name: str
    layer: ApiLayer
    template_args: list[TemplateArg]
    threads: set[ComputeThread] = field(default_factory=set)
    kernels: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layer": self.layer.value,
            "template_args": [t.to_dict() for t in self.template_args],
            "threads": sorted(t.value for t in self.threads),
            "kernel_count": len(self.kernels),
            "kernels": sorted(self.kernels),
        }


@dataclass
class RunAnalysis:
    """The aggregate analysis of all compute kernels from a model/TTNN run."""

    root: str
    kernels: list[KernelAnalysis] = field(default_factory=list)

    def aggregate(self) -> list[AggregatedApi]:
        """Collapse identical (API, template-config) tuples across all kernels."""
        agg: dict[tuple, AggregatedApi] = {}
        for kernel in self.kernels:
            for call in kernel.api_calls:
                key = call.signature_key
                entry = agg.get(key)
                if entry is None:
                    entry = AggregatedApi(call.name, call.layer, list(call.template_args))
                    agg[key] = entry
                entry.threads.add(call.thread)
                entry.kernels.add(kernel.name)
        return sorted(agg.values(), key=lambda a: (_LAYER_ORDER.get(a.layer, 99), a.name))

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "kernel_count": len(self.kernels),
            "kernels": [k.to_dict() for k in self.kernels],
            "aggregated_apis": [a.to_dict() for a in self.aggregate()],
        }
