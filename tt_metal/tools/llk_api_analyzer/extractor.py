# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extract LLK API invocations from a compute thread's DWARF.

Compute-kernel APIs are all ``ALWI`` (always-inline) and the kernels are built
with ``-O3 -flto``. As a result, every LLK call that survives into the final
binary appears in DWARF as a ``DW_TAG_inlined_subroutine`` nested under the
thread entry point, and dead code is eliminated. Walking that inlined tree
therefore yields the APIs that are actually present in the running kernel, with
their template arguments attached to the abstract definition.

The extractor is intentionally generic: it has no per-op or per-API knowledge.
New LLKs, ops and models are picked up automatically because classification is
based only on naming conventions (``_llk_*`` / ``llk_*``) and DWARF source
paths.
"""

from __future__ import annotations

from dataclasses import dataclass

from elftools.dwarf.die import DIE

from .arg_values import ConstantArgEvaluator
from .dwarf_helpers import (
    EnumTable,
    SourceResolver,
    attr_str,
    definition_die,
    resolve_name,
    strip_type_qualifiers,
    follow_ref,
    type_name,
)
from .model import ApiCall, ApiLayer, ComputeThread, RuntimeArg, TemplateArg

# Source-path fragments that identify the compute-API layer (the user-facing
# ckernel ops that call into the LLKs).
_COMPUTE_API_PATH_HINTS = ("/api/compute/", "compute_kernel_api")


@dataclass
class ExtractorConfig:
    """Controls which call layers are collected. LLKs are the default focus."""

    include_layers: frozenset[ApiLayer] = frozenset({ApiLayer.LLK_API})


class LlkApiExtractor:
    """Extracts :class:`ApiCall` records from one thread's DWARF info."""

    def __init__(self, dwarf, thread: ComputeThread, config: ExtractorConfig | None = None):
        self._dwarf = dwarf
        self._thread = thread
        self._config = config or ExtractorConfig()
        self._const_args = ConstantArgEvaluator(dwarf)

    def extract(self) -> list[ApiCall]:
        enums = self._build_enum_table()
        sources = SourceResolver(self._dwarf)
        calls: list[ApiCall] = []
        for cu in self._dwarf.iter_CUs():
            self._walk(cu.get_top_DIE(), enums, sources, enclosing_op=None, out=calls)
        return calls

    def _build_enum_table(self) -> EnumTable:
        """Build a single enum table spanning every CU in the thread's DWARF."""
        table = EnumTable()
        for cu in self._dwarf.iter_CUs():
            for die in cu.iter_DIEs():
                if die.tag == "DW_TAG_enumeration_type":
                    table.add(die)
        return table

    def _walk(self, die: DIE, enums: EnumTable, sources: SourceResolver, enclosing_op: str | None, out: list) -> None:
        next_op = enclosing_op
        if die.tag == "DW_TAG_inlined_subroutine":
            name = resolve_name(die)
            if name is not None:
                layer = self._classify(name, die, sources)
                if layer == ApiLayer.COMPUTE_API:
                    next_op = _base_name(name)
                if layer in self._config.include_layers:
                    out.append(self._build_call(die, name, layer, enums, sources, enclosing_op))
        for child in die.iter_children():
            self._walk(child, enums, sources, next_op, out)

    def _classify(self, name: str, die: DIE, sources: SourceResolver) -> ApiLayer:
        base = _base_name(name)
        if base.startswith("_llk_"):
            return ApiLayer.LLK_CORE
        if base.startswith("llk_"):
            return ApiLayer.LLK_API
        source = sources.path_for(die)
        if source and any(hint in source for hint in _COMPUTE_API_PATH_HINTS):
            return ApiLayer.COMPUTE_API
        return ApiLayer.OTHER

    def _build_call(
        self,
        die: DIE,
        name: str,
        layer: ApiLayer,
        enums: EnumTable,
        sources: SourceResolver,
        enclosing_op: str | None,
    ) -> ApiCall:
        definition = definition_die(die)
        call_file, call_line, call_column = sources.call_site(die)
        return ApiCall(
            name=_base_name(name),
            qualified_name=name,
            layer=layer,
            thread=self._thread,
            template_args=self._template_args(definition, enums),
            runtime_args=self._runtime_args(die),
            operation=enclosing_op,
            source_file=sources.path_for(die),
            call_file=call_file,
            call_line=call_line,
            call_column=call_column,
        )

    def _template_args(self, definition: DIE, enums: EnumTable) -> list[TemplateArg]:
        args: list[TemplateArg] = []
        for child in definition.iter_children():
            if child.tag == "DW_TAG_template_value_param":
                args.append(self._template_value_arg(child, enums))
            elif child.tag == "DW_TAG_template_type_param":
                type_die = follow_ref(child, "DW_AT_type")
                args.append(
                    TemplateArg(
                        name=attr_str(child, "DW_AT_name") or "?",
                        value=type_name(type_die),
                        type_name="type",
                    )
                )
        return args

    @staticmethod
    def _template_value_arg(child: DIE, enums: EnumTable) -> TemplateArg:
        type_die = strip_type_qualifiers(follow_ref(child, "DW_AT_type"))
        type_str = type_name(type_die) if type_die is not None else None
        name = attr_str(child, "DW_AT_name") or "?"

        const = child.attributes.get("DW_AT_const_value")
        if const is None or not isinstance(const.value, int):
            return TemplateArg(name=name, value="?", type_name=type_str)
        raw_value = const.value

        enum_name = None
        value: int | bool | str = raw_value
        if type_die is not None and type_die.tag == "DW_TAG_enumeration_type":
            enum_name = enums.name_for(type_die.offset, raw_value, type_str)
        elif type_str == "bool":
            value = bool(raw_value)

        return TemplateArg(
            name=name,
            value=value,
            type_name=type_str,
            enum_name=enum_name,
        )

    def _runtime_args(self, die: DIE) -> list[RuntimeArg]:
        args: list[RuntimeArg] = []
        for child in die.iter_children():
            if child.tag != "DW_TAG_formal_parameter":
                continue
            values = self._const_args.evaluate(child)
            args.append(
                RuntimeArg(
                    name=resolve_name(child) or "?",
                    static_values=values or (),
                    is_static=values is not None,
                )
            )
        return args


def _base_name(name: str) -> str:
    """Strip template arguments from a function name (``foo<int>`` -> ``foo``)."""
    angle = name.find("<")
    return name[:angle] if angle != -1 else name
