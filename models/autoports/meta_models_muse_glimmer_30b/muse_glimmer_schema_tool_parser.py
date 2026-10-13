# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Schema-aware ATEM tool-call parser for Muse-Glimmer-30B.

vLLM's ``muse_glimmer`` parser JSON-decodes every ``<atem:parameter>`` value
without looking at the tool schema. A parameter declared ``"type": "string"``
whose value happens to be JSON text (e.g. tau3's
``call_discoverable_agent_tool(arguments="{...}")``) therefore reaches the
client as a nested object instead of the declared string.

This subclass keeps string-typed parameters as strings and leaves every other
parameter on the upstream decoding path. Load it with::

    --tool-parser-plugin <this file> --tool-call-parser muse_glimmer_schema
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import FunctionCall, ToolCall
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.muse_glimmer_tool_parser import (
    _INVOKE_RE,
    _NAME_RE,
    _PARAM_RE,
    MuseGlimmerToolParser,
    _decode_value,
)

_STRING_TYPES = {"string", "null"}


def _declared_types(schema: dict) -> set:
    declared = schema.get("type")
    if isinstance(declared, list):
        return set(declared)
    if declared:
        return {declared}
    return {s.get("type") for s in schema.get("anyOf") or [] if isinstance(s, dict)}


def _param_schemas(request) -> dict[str, dict]:
    """Map tool name -> its ``parameters.properties`` for this request."""
    schemas: dict[str, dict] = {}
    for tool in getattr(request, "tools", None) or []:
        fn = getattr(tool, "function", None) or tool
        if isinstance(fn, dict):
            name, params = fn.get("name"), fn.get("parameters")
        else:
            name, params = getattr(fn, "name", None), getattr(fn, "parameters", None)
        if name and isinstance(params, dict):
            schemas[name] = params.get("properties") or {}
    return schemas


def _decode_for_schema(raw: str, schema: dict | None):
    value = _decode_value(raw)
    types = _declared_types(schema) if isinstance(schema, dict) else set()
    if not types or not types <= _STRING_TYPES:
        return value
    if value is None and "null" in types:
        return None
    # A quoted JSON string decodes to the intended str; anything else
    # (object, array, number, bool) is the literal text the model wrote.
    return value if isinstance(value, str) else raw


@ToolParserManager.register_module("muse_glimmer_schema")
class MuseGlimmerSchemaToolParser(MuseGlimmerToolParser):
    _properties: dict[str, dict] = {}

    def extract_tool_calls(self, model_output: str, request):
        self._properties = _param_schemas(request)
        return super().extract_tool_calls(model_output, request)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ):
        self._properties = _param_schemas(request)
        return super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    def _parse_tool_calls(self, text: str, registered: set[str] | None = None) -> list[ToolCall]:
        registered = registered or set()
        scoped = self._tool_channel_text(text)
        tool_calls: list[ToolCall] = []
        for invoke in _INVOKE_RE.findall(scoped):
            name_m = _NAME_RE.search(invoke)
            if not name_m:
                continue
            name = self._normalize_name(name_m.group(1), registered)
            properties = self._properties.get(name, {})
            args: dict = {}
            for pm in _PARAM_RE.finditer(invoke):
                key = pm.group("key")
                args[key] = _decode_for_schema(pm.group("value"), properties.get(key))
            tool_calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    )
                )
            )
        return tool_calls
