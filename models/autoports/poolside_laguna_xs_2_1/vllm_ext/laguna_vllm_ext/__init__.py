# SPDX-License-Identifier: Apache-2.0
"""tt-metal-owned vLLM general plugin for Laguna-XS-2.1
(stock vLLM 0.24.0 + public vllm-tt-plugin).

Registered via the ``vllm.general_plugins`` entry point (see pyproject.toml), so vLLM imports and calls
``register()`` in every process that loads plugins — crucially the API-server / frontend process, which is
where tool-call parsing runs.

Why this exists
---------------
Stock vLLM 0.24.0 ships a ``poolside_v1`` tool parser whose ``func_detail_regex`` is::

    <tool_call>([^\\n]*)\\n(.*)</tool_call>

i.e. it REQUIRES a newline after the function name. This Laguna checkpoint (and its chat template) emit the
tool call with the arg tags immediately after the name, no newline::

    <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>...</tool_call>

so the stock regex misses and ``auto`` tool-calling silently returns finish_reason=stop with the raw
``<tool_call>`` text left in ``content`` (verified on device). This override subclasses the stock parser and
swaps in a newline-TOLERANT detail regex that parses BOTH the newline-free (Laguna) and the newline
(stock-expected) grammars, then EAGERLY re-registers it under the same name ``poolside_v1``. Eager
registration lands in ToolParserManager.tool_parsers, which get_tool_parser() checks BEFORE the stock lazy
entry — so existing serve flags (--tool-call-parser poolside_v1) keep working unchanged. The reasoning parser
is unaffected (the tool call lands in content, not reasoning_content).
"""

import logging

logger = logging.getLogger(__name__)

# Newline-TOLERANT: name = chars up to the first '<' or newline; optional newline; args = rest (non-greedy).
_FIXED_DETAIL_PATTERN = r"<tool_call>([^\n<]*)\n?(.*?)</tool_call>"

try:
    import regex as _re

    from vllm.tool_parsers.poolside_v1_tool_parser import PoolsideV1ToolParser

    class PoolsideV1LagunaToolParser(PoolsideV1ToolParser):
        """poolside_v1 tool parser with a newline-tolerant <tool_call> detail regex for Laguna-XS-2.1."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.func_detail_regex = _re.compile(_FIXED_DETAIL_PATTERN, _re.DOTALL)

    _IMPORT_OK = True
except Exception:  # pragma: no cover - defensive: never break plugin loading
    PoolsideV1LagunaToolParser = None  # type: ignore[assignment]
    _IMPORT_OK = False


def register() -> None:
    if not _IMPORT_OK:
        logger.warning("laguna_vllm_ext: poolside_v1 tool parser unavailable; override NOT installed")
        return
    try:
        from vllm.tool_parsers import ToolParserManager

        # Immediate/eager registration (module=...): stores the class object directly in
        # ToolParserManager.tool_parsers, which get_tool_parser() resolves before the stock lazy entry.
        ToolParserManager.register_module("poolside_v1", module=PoolsideV1LagunaToolParser)
        logger.info("laguna_vllm_ext: registered newline-tolerant poolside_v1 tool parser override")
    except Exception:
        logger.exception("laguna_vllm_ext: failed to register poolside_v1 override")
