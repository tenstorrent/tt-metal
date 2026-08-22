# SPDX-License-Identifier: Apache-2.0
"""tt-metal-owned vLLM general plugin for Laguna-XS-2.1.

Registered via the ``vllm.general_plugins`` entry point (see pyproject.toml), so vLLM imports and calls
``register()`` in every process that loads plugins — crucially the API-server / frontend process, which is
where tool-call parsing runs, and before ``VllmConfig`` calls the TT platform's config hook.

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

The public TT plugin commit pinned by this model also disables prefix caching for every model with a
sliding window. Laguna's p150x2 profile qualifies that combination with canonical 8192-token cache
admission, runtime-stable resume inputs, and a frozen post-trace program cache.
``TT_LAGUNA_PREFIX_CACHE=1`` advertises both model capabilities and lets this wrapper restore the
requested setting. Models without both capabilities retain the public plugin's behavior unchanged.
"""

import logging
import os

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
    _IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - exercised through failure injection
    PoolsideV1LagunaToolParser = None  # type: ignore[assignment]
    _IMPORT_OK = False
    _IMPORT_ERROR = error


def _register_tool_parser_override() -> None:
    """Install the parser required by every advertised Laguna serving profile."""

    if not _IMPORT_OK:
        raise RuntimeError("laguna_vllm_ext: required poolside_v1 tool parser is unavailable") from _IMPORT_ERROR

    from vllm.tool_parsers import ToolParserManager

    try:
        # Immediate/eager registration (module=...): stores the class object directly in
        # ToolParserManager.tool_parsers, which get_tool_parser() resolves before the stock lazy entry.
        ToolParserManager.register_module("poolside_v1", module=PoolsideV1LagunaToolParser)
    except Exception as error:
        logger.exception("laguna_vllm_ext: failed to register poolside_v1 override")
        raise RuntimeError("laguna_vllm_ext: required poolside_v1 tool parser registration failed") from error
    logger.info("laguna_vllm_ext: registered newline-tolerant poolside_v1 tool parser override")


def register() -> None:
    # General plugins are loaded before VllmConfig invokes the active platform's
    # check_and_update_config hook. Install the model-capability wrapper here so
    # it is present in the API, engine-core, and worker processes.
    try:
        from .hybrid_kv import install_hybrid_kv_patch
        from .prefix_cache import install_sliding_window_prefix_cache_patch
        from .prefix_cache_quantum import install_prefix_cache_quantum_patch

        install_sliding_window_prefix_cache_patch()
        install_prefix_cache_quantum_patch()
        install_hybrid_kv_patch()
    except Exception:
        logger.exception("laguna_vllm_ext: failed to install KV-cache support")
        # Cache admission and shared-pool sizing are correctness boundaries,
        # not optional optimizations. Explicitly enabled profiles fail closed.
        if os.environ.get("TT_LAGUNA_PREFIX_CACHE", "0") == "1" or os.environ.get("TT_LAGUNA_HYBRID_KV", "0") == "1":
            raise

    try:
        from .lifecycle import install_worker_lifecycle_patch

        install_worker_lifecycle_patch()
    except Exception:
        logger.exception("laguna_vllm_ext: failed to install adapter lifecycle")
        raise

    # Every launcher profile enables --tool-call-parser poolside_v1 and the
    # checkpoint emits a grammar the stock parser does not understand. This is
    # therefore a production correctness dependency, not a best-effort feature.
    _register_tool_parser_override()
