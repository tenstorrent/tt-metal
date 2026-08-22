# SPDX-License-Identifier: Apache-2.0
"""Failure-injection contracts for Laguna's required general plugin."""

from __future__ import annotations

import laguna_vllm_ext


def test_required_tool_parser_import_failure_is_fatal(monkeypatch, expect_error):
    cause = ImportError("injected parser import failure")
    monkeypatch.setattr(laguna_vllm_ext, "_IMPORT_OK", False)
    monkeypatch.setattr(laguna_vllm_ext, "_IMPORT_ERROR", cause)

    with expect_error(RuntimeError, "required poolside_v1 tool parser is unavailable") as raised:
        laguna_vllm_ext._register_tool_parser_override()

    assert raised.value.__cause__ is cause


def test_required_tool_parser_registration_failure_is_fatal(monkeypatch, expect_error):
    from vllm.tool_parsers import ToolParserManager

    def fail_registration(*args, **kwargs):
        raise ValueError("injected parser registry failure")

    monkeypatch.setattr(ToolParserManager, "register_module", fail_registration)

    with expect_error(RuntimeError, "required poolside_v1 tool parser registration failed") as raised:
        laguna_vllm_ext._register_tool_parser_override()

    assert isinstance(raised.value.__cause__, ValueError)


def test_general_plugin_does_not_swallow_required_parser_failure(monkeypatch, expect_error):
    def fail_required_parser():
        raise RuntimeError("injected required parser failure")

    monkeypatch.setattr(laguna_vllm_ext, "_register_tool_parser_override", fail_required_parser)
    monkeypatch.setenv("TT_LAGUNA_PREFIX_CACHE", "0")
    monkeypatch.setenv("TT_LAGUNA_HYBRID_KV", "0")

    with expect_error(RuntimeError, "injected required parser failure"):
        laguna_vllm_ext.register()
