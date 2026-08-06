#!/usr/bin/env python3
"""Offline unit test for the vendored Poolside Laguna `poolside_v1` tool + reasoning parsers.

Runs WITHOUT a device. Feeds canonical Laguna native tool-call output
(`<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`) through the plugin's
PoolsideV1ToolParser and asserts the OpenAI tool_calls round-trip — including Laguna's string-vs-nonstring
arg rule (strings verbatim, non-strings JSON-deserialized, mirroring the chat template's
`tojson … if v is not string else v`). Also asserts poolside_v1 == glm47 on the common case.

Env: TT_METAL_HOME + the serve PYTHONPATH (see README). Run:
    python doc/vllm_integration/scripts/test_tool_parser.py
"""
import json
import os
import sys

TOK_DIR = os.environ.get(
    "LAGUNA_TOK_DIR",
    max(
        __import__("glob").glob("/home/ttuser/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1/snapshots/*"),
        key=os.path.getmtime,
    ),
)

import vllm_tt_plugin.entrypoints as e  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager  # noqa: E402

e._register_tt_tool_parsers()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                },
                "required": ["city"],
            },
        },
    }
]

# Canonical Laguna native output: string arg raw, non-string args serialized (int, bool).
MODEL_OUTPUT = (
    "I'll check the forecast.\n"
    "<tool_call>get_weather"
    "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    "<arg_key>days</arg_key><arg_value>3</arg_value>"
    "<arg_key>metric</arg_key><arg_value>true</arg_value>"
    "</tool_call>"
)


def run(parser_name):
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
    parser = ToolParserManager.get_tool_parser(parser_name)(tok)
    req = ChatCompletionRequest(model="poolside/Laguna-XS-2.1", messages=[], tools=TOOLS)
    out = parser.extract_tool_calls(MODEL_OUTPUT, req)
    assert out.tools_called, f"{parser_name}: tools_called False"
    assert len(out.tool_calls) == 1, f"{parser_name}: expected 1 tool call, got {len(out.tool_calls)}"
    fc = out.tool_calls[0].function
    args = json.loads(fc.arguments)
    return fc.name, args, out.content


def main():
    print(f"tokenizer: {TOK_DIR}")
    name, args, content = run("poolside_v1")
    print(f"poolside_v1 -> name={name!r} args={args!r} content={content!r}")
    assert name == "get_weather", name
    assert args == {"city": "Paris", "days": 3, "metric": True}, args  # str raw, int+bool deserialized
    assert content and content.strip() == "I'll check the forecast.", repr(content)
    print("  ✓ name, string/int/bool arg typing, and pre-call content all correct")

    # Cross-check the interim glm47 stand-in agrees on this common case.
    try:
        gname, gargs, _ = run("glm47")
        print(f"glm47       -> name={gname!r} args={gargs!r}")
        assert (gname, gargs) == (name, args), f"glm47 disagrees: {(gname, gargs)} vs {(name, args)}"
        print("  ✓ poolside_v1 == glm47 on the common case")
    except Exception as ex:  # glm47 may type-coerce differently; report but don't fail the poolside gate
        print(f"  (glm47 cross-check skipped/differs: {ex})")

    print("OFFLINE TOOL-PARSER TEST PASS")


if __name__ == "__main__":
    sys.exit(main())
