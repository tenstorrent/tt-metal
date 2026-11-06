"""Host-only audit using unchanged upstream parser functions; never runs commands."""
import argparse
import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
HARBOR = ROOT / "bringup/artifacts/parser-protocol-harbor/src/harbor"
MINI = ROOT / "bringup/artifacts/parser-protocol-mini-swe/src/minisweagent"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def body(path):
    value = json.loads(Path(path).read_text())
    return value.get("body", value)


def terminal(path):
    module = load("terminus_json_plain_parser", HARBOR / "agents/terminus_2/terminus_json_plain_parser.py")
    message = body(path)["choices"][0]["message"]
    result = module.TerminusJSONPlainParser().parse_response(message.get("content") or "")
    return {
        "source": str(path),
        "parsed": dataclasses.asdict(result),
        "has_separate_reasoning": bool(message.get("reasoning") or message.get("reasoning_content")),
    }


def swe(path):
    # Namespace shells avoid upstream __init__ startup/config side effects; all
    # parser and exception implementations below are loaded unchanged from source.
    for name, directory in [
        ("minisweagent", MINI),
        ("minisweagent.models", MINI / "models"),
        ("minisweagent.models.utils", MINI / "models/utils"),
    ]:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    module = load("minisweagent.models.utils.actions_toolcall", MINI / "models/utils/actions_toolcall.py")
    message = body(path)["choices"][0]["message"]
    calls = [
        types.SimpleNamespace(**(call | {"function": types.SimpleNamespace(**call["function"])}))
        for call in message.get("tool_calls", [])
    ]
    try:
        actions = module.parse_toolcall_actions(calls, format_error_template="{{ error }}")
        return {
            "source": str(path),
            "ok": True,
            "actions": actions,
            "has_separate_reasoning": bool(message.get("reasoning") or message.get("reasoning_content")),
        }
    except Exception as exc:
        return {
            "source": str(path),
            "ok": False,
            "error_type": type(exc).__name__,
            "messages": getattr(exc, "messages", [str(exc)]),
        }


def pair(before, after):
    b, a = body(before)["choices"][0], body(after)["choices"][0]
    bt, at = b.get("token_ids"), a.get("token_ids")
    return {
        "baseline": str(before),
        "candidate": str(after),
        "baseline_tokens": len(bt) if bt is not None else None,
        "candidate_tokens": len(at) if at is not None else None,
        "exact_token_ids": bt is not None and at is not None and bt == at,
        "same_finish_reason": b.get("finish_reason") == a.get("finish_reason"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", action="append", default=[])
    parser.add_argument("--swe", action="append", default=[])
    parser.add_argument("--pair", nargs=2, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "terminal": [terminal(p) for p in args.terminal],
                "swe": [swe(p) for p in args.swe],
                "pairs": [pair(*p) for p in args.pair],
            },
            indent=2,
        )
    )
