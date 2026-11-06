"""Strict offline mini2.2.8 reconstruction; never an exact HTTP capture."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("bringup/artifacts/parser-protocol-mini-swe/src/minisweagent/models")


def formatter():
    namespace = {}
    for name in ["anthropic_utils", "cache_control"]:
        source = ROOT / "utils" / (name + ".py")
        exec(compile(source.read_text(), str(source), "exec"), namespace)
    tree = ast.parse((ROOT / "litellm_model.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LitellmModel")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_messages_for_api")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "litellm_model.py"), "exec"), namespace)
    tools = ast.parse((ROOT / "utils/actions_toolcall.py").read_text())
    tool = next(
        ast.literal_eval(n.value)
        for n in tools.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BASH_TOOL" for t in n.targets)
    )
    return namespace["_prepare_messages_for_api"], tool


def reconstruct(data, allow_recorded_subset=False):
    info = data["info"]
    config = info["config"]
    assert info["mini_version"] == "2.2.8"
    assert data["trajectory_format"] == "mini-swe-agent-1.1"
    assert config["model_type"] == "minisweagent.models.litellm_model.LitellmModel"
    assert config["agent_type"] in {
        "minisweagent.agents.default.DefaultAgent",
        "minisweagent.agents.interactive.InteractiveAgent",
    }
    if config["agent_type"].endswith(".InteractiveAgent"):
        assert config["agent"]["mode"] == "yolo"
    settings = config["model"]
    kwargs = copy.deepcopy(settings.get("model_kwargs", {}))
    # Transport configuration is omitted explicitly; never serialize credentials.
    transport = {
        "api_key",
        "api_base",
        "base_url",
        "timeout",
        "request_timeout",
        "num_retries",
        "drop_params",
        "extra_headers",
    }
    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "stop",
        "stream",
        "n",
        "extra_body",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "tool_choice",
        "parallel_tool_calls",
    }
    assert not (set(kwargs) - transport - allowed), "Unreviewed request setting"
    omitted = sorted(set(kwargs) & transport)
    for key in omitted:
        kwargs.pop(key)
    assert not settings.get("multimodal_regex"), "Multimodal extraction needs separate validation"
    prepare, tool = formatter()
    obj = SimpleNamespace(config=SimpleNamespace(set_cache_control=settings.get("set_cache_control")))
    messages = data["messages"]
    rows = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assert message.get("extra", {}).get("response"), "Assistant lacks recorded response evidence"
        prefix = messages[:index]
        assert all(m.get("role") in {"system", "user", "assistant", "tool"} for m in prefix)
        rows.append(
            dict(
                model=settings["model_name"],
                messages=prepare(obj, copy.deepcopy(prefix)),
                tools=[copy.deepcopy(tool)],
                **copy.deepcopy(kwargs),
            )
        )
    assert (
        len(rows) == info["model_stats"]["api_calls"] or allow_recorded_subset
    ), "Missing/unrecorded calls; cannot reconstruct faithfully"
    assert rows
    return rows, omitted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--allow-recorded-subset", action="store_true")
    args = p.parse_args()
    private = Path("bringup/artifacts").resolve()
    assert args.output.resolve().is_relative_to(private), "Raw histories must stay under bringup/artifacts"
    data = json.loads(args.source.read_text())
    rows, omitted = reconstruct(data, args.allow_recorded_subset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    metadata = dict(
        kind="reconstructed_litellm_input_not_http_capture",
        calls=len(rows),
        recorded_subset=args.allow_recorded_subset,
        api_calls=data["info"]["model_stats"]["api_calls"],
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        omitted_transport_field_names=omitted,
        limitations=[
            "Provider normalization/retries and transport defaults not reproduced.",
            "Replay overrides model/max_tokens/temperature/stream; original body is retained here.",
        ],
    )
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
