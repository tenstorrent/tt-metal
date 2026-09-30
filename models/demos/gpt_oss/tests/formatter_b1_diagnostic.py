# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Temporary PR58542 diagnosis; never performance or qualification evidence."""

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch

PREFIX = "formatter-diag-structured-"
TARGET = PREFIX + "0"
OUT = Path("output/formatter_b1_diagnostic")
METHODS = Path(__file__).with_name("formatter_b1_methods.json")
_context = None
_captured = False
_token_steps = 0
_stream_counts = {}


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _append(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("a") as handle:
        handle.write(json.dumps(value) + "\n")


def _is_target(request_id):
    return bool(re.search(re.escape(TARGET) + r"(?:$|[-_:])", request_id))


def set_request_context(request_ids, is_prompt, has_structured):
    global _context
    if os.environ.get("TT_FORMATTER_B1_DIAGNOSTIC") != "1":
        return
    _context = None
    if not is_prompt and has_structured and any(_is_target(r) for r in request_ids):
        _context = {"request_ids": list(request_ids), "is_prompt": is_prompt, "has_structured": has_structured}


def _metadata(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "strides": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
        "bytes_sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _methods():
    snapshots = json.loads(METHODS.read_text())
    functions = []
    for side in ("main", "pr"):
        source = snapshots[side + "_method"]
        assert hashlib.sha256(source.encode()).hexdigest() == snapshots[side + "_method_sha256"]
        namespace = {"torch": torch, "Mode": SimpleNamespace(DECODE="decode")}
        exec(compile(source, side + "-exact-process-output-decode", "exec"), namespace)
        functions.append(namespace["process_output_decode"])
    return snapshots, functions


def compare_post_gather(tensor, B, S, vocab_size):
    """Run unchanged exact methods after substituting their already-completed gather."""
    snapshots, functions = _methods()
    # TP/EP gather has already completed once. Both exact methods receive this
    # identical tensor via the TP=1 gather endpoint; no native/device call occurs.
    proxy = SimpleNamespace(
        vocab_size=vocab_size,
        mesh_config=SimpleNamespace(get_config=lambda mode: SimpleNamespace(tp=1)),
        concat_device_output=lambda ignored: tensor,
    )
    outputs = {}
    result = {"B": B, "S": S, "vocab_size": vocab_size, "input": _metadata(tensor)}
    for side, function in zip(("main", "pr"), functions):
        try:
            outputs[side] = function(proxy, None, B, S)
            result[side] = _metadata(outputs[side])
        except Exception as error:
            result[side] = {"error_type": type(error).__name__, "error": str(error)}
    if len(outputs) == 2:
        result["same_shape_dtype_values_bits"] = all(
            result["main"][key] == result["pr"][key] for key in ("shape", "dtype", "bytes_sha256")
        )
        result["same_strides"] = result["main"]["strides"] == result["pr"]["strides"]
        result["torch_equal"] = torch.equal(outputs["main"], outputs["pr"])
    result["main_head"] = snapshots["main_head"]
    result["pr_head"] = snapshots["pr_head"]
    result["method_sha256"] = {side: snapshots[side + "_method_sha256"] for side in ("main", "pr")}
    return result, outputs


def capture_post_gather(tensor, B, S, vocab_size):
    global _captured
    if _captured or _context is None:
        return
    _captured = True
    result, outputs = compare_post_gather(tensor, B, S, vocab_size)
    result["request_context"] = _context
    result["scope"] = "First post-gather host decode for measured structured request0; diagnostic only"
    OUT.mkdir(parents=True, exist_ok=True)
    # Preserve exact values, not just rounded text or aggregate similarities.
    values = {"post_gather": tensor.detach().clone()}
    values.update({side: value.detach().clone() for side, value in outputs.items()})
    file = OUT / "same-post-gather-values.pt"
    torch.save(values, file)
    result["exact_values_file"] = file.name
    result["exact_values_file_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    _json(OUT / "same-post-gather-comparison.json", result)
    print("FORMATTER_B1_DIAGNOSTIC " + json.dumps(result), flush=True)


def record_sampled_tokens(request_ids, tokens, is_decode, perform_device_sampling):
    global _token_steps
    if _token_steps >= 64 or not any(_is_target(r) for r in (request_ids or [])):
        return
    _token_steps += 1
    _append(
        "request0-sampled-tokens.jsonl",
        {
            "step": _token_steps,
            "request_ids": request_ids,
            "is_decode": is_decode,
            "device_sampling": perform_device_sampling,
            "token_ids": tokens.detach().cpu().tolist(),
        },
    )


def record_request(request_id, prompt, schema):
    _append(
        "structured-request-identities.jsonl",
        {
            "request_id": request_id,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        },
    )


def record_stream(request_id, data):
    if not request_id or not request_id.startswith(PREFIX):
        return
    counts = _stream_counts.setdefault(
        request_id, {"packets": 0, "content_chars": 0, "reasoning_chars": 0, "error_packets": 0}
    )
    counts["packets"] += 1
    choices = []
    for choice in data.get("choices", []):
        delta = choice.get("delta", {})
        counts["content_chars"] += len(delta.get("content") or "")
        counts["reasoning_chars"] += len(delta.get("reasoning") or delta.get("reasoning_content") or "")
        choices.append(
            {
                "delta": {
                    key: delta[key] for key in ("role", "content", "reasoning", "reasoning_content") if key in delta
                },
                "finish_reason": choice.get("finish_reason"),
            }
        )
    if data.get("error"):
        counts["error_packets"] += 1
    if request_id == TARGET:
        # Deliberate whitelist: never headers, auth, arbitrary error messages,
        # environment, or the unfiltered response object.
        error = data.get("error") or {}
        if not isinstance(error, dict):
            error = {}
        _append(
            "request0-stream.jsonl",
            {
                "packet": counts["packets"],
                "choices": choices,
                "error_type": error.get("type"),
                "error_code": error.get("code"),
                "usage": {
                    key: data["usage"][key]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if key in data.get("usage", {})
                }
                if isinstance(data.get("usage"), dict)
                else None,
            },
        )


def record_stream_end(request_id, generated):
    if request_id and request_id.startswith(PREFIX):
        _append(
            "structured-stream-summaries.jsonl",
            {
                "request_id": request_id,
                **_stream_counts.get(request_id, {}),
                "generated_chars": len(generated),
                "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            },
        )


def _patch(path, expected_hash, transform):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_hash, str(path)
    output = transform(data.decode())
    ast.parse(output)
    path.write_text(output)
    return {
        "path": str(path),
        "before_sha256": expected_hash,
        "after_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def _replace_once(source, needle, replacement):
    assert source.count(needle) == 1, needle
    return source.replace(needle, replacement)


def _patch_plugin(source):
    source = _replace_once(
        source,
        "        perform_device_sampling = self.check_perform_device_sampling(\n",
        "        from models.demos.gpt_oss.tests.formatter_b1_diagnostic import set_request_context\n"
        "        set_request_context(row_req_ids, is_prompt, has_structured)\n\n"
        "        perform_device_sampling = self.check_perform_device_sampling(\n",
    )
    return _replace_once(
        source,
        "            sampled_token_ids_per_dp.append(next_token_ids.reshape(sz, -1))\n",
        "            from models.demos.gpt_oss.tests.formatter_b1_diagnostic import record_sampled_tokens\n"
        "            record_sampled_tokens(model_input.row_req_ids, next_token_ids, is_decode, perform_device_sampling)\n"
        "            sampled_token_ids_per_dp.append(next_token_ids.reshape(sz, -1))\n",
    )


def _patch_benchmark(source):
    # Initial single-prompt probe remains unchanged; only the original measured
    # 32-request loop receives observability IDs. No prompt/schema/sampling edit.
    source = _replace_once(
        source,
        "            extra_body=extra_body,\n        )\n        expected.append(request.completion)\n",
        "            extra_body=extra_body,\n"
        "            request_id=f'formatter-diag-structured-{i}',\n        )\n"
        "        from models.demos.gpt_oss.tests.formatter_b1_diagnostic import record_request\n"
        "        record_request(request_func_input.request_id, request.prompt, request.schema)\n"
        "        expected.append(request.completion)\n",
    )
    return source


def _patch_backend(source):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "async_request_openai_chat_completions"
    )
    lines = source.splitlines(keepends=True)
    part = "".join(lines[node.lineno - 1 : node.end_lineno])
    part = _replace_once(
        part,
        "                            data = json.loads(chunk)\n",
        "                            data = json.loads(chunk)\n"
        "                            from models.demos.gpt_oss.tests.formatter_b1_diagnostic import record_stream\n"
        "                            record_stream(request_func_input.request_id, data)\n",
    )
    part = _replace_once(
        part,
        "                    output.generated_text = generated_text\n",
        "                    from models.demos.gpt_oss.tests.formatter_b1_diagnostic import record_stream_end\n"
        "                    record_stream_end(request_func_input.request_id, generated_text)\n"
        "                    output.generated_text = generated_text\n",
    )
    return "".join(lines[: node.lineno - 1]) + part + "".join(lines[node.end_lineno :])


def self_check():
    cases = []
    for rows in (1, 8, 32):
        for width in (100, 128):
            tensor = torch.arange(rows * width, dtype=torch.float32).reshape(1, 1, rows, width)
            result, outputs = compare_post_gather(tensor, 1, 1, 100)
            assert result["same_shape_dtype_values_bits"], result
            cases.append({"rows": rows, "width": width, "same_bits": True, "same_strides": result["same_strides"]})
    # Demonstrate that this comparison can detect the original B32 row bug.
    result, outputs = compare_post_gather(torch.arange(128.0).reshape(1, 1, 1, 128), 32, 1, 100)
    assert "error_type" in result["pr"] or not result.get("same_shape_dtype_values_bits", False)
    _json(OUT / "host-comparison-self-check.json", {"B1S1_cases": cases, "B32_row_bug_distinguished": True})


def prepare(plugin, benchmark):
    self_check()
    receipts = [
        _patch(
            Path(plugin) / "src/vllm_tt_plugin/model_runner.py",
            "88ef54cc94258591a27a6ee0ee3b122c39adf815a93546f024cb5ff2e30a2751",
            _patch_plugin,
        ),
        _patch(
            Path(benchmark) / "benchmarks/benchmark_serving_structured_output.py",
            "629732fd2983d8bbc6a698efbe40eed0c4c242ff8738ee8f4b10f67e7ebc2c07",
            _patch_benchmark,
        ),
        _patch(
            Path(benchmark) / "benchmarks/backend_request_func.py",
            "f73736e66e0df650f1f41ee6f0439e3480283dce6d35354be3da8f5472c25a40",
            _patch_backend,
        ),
    ]
    _json(
        OUT / "runtime-instrumentation.json",
        {"diagnostic_only": True, "source_plugin_pin": "c1c85eb6bfe14a3e47afc450e25f7a8ddc81d974", "patches": receipts},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", nargs=2, metavar=("PLUGIN", "BENCHMARK"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(*args.prepare)
    elif args.self_check:
        self_check()
