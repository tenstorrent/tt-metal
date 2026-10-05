"""Offline native prompt and unmodified parser verification from returned IDs."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from models.demos.k2_horizon_7b_qb2.tests.benchmark_chat_middleware import normalize_assistant_history
from models.demos.k2_horizon_7b_qb2.tests.benchmark_reasoning_parser import K2HorizonBenchmarkReasoningParser

MODEL_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(
    "/mnt/models/huggingface/hub/models--IFM--K2-Horizon-7B/snapshots/" "036114ce8d46c32b24c15423211069abb9c5d25e"
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    # Iterating physical lines preserves literal Unicode line separators inside
    # response strings, unlike str.splitlines().
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_response(tokenizer, parser, eos_ids, payload, response, metadata):
    response_id = response.get("id")
    normalized = normalize_assistant_history(payload)
    template_kwargs = payload.get("chat_template_kwargs", {})
    require(template_kwargs.get("reasoning_effort") == "high", f"{response_id}: expected native high mode")
    expected = tokenizer.apply_chat_template(
        normalized["messages"], tokenize=True, return_dict=False, add_generation_prompt=True, **template_kwargs
    )
    actual = response.get("prompt_token_ids")
    require(isinstance(actual, list) and actual == expected, f"{response_id}: native prompt token IDs differ")
    require(response["usage"]["prompt_tokens"] == len(actual), f"{response_id}: prompt usage count differs")
    require(not parser.is_reasoning_end(actual), f"{response_id}: native prompt has no active reasoning opener")
    choices = response.get("choices", [])
    require(len(choices) == 1 and choices[0]["index"] == 0, f"{response_id}: expected one completion")
    choice = choices[0]
    ids = choice.get("token_ids")
    require(
        isinstance(ids, list) and ids and all(type(value) is int for value in ids), f"{response_id}: missing token IDs"
    )
    usage = response["usage"]["completion_tokens"]
    native_stop = choice["finish_reason"] == "stop" and choice.get("stop_reason") in eos_ids
    count_relation = "equal"
    if usage != len(ids):
        if native_stop and ids[-1] in eos_ids and usage == len(ids) - 1:
            count_relation = "usage_excludes_returned_terminal_eos"
        elif native_stop and ids[-1] not in eos_ids and usage == len(ids) + 1:
            count_relation = "returned_ids_omit_counted_terminal_eos"
        else:
            raise ValueError(f"{response_id}: generated ID/usage counts differ without a single native EOS explanation")
    # vLLM suppresses a terminal stop token from output.text unless explicitly
    # requested. Preserve every other generated token, including think markers.
    text_ids = ids
    if native_stop and ids[-1] == choice.get("stop_reason") and not payload.get("include_stop_str_in_output", False):
        text_ids = ids[:-1]
    decoded = tokenizer.decode(text_ids, skip_special_tokens=payload.get("skip_special_tokens", True))
    reasoning, content = parser.extract_reasoning(decoded, None)
    message = choice["message"]
    require(
        (reasoning, content) == (message.get("reasoning"), message.get("content")),
        f"{response_id}: decoded tokens/parser fields differ from the actual API response",
    )
    return {
        **metadata,
        "response_id": response_id,
        "prompt_tokens": len(actual),
        "prompt_token_ids_sha256": digest(actual),
        "normalized_messages_sha256": digest(normalized["messages"]),
        "generated_token_ids": len(ids),
        "generated_token_ids_sha256": digest(ids),
        "usage_completion_tokens": usage,
        "completion_count_relation": count_relation,
        "finish_reason": choice["finish_reason"],
        "stop_reason": choice.get("stop_reason"),
        "initial_generated_close": ids[0] == parser.end_token_id,
        "generated_open_markers": ids.count(parser.start_token_id),
        "generated_close_markers": ids.count(parser.end_token_id),
        "literal_open_markers_in_content": (content or "").count(parser.start_token),
        "literal_close_markers_in_content": (content or "").count(parser.end_token),
        "exact_native_prompt_match": True,
        "exact_parser_response_match": True,
    }


def run_cases(run_dir, sources):
    config_path = run_dir / "run_config.json"
    inputs_path = run_dir / "benchmark-inputs.jsonl"
    manifest_path = run_dir / "manifest.json"
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    inputs = read_jsonl(inputs_path)
    by_hash = {row["request_sha256"]: row for row in inputs}
    expected = sum(manifest["groups"][task]["sample_count"] for task in config["tasks"])
    require(len(inputs) == len(by_hash) == expected, "Input count or unique request hashes differ from frozen subset")
    for path in (config_path, inputs_path, manifest_path):
        sources[str(path)] = sha256(path)
    seen = set()
    for task in config["tasks"]:
        responses_path = run_dir / task / "responses.jsonl"
        links_path = run_dir / task / "request_links.jsonl"
        responses = read_jsonl(responses_path)
        links = read_jsonl(links_path)
        response_map = {row["id"]: row for row in responses}
        require(
            len(response_map) == len(responses) == len(links) == manifest["groups"][task]["sample_count"],
            f"{task}: response/link count differs",
        )
        sources[str(responses_path)] = sha256(responses_path)
        sources[str(links_path)] = sha256(links_path)
        linked_responses = set()
        for link in links:
            key = link["request_sha256"]
            require(key not in seen, "Repeated linked input request")
            seen.add(key)
            row = by_hash[key]
            require(digest([row["arguments"][0], row["arguments"][1]]) == key, "Input request hash differs")
            require(
                row["doc_id"] == link["doc_id"] and row["task"] == link["task"], "Input/link document identity differs"
            )
            require(link["response_id"] not in linked_responses, "Repeated linked response")
            linked_responses.add(link["response_id"])
            payload = {"model": config["model"], "messages": row["messages"], **config["generation"][task]}
            yield payload, response_map[link["response_id"]], {
                "task": task,
                "doc_id": row["doc_id"],
                "doc_sha256": row["doc_sha256"],
                "request_sha256": key,
            }
        require(linked_responses == set(response_map), f"{task}: response/link IDs differ")
    require(seen == set(by_hash), "Unverified frozen request inputs remain")


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--run-dir", type=Path, default=MODEL_DIR / "doc/benchmark/run")
    arguments.add_argument("--control-dir", type=Path)
    args = arguments.parse_args()
    output = (args.control_dir or args.run_dir) / "chat-token-verification.json"
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
    parser = K2HorizonBenchmarkReasoningParser(tokenizer)
    eos = json.loads((SNAPSHOT / "generation_config.json").read_text())["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    sources = {
        str(path): sha256(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("benchmark_reasoning_parser.py"),
            Path(__file__).with_name("benchmark_chat_middleware.py"),
            SNAPSHOT / "chat_template.jinja",
        )
    }
    summary = {"status": "running", "inference_calls": 0, "sources_sha256": sources, "requests": []}
    try:
        if args.control_dir:
            cases = []
            for path in sorted(args.control_dir.glob("response-*.json")):
                entry = json.loads(path.read_text())
                sources[str(path)] = sha256(path)
                cases.append((entry["request"], entry["response"], {"doc_id": entry["doc_id"]}))
            require(len(cases) == 4, "Expected four actual-token diagnostic controls")
        else:
            cases = run_cases(args.run_dir, sources)
        for payload, response, metadata in cases:
            summary["requests"].append(verify_response(tokenizer, parser, eos_ids, payload, response, metadata))
        summary.update(
            status="passed",
            verified_requests=len(summary["requests"]),
            literal_close_responses=sum(row["literal_close_markers_in_content"] > 0 for row in summary["requests"]),
            initial_generated_close_responses=sum(row["initial_generated_close"] for row in summary["requests"]),
            conclusion="Actual prompt tokens and unmodified parser outputs match exactly; literal generated delimiters remain in upstream-scored content.",
        )
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("sources_sha256", "requests")}))


if __name__ == "__main__":
    main()
