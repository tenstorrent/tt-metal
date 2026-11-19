# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic HTTP repeat/reorder and full-model teacher-forced logit controls.

Run HTTP against the existing server; run TT only after it releases the devices.
Explicit token prompts are continuation stress tests, not chat quality verdicts.
Full-vocabulary readback is an oracle only, never a performance measurement.
The fixed 1e-4 logprob criterion is reported separately from exact equality;
The standalone control matches B32/context262144/8224-block serving geometry;
different scheduler row assignments and prefill grouping remain explicit.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ATOL = 1e-4
MODEL = "ornith-ai/Ornith-1.5-9B"
PROMPTS = {
    "p129": [9707 + i % 13 for i in range(129)],
    "p131": [9713 + i % 17 for i in range(131)],
    "p130": [9811 + i % 11 for i in range(130)],
}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def token_id(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("token_id:"):
        return int(value.split(":", 1)[1])
    raise ValueError(f"Expected return_tokens_as_token_ids representation, got {value!r}")


def positions_from_response(response, expected):
    choice = response["choices"][0]
    logs = choice["logprobs"]
    tokens = [token_id(value) for value in logs["tokens"]]
    if choice.get("token_ids") is not None and choice["token_ids"] != tokens:
        raise ValueError("Returned token_ids disagree with logprob token identifiers")
    if not len(tokens) == len(logs["token_logprobs"]) == len(logs["top_logprobs"]) == expected:
        raise ValueError("Response did not contain every requested token/logprob position")
    result = []
    for token, logprob, top in zip(tokens, logs["token_logprobs"], logs["top_logprobs"]):
        candidates = [{"token": token_id(key), "logprob": float(value)} for key, value in top.items()]
        if len(candidates) < 20:
            raise ValueError("Response returned fewer than 20 top-logprob candidates")
        result.append({"token": token, "logprob": float(logprob), "top": candidates})
    return result


def compare_positions(reference, actual):
    """Keep every position's difference; never pass a missing position."""
    rows = []
    for index, (left, right) in enumerate(zip(reference, actual)):
        a = {item["token"]: item["logprob"] for item in left["top"]}
        b = {item["token"]: item["logprob"] for item in right["top"]}
        shared = sorted(a.keys() & b.keys())
        errors = [{"token": token, "delta": b[token] - a[token]} for token in shared]
        selected_error = right["logprob"] - left["logprob"]
        rows.append(
            {
                "position": index,
                "same_token": left["token"] == right["token"],
                "selected_logprob_delta": selected_error,
                "exact_top_order": [item["token"] for item in left["top"]] == [item["token"] for item in right["top"]],
                "missing_reference_candidates": sorted(a.keys() - b.keys()),
                "new_candidates": sorted(b.keys() - a.keys()),
                "shared_candidate_deltas": errors,
                "exact": left == right,
                "within_1e_4": left["token"] == right["token"]
                and a.keys() == b.keys()
                and abs(selected_error) <= ATOL
                and all(abs(item["delta"]) <= ATOL for item in errors),
            }
        )
    lengths_match = len(reference) == len(actual) and bool(reference)
    return {
        "lengths": [len(reference), len(actual)],
        "exact": lengths_match and all(row["exact"] for row in rows),
        "within_1e_4": lengths_match and all(row["within_1e_4"] for row in rows),
        "positions": rows,
    }


def run_http(args, report):
    import requests

    def request(name):
        payload = {
            "model": args.model,
            "prompt": PROMPTS[name],
            "max_tokens": args.positions,
            "temperature": 0,
            "top_p": 1,
            "top_k": 20,
            "seed": 0,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "repetition_penalty": 1,
            "ignore_eos": True,
            "stream": False,
            "logprobs": 20,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
        response = requests.post(args.server_url.rstrip("/") + "/v1/completions", json=payload, timeout=600)
        record = {"prompt_id": name, "request": payload, "status_code": response.status_code}
        try:
            record["response"] = response.json()
            response.raise_for_status()
            record["positions"] = positions_from_response(record["response"], args.positions)
        except Exception as error:
            record["error"] = repr(error)
            record.setdefault("response_text", response.text)
        return record

    names = list(PROMPTS)
    for label, order, concurrent in (
        ("solo1", names, False),
        ("solo2", names, False),
        ("concurrent_reverse", names[::-1], True),
        ("concurrent_rotate", names[1:] + names[:1], True),
    ):
        if concurrent:
            with ThreadPoolExecutor(max_workers=3) as pool:
                runs = list(pool.map(request, order))
        else:
            runs = [request(name) for name in order]
        report["runs"][label] = {run["prompt_id"]: run for run in runs}
        save(args.output, report)
    reference = report["runs"]["solo1"]
    for label, runs in report["runs"].items():
        for name, run in runs.items():
            key = f"{label}/{name}"
            report["comparisons"][key] = (
                compare_positions(reference[name]["positions"], run["positions"])
                if "positions" in run and "positions" in reference[name]
                else {"exact": False, "within_1e_4": False, "error": "Missing valid HTTP response"}
            )
    report["status"] = "pass" if all(item["exact"] for item in report["comparisons"].values()) else "differences"
    report["route_note"] = "Explicit logprobs selects the plugin's host sampling path on TP4."


def summarize_logits(logits, http_position):
    import torch

    logits = logits.detach().float().reshape(-1).contiguous()
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite canonical full-vocabulary logits")
    logprobs = logits.log_softmax(-1)
    values, indices = logprobs.topk(20)
    selected = http_position["token"]
    top = [{"token": int(token), "logprob": float(value)} for token, value in zip(indices, values)]
    comparison_ids = sorted({item["token"] for item in http_position["top"]} | {selected})
    http_values = {item["token"]: item["logprob"] for item in http_position["top"]}
    http_values[selected] = http_position["logprob"]
    errors = [
        {
            "token": token,
            "http_logprob": http_values[token],
            "tt_logprob": float(logprobs[token]),
            "delta": float(logprobs[token]) - http_values[token],
            "tt_rank_min": 1 + int((logits > logits[token]).sum()),
        }
        for token in comparison_ids
    ]
    return {
        "token": selected,
        "logprob": float(logprobs[selected]),
        "top": top,
        "top_logits": [{"token": item["token"], "logit": float(logits[item["token"]])} for item in top],
        "argmax_token": int(logits.argmax()),
        "http_selected_is_argmax": bool(logits[selected] == logits.max()),
        "http_selected_rank_min": 1 + int((logits > logits[selected]).sum()),
        "raw_logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        "http_candidate_errors": errors,
        "http_candidates_within_1e_4": all(abs(item["delta"]) <= ATOL for item in errors),
    }


def run_tt(args, report):
    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.precision import load_policy

    source = json.loads(args.http_result.read_text())
    reference = source["runs"]["solo1"]
    names = list(PROMPTS)
    for name in names:
        run = reference[name]
        if run["request"]["prompt"] != PROMPTS[name] or len(run["positions"]) < args.positions:
            raise ValueError("HTTP source must contain the same prompts and enough valid token positions")
    report.update(
        http_source=str(args.http_result.resolve()),
        http_source_sha256=hashlib.sha256(args.http_result.read_bytes()).hexdigest(),
        layers=list(range(32)),
        batch=32,
        context=262144,
        physical_blocks=8224,
        active_slots=[0, 1, 2],
        geometry={
            "page_table_shape": [32, 8192],
            "prefill_page_table_shape": [3, 8192],
            "prefill_tokens_shape": [3, max(map(len, PROMPTS.values()))],
            "decode_tokens_shape": [32, 1],
            "decode_positions_shape": [32],
            "decode_host_logits_shape": [32, 1, "vocab_size"],
            "inactive_decode_position": -1,
            "warmup_mapped_pages_per_slot": 8,
            "page_mapping": "slot s owns pages [8*s, 8*s+8); unused table tail is zero, matching serving",
            "decode_masked": True,
            "sdpa_key_chunk": 512,
            "trace_region_size": 200000000,
        },
        precision_policy=load_policy(),
        comparison_scope="B32 external-cache canonical control; HTTP scheduler row assignments/grouping may differ",
    )
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    baseline_logits = {}
    try:
        model = OrnithModel(mesh, checkpoint=args.checkpoint)
        cache = model.allocate_cache(batch=32, context=262144, global_pages=True, num_blocks=8224)
        # Preparation/capture executes all 32 rows at position zero. Each row
        # needs valid private pages even though only three requests are admitted.
        # Match the plugin's zero-padded unused tail, including any masked
        # pages loaded by the SDPA chunk beyond the short causal prefix.
        table = torch.zeros((32, 8192), dtype=torch.int32)
        table[:, :8] = torch.arange(256, dtype=torch.int32).reshape(32, 8)
        gen = OrnithGenerator(model, max_batch_size=32, max_seq_len=262144, kv_cache=cache, page_table=table)
        if len(gen.model.layers) != 32:
            raise AssertionError("This control requires all 32 layers")
        report["geometry"]["decode_host_logits_shape"] = [32, 1, gen.model.vocab_size]
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=tuple(range(32)))
        gen.capture(state)
        report["checkpoint"] = str(gen.model.checkpoint.path)
        for label, order in (("batch1", names), ("batch2", names), ("batch_reorder", names[::-1])):
            lengths = [len(PROMPTS[name]) for name in order]
            tokens = torch.zeros((3, max(lengths)), dtype=torch.int64)
            for row, name in enumerate(order):
                tokens[row, : lengths[row]] = torch.tensor(PROMPTS[name])
            logits = gen.prefill_forward(
                tokens,
                page_table=gen.page_table[:3],
                kv_cache=gen.cache,
                prompt_lens=lengths,
                original_prompt_lens=lengths,
                empty_slots=[0, 1, 2],
                sampling_params=None,
            )
            if tuple(logits.shape) != (3, 1, gen.model.vocab_size):
                raise AssertionError("Prefill must return the three submitted host-logit rows")
            runs = {name: {"prompt_id": name, "slot": row, "positions": []} for row, name in enumerate(order)}
            report["runs"][label] = runs
            for position in range(args.positions):
                for row, name in enumerate(order):
                    raw = logits[row, 0].float().clone()
                    item = summarize_logits(raw, reference[name]["positions"][position])
                    key = (name, position)
                    if label == "batch1":
                        baseline_logits[key] = raw
                    else:
                        delta = (raw - baseline_logits[key]).abs()
                        item["repeat_raw_logits"] = {
                            "exact": bool(torch.equal(raw, baseline_logits[key])),
                            "max_abs": float(delta.max()),
                            "mean_abs": float(delta.mean()),
                            "within_1e_4": bool(delta.max() <= ATOL),
                        }
                    runs[name]["positions"].append(item)
                save(args.output, report)
                if position + 1 == args.positions:
                    break
                # Teacher force HTTP outputs; do not follow the control's own argmax.
                inputs = torch.zeros((32, 1), dtype=torch.int64)
                inputs[:3, 0] = torch.tensor([reference[name]["positions"][position]["token"] for name in order])
                decode_positions = [length + position for length in lengths] + [-1] * 29
                handle = gen.decode_forward(
                    inputs,
                    decode_positions,
                    page_table=gen.page_table,
                    kv_cache=gen.cache,
                    sampling_params=None,
                    reload_inputs=True,
                    read_from_device=False,
                )
                if not state.masked:
                    raise AssertionError("Three active slots in the B32 pool must use masked decode")
                read, events = gen.read_decode_output(handle, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                logits = gen.process_decode_output_host(read, is_tokens=False)
                if tuple(logits.shape) != (32, 1, gen.model.vocab_size):
                    raise AssertionError("Decode must retain all 32 pool rows")
            for name in order:
                report["comparisons"][f"http/{label}/{name}"] = compare_positions(
                    reference[name]["positions"][: args.positions],
                    [{key: item[key] for key in ("token", "logprob", "top")} for item in runs[name]["positions"]],
                )
        positions = [item for run in report["runs"].values() for entry in run.values() for item in entry["positions"]]
        report["identical_order_repeat_exact"] = all(
            item["repeat_raw_logits"]["exact"] for run in report["runs"]["batch2"].values() for item in run["positions"]
        )
        report["http_all_candidates_within_1e_4"] = all(item["http_candidates_within_1e_4"] for item in positions)
        report["status"] = "measured"
        report["verdict_note"] = "Inspect all errors/ranks; measured is not a serving correctness pass."
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("http", "tt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--http-result", type=Path)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if not 1 <= args.positions <= 16:
        parser.error("--positions must be between 1 and 16 for this small diagnostic")
    if args.mode == "tt" and args.http_result is None:
        parser.error("TT mode requires --http-result from the completed HTTP mode")
    report = {
        "status": "running",
        "mode": args.mode,
        "logprob_atol": ATOL,
        "prompts": PROMPTS,
        "runs": {},
        "comparisons": {},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save(args.output, report)
    try:
        (run_http if args.mode == "http" else run_tt)(args, report)
    except Exception as error:
        report.update(status="error", error=repr(error))
        raise
    finally:
        save(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    if report["status"] == "differences":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
