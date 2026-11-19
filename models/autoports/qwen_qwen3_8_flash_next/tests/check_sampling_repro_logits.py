# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy prefill/decode logit diagnostic; no device-RNG or performance claim.

Served mode deliberately requests host top20 logprobs. Standalone mode reads
full logits for exact repeat/row-swap checks. Hardware is owned by the caller.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .check_vllm_logit_determinism import MODEL_DIR, compare, read_events

TEXTS = {"random": "Random: ", "generate": "Generate: ", "text0": "Text 0:", "number": "Number: "}
LAYOUTS = {"solo": ["target"], "ab": ["target", "companion"], "ba": ["companion", "target"]}


def encoded_cases(tokenizer):
    cases = {}
    for name, text in TEXTS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        for companion in ("A", "I am", "One two three"):
            other = tokenizer.encode(companion, add_special_tokens=False)
            if len(ids) != len(other) and len(ids) + len(other) <= 9:
                break
        else:
            raise ValueError(f"No distinct-length budget9 companion for {name}")
        cases[name] = dict(text=text, companion_text=companion, target=ids, companion=other)
    return cases


def provenance(execution_manifest):
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    owner = json.loads(execution_manifest.read_text())
    paths = sorted((MODEL_DIR / "tt").rglob("*.py")) + [
        Path(__file__),
        MODEL_DIR / "doc/context_contract.json",
        MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json",
        repo / "models/common/sampling/generator.py",
        repo / "models/common/sampling/tt_sampling.py",
    ]
    hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    checked = {p: value for p, value in hashes.items() if p in owner["source_sha256"]}
    if not checked or any(owner["source_sha256"][p] != value for p, value in checked.items()):
        raise ValueError("Current model/shared source differs from the owned server launch manifest")
    return dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=hashes,
        owner_manifest=str(execution_manifest.resolve()),
        owner_source_matches=sorted(checked),
        shared_runtime_reference=owner["shared_runtime"],
        runtime_scope="Reference to owner launch inventory, not a new native-runtime measurement",
    )


def model_sources(source):
    # Test-client revisions are recorded, but are not model/runtime identity.
    # This permits replaying saved HTTP responses after fixing standalone setup.
    diagnostic = str(Path(__file__).resolve().relative_to(MODEL_DIR.resolve().parents[2]))
    return {key: value for key, value in source["source_sha256"].items() if key != diagnostic}


def row(prompt, token, values):
    ranked = sorted(values.items(), key=lambda item: (-item[1], int(item[0].removeprefix("token_id:"))))
    return dict(
        prompt_token_ids=list(prompt),
        sampled_token_id=int(token),
        top_logprobs=values,
        argmax_token_id=int(token),
        top2_logprob_margin=ranked[0][1] - ranked[1][1],
    )


def parse_response(response, prompts, output_tokens):
    choices = response["choices"]
    if len(choices) != len(prompts) or sorted(c["index"] for c in choices) != list(range(len(prompts))):
        raise ValueError("Missing or duplicate choices")
    if (
        response["usage"]["prompt_tokens"] != sum(map(len, prompts))
        or response["usage"]["completion_tokens"] != len(prompts) * output_tokens
    ):
        raise ValueError("Unexpected prompt/output token usage")
    rows = []
    for choice, prompt in zip(sorted(choices, key=lambda c: c["index"]), prompts, strict=True):
        tokens, lp = choice["token_ids"], choice["logprobs"]
        if (
            len(tokens) != output_tokens
            or len(lp["top_logprobs"]) != output_tokens
            or len(lp["token_logprobs"]) != output_tokens
        ):
            raise ValueError("Unexpected output token count")
        steps = []
        for step, values in enumerate(lp["top_logprobs"]):
            if len(values) < 20 or any(not k.startswith("token_id:") for k in values):
                raise ValueError("Expected token-ID keyed top20 logprobs")
            for key, value in values.items():
                int(key.removeprefix("token_id:"))
                if not math.isfinite(value) or value > 1e-5:
                    raise ValueError("Invalid logprob")
            selected = values[f"token_id:{tokens[step]}"]
            if abs(selected - max(values.values())) > 1e-5 or abs(lp["token_logprobs"][step] - selected) > 1e-5:
                raise ValueError("Greedy token/logprob is not the returned maximum")
            steps.append(row(prompt + tokens[:step], tokens[step], values))
        rows.append(dict(prompt_token_ids=prompt, token_ids=tokens, steps=steps))
    return rows


def position_evidence(events, prompts):
    lengths = list(map(len, prompts))
    matches = [
        e
        for e in events
        if e.get("kind") == "prefill_submission"
        and sorted(e["ends"]) == sorted(lengths)
        and e["starts"] == [0] * len(prompts)
        and e.get("sample_mask") == [True] * len(prompts)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one actual final B{len(prompts)} prefill, found {len(matches)}")
    event = matches[0]
    if event["sampling_mode"] != "host" or len(set(event["slots"])) != len(prompts):
        raise ValueError("Expected distinct slots and explicit host-logprob routing")
    return {
        str(length): dict(submitted_index=i, physical_slot=event["slots"][i]) for i, length in enumerate(event["ends"])
    }


def decode_evidence(events, prompts, positions, output_tokens):
    prefills = [e for e in events if e.get("kind") == "prefill_submission"]
    decodes = [e for e in events if e.get("kind") == "decode_submission"]
    checks = dict(
        one_initial_prefill=len(prefills) == 1,
        exact_decode_count=len(decodes) == output_tokens - 1,
        host_route_unchanged=bool(decodes) and all(e["sampling_mode"] == "host" for e in decodes),
    )
    direct = all("active_slots" in e and "submitted_positions" in e for e in decodes)
    if direct and decodes:
        checks["explicit_slots_positions"] = all(
            dict(zip(e["active_slots"], [p for p in e["submitted_positions"] if p >= 0], strict=True))
            == {positions[str(len(p))]["physical_slot"]: len(p) + step for p in prompts}
            for step, e in enumerate(decodes)
        )
    return dict(
        checks=checks,
        passed=all(checks.values()),
        decode_count=len(decodes),
        expected_decode_count=output_tokens - 1,
        expected_active_width=len(prompts),
        coverage=("explicit_decode_slots_and_positions" if direct else "lifecycle_inference"),
        inference="Initial observed slots, equal ignore_eos output lengths, exactly N-1 host decode submissions, and no intervening prefills imply fixed active width under the required exclusive-client execution.",
        limitation=(
            None
            if direct
            else "Tracking0 events omit decode slots/positions; physical decode bucket is inferred from lifecycle, not directly observed."
        ),
    )


def served(args, result):
    def post(suffix, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            args.server_url.rstrip("/") + suffix, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=args.timeout if body else 30) as response:
            return response.read().decode()

    for name, case in result["cases"].items():
        for repeat in range(2):
            for layout, roles in LAYOUTS.items():
                prompts = [case[role] for role in roles]
                body = dict(
                    model=args.model,
                    prompt=prompts,
                    temperature=0,
                    top_k=1,
                    top_p=1.0,
                    max_tokens=args.decode_tokens,
                    ignore_eos=True,
                    logprobs=20,
                    return_token_ids=True,
                    return_tokens_as_token_ids=True,
                    add_special_tokens=False,
                    repetition_penalty=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    request_id=f"repro-logits-{name}-{layout}-{repeat}-{time.time_ns()}",
                )
                run = dict(case=name, repeat=repeat, layout=layout, roles=roles, request=body)
                result["runs"].append(run)
                offset = args.server_log.stat().st_size
                try:
                    if len(prompts) == 2:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            post("/pause?mode=keep&clear_cache=false")
                            try:
                                pending = executor.submit(post, "/v1/completions", body)
                                time.sleep(1)
                                if pending.done():
                                    raise RuntimeError("Request finished before admission resumed")
                            finally:
                                post("/resume")
                            run["raw_response"] = pending.result()
                    else:
                        run["raw_response"] = post("/v1/completions", body)
                finally:
                    run["server_events"] = read_events(args.server_log, offset)
                run["response"] = json.loads(run["raw_response"])
                run["rows"] = parse_response(run["response"], prompts, args.decode_tokens)
                run["observed_positions"] = position_evidence(run["server_events"], prompts)
                run["decode_coverage"] = decode_evidence(
                    run["server_events"], prompts, run["observed_positions"], args.decode_tokens
                )
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def standalone(args, result):
    import torch

    import ttnn

    from ..tt.generator import build_generator
    from .shared_pool_layout import shared_pool_layout

    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        raise ValueError("Standalone requires TT_METAL_TRACE_ALLOC_TRACKING=1")
    for flag in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_PROFILER",
        "TT_METAL_STREAMING_PROFILER",
    ):
        if (flag == "TT_METAL_WATCHER" and flag in os.environ) or os.environ.get(flag) not in (None, "", "0"):
            raise ValueError("Standalone requires watcher/profiler disabled")
    mesh = gen = None
    raw = {}
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        gen = build_generator(
            MODEL_DIR,
            mesh,
            checkpoint=args.tokenizer,
            precision_config=result["precision"],
            layer_indices=tuple(range(48)),
            max_batch_size=2,
            max_seq_len=result["capacity"],
            defer_kv_cache=True,
        )
        result["runtime_precision"] = gen.model.precision_summary()
        if result["runtime_precision"]["requested"] != result["precision"] or list(gen.model.layer_indices) != list(
            range(48)
        ):
            raise ValueError("Runtime differs from selected all48 policy")
        specs, tables = shared_pool_layout(2, result["capacity"], 12)
        if args.pool_blocks is not None:
            if args.pool_blocks < (result["capacity"] + 31) // 32:
                raise ValueError("Pool must cover at least one full contract-length request")
            specs = [((args.pool_blocks, *shape[1:]), dtype, i) for shape, dtype, i in specs]
            tables = [table % args.pool_blocks for table in tables]
        result["pool_blocks"] = specs[0][0][0]
        pool = gen.allocate_kv_cache_per_layer(
            [(shape, result["precision"]["kv_cache_dtype"], i) for shape, _, i in specs]
        )
        result["pool_bytes_per_rank"] = pool.bytes_per_rank
        gen.prepare()
        for name, case in result["cases"].items():
            for repeat in range(2):
                for layout, roles in LAYOUTS.items():
                    prompts = [case[role] for role in roles]
                    slots = tuple(range(len(prompts)))
                    selected = [table[list(slots)] for table in tables]
                    gen.reset()
                    tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in prompts], batch_first=True)
                    logits = (
                        gen.prefill_forward(
                            tokens,
                            prompt_lens=list(map(len, prompts)),
                            slots=slots,
                            page_table=selected[0],
                            page_tables_per_layer=selected,
                            kv_cache=pool,
                        )
                        .float()
                        .cpu()
                    )
                    rows = [dict(prompt_token_ids=p, token_ids=[], steps=[]) for p in prompts]
                    full_steps = []
                    for step in range(args.decode_tokens):
                        if (
                            logits.shape
                            not in (
                                (len(prompts), 1, gen.model.config.vocab_size),
                                (len(prompts), gen.model.config.vocab_size),
                            )
                            or not torch.isfinite(logits).all()
                        ):
                            raise ValueError("Malformed or nonfinite logits")
                        logits = logits.reshape(len(prompts), gen.model.config.vocab_size)
                        full_steps.append(logits.clone())
                        predictions = logits.argmax(-1)
                        for output, values, prediction in zip(rows, logits, predictions, strict=True):
                            probabilities, ids = values.log_softmax(-1).topk(20)
                            item = row(
                                output["prompt_token_ids"] + output["token_ids"],
                                prediction,
                                {
                                    f"token_id:{token}": value
                                    for token, value in zip(ids.tolist(), probabilities.tolist(), strict=True)
                                },
                            )
                            top2 = values.topk(2).values
                            item["top2_logit_margin"] = float(top2[0] - top2[1])
                            output["steps"].append(item)
                            output["token_ids"].append(int(prediction))
                        if step + 1 < args.decode_tokens:
                            logits = (
                                gen.decode_forward(
                                    predictions.to(torch.int64),
                                    [len(p) + step for p in prompts],
                                    slots=slots,
                                    page_table=selected[0],
                                    page_tables_per_layer=selected,
                                    kv_cache=pool,
                                    sample_on_device=False,
                                    reload_inputs=True,
                                    reload_page_table=False,
                                    read_from_device=True,
                                )
                                .float()
                                .cpu()
                            )
                    raw[name, repeat, layout] = torch.stack(full_steps, dim=1)
                    result["runs"].append(
                        dict(
                            case=name,
                            repeat=repeat,
                            layout=layout,
                            roles=roles,
                            rows=rows,
                            submitted_slots=slots,
                            shape=list(raw[name, repeat, layout].shape),
                            decode_geometry=dict(
                                active_slots=slots,
                                active_width=len(prompts),
                                decode_calls=args.decode_tokens - 1,
                                reload_inputs=True,
                            ),
                        )
                    )
        result["checks"]["full_vocabulary_repeat_exact"] = all(
            torch.equal(raw[n, 0, layout], raw[n, 1, layout]) for n in result["cases"] for layout in LAYOUTS
        )
        result["checks"]["full_vocabulary_row_swap_exact"] = all(
            torch.equal(raw[n, r, "ab"], raw[n, r, "ba"].flip(0)) for n in result["cases"] for r in range(2)
        )
        result["full_vocabulary_solo_vs_pair_prefill_exact"] = {
            n: all(torch.equal(raw[n, r, "solo"][0, 0], raw[n, r, "ab"][0, 0]) for r in range(2))
            for n in result["cases"]
        }
        indexed = {(r["case"], r["repeat"], r["layout"]): r for r in result["runs"]}
        result["full_vocabulary_cross_width_shared_prefix"] = []
        for name in result["cases"]:
            for repeat in range(2):
                left, right = indexed[name, repeat, "solo"]["rows"][0], indexed[name, repeat, "ab"]["rows"][0]
                for step, (a, b) in enumerate(zip(left["steps"], right["steps"], strict=True)):
                    if a["prompt_token_ids"] != b["prompt_token_ids"]:
                        break
                    va, vb = raw[name, repeat, "solo"][0, step], raw[name, repeat, "ab"][0, step]
                    result["full_vocabulary_cross_width_shared_prefix"].append(
                        dict(
                            case=name,
                            repeat=repeat,
                            step=step,
                            exact=torch.equal(va, vb),
                            max_abs_logit_delta=float((va - vb).abs().max()),
                            solo_argmax=a["sampled_token_id"],
                            paired_argmax=b["sampled_token_id"],
                            solo_top2_margin=a["top2_logit_margin"],
                            paired_top2_margin=b["top2_logit_margin"],
                            sampled_token_equal=a["sampled_token_id"] == b["sampled_token_id"],
                        )
                    )
    finally:
        try:
            if gen is not None:
                gen.close()
        finally:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)


def compare_sequences(left, right):
    if len(left["steps"]) != len(right["steps"]):
        raise ValueError("Sequence lengths differ")
    results = []
    for step, (a, b) in enumerate(zip(left["steps"], right["steps"], strict=True)):
        same_history = a["prompt_token_ids"] == b["prompt_token_ids"]
        result = dict(
            step=step,
            same_history=same_history,
            left_argmax=a["sampled_token_id"],
            right_argmax=b["sampled_token_id"],
            left_top2_margin=a["top2_logprob_margin"],
            right_top2_margin=b["top2_logprob_margin"],
        )
        result.update(compare(a, b, 1e-5) if same_history else dict(passed=False, compared=False))
        results.append(result)
    return results


def evaluate(result, baseline=None):
    indexed = {(r["case"], r["repeat"], r["layout"]): r for r in result["runs"]}
    expected = {(name, repeat, layout) for name in result["cases"] for repeat in range(2) for layout in LAYOUTS}
    if set(indexed) != expected or len(indexed) != len(result["runs"]):
        raise ValueError("Incomplete or duplicate runs")
    comparisons = {"repeat": [], "row_swap": [], "solo_vs_pair_shared_prefix": []}
    divergence = []
    for name in result["cases"]:
        for layout in LAYOUTS:
            comparisons["repeat"].extend(
                dict(case=name, layout=layout, row=i, **comparison)
                for i, (a, b) in enumerate(
                    zip(indexed[name, 0, layout]["rows"], indexed[name, 1, layout]["rows"], strict=True)
                )
                for comparison in compare_sequences(a, b)
            )
        for repeat in range(2):
            a, b = indexed[name, repeat, "ab"]["rows"], indexed[name, repeat, "ba"]["rows"]
            comparisons["row_swap"].extend(
                dict(case=name, repeat=repeat, row=i, **comparison)
                for i, (x, y) in enumerate(zip(a, reversed(b), strict=True))
                for comparison in compare_sequences(x, y)
            )
            cross_width = compare_sequences(indexed[name, repeat, "solo"]["rows"][0], a[0])
            comparisons["solo_vs_pair_shared_prefix"].extend(
                dict(case=name, repeat=repeat, **comparison) for comparison in cross_width if comparison["same_history"]
            )
            divergence.append(
                dict(
                    case=name,
                    repeat=repeat,
                    first_token_divergence=next(
                        (c["step"] for c in cross_width if c["left_argmax"] != c["right_argmax"]), None
                    ),
                    equal_output_tokens=indexed[name, repeat, "solo"]["rows"][0]["token_ids"] == a[0]["token_ids"],
                )
            )
    result["comparisons"] = comparisons
    result["cross_width_divergence_diagnostic"] = divergence
    for key in ("repeat", "row_swap"):
        result["checks"][f"selected_logprobs_{key}"] = all(c["passed"] for c in comparisons[key])
    if result["mode"] == "served":
        result["checks"]["constant_active_width_host_decode"] = all(
            r["decode_coverage"]["passed"] for r in result["runs"]
        )
        result["checks"]["actual_physical_slot_reversal"] = all(
            {
                indexed[name, repeat, layout]["observed_positions"][str(len(case["target"]))]["physical_slot"]
                for layout in ("ab", "ba")
            }
            == {0, 1}
            for name, case in result["cases"].items()
            for repeat in range(2)
        )
        result["checks"]["solo_physical_slot_zero"] = all(
            r["observed_positions"][str(len(result["cases"][r["case"]]["target"]))]["physical_slot"] == 0
            for r in result["runs"]
            if r["layout"] == "solo"
        )
    if baseline is not None:
        if (
            baseline["mode"] != "standalone"
            or not baseline["passed"]
            or baseline["cases"] != result["cases"]
            or baseline["precision"] != result["precision"]
            or baseline["capacity"] != result["capacity"]
            or baseline["decode_tokens"] != result["decode_tokens"]
        ):
            raise ValueError("Baseline does not match selected-policy cases/context or did not pass")
        if model_sources(baseline["source"]) != model_sources(result["source"]):
            raise ValueError("Baseline and served source identities differ")
        result["baseline_diagnostic_source"] = baseline["source"]
        source = {(r["case"], r["repeat"], r["layout"]): r for r in baseline["runs"]}
        result["baseline_comparisons"] = [
            dict(case=key[0], repeat=key[1], layout=key[2], row=i, **comparison)
            for key, run in indexed.items()
            for i, (a, b) in enumerate(zip(source[key]["rows"], run["rows"], strict=True))
            for comparison in compare_sequences(a, b)
        ]
        result["checks"]["standalone_selected_logprobs"] = (
            result["checks"].get("constant_active_width_host_decode", False)
            and result["checks"].get("solo_physical_slot_zero", False)
            and result["checks"].get("actual_physical_slot_reversal", False)
            and all(c["passed"] for c in result["baseline_comparisons"])
        )
    result["standalone_comparison_pending"] = result["mode"] == "served" and baseline is None
    result["passed"] = all(result["checks"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("served", "standalone"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument(
        "--tokenizer",
        default="/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--replay-artifact", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--decode-tokens", type=int, default=10, help="Total output tokens, including prefill output")
    parser.add_argument("--pool-blocks", type=int, help="Standalone global pool pages; context capacity is unchanged")
    args = parser.parse_args()
    if args.output.exists() or (args.mode == "served" and not args.replay_artifact and not args.server_log):
        parser.error("Use a fresh output and a server log for live served mode")
    if args.mode != "served" and (args.baseline or args.replay_artifact):
        parser.error("Baseline/replay applies only to served mode")
    if not 2 <= args.decode_tokens <= 32:
        parser.error("Use 2..32 total output tokens")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False, mode=args.mode, performance_claim=False, runs=[], checks={})
    try:
        current = provenance(args.execution_manifest)
        if args.replay_artifact:
            result = json.loads(args.replay_artifact.read_text())
            if result["mode"] != "served" or model_sources(result["source"]) != model_sources(current):
                raise ValueError("Saved serving artifact mode/source differs")
            result["replay_diagnostic_source"] = current
            result["passed"] = False
            result["replayed_from"] = str(args.replay_artifact)
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            contract = json.loads((MODEL_DIR / "doc/context_contract.json").read_text())
            if Path(args.tokenizer).resolve().name != contract["hf_revision"]:
                raise ValueError("Tokenizer/checkpoint path is not the pinned revision")
            result.update(
                source=current,
                cases=encoded_cases(tokenizer),
                capacity=contract["supported_context"],
                precision=json.loads((MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json").read_text()),
                tokenizer=str(Path(args.tokenizer).resolve()),
                decode_tokens=args.decode_tokens,
                scope="greedy_prefill_and_constant_width_decode_selected_host_logprobs_in_serving_full_logits_in_standalone",
                limitations=[
                    "Tests greedy histories for the failing prompt families, not the original stochastic divergent prefixes or raw device-sampler RNG.",
                    "Selected top20 logprobs cannot establish full serving-logit equality or expose a uniform additive shift.",
                    "Solo/pair numerical comparison is diagnostic; no cross-active-width decode guarantee is implied.",
                    "Run with no unrelated clients. Tracking0 constant-width coverage is lifecycle inference from actual initial slots, equal fixed outputs and decode counts.",
                ],
            )
            (served if args.mode == "served" else standalone)(args, result)
        evaluate(result, json.loads(args.baseline.read_text()) if args.baseline else None)
    except Exception:
        result["passed"] = False
        result["error"] = traceback.format_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result.get(k) for k in ("passed", "checks", "standalone_comparison_pending", "error")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
