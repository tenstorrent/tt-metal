# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit host-logprob diagnostic; never a device-sampling performance test."""

import argparse
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
# readiness_aime24_chat.refpt entry 0, same histories as
# doc/full_model/prefill_factored_full48_v1.json and probe_prefill_trace_signatures.
PREFIX = [
    248045,
    8678,
    198,
    24342,
    286,
    4879,
    369,
    716,
    310,
    830,
    11553,
    13,
    5044,
    1683,
    15060,
    1472,
    279,
    3274,
    11,
    9307,
    1328,
    30800,
    11,
    2814,
    47675,
    25605,
    11,
    321,
    60445,
    55404,
    11,
    27224,
    11,
    321,
    30246,
    303,
    279,
    1534,
    4087,
    13,
    248046,
    198,
    248045,
    846,
    198,
    11179,
    6353,
]
PROMPTS = {"a33": PREFIX[:33], "b47": PREFIX[:47]}
LAYOUTS = {"solo_a": ["a33"], "solo_b": ["b47"], "ab": ["a33", "b47"], "ba": ["b47", "a33"]}


def source_manifest():
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        "precision": json.loads((MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json").read_text()),
        "sources": [
            {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted((MODEL_DIR / "tt").rglob("*"))
            if p.is_file() and p.suffix in (".py", ".cpp", ".hpp", ".h")
        ],
    }


def compare(left, right, atol):
    """Compare named token probabilities, allowing only equal-score cutoff ties."""
    if left["prompt_token_ids"] != right["prompt_token_ids"]:
        raise ValueError("Cannot compare different token histories")
    a, b = left["top_logprobs"], right["top_logprobs"]
    common = sorted(set(a) & set(b))
    delta = max((abs(a[k] - b[k]) for k in common), default=None)
    boundary_ties = (
        bool(common)
        and (a.keys() == b.keys() or abs(min(a.values()) - min(b.values())) <= atol)
        and all(abs(a[k] - min(a.values())) <= atol for k in a.keys() - b.keys())
        and all(abs(b[k] - min(b.values())) <= atol for k in b.keys() - a.keys())
    )
    inversions = sum(
        (a[x] - a[y]) * (b[x] - b[y]) < 0 and abs(a[x] - a[y]) > atol and abs(b[x] - b[y]) > atol
        for i, x in enumerate(common)
        for y in common[i + 1 :]
    )
    same_token = left["sampled_token_id"] == right["sampled_token_id"]
    passed = same_token and delta is not None and delta <= atol and boundary_ties and not inversions
    return dict(
        passed=passed,
        exact_maps=a == b,
        sampled_token_equal=same_token,
        top_set_equal=a.keys() == b.keys(),
        common_tokens=len(common),
        max_abs_logprob_delta=delta,
        rank_inversions=inversions,
        unmatched_only_at_cutoff_ties=boundary_ties,
        left_only=sorted(a.keys() - b.keys()),
        right_only=sorted(b.keys() - a.keys()),
    )


def comparisons(runs, atol):
    indexed = {(r["layout"], r["repeat"], row["prompt_id"]): row for r in runs for row in r["rows"]}
    groups = {"repeat": [], "cross_position": [], "solo_vs_batch": []}
    for run in runs:
        for row in run["rows"]:
            key = row["prompt_id"]
            if run["repeat"]:
                other = indexed[(run["layout"], 0, key)]
                groups["repeat"].append(
                    dict(layout=run["layout"], repeat=run["repeat"], prompt_id=key, **compare(other, row, atol))
                )
            if run["layout"] == "ba":
                groups["cross_position"].append(
                    dict(repeat=run["repeat"], prompt_id=key, **compare(indexed[("ab", run["repeat"], key)], row, atol))
                )
            if run["layout"] in ("ab", "ba"):
                solo = "solo_a" if key == "a33" else "solo_b"
                groups["solo_vs_batch"].append(
                    dict(
                        layout=run["layout"],
                        repeat=run["repeat"],
                        prompt_id=key,
                        **compare(indexed[(solo, run["repeat"], key)], row, atol),
                    )
                )
    return groups


def normalize(response, prompt_ids, top_k, atol):
    choices = {c["index"]: c for c in response["choices"]}
    if (
        len(response["choices"]) != len(prompt_ids)
        or len(choices) != len(prompt_ids)
        or set(choices) != set(range(len(prompt_ids)))
    ):
        raise ValueError("Missing, duplicate or reordered completion choice indices")
    expected_prompt_tokens = sum(len(PROMPTS[p]) for p in prompt_ids)
    if response["usage"]["prompt_tokens"] != expected_prompt_tokens or response["usage"]["completion_tokens"] != len(
        prompt_ids
    ):
        raise ValueError("Server token usage differs from exact input/output shapes")
    rows = []
    for i, key in enumerate(prompt_ids):
        choice = choices[i]
        ids = choice.get("token_ids")
        lp = choice["logprobs"]
        if not ids or len(ids) != 1 or len(lp["top_logprobs"]) != 1:
            raise ValueError("Expected one generated token and one top-logprob map")
        values = lp["top_logprobs"][0]
        if len(values) < top_k or any(not k.startswith("token_id:") for k in values):
            raise ValueError("Missing token-ID keyed top-logprobs; text keys are ambiguous")
        for k, v in values.items():
            int(k.removeprefix("token_id:"))
            if not isinstance(v, (float, int)) or not math.isfinite(v) or v > atol:
                raise ValueError("Non-finite or positive selected logprob")
        selected = f"token_id:{ids[0]}"
        if selected not in values or abs(values[selected] - max(values.values())) > atol:
            raise ValueError("Greedy token is not a returned maximum")
        if len(lp["token_logprobs"]) != 1 or abs(lp["token_logprobs"][0] - values[selected]) > atol:
            raise ValueError("Selected-token logprob disagrees with top-logprob map")
        rows.append(
            dict(
                prompt_id=key,
                prompt_token_ids=PROMPTS[key],
                submitted_index=i,
                sampled_token_id=ids[0],
                top_logprobs=values,
            )
        )
    return rows


def read_events(path, offset):
    if path is None:
        return []
    with path.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read().decode(errors="replace").splitlines()
    return [json.loads(line.split("QWEN_VLLM_EVENT ", 1)[1]) for line in lines if "QWEN_VLLM_EVENT " in line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--standalone-baseline", type=Path)
    parser.add_argument("--replay-artifact", type=Path, help="Reanalyze saved responses without HTTP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--pause-batched-admission", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or not 2 <= args.top_logprobs <= 20 or not 0 <= args.atol < 1:
        parser.error("Need repeats>=2, 2..20 top logprobs, and 0<=atol<1")
    if args.output.exists():
        parser.error("Output must be fresh")
    result = dict(
        passed=False,
        scope="selected_host_logprobs_not_full_logits_or_performance",
        prompts=PROMPTS,
        source=source_manifest(),
        runs=[],
        checks={},
        atol=args.atol,
        top_logprobs=args.top_logprobs,
        limitations=[
            "One output tests prefill next-token logits, not steady decode.",
            "Top logprobs do not expose unreturned logits or an additive logit offset.",
            "API batch order alone does not prove actual device batch positions.",
            "Solo-vs-batch differences are geometry-dependent diagnostics, not stochastic nondeterminism.",
        ],
    )
    try:
        if args.replay_artifact:
            saved = json.loads(args.replay_artifact.read_text())
            if saved["prompts"] != PROMPTS:
                raise ValueError("Replay artifact has different prompts")
            result["source"] = saved["source"]
            result["runs"] = saved["runs"]
        else:
            for repeat in range(args.repeats):
                for layout, prompt_ids in LAYOUTS.items():
                    body = dict(
                        model=args.model,
                        prompt=[PROMPTS[p] for p in prompt_ids],
                        temperature=0,
                        top_k=1,
                        top_p=1.0,
                        max_tokens=1,
                        ignore_eos=True,
                        logprobs=args.top_logprobs,
                        return_token_ids=True,
                        return_tokens_as_token_ids=True,
                        add_special_tokens=False,
                        repetition_penalty=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        request_id=f"logit-check-{time.time_ns()}-{layout}",
                    )
                    offset = args.server_log.stat().st_size if args.server_log else 0
                    run = dict(
                        layout=layout,
                        repeat=repeat,
                        prompt_ids=prompt_ids,
                        request=body,
                        requested_shape=[len(prompt_ids), [len(PROMPTS[p]) for p in prompt_ids], 1],
                    )
                    result["runs"].append(run)
                    start = time.monotonic()
                    req = urllib.request.Request(
                        args.server_url.rstrip("/") + "/v1/completions",
                        data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    try:

                        def send():
                            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                                return response.status, response.read().decode()

                        if args.pause_batched_admission and len(prompt_ids) > 1:
                            # Pause only before admission, retaining cache state.
                            # Actual B2 events below remain the coverage oracle.
                            def control(suffix):
                                request = urllib.request.Request(
                                    args.server_url.rstrip("/") + suffix, data=b"", method="POST"
                                )
                                with urllib.request.urlopen(request, timeout=30) as response:
                                    response.read()

                            with ThreadPoolExecutor(max_workers=1) as pool:
                                control("/pause?mode=keep&clear_cache=false")
                                try:
                                    pending = pool.submit(send)
                                    time.sleep(1)
                                    if pending.done():
                                        raise RuntimeError("Batched request completed before admission resume")
                                finally:
                                    control("/resume")
                                run["status"], run["raw_response"] = pending.result()
                            run["paused_before_admission"] = True
                        else:
                            run["status"], run["raw_response"] = send()
                    except urllib.error.HTTPError as error:
                        run["status"] = error.code
                        run["raw_response"] = error.read().decode()
                        raise
                    finally:
                        run["seconds"] = time.monotonic() - start
                        run["server_events"] = read_events(args.server_log, offset)
                    run["response"] = json.loads(run["raw_response"])
                    run["rows"] = normalize(run["response"], prompt_ids, args.top_logprobs, args.atol)
        runs = result["runs"]
        expected = {(layout, r) for r in range(args.repeats) for layout in LAYOUTS}
        if {(r["layout"], r["repeat"]) for r in runs} != expected or len(runs) != len(expected):
            raise ValueError("Incomplete or duplicate workload")
        for run in runs:
            run["rows"] = normalize(run["response"], run["prompt_ids"], args.top_logprobs, args.atol)
        result["comparisons"] = comparisons(runs, args.atol)
        positions = {key: [] for key in PROMPTS}
        for run in runs:
            if run["layout"] not in ("ab", "ba"):
                continue
            run["observed_final_batch"] = False
            for event in run.get("server_events", []):
                if event.get("kind") != "prefill_submission" or len(event["ends"]) != 2:
                    continue
                if sorted(event["ends"]) != [33, 47] or not all(event.get("sample_mask", [])):
                    continue
                if event["starts"] != [0, 0]:
                    continue
                run["observed_final_batch"] = True
                for i, end in enumerate(event["ends"]):
                    positions["a33" if end == 33 else "b47"].append(
                        dict(
                            layout=run["layout"],
                            repeat=run["repeat"],
                            submitted_index=i,
                            physical_slot=event["slots"][i],
                            starts=event["starts"],
                            sampling_mode=event["sampling_mode"],
                        )
                    )
        result["observed_positions"] = positions
        checks = result["checks"]
        checks["run_to_run_selected_logprobs"] = all(c["passed"] for c in result["comparisons"]["repeat"])
        checks["reversed_order_selected_logprobs"] = all(c["passed"] for c in result["comparisons"]["cross_position"])
        checks["actual_cross_batch_positions"] = all(
            {p["submitted_index"] for p in rows} == {0, 1} and len({p["physical_slot"] for p in rows}) >= 2
            for rows in positions.values()
        ) and all(r["observed_final_batch"] for r in runs if r["layout"] in ("ab", "ba"))
        checks["explicit_host_logprob_route"] = bool(sum(positions.values(), [])) and all(
            p["sampling_mode"] == "host" for rows in positions.values() for p in rows
        )
        checks["standalone_selected_policy_baseline"] = False
        if args.standalone_baseline:
            baseline = json.loads(args.standalone_baseline.read_text())
            result["standalone_baseline"] = str(args.standalone_baseline)
            checks["same_source_manifest"] = baseline["source"]["sources"] == result["source"]["sources"]
            if (
                not baseline["passed"]
                or baseline["precision"] != result["source"]["precision"]
                or baseline["layer_indices"] != list(range(48))
            ):
                raise ValueError("Standalone baseline did not pass with the same selected all48 policy")
            base = {(r["layout"], r["repeat"], row["prompt_id"]): row for r in baseline["runs"] for row in r["rows"]}
            result["standalone_comparisons"] = [
                dict(
                    layout=r["layout"],
                    repeat=r["repeat"],
                    prompt_id=row["prompt_id"],
                    **compare(base[(r["layout"], r["repeat"], row["prompt_id"])], row, args.atol),
                )
                for r in runs
                for row in r["rows"]
            ]
            checks["standalone_selected_policy_baseline"] = all(c["passed"] for c in result["standalone_comparisons"])
        result["passed"] = all(checks.values())
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(passed=result["passed"], checks=result["checks"], error=result.get("error"), output=str(args.output))
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
