# SPDX-License-Identifier: Apache-2.0
"""Live P32 allocator/reordering differential with matched physical prefill shapes.

Control: sequential requests on the SAME model/config launched with
--no-async-scheduling. A startup-log async_scheduling=False record is mandatory.
Mixed: arrivals/completions on the normal async server. Can collect mixed first,
then compare offline after a control-server run. Neither phase launches servers.

Every prompt is exactly128 tokens; at most three arrive together (384<512).
Actual adapter events must prove every prefill is sixteen8-token native chunks.
This avoids the independently diagnosed singleton/chunk-tail numerical effect.
The mixed-shape lifecycle check remains a separate required test.

Events expose page-only refresh decisions/counters and physical state-slot row
order, not allocator block IDs. We combine those with completed token counts and
source-backed P32 addressing. No per-request physical page-ID mapping is claimed.

Optional --control-strategy host-logprobs collects sequential diagnostic controls
on the same async server (requires QWEN_FLASH_ALLOW_HOST_SAMPLING=1 at launch).
It requests logprobs=2, proves the pinned plugin drains/fully reloads host steps,
and records greedy tie diagnostics. It is never a performance measurement.
The default no-async control remains available if host/device tie choices differ.
"""
import argparse
import ast
import asyncio
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path

import aiohttp

PROMPTS = {
    "A": ("Explain how rain forms and reaches rivers in a clear paragraph.", 224),
    "B": ("Explain how a bicycle carries a rider in a clear paragraph.", 112),
    "C": ("Explain why tree leaves are green in a clear paragraph.", 176),
    "D": ("Explain how a baker makes bread in a clear paragraph.", 160),
    "E": ("Explain how Earth travels around the Sun in a clear paragraph.", 112),
}
OVERLAP = re.compile(r"TT async decode: (\d+) submission\(s\) overlapped an outstanding step, (\d+) of them")
SUBMISSIONS = re.compile(r"TT submissions: (\d+) ordinary decode, (\d+) verify, (\d+) overlapped an outstanding step")
ASYNC_FALSE = re.compile(r"[\"']?async_scheduling[\"']?\s*[:=]\s*(?:False|false)\b")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_contract(path):
    data = json.loads(path.read_text())
    return {key: data[key] for key in ("adapter", "runtime", "batch", "context", "public_pool_tokens", "cache_shapes")}


def read_events(path):
    # The writer flushes one JSON line per event; ignore only an unfinished tail.
    raw = path.read_text().splitlines(keepends=True)
    return [json.loads(line) for line in raw if line.endswith("\n")]


def overlap_counts(text):
    rows = [(int(a), int(b)) for a, b in OVERLAP.findall(text)]
    # Regular submission logs prevent a quiet power-of-two interval from being
    # mistaken for no overlap during a workload on an already-running server.
    counts = [a for a, _ in rows] + [int(overlap) for _, _, overlap in SUBMISSIONS.findall(text)]
    return dict(overlapped=max(counts, default=0), unsafe=max((b for _, b in rows), default=0))


def host_control_source_proof():
    """Record and guard the exact local plugin/adapter clauses used by this test."""
    spec = importlib.util.find_spec("vllm_tt_plugin")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot inspect the installed TT plugin for host-control proof")
    plugin = Path(spec.origin).parent
    adapter = Path(__file__).resolve().parents[1] / "tt/generator_vllm.py"
    sources = {
        "runner": plugin / "model_runner.py",
        "async": plugin / "async_decode.py",
        "adapter": adapter,
    }
    functions = {}
    for label, path in sources.items():
        functions[label] = {
            node.name: ast.unparse(node)
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    clauses = [
        ("runner", "check_perform_device_sampling", "if num_devices not in (8, 32):\n    return False"),
        (
            "runner",
            "check_perform_device_sampling",
            "if max_lp > 0 and (not self.supports_topk_logprobs):\n    return False",
        ),
        ("async", "steady_decode_scheduler_invariants_met", "if max_num_logprobs is not None:\n    return False"),
        ("async", "can_use_steady_decode_fast_path", "if not model_input.perform_device_sampling:\n    return False"),
        ("async", "must_drain_pending_async_steps", "if not steady_decode_candidate:\n    return True"),
        ("async", "wait_for_all_pending_async_steps", "step.ensure_finalized()"),
        ("async", "plan_decode_reload", "or not device_sampling"),
        ("runner", "execute_model", "self.async_decode.wait_for_all_pending_async_steps()"),
        (
            "runner",
            "build_model_input",
            "self.async_decode.apply_completed_decode_steps_before_build(scheduler_output)",
        ),
        ("adapter", "_sampling", "return 'host'"),
        ("adapter", "decode_forward", "tokens if reload_inputs else None"),
    ]
    checked = []
    for label, name, clause in clauses:
        source = functions[label][name]
        if re.sub(r"\s+", "", clause) not in re.sub(r"\s+", "", source):
            raise RuntimeError(f"Host-control source guard changed: {label}.{name}: {clause}")
        checked.append(dict(source=label, function=name, required_clause=clause, implementation=source))
    execute = functions["runner"]["execute_model"]
    build = functions["runner"]["build_model_input"]
    if execute.index("self.async_decode.wait_for_all_pending_async_steps()") > execute.index(
        "self.build_model_input(scheduler_output, None)"
    ) or build.index("self.async_decode.apply_completed_decode_steps_before_build(scheduler_output)") > build.index(
        "self._prepare_model_inputs(scheduler_output, grammar_output)"
    ):
        raise RuntimeError("Host-control drain/apply-before-build ordering changed")
    return dict(
        status="pass",
        files={label: dict(path=str(path), sha256=digest(path)) for label, path in sources.items()},
        checked=checked,
        interpretation="Qwen TP4 positive logprobs selects explicit host compatibility; every host step is ineligible for overlap and reloads authoritative inputs after pending results are finalized/applied before build.",
        tie_limit="CPU greedy uses lowest-index argmax; Sampling1D uses default unstable top-k. A tie can cause exact-token failure; this diagnostic never excuses a mismatch.",
    )


def logprob_rows(value):
    """Keep each returned token's raw top-logprobs and top-two margin."""
    if not value:
        return []
    rows = []
    for i, token in enumerate(value.get("tokens") or []):
        top = (value.get("top_logprobs") or [None] * len(value["tokens"]))[i]
        ranked = sorted((top or {}).items(), key=lambda pair: pair[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) >= 2 else None
        rows.append(dict(token=token, logprob=value["token_logprobs"][i], top_logprobs=top, top_two_margin=margin))
    return rows


def audit_events(before, during, phase, control_strategy="no-async"):
    failures = []
    captures = {e["signature"]: e["trace_id"] for e in before if e["event"] == "capture"}
    new_captures = []
    for event in during:
        if event["event"] in ("shutdown_release", "release", "retire", "evict"):
            failures.append("Trace release/retirement during traffic")
        if event["event"] == "capture":
            new_captures.append(event)
            if event["signature"] in captures:
                failures.append("Recapture of existing signature " + event["signature"])
            captures[event["signature"]] = event["trace_id"]
    schedules = [e for e in during if e["event"] == "prefill_schedule"]
    chunks = [e for e in during if e["event"] == "prefill_chunk"]
    decodes = [e for e in during if e["event"] == "decode_submit"]
    if sum(len(e["ends"]) for e in schedules) != len(PROMPTS):
        failures.append(
            "Expected exactly five full-prompt admissions; external traffic or scheduler splitting detected"
        )
    if any(any(s != 0 for s in e["starts"]) or any(end != 128 for end in e["ends"]) for e in schedules):
        failures.append("Prefill was externally split; physical signature control does not apply")
    observed_positions = []
    for e in chunks:
        if e["chunk_length"] != 8:
            failures.append("Native prefill chunk length differed from8")
        for position, active in zip(e["start_positions"], e["active"]):
            if active:
                observed_positions.append(position)
                if position not in range(0, 128, 8):
                    failures.append("Native active prefill position outside0,8,...,120")
    if sorted(observed_positions) != sorted(list(range(0, 128, 8)) * len(PROMPTS)):
        failures.append("Incomplete per-prompt native shape coverage")
    page_only, unchanged, reorder, concurrent = [], [], [], []
    previous_order = None
    previous_refresh = None
    for e in decodes:
        active_order = [slot for slot, active in zip(e["slots"], e["active"]) if active]
        refresh = e["counters"].get("page_table_refreshes", 0)
        row = dict(
            time=e["time"],
            active_slots=active_order,
            page_table_refreshes=refresh,
            reload_inputs=e["reload_inputs"],
            reload_page_table=e["reload_page_table"],
        )
        if len(active_order) >= 2:
            concurrent.append(row)
        if e["reload_page_table"] and not e["reload_inputs"]:
            if previous_refresh is not None and refresh <= previous_refresh:
                failures.append("Page-only reload did not refresh the table")
            page_only.append(row)
        if not e["reload_page_table"] and not e["reload_inputs"]:
            if previous_refresh is not None and refresh != previous_refresh:
                # A prefill can legitimately change the shared backing table
                # between decode events; do not infer an every-token copy here.
                row["intervening_activity_or_refresh"] = True
            else:
                unchanged.append(row)
        if previous_order is not None:
            moved = [
                slot
                for slot in active_order
                if slot in previous_order and active_order.index(slot) != previous_order.index(slot)
            ]
            if moved:
                reorder.append(dict(**row, previous_active_slots=previous_order, moved_slots=moved))
        previous_order, previous_refresh = active_order, refresh
    if not decodes:
        failures.append("No adapter decode events")
    if phase == "control" and control_strategy == "host-logprobs":
        if any(e.get("sampling") != "host" for e in schedules):
            failures.append("Host control prefill did not use explicit host compatibility")
        if any(e.get("sampling") != "host" or not e["reload_inputs"] or e["reload_page_table"] for e in decodes):
            failures.append("Host control decode did not use authoritative host/full-input reload")
        if any(sum(e["active"]) != 1 for e in decodes):
            failures.append("Host control contains concurrent active requests")
    if phase == "mixed":
        if any(e.get("sampling") != "greedy" for e in schedules + decodes):
            failures.append("Mixed allocator workload did not remain on canonical device greedy sampling")
        if len(page_only) < 3:
            failures.append("Fewer than3 automatic page-only growth refreshes")
        if len(unchanged) < 16:
            failures.append("Insufficient unchanged-table steady decode evidence")
        if not reorder:
            failures.append("No existing physical state slot moved to a different active row")
        if not concurrent or max(len(e["active_slots"]) for e in concurrent) < 3:
            failures.append("Three-request concurrent decode was not observed")
    return dict(
        status="pass" if not failures else "failed",
        failures=sorted(set(failures)),
        prefill_admissions=schedules,
        native_chunk_count=len(chunks),
        active_native_chunk_rows=len(observed_positions),
        physical_prefill_signature=dict(
            fixed_batch="runtime.batch", native_sequence=8, absolute_positions=list(range(0, 128, 8))
        ),
        page_only_refreshes=page_only,
        unchanged_table_steps=len(unchanged),
        row_reorders=reorder,
        max_active_decode=max((sum(e["active"]) for e in decodes), default=0),
        baseline_trace_ids={e["signature"]: e["trace_id"] for e in before if e["event"] == "capture"},
        new_captures=new_captures,
        evidence_limit="Events expose whole-table refreshes and state-slot order, not per-request allocator block IDs; consumed logical P32 crossings follow returned token counts.",
    )


def compare(control, mixed):
    problems = []
    if control.get("phase") != "control" or control.get("status") != "control-valid":
        problems.append("Control artifact has not passed non-overlapped and shape gates")
    if (
        mixed.get("phase") != "mixed"
        or mixed.get("event_audit", {}).get("status") != "pass"
        or mixed.get("status") not in ("collected-awaiting-control", "pass")
        or mixed.get("failures")
    ):
        problems.append("Mixed artifact lacks passing allocator/reordering/overlap evidence")
    for key in ("runtime_contract", "prompts"):
        if control.get(key) != mixed.get(key):
            problems.append("Control/mixed mismatch: " + key)
    sampling = dict(control.get("sampling", {}))
    if control.get("control_strategy", "no-async") == "host-logprobs":
        if (
            sampling.pop("logprobs", None) != 2
            or control.get("control_proof", {}).get("source_guards", {}).get("status") != "pass"
        ):
            problems.append("Host control lacks explicit logprobs=2/source proof")
        if any(sampling.get(key) != value for key, value in (("temperature", 0), ("top_k", 1), ("top_p", 1))):
            problems.append("Host control is not the required greedy profile")
        if not control.get("server_instance", {}).get("pid") or control.get("server_instance") != mixed.get(
            "server_instance"
        ):
            problems.append("Host control and mixed workload did not use the same running server/captures")
    if sampling != mixed.get("sampling"):
        problems.append("Control/mixed mismatch: sampling (only explicit host-control logprobs may differ)")
    rows = []
    for name in PROMPTS:
        left = next((r for r in control.get("requests", []) if r["name"] == name), None)
        right = next((r for r in mixed.get("requests", []) if r["name"] == name), None)
        if left is None or right is None:
            problems.append("Missing completed request " + name)
            continue
        a, b = left["token_ids"], right["token_ids"]
        same = a == b
        divergence = next(
            (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if len(a) != len(b) else None
        )
        diagnostics = left.get("logprobs", [])
        rows.append(
            dict(
                name=name,
                exact_tokens=same,
                output_tokens=len(b),
                first_divergence=divergence,
                control_logprobs_at_divergence=diagnostics[divergence]
                if divergence is not None and divergence < len(diagnostics)
                else None,
                control_top_tie_positions=[i for i, row in enumerate(diagnostics) if row["top_two_margin"] == 0],
            )
        )
        if not same:
            problems.append("Exact token mismatch for " + name)
    return dict(status="pass" if not problems else "failed", failures=problems, comparisons=rows)


async def collect(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompts = {}
    for name, (topic, _) in PROMPTS.items():
        text = tokenizer.apply_chat_template(
            [dict(role="user", content=topic)], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.encode(text)
        if len(ids) > 128:
            raise ValueError("Prompt exceeds128 tokens")
        end = ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        ids[end:end] = [198] * (128 - len(ids))
        assert len(ids) == 128
        prompts[name] = ids
    before = read_events(args.events)
    log_before = args.server_log.read_text()
    command = args.server_command.read_text()
    host_control = args.phase == "control" and args.control_strategy == "host-logprobs"
    source_proof = host_control_source_proof() if host_control else None
    if args.phase == "control" and not host_control:
        if "--no-async-scheduling" not in command or not ASYNC_FALSE.search(log_before):
            raise RuntimeError(
                "Control requires --no-async-scheduling command AND actual startup-log async_scheduling=False"
            )
        if overlap_counts(log_before)["overlapped"]:
            raise RuntimeError("Control server log already reports overlap; use a fresh synchronous-server log")
    runtime = runtime_contract(args.runtime_report)
    if runtime["batch"] < 3:
        raise ValueError("At least three fixed serving lanes required")
    report = dict(
        status="running",
        phase=args.phase,
        control_strategy=args.control_strategy if args.phase == "control" else None,
        diagnostic_only=True,
        url=args.url,
        model=args.model,
        runtime_contract=runtime,
        server_instance=dict(
            pid=json.loads(args.runtime_report.read_text()).get("pid"),
            capture_ids={e["signature"]: e["trace_id"] for e in before if e["event"] == "capture"},
        ),
        prompts=prompts,
        sampling=dict(temperature=0, top_k=1, top_p=1, ignore_eos=True, return_token_ids=True),
        requests=[],
        arrivals=[],
        source_sha256=digest(Path(__file__)),
        server_command=command,
        server_log=str(args.server_log),
        event_path=str(args.events),
        runtime_path=str(args.runtime_report),
        initial_event_count=len(before),
        control_proof=dict(
            startup_async_false=bool(ASYNC_FALSE.search(log_before)),
            no_async_command="--no-async-scheduling" in command,
            source_contract="Positive logprobs selects drained, authoritative host compatibility"
            if host_control
            else "TTModelRunner.async_decode_scheduling=scheduler_config.async_scheduling; TTAsyncDecodeController.steady_decode_base_enabled returnsFalse when disabled",
            source_guards=source_proof,
        ),
    )
    if host_control:
        report["sampling"]["logprobs"] = 2

    def persist():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    persist()
    live = {}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:

            async def request(name):
                count = PROMPTS[name][1]
                row = dict(
                    name=name,
                    prompt_tokens=128,
                    requested_output_tokens=count,
                    start_time=time.monotonic(),
                    token_ids=[],
                    token_arrivals=[],
                    logprobs=[],
                    text="",
                    complete=False,
                )
                live[name] = row
                payload = dict(
                    model=args.model,
                    prompt=prompts[name],
                    max_tokens=count,
                    stream=True,
                    stream_options=dict(include_usage=True),
                    **report["sampling"],
                )
                usage = None
                async with session.post(args.url + "/v1/completions", json=payload) as response:
                    if response.status != 200:
                        raise RuntimeError(await response.text())
                    async for line in response.content:
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                            continue
                        data = json.loads(line[6:])
                        if "error" in data:
                            raise RuntimeError(data)
                        row["request_id"] = data.get("id", row.get("request_id"))
                        if data.get("usage"):
                            usage = data["usage"]
                        for choice in data.get("choices", []):
                            ids = choice.get("token_ids") or []
                            if ids:
                                row["token_ids"].extend(ids)
                                row["token_arrivals"].append(dict(time=time.monotonic(), count=len(row["token_ids"])))
                            row["text"] += choice.get("text", "")
                            row["logprobs"].extend(logprob_rows(choice.get("logprobs")))
                            if (
                                choice.get("prompt_token_ids") is not None
                                and choice["prompt_token_ids"] != prompts[name]
                            ):
                                raise AssertionError("Server returned different prompt token IDs")
                row.update(end_time=time.monotonic(), usage=usage, complete=True)
                assert usage and usage["prompt_tokens"] == 128 and usage["completion_tokens"] == count, (name, usage)
                assert len(row["token_ids"]) == count, (name, len(row["token_ids"]), count)
                if row["logprobs"] and len(row["logprobs"]) != count:
                    raise AssertionError((name, "Logprob/token count mismatch", len(row["logprobs"]), count))
                for index, diagnostic in enumerate(row["logprobs"]):
                    diagnostic.update(output_index=index, token_id=row["token_ids"][index])
                row["logprobs_returned"] = bool(row["logprobs"])
                # Prefill emits output0; decode consumes positions128..128+N-2.
                row["consumed_new_page_boundaries"] = list(range(128, 128 + count - 1, 32))
                assert len(row["consumed_new_page_boundaries"]) >= 3
                report["requests"].append(row)
                persist()
                print(
                    json.dumps(dict(name=name, completed=count, page_boundaries=row["consumed_new_page_boundaries"])),
                    flush=True,
                )
                return row

            if args.phase == "control":
                for name in PROMPTS:
                    await request(name)
            else:
                tasks = {name: asyncio.create_task(request(name)) for name in ("A", "B", "C")}
                await tasks["B"]
                remaining = [n for n, t in tasks.items() if not t.done()]
                if len(remaining) < 2:
                    raise AssertionError("B did not complete while A/C still decode")
                report["arrivals"].append(
                    dict(name="D", trigger="B completion", still_running=remaining, time=time.monotonic())
                )
                tasks["D"] = asyncio.create_task(request("D"))
                await tasks["C"]
                remaining = [n for n, t in tasks.items() if not t.done()]
                if not remaining:
                    raise AssertionError("C completion left no decode work for E arrival")
                report["arrivals"].append(
                    dict(name="E", trigger="C completion", still_running=remaining, time=time.monotonic())
                )
                tasks["E"] = asyncio.create_task(request("E"))
                await asyncio.gather(*tasks.values())
        after = read_events(args.events)
        if after[: len(before)] != before:
            raise RuntimeError("Event file rotated/reset during workload")
        during = after[len(before) :]
        segment = args.output.with_suffix(".adapter_events.jsonl")
        segment.write_text("".join(json.dumps(e) + "\n" for e in during))
        report["event_segment"] = str(segment)
        report["event_audit"] = audit_events(before, during, args.phase, args.control_strategy)
        before_overlap = overlap_counts(log_before)
        log_after = args.server_log.read_text()
        if not log_after.startswith(log_before):
            raise RuntimeError("Server log rotated/reset during workload")
        after_overlap = overlap_counts(log_after)
        report["overlap_evidence"] = dict(before=before_overlap, after=after_overlap)
        failures = list(report["event_audit"]["failures"])
        if after_overlap["unsafe"] > before_overlap["unsafe"]:
            failures.append("Unsafe async overlap reported")
        if host_control:
            observations = []
            for line in log_after[len(log_before) :].splitlines():
                old, regular = OVERLAP.search(line), SUBMISSIONS.search(line)
                if old or regular:
                    observations.append(int(old.group(1)) if old else int(regular.group(3)))
            report["overlap_evidence"].update(
                in_control_interval=observations,
                observed_interval_no_increase=len(observations) >= 2 and min(observations) == max(observations),
                coverage_limit="The first interval log may expose prior unlogged overlaps. Compare observations inside this interval; source guards and every decode event cover unlogged edges.",
            )
            if len(observations) >= 2 and min(observations) != max(observations):
                failures.append("Host control interval reports an overlap counter change")
        elif args.phase == "control" and after_overlap["overlapped"]:
            failures.append("Control server overlapped decode")
        if args.phase == "mixed" and after_overlap["overlapped"] <= before_overlap["overlapped"]:
            failures.append("No measured async overlap increment; configuration alone is insufficient")
        report["failures"] = failures
        report["status"] = (
            "failed" if failures else ("control-valid" if args.phase == "control" else "collected-awaiting-control")
        )
        if args.phase == "mixed" and args.control:
            report["control_comparison"] = compare(json.loads(args.control.read_text()), report)
            report["status"] = "pass" if not failures and report["control_comparison"]["status"] == "pass" else "failed"
        persist()
        if report["status"] == "failed":
            raise AssertionError(report.get("control_comparison", failures))
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        report["partial_requests"] = [r for r in live.values() if not r["complete"]]
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("control", "mixed", "compare"), required=True)
    parser.add_argument("--control-strategy", choices=("no-async", "host-logprobs"), default="no-async")
    parser.add_argument("--url", default="http://localhost:8019")
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument(
        "--checkpoint", default="/home/vkovacevic/qwen38-flash/hf-local/de4b8e4d43b917e7706784d8bb445c9af86a3540"
    )
    parser.add_argument("--events", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--server-command", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--mixed", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Use a fresh output")
    if args.phase == "compare":
        if not args.control or not args.mixed:
            parser.error("compare requires --control and --mixed")
        result = compare(json.loads(args.control.read_text()), json.loads(args.mixed.read_text()))
        result.update(
            control=str(args.control),
            mixed=str(args.mixed),
            control_sha256=digest(args.control),
            mixed_sha256=digest(args.mixed),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if result["status"] != "pass":
            raise SystemExit(1)
    else:
        for name in ("events", "server_log", "server_command", "runtime_report"):
            if getattr(args, name) is None:
                parser.error("collection requires --" + name.replace("_", "-"))
        asyncio.run(collect(args))


if __name__ == "__main__":
    main()
