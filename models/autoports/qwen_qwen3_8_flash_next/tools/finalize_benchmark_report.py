#!/usr/bin/env python3
"""Append observed provenance/caveats to an already generated benchmark report.

Never rewrites upstream tables, results, raw responses or summary.json. Repeated
execution replaces only this helper's marked appendix. No device APIs or model
runtime imports are used. Invocation UTC is an explicit recorded input, never
inferred from filesystem modification times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO = MODEL_DIR.parents[2]
START = "<!-- qwen-benchmark-provenance:start -->"
END = "<!-- qwen-benchmark-provenance:end -->"


def read(path):
    return json.loads(Path(path).read_text())


def escape(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Recorded timestamps must include a UTC offset")
    return result


def link(path, output, label=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing appendix evidence: {path}")
    relative = os.path.relpath(path, output)
    return f"[{label or path.name}](<{relative}>)"


def commit_identity(value):
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", value + "^{commit}"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    subject = subprocess.run(
        ["git", "show", "-s", "--format=%s", resolved], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    return resolved, subject


def precise_throughput(summary):
    """Display recorded rates without changing or recomputing measurements."""
    rows = [
        (f"B{batch}", row)
        for batch, row in sorted(summary.get("performance", {}).items(), key=lambda item: int(item[0]))
    ]
    for name, row in summary.get("lone_user", {}).items():
        label = {"functional": "FUNCTIONAL", "long_context": "Long context"}.get(name, name)
        shape = f"{row.get('requested_input_tokens', 'unrecorded')}/{row.get('requested_output_tokens', 'unrecorded')}"
        rows.append((f"{label} ({shape}, concurrency 1)", row))
    if not rows:
        return []
    lines = [
        "Recorded throughput with six decimal places (the upstream tables round to two):",
        "",
        "| Profile | Requests/s | Output tokens/s | Total tokens/s |",
        "|---|---:|---:|---:|",
    ]
    for profile, row in rows:
        values = [row.get(key) for key in ("request_throughput", "output_throughput", "total_token_throughput")]
        formatted = [f"{value:.6f}" if isinstance(value, (int, float)) else "—" for value in values]
        lines.append(f"| {escape(profile)} | " + " | ".join(formatted) + " |")
    return lines + [""]


def render(output, *, invoked_utc, setup_path, template_path, local_commits):
    output = Path(output).resolve()
    bench = output.parent
    identity_path = bench / "identity.json"
    identity, config, summary = read(identity_path), read(output / "run_config.json"), read(output / "summary.json")
    if summary.get("status") in (None, "running"):
        raise ValueError("Finalize only after benchmark_stage writes its completed/failed summary")
    setup, template = read(setup_path), read(template_path)
    begun = timestamp(setup["stage_setup_started_utc"])
    invoked = timestamp(invoked_utc)
    elapsed = (invoked - begun).total_seconds()
    if elapsed < 0:
        raise ValueError("Recorded invocation precedes setup")
    if not template.get("identical_rendered_text_to_client_smoke") or not template.get("chat_template_sha256"):
        raise ValueError("Missing matching native-server template proof")
    lines = [
        START,
        "",
        "## Provenance and measurement qualifications",
        "",
        f"The observed server imported `{identity['generator_module']}` with "
        f"{identity['layer_count']}/{identity['configured_layer_count']} layers, precision "
        f"`{identity['precision']}`, and context capacity {identity['max_model_len']}. "
        f"Hardware: {escape(identity['hardware'])}. "
        f"Model revision `{identity['model_revision']}`; tokenizer revision `{identity['tokenizer_revision']}`. "
        f"See {link(identity_path, output, 'identity')} and "
        f"{link(identity['implementation_evidence'], output, 'constructed module/precision evidence')}. "
        "Per-profile records above retain the observed launch commands, server slots and prefix-caching settings.",
        "",
        "Runtime/source commits: "
        + "; ".join(f"{escape(name)} `{sha}`" for name, sha in identity["source_commits"].items())
        + ".",
        "",
        f"Setup before invocation spans {begun.isoformat()} through {invoked.isoformat()} "
        f"({elapsed:.1f} s elapsed). This separate wall interval includes installation, downloads, "
        "instrumentation repairs, initial launches, warmup and pilots; it is not active compute time "
        "and is not added to the runner's client/setup clocks printed above. "
        f"See {link(setup_path, output, 'setup timestamps')} and {link(bench / 'RUN_NOTES.md', output, 'run notes')}.",
        "",
    ]
    lines += precise_throughput(summary)
    launches = []
    for path in sorted((bench / "setup").glob("b*/launch.json")):
        ready_path = path.parent / "ready.json"
        if not ready_path.exists():
            continue
        launch, ready = read(path), read(ready_path)
        begin, finish = launch.get("started_unix"), ready.get("ready_unix")
        if begin is not None and finish is not None and finish <= invoked.timestamp():
            if finish < begin:
                raise ValueError(f"Invalid server readiness timing: {path}")
            launches.append((path, ready_path, launch.get("max_num_seqs", "unrecorded"), finish - begin))
    if launches:
        lines += [
            "Initial launch-to-readiness intervals completed before invocation:",
            "",
            "| Launch | Server slots | Seconds |",
            "|---|---:|---:|",
        ]
        for launch, ready, slots, seconds in launches:
            lines.append(
                f"| {link(launch, output, launch.parent.name)} / {link(ready, output, 'readiness')} | {slots} | {seconds:.1f} |"
            )
        lines += [
            "",
            "These readiness intervals include any ready servers later replaced; they are setup evidence, not serving measurements.",
            "",
        ]
    protocol = []
    for task in config["tasks"]:
        path = output / task / "results.json"
        if not path.exists():
            continue
        result = read(path)
        meta = result.get("benchmark_stage", {})
        if meta.get("template") != "native server chat template; structured messages exactly once":
            raise ValueError(f"Missing native-once protocol evidence: {task}")
        if meta.get("fewshot_as_multiturn") is not True:
            raise ValueError(f"Missing structured fewshot evidence: {task}")
        protocol.append(f"{link(path, output, task)} records {escape(result.get('n-shot', {}))} few-shot settings")
    lines += [
        f"Template evidence: {link(template_path, output, 'server-interpreter smoke')} records "
        f"`{template['tokenizer_class']}` and native template SHA-256 `{template['chat_template_sha256']}` "
        "with identical rendering to the client smoke. The evaluator sends structured messages with "
        "its tokenizer backend disabled; the server applies the native HF template once. "
        "This setup smoke establishes rendering compatibility; per-task result metadata records the actual run's template contract. "
        + (
            "; ".join(protocol) + "."
            if protocol
            else "No scored-task protocol metadata was available in this failed run."
        ),
        "",
        "Generation dictionaries (including thinking, sampling, token caps and textual stops) are frozen in "
        f"{link(output / 'run_config.json', output)}. "
        "Final content is scored separately from retained reasoning; no raw reasoning is substituted for an empty final answer. "
        "The task variants, document hashes, populations and upstream scorer settings remain in "
        f"{link(output / 'manifest.json', output)} and task artifacts. Answer extraction and truncation facts are reported above.",
        "",
    ]
    protocols = {}
    for task, generation in config.get("generation", {}).items():
        # One compact row per distinct recipe, avoiding repeated identical
        # shared-pool dictionaries. All omitted overrides stay in run_config.
        recipe = {
            key: generation[key]
            for key in (
                "chat_template_kwargs",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
                "do_sample",
                "until",
            )
            if key in generation
        }
        protocols.setdefault(json.dumps(recipe, sort_keys=True), []).append(task)
    if protocols:
        lines += ["Recorded thinking/sampling settings:", ""]
        for recipe, tasks in protocols.items():
            lines.append(f"- {', '.join(tasks)}: `{recipe}`.")
        lines += [""]
    rows = []
    for batch in summary.get("measured_batch_sizes", []):
        path = output / f"phase-accounting-b{batch}.json"
        if not path.exists():
            if summary["status"] == "completed":
                raise ValueError(f"Missing completed-profile phase accounting: {path}")
            continue
        phase = read(path)
        counts = Counter(len(event["metadata"]["requests"]) for event in phase["events"] if event["phase"] == "decode")
        if not counts:
            raise ValueError(f"Missing active decode-batch observations: {path}")
        distribution = "; ".join(f"{size} active: {count} steps" for size, count in sorted(counts.items()))
        rows.append((batch, path, phase, distribution))
    if rows:
        lines += [
            "Measured decode batches are derived from server events, separately from HTTP concurrency and configured slots:",
            "",
            "| Profile | Server slots | Observed active decode batches | Maximum polling gap s | Maximum completion bracket s |",
            "|---|---:|---|---:|---:|",
        ]
        for batch, path, phase, distribution in rows:
            lines.append(
                f"| {link(path, output, 'B' + str(batch))} | {phase['runtime']['slots']} | {distribution} | "
                f"{phase['max_observer_poll_gap_seconds']:.6f} | {phase['max_completion_uncertainty_seconds']:.6f} |"
            )
        lines += [
            "",
            "| Profile | Phase | Nominal FIFO attribution s | Completion-bracket lower s | Completion-bracket upper s |",
            "|---|---|---:|---:|---:|",
        ]
        for batch, _, phase, _ in rows:
            for name in ("prefill", "decode"):
                bounds = phase["phase_duration_bounds_seconds"][name]
                lines.append(
                    f"| B{batch} | {name} | {phase['seconds'][name]:.6f} | {bounds['lower']:.6f} | {bounds['upper']:.6f} |"
                )
        lines += [
            "",
            "The headline roofline percentages use nominal FIFO attribution from host-observed completion events. "
            "They are not precise phase occupancy measurements. Delayed observer polling widens completion brackets; "
            "conservative summed upper brackets can exceed the client wall interval. The retained evidence includes "
            "FIFO validation, event timestamps, uncertainty and client-request mapping. No device synchronization or live device profiler was added.",
            "",
            "Decode bytes model stored precision, padded weight scans, active-expert sharing, selected KV and explicit materialized traffic. "
            "They are conservative payload/materialization estimates, not measured DRAM transactions. Expert-union sensitivity "
            "bounds apply only to that category; unmodeled staging, scratch, cache effects and repeated reads remain explicit omissions. "
            "Prefill uses useful executed-row FLOPs, including replayed chunks, and a fidelity-weighted nominal four-chip peak; "
            "its matrix-equivalent normalization does not establish an achievable SFPU ceiling.",
            "",
        ]
    if local_commits:
        lines += [
            "Stage-local helper/evidence commits (separate from the launched runtime identity):",
            "",
            "| Commit | Purpose |",
            "|---|---|",
        ]
        for sha, subject in local_commits:
            lines.append(f"| `{sha}` | {escape(subject)} |")
        lines += [""]
    lines += [
        "This appendix adds provenance and measurement qualifications; upstream scores, denominators, raw responses and summary values are unchanged.",
        "",
        END,
    ]
    return "\n".join(lines)


def append(report, appendix):
    original = report.read_text()
    if START in original:
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError("Malformed existing provenance markers")
        before, rest = original.split(START, 1)
        _, after = rest.split(END, 1)
        if after.strip():
            raise ValueError("Refusing to replace an appendix followed by unowned content")
        original = before.rstrip() + "\n"
    elif END in original:
        raise ValueError("Unmatched appendix marker")
    report.write_text(original.rstrip() + "\n\n" + appendix + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=MODEL_DIR / "doc/benchmark/run")
    parser.add_argument(
        "--invoked-utc", required=True, help="UTC timestamp recorded immediately before benchmark_stage run"
    )
    parser.add_argument(
        "--setup-evidence", type=Path, default=MODEL_DIR / "doc/benchmark/setup/pre-invocation-timing.json"
    )
    parser.add_argument(
        "--template-evidence", type=Path, default=MODEL_DIR / "doc/benchmark/setup/template-smoke-server.json"
    )
    parser.add_argument("--commit", action="append", default=[], help="Repeat for verified local stage commits")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary_path = run / "summary.json"
    before = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    appendix = render(
        run,
        invoked_utc=args.invoked_utc,
        setup_path=args.setup_evidence,
        template_path=args.template_evidence,
        local_commits=[commit_identity(x) for x in args.commit],
    )
    append(run / "REPORT.md", appendix)
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != before:
        raise RuntimeError("Summary changed during finalization; investigate concurrent mutation")
    print(f"Appended provenance; summary unchanged: {run / 'REPORT.md'}")


if __name__ == "__main__":
    main()
