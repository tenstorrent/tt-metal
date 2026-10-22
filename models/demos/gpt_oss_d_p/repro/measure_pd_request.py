#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure one isolated streaming request without opening or restarting TT devices.

Run on the prefill host so the client, prefill intervals and source-manager log
use one host clock. The output directory must be new; failed evidence is kept.
"""
import argparse
import fcntl
import json
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from perf_analysis import counter_delta, expected_migration_positions, interval_metrics, migration_intervals


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def lines_from_stream(response):
    """Process HTTP chunks immediately; do not wait for requests' 512-byte buffer."""
    pending = b""
    for chunk in response.iter_content(chunk_size=None):
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.rstrip(b"\r")
    if pending:
        yield pending.rstrip(b"\r")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prefill-ip", required=True)
    parser.add_argument("--decode-ip", required=True)
    parser.add_argument("--prefill-host", required=True)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != args.prefill_host:
        raise RuntimeError("client must run on prefill host for comparable epoch clocks")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.run / "measurement.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prompt = json.loads(args.prompt.read_text())
    if prompt["input_tokens"] < args.chunk:
        raise ValueError("chunk exceeds requested input length")
    write_json(args.output / "prompt.json", prompt)
    started = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        status="IN PROGRESS",
        warmup=args.warmup,
        input_tokens=prompt["input_tokens"],
        chunk=args.chunk,
        host=socket.gethostname(),
        layers=args.layers,
    )
    write_json(args.output / "status.json", started)
    endpoints = dict(
        prefill=f"http://{args.prefill_ip}:8089/metrics",
        decode=f"http://{args.decode_ip}:18089/metrics",
        manager_prefill=f"http://{args.prefill_ip}:19100/metrics",
        manager_decode=f"http://{args.decode_ip}:19100/metrics",
    )
    logs = {
        name: args.run / "services" / f"{name}.log"
        for name in ["manager-prefill", "manager-decode", "worker-prefill", "worker-decode"]
    }
    logs["prefill-chunks"] = args.run / "prefill/chunk-times.jsonl"
    session = requests.Session()
    session.trust_env = False

    def snapshot(label):
        result = {}
        for name, url in endpoints.items():
            response = session.get(url, timeout=10)
            response.raise_for_status()
            result[name] = response.text
            (args.output / f"{label}-{name}.prom").write_text(response.text)
        return result

    before = snapshot("before")
    offsets = {name: path.stat().st_size if path.exists() else 0 for name, path in logs.items()}
    write_json(args.output / "log-offsets-before.json", offsets)
    events, usage, first, last, finish, error = [], None, None, None, None, None
    content, reasoning = [], []
    epoch, start = time.time(), time.perf_counter()
    done = False
    try:
        with session.post(
            f"http://{args.prefill_ip}:8000/v1/chat/completions",
            json=prompt["payload"],
            stream=True,
            timeout=(10, args.timeout),
        ) as response:
            write_json(args.output / "http.json", dict(status=response.status_code, headers=dict(response.headers)))
            response.raise_for_status()
            with (args.output / "events.jsonl").open("w") as output:
                for line in lines_from_stream(response):
                    now = time.perf_counter()
                    if now - start > args.timeout:
                        raise TimeoutError("request duration exceeded bound")
                    if not line.startswith(b"data:"):
                        continue
                    body = line[5:].strip()
                    if body == b"[DONE]":
                        done = True
                        break
                    event = json.loads(body)
                    output.write(json.dumps(dict(elapsed_s=now - start, event=event)) + "\n")
                    output.flush()
                    events.append(event)
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if text or thought:
                        first = first if first is not None else now
                        last = now
                        content.append(text)
                        reasoning.append(thought)
        elapsed = time.perf_counter() - start
    except Exception as exc:
        elapsed = time.perf_counter() - start
        error = repr(exc)
    # Allow final counter publication after the stream; this is outside timings.
    time.sleep(1)
    after = snapshot("after")
    windows = {}
    for name, path in logs.items():
        if not path.exists() or path.stat().st_size < offsets[name]:
            raise RuntimeError("log missing/truncated during request: " + str(path))
        with path.open("rb") as log:
            log.seek(offsets[name])
            windows[name] = log.read().decode(errors="replace")
        (args.output / f"{name}.window.log").write_text(windows[name])
    result = dict(
        **started,
        request_start_epoch=epoch,
        total_s=elapsed,
        ttft_s=None if first is None else first - start,
        stream_interval_s=None if first is None else last - first,
        finish_reason=finish,
        usage=usage,
        done_marker=done,
        error=error,
        content="".join(content),
        reasoning="".join(reasoning),
        sse_events=len(events),
        definitions=dict(
            ttft="request submission to first nonempty content/reasoning delta",
            migration="source manager command to successful completion; control-plane inclusive",
            prefill="host compute-call boundary, existing async layer drain; no extra sync",
            tsu="native inter-token interval count divided by native interval seconds",
        ),
    )
    write_json(args.output / "response.json", result)
    try:
        if error or not done or first is None or not usage or finish not in ("stop", "length"):
            raise ValueError("incomplete stream, missing usage, or request failure")
        if usage["prompt_tokens"] != prompt["input_tokens"]:
            raise ValueError("server/local tokenizer input count mismatch")
        if not 0 < usage["completion_tokens"] <= prompt["payload"]["max_tokens"]:
            raise ValueError("completion count outside requested cap")
        chunks = [json.loads(line) for line in windows["prefill-chunks"].splitlines() if line.strip()]
        if not chunks or any(x["rank"] != 0 for x in chunks):
            raise ValueError("expected one-rank prefill interval evidence")
        if any(x["start_epoch"] < epoch - 0.1 or x["end_epoch"] > epoch + elapsed + 1 for x in chunks):
            raise ValueError("prefill intervals outside isolated request window")
        for name, body in windows.items():
            if name == "prefill-chunks":
                continue
            faults = [
                line
                for line in body.splitlines()
                if re.search(r"successful=false|queue has no chunks|stays stale|condemn|reclaim", line, re.I)
            ]
            if faults:
                raise ValueError(f"{name} faults: " + "\n".join(faults[:5]))
        positions = [(x["from_token"], x["to_token"]) for x in chunks]
        if len(positions) != len(set(positions)):
            raise ValueError("duplicate prefill chunk coverage")
        block_size = json.loads((args.run / "worker-decode.json").read_text())["runtime"]["kv_block_size"]
        expected_positions = expected_migration_positions(positions, prompt["input_tokens"], block_size)
        migrations = migration_intervals(windows["manager-prefill"], args.layers, expected_positions)
        write_json(args.output / "migrations.json", migrations)
        metrics = interval_metrics(
            [(x["start_epoch"], x["end_epoch"]) for x in chunks],
            [(x["start_epoch"], x["end_epoch"]) for x in migrations],
        )
        metrics["migrations"] = len(migrations)
        metrics["migration_positions"] = expected_positions
        metrics["migration_command_log_count"] = sum(x["command_log_count"] for x in migrations)
        metrics["migration_deferred_log_count"] = windows["manager-prefill"].count("deferred reason=")
        metrics["kv_payload_bytes"] = counter_delta(
            before["manager_prefill"], after["manager_prefill"], "kv_manager_bytes_moved_total", {"leg": "wire"}
        )
        if metrics["kv_payload_bytes"] <= 0:
            raise ValueError("no measured wire payload bytes")
        missing = {}
        for key, source, name in [
            ("decode_interval_s", "decode", "decode_time_between_tokens_seconds_total"),
            ("decode_intervals", "decode", "decode_time_between_tokens_intervals_total"),
            ("decode_generated", "decode", "generated_tokens_total"),
            ("prefill_tokens_processed", "prefill", "prompt_tokens_prefilled_total"),
            ("prefill_reused", "prefill", "reused_tokens_total"),
            ("decode_reused", "decode", "reused_tokens_total"),
            ("decode_prompt_processed", "decode", "prompt_tokens_prefilled_total"),
            ("decode_migrated", "decode", "migrated_tokens_total"),
            ("prefill_engine_busy_s", "prefill", "prefill_device_busy_seconds_total"),
        ]:
            try:
                metrics[key] = counter_delta(before[source], after[source], "tt_engine_" + name)
            except ValueError as exc:
                metrics[key] = None
                missing[key] = str(exc)
        metrics["missing_metrics"] = missing
        metrics["tsu"] = (
            metrics["decode_intervals"] / metrics["decode_interval_s"]
            if metrics["decode_intervals"] and metrics["decode_interval_s"]
            else None
        )
        metrics["prefill_actual_tokens"] = sum(b - a for a, b in positions)
        metrics["prefill_tps"] = metrics["prefill_actual_tokens"] / metrics["prefill_span_s"]
        metrics["prefill_processed_tps"] = (
            metrics["prefill_tokens_processed"] / metrics["prefill_span_s"]
            if metrics["prefill_tokens_processed"]
            else None
        )
        metrics["prefill_positions"] = positions
        result.update(metrics, status="DONE")
        # Preserve the scope: stream/migration/performance validation, no numerical claim.
        result["validation"] = "complete stream, exact input count, full layer/range migration; no PCC or golden"
    except Exception as exc:
        result.update(status="FAILED", analysis_error=repr(exc))
    write_json(args.output / "result.json", result)
    write_json(
        args.output / "status.json", {k: result[k] for k in ["status", "utc", "warmup", "input_tokens", "chunk"]}
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("content", "reasoning")}), flush=True)
    if result["status"] != "DONE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
