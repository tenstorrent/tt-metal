# SPDX-License-Identifier: Apache-2.0
"""HTTP-only agentic load costs; no device access or profiler collection.

Run against an otherwise idle, already warmed server. Engine CPU is process CPU
(user + system, all its threads), not elapsed host overhead or device time.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check_served_state import log_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--revision", default="842da3794eaa0b77d5f08bae87a17459d91ff475")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--engine-pid", type=int, required=True)
    parser.add_argument("--stages", default="prefill,overlap,contention")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--overlap-decode-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument(
        "--workload", type=Path, default=Path(__file__).resolve().parents[1] / "doc/agentic_serving.json"
    )
    args = parser.parse_args()
    stages = args.stages.split(",")
    if set(stages) - {"prefill", "overlap", "contention"}:
        parser.error("Unknown stage")
    if args.batch < 2 or args.decode_tokens < 2 or args.request_timeout <= 0:
        parser.error("Require batch >= 2, decode tokens >= 2 and positive timeout")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    vocabulary_size = len(tokenizer)
    workload = json.loads(args.workload.read_text())["workload"]
    run_nonce = uuid.uuid4().hex
    started = time.monotonic()
    records = []
    lock = threading.Lock()
    report = dict(
        status="running",
        workload=workload,
        stages=stages,
        requests=records,
        engine_pid=args.engine_pid,
        server_log=str(args.server_log),
        prefill_by_length=[],
        host_contention={},
        missing_reasons=[],
        limitations=[
            "Synthetic exact-token raw continuation prompts measure load, not chat quality.",
            "Streaming arrival counts measure HTTP delivery; not device timing.",
            "Engine CPU/step is process CPU from /proc, including all engine threads; not exclusive Python overhead.",
            "Decode window starts after all first tokens and ends at first completion; boundary events may straddle snapshots.",
            "Requires no other server clients; stress workers inherit CPU affinity and are not individually pinned.",
        ],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with lock:
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)

    def reset():
        request = urllib.request.Request(args.url + "/reset_prefix_cache", data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError("Prefix cache reset failed")

    def prompt(length, label):
        prefix = tokenizer.encode(
            "Unique request "
            + label.replace("idle_", "contention_").replace("busy_", "contention_")
            + " "
            + run_nonce
            + ". ",
            add_special_tokens=False,
        )
        filler = tokenizer.encode("Explain how a computer stores and processes information. ", add_special_tokens=False)
        return (prefix + filler * (length // len(filler) + 1))[:length]

    def request(label, length, count, first=None, done=None):
        ids = prompt(length, label)
        began = time.monotonic()
        row = dict(
            label=label,
            input_tokens=len(ids),
            requested_output_tokens=count,
            prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            start_seconds=began - started,
            token_events=[],
            output_tokens=0,
            status="running",
        )
        with lock:
            records.append(row)
        body = dict(
            model=args.model,
            prompt=ids,
            max_tokens=count,
            temperature=0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            stream=True,
            return_token_ids=True,
        )
        try:
            req = urllib.request.Request(
                args.url + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=min(60, args.request_timeout)) as response:
                for raw in response:
                    if time.monotonic() - began > args.request_timeout:
                        raise TimeoutError("Request total deadline exceeded")
                    line = raw.decode().strip()
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    for choice in chunk.get("choices", []):
                        tokens = choice.get("token_ids") or []
                        if tokens:
                            if any(token < 0 or token >= vocabulary_size for token in tokens):
                                raise ValueError("Out-of-vocabulary token")
                            now = time.monotonic()
                            with lock:
                                row["token_events"].append([now - started, len(tokens)])
                                row["output_tokens"] += len(tokens)
                                if "ttft_ms" not in row:
                                    row["ttft_ms"] = (now - began) * 1000
                                    row["first_token_seconds"] = now - started
                            if first:
                                first.set()
            if row["output_tokens"] != count:
                raise ValueError(f"Expected {count} output tokens, got {row['output_tokens']}")
            row["status"] = "pass"
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            row["end_seconds"] = time.monotonic() - started
            if done:
                done.set()
            save()
        return row

    def snapshot():
        # comm may contain spaces; fields after its closing parenthesis start at field 3.
        stat = Path(f"/proc/{args.engine_pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(
            time=time.monotonic() - started,
            cpu_seconds=(int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
            process_start_ticks=stat[19],
            log_offset=args.server_log.stat().st_size,
        )

    def window_metrics(before, after, rows):
        with args.server_log.open("rb") as stream:
            stream.seek(before["log_offset"])
            data = stream.read(after["log_offset"] - before["log_offset"])
        events = log_events(data)
        decode = [e for e in events if e.get("event") == "sampling_mode" and e.get("phase") == "decode"]
        prefill = [e for e in events if e.get("event") == "sampling_mode" and e.get("phase") == "prefill"]
        elapsed = after["time"] - before["time"]
        token_count = sum(n for r in rows for t, n in r["token_events"] if before["time"] < t <= after["time"])
        valid = bool(decode) and not prefill and before["process_start_ticks"] == after["process_start_ticks"]
        return dict(
            start_seconds=before["time"],
            end_seconds=after["time"],
            seconds=elapsed,
            delivered_tokens=token_count,
            output_tokens_per_second=token_count / elapsed,
            logged_decode_steps=len(decode),
            logged_prefill_steps=len(prefill),
            active_rows=sorted(set(e.get("active_rows", 0) for e in decode)),
            engine_cpu_seconds=after["cpu_seconds"] - before["cpu_seconds"],
            engine_host_cpu_ms_per_decode_step=(after["cpu_seconds"] - before["cpu_seconds"]) * 1000 / len(decode)
            if valid
            else None,
            log_byte_range=[before["log_offset"], after["log_offset"]],
            valid_decode_only_cpu_window=valid,
        )

    def decode_wave(label):
        reset()
        first = [threading.Event() for _ in range(args.batch)]
        done = threading.Event()
        with ThreadPoolExecutor(max_workers=args.batch) as pool:
            futures = [
                pool.submit(request, f"{label}_{i}", 128, args.decode_tokens, first[i], done) for i in range(args.batch)
            ]
            deadline = time.monotonic() + args.request_timeout
            for event in first:
                if not event.wait(max(0, deadline - time.monotonic())):
                    raise TimeoutError("Not all rows began decoding")
            if done.is_set():
                raise RuntimeError("A request finished before all rows began decoding; increase decode length")
            before = snapshot()
            if not done.wait(args.request_timeout):
                raise TimeoutError("Decode wave did not finish")
            after = snapshot()
            rows = [future.result() for future in futures]
        metrics = window_metrics(before, after, rows)
        metrics.update(
            batch=args.batch,
            input_tokens=128,
            output_tokens=args.decode_tokens,
            completed_requests=sum(r["status"] == "pass" for r in rows),
        )
        if args.batch not in metrics["active_rows"]:
            report["missing_reasons"].append(label + ": no logged full-batch decode step in window")
        if not metrics["valid_decode_only_cpu_window"]:
            report["missing_reasons"].append(label + ": no clean decode-only CPU/log window")
        return metrics

    workers = []

    def terminate(signum, frame):
        raise InterruptedError(f"Received signal {signum}; preserve evidence and clean owned stress workers")

    signal.signal(signal.SIGTERM, terminate)
    save()
    try:
        if "prefill" in stages:
            for length in dict.fromkeys([4096, workload["typical_context_tokens"], workload["working_set_tokens"]]):
                reset()
                row = request(f"prefill_{length}", length, 1)
                report["prefill_by_length"].append(
                    dict(
                        input_tokens=length,
                        ttft_ms=row.get("ttft_ms"),
                        input_tokens_per_second=length * 1000 / row["ttft_ms"] if row.get("ttft_ms") else None,
                        status=row["status"],
                        prefix_cache_bypass="reset while idle plus unique initial prefix",
                    )
                )
                save()
        if "overlap" in stages:
            reset()
            first = [threading.Event() for _ in range(args.batch - 1)]
            with ThreadPoolExecutor(max_workers=args.batch) as pool:
                anchors = [
                    pool.submit(request, f"overlap_anchor_{i}", 128, args.overlap_decode_tokens, first[i])
                    for i in range(args.batch - 1)
                ]
                deadline = time.monotonic() + args.request_timeout
                for event in first:
                    if not event.wait(max(0, deadline - time.monotonic())):
                        raise TimeoutError("Overlap anchors did not start")
                long = request("overlap_long", workload["working_set_tokens"], 1)
                rows = [f.result() for f in anchors]
            if "first_token_seconds" not in long:
                raise RuntimeError("Long request never produced a token")
            begin, end = long["start_seconds"], long["first_token_seconds"]
            count = sum(n for r in rows for t, n in r["token_events"] if begin < t <= end)
            report["decode_during_working_set_prefill"] = dict(
                input_tokens=long["input_tokens"],
                concurrent_decode_requests=args.batch - 1,
                window_seconds=end - begin,
                other_delivered_tokens=count,
                other_output_tokens_per_second=count / (end - begin),
                all_anchors_live_through_window=all(r["end_seconds"] >= end for r in rows),
                timing_boundary="long request submit through first token, including scheduler wait",
            )
            if not report["decode_during_working_set_prefill"]["all_anchors_live_through_window"]:
                report["missing_reasons"].append(
                    "Overlap anchors ended before long prefill completed; increase overlap decode length"
                )
            save()
        if "contention" in stages:
            report["host_contention"]["idle"] = decode_wave("idle")
            save()
            cpu_count = len(os.sched_getaffinity(0))
            report["host_contention"].update(
                stress_processes=cpu_count,
                inherited_cpu_affinity=sorted(os.sched_getaffinity(0)),
                stress_mode="one unpinned busy process per available CPU",
            )
            for _ in range(cpu_count):
                workers.append(
                    subprocess.Popen(
                        [sys.executable, "-c", "while True: pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                )
            report["host_contention"]["busy"] = decode_wave("busy")
        report["status"] = (
            "pass" if all(r["status"] == "pass" for r in records) and not report["missing_reasons"] else "incomplete"
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        report["stress_workers_cleaned"] = all(w.poll() is not None for w in workers)
        save()
    print(json.dumps(dict(status=report["status"], output=str(args.output))))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
