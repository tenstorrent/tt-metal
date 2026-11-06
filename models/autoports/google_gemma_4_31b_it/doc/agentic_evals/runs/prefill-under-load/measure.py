"""HTTP-only diagnostic; coordinator provides an otherwise idle warmed B8 server."""

import argparse
import json
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", default="http://localhost:8000")
    args = p.parse_args()
    revision = "842da3794eaa0b77d5f08bae87a17459d91ff475"
    snapshot = "/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/" + revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot, revision=revision, local_files_only=True)
    clock = time.monotonic()
    nonce = uuid.uuid4().hex
    report = dict(
        status="running",
        planned_clients=2,
        batch=8,
        revision=revision,
        timing="HTTP wall/arrival events, not device time",
        arms=[],
        errors=[],
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def prompt(n, label):
        prefix = tokenizer.encode(nonce + label, add_special_tokens=False)
        filler = tokenizer.encode("Explain experimental measurements carefully. ", add_special_tokens=False)
        return (prefix + filler * (n // len(filler) + 1))[:n]

    def request(label, length, output, ready=None):
        row = dict(
            label=label,
            prompt_tokens=length,
            requested_output_tokens=output,
            start=time.monotonic() - clock,
            events=[],
            status="running",
        )
        body = dict(
            model="google/gemma-4-31B-it",
            prompt=prompt(length, label),
            max_tokens=output,
            ignore_eos=True,
            temperature=0,
            top_k=1,
            top_p=1,
            stream=True,
            return_token_ids=True,
        )
        try:
            req = urllib.request.Request(
                args.url + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=240) as response:
                for raw in response:
                    if time.monotonic() - clock - row["start"] > 240:
                        raise TimeoutError("Total request deadline")
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
                            assert all(0 <= x < len(tokenizer) for x in tokens)
                            row["events"].append([time.monotonic() - clock, len(tokens)])
                            if ready:
                                ready.set()
            row["output_tokens"] = sum(n for _, n in row["events"])
            assert row["output_tokens"] == output
            row["status"] = "pass"
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["end"] = time.monotonic() - clock
        return row

    def window(events, start, end):
        gaps = [(b[0] - a[0]) * 1000 for a, b in zip(events, events[1:]) if start < b[0] <= end]
        ordered = sorted(gaps)

        def pct(q):
            return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))] if ordered else None

        count = sum(n for t, n in events if start < t <= end)
        return dict(
            start=start,
            end=end,
            seconds=end - start,
            delivered_tokens=count,
            tokens_per_second=count / (end - start),
            gap_count=len(gaps),
            gap_p50_ms=pct(0.5),
            gap_p95_ms=pct(0.95),
            gap_max_ms=max(gaps) if gaps else None,
            boundary_straddling_gaps_ms=[
                (b[0] - a[0]) * 1000 for a, b in zip(events, events[1:]) if a[0] <= start < b[0] <= end
            ],
        )

    save()
    for length in [4096, 32768, 86016]:
        ready = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            anchor = pool.submit(request, f"anchor-{length}", 128, 2048, ready)
            if not ready.wait(60):
                a = anchor.result()
                report["errors"].append("Anchor failed to start")
                report["arms"].append({"anchor": a})
                break
            baseline_start = time.monotonic() - clock
            time.sleep(2)
            long = request(f"long-{length}", length, 1)
            a = anchor.result()
        arm = dict(prefill_prompt_tokens=length, anchor=a, prefill=long)
        if a["status"] == "pass" and long["status"] == "pass":
            end = long["events"][0][0]
            arm.update(
                baseline=window(a["events"], baseline_start, long["start"]),
                interference=window(a["events"], long["start"], end),
                anchor_live_through_prefill=a["end"] >= end,
            )
            if not arm["anchor_live_through_prefill"]:
                report["errors"].append(f"{length}: anchor ended before prefill")
        else:
            report["errors"].append(f"{length}: request failed")
        report["arms"].append(arm)
        save()
        if report["errors"]:
            break
    report["elapsed_seconds"] = time.monotonic() - clock
    report["status"] = "pass" if len(report["arms"]) == 3 and not report["errors"] else "incomplete"
    save()
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
