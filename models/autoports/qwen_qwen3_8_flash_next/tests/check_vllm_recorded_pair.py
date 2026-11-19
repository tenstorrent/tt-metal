# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record one staggered513/129 (or reversed) pair against the diagnostic server."""

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--reverse-prompts", action="store_true")
    args = parser.parse_args()
    left_length, right_length = (129, 513) if args.reverse_prompts else (513, 129)
    workload = f"staggered{left_length}/{right_length} input,96 output each, greedy seed42"
    output = args.recording / "client.json"
    marker = args.recording / "client.attempt.json"
    if output.exists() or marker.exists():
        parser.error("Preserve the previous diagnostic pair; use a fresh server/recording directory")
    if not (args.recording / "setup.pkl").exists():
        parser.error("The diagnostic server has not recorded cache initialization")
    with marker.open("x") as handle:
        handle.write(json.dumps(dict(started=time.time(), workload=workload)) + "\n")
    server = args.server_url.rstrip("/")
    report = dict(
        passed=False,
        requests=[],
        server_url=server,
        workload=workload,
        scope="raw-token correctness diagnostic, not qualitative/performance evidence",
    )
    first = threading.Event()

    def request(name, length, pattern):
        prompt = (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
        body = dict(
            model=model,
            request_id="history-" + name,
            prompt=prompt,
            temperature=0,
            top_k=1,
            top_p=1.0,
            seed=42,
            max_tokens=96,
            ignore_eos=True,
            return_token_ids=True,
            stream=True,
            stream_options={"include_usage": True},
        )
        row = dict(name=name, body=body, token_ids=[], chunks=[], response_id=None, usage=None)
        # Keep partial streaming evidence even when a request fails.
        report["requests"].append(row)
        with requests.post(server + "/v1/completions", json=body, stream=True, timeout=(30, 1200)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(item["error"])
                row["chunks"].append(item)
                row["response_id"] = item.get("id", row["response_id"])
                if item.get("usage"):
                    row["usage"] = item["usage"]
                for choice in item.get("choices", []):
                    ids = choice.get("token_ids") or []
                    row["token_ids"].extend(ids)
                    if ids and name == "left":
                        first.set()
        assert len(row["token_ids"]) == 96
        assert row["usage"]["prompt_tokens"] == length and row["usage"]["completion_tokens"] == 96
        assert row["response_id"]
        return row

    try:
        metadata = requests.get(server + "/v1/models", timeout=30)
        metadata.raise_for_status()
        model = metadata.json()["data"][0]["id"]
        report["model"] = model
        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(request, "left", left_length, [785, 13458, 13])
            if not first.wait(600):
                raise RuntimeError("Left request did not produce its first token")
            report["left_pending_at_right_admission"] = not left.done()
            assert report["left_pending_at_right_admission"], "No live request overlap"
            right = pool.submit(request, "right", right_length, [785, 3432, 13])
            for future in (left, right):
                future.result()
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, output=str(output))))


if __name__ == "__main__":
    main()
