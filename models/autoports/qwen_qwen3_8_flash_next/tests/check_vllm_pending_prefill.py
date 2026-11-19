# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare a relocated unfinished prefill with solo greedy serving controls."""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .check_vllm_mixed import events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    response = requests.get(args.server_url + "/v1/models", timeout=30)
    response.raise_for_status()
    model = response.json()["data"][0]["id"]
    report = dict(passed=False, workload="128 input / 32 output tokens; solo controls and staggered concurrency2")

    def request(variant, first=None):
        pattern = ([785, 13458, 13], [785, 3432, 13])[variant]
        body = dict(
            model=model,
            prompt=(pattern * 43)[:128],
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_tokens=32,
            ignore_eos=True,
            return_token_ids=True,
            stream=True,
            stream_options={"include_usage": True},
        )
        row = dict(variant=variant, token_ids=[], text="", usage=None)
        with requests.post(args.server_url + "/v1/completions", json=body, stream=True, timeout=1200) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if item.get("choices"):
                    choice = item["choices"][0]
                    row["token_ids"].extend(choice.get("token_ids") or [])
                    row["text"] += choice.get("text", "")
                    if first is not None and row["token_ids"]:
                        first.set()
                if item.get("usage"):
                    row["usage"] = item["usage"]
        assert len(row["token_ids"]) == 32 and row["usage"]["prompt_tokens"] == 128
        assert row["usage"]["completion_tokens"] == 32
        return row

    try:
        report["solo"] = [request(variant) for variant in range(2)]
        offset = len(events(args.server_log))
        first = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(request, 0, first)
            assert first.wait(600), "First request did not reach decode"
            assert not left.done(), "No concurrent lifecycle transition"
            right = pool.submit(request, 1)
            report["concurrent"] = [left.result(), right.result()]
        observed = events(args.server_log)[offset:]
        report["relocations"] = [event for event in observed if event["kind"] == "pending_prefill_relocation"]
        report["checks"] = dict(
            relocation_exercised=bool(report["relocations"]),
            token_exact=all(
                solo["token_ids"] == concurrent["token_ids"]
                for solo, concurrent in zip(report["solo"], report["concurrent"], strict=True)
            ),
            no_recapture=not any(event["kind"] == "capture" for event in observed),
            no_retirement=not any("release" in event["kind"] or "retir" in event["kind"] for event in observed),
            device_only=all(
                event["sampling_mode"] == "device"
                or (event["sampling_mode"] == "none" and not any(event.get("sample_mask", [True])))
                for event in observed
                if event["kind"] in ("prefill_submission", "decode_submission")
            ),
            no_oov=all(event["out_of_vocabulary"] == 0 for event in observed if event["kind"] == "sampled_tokens"),
        )
        report["passed"] = all(report["checks"].values())
        assert report["passed"], report["checks"]
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, output=str(args.output))))


if __name__ == "__main__":
    main()
