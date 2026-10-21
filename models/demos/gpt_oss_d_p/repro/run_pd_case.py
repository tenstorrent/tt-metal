#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run one warmup and three measured requests against already-audited owners."""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--prefill-host", required=True)
    parser.add_argument("--prefill-ip", required=True)
    parser.add_argument("--decode-ip", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "worker-prefill.json").read_text())["runtime"]
    if config["chunk_size"] != args.chunk or config["layers_per_chunk"] != 36 or args.isl < args.chunk:
        raise ValueError("case does not match active full-model prefill config")
    if not (args.run / "decode/external.ready.json").is_file():
        raise RuntimeError("decoder ready manifest missing")
    args.output.mkdir(parents=True, exist_ok=False)
    progress = dict(
        status="IN PROGRESS",
        input_tokens=args.isl,
        chunk=args.chunk,
        utc=datetime.now(timezone.utc).isoformat(),
        runs=[],
        definitions="One warmup plus three measured requests; no golden/PCC",
    )

    def save():
        temporary = args.output / "case.json.tmp"
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(args.output / "case.json")

    save()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location("prompt_builder", here / "make-perf-prompt.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        for index in range(4):
            timeout = 900 if index == 0 else 300
            if time.time() + timeout + 20 >= args.deadline:
                raise TimeoutError("insufficient time before serving deadline")
            prompt = builder.create_prompt(tokenizer, args.isl, args.model)
            prompt_path = args.output / f"prompt-{index}.json"
            prompt_path.write_text(json.dumps(prompt) + "\n")
            out = args.output / f"run-{index}"
            progress.update(current_run=index, current_run_is_warmup=index == 0)
            save()
            command = [
                sys.executable,
                str(here / "measure_pd_request.py"),
                "--prompt",
                str(prompt_path),
                "--output",
                str(out),
                "--run",
                str(args.run),
                "--chunk",
                str(args.chunk),
                "--prefill-host",
                args.prefill_host,
                "--prefill-ip",
                args.prefill_ip,
                "--decode-ip",
                args.decode_ip,
                "--timeout",
                str(timeout),
            ]
            if index == 0:
                command.append("--warmup")
            (args.output / f"command-{index}.json").write_text(json.dumps(command) + "\n")
            with (args.output / f"run-{index}.log").open("xb") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout + 60)
            if (out / "result.json").exists():
                result = json.loads((out / "result.json").read_text())
                result.update(index=index, measured=index > 0, exit_code=completed.returncode, evidence=str(out))
                progress["runs"].append({k: v for k, v in result.items() if k not in ("content", "reasoning")})
                save()
            if completed.returncode or not (out / "result.json").exists() or result["status"] != "DONE":
                raise RuntimeError(f"run {index} failed; preserved {out}")
        measured = [x for x in progress["runs"] if x["measured"]]
        means, variances = {}, {}
        for key in [
            "ttft_s",
            "prefill_span_s",
            "prefill_active_s",
            "prefill_tps",
            "prefill_processed_tps",
            "kv_payload_bytes",
            "transfer_sum_s",
            "transfer_active_s",
            "transfer_span_s",
            "overlap_s",
            "nonoverlap_prefill_s",
            "nonoverlap_transfer_s",
            "initial_prefill_s",
            "final_transfer_tail_s",
            "decode_interval_s",
            "tsu",
            "total_s",
        ]:
            values = [x[key] for x in measured if x.get(key) is not None]
            means[key] = statistics.mean(values) if len(values) == 3 else None
            variances[key] = statistics.variance(values) if len(values) == 3 else None
        progress.update(
            status="DONE",
            mean=means,
            sample_variance=variances,
            warmup_s=progress["runs"][0]["total_s"],
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        save()
    except Exception as error:
        progress.update(status="FAILED", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
