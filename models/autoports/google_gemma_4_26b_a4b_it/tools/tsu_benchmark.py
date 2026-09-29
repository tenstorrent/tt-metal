# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run comparable TTI-shaped streaming benchmarks against an existing server."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", nargs=4, type=int, action="append", metavar=("ISL", "OSL", "C", "N"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urlopen("http://127.0.0.1:8000/server_info", timeout=10) as response:
        server = json.load(response)
    (args.output / "server_info.json").write_text(json.dumps(server, indent=2) + "\n")
    rows = []
    for isl, osl, concurrency, requests in args.shape or [(4096, 128, 1, 4)]:
        for repeat in range(args.repeats):
            name = f"s{isl}-o{osl}-c{concurrency}-n{requests}-r{repeat}"
            result_path = args.output / f"{name}.json"
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "bench",
                "serve",
                "--backend",
                "openai-chat",
                "--endpoint",
                "/v1/chat/completions",
                "--model",
                "google/gemma-4-26B-A4B-it",
                "--tokenizer",
                "/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--dataset-name",
                "random",
                "--random-input-len",
                str(isl),
                "--random-output-len",
                str(osl),
                "--random-range-ratio",
                "0.0",
                "--max-concurrency",
                str(concurrency),
                "--num-prompts",
                str(requests),
                "--extra-body",
                json.dumps({"truncate_prompt_tokens": isl}),
                "--header",
                "Accept-Encoding=identity",
                "--temperature",
                "0",
                "--ignore-eos",
                "--seed",
                "0",
                "--num-warmups",
                str(args.warmups),
                "--ready-check-timeout-sec",
                "0",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--metric-percentiles",
                "50,99",
                "--save-result",
                "--save-detailed",
                "--result-filename",
                str(result_path),
                "--request-id-prefix",
                f"tsu-{name}-",
            ]
            (args.output / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (args.output / f"{name}.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads(result_path.read_text())
            if result.get("completed") != requests:
                raise RuntimeError(f"Incomplete requests: {name}: {result.get('completed')} / {requests}")
            row = {"shape": [isl, osl, concurrency, requests], "repeat": repeat, "source": result_path.name}
            row.update(
                {
                    key: result.get(key)
                    for key in (
                        "completed",
                        "mean_ttft_ms",
                        "median_ttft_ms",
                        "p99_ttft_ms",
                        "mean_tpot_ms",
                        "p99_tpot_ms",
                        "median_itl_ms",
                        "p99_itl_ms",
                        "output_throughput",
                        "mean_e2el_ms",
                    )
                }
            )
            row["tsu"] = 1000 / result["mean_tpot_ms"]
            rows.append(row)
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
