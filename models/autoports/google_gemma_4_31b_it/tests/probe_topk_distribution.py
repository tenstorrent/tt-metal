# SPDX-License-Identifier: Apache-2.0
"""Retained exact top-k distribution diagnostic used for PC-05 (not a benchmark)."""
import collections
import concurrent.futures
import json
import math
import pathlib
import urllib.request


def main():
    r = pathlib.Path("models/autoports/google_gemma_4_31b_it/readiness_vllm")
    log = r / "final_full/server.log"
    out = r / "topk_distribution_probe.json"
    prompt = "<question>Random letter a-z:</question><answer>"

    def call(body):
        b = {"model": "google/gemma-4-31B-it", "prompt": prompt, "top_p": 1.0, "return_token_ids": True, **body}
        req = urllib.request.Request(
            "http://localhost:8000/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=180) as h:
            return json.load(h)

    def modes(start):
        events = []
        for line in log.read_bytes()[start:].decode(errors="replace").splitlines():
            try:
                e = json.loads(line[line.index("{") :])
                if e.get("event") == "sampling_mode":
                    events.append(e)
            except (ValueError, AttributeError):
                pass
        return events

    report = {"scope": "First-token top-k correctness diagnostic, not performance", "prompt": prompt, "waves": []}
    start = log.stat().st_size
    reference = call({"temperature": 1.0, "top_k": -1, "max_tokens": 1, "logprobs": 20, "seed": 42})
    report["reference"] = reference
    report["reference_modes"] = modes(start)
    lp = reference["choices"][0]["logprobs"]["top_logprobs"][0]
    values = sorted(lp.items(), key=lambda x: x[1], reverse=True)[:10]
    z = sum(math.exp((v - values[0][1]) / 2) for _, v in values)
    probs = [math.exp((v - values[0][1]) / 2) / z for _, v in values]
    report["derived"] = {
        "basis": "Softmax of returned raw log probabilities divided by temperature2 over top10; verify upstream logprob mode before interpreting.",
        "top10_tokens_and_logprobs": values,
        "temperature2_top10_probabilities": probs,
        "pmax": max(probs),
        "probability_all10_same_if_independent": sum(p**10 for p in probs),
    }
    for mode in ["native", "neutral_host"]:
        for rep in range(3):
            configs = [{"temperature": 0, "max_tokens": 10} for _ in range(9)] + [
                {"temperature": 2.0, "top_k": 10, "max_tokens": 10} for _ in range(10)
            ]
            if mode == "neutral_host":
                configs = [dict(x, logit_bias={"0": 0.0}) for x in configs]
            start = log.stat().st_size
            with concurrent.futures.ThreadPoolExecutor(max_workers=19) as pool:
                results = list(pool.map(call, configs))
            ids = [x["choices"][0]["token_ids"][0] for x in results[9:]]
            report["waves"].append(
                {
                    "mode": mode,
                    "repeat": rep,
                    "responses": results,
                    "first_token_counts": dict(collections.Counter(ids)),
                    "log_byte_start": start,
                    "log_byte_end": log.stat().st_size,
                    "sampling_modes": modes(start),
                }
            )
            print(mode, rep, collections.Counter(ids), flush=True)
            out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["derived"]), flush=True)


if __name__ == "__main__":
    main()
