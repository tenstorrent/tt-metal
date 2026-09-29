"""Compare live serving logprobs with the direct adapter/standalone controls."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    controls = json.loads(Path(args.control).read_text())["serving_logprob_controls"]

    def request(index):
        control = controls[index]
        payload = dict(
            model="IFM/K2-Horizon-7B",
            prompt=control["prompt"],
            max_tokens=1,
            temperature=0,
            logprobs=20,
            return_tokens_as_token_ids=True,
        )
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        assert "error" not in data, data
        actual = data["choices"][0]["logprobs"]["top_logprobs"][0]
        expected = control["top_logprobs"]
        differences = [abs(actual["token_id:" + token] - value) for token, value in expected.items()]
        assert max(differences) < 1e-4, (index, max(differences))
        return dict(index=index, maximum_absolute_error=max(differences), request=payload, response=data)

    result = dict(control=args.control, phases={})
    result["phases"]["serial"] = [request(i) for i in range(len(controls))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(controls)) as pool:
        result["phases"]["reordered_concurrent"] = list(pool.map(request, reversed(range(len(controls)))))
        result["phases"]["repeat_concurrent"] = list(pool.map(request, range(len(controls))))
    result["passed"] = True
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("SERVING_LOGIT_CONTROLS_PASS", args.output)


if __name__ == "__main__":
    main()
