"""Setup-only parser and phase-collector probes, never scored benchmarks."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from benchmark_stage.responses import scoring_response

ROOT = Path(__file__).resolve().parents[1] / "doc/benchmark/setup"
MODEL = "IFM/K2-Horizon-7B"
BASE_URL = "http://127.0.0.1:8000"


def main():
    start = time.monotonic()
    metadata = json.loads((ROOT / "prompt-format.json").read_text())
    chat = dict(
        model=MODEL,
        messages=metadata["smoke_messages"],
        max_tokens=32768,
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        seed=1234,
        chat_template_kwargs={"reasoning_effort": "high"},
    )
    response = requests.post(
        BASE_URL + "/v1/chat/completions", json=chat, headers={"x-request-id": "setup-native-chat"}, timeout=300
    )
    (ROOT / "native-chat-smoke.json").write_text(
        json.dumps(
            {
                "request": chat,
                "status": response.status_code,
                "response": response.json(),
                "wall_seconds": time.monotonic() - start,
            },
            indent=2,
        )
        + "\n"
    )
    response.raise_for_status()
    scoring_response(response.json())
    print("Native chat parser smoke complete", flush=True)

    def probe(index):
        payload = dict(model=MODEL, prompt=[20000 + index] * 4096, max_tokens=8, temperature=0, ignore_eos=True)
        before = time.monotonic()
        result = requests.post(
            BASE_URL + "/v1/completions", json=payload, headers={"x-request-id": f"setup-phase-32-{index}"}, timeout=600
        )
        row = dict(
            index=index,
            request=payload,
            status=result.status_code,
            response=result.json(),
            wall_seconds=time.monotonic() - before,
        )
        (ROOT / f"phase-probe-{index}.json").write_text(json.dumps(row, indent=2) + "\n")
        result.raise_for_status()
        usage = result.json()["usage"]
        assert usage["prompt_tokens"] == 4096 and usage["completion_tokens"] == 8, usage
        return {k: v for k, v in row.items() if k != "request"}

    with ThreadPoolExecutor(max_workers=32) as pool:
        rows = list(pool.map(probe, range(32)))
    (ROOT / "phase-probe-summary.json").write_text(
        json.dumps(
            {
                "scope": "setup collector validation only; 8 output tokens",
                "completed": len(rows),
                "total_setup_probe_seconds": time.monotonic() - start,
                "responses": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("Completed 32 phase probes, 4096 input / 8 output tokens", flush=True)


if __name__ == "__main__":
    main()
