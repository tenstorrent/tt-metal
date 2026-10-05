"""Unsupported strict-device requests must fail before EngineCore submission."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--host-compat", action="store_true")
    args = parser.parse_args()
    result = {"mode": "strict device sampling", "cases": [], "supported": []}
    common = {
        "model": "IFM/K2-Horizon-7B",
        "prompt": list(range(500, 531)),
        "max_tokens": 8,
    }
    candidates = [
        ("default_sampling", {}),
        ("unrestricted_zero", {"temperature": 0.7, "top_k": 0}),
        ("unrestricted_minus_one", {"temperature": 0.7, "top_k": -1}),
        ("large_top_k", {"temperature": 0.7, "top_k": 33}),
        ("logprobs_zero", {"logprobs": 0}),
        ("logprobs_one", {"logprobs": 1}),
        ("logprobs_twenty", {"logprobs": 20}),
        ("presence_penalty", {"presence_penalty": 0.5}),
        ("frequency_penalty", {"frequency_penalty": 0.5}),
        ("repetition_penalty", {"repetition_penalty": 1.1}),
        ("min_p", {"temperature": 0.7, "top_k": 32, "min_p": 0.1}),
        ("allowed_tokens", {"allowed_token_ids": [17, 18]}),
        ("logit_bias", {"logit_bias": {"17": 1.0}}),
        ("min_tokens", {"min_tokens": 2}),
        ("structured_outputs", {"structured_outputs": {"choice": ["yes", "no"]}}),
        ("seed_overflow", {"seed": 2**31}),
        ("seed_underflow", {"seed": -(2**31) - 1}),
        (
            "seed_child_overflow",
            {"temperature": 0.7, "top_k": 32, "seed": 2**31 - 1, "n": 2},
        ),
        ("temperature_overflow", {"temperature": 1e39, "top_k": 32}),
    ]

    def save():
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")

    def supported(phase, sampled=False):
        payload = dict(
            common,
            temperature=0.7 if sampled else 0,
            top_k=32,
            top_p=0.9,
            seed=17,
            ignore_eos=True,
            return_token_ids=True,
        )
        if not sampled:
            payload.pop("top_k")  # Default top-k is valid for semantic greedy.
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=120)
        data = response.json()
        result["supported"].append(dict(phase=phase, request=payload, status=response.status_code, response=data))
        save()
        assert response.status_code == 200, data
        ids = data["choices"][0]["token_ids"]
        assert len(ids) == 8, data
        return ids

    if args.host_compat:
        result["mode"] = "explicit host compatibility"
        for label, extra in [
            ("default_sampling", {}),
            ("host_logprobs", {"temperature": 0, "logprobs": 1}),
            ("host_constraint", {"temperature": 0, "allowed_token_ids": [17, 18]}),
        ]:
            payload = dict(common, ignore_eos=True, return_token_ids=True, **extra)
            response = requests.post(args.url + "/v1/completions", json=payload, timeout=120)
            data = response.json()
            result["cases"].append(dict(id=label, request=payload, status=response.status_code, response=data))
            save()
            assert response.status_code == 200, data
            ids = data["choices"][0]["token_ids"]
            assert len(ids) == 8, data
            if label == "host_logprobs":
                assert data["choices"][0]["logprobs"]["token_logprobs"], data
            if label == "host_constraint":
                assert set(ids).issubset({17, 18}), data
        result["passed"] = True
        save()
        print("HOST_ADMISSION_PASS", args.output)
        return

    baseline = supported("before")
    for label, overrides in candidates:
        payload = (
            dict(common, temperature=0, **overrides) if "temperature" not in overrides else dict(common, **overrides)
        )
        if label == "default_sampling":
            payload = dict(common)
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=30)
        result["cases"].append(
            dict(
                id=label,
                request=payload,
                status=response.status_code,
                response=response.json(),
            )
        )
        save()
        assert response.status_code == 400, result["cases"][-1]
        assert "K2 " in response.text and "sampling" in response.text, response.text
        assert requests.get(args.url + "/health", timeout=10).status_code == 200
        assert supported("after_" + label) == baseline
    # Both endpoints reject ordinary streamed requests before selecting SSE.
    chat_common = dict(
        model=common["model"],
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=8,
    )
    for label, endpoint, payload in [
        ("chat_default_sampling", "/v1/chat/completions", chat_common),
        (
            "chat_bad_words",
            "/v1/chat/completions",
            dict(chat_common, temperature=0, bad_words=["hello"]),
        ),
        (
            "chat_penalty",
            "/v1/chat/completions",
            dict(chat_common, temperature=0, frequency_penalty=0.5),
        ),
        (
            "chat_logprobs_stream",
            "/v1/chat/completions",
            dict(chat_common, temperature=0, logprobs=True, stream=True),
        ),
        (
            "completion_logprobs_stream",
            "/v1/completions",
            dict(common, temperature=0, logprobs=0, stream=True),
        ),
        (
            "chat_beam_nonstream",
            "/v1/chat/completions",
            dict(chat_common, temperature=0, use_beam_search=True),
        ),
    ]:
        response = requests.post(args.url + endpoint, json=payload, timeout=30)
        result["cases"].append(
            dict(
                id=label,
                endpoint=endpoint,
                request=payload,
                status=response.status_code,
                response=response.json(),
            )
        )
        save()
        assert response.status_code == 400, result["cases"][-1]
        assert "K2 " in response.text and "sampling" in response.text, response.text
        assert supported("after_" + label) == baseline
    # Chat beam search is evaluated lazily by the pinned API. Its documented
    # SSE error delivery still rejects before EngineCore and must not poison it.
    payload = dict(chat_common, temperature=0, use_beam_search=True, stream=True)
    response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=30)
    result["beam_stream"] = dict(request=payload, status=response.status_code, response=response.text)
    save()
    assert response.status_code == 200 and '"code": 400' in response.text, response.text
    assert "K2 strict device sampling" in response.text and "[DONE]" in response.text
    assert supported("after_beam_stream") == baseline
    # Receive the first streamed output before rejecting another request, so
    # a supported generation is demonstrably live during the bad submission.
    payload = dict(
        common,
        temperature=0,
        max_tokens=65,
        ignore_eos=True,
        stream=True,
        stream_options={"include_usage": True},
    )
    stream = requests.post(args.url + "/v1/completions", json=payload, stream=True, timeout=120)
    assert stream.status_code == 200
    chunks = []
    rejected = False
    for line in stream.iter_lines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        chunks.append(json.loads(line[6:]))
        if not rejected and chunks[-1].get("choices"):
            invalid = dict(common, logprobs=0, temperature=0)
            response = requests.post(args.url + "/v1/completions", json=invalid, timeout=30)
            result["cases"].append(
                dict(
                    id="reject_during_live_stream",
                    request=invalid,
                    status=response.status_code,
                    response=response.json(),
                )
            )
            save()
            assert response.status_code == 400, result["cases"][-1]
            assert "K2 " in response.text and "sampling" in response.text, response.text
            rejected = True
    stream.close()
    result["live_stream"] = dict(request=payload, chunks=chunks)
    save()
    assert rejected and chunks[-1]["usage"]["completion_tokens"] == 65, chunks
    assert supported("after_live_stream") == baseline
    supported("supported_stochastic", sampled=True)
    result["passed"] = True
    save()
    print("STRICT_ADMISSION_PASS", args.output)


if __name__ == "__main__":
    main()
