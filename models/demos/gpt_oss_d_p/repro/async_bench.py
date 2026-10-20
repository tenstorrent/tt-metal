import json
import os
import sys
import time

import requests
from transformers import AutoTokenizer


target = int(sys.argv[1])
frontend_url = os.environ["FRONTEND_URL"].rstrip("/")
tokenizer_path = os.environ["TOKENIZER_PATH"]
served_model = os.environ.get("SERVED_MODEL_NAME", "gpt-oss-120b-4x2-chat")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
nonce = time.time_ns()


def make_prompt(repeats):
    return f"Unique request {nonce}. " + "hello " * repeats + "Briefly summarize the preceding text."


def token_count(prompt):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(encoded["input_ids"])


lo, hi = 1, target + 1024
while lo < hi:
    mid = (lo + hi + 1) // 2
    if token_count(make_prompt(mid)) <= target:
        lo = mid
    else:
        hi = mid - 1

prompt = make_prompt(lo)
input_tokens = token_count(prompt)
body = {
    "model": served_model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 32,
    "stream": True,
    "stream_options": {"include_usage": True},
}
t0 = time.perf_counter()
first = None
last = None
usage = None
events = 0
with requests.post(
    f"{frontend_url}/v1/chat/completions", json=body, stream=True, timeout=300
) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        event = json.loads(payload)
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if any(value for key, value in delta.items() if key != "role"):
                now = time.perf_counter()
                first = first or now
                last = now
                events += 1
t1 = time.perf_counter()
print(
    json.dumps(
        {
            "target_tokens": target,
            "templated_input_tokens": input_tokens,
            "ttft_s": None if first is None else first - t0,
            "total_s": t1 - t0,
            "content_events": events,
            "usage": usage,
        }
    )
)
