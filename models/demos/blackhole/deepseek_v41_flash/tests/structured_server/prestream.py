#!/usr/bin/env python3
"""Pre-stream validation: requests the launch still cannot serve must return a clean HTTP 400 JSON error in streaming mode too (not 200 + one SSE error event).
Also: supported structured streams are still 200 event streams. Exit 1 on any failure."""
import json
import sys
import urllib.error
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from matrix import PERSON  # noqa: E402

U = "http://127.0.0.1:8000/v1/chat/completions"
M = "deepseek-ai/DeepSeek-V4.1-Flash"
MSG = [{"role": "user", "content": "Invent a person."}]
RF = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON, "strict": True}}


def call(body):
    req = urllib.request.Request(
        U, json.dumps(dict(body, model=M, max_tokens=64, temperature=0)).encode(), {"Content-Type": "application/json"}
    )
    try:
        r = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, e.headers.get("content-type", ""), json.loads(raw)
        except ValueError:
            return e.code, e.headers.get("content-type", ""), raw[:200]
    first = r.readline().decode()
    ctype = r.headers.get("content-type", "")
    r.read()
    return r.status, ctype, first[:160]


REFUSED = [
    ("logprobs alone", dict(messages=MSG, logprobs=True, top_logprobs=1)),
    ("logprobs + json_schema", dict(messages=MSG, logprobs=True, top_logprobs=1, response_format=RF)),
    ("prompt_logprobs + json_schema", dict(messages=MSG, prompt_logprobs=1, response_format=RF)),
    ("min_tokens + regex", dict(messages=MSG, min_tokens=5, structured_outputs={"regex": "[a-z]+"})),
    ("bad_words + choice", dict(messages=MSG, bad_words=["no"], structured_outputs={"choice": ["a", "b"]})),
    ("logit_bias + json_schema", dict(messages=MSG, logit_bias={"5": 1.0}, response_format=RF)),
]
GOOD = [
    ("json_schema", dict(messages=MSG, response_format=RF)),
    ("regex", dict(messages=MSG, structured_outputs={"regex": "[A-Z]{3}-[0-9]{4}"})),
]
bad = 0
for name, body in REFUSED:
    for stream in (False, True):
        code, ctype, payload = call(dict(body, stream=stream))
        ok = code == 400 and "json" in ctype and isinstance(payload, dict) and "error" in payload
        bad += not ok
        print(
            "%-4s refused %-32s stream=%-5s -> HTTP %s %s %s"
            % (
                "PASS" if ok else "FAIL",
                name,
                stream,
                code,
                ctype.split(";")[0],
                str(payload)[:140].replace("\n", " "),
            ),
            flush=True,
        )
for name, body in GOOD:
    code, ctype, payload = call(dict(body, stream=True))
    ok = code == 200 and "event-stream" in ctype and "error" not in str(payload)[:30]
    bad += not ok
    print(
        "%-4s served  %-32s stream=True  -> HTTP %s %s %s"
        % ("PASS" if ok else "FAIL", name, code, ctype.split(";")[0], str(payload)[:80]),
        flush=True,
    )
sys.exit(1 if bad else 0)
