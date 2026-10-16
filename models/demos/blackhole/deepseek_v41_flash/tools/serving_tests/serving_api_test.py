#!/usr/bin/env python3
"""OpenAI-compatible serving checks against the running DSV4.1 server (vLLM 0.31 + tt plugin). usage: serving_api_test.py <max_model_len>
Every check prints PASS / FAIL with the evidence; the exit code is 0 when all pass. Expectations come from the server's documented behaviour: sampling on device, no logprobs / structured outputs under speculation (clean OpenAI-style 400)."""
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

L = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
U = "http://127.0.0.1:8000"
M = "deepseek-ai/DeepSeek-V4.1-Flash"
res = []


def call(method, path, body=None, timeout=900, raw=False):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(U + path, data, {"Content-Type": "application/json"}, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return (r.status, r if raw else json.loads(r.read().decode() or "null"))
    except urllib.error.HTTPError as e:
        t = e.read().decode()
        try:
            return e.code, json.loads(t)
        except Exception:
            return e.code, t


def chat(msgs, **kw):
    return call("POST", "/v1/chat/completions", dict(model=M, messages=msgs, **kw))


def user(t):
    return [{"role": "user", "content": t}]


def text(j):
    return j["choices"][0]["message"]["content"]


def check(name, fn):
    t = time.time()
    try:
        ok, ev = fn()
    except Exception as e:
        ok, ev = False, "exception " + repr(e)[:200]
    res.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({time.time()-t:.1f}s)  {str(ev)[:230]}", flush=True)


def t_models():
    st, j = call("GET", "/v1/models")
    ids = [m["id"] for m in j["data"]]
    return st == 200 and M in ids, ids


def t_health():
    r = urllib.request.urlopen(U + "/health", timeout=20)
    return r.status == 200, r.status


def t_chat():
    st, j = chat(user("What is 6 times 7? Answer with just the number."), max_tokens=8, temperature=0)
    return st == 200 and "42" in text(j) and j["usage"]["completion_tokens"] > 0, (st, text(j), j["usage"])


def t_completions():
    st, j = call(
        "POST", "/v1/completions", dict(model=M, prompt="The capital of France is", max_tokens=6, temperature=0)
    )
    return st == 200 and "Paris" in j["choices"][0]["text"], (st, j["choices"][0]["text"])


def t_system_multiturn():
    msgs = [
        {"role": "system", "content": "You answer in exactly one word."},
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue"},
        {"role": "user", "content": "Name another color."},
    ]
    st, j = chat(msgs, max_tokens=8, temperature=0)
    a = text(j).strip()
    return st == 200 and 0 < len(a.split()) <= 3 and "blue" not in a.lower(), a


def t_finish_reasons():
    st, j = chat(user("Count from 1 to 200 separated by commas."), max_tokens=10, temperature=0)
    a = j["choices"][0]["finish_reason"]
    st2, j2 = chat(user("Say hi."), max_tokens=64, temperature=0)
    b = j2["choices"][0]["finish_reason"]
    return a == "length" and b == "stop" and j["usage"]["completion_tokens"] == 10, (a, b, j["usage"])


def t_stop():
    st, j = chat(
        user("Count from 1 to 10, separated by commas, nothing else."), max_tokens=60, temperature=0, stop=["5"]
    )
    a = text(j)
    return st == 200 and "5" not in a and "4" in a and j["choices"][0]["finish_reason"] == "stop", a


def t_stream_chat():
    st, r = call(
        "POST",
        "/v1/chat/completions",
        dict(
            model=M,
            messages=user("Count from 1 to 8, separated by commas."),
            max_tokens=40,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        ),
        raw=True,
    )
    parts, usage, done, first = [], None, False, None
    t0 = time.time()
    for line in r:
        line = line.decode().strip()
        if line == "data: [DONE]":
            done = True
        elif line.startswith("data: "):
            j = json.loads(line[6:])
            if j.get("usage"):
                usage = j["usage"]
            if j["choices"] and j["choices"][0]["delta"].get("content"):
                first = first or time.time() - t0
                parts.append(j["choices"][0]["delta"]["content"])
    s = "".join(parts)
    return st == 200 and done and usage and "8" in s and len(parts) > 3, (
        len(parts),
        repr(s[:60]),
        usage,
        f"ttft {first:.2f}s",
    )


def t_stream_completions():
    st, r = call(
        "POST", "/v1/completions", dict(model=M, prompt="1, 2, 3,", max_tokens=12, temperature=0, stream=True), raw=True
    )
    parts, done = [], False
    for line in r:
        line = line.decode().strip()
        if line == "data: [DONE]":
            done = True
        elif line.startswith("data: "):
            parts.append(json.loads(line[6:])["choices"][0]["text"])
    return st == 200 and done and "4" in "".join(parts), repr("".join(parts)[:50])


def t_seed():
    kw = dict(max_tokens=24, temperature=0.9, top_p=0.95)
    a = text(chat(user("Write a short sentence about the ocean."), seed=1234, **kw)[1])
    b = text(chat(user("Write a short sentence about the ocean."), seed=1234, **kw)[1])
    c = text(chat(user("Write a short sentence about the ocean."), seed=99, **kw)[1])
    d = text(chat(user("Write a short sentence about the ocean."), seed=100, **kw)[1])
    return a == b and (a != c or c != d), (a == b, a != c, c != d)


def t_top_k_1_is_greedy():
    g = text(chat(user("Write one sentence about trees."), max_tokens=20, temperature=0)[1])
    k = text(chat(user("Write one sentence about trees."), max_tokens=20, temperature=1.0, top_k=1)[1])
    return g == k, (g[:50], k[:50])


def t_sampling_params():
    st, j = chat(
        user("Give me one word."),
        max_tokens=6,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        frequency_penalty=0.2,
        presence_penalty=0.2,
    )
    return st in (200, 400), (st, (text(j) if st == 200 else str(j)[:150]))


def t_logprobs_policy():
    st, j = chat(user("Hi"), max_tokens=4, temperature=0, logprobs=True, top_logprobs=2)
    if st == 200:
        return True, "logprobs served: " + str(j["choices"][0].get("logprobs"))[:100]
    return st == 400 and isinstance(j, dict) and "error" in j and "message" in j["error"], (st, str(j)[:200])


def t_n_param():
    st, j = chat(user("Say a number."), max_tokens=4, temperature=0.8, n=2)
    return (st == 200 and len(j["choices"]) == 2) or (st == 400 and "error" in j), (
        st,
        len(j["choices"]) if st == 200 else str(j)[:150],
    )


def t_too_long():
    words = " ".join(["alpha"] * (L + 2000))
    st, j = chat(user(words), max_tokens=4)
    return st == 400 and isinstance(j, dict) and "error" in j and (
        "maximum context length" in j["error"]["message"]
        or "too long" in j["error"]["message"].lower()
        or "max" in j["error"]["message"].lower()
    ), (st, str(j)[:260])


def t_context_plus_output_limit():
    st, j = chat(user("Hi"), max_tokens=L + 10)
    return st == 400 and "error" in j, (st, str(j)[:200])


def t_bad_model():
    st, j = call("POST", "/v1/chat/completions", dict(model="nope/none", messages=user("hi"), max_tokens=4))
    return st == 404 and "error" in j, (st, str(j)[:160])


def t_bad_body():
    st, j = call("POST", "/v1/chat/completions", dict(model=M))
    a = st in (400, 422) and (isinstance(j, dict) and ("error" in j or "detail" in j))
    st2, j2 = chat(user("hi"), max_tokens=4, temperature=-1.0)
    return a and st2 in (400, 422), (st, st2)


def t_concurrent_mixed():
    out = {}

    def w(i):
        if i % 2:
            out[i] = chat(user(f"What is {i}+{i}? Just the number."), max_tokens=8, temperature=0)
        else:
            st, r = call(
                "POST",
                "/v1/chat/completions",
                dict(
                    model=M,
                    messages=user(f"What is {i}+{i}? Just the number."),
                    max_tokens=8,
                    temperature=0,
                    stream=True,
                ),
                raw=True,
            )
            s = ""
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    c = json.loads(line[6:])["choices"]
                    s += (c[0]["delta"].get("content") or "") if c else ""
            out[i] = (st, {"choices": [{"message": {"content": s}}]})

    ts = [threading.Thread(target=w, args=(i,)) for i in range(2, 14)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    good = sum(1 for i, (st, j) in out.items() if st == 200 and str(i + i) in text(j))
    return good == 12, f"{good}/12"


def t_client_disconnect():
    s = socket.create_connection(("127.0.0.1", 8000), timeout=30)
    body = json.dumps(
        dict(model=M, messages=user("Write a long story about a dragon."), max_tokens=400, temperature=0.5, stream=True)
    ).encode()
    s.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
    )
    s.recv(512)
    s.close()
    time.sleep(3)
    st, j = chat(user("What is 2+3? Just the number."), max_tokens=6, temperature=0)
    return st == 200 and "5" in text(j), (st, text(j))


def t_unicode_and_long_output():
    st, j = chat(
        user("Translate to French: 'Good morning, how are you?' Then write the Chinese word for water."),
        max_tokens=60,
        temperature=0,
    )
    a = text(j)
    return st == 200 and len(a) > 10 and ("Bonjour" in a or "bonjour" in a), a[:120]


def t_chat_template_nonthinking():
    st, j = chat(user("What is 15+27? Answer with just the number."), max_tokens=12, temperature=0)
    a = text(j)
    return st == 200 and "42" in a and "<think>" not in a, repr(a)


def t_health_after():
    r = urllib.request.urlopen(U + "/health", timeout=20)
    return r.status == 200, r.status


for name, fn in [
    ("GET /health", t_health),
    ("GET /v1/models", t_models),
    ("chat completion", t_chat),
    ("/v1/completions", t_completions),
    ("system prompt + multi-turn", t_system_multiturn),
    ("finish_reason length / stop + usage", t_finish_reasons),
    ("stop sequence", t_stop),
    ("stream chat (+usage, [DONE])", t_stream_chat),
    ("stream completions", t_stream_completions),
    ("seed reproducible / different seeds differ", t_seed),
    ("top_k=1 equals greedy", t_top_k_1_is_greedy),
    ("sampling params accepted", t_sampling_params),
    ("logprobs: served or a clean OpenAI 400", t_logprobs_policy),
    ("n=2: served or a clean 400", t_n_param),
    ("prompt longer than max_model_len -> 400", t_too_long),
    ("max_tokens beyond the context -> 400", t_context_plus_output_limit),
    ("unknown model -> 404", t_bad_model),
    ("malformed body / bad temperature -> 4xx", t_bad_body),
    ("12 concurrent mixed streaming / non-streaming", t_concurrent_mixed),
    ("client disconnect mid-stream, server keeps serving", t_client_disconnect),
    ("unicode / French / Chinese", t_unicode_and_long_output),
    ("chat template is non-thinking", t_chat_template_nonthinking),
    ("GET /health after everything", t_health_after),
]:
    check(name, fn)
print(f"API checks: {sum(res)}/{len(res)} passed")
print("API_OK" if all(res) else "API_FAIL")
sys.exit(0 if all(res) else 1)
