#!/usr/bin/env python3
"""Real HumanEval-164 pass@1 against the served Laguna-XS-2.1 vLLM endpoint.

Reads the canonical human_eval dataset (164 problems), generates a greedy completion for each prompt via
/v1/completions, assembles prompt+completion, and executes the problem's hidden test in a sandboxed
subprocess (like eval_code.py) — avoiding the human_eval package's disabled-exec. Reports pass@1.

Run (swebench-venv has human_eval + requests):
  /home/ttuser/swebench-venv/bin/python humaneval_served.py --base-url http://localhost:8000 \
      --concurrency 8 --out <dir>/humaneval.json
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from human_eval.data import read_problems

STOP = ["\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n@", "\nassert "]


def _extract_code(text, prompt):
    """Strip a reasoning-model's leakage: drop everything up to the last </think>, then pull the fenced
    ```python block if present. Returns a function body/definition suitable to append to the prompt."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    import re as _re

    m = _re.search(r"```(?:python)?\s*(.+?)```", text, _re.S)
    if m:
        return m.group(1)
    # Unclosed fence (model truncated before the closing ```): take everything after the opening fence.
    m = _re.search(r"```(?:python)?\s*\n(.+)", text, _re.S)
    if m:
        return m.group(1)
    return text


def generate(base_url, model, prompt, max_tokens, temperature, timeout, chat=False):
    if not chat:
        r = requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stop": STOP},
            timeout=timeout,
        )
        r.raise_for_status()
        return prompt + r.json()["choices"][0]["text"], False  # program, is_full
    # CHAT mode with thinking DISABLED — the correct way to eval a reasoning model (no CoT leak into code).
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Complete this Python function. Return ONLY the complete function "
                    "in a ```python code block, no explanation.\n\n" + prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _extract_code(content, prompt), True  # full program


def run_test(program, test_src, entry_point, timeout=12):
    full = program + "\n\n" + test_src + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(full)
        f.flush()
        try:
            p = subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout)
            return p.returncode == 0, (p.stderr.decode()[-200:] if p.returncode else "")
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}:{e}"


def one(task_id, prob, args):
    t0 = time.time()
    try:
        program, _is_full = generate(
            args.base_url,
            args.model,
            prob["prompt"],
            args.max_tokens,
            args.temperature,
            args.req_timeout,
            chat=args.chat,
        )
    except Exception as e:  # noqa: BLE001
        return task_id, False, f"gen_err:{type(e).__name__}", time.time() - t0
    if _is_full:
        # Chat mode returns just the function; the model re-writes the signature but drops the prompt's
        # import preamble (`from typing import List`, `import math`, ...), so `List[int]` hints NameError.
        # Restore the prompt's imports (the completion path kept them via prompt+completion). Idempotent.
        imports = "\n".join(ln for ln in prob["prompt"].splitlines() if ln.startswith(("import ", "from ")))
        if imports:
            program = imports + "\n" + program
    ok, err = run_test(program, prob["test"], prob["entry_point"])
    return task_id, ok, err, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="poolside/Laguna-XS-2.1")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--chat", action="store_true", help="use /v1/chat/completions with enable_thinking=false (no CoT leak)"
    )
    ap.add_argument("--req-timeout", type=int, default=300)
    ap.add_argument("-n", type=int, default=0, help="limit problems (0=all 164)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    problems = read_problems()
    items = list(problems.items())
    if args.n:
        items = items[: args.n]
    total = len(items)
    print(f"[humaneval] {total} problems, concurrency={args.concurrency}, greedy, model={args.model}", flush=True)

    results = {}
    passed = 0
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, tid, prob, args) for tid, prob in items]
        for fut in as_completed(futs):
            tid, ok, err, dt = fut.result()
            results[tid] = {"pass": ok, "err": err, "s": round(dt, 1)}
            passed += int(ok)
            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"[humaneval] {done}/{total} done, pass@1={passed}/{done}={100*passed/done:.1f}% "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )
    pass1 = 100 * passed / total
    print(f"[humaneval] DONE pass@1 = {passed}/{total} = {pass1:.1f}%  in {time.time()-t0:.0f}s", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"pass_at_1": pass1, "passed": passed, "total": total, "results": results}, f, indent=2)
        print(f"[humaneval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
