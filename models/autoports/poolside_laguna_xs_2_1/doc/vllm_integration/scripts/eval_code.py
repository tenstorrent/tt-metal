#!/usr/bin/env python3
"""Self-contained code-accuracy eval for the served Laguna-XS-2.1 vLLM endpoint.

For each problem we send the function signature + docstring, take the model's completion,
execute it against hidden asserts in a sandboxed subprocess, and report pass@1. No external
datasets — the problems + tests are embedded, so it runs anywhere the server is reachable.

Usage:
  python eval_code.py                 # all problems, greedy
  python eval_code.py -n 5            # first 5
  python eval_code.py --base-url http://localhost:8000 --temperature 0.0 --max-tokens 512
"""
import argparse
import re
import subprocess
import sys
import tempfile
import textwrap
import time

from openai import OpenAI

# (entry_point, prompt, test_body) — test_body asserts against the defined function.
PROBLEMS = [
    (
        "solve",
        'def solve(nums):\n    """Return the sum of all even numbers in the list nums."""\n',
        "assert solve([1,2,3,4,5,6])==12\nassert solve([1,3,5])==0\nassert solve([])==0\nassert solve([2,4,-6])==0",
    ),
    (
        "is_palindrome",
        'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards, ignoring case and non-alphanumeric characters."""\n',
        "assert is_palindrome('A man, a plan, a canal: Panama')==True\nassert is_palindrome('hello')==False\nassert is_palindrome('')==True",
    ),
    (
        "gcd",
        'def gcd(a, b):\n    """Return the greatest common divisor of positive integers a and b."""\n',
        "assert gcd(12,18)==6\nassert gcd(17,5)==1\nassert gcd(100,10)==10",
    ),
    (
        "fib",
        'def fib(n):\n    """Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)."""\n',
        "assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55\nassert fib(15)==610",
    ),
    (
        "two_sum",
        'def two_sum(nums, target):\n    """Return indices [i, j] of the two numbers in nums that add up to target (i<j). Exactly one solution exists."""\n',
        "assert two_sum([2,7,11,15],9)==[0,1]\nassert two_sum([3,2,4],6)==[1,2]",
    ),
    (
        "count_vowels",
        'def count_vowels(s):\n    """Return the number of vowels (a,e,i,o,u, case-insensitive) in the string s."""\n',
        "assert count_vowels('hello')==2\nassert count_vowels('XYZ')==0\nassert count_vowels('AeIoU')==5",
    ),
    (
        "flatten",
        'def flatten(lst):\n    """Flatten an arbitrarily nested list of integers into a single flat list, preserving order."""\n',
        "assert flatten([1,[2,[3,4]],5])==[1,2,3,4,5]\nassert flatten([])==[]\nassert flatten([[1],[2],[3]])==[1,2,3]",
    ),
    (
        "is_prime",
        'def is_prime(n):\n    """Return True if n is a prime number, else False."""\n',
        "assert is_prime(2)==True\nassert is_prime(15)==False\nassert is_prime(97)==True\nassert is_prime(1)==False",
    ),
    (
        "run_length",
        'def run_length(s):\n    """Run-length encode a string, e.g. \'aaabbc\' -> \'a3b2c1\'."""\n',
        "assert run_length('aaabbc')=='a3b2c1'\nassert run_length('x')=='x1'\nassert run_length('')==''",
    ),
    (
        "max_subarray",
        'def max_subarray(nums):\n    """Return the maximum sum of any contiguous non-empty subarray of nums."""\n',
        "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6\nassert max_subarray([1])==1\nassert max_subarray([-1,-2,-3])==-1",
    ),
]


def extract_code(text, entry):
    """Pull runnable python out of the completion: prefer a fenced block, else use the raw text."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    # Ensure the entry-point definition is present; if the model only returned a body, we still
    # have the signature from the prompt (caller prepends it).
    return body


def run_tests(code, test_body, entry, timeout=15):
    src = code + "\n\n" + test_body + "\nprint('PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and "PASS" in r.stdout, (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="poolside/Laguna-XS-2.1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)  # headroom for always-on thinking + code
    ap.add_argument("-n", type=int, default=len(PROBLEMS))
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="x")
    probs = PROBLEMS[: args.n]
    passed = 0
    t0 = time.time()
    print(f"code eval · {args.model} · {len(probs)} problems · temp={args.temperature}\n" + "-" * 62)
    for i, (entry, prompt, tests) in enumerate(probs, 1):
        instr = (
            "Complete the following Python function. Return ONLY the full function "
            "definition inside a ```python code block.\n\n```python\n" + prompt + "```"
        )
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": instr}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                # enable_thinking:true is REQUIRED — without it the template puts </think> in the
                # prompt, the deepseek_r1 reasoning parser then classifies the whole output as
                # reasoning, and message.content comes back EMPTY (→ 0/10 AssertionErrors).
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
            )
            msg = resp.choices[0].message
            # code lands in content once thinking is on; fall back to reasoning just in case.
            out = msg.content or getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", "") or ""
        except Exception as e:
            print(f"[{i:2}/{len(probs)}] {entry:<14} ERROR calling server: {e}")
            continue
        code = extract_code(out, entry)
        if f"def {entry}" not in code:  # model returned only a body → prepend signature
            code = prompt + textwrap.indent(code, "    ") if not code.lstrip().startswith("def") else code
        ok, err = run_tests(code, tests, entry)
        passed += ok
        print(f"[{i:2}/{len(probs)}] {entry:<14} {'PASS ✓' if ok else 'FAIL ✗  ' + err[:48]}")
    dt = time.time() - t0
    print("-" * 62)
    print(f"pass@1: {passed}/{len(probs)} = {100*passed/len(probs):.1f}%   ({dt:.1f}s)")


if __name__ == "__main__":
    main()
