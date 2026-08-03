# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage C — verify concurrent tool-calling decode is CLEAN on the live server.

Memory `laguna-batched-decode-corruption`: under concurrent decode the plugin's changed-only reset_batch
duplicated tokens (`/testbedbed`, `separableable`) and wrecked ~94% of turns -> 0 patches; the fix was
reset_batch=True (full per-step refresh). This probe confirms whether that fix is LIVE on the served
plugin, so Stage E can run the agent benches concurrently (else fall back to batch-1).

Method: fire N concurrent chat completions (thinking on) that each generate a longish coded answer, plus
a 2-shot batch-1 control. Detector = adjacent duplicated substrings `(\\w{5,})\\1` (the corruption
signature). If the concurrent dup-rate is not materially worse than batch-1, concurrent decode is clean.
Also checks a tool-call round-trips concurrently (tool_calls present + JSON-parseable args).

Run: /home/ttuser/.tenstorrent-venv/bin/python stage_c_probe.py [N]
"""
import concurrent.futures as cf
import json
import re
import sys

from openai import OpenAI

CLIENT = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "poolside/Laguna-XS-2.1"
EXTRA = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
DUP = re.compile(r"(\w{5,})\1")

CODE_PROMPT = (
    "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals, "
    "with a short docstring and three example asserts. Then briefly explain the algorithm."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command in the repo and return stdout.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


def one_completion(i, use_tool=False):
    try:
        kwargs = dict(model=MODEL, temperature=1.0, top_p=1.0, max_tokens=1200, extra_body=EXTRA)
        if use_tool:
            r = CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": "List the repo's python files. Use the run_bash tool."}],
                tools=TOOLS,
                tool_choice="auto",
                **{**kwargs, "max_tokens": 800},
            )
            m = r.choices[0].message
            tc = m.tool_calls or []
            ok_json = False
            if tc:
                try:
                    json.loads(tc[0].function.arguments)
                    ok_json = True
                except Exception:
                    ok_json = False
            return {
                "i": i,
                "kind": "tool",
                "n_tool_calls": len(tc),
                "args_json_ok": ok_json,
                "content_len": len(m.content or ""),
                "dups": len(DUP.findall(m.content or "")),
            }
        r = CLIENT.chat.completions.create(messages=[{"role": "user", "content": CODE_PROMPT}], **kwargs)
        txt = r.choices[0].message.content or ""
        return {
            "i": i,
            "kind": "code",
            "content_len": len(txt),
            "dups": len(DUP.findall(txt)),
            "dup_rate": round(len(DUP.findall(txt)) / max(1, len(txt)) * 1000, 3),
        }
    except Exception as e:
        return {"i": i, "error": str(e)[:200]}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"=== Stage C concurrency probe: N={n} concurrent + 2 batch-1 control ===", flush=True)

    print("--- batch-1 control (2 sequential) ---", flush=True)
    ctrl = [one_completion(f"ctrl{i}") for i in range(2)]
    for r in ctrl:
        print(" ", r, flush=True)

    print(f"--- {n} CONCURRENT code completions ---", flush=True)
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        conc = list(ex.map(lambda i: one_completion(f"conc{i}"), range(n)))
    for r in conc:
        print(" ", r, flush=True)

    print("--- 4 CONCURRENT tool-calling requests ---", flush=True)
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        tools = list(ex.map(lambda i: one_completion(f"tool{i}", use_tool=True), range(4)))
    for r in tools:
        print(" ", r, flush=True)

    # verdict
    errs = [r for r in conc + tools + ctrl if "error" in r]
    ctrl_dupr = sum(r.get("dup_rate", 0) for r in ctrl) / max(1, len(ctrl))
    conc_dupr = sum(r.get("dup_rate", 0) for r in conc) / max(1, len(conc))
    tool_ok = sum(1 for r in tools if r.get("n_tool_calls", 0) >= 1 and r.get("args_json_ok"))
    print("\n=== VERDICT ===", flush=True)
    print(f"errors: {len(errs)}", flush=True)
    print(f"batch-1 dup_rate (per 1k chars): {ctrl_dupr:.3f}", flush=True)
    print(f"concurrent dup_rate (per 1k chars): {conc_dupr:.3f}", flush=True)
    print(f"concurrent tool-calls valid: {tool_ok}/4", flush=True)
    clean = (len(errs) == 0) and (conc_dupr <= ctrl_dupr + 0.5) and (tool_ok >= 3)
    print(
        f"CONCURRENT DECODE: {'CLEAN -> run benches concurrent' if clean else 'SUSPECT -> fall back to batch-1'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
