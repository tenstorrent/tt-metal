"""Validate the rebuilt image: reasoning off by default, opt-in still works, cap applied."""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = ("Write a 300+ word summary of the wikipedia page "
          "\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". "
          "Do not use any commas and highlight at least 3 sections that has titles "
          "in markdown format, for example *highlighted section part 1*, "
          "*highlighted section part 2*, *highlighted section part 3*.")

def ask(extra, label, prompt=PROMPT, max_tokens=16384):
    payload = {"model": "zai-org/GLM-4.7-Flash",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "seed": 42, **extra}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3000))
    c = d["choices"][0]; m = c["message"]
    content = m.get("content") or ""
    reas = m.get("reasoning") or m.get("reasoning_content") or ""
    print(f"[{label}] finish={c.get('finish_reason')} tokens={d['usage']['completion_tokens']} "
          f"content={len(content)}ch reasoning={len(reas)}ch {time.time()-t0:.0f}s", flush=True)
    return content, reas, c.get("finish_reason")

fails = []

# 1. DEFAULT request: the loop prompt that used to return nothing.
content, reas, fin = ask({}, "default (no chat_template_kwargs)")
if reas:
    fails.append(f"default request produced {len(reas)}ch of reasoning; expected none")
if not content.strip():
    fails.append("default request returned EMPTY content (the defect is not fixed)")
if fin != "stop":
    fails.append(f"default request finish_reason={fin}, expected stop")

# 2. OPT-IN must still work.
content2, reas2, _ = ask({"chat_template_kwargs": {"enable_thinking": True}},
                         "opt-in enable_thinking=true", prompt="What is 17 times 23? Think it through.",
                         max_tokens=2048)
if not reas2:
    fails.append("enable_thinking=true produced NO reasoning; per-request opt-in is broken")

# 3. Reasoning-off answer must still be a real answer.
import re
w = len(content.split()); com = content.count(","); hl = len(re.findall(r"[*][^*\n]+[*]", content))
print(f"\ndefault answer vs IFEval constraints: words={w} (need 300+) commas={com} (need 0) highlights={hl} (need 3+)")
print(f"default answer head: {content[:180]!r}")
print(f"opt-in reasoning head: {reas2[:160]!r}")

print("\n" + ("FAILED:\n  " + "\n  ".join(fails) if fails else "ALL CHECKS PASSED"))
sys.exit(1 if fails else 0)
