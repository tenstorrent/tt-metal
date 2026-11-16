# Sends MMLU prompts that gave garbage on TP32: cold, repeated (prefix hit), and after a sibling warms the shared 5-shot prefix.
import glob, json, random, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.3-70B-Instruct"


def ask(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 10, "temperature": 0, "top_k": 1}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return r["choices"][0]["text"], r["usage"]["prompt_tokens"], cached


by_task = {}
for f in glob.glob("/home/ubuntu/teja/runs/mmlu_tp32_v5/llama70b/samples_*.jsonl"):
    for line in open(f):
        s = json.loads(line)
        by_task.setdefault(f, []).append((s["arguments"]["gen_args_0"]["arg_0"], "\\\\" in s["resps"][0][0]))

random.seed(0)
tasks = [t for t, rows in by_task.items() if sum(g for _, g in rows) >= 2]
for t in random.sample(tasks, 3):
    bad = [p for p, g in by_task[t] if g][:2]
    print("==", t.split("samples_mmlu_llama_")[1].rsplit("_2026", 1)[0])
    for tag, p in [("cold A", bad[0]), ("repeat A", bad[0]), ("sibling B", bad[1])]:
        text, n, c = ask(p)
        garbage = "\\\\" in text
        print(f"  {tag:10} tokens={n} cached={c} garbage={garbage} {text[:30]!r}")
