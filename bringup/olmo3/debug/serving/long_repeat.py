import json, requests

d = json.load(open("/home/ubuntu/ssinghal/olmo3/scratch/prefix_24450.json"))
P = d["prompt_len"]
U = "http://localhost:8000/v1/completions"


def go(label, prompt, n=5):
    r = requests.post(
        U, json=dict(model="allenai/Olmo-3.1-32B-Think", prompt=prompt, max_tokens=n, temperature=0), timeout=900
    ).json()
    try:
        print(label, repr(r["choices"][0]["text"]), flush=True)
    except Exception:
        print(label, str(r)[:120])


short = "<|im_start|>user\nWhat is the capital of France? Explain briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n"
go("short before     ", short, 8)
for i in range(4):
    go(f"8.2K greedy #{i}  ", d["ids"][: P + 8000], 5)
go("short after      ", short, 8)
