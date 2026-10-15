import requests, sys

URL = "http://localhost:8000/v1/completions"
PR = "<|im_start|>user\nWhat is the capital of France? Explain briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n"


def go(label, n=6, prompt=PR, **kw):
    for i in range(n):
        r = requests.post(
            URL, json=dict(model="allenai/Olmo-3.1-32B-Think", prompt=prompt, max_tokens=50, **kw), timeout=900
        )
        try:
            print(label, i, repr(r.json()["choices"][0]["text"][:110]), flush=True)
        except Exception:
            print(label, i, r.text[:120])
            return


if __name__ == "__main__":
    go("dev-sampled", temperature=0.6, top_p=0.95)
    go("greedy", 3, temperature=0)
    go("dev-sampled-after-greedy", 3, temperature=0.6, top_p=0.95)
    go("host-sampled(top_k=0)", 3, temperature=0.6, top_p=0.95, top_k=0)
    go("dev-sampled-again", 3, temperature=0.6, top_p=0.95)
