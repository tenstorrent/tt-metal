import json, urllib.request
from transformers import AutoTokenizer
import glob

snap = sorted(glob.glob("/home/tt-admin/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*"))[
    -1
]
tok = AutoTokenizer.from_pretrained(snap)
lines, total = [], 0
for line in open("/home/tt-admin/smanoj/metal_1/top-200000.txt"):
    total += len(tok(line, add_special_tokens=False)["input_ids"])
    if total > 250_000:
        break
    lines.append(line)
print(f"sending {len(lines)} words, about {total} tokens")
body = {
    "model": "deepseek-v4.1-flash",
    "messages": [
        {
            "role": "user",
            "content": "".join(lines)
            + '\nWhat word comes right after and before "'
            + lines[69000].strip()
            + '" in the list above?',
        }
    ],
}
req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
)
print(json.load(urllib.request.urlopen(req, timeout=3600))["choices"][0]["message"]["content"])
