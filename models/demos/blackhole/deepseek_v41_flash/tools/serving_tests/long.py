import json
import random
import time
import urllib.request

random.seed(1)
words = "alpha beta gamma delta river stone cloud forest engine window garden bridge market".split()
n = 20000
text = (
    " ".join(random.choice(words) for _ in range(n))
    + "\n\nThe secret code is 4817. Repeat the secret code mentioned above."
)
body = dict(
    model="deepseek-ai/DeepSeek-V4.1-Flash", messages=[{"role": "user", "content": text}], max_tokens=24, temperature=0
)
t = time.time()
d = json.load(
    urllib.request.urlopen(
        urllib.request.Request(
            "http://localhost:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        ),
        timeout=1800,
    )
)
print(d["usage"], repr(d["choices"][0]["message"]["content"]), "%.1fs" % (time.time() - t))
