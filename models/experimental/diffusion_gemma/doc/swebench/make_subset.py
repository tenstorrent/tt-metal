"""Pick a fixed random 100-instance subset of SWE-Bench Verified.

Deterministic: sorted instance ids, random.Random(0).sample(..., 100).
Writes the id list and a --filter regex so the run is reproducible.
"""

import json
import os
import random

os.environ.setdefault("HF_HOME", "/home/ttuser/zni/benchmarks/hfcache")
from datasets import load_dataset  # noqa: E402

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
ids = sorted(ds["instance_id"])
assert len(ids) == 500, len(ids)

subset = sorted(random.Random(0).sample(ids, 100))
repos = {}
for i in subset:
    repos[i.split("__")[0]] = repos.get(i.split("__")[0], 0) + 1

out = "/home/ttuser/zni/benchmarks/swebench_verified_subset100.json"
with open(out, "w") as f:
    json.dump({"seed": 0, "n": 100, "instance_ids": subset, "repo_counts": repos}, f, indent=1)

regex = "^(" + "|".join(i.replace(".", r"\.") + "$" for i in subset) + ")"
with open("/home/ttuser/zni/benchmarks/swebench_verified_subset100.regex", "w") as f:
    f.write(regex)

print("wrote", out, "n =", len(subset))
print("repo counts:", json.dumps(repos, indent=1))
print("regex chars:", len(regex))
