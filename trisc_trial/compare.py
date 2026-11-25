import hashlib, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

objcopy, temp, out = sys.argv[1], sys.argv[2], sys.argv[3]
builds = ["build-plain", "build-old", "build-new"]


def elfs(b):
    r = os.path.join(temp, b)
    return {
        os.path.relpath(os.path.join(d, f), r)
        for d, _, fs in os.walk(r)
        if os.path.basename(d) == "elf"
        for f in fs
        if f.endswith(".elf")
    }


def digest(path):
    with tempfile.NamedTemporaryFile() as t:
        subprocess.run([objcopy, "--strip-debug", path, t.name], check=True)
        return hashlib.sha256(open(t.name, "rb").read()).hexdigest()


sets = {b: elfs(b) for b in builds}
res = {"elf_count": {b: len(s) for b, s in sets.items()}, "same_file_set": len({frozenset(s) for s in sets.values()}) == 1}
common = sorted(set.intersection(*sets.values()))
with ThreadPoolExecutor(os.cpu_count()) as pool:
    hashes = {b: dict(zip(common, pool.map(digest, [os.path.join(temp, b, e) for e in common]))) for b in builds}
for a, b in [("build-plain", "build-old"), ("build-plain", "build-new")]:
    diff = [e for e in common if hashes[a][e] != hashes[b][e]]
    res[f"{a} vs {b} (strip-debug)"] = {"compared": len(common), "different": len(diff), "examples": diff[:10]}
for b in ("build-old", "build-new"):
    store = os.path.join(temp, b, "trisc")
    files = os.listdir(store) if os.path.isdir(store) else []
    res[f"{b} store"] = {
        "elfs": sum(f.endswith(".elf") for f in files),
        "names_files": sum(f.endswith(".names") for f in files),
        "names_bytes": sum(os.path.getsize(os.path.join(store, f)) for f in files if f.endswith(".names")),
    }
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
