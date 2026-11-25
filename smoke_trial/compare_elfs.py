import hashlib, json, os, subprocess, sys, tempfile

objcopy, temp, out = sys.argv[1], sys.argv[2], sys.argv[3]
builds = ["build-main", "build-main-sparse", "build-pr", "build-nodedup"]


def elfs(b):
    r = os.path.join(temp, b)
    return {
        os.path.relpath(os.path.join(d, f), r)
        for d, _, fs in os.walk(r)
        if os.path.basename(d) == "elf"
        for f in fs
        if f.endswith(".elf")
    }


def digest(path, strip=False):
    if strip:
        with tempfile.NamedTemporaryFile() as t:
            subprocess.run([objcopy, "--strip-debug", path, t.name], check=True)
            return hashlib.sha256(open(t.name, "rb").read()).hexdigest()
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


sets = {b: elfs(b) for b in builds}
res = {"elf_count": {b: len(s) for b, s in sets.items()}}
res["same_file_set"] = len({frozenset(s) for s in sets.values()}) == 1
common = sorted(set.intersection(*sets.values()))
pairs = [
    ("build-main", "build-main-sparse", False),
    ("build-pr", "build-nodedup", False),
    ("build-main", "build-pr", False),
    ("build-main", "build-pr", True),
    ("build-pr", "build-nodedup", True),
    ("build-main", "build-nodedup", True),
]
for a, b, strip in pairs:
    diff = [e for e in common if digest(os.path.join(temp, a, e), strip) != digest(os.path.join(temp, b, e), strip)]
    res[f"{a} vs {b}{' (strip-debug)' if strip else ''}"] = {
        "compared": len(common),
        "different": len(diff),
        "examples": diff[:5],
        "files": diff,
        "outside_fused_tests": [e for e in diff if "/fused_tests/" not in e],
    }
readelf = objcopy.replace("objcopy", "readelf")
for a, b in [("build-pr", "build-nodedup")]:
    ex = res[f"{a} vs {b}"]["examples"]
    if ex:
        dumps = [
            subprocess.run(
                [readelf, "-wi", "-S", os.path.join(temp, x, ex[0])], capture_output=True, text=True
            ).stdout.splitlines()
            for x in (a, b)
        ]
        import difflib

        res["debug_diff_example"] = list(difflib.unified_diff(*dumps, lineterm="", n=0))[:80]
store = os.path.join(temp, "build-pr", "trisc")
res["pr_unique_trisc_elfs"] = len(os.listdir(store)) if os.path.isdir(store) else 0
res["unique_by_content"] = {b: len({digest(os.path.join(temp, b, e)) for e in sets[b]}) for b in builds}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
