import json, os, re, sys

root, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
paths = set()
quoted = re.compile(r'"((?:[^"\\]|\\.)*)"')
for line in sys.stdin if src == "-" else open(src):
    m = quoted.search(line)
    if m:
        paths.add(m.group(1))
keep = [
    ".github",
    "tt_metal/tt-llk",
    "tt_metal/hw",
    "tt_metal/hostdevcommon",
    "ttnn/cpp/ttnn/operations/experimental",
    "tt_metal/sfpi-info.sh",
    "tt_metal/sfpi-version",
]
cwds = [f"{root}/tt_metal/tt-llk/tests", f"{root}/tt_metal/tt-llk/tests/python_tests"]
outside, inside = {}, 0
for p in paths:
    cands = [p] if p.startswith("/") else [os.path.join(c, p) for c in cwds]
    for c in cands:
        a = os.path.normpath(c)
        if not a.startswith(root + "/") or not os.path.lexists(a):
            continue
        rel = os.path.relpath(a, root)
        if rel.startswith(".git/") or rel == ".git":
            continue
        parent_file = False
        if any(rel == k or rel.startswith(k + "/") for k in keep) or parent_file:
            inside += 1
        elif not any(k.startswith(rel + "/") for k in keep):
            outside[rel] = "dir" if os.path.isdir(a) else "file"
json.dump(
    {"accessed": len(paths), "inside": inside, "outside": dict(sorted(outside.items()))},
    open(out, "w"),
    indent=1,
)
print(json.dumps({"check": os.path.basename(out), "outside": len(outside)}))
