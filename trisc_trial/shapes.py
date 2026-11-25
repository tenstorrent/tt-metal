"""Classify every distinct build.h statement of a build, old key vs new key."""
import ast, collections, json, re, sys
from pathlib import Path

old_py, new_py, build, out = sys.argv[1:5]
OLD = r"\s*(?:(?:static|inline)\s+)*constexpr\b[^=;{]*?\b(\w+)\s*[={]"


def load_declared(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TestConfig")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_declared_name")
    fn.decorator_list = []
    ns = {"re": re}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["_declared_name"]


declared = load_declared(new_py)


def statements(text, new):
    out, statement, depth = [], "", 0
    if new:
        text = text.replace("\\\n", " ")
    for line in text.splitlines():
        if (line.lstrip() if new else line).startswith("#"):
            out.append(("#", line))
            continue
        statement += line + "\n"
        depth += line.count("{") - line.count("}")
        if depth == 0 and line.rstrip().endswith(";"):
            out.append(("s", statement))
            statement = ""
    return out


headers = {p.read_text() for p in Path(build).rglob("build.h")}
seen, stats, lost = set(), collections.Counter(), collections.Counter()
for h in headers:
    for kind, s in statements(h, True):
        if kind != "s" or s in seen:
            continue
        seen.add(s)
        o = re.match(OLD, s)
        n = declared(s)
        stats[f"old {'named' if o else 'unnamed'} / new {'named' if n else 'unnamed'}"] += 1
        if o and not n:
            lost[re.sub(r"\b\d+u?\b", "N", s.strip())[:200]] += 1
        if o and n and o.group(1) != n:
            stats["named differently"] += 1
res = dict(build_h_files=len(headers), distinct_statements=len(seen), stats=dict(stats), old_named_new_unnamed=lost.most_common(60))
json.dump(res, open(f"{out}/shapes.json", "w"), indent=1)
import tarfile, io
with tarfile.open(f"{out}/build_h.tar.gz", "w:gz") as tar:
    for i, h in enumerate(sorted(headers)):
        data = h.encode()
        info = tarfile.TarInfo(f"build_h/{i}.h"); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
print(json.dumps({k: v for k, v in res.items() if k != "old_named_new_unnamed"}))
