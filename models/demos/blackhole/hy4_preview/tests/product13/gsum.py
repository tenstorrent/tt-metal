import ast
import sys

for f in sys.argv[1:]:
    rows = []
    for l in open(f):
        if l.startswith("{'i'"):
            d = ast.literal_eval(l.strip())
            rows.append((d["ntok"], d["finish"], d["cjk"], d["boxed"], d["loop"]))
    print(
        f,
        "n",
        len(rows),
        "eos",
        sum(r[1] == "stop" for r in rows),
        "len",
        sum(r[1] == "length" for r in rows),
        "cjk",
        sum(r[2] > 0 for r in rows),
        "loop",
        sum(r[4] for r in rows),
        [r[0] for r in rows],
    )
