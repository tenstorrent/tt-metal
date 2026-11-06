#!/usr/bin/env python3
"""mk_pr_yamls.py: the 4096 B suites of the PR perf refresh (user, 2026-11-05: add 4096 B, the one size missing).
4096 B versions of the main bandwidth suites, with each source suite's rows, settings and packet counts unchanged:
  pr_2d_4096.yaml    exp_2d_4352.yaml (2D mesh), size 4352 -> 4096
  pr_1d_4096.yaml    exp_1d_4352.yaml (1D: lines, the half-ring row, NeighborExchange), size 4352 -> 4096
  pr_ring_4096.yaml  v4/exp_ring.yaml (ring, two ERISCs), size 4352 -> 4096
and the suites that sweep sizes get 4096 added (1024, 4096, 4352):
  pr_ring_se.yaml    v4/exp_ring_se.yaml (ring, one ERISC)
  pr_mux.yaml        v4/exp_mux.yaml (Tensix mux)
Text edits only (comments and layout kept); each output is checked to parse and to hold the expected sizes."""
import os, re, yaml

FE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def swap(src, dst, note):
    s = open(os.path.join(FE, src)).read()
    n0 = len(re.findall(r"\b4352\b", s))
    s = re.sub(r"(size: \[)4352(\])", r"\g<1>4096\2", s)
    s = re.sub(r"(\n(\s*)size:\n\2- )4352\n", r"\g<1>4096\n", s)
    assert "4352" not in re.sub(r"#.*", "", s), (src, "4352 left")
    write(dst, f"# {note}\n" + s, {4096})
    return n0


def add4096(src, dst, note):
    s = open(os.path.join(FE, src)).read()
    s, n = re.subn(r"(\n(\s*)size:\n\2- 1024\n)(\2- 4352\n)", r"\1\g<2>- 4096\n\3", s)
    assert n > 0, (src, "no 1024/4352 sweep")
    write(dst, f"# {note}\n" + s, {1024, 4096, 4352})


def write(dst, text, want):
    open(os.path.join(HERE, dst), "w").write(text)
    d = yaml.safe_load(text)
    sizes = set()
    for t in d["Tests"]:
        pp = t.get("parametrization_params", {})
        if "size" in pp:
            sizes |= set(pp["size"])
        if "size" in t.get("defaults", {}):
            sizes.add(t["defaults"]["size"])
    assert sizes == want, (dst, sizes)
    print(f"{dst}: {len(d['Tests'])} tests, sizes {sorted(sizes)}")


swap("exp_2d_4352.yaml", "pr_2d_4096.yaml", "PR perf refresh (mk_pr_yamls.py): exp_2d_4352.yaml at 4096 B.")
swap("exp_1d_4352.yaml", "pr_1d_4096.yaml", "PR perf refresh (mk_pr_yamls.py): exp_1d_4352.yaml at 4096 B.")
swap("v4/exp_ring.yaml", "pr_ring_4096.yaml", "PR perf refresh (mk_pr_yamls.py): v4/exp_ring.yaml at 4096 B.")
add4096("v4/exp_ring_se.yaml", "pr_ring_se.yaml", "PR perf refresh (mk_pr_yamls.py): v4/exp_ring_se.yaml with 4096 B added.")
add4096("v4/exp_mux.yaml", "pr_mux.yaml", "PR perf refresh (mk_pr_yamls.py): v4/exp_mux.yaml with 4096 B added.")
