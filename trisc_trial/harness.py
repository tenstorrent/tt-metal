"""Property test of _compile_kernel_part_once: equal key (ELF reused) must imply equal object code.

usage: python3 harness.py <test_config.py> <trials> <seed> <workdir> [compiler] [form ...]
Each trial: random forms (random values per variant), one TRISC-like unit using a random subset of the
names. Variant 1 then variant 2 go through the real key code against one store; ground truth compiles
both variants with no reuse and compares the objects.
"""

import hashlib, json, random, shutil, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import forms
import load_once

PRELUDE = """#pragma once
namespace std { using uint32_t = unsigned; using uint8_t = unsigned char; }
template <class T, int N> struct Arr { T d[N]; };
"""


def names(i):
    return SimpleNamespace(**{k: f"{k}{i}" for k in "ABCEMNPST"})


def render(picks, values):
    pre, build, uses = [PRELUDE], [], []
    for i, (fname, v) in enumerate(zip(picks, values)):
        p, b, u = forms.F[fname][2](names(i), v)
        pre.append(p)
        build.append(b)
        uses.append(u)
    return "\n".join(pre) + "\n", "#pragma once\n" + "\n".join(build) + "\n", uses


def compile_cmd(cxx, wd, vdir, out):
    return [cxx, "-std=c++20", "-O2", "-g0", "-w", f"-I{wd}", f"-I{vdir}", "-c", "-x", "c++", "-", "-o", str(out)]


def obj(cxx, wd, vdir, source, out):
    r = subprocess.run(compile_cmd(cxx, wd, vdir, out), cwd=wd, input=source, capture_output=True, text=True)
    return hashlib.sha256(Path(out).read_bytes()).hexdigest() if r.returncode == 0 else "ERROR"


def trial(once, cxx, root, rng, pool):
    if root.exists():
        shutil.rmtree(root)
    wd, art = root / "wd", root / "art"
    wd.mkdir(parents=True)
    picks = [rng.choice(pool) for _ in range(rng.randint(1, 4))]
    v1 = [[rng.randint(0, 3) for _ in range(forms.F[f][0])] for f in picks]
    v2 = [[x if rng.random() < 0.5 else rng.randint(0, 3) for x in v] for v in v1]
    if v1 == v2:
        return None
    pre1, b1, uses = render(picks, v1)
    pre2, b2, _ = render(picks, v2)
    if pre1 != pre2:
        return None
    (wd / "prelude.h").write_text(pre1)
    chosen = [u for us in uses for u in us if rng.random() < 0.5]
    (wd / "unit.cpp").write_text('#include "prelude.h"\n#include "build.h"\n'
                                 f'extern "C" int use() {{ return 0 {"".join(" + (" + u + ")" for u in chosen)}; }}\n')
    source = '#include "unit.cpp"\n'
    truth = []
    for i, b in enumerate((b1, b2)):
        vdir = art / "t" / f"v{i}"
        (vdir / "elf").mkdir(parents=True)
        (vdir / "build.h").write_text(b)
        truth.append(obj(cxx, wd, vdir, source, root / f"truth{i}.o"))
    if "ERROR" == truth[0]:
        return None  # the first variant must build for anything to be stored
    load_once.FakeTestConfig.ARTEFACTS_DIR = art
    load_once.FakeTestConfig.TESTS_WORKING_DIR = wd
    got = []
    for i in range(2):
        vdir = art / "t" / f"v{i}"
        out = vdir / "elf" / "x.o"
        try:
            once(SimpleNamespace(test_name="t", variant_id=f"v{i}"), compile_cmd(cxx, wd, vdir, out), source)
            got.append(hashlib.sha256(out.read_bytes()).hexdigest())
        except RuntimeError:
            got.append("ERROR")
    reused = len(list((art / "trisc").glob("*.elf"))) == 1
    bad = got[1] != truth[1]
    return dict(picks=picks, v1=v1, v2=v2, uses=chosen, reused=reused, code_equal=truth[0] == truth[1],
                wrong=bad, b1=b1, b2=b2, prelude=pre1)


def main():
    tc, n, seed, work = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    cxx = sys.argv[5] if len(sys.argv) > 5 else "c++"
    pool = sys.argv[6:] or sorted(forms.F)
    once = load_once.load(tc)
    rng = random.Random(seed)
    stats, bad = {}, []
    for _ in range(n):
        r = trial(once, cxx, work, rng, pool)
        if r is None:
            continue
        for f in set(r["picks"]):
            s = stats.setdefault(f, dict(cls=forms.F[f][1], trials=0, wrong=0, reused=0, code_equal=0))
            s["trials"] += 1
            s["wrong"] += r["wrong"]
            s["reused"] += r["reused"]
            s["code_equal"] += r["code_equal"]
        if r["wrong"]:
            bad.append(r)
    print(json.dumps(dict(stats=stats, wrong=len(bad)), indent=1))
    Path(work.parent / f"counterexamples-{seed}.json").write_text(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
