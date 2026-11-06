#!/usr/bin/env python3
"""hunks.py: every diff hunk between STUDY (worktree + v4txa11 overlay) and PORT (22cdb1e55c3) over the differing files of
the router's include closure, with an automatic first-pass origin:
  port side: git blame on 22cdb1e55c3 -> the commits that wrote the hunk's port lines (port commits d2bd12fcf1f..22cdb1e55c3
             by sha and subject; others = main, with the PR number from the subject)
  study side: whether the hunk's study lines exist in main at the study base 6d8a1032acc (main's old code) or not (study)
and a 'comment-only' flag (every changed line on both sides blank or a // or /* comment line).
Writes hunks.json and hunks_code.txt (the non-comment hunks, with both sides' text) beside it."""
import json, os, re, subprocess, sys

TREE = "/home/aagarwal/tt-metal"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TREE + "/fabric_exp/consolidation")
from offline_build import resolve  # noqa: E402

PORT, MAIN_P, MAIN_S = "22cdb1e55c3", "d2bd12fcf1f", "6d8a1032acc"
K_S = "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp"
K_P = "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp"
over = {dst: src for src, dst in resolve(TREE + "/fabric_exp/variants/v4txa11")}
port_commits = {}
for l in subprocess.run(["git", "-C", TREE, "log", "--format=%H %s", f"{MAIN_P}..{PORT}"], capture_output=True,
                        text=True).stdout.splitlines():
    h, s = l.split(" ", 1)
    port_commits[h] = s


def show(rev, path):
    r = subprocess.run(["git", "-C", TREE, "show", f"{rev}:{path}"], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None


def blame(path):
    out = subprocess.run(["git", "-C", TREE, "blame", "--line-porcelain", PORT, "--", path], capture_output=True,
                         text=True, errors="replace").stdout
    res, cur, summ = [], None, {}
    for l in out.splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", l)
        if m:
            cur = m.group(1)
        elif l.startswith("summary "):
            summ[cur] = l[8:]
        elif l.startswith("\t"):
            res.append(cur)
    return res, summ


def is_comment(line):
    t = line.strip()
    return t == "" or t.startswith("//") or t.startswith("/*") or t.startswith("*") or t.startswith("*/")


files = [l.split()[-1] for l in open(os.path.join(HERE, "differ_files.txt")) if l.strip()]
out, code = [], []
for f in files:
    sp = over.get(f, os.path.join(TREE, f))
    pf = K_P if f == K_S else f
    s_txt = open(sp, errors="replace").read()
    p_txt = show(PORT, pf)
    m0 = show(MAIN_S, pf if f == K_S else f) or ""
    m0_lines = set(l.strip() for l in m0.splitlines())
    open("/tmp/_s", "w").write(s_txt)
    open("/tmp/_p", "w").write(p_txt)
    d = subprocess.run(["diff", "-U0", "/tmp/_s", "/tmp/_p"], capture_output=True, text=True).stdout
    bl, summ = blame(pf)
    for h in re.split(r"(?m)^(?=@@ )", d):
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", h)
        if not m:
            continue
        s0, sn = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        p0, pn = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
        body = h.splitlines()[1:]
        sl = [l[1:] for l in body if l.startswith("-")]
        pl = [l[1:] for l in body if l.startswith("+")]
        commits = sorted({bl[i - 1] for i in range(p0, p0 + pn)} if pn else set())
        port_sh = [c[:11] for c in commits if c in port_commits]
        main_c = [c for c in commits if c not in port_commits]
        main_pr = sorted({(re.findall(r"\(#(\d+)\)", summ.get(c, "")) or [c[:11]])[0] for c in main_c})
        study_new = [l for l in sl if l.strip() and l.strip() not in m0_lines]
        comment = all(is_comment(l) for l in sl + pl)
        rec = {"file": f, "s": [s0, sn], "p": [p0, pn], "port_commits": port_sh, "main": main_pr,
               "study_lines_not_in_main": len(study_new), "comment_only": comment, "ns": len(sl), "np": len(pl)}
        out.append(rec)
        if not comment:
            code.append((rec, sl, pl))
json.dump(out, open(os.path.join(HERE, "hunks.json"), "w"), indent=1)
with open(os.path.join(HERE, "hunks_code.txt"), "w") as fo:
    for rec, sl, pl in code:
        fo.write(f"### {rec['file']} S:{rec['s'][0]}+{rec['s'][1]} P:{rec['p'][0]}+{rec['p'][1]} port={rec['port_commits']} "
                 f"main={rec['main']} study_new={rec['study_lines_not_in_main']}\n")
        fo.writelines(f"-S {l}\n" for l in sl)
        fo.writelines(f"+P {l}\n" for l in pl)
print(len(out), "hunks;", sum(r["comment_only"] for r in out), "comment-only;", len(code), "with code")
