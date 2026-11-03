#!/usr/bin/env python3
"""Second-LB harness (LB2_HARNESS.md): tree snapshots as zstd tarballs on /data (NFS creates small files slowly, so
the stage holds one archive per scope, not a file tree). Read-only on the tree it packs.

  lb2_pack.py pack TREE OUT.tar.zst REF.sha256 [--backup BK] [--toplevel LIST] DIR...
      every regular file and symlink under TREE/DIR (plus the TREE-relative paths in LIST), with the stage excludes
      (fabric_exp/results, fabric_exp/consolidation/builds, fabric_exp/variants/*/elf, __pycache__, *.pyc). Each file
      is read once: the bytes archived are the bytes hashed into REF (`sha256sum -c` format, regular files only).
      --backup BK (swap.py's backup dir): the unswapped tree; a path in the active swap's "saved" list is archived
      from the backup, a path in its "created" list is left out.
  lb2_pack.py check OUT.tar.zst REF.sha256
      read the archive back and compare every member's hash with REF (catches a bad write to /data)
  lb2_pack.py live TREE REF.sha256 [--backup BK] [--toplevel LIST] DIR...
      hash the tree again (same view) and compare with REF (catches a file that changed after the pack); only input
      files count (is_input: not fabric_exp notes, logs or data tables, which agents write while a stage is taken)
  lb2_pack.py gitsum GITDIR WORKTREE OUT.json [--exclude PATH ...] DIR...
      git status lines and per-file diff hashes (submodules ignored, fabric_exp/results left out)
  lb2_pack.py gitcmp LIVE.json NODE.json
      compare two gitsum outputs; exit 1 only when an input file differs
"""
import argparse
import fnmatch
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile

EXCLUDE_DIRS = ("fabric_exp/results", "fabric_exp/consolidation/builds")


def excluded(rel, is_dir):
    if any(rel == d or rel.startswith(d + "/") for d in EXCLUDE_DIRS):
        return True
    base = os.path.basename(rel)
    if is_dir:
        return base == "__pycache__" or fnmatch.fnmatch(rel, "fabric_exp/variants/*/elf")
    return base.endswith(".pyc")


NON_INPUT = (".md", ".txt", ".html", ".png", ".svg", ".log", ".csv", ".bak", ".out", ".json.done")


def is_input(rel):
    """Could a job read this file? Everything outside fabric_exp (sources, tests, top level) and every fabric_exp file
    but notes, logs and data tables (an agent writing its notes or loopvars_maps while the stage is taken)."""
    if rel.startswith("fabric_exp/variants/"):
        return True
    if rel.startswith(("fabric_exp/report/data/", "fabric_exp/report/page/")):
        return False  # the report page bundle (agents rebuild it); report/sweep yamls stay inputs
    return not rel.startswith("fabric_exp/") or not rel.endswith(NON_INPUT)


def swap_lists(backup):
    if backup and os.path.exists(os.path.join(backup, "manifest.json")):
        m = json.load(open(os.path.join(backup, "manifest.json")))
        return set(m["saved"]), set(m["created"])
    return set(), set()


def walk(tree, dirs, toplevel):
    """(rel, is_dir) for every entry in scope, parents first."""
    out = []
    if toplevel:
        out += [(l.strip(), False) for l in open(toplevel) if l.strip()]
    for d in dirs:
        out.append((d, True))
        for root, ds, fs in os.walk(os.path.join(tree, d)):
            rr = os.path.relpath(root, tree)
            keep = []
            for x in sorted(ds):
                rel = os.path.join(rr, x)
                if excluded(rel, True):
                    continue
                if os.path.islink(os.path.join(tree, rel)):
                    out.append((rel, False))  # a symlink to a directory: archived as the link
                else:
                    keep.append(x)
                    out.append((rel, True))
            ds[:] = keep
            out += [(os.path.join(rr, f), False) for f in sorted(fs) if not excluded(os.path.join(rr, f), False)]
    return out


def hashes(tree, a):
    saved, created = swap_lists(a.backup)
    for rel, is_dir in walk(tree, a.dirs, a.toplevel):
        p = os.path.join(tree, rel)
        if is_dir or rel in created or os.path.islink(p) or not os.path.isfile(p):
            continue
        src = os.path.join(a.backup, rel) if rel in saved else p
        with open(src, "rb") as f:
            yield rel, hashlib.sha256(f.read()).hexdigest()


def load_ref(path):
    ref = {}
    for l in open(path):
        h, rel = l.rstrip("\n").split("  ", 1)
        ref[rel] = h
    return ref


def pack(a):
    saved, created = swap_lists(a.backup)
    z = subprocess.Popen(["zstd", "-q", "-T0", "-3", "-f", "-o", a.out + ".tmp"], stdin=subprocess.PIPE)
    n = nbytes = nlinks = 0
    with tarfile.open(fileobj=z.stdin, mode="w|", format=tarfile.PAX_FORMAT) as t, open(a.ref + ".tmp", "w") as r:
        for rel, is_dir in walk(a.tree, a.dirs, a.toplevel):
            p = os.path.join(a.tree, rel)
            if rel in created:
                continue
            if is_dir or os.path.islink(p):
                t.add(p, arcname=rel, recursive=False)
                nlinks += os.path.islink(p)
                continue
            if not os.path.isfile(p):
                continue
            src = os.path.join(a.backup, rel) if rel in saved else p
            with open(src, "rb") as f:
                data = f.read()
            ti = t.gettarinfo(src, arcname=rel)
            ti.type, ti.linkname, ti.size = tarfile.REGTYPE, "", len(data)  # hard links archived as plain copies
            t.addfile(ti, io.BytesIO(data))
            r.write(f"{hashlib.sha256(data).hexdigest()}  {rel}\n")
            n += 1
            nbytes += len(data)
    z.stdin.close()
    if z.wait() != 0:
        sys.exit("!!! zstd failed")
    os.replace(a.out + ".tmp", a.out)
    os.replace(a.ref + ".tmp", a.ref)
    print(f"   pack {os.path.basename(a.out)}: {n} files ({nbytes / 1e6:.1f} MB, {len(saved & set(load_ref(a.ref)))} "
          f"from the swap backup, {len(created)} swap-created left out), {nlinks} symlinks; "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB written")


def check(a):
    ref = load_ref(a.ref)
    z = subprocess.Popen(["zstd", "-q", "-dc", a.out], stdout=subprocess.PIPE)
    seen, bad = 0, []
    with tarfile.open(fileobj=z.stdout, mode="r|") as t:
        for m in t:
            if not m.isfile():
                continue
            h = hashlib.sha256(t.extractfile(m).read()).hexdigest()
            seen += 1
            if ref.get(m.name) != h:
                bad.append(m.name)
    z.stdout.read()  # drain, so zstd never blocks on a full pipe
    z.wait()
    if bad or seen != len(ref):
        print(f"!!! {a.out}: {len(bad)} members differ from the reference, {seen} read of {len(ref)}: {bad[:10]}")
        return 1
    print(f"   check {os.path.basename(a.out)}: {seen} members = reference")
    return 0


def live(a):
    ref = load_ref(a.ref)
    now = dict(hashes(a.tree, a))
    diff = sorted(k for k in set(ref) | set(now) if ref.get(k) != now.get(k))
    inp = [k for k in diff if is_input(k)]
    if inp:
        print(f"!!! live tree differs from {os.path.basename(a.ref)} in {len(inp)} input files: {inp[:10]}")
        return 1
    note = f"; {len(diff)} non-input files changed meanwhile (notes/logs/tables): {diff[:5]}" if diff else ""
    print(f"   live tree = {os.path.basename(a.ref)} ({len(ref)} files){note}")
    return 0


def gitsum(a):
    """git status lines + per-file diff hashes of WORKTREE against GITDIR (submodules ignored, results and the
    pathspecs given excluded), as JSON."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    g = ["git", "--git-dir", a.gitdir, "--work-tree", a.tree]
    spec = ["--"] + a.dirs + [":(exclude)fabric_exp/results"] + [":(exclude)" + p for p in (a.exclude or [])]
    st = subprocess.run(g + ["status", "--porcelain=v1", "-uall", "--ignore-submodules=all"] + spec, env=env,
                        capture_output=True, text=True, check=True).stdout.splitlines()
    df = subprocess.run(g + ["diff", "--ignore-submodules=all"] + spec, env=env, capture_output=True, check=True).stdout
    per, cur = {}, None
    for chunk in df.split(b"\ndiff --git ") if df else []:
        head = chunk.split(b"\n", 1)[0].decode(errors="replace")
        name = head.split(" b/", 1)[-1] if " b/" in head else head
        per[name] = hashlib.sha256(chunk.removeprefix(b"diff --git ")).hexdigest()[:16]
    json.dump({"status": sorted(st), "diff": per}, open(a.out, "w"), indent=0)
    print(f"   git: {len(st)} status lines, {len(per)} files in the diff -> {os.path.basename(a.out)}")
    return 0


def gitcmp(a):
    r, n = json.load(open(a.ref)), json.load(open(a.out))
    st = sorted(set(r["status"]) ^ set(n["status"]))
    df = sorted(k for k in set(r["diff"]) | set(n["diff"]) if r["diff"].get(k) != n["diff"].get(k))
    paths = [l[3:] for l in st] + df
    inp = [p for p in paths if is_input(p)]
    print(f"   git status: {len(n['status'])} lines, {len(st)} differ from the live tree's; diff: {len(n['diff'])} files, "
          f"{len(df)} differ" + (f" (non-input only: {paths[:5]})" if paths and not inp else ""))
    if inp:
        print(f"!!! git status/diff differ from the live tree's in input files: {inp[:10]}")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "check", "live", "gitsum", "gitcmp"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--backup")
    ap.add_argument("--toplevel")
    ap.add_argument("--exclude", action="append", help="gitsum: a path to leave out (repeatable)")
    a = ap.parse_intermixed_args()
    if a.cmd == "pack":
        a.tree, a.out, a.ref, a.dirs = a.args[0], a.args[1], a.args[2], a.args[3:]
        return pack(a)
    if a.cmd == "gitsum":  # gitsum GITDIR WORKTREE OUT.json DIR...
        a.gitdir, a.tree, a.out, a.dirs = a.args[0], a.args[1], a.args[2], a.args[3:]
        return gitsum(a)
    if a.cmd == "gitcmp":  # gitcmp LIVE.json NODE.json
        a.ref, a.out = a.args
        return gitcmp(a)
    if a.cmd == "check":
        a.out, a.ref = a.args
        return check(a)
    a.tree, a.ref, a.dirs = a.args[0], a.args[1], a.args[2:]
    return live(a)


if __name__ == "__main__":
    sys.exit(main())
