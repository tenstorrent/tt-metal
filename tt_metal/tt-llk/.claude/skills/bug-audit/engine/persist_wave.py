#!/usr/bin/env python3
"""Persist a finished wave. This is the step that must NOT be skipped: until it runs, the wave's hunt results and
verdicts exist only in the workflow's return value.

  persist_wave.py [--run DIR] <raw_output.json>

<raw_output.json> is the workflow task's output file (copy it into raw_wave_outputs/ first). For each batch it writes
  findings/<batch>.json   the hunter's candidates and read log
  verdicts/<batch>.json   every candidate with its status (confirmed / uncertain / refuted / needs_recheck) and votes
  done/<batch>.done       only when every assigned file passed the read check below
The read check compares each file's reported line count and last non-blank line against the pinned tree. The
contract-trace ledger is checked too: every boundary's "other side" must be a real file:line (or file:symbol, or a
document outside the tree). A file that fails the read check (or was skipped) is recorded in reread.json, and its
batch stays un-done so the next wave re-hunts it; an invalid boundary is set aside and reported, not re-hunted.
Persisting the same output twice is refused, so a rerun after a crash cannot duplicate verdicts.
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spawn  # noqa: E402

from common import load, manifest, run_dir, save, state  # noqa: E402

out = run_dir()
if len(sys.argv) != 2:
    sys.exit(__doc__)
st = state(out)
man = manifest(out)
raw_sha = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
prior = next((w for w in st["waves"] if w.get("sha") == raw_sha), None)
if prior:
    print(f"already persisted as wave {prior['wave']}; nothing done")
    sys.exit(0)
raw = load(sys.argv[1])
r = raw.get("result", raw) if isinstance(raw, dict) else raw
if isinstance(r, str):
    r = json.loads(r)


def last_nonblank(path):
    try:
        with open(path, errors="replace", newline="") as fh:
            text = fh.read()
    except OSError:
        return None, None
    # count "\n" lines, as the hunter does: splitlines() also breaks on form feeds, U+2028 and the like, and a file
    # holding them would fail the read check on every hunt and be re-hunted forever
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    tail = next((ln for ln in reversed(lines) if ln.strip()), "")
    return len(lines), tail.strip()


_suffix_cache = {}


def ledger_site_ok(tree, site):
    """True if a ledger "other side" names a real place: path:line (or :a-b, :a,b-c) with the first line inside the
    file, or path:symbol where the symbol occurs in the file. The path is repo-relative or a unique suffix of a tracked
    file; an absolute path outside the tree (the ISA spec, other docs) is an external reference and needs no line.
    """
    first = site.split(" and ")[0].split(";")[0].strip()
    m = re.match(
        r"^(?P<path>[^:\s]+)(?::(?:(?P<line>\d+)|(?P<sym>[A-Za-z_][\w:~]*)))?", first
    )
    if not m:
        return False
    path, line = m.group("path"), m.group("line")
    full = os.path.join(tree, path)
    if not os.path.isfile(full):
        if tree not in _suffix_cache:
            r = spawn.run(
                "git", ["-C", tree, "ls-files"], capture_output=True, text=True
            )
            _suffix_cache[tree] = r.stdout.splitlines()
        hits = [f for f in _suffix_cache[tree] if f == path or f.endswith("/" + path)]
        if len(hits) != 1:
            return False
        full = os.path.join(tree, hits[0])
    if os.path.isabs(path) and not os.path.abspath(path).startswith(
        os.path.abspath(tree) + os.sep
    ):
        return True  # an external reference (the ISA spec, a doc outside the tree): a real place, line optional
    if line is None and m.group("sym"):
        try:  # path:function -- as checkable as a line number, if the name really is in that file
            return m.group("sym").split("::")[-1] in open(full, errors="replace").read()
        except OSError:
            return False
    if line is None:
        return False  # an in-tree boundary names the LINE (or symbol) the hunter read, not just the file
    n, _ = last_nonblank(full)
    return n is not None and 1 <= int(line) <= n


reread = load(os.path.join(out, "reread.json"), {})
wave_no = len(st["waves"]) + 1
stats = {
    "batches": 0,
    "done": 0,
    "reread_files": 0,
    "confirmed": 0,
    "uncertain": 0,
    "refuted": 0,
    "needs_recheck": 0,
}
for res in r.get("results", []):
    b = res["batch"]
    if b not in man:
        print(f"  !! {b} not in manifest — skipped")
        continue
    stats["batches"] += 1
    hunt = res.get("hunt")
    if not hunt:
        print(f"  !! {b}: hunter returned nothing (agent died?) — stays pending")
        continue
    fpath = os.path.join(out, "findings", f"{b}.json")
    if os.path.exists(fpath):  # a second pass: keep every earlier hunt's raw output
        with open(os.path.join(out, "findings", f"{b}.history.jsonl"), "a") as fh:
            fh.write(json.dumps(load(fpath)) + "\n")
    save(fpath, {**hunt, "wave": wave_no})
    reads = {e["path"]: e for e in hunt.get("files_read", [])}
    bad = []
    for f in man[b]["files"]:
        e = reads.get(f)
        n, tail = last_nonblank(os.path.join(man[b].get("root", st["root"]), f))
        if e is None:
            bad.append((f, "not reported as read"))
        elif n is None:
            bad.append((f, "file missing from tree"))
        elif abs(e["lines"] - n) > 1 or (tail and e["last_line"].strip() != tail):
            bad.append(
                (
                    f,
                    f"read check failed: reported {e['lines']} lines / last {e['last_line'][:40]!r}, "
                    f"tree has {n} / {tail[:40]!r}",
                )
            )
    # contract-trace ledger: every "other side" must be a real file:line in the tree
    tree = man[b].get("root", st["root"])
    # An invalid boundary is set aside and reported, not trusted -- and it does not re-hunt the batch: file coverage
    # is proven by the read check above, and a re-hunt costs a full hunt for one bad line of evidence.
    invalid = []
    for e in hunt.get("boundaries", []):
        stats["ledger_entries"] = stats.get("ledger_entries", 0) + 1
        if not ledger_site_ok(tree, e.get("other_side", "")):
            stats["ledger_bad"] = stats.get("ledger_bad", 0) + 1
            invalid.append(e)
    if invalid:
        hunt["ledger_invalid"] = invalid
        save(fpath, {**hunt, "wave": wave_no})
    stats["ledger_skipped"] = stats.get("ledger_skipped", 0) + len(
        hunt.get("boundaries_skipped", [])
    )
    ta = hunt.get("trace_audit") or {}
    if ta.get("died"):
        stats["trace_died"] = stats.get("trace_died", 0) + 1
        print(
            f"  !! {b}: the trace auditor died twice; its {ta.get('sampled')} sampled boundaries were not re-traced"
        )
    stats["trace_rechecked"] = stats.get("trace_rechecked", 0) + len(
        ta.get("rechecked", [])
    )
    stats["trace_overturned"] = stats.get("trace_overturned", 0) + sum(
        1 for x in ta.get("rechecked", []) if not x.get("agrees")
    )
    judged = res.get("judged", [])
    for f in judged:
        stats[f["status"]] = stats.get(f["status"], 0) + 1
    if judged:
        # merge with earlier passes rather than overwrite; consolidate.py dedupes by file:line
        vpath = os.path.join(out, "verdicts", f"{b}.json")
        prev = load(vpath, {}).get("findings", []) if os.path.exists(vpath) else []
        for f in judged:
            f.setdefault("wave", wave_no)
        save(vpath, {"batch": b, "wave": wave_no, "findings": prev + judged})
    if bad:
        reread[b] = [{"file": f, "why": why} for f, why in bad]
        stats["reread_files"] += len(bad)
        print(
            f"  RE-HUNT {b}: {len(bad)} file(s) failed the read check, e.g. {bad[0][0]}: {bad[0][1]}"
        )
    else:
        reread.pop(b, None)
        with open(os.path.join(out, "done", f"{b}.done"), "w") as fh:
            fh.write(f"{len(man[b]['files'])}\n")
        stats["done"] += 1
save(os.path.join(out, "reread.json"), reread)
# keep the per-file ledger in step with the markers: audited once the batch is done, with its candidate count
lp = os.path.join(out, "ledger.tsv")
if os.path.exists(lp):
    import csv

    done_now = {
        f[:-5] for f in os.listdir(os.path.join(out, "done")) if f.endswith(".done")
    }
    with open(lp) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    per_file = {}
    for vf in os.listdir(os.path.join(out, "verdicts")):
        if vf.endswith(".json"):
            for f in load(os.path.join(out, "verdicts", vf), {}).get("findings", []):
                if f.get("status") == "confirmed":
                    per_file.setdefault(f["file"], set()).add(
                        f["line"]
                    )  # unique lines: passes merge duplicates
    for row in rows:
        if row["batch"] in done_now:
            row["status"] = "audited"
            row["findings"] = str(len(per_file.get(row["file"], ())))
        elif row["batch"] in reread:
            row["status"] = "reread"
    if rows:
        with open(lp + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        os.replace(lp + ".tmp", lp)
ids = set(r.get("batches", []))
st["in_flight"] = [b for b in st.get("in_flight", []) if b not in ids]
st["waves"].append(
    {"wave": wave_no, "raw": os.path.relpath(sys.argv[1], out), "sha": raw_sha, **stats}
)
save(os.path.join(out, "state.json"), st)
print(f"wave {wave_no}: " + ", ".join(f"{k} {v}" for k, v in stats.items()))
if stats.get("ledger_entries"):
    print(
        f"contract-trace ledger: {stats['ledger_entries']} boundaries traced, {stats.get('ledger_skipped', 0)} skipped, "
        f"{stats.get('ledger_bad', 0)} cite an other side that cannot be located (fabricated, an ambiguous short path, or an out-of-tree file not given as an absolute path); "
        f"trace audit overturned {stats.get('trace_overturned', 0)} of {stats.get('trace_rechecked', 0)} re-checked verdicts"
    )
print(
    "next: python3 consolidate.py && python3 status.py   (and snapshot the run directory)"
)
