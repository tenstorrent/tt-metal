#!/usr/bin/env python3
"""Persist a finished wave. This is the step that must NOT be skipped: until it runs, the wave's hunt results and
verdicts exist only in the workflow's return value.

  persist_wave.py [--run DIR] <raw_output.json>

<raw_output.json> is the workflow task's output file (copy it into raw_wave_outputs/ first). For each batch it writes
  findings/<batch>.json   the hunter's candidates and read log
  verdicts/<batch>.json   every candidate with its status (confirmed / uncertain / refuted / needs_recheck) and votes
  done/<batch>.done       only when every assigned file passed the read check below
The read check compares each file's reported line count and last non-blank line against the pinned tree. The
contract-trace ledger is checked the same way: every boundary's "other side" must be a real file:line. A file
that fails it (or was skipped) is recorded in reread.json, and its batch stays un-done so the next wave re-hunts it.
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, manifest, run_dir, save, state  # noqa: E402

out = run_dir()
if len(sys.argv) != 2:
    sys.exit(__doc__)
st = state(out)
man = manifest(out)
raw = load(sys.argv[1])
r = raw.get("result", raw) if isinstance(raw, dict) else raw
if isinstance(r, str):
    r = json.loads(r)


def last_nonblank(path):
    try:
        with open(path, errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, None
    tail = next((ln for ln in reversed(lines) if ln.strip()), "")
    return len(lines), tail.strip()


_suffix_cache = {}


def ledger_site_ok(tree, site):
    """True if a ledger "other side" names a real place: path[:line | :a-b | :a,b-c] where the path is repo-relative
    or a unique suffix of a tracked file, and the first line number is within the file.
    """
    first = site.split(" and ")[0].split(";")[0].strip()
    m = re.match(r"^(?P<path>[^:\s]+)(?::(?P<line>\d+))?", first)
    if not m:
        return False
    path, line = m.group("path"), m.group("line")
    full = os.path.join(tree, path)
    if not os.path.isfile(full):
        if tree not in _suffix_cache:
            r = subprocess.run(
                ["git", "-C", tree, "ls-files"], capture_output=True, text=True
            )
            _suffix_cache[tree] = r.stdout.splitlines()
        hits = [f for f in _suffix_cache[tree] if f == path or f.endswith("/" + path)]
        if len(hits) != 1:
            return False
        full = os.path.join(tree, hits[0])
    if line is None:
        return True
    n, _ = last_nonblank(full)
    return n is not None and int(line) <= n + 1


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
    for e in hunt.get("boundaries", []):
        stats["ledger_entries"] = stats.get("ledger_entries", 0) + 1
        if not ledger_site_ok(tree, e.get("other_side", "")):
            stats["ledger_bad"] = stats.get("ledger_bad", 0) + 1
    stats["ledger_skipped"] = stats.get("ledger_skipped", 0) + len(
        hunt.get("boundaries_skipped", [])
    )
    ta = hunt.get("trace_audit") or {}
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
st["waves"].append({"wave": wave_no, "raw": os.path.relpath(sys.argv[1], out), **stats})
save(os.path.join(out, "state.json"), st)
print(f"wave {wave_no}: " + ", ".join(f"{k} {v}" for k, v in stats.items()))
if stats.get("ledger_entries"):
    print(
        f"contract-trace ledger: {stats['ledger_entries']} boundaries traced, {stats.get('ledger_skipped', 0)} skipped, "
        f"{stats.get('ledger_bad', 0)} cite an other side that cannot be located in the tree (fabricated, or an ambiguous short path); "
        f"trace audit overturned {stats.get('trace_overturned', 0)} of {stats.get('trace_rechecked', 0)} re-checked verdicts"
    )
print(
    "next: python3 consolidate.py && python3 status.py   (and snapshot the run directory)"
)
