#!/usr/bin/env python3
"""proof_counts.py CAPTURE_DIR [--prev PREV_JSON] [--json OUT]: the completion-proof counters of the diagnostic receiver
(proof_count root's router_receiver.hpp: dump_proof_counts) from a hang_capture.py L1 dump of every Ethernet core.

Each receiver ERISC writes, at the end of its loop, into the first slot of its VC0 receiver ring: magic 0x50524F46,
version, the proof cap, the number of VC records, then per VC 16 words (runs, slots, group size, attempts, reads,
hist[1, 2-4, 5-8, 9-16, 17-32], cap hits, G 1 checks, G 1 passes, G 1 deferrals, slots credited) and the wall clock
(2 words). The record is found by scanning each core's L1 for the magic. --prev: an earlier run's JSON; a record
identical to that run's for the same core (same clock) was left by a router this run did not launch: marked stale."""
import argparse, glob, json, os, re, struct

MAGIC = 0x50524F46
FIELDS = ["attempts", "reads", "h1", "h2_4", "h5_8", "h9_16", "h17_32", "cap_hits", "g1_checks", "g1_pass",
          "g1_deferred", "slots_credited"]


def parse_l1(blob):
    out = []
    key = struct.pack("<I", MAGIC)
    off = blob.find(key)
    while off != -1:
        here, off = off, blob.find(key, off + 1)
        if here % 4 or here + 16 + 64 * 3 + 8 > len(blob):
            continue
        off_ = off
        off = here
        w = struct.unpack_from("<4I", blob, off)
        version, cap, nvc = w[1], w[2], w[3]
        if version != 1 or nvc not in (1, 2, 3) or cap not in (1, 8, 16, 32, 64, 128, 256):
            off = off_
            continue
        rec = {"addr": off, "cap": cap, "vcs": {}}
        for vc in range(nvc):
            v = struct.unpack_from("<16I", blob, off + 16 + 64 * vc)
            if v[0] != 1:
                continue  # (the next VC)
            rec["vcs"][vc] = {"slots": v[1], "group": v[2], **dict(zip(FIELDS, v[3:3 + len(FIELDS)]))}
        lo, hi = struct.unpack_from("<2I", blob, off + 16 + 64 * nvc)
        rec["clock"] = lo | hi << 32
        out.append(rec)
        off = off_
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture")
    ap.add_argument("--prev", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.capture, "index.json")))
    remote = {(c[0], c[1]): (c[2], c[3]) for c in idx.get("eth_connections", [])}
    prev = {}
    if a.prev and os.path.exists(a.prev):
        prev = {(r["chip"], r["eth"]): r for r in json.load(open(a.prev))}
    recs = []
    for f in sorted(glob.glob(os.path.join(a.capture, "chip*", "eth*_l1.bin"))):
        chip = int(re.search(r"chip(\d+)", f).group(1)); eth = int(re.search(r"eth(\d+)_l1", f).group(1))
        found = parse_l1(open(f, "rb").read())
        if not found:
            continue
        r = found[0]
        r.update(chip=chip, eth=eth, records=len(found), remote=remote.get((chip, eth)))
        p = prev.get((chip, eth))
        r["stale"] = bool(p and p.get("clock") == r["clock"])
        recs.append(r)
    if a.json:
        json.dump(recs, open(a.json, "w"), indent=1)
    for r in recs:
        for vc, c in sorted(r["vcs"].items()):
            if c["attempts"] == 0 and c["g1_checks"] == 0:
                continue
            print(f"chip {r['chip']} eth {r['eth']:2d} (from chip {r['remote'][0] if r['remote'] else '?'}) VC{vc} "
                  f"G{c['group']} slots {c['slots']}"
                  f"{' STALE' if r['stale'] else ''}: " + " ".join(f"{k} {c[k]}" for k in FIELDS))


if __name__ == "__main__":
    main()
