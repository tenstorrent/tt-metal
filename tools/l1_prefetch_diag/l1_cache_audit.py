#!/usr/bin/env python3
"""
DIAGNOSTIC 1 -- audit every kernel binary cached in every dispatch core's L1.

Self-discovering: nothing is hardcoded. It finds the cached binaries by content, so it
keeps working after a reset, a rebuild, or a different workload, when every DRAM address,
L1 offset and bank assignment has changed.

Ground truth is the on-disk ELF corpus. The prefetcher fills its L1 ring buffer once from
the DRAM kernels_buffer (CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER, dispatch.cpp:2890) and
thereafter relays from L1 (relay_ringbuffer, dispatch.cpp:1766) -- DRAM is never re-read and
nothing verifies the copy, so any disagreement between L1 and the ELF is a real transport or
storage error, latched permanently for the life of the cache entry.

READ-ONLY. Safe against a live hang and alongside a running workload.

  Anchors are 64-byte windows proven unique across the whole corpus, so a blob is only
  matched where it genuinely sits -- a short anchor collides between kernels that share a
  function prologue and produces megabytes of phantom diffs.
"""
import argparse, collections, glob, json, os, sys, time

DEF_CACHE = os.path.expanduser("~/.cache/tt-metal-cache")
HDR = 0x20  # XIP header at the start of .text; never loaded into L1
ANCHOR = 64
MIN_BODY = 512


def load_corpus(roots, verbose):
    from ttexalens.elf import ElfFile

    bodies = {}  # path -> body bytes
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*.elf.xip.elf"), recursive=True):
            try:
                t = bytes(ElfFile.from_bytes(open(p, "rb").read()).get_section_by_name(".text").data)
            except Exception:
                continue
            if len(t) >= HDR + MIN_BODY:
                bodies[p] = t[HDR:]
    # choose, per blob, an anchor window that is unique across the whole corpus
    seen = collections.Counter()
    cands = {}
    for p, b in bodies.items():
        ks = [k for k in range(0, min(len(b) - ANCHOR, 4096), ANCHOR)]
        cands[p] = ks
        for k in ks:
            seen[b[k : k + ANCHOR]] += 1
    picked = {}
    for p, b in bodies.items():
        for k in cands[p]:
            w = b[k : k + ANCHOR]
            if seen[w] == 1:
                picked[p] = (k, w)
                break
    if verbose:
        print(
            f"corpus: {len(bodies)} blobs, {len(picked)} with a unique anchor "
            f"({len(bodies) - len(picked)} skipped as ambiguous)",
            file=sys.stderr,
        )
    return bodies, picked


def audit_core(dump, bodies, picked, halo):
    """Return (n_blobs_found, n_bytes_compared, [findings]).

    Only the region of a blob that is actually resident is judged. The relay length is
    kg_transfer_info.lengths[], not the whole .text body, so a blob's tail can run past its
    cache entry into the next program -- comparing the full body reports that boundary as
    megabytes of phantom corruption. A genuine transport/storage error is instead an ISOLATED
    byte sitting inside a long agreeing run, so a mismatch is only reported when `halo` bytes
    on BOTH sides of it match. That one rule rejects truncation tails, misalignment and
    entry boundaries, and keeps single-bit errors.
    """
    found = 0
    nbytes = 0
    out = []
    for p, (k, w) in picked.items():
        body = bodies[p]
        # The ring buffer can hold more than one copy of a blob, so score EVERY occurrence of
        # the anchor and keep the best-agreeing alignment. Taking the first hit mis-aligns the
        # comparison against a stale copy and silently hides a real finding.
        best = None
        pos = dump.find(w)
        while pos >= 0:
            start = pos - k
            if start >= 0:
                n = min(len(body), len(dump) - start)
                if n >= 2 * halo + 1:
                    got = dump[start : start + n]
                    ok = [got[i] == body[i] for i in range(n)]
                    score = sum(ok)
                    if best is None or score > best[0]:
                        best = (score, start, n, got, ok)
            pos = dump.find(w, pos + 1)
        if best is None:
            continue
        _, start, n, got, ok = best
        found += 1
        nbytes += n
        diffs = []
        for i in range(halo, n - halo):
            if ok[i]:
                continue
            if all(ok[i - halo : i]) and all(ok[i + 1 : i + 1 + halo]):
                diffs.append(i)
        if not diffs:
            continue
        # how much of this blob is actually resident/agreeing, for context
        agree = sum(ok)
        out.append(
            {
                "elf": p,
                "l1_base": start,
                "cmp_len": n,
                "agreeing": agree,
                "n_bytes_wrong": len(diffs),
                "n_bits_wrong": sum(bin(got[i] ^ body[i]).count("1") for i in diffs),
                "diffs": [
                    {
                        "l1": start + i,
                        "blob_off": i,
                        "text_off": HDR + i,
                        "expected": body[i],
                        "observed": got[i],
                        "xor": body[i] ^ got[i],
                        "bits": [b for b in range(8) if (body[i] ^ body[i] ^ (got[i] ^ body[i])) >> b & 1],
                    }
                    for i in diffs[:32]
                ],
            }
        )
    return found, nbytes, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="all")
    ap.add_argument(
        "--cores", default="auto", help="'auto' probes the usual dispatch columns, or a noc0 list like 16-2,16-3"
    )
    ap.add_argument(
        "--cache-root", action="append", default=None, help=f"kernel cache dir (repeatable). default: {DEF_CACHE}/*"
    )
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--halo", type=int, default=256, help="matching bytes required on both sides of a reported mismatch"
    )
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    roots = a.cache_root or sorted(glob.glob(os.path.join(DEF_CACHE, "*", "kernels")))
    if not roots:
        print(f"ERROR: no kernel cache found under {DEF_CACHE}", file=sys.stderr)
        return 2

    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.coordinate import OnChipCoordinate
    from ttexalens.memory_access import create_l1_memory_access

    bodies, picked = load_corpus(roots, not a.quiet)
    if not picked:
        print("ERROR: no usable blobs in the corpus", file=sys.stderr)
        return 2

    ctx = init_ttexalens()
    by = {d.id: d for d in ctx.devices.values()}
    ids = sorted(by) if a.devices == "all" else [int(x, 0) for x in a.devices.split(",")]
    cores = None if a.cores == "auto" else [c.strip() for c in a.cores.split(",")]

    print(f"auditing {len(picked)} cached-binary candidates against on-disk ELFs\n")
    report = {"roots": roots, "findings": [], "devices": {}}
    total_bad = 0

    for dev_id in ids:
        dev = by[dev_id]
        cl = cores or [f"{x}-{y}" for x in (15, 16, 17) for y in (2, 3)]
        for core in cl:
            try:
                buf = bytearray(0x180000)
                create_l1_memory_access(OnChipCoordinate.create(core, dev, "noc0")).read(0, buf)
            except Exception:
                continue
            t0 = time.time()
            nfound, ncmp, findings = audit_core(bytes(buf), bodies, picked, a.halo)
            if nfound == 0:
                continue
            key = f"{dev_id}:{core}"
            report["devices"][key] = {"blobs": nfound, "bytes": ncmp, "bad": len(findings)}
            if findings:
                total_bad += 1
                print(
                    f"  device {dev_id:3d} {core}: *** {len(findings)} corrupt blob(s) *** "
                    f"({nfound} found, {ncmp} B compared, {time.time()-t0:.1f}s)"
                )
                for f in findings:
                    f["device"] = dev_id
                    f["core"] = core
                    report["findings"].append(f)
                    print(
                        f"      {f['n_bytes_wrong']} byte(s) / {f['n_bits_wrong']} bit(s) wrong in "
                        f".../{f['elf'].split('/kernels/')[-1]}"
                    )
                    print(
                        f"      resident window {f['cmp_len']} B, {f['agreeing']} agreeing "
                        f"({100.0*f['agreeing']/f['cmp_len']:.2f}%)"
                    )
                    for d in f["diffs"]:
                        print(
                            f"        L1 0x{d['l1']:06x}  .text+0x{d['text_off']:05x}  "
                            f"{d['expected']:02x} -> {d['observed']:02x}  xor {d['xor']:02x}  "
                            f"bit(s) {d['bits']}"
                        )
            elif not a.quiet:
                print(f"  device {dev_id:3d} {core}: clean  " f"({nfound} blobs, {ncmp} B, {time.time()-t0:.1f}s)")

    print(f"\n{total_bad} core(s) hold a corrupted kernel binary")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
