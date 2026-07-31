#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Compare D2D-socket SEND vs RECV activations by PCC (issue #49586, DFlash prefill Stage 2).

The inter-galaxy D2D socket has no on-the-wire integrity (no CRC/ECC); it only guarantees lossless,
ordered delivery. To *prove* a packed activation crossed the galaxies correctly we capture it at both
ends and compare it here. Set ``PREFILL_D2D_DUMP_DIR=<shared-fs dir>`` on the run and prefill_runner
writes, per rank boundary (sender rank r -> receiver rank r+1) and per chunk:

    d2d_send_hop{r}-{r+1}_slot{S}_{start}_{end}.pt   (written by rank r,   on galaxy g)
    d2d_recv_hop{r}-{r+1}_slot{S}_{start}_{end}.pt   (written by rank r+1, on galaxy g+1)

Each file holds the FULL composed ``[1,1,chunk,2H]`` host tensor (native bf16 — exactly the transported
bits) plus metadata and the per-shard CRC32s. Because the transport must be bit-exact, for every matched
``(hop, slot, token-range)`` pair the SEND and RECV tensors should be numerically identical:
PCC == 1.0, max_abs_diff == 0, torch.equal == True.

DFlash packs ``[hidden ‖ drafter-partial]`` into a 2H-wide activation. This script therefore reports the
**hidden half** ``[...,:H]`` and the **drafter-partial half** ``[...,H:]`` PCC *separately*, so you can
see that BOTH the verifier hidden state and the drafter FC partial crossed the galaxies correctly. It
also cross-checks the per-shard CRC32 recorded at each end to localize a divergent chip.

Usage:
    python3 models/demos/common/prefill/tools/compare_d2d_pcc.py <dump_dir> [--pcc 0.9999] [--json out.json]
    python3 models/demos/common/prefill/tools/compare_d2d_pcc.py --selftest   # no device / no dumps needed

    <dump_dir>   the PREFILL_D2D_DUMP_DIR the run wrote to (shared NFS path).
    --pcc        per-half PASS threshold (default 0.9999; transport is bit-exact so real pairs hit 1.0).
    --json       also write the full per-pair result table as JSON.

Exit code 0 iff every matched pair passes (and no SEND/RECV is left unmatched); non-zero otherwise, so
this doubles as a CI gate. Depends only on ``torch`` (no ttnn / no device).
"""

import argparse
import glob
import json
import os
import sys

import torch


# ---------------------------------------------------------------------------
# PCC — Pearson correlation of the two flattened tensors, matching the semantics of
# tt-metal's models.utility_functions.comp_pcc (bit-identical -> 1.0; mismatched finiteness -> 0.0;
# both-constant -> 1.0 iff equal). Kept dependency-free (torch only) so this runs anywhere.
# ---------------------------------------------------------------------------
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    if a.numel() != b.numel():
        return float("nan")
    if a.numel() == 0:
        return 1.0
    if torch.equal(a, b):
        return 1.0
    fin_a, fin_b = torch.isfinite(a), torch.isfinite(b)
    if not torch.equal(fin_a, fin_b):
        return 0.0  # a nan/inf appeared on one side but not the other -> corruption
    mask = fin_a & fin_b
    if not torch.all(mask):
        a, b = a[mask], b[mask]
    if a.numel() == 0:
        return 1.0
    a_const = bool(torch.all(a == a[0]))
    b_const = bool(torch.all(b == b[0]))
    if a_const or b_const:
        # Correlation is undefined if either side is constant; fall back to exactness.
        return 1.0 if torch.allclose(a, b) else 0.0
    va, vb = a - a.mean(), b - b.mean()
    denom = torch.sqrt(torch.sum(va * va) * torch.sum(vb * vb))
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((torch.sum(va * vb) / denom).item())


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    fa, fb = a.detach().to(torch.float32), b.detach().to(torch.float32)
    d = (fa - fb).abs()
    return {
        "pcc": pcc(a, b),
        "max_abs_diff": float(d.max().item()) if d.numel() else 0.0,
        "num_mismatch": int((d != 0).sum().item()),
        "numel": int(d.numel()),
        "bit_exact": bool(torch.equal(fa, fb)),
    }


def _key(p: dict) -> tuple:
    """Pair a SEND with its RECV: same boundary (hop) and same chunk (slot + token range)."""
    return (tuple(p["hop"]), p["slot_id"], p["actual_start"], p["actual_end"])


def _crc_compare(send: dict, recv: dict) -> tuple:
    sc, rc = send.get("crc32_per_shard"), recv.get("crc32_per_shard")
    if not sc or not rc or len(sc) != len(rc):
        return None, None
    mismatched = [i for i, (x, y) in enumerate(zip(sc, rc)) if x != y]
    return len(sc) - len(mismatched), mismatched


def compare_dir(dump_dir: str, threshold: float) -> dict:
    files = sorted(glob.glob(os.path.join(dump_dir, "d2d_*.pt")))
    if not files:
        raise FileNotFoundError(f"no d2d_*.pt dumps found in {dump_dir}")
    sends, recvs = {}, {}
    for f in files:
        p = torch.load(f, map_location="cpu", weights_only=False)
        p["_file"] = os.path.basename(f)
        (sends if p["tag"] == "send" else recvs)[_key(p)] = p

    keys = sorted(set(sends) | set(recvs))
    pairs, unmatched = [], []
    for k in keys:
        s, r = sends.get(k), recvs.get(k)
        if s is None or r is None:
            unmatched.append({"key": k, "have": "recv-only" if s is None else "send-only", "file": (r or s)["_file"]})
            continue
        st, rt = s["tensor"], r["tensor"]
        dflash = bool(s.get("dflash")) and bool(r.get("dflash"))
        split = int(s.get("split_at", st.shape[-1]))
        res = {
            "hop": list(k[0]),
            "slot": k[1],
            "range": [k[2], k[3]],
            "shape": list(st.shape),
            "dflash": dflash,
            "full": _diff_stats(st, rt),
        }
        if dflash and split < st.shape[-1]:
            res["hidden"] = _diff_stats(st[..., :split], rt[..., :split])
            res["partial"] = _diff_stats(st[..., split:], rt[..., split:])
        crc_ok, crc_bad = _crc_compare(s, r)
        res["crc_shards_ok"] = crc_ok
        res["crc_shards_bad"] = crc_bad

        halves = [res["full"]] + ([res["hidden"], res["partial"]] if "hidden" in res else [])
        res["pass"] = all(h["pcc"] >= threshold for h in halves)
        pairs.append(res)

    return {"dump_dir": dump_dir, "threshold": threshold, "pairs": pairs, "unmatched": unmatched}


def _fmt_half(name: str, h: dict) -> str:
    flag = "OK " if h["pcc"] >= 0 else "  "  # placeholder; caller prints PASS/FAIL per pair
    return (
        f"    {name:<8} pcc={h['pcc']:.10f}  max_abs_diff={h['max_abs_diff']:.3e}  "
        f"mismatch={h['num_mismatch']}/{h['numel']}  bit_exact={h['bit_exact']}"
    )


def print_report(report: dict) -> bool:
    thr = report["threshold"]
    print(f"\nD2D SEND vs RECV PCC report — {report['dump_dir']}  (threshold {thr})")
    print("=" * 88)
    all_pass = True
    for r in report["pairs"]:
        ok = r["pass"]
        all_pass &= ok
        tag = "PASS" if ok else "FAIL"
        print(
            f"[{tag}] hop {r['hop'][0]}->{r['hop'][1]}  slot={r['slot']}  "
            f"tokens[{r['range'][0]},{r['range'][1]})  shape={r['shape']}  dflash={r['dflash']}"
        )
        print(_fmt_half("full", r["full"]))
        if "hidden" in r:
            print(_fmt_half("hidden", r["hidden"]))
            print(_fmt_half("partial", r["partial"]))
        if r["crc_shards_ok"] is not None:
            n_bad = len(r["crc_shards_bad"])
            crc_line = f"    crc32    {r['crc_shards_ok']} shards identical"
            crc_line += "" if n_bad == 0 else f", {n_bad} DIFFER at shards {r['crc_shards_bad']}"
            print(crc_line)
    for u in report["unmatched"]:
        all_pass = False
        print(f"[FAIL] UNMATCHED {u['have']}  key={u['key']}  ({u['file']})")
    print("=" * 88)
    n = len(report["pairs"])
    n_ok = sum(1 for r in report["pairs"] if r["pass"])
    print(
        f"{n_ok}/{n} pairs passed, {len(report['unmatched'])} unmatched  ->  "
        f"{'ALL GOOD' if all_pass else 'FAILURES PRESENT'}\n"
    )
    return all_pass


# ---------------------------------------------------------------------------
# Self-test: fabricate the exact dump schema (an identical pair, a corrupted-partial pair, an
# unmatched send) into a temp dir and assert the comparator classifies them correctly. Runs with no
# device and no real dumps, so it validates the comparison logic host-side before a real capture.
# ---------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    H, chunk = 64, 128  # tiny stand-ins for hidden_size and chunk
    torch.manual_seed(0)
    d = tempfile.mkdtemp(prefix="d2d_selftest_")

    def save(tag, hop, rng, tensor, crcs):
        torch.save(
            {
                "tag": tag,
                "hop": hop,
                "rank": hop[0] if tag == "send" else hop[1],
                "slot_id": 0,
                "actual_start": rng[0],
                "actual_end": rng[1],
                "dflash": True,
                "split_at": H,
                "packed_width": 2 * H,
                "mesh_shape": (8, 4),
                "dtype": "bf16",
                "layout": "TILE",
                "shape": tuple(tensor.shape),
                "crc32_per_shard": crcs,
                "tensor": tensor,
            },
            os.path.join(d, f"d2d_{tag}_hop{hop[0]}-{hop[1]}_slot0_{rng[0]}_{rng[1]}.pt"),
        )

    # Pair 1 — clean hop (0->1, chunk 0): SEND == RECV exactly.
    t0 = torch.randn(1, 1, chunk, 2 * H, dtype=torch.bfloat16)
    save("send", (0, 1), (0, chunk), t0.clone(), ["0x1"] * 32)
    save("recv", (0, 1), (0, chunk), t0.clone(), ["0x1"] * 32)

    # Pair 2 — corrupted partial half on the RECV (0->1, chunk 1): hidden clean, partial mangled.
    t1 = torch.randn(1, 1, chunk, 2 * H, dtype=torch.bfloat16)
    bad = t1.clone()
    bad[..., H:] = torch.randn(1, 1, chunk, H, dtype=torch.bfloat16)  # trash the drafter partial
    crc_send = [hex(i) for i in range(32)]
    crc_recv = crc_send.copy()
    crc_recv[18] = "0xdeadbeef"  # a partial-half shard (TP col 2/3) diverged
    save("send", (0, 1), (chunk, 2 * chunk), t1.clone(), crc_send)
    save("recv", (0, 1), (chunk, 2 * chunk), bad, crc_recv)

    # Pair 3 — a SEND with no matching RECV (should be flagged unmatched).
    save("send", (0, 1), (2 * chunk, 3 * chunk), torch.randn(1, 1, chunk, 2 * H, dtype=torch.bfloat16), ["0x0"] * 32)

    report = compare_dir(d, threshold=0.9999)
    ok = print_report(report)

    by_range = {tuple(r["range"]): r for r in report["pairs"]}
    clean = by_range[(0, chunk)]
    corrupt = by_range[(chunk, 2 * chunk)]
    checks = {
        "clean pair passes": clean["pass"] is True,
        "clean full PCC == 1.0": clean["full"]["pcc"] == 1.0,
        "clean bit_exact": clean["full"]["bit_exact"] is True,
        "clean crc all identical": clean["crc_shards_ok"] == 32 and clean["crc_shards_bad"] == [],
        "corrupt pair fails": corrupt["pass"] is False,
        "corrupt hidden still clean": corrupt["hidden"]["pcc"] == 1.0,
        "corrupt partial degraded": corrupt["partial"]["pcc"] < 0.9999,
        "corrupt crc localizes shard 18": corrupt["crc_shards_bad"] == [18],
        "unmatched send flagged": len(report["unmatched"]) == 1 and report["unmatched"][0]["have"] == "send-only",
        "overall report not all-pass": ok is False,
    }
    print("self-test assertions:")
    all_ok = True
    for name, passed in checks.items():
        all_ok &= passed
        print(f"  [{'ok' if passed else 'XX'}] {name}")
    print(f"\nSELF-TEST {'PASSED' if all_ok else 'FAILED'}\n")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare D2D SEND vs RECV activations by PCC (issue #49586).")
    ap.add_argument("dump_dir", nargs="?", help="PREFILL_D2D_DUMP_DIR the run wrote to")
    ap.add_argument("--pcc", type=float, default=0.9999, help="per-half PASS threshold (default 0.9999)")
    ap.add_argument("--json", type=str, default=None, help="also write the result table to this JSON file")
    ap.add_argument("--selftest", action="store_true", help="fabricate dumps and validate the comparator (no device)")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.dump_dir:
        ap.error("dump_dir is required (or pass --selftest)")

    report = compare_dir(args.dump_dir, args.pcc)
    ok = print_report(report)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
