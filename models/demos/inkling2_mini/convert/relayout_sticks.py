# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Copy a weight cache with the expert tensors of fused-MoE layers rewritten in the stick layout. Host only.

    python -m models.demos.inkling2_mini.convert.relayout_sticks --src cache/v1 --dst cache/v1-stick [--dtypes BFP4]

The source cache is only read. A layer is rewritten when its gate_up and down share a dtype listed in --dtypes
(the fused kernel needs one dtype for both); every other file is hard-linked (copied across filesystems). Each
rewritten tensor is checked: un-sticking it gives back the source tile bytes exactly. --dry-run prints the plan and
the disk space it needs. --check re-verifies an existing --dst from disk (e.g. after a crash) and writes nothing.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import resource
import shutil
import time
from pathlib import Path

from models.demos.inkling2_mini.convert.stick import N_BANKS, TILE_BYTES, check_sticks, stick_pages, write_sticks

MARGIN = 2 << 30  # free space left on the destination filesystem after the run


def plan(m: dict, dtypes) -> set[tuple[int, str]]:
    """(layer, name) of the expert tensors to rewrite."""
    by_layer: dict[int, list[dict]] = {}
    for e in m["tensors"]:
        if e["layer"] is not None and e["name"].startswith("experts."):
            by_layer.setdefault(e["layer"], []).append(e)
    out = set()
    for L, ex in by_layer.items():
        kinds = {(e["dtype"], e["layout"]) for e in ex}
        if len(ex) == 2 and len(kinds) == 1 and kinds <= {(d, "TILE") for d in dtypes}:
            out |= {(L, e["name"]) for e in ex}
    return out


def stick_bytes(e: dict, n_banks: int) -> int:
    _, E, K, N = e["shape"]
    return len(stick_pages(E, K // 32, N // 32, n_banks)) * TILE_BYTES[e["dtype"]] + 4096  # + header


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError as x:
        if x.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)


def check(src: Path, dst: Path) -> int:
    """Re-read every file of ``dst``: stick tensors un-stick to the ``src`` tiles, the rest match ``src``."""
    srcm = {(e["layer"], e["name"]): e for e in json.loads((src / "manifest.json").read_text())["tensors"]}
    bad = 0
    for e in json.loads((dst / "manifest.json").read_text())["tensors"]:
        s, d = src / e["file"], dst / e["file"]
        try:
            if e["layout"] == "STICK":
                check_sticks(s, d, srcm[(e["layer"], e["name"])], e["stick_banks"])
            elif not os.path.samefile(s, d) and s.read_bytes() != d.read_bytes():
                raise ValueError("differs from source")
        except Exception as x:
            bad += 1
            print(f"BAD {d}: {x}", flush=True)
    print(f"check {dst}: {bad} bad files")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="cache to read")
    ap.add_argument("--dst", type=Path, required=True, help="new cache to write")
    ap.add_argument("--dtypes", default="BFP4", help="comma list of BFP4, BFP8")
    ap.add_argument("--n-banks", type=int, default=N_BANKS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="only re-verify an existing --dst")
    args = ap.parse_args(argv)
    dtypes = tuple(x for x in args.dtypes.split(",") if x)
    if not dtypes or set(dtypes) - set(TILE_BYTES):
        raise SystemExit(f"--dtypes: BFP4 and/or BFP8, got {args.dtypes}")
    if args.src.resolve() == args.dst.resolve():
        raise SystemExit("--dst must differ from --src")
    if args.check:
        raise SystemExit(1 if check(args.src, args.dst) else 0)
    m = json.loads((args.src / "manifest.json").read_text())
    if not m.get("complete"):
        raise SystemExit(f"{args.src}: manifest not complete")
    todo = plan(m, dtypes)
    entries = {(e["layer"], e["name"]): e for e in m["tensors"]}
    need = sum(stick_bytes(entries[k], args.n_banks) for k in todo)
    dst_fs = args.dst if args.dst.exists() else args.dst.parent
    free = shutil.disk_usage(dst_fs).free
    same_fs = os.stat(args.src).st_dev == os.stat(dst_fs).st_dev
    if not same_fs:
        need += sum((args.src / e["file"]).stat().st_size for k, e in entries.items() if k not in todo)
    layers = sorted({L for L, _ in todo})
    print(f"rewrite {len(todo)} tensors (layers {layers}), need {need / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB")
    if args.dry_run:
        return
    if need + MARGIN > free:
        raise SystemExit(f"not enough disk on {dst_fs}: need {need / 2**30:.2f} GiB + 2 GiB margin")
    if (args.dst / "manifest.json").exists():
        raise SystemExit(f"{args.dst} already has a manifest; remove it or pick a new --dst")

    t0 = time.time()
    out = []
    for e in m["tensors"]:
        src, dst = args.src / e["file"], args.dst / e["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (e["layer"], e["name"]) in todo:
            t1 = time.time()
            out.append(write_sticks(src, dst, e, args.n_banks))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            print(f"L{e['layer']:02d} {e['name']} {e['dtype']}: {time.time() - t1:.1f}s, peak RSS {rss:.2f} GiB",
                  flush=True)  # fmt: skip
        else:
            if dst.exists():
                dst.unlink()  # a partial earlier run; the manifest is written last
            link_or_copy(src, dst)
            out.append(e)
    new = dict(m, tensors=out, stick_source=str(args.src), stick_banks=args.n_banks)
    new["expert_layout"] = {
        f"L{e['layer']:02d}/{e['name']}": e["layout"] for e in out if e["name"].startswith("experts.")
    }
    new["peak_rss_gib_last_run"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    tmp = args.dst / "manifest.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(new, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.sync()  # hard links and copies too, before the manifest marks the cache usable
    tmp.replace(args.dst / "manifest.json")
    print(f"done in {time.time() - t0:.0f}s, peak RSS {new['peak_rss_gib_last_run']} GiB -> {args.dst}")


if __name__ == "__main__":
    main()
