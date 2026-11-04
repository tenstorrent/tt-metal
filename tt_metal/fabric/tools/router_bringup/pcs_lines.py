#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md). BLACKHOLE ONLY (reads pc_sampler.py runs).

pcs_lines.py SAMPLES_PREFIX CACHE_DIR CHIP/CHAN/ERISC [--top N] [--depth D] [--dump F]: one router ERISC's PC samples
by source line.

For one target of a pc_sampler.py run (SAMPLES_PREFIX.{bin,json}): its build (the kernel cache's image whose relocated
text starts with the image head read at the launch message's load address), its kernel samples as ELF addresses, the
first and last 5% by time dropped (outside the traffic window), then addr2line -i on the cache's own ELF (the JIT
builds with -g). Prints the share of samples per innermost source line, and per inline chain cut to D frames from the
innermost (the helper and its callers), the top N of each. --dump writes every window sample's address with its whole
chain (JSON: [[count, [[function, file:line], ...], ELF address], ...]) to F.
Use PC samples to split cycles within one ERISC, not to say which ERISC bounds a row: a router polls without stopping,
so its samples always add up to the row's rate.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcs_common as pc  # noqa: E402


def short_fn(fn):
    while re.search(r"<[^<>]*>", fn):  # template arguments, innermost first
        fn = re.sub(r"<[^<>]*>", "", fn)
    return re.sub(r"\(.*", "", fn).replace("fabric_router::", "").split()[-1] if fn.strip() else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    ap.add_argument("cache")
    ap.add_argument("target", help="CHIP/CHAN/ERISC, e.g. 3/9/0")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    chip, chan, erisc = (int(x) for x in a.target.replace("E", "").split("/"))
    meta, samples = pc.load_run(a.prefix)
    idx = next(
        (i for i, t in enumerate(meta["targets"]) if (t["chip"], t["chan"], t["erisc"]) == (chip, chan, erisc)), None
    )
    if idx is None:
        sys.exit(f"no target {a.target} in {a.prefix}.json")
    lay = pc.layout(meta, meta["targets"][idx])
    build = pc.match_build(a.cache, erisc, lay)
    if build is None:
        sys.exit("no build in the cache matches the target's image")
    elf, base, h = build
    kern = pc.kernel_addrs(samples.get(idx, []), lay, base)
    chains = pc.a2l(elf, sorted({x for _, x in kern}))
    n = len(kern)
    if not n:
        sys.exit("no kernel samples in the window")
    by_line, by_chain = collections.Counter(), collections.Counter()
    for _, x in kern:
        f = chains.get(x, [])
        by_line[f"{f[0][1]}:{f[0][2]}  ({short_fn(f[0][0])})" if f else "?"] += 1
        by_chain[" < ".join(short_fn(fn) for fn, _, _ in f[: a.depth]) if f else "?"] += 1
    if a.dump:
        counts = collections.Counter(x for _, x in kern)
        json.dump(
            [[c, [[short_fn(fn), f"{fl}:{ln}"] for fn, fl, ln in chains.get(x, [])], x] for x, c in counts.items()],
            open(a.dump, "w"),
        )
    print(f"{a.target}: {len(samples.get(idx, []))} samples, {n} kernel samples in the window; build {h}")
    print(f"\nby innermost line (top {a.top}):")
    for k, c in by_line.most_common(a.top):
        print(f"  {100 * c / n:5.1f}%  {k}")
    print(f"\nby inline chain, {a.depth} frames (top {a.top}):")
    for k, c in by_chain.most_common(a.top):
        print(f"  {100 * c / n:5.1f}%  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
