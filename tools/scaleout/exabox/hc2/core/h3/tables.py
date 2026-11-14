#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""h3/tables.py: the two fixed-seed tables of reliability.sh's H3 steps, checked into h3/.
  random-table.tsv    h3-random: rows 'gens<TAB>pause_s', 1-5 gens back to back, then 10-30 s with no load; repeats.
  full-cooldowns.tsv  h3-full: rows 'offset_s<TAB>duration_s', 5 zero-load cooldowns of 60-180 s in the first 30 min,
                      offsets from the step's first gen, one per 6 min slot, at least 60 s of load between two; none
                      after 30 min, so the rest of the step is continuous full load.
Same seed, same table: tables.py --write rewrites both files; tables.py --check exits 1 when a file differs.
Python's random.Random(seed) only (stdlib, the same on every box)."""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED, RANDOM_ROWS, GENS, PAUSE_S = 4912, 40, (1, 5), (10, 30)
COOL_SEED, COOL_N, COOL_S, COOL_END_S, GAP_S = 5534, 5, (60, 180), 1800, 30  # COOL_END_S = reliability.sh H3_COOL_END


def random_rows(seed=RANDOM_SEED, n=RANDOM_ROWS):
    r = random.Random(seed)
    return [(r.randint(*GENS), r.randint(*PAUSE_S)) for _ in range(n)]


def cooldown_rows(seed=COOL_SEED, n=COOL_N, hour=COOL_END_S):
    """One cooldown per slot of hour/n s: duration 60-180 s, start at least GAP_S after the slot opens and the end at
    least GAP_S before it closes, so two cooldowns never touch and the last ends inside the window."""
    r = random.Random(seed)
    slot = hour // n
    rows = []
    for k in range(n):
        d = r.randint(*COOL_S)
        rows.append((k * slot + r.randint(GAP_S, slot - GAP_S - d), d))
    return rows


def render(name):
    if name == "random-table.tsv":
        head = (
            f"# h3/random-table.tsv: reliability.sh h3-random load (h3/tables.py, seed {RANDOM_SEED}). Rows 'gens<TAB>pause_s':\n"
            f"# run that many gens back to back, then pause_s with no load. Gens {GENS[0]}-{GENS[1]}, pause "
            f"{PAUSE_S[0]}-{PAUSE_S[1]} s, {RANDOM_ROWS} rows; repeats past its end.\n"
        )
        rows = random_rows()
    else:
        head = (
            f"# h3/full-cooldowns.tsv: reliability.sh h3-full cooldowns (h3/tables.py, seed {COOL_SEED}).\n"
            f"# Rows 'offset_s<TAB>duration_s': {COOL_N} zero-load cooldowns of {COOL_S[0]}-{COOL_S[1]} s in the first {COOL_END_S // 60} min, "
            f"offset from the first gen;\n# each starts at the first gen boundary at or after its offset. None after {COOL_END_S // 60} min "
            f"(h3.sh --cooldowns-end): then continuous full load.\n"
        )
        rows = cooldown_rows()
    return head + "".join(f"{a}\t{b}\n" for a, b in rows)


def parse(path):
    return [tuple(int(x) for x in ln.split("\t")[:2]) for ln in open(path) if ln.strip() and not ln.startswith("#")]


def main(argv):
    bad = 0
    for name in ("random-table.tsv", "full-cooldowns.tsv"):
        p = os.path.join(HERE, name)
        want = render(name)
        if argv[:1] == ["--write"]:
            open(p + ".tmp", "w").write(want)
            os.replace(p + ".tmp", p)
        elif not os.path.exists(p) or open(p).read() != want:
            print(f"h3/tables.py: {name} differs from seed output (h3/tables.py --write)")
            bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
