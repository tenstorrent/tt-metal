# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side readback for CHUNK_SKIP_TELEMETRY: reduce CSTL DPRINT logs to a
per-position skip curve plus the amortization-law fit. Underscore-prefixed:
not collected by pytest.

Usage:
    _topk_large_indices_skip_telemetry_parse.py [--user-k N] [--csv OUT.csv] \
        <dprint_log> [<dprint_log> ...]

Parses the lines emitted by the CHUNK_SKIP_TELEMETRY recorder in
topk_large_indices_chunk_skip.hpp (MATH, once per row):
    CSTL r <row> n <num_chunks> f <first_tested> s <skipped>
    CSTLM <word_idx> <mask_word>
DPRINT prefixes device/core/RISC info; patterns match anywhere in the line.

Outputs:
  - per-position empirical P(skip | c) across all rows in all logs, with the
    amortization law e^(-USER_K/(c+1)) alongside (USER_K via --user-k,
    default 32),
  - aggregate E[#skips]/row observed vs the law's sum over tested positions,
  - optional --csv machine-readable per-position dump for downstream tooling
    (the gate A/B harness reads it).

Hard validation per row (parse fails loudly instead of skewing the curve):
  - popcount(mask) == s,
  - no skip bit below first_tested or at/after num_chunks,
  - mask words arrive in order, exactly ceil(num_chunks/32) of them.
"""

import math
import re
import sys

user_k = 32
csv_out = None
args = list(sys.argv[1:])
if "--user-k" in args:
    i = args.index("--user-k")
    user_k = int(args[i + 1])
    del args[i : i + 2]
if "--csv" in args:
    i = args.index("--csv")
    csv_out = args[i + 1]
    del args[i : i + 2]
if not args:
    sys.exit(__doc__)

hdr_re = re.compile(r"CSTL r (\d+) n (\d+) f (\d+) s (\d+)")
msk_re = re.compile(r"CSTLM (\d+) (\d+)")

rows = []  # (num_chunks, first, skipped, [mask words])


def finish(rec, path):
    if rec is None:
        return
    num_chunks, first, skipped, words = rec
    want_words = (num_chunks + 31) // 32
    assert len(words) == want_words, f"{path}: {len(words)} mask words, want {want_words} (n={num_chunks})"
    rows.append(rec)


# Multi-core row-parallel launches interleave per-core DPRINT streams in one
# file. The default "<dev>:(x,y):<RISC>: " prefix identifies each stream —
# demux records per (path, prefix) so one core's CSTLM lines never attach to
# another core's CSTL header. Within a single core's stream, lines are in
# emission order.
for path in args:
    cur_by_stream = {}
    with open(path) as f:
        for line in f:
            m = hdr_re.search(line)
            if m:
                stream = line[: m.start()]
                finish(cur_by_stream.get(stream), path)
                cur_by_stream[stream] = [int(m.group(2)), int(m.group(3)), int(m.group(4)), []]
                continue
            m = msk_re.search(line)
            if m:
                stream = line[: m.start()]
                cur = cur_by_stream.get(stream)
                if cur is not None:
                    w = int(m.group(1))
                    assert w == len(cur[3]), f"{path}: mask word {w} out of order (have {len(cur[3])})"
                    cur[3].append(int(m.group(2)))
    for cur in cur_by_stream.values():
        finish(cur, path)

if not rows:
    sys.exit("no CSTL records found")

max_chunks = max(r[0] for r in rows)
skips_at = [0] * max_chunks
tested_at = [0] * max_chunks
tot_skips = 0

for num_chunks, first, skipped, words in rows:
    bits = 0
    for w, word in enumerate(words):
        for b in range(32):
            c = w * 32 + b
            if word >> b & 1:
                assert first <= c < num_chunks, f"skip bit at c={c} outside tested range [{first}, {num_chunks})"
                skips_at[c] += 1
                bits += 1
    assert bits == skipped, f"mask popcount {bits} != s {skipped}"
    for c in range(first, num_chunks):
        tested_at[c] += 1
    tot_skips += skipped

n_rows = len(rows)
print(f"rows={n_rows} user_k={user_k} max_chunks={max_chunks}")
print(f"{'c':>5} {'tested':>7} {'skips':>6} {'P_obs':>8} {'P_law':>8}")
law_sum = 0.0
csv_lines = ["c,tested,skips,p_obs,p_law"]
for c in range(max_chunks):
    if tested_at[c] == 0:
        continue
    p_obs = skips_at[c] / tested_at[c]
    p_law = math.exp(-user_k / (c + 1))
    law_sum += p_law * tested_at[c] / n_rows
    print(f"{c:>5} {tested_at[c]:>7} {skips_at[c]:>6} {p_obs:>8.4f} {p_law:>8.4f}")
    csv_lines.append(f"{c},{tested_at[c]},{skips_at[c]},{p_obs:.6f},{p_law:.6f}")

obs = tot_skips / n_rows
print(f"\nE[#skips]/row observed = {obs:.2f}   law = {law_sum:.2f}")
csv_lines.append(f"TOTAL,{n_rows},{tot_skips},{obs:.6f},{law_sum:.6f}")

if csv_out:
    with open(csv_out, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"csv written: {csv_out}")
