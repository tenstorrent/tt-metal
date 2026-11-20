# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the receive slots of sparse_mla's SM_ROWS form (16 token rows on 8 S-blocks). The compute kernel
waits for one slot per (user, step) pair in which a merge partial is due on its S-block, and nothing recovers a
compute kernel that waits for a slot nobody publishes. tests/host_due_slots.cpp includes the kernels' own header
(kernels/sm_common.hpp, sm::DueList: the one list the io kernel publishes from, the compute kernel consumes and a
sender numbers its partner's slot by) and checks, for every count row and S-block, that what the io kernel
publishes (partials arriving in any order, and when it gives up waiting) is what the compute kernel consumes, and
that every slot has exactly one sender. Here: 100,000 count rows of 16 users, and the lists against a model written
independently in Python.

    python -m pytest models/experimental/hy4_preview/fused/sparse_mla/tests/test_due_slots.py
"""

import random
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
USERS, GROUPS, CHUNK, TOPK, STEPS = 16, 8, 128, 2048, 3
SPECIAL = [
    0,
    0,
    0,
    1,
    2,
    64,
    127,
    128,
    129,
    255,
    256,
    257,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4096,
    65535,
    65536,
    0xFFFFFFFF,
]


def count_rows(n, seed=0):
    rng = random.Random(seed)
    rows = [[0] * USERS, [TOPK] * USERS, [0xFFFFFFFF] * USERS, [1] * USERS, [2047] * USERS, [129] * USERS]
    for u in range(USERS):  # one live row
        for v in (1, 128, 129, 2048, 65536):
            rows.append([v if i == u else 0 for i in range(USERS)])
    while len(rows) < n:
        kind = rng.random()
        if kind < 0.35:
            rows.append([rng.choice(SPECIAL) for _ in range(USERS)])
        elif kind < 0.7:
            rows.append([rng.randint(0, TOPK) if rng.random() < 0.6 else 0 for _ in range(USERS)])
        elif kind < 0.85:
            rows.append([rng.randint(0, 2 * CHUNK) for _ in range(USERS)])
        else:
            rows.append([rng.choice(SPECIAL) if rng.random() < 0.3 else rng.randint(0, 70000) for _ in range(USERS)])
    return rows


def model_lists(row):
    """Written from the piece's description, not from the header: user u's chunks sit on S-blocks u, u + 1, ...
    (mod 8); relative index r = (b - u) mod 8; a user's active S-blocks are r < min(chunks, 8); at step s the block
    with r a multiple of 2^(s + 1) receives the partial of r + 2^s if that block is active."""
    out = []
    for b in range(GROUPS):
        pairs = []
        for u, raw in enumerate(row):
            n = min(raw, TOPK)
            active = min(-(-n // CHUNK), GROUPS)
            r = (b - u) % GROUPS
            for s in range(STEPS):
                if r < active and r % (2 << s) == 0 and r + (1 << s) < active:
                    pairs.append((u, s))
        out.append(pairs)
    return out


def checksum(rows):
    total, longest, pairs = 0, 0, 0
    for row in rows:
        for b, lst in enumerate(model_lists(row)):
            pairs += len(lst)
            longest = max(longest, len(lst))
            for u, s in lst:
                total = (total * 1000003 + (b * 64 + u * 4 + s + 1)) % (1 << 64)
    return pairs, longest, total


def test_due_slots(tmp_path):
    exe = tmp_path / "host_due_slots"
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-o", str(exe), str(HERE / "host_due_slots.cpp")], check=True)
    rows = count_rows(100_000)
    text = "\n".join(" ".join(str(v) for v in row) for row in rows) + "\n"
    run = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    words = run.stdout.split()
    got = {words[i]: int(words[i + 1]) for i in range(0, len(words), 2)}
    assert got["rows"] == len(rows), got
    # the lists themselves against the independent model, on the first 20,000 rows
    part = rows[:20_000]
    again = subprocess.run(
        [str(exe)],
        input="\n".join(" ".join(str(v) for v in row) for row in part) + "\n",
        capture_output=True,
        text=True,
    )
    words = again.stdout.split()
    have = {words[i]: int(words[i + 1]) for i in range(0, len(words), 2)}
    pairs, longest, total = checksum(part)
    assert (have["pairs"], have["longest"], have["checksum"]) == (pairs, longest, total), (have, pairs, longest, total)
    assert got["longest"] <= 14, got
    print(
        f"HY4_MK3 due slots: {got['rows']} count rows x {GROUPS} S-blocks, {got['pairs']} due pairs, longest list {got['longest']}: "
        f"published == consumed in both branches, one sender per slot; lists equal to the model on {len(part)} rows"
    )
