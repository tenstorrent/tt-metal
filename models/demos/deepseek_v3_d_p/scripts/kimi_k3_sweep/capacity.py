#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Per-chip DRAM capacity model for Kimi-K3 prefill on 4xGLX, fitted to the rank-0 DRAM logs."""
GIB = 2**30
TOTAL = 31.581  # DRAM per chip, allocator view
WEIGHTS = 19.016 - 24 * 0.0290  # rank 0 "after weights" minus 24 users of KDA carries
COMPILE = 0.048  # rank 0 "after compile" - "after KV cache"
RESERVE = (0.637, 1.093)  # 25 users OOM in compile with 0.637 free; 24 run with 1.093 free
SP, MLA_L, KDA_L = 8, 6, 18  # rank 0: 6 MLA + 18 KDA layers
ROW = 576 * 1.0625  # kv_lora 512 + rope 64, bfp8 incl. exponents
CARRY = 0.0290  # per user: 18 native KDA carries (measured, after-weights slope)
SLAB = 0.4760 - MLA_L * 131200 * ROW / GIB  # per user: KV+slab slope at 1M minus KV


def kv(seq):
    return MLA_L * (seq / SP) * ROW / GIB


print(f"fixed {WEIGHTS:.3f} GiB weights + {COMPILE:.3f} compile; KDA carry {CARRY:.4f} + slab {SLAB:.4f} GiB/user")
print("| max_seq_len | KV/user/chip GiB | KDA/user/chip GiB | users (reserve 1.09) | users (reserve 0.64) |")
print("|---|---|---|---|---|")
for seq in (131072, 262144, 524288, 1049600):
    per = kv(seq) + CARRY + SLAB
    free = TOTAL - WEIGHTS - COMPILE
    print(
        f"| {seq:,} | {kv(seq):.4f} | {CARRY + SLAB:.4f} | {int((free - RESERVE[1]) // per)} | {int((free - RESERVE[0]) // per)} |"
    )

print("\nreplication what-ifs at reserve 1.09 GiB (users per max_seq_len)")
print("| max_seq_len | today | KV TP-sharded (/4) | + one KDA copy (no separate carry) |")
print("|---|---|---|---|")
free = TOTAL - WEIGHTS - COMPILE - RESERVE[1]
for seq in (131072, 262144, 524288, 1049600):
    today = kv(seq) + CARRY + SLAB
    tp = kv(seq) / 4 + CARRY + SLAB
    one = kv(seq) / 4 + SLAB
    print(f"| {seq:,} | {int(free // today)} | {int(free // tp)} | {int(free // one)} |")
logical_kv = 24 * 1049600 * ROW / GIB
logical_kda = 69 * 96 * 128 * 128 * 4 / GIB
print(
    f"\nper user at 1M: KV logical {logical_kv:.2f} GiB, stored {logical_kv * 4:.2f} GiB (x4 TP replicas); "
    f"KDA recurrent logical {logical_kda:.3f} GiB, stored {logical_kda * 8 * 2:.3f} GiB (x8 SP replicas x2 copies)"
)
