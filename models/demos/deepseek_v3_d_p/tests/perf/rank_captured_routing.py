# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Rank captured expert-routing by dispatch-group load ("in-col share").

Reads expert_routing.safetensors (one `expert_ids_layer_<L>` tensor per MoE
layer, shape (25600, 8) int32, Galaxy-global expert IDs in [0, 256)) and, for
every (layer, Galaxy column), computes the fraction of the 25600*8 top-k picks
that land in that column's 64 experts. That fraction is exactly the
`in_col_share` that load_captured_routing() logs, and it is the "most data
routing" metric behind _REAL_INDICES_PICKS in test_dispatch_combine_perf.py.

It also breaks each (layer, col) down per chunk — the per-chip token slice the
worker sees when it views (25600, 8) -> (dispatch_group_size=8, 3200, top_k=8) —
so you can see the worst chunk (hottest chip) inside a dispatch group.

Run:
    python models/demos/deepseek_v3_d_p/tests/perf/rank_captured_routing.py
    python .../rank_captured_routing.py --path /some/expert_routing.safetensors --top 10
"""

import argparse
import os
from pathlib import Path

from safetensors import safe_open

NUM_ROUTED_EXPERTS = 256
NUM_DISPATCH_GROUPS = 4  # Galaxy columns
EXPERTS_PER_COL = NUM_ROUTED_EXPERTS // NUM_DISPATCH_GROUPS  # 64
DISPATCH_GROUP_SIZE = 8  # chips per column == number of chunks
TOP_K = 8


def default_path() -> Path:
    base = Path(os.getenv("DEEPSEEK_V3_TRACE_DIR", "/mnt/MLPerf/deepseek-prefill-cache"))
    return base / "longbook_qa_eng_prefill_25600_nopad" / "expert_routing.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=default_path(), help="expert_routing.safetensors")
    ap.add_argument("--top", type=int, default=8, help="how many (layer, col) pairs to print")
    args = ap.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Not found: {args.path}  (set DEEPSEEK_V3_TRACE_DIR or pass --path)")

    rows = []  # (share_pct, layer, col, per_chunk_pct[8])
    with safe_open(str(args.path), framework="pt") as f:
        layer_keys = sorted(
            (k for k in f.keys() if k.startswith("expert_ids_layer_")),
            key=lambda k: int(k.rsplit("_", 1)[1]),
        )
        for key in layer_keys:
            layer = int(key.rsplit("_", 1)[1])
            flat = f.get_tensor(key)  # (25600, 8) int32
            # (dispatch_group_size=8 chips/chunks, seq_len_per_chip=3200, top_k=8)
            picks = flat.view(DISPATCH_GROUP_SIZE, -1, TOP_K)
            for col in range(NUM_DISPATCH_GROUPS):
                lo, hi = col * EXPERTS_PER_COL, (col + 1) * EXPERTS_PER_COL
                in_col = ((picks >= lo) & (picks < hi)).float()
                share = in_col.mean().item() * 100.0
                per_chunk = (in_col.mean(dim=(1, 2)) * 100.0).tolist()  # one value per chip/chunk
                rows.append((share, layer, col, per_chunk))

    rows.sort(reverse=True)
    print(f"src: {args.path}\n")
    print(f"{'rank':>4}  {'layer':>5}  {'col':>3}  {'in-col%':>7}   worst-chunk(chip: %)   per-chunk %")
    for rank, (share, layer, col, per_chunk) in enumerate(rows[: args.top], 1):
        worst_chip = max(range(len(per_chunk)), key=lambda c: per_chunk[c])
        per_chunk_str = " ".join(f"{p:4.1f}" for p in per_chunk)
        print(
            f"{rank:>4}  {layer:>5}  {col:>3}  {share:7.1f}   "
            f"chip {worst_chip} = {per_chunk[worst_chip]:5.1f}%   [{per_chunk_str}]"
        )

    print("\nPaste-ready _REAL_INDICES_PICKS (top 4):")
    for share, layer, col, _ in rows[:4]:
        print(f"    ({layer}, {col}),  # {share:.1f}% in-col share")


if __name__ == "__main__":
    main()
