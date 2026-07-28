# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Select 4 balanced + 4 edge dispatch-group cases and write a test_dispatch_combine input.

Reads any expert-routing safetensors (one `expert_ids_layer_<L>` (total_tokens, top_k) int32
tensor per MoE layer, Galaxy-global expert IDs in [0, num_routed_experts)) -- either the
longbook capture test_dispatch_combine already uses, or one produced by routing_capture.
save_safetensors from a fresh kimi run.

For every (layer, Galaxy column/dispatch group) it computes the in-col share: the fraction of
all total_tokens*top_k picks landing in that group's experts_per_group experts. Then it selects:

  * EDGE     : the `--n` (layer, col) pairs with the HIGHEST in-col share (heaviest dispatch group)
  * BALANCED : the `--n` (layer, col) pairs whose in-col share is CLOSEST to the uniform
               1/num_dispatch_groups (= 25% for 4 groups)

It writes a curated safetensors containing exactly the layers referenced by those 8 pairs (so it
is small but still load_captured_routing-compatible), and prints paste-ready _REAL_INDICES_PICKS
blocks for test_dispatch_combine_perf.py.

Run:
    python .../extract_dispatch_cases.py                          # defaults to the longbook capture
    python .../extract_dispatch_cases.py --path captured_expert_routing.safetensors
    python .../extract_dispatch_cases.py --n 4 --out dispatch_cases.safetensors
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NUM_ROUTED_EXPERTS = 256
NUM_DISPATCH_GROUPS = 4  # Galaxy columns
EXPERTS_PER_GROUP = NUM_ROUTED_EXPERTS // NUM_DISPATCH_GROUPS  # 64
TOP_K = 8
UNIFORM_SHARE = 100.0 / NUM_DISPATCH_GROUPS  # 25.0


def default_path() -> Path:
    base = Path(os.getenv("DEEPSEEK_V3_TRACE_DIR", "/mnt/MLPerf/deepseek-prefill-cache"))
    return base / "longbook_qa_eng_prefill_25600_nopad" / "expert_routing.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=default_path(), help="input expert_routing.safetensors")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/nostojic/dispatch_cases.safetensors"),
        help="curated output safetensors",
    )
    ap.add_argument("--n", type=int, default=4, help="number of balanced and of edge cases (each)")
    args = ap.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Not found: {args.path}  (set DEEPSEEK_V3_TRACE_DIR or pass --path)")

    # (share_pct, layer, col)
    rows: list[tuple[float, int, int]] = []
    with safe_open(str(args.path), framework="pt") as f:
        layer_keys = sorted(
            (k for k in f.keys() if k.startswith("expert_ids_layer_")),
            key=lambda k: int(k.rsplit("_", 1)[1]),
        )
        if not layer_keys:
            raise SystemExit(f"No expert_ids_layer_* keys in {args.path}")
        for key in layer_keys:
            layer = int(key.rsplit("_", 1)[1])
            picks = f.get_tensor(key)  # (total_tokens, top_k)
            for col in range(NUM_DISPATCH_GROUPS):
                lo, hi = col * EXPERTS_PER_GROUP, (col + 1) * EXPERTS_PER_GROUP
                share = ((picks >= lo) & (picks < hi)).float().mean().item() * 100.0
                rows.append((share, layer, col))

    edge = sorted(rows, key=lambda r: r[0], reverse=True)[: args.n]
    balanced = sorted(rows, key=lambda r: abs(r[0] - UNIFORM_SHARE))[: args.n]

    print(f"src: {args.path}\n")
    print(f"EDGE (heaviest dispatch group, top {args.n} by in-col share):")
    for share, layer, col in edge:
        print(f"    ({layer}, {col}),  # {share:.1f}% in-col share")
    print(f"\nBALANCED (closest to uniform {UNIFORM_SHARE:.0f}%, top {args.n}):")
    for share, layer, col in balanced:
        print(f"    ({layer}, {col}),  # {share:.1f}% in-col share")

    # Curated safetensors: only the layers referenced by the selected pairs.
    layers_needed = sorted({layer for _, layer, _ in edge + balanced})
    tensors = {}
    with safe_open(str(args.path), framework="pt") as f:
        for layer in layers_needed:
            tensors[f"expert_ids_layer_{layer}"] = (
                f.get_tensor(f"expert_ids_layer_{layer}").to(torch.int32).contiguous()
            )
    save_file(tensors, str(args.out))
    any_key = next(iter(tensors))
    print(
        f"\nwrote {len(tensors)} layers ({layers_needed}) -> {args.out}"
        f"  (e.g. {any_key} shape={tuple(tensors[any_key].shape)})"
    )
    total_tokens = tensors[any_key].shape[0]
    print(
        f"feed to test_dispatch_combine via captured_indices_path/TT_DS_USE_CAPTURED_INDICES={args.out}; "
        f"set seq_len_per_chip = total_tokens/dispatch_group_size = {total_tokens}//8 = {total_tokens // 8}"
    )


if __name__ == "__main__":
    main()
