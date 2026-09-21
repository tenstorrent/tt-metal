# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host estimate of BF16-packed online-max identity; not a device guard counter."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--distributions", default="normal,scaled_qk,outliers,common_q,common_k")
    parser.add_argument("--k-chunks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1236)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    target = here / (args.label + ".json")
    assert not target.exists()
    spec = importlib.util.spec_from_file_location(
        "sprint_repro", root / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
    )
    repro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro)
    torch.set_num_threads(8)
    records = []
    for distribution in args.distributions.split(","):
        q, k, _ = repro.make_inputs(1, 256, 512 * args.k_chunks, 128, args.seed, distribution)
        q = q.double()
        previous = torch.full((256,), -torch.inf, dtype=torch.bfloat16)
        all_equal, rows_equal, changed_queries = [], [], []
        for chunk in range(args.k_chunks):
            block = k[..., chunk * 512 : (chunk + 1) * 512, :].double()
            # The packed max recurrence compares the previous BF16 maximum
            # with the new raw (unscaled) QK maximum before packing BF16 again.
            chunk_max = (q @ block.transpose(-2, -1)).amax(dim=-1).reshape(256)
            current = torch.maximum(previous.double(), chunk_max).bfloat16()
            identity = current.view(torch.uint16) == previous.view(torch.uint16)
            if chunk:
                all_equal.append(bool(identity.all()))
                rows_equal.append(float(identity.reshape(8, 32).all(dim=-1).float().mean()))
                changed_queries.append(int((~identity).sum()))
            previous = current
        record = dict(
            distribution=distribution, seed=args.seed, q_length=256, k_length=512 * args.k_chunks,
            whole_q_identity_fraction=sum(all_equal) / len(all_equal),
            row_group_identity_fraction=sum(rows_equal) / len(rows_equal),
            whole_q_identity_by_chunk=all_equal, changed_queries_by_chunk=changed_queries,
            warning="Host double-QK/BF16-max estimate, not exact device FPU accumulation or hardware guard measurement",
        )
        records.append(record)
        target.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if not k.endswith("by_chunk")}), flush=True)


if __name__ == "__main__":
    main()
