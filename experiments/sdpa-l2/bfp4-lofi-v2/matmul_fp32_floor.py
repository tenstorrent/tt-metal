# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnose the observed ~0.03% public-matmul FP32-output error floor.

Exact BF16 upload checks, one-term products, identity copies and dense sums.
No performance or integrated-attention claims.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import ttnn


def metric(actual, expected):
    a, b = actual.double(), expected.double()
    raw = actual.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    return dict(
        l2_pct=float(100 * (a - b).norm() / b.norm()),
        max_abs=float((a - b).abs().max()),
        exact=int((a == b).sum()), elements=a.numel(),
        low_bits_nonzero={str(n): int(((raw & ((1 << n) - 1)) != 0).sum()) for n in (8, 10, 12, 13, 16)},
        output_sha256=hashlib.sha256(actual.float().numpy().tobytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    path = Path(__file__).with_name(args.label + ".jsonl")
    assert Path(args.label).name == args.label and not path.exists()
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(1240)
    device = ttnn.open_device(device_id=0)
    records = []
    try:
        for case in ("identity", "one_term", "dense32", "dense128", "positive128", "same_eight", "opposite_eight"):
            reduction = 128 if case in ("dense128", "positive128") else 32
            a = torch.randn((1, 1, 32, reduction), generator=generator).bfloat16()
            b = torch.randn((1, 1, reduction, 64), generator=generator).bfloat16()
            if case == "identity":
                a = torch.eye(32).reshape(1, 1, 32, 32).bfloat16()
            if case == "one_term":
                a[..., 1:] = 0
                b[..., 1:, :] = 0
            if case == "positive128":
                a, b = a.abs(), b.abs()
            if case in ("same_eight", "opposite_eight"):
                position = 1 if case == "same_eight" else 8
                a.zero_()
                b.zero_()
                a[..., 0] = 1
                a[..., position] = 2.0**-12
                b[..., 0, :] = 1
                b[..., position, :] = 1
            expected = a.double() @ b.double()
            da, db = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (a, b)]
            assert torch.equal(ttnn.to_torch(da), a) and torch.equal(ttnn.to_torch(db), b)
            for fidelity in ("LoFi", "HiFi2", "HiFi4"):
                config = ttnn.init_device_compute_kernel_config(
                    device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False,
                )
                out = ttnn.matmul(da, db, dtype=ttnn.float32, core_grid=ttnn.CoreGrid(y=1, x=1),
                                  compute_kernel_config=config)
                actual = ttnn.to_torch(out).float()
                record = dict(case=case, fidelity=fidelity, fp32_dst=True, **metric(actual, expected))
                # Independent post-result truncation comparisons: diagnostic,
                # not a claim that hardware performs this specific operation.
                raw_expected = expected.float().contiguous().view(torch.int32)
                for fraction_bits in (10, 11, 12, 13, 14, 16, 19):
                    cut = 23 - fraction_bits
                    truncated = (raw_expected & ~((1 << cut) - 1)).view(torch.float32)
                    record[f"matches_trunc_frac{fraction_bits}"] = int((actual == truncated).sum())
                records.append(record)
                print("MATMUL_FP32_FLOOR", json.dumps(record), flush=True)
                ttnn.deallocate(out)
            ttnn.deallocate(da)
            ttnn.deallocate(db)
        records.insert(0, dict(kind="provenance", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               contract=__doc__, label=args.label))
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
