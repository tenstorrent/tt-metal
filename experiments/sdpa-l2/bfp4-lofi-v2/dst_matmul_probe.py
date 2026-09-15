# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Separate matmul DST/fidelity arithmetic error from representation loss.

One-core TTNN matmuls, not streaming-SDPA timing. No performance claims.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dst_probe_model", HERE / "numerics.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=1240)
    args = parser.parse_args()
    path = HERE / (args.label + ".jsonl")
    assert not path.exists()
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(args.seed)
    q = torch.randn((1, 1, 256, 128), generator=generator).bfloat16()
    k = torch.randn((1, 1, 512, 128), generator=generator).bfloat16()
    v = torch.randn((1, 1, 512, 128), generator=generator).bfloat16()
    p = torch.softmax(q.double() @ k.double().transpose(-2, -1) / math.sqrt(128), -1).bfloat16()
    device = ttnn.open_device(device_id=0)
    try:
        with path.open("x") as output:
            output.write(json.dumps(dict(kind="provenance", args=vars(args),
                contract=__doc__, source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).resolve(), HERE / "numerics.py")})) + "\n")
            for operation, left, right, transpose in (("QK", q, k, True), ("PV", p, v, False)):
                original = left.double() @ (right.double().transpose(-2, -1) if transpose else right.double())
                for representation in ("bf16", "lofi_rne", "bfp8", "bfp4_right"):
                    a, b = left.float(), right.float()
                    adtype = bdtype = ttnn.bfloat16
                    if representation == "lofi_rne":
                        a, b = M.round_significand(a, 7), M.round_significand(b, 5)
                    elif representation == "bfp8":
                        adtype = bdtype = ttnn.bfloat8_b
                    elif representation == "bfp4_right":
                        a = M.round_significand(a, 7)
                        bdtype = ttnn.bfloat4_b
                    ta = ttnn.from_torch(a, dtype=adtype, device=device, layout=ttnn.TILE_LAYOUT)
                    tb = ttnn.from_torch(b, dtype=bdtype, device=device, layout=ttnn.TILE_LAYOUT)
                    da, db = ttnn.to_torch(ta).double(), ttnn.to_torch(tb).double()
                    represented = da @ (db.transpose(-2, -1) if transpose else db)
                    for fidelity in ("LoFi", "HiFi2", "HiFi4"):
                        for fp32_dst in (False, True):
                            config = ttnn.WormholeComputeKernelConfig(
                                math_fidelity=getattr(ttnn.MathFidelity, fidelity), math_approx_mode=False,
                                fp32_dest_acc_en=fp32_dst, packer_l1_acc=False)
                            tensor = ttnn.matmul(ta, tb, transpose_b=transpose, dtype=ttnn.float32,
                                compute_kernel_config=config, core_grid=ttnn.CoreGrid(y=1, x=1))
                            actual = ttnn.to_torch(tensor).double()
                            assert torch.isfinite(actual).all()
                            record = dict(kind="matmul_probe", operation=operation, representation=representation,
                                fidelity=fidelity, fp32_dst=fp32_dst, output_dtype="FP32", seed=args.seed,
                                arithmetic=M.metrics(actual, represented), end_to_end=M.metrics(actual, original),
                                representation_error=M.metrics(represented, original),
                                output_sha256=hashlib.sha256(actual.float().numpy().tobytes()).hexdigest())
                            output.write(json.dumps(record, allow_nan=False) + "\n")
                            output.flush()
                            print(json.dumps(record, allow_nan=False), flush=True)
                            ttnn.deallocate(tensor)
                    ttnn.deallocate(ta)
                    ttnn.deallocate(tb)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
