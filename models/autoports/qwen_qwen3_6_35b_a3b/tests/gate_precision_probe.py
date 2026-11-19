# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-input diagnostic for recurrent softplus and exponential arithmetic."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frontend_precision_probe import compare
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import weights


def run(args):
    record(args.output)
    torch.set_num_threads(8)
    sd = weights(0)
    dt = sd["linear_attn.dt_bias"].float().reshape(1, 1, 32)
    scale = -sd["linear_attn.A_log"].float().exp().reshape(1, 1, 32)
    rows = []
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:

        def upload(x):
            return ttnn.from_torch(x.contiguous(), device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        ts = upload(scale)
        for pos in (128, 129, 130, 1024, 3979, 5011, 7551, 7703, 8157):
            capture = torch.load(f"{args.input}.pos{pos}.pt", weights_only=False)
            a = capture["hf_values"]["projection_a"].float().reshape(1, 1, 32) + dt
            expected = scale * torch.nn.functional.softplus(a)
            ta = upload(a)
            bound = ttnn.minimum(ta, 20.0)
            sp = ttnn.log1p(ttnn.exp(bound, fast_and_approximate_mode=False), fast_and_approximate_mode=False)
            candidates = {
                "fused_softplus": ttnn.multiply(
                    ts, ta, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]
                ),
                "separate_softplus": ttnn.multiply(ts, ttnn.softplus(ta, beta=1.0, threshold=20.0)),
                "log1p_exp": ttnn.multiply(ts, ttnn.where(ttnn.gt(ta, 20.0), ta, sp)),
            }
            for name, value in candidates.items():
                got = ttnn.to_torch(value).reshape(expected.shape)
                exp = ttnn.to_torch(ttnn.exp(value, fast_and_approximate_mode=False)).reshape(expected.shape)
                rows.append(
                    dict(
                        position=pos,
                        candidate=name,
                        g=compare(expected, got),
                        fp32_different=int((expected != got).sum()),
                        exp=compare(expected.exp(), exp),
                        exp_fp32_different=int((expected.exp() != exp).sum()),
                    )
                )
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
        print(
            "GATE_PRECISION",
            [
                (
                    name,
                    sum(r["fp32_different"] for r in rows if r["candidate"] == name),
                    sum(r["exp_fp32_different"] for r in rows if r["candidate"] == name),
                )
                for name in candidates
            ],
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance_precise_probe.json"
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
