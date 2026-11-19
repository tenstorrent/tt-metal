# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Zero-update identity-state discriminator for native FP32 recurrent arithmetic."""
import argparse
import json
import math
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import pcc


def run(args):
    record(args.output)
    torch.manual_seed(831)
    torch.set_num_threads(8)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:

        def up(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        initial = torch.randn(1, 32, 128, 128) * 0.1
        state = up(initial)
        q = up(torch.zeros(1, 1, 16, 128))
        v = up(torch.zeros(1, 1, 32, 128))
        g = up(torch.full((1, 1, 32), args.decay))
        beta = up(torch.zeros(1, 1, 32))
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder

        rows = []
        for step in range(1, args.steps + 1):
            if args.sfpu:
                out, new = FusedDecoder._recurrent_step(None, q, q, v, g, beta, state)
            else:
                out, new = ttnn.transformer.fused_recurrent_gated_delta_rule(
                    q, q, v, g, beta, initial_state=state, output_final_state=True, compute_kernel_config=compute
                )
            ttnn.copy(ttnn.reshape(new, (1, 32, 128, 128)), state)
            del out, new
            if step in (1, 32, 128, args.steps):
                got = ttnn.to_torch(state).reshape(initial.shape)
                expected = initial * math.exp(args.decay * step)
                rows.append(
                    dict(
                        step=step,
                        decay=args.decay,
                        pcc=pcc(expected, got),
                        relative_l2=float((got - expected).norm() / expected.norm()),
                        norm_ratio=float(got.norm() / expected.norm()),
                        max_abs=float((got - expected).abs().max()),
                        bit_equal=torch.equal(expected, got),
                    )
                )
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sfpu", action="store_true")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--decay", type=float, default=0.0)
    p.add_argument("--output", required=True)
    run(p.parse_args())
