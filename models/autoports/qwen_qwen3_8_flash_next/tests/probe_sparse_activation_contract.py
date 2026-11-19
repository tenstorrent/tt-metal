# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-op diagnostic: does sparse matmul honor a program's SiLU field?"""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
from models.demos.gpt_oss.tt.experts.config import ProgramConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(812)
    weights = load_weights(a.snapshot, 0, "mlp.experts.gate_up_proj")
    gate = weights["mlp.experts.gate_up_proj"][:, :640].transpose(-1, -2).unsqueeze(0).contiguous()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    rows = []
    try:
        loader = WeightLoader({}, mesh)
        w = loader.tensor(gate)
        active = ttnn.from_torch(
            torch.tensor([1] * 10 + [0] * 502, dtype=torch.bfloat16).reshape(1, 1, 1, 512),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        for n in (1, 32):
            x = loader.tensor((torch.randn(1, 1, n, 2560) * 0.02).bfloat16())
            cfg = ProgramConfig(decode_gate_up_cores=(5, 1), decode_down_cores=(8, 1))

            def run(activation):
                program = cfg.get_decode_gate_up_config(n, 640, k=2560)
                program.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None
                return ttnn.sparse_matmul(
                    x,
                    w,
                    sparsity=active,
                    nnz=None,
                    program_config=program,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=loader.compute,
                )

            plain = run(False)
            expected = ttnn.to_torch(ttnn.silu(plain))[:, :10]
            original = ttnn.to_torch(plain)[:, :10]
            fused = run(True)
            observed = ttnn.to_torch(fused)[:, :10]
            rows.append(
                dict(
                    tokens=n,
                    activation_field_honored=torch.equal(expected, observed),
                    flag_output_equals_plain=torch.equal(original, observed),
                    max_error_vs_silu=float((observed.float() - expected.float()).abs().max()),
                )
            )
            del plain, fused, x
    finally:
        ttnn.close_mesh_device(mesh)
    a.output.write_text(json.dumps(dict(rows=rows, scope=__doc__), indent=2) + "\n")
    print(rows)


if __name__ == "__main__":
    main()
