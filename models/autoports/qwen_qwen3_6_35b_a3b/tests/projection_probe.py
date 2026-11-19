# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Discriminate packed-projection arithmetic, bias and layout regressions."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder


def run(args):
    record(args.output)
    torch.set_num_threads(8)
    torch.manual_seed(41)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        dec = FusedDecoder.from_state_dict(
            weights(args.layer), hf_config=config(), layer_idx=args.layer, mesh_device=mesh
        )
        for key, (w, im, om, pc) in dec.decode_linear.items():
            x = torch.randn(1, 1, w.shape[-2], dtype=torch.bfloat16) * 0.05
            tx = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            bias = dec.attention_bias if key == "attention_packed" else None
            dtype = ttnn.float32 if key == "attention_packed" and dec.kind == "linear_attention" else ttnn.bfloat16
            ref = ttnn.to_torch(ttnn.linear(tx, dec.w[key], bias=bias, dtype=dtype, compute_kernel_config=dec.compute))
            fused_bias = bias
            if bias is not None and args.shard_bias:
                spec = w.memory_config().shard_spec
                bias_mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(spec.grid, (32, spec.shape[1]), ttnn.ShardOrientation.ROW_MAJOR),
                )
                fused_bias = ttnn.to_memory_config(bias, bias_mem)
            for use_bias in (True, False) if bias is not None else (False,):
                out = ttnn.linear(
                    ttnn.to_memory_config(tx, im),
                    w,
                    bias=fused_bias if use_bias else None,
                    dtype=dtype,
                    compute_kernel_config=dec.compute,
                    memory_config=om,
                    program_config=pc,
                )
                out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
                if bias is not None and not use_bias:
                    out = ttnn.add(out, bias)
                got = ttnn.to_torch(out)
                bounds = (
                    [(0, 8192), (8192, 12288), (12288, 12320), (12320, 12352)]
                    if dtype == ttnn.float32
                    else [(0, got.shape[-1])]
                )
                for lo, hi in bounds:
                    a, b = ref[..., lo:hi], got[..., lo:hi]
                    rows.append(
                        dict(
                            key=key,
                            fused_bias=use_bias,
                            begin=lo,
                            end=hi,
                            pcc=pcc(a, b),
                            max_abs=float((a - b).abs().max()),
                            ref_abs=float(a.abs().max()),
                            got_abs=float(b.abs().max()),
                        )
                    )
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard-bias", action="store_true")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--output", required=True)
    run(p.parse_args())
