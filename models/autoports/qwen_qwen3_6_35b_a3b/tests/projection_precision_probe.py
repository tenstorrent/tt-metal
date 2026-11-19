# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured same-input packed projection accumulation precision probe."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frontend_precision_probe import compare
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder


def run(args):
    record(args.output)
    torch.set_num_threads(4)
    sd = weights(0)
    probe = torch.load(args.input, weights_only=False)
    x = probe["hf_values"]["input_norm"].contiguous()
    keys = ["linear_attn.in_proj_" + suffix + ".weight" for suffix in ("qkv", "z", "a", "b")]
    packed = torch.cat([sd[k] for k in keys])
    ref32 = torch.nn.functional.linear(x.float(), packed.float())
    ref16 = torch.cat([torch.nn.functional.linear(x, sd[k]) for k in keys], dim=-1)
    rows = []
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        dec = FusedDecoder.from_state_dict(sd, hf_config=config(), layer_idx=0, mesh_device=mesh)
        tx = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        w, im, om, original_pc = dec.decode_linear["attention_packed"]
        sharded = ttnn.to_memory_config(tx, im)
        assert torch.equal(ttnn.to_torch(tx).reshape(x.shape), x), "input upload differs"
        assert torch.equal(ttnn.to_torch(dec.w["attention_packed"]), packed.T), "weight upload differs"
        for packer in (False, True):
            compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=packer,
            )
            for block in (None, 1, 2, 4):
                label = f"generic_packer{packer}" if block is None else f"pipeline_block{block}_packer{packer}"
                row = dict(candidate=label)
                try:
                    if block is None:
                        out = ttnn.linear(
                            tx, dec.w["attention_packed"], dtype=ttnn.float32, compute_kernel_config=compute
                        )
                    else:
                        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=block,
                            per_core_M=1,
                            per_core_N=7,
                            fused_activation=None,
                            num_workers_per_dram_bank=3,
                            use_multicore_pipeline=True,
                        )
                        out = ttnn.linear(
                            sharded,
                            w,
                            dtype=ttnn.float32,
                            compute_kernel_config=compute,
                            memory_config=om,
                            program_config=pc,
                        )
                    got = ttnn.to_torch(out).reshape(ref32.shape)
                    row.update(float32=compare(ref32, got), canonical_bf16=compare(ref16, got.bfloat16()))
                    row["segments"] = {
                        name: compare(ref16[..., lo:hi], got[..., lo:hi].bfloat16())
                        for name, lo, hi in (
                            ("qkv", 0, 8192),
                            ("z", 8192, 12288),
                            ("a", 12288, 12320),
                            ("b", 12320, 12352),
                        )
                    }
                except RuntimeError as exc:
                    row["error"] = str(exc)
                rows.append(row)
                Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
                print("PROJECTION_PRECISION", json.dumps(row), flush=True)
        # Keep values identical while testing native format and reduction boundaries.
        for label, ax, bw in (
            ("fp32_a", ttnn.typecast(tx, ttnn.float32), dec.w["attention_packed"]),
            ("fp32_b", tx, ttnn.typecast(dec.w["attention_packed"], ttnn.float32)),
            ("fp32_ab", ttnn.typecast(tx, ttnn.float32), ttnn.typecast(dec.w["attention_packed"], ttnn.float32)),
        ):
            out = ttnn.linear(ax, bw, dtype=ttnn.float32, compute_kernel_config=dec.compute)
            got = ttnn.to_torch(out).reshape(ref32.shape)
            rows.append(
                dict(candidate=label, float32=compare(ref32, got), canonical_bf16=compare(ref16, got.bfloat16()))
            )
        for chunks in (2, 4, 8):
            width = 2048 // chunks
            out = None
            for i in range(chunks):
                part = ttnn.linear(
                    tx[..., i * width : (i + 1) * width],
                    dec.w["attention_packed"][i * width : (i + 1) * width, :],
                    dtype=ttnn.float32,
                    compute_kernel_config=dec.compute,
                )
                out = part if out is None else ttnn.add(out, part)
            got = ttnn.to_torch(out).reshape(ref32.shape)
            rows.append(
                dict(
                    candidate=f"split_k_{chunks}",
                    float32=compare(ref32, got),
                    canonical_bf16=compare(ref16, got.bfloat16()),
                )
            )
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
        for fidelity in (
            ttnn.MathFidelity.LoFi,
            ttnn.MathFidelity.HiFi2,
            ttnn.MathFidelity.HiFi3,
            ttnn.MathFidelity.HiFi4,
        ):
            compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            out = ttnn.linear(tx, dec.w["attention_packed"], dtype=ttnn.float32, compute_kernel_config=compute)
            got = ttnn.to_torch(out).reshape(ref32.shape)
            rows.append(
                dict(
                    candidate=str(fidelity),
                    compute=str(compute),
                    float32=compare(ref32, got),
                    canonical_bf16=compare(ref16, got.bfloat16()),
                )
            )
        a = torch.diag(x.flatten()[:32]).bfloat16()
        b = packed[:32, :32].T.contiguous()
        ta, tb = [ttnn.from_torch(v, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for v in (a, b)]
        single = ttnn.to_torch(ttnn.matmul(ta, tb, dtype=ttnn.float32, compute_kernel_config=dec.compute))
        expected = a.float() @ b.float()
        rows.append(dict(candidate="single_product_diagonal", float32=compare(expected, single)))
        torch.save(dict(a=a, b=b, reference=expected, actual=single), str(args.output) + ".products.pt")
        column = ttnn.from_torch(x.reshape(2048, 1).float(), device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        products = ttnn.multiply(ttnn.typecast(dec.w["attention_packed"], ttnn.float32), column)
        out = ttnn.sum(products, dim=0, keepdim=True, compute_kernel_config=dec.compute)
        got = ttnn.to_torch(out).reshape(ref32.shape)
        rows.append(
            dict(
                candidate="sfpu_multiply_sum",
                float32=compare(ref32, got),
                canonical_bf16=compare(ref16, got.bfloat16()),
            )
        )
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance_exact_frontend_probe.json.pos3979.pt",
    )
    p.add_argument("--output", required=True)
    run(p.parse_args())
