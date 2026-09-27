# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact decode-shaped comparison of sharded and interleaved native RoPE."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedAttention
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _LocalAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format-controls", action="store_true")
    parser.add_argument("--adapted", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    results = []
    try:
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(value, dtype=ttnn.float32):
            return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

        configs = (
            [
                (64, True, False, "float32", "float32"),
                (64, True, False, "bfloat16", "bfloat16"),
                (64, False, False, "bfloat16", "bfloat16"),
                (128, True, False, "bfloat16", "bfloat16"),
                (256, True, False, "bfloat16", "bfloat16"),
                (256, False, False, "bfloat16", "bfloat16"),
                (512, False, False, "bfloat16", "bfloat16"),
                (128, True, False, "float32", "bfloat16"),
            ]
            if args.format_controls
            else [
                (w, f, s, "float32", "float32")
                for w, f, s in (
                    (64, True, False),
                    (128, True, False),
                    (256, True, False),
                    (256, True, True),
                    (256, False, False),
                    (512, True, False),
                    (512, False, False),
                )
            ]
        )
        if args.adapted:
            configs = [(width, True, False, "float32", "float32") for width in (256, 512)]
        for width, fp32, sync, input_dtype, table_dtype in configs:
            compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=False,
                dst_full_sync_en=sync,
            )
            target = SimpleNamespace(compute=compute, fuse_rope=True, sharded_decode_rope=True)
            for heads in (2,) if args.format_controls else (1, 2, 4):
                value = torch.randn(1, 1, heads, width)
                if input_dtype == "bfloat16":
                    value = value.bfloat16().float()
                theta = torch.randn(1, 1, 1, width // 2).repeat(1, 1, 1, 2)
                cos, sin = (getattr(theta, name)().bfloat16().float() for name in ("cos", "sin"))
                golden = value * cos + torch.cat((-value[..., width // 2 :], value[..., : width // 2]), dim=-1) * sin
                x = upload(value, getattr(ttnn, input_dtype))
                c, s = (upload(v, getattr(ttnn, table_dtype)) for v in (cos, sin))
                reference = FusedAttention.rotary(target, x, c, s, decode=True)
                if not args.adapted:
                    memory = ttnn.create_sharded_memory_config(
                        (32, width),
                        ttnn.CoreGrid(x=1, y=1),
                        ttnn.ShardStrategy.HEIGHT,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    actual = ttnn.experimental.rotary_embedding_hf(
                        ttnn.to_memory_config(x, memory),
                        ttnn.to_memory_config(c, memory),
                        ttnn.to_memory_config(s, memory),
                        is_decode_mode=True,
                        compute_kernel_config=compute,
                    )
                else:
                    actual = _LocalAttention.rotary(target, x, c, s, decode=True)
                a = [ttnn.to_torch(v).float() for v in ttnn.get_device_tensors(actual)]
                b = ttnn.to_torch(ttnn.get_device_tensors(reference)[0]).float()
                finite = bool(torch.isfinite(a[0]).all())
                record = dict(
                    width=width,
                    heads=heads,
                    input_dtype=input_dtype,
                    table_dtype=table_dtype,
                    fp32_dest_acc_en=fp32,
                    requested_full_sync=sync,
                    finite=finite,
                    replicas_equal=all(torch.equal(a[0], v) for v in a[1:]),
                    baseline_vs_cpu_pcc=torch.corrcoef(torch.stack((b.flatten().double(), golden.flatten().double())))[
                        0, 1
                    ].item(),
                    sharded_vs_cpu_pcc=torch.corrcoef(
                        torch.stack((a[0].flatten().double(), golden.flatten().double()))
                    )[0, 1].item(),
                    max_abs_difference=float((a[0] - b).abs().max()),
                    changed=int((a[0] != b).sum()),
                )
                results.append(record)
                print(json.dumps(record), flush=True)
                del x, c, s, reference, actual
    finally:
        ttnn.close_mesh_device(mesh)
    report = dict(
        runtime_sha256=hashlib.sha256(
            Path(__file__).parents[1].joinpath("tt/multichip_decoder.py").read_bytes()
        ).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=results,
        adapted=args.adapted,
        passed=all(r["replicas_equal"] and r["finite"] and r["sharded_vs_cpu_pcc"] >= 0.999 for r in results),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.adapted:
        assert report["passed"], report


if __name__ == "__main__":
    main()
