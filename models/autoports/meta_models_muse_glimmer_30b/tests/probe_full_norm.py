# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small real-input normalization reproducer extracted from full-model prefill."""

import argparse
import json
import traceback
from pathlib import Path

import torch

import ttnn


def main(args):
    payload = torch.load(args.input, weights_only=False)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    results = []
    try:
        weight = ttnn.from_torch(
            payload["weight"],
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for rows in (32, 64):
            source = payload["input"][:, :, :rows].contiguous()
            expected = source.float() * torch.rsqrt(source.float().square().mean(-1, keepdim=True) + payload["epsilon"])
            expected *= payload["weight"].flatten().float()
            for inplace, fp32, fidelity in (
                (True, True, "HiFi2"),
                (False, True, "HiFi2"),
                (True, False, "HiFi2"),
                (True, True, "HiFi4"),
            ):
                cores = ttnn.num_cores_to_corerangeset(args.cores, ttnn.CoreCoord(8, 8), row_wise=True)
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, [rows, 6656 // args.cores], ttnn.ShardOrientation.ROW_MAJOR),
                )
                x = ttnn.from_torch(
                    source,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                config = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                    math_approx_mode=False,
                    fp32_dest_acc_en=fp32,
                    packer_l1_acc=True,
                )
                y = ttnn.rms_norm(
                    x,
                    epsilon=payload["epsilon"],
                    weight=weight,
                    program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=[8, 8],
                        subblock_w=1,
                        block_h=rows // 32,
                        block_w=6656 // args.cores // 32,
                        inplace=inplace,
                    ),
                    compute_kernel_config=config,
                    memory_config=memory,
                )
                actual = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
                row = dict(
                    rows=rows,
                    cores=args.cores,
                    inplace=inplace,
                    fp32=fp32,
                    fidelity=fidelity,
                    finite=bool(torch.isfinite(actual).all()),
                    min=actual.min().item(),
                    max=actual.max().item(),
                    cosine=torch.nn.functional.cosine_similarity(
                        actual.flatten().double(), expected.flatten().double(), dim=0
                    ).item(),
                    max_abs=(actual - expected).abs().max().item(),
                )
                results.append(row)
                print("NORM_CONTROL", row, flush=True)
                del x, y
        args.output.write_text(
            json.dumps(dict(source=str(args.input), original_config=payload["config"], rows=results), indent=2) + "\n"
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", type=int, choices=(8, 13, 26, 52), default=52)
    main(parser.parse_args())
