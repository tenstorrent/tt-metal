# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate distributed RMSNorm grid/precision effects from attention and sampling."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric
from models.common.modules.tt_ccl import TT_CCL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="1x128,1x512,1x2048,1x4096,16x128,8x4096,16x4096")
    parser.add_argument("--output", type=Path, required=True)
    # Pre-2D is explicitly opt-in: multi-row/core cases can hang this runtime.
    parser.add_argument(
        "--variants", default="default_bf16,padded_full_grid_bf16,post_rect_bf16,default_fp32,accurate_fp32_input"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(123)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    ccl = TT_CCL(mesh)
    grid = mesh.compute_with_storage_grid_size()
    report = dict(grid=[grid.x, grid.y], rows=[])
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    def upload(x, shard=False):
        return ttnn.from_torch(
            x.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if shard else ttnn.ReplicateTensorToMesh(mesh),
        )

    def gather(x):
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(1),
        )

    def metrics(actual, ref):
        error = actual - ref
        row_relative = error.square().mean(-1).sqrt() / ref.square().mean(-1).sqrt().clamp_min(1e-12)
        stride = max(1, actual.numel() // 1000000)
        return dict(
            finite=bool(torch.isfinite(actual).all()),
            sampled_pcc_fp64=torch.corrcoef(
                torch.stack([actual.flatten()[::stride].double(), ref.flatten()[::stride].double()])
            )[0, 1].item(),
            max_abs=error.abs().max().item(),
            relative_l2=(error.norm() / ref.norm()).item(),
            worst_row_relative_l2=row_relative.max().item(),
            exact=torch.equal(actual, ref),
        )

    try:
        weight = (1 + torch.randn(1, 1, 1, 5120) * 0.1).bfloat16()
        wlocal, wfull = upload(weight, True), upload(weight)
        for shape in args.shapes.split(","):
            batch, length = map(int, shape.split("x"))
            # Distinct rows and chip slices expose misrouted or omitted work.
            host = torch.randn(1, batch, length, 5120)
            host += torch.arange(4).repeat_interleave(1280).reshape(1, 1, 1, -1) * 0.2
            host *= torch.logspace(-1, 1, batch * length).reshape(1, batch, length, 1)
            host = host.bfloat16()
            x, full = upload(host, True), upload(host)
            hf = host.float()
            reference = (
                (hf * torch.rsqrt(hf.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).bfloat16().float()
            )
            norm_full = ttnn.rms_norm(
                full, weight=wfull, epsilon=1e-6, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            reference_tt = ttnn.to_torch(ttnn.get_device_tensors(norm_full)[0]).float()
            baseline_error = metrics(reference_tt, reference)
            tile_rows = batch * ((length + 31) // 32)
            active = min(tile_rows, grid.x * grid.y)
            default_actual = None
            for name, pre2d, post2d, dtype in (
                ("default_bf16", False, False, ttnn.bfloat16),
                ("rect_bf16", True, True, ttnn.bfloat16),
                ("pre_rect_bf16", True, False, ttnn.bfloat16),
                ("post_rect_bf16", False, True, ttnn.bfloat16),
                ("default_fp32", False, False, ttnn.float32),
                ("rect_fp32", True, True, ttnn.float32),
                ("accurate_fp32_input", False, False, ttnn.float32),
                ("padded_full_grid_bf16", False, False, ttnn.bfloat16),
            ):
                if name not in args.variants.split(","):
                    continue
                times = []
                norm_input = ttnn.typecast(x, ttnn.float32) if name == "accurate_fp32_input" else x
                post_input = x
                if name == "padded_full_grid_bf16":
                    padded_rows = (
                        ((batch * length + grid.x * grid.y * 32 - 1) // (grid.x * grid.y * 32)) * grid.x * grid.y * 32
                    )
                    norm_input = upload(
                        torch.nn.functional.pad(host.reshape(1, 1, -1, 5120), (0, 0, 0, padded_rows - batch * length)),
                        True,
                    )
                    post_input = norm_input
                for repeat in range(2):
                    print("NORM_BEGIN", shape, name, repeat, flush=True)
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    stats = ttnn.rms_norm_pre_all_gather(
                        norm_input, dtype=dtype, compute_kernel_config=compute, use_2d_core_grid=pre2d
                    )
                    if repeat == 0:
                        ttnn.synchronize_device(mesh)
                        print("NORM_PRE_DONE", flush=True)
                    stats = gather(stats)
                    if repeat == 0:
                        ttnn.synchronize_device(mesh)
                        print("NORM_GATHER_DONE", flush=True)
                    out = ttnn.rms_norm_post_all_gather(
                        post_input,
                        stats,
                        epsilon=1e-6,
                        weight=wlocal,
                        compute_kernel_config=compute,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        use_2d_core_grid=post2d,
                    )
                    ttnn.synchronize_device(mesh)
                    times.append(time.perf_counter() - begin)
                if name == "padded_full_grid_bf16":
                    out = ttnn.reshape(out[:, :, : batch * length, :], [1, batch, length, 1280])
                actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
                if name == "default_bf16":
                    default_actual = actual.clone()
                row = dict(
                    batch=batch,
                    length=length,
                    variant=name,
                    seconds=times,
                    tile_rows=tile_rows,
                    exact_to_default=None if default_actual is None else torch.equal(actual, default_actual),
                    default_active_cores=active,
                    default_active_rectangle=active <= grid.x or active % grid.x == 0,
                    replicated_versus_torch=baseline_error,
                    versus_torch=metrics(actual, reference),
                    versus_replicated=metrics(actual, reference_tt),
                )
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("NORM_GRID", json.dumps(row), flush=True)
                del actual, out, stats
            del host, hf, reference, reference_tt, norm_full, x, full, default_actual
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
