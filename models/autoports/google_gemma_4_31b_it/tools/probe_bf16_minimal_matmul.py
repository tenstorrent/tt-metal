# SPDX-License-Identifier: Apache-2.0
"""Serialized standalone chip-0 BF16/FP32 projection regression; never resets devices.

Run only after checking device ownership. --include-gate adds the larger L1 case.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-gate", action="store_true")
    args = parser.parse_args()
    import torch
    from ttnn.types import BlackholeComputeKernelConfig

    import ttnn

    torch.manual_seed(195)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
    results = []
    try:
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        dram = mesh.dram_grid_size()
        banks = dram.x * dram.y
        if banks != 8:
            raise RuntimeError(f"Expected 8 DRAM banks, got {banks}")
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
        kernel = BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            packer_l1_acc=True,
        )
        config = ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        )

        def upload(host, memory):
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        shapes = [(32, 5376, 4096, False), (64, 5376, 4096, False)]
        if args.include_gate:
            shapes.append((1024, 5376, 5632, True))
        for m, k, n, gate in shapes:
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(cores, (k, n // banks), ttnn.ShardOrientation.ROW_MAJOR),
            )
            # Both generations stay allocated: cache-hit pointers must be genuinely different.
            tensors = []
            cases = []
            residual_host = torch.randn(1, 1, m, k).bfloat16()
            residual = upload(residual_host, ttnn.L1_MEMORY_CONFIG)
            tensors.append(residual)
            if gate:
                live_gate = upload(torch.randn(1, 1, m, n).bfloat16(), ttnn.L1_MEMORY_CONFIG)
                tensors.append(live_gate)
            for repeat in range(2):
                a = torch.randn(1, 1, m, k).bfloat16()
                b = (torch.randn(1, 1, k, n) / k**0.5).bfloat16()
                x, w = upload(a, ttnn.L1_MEMORY_CONFIG), upload(b, memory)
                tensors.extend([x, w])
                cases.append((a, b, x, w))
            cached = None
            for repeat, (a, b, x, w) in enumerate(cases):
                out = ttnn.experimental.minimal_matmul(
                    input_tensor=x,
                    weight_tensor=w,
                    config=config,
                    compute_kernel_config=kernel,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG if gate else ttnn.DRAM_MEMORY_CONFIG,
                )
                actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
                expected = a.float() @ b.float()
                pcc = float(torch.corrcoef(torch.stack([expected.flatten(), actual.flatten()]))[0, 1])
                rrmse = float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt())
                entries = mesh.num_program_cache_entries()
                if cached is not None and entries != cached:
                    raise AssertionError("Second call compiled a new program instead of hitting cache")
                cached = entries
                # Live residual must survive kernel CB allocations unchanged.
                observed_residual = ttnn.to_torch(residual, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                residual_ok = torch.equal(observed_residual, residual_host)
                row = dict(
                    m=m,
                    k=k,
                    n=n,
                    output="L1" if gate else "DRAM",
                    repeat=repeat,
                    pcc=pcc,
                    rrmse=rrmse,
                    cache_entries=entries,
                    live_residual_unchanged=residual_ok,
                )
                results.append(row)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                if not (pcc > 0.9995 and rrmse < 0.02 and residual_ok):
                    raise AssertionError("BF16 minimal_matmul accuracy/residual check failed")
                ttnn.deallocate(out)
            for tensor in reversed(tensors):
                ttnn.deallocate(tensor)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
