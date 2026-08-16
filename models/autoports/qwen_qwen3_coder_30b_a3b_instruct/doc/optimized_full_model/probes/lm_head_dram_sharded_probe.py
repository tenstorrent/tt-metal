import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.optimized_decoder import (
    _DRAM_BANKS,
    _dram_sharded_program_config,
)

K, N = 2048, 37984
print(f"RESULT K={K} N={N}  N/32 tiles = {N/32}  tiles/bank = {N/32/_DRAM_BANKS}  banks={_DRAM_BANKS}")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))


def bench(x, w, pc, mc, iters=50, cc=None):
    def call():
        o = ttnn.linear(x, w, program_config=pc, memory_config=mc, dtype=ttnn.bfloat16, compute_kernel_config=cc)
        ttnn.deallocate(o)

    call()
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(iters):
        call()
    ttnn.synchronize_device(mesh)
    return 1e3 * (time.perf_counter() - t) / iters


try:
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn((1, 1, 32, K)).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    wt = torch.randn((K, N)).float() * 0.02
    w = ttnn.from_torch(
        wt,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    cc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    print(f"RESULT shipped interleaved            {bench(x,w,None,ttnn.DRAM_MEMORY_CONFIG,cc=cc)*1000:8.2f} us")
    # DRAM-sharded, the report's recommendation
    for label, n in [("as-is N=37984", N), ("padded N=38912 (=8*32*152)", 38912)]:
        try:
            wn = (
                w
                if n == N
                else ttnn.from_torch(
                    torch.randn((K, n)).float() * 0.02,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            )
            shard = ttnn.create_sharded_memory_config(
                (1, 1, 32, K),
                core_grid=ttnn.CoreGrid(y=1, x=_DRAM_BANKS),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            xs = ttnn.to_memory_config(x, shard)
            ws = ttnn.to_memory_config(
                wn,
                ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_DRAM_BANKS - 1, 0))}),
                        (K, n // _DRAM_BANKS),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
            out_mc = ttnn.create_sharded_memory_config(
                (1, 1, 32, n),
                core_grid=ttnn.CoreGrid(y=1, x=_DRAM_BANKS),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            ms = bench(xs, ws, _dram_sharded_program_config(K, n), out_mc, cc=cc)
            print(f"RESULT dram-sharded {label:<28} {ms*1000:8.2f} us")
        except Exception as e:
            print(f"RESULT dram-sharded {label:<28} RAISE {str(e).strip().splitlines()[0][:150]}")
            for l in str(e).splitlines():
                if "info:" in l or "must" in l.lower() or "TT_FATAL" in l:
                    print("RESULT   ", l.strip()[:170])
finally:
    ttnn.close_mesh_device(mesh)
