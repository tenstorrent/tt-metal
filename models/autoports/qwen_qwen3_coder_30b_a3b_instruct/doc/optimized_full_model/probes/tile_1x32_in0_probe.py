import sys

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.optimized_decoder import (
    EXPERT_IN0_BLOCK_W_GATE_UP,
    _tuned_sparse_matmul_config,
)

E, H, I, B = 32, 2048, 768, 1
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
try:
    torch.manual_seed(0)
    t1 = ttnn.Tile([1, 32])
    x = ttnn.from_torch(
        torch.randn((1, B, 1, H)).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        tile=t1,
    )
    print("in0 with Tile([1,32]) built:", x.shape, x.tile if hasattr(x, "tile") else "")
    w = ttnn.from_torch(
        torch.randn((1, E, H, 2 * I)).float() * 0.02,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    r = torch.zeros((1, 1, B, E))
    r[..., :2] = 0.5
    sp = ttnn.from_torch(
        r,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out = ttnn.sparse_matmul(
        x,
        w,
        sparsity=sp,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        program_config=_tuned_sparse_matmul_config(1, 2 * I, H, EXPERT_IN0_BLOCK_W_GATE_UP),
        output_tile=t1,
        dtype=ttnn.bfloat16,
    )
    print("RESULT produced", out.shape)
    for nm, f in [
        ("reshape", lambda: ttnn.reshape(out, (B, E, 2 * I)).shape),
        ("sum", lambda: ttnn.sum(out, dim=1).shape),
        ("untilize", lambda: ttnn.untilize(out).shape),
        ("silu", lambda: ttnn.silu(out).shape),
        ("to_torch", lambda: ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).shape),
    ]:
        try:
            print(f"RESULT   {nm:<10} OK {f()}")
        except Exception as e:
            print(f"RESULT   {nm:<10} RAISE {str(e).strip().splitlines()[0][:120]}")
except Exception as e:
    print("RESULT FAILED:", str(e).strip().splitlines()[0][:200])

    m = [l for l in str(e).splitlines() if "info:" in l or "must" in l][:3]
    for l in m:
        print("RESULT  ", l.strip()[:180])
finally:
    ttnn.close_mesh_device(mesh)
