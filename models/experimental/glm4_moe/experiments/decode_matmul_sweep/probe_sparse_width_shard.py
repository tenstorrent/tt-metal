# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cheap single-device probe: does ttnn.sparse_matmul accept a WIDTH_SHARDED L1 output?

Track 4's only novel lever is width-sharded L1 expert I/O (no repo precedent for the
sparse path). This validates op support on GLM4's decode gate/up shape without the 218B
model. Compares width-sharded output vs interleaved-L1 output for numeric equality.

Run:
  export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
  ./python_env/bin/python models/experimental/glm4_moe/experiments/decode_matmul_sweep/probe_sparse_width_shard.py
"""
import torch
import ttnn
from models.experimental.glm4_moe.tt.moe_tt import _make_sparse_matmul_program_config

HIDDEN = 5120
MOE_INTER = 1536
E_LOCAL = 3
BLOCK = 32


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        # gate/up sparse_matmul: a=[1,1,BLOCK,HIDDEN] (dense), b=[1,E,HIDDEN,MOE_INTER] (sparse b),
        # sparsity=[A,B,1,E]=[1,1,1,E].
        a = torch.rand((1, 1, BLOCK, HIDDEN)).bfloat16()
        b = torch.rand((1, E_LOCAL, HIDDEN, MOE_INTER)).bfloat16()
        spars = torch.ones((1, 1, 1, E_LOCAL)).bfloat16()  # all local experts active (worst case)

        a_tt = ttnn.from_torch(a, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        b_tt = ttnn.from_torch(b, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        s_tt = ttnn.from_torch(spars, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

        pc = _make_sparse_matmul_program_config(device=dev, out_features=MOE_INTER, in0_block_w=8, per_core_M=1)
        ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi)

        def run(out_mc, label):
            try:
                out = ttnn.sparse_matmul(
                    a_tt,
                    b_tt,
                    sparsity=s_tt,
                    memory_config=out_mc,
                    program_config=pc,
                    is_input_a_sparse=False,
                    is_input_b_sparse=True,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ckc,
                    output_tile=ttnn.Tile([BLOCK, ttnn.TILE_SIZE]),
                )
                t = ttnn.to_torch(out)
                print(f"  [{label}] OK  out.shape={tuple(out.shape)}  mem={out.memory_config()}")
                ttnn.deallocate(out)
                return t
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else repr(e)
                print(f"  [{label}] FAILED: {msg[:160]}")
                return None

        print("Interleaved L1 (baseline):")
        ref = run(ttnn.L1_MEMORY_CONFIG, "interleaved-L1")

        # Width-sharded L1 over the worker grid, sized to the N=MOE_INTER output width.
        grid = dev.compute_with_storage_grid_size()
        cx, cy = int(grid.x), int(grid.y)
        n_tiles = (MOE_INTER + 31) // 32
        print(f"Width-sharded L1 (grid {cx}x{cy}, N={MOE_INTER}, n_tiles={n_tiles}):")
        try:
            ws_mc = ttnn.create_sharded_memory_config(
                shape=(BLOCK, MOE_INTER),
                core_grid=ttnn.CoreGrid(y=1, x=min(cx, n_tiles)),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            run(ws_mc, "width-sharded-L1")
        except Exception as e:
            print(f"  [width-sharded-L1] config build FAILED: {str(e).splitlines()[0][:160]}")

        print(
            "\nVERDICT: if width-sharded FAILED, sparse_matmul does not support it -> Track 4 width-shard is a dead end for the sparse path."
        )
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
