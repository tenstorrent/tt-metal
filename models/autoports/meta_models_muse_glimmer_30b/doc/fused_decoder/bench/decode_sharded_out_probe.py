# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can the decode ``o_proj`` / ``mlp_down`` write straight into the sharded residual?

The decode residual stream is width-sharded in L1, so both projections
currently write DRAM-interleaved and are immediately followed by an
``interleaved_to_sharded`` into ``norm_memcfg``.  The decode QKV projection
already writes to L1 directly (rewrite H), so the same merge is worth asking
for here: if ``ttnn.linear`` accepts the width-sharded output memory config,
two ``InterleavedToShardedDeviceOperation`` disappear from every decode step.

Measured on device kernel time (run under ``python -m tracy -r -p -v``), the
DRAM matmul + reshard pair against the single sharded-output matmul, at the
shipped decode shapes and the shipped 4x2 norm grid.
"""
from __future__ import annotations

import torch

import ttnn

BATCH = 32
HIDDEN = 6656
NORM_GRID = (4, 2)  # choose_decode_norm_grid(6656, 11x10)
CASES = [("o_proj  ", 4096), ("mlp_down", 19968)]
REPS = 16


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        gx, gy = NORM_GRID
        cores = gx * gy
        norm_memcfg = ttnn.create_sharded_memory_config(
            shape=(BATCH, HIDDEN // cores),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        for label, k in CASES:
            torch.manual_seed(0)
            a = torch.randn(1, 1, BATCH, k).to(torch.bfloat16) * 0.1
            b = torch.randn(1, 1, k, HIDDEN).to(torch.bfloat16) * 0.02
            ta = ttnn.from_torch(
                a, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            tb = ttnn.from_torch(
                b, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ref = a.float() @ b.float()

            # Every device call this script makes is announced, so
            # ``summarize_device_probe.py`` can slice the ops CSV back apart.
            print(f"GROUP {2 * REPS} {label} shipped_dram_then_reshard", flush=True)
            for _ in range(REPS):
                out = ttnn.linear(ta, tb, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                sh = ttnn.interleaved_to_sharded(out, norm_memcfg)
                ttnn.deallocate(out)
                ttnn.deallocate(sh)
            ttnn.synchronize_device(mesh)

            # Does the requested shard spec survive?  ``ttnn.linear`` picks its
            # own program config, and the output shard grid comes from *that*,
            # not from the memory config handed in.
            print(f"GROUP 1 {label} shardspec_probe", flush=True)
            probe_out = ttnn.linear(ta, tb, dtype=ttnn.bfloat16, memory_config=norm_memcfg)
            print(
                f"SHARDSPEC {label} requested {norm_memcfg.shard_spec.grid} "
                f"got {probe_out.memory_config().shard_spec.grid}",
                flush=True,
            )
            ttnn.deallocate(probe_out)

            # Adapt and retry: force the matmul onto the norm's 4x2 grid with an
            # explicit 1D width-sharded program config, which is the only way to
            # make the output shard spec match what the norm requires.
            per_core_n = (HIDDEN // cores) // 32  # 26 tiles at the 4x2 grid
            adapted = None
            for in0_block_w in (1, 2, 4, 8, 16):
                if (k // 32) % in0_block_w:
                    continue
                for sub_w in (2, 1):
                    if per_core_n % sub_w:
                        continue
                    cand = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=in0_block_w,
                        out_subblock_h=1,
                        out_subblock_w=sub_w,
                        per_core_M=1,
                        per_core_N=per_core_n,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                    print(f"GROUP 1 {label} search_in0w{in0_block_w}_sw{sub_w}", flush=True)
                    try:
                        ttnn.deallocate(
                            ttnn.linear(ta, tb, dtype=ttnn.bfloat16, memory_config=norm_memcfg, program_config=cand)
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"BLOCKED {label} search_in0w{in0_block_w}_sw{sub_w}: "
                            f"{' '.join(str(exc).split())[:170]}",
                            flush=True,
                        )
                        continue
                    print(f"ADAPTED {label} in0_block_w={in0_block_w} sub_w={sub_w} runs", flush=True)
                    adapted = cand
                    break
                if adapted is not None:
                    break
            if adapted is None:
                for t in (ta, tb):
                    ttnn.deallocate(t)
                continue
            print(f"GROUP 1 {label} pcc_probe", flush=True)
            try:
                out = ttnn.linear(ta, tb, dtype=ttnn.bfloat16, memory_config=norm_memcfg, program_config=adapted)
                pcc = torch.nn.functional.cosine_similarity(
                    ref.flatten().float(), ttnn.to_torch(out).flatten().float(), dim=0
                ).item()
                ttnn.deallocate(out)
                print(
                    f"PROBE {label} adapted_4x2 OK cos={pcc:.6f} grid={out.memory_config().shard_spec.grid}", flush=True
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"BLOCKED {label} adapted_4x2: {type(exc).__name__}: {' '.join(str(exc).split())[:260]}", flush=True
                )
                for t in (ta, tb):
                    ttnn.deallocate(t)
                continue

            print(f"GROUP {REPS} {label} adapted_4x2", flush=True)
            for _ in range(REPS):
                ttnn.deallocate(
                    ttnn.linear(ta, tb, dtype=ttnn.bfloat16, memory_config=norm_memcfg, program_config=adapted)
                )
            ttnn.synchronize_device(mesh)
            for t in (ta, tb):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
