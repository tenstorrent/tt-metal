# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: can ``paged_fused_update_cache`` be reached without a reshard?

That op asserts its two update tensors live on disjoint cores
(``paged_fused_update_cache_device_operation.cpp:341-348``).
``nlp_create_qkv_heads_decode`` emits V on Q's grid unconditionally
(``nlp_create_qkv_heads_decode_device_operation.cpp:152``) but has an
``overlap_qk_coregrid=False`` mode that moves **K** to a disjoint grid, which
would make K and V disjoint and the fused cache write legal for free — but the
frontend *drops* that flag for an interleaved input
(``nlp_create_qkv_heads_decode.cpp:23``), and the device op then constrains it
to a width-sharded QKV whose shard width divides ``head_dim``.

This probe checks whether that mode is reachable from the layout this layer's
decode QKV projection actually produces (L1 *interleaved*, which is what
``nlp_create_qkv_heads_decode`` needs after the Blackhole tt-metal #16667
workaround), and prints the resulting Q/K/V core grids either way.

Output: ``doc/fused_decoder/logs/kv_coregrid_probe.log``.
"""

from __future__ import annotations

import torch

import ttnn

QKV_WIDTH = 4608  # 32 q heads + 2 kv heads + 2 kv heads, head_dim 128
NUM_HEADS = 32
NUM_KV_HEADS = 2


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        print(f"PROBE device compute grid = {grid.x}x{grid.y} = {grid.x * grid.y} cores", flush=True)
        for batch in (1, 4, 32):
            x = torch.randn(1, 1, batch, QKV_WIDTH).to(torch.bfloat16)
            tx = ttnn.from_torch(
                x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            for overlap in (True, False):
                try:
                    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                        tx,
                        num_heads=NUM_HEADS,
                        num_kv_heads=NUM_KV_HEADS,
                        overlap_qk_coregrid=overlap,
                        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    )
                    kg = k.memory_config().shard_spec.grid
                    vg = v.memory_config().shard_spec.grid
                    print(
                        f"PROBE batch={batch:2d} overlap_qk_coregrid={overlap!s:5s} "
                        f"q={q.memory_config().shard_spec.grid} k={kg} v={vg} "
                        f"k_and_v_disjoint={str(kg) != str(vg)}",
                        flush=True,
                    )
                    try:
                        cache = ttnn.from_torch(
                            torch.zeros(batch, NUM_KV_HEADS, 64, 128),
                            device=mesh,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        pos = ttnn.from_torch(
                            torch.zeros(batch, dtype=torch.int32),
                            device=mesh,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            dtype=ttnn.int32,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        ttnn.experimental.paged_fused_update_cache(cache, k, cache, v, update_idxs_tensor=pos)
                        print(f"PROBE batch={batch:2d} overlap={overlap!s:5s} paged_fused_update_cache OK", flush=True)
                        ttnn.deallocate(cache)
                        ttnn.deallocate(pos)
                    except Exception as exc:  # noqa: BLE001
                        detail = next(
                            (l.strip() for l in str(exc).splitlines() if "must" in l and "overlap" in l),
                            str(exc).splitlines()[0],
                        )
                        print(
                            f"PROBE batch={batch:2d} overlap={overlap!s:5s} paged_fused_update_cache "
                            f"REJECTED: {detail[:150]}",
                            flush=True,
                        )
                    for t in (q, k, v):
                        ttnn.deallocate(t)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"PROBE batch={batch:2d} overlap_qk_coregrid={overlap!s:5s} "
                        f"create_heads FAILED: {str(exc)[:200]}",
                        flush=True,
                    )
            ttnn.deallocate(tx)

            # The same call from a WIDTH_SHARDED QKV, which is the only layout
            # for which the op honours overlap_qk_coregrid=False at all — the
            # frontend drops it for an interleaved input
            # (nlp_create_qkv_heads_decode.cpp:23).  That layout is what a
            # DRAM-sharded decode matmul would produce, i.e. the
            # optimized-decoder stage.
            try:
                # overlap_qk_coregrid=False additionally requires
                # head_dim % shard_width == 0 (no partial heads in a shard), so
                # the shard width must be 128 -> 4608/128 = 36 cores (9x4).
                cores = QKV_WIDTH // 128
                width_sharded = ttnn.create_sharded_memory_config(
                    shape=(32, 128),
                    core_grid=ttnn.CoreGrid(y=cores // 9, x=9),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                txs = ttnn.from_torch(
                    x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=width_sharded
                )
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    txs,
                    num_heads=NUM_HEADS,
                    num_kv_heads=NUM_KV_HEADS,
                    overlap_qk_coregrid=False,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                )
                kg, vg = k.memory_config().shard_spec.grid, v.memory_config().shard_spec.grid
                print(
                    f"PROBE batch={batch:2d} WIDTH_SHARDED input, overlap=False: k={kg} v={vg} "
                    f"k_and_v_disjoint={str(kg) != str(vg)}",
                    flush=True,
                )
                for t in (q, k, v, txs):
                    ttnn.deallocate(t)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"PROBE batch={batch:2d} WIDTH_SHARDED input, overlap=False: FAILED "
                    f"{str(exc).splitlines()[0][:80]} {str(exc).splitlines()[2][:130] if len(str(exc).splitlines()) > 2 else ''}",
                    flush=True,
                )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
