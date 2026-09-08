# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Can the SHARDED nlp_create_qkv_heads run at prefill seq 64/128? (PERF_NOTES 6.6)

Prefill spends 47-49 us/layer in `nlp_create_qkv_heads` — 1.3 ms of the 16.9 ms
prefill — and it is flat in sequence length because the interleaved factory's unit of
work is a tile-row of the SEQUENCE
(`nlp_create_qkv_heads_program_factory.cpp:30`: `num_blocks = shape[0]*shape[1]*shape[2]
/ TILE_HEIGHT`), so the 4096-wide axis contributes no parallelism. The sharded factory
parallelises over HEADS instead and runs the same split in ~3 us.

Two things blocked it, both on the CALLER's side:

1. `attention.py` builds `q_grid` as ``CoreRange((0,0), (num_heads-1, 0))`` — a
   16-wide single ROW. On an 8x8 compute grid x=8..15 do not exist, so that spec is
   illegal for 16 heads. `concat_grid` three lines above already does it correctly with
   `ttnn.num_cores_to_corerangeset`. This probe uses the legal grid.

2. The op's `compute_output_specs` ignores the shard spec you hand it in
   `memory_config` and rebuilds ``{TILE_HEIGHT, head_dim}``
   (`nlp_create_qkv_heads_device_operation.cpp:229`). One tile-row per head means
   seq=64 needs 2 shards per head = 32 shards on a 16-core grid, and it dies. BUT the
   same function returns the caller's specs verbatim when three output tensors are
   supplied (`:201`), and the Python binding exposes that as `output_tensors=`
   (`nlp_create_qkv_heads_nanobind.cpp:35`). So the caller can ask for
   ``{seq, head_dim}`` and bypass the hardcode entirely — no ttnn change.

None of the sharded-input TT_FATALs constrain the output shard height, so this is
legal on paper. What it does NOT tell us is whether the sharded KERNEL addresses more
than one tile-row per head correctly: `build_sharded_core_args` derives
`k_num_tiles` from the shard shape (so it scales), but the per-head offsets are built
from `head_size = head_tiles * single_tile_size`, which is one tile-row's worth. That
is a question about generated addresses, so this probe answers it the only reliable
way — by comparing against torch's own split.

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_qkv_split_outtensors.py
"""

from __future__ import annotations

import pytest
import torch

import ttnn

TILE = 32
HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
FUSED_QKV = (HEADS + 2 * KV_HEADS) * HEAD_DIM  # 4096
REPS = 5


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)


def _why(e: Exception) -> str:
    for line in str(e).splitlines():
        s = line.strip()
        if "TT_FATAL" in s or "must" in s or "Statically allocated" in s:
            return s[:170]
    return (str(e).splitlines() or [repr(e)])[0][:170]


def _hs(grid, m, w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (m, w), ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("m", [32, 64, 128], ids=["seq32", "seq64", "seq128"])
def test_sharded_split_via_output_tensors(device, m):
    cg = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    x = torch.randn(1, 1, m, FUSED_QKV, dtype=torch.bfloat16)

    # Reference: the layout nlp_create_qkv_heads produces from a KV-group-interleaved
    # fused QKV is just the plain [Q | K | V] split when the input is plain, so build the
    # reference from the interleaved op itself AND from torch, and require both.
    q_ref = x[..., : HEADS * HEAD_DIM].reshape(1, m, HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    k_ref = (
        x[..., HEADS * HEAD_DIM : (HEADS + KV_HEADS) * HEAD_DIM]
        .reshape(1, m, KV_HEADS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    v_ref = x[..., (HEADS + KV_HEADS) * HEAD_DIM :].reshape(1, m, KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()

    # The sharded kernel reads a KV-GROUP-INTERLEAVED fused QKV -- [q..q, k, v] per KV
    # group -- not the plain [Q|K|V] the interleaved kernel takes (PERF_NOTES 2.1; the
    # model keeps a second permuted weight copy `wqkv_kvgi` for exactly this). Feeding it
    # the plain order produces a valid-looking but wrong answer, so build the permuted
    # input here. The OUTPUT head order is the same either way, so the reference stands.
    q_per_kv = HEADS // KV_HEADS
    groups = []
    for g in range(KV_HEADS):
        for j in range(q_per_kv):
            h = g * q_per_kv + j
            groups.append(x[..., h * HEAD_DIM : (h + 1) * HEAD_DIM])
        groups.append(x[..., (HEADS + g) * HEAD_DIM : (HEADS + g + 1) * HEAD_DIM])
        groups.append(x[..., (HEADS + KV_HEADS + g) * HEAD_DIM : (HEADS + KV_HEADS + g + 1) * HEAD_DIM])
    x_kvgi = torch.cat(groups, dim=-1).contiguous()

    qkv_shard_w = (HEADS // KV_HEADS + 2) * HEAD_DIM  # 512
    qkv_cores = FUSED_QKV // qkv_shard_w  # 8
    qkv_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(qkv_cores - 1, 0))})
    in_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(qkv_grid, (m, qkv_shard_w), ttnn.ShardOrientation.ROW_MAJOR),
    )
    q_grid_legal = ttnn.num_cores_to_corerangeset(HEADS, cg, True)
    kv_grid_legal = ttnn.num_cores_to_corerangeset(KV_HEADS, cg, True)
    q_grid_row = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HEADS - 1, 0))})

    print(f"\n### m={m}   compute grid {cg.x}x{cg.y}")
    print(f"    shipped q_grid  = 16-wide row, bbox {q_grid_row.bounding_box()}   <- x>7 does not exist")
    print(f"    legal   q_grid  = num_cores_to_corerangeset, bbox {q_grid_legal.bounding_box()}")

    def _check(label, got_q, got_k, got_v):
        ok = []
        for name, got, ref in (("q", got_q, q_ref), ("k", got_k, k_ref), ("v", got_v, v_ref)):
            g = ttnn.to_torch(got).reshape(ref.shape).to(torch.float32)
            exact = bool(torch.equal(g, ref.to(torch.float32)))
            md = float((g - ref.to(torch.float32)).abs().max())
            ok.append((name, exact, md))
        allx = all(e for _n, e, _d in ok)
        detail = " ".join(f"{n}{'=OK' if e else f'=BAD({d:.1e})'}" for n, e, d in ok)
        print(f"  {label:44s} -> {'BIT-EXACT' if allx else 'WRONG'}  {detail}")
        return allx

    # --- arm A: interleaved (what prefill 64/128 runs today) ---
    xt_il = ttnn.from_torch(
        x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    for _ in range(REPS):
        a, b, c = ttnn.experimental.nlp_create_qkv_heads(
            xt_il, num_heads=HEADS, num_kv_heads=KV_HEADS, transpose_k_heads=False, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.synchronize_device(device)
        last = (a, b, c)
        if _ < REPS - 1:
            for t in last:
                ttnn.deallocate(t)
    _check("A interleaved (shipped at 64/128)", *last)
    for t in last:
        ttnn.deallocate(t)
    ttnn.deallocate(xt_il)

    # --- arms B/C: sharded input, legal grids ---
    xt_sh = ttnn.from_torch(
        x_kvgi, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_sharded
    )

    # B: memory_config= route -> compute_output_specs rebuilds {TILE_HEIGHT, head_dim}
    try:
        a, b, c = ttnn.experimental.nlp_create_qkv_heads(
            xt_sh,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            transpose_k_heads=False,
            memory_config=_hs(q_grid_legal, m, HEAD_DIM),
        )
        ttnn.synchronize_device(device)
        _check("B sharded via memory_config=", a, b, c)
        for t in (a, b, c):
            ttnn.deallocate(t)
    except Exception as e:
        print(f"  {'B sharded via memory_config=':44s} -> REFUSED {_why(e)}")

    # C: output_tensors= route -> our own {m, head_dim} specs are used verbatim
    try:

        def _alloc(nh, mcfg):
            return ttnn.from_torch(
                torch.zeros(1, nh, m, HEAD_DIM, dtype=torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mcfg,
            )

        oq = _alloc(HEADS, _hs(q_grid_legal, m, HEAD_DIM))
        ok_ = _alloc(KV_HEADS, _hs(kv_grid_legal, m, HEAD_DIM))
        ov = _alloc(KV_HEADS, _hs(kv_grid_legal, m, HEAD_DIM))
        for _ in range(REPS):
            ttnn.experimental.nlp_create_qkv_heads(
                xt_sh,
                num_heads=HEADS,
                num_kv_heads=KV_HEADS,
                transpose_k_heads=False,
                memory_config=_hs(q_grid_legal, m, HEAD_DIM),
                output_tensors=(oq, ok_, ov),
            )
            ttnn.synchronize_device(device)
        _check("C sharded via output_tensors=", oq, ok_, ov)
    except Exception as e:
        print(f"  {'C sharded via output_tensors=':44s} -> REFUSED {_why(e)}")

    ttnn.deallocate(xt_sh)
