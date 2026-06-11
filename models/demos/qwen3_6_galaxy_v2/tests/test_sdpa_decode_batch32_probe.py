# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness PROBE: paged SDPA-DECODE *op* at qwen3.6 full-attn dims, BATCH-32.

GOAL
----
DECISIVELY isolate whether ``ttnn.transformer.paged_scaled_dot_product_attention_decode``
itself handles **batch-32 correctly**, independent of the qwen3.6 model integration.

Background: full-attn decode at N=32 produces *per-user-divergent* output (most users
PCC ~0.98, a few PCC 0.0) even though Q is bit-identical across the 32 users and the KV
cache is identical across the 32 users. The model path (``llama_attention.py::
_forward_decode_qwen36`` ~L2078-2146) height-shards q one user-row per core and runs the
op on a 48-core (8,6)+sub_core_grids program config. We need to know if the OP diverges
per-user at batch-32 in isolation, or only inside the model integration.

METHOD (CPU-free oracle)
------------------------
Build a batch-32 input where:
  * all 32 batch entries of Q are the **SAME** random vector, and
  * all 32 users' paged KV is **IDENTICAL** (same keys/values at the same ctx).
Then identical-Q + identical-KV  ⇒  the SDPA output MUST be identical for all 32 users.
So the oracle is simply: every batch row 1..31 must equal row 0. No CPU SDPA needed
(a torch reference is also computed for row 0 as a sanity check on the test setup).

We mirror the model's *exact* decode call:
  * q laid out ``[1, B=32, n_q_pc=3, hd=256]``, then height-sharded one user-row per core
    (tile-pad n_q_pc→32) on ``num_cores_to_corerangeset_in_subcoregrids(start, N=32, sub)``
    — identical to ``_forward_decode_qwen36`` at N>1.
  * sharded output memcfg = SCORES_BATCHED_MM_OUTPUT_MEMCFG(N=32) shape (32, hd) on N cores.
  * SDPA_DECODE_COMPUTE_PROGCFG (HiFi2, fp32_dest_acc_en=False) — the model default.

PARAMETRIZE the grid/core-count (the lever under test):
  * "grid48_8x6"  — (8,6)+sub_core_grids(48): the CURRENT model default (PAGED_SDPA_DECODE_PROGCFG).
  * "grid32_8x4"  — (8,4)+sub_core_grids(32): q sharded to 32 cores (one core per user).

VERDICT
-------
  * BOTH grids diverge per-user in isolation  → TTNN SDPA-decode OP limitation at batch-32
        (escalate / op fix). Model integration is NOT the cause.
  * grid32 all-rows-identical, grid48 not       → fix = use a 32-core (N-core) SDPA decode
        program config for batch-32 (a model-config lever).
  * BOTH grids give identical rows in isolation → the OP is fine; the per-user divergence
        lives in the model integration (q layout / core alignment upstream of the op).

This is a PROBE — it does NOT touch model code and does NOT commit.

Run (48-core, current model default):
    cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    export HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
    python3 -m pytest --noconftest -q -s \
        models/demos/qwen3_6_galaxy_v2/tests/test_sdpa_decode_batch32_probe.py \
        -k grid48_8x6

Run (32-core variant):
    ... same env ...
    python3 -m pytest --noconftest -q -s \
        models/demos/qwen3_6_galaxy_v2/tests/test_sdpa_decode_batch32_probe.py \
        -k grid32_8x4
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn

# --- qwen3.6 full-attn decode dims (fixed) ---------------------------------
MESH_SHAPE = (8, 4)  # production BH GLX 32-chip; rows=cluster_axis0, cols=cluster_axis1
B = 32  # batch (users) — the corner under test
N_Q_PC = 3  # n_q_per_chip
N_KV_PC = 1  # n_kv_per_chip
N_KV_FULL = 8  # n_kv padded; cache dim=1 sharded across 8 rows
HD = 256  # head_dim
SCALE = HD**-0.5
TILE = 32

# Paged config. block_size=32 per the task spec.
BLOCK_SIZE = 32

# Context length to probe (cur_pos = ctx-1). 128 is cheap and sufficient: the
# oracle is row0-vs-rowi divergence, not a ctx sweep.
CTX = int(os.environ.get("QWEN36_PROBE_CTX", "128"))

PCC_THRESH = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    am, bm = a - a.mean(), b - b.mean()
    denom = am.norm() * bm.norm()
    if denom == 0:
        return 1.0 if a.norm() == 0 and b.norm() == 0 else 0.0
    return float((am @ bm) / denom)


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE))
    assert tuple(mesh.shape) == MESH_SHAPE, f"mesh shape {tuple(mesh.shape)} != {MESH_SHAPE}"
    print(f"\n[probe] opened mesh {tuple(mesh.shape)} ({mesh.get_num_devices()} devices)")
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _sub_core_grids(mesh):
    """Mirror qwen36_model_config.py L204-218: full BH compute grid, start (1,0).

    This is what _forward_decode_qwen36 actually uses (NOT the narrow llama70b
    3-col band), so the op runs on the exact cores the model integration uses.
    """
    g = mesh.compute_with_storage_grid_size()
    sub = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    start = ttnn.CoreCoord(1, 0)
    return sub, start


def _compute_kernel_cfg():
    # Model default for multi-core decode (SDPA_DECODE_COMPUTE_PROGCFG):
    # HiFi2, fp32_dest_acc_en=False. HiFi4+fp32 corrupts the 128k cross-core combine.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


# (grid_name, bounding_rect, n_cores) — the lever under test.
_GRID_VARIANTS = [
    ("grid48_8x6", (8, 6), 48),  # current model default (PAGED_SDPA_DECODE_PROGCFG)
    ("grid32_8x4", (8, 4), 32),  # one core per user (B=32)
]


@pytest.mark.parametrize("grid_name,bounding,n_cores", _GRID_VARIANTS, ids=[g[0] for g in _GRID_VARIANTS])
def test_sdpa_decode_batch32(bh_glx_mesh, grid_name, bounding, n_cores):
    mesh = bh_glx_mesh
    cluster_shape = list(mesh.shape)  # [8, 4]
    cur_pos_int = CTX - 1
    n_blocks_per_user = (CTX + BLOCK_SIZE - 1) // BLOCK_SIZE
    ctx_pad = n_blocks_per_user * BLOCK_SIZE
    sub, start = _sub_core_grids(mesh)

    print(
        f"\n[probe] grid={grid_name} bounding={bounding} n_cores={n_cores} "
        f"B={B} ctx={CTX} cur_pos={cur_pos_int} block_size={BLOCK_SIZE} "
        f"n_blocks/user={n_blocks_per_user} head_dim={HD} n_q_pc={N_Q_PC} n_kv_pc={N_KV_PC}"
    )

    torch.manual_seed(1234)

    # --- Q: ONE random vector, replicated across ALL 32 users. ---
    # Model permutes q_rot to [1, B, n_q_pc, hd] before paged SDPA decode.
    q_one = torch.randn(1, 1, N_Q_PC, HD, dtype=torch.float32) * 0.5
    q_torch = q_one.expand(1, B, N_Q_PC, HD).contiguous()  # all 32 users identical

    # --- KV: ONE user's keys/values, IDENTICAL for all 32 users. ---
    # The paged cache is shared by all users via per-user page tables that map
    # each user's virtual blocks to DISTINCT physical blocks holding the SAME data.
    k_real = torch.randn(1, 1, CTX, HD, dtype=torch.float32) * 0.5
    v_real = torch.randn(1, 1, CTX, HD, dtype=torch.float32) * 0.5

    # --- torch reference SDPA for the single decode position (row-0 sanity only).
    qf = q_one[0, 0]  # [n_q_pc, hd]
    scores = (qf @ k_real[0, 0].transpose(0, 1)) * SCALE  # [n_q_pc, ctx]
    probs = torch.softmax(scores, dim=-1)
    ref_out = probs @ v_real[0, 0]  # [n_q_pc, hd]

    # --- Build paged KV cache: distinct physical blocks per user, same content. ---
    # Each user gets n_blocks_per_user physical blocks; the page_table is [B, n_blocks].
    max_num_blocks = n_blocks_per_user * B
    permutation = torch.randperm(max_num_blocks)
    page_table = permutation.reshape(B, n_blocks_per_user).to(torch.int32)

    cache_k = torch.zeros(max_num_blocks, N_KV_FULL, BLOCK_SIZE, HD, dtype=torch.float32)
    cache_v = torch.zeros(max_num_blocks, N_KV_FULL, BLOCK_SIZE, HD, dtype=torch.float32)
    k_padded = torch.zeros(1, 1, ctx_pad, HD)
    v_padded = torch.zeros(1, 1, ctx_pad, HD)
    k_padded[:, :, :CTX] = k_real
    v_padded[:, :, :CTX] = v_real
    # Fill every user's physical blocks with the SAME (single-user) content,
    # replicated across all N_KV_FULL head slots so each row-device holds it.
    for u in range(B):
        for vblk in range(n_blocks_per_user):
            phys = int(page_table[u, vblk])
            t0 = vblk * BLOCK_SIZE
            cache_k[phys, :, :, :] = k_padded[0, 0, t0 : t0 + BLOCK_SIZE].unsqueeze(0)
            cache_v[phys, :, :, :] = v_padded[0, 0, t0 : t0 + BLOCK_SIZE].unsqueeze(0)

    row_shard_kv = ttnn.ShardTensor2dMesh(mesh, dims=(1, None), mesh_shape=cluster_shape)
    keys_cache = ttnn.from_torch(
        cache_k,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=row_shard_kv,
    )
    values_cache = ttnn.from_torch(
        cache_v,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=row_shard_kv,
    )

    # page_table [B, n_blocks]: replicated across mesh (mirror model dims=(None,None)).
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )

    # cur_pos [B] = [128]*32, replicated across mesh.
    cur_pos = torch.tensor([cur_pos_int] * B, dtype=torch.int32)
    cur_pos_tt = ttnn.from_torch(
        cur_pos,
        device=mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )

    # Q [1, B, n_q_pc, hd] replicated across mesh, bf16, TILE — DRAM first.
    q_tt = ttnn.from_torch(
        q_torch,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )

    # --- Height-shard q one user-row per core (mirror _forward_decode_qwen36, N>1). ---
    # tile-pad n_q_pc rows to 32, height-shard the B user-rows one-per-core on the
    # SAME sub_core_grids + row_wise order the output uses.
    q_tile_rows = ((N_Q_PC + TILE - 1) // TILE) * TILE  # 32
    q_shard_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(start, B, sub, row_wise=True)
    q_shard_cfg = ttnn.create_sharded_memory_config(
        shape=[q_tile_rows, HD],
        core_grid=q_shard_cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q_sharded = ttnn.to_memory_config(q_tt, memory_config=q_shard_cfg)
    q_tt.deallocate(True)

    # --- SDPA program config for this grid variant. ---
    progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=bounding,
        sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, n_cores, sub, row_wise=True),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )

    # --- Sharded output memcfg = SCORES_BATCHED_MM_OUTPUT_MEMCFG(N=B): (32, hd) on B cores.
    out_memcfg = ttnn.create_sharded_memory_config(
        shape=(math.ceil(N_Q_PC / TILE) * TILE, HD),  # (32, 256)
        core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, B, sub, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    compute_cfg = _compute_kernel_cfg()

    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_sharded,
        keys_cache,
        values_cache,
        page_table_tt,
        cur_pos_tensor=cur_pos_tt,
        scale=SCALE,
        program_config=progcfg,
        compute_kernel_config=compute_cfg,
        memory_config=out_memcfg,
    )
    q_sharded.deallocate(True)

    # Convert sharded output → DRAM (mirror model), then read device0.
    out_dram = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
    out.deallocate(True)
    ttnn.synchronize_device(mesh)

    # out shape [1, B, n_q_pc(padded), hd]; concat-mesh on dim0 → [num_dev, B, nq, hd]; device0.
    out_t = ttnn.to_torch(out_dram, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    dev0 = out_t[0:1]  # [1, B, nq, hd]
    got = dev0[0, :, :N_Q_PC, :].to(torch.float32)  # [B, n_q_pc, hd]

    ttnn.deallocate(out_dram)
    ttnn.deallocate(keys_cache)
    ttnn.deallocate(values_cache)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(cur_pos_tt)

    # --- Oracle: every user row must equal row 0 (identical q + identical KV). ---
    row0 = got[0]  # [n_q_pc, hd]
    print(f"\n=== batch-32 SDPA-decode OP probe [{grid_name}] @ ctx={CTX} (head_dim={HD}) ===")
    print(f"row0 PCC vs torch ref (setup sanity): {_pcc(row0, ref_out):.5f}")
    print(f"{'user':>4} {'max_abs_diff_vs_row0':>22} {'PCC_vs_row0':>14}")
    max_diffs = []
    pccs = []
    n_bad = 0
    for u in range(B):
        d = (got[u] - row0).abs().max().item()
        p = _pcc(got[u], row0)
        max_diffs.append(d)
        pccs.append(p)
        flag = "" if p > PCC_THRESH else "  <-- DIVERGES"
        if p <= PCC_THRESH:
            n_bad += 1
        # Print all 32 so the per-user pattern (which users go to 0.0) is visible.
        print(f"{u:>4} {d:>22.6e} {p:>14.5f}{flag}")

    worst_pcc = min(pccs)
    worst_diff = max(max_diffs)
    print(
        f"\n[{grid_name}] VERDICT: {B - n_bad}/{B} users match row0 (PCC>{PCC_THRESH}); "
        f"min PCC={worst_pcc:.5f}; max abs-diff vs row0={worst_diff:.6e}"
    )

    # row-0 sanity: if this fails, the test SETUP is wrong, not an op verdict.
    assert _pcc(row0, ref_out) > PCC_THRESH, (
        f"row0 PCC vs torch = {_pcc(row0, ref_out):.5f} (< {PCC_THRESH}); " f"test setup is wrong, not a kernel verdict"
    )

    # The DECISIVE op verdict: identical q + identical KV ⇒ all 32 rows identical.
    assert n_bad == 0, (
        f"[{grid_name}] SDPA-decode OP DIVERGES per-user at batch-32 in ISOLATION: "
        f"{n_bad}/{B} users have PCC<={PCC_THRESH} vs row0 (min PCC={worst_pcc:.5f}). "
        f"Identical-q + identical-KV must give identical output — this is an OP-level "
        f"batch-32 bug on the {grid_name} grid, NOT a model-integration artifact."
    )
