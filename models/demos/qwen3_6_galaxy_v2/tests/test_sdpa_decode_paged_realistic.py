# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""REALISTIC paged SDPA decode probe at 128k — block-indexing correctness.

Why a second probe?
-------------------
The original ``test_sdpa_decode_grid.py`` fills EVERY physical block with the
SAME content per row (``cache_k[phys, :, :, :] = blk.unsqueeze(0)`` where blk
is the same tensor pattern), and uses ``BLOCK_SIZE=64``. That probe cannot
expose a page-table / block-indexing bug: if the multi-core flash-decode read
the wrong physical block, it would read identical data and still match.

The garbling demo (``text_demo_qwen36.py``) uses ``BLOCK_SIZE=32`` (NOT 64),
which at 131072 ctx yields ~4100 physical blocks (vs ~2052 at block_size 64) —
2x the page-table fan-out and 2x the per-step block iteration.

This probe:
  * uses the DEMO's block_size=32 + max_num_blocks formula,
  * fills each physical block with DISTINCT content (so reading the wrong block
    is detectable),
  * builds a real randperm page_table,
  * compares single-core (1,1) vs multi-core (8,6)+sub_core(48) output PCC vs a
    torch reference that gathers V through the SAME page table,
  * sweeps ctx and (optionally) max_cores_per_head_batch.

Run:
    cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    export HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
    python3 -m pytest --noconftest -q -s \
        models/demos/qwen3_6_galaxy_v2/tests/test_sdpa_decode_paged_realistic.py
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn

MESH_SHAPE = (8, 4)
B = 1
N_Q_PC = 3
N_KV_PC = 1
N_KV_FULL = 8
HD = 256
SCALE = HD**-0.5

# Demo uses block_size=32 (text_demo_qwen36.py _PAGED_BLOCK_SIZE).
BLOCK_SIZE = int(os.environ.get("QWEN36_PROBE_BLOCK_SIZE", "32"))
_DECODE_STEPS = 32  # match demo headroom

# 128k is the failing context; include 64k (coherent) as a control.
_CTXS = [int(c) for c in os.environ.get("QWEN36_PROBE_CTXS", "65536,131072").split(",")]
# max_cores_per_head_batch override sweep (empty -> default kernel value)
_MAXCORES = [s for s in os.environ.get("QWEN36_PROBE_MAXCORES", "").split(",") if s]

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


def _sub_core_grids():
    sub = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    start = ttnn.CoreCoord(1, 0)
    return sub, start


def _single_core_cfg():
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )


def _multi_cfg(max_cores=None):
    sub, start = _sub_core_grids()
    kwargs = dict(
        compute_with_storage_grid_size=(8, 6),
        sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, 48, sub, row_wise=True),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    if max_cores is not None:
        kwargs["max_cores_per_head_batch"] = max_cores
    return ttnn.SDPAProgramConfig(**kwargs)


def _compute_kernel_cfg():
    # Match the model's compute_kernel_config_hifi4.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _shard_out_memcfg():
    sub, start = _sub_core_grids()
    return ttnn.create_sharded_memory_config(
        shape=(math.ceil(N_Q_PC / 32) * 32, HD),
        core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, B, sub, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("ctx", _CTXS)
def test_realistic_paged(bh_glx_mesh, ctx):
    mesh = bh_glx_mesh
    cluster_shape = list(mesh.shape)
    cur_pos_int = ctx - 1

    # Demo formula: max(32, (T + decode_steps + bs - 1)//bs + 4)
    n_blocks_per_user = (ctx + _DECODE_STEPS + BLOCK_SIZE - 1) // BLOCK_SIZE + 4
    n_blocks_per_user = max(32, n_blocks_per_user)
    max_num_blocks = n_blocks_per_user * B
    ctx_pad = n_blocks_per_user * BLOCK_SIZE
    print(
        f"\n[probe] ctx={ctx} cur_pos={cur_pos_int} block_size={BLOCK_SIZE} "
        f"blocks/user={n_blocks_per_user} max_blocks={max_num_blocks} head_dim={HD}"
    )

    torch.manual_seed(1234)

    # Distinct per-position K/V — magnitudes vary per block so a wrong-block read
    # is numerically visible. Q small so softmax spreads (avoids one-hot masking
    # the block-index error).
    # Magnitudes chosen so the 131072-deep softmax is neither one-hot nor uniform:
    # qk scores ~ N(0, (qs*ks)^2 * HD * SCALE^2). With qs=ks=1.0, HD=256,
    # SCALE=1/16 -> score std ~1.0, giving real attention structure (output has
    # variance, so PCC is meaningful) without collapsing to a single position.
    qs = float(os.environ.get("QWEN36_PROBE_QS", "1.0"))
    ks = float(os.environ.get("QWEN36_PROBE_KS", "1.0"))
    q_torch = torch.randn(1, B, N_Q_PC, HD, dtype=torch.float32) * qs
    k_real = torch.randn(B, 1, ctx, HD, dtype=torch.float32) * ks
    v_real = torch.randn(B, 1, ctx, HD, dtype=torch.float32) * 0.4
    if os.environ.get("QWEN36_PROBE_RAMP", "0") == "1":
        # Make each position's V mean distinct (ramp) so mis-indexed reads shift output.
        pos = torch.arange(ctx, dtype=torch.float32).view(1, 1, ctx, 1)
        v_real = v_real + (pos / ctx)

    # Torch reference in bf16 to match the device cache dtype (the kernel reads a
    # bf16 paged cache; an fp32 reference over a 131072-deep softmax diverges
    # purely from cache quantization, which is NOT the bug under test).
    qf = q_torch[0, 0].to(torch.bfloat16).to(torch.float32)
    k_bf = k_real[0, 0].to(torch.bfloat16).to(torch.float32)
    v_bf = v_real[0, 0].to(torch.bfloat16).to(torch.float32)
    scores = (qf @ k_bf.transpose(0, 1)) * SCALE  # [n_q, ctx]
    probs = torch.softmax(scores, dim=-1)
    ref_out = probs @ v_bf  # [n_q, hd]

    # Page table: virtual block i -> physical perm[i].
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(B, max_num_blocks // B).to(torch.int32)

    cache_k = torch.zeros(max_num_blocks, N_KV_FULL, BLOCK_SIZE, HD, dtype=torch.float32)
    cache_v = torch.zeros(max_num_blocks, N_KV_FULL, BLOCK_SIZE, HD, dtype=torch.float32)
    k_padded = torch.zeros(B, 1, ctx_pad, HD)
    v_padded = torch.zeros(B, 1, ctx_pad, HD)
    k_padded[:, :, :ctx] = k_real
    v_padded[:, :, :ctx] = v_real
    for vblk in range(n_blocks_per_user):
        phys = int(page_table[0, vblk])
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
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )
    cur_pos = torch.tensor([cur_pos_int] * B, dtype=torch.int32)
    cur_pos_tt = ttnn.from_torch(
        cur_pos,
        device=mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )
    q_tt = ttnn.from_torch(
        q_torch,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )

    compute_cfg = _compute_kernel_cfg()
    shard_mc = _shard_out_memcfg()

    def run(progcfg, sharded):
        out_mc = shard_mc if sharded else ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            keys_cache,
            values_cache,
            page_table_tensor=page_table_tt,
            cur_pos_tensor=cur_pos_tt,
            scale=SCALE,
            program_config=progcfg,
            compute_kernel_config=compute_cfg,
            memory_config=out_mc,
        )
        if sharded:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh)
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        got = out_t[0, 0, :N_Q_PC, :].to(torch.float32)
        ttnn.deallocate(out)
        return got

    variants = {"single_core_1x1": (_single_core_cfg(), False), "multi_8x6_subcore_shardout": (_multi_cfg(), True)}
    for mc in _MAXCORES:
        variants[f"multi_maxcores_{mc}"] = (_multi_cfg(int(mc)), True)

    results = {}
    for name, (cfg, sharded) in variants.items():
        try:
            results[name] = run(cfg, sharded)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] {name} CRASHED: {e!r}")
            results[name] = None

    print(f"\n=== REALISTIC paged probe @ ctx={ctx} block_size={BLOCK_SIZE} blocks={max_num_blocks} ===")
    ref_single = results.get("single_core_1x1")
    print(f"{'config':<28} {'PCC_vs_torch':>13} {'PCC_vs_single':>14}")
    for name, got in results.items():
        if got is None:
            print(f"{name:<28} {'CRASH':>13} {'CRASH':>14}")
            continue
        pcc_t = _pcc(got, ref_out)
        pcc_s = _pcc(got, ref_single) if ref_single is not None else float("nan")
        print(f"{name:<28} {pcc_t:>13.5f} {pcc_s:>14.5f}")

    for t in (q_tt, keys_cache, values_cache, page_table_tt, cur_pos_tt):
        ttnn.deallocate(t)

    assert ref_single is not None, "single-core reference failed to run"
    sref = _pcc(ref_single, ref_out)
    print(f"[probe] single-core vs torch = {sref:.5f} (setup sanity; want > {PCC_THRESH})")
