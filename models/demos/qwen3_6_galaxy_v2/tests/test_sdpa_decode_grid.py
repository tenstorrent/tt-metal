# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness PROBE: multi-core paged SDPA decode at qwen3.6 head_dim=256.

GOAL
----
Determine whether multi-core ``paged_scaled_dot_product_attention_decode`` is
numerically correct at qwen3.6's full-attn decode dims (head_dim=256,
n_q_per_chip=3, n_kv_per_chip=1, GQA 3:1, batch=1) — i.e. whether the prior
model-level quality regression (alpha 118 -> 34 when bumping the SDPA decode
grid (1,1) -> (4,1)/(8,8)) is a *config* issue or an *inherent kernel bug*.

The current model path (``llama_attention.py::_forward_decode_qwen36`` ~L1931)
runs paged SDPA decode on a ``(1,1)`` single-core grid — BW-bound, reads the
whole KV cache per step. llama70b runs paged SDPA decode coherently on
``(8,6)``=48 cores *with* ``sub_core_grids`` (see llama3_70b_galaxy model_config
``PAGED_SDPA_DECODE_PROGCFG``). This probe mirrors that.

This is a PROBE — it does NOT touch model code. It builds synthetic inputs,
runs the op under several program configs, and compares output PCC against the
single-core reference (the trusted model path) plus a torch reference.

Run:
    cd /home/tt-admin/ssinghal/qwen36/new/tt-metal
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    export HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
    python3 -m pytest --noconftest -q -s \
        models/demos/qwen3_6_galaxy_v2/tests/test_sdpa_decode_grid.py
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

# --- qwen3.6 full-attn decode dims (fixed) ---------------------------------
MESH_SHAPE = (8, 4)  # production BH GLX 32-chip; rows=cluster_axis0, cols=cluster_axis1
B = 1  # batch per device group
N_Q_PC = 3  # n_q_per_chip
N_KV_PC = 1  # n_kv_per_chip
N_KV_FULL = 8  # n_kv padded; cache dim=1 sharded across 8 rows
HD = 256  # head_dim
SCALE = HD**-0.5

# Paged config (mirror demo_qwen_decode: block_size=64, larger ctx -> more blocks)
BLOCK_SIZE = 64

# Contexts to probe (cur_pos = ctx-1). 8k cheap to iterate; 32k to surface a
# timing delta if 8k is too small.
_CTXS = [int(c) for c in os.environ.get("QWEN36_PROBE_CTXS", "8192,32768").split(",")]
_ITERS = int(os.environ.get("QWEN36_PROBE_ITERS", "20"))
_WARMUP = int(os.environ.get("QWEN36_PROBE_WARMUP", "5"))

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
    # Probe the mesh once before any heavy work.
    assert tuple(mesh.shape) == MESH_SHAPE, f"mesh shape {tuple(mesh.shape)} != {MESH_SHAPE}"
    print(f"\n[probe] opened mesh {tuple(mesh.shape)} ({mesh.get_num_devices()} devices)")
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _sub_core_grids():
    """Mirror llama3_70b_galaxy model_config self.sub_core_grids + start_core."""
    sub = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    start = ttnn.CoreCoord(1, 0)
    return sub, start


def _build_progcfgs():
    """Return {name: SDPAProgramConfig}. single-core is the reference."""
    sub, start = _sub_core_grids()
    cfgs = {}
    # Reference: current model path.
    cfgs["single_core_1x1"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    # Multi-core mirroring llama70b PAGED_SDPA_DECODE_PROGCFG (48 cores + sub_core_grids).
    cfgs["multi_8x6_subcore"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, 48, sub, row_wise=True),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    # Simpler 8x4 + sub_core_grids (32 cores).
    cfgs["multi_8x4_subcore"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, 32, sub, row_wise=True),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    # 8x6 WITHOUT sub_core_grids — isolate whether sub_core_grids is the missing ingredient.
    cfgs["multi_8x6_nosubcore"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    # 8x6 + sub_core_grids, run with a SHARDED output memcfg (the llama70b path:
    # subcore grids assert is_q_sharded || is_output_sharded). This variant pairs
    # with shard_out=True below.
    cfgs["multi_8x6_subcore_shardout"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, 48, sub, row_wise=True),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    return cfgs


# configs that need a sharded output memory_config (subcore grids assert it)
_SHARD_OUT_CONFIGS = {"multi_8x6_subcore_shardout"}


def _compute_kernel_cfg():
    # Match the model's SDPA decode compute config (hifi4-ish; fp32 acc off as model uses).
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


@pytest.mark.parametrize("ctx", _CTXS)
def test_sdpa_decode_grid_pcc(bh_glx_mesh, ctx):
    mesh = bh_glx_mesh
    cluster_shape = list(mesh.shape)  # [8, 4]
    cur_pos_int = ctx - 1  # last valid position
    n_blocks_per_user = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_num_blocks = n_blocks_per_user * B  # B=1
    print(
        f"\n[probe] ctx={ctx} cur_pos={cur_pos_int} block_size={BLOCK_SIZE} "
        f"max_num_blocks={max_num_blocks} head_dim={HD} n_q_pc={N_Q_PC} n_kv_pc={N_KV_PC}"
    )

    torch.manual_seed(1234)

    # --- Synthetic Q: model permutes q_rot to [1, B, n_q_pc, hd] for paged SDPA decode.
    q_torch = torch.randn(1, B, N_Q_PC, HD, dtype=torch.float32) * 0.5

    # --- Synthetic K/V content: one kv head's worth (n_kv_pc=1) over the real ctx.
    # Shape [B, 1, ctx, hd]. Pad time to BLOCK_SIZE multiple for the paged cache.
    ctx_pad = n_blocks_per_user * BLOCK_SIZE
    k_real = torch.randn(B, 1, ctx, HD, dtype=torch.float32) * 0.5
    v_real = torch.randn(B, 1, ctx, HD, dtype=torch.float32) * 0.5

    # --- torch reference SDPA for the single decode position (GQA 3:1, 1 kv head).
    # q: [n_q_pc, hd]; k/v: [ctx, hd]; causal up to cur_pos (=ctx-1 -> all positions valid).
    qf = q_torch[0, 0]  # [n_q_pc, hd]
    scores = (qf @ k_real[0, 0].transpose(0, 1)) * SCALE  # [n_q_pc, ctx]
    probs = torch.softmax(scores, dim=-1)
    ref_out = probs @ v_real[0, 0]  # [n_q_pc, hd]

    # --- Build paged KV cache on device.
    # Cache torch shape: [max_num_blocks, N_KV_FULL, BLOCK_SIZE, HD], zero-init,
    # sharded on dim=1 across 8 rows (each device sees 1 kv head), replicated on cols.
    # We place the real kv content (1 kv head) into ALL N_KV_FULL slots so every
    # row-device holds the same single-head content -> device0 readout == ref.
    # Page table maps virtual block i -> physical block perm[i] (mirror demo).
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(B, max_num_blocks // B).to(torch.int32)

    cache_k = torch.zeros(max_num_blocks, N_KV_FULL, BLOCK_SIZE, HD, dtype=torch.float32)
    cache_v = torch.zeros(max_num_blocks, N_KV_FULL, BLOCK_SIZE, HD, dtype=torch.float32)
    # Fill physical blocks per the page table for user 0.
    k_padded = torch.zeros(B, 1, ctx_pad, HD)
    v_padded = torch.zeros(B, 1, ctx_pad, HD)
    k_padded[:, :, :ctx] = k_real
    v_padded[:, :, :ctx] = v_real
    for vblk in range(n_blocks_per_user):
        phys = int(page_table[0, vblk])
        t0 = vblk * BLOCK_SIZE
        blk_k = k_padded[0, 0, t0 : t0 + BLOCK_SIZE]  # [BLOCK_SIZE, HD]
        blk_v = v_padded[0, 0, t0 : t0 + BLOCK_SIZE]
        cache_k[phys, :, :, :] = blk_k.unsqueeze(0)  # all N_KV_FULL heads identical
        cache_v[phys, :, :, :] = blk_v.unsqueeze(0)

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

    # page_table tensor: replicated across mesh (mirror demo dims=(None,None)).
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )

    # cur_pos tensor: [B], replicated (B=1 -> dims=(None,None) like demo).
    cur_pos = torch.tensor([cur_pos_int] * B, dtype=torch.int32)
    cur_pos_tt = ttnn.from_torch(
        cur_pos,
        device=mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )

    # Q tensor: [1, B, n_q_pc, hd] replicated across mesh, bf16, TILE.
    q_tt = ttnn.from_torch(
        q_torch,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=cluster_shape),
    )

    compute_cfg = _compute_kernel_cfg()
    progcfgs = _build_progcfgs()

    # Sharded output memcfg mirroring qwen36 SCORES_BATCHED_MM_OUTPUT_MEMCFG:
    # height-sharded [ceil(n_local_heads/32)*32, hd] on B cores of the sub_core_grid.
    sub, start = _sub_core_grids()
    import math as _math

    shard_out_memcfg = ttnn.create_sharded_memory_config(
        shape=(_math.ceil(N_Q_PC / 32) * 32, HD),
        core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(start, B, sub, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    def run_one(name, progcfg):
        out_mc = shard_out_memcfg if name in _SHARD_OUT_CONFIGS else ttnn.DRAM_MEMORY_CONFIG
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
        return out

    # device0 readout: ConcatMeshToTensor over dim0 then take device0 row slice.
    results = {}
    timings = {}
    ref_torch_pcc = {}
    crashed = {}
    for name, cfg in progcfgs.items():
        try:
            out = run_one(name, cfg)
            if name in _SHARD_OUT_CONFIGS:
                out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(mesh)
            # output shape [1, B, n_q_pc(padded), hd]; read device0.
            out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            # device0 is first chunk along dim0; out_t dim0 == 1*num_devices.
            # Each device produced [1,B,nq,hd]; concat on dim0 -> [num_dev, B, nq, hd].
            dev0 = out_t[0:1]  # [1, B, nq, hd]
            got = dev0[0, 0, :N_Q_PC, :].to(torch.float32)  # [n_q_pc, hd]
            results[name] = got
            ref_torch_pcc[name] = _pcc(got, ref_out)
            ttnn.deallocate(out)

            # timing (eager)
            for _ in range(_WARMUP):
                o = run_one(name, cfg)
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(_ITERS):
                o = run_one(name, cfg)
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            timings[name] = (time.perf_counter() - t0) / _ITERS * 1e3  # ms/iter
        except Exception as e:  # noqa: BLE001
            crashed[name] = repr(e)
            print(f"[probe] CONFIG {name} CRASHED: {e!r}")

    # --- Report ---
    print(f"\n=== SDPA decode grid probe @ ctx={ctx} (head_dim={HD}) ===")
    ref = results.get("single_core_1x1")
    print(f"{'config':<22} {'PCC_vs_torch':>13} {'PCC_vs_single':>14} {'ms/iter':>10}")
    for name in progcfgs:
        if name in crashed:
            print(f"{name:<22} {'CRASH':>13} {'CRASH':>14} {'--':>10}  {crashed[name]}")
            continue
        pcc_torch = ref_torch_pcc[name]
        pcc_single = _pcc(results[name], ref) if ref is not None else float("nan")
        ms = timings.get(name, float("nan"))
        print(f"{name:<22} {pcc_torch:>13.5f} {pcc_single:>14.5f} {ms:>10.4f}")

    ttnn.deallocate(q_tt)
    ttnn.deallocate(keys_cache)
    ttnn.deallocate(values_cache)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(cur_pos_tt)

    # --- Assertions (informational verdict; do not hard-fail multi-core so all variants report) ---
    assert "single_core_1x1" in results, "single-core reference itself failed to run"
    assert ref_torch_pcc["single_core_1x1"] > PCC_THRESH, (
        f"single-core reference PCC vs torch = {ref_torch_pcc['single_core_1x1']:.5f} "
        f"(< {PCC_THRESH}); test setup is wrong, not a kernel verdict"
    )
