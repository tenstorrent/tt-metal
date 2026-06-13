# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Probe: can the FF2 all_gather_matmul read its w2 weight from a SHARDED-DRAM
layout (no per-token L1 reshard)?

The fused FF2 step uses tt_ccl.all_gather_matmul (llama_all_gather_matmul_async),
which REQUIRES a weight with a shard_spec — a DRAM-INTERLEAVED w2 fails with
"bad optional access". So w2 currently reshards DRAM->L1 width-sharded every
decode token (the last remaining reshard after w1/w3 went DRAM-direct).

w2_ring40 is used ONLY by FF2, so it can be stored in whatever layout
all_gather_matmul accepts. Candidates (output PCC vs torch gather-then-matmul):
  * BASELINE: L1 width-sharded (W2_RING40_L1_MEMCFG) — current per-token-reshard path.
  * VARIANT A: DRAM-bank-sharded (W2_RING40_MEMCFG / create_dram_sharded_mem_config)
    — HAS a shard_spec (no bad-optional), question is whether the kernel reads it
    correctly or scrambles columns (as it did for the FF12 plain ring matmul).
  * VARIANT B: DRAM-interleaved (as-built) — expected FAIL (bad optional access).

If VARIANT A passes PCC > 0.99, build w2_ring40 DRAM-bank-sharded and feed it
directly -> drops the last per-token reshard.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_ff2_w2_dram_weight_micro.py -s -x
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    bh_glx_mesh,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_CLUSTER_SHAPE = (8, 4)


@pytest.mark.hardware
def test_ff2_w2_dram_weight(bh_glx_mesh):
    mesh_device = bh_glx_mesh
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FUSE_RS_MATMUL": "1",
    }.items():
        os.environ.setdefault(_k, _v)

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    state_dict = _load_full_state_dict(_SNAPSHOT)
    pfx0 = "model.language_model.layers.0."
    all_pfx = "model.language_model.layers."
    state_dict = {k: v for k, v in state_dict.items() if (not k.startswith(all_pfx)) or k.startswith(pfx0)}
    paged = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh_device, state_dict, ["linear_attention"], 1, paged)

    mlp = model.layers[0].feed_forward
    mc = model.model_config
    ck = args.compute_kernel_config_hifi2
    M, dim_per_tp = 32, args.dim_per_tp  # 1280
    swiglu_per_col = 2560 // _CLUSTER_SHAPE[1]  # 640

    torch.manual_seed(0)
    swiglu_full = torch.randn(*_CLUSTER_SHAPE, M, swiglu_per_col)  # [8,4,32,640]
    ff_ring = ttnn.from_torch(
        swiglu_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=mc["REDUCE_SCATTER_OUT_RING40_FF12_MEMCFG"],
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_CLUSTER_SHAPE),
    )

    # AG input layout = the proven 30-core WIDTH_SHARDED [32,32] band.
    ag_in_crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0),
        30,
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        ),
        row_wise=True,
    )
    ag_in_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ag_in_crs, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    ag_out_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mc["RING40_AG_INTERIM_CRS"], [32, 640], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # ---- Torch golden for the all_gather_matmul OUTPUT (before all-reduce) ----
    # gather the 4 cols' 640-shards in each row -> [32,2560], then @ per-device w2.
    w2_dev = [ttnn.get_device_tensors(mlp.w2_ring40)[i].cpu().to_torch().float() for i in range(32)]
    w2_dev = [w.reshape(w.shape[-2], w.shape[-1]) for w in w2_dev]  # [2560,1280] each
    gathered = {}
    for r in range(_CLUSTER_SHAPE[0]):
        cols = [swiglu_full[r, c] for c in range(_CLUSTER_SHAPE[1])]  # each [32,640]
        gathered[r] = torch.cat(cols, dim=-1)  # [32,2560]

    def golden(i):
        r, c = i // _CLUSTER_SHAPE[1], i % _CLUSTER_SHAPE[1]
        return gathered[r] @ w2_dev[i]  # [32,1280] (per-device partial, pre all-reduce)

    def run(name, w2_weight, key):
        try:
            out = mlp.tt_ccl.all_gather_matmul(
                ttnn.to_memory_config(ff_ring, ag_in_memcfg),
                w2_weight,
                dim=3,
                cluster_axis=1,
                num_links=mc["GALAXY_NUM_LINKS"],
                ag_memory_config=ag_out_memcfg,
                mm_memory_config=mc["REDUCE_SCATTER_OUT_RING40_MEMCFG"],
                program_config=mc["FF2_RING40_PROGCFG"],
                compute_kernel_config=ck,
                dtype=ttnn.bfloat8_b,
                global_cb=None,
                buffer_key=key,
            )
            ttnn.synchronize_device(mesh_device)
        except Exception as exc:  # noqa: BLE001
            print(f"[ff2-w2] {name}: ERROR {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return False
        worst = 1.0
        for i, t in enumerate(ttnn.get_device_tensors(out)):
            got = t.cpu().to_torch().float().reshape(M, -1)[:, :dim_per_tp]
            eq, msg = comp_pcc(got, golden(i))
            try:
                pcc = float(msg.split("PCC:")[-1].strip())
            except Exception:  # noqa: BLE001
                pcc = -1.0
            worst = min(worst, pcc)
        out.deallocate(True)
        print(f"[ff2-w2] {name}: worst-device PCC = {worst:.5f}  -> {'PASS' if worst > 0.99 else 'FAIL'}", flush=True)
        return worst > 0.99

    print(
        f"[ff2-w2] mlp.w2_ring40 layout = {mlp.w2_ring40.memory_config().memory_layout} "
        f"buffer={mlp.w2_ring40.memory_config().buffer_type}"
    )

    # BASELINE: L1 width-sharded (current per-token reshard)
    w2_l1 = ttnn.to_memory_config(mlp.w2_ring40, mc["W2_RING40_L1_MEMCFG"])
    base = run("BASELINE_L1_reshard", w2_l1, "FF2_W2_BASE")
    w2_l1.deallocate(True)

    # VARIANT A: DRAM-bank-sharded (has shard_spec)
    try:
        w2_bank = ttnn.to_memory_config(mlp.w2_ring40, mc["W2_RING40_MEMCFG"])
        varA = run("VARIANT_A_dram_bank_sharded", w2_bank, "FF2_W2_BANK")
        w2_bank.deallocate(True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ff2-w2] VARIANT_A build ERROR: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        varA = False

    # VARIANT B: DRAM-interleaved direct (expected FAIL: bad optional access)
    varB = run("VARIANT_B_dram_interleaved_direct", mlp.w2_ring40, "FF2_W2_INTERLEAVED")

    # VARIANT C: DRAM-buffer with the L1-style width-shard spec (has shard_spec,
    # column-order-preserving). Open question: does ttnn support a DRAM tensor
    # sharded on a CORE GRID (vs banks)?
    try:
        l1_spec = mc["W2_RING40_L1_MEMCFG"].shard_spec
        dram_ws = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, l1_spec)
        w2_dws = ttnn.to_memory_config(mlp.w2_ring40, dram_ws)
        varC = run("VARIANT_C_dram_width_sharded_coregrid", w2_dws, "FF2_W2_DRAMWS")
        w2_dws.deallocate(True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ff2-w2] VARIANT_C build ERROR: {type(exc).__name__}: {str(exc)[:140]}", flush=True)
        varC = False

    print(
        f"\n[ff2-w2] SUMMARY: baseline(L1)={base}  A(DRAM-bank)={varA}  B(DRAM-interleaved)={varB}  C(DRAM-width-sharded)={varC}"
    )
    print("[ff2-w2] => If A is True, build w2_ring40 DRAM-bank-sharded and feed directly (drop last reshard).")
    assert base, "baseline L1 path should pass (sanity)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
