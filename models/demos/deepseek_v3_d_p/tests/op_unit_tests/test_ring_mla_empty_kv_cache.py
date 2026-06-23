# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level repro of the init_kvpe_cache(ttnn.empty) bug.

Wires ONLY the three cache-path ops -- init_kvpe_cache -> update_padded_kv_cache -> ring_mla -- with
synthetic tensors (no ttMLA, no projections/rope/weights). Runs the SAME rotated multi-chunk prefill
twice, identical inputs, differing ONLY in cache init (zero vs ttnn.empty), and asserts the ring_mla
outputs match.

Why all three ops are needed (established empirically):
  * ring_mla alone masks cache cells beyond logical_n -- garbage there is harmless (see
    test_ring_joint_sdpa.py::test_ring_mla_robust_to_uninitialized_kv_cache).
  * update_padded_kv_cache writes FULL 128-aligned tiles. A non-128-aligned chunk leaves the pad rows
    between the last real token and the next tile boundary = whatever was already in the cache.
  * With a zero cache those rows are 0 (benign); with ttnn.empty they hold DRAM garbage (huge bfloat8
    shared exponent), and a LATER chunk's ring_mla read covers them INSIDE [0, logical_n) -> collapse.

The padded_partial rotation (new_actual_isls below) is the same one the dedicated update_padded_kv_cache
op test uses; iter 0 leaves a non-128-aligned frontier and a later iter reads across it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config as Cfg
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import BH_NUM_DRAM_BANKS, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

KV_LORA_RANK = Cfg.KV_LORA_RANK  # 512  (= ring_mla head_dim_v)
QK_ROPE = Cfg.QK_ROPE_HEAD_DIM  # 64
KVPE_DIM = KV_LORA_RANK + QK_ROPE  # 576
QK_HEAD_DIM = Cfg.QK_NOPE_HEAD_DIM + Cfg.QK_ROPE_HEAD_DIM  # 192 (drives the scale)

DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
}


def _kvpe_cache_mem_config():
    """The DRAM-bank nd-shard config init_kvpe_cache builds for the KV cache (replicated here so the
    zero reference matches the empty cache's layout exactly)."""
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(BH_NUM_DRAM_BANKS)}
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, KVPE_DIM],
            grid=grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _alloc_kvpe_cache(mesh_device, init_mode, *, seq_len, mesh_shape, sp_axis):
    """Allocate the MLA KV cache with init_kvpe_cache's exact shape / dtype / TILE layout / DRAM-bank
    nd-shard config, differing ONLY in initialization:
      * empty -> ttnn.empty (uninitialized; unwritten cells hold DRAM garbage) -- what the proposed
                 init_kvpe_cache(ttnn.empty) change does.
      * zero  -> from_torch(zeros) -- the current main behavior / correct reference.
    Self-contained (does not call init_kvpe_cache) so the repro is independent of which init the branch
    under test uses -- it demonstrates the failure mode of an empty-initialized cache directly."""
    seq_len_local = seq_len // mesh_shape[sp_axis]
    expected_shape = [1, 1, seq_len_local, KVPE_DIM]
    mem_config = _kvpe_cache_mem_config()
    if init_mode == "empty":
        return ttnn.empty(
            shape=expected_shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=mem_config,
        )
    return ttnn.from_torch(
        torch.zeros(*expected_shape),
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rotated_chip_positions(kv_actual, sp, chunk_local):
    """Global token position each chip-local input row lands at, mirroring the writer kernel
    (copied from test_deepseek_prefill_update_padded_kv_cache.py)."""
    C = chunk_local
    chunk_global = sp * C
    boundary_slab = kv_actual // chunk_global
    boundary_chip = (kv_actual // C) % sp
    boundary_offset = kv_actual % C
    positions = [[0] * C for _ in range(sp)]
    for c in range(sp):
        if c < boundary_chip:
            update_idxt = (boundary_slab + 1) * C
        elif c == boundary_chip:
            update_idxt = boundary_slab * C + boundary_offset
        else:
            update_idxt = boundary_slab * C
        for r in range(C):
            lr = update_idxt + r
            positions[c][r] = (lr // C) * chunk_global + c * C + (lr % C)
    return positions


def _run_op_level(mesh_device, *, init_mode, new_isl_tiles_per_dev=4, cache_tokens_per_dev=512):
    sp_axis, tp_axis = 0, 1
    sp, tp = list(mesh_device.shape)
    if sp < 2:
        pytest.skip(f"ring_mla needs SP>=2; got sp={sp}")
    tile = ttnn.TILE_SIZE
    num_links = 2 if is_blackhole() else 1

    C = new_isl_tiles_per_dev * tile  # per-device chunk
    chunk_global = C * sp
    cache_global = cache_tokens_per_dev * sp
    # padded_partial rotation: iter0 frontier is non-128-aligned -> stale pad rows a later iter reads.
    new_actual_isls = [(sp - 1) * C + tile, 2 * C, sp * C]
    assert all(v <= chunk_global for v in new_actual_isls)
    nhq = 4 * tp  # global Q heads (heads sharded over tp)
    scale = QK_HEAD_DIM**-0.5

    # --- ring-attention CCL scaffolding (mirrors TtMLA / _run_ring_mla) ---
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    ccl_sems = [ttnn.create_global_semaphore(mesh_device, ccl_crs, 0) for _ in range(2)]
    ccl_core_grid_offset = (grid.x - 1, 0)
    sdpa_compute_grid = (grid.x - 1, grid.y)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid, q_chunk_size=tile, k_chunk_size=C, exp_approx_mode=False
    )

    try:
        kvpe_cache = _alloc_kvpe_cache(
            mesh_device, init_mode, seq_len=cache_global, mesh_shape=list(mesh_device.shape), sp_axis=sp_axis
        )
        # Gathered-KV scratch (== TtMLA make_chunked_kv_buf): full cache, replicated.
        chunked_kv_buf = ttnn.from_torch(
            torch.zeros(1, 1, cache_global, KVPE_DIM),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
        )

        kv_in_shard = [None, None]
        kv_in_shard[sp_axis] = 2  # chunk seq across the ring
        q_shard = [None, None]
        q_shard[sp_axis] = 2  # chunk seq across the ring
        q_shard[tp_axis] = 1  # heads across tp
        out_concat = [None, None]
        out_concat[sp_axis] = 2
        out_concat[tp_axis] = 1

        cum_total = sum(new_actual_isls)
        torch.manual_seed(0)
        kv_sent = torch.randn(cum_total, KVPE_DIM, dtype=torch.bfloat16)  # natural-order KV tokens
        q_sent = torch.randn(nhq, cum_total, KVPE_DIM, dtype=torch.bfloat16)  # natural-order queries

        outputs = []
        kv_actual = 0
        for it, new_isl in enumerate(new_actual_isls):
            valid_end = kv_actual + new_isl
            positions = _rotated_chip_positions(kv_actual, sp, C)
            flat = [positions[c][r] for c in range(sp) for r in range(C)]
            gather_idx = torch.tensor([min(g, cum_total - 1) for g in flat], dtype=torch.long)
            pad_mask = torch.tensor([g >= valid_end for g in flat])

            # --- write this chunk via the real op (full-tile write leaves stale pad rows) ---
            kv_chunk = kv_sent[gather_idx].clone()
            kv_chunk[pad_mask] = 0.0
            tt_kvpe = ttnn.from_torch(
                kv_chunk.reshape(1, 1, chunk_global, KVPE_DIM),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=kv_in_shard),
            )
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kvpe_cache,
                tt_kvpe,
                slot_idx=0,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=kv_actual,
                cluster_axis=sp_axis,
            )

            # --- read via ring_mla over the populated prefix (same as TtMLA._chunked_attn) ---
            q_chunk = q_sent[:, gather_idx, :].clone()  # [nhq, chunk_global, KVPE_DIM]
            tt_q = ttnn.from_torch(
                q_chunk.reshape(1, nhq, chunk_global, KVPE_DIM),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=q_shard),
            )
            attn_out, _ = ttnn.transformer.ring_mla(
                tt_q,
                kvpe_cache,
                persistent_output_buffer_kv=chunked_kv_buf,
                head_dim_v=KV_LORA_RANK,
                logical_n=kv_actual + chunk_global,
                program_config=program_config,
                scale=scale,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=ccl_sems,
                num_links=num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=ttnn.Topology.Linear,
                ccl_core_grid_offset=ccl_core_grid_offset,
                use_column_major_ccl=True,
                is_balanced=False,
                kv_cache_batch_idx=0,
                kv_actual_isl=kv_actual,
            )
            out = ttnn.to_torch(
                attn_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(out_concat), mesh_shape=(sp, tp)),
            ).to(torch.float32)
            outputs.append(out)
            logger.info(f"[{init_mode}] iter {it} kv_actual={kv_actual} new_isl={new_isl} -> out {list(out.shape)}")
            kv_actual = valid_end
        ttnn.synchronize_device(mesh_device)
        return outputs
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.timeout(0)
def test_ring_mla_empty_kv_cache_op_level(mesh_device, device_params):
    """Op-level repro: init_kvpe_cache + update_padded_kv_cache (rotated, partial) + ring_mla. Runs the
    same rotated multi-chunk prefill with a ttnn.empty cache and a zero cache; the outputs must match
    (unwritten cells must not affect the result). Currently FAILS with empty init, exposing the bug.

    IMPORTANT: reset the device before running (e.g. `tt-smi -r`) so DRAM holds real garbage. ttnn.empty
    returns whatever is already in the banks; on a freshly-zeroed device it can come back zero and the
    bug won't surface. The EMPTY run goes FIRST so it captures the post-reset garbage before the zero
    run overwrites those banks."""
    out_empty = _run_op_level(mesh_device, init_mode="empty")  # first: sees real post-reset DRAM garbage
    out_zero = _run_op_level(mesh_device, init_mode="zero")  # reference: correct (zero-padded) output
    assert len(out_zero) == len(out_empty)
    for it, (z, e) in enumerate(zip(out_zero, out_empty)):
        passed, msg = comp_pcc(z, e, 0.99)
        logger.info(f"iter {it} zero-vs-empty ring_mla PCC: {msg}")
        assert passed, (
            f"ring_mla output depends on KV-cache init at iter {it}: zero vs empty diverge ({msg}). "
            f"update_padded_kv_cache leaves garbage in partial-tile pad rows that a later ring_mla reads; "
            f"the cache must be zero-initialized (or the pad rows scrubbed)."
        )
    logger.success("✓ op-level: ring_mla invariant to KV-cache init (empty == zero)")
