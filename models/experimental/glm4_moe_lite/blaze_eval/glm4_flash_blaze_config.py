# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""BlazeConfig for GLM-4.7-Flash (zai-org/GLM-4.7-Flash) on a 1x-harvested BH Galaxy.

Written for a **12x10 = 120-core** device (every chip in this Galaxy is 1x-harvested), not the
13x10 = 130 that DSv3 and GLM-5's configs assume. GridConfig puts the sender at (11, 9) and
the idle phantom column at (11, 0..8), so column 11 rows 0-1 are the natural home for the
gate-MM cores -- the same relationship GLM-5 has on its 13-wide grid, where gate MM sits on
its phantom column 12.

SCOPE: this exists to drive the **routed-expert / router** MoE path, which is the part that
runs on a harvested grid. The shared-expert coords are deliberately EMPTY:

  blaze/weights/moe_grid_layout.py hardcodes NUM_SHARED_GATE_UP_MM_CORES = 64 with
  `assert len(gate_coords) == len(up_coords) == 64`. That balances only at 130 cores -- on
  12x10 the same column pattern yields gate=64 / up=54, and the op then rejects it with
  "gate/up KNMatmul branches must use matching parallelism". 64/64 needs 128 cores against
  118 usable here (120 - sender - phantom). A balanced 56/56 = 112 would fit, so it is
  fixable upstream, but it needs preprocess_gate_up's placement spec regenerated to match.

`sanity_check_model_config` does not require those coords to be non-empty (only the MoE gate /
DRAM-worker relations), so a routed-only config is valid. `dflash/config.py` sets them empty
too.

DIMS come from the HF config.json: hidden 2048, 20 heads, qk_nope 192, qk_rope 64, v_head 256,
kv_lora 512, q_lora 768, 64 routed experts, top-4, 1 shared, moe intermediate 1536, 47 layers.
"""

from pathlib import Path

import ttnn

from ..config.blaze_config import BlazeConfig

# ---------------------------------------------------------------------------
# MoE cores
# ---------------------------------------------------------------------------

# Gate MM: one core per 32-expert tile. sanity_check_model_config asserts
#   num_gate_mm_cores >= ceil(routed_experts_total / 32)  and  experts % len(cores) == 0
# 64 experts -> 2. Using MORE than needed is what hangs the device: the handshake is sized
# from num_experts // 32 while a different set of cores participates, and everyone waits with
# no error raised. Keep this exactly 2.
_MOE_GATE_MM_CORES = ((11, 0), (11, 1))

# One core per DRAM bank; BH has 8. ORDER IS LOAD-BEARING -- it must match
# get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, NOC_0), which is how
# DRAMStreamingMatmul assigns bank ids. A sorted list silently mispairs workers with banks.
_MOE_DRAM_WORKER_GRID = ((0, 9), (0, 0), (0, 7), (0, 3), (7, 9), (7, 1), (7, 6), (7, 4))
_MOE_DRAM_SHARD_GRID = tuple((x, 0) for x in range(8))
_MOE_RED2ONE_AGGREGATOR_GRID = ((0, 2),)

# ---------------------------------------------------------------------------
# MLA cores
#
# These satisfy the config's own divisibility checks so the object is constructible, but the
# MLA path is NOT usable for this model yet: tests/blaze/backed/layout_plan.py:186 requires
# n_heads_per_device % 8 == 0 and GLM has 20 heads, which no TP divisor fixes. GLM's 3:1
# nope:rope ratio additionally makes qrope_grid_cols = heads/(8*3) need a multiple of 24.
# Kept plausible rather than empty so the checks below have real values to validate:
#   q_lora_rank 768 % len(q_a_cores) == 0                -> 48 cores, 16 each
#   nope 20*192 = 3840, 3840/20 = 192 % qk_nope(192) == 0 -> 20 cores, one head each
#   rope 20* 64 = 1280, 1280/20 =  64 % qk_rope( 64) == 0 -> 20 cores, one head each
# nope and rope must not overlap; q_a may overlap them (different pipeline stages), exactly
# as DSv3's q_a x0-11 y0-7 overlaps its qb_nope x0-7 y0-7.
# ---------------------------------------------------------------------------
_QB_NOPE_CORES = tuple((col, row) for col in (0, 1) for row in range(10))
_QB_ROPE_CORES = tuple((col, row) for col in (2, 3) for row in range(10))
_Q_A_CORES = tuple((col, row) for col in range(6) for row in range(8))

# Stays inside columns 0-10, off the phantom column and the sender.
_LM_HEAD_MATMUL_CORES = tuple((col, row) for row in range(10) for col in range(10))


GLM4_FLASH_BLAZE_CONFIG = BlazeConfig(
    model_config_path=str(Path(__file__).with_name("glm4_flash.model_config.json")),
    # Single device: this config is used for per-device op benchmarks, and the checks
    # assert num_devices % attn_sdpa_tp == 0 and % attn_sdpa_cp == 0.
    mesh_shape=(1, 1),
    root_mesh_coord=(0, 0),
    sender_core=(11, 9),
    # LM head / embedding
    lm_head_mesh_shape=(1, 1),
    embedding_mesh_shape=(1, 1),
    embedding_core_coord=(0, 0),
    embedding_weight_dtype=ttnn.bfloat16,
    lm_head_matmul_cores=_LM_HEAD_MATMUL_CORES,
    lm_head_weight_dtype=ttnn.bfloat8_b,
    # Attention. 20 heads is not divisible by 8, so tp/cp are held at 1: the checks require
    # n_heads % attn_sdpa_tp == 0 and n_heads % attn_sdpa_cp == 0.
    attn_qheads_tp=1,
    attn_kvheads_tp=1,
    attn_sdpa_tp=1,
    attn_sdpa_cp=1,
    attn_ar_kind="all_reduce",
    attn_ar_cluster_axis=0,
    attn_q_cores=(),
    attn_k_cores=(),
    attn_v_cores=(),
    attn_o_cores=(),
    attn_q_a_cores=_Q_A_CORES,
    attn_qb_cores=(),
    attn_qb_nope_cores=_QB_NOPE_CORES,
    attn_qb_rope_cores=_QB_ROPE_CORES,
    attn_kv_knope_cores=(),
    attn_kv_rope_cores=(),
    attn_sdpa_forwarder_cores=(),
    attn_sdpa_r3_forwarder_cores=(),
    attn_sdpa_k_chunk_size=64,
    attn_override_max_seq_len=0,
    attn_override_dense_max_positions=0,
    # GLM-4.7-Flash has no lightning indexer (that is a GLM-5 / DSA feature).
    indexer_wq_b_cores=(),
    indexer_wk_cores=(),
    indexer_weights_proj_cores=(),
    # MoE
    moe_router_gate_mm_cores=_MOE_GATE_MM_CORES,
    moe_router_gate_eps=1e-20,
    moe_routed_experts_dram_shard_grid=_MOE_DRAM_SHARD_GRID,
    moe_routed_experts_dram_worker_grid=_MOE_DRAM_WORKER_GRID,
    moe_routed_expert_tp=1,
    moe_shared_expert_tp=1,
    moe_routed_experts_red2one_aggregator_grid=_MOE_RED2ONE_AGGREGATOR_GRID,
    # Empty on purpose -- see the module docstring.
    moe_shared_expert_gate_coords=(),
    moe_shared_expert_up_coords=(),
    moe_shared_expert_down_coords=(),
    # Runtime / fabric, carried over from GLM-5 unchanged.
    max_packet_payload_size_bytes=15232,
    fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    trace_region_size=573440,
    worker_l1_size=1419280,
    fabric_router_sync_timeout_ms=30000,
    fold_rmsnorm_gamma=False,
    defer_norm=False,
)
