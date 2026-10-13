# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-turn prefill through TtPrefillRuntime.prefill_chunk: a multi-turn continuation resumed mid-slab must fill
the KV cache exactly like one uninterrupted prefill of the same tokens.

Turn 1 is 5119 tokens (one ragged chunk); the scheduler resumes turn 2 at align_down(5119, 32) = 5088, so turn 2's
chunk [5088, 9088) starts mid-slab AND is ragged. The same 9088 tokens prefilled in one pass (chunks at 0 and
5120) are the reference, compared per layer on the device cache read back in natural order (K, V and, on the MSA
layers, index_k). This runs the whole runtime path the lower-level tests bypass: make_chunk_input's input rotation,
the KV write + indexed RoPE at rotated positions, the dense ring-joint and MSA cache reads, and the MoE padding
config for a rotated ragged chunk (layer 3's MoE feeds layer 4's KV).

A third slot feeds turn 2 WITHOUT the input rotation (the contiguous shard the runtime used before), which must
diverge -- so the test keeps its power to catch a regression in the input plumbing.

Real weights, first 5 layers (3 dense + 2 MSA/MoE), from the warm tilized weight cache (HF_MODEL / TT_CACHE_PATH);
skipped when that cache is not available rather than paying the multi-hour bf16 source read.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ..test_factory import parametrize_mesh_with_fabric

NUM_LAYERS = 5
CHUNK = 5120
TURN1_END = 5119
RESUME = TURN1_END // ttnn.TILE_SIZE * ttnn.TILE_SIZE  # 5088: where the prefill scheduler resumes turn 2
TOTAL = 9088  # turn 2 = one ragged chunk [5088, 9088)
CAPACITY = 3 * CHUNK  # holds RESUME + CHUNK
PCC = 0.99
SLOT_RESUMED, SLOT_ONE_PASS, SLOT_UNROTATED = 0, 1, 2


@pytest.mark.timeout(1800)  # 5-layer real-weight build + 6 chunks + 3 slot read-backs exceed the default 300s
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_prefill_runtime_multiturn_resume(mesh_device, device_params, reset_seeds):
    if not os.getenv("HF_MODEL"):
        pytest.skip("needs real MiniMax-M3 weights (HF_MODEL + a warm TT_CACHE_PATH tilized cache)")
    from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    model_args = ModelArgs(mesh_device=mesh_device)
    hf_config = model_args.hf_config
    hf_config.num_hidden_layers = NUM_LAYERS
    os.environ.setdefault("M3_LOAD_NLAYERS", str(NUM_LAYERS))
    cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
    if not weight_cache_is_complete(cache_path, hf_config, NUM_LAYERS, ttnn.bfloat4_b):
        pytest.skip(f"tilized weight cache for the first {NUM_LAYERS} layers is not complete at {cache_path}")

    rows, cols = tuple(mesh_device.shape)
    cfg = TtPrefillRuntimeConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=CAPACITY,
        mesh_shape=(rows, cols),
        chunk_size=CHUNK,
        num_users=3,
        expert_weight_dtype=ttnn.bfloat4_b,
        weight_cache_path=cache_path,
        topology=ttnn.Topology.Linear,
    )
    runtime = TtPrefillRuntime(mesh_device, hf_config, {}, cfg)
    kv_cache = allocate_kv_caches(
        mesh_device, num_layers=NUM_LAYERS, max_seq_len=CAPACITY, num_users=3, head_dim=hf_config.head_dim
    )

    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, hf_config.vocab_size, (TOTAL,), generator=gen).tolist()

    def prefill(slot, start, end, rotate=True):
        chunk = tokens[start : start + CHUNK]
        chunk = chunk + [0] * (CHUNK - len(chunk))
        inp = runtime.make_chunk_input(chunk, start if rotate else 0)
        runtime.prefill_chunk(inp, kv_cache, slot_id=slot, actual_start=start, actual_end=end)

    prefill(SLOT_RESUMED, 0, TURN1_END)
    prefill(SLOT_RESUMED, RESUME, TOTAL)
    prefill(SLOT_ONE_PASS, 0, CHUNK)
    prefill(SLOT_ONE_PASS, CHUNK, TOTAL)
    prefill(SLOT_UNROTATED, 0, TURN1_END)
    prefill(SLOT_UNROTATED, RESUME, TOTAL, rotate=False)
    ttnn.synchronize_device(mesh_device)

    # One host read per slot (k, v, index_k for every layer), un-rotated to natural order per layer.
    def natural(slot):
        blocks = runtime.read_slot_kv(kv_cache, slot)
        return [
            [naturalize_kv_block(blk[layer], TOTAL, cfg.sp_factor, CHUNK, CAPACITY) for blk in blocks]
            for layer in range(NUM_LAYERS)
        ]

    ref_all, got_all, bad_all = natural(SLOT_ONE_PASS), natural(SLOT_RESUMED), natural(SLOT_UNROTATED)
    worst, worst_unrotated = 1.0, 1.0
    for layer in range(NUM_LAYERS):
        names = ("k", "v", "index_k") if layer >= 3 else ("k", "v")
        for name, r, g, b in zip(names, ref_all[layer], got_all[layer], bad_all[layer]):
            _, p = comp_pcc(r.float(), g.float(), PCC)
            _, p_bad = comp_pcc(r[..., RESUME:, :].float(), b[..., RESUME:, :].float(), PCC)
            logger.info(f"[multiturn] layer {layer} {name}: resumed pcc={p:.5f} (unrotated turn 2: {p_bad:.5f})")
            worst, worst_unrotated = min(worst, p), min(worst_unrotated, p_bad)
    logger.info(f"[multiturn] worst resumed pcc={worst:.5f}, worst unrotated pcc={worst_unrotated:.5f}")
    assert worst_unrotated < PCC, "test lost its power: an unrotated turn-2 input would pass too"
    assert worst >= PCC, f"resumed two-turn prefill diverges from the one-pass prefill (worst pcc={worst:.5f})"
