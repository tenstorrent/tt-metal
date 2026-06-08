# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep SdpaDecode program configs (the WAN decode attention op, report ID [12340]).

Geometry from test_wan_decode_ops_sequence_univ.py (batch=32, decode = 1 token/user):
  D=1536, NUM_HEADS=12, NUM_KV_HEADS=2, HEAD_DIM=128 (GQA), cur_pos=128.
  Q is height-sharded [padded_heads, head_dim]; K/V are paged KV caches
  [BATCH, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM]; out is DRAM-interleaved (GQA can't shard out).

Optimization strategy — why these levers:
  At batch=32 on a 64-core grid, batch*num_kv_heads = 32*2 = 64 already saturates the
  grid (one (batch, kv_head) pair per core, cores_per_head=1). So the flash-decode
  parallelism knob `max_cores_per_head_batch` is a NO-OP here — confirmed by the program
  factory's core-allocation math — and the op is purely KV-cache-bandwidth-bound: each
  core serially reads its full KV slice. The dominant lever is therefore the KV-cache
  dtype (bf16 -> bfp8_b roughly halves the bytes read, ~halving latency). Secondary
  levers: k_chunk_size (how the KV length is tiled — pure perf), exp_approx_mode, and
  math fidelity (LoFi). dtype/approx/LoFi trade accuracy; k_chunk_size does not.

The decode inputs are built ONCE (layernorm -> qkv -> create-heads -> rope -> cache
update), and the KV caches are materialized in each swept dtype. The sweep loop then
varies only the SDPA-decode program_config, compute kernel, and KV-cache dtype, timing
the single op via the device profiler.

Each config runs once; CSV-style rows printed fastest-first, ERROR rows last.
Baseline: bf16 cache, k128, HiFi2, exp_approx=False (matches the WAN decode report).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from tracy.process_device_log import import_log_run_stats
import tracy.device_post_proc_config as device_post_proc_config
from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

# Shapes (decode: seq_len 1, batch in the height dim)
BATCH = 32
D = 1536
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = D // NUM_HEADS  # 128
QKV = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM  # 2048

# KV-cache geometry for the decode step
MAX_SEQ = 1024
CUR_POS = 128  # current decode position for all users

HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
)
LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _mesh_mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if getattr(device, "get_num_devices", lambda: 1)() > 1 else None


def _to_dev(t, device, dtype=ttnn.bfloat8_b):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=_mesh_mapper(device))


def sdpa_decode_pc(grid, k, exp_approx):
    # q_chunk_size is unused in decode; carried at 32 for a well-formed config.
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=k,
        exp_approx_mode=exp_approx,
    )


BF16, BFP8 = ttnn.bfloat16, ttnn.bfloat8_b

# (label, k_chunk_size, exp_approx_mode, compute_kernel, kv_cache_dtype)
SDPA_CONFIGS = [
    ("BASE bf16 k128 HIFI2", 128, False, HIFI2, BF16),  # WAN decode report config
    ("bf16 k32 HIFI2", 32, False, HIFI2, BF16),
    ("bf16 k64 HIFI2", 64, False, HIFI2, BF16),
    ("bf16 k128 HIFI2 approx", 128, True, HIFI2, BF16),
    ("bf16 k64 LoFi approx", 64, True, LOFI, BF16),
    # --- bfp8 KV cache: ~half the cache bandwidth (the dominant decode cost) ---
    ("bfp8 k128 HIFI2", 128, False, HIFI2, BFP8),
    ("bfp8 k32 HIFI2", 32, False, HIFI2, BFP8),
    ("bfp8 k64 HIFI2", 64, False, HIFI2, BFP8),
    ("bfp8 k128 HIFI2 approx", 128, True, HIFI2, BFP8),
    ("bfp8 k64 HIFI2 approx", 64, True, HIFI2, BFP8),
    ("bfp8 k128 LoFi approx", 128, True, LOFI, BFP8),  # most aggressive
    ("bfp8 k64 LoFi approx", 64, True, LOFI, BFP8),
]


def _read_device_kernel_us():
    """Device kernel duration [us] of the single op currently in the profiler log."""
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG
    data = import_log_run_stats(setup)
    freq_mhz = data["deviceInfo"]["freq"]
    cycles = data["devices"][0]["cores"]["DEVICE"]["analysis"]["device_kernel_duration"]["stats"]["Average"]
    return cycles / freq_mhz  # cycles / MHz == us


def _build_decode_inputs(device):
    """Run the WAN-decode prep ops once to produce height-sharded Q and paged K/V caches."""
    x = _to_dev(torch.randn(1, 1, BATCH, D), device, dtype=ttnn.bfloat16)
    ln_w = _to_dev(torch.randn(D), device, dtype=ttnn.bfloat16)
    ln_b = _to_dev(torch.randn(D), device, dtype=ttnn.bfloat16)
    qkv_w = _to_dev(torch.randn(D, QKV), device, dtype=ttnn.bfloat8_b)
    cos = _to_dev(torch.randn(1, 1, MAX_SEQ, HEAD_DIM), device, dtype=ttnn.bfloat16)
    sin = _to_dev(torch.randn(1, 1, MAX_SEQ, HEAD_DIM), device, dtype=ttnn.bfloat16)
    k_cache = _to_dev(torch.zeros(BATCH, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM), device, dtype=ttnn.bfloat16)
    v_cache = _to_dev(torch.zeros(BATCH, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM), device, dtype=ttnn.bfloat16)
    cur_pos = ttnn.from_torch(
        torch.full((BATCH,), CUR_POS, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=_mesh_mapper(device),
    )

    x_ln = ttnn.layer_norm(x, weight=ln_w, bias=ln_b, epsilon=1e-6)
    qkv = ttnn.matmul(x_ln, qkv_w, dtype=ttnn.bfloat16, compute_kernel_config=HIFI2)
    qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
    )
    head_sharded_mem = q.memory_config()

    q = ttnn.sharded_to_interleaved(q, DRAM)
    k = ttnn.sharded_to_interleaved(k, DRAM)
    q = ttnn.experimental.rotary_embedding(q, cos, sin, CUR_POS)
    k = ttnn.experimental.rotary_embedding(k, cos, sin, CUR_POS)

    k_upd = ttnn.to_memory_config(k, head_sharded_mem)
    v = ttnn.sharded_to_interleaved(v, DRAM)
    v_upd = ttnn.to_memory_config(v, head_sharded_mem)
    ttnn.experimental.paged_update_cache(k_cache, k_upd, update_idxs_tensor=cur_pos)
    ttnn.experimental.paged_update_cache(v_cache, v_upd, update_idxs_tensor=cur_pos)

    q_sdpa = ttnn.to_memory_config(q, head_sharded_mem)  # height-sharded Q for SDPA decode

    # Materialize the populated caches in every swept dtype (bf16 built above; bfp8 by
    # typecast) so the loop can compare cache-bandwidth without re-running the prep.
    caches = {BF16: (k_cache, v_cache), BFP8: (ttnn.typecast(k_cache, BFP8), ttnn.typecast(v_cache, BFP8))}
    ttnn.synchronize_device(device)
    return q_sdpa, caches, cur_pos


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_sdpa_decode_sweep(mesh_device):
    if os.getenv("TT_METAL_DEVICE_PROFILER") is None:
        pytest.skip(
            "device-time sweep needs a profiler build: rebuild with --enable-profiler and set TT_METAL_DEVICE_PROFILER=1"
        )

    device = mesh_device  # 4x n150 (TP4); the decode SDPA runs replicated -> per-device time
    if int(device.get_num_devices()) < 4:
        pytest.skip("requires 4 devices (4x n150)")

    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    q_sdpa, caches, cur_pos = _build_decode_inputs(device)

    # Run each config's SDPA-decode op once (warmup compiles the kernel; timed run is the one
    # captured by the Tracy ops-perf report). The device kernel duration is read from the
    # ops_perf_results CSV after the run (filter to SDPAOperation rows, device 0). The Python
    # side just prints the SUCCESSFUL config order so CSV rows map to configs.
    run_order = []
    for idx, (label, k_chunk, exp_approx, ck, kv_dtype) in enumerate(SDPA_CONFIGS):
        logger.info(f"=== CFG {idx}: {label} ===")
        pc = sdpa_decode_pc(grid, k_chunk, exp_approx)
        k_cache, v_cache = caches[kv_dtype]
        try:
            for _ in range(2):  # [warmup, timed] -> 2 SDPA rows per config in the CSV
                out = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_sdpa,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=cur_pos,
                    scale=HEAD_DIM**-0.5,
                    program_config=pc,
                    compute_kernel_config=ck,
                    memory_config=DRAM,
                )
                ttnn.synchronize_device(device)
                ttnn.deallocate(out)
            run_order.append((idx, label, "OK"))
            logger.info(f"=== CFG {idx} OK ===")
        except Exception as e:  # OOM / FATAL (e.g. k_chunk_size > padded seq) — record, keep sweeping
            run_order.append((idx, label, f"ERR: {str(e).strip().splitlines()[0][:80]}"))
            logger.error(f"=== CFG {idx} FAILED: {str(e)[:140]} ===")
        ttnn.synchronize_device(device)

    logger.info(
        f"SDPA-decode sweep: batch={BATCH} heads={NUM_HEADS} kv_heads={NUM_KV_HEADS} "
        f"head_dim={HEAD_DIM} cur_pos={CUR_POS}  (2 SDPA runs/config: warmup+timed)"
    )
    logger.info("SDPA_SWEEP_ORDER (successful configs, in CSV order):")
    for idx, label, status in run_order:
        logger.info(f"  CFG {idx:>2}  {label:<24} {status}")
    else:
        logger.info("=== BEST: none (every config errored) ===")
