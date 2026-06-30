# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage wall-clock perf on the BH Galaxy host-bounce pipeline.

Runs each stage in isolation on its assigned submesh, with warmup + N timed
iters, and reports per-stage latency + per-host-bounce overhead. Useful for
isolating which stage dominates wall-clock and how much the host-bounce
transport costs vs the on-device compute.

All tests require:
  - PI05_CHECKPOINT_DIR pointed at pi05_libero_upstream
  - BH Galaxy 8x4 = 32 chips
  - `source models/experimental/pi0_5/pi05_production.env` (recommended) for the
    validated 97.2% LIBERO production knobs.

Run all:
    PYTHONPATH=$PWD TT_METAL_HOME=$PWD python_env/bin/pytest \
      -xvs models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_stages.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx import stages
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.tt_bh_glx.transport import send_via_host


CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
NUM_WARMUP = int(os.environ.get("PI05_GLX_NUM_WARMUP", "1"))
NUM_ITERS = int(os.environ.get("PI05_GLX_NUM_ITERS", "5"))

_skip_if_no_ckpt = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)


def _summary(name: str, times_ms: List[float]) -> None:
    if not times_ms:
        return
    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    print(f"\n  {name}")
    print(f"    iters={len(times_ms)}  avg={avg:7.2f} ms  min={mn:7.2f}  max={mx:7.2f}  stddev={sd:6.2f}")


def _sync(submesh) -> None:
    ttnn.synchronize_device(submesh)


def test_perf_host_bounce_microbench():
    """Time send_via_host across distinct chip pairs at a few tensor sizes.

    Isolates the PCIe roundtrip cost without any device compute. Useful as
    a baseline to subtract from end-to-end stage latencies.
    """
    print("\n--- HOST BOUNCE MICROBENCH (no compute, just PCIe roundtrip) ---")
    with open_galaxy_mesh(l1_small_size=24576) as h:
        # Pairs: (label, src, dst, source-distance-class)
        pairs = [
            ("vision[0]→vision[1]", h.vision_per_chip[0], h.vision_per_chip[1]),
            ("vision[3]→prefill[0]", h.vision_per_chip[3], h.prefill_per_chip[0]),
            ("prefill[17]→denoise[0]", h.prefill_per_chip[17], h.denoise_per_chip[0]),
            ("denoise[0]→denoise[5]", h.denoise_per_chip[0], h.denoise_per_chip[5]),
        ]
        # Production shapes from the pipeline.
        shapes_bf16 = [
            ("vision_out_3cam", (3, 256, 2048)),  # vision → prefill activation
            ("prefill_hidden_1024", (1, 1024, 2048)),  # VLM hidden between blocks
            ("expert_hidden_32", (1, 32, 1024)),  # expert hidden between chunks
            ("prefix_kv_1024", (1, 1, 1024, 256)),  # one K or V tensor in migration
        ]
        torch.manual_seed(SEED)
        for label, src, dst in pairs:
            print(f"\n  pair: {label}")
            for sname, shape in shapes_bf16:
                host_t = torch.randn(*shape, dtype=torch.float32)
                src_t = ttnn.from_torch(
                    host_t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=src,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                # Warmup
                for _ in range(NUM_WARMUP):
                    out = send_via_host(src_t, dst)
                    _sync(dst)
                    ttnn.deallocate(out)
                times: List[float] = []
                for _ in range(NUM_ITERS):
                    _sync(src)
                    t0 = time.perf_counter()
                    out = send_via_host(src_t, dst)
                    _sync(dst)
                    times.append((time.perf_counter() - t0) * 1000.0)
                    ttnn.deallocate(out)
                ttnn.deallocate(src_t)
                n_bytes = int(torch.tensor(shape).prod()) * 2  # bf16
                avg = statistics.mean(times)
                gbps = (n_bytes / 1e9) / (avg / 1000.0) if avg > 0 else 0.0
                print(
                    f"    shape={sname:<22} {tuple(shape)!s:<24} "
                    f"avg={avg:6.2f} ms  ({n_bytes/1e6:5.2f} MB → {gbps:5.2f} GB/s)"
                )


@_skip_if_no_ckpt
def test_perf_vision_stage():
    """4-chip SigLIP + mm_projector wall-clock (steady-state)."""
    from models.experimental.pi0_5.common.configs import SigLIPConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.stage_vision import StageVision

    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    bs = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    torch.manual_seed(SEED)
    pixel_values = torch.randn(bs, 3, cfg.image_size, cfg.image_size)

    print(f"\n--- VISION STAGE (4 chips, bs={bs}) ---")
    with open_galaxy_mesh(l1_small_size=24576) as h:
        stage = StageVision(cfg, loader.categorized_weights, h)
        # Warmup includes JIT compile of all 27 SigLIP layers + mm_projector.
        for _ in range(NUM_WARMUP):
            out = stage.run(pixel_values)
            _sync(h.vision_per_chip[3])
            ttnn.deallocate(out)
        times: List[float] = []
        for i in range(NUM_ITERS):
            t0 = time.perf_counter()
            out = stage.run(pixel_values)
            _sync(h.vision_per_chip[3])
            times.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(out)
            print(f"    iter {i + 1:2d}: {times[-1]:7.2f} ms")
        _summary("vision (embed + 9+9+9 SigLIP layers + post_ln + mm_projector + 3 host bounces)", times)


@_skip_if_no_ckpt
def test_perf_prefill_stage():
    """18-chip VLM prefill wall-clock at production VLM chunk size."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.stage_prefill import StagePrefill

    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5")),
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    seq_len = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "1024"))
    torch.manual_seed(SEED)
    prefix_embs = torch.randn(1, seq_len, cfg.vlm_config.width) * 0.5

    print(f"\n--- PREFILL STAGE (18 chips, seq_len={seq_len}) ---")
    with open_galaxy_mesh(l1_small_size=24576) as h:
        stage = StagePrefill(cfg, loader.categorized_weights, h)
        for _ in range(NUM_WARMUP):
            prefix_ttnn = ttnn.from_torch(
                prefix_embs,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=h.prefill_per_chip[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out, _ = stage.run(prefix_ttnn, attention_mask=None, position_ids=None)
            _sync(h.prefill_per_chip[-1])
            ttnn.deallocate(out)
            ttnn.deallocate(prefix_ttnn)
        times: List[float] = []
        for i in range(NUM_ITERS):
            prefix_ttnn = ttnn.from_torch(
                prefix_embs,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=h.prefill_per_chip[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            t0 = time.perf_counter()
            out, _ = stage.run(prefix_ttnn, attention_mask=None, position_ids=None)
            _sync(h.prefill_per_chip[-1])
            times.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(out)
            ttnn.deallocate(prefix_ttnn)
            print(f"    iter {i + 1:2d}: {times[-1]:7.2f} ms")
        _summary(
            f"prefill (18 Gemma-2B blocks + final RMS norm + 17 host bounces) @ seq={seq_len}",
            times,
        )


@_skip_if_no_ckpt
def test_perf_denoise_expert_chain():
    """6-chip 18-layer AdaRMS expert chain wall-clock (one denoise step)."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.stage_denoise import StageDenoise

    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5")),
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))

    B = 1
    suffix_len = ((cfg.action_horizon + 31) // 32) * 32
    prefix_len = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "1024"))
    torch.manual_seed(SEED)
    suffix_hidden = torch.randn(B, suffix_len, cfg.expert_config.width) * 0.5
    adarms_cond = torch.randn(B, cfg.expert_config.width) * 0.5
    n_per = stages.EXPERT_LAYERS_PER_CHIP  # 3
    head_dim = cfg.expert_config.head_dim
    num_kv_heads = cfg.expert_config.num_kv_heads
    prefix_kv_torch = [
        (
            torch.randn(B, num_kv_heads, prefix_len, head_dim) * 0.1,
            torch.randn(B, num_kv_heads, prefix_len, head_dim) * 0.1,
        )
        for _ in range(cfg.expert_config.depth)
    ]

    print(f"\n--- DENOISE EXPERT CHAIN (6 chips × 3 layers, suffix={suffix_len}, prefix={prefix_len}) ---")
    with open_galaxy_mesh(l1_small_size=24576) as h:
        stage = StageDenoise(cfg, loader.categorized_weights, h)
        # Upload static synthetic prefix KV once (bf8_b to match expert internal dtype).
        prefix_kv_per_chip = []
        for c in range(stages.DENOISE_NUM_CHIPS):
            chip_kv = []
            for j in range(n_per):
                kt, vt = prefix_kv_torch[c * n_per + j]
                k = ttnn.from_torch(kt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=h.denoise_per_chip[c])
                v = ttnn.from_torch(vt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=h.denoise_per_chip[c])
                chip_kv.append((k, v))
            prefix_kv_per_chip.append(chip_kv)
        adarms_per_chip = [
            ttnn.from_torch(
                adarms_cond,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=h.denoise_per_chip[c],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for c in range(stages.DENOISE_NUM_CHIPS)
        ]
        for _ in range(NUM_WARMUP):
            hidden_t = ttnn.from_torch(
                suffix_hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=h.denoise_per_chip[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = stage.run_expert_chain(hidden_t, adarms_per_chip, prefix_kv_per_chip)
            _sync(h.denoise_per_chip[-1])
            ttnn.deallocate(out)
        times: List[float] = []
        for i in range(NUM_ITERS):
            hidden_t = ttnn.from_torch(
                suffix_hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=h.denoise_per_chip[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            t0 = time.perf_counter()
            out = stage.run_expert_chain(hidden_t, adarms_per_chip, prefix_kv_per_chip)
            _sync(h.denoise_per_chip[-1])
            times.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(out)
            print(f"    iter {i + 1:2d}: {times[-1]:7.2f} ms")
        _summary(
            f"denoise expert chain (18 AdaRMS blocks + 5 host bounces) @ suffix={suffix_len}, prefix={prefix_len}",
            times,
        )
