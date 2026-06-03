# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""L1 footprint probe — Option B TP=8 with L1-resident weights.

Tests the hypothesis from docs/OPTION_B_L1_ASSESSMENT.md: at TP=8 the
matmul kernel's static circular-buffer region shrinks 7x (out_block_w
drops from ~43 single-chip to ~6 sharded), which should leave enough
L1 above the CB region for the L1-resident weights to fit.

If this hypothesis holds:
- initialize() succeeds with L1 weights migrated (no upload OOM).
- run_one_step() completes without the CB clash that blocks Option C.
- per-chip L1 footprint reports ~125 MB weights + small transients,
  well within the 175.4 MB L1 cap.

If it fails:
- A different bug surfaces (CB clash address, all_reduce CB conflict,
  fabric init mismatch, etc.). The failure address pins down which
  kernel's CB region is the actual bottleneck.

Opt in via PI0_OB_L1_PROBE=1. Skipped otherwise.

Workload — identical constants to test_option_b_benchmark_e2e.py and
test_option_c_l1_footprint_probe.py so numbers compare directly.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_b.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.option_b.pipeline import Pi0_5PipelineB
from models.experimental.pi0_5.tt.option_b.stages import build_shrunk_layout, build_default_layout

# ----------------------------------------------------------------------------#
# Skip gate                                                                   #
# ----------------------------------------------------------------------------#

PROBE_ENABLED = os.environ.get("PI0_OB_L1_PROBE") == "1"
pytestmark = pytest.mark.skipif(
    not PROBE_ENABLED,
    reason="set PI0_OB_L1_PROBE=1 to run the Option B L1 footprint probe",
)

# ----------------------------------------------------------------------------#
# Workload — matches test_option_b_benchmark_e2e.py and the trace test        #
# ----------------------------------------------------------------------------#

LANG_SEQ_LEN = 256
NUM_IMAGE_TOKENS = 256
PREFIX_LEN = NUM_IMAGE_TOKENS + LANG_SEQ_LEN  # 512
ACTION_DIM = 32
ACTION_HORIZON = 10
ACTION_HORIZON_PADDED = 32
NUM_DENOISE_STEPS = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "10"))
BATCH_SIZE = 1

# ----------------------------------------------------------------------------#
# Probe knobs                                                                 #
# ----------------------------------------------------------------------------#

_DEFAULT_CKPT = "/home/tt-admin/pi05_cache/pi05_libero_upstream"
CHECKPOINT_DIR = Path(
    os.environ.get("PI0_OB_L1_PROBE_CHECKPOINT") or os.environ.get("PI05_CHECKPOINT_DIR") or _DEFAULT_CKPT
)
# Default to full depth — the whole point of this probe is to validate
# the L1 placement at the deployment-target shape, not the shrunk bench
# shape. Both depths support the same code paths.
VLM_DEPTH = int(os.environ.get("PI0_OB_L1_PROBE_VLM_DEPTH", "18"))
EXPERT_DEPTH = int(os.environ.get("PI0_OB_L1_PROBE_EXPERT_DEPTH", "18"))
RUN_FORWARD = os.environ.get("PI0_OB_L1_PROBE_RUN_FORWARD", "1") == "1"
# Option B's pipeline flags. tp_shard=1 builds the TP=8 path (otherwise
# replicated, which is meaningless for L1 placement). weights_l1=1
# triggers the post-construction migration via _l1_migration.
TP_SHARD = os.environ.get("PI0_OB_L1_PROBE_TP_SHARD", "1") == "1"
WEIGHTS_L1 = os.environ.get("PI0_OB_L1_PROBE_WEIGHTS_L1", "1") == "1"
# Standard pi0_5 value; required for fabric + all_reduce on the TP=8 path.
L1_SMALL_SIZE = int(os.environ.get("PI0_OB_L1_PROBE_L1_SMALL_SIZE", "24576"))

STAGE_NAMES = ("vision", "vlm_first_half", "vlm_second_half", "expert_denoise")

# ----------------------------------------------------------------------------#
# L1+DRAM measurement (mirrors Option C probe)                                #
# ----------------------------------------------------------------------------#


def _read_per_chip_view(submesh, buffer_type) -> Tuple[float, float, float]:
    mv = ttnn._ttnn.device.GetMemoryView(submesh, buffer_type)
    num_banks = int(mv.num_banks)
    used = (int(mv.total_bytes_allocated_per_bank) * num_banks) / 1e6
    free = (int(mv.total_bytes_free_per_bank) * num_banks) / 1e6
    cap = (int(mv.total_bytes_per_bank) * num_banks) / 1e6
    return used, free, cap


def _summarise_submesh(submesh, name: str) -> Dict[str, float]:
    l1_used, l1_free, l1_cap = _read_per_chip_view(submesh, ttnn.BufferType.L1)
    dram_used, dram_free, dram_cap = _read_per_chip_view(submesh, ttnn.BufferType.DRAM)
    return {
        "stage": name,
        "n_chips": submesh.get_num_devices(),
        "l1_used_mb": l1_used,
        "l1_free_mb": l1_free,
        "l1_cap_mb": l1_cap,
        "dram_used_mb": dram_used,
        "dram_free_mb": dram_free,
        "dram_cap_mb": dram_cap,
    }


def _print_phase(phase: str, snapshots: List[Dict[str, float]]) -> None:
    print(f"\n[{phase}]")
    for s in snapshots:
        print(
            f"  stage={s['stage']:<18}  chips={s['n_chips']:2d}  "
            f"L1  used = {s['l1_used_mb']:7.1f} MB / chip   "
            f"(cap {s['l1_cap_mb']:.1f} MB, free {s['l1_free_mb']:7.1f} MB)"
        )
        print(
            f"  stage={s['stage']:<18}  chips={s['n_chips']:2d}  "
            f"DRAM used = {s['dram_used_mb']:7.1f} MB / chip   "
            f"(cap {s['dram_cap_mb']:.1f} MB, free {s['dram_free_mb']:7.1f} MB)"
        )


@contextmanager
def _phase(phase_name: str, submeshes: List, results: Dict[str, List[Dict]]):
    try:
        yield
    finally:
        snaps: List[Dict[str, float]] = []
        for sm, sname in zip(submeshes, STAGE_NAMES):
            snaps.append(_summarise_submesh(sm, sname))
        results[phase_name] = snaps
        _print_phase(phase_name, snaps)


# ----------------------------------------------------------------------------#
# Inputs (same shape contract as test_option_b_benchmark_e2e.py)              #
# ----------------------------------------------------------------------------#


def _matched_inputs(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    pixel_values = torch.randn(BATCH_SIZE, 3, 224, 224, generator=gen, dtype=torch.float32)
    lang_tokens = torch.randint(0, 256000, (BATCH_SIZE, LANG_SEQ_LEN), generator=gen, dtype=torch.int32)
    noisy_actions = torch.zeros(BATCH_SIZE, ACTION_HORIZON_PADDED, ACTION_DIM, dtype=torch.float32)
    noisy_actions[:, :ACTION_HORIZON, :] = torch.randn(
        BATCH_SIZE, ACTION_HORIZON, ACTION_DIM, generator=gen, dtype=torch.float32
    )
    return pixel_values, lang_tokens, noisy_actions


def _load_real_weights():
    if not (CHECKPOINT_DIR / "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {CHECKPOINT_DIR}")
    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    return Pi0_5WeightLoader(CHECKPOINT_DIR).categorized_weights


def _echo_env():
    flags = [
        "PI0_UPSTREAM_MASKS",
        "QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT",
        "QWEN_NLP_CREATE_HEADS_HEAD_SPLIT",
        "PI05_CHECKPOINT_DIR",
        "PI05_NUM_DENOISE_STEPS",
        "TT_VISIBLE_DEVICES",
        "TT_METAL_HOME",
    ]
    print("[env]")
    for k in flags:
        print(f"  {k} = {os.environ.get(k, '<unset>')}")
    print("[probe knobs]")
    print(f"  PI0_OB_L1_PROBE_CHECKPOINT       = {CHECKPOINT_DIR}")
    print(f"  PI0_OB_L1_PROBE_VLM_DEPTH        = {VLM_DEPTH}")
    print(f"  PI0_OB_L1_PROBE_EXPERT_DEPTH     = {EXPERT_DEPTH}")
    print(f"  PI0_OB_L1_PROBE_RUN_FORWARD      = {RUN_FORWARD}")
    print(f"  PI0_OB_L1_PROBE_TP_SHARD         = {TP_SHARD}")
    print(f"  PI0_OB_L1_PROBE_WEIGHTS_L1       = {WEIGHTS_L1}")
    print(f"  PI0_OB_L1_PROBE_L1_SMALL_SIZE    = {L1_SMALL_SIZE}")


# ----------------------------------------------------------------------------#
# Main probe                                                                  #
# ----------------------------------------------------------------------------#


@pytest.mark.timeout(3600)
def test_ob_l1_footprint_probe_full_depth():
    """Probe Option B TP=8 with L1-resident weights at full depth.

    Phases:
        baseline (pre-init)
        after Pi0_5PipelineB.initialize  (incl. L1 migration when weights_l1)
        after warmup forward             (run_one_step + denoise loop)
    """
    _echo_env()

    cfg = PaliGemmaConfig()  # gemma_2b VLM + gemma_300m expert, full depth
    # Option B's StageLayout supports build_default_layout (full depth)
    # vs build_shrunk_layout (used by bench). For the L1 hypothesis test
    # we want full depth so per-chip footprint is the real number.
    if VLM_DEPTH == 18 and EXPERT_DEPTH == 18:
        layout = build_default_layout()
    else:
        layout = build_shrunk_layout(vlm_depth=VLM_DEPTH, expert_depth=EXPERT_DEPTH)
    print(f"\n[cfg] vlm_depth={cfg.vlm_config.depth} expert_depth={cfg.expert_config.depth}")
    print(f"[layout] parent={layout.parent_mesh_shape} stages={len(layout.stages)}")
    for s in layout.stages:
        n_chips = s.submesh_shape[0] * s.submesh_shape[1]
        print(f"  - {s.name:<18} submesh_shape={s.submesh_shape} num_chips={n_chips}")

    weights = _load_real_weights()
    print(f"[weights] keys: {sorted(weights.keys())}")

    pixel_values, lang_tokens, noisy_actions = _matched_inputs()
    print(
        f"[inputs] pixel_values={tuple(pixel_values.shape)} "
        f"lang_tokens={tuple(lang_tokens.shape)} noisy_actions={tuple(noisy_actions.shape)}"
    )

    results: Dict[str, List[Dict]] = {}

    # TP=8 requires fabric for the all_reduces inside each block.
    with open_galaxy_mesh(layout, enable_fabric=True, l1_small_size=L1_SMALL_SIZE) as (
        _parent,
        submeshes,
    ):
        assert len(submeshes) == 4

        with _phase("baseline (pre-init)", submeshes, results):
            pass

        pipe = Pi0_5PipelineB(
            layout=layout,
            submeshes=submeshes,
            config=cfg,
            weights=weights,
            tp_shard=TP_SHARD,
            weights_l1=WEIGHTS_L1,
        )

        try:
            with _phase("after Pi0_5PipelineB.initialize", submeshes, results):
                pipe.initialize()
        except Exception as e:
            print(f"\n[FAIL] initialize() raised: {type(e).__name__}: {e}")
            print("[FAIL] L1 snapshot above shows state at fault.")
            raise

        if RUN_FORWARD:
            # We just want to verify the L1-resident matmul forward works.
            # Run stage 0 (host vision) + transport + stage 1 (VLM with TP=8
            # L1 weights). The matmul + all_reduce inside stage_1's forward
            # is the path that would trip the CB clash if the hypothesis
            # were wrong.
            try:
                with _phase("after warmup forward", submeshes, results):
                    from models.experimental.pi0_5.tt.option_b.transport import (
                        send_activation_via_host,
                    )

                    prefix_s0 = pipe.stage_0.forward(pixel_values, lang_tokens)
                    sm1 = submeshes[1]
                    h_on_1 = send_activation_via_host(prefix_s0, sm1)
                    ttnn.deallocate(prefix_s0)
                    S_prefix = h_on_1.shape[1]
                    # Mask in DRAM (SDPA precondition; see option_c/pipeline.py).
                    mask_torch = torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32)
                    mask_s1 = ttnn.from_torch(
                        mask_torch,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=sm1,
                        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(sm1),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    h_after_1 = pipe.stage_1.forward(h_on_1, attention_mask=mask_s1, use_cache=True)
                    ttnn.deallocate(h_on_1)
                    ttnn.deallocate(mask_s1)
                    ttnn.deallocate(h_after_1)
            except Exception as e:
                print(f"\n[FAIL] forward raised: {type(e).__name__}: {e}")
                print("[FAIL] L1 snapshot above shows state at fault.")
                raise

    # Final summary table
    print("\n" + "=" * 90)
    print("== Option B L1 footprint — TP=8 + L1 weights, full depth, real ckpt ==")
    print("=" * 90)
    l1_cap = results["baseline (pre-init)"][0]["l1_cap_mb"]
    dram_cap = results["baseline (pre-init)"][0]["dram_cap_mb"]
    print(f"L1   cap per chip: {l1_cap:.1f} MB")
    print(f"DRAM cap per chip: {dram_cap:.1f} MB")
    print()
    header = (
        f"{'phase':<36}  {'stage':<18}  {'chips':>5}  "
        f"{'L1 used':>9}  {'L1 free':>9}  {'DRAM used':>10}  {'DRAM free':>10}  (MB/chip)"
    )
    print(header)
    print("-" * len(header))
    for phase_name, snaps in results.items():
        for s in snaps:
            print(
                f"{phase_name:<36}  {s['stage']:<18}  {s['n_chips']:>5d}  "
                f"{s['l1_used_mb']:>9.1f}  {s['l1_free_mb']:>9.1f}  "
                f"{s['dram_used_mb']:>10.1f}  {s['dram_free_mb']:>10.1f}"
            )

    init_snap = results.get("after Pi0_5PipelineB.initialize", [])
    if init_snap:
        total_post_init = sum(s["l1_used_mb"] + s["dram_used_mb"] for s in init_snap)
        assert total_post_init > 0, "post-init L1 + DRAM usage is zero on every chip — MemoryView silently empty?"
