# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""L1 footprint probe — Option C at full VLM and expert depth on real weights.

Goal: produce an *empirical* per-chip L1 budget table for Option C at the
deployment-target depth (vlm_depth=18, expert_depth=18) on the real
libero_upstream checkpoint, matching the single-device trace test's
workload exactly so the numbers are directly comparable.

The bench paths run at shrunk depth (vlm_depth=2 expert_depth=1) because
the replicated upload was not known to fit at full depth — this probe
exists precisely to find out *whether* it fits and, if not, which stage
hits the cliff and how far over.

Opt in via `PI0_OC_L1_PROBE=1`. Skipped otherwise.

What this probe measures
------------------------

Per submesh, per chip:

  Phase                    L1 used (per chip)   L1 free (per chip)
  ─────────────────────────────────────────────────────────────────
  baseline (pre-init)      —                    cap
  after stage_0 init       vision weights only
  after stage_1 init       (same — different submesh)
  after stage_2 init       expert + suffix
  after warmup forward     + activations + KV + masks (peak)

Each chip's L1 view is read via `ttnn.get_memory_view(dev,
ttnn.BufferType.L1)`. We summarise across the submesh as min / median /
max bytes used (sharded uploads can produce non-uniform per-chip
footprints, especially in vision where layer-paired SigLIP splits).

Workload — identical to test_perf_ttnn_full_e2e_trace.py
--------------------------------------------------------

  LANG_SEQ_LEN          = 256
  prefix_len            = 256 + LANG_SEQ_LEN     # 512
  action_dim            = 32
  action_horizon        = 10                     # libero ckpt
  num_denoise_steps     = 10
  batch_size            = 1
  one 224x224 image

Env flags matched (must be set BEFORE `python -m pytest`):

  PI0_UPSTREAM_MASKS=1
  QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1
  QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
  PI05_NUM_DENOISE_STEPS=10
  PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream

Probe-specific knobs
--------------------

  PI0_OC_L1_PROBE              = unset      `1` to opt in
  PI0_OC_L1_PROBE_CHECKPOINT   = /home/tt-admin/pi05_cache/pi05_libero_upstream
  PI0_OC_L1_PROBE_VLM_DEPTH    = 18         full gemma_2b
  PI0_OC_L1_PROBE_EXPERT_DEPTH = 18         full gemma_300m
  PI0_OC_L1_PROBE_RUN_FORWARD  = 1          run one warmup forward
  PI0_OC_L1_PROBE_DEPTH_SWEEP  = unset      `1` to also try a depth sweep
                                            (2, 4, 8, 12, 18) if full depth
                                            OOMs — separate test
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_c.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.option_c.pipeline import Pi0_5PipelineC
from models.experimental.pi0_5.tt.option_c.stages import build_default_layout

# ----------------------------------------------------------------------------#
# Skip gate                                                                   #
# ----------------------------------------------------------------------------#

PROBE_ENABLED = os.environ.get("PI0_OC_L1_PROBE") == "1"
pytestmark = pytest.mark.skipif(
    not PROBE_ENABLED,
    reason="set PI0_OC_L1_PROBE=1 to run the Option C L1 footprint probe",
)

# ----------------------------------------------------------------------------#
# Workload — match test_perf_ttnn_full_e2e_trace.py exactly                   #
# ----------------------------------------------------------------------------#

LANG_SEQ_LEN = 256
NUM_IMAGE_TOKENS = 256
PREFIX_LEN = NUM_IMAGE_TOKENS + LANG_SEQ_LEN  # 512
ACTION_DIM = 32
ACTION_HORIZON = 10  # libero_upstream checkpoint
ACTION_HORIZON_PADDED = 32  # tile-aligned
NUM_DENOISE_STEPS = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "10"))
BATCH_SIZE = 1

# ----------------------------------------------------------------------------#
# Probe knobs                                                                 #
# ----------------------------------------------------------------------------#

_DEFAULT_CKPT = "/home/tt-admin/pi05_cache/pi05_libero_upstream"
CHECKPOINT_DIR = Path(
    os.environ.get("PI0_OC_L1_PROBE_CHECKPOINT") or os.environ.get("PI05_CHECKPOINT_DIR") or _DEFAULT_CKPT
)
VLM_DEPTH_PROBE = int(os.environ.get("PI0_OC_L1_PROBE_VLM_DEPTH", "18"))
EXPERT_DEPTH_PROBE = int(os.environ.get("PI0_OC_L1_PROBE_EXPERT_DEPTH", "18"))
RUN_FORWARD = os.environ.get("PI0_OC_L1_PROBE_RUN_FORWARD", "1") == "1"
DEPTH_SWEEP = os.environ.get("PI0_OC_L1_PROBE_DEPTH_SWEEP") == "1"
# Pi0_5PipelineC placement flags — toggle to compare DRAM-replicated vs
# L1-paired footprints. Defaults match the bench / smoke path so a fresh
# probe run reproduces the baseline numbers in OPTION_C_L1_FOOTPRINT_PROBE.md.
LAYER_PAIRED_L1 = os.environ.get("PI0_OC_L1_PROBE_LAYER_PAIRED") == "1"
DEVICE_SIGLIP = os.environ.get("PI0_OC_L1_PROBE_DEVICE_SIGLIP") == "1"
EXPERT_LAYERS_PER_CHIP = int(os.environ.get("PI0_OC_L1_PROBE_EXPERT_LAYERS_PER_CHIP", "3"))
# L1 small / static-CB reservation, per-bank. 24576 bytes = 24 KB is the
# value every working pi0.5 single-device test uses (see README.md:172,
# test_perf_ttnn_full_e2e_trace.py:95, libero_rollout.py:979, all of
# tests/pcc). The mesh path needs the same reservation when L1-resident
# weights are live or the matmul kernel's static CB region collides with
# L1-allocated buffers. 1 MB / bank (~120 MB / chip) is way too much and
# OOMs the layer-paired weights — DON'T do that. None = ttnn default (0).
_default_l1_small = "24576" if LAYER_PAIRED_L1 else ""
L1_SMALL_SIZE_RAW = os.environ.get("PI0_OC_L1_PROBE_L1_SMALL_SIZE", _default_l1_small)
L1_SMALL_SIZE: Optional[int] = int(L1_SMALL_SIZE_RAW) if L1_SMALL_SIZE_RAW else None

STAGE_NAMES = ("vision", "prefill", "denoise")

# ----------------------------------------------------------------------------#
# L1 measurement                                                              #
# ----------------------------------------------------------------------------#


def _read_per_chip_view(submesh, buffer_type) -> Tuple[float, float, float]:
    """Return (used_mb, free_mb, cap_mb) PER CHIP for a given buffer_type.

    `GetMemoryView(mesh, bt)` exposes scalar per-bank stats (uniform
    across banks). Per-chip total = per-bank × num_banks. The Python
    binding doesn't expose per-device variation across a multi-chip mesh
    — for replicated uploads (Option C's path today) that's fine.
    """
    mv = ttnn._ttnn.device.GetMemoryView(submesh, buffer_type)
    num_banks = int(mv.num_banks)
    used = (int(mv.total_bytes_allocated_per_bank) * num_banks) / 1e6
    free = (int(mv.total_bytes_free_per_bank) * num_banks) / 1e6
    cap = (int(mv.total_bytes_per_bank) * num_banks) / 1e6
    return used, free, cap


def _summarise_submesh(submesh, name: str) -> Dict[str, float]:
    """Per-chip L1 + DRAM stats for `submesh` — read in one pass."""
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
            f"  stage={s['stage']:<8}  chips={s['n_chips']:2d}  "
            f"L1  used = {s['l1_used_mb']:7.1f} MB / chip   "
            f"(cap {s['l1_cap_mb']:.1f} MB, free {s['l1_free_mb']:7.1f} MB)"
        )
        print(
            f"  stage={s['stage']:<8}  chips={s['n_chips']:2d}  "
            f"DRAM used = {s['dram_used_mb']:7.1f} MB / chip   "
            f"(cap {s['dram_cap_mb']:.1f} MB, free {s['dram_free_mb']:7.1f} MB)"
        )


# ----------------------------------------------------------------------------#
# Inputs — same builder as test_perf_ttnn_full_e2e_trace.py                   #
# ----------------------------------------------------------------------------#


def _trace_matched_inputs(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One 224x224 image, LANG_SEQ_LEN lang tokens, padded noisy actions.

    Mirrors `_build_inputs` in test_perf_ttnn_full_e2e_trace.py for
    shape/dtype. The pi0.5 pipeline's vision stage handles the image →
    256-token mapping internally.
    """
    gen = torch.Generator().manual_seed(seed)
    pixel_values = torch.randn(BATCH_SIZE, 3, 224, 224, generator=gen, dtype=torch.float32)
    lang_tokens = torch.randint(0, 256000, (BATCH_SIZE, LANG_SEQ_LEN), generator=gen, dtype=torch.int32)
    # Match ttnn_pi0_5_model.sample_actions: zero-pad to action_horizon_padded
    # and only fill the first action_horizon rows with N(0,1). The padding
    # rows should never affect output (attention masks null them) but
    # matching exactly avoids one variable when comparing numerics later.
    noisy_actions = torch.zeros(BATCH_SIZE, ACTION_HORIZON_PADDED, ACTION_DIM, dtype=torch.float32)
    noisy_actions[:, :ACTION_HORIZON, :] = torch.randn(
        BATCH_SIZE, ACTION_HORIZON, ACTION_DIM, generator=gen, dtype=torch.float32
    )
    return pixel_values, lang_tokens, noisy_actions


# ----------------------------------------------------------------------------#
# Weight loading                                                              #
# ----------------------------------------------------------------------------#


def _load_real_weights() -> Dict:
    if not (CHECKPOINT_DIR / "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {CHECKPOINT_DIR}")
    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    return Pi0_5WeightLoader(CHECKPOINT_DIR).categorized_weights


# ----------------------------------------------------------------------------#
# Env flag echo                                                               #
# ----------------------------------------------------------------------------#


def _echo_env() -> None:
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
        v = os.environ.get(k, "<unset>")
        print(f"  {k} = {v}")
    print(f"[probe knobs]")
    print(f"  PI0_OC_L1_PROBE_CHECKPOINT             = {CHECKPOINT_DIR}")
    print(f"  PI0_OC_L1_PROBE_VLM_DEPTH              = {VLM_DEPTH_PROBE}")
    print(f"  PI0_OC_L1_PROBE_EXPERT_DEPTH           = {EXPERT_DEPTH_PROBE}")
    print(f"  PI0_OC_L1_PROBE_RUN_FORWARD            = {RUN_FORWARD}")
    print(f"  PI0_OC_L1_PROBE_DEPTH_SWEEP            = {DEPTH_SWEEP}")
    print(f"  PI0_OC_L1_PROBE_LAYER_PAIRED           = {LAYER_PAIRED_L1}")
    print(f"  PI0_OC_L1_PROBE_DEVICE_SIGLIP          = {DEVICE_SIGLIP}")
    print(f"  PI0_OC_L1_PROBE_EXPERT_LAYERS_PER_CHIP = {EXPERT_LAYERS_PER_CHIP}")


# ----------------------------------------------------------------------------#
# Helpers — snapshot wrapper                                                  #
# ----------------------------------------------------------------------------#


@contextmanager
def _phase(phase_name: str, submeshes: List, results: Dict[str, List[Dict]], l1_targets: Optional[List] = None):
    """Run a block, then snapshot L1+DRAM on every submesh and stash under phase_name.

    `l1_targets` is a parallel list of submeshes to read L1 from — defaults
    to `submeshes` (full stage submeshes). When paired-mode stages have
    carved single-chip micro-submeshes, the caller swaps those in so the
    reads see the per-chip allocations the parent submesh can't.
    """
    targets = list(l1_targets) if l1_targets is not None else list(submeshes)
    try:
        yield
    finally:
        snaps: List[Dict[str, float]] = []
        for target, sname in zip(targets, STAGE_NAMES):
            snaps.append(_summarise_submesh(target, sname))
        results[phase_name] = snaps
        _print_phase(phase_name, snaps)


# ----------------------------------------------------------------------------#
# Main probe                                                                  #
# ----------------------------------------------------------------------------#


@pytest.mark.timeout(3600)
def test_oc_l1_footprint_probe_full_depth():
    """Run Option C at full depth and report per-chip L1 after each phase.

    The probe never asserts on absolute footprint — it is diagnostic.
    The only assertion is that we get *some* measurement back, so that a
    silently-empty MemoryView wouldn't pass for "fits".
    """
    _echo_env()

    cfg = PaliGemmaConfig()  # gemma_2b VLM + gemma_300m expert, full depth
    layout = build_default_layout()  # full depth: 18 VLM + 18 expert

    print(f"\n[cfg] vlm_depth = {cfg.vlm_config.depth}  expert_depth = {cfg.expert_config.depth}")
    print(f"[layout] parent = {layout.parent_mesh_shape}  stages = {len(layout.stages)}")
    for s in layout.stages:
        print(f"  - {s.name:<8} submesh_shape={s.submesh_shape} num_chips={s.num_chips}")

    weights = _load_real_weights()
    print("[weights] categorized weight buckets:", sorted(weights.keys()))

    pixel_values, lang_tokens, noisy_actions = _trace_matched_inputs()
    print(
        f"[inputs] pixel_values={tuple(pixel_values.shape)}  "
        f"lang_tokens={tuple(lang_tokens.shape)}  noisy_actions={tuple(noisy_actions.shape)}"
    )

    results: Dict[str, List[Dict]] = {}

    with open_galaxy_mesh(layout, l1_small_size=L1_SMALL_SIZE) as (_parent, submeshes):
        assert len(submeshes) == 3, f"expected 3 submeshes, got {len(submeshes)}"

        # Stage submeshes we read for each phase. After Pi0_5PipelineC.initialize()
        # the paired-mode stages carve single-chip micro-submeshes inside the
        # parent prefill / denoise submeshes — query those instead of the
        # parents, because GetMemoryView on a parent submesh doesn't see
        # allocations made on its carved children. Filled in below.
        l1_targets: List = list(submeshes)

        with _phase("baseline (pre-init)", submeshes, results, l1_targets):
            pass

        pipe = Pi0_5PipelineC(
            layout=layout,
            submeshes=submeshes,
            config=cfg,
            weights=weights,
            denoise_steps=NUM_DENOISE_STEPS,
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            layer_paired_l1=LAYER_PAIRED_L1,
            device_siglip=DEVICE_SIGLIP,
            expert_layers_per_chip=EXPERT_LAYERS_PER_CHIP,
        )

        # initialize is one call today; if it OOMs we capture state at the
        # failure point. The pipeline initializes stage_0 → stage_1 → stage_2
        # internally (see Pi0_5PipelineC.initialize).
        try:
            with _phase("after Pi0_5PipelineC.initialize", submeshes, results, l1_targets):
                pipe.initialize()
        except Exception as e:
            print(f"\n[FAIL] initialize() raised: {type(e).__name__}: {e}")
            print("[FAIL] L1 snapshot above shows state at fault.")
            print("[hint] re-run with PI0_OC_L1_PROBE_DEPTH_SWEEP=1 to bisect.")
            raise

        # Re-target L1 reads to per-chip micro-submeshes when paired-mode
        # stages have carved them. Each stage's first micro-submesh is
        # representative of the per-chip footprint (in paired mode every
        # chip in the stage holds a different layer's weights, so any one
        # is a reasonable proxy for "is the per-chip placement happening").
        for i, stage in enumerate((pipe.stage_0, pipe.stage_1, pipe.stage_2)):
            micro = getattr(stage, "micro_submeshes", None)
            if micro:
                l1_targets[i] = micro[0]
                print(f"[probe] stage={STAGE_NAMES[i]} → reading L1 from " f"micro_submeshes[0] (1 of {len(micro)})")

        if RUN_FORWARD:
            try:
                with _phase("after warmup forward", submeshes, results, l1_targets):
                    actions, _t = pipe.run_inference(pixel_values, lang_tokens, noisy_actions)
                    ttnn.deallocate(actions)
            except Exception as e:
                print(f"\n[FAIL] run_inference() raised: {type(e).__name__}: {e}")
                print("[FAIL] L1 snapshot above shows state at fault.")
                raise

    # Final summary table — most useful artifact for follow-up work.
    print("\n" + "=" * 78)
    print("== Option C L1 footprint — full depth (vlm=18, expert=18), real ckpt ==")
    print("=" * 78)
    l1_cap = results["baseline (pre-init)"][0]["l1_cap_mb"]
    dram_cap = results["baseline (pre-init)"][0]["dram_cap_mb"]
    print(f"L1   cap per chip: {l1_cap:.1f} MB")
    print(f"DRAM cap per chip: {dram_cap:.1f} MB")
    print()
    header = (
        f"{'phase':<32}  {'stage':<8}  {'chips':>5}  "
        f"{'L1 used':>9}  {'L1 free':>9}  {'DRAM used':>10}  {'DRAM free':>10}  (MB/chip)"
    )
    print(header)
    print("-" * len(header))
    for phase_name, snaps in results.items():
        for s in snaps:
            print(
                f"{phase_name:<32}  {s['stage']:<8}  {s['n_chips']:>5d}  "
                f"{s['l1_used_mb']:>9.1f}  {s['l1_free_mb']:>9.1f}  "
                f"{s['dram_used_mb']:>10.1f}  {s['dram_free_mb']:>10.1f}"
            )

    # Sanity: weights got uploaded somewhere — either L1 or DRAM.
    # If both are zero post-init, the MemoryView call is silently empty.
    init_snap = results.get("after Pi0_5PipelineC.initialize", [])
    if init_snap:
        total_post_init = sum(s["l1_used_mb"] + s["dram_used_mb"] for s in init_snap)
        assert total_post_init > 0, "post-init L1 + DRAM usage is zero on every chip — MemoryView is silently empty"
        # Diagnostic: print where the weights ended up so the user knows
        # what to do next.
        print()
        print("== Conclusion ==")
        for s in init_snap:
            placement = (
                "L1-resident"
                if s["l1_used_mb"] > s["dram_used_mb"]
                else "DRAM-resident"
                if s["dram_used_mb"] > 0
                else "unused (host-resident or never touched)"
            )
            print(
                f"  stage={s['stage']:<8}  weights are {placement}  "
                f"(L1 {s['l1_used_mb']:.1f} MB, DRAM {s['dram_used_mb']:.1f} MB)"
            )


# ----------------------------------------------------------------------------#
# Optional depth sweep — bisect the OOM cliff                                 #
# ----------------------------------------------------------------------------#


@pytest.mark.skipif(not DEPTH_SWEEP, reason="set PI0_OC_L1_PROBE_DEPTH_SWEEP=1 to run sweep")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("vlm_depth,expert_depth", [(2, 1), (4, 2), (8, 4), (12, 8), (18, 18)])
def test_oc_l1_depth_sweep(vlm_depth: int, expert_depth: int):
    """Try a few (vlm_depth, expert_depth) pairs and report post-init L1.

    Lets us bisect *where* the cliff sits if full depth OOMs. Each pair
    re-opens the mesh fresh so a prior OOM doesn't poison state.
    """
    from models.experimental.pi0_5.tt.option_c.stages import build_shrunk_layout

    _echo_env()
    cfg = PaliGemmaConfig()
    layout = build_shrunk_layout(vlm_depth=vlm_depth, expert_depth=expert_depth)
    print(f"\n[sweep] vlm_depth={vlm_depth}  expert_depth={expert_depth}")
    weights = _load_real_weights()

    results: Dict[str, List[Dict]] = {}
    with open_galaxy_mesh(layout) as (_parent, submeshes):
        with _phase(f"baseline d={vlm_depth}/{expert_depth}", submeshes, results):
            pass
        pipe = Pi0_5PipelineC(
            layout=layout,
            submeshes=submeshes,
            config=cfg,
            weights=weights,
            denoise_steps=NUM_DENOISE_STEPS,
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
        )
        with _phase(f"post-init d={vlm_depth}/{expert_depth}", submeshes, results):
            pipe.initialize()
