# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0 per-stage wall-clock breakdown.

Replays the steps of `sample_actions` with `ttnn.synchronize_device` between each
stage so we can see how the e2e time splits across:

    embed_prefix → forward_vlm → (embed_suffix × 10, forward_expert × 10, project/step × 10)

NOTE: trace is *off* here on purpose — trace records the whole thing as one
unit, so sync points inside the trace are not possible. Absolute numbers are
therefore larger than the traced e2e test (`test_perf_e2e.py`), but the
relative split is what we need.

Run:
    PI0_CHECKPOINT=pi0_aloha_sim pytest \\
        models/experimental/pi0/tests/perf/test_stage_breakdown.py \\
        --device-id=31 -v -s
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
if not TT_METAL_HOME:
    raise EnvironmentError("TT_METAL_HOME environment variable is not set")
CHECKPOINT_NAME = os.environ.get("PI0_CHECKPOINT", "pi0_base")
CHECKPOINT_PATH = os.path.join(TT_METAL_HOME, "models/experimental/pi0/weights", CHECKPOINT_NAME)


def create_config() -> PI0ModelConfig:
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=False,
    )
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    return config


def create_inputs(config: PI0ModelConfig, device, batch_size: int = 1):
    image_size = config.siglip_config.image_size
    images_torch = [torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32) for _ in range(2)]
    img_masks_torch = [torch.ones(batch_size, dtype=torch.bool) for _ in range(2)]
    lang_tokens_torch = torch.randint(0, 256000, (batch_size, 32))
    lang_masks_torch = torch.ones(batch_size, 32, dtype=torch.bool)
    state_torch = torch.randn(batch_size, config.state_dim, dtype=torch.float32)

    images = [
        ttnn.from_torch(
            img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for img in images_torch
    ]
    img_masks = [
        ttnn.from_torch(
            m.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for m in img_masks_torch
    ]
    lang_tokens = ttnn.from_torch(lang_tokens_torch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    lang_masks = ttnn.from_torch(lang_masks_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    state = ttnn.from_torch(state_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
    }


def _sync_time(device) -> float:
    """Return wall clock after a device sync — so the previous block's
    enqueued work is genuinely finished before we read the clock."""
    ttnn.synchronize_device(device)
    return time.perf_counter()


def run_one_iter_instrumented(model: PI0ModelTTNN, inputs, device) -> Dict[str, float]:
    """Replays sample_actions but with sync points between stages.
    Returns a dict of ms per stage for this single iter."""
    stage = {}

    # ── stage 1: embed prefix ────────────────────────────────
    t0 = _sync_time(device)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"]
    )
    t1 = _sync_time(device)
    stage["embed_prefix"] = (t1 - t0) * 1000

    # ── stage 2: vlm prefill ─────────────────────────────────
    _, prefix_kv_cache = model.backbone.forward_vlm(prefix_embs, use_cache=True)
    t2 = _sync_time(device)
    stage["forward_vlm_prefill"] = (t2 - t1) * 1000

    # ── stage 3: denoise loop, broken into sub-stages ────────
    num_steps = model.denoise_config.num_steps
    state_emb_cached = model.suffix_embedding.embed_state(inputs["state"])
    ttnn.synchronize_device(device)

    suffix_total_ms = 0.0
    expert_total_ms = 0.0
    project_total_ms = 0.0
    x_t = model.x_t_ttnn
    action_slice_end = (1, 1 + model.config.action_horizon, model.config.expert_config.width)
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

    for i in range(num_steps):
        dt = timesteps[i + 1] - timesteps[i]
        ts_a = _sync_time(device)
        suffix_embs, suffix_att = model.suffix_embedding.embed_suffix_fast_bs1(
            state_emb_cached, model._time_expanded_per_step_bs1[i], x_t
        )
        ts_b = _sync_time(device)
        suffix_total_ms += (ts_b - ts_a) * 1000

        expert_output, _ = model.backbone.forward_expert(suffix_embs, past_key_values=prefix_kv_cache)
        ts_c = _sync_time(device)
        expert_total_ms += (ts_c - ts_b) * 1000

        action_output = ttnn.slice(expert_output, [0, 1, 0], list(action_slice_end))
        velocity = model.suffix_embedding.project_output(action_output)
        velocity_scaled = ttnn.mul(velocity, dt)
        x_t = ttnn.add(x_t, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
        ts_d = _sync_time(device)
        project_total_ms += (ts_d - ts_c) * 1000

    stage["denoise_embed_suffix_10x"] = suffix_total_ms
    stage["denoise_forward_expert_10x"] = expert_total_ms
    stage["denoise_project_step_10x"] = project_total_ms

    # ── stage 4: device → host copy of final actions ─────────
    t_h0 = _sync_time(device)
    _ = ttnn.to_torch(x_t)
    t_h1 = _sync_time(device)
    stage["output_d2h"] = (t_h1 - t_h0) * 1000

    stage["TOTAL"] = sum(v for k, v in stage.items() if k != "TOTAL")
    return stage


def _print_breakdown(rows: List[Dict[str, float]], label: str):
    keys = [
        "embed_prefix",
        "forward_vlm_prefill",
        "denoise_embed_suffix_10x",
        "denoise_forward_expert_10x",
        "denoise_project_step_10x",
        "output_d2h",
        "TOTAL",
    ]
    n = len(rows)
    means = {k: sum(r[k] for r in rows) / n for k in keys}
    mins = {k: min(r[k] for r in rows) for k in keys}
    maxs = {k: max(r[k] for r in rows) for k in keys}
    total = means["TOTAL"]

    print(f"\n{'='*80}\n  PI0 per-stage breakdown — {label} (n={n})\n{'='*80}")
    print(f"{'stage':<32} {'mean_ms':>10} {'min_ms':>10} {'max_ms':>10} {'%total':>8}")
    print("-" * 80)
    for k in keys:
        pct = 100 * means[k] / total if k != "TOTAL" and total > 0 else 100.0
        print(f"{k:<32} {means[k]:>10.2f} {mins[k]:>10.2f} {maxs[k]:>10.2f} {pct:>7.1f}%")
    print("=" * 80)
    print(f"  Inference rate at this measurement: {1000.0 / total:.2f} fps")
    print(f"  Per-step expert avg: {means['denoise_forward_expert_10x']/10:.2f} ms")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pi0_stage_breakdown(device):
    """Per-stage timing of pi0 sample_actions on a single device."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    config = create_config()
    inputs = create_inputs(config, device, batch_size=1)
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    model = PI0ModelTTNN(config, weight_loader, device)

    # warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model.sample_actions(
                images=inputs["images"],
                img_masks=inputs["img_masks"],
                lang_tokens=inputs["lang_tokens"],
                lang_masks=inputs["lang_masks"],
                state=inputs["state"],
            )
    ttnn.synchronize_device(device)

    # measure
    n_iters = int(os.environ.get("PI0_STAGE_ITERS", "10"))
    rows = []
    for _ in range(n_iters):
        with torch.no_grad():
            rows.append(run_one_iter_instrumented(model, inputs, device))

    _print_breakdown(rows, f"checkpoint={CHECKPOINT_NAME}")
