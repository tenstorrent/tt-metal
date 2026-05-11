# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0 per-stage breakdown WITH trace.

Captures one trace per logical stage of `sample_actions` and measures replay time.
This gives the same breakdown as `test_stage_breakdown.py` but for the
production-realistic traced execution.

Stages:
  - trace_embed_prefix    : embed_prefix
  - trace_forward_vlm     : forward_vlm (Gemma 2B prefill)
  - trace_denoise_step    : ONE iteration of the denoise loop
                            (timing × 10 gives the loop cost)

Run:
    PI0_CHECKPOINT=pi0_aloha_sim pytest \\
        models/experimental/pi0/tests/perf/test_stage_breakdown_traced.py \\
        --device-id=31 -v -s
"""

import os
import sys
import time
from pathlib import Path
from typing import List

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


def _capture(device, fn, *args, **kwargs):
    """Capture a trace of `fn(*args, **kwargs)` and return (trace_id, outputs)."""
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn(*args, **kwargs)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    return trace_id, out


def _time_trace(device, trace_id, n_iters: int) -> List[float]:
    """Replay a trace n_iters times, returning wall-clock ms per replay."""
    times = []
    for _ in range(n_iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 80_000_000, "num_command_queues": 2}],
    indirect=True,
)
def test_pi0_stage_breakdown_traced(device):
    """Capture one trace per stage, time replays, print breakdown table."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    config = create_config()
    inputs = create_inputs(config, device, batch_size=1)
    weight_loader = PI0WeightLoader(str(checkpoint_path))
    model = PI0ModelTTNN(config, weight_loader, device)

    # ── compile pass: run sample_actions once so kernels exist ──
    with torch.no_grad():
        _ = model.sample_actions(
            images=inputs["images"],
            img_masks=inputs["img_masks"],
            lang_tokens=inputs["lang_tokens"],
            lang_masks=inputs["lang_masks"],
            state=inputs["state"],
        )
    ttnn.synchronize_device(device)

    # Need a fresh prefix_kv_cache that lives across stage traces — produce it once
    # (NOT inside any trace) so the addresses are stable for the denoise trace.
    prefix_embs0, _, _ = model.embed_prefix(
        inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"]
    )
    _, kv_cache = model.backbone.forward_vlm(prefix_embs0, use_cache=True)
    state_emb = model.suffix_embedding.embed_state(inputs["state"])
    ttnn.synchronize_device(device)

    n_iters = int(os.environ.get("PI0_STAGE_ITERS", "20"))

    # ── Trace A: embed_prefix ──
    trace_a, _ = _capture(
        device,
        lambda: model.embed_prefix(inputs["images"], inputs["img_masks"], inputs["lang_tokens"], inputs["lang_masks"]),
    )
    times_a = _time_trace(device, trace_a, n_iters)
    ttnn.release_trace(device, trace_a)

    # ── Trace B: forward_vlm (uses precomputed prefix_embs0) ──
    trace_b, _ = _capture(device, lambda: model.backbone.forward_vlm(prefix_embs0, use_cache=True))
    times_b = _time_trace(device, trace_b, n_iters)
    ttnn.release_trace(device, trace_b)

    # ── Trace C: one denoise step (uses kv_cache, state_emb, time_expanded[0]) ──
    x_t = model.x_t_ttnn
    action_slice_end = (1, 1 + model.config.action_horizon, model.config.expert_config.width)

    def one_denoise_step():
        suffix_embs, _ = model.suffix_embedding.embed_suffix_fast_bs1(
            state_emb, model._time_expanded_per_step_bs1[0], x_t
        )
        expert_output, _ = model.backbone.forward_expert(suffix_embs, past_key_values=kv_cache)
        action_output = ttnn.slice(expert_output, [0, 1, 0], list(action_slice_end))
        velocity = model.suffix_embedding.project_output(action_output)
        velocity_scaled = ttnn.mul(velocity, 0.1)  # fixed dt for timing
        return ttnn.add(x_t, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

    trace_c, _ = _capture(device, one_denoise_step)
    times_c = _time_trace(device, trace_c, n_iters)
    ttnn.release_trace(device, trace_c)

    # ── Aggregate ──
    def stat(name, times):
        n = len(times)
        mean = sum(times) / n
        mn = min(times)
        mx = max(times)
        return name, mean, mn, mx, n

    rows = [
        stat("trace_embed_prefix", times_a),
        stat("trace_forward_vlm_prefill", times_b),
        stat("trace_denoise_step (×1)", times_c),
    ]

    num_steps = model.denoise_config.num_steps
    denoise_total = rows[2][1] * num_steps
    grand_total = rows[0][1] + rows[1][1] + denoise_total

    print(f"\n{'='*80}\n  PI0 per-stage breakdown (TRACED) — checkpoint={CHECKPOINT_NAME} (n={n_iters})\n{'='*80}")
    print(f"{'stage':<32} {'mean_ms':>10} {'min_ms':>10} {'max_ms':>10} {'% est_total':>12}")
    print("-" * 80)
    for name, mean, mn, mx, _ in rows:
        pct_share = 100 * mean / grand_total if grand_total > 0 else 0
        # for the per-step row, also show what 10× looks like below
        print(f"{name:<32} {mean:>10.2f} {mn:>10.2f} {mx:>10.2f} {pct_share:>11.1f}%")
    print("-" * 80)
    print(
        f"{'denoise_loop (×10 est)':<32} {denoise_total:>10.2f} {'':>10} {'':>10} {100*denoise_total/grand_total:>11.1f}%"
    )
    print(f"{'ESTIMATED TOTAL':<32} {grand_total:>10.2f}")
    print("=" * 80)
    print(f"  Estimated traced fps if stages run back-to-back: {1000.0 / grand_total:.2f}")
    print(
        "\n  NOTE: each trace is captured in isolation, so per-stage replay overhead "
        "(execute_trace dispatch, ~tens of µs) is paid 3 times here vs once in a "
        "single full-model trace. The full-model trace via test_perf_e2e.py is the "
        "ground truth for total latency; this table just decomposes where time goes."
    )
