# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 TTNN end-to-end denoise-loop performance with trace (+ optional 2CQ).

Builds on test_perf_ttnn_trace.py by:
  1) Hoisting deterministic work — per-(step, layer) adaRMS modulations are
     precomputed at model init (Pi0_5ModelTTNN._precompute_bs1_modulations),
     so the per-step path is only `action_in_proj → forward_expert →
     action_out_proj → Euler update`. The time-MLP + per-block mod-Dense chain
     runs on host once at init, not 10× per chunk.
  2) Capturing the FULL 10-step denoise loop as a single trace, so we measure
     a real chunk latency (not a 1-step × 10 estimate).
  3) An optional 2CQ variant: CQ 1 uploads the next chunk's initial noise to
     a pre-allocated device buffer while CQ 0 replays the trace for the
     current chunk.

Also includes `test_pi0_5_ttnn_perf_prefill_stages` — times the real prefix
path (SigLIP, language embed, VLM prefill, KV-cache lift) that runs once per
`sample_actions` call before the denoise loop.
"""

import statistics
import time
from pathlib import Path
from typing import Callable, List, Tuple

import pytest
import torch
import ttnn

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"

NUM_WARMUP = 2
NUM_ITERS = 20
PREFIX_LEN = 32
LANG_SEQ_LEN = 256
SEED = 0
TRACE_REGION_SIZE = 80_000_000

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_e2e_inputs(device, batch_size: int = 1):
    """Realistic prefix inputs — same layout as the full e2e perf tests."""
    torch.manual_seed(SEED)
    image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    img_mask = torch.ones(batch_size, dtype=torch.bool)
    lang_tokens = torch.randint(0, 256000, (batch_size, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(batch_size, LANG_SEQ_LEN, dtype=torch.bool)

    image_ttnn = ttnn.from_torch(
        image,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    img_mask_ttnn = ttnn.from_torch(
        img_mask.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    lang_tokens_ttnn = ttnn.from_torch(
        lang_tokens.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    lang_masks_ttnn = ttnn.from_torch(
        lang_masks.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return image_ttnn, img_mask_ttnn, lang_tokens_ttnn, lang_masks_ttnn


def _time_stage(
    device, fn: Callable[[], None], warmup: int = NUM_WARMUP, iters: int = NUM_ITERS
) -> Tuple[float, float, float, float]:
    """Warm up `fn`, then return (avg, min, max, stddev) in ms."""
    for _ in range(warmup):
        fn()
        ttnn.synchronize_device(device)

    times_ms: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return avg, mn, mx, sd


def _print_stage_row(label: str, avg: float, mn: float, mx: float, sd: float, width: int = 22) -> None:
    print(f"   {label:<{width}} {avg:7.2f} ms  (min {mn:.2f}, max {mx:.2f}, std {sd:.2f})")


def _run_prefill_like_sample_actions(model, image_ttnn, img_mask_ttnn, lang_tokens_ttnn, lang_masks_ttnn):
    """Mirror Pi0_5ModelTTNN.sample_actions prefix path (through KV-cache lift)."""
    prefix_embs, _, _ = model.embed_prefix(
        images=[image_ttnn],
        img_masks=[img_mask_ttnn],
        lang_tokens=lang_tokens_ttnn,
        lang_masks=lang_masks_ttnn,
    )
    if prefix_embs.layout != ttnn.TILE_LAYOUT:
        prefix_embs = ttnn.to_layout(prefix_embs, ttnn.TILE_LAYOUT)
    _, prefix_kv_cache = model.backbone.forward_vlm(prefix_embs, use_cache=True)
    prefix_logical_len = prefix_embs.shape[1]
    prefix_kv_cache = [
        (
            ttnn.fill_implicit_tile_padding(k, 0.0),
            ttnn.fill_implicit_tile_padding(v, 0.0),
        )
        for k, v in prefix_kv_cache
    ]
    model._sdpa_attn_mask = model._build_sdpa_phantom_mask(prefix_logical_len)
    model._sdpa_mask_kv_len = prefix_kv_cache[0][0].shape[2]
    return prefix_kv_cache


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_perf_prefill_stages(device):
    """
    Time the prefix prefill stages that run once per sample_actions call.

    Breakdown:
      - siglip:           SigLIP vision encoder (image -> patch embeddings)
      - lang_embed:       Gemma token embedding + sqrt(d) scaling
      - embed_prefix:     siglip + lang + concat (full prefix embedding)
      - vlm_forward:      Gemma-2B VLM forward with KV-cache (18 layers), given
                          fixed prefix embeddings on device
      - prefill_total:    full production path through KV-cache lift + SDPA mask
    """
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded")

    image_ttnn, img_mask_ttnn, lang_tokens_ttnn, lang_masks_ttnn = _build_e2e_inputs(device)
    num_patches = cfg.siglip_config.num_patches
    prefix_tokens = num_patches + LANG_SEQ_LEN
    print(f"   Prefix tokens:       {prefix_tokens} ({num_patches} image + {LANG_SEQ_LEN} lang)")

    def _siglip():
        out = model.backbone.embed_image(image_ttnn)
        ttnn.deallocate(out)

    def _lang_embed():
        out = model.prefix_embedding.embed_language(lang_tokens_ttnn, lang_masks_ttnn)
        ttnn.deallocate(out)

    def _embed_prefix():
        out, pad, att = model.embed_prefix(
            images=[image_ttnn],
            img_masks=[img_mask_ttnn],
            lang_tokens=lang_tokens_ttnn,
            lang_masks=lang_masks_ttnn,
        )
        ttnn.deallocate(out)
        ttnn.deallocate(pad)
        ttnn.deallocate(att)

    def _prefill_total():
        prefix_kv_cache = _run_prefill_like_sample_actions(
            model, image_ttnn, img_mask_ttnn, lang_tokens_ttnn, lang_masks_ttnn
        )
        for k, v in prefix_kv_cache:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        if model._sdpa_attn_mask is not None:
            ttnn.deallocate(model._sdpa_attn_mask)
            model._sdpa_attn_mask = None
            model._sdpa_mask_kv_len = 0

    stages = [
        ("siglip", _siglip),
        ("lang_embed", _lang_embed),
        ("embed_prefix", _embed_prefix),
        ("prefill_total", _prefill_total),
    ]

    print(f"\n⏱️  Measuring prefill stages ({NUM_ITERS} iters, {NUM_WARMUP} warmup each)")
    results = {}
    for name, fn in stages:
        print(f"   … {name}")
        results[name] = _time_stage(device, fn)

    # VLM forward only — prefix embeddings are fixed on device so this isolates
    # the Gemma-2B KV-cache prefill from SigLIP/lang embedding cost.
    print("   … vlm_forward")
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        images=[image_ttnn],
        img_masks=[img_mask_ttnn],
        lang_tokens=lang_tokens_ttnn,
        lang_masks=lang_masks_ttnn,
    )
    ttnn.deallocate(prefix_pad)
    ttnn.deallocate(prefix_att)
    if prefix_embs.layout != ttnn.TILE_LAYOUT:
        prefix_embs = ttnn.to_layout(prefix_embs, ttnn.TILE_LAYOUT)

    def _vlm_forward():
        _, prefix_kv_cache = model.backbone.forward_vlm(prefix_embs, use_cache=True)
        for k, v in prefix_kv_cache:
            ttnn.deallocate(k)
            ttnn.deallocate(v)

    results["vlm_forward"] = _time_stage(device, _vlm_forward)
    ttnn.deallocate(prefix_embs)

    print("\n" + "=" * 72)
    print("  PI0.5 TTNN PREFILL STAGE BREAKDOWN (real pi05_base weights)")
    print("=" * 72)
    print(f"   Image patches/view:  {num_patches}")
    print(f"   Language tokens:     {LANG_SEQ_LEN}")
    print(f"   Total prefix tokens: {prefix_tokens}")
    print(f"   Iterations/stage:    {NUM_ITERS}")
    print("-" * 72)
    for name, _ in stages:
        avg, mn, mx, sd = results[name]
        _print_stage_row(name, avg, mn, mx, sd)
    avg, mn, mx, sd = results["vlm_forward"]
    _print_stage_row("vlm_forward", avg, mn, mx, sd)
    print("=" * 72)
    assert results["prefill_total"][0] > 0


def _build_prefix_kv(model, device, batch_size: int = 1):
    """Synthetic prefix KV cache for the bs=1 keep_padded expert path."""
    ec = model.config.expert_config
    torch.manual_seed(SEED)
    prefix_kv = []
    for _ in range(ec.depth):
        k = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        v = torch.randn(batch_size, ec.num_kv_heads, PREFIX_LEN, ec.head_dim) * 0.1
        k_ttnn = ttnn.from_torch(k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        v_ttnn = ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        prefix_kv.append(
            (
                ttnn.fill_implicit_tile_padding(k_ttnn, 0.0),
                ttnn.fill_implicit_tile_padding(v_ttnn, 0.0),
            )
        )
    return prefix_kv


def _ensure_sdpa_mask(model, prefix_kv):
    """Build the phantom SDPA mask once for the synthetic prefix length."""
    kv_len = prefix_kv[0][0].shape[2]
    if model._sdpa_attn_mask is None or model._sdpa_mask_kv_len != kv_len:
        model._sdpa_attn_mask = model._build_sdpa_phantom_mask(PREFIX_LEN)
        model._sdpa_mask_kv_len = kv_len


def _run_denoise_loop(model, x_t, prefix_kv):
    """Inline 10-step denoise loop using the production bs=1 fast path."""
    cfg = model.config
    num_steps = cfg.num_denoising_steps
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

    for i in range(num_steps):
        dt = timesteps[i + 1] - timesteps[i]
        suffix_embs = model.suffix_embedding.embed_actions(x_t)
        expert_out, _ = model.backbone.forward_expert(
            suffix_embs,
            adarms_cond=None,
            past_key_values=prefix_kv,
            precomputed_block_mods=model._block_mods_per_step[i],
            precomputed_final_mod=model._final_mod_per_step[i],
            attention_mask=model._sdpa_attn_mask,
            keep_padded=True,
        )
        velocity = model.suffix_embedding.project_output(expert_out)
        v_scaled = ttnn.mul(velocity, dt)
        x_t = ttnn.add(x_t, v_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
    return x_t


def _print_summary(label, capture_ms, times_ms, cfg):
    avg = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    chunks_per_sec = 1000.0 / avg if avg > 0 else 0.0
    actions_per_sec = chunks_per_sec * cfg.action_horizon
    per_step_avg = avg / cfg.num_denoising_steps

    print("\n" + "=" * 72)
    print(f"  PI0.5 TTNN PERFORMANCE — {label}")
    print("=" * 72)
    print(f"   Denoise steps/chunk: {cfg.num_denoising_steps}")
    print(f"   Action horizon:      {cfg.action_horizon}")
    print(f"   Trace capture:       {capture_ms:7.2f} ms (one-time)")
    print(f"   Iterations:          {len(times_ms)} replays")
    print("-" * 72)
    print(f"   Chunk avg:           {avg:7.2f} ms  ({per_step_avg:.2f} ms/step)")
    print(f"   Chunk min:           {mn:7.2f} ms")
    print(f"   Chunk max:           {mx:7.2f} ms")
    print(f"   Chunk stddev:        {sd:7.2f} ms")
    print("-" * 72)
    print(f"   Chunk throughput:    {chunks_per_sec:7.2f} chunks/s")
    print(f"   Action throughput:   {actions_per_sec:7.2f} actions/s")
    print("=" * 72)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_ttnn_perf_trace_e2e(device):
    """Full 10-step denoise loop captured as a single trace, single CQ."""
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded; {len(model._block_mods_per_step)} step modulations precomputed")

    prefix_kv = _build_prefix_kv(model, device)
    _ensure_sdpa_mask(model, prefix_kv)

    # Initial x_t (noise) lives at a fixed device buffer so trace can reference it.
    x_t = model.x_t_ttnn

    print(f"\n🔥 Warmup ({NUM_WARMUP} chunks) — JIT compile")
    for i in range(NUM_WARMUP):
        out = _run_denoise_loop(model, x_t, prefix_kv)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        print(f"   warmup chunk {i + 1} done")

    print(f"\n📷 Capturing trace of full 10-step denoise loop…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _run_denoise_loop(model, x_t, prefix_kv)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    print(f"\n⏱️  Measuring traced chunk replay ({NUM_ITERS} chunks)")
    times_ms: List[float] = []
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   chunk {i + 1:2d}: {elapsed_ms:7.2f} ms")

    ttnn.release_trace(device, tid)
    _print_summary("trace, full 10-step denoise loop (1 CQ)", capture_ms, times_ms, cfg)
    assert statistics.mean(times_ms) > 0


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "trace_region_size": TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
def test_pi0_5_ttnn_perf_trace_2cq(device):
    """
    Full 10-step denoise loop with trace, with CQ 1 uploading the next
    chunk's initial noise while CQ 0 replays the trace for the current chunk.

    Each chunk in steady state issues:
      - CQ 1: copy_host_to_device_tensor(host_noise[i+1] -> x_t buffer)
      - CQ 0: execute_trace (10 denoise steps)
    The two operations overlap when chunk compute > host upload (which is
    always true here — upload is ~3KB, compute is ~120ms).
    """
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig()
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded")

    prefix_kv = _build_prefix_kv(model, device)
    _ensure_sdpa_mask(model, prefix_kv)
    x_t = model.x_t_ttnn

    # Pre-allocate host noise tensors for NUM_ITERS chunks.
    torch.manual_seed(SEED)
    host_noise = [torch.randn(1, cfg.action_horizon, cfg.action_dim, dtype=torch.float32) for _ in range(NUM_ITERS + 1)]
    # Move each to a host TTNN tensor so copy_host_to_device_tensor can target x_t.
    host_noise_ttnn = [ttnn.from_torch(n, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for n in host_noise]

    print(f"\n🔥 Warmup ({NUM_WARMUP} chunks)")
    for i in range(NUM_WARMUP):
        out = _run_denoise_loop(model, x_t, prefix_kv)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    print(f"\n📷 Capturing trace…")
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _run_denoise_loop(model, x_t, prefix_kv)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0
    print(f"   trace captured in {capture_ms:.2f} ms")

    print(f"\n⏱️  Measuring 2CQ + trace ({NUM_ITERS} chunks)")
    times_ms: List[float] = []
    # Pre-stage chunk 0 noise on CQ 1.
    ttnn.copy_host_to_device_tensor(host_noise_ttnn[0], x_t, cq_id=1)
    write_event = ttnn.record_event(device, 1)

    for i in range(NUM_ITERS):
        start = time.perf_counter()
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)

        # Pre-stage next chunk's noise on CQ 1 in parallel with the trace.
        if i + 1 < NUM_ITERS:
            ttnn.wait_for_event(1, op_event)
            ttnn.copy_host_to_device_tensor(host_noise_ttnn[i + 1], x_t, cq_id=1)
            write_event = ttnn.record_event(device, 1)

        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        print(f"   chunk {i + 1:2d}: {elapsed_ms:7.2f} ms")

    ttnn.release_trace(device, tid)
    _print_summary("trace + 2CQ, full 10-step denoise loop", capture_ms, times_ms, cfg)
    assert statistics.mean(times_ms) > 0
