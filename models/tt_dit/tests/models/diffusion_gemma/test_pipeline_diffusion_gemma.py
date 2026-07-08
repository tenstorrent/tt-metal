# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end + perf test for the DiffusionGemma pipeline vs HF.

Correctness: PCC on a single seeded-canvas decoder forward + top-K token match.
Perf: end-to-end diffusion-loop wall time asserted against per-mesh thresholds.

Requires the 51 GB google/diffusiongemma-26B-A4B-it checkpoint.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from ....pipelines.diffusion_gemma.pipeline_diffusion_gemma import DiffusionGemmaPipeline, DiffusionGemmaPipelineConfig
from ....utils.check import assert_quality
from ....utils.test import line_params, ring_params

# ----- Configuration ---------------------------------------------------------------------

MODEL_ID = os.environ.get("DIFFUSIONGEMMA_MODEL_ID", "google/diffusiongemma-26B-A4B-it")
# Top-K containment check: HF's argmax token must be in TT's top-K at each of the first
# N positions; require MIN_MATCH_RATIO of positions to pass. Tolerates bfp8 argmax flips
# on near-tie tokens ("Tokyo." vs "**Tokyo**.") while still catching qualitative drift.
N_TOKEN_MATCH = 16
TOP_K_MATCH = 3
MIN_MATCH_RATIO = 1.0
# Loose PCC bar: chained bf16 over 30 layers + tanh softcap + 262144-vocab projection lands
# around 0.88 even when argmax is correct. TODO: full_attention layers (5, 11, 17, 23, 29)
# introduce most of the depth-wise drift; tightening requires higher-fidelity SDPA there.
PCC_THRESHOLD = 0.80
PROMPT = "Briefly: what is the capital of Japan?"
MAX_NEW_TOKENS = 32
SEED = 0

# Per-mesh × expert-dtype total-latency thresholds (seconds). Only ``total`` is asserted;
# ``encoder``/``denoising`` are informational. (1, 8, "ring", "bfp8") is calibrated from
# hardware; others are estimates.
_DEFAULT_METRICS = {"encoder": 20.0, "denoising": 120.0, "total": 150.0}
EXPECTED_METRICS = {
    (2, 4, "linear", "bfp8"): _DEFAULT_METRICS,
    (2, 4, "linear", "bf16"): {"encoder": 30.0, "denoising": 180.0, "total": 220.0},
    (4, 8, "linear", "bfp8"): {"encoder": 8.0, "denoising": 60.0, "total": 80.0},
    (4, 8, "linear", "bf16"): {"encoder": 12.0, "denoising": 90.0, "total": 120.0},
    (2, 4, "ring", "bfp8"): {"encoder": 30.0, "denoising": 180.0, "total": 220.0},
    (1, 8, "ring", "bfp8"): {"encoder": 5.0, "denoising": 35.0, "total": 32.0},
    (1, 8, "ring", "bf16"): {"encoder": 8.0, "denoising": 55.0, "total": 40.0},
}


def _expert_dtype_key(dtype: ttnn.DataType) -> str:
    return {ttnn.bfloat8_b: "bfp8", ttnn.bfloat16: "bf16"}[dtype]


# ----- Test ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expert_dtype",
    [
        pytest.param(ttnn.bfloat8_b, id="expert_bfp8"),
        pytest.param(ttnn.bfloat16, id="expert_bf16"),
    ],
)
@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2"),
        pytest.param((4, 8), 0, 2, line_params, ttnn.Topology.Linear, id="bh_galaxy"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k"),
        # WH T3K flat 1x8, tp=8: splits MoE intermediate 8-way; ~4x less per-device DRAM
        # than 2x4 → bf16 fits here. Attention with num_global_kv_heads=2 uses kv_replication=4.
        pytest.param((1, 8), 1, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_1x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_diffusion_gemma(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    expert_dtype: ttnn.DataType,
) -> None:
    """End-to-end + perf: TT pipeline vs HF ``DiffusionGemmaForBlockDiffusion.generate``."""
    from transformers import AutoProcessor
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion as HFForBlockDiffusion,
    )

    mesh_shape = tuple(mesh_device.shape)

    # bf16 experts (~91 GB MoE alone) don't fit on 2x4 meshes; skip unless force-enabled.
    if expert_dtype == ttnn.bfloat16 and mesh_shape == (2, 4):
        if os.environ.get("DIFFUSIONGEMMA_FORCE_BF16", "0") != "1":
            pytest.skip(f"bf16 experts OOM on {mesh_shape}; use bfp8 or DIFFUSIONGEMMA_FORCE_BF16=1")

    torch.manual_seed(SEED)
    benchmark_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    # 1. Build TT pipeline + HF reference model.
    config = DiffusionGemmaPipelineConfig(
        mesh_device=mesh_device,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        max_denoising_steps=8,  # cut for test runtime; full would be 48
        expert_dtype=expert_dtype,
    )
    benchmark_profiler.start("setup")
    pipeline = DiffusionGemmaPipeline.from_pretrained(MODEL_ID, config=config)
    hf_processor = AutoProcessor.from_pretrained(MODEL_ID)
    hf_model = HFForBlockDiffusion.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    benchmark_profiler.end("setup")

    # Enable MoE routing-histogram diagnostic to print per-expert token assignment counts
    for layer in pipeline.tt_model.model.encoder.language_model.layers:
        layer.experts_and_router.log_routing_histogram = True

    # 2. Correctness: one decoder forward at a seeded canvas vs HF.
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    proc_inputs = hf_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    input_ids = proc_inputs["input_ids"]

    text_cfg = hf_model.config.text_config
    canvas_length = hf_model.config.canvas_length
    seed_canvas = torch.randint(0, text_cfg.vocab_size, (1, canvas_length), dtype=torch.long)

    encoder_kv, tt_enc_masks, enc_pos = pipeline._run_encoder(input_ids, None, None)
    decoder_position_ids = torch.arange(
        input_ids.shape[1], input_ids.shape[1] + canvas_length, dtype=torch.long
    ).unsqueeze(0)
    tt_logits = pipeline._run_decoder(
        seed_canvas,
        encoder_kv,
        decoder_position_ids,
        tt_enc_masks,
        encoder_seq_len=input_ids.shape[1],
        self_cond_logits=None,
    )
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, decoder_input_ids=seed_canvas).logits

    logger.info(f"hf_logits {hf_logits.shape}, tt_logits {tt_logits.shape}")

    # 3. Full diffusion loop with per-step decoded text (also consumed by visualize_diffusion.py).
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
        LinearTemperatureScheduleLogitsProcessor,
        StableAndConfidentStoppingCriteria,
    )

    torch.manual_seed(SEED)
    sampler = EntropyBoundSampler(
        EntropyBoundSamplerConfig(entropy_bound=pipeline.config.entropy_bound),
        canvas_length=canvas_length,
        vocab_size=text_cfg.vocab_size,
        max_denoising_steps=pipeline.config.max_denoising_steps,
    )
    logits_processor = LinearTemperatureScheduleLogitsProcessor(
        t_min=pipeline.config.temperature_start,
        t_max=pipeline.config.temperature_end,
        max_denoising_steps=pipeline.config.max_denoising_steps,
    )
    stopping = StableAndConfidentStoppingCriteria(
        stability_threshold=pipeline.config.stability_threshold,
        confidence_threshold=pipeline.config.confidence_threshold,
    )

    current_canvas = sampler.initialize_canvas(batch_size=1, device=torch.device("cpu"))
    self_cond_logits: torch.Tensor | None = None
    argmax_canvas = current_canvas.clone()

    # Warmup the self_cond decoder code path (section 2 already warmed encoder + self_cond=None).
    _ = pipeline._run_decoder(
        current_canvas,
        encoder_kv,
        decoder_position_ids,
        tt_enc_masks,
        encoder_seq_len=input_ids.shape[1],
        self_cond_logits=tt_logits,
    )

    logger.info(f"[3] Diffusion loop ({pipeline.config.max_denoising_steps} steps max)")
    logger.info(f"    Prompt: {PROMPT!r}")

    benchmark_profiler.start("e2e_generate")
    for step in range(pipeline.config.max_denoising_steps):
        logits = pipeline._run_decoder(
            current_canvas,
            encoder_kv,
            decoder_position_ids,
            tt_enc_masks,
            encoder_seq_len=input_ids.shape[1],
            self_cond_logits=self_cond_logits,
        )
        argmax_canvas = logits.argmax(dim=-1)
        processed_logits = logits_processor(current_canvas, logits, cur_step=step)
        probs = torch.softmax(processed_logits.float(), dim=-1)
        denoiser_canvas = torch.distributions.Categorical(probs=probs).sample()
        accepted = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, step)
        current_canvas = sampler.renoise_canvas(accepted, step)
        finished = stopping(argmax_canvas, processed_logits)
        self_cond_logits = processed_logits

        argmax_text = hf_processor.batch_decode(argmax_canvas, skip_special_tokens=False)[0]
        logger.info(f"[step {step:02d}] argmax : {argmax_text!r}")
        if torch.all(finished):
            logger.info(f"[step {step:02d}] stopping criteria satisfied — halting")
            break
    benchmark_profiler.end("e2e_generate")

    final_text = hf_processor.batch_decode(argmax_canvas, skip_special_tokens=False)[0]
    logger.info(f"[3] Final TT decoded canvas: {final_text!r}")

    # 4. Correctness assertions.
    # 4a. Loose PCC smoke test on the section-2 seed-canvas logits.
    logger.info(f"[4a] Loose PCC smoke test (threshold={PCC_THRESHOLD:.2f})")
    assert_quality(hf_logits, tt_logits, pcc=PCC_THRESHOLD)

    # 4b. Top-K token containment: HF argmax must appear in TT's top-K at each of the first
    # N positions; require MIN_MATCH_RATIO of positions to pass. Tolerates bfp8 near-tie flips.
    hf_argmax = hf_logits.argmax(dim=-1)[0, :N_TOKEN_MATCH].tolist()
    tt_topk = torch.topk(tt_logits[0, :N_TOKEN_MATCH], k=TOP_K_MATCH, dim=-1).indices.tolist()
    tt_argmax = [row[0] for row in tt_topk]
    match_count = sum(hf_tok in top_row for hf_tok, top_row in zip(hf_argmax, tt_topk))
    logger.info(f"[4b] HF argmax: {hf_argmax}")
    logger.info(f"[4b] TT top-{TOP_K_MATCH}: {tt_topk}")
    logger.info(f"[4b] {match_count}/{N_TOKEN_MATCH} positions had HF's argmax in TT's top-{TOP_K_MATCH}")
    required = max(1, int(N_TOKEN_MATCH * MIN_MATCH_RATIO + 0.5))
    assert match_count >= required, (
        f"Only {match_count}/{N_TOKEN_MATCH} positions match (required {required}).\n"
        f"  TT top-1: {hf_processor.batch_decode(torch.tensor([tt_argmax]))[0]!r}\n"
        f"  HF top-1: {hf_processor.batch_decode(torch.tensor([hf_argmax]))[0]!r}"
    )

    # 5. Perf: assert e2e diffusion loop time.
    topology_key = "ring" if topology == ttnn.Topology.Ring else "linear"
    metrics_key = (*mesh_shape, topology_key, _expert_dtype_key(expert_dtype))
    expected = EXPECTED_METRICS.get(metrics_key, _DEFAULT_METRICS)

    e2e_time = benchmark_profiler.get_duration("e2e_generate")
    logger.info(f"E2E generate time: {e2e_time:.2f}s vs threshold {expected['total']}s")
    benchmark_data.add_measurement(
        profiler=benchmark_profiler,
        iteration=0,
        step_name="e2e_generate",
        name="diffusiongemma_e2e_seconds",
        value=e2e_time,
    )

    assert (
        e2e_time <= expected["total"]
    ), f"E2E time {e2e_time:.2f}s exceeds threshold {expected['total']}s for {metrics_key}"
    logger.info(f"Perf ✓ ({e2e_time:.2f}s ≤ {expected['total']}s)")
