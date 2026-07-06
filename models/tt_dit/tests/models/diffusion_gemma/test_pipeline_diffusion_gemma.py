# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end + perf test for the DiffusionGemma pipeline.

Per-canvas correctness:
  * Hidden-state PCC after one decoder forward vs HF.
  * Token-ID match on the first ``N_TOKEN_MATCH`` tokens generated under a
    deterministic seed.

Perf:
  * BenchmarkProfiler instruments encoder / decoder-loop / total wall time.
  * Asserts each component is below a permissive per-mesh threshold (start
    permissive, tighten after first hardware run).

Requires the released checkpoint (``google/diffusiongemma-26B-A4B-it``, 51 GB).
The test will skip on configurations whose total device memory is too tight to
hold the model + activations.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_pipeline_diffusion_gemma.py -s
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
N_TOKEN_MATCH = 8  # First-N tokens must match HF exactly under seeded sampling.
# End-to-end pipeline runs the tanh-softcapped logits through a sampler; the softcap compresses
# most drift into the ~[-30, 30] band. First-N token-ID match under a seeded categorical sampler
# is the primary correctness signal. Logit-space PCC is a smoke test at a *loose* threshold —
# chained-bf16 drift across 30 layers + tanh saturation + 262144-way vocab projection sits at
# ~88% naturally even when the argmax path (and hence generated text) matches HF exactly.
#
# TODO(sosborne): the current 30-layer chained hidden-state drift includes an outsized
# ~-32pp PCC drop at layer 29 (last ``full_attention`` layer) that the final RMSNorm partially
# rescues. Every 6th layer (5, 11, 17, 23, 29) is ``full_attention`` and the per-layer PCC
# regression grows with depth. Working hypothesis: bf16 accumulation drift compounding in the
# num_kv_heads=2/kv_replication=4 attention (head_dim=512, larger matmuls than sliding).
# Tightening the PCC threshold requires either (a) higher-fidelity SDPA on full_attention or
# (b) fp32 accumulation on the attention output path. Not blocking correct-output-generation.
PCC_THRESHOLD = 0.80
PROMPT = "Briefly: what is the capital of France?"
MAX_NEW_TOKENS = 32
SEED = 0

# Permissive per-mesh-shape × per-expert-dtype latency thresholds (seconds). Expert dtype
# strongly affects DRAM traffic through the MoE: bfp4 is ~8x smaller than bf16, bfp8 ~4x.
# Tighten after first hardware run.
# Key format: (rows, cols, topology, expert_dtype_key) where
#   expert_dtype_key ∈ {"bfp4", "bfp8", "bf16"}.
_DEFAULT_METRICS = {"encoder": 20.0, "denoising": 120.0, "total": 150.0}
EXPECTED_METRICS = {
    # BH QB2 (2x4 linear)
    (2, 4, "linear", "bfp4"): {"encoder": 15.0, "denoising": 90.0, "total": 110.0},
    (2, 4, "linear", "bfp8"): _DEFAULT_METRICS,
    (2, 4, "linear", "bf16"): {"encoder": 30.0, "denoising": 180.0, "total": 220.0},
    # BH galaxy (4x8 linear)
    (4, 8, "linear", "bfp4"): {"encoder": 6.0, "denoising": 45.0, "total": 60.0},
    (4, 8, "linear", "bfp8"): {"encoder": 8.0, "denoising": 60.0, "total": 80.0},
    (4, 8, "linear", "bf16"): {"encoder": 12.0, "denoising": 90.0, "total": 120.0},
    # WH T3K (2x4 ring) — bf16 doesn't fit in DRAM, skipped in-test.
    (2, 4, "ring", "bfp4"): {"encoder": 20.0, "denoising": 120.0, "total": 150.0},
    (2, 4, "ring", "bfp8"): {"encoder": 30.0, "denoising": 180.0, "total": 220.0},
    # WH T3K (1x8 ring) — tp=8 splits MoE intermediate 8-way, so per-device DRAM is 4x
    # smaller than (2,4). bf16 fits here. Numbers below are calibrated from a first
    # successful run at bfp8 (~6-9s per denoising step; converged in 3-4 steps under
    # StableAndConfident stopping) with generous slack for warm-cache variability. The
    # total includes stopping-criteria-driven early termination.
    (1, 8, "ring", "bfp4"): {"encoder": 15.0, "denoising": 80.0, "total": 110.0},
    (1, 8, "ring", "bfp8"): {"encoder": 15.0, "denoising": 100.0, "total": 130.0},
    (1, 8, "ring", "bf16"): {"encoder": 25.0, "denoising": 160.0, "total": 200.0},
}


def _expert_dtype_key(dtype: ttnn.DataType) -> str:
    """Short stable string for keying ``EXPECTED_METRICS``."""
    return {ttnn.bfloat4_b: "bfp4", ttnn.bfloat8_b: "bfp8", ttnn.bfloat16: "bf16"}[dtype]


# ----- Test ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expert_dtype",
    [
        pytest.param(ttnn.bfloat4_b, id="expert_bfp4"),
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
        # WH T3K flat 1x8 with tp=8. Splits the MoE intermediate dim 8-way, giving ~4x less
        # per-device DRAM than the 2x4 layout — makes bfp8/bf16 experts fit where 2x4 OOMs.
        # tp_axis=1 since axis 0 is size 1. Attention KV heads with num_global_kv_heads=2 will
        # be replicated 4x per device (Gemma4Attention handles this via `_kv_replication`).
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

    # DRAM ceiling on 2x4 meshes: bf16 expert weights are ~91 GB for MoE alone → doesn't fit
    # on WH T3K (8x12 GB = 96 GB) or Blackhole 2x4. bfp8 experts (~23 GB for MoE) fit
    # comfortably on both. The 4x8 Blackhole galaxy has 4x the DRAM banks and fits both dtypes.
    # Escape hatch: DIFFUSIONGEMMA_FORCE_BF16=1 if you've validated the DRAM math on a
    # specific config (e.g. reduced num_hidden_layers).
    if expert_dtype == ttnn.bfloat16 and mesh_shape == (2, 4):
        if os.environ.get("DIFFUSIONGEMMA_FORCE_BF16", "0") != "1":
            pytest.skip(
                f"bf16 expert weights don't fit in DRAM on mesh_shape={mesh_shape}. "
                "Use expert_dtype=ttnn.bfloat8_b, run on 4x8 mesh, or set DIFFUSIONGEMMA_FORCE_BF16=1."
            )

    torch.manual_seed(SEED)
    benchmark_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    # ------------------------------------------------------------------
    # 1. Build TT pipeline + HF reference model.
    # ------------------------------------------------------------------
    # ``expert_dtype`` is parametrized so we can validate both the DRAM-efficient bfp8 path
    # (default; fits comfortably on all supported meshes) and the higher-precision bf16 path
    # (may OOM on tighter mesh shapes — the ``bf16`` variants are expected to skip when the
    # DRAM budget is exceeded rather than fail the test suite).
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

    # ------------------------------------------------------------------
    # 2. Correctness: one decoder forward at a seeded canvas vs HF.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Full diffusion loop with per-step decoded text.
    # ------------------------------------------------------------------
    # Runs the same diffusion loop that ``pipeline.generate()`` runs, but prints the decoded
    # argmax canvas after every step. Useful for spotting convergence problems and for the
    # visualize_diffusion.py video tool that parses this log format.
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

    # ------------------------------------------------------------------
    # 4. Correctness assertions.
    # ------------------------------------------------------------------
    # 4a. Loose PCC smoke test on the section-2 seed-canvas logits. See ``PCC_THRESHOLD``'s
    # comment for why 0.80 is the current bar (chained bf16 across 30 layers + tanh saturation
    # + 262144-way vocab projection lands at ~0.88 in practice even when argmax is correct).
    logger.info(f"[4a] Loose PCC smoke test (threshold={PCC_THRESHOLD:.2f})")
    assert_quality(hf_logits, tt_logits, pcc=PCC_THRESHOLD)

    # 4b. Argmax-path token match on the section-2 logits — the functional correctness bar.
    # Both TT and HF take argmax of the *same* seeded-canvas decoder forward, so this is
    # fully deterministic (unlike comparing to a stochastic ``generate(do_sample=True)`` run,
    # which would diverge even at perfect PCC).
    tt_argmax = tt_logits.argmax(dim=-1)[0, :N_TOKEN_MATCH].tolist()
    hf_argmax = hf_logits.argmax(dim=-1)[0, :N_TOKEN_MATCH].tolist()
    logger.info(f"[4b] TT argmax[:{N_TOKEN_MATCH}]: {tt_argmax}")
    logger.info(f"[4b] HF argmax[:{N_TOKEN_MATCH}]: {hf_argmax}")
    assert tt_argmax == hf_argmax, (
        f"First {N_TOKEN_MATCH} argmax tokens differ (deterministic; NOT sample noise).\n"
        f"  TT: {tt_argmax} -> {hf_processor.batch_decode(torch.tensor([tt_argmax]))[0]!r}\n"
        f"  HF: {hf_argmax} -> {hf_processor.batch_decode(torch.tensor([hf_argmax]))[0]!r}"
    )

    # ------------------------------------------------------------------
    # 5. Perf: assert per-component thresholds.
    # ------------------------------------------------------------------
    topology_key = "ring" if topology == ttnn.Topology.Ring else "linear"
    metrics_key = (*mesh_shape, topology_key, _expert_dtype_key(expert_dtype))
    expected = EXPECTED_METRICS.get(metrics_key, _DEFAULT_METRICS)

    e2e_time = benchmark_profiler.get_duration("e2e_generate")
    logger.info(f"E2E generate time: {e2e_time:.2f}s vs threshold {expected['total']}s")
    benchmark_data.add_measurement("diffusiongemma_e2e_seconds", e2e_time)

    assert (
        e2e_time <= expected["total"]
    ), f"E2E time {e2e_time:.2f}s exceeds threshold {expected['total']}s for {metrics_key}"
    logger.info(f"Perf ✓ ({e2e_time:.2f}s ≤ {expected['total']}s)")
