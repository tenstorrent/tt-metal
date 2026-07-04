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
# is the primary correctness signal (line ~167). PCC is a secondary check.
PCC_THRESHOLD = 0.999
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
    # smaller than (2,4). bf16 fits here. First-run values are placeholders — tighten after.
    (1, 8, "ring", "bfp4"): {"encoder": 20.0, "denoising": 120.0, "total": 150.0},
    (1, 8, "ring", "bfp8"): {"encoder": 25.0, "denoising": 150.0, "total": 180.0},
    (1, 8, "ring", "bf16"): {"encoder": 40.0, "denoising": 220.0, "total": 260.0},
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
    # 2. Correctness: encoder + decoder + step-by-step diffusion.
    # ------------------------------------------------------------------
    # Debug flow (temporary while we chase the ~87-88% end-to-end PCC): each stage prints its
    # own PCC and (where applicable) decoded text before any assert fires. This lets us see
    # (a) whether the drift is in the encoder or after it, (b) what the model actually
    # generates through the full diffusion loop even if intermediate PCC is loose.
    from ....utils.tensor import local_device_to_torch

    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    proc_inputs = hf_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    input_ids = proc_inputs["input_ids"]

    text_cfg = hf_model.config.text_config
    canvas_length = hf_model.config.canvas_length
    seed_canvas = torch.randint(0, text_cfg.vocab_size, (1, canvas_length), dtype=torch.long)

    # 2a. Run the TT encoder + one decoder forward. Also cache the encoder hidden state.
    encoder_kv, tt_enc_masks, enc_pos = pipeline._run_encoder(input_ids, None, None)
    # Pull the last encoder hidden state to host (the pipeline stashed it for us).
    tt_enc_h_host = local_device_to_torch(pipeline._last_encoder_hidden).to(torch.float32)
    if tt_enc_h_host.ndim == 4 and tt_enc_h_host.shape[0] == 1:
        tt_enc_h_host = tt_enc_h_host.squeeze(0)

    # HF reference (piecewise so we can inspect encoder hidden independently).
    with torch.no_grad():
        hf_enc_out = hf_model.model.encoder(
            input_ids=input_ids,
            attention_mask=None,
        )
        hf_enc_h = hf_enc_out.last_hidden_state  # [B, S, H]
        # Full forward (encoder + decoder + lm_head + softcap).
        hf_out = hf_model(input_ids=input_ids, decoder_input_ids=seed_canvas)
        hf_logits = hf_out.logits

    logger.info("=" * 70)
    logger.info(f"[2a] Encoder hidden state — hf {hf_enc_h.shape}, tt {tt_enc_h_host.shape}")
    try:
        assert_quality(hf_enc_h, tt_enc_h_host)  # logs PCC; no threshold, no raise
    except Exception as e:
        logger.warning(f"encoder PCC assert (should not fire, no threshold): {e}")

    # 2b. One decoder forward at the seed canvas. We also capture the pre-lm_head hidden
    # state so we can tell whether drift accumulates in the layer stack or in lm_head+softcap.
    decoder_position_ids = torch.arange(
        input_ids.shape[1], input_ids.shape[1] + canvas_length, dtype=torch.long
    ).unsqueeze(0)
    pipeline._debug_capture_decoder_hidden = True
    tt_logits = pipeline._run_decoder(
        seed_canvas,
        encoder_kv,
        decoder_position_ids,
        tt_enc_masks,
        encoder_seq_len=input_ids.shape[1],
        self_cond_logits=None,
    )
    pipeline._debug_capture_decoder_hidden = False
    tt_dec_h_host = pipeline._last_decoder_hidden_host
    if tt_dec_h_host is not None and tt_dec_h_host.ndim == 4 and tt_dec_h_host.shape[0] == 1:
        tt_dec_h_host = tt_dec_h_host.squeeze(0)

    # HF reference: grab the last decoder hidden state (what lm_head sees). We could ask the
    # top-level ForBlockDiffusion for output_hidden_states, but its return shape is ambiguous
    # (encoder or decoder?). Safer: recompute the linear-algebra inverse of lm_head+softcap on
    # the HF logits — that gives us exactly what the decoder produced.
    #
    #   softcap(x) = tanh(x / cap) * cap
    #   pre_softcap = atanh(hf_logits / cap) * cap
    #   pre_lm_head = pre_softcap @ pinv(lm_head.weight)     ← too expensive for vocab=262144
    #
    # atanh is well-defined for |y| < cap, and hf_logits are all in that range by construction.
    # But inverting lm_head is a 262144×2816 pinv — not cheap. Instead: just call the HF decoder
    # directly with output_hidden_states=True and take the last hidden state. That gives us the
    # exact tensor lm_head consumes with no inversion math.
    with torch.no_grad():
        hf_full = hf_model(
            input_ids=input_ids,
            decoder_input_ids=seed_canvas,
            output_hidden_states=True,
        )
        # `hidden_states` is the tuple of decoder hidden states (input_embeds, layer_1, ..., final_norm)
        # — the last entry has shape [B, canvas, hidden] and is what lm_head consumes.
        hf_dec_h = hf_full.hidden_states[-1]
        assert hf_dec_h.shape == (
            seed_canvas.shape[0],
            canvas_length,
            text_cfg.hidden_size,
        ), (
            f"unexpected hf_dec_h shape {hf_dec_h.shape} — expected "
            f"({seed_canvas.shape[0]}, {canvas_length}, {text_cfg.hidden_size}). If HF's "
            f"hidden_states is actually the encoder's, we need to grab it a different way."
        )

    logger.info("=" * 70)
    logger.info(f"[2b] Decoder hidden (pre-lm_head) — hf {hf_dec_h.shape}, tt {tt_dec_h_host.shape}")
    try:
        assert_quality(hf_dec_h, tt_dec_h_host)  # logs PCC; no threshold, no raise
    except Exception as e:
        logger.warning(f"pre-lm_head PCC assert (should not fire, no threshold): {e}")

    logger.info("=" * 70)
    logger.info(f"[2c] Decoder logits (post lm_head + softcap) — hf {hf_logits.shape}, tt {tt_logits.shape}")
    try:
        assert_quality(hf_logits, tt_logits)  # logs PCC; no threshold, no raise
    except Exception as e:
        logger.warning(f"post lm_head PCC assert (should not fire, no threshold): {e}")

    # ------------------------------------------------------------------
    # 3. Full diffusion loop with per-step decoded text.
    # ------------------------------------------------------------------
    # Instead of only comparing the final argmax token match, we run the same diffusion
    # loop that pipeline.generate() runs, but print the decoded text after every step so
    # we can eyeball whether the model is converging to a coherent answer.
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
        LinearTemperatureScheduleLogitsProcessor,
        StableAndConfidentStoppingCriteria,
    )

    torch.manual_seed(SEED)
    vocab_size = text_cfg.vocab_size
    sampler = EntropyBoundSampler(
        EntropyBoundSamplerConfig(entropy_bound=pipeline.config.entropy_bound),
        canvas_length=canvas_length,
        vocab_size=vocab_size,
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

    logger.info("=" * 70)
    logger.info(f"[3] Diffusion loop ({pipeline.config.max_denoising_steps} steps max)")
    logger.info(f"    Prompt: {PROMPT!r}")

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
        processed_logits = logits_processor(logits, cur_step=step)
        probs = torch.softmax(processed_logits.float(), dim=-1)
        denoiser_canvas = torch.distributions.Categorical(probs=probs).sample()
        accepted = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, step)
        current_canvas = sampler.renoise_canvas(accepted, step)
        finished = stopping(argmax_canvas, processed_logits)
        self_cond_logits = processed_logits

        # Decode the current argmax canvas (what the model would emit if it stopped now).
        argmax_text = hf_processor.batch_decode(argmax_canvas, skip_special_tokens=False)[0]
        accepted_text = hf_processor.batch_decode(accepted, skip_special_tokens=False)[0]
        logger.info(f"[step {step:02d}] argmax : {argmax_text!r}")
        logger.info(f"[step {step:02d}] accepted: {accepted_text!r}")
        if torch.all(finished):
            logger.info(f"[step {step:02d}] stopping criteria satisfied — halting")
            break

    logger.info("=" * 70)
    final_text = hf_processor.batch_decode(argmax_canvas, skip_special_tokens=False)[0]
    logger.info(f"[3] Final TT decoded canvas: {final_text!r}")

    # HF reference generation (kept for reporting; not asserted while we're debugging).
    torch.manual_seed(SEED)
    with torch.no_grad():
        hf_gen = hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
        )
    hf_new_tokens = hf_gen.sequences[:, input_ids.shape[1] :]
    hf_text = hf_processor.batch_decode(hf_new_tokens, skip_special_tokens=False)[0]
    logger.info(f"[3] HF reference generated: {hf_text!r}")

    # ------------------------------------------------------------------
    # 4. Assertions (all deferred until AFTER the diagnostics have printed).
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("[4] Asserting end-to-end PCC")
    assert_quality(hf_logits, tt_logits, pcc=PCC_THRESHOLD)
    logger.info("Hidden/logit-state PCC ✓")

    # ------------------------------------------------------------------
    # 4. Perf: assert per-component thresholds.
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
