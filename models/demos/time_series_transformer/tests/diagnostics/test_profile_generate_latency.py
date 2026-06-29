# tests/test_profile_generate_latency.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest

pytestmark = pytest.mark.skip(
    reason="diagnostic script: references _apply_layernorm_f32 removed in refactor; "
    "update if needed for local profiling, not part of CI suite"
)

"""
DIAGNOSTIC -- not part of the real bounty test suite.

Phase 1 of the Stage-1-closeout roadmap: profile BEFORE optimizing.
test_single_sequence_latency measures total wall time (~348ms, batch=1)
but doesn't say WHERE that time goes. This script re-implements generate()'s
logic with timers inserted around the three suspected cost centers:

  1. Encoder pass (1 call, should be cheap -- runs once)
  2. Decoder loop (24 sequential autoregressive steps -- HYPOTHESIS: this
     dominates, since each step does a full forward pass + at least one
     ttnn.to_torch/from_torch round trip, and TTNN op-dispatch overhead
     tends to dominate on tiny ops like this model's d_model=26)
  3. Host<->device transfer count and approximate time (every
     ttnn.to_torch()/ttnn.from_torch() call is real data movement, and
     generate()'s decode loop has several per step)

This is NOT a rewrite of generate() -- it's the same logic, instrumented,
run against the exact same single-sequence (batch=1) input used by
test_single_sequence_latency, so the total should roughly match that
test's ~348ms (if it doesn't, something about this instrumented copy
diverged from the real path, and the breakdown below would be suspect).
"""

import time
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import build_causal_mask
from tt.tst_embedding import prepare_decoder_input, prepare_encoder_input
from tt.tst_model import (
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_PARALLEL_SAMPLES,
    PREDICTION_LENGTH,
    _apply_layernorm_f32,
    _distribution_head,
    load_weights,
    run_decoder_step,
    run_encoder,
)

import ttnn

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def profiled_generate(
    device,
    weights,
    past_values,
    past_time_features,
    future_time_features,
    past_observed_mask,
    static_categorical_features,
    static_real_features,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_parallel_samples=NUM_PARALLEL_SAMPLES,
):
    """
    Mirrors generate() in tt/tst_model.py exactly, with timers added.
    Returns (samples, timing_dict) instead of just samples.
    """
    timing = {
        "mask_cache_build": 0.0,
        "encoder_total": 0.0,
        "encoder_prepare_input": 0.0,
        "encoder_run": 0.0,
        "repeat_and_transfer_setup": 0.0,
        "decode_steps": [],  # one entry per step, each a dict of sub-timings
        "total": 0.0,
    }
    t_start_all = time.perf_counter()

    B = past_values.shape[0]
    S = num_parallel_samples
    device_cpu = past_values.device

    t0 = time.perf_counter()
    causal_mask_cache = {T: build_causal_mask(device, T) for T in range(1, prediction_length + 1)}
    timing["mask_cache_build"] = time.perf_counter() - t0

    # --- Encoder ---
    t_enc_start = time.perf_counter()
    t0 = time.perf_counter()
    enc_emb, loc, scale = prepare_encoder_input(
        past_values=past_values,
        past_time_features=past_time_features,
        past_observed_mask=past_observed_mask,
        static_cat_features=static_categorical_features,
        static_real_features=static_real_features,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["encoder_value_proj"],
        pos_emb_weight=weights["encoder_pos_emb"],
        context_length=context_length,
    )
    enc_emb = _apply_layernorm_f32(enc_emb, weights["encoder_layernorm_f32"])
    timing["encoder_prepare_input"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    encoder_hidden = run_encoder(device, enc_emb, weights)
    timing["encoder_run"] = time.perf_counter() - t0
    timing["encoder_total"] = time.perf_counter() - t_enc_start

    # --- Repeat for parallel samples + initial host<->device transfer ---
    t0 = time.perf_counter()
    repeated_loc = loc.repeat_interleave(S, dim=0)
    repeated_scale = scale.repeat_interleave(S, dim=0)
    _sc = repeated_scale.squeeze(-1).squeeze(-1)
    _lc = repeated_loc.squeeze(-1).squeeze(-1)

    repeated_past_values_norm = (past_values.repeat_interleave(S, dim=0).float() - _lc.unsqueeze(-1)) / _sc.unsqueeze(
        -1
    )

    repeated_future_time = future_time_features.repeat_interleave(S, dim=0)
    repeated_static_cat = static_categorical_features.repeat_interleave(S, dim=0)
    repeated_static_real = static_real_features.repeat_interleave(S, dim=0)

    enc_hidden_t = ttnn.to_torch(encoder_hidden).float()  # device->host transfer #1
    enc_hidden_rep = ttnn.from_torch(
        enc_hidden_t.repeat_interleave(S, dim=0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )  # host->device transfer #1
    timing["repeat_and_transfer_setup"] = time.perf_counter() - t0

    future_samples = []

    for k in range(prediction_length):
        step_timing = {}
        t_step_start = time.perf_counter()
        T_dec = k + 1

        t0 = time.perf_counter()
        if k == 0:
            future_vals_k = torch.zeros(B * S, 1, device=device_cpu)
        else:
            prev_raw = torch.stack(future_samples, dim=1)
            future_vals_k = torch.cat([prev_raw, torch.zeros(B * S, 1, device=device_cpu)], dim=1)
        repeated_past_raw = repeated_past_values_norm * _sc.unsqueeze(-1) + _lc.unsqueeze(-1)
        step_timing["build_future_vals"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dec_emb_k = prepare_decoder_input(
            future_values=future_vals_k,
            future_time_features=repeated_future_time[:, : k + 1, :],
            past_values=repeated_past_raw,
            loc=repeated_loc,
            scale=repeated_scale,
            static_cat_features=repeated_static_cat,
            static_real_features=repeated_static_real,
            cat_embedder_weight=weights["cat_embedder"],
            value_proj_weight=weights["decoder_value_proj"],
            pos_emb_weight=weights["decoder_pos_emb"],
            context_length=context_length,
        )
        dec_emb_k = _apply_layernorm_f32(dec_emb_k, weights["decoder_layernorm_f32"])
        step_timing["prepare_decoder_input"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mask_k = causal_mask_cache[T_dec]
        dec_out = run_decoder_step(
            device,
            dec_emb_k,
            enc_hidden_rep,
            weights,
            causal_mask=mask_k,
            apply_layernorm=False,
        )
        step_timing["run_decoder_step"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dec_out_torch = ttnn.to_torch(dec_out).float()[..., :D_MODEL]  # device->host #2
        step_timing["decoder_output_to_torch"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        df_k, loc_d_k, scale_d_k = _distribution_head(dec_out_torch[:, -1:, :], weights)
        raw_loc_k = _lc + _sc * loc_d_k.squeeze(-1)
        raw_scale_k = _sc * scale_d_k.squeeze(-1)
        from torch.distributions import StudentT

        next_sample = StudentT(df=df_k.squeeze(-1), loc=raw_loc_k, scale=raw_scale_k).sample()
        future_samples.append(next_sample)
        step_timing["distribution_and_sample"] = time.perf_counter() - t0

        step_timing["total"] = time.perf_counter() - t_step_start
        timing["decode_steps"].append(step_timing)

    samples = torch.stack(future_samples, dim=1)
    timing["total"] = time.perf_counter() - t_start_all
    return samples.reshape(B, S, prediction_length), timing


def print_timing_breakdown(timing, batch_size, num_samples):
    logger.info(f"=== Latency breakdown (batch={batch_size}, samples={num_samples}) ===")
    logger.info(f"Total: {timing['total']*1000:.2f} ms")
    logger.info(f"  mask_cache_build:          {timing['mask_cache_build']*1000:.2f} ms")
    logger.info(
        f"  encoder_total:             {timing['encoder_total']*1000:.2f} ms "
        f"(prepare_input={timing['encoder_prepare_input']*1000:.2f}ms, "
        f"run={timing['encoder_run']*1000:.2f}ms)"
    )
    logger.info(f"  repeat_and_transfer_setup: {timing['repeat_and_transfer_setup']*1000:.2f} ms")

    decode_total = sum(s["total"] for s in timing["decode_steps"])
    logger.info(
        f"  decode_loop_total ({len(timing['decode_steps'])} steps): {decode_total*1000:.2f} ms "
        f"({decode_total*1000/len(timing['decode_steps']):.2f} ms/step avg)"
    )

    # Aggregate per-substep across all 24 decode steps
    substeps = [
        "build_future_vals",
        "prepare_decoder_input",
        "run_decoder_step",
        "decoder_output_to_torch",
        "distribution_and_sample",
    ]
    for sub in substeps:
        total_sub = sum(s[sub] for s in timing["decode_steps"])
        logger.info(
            f"    {sub}: {total_sub*1000:.2f} ms total "
            f"({total_sub*1000/len(timing['decode_steps']):.2f} ms/step avg, "
            f"{100*total_sub/decode_total:.1f}% of decode loop)"
        )

    logger.info(
        f"  Sum of measured parts: "
        f"{(timing['mask_cache_build']+timing['encoder_total']+timing['repeat_and_transfer_setup']+decode_total)*1000:.2f} ms "
        f"(vs total {timing['total']*1000:.2f} ms -- gap is unmeasured overhead, e.g. Python/dispatch glue)"
    )


def main():
    print("Loading HuggingFace reference model...")
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()

    device = ttnn.open_device(device_id=0)
    try:
        weights = load_weights(hf_model, device)
        inputs = load_ref("inputs.safetensors")

        batch_size = 1
        pv = inputs["input_past_values"][:batch_size]
        pt = inputs["input_past_time_features"][:batch_size]
        ft = inputs["input_future_time_features"][:batch_size]
        pm = inputs["input_past_observed_mask"][:batch_size]
        sc = inputs["input_static_categorical_features"][:batch_size].long()
        sr = inputs["input_static_real_features"][:batch_size]

        # Warmup (matches test_single_sequence_latency's warmup pattern --
        # first call pays one-time kernel compilation cost, shouldn't be
        # counted in the "real" measurement)
        print("\nWarmup run (excluded from timing)...")
        _, _ = profiled_generate(device, weights, pv, pt, ft, pm, sc, sr, num_parallel_samples=100)

        print("\nTimed run...")
        samples, timing = profiled_generate(device, weights, pv, pt, ft, pm, sc, sr, num_parallel_samples=100)
        print_timing_breakdown(timing, batch_size, 100)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
