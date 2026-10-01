#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible end-to-end (prefill + decode) benchmark for Qwen3.5-2B on ONE Blackhole P150.

Plain python (argparse, no pytest) so it can be run directly:

    python3 models/demos/blackhole/qwen36/demo/bench_e2e_p150.py --isl 4096 --osl 8 \
        --out models/demos/blackhole/qwen36/demo/bench_results/run.json

Prefer the wrapper models/demos/blackhole/qwen36/demo/run_bench_e2e_p150.sh, which pins every
QWEN36_*/QWEN_GDN_*/QWEN_* env flag explicitly first (see README_P150_PERF.md).

WHY REUSE, NOT REIMPLEMENT
---------------------------
This script builds the model, captures the prefill trace, and runs prefill/decode by calling the
SAME functions models/demos/blackhole/qwen36/demo/text_demo.py uses for its "traced_*" pytest
cases (Qwen36Model.from_pretrained, text_demo._warmup_prefill, Qwen36Model.capture_prefill_trace_
chunked, Qwen36Model.prefill_traced_chunked / prefill_masked_bucket, generator_interface.
prime_decode_trace, tt_transformers.generator.Generator.decode_forward). The one difference from
text_demo.py is structural, not numerical: text_demo.py's test function runs ONE generation per
process; this script splits that same call sequence into a one-time "capture" step (trace capture
+ decode-trace priming) followed by N repeatable "request" calls in the SAME process, so it can
report warmup vs. timed statistics. This mirrors the pattern already exercised by
/local/ttuser/atupe/qwen35_2b_handoff/scripts/step1/ttft_chunked_one.py's --multi mode (capture
once, then call model.prefill_traced_chunked repeatedly against the parked trace).

TTFT / TPOT / E2E DEFINITIONS (taken verbatim from text_demo.py's single-device traced path,
`_run_traced_generation`, so this script's numbers ARE the demo's numbers):

    signpost("inference_prefill")
    t0 = time.time()
    if T < chunk_size:
        logits = model.prefill_masked_bucket(token_ids, page_table, actual_len=T)
    else:
        logits = model.prefill_traced_chunked(padded_token_ids, page_table, actual_len=T)
    logits_torch = ttnn.to_torch(logits).squeeze()
    next_token = logits_torch.argmax().item()
    ttft = time.time() - t0

    ...
    for i in range(max_generated_tokens - 1):
        # Timing includes forward + sampling
        t_step = time.time()
        out = gen.decode_forward(...)
        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        next_token = int(dl.argmax())
        decode_times.append(time.time() - t_step)

So, in this script:
  TTFT  = host-side wall time from just before the (only) prefill call for a request, through
          chunk-trace replay(s) + eager masked-bucket tail (if any) + LM head + device->host
          readback (`ttnn.to_torch`) + host argmax for the FIRST generated token. No explicit
          ttnn.synchronize_device is added here because text_demo.py's traced path does not add
          one either -- `ttnn.to_torch` itself blocks for the readback.
  TPOT  = mean, over generated tokens 2..osl, of (forward + readback + host argmax) wall time for
          one decode step (`gen.decode_forward(..., read_from_device=True)`).
  E2E   = host wall time from the same t0 used for TTFT through the LAST decode step's token being
          available on host (i.e. TTFT + sum of the (osl-1) per-token decode times, measured
          directly rather than reconstructed).
  E2E_model = TTFT_median + (osl - 1) * TPOT_median  (reported alongside measured E2E for sanity).

QWEN36_ONDEV_ARGMAX (default "1"; recorded in the JSON header as "ondev_argmax"): the greedy argmax
runs ON DEVICE (Qwen36Model.set_greedy_token_output): after the LM head, each vocab split is
untilized to its one logical row, the rows are concatenated, and one ttnn.argmax gives a uint32
token (the first max index, the same token as the host torch.argmax of the same bf16 logits). The
prefill returns that token, and the decode trace captures the token ops. The host reads 4 bytes
instead of the logits, so with the flag on:
  TTFT  = the same window as above, but the readback is the 4-byte token and there is no host argmax.
  TPOT  = the same window as above, with the 4-byte token read instead of the logits + host argmax.
QWEN36_ONDEV_ARGMAX=0 restores the host-argmax path above exactly (no new device ops, same traces).

Determinism check: sampling is greedy argmax (temperature 0; on device, or on host with
QWEN36_ONDEV_ARGMAX=0), so every timed run must
generate byte-identical token ids to the (last) warmup run in the same process -- this is checked
and printed as PASS/FAIL, and optionally compared to a --check-ref JSON
({"expected_first_tokens": [...]}).

PROMPT CONSTRUCTION (--prompt-mode corpus, the default): the prompt must be a legitimate one a
person can judge the model's output of, not tiled/repeated filler. This script takes the SAME real
document text_demo.py's own long-context evals use (the Frankenstein corpus, downloaded/cached by
text_demo._load_and_cache_context -- reused here via the same cache-file convention, see
`_load_source_document`), appends the demo's own chat-template mechanism (the
`tokenizer.apply_chat_template` pattern from text_demo._get_prompt's QWEN35_REF_PROMPT branch) with
the instruction "Summarize the text above in a few bullet points.", and picks the document cut
point by binary search on character count so the TOTAL prompt is exactly --isl tokens -- no
repetition anywhere. It prefers a sentence boundary, falling back to a word boundary, falling back
(when neither lands exactly on --isl -- verified this is actually needed for isl=4096/2642/8000 on
this corpus, since a single word can be >1 BPE token and skip clean over the target) to an
exhaustive character-level scan within the final few-character bracket; see
`_exact_isl_legitimate_corpus_prompt` for the full explanation and exactly which fallback tier
text_demo.py has no equivalent of. --prompt-mode synthetic remains as a fallback-only mode
(deterministic random token ids, not a prompt with a legitimate answer). --demo-prompt bypasses all
of this and uses text_demo._get_prompt(4096, ...) unchanged (2642 tokens) for byte-for-byte
comparability with a `pytest text_demo.py -k traced_4k` run. Everything else here (model
construction, trace capture, prefill, decode, host argmax) is the demo's own code.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Must be set before importing ttnn / text_demo (text_demo reads MESH_DEVICE at import time to
# pick its DEVICE_PARAMS / mesh shape). setdefault() so a caller (e.g. run_bench_e2e_p150.sh) that
# already exported these keeps control.
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-2B")
os.environ.setdefault("MESH_DEVICE", "P150")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TT_METAL_HOME", str(REPO_ROOT))
os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))

# ---------------------------------------------------------------------------------------------
# Full QWEN36_*/QWEN_GDN_*/QWEN35_*/QWEN9B_*/QWEN_* env flag table.
#
# Regenerate with:
#   grep -h "os.environ.get(\"QWEN" -r models/demos/blackhole/qwen36 \
#       models/experimental/gated_attention_gated_deltanet | sed 's/.*environ.get(//' | cut -d, -f1 \
#       | sort -u
#
# name -> (documented_default, one-line meaning). "documented_default" is a literal string that is
# SAFE to `export NAME=<value>` and reproduces the unset behaviour, EXCEPT for the flags in
# UNSAFE_TO_EXPORT_UNSET below: those are read via a bare truthiness check (`if os.environ.get(X):`)
# or fall back to a value computed from OTHER locals (not a fixed literal) -- exporting ANY
# non-empty string for a bare-truthy flag flips it on, and exporting a made-up literal for a
# dynamic-default flag would silently override the real (correct) computed default. Those must be
# left truly unset (not even exported as "").
# ---------------------------------------------------------------------------------------------
ALL_QWEN_FLAG_DEFAULTS = {
    "QWEN35_GDN_DECODE_BF16": (
        "0",
        "GDN decode matmul precision; 1=allow bf16 (faster/less precise), default=high precision",
    ),
    "QWEN35_GDN_STATE_BF16": ("0", "GDN recurrent-state dtype; 1=bf16 state, default=higher precision"),
    "QWEN35_NO_REPEAT_NGRAM": ("0", "n-gram size to block repeats during sampling; 0=disabled"),
    "QWEN35_NO_THINK": (
        "<unset>",
        "if set (any value), seed an empty <think></think> block for the Frankenstein long-context prompt",
    ),
    "QWEN35_REF_PROMPT": (
        "<unset>",
        "if set (and seqlen>=4096), use the 64k reference extractive prompt instead of the Frankenstein corpus",
    ),
    "QWEN35_REP_PENALTY": ("1.0", "repetition penalty applied to sampled logits; 1.0=off"),
    "QWEN35_TEMP": ("0", "sampling temperature; 0=greedy argmax (this bench always uses greedy)"),
    "QWEN35_TOP_K": ("0", "top-k sampling cutoff; 0=disabled"),
    "QWEN35_TOP_P": ("1.0", "nucleus top-p cutoff; 1.0=disabled"),
    "QWEN35_TP_DECODE_EAGER": ("0", "TP-only: force eager (non-traced) decode; default traced"),
    "QWEN35_TP_PREFILL_EAGER": ("0", "TP-only: force eager (non-traced) chunk prefill; default traced"),
    "QWEN36_ATTN_FUSED_QKV": (
        "1",
        "fuse Q/K/V projection into one matmul when a fused weight exists; 0=separate matmuls",
    ),
    "QWEN36_ATTN_GATE_FUSED": ("1", "fuse the attention output-gate multiply into the op; 0=separate"),
    "QWEN36_ATTN_KV_BF8": ("0", "store attention KV cache in bfp8 instead of bf16; 1=enable"),
    "QWEN36_ATTN_L1_MAX_T": ("2048", "max token count for L1-resident (vs DRAM) attention intermediates"),
    "QWEN36_ATTN_QKNORM_HIFI2": ("1", "use HiFi2 math fidelity for Q/K RMSNorm; 0=default fidelity"),
    "QWEN36_BATCHED_DECODE_MODE": ("shard", "batched (TP, B>1) decode dispatch mode; irrelevant on single P150"),
    "QWEN36_BUCKET_TEST_CTX": ("8192", "context length used by the decode-bucketing TEST harness (not the demo path)"),
    "QWEN36_BUCKET_TEST_WIDTHS": ("1,8", "batch widths swept by the decode-bucketing TEST harness"),
    "QWEN36_CAPACITY_TEST_BMAX": ("8", "max batch size swept by the capacity TEST harness"),
    "QWEN36_CAPACITY_TEST_EAGER_PROFILE": ("0", "capacity TEST harness: profile the eager (untraced) path"),
    "QWEN36_CAPACITY_TEST_ITERS": ("20", "iterations per trial in the capacity TEST harness"),
    "QWEN36_CAPACITY_TEST_LAYER_INDEX": ("<unset>", "capacity TEST harness: run only this layer index"),
    "QWEN36_CAPACITY_TEST_N_LAYERS": ("<unset>", "capacity TEST harness: truncate to N layers"),
    "QWEN36_CAPACITY_TEST_TRIALS": ("5", "trials in the capacity TEST harness"),
    "QWEN36_CAPACITY_TEST_WARMUP": ("3", "warmup iterations in the capacity TEST harness"),
    "QWEN36_DEBUG_DECODE_TIMING": ("0", "print per-substep decode timing breakdown to stdout"),
    "QWEN36_DECODE_PROGCFG": ("1", "use the tuned decode matmul program config; 0=ttnn auto-config"),
    "QWEN36_GDN_CONV_CHUNKS": (
        "<unset>",
        "override the causal-conv1d chunk count; unset=auto-derived from sequence length",
    ),
    "QWEN36_GDN_CONV_KDA": ("1", "use the fused KDA causal-conv1d+SiLU kernel for kernel_size==4; 0=generic conv"),
    "QWEN36_GDN_CONV_KDA_CCS": (
        "<unset>",
        "override the KDA conv kernel's channel-chunk-size; unset=model-computed default",
    ),
    "QWEN36_GDN_CONV_KDA_FP32ACC": ("0", "accumulate the KDA conv kernel in fp32; 1=enable"),
    "QWEN36_GDN_CONV_KDA_TILED": (
        "1",
        "INT-3: KDA conv reads the TILE in-proj output + TILE conv state directly (tiled kernel, "
        "channel_chunk_size=128, returns new_state; bit-identical); 0=untilize + ROW_MAJOR op + slice/tilize glue",
    ),
    "QWEN36_GDN_CONV_LEGACY": ("0", "force the legacy (pre-native) conv1d implementation"),
    "QWEN36_GDN_CONV_SILU_SHARDED": ("0", "run the conv+SiLU on a sharded memory config; 1=enable"),
    "QWEN36_GDN_CONV_T3_MAX": ("0", "cap on the conv kernel's T3 tiling dimension; 0=no cap"),
    "QWEN36_GDN_CONV_REPACK": (
        "perlayer",
        "fused GDN decode only: conv-history repack after prefill, perlayer (8 ops/layer), batched, or "
        "gather (one shared index table + one ttnn.embedding per layer; P6_INT1C item 3)",
    ),
    "QWEN36_GDN_CONV_TILED_SPLIT": ("0", "split the conv1d input into tiles; 1=enable"),
    "QWEN36_GDN_CONV_XIN_L1_MAX_T": (
        "<unset>",
        "max token count for L1-resident conv input; unset=model-computed default (xin_l1_max_t)",
    ),
    "QWEN36_TRACE_GUARD": (
        "0",
        "trace-allocation guard: 1 = needs TT_METAL_TRACE_ALLOC_TRACKING=1 before the ttnn import (the bench "
        "asserts it at start; execute_trace then fails on unsafe live buffers); run_bench_e2e_p150.sh pins it "
        "from BENCH_TRACE_GUARD (runner default 1) and exports the tracker env",
    ),
    "QWEN36_GDN_DECODE_FUSED": (
        "2",
        "2 (default, user decision 2026-09-25)=GDN decode via ttnn.experimental.kda.gdn_decode_step (FP32 GDN "
        "state, 1 device, B=1); 0=composite GDN decode (pre-T8 path)",
    ),
    "QWEN36_GDN_FLA_INPUTS_DRAM": (
        "<unset>",
        "force FLA (chunk-scan) inputs into DRAM instead of L1; unset=auto (1 if QWEN_GDN_PATH=fused else 0)",
    ),
    "QWEN36_GDN_FUSED_PREFILL": ("1", "use the fused chunk-parallel GDN prefill kernel; 0=legacy per-op prefill"),
    "QWEN36_GDN_GATE_CLIP": ("0", "clip the GDN output gate; 1=enable"),
    "QWEN36_GDN_GATE_FUSED": ("1", "fuse the GDN output-gate multiply; 0=separate op"),
    "QWEN36_GDN_GB_LAYOUT": ("0", "alternate gate/beta tensor layout for GDN; 1=enable"),
    "QWEN36_GDN_L1_MAX_T": ("0", "max token count for L1-resident GDN intermediates; 0=disabled (always DRAM)"),
    "QWEN36_GDN_NATIVE_CONV1D": ("1", "use the native fused conv1d weight path; 0=legacy"),
    "QWEN36_GDN_PCFG": (
        "auto",
        "PR #57440 port: fused FLA prefill program_config (tt/gdn/gated_deltanet.py); auto=op cost model, "
        "nv1np6|nv1np5|nv2np4|nv2np6 = pinned ChunkGdnFusedProgramConfig. run_bench_e2e_p150.sh pins it "
        "(runner default nv1np5; nv1np6 = the pre-#57440 geometry)",
    ),
    "QWEN36_GDN_WYINV": (
        "auto",
        "PR #57445 WY-inverse selector of the fused FLA prefill (tt/gdn/decode.py): auto=not passed (op default "
        "AUTO), horner|sfpu = ttnn.ChunkGdnWyInverse.HORNER|SFPU. run_bench_e2e_p150.sh pins it "
        "(runner default horner)",
    ),
    "QWEN36_GDN_POST_L1": ("1", "keep GDN post-scan tensors in L1; 0=DRAM"),
    "QWEN36_GDN_POST_L1_OUTPROJ": ("1", "keep the GDN output-projection input in L1; 0=DRAM"),
    "QWEN36_GDN_POST_L1_SCAN": ("1", "keep the GDN scan output in L1; 0=DRAM"),
    "QWEN36_GDN_SPLIT_PROJ": ("1", "split the GDN QKVBA projection matmul; 0=single matmul"),
    # I-1 item flags (tt/tp_common.py I1_FLAG_DEFAULTS; read via tp_common.i1_enabled); 0 = pre-I-1 path.
    "QWEN36_I1_D15": ("1", "I-1 D15: decode full-attn q|k|v via 2 matmuls (qkv_fused + gate_deint); 0=3 matmuls"),
    "QWEN36_I1_D3": ("1", "I-1 D3: decode decoder/final RMSNorm width-sharded on 8 cores; 0=1-core norm"),
    "QWEN36_I1_D4A": (
        "0",
        "I-1 D4a+D5: decode full-attn RoPE via rotary_embedding_hf on [1,1,H,64], row-replicated cos/sin, no transposes",
    ),
    "QWEN36_I1_D6": ("1", "I-1 D6: decode full-attn head concat via one reshape; 0=transpose + concatenate_heads"),
    "QWEN36_I1_P5": ("1", "I-1 P5: prefill GDN [g|a|0|b|0] tile-padded weight (no a/b untilize); 0=mega slice"),
    "QWEN36_I1_ROPEWARM": ("1", "I-1: compile the chunk RoPE slice + cos/sin copy before the prefill trace capture"),
    # I-2 item flags (tt/tp_common.py I2_FLAG_DEFAULTS; read via tp_common.i2_value); 0 = pre-I-2 path.
    "QWEN36_I2_PICKER": ("1", "I-2: prefill picker output-subblock area cap 8 (fp32 dest off; bit-exact); 0=cap 4"),
    "QWEN36_I2_S1": (
        "2",
        "I-2 S1: fused SwiGLU minimal_matmul blocking (1=7/8/10 1x5, 2=7/8/16 1x8 bit-exact); 0=4/8/16 2x4",
    ),
    "QWEN36_I2_S2": ("0", "I-2 S2: MLP down-proj 2D-mcast linear 13x10 (1=in0_block_w 16, 8=bw 8); 0=minimal_matmul"),
    "QWEN36_I2_S3": ("0", "I-2 S3: GDN qkv in-proj 2D-mcast linear 13x10 (1=in0_block_w 16, 8=bw 8); 0=minimal_matmul"),
    # M1 item flags (tt/tp_common.py M1_FLAG_DEFAULTS; read via tp_common.m1_value); 0 = current path.
    "QWEN36_M1_S2": (
        "0",
        "M1 S2: MLP down-proj 2D-mcast matmul 13x10 bw8 pcM7 pcN5 1x5 at T=2048 (numerics change); 0=minimal_matmul",
    ),
    "QWEN36_M1_S3": (
        "0",
        "M1 S3: GDN qkv in-proj 2D-mcast matmul 13x10 bw8 pcM7 pcN15 1x5 at T=2048, attention_norm output + "
        "in-proj output in L1 (numerics change); 0=minimal_matmul, DRAM",
    ),
    "QWEN36_M1_S4": (
        "0",
        "M1 S4: GDN z|a|0|b|0 in-proj 2D-mcast matmul 13x10 bw8 pcM7 pcN6 1x6 at T=2048, attention_norm output + "
        "in-proj output in L1; 0=picker linear, DRAM",
    ),
    # M2 item flags (tt/tp_common.py M2_FLAG_DEFAULTS; read via tp_common.m2_value); 0 = current path.
    "QWEN36_M2_LASTROW": (
        "0",
        "M2: traced chunk computes the last layer's gate/o_proj/MLP for row chunk-1 only (1-row decode "
        "progcfgs; trace output [1,1,dim]; numerics change); 0=all rows",
    ),
    "QWEN36_M2_NOWHERE": (
        "0",
        "M2: text-only model -> no vision-splice where in the traced chunk (bit-exact); 0=where",
    ),
    "QWEN36_M2_REPACK_LATE": (
        "0",
        "M2: exact-multiple traced prefill defers the GDN conv-history repack to the next decode entry "
        "(after the first-token read; bit-exact); 0=repack before the LM head",
    ),
    # M3 item flags (tt/tp_common.py M3_FLAG_DEFAULTS; read via tp_common.m3_value); 0 = current path.
    "QWEN36_M3_REPACK_TRACE": (
        "0",
        "M3: exact-multiple traced prefill replays a captured trace of the GDN conv-history repack (same "
        "point, same ops; bit-exact); 0=eager repack",
    ),
    "QWEN36_M3_ZB": (
        "0",
        "M3: with QWEN36_M1_S2 / _S3 on, the M1 matmul runs as ttnn.linear + a zero bias allocated at load "
        "(FUSE_BIAS; bit-identical to minimal_matmul); 0=ttnn.matmul",
    ),
    # C2 item flags (tt/tp_common.py C2_FLAG_DEFAULTS; read via tp_common.c2_value); 0 = current path.
    "QWEN36_C2_SGRN": (
        "0",
        "C2: unmasked T=2048 GDN chunks with M1 S4: z slice + typecast + per-head rms_norm + nlp_concat_heads + "
        "SILU gate multiply as one sigmoid_gated_rms_norm(gate_activation=silu) reading z in place from the L1 "
        "z|a|0|b|0 in-proj output (numerics change); 0=5 ops",
    ),
    # R3 item flags (tt/tp_common.py R3_FLAG_DEFAULTS; read via tp_common.r3_value); 0 = current path.
    "QWEN36_R3_FLA_IN_L1": (
        "0",
        "R3: unmasked T=2048 GDN chunks: tiled KDA conv q/k/v and the beta/g chain stay L1 (FLA inputs; needs "
        "the Ct==1 ChunkGdnFused CB shrink; bit-exact); 0=DRAM (QWEN36_GDN_FLA_INPUTS_DRAM behavior)",
    ),
    "QWEN36_R3_O_L1": (
        "0",
        "R3: unmasked T=2048 GDN chunks: ChunkGdnFused writes o + final state to L1 (needs the CB shrink; "
        "bit-exact); 0=DRAM",
    ),
    "QWEN36_R3_SGRN_GAB_L1": (
        "0",
        "R3: with QWEN36_C2_SGRN=1 and QWEN36_LAYER_RESID_L1=1, gab stays L1 (C2 variant a; needs the CB "
        "shrink; bit-exact); 0=C4 variant b (gab DRAM)",
    ),
    # M4 item flags (tt/tp_common.py M4_FLAG_DEFAULTS; read via tp_common.m4_value); 0 = current path.
    "QWEN36_M4_R4A": (
        "0",
        "M4: with M2 LASTROW, the last layer's 3 one-row reads use tile-aligned [T-32:T] block slices + row 31 "
        "(no untilize of the whole tensor; bit-exact); 0=[T-1:T] slices",
    ),
    "QWEN36_M4_R4B": (
        "0",
        "M4: with M2 LASTROW, the last layer's prefill SDPA is the paged decode SDPA for the chunk's last row "
        "(position buffer written per chunk; numerics change); 0=chunked SDPA over all rows",
    ),
    # M5 item flags (tt/tp_common.py M5_FLAG_DEFAULTS; read via tp_common.m5_value); 0 = current path.
    "QWEN36_M5_TAIL_TRACE": (
        "0",
        "M5: exact-multiple prompts, prepared order: final norm + LM head (+ argmax) after the last chunk replay "
        "run as one captured trace into a persistent DRAM output (same ops + 1 copy; bit-exact); 0=eager tail",
    ),
    "QWEN36_M5_ADDNORM": (
        "0",
        "M5: T=2048 traced chunks: each full-T residual add + the RMSNorm that reads it (attn add -> ffn_norm, "
        "MLP add -> next attention_norm) as one rms_norm(residual_output_tensor=h) (bit-exact); 0=add + norm",
    ),
    # I-3 item flags (tt/tp_common.py I3_FLAG_DEFAULTS; read via tp_common.i3_value); 0 = pre-I-3 path.
    "QWEN36_I3_FA_PROGCFG": (
        "1",
        "I-3: full-attn decode projections via swept 1D progcfgs (bit-exact); 0=ttnn auto-config",
    ),
    "QWEN36_I3_LMHEAD": (
        "0",
        "I-3 LM head: A=4x DRAM-sharded linear (LoFi), A2=A with HiFi2/fp32, A3=10x DRAM-sharded linear (HiFi2/fp32, "
        "bw2, in0 8x8, chunks built at load, unsplit weight freed), B=unsplit 1D linear 13x10 (LoFi), "
        "C=4x minimal 13x2 K16 (1-ulp change); all change numerics; 0=QWEN36_LMHEAD_SPLIT minimal path",
    ),
    "QWEN36_I3_MLP_FUSED_GU": (
        "0",
        "I-3: MLP decode gate|up as one DRAM-sharded matmul + slices + silu*mul (numerics change); 0=2 matmuls",
    ),
    "QWEN36_I3_MLP_PROGCFG": ("1", "I-3: MLP decode gate/up 1D 13x4 bw4 pcN4 progcfg (bit-exact); 0=13x3 bw8 pcN5"),
    # F item flags (tt/tp_common.py F_FLAG_DEFAULTS; read via tp_common.f_value); 0 = current path.
    "QWEN36_F_MLP_GU_BF8": (
        "0",
        "F: single device, MLP gate/up weights (decode w1/w3 + prefill packed w_gate_up) bfloat8_b (numerics change, "
        "new weight-cache files); 0=bfloat4_b",
    ),
    # W flag (tt/tp_common.py w_bf4_enabled / mm_weight_dtype / mlp_gate_up_dtype); 0 = current dtypes.
    "QWEN36_W_BF4": (
        "0",
        "W: single device, every matmul weight (GDN/FA projections, MLP gate/up/down, LM head) bfloat4_b "
        "(numerics change, new weight-cache files; overrides QWEN36_F_MLP_GU_BF8); 0=current dtypes (bfloat8_b "
        "for the projections, down and LM head; gate/up per QWEN36_F_MLP_GU_BF8)",
    ),
    # LM flag (tt/tp_common.py lm_bf4fast_enabled; T24_LMFAST); 0 = current A3 LM head.
    "QWEN36_LM_BF4FAST": (
        "0",
        "LM: single device, I-3 A3 LM head with a bfloat4_b weight (QWEN36_W_BF4=1): LoFi compute config + 3 workers "
        "per DRAM bank (bit-exact vs HiFi2 / 2 workers); no effect with a bfloat8_b LM head; 0=HiFi2 / 2 workers",
    ),
    # R5 item flags (tt/tp_common.py R5_FLAG_DEFAULTS; read via tp_common.r5_value); 0 = current path.
    "QWEN36_R5_GLU": (
        "0",
        "R5: single device, T == 2048 prefill chunks with bfloat8_b gate/up: fused-SwiGLU gate/up as the 2D-mcast "
        "ttnn.matmul fuse_swiglu epilogue (needs the C++ config field; numerics change); 0=minimal_matmul fuse_swiglu; 1=in0_block_w 4 config; 2=in0_block_w 16 + glu_last_block + glu_sfpu_on_pack",
    ),
    # MM item flags (tt/tp_common.py MM_FLAG_DEFAULTS; read via tp_common.mm_value); 0 = current path.
    "QWEN36_MM_BW16": (
        "0",
        "MM: single device, T == 2048 prefill chunks: in0_block_w 16 (instead of 8) for MLP down (M1 S2), GDN "
        "z|a|0|b|0 in-proj (M1 S4), the o-proj family (FA/GDN o_proj) and the FA q|k|v fused proj (numerics change, "
        "PCC ~0.99995); excludes GDN q|k|v in-proj (M1 S3); 0=in0_block_w 8 for all of them",
    ),
    "QWEN36_ACT_BF8_RESID": (
        "1",
        "P6_BF8ACT: 1 = bfloat8_b output for the T>1 prefill o-proj / MLP down matmuls (G3, F3, M2); 0 = bf16 (no change)",
    ),
    "QWEN36_ACT_BF8_NORM": (
        "1",
        "P6_BF8ACT: 1 = bfloat8_b fused add+RMSNorm / layer-0 norm output n at T == 2048 (residual h stays bf16); 0 = bf16 (no change)",
    ),
    "QWEN36_RESID_HS": (
        "1",
        "P11_SHARDRES_B: 1 = T == 2048 prefill residual stream h and the o-proj / down-proj outputs (G3, F3, M2) HEIGHT_SHARDED L1 "
        "[32, 2048] on the 64 fused add+RMSNorm cores (bit-exact); 0 = interleaved (no change)",
    ),
    # SGRN item flag (tt/tp_common.py sgrn_kernel_variant(); only with QWEN36_C2_SGRN=1); unset = op default 4.
    "QWEN36_SGRN_VARIANT": (
        "5",
        "P6_INT1C item 2 (runner default 5 since P9_INT1G): kernel_variant passed to sigmoid_gated_rms_norm. 0=legacy 7-pass kernel (bit-exact "
        "with the pre-P6_INT1C op); 1-3=fused kernel (bit-exact with each other); 4=fused kernel with an "
        "exp_21f sigmoid (within 1 bf16 ulp of 0 for >99.9% of values; not bit-exact); 5=fused gated RMSNorm compute kernel (P9_SGRN2); unset=op default (4)",
    ),
    # P7_INT1D item flags (merged into r3 2026-09-28).
    "QWEN36_REPACK_AFTER_TTFT": (
        "1",
        "P23_REPACK: 1 = replay the M3 conv-history repack trace at the first decode step (after the first-token "
        "readback) instead of inside TTFT; needs M3 REPACK_TRACE; same ops/buffers. run_bench_e2e_p150.sh pins it "
        "(runner default 1 since INT2j; code default 0)",
    ),
    "QWEN36_GDN_GATE_FUSE": (
        "0",
        "P7_INT1D (P5_GATING): fuse the GDN beta sigmoid+scale and the a+dt_bias+softplus into single BinaryNg "
        "ops (same math, bit-exact; distinct from the pre-existing QWEN36_GDN_GATE_FUSED, which fuses the "
        "output-gate multiply). run_bench_e2e_p150.sh pins it (runner default 1); 0=separate ops",
    ),
    "QWEN36_GDN_GATES_OP": (
        "1",
        "P10_GDNGATE: one ttnn.experimental.gdn_gates op makes the GDN beta and g (fp32) straight from the a/b columns "
        "of gab at T>1 prefill (M1 S4 gab branch, fused FLA path), replacing the 2 slices + sigmoid-mul + add-softplus "
        "+ mul (+ the FLA op's 2 typecasts); bit-identical to the chain. Runner default 1; 0=slice/eltwise chain",
    ),
    "QWEN36_PRELUDE_TRACE": (
        "1",
        "P18_PRELUDE: the per-request GDN state reset (recurrent + conv state copies of every GDN layer; the "
        "conv_hist copies are dropped, the repack overwrites conv_hist) and the chunk-0 RoPE slice/copy are "
        "captured into one trace at prepare and replayed at request start (bit-exact). "
        "run_bench_e2e_p150.sh pins it (runner default 1); 1=trace, 0=eager ops",
    ),
    "QWEN36_ROPE_L1": (
        "0",
        "P7_INT1D (P7_ROPE): place the persistent per-chunk RoPE cos/sin buffers in L1 interleaved instead of "
        "DRAM interleaved (bit-exact). run_bench_e2e_p150.sh pins it (runner default 1); 0=DRAM",
    ),
    "QWEN36_FA_GATE_FAST": (
        "1",
        "P14_FAGATE2 B2: FA prefill (T>1) gate: SIGMOID fused into the gate matmul program config, then a plain "
        "multiply (bit-identical). "
        "run_bench_e2e_p150.sh pins it (runner default 1); 1=fused",
    ),
    "QWEN36_SDPA_CONCAT_OUT": (
        "1",
        "P15: the flexible chunked SDPA writes the head-concatenated [B,1,T,H*D] output directly "
        "(concat_heads_output=True) and the concatenate_heads op is skipped at T>1 prefill (bit-exact layout change). "
        "run_bench_e2e_p150.sh pins it (runner default 1); 1=direct concat write",
    ),
    "QWEN36_ROPE_PARTIAL_INPLACE": (
        "1",
        "P10_ROPE: prefill partial RoPE (rotary 64 of head 256) as ONE in-place ttnn.experimental.rotary_embedding_hf "
        "call per tensor (rotary_dim=64) instead of slice+rope+slice+concat (bit-exact). run_bench_e2e_p150.sh pins "
        "it (runner default 1); 1=in-place op",
    ),
    "QWEN36_GDN_STATE_INPLACE": (
        "0",
        "P9_INT1F (P7_STATECOPY): on the traced chunked prefill the fused FLA op and the KDA conv op write the "
        "recurrent / conv state straight into the persistent buffers (bit-exact). run_bench_e2e_p150.sh pins it "
        "(runner default 1); 0=new tensor plus copy",
    ),
    "QWEN36_FLA_SCAN_FID": (
        "<unset>",
        "R10B experiment hook (chunk_gdn_fused_program_factory.cpp): math-fidelity override for the fused FLA "
        "scan/receiver compute kernel only; HiFi4|HiFi3|HiFi2|LoFi, unset=HiFi4 (today's fixed behaviour). "
        "run_bench_e2e_p150.sh pins it only with QWEN36_FLA_SCAN_FID_BY_LEN=0 (then runner default HiFi3 since "
        "P7_INT1D item B2: +/-2% vs f64, -1.7 ms/4k vs HiFi4); with BY_LEN=1 (runner default) it stays unset, "
        "because a set env var overrides the by-length choice. QWEN36_FLA_PREP_FID is the same hook for the "
        "prep/producer kernel; the runner leaves it unset (no pin)",
    ),
    "QWEN36_FLA_SCAN_FID_BY_LEN": (
        "0",
        "P11_FLALEN (tt/gdn/gated_deltanet.py + tt/model.py): the fused FLA fidelity follows each request's "
        "prompt length, scan HiFi2 up to 65536 tokens and scan HiFi3 above (prep HiFi4 in both), through the "
        "hashed ChunkGdnFusedProgramConfig.scan_math_fidelity / prep_math_fidelity; prepare compiles both when "
        "max_prompt_len > 65536 and the chunk trace is re-captured when a request needs the other one. "
        "run_bench_e2e_p150.sh pins it (runner default 1); 0=the fixed QWEN36_FLA_SCAN_FID",
    ),
    # INT-4 SDPA flags (ttnn_gated_attention.py; need upstream PR #57395 + the T3d chunked K/V chains in the op).
    "QWEN36_I4_SDPA_EXP_COMPAT": (
        "1",
        "INT-4: gated-attention SDPA configs pass exp_approx_mode=True = the kernel the pre-PR #57395 False ran; "
        "0=literal False (post-PR accurate rescale exp)",
    ),
    "QWEN36_I4_SDPA_Q64": (
        "1",
        "INT-4: flexible chunked prefill SDPA q_chunk 64 / k_chunk 128, exp_approx_mode=True; 0=q/k_chunk 128",
    ),
    "QWEN36_LAYER_L1_MAX_T": ("0", "max token count for L1-resident layer activations; 0=disabled"),
    "QWEN36_LAYER_RESID_L1": ("0", "keep the residual stream in L1 for short sequences; 1=enable"),
    "QWEN36_LMHEAD_MINIMAL": ("1", "use the minimal (narrow) LM-head matmul config; 0=default config"),
    "QWEN36_LMHEAD_SPLIT": ("8", "number of column-splits for the LM-head matmul"),
    "QWEN36_MAX_TOKENS_ALL_USERS": (
        "<unset>",
        "vLLM integration: override the max-tokens-per-batch budget; unset=auto",
    ),
    "QWEN36_MLP_FUSED_SWIGLU": ("1", "fuse the MLP SwiGLU gate+up multiply; 0=separate ops"),
    "QWEN36_MLP_L1_OUT": ("1", "keep MLP down-proj output in L1 for T<=2048; 0=DRAM"),
    "QWEN36_MLP_LEGACY_SHORT": ("0", "force the legacy MLP path for short sequences (T<=512)"),
    "QWEN36_MLP_MINIMAL_MM": ("1", "use the minimal MLP matmul program config; 0=default"),
    "QWEN36_ONDEV_ARGMAX": (
        "1",
        "bench/text_demo single-device greedy: argmax on device, read a 4-byte token; 0=read logits, host argmax",
    ),
    "QWEN36_PREFILL_DEBUG": ("0", "print chunk-by-chunk trace-execute debug lines during prefill"),
    "QWEN36_PREFILL_MINIMAL_CFG": ("1", "use the minimal prefill matmul program config; 0=default"),
    "QWEN36_PREFILL_MM_FP32_ACC": ("0", "accumulate prefill matmuls in fp32; 1=enable"),
    "QWEN36_PREFILL_MM_PACKER_L1_ACC": ("1", "accumulate prefill matmuls in the packer's L1 accumulator; 0=disable"),
    "QWEN36_PREFILL_OVERLAP": ("1", "overlap prefill chunk compute/dispatch; 0=serial"),
    "QWEN36_PREFILL_PROGCFG": ("1", "use the tuned prefill matmul program config; 0=ttnn auto-config"),
    "QWEN36_PREFILL_PROGCFG_OVERRIDES": ("1", "apply the 13x10-grid prefill program-config overrides; 0=disable"),
    "QWEN36_PREFIX_WRITE_ITERS": (
        "100",
        "iteration count for the prefix-cache-write micro-benchmark (not the demo path)",
    ),
    "QWEN36_PREFIX_WRITE_WIDTH": ("1", "batch width for the prefix-cache-write micro-benchmark"),
    "QWEN36_ROPE_DEVICE_TABLE": (
        "1",
        "slice RoPE cos/sin from a persistent on-device table (device-to-device copy); 0=host recompute+upload",
    ),
    "QWEN36_ROPE_LEGACY": ("0", "force the legacy RoPE implementation"),
    "QWEN36_SDPA_PREFILL_CHUNKS": (
        "<unset>",
        "experimental gated-attention SDPA: override prefill chunk count; unset=auto",
    ),
    "QWEN9B_GDN_DBG": ("<unset>", "experimental (Qwen3.5-9B) GDN debug print switch"),
    "QWEN9B_MLP_DOWN_AUTO": ("0", "experimental (9B): force ttnn auto-config for the MLP down-proj matmul"),
    "QWEN9B_MLP_UP_AUTO": ("0", "experimental (9B): force ttnn auto-config for the MLP up-proj matmul"),
    "QWEN9B_SDPA_QK64": ("0", "experimental (9B): use qk_head_dim=64 for SDPA instead of 128"),
    "QWEN_BATCHED_GROUPED": ("1", "TP batched serving: use grouped single-pass prefill for T<=256; 0=per-user"),
    "QWEN_GDN_DIAG_ALPHA": (
        "0.25",
        "experimental fused-GDN: diagonal-inverse mixing coefficient (only used when QWEN_GDN_INV_DOUBLING is set)",
    ),
    "QWEN_GDN_FLAT_GB": (
        "0",
        "fused/phased FLA op reads g/beta flat [B,T,HV] (no host permute+reshape); 1=enable. "
        "run_bench_e2e_p150.sh pins it from BENCH_GDN_FLAT_GB (runner default 1)",
    ),
    "QWEN_GDN_FP32_STATE": ("0", "GDN recurrent-state dtype for the experimental fused path; 1=fp32"),
    "QWEN_GDN_INV_DOUBLING": ("0", "experimental fused-GDN: use doubling-based matrix inversion; 0=default"),
    "QWEN_GDN_PATH": (
        "<unset>",
        "model-side only since the PR #57440 port (the C++ op ignores it): fused => QWEN36_GDN_FLA_INPUTS_DRAM "
        "defaults to 1; the FLA path/geometry comes from QWEN36_GDN_PCFG",
    ),
    "QWEN_SDPA_BF8": ("0", "experimental gated-attention: store SDPA KV in bfp8; 1=enable"),
}

# Flags that must NEVER be exported with a placeholder value (see the note above the table): a
# bare `if os.environ.get(X):` check, or a fallback computed from other locals rather than a fixed
# literal. run_bench_e2e_p150.sh leaves these truly unset rather than exporting their "default".
UNSAFE_TO_EXPORT_UNSET = {name for name, (default, _meaning) in ALL_QWEN_FLAG_DEFAULTS.items() if default == "<unset>"}


def _conv_kda_path_counts():
    """INT-3: Python-side call counts of the KDA conv paths (tiled / row_major / native_fallback)."""
    from models.demos.blackhole.qwen36.tt.gdn import conv1d_kda

    return conv1d_kda.path_counts()


def _header_flag_names():
    """QWEN36_*/QWEN_GDN_* subset reported in the bench header (per spec)."""
    return sorted(n for n in ALL_QWEN_FLAG_DEFAULTS if n.startswith("QWEN36_") or n.startswith("QWEN_GDN_"))


TRACE_ALLOC_ENV_VARS = (
    "TT_METAL_TRACE_ALLOC_TRACKING",
    "TT_METAL_TRACE_ALLOC_TRACEBACKS",
    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
)


def _check_trace_guard_env():
    """QWEN36_TRACE_GUARD=1 (T7 trace guard): make sure Metal's trace-allocation tracker will be ON.

    Metal reads TT_METAL_TRACE_ALLOC_TRACKING once (first query; ttnn queries it at import) and
    enables it only when the value starts with '1'. So this must run BEFORE `import ttnn`. Like
    text_demo.py, a missing value is defaulted to "1"; an explicit other value fails. Returns the
    header record {"guard", "env", "env_source"}; main() adds "tracking_effective" after the import
    (text_demo._check_trace_guard asserts Metal's cached snapshot)."""
    guard = os.environ.get("QWEN36_TRACE_GUARD", "0") == "1"
    source = "caller" if "TT_METAL_TRACE_ALLOC_TRACKING" in os.environ else "unset"
    if guard:
        if source == "unset":
            # Defaulting only helps before the first ttnn import (a wrapper that imported ttnn first
            # must export the tracker env itself; main() then re-checks Metal's effective setting).
            assert "ttnn" not in sys.modules, (
                "QWEN36_TRACE_GUARD=1 but TT_METAL_TRACE_ALLOC_TRACKING is unset and ttnn is already "
                "imported, so the tracker is off. Export TT_METAL_TRACE_ALLOC_TRACKING=1 before starting python."
            )
            os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] = "1"
            source = "bench default (QWEN36_TRACE_GUARD=1)"
        val = os.environ["TT_METAL_TRACE_ALLOC_TRACKING"]
        assert val.startswith("1"), (
            f"QWEN36_TRACE_GUARD=1 but TT_METAL_TRACE_ALLOC_TRACKING={val!r}: Metal would leave the "
            "trace-allocation tracker off (it needs a value starting with '1')"
        )
    env = {name: os.environ.get(name, "<unset>") for name in TRACE_ALLOC_ENV_VARS}
    return {"guard": guard, "env": env, "env_source": source}


def _fused_fla_available(repo_root):
    """True iff this tree's chunk_gated_delta_rule.cpp mentions ChunkGdnFusedProgramConfig (PR #57440
    fused prim wired in; the program config, QWEN36_GDN_PCFG, selects it -- not the removed env knobs)."""
    p = repo_root / "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp"
    try:
        return "ChunkGdnFusedProgramConfig" in p.read_text()
    except OSError:
        return False


def _git_info(repo_root):
    def _run(args):
        try:
            out = subprocess.run(args, cwd=str(repo_root), capture_output=True, text=True, timeout=10)
            return out.stdout.strip()
        except Exception as e:
            return f"<error: {e}>"

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["git", "rev-parse", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    dirty_files = len(porcelain.splitlines()) if porcelain and not porcelain.startswith("<error") else 0
    return {"branch": branch, "commit": commit, "dirty_files": dirty_files}


def _tt_smi_summary():
    """Parse a few fields out of `tt-smi -s` JSON. Returns None if tt-smi is missing/unparseable."""
    exe = shutil.which("tt-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-s"], capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout)
        dev0 = (data.get("device_info") or [{}])[0]
        return {
            "board_type": (dev0.get("board_info") or {}).get("board_type", "N/A"),
            "arc_fw": (dev0.get("firmwares") or {}).get("arc_fw", "N/A"),
            "aiclk": (dev0.get("telemetry") or {}).get("aiclk", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


def _sha256_ids(token_ids):
    """Deterministic hash of a [1, N] long tensor's token ids (list-of-int string, dtype/endian-safe)."""
    return hashlib.sha256(str(token_ids.flatten().tolist()).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Legitimate (non-tiled) corpus prompt: a real document, cut to an exact length, wrapped with a
# real instruction a person can judge the model's answer against.
# ---------------------------------------------------------------------------------------------
CORPUS_INSTRUCTION = "Summarize the text above in a few bullet points."
CORPUS_SYS_MSG = "You are a helpful assistant."
# Same URL/mechanism as text_demo._load_and_cache_context's Frankenstein entries
# (sample_prompts/eval_frankenstein_long.json[0]["context"]) -- reusing its own cache-file naming
# (md5 of the URL) so this script picks up a cache the demo itself already populated, or -- since
# this script never touches the network on its own during this repo's authoring/CI use -- falls
# back gracefully when there is none.
FRANKENSTEIN_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"


_GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _strip_gutenberg_boilerplate(text):
    """Trim the Project Gutenberg license header/footer, keeping only the actual novel text
    between its standard START/END markers -- both more legitimate (a person asked to judge a
    summary of "the text above" should see the novel, not license boilerplate) and, empirically,
    necessary: searching for an exact-isl cut point WITH the boilerplate included failed for
    several isl values tested (e.g. isl=8000 has no reachable sentence/word/character cut in that
    version at all, verified directly) that all succeed once the boilerplate is removed (different
    text changes which token counts are reachable at a nearby cut). Returns `text` unchanged if
    the markers are not found (e.g. the non-Gutenberg fallback document)."""
    try:
        s = text.index(_GUTENBERG_START)
        s = text.index("\n", s) + 1
        e = text.index(_GUTENBERG_END)
        return text[s:e]
    except ValueError:
        return text


def _load_source_document(sample_prompts_dir):
    """The real source document used to build a legitimate corpus prompt.

    Prefers the SAME long-context source text_demo._load_and_cache_context downloads/caches for
    its Frankenstein evals (sample_prompts/eval_frankenstein_long.json's first entry) -- reusing
    its cache file if this tree already has one (same md5-of-URL cache filename), or downloading
    it fresh via the same URL/mechanism, then trims it to just the novel text (see
    _strip_gutenberg_boilerplate). This document is long enough (~99k tokens once trimmed) to hit
    ANY --isl in this benchmark's range at a real cut point, unlike the short static "Nk" prompt
    files, which is why it -- not e.g. sample_prompts/input_data_long_4k.json -- is the base
    document for the general (arbitrary --isl) corpus mode.

    Falls back to the longest single prompt-text file under sample_prompts/ if the Frankenstein
    corpus is neither cached nor downloadable (e.g. no network) -- and says which file.

    Returns (text, human-readable description of where it came from).
    """
    cache_dir = Path(sample_prompts_dir) / ".context_cache"
    cache_file = cache_dir / hashlib.md5(FRANKENSTEIN_URL.encode()).hexdigest()
    if cache_file.exists() and cache_file.stat().st_size > 50_000:
        return (
            _strip_gutenberg_boilerplate(cache_file.read_text(errors="replace")),
            f"text_demo.py's own cached Frankenstein corpus ({cache_file}, from {FRANKENSTEIN_URL}), "
            f"license boilerplate trimmed",
        )
    try:
        import requests

        resp = requests.get(FRANKENSTEIN_URL, timeout=20)
        resp.raise_for_status()
        cache_dir.mkdir(exist_ok=True)
        cache_file.write_text(resp.text)
        return (
            _strip_gutenberg_boilerplate(resp.text),
            f"downloaded text_demo.py's Frankenstein corpus ({FRANKENSTEIN_URL}), license boilerplate trimmed",
        )
    except Exception as e:
        candidates = []
        for p in sorted(Path(sample_prompts_dir).glob("*.json")):
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list) and data and isinstance(data[0].get("prompt"), str):
                    candidates.append((len(data[0]["prompt"]), p, data[0]["prompt"]))
            except Exception:
                continue
        if not candidates:
            raise RuntimeError(
                f"no source document available: text_demo.py's Frankenstein corpus is not cached "
                f"and could not be downloaded ({e}), and no fallback prompt-text file was found "
                f"under {sample_prompts_dir}"
            ) from e
        candidates.sort(reverse=True)
        _len, path, text = candidates[0]
        return (
            text,
            f"FALLBACK (Frankenstein corpus unavailable: {e}): longest demo prompt-text file "
            f"under {sample_prompts_dir}: {path.name} ({_len} chars)",
        )


def _wrap_with_instruction(tokenizer, doc_text):
    """The demo's own chat-template mechanism (text_demo._get_prompt's QWEN35_REF_PROMPT branch:
    apply_chat_template with a system message + a user message of document+instruction,
    add_generation_prompt=True) -- reused verbatim, with CORPUS_INSTRUCTION in place of that
    branch's task-specific instruction."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CORPUS_SYS_MSG},
            {"role": "user", "content": doc_text + "\n\n" + CORPUS_INSTRUCTION},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )


def _token_count_for_cut(tokenizer, doc, cut):
    text = _wrap_with_instruction(tokenizer, doc[:cut].rstrip())
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _binary_search_boundary(tokenizer, doc, boundaries, isl):
    """boundaries: sorted increasing char cut positions. Token count is non-decreasing as the
    prefix is extended by whole sentences/words (never decreasing -- more real text in front of
    the same fixed instruction/template suffix cannot tokenize to fewer tokens), so a standard
    binary search for the leftmost boundary whose count is >= isl is valid at this granularity.
    Returns (exact_cut_or_None, (lo_idx, hi_idx)) -- the tightest bracket straddling isl when no
    boundary at this granularity lands exactly on it."""
    lo, hi = 0, len(boundaries) - 1
    lo_v = _token_count_for_cut(tokenizer, doc, boundaries[lo])
    hi_v = _token_count_for_cut(tokenizer, doc, boundaries[hi])
    if isl < lo_v or isl > hi_v:
        return None, (lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        v = _token_count_for_cut(tokenizer, doc, boundaries[mid])
        if v >= isl:
            hi = mid
        else:
            lo = mid + 1
    if _token_count_for_cut(tokenizer, doc, boundaries[lo]) == isl:
        return boundaries[lo], None
    return None, (max(0, lo - 1), lo)


def _word_then_char_search(tokenizer, doc, char_lo, char_hi, isl):
    """Within [char_lo, char_hi]: WORD-boundary binary search, then an exhaustive character-by-
    character scan of the final bracket (BPE retokenization right at a cut is not monotonic
    character-by-character -- verified: counts can wobble down and back up across a handful of
    characters -- so this tier scans rather than bisects; the bracket is tiny by construction).
    Returns (cut_or_None, boundary_type_or_None)."""
    segment = doc[char_lo:char_hi]
    word_ends = [char_lo + m.end() for m in re.finditer(r"\S+\s*", segment)]
    if not word_ends or word_ends[-1] != char_hi:
        word_ends.append(char_hi)
    if word_ends[0] != char_lo:
        word_ends.insert(0, char_lo)
    cut, bracket = _binary_search_boundary(tokenizer, doc, word_ends, isl)
    if cut is not None:
        return cut, "word"
    lo_j, hi_j = bracket
    c_lo, c_hi = word_ends[lo_j], word_ends[hi_j]
    for c in range(c_lo, c_hi + 1):
        if _token_count_for_cut(tokenizer, doc, c) == isl:
            return c, "character (mid-word fallback: no sentence or word boundary lands exactly on --isl here)"
    return None, None


def _exact_isl_legitimate_corpus_prompt(tokenizer, isl, sample_prompts_dir, max_widen=10):
    """EXACT --isl-token prompt built from a REAL document -- no tiling/repetition. Cuts the
    demo's own Frankenstein corpus (or its fallback, see _load_source_document) at a SENTENCE
    boundary chosen by binary search so the full chat-templated prompt (system message + the
    document up to that cut + "\n\n" + CORPUS_INSTRUCTION + generation prompt) is exactly --isl
    tokens; falls back to a WORD boundary (then a character-level scan -- see
    _word_then_char_search) within the bracketing sentence if no sentence boundary lands exactly
    on isl.

    Verified empirically against the (boilerplate-trimmed) Frankenstein corpus: sentence
    boundaries essentially never land exactly on an arbitrary isl (checked directly for a couple
    dozen isl values spanning 50-80000), and a WORD boundary alone also sometimes misses -- a
    single word can be >1 BPE token, so the cumulative count can jump clean over the target (e.g.
    4095 -> 4097, skipping 4096). Since an exact --isl is a hard requirement everywhere else in
    this benchmark (and is NOT relaxed for corpus mode), this adds fallback tiers the base
    instructions did not spell out: character-level scan within the word bracket, and -- if even
    that narrow bracket has no exact hit (a real but rare occurrence: verified some isl values,
    e.g. very close to the document's start or its max capacity, have NO reachable cut in a
    single sentence at all) -- widening to include up to `max_widen` additional sentences on each
    side before giving up. In stress-testing across isl in {50..80000} this combination resolved
    all but the most extreme edge cases (isl within roughly the fixed overhead's distance of 0 or
    of the document's max capacity), which raise a clear, actionable error instead of silently
    falling back to repeated text. This is always real, uninterrupted (mostly whole-sentence, with
    some pull of a further -- but still uninterrupted for the accepted span -- appended sentences
    when widened) document text; it may occasionally cut mid-word, which is reported via
    `boundary_type` alongside "sentence"/"word" (see the printed/JSON boundary_type).

    Returns (token_ids [1, isl], doc_text_included, boundary_type, source_description).
    """
    doc, source_desc = _load_source_document(sample_prompts_dir)

    overhead = _token_count_for_cut(tokenizer, doc, 0)
    max_isl = _token_count_for_cut(tokenizer, doc, len(doc))
    if isl < overhead:
        raise ValueError(
            f"--isl={isl} is smaller than the fixed chat-template+instruction overhead "
            f"({overhead} tokens) -- cannot build a legitimate corpus prompt this short from any "
            f"document. Use a larger --isl or --prompt-mode synthetic."
        )
    if isl > max_isl:
        raise ValueError(
            f"--isl={isl} exceeds the source document's max token capacity ({max_isl} tokens; "
            f"{source_desc}). Use a smaller --isl or --prompt-mode synthetic."
        )

    sentence_ends = [m.end() for m in re.finditer(r'[.!?]["\')]*\s+', doc)]
    if not sentence_ends or sentence_ends[-1] != len(doc):
        sentence_ends.append(len(doc))
    if sentence_ends[0] != 0:
        sentence_ends.insert(0, 0)

    cut, bracket = _binary_search_boundary(tokenizer, doc, sentence_ends, isl)
    boundary_type = "sentence"

    if cut is None:
        lo_i, hi_i = bracket
        for w in range(0, max_widen + 1):
            char_lo = sentence_ends[max(0, lo_i - w)]
            char_hi = sentence_ends[min(len(sentence_ends) - 1, hi_i + w)]
            cut, boundary_type = _word_then_char_search(tokenizer, doc, char_lo, char_hi, isl)
            if cut is not None:
                if w > 0:
                    boundary_type += f" (widened search: +/-{w} sentences)"
                break
        if cut is None:
            raise RuntimeError(
                f"could not hit --isl={isl} exactly at a sentence, word, or character boundary "
                f"even widening +/-{max_widen} sentences around the natural cut point in the "
                f"source document ({source_desc}). This isl value happens to be structurally "
                f"unreachable in this document without repeating text -- try --isl +/- a few "
                f"tokens, or --prompt-mode synthetic."
            )

    doc_included = doc[:cut].rstrip()
    text = _wrap_with_instruction(tokenizer, doc_included)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    assert ids.shape[1] == isl, f"internal error: built {ids.shape[1]} tokens, expected {isl}"
    return ids, doc_included, boundary_type, source_desc


def _demo_prompt_unchanged(tokenizer, get_prompt_fn):
    """The demo's OWN traced_4k prompt, byte-for-byte: text_demo._get_prompt(4096, tokenizer,
    max_prompt_len=None) -- exactly the call the traced_4k pytest case makes. This is the static
    sample_prompts/input_data_long_4k.json text, which tokenizes to 2642 tokens (_get_prompt
    clips, never pads, so the "4k" bucket name does not mean 4096 actual tokens). --isl is
    IGNORED for prompt construction in this mode: the point of --demo-prompt is byte-for-byte
    comparability with a `pytest text_demo.py -k traced_4k` run, not a configurable length."""
    return get_prompt_fn(4096, tokenizer, max_prompt_len=None)


def _exact_isl_synthetic_prompt(isl, seed=0):
    """Deterministic EXACT-isl synthetic prompt (uniform token ids in [1, 1000), matching the
    style used by ttft_chunked_one.py's random prompts). Local Generator so this never touches
    global torch RNG state."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 1000, (1, isl), dtype=torch.long, generator=g)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--isl", type=int, default=4096, help="exact input token count (default 4096)")
    p.add_argument("--osl", type=int, default=8, help="tokens to generate, including the first (default 8)")
    p.add_argument("--runs", type=int, default=5, help="timed requests after warmup (default 5)")
    p.add_argument("--warmup", type=int, default=1, help="warmup requests; the first captures the traces (default 1)")
    p.add_argument("--chunk", type=int, default=2048, help="prefill chunk size (default 2048)")
    p.add_argument("--out", type=str, required=True, help="output JSON path")
    p.add_argument(
        "--prompt-mode",
        choices=["corpus", "synthetic"],
        default="corpus",
        help="corpus (default): a REAL document (the demo's own Frankenstein long-context "
        "corpus) cut at a sentence/word boundary + a summarization instruction, so the total "
        "chat-templated prompt is exactly --isl tokens -- no tiling/repetition; see "
        "_exact_isl_legitimate_corpus_prompt. synthetic: fallback only -- deterministic random "
        "token ids, not a prompt a person can judge the output of.",
    )
    p.add_argument(
        "--demo-prompt",
        action="store_true",
        help="use the demo's own traced_4k prompt unchanged (text_demo._get_prompt(4096, ...), "
        "which tokenizes to 2642 tokens) for direct comparability with text_demo.py; overrides "
        "--prompt-mode and ignores --isl for prompt construction (--isl still sizes the paged KV "
        "cache budget unless it is smaller than 4096, in which case 4096 is used instead so the "
        "2642-token prompt always fits).",
    )
    p.add_argument("--check-ref", type=str, default=None, help="optional JSON with {'expected_first_tokens': [...]}")
    return p.parse_args()


def main():
    args = parse_args()
    if args.warmup < 1:
        print(
            "[WARN] --warmup < 1: the decode trace is primed on the first request regardless of "
            "whether it is a warmup or timed request, so its trace-capture cost would leak into "
            "a timed measurement. Use --warmup >= 1 for numbers comparable to the demo.",
            file=sys.stderr,
        )

    # Resolve user-supplied paths against the ORIGINAL cwd before chdir'ing to the repo root below
    # (text_demo._get_prompt/_load_and_cache_context use paths relative to the repo root, matching
    # how the demo itself is always run -- so this script matches that convention rather than
    # inventing its own).
    out_path_abs = Path(args.out).resolve()
    check_ref_abs = Path(args.check_ref).resolve() if args.check_ref else None

    repo_root = REPO_ROOT
    os.chdir(repo_root)
    git_info = _git_info(repo_root)
    fused_fla_available = _fused_fla_available(repo_root)
    tt_smi = _tt_smi_summary()
    py_ver = platform.python_version()

    header_flags = {}
    for name in _header_flag_names():
        default, _meaning = ALL_QWEN_FLAG_DEFAULTS[name]
        header_flags[name] = os.environ.get(name, f"<unset, documented default={default}>")

    print("=" * 78)
    print("Qwen3.5-2B P150 end-to-end benchmark")
    print(f"  git branch={git_info['branch']} commit={git_info['commit']} dirty_files={git_info['dirty_files']}")
    print(f"  TT_METAL_HOME={os.environ.get('TT_METAL_HOME')}")
    print(f"  python={py_ver}")
    print(f"  HF_MODEL={os.environ.get('HF_MODEL')} MESH_DEVICE={os.environ.get('MESH_DEVICE')}")
    print(
        f"  fused_fla_available={fused_fla_available} (ChunkGdnFusedProgramConfig in the op) "
        f"QWEN36_GDN_PCFG={os.environ.get('QWEN36_GDN_PCFG', '<unset>')}"
    )
    if tt_smi:
        print(f"  tt-smi: {tt_smi}")
    else:
        print("  tt-smi: not available (binary not found on PATH)")
    print(f"  isl={args.isl} osl={args.osl} chunk={args.chunk} warmup={args.warmup} runs={args.runs}")
    ondev_argmax = os.environ.get("QWEN36_ONDEV_ARGMAX", "1") == "1"
    print(f"  ondev_argmax={ondev_argmax} (QWEN36_ONDEV_ARGMAX={os.environ.get('QWEN36_ONDEV_ARGMAX', '<unset>')})")
    # Trace guard self-check, part 1 (before ttnn is imported: Metal reads the tracker env once, at the
    # first query, and ttnn queries it at import). With the tracker on, Metal does NOT log "Allocating
    # device buffers is potentially unsafe"; execute_trace raises instead, so the check is the env.
    trace_guard = _check_trace_guard_env()
    print(f"  trace_guard={trace_guard['guard']} tracker_env={trace_guard['env']}")
    gdn_decode_fused = os.environ.get("QWEN36_GDN_DECODE_FUSED", "2")  # = tt/gdn/decode_fused.py default
    gdn_conv_repack = os.environ.get("QWEN36_GDN_CONV_REPACK", "perlayer")
    print(
        f"  gdn_decode_fused={gdn_decode_fused} gdn_conv_repack={gdn_conv_repack} (QWEN36_GDN_DECODE_FUSED=2 = fused op)"
    )
    gdn_conv_kda_tiled = os.environ.get("QWEN36_GDN_CONV_KDA_TILED", "1")  # = tt/gdn/conv1d_kda.py default
    print(f"  gdn_conv_kda_tiled={gdn_conv_kda_tiled} (QWEN36_GDN_CONV_KDA_TILED=1 = tiled KDA conv, no glue ops)")
    print("=" * 78)

    # ---- heavy / device-related imports (after env is settled) ----
    from transformers import AutoTokenizer

    import ttnn
    from models.demos.blackhole.qwen36.demo.text_demo import (
        BLOCK_SIZE,
        SAMPLE_PROMPTS_DIR,
        _blocks_for,
        _check_trace_guard,
        _get_prompt,
        _should_use_chunked_trace,
        _warmup_prefill,
    )

    # Trace guard self-check, part 2: Metal's cached snapshot says the tracker is really on.
    _check_trace_guard()
    from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

    trace_guard["tracking_effective"] = bool(TRACE_ALLOC_TRACKING)
    print(f"  trace_alloc_tracking_effective={trace_guard['tracking_effective']}")
    from models.demos.blackhole.qwen36.tt import tp_common as _tp_common
    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.tt_transformers.tt.generator import Generator
    from tests.scripts.common import get_updated_device_params

    ttnn_ver = getattr(ttnn, "__version__", "unknown")
    print(f"  ttnn={ttnn_ver}")

    module_defaults = {
        "tp_common.PREFILL_MM_PACKER_L1_ACC": _tp_common.PREFILL_MM_PACKER_L1_ACC,
        "tp_common.PREFILL_MM_FP32_ACC": _tp_common.PREFILL_MM_FP32_ACC,
        "tp_common.PREFILL_MINIMAL_CFG": _tp_common.PREFILL_MINIMAL_CFG,
    }
    print(f"  module defaults (evaluated at import time): {module_defaults}")
    # I-1 item flags, effective values (QWEN36_I1_<item>, default in tp_common.I1_FLAG_DEFAULTS).
    i1_flags = {item: _tp_common.i1_enabled(item) for item in _tp_common.I1_FLAG_DEFAULTS}
    print(f"  i1_flags (effective): {i1_flags}")
    # I-2 item flags, effective raw values (QWEN36_I2_<item>, default in tp_common.I2_FLAG_DEFAULTS).
    i2_flags = {item: _tp_common.i2_value(item) for item in _tp_common.I2_FLAG_DEFAULTS}
    print(f"  i2_flags (effective): {i2_flags}")
    # M1 item flags, effective raw values (QWEN36_M1_<item>, default in tp_common.M1_FLAG_DEFAULTS).
    m1_flags = {item: _tp_common.m1_value(item) for item in _tp_common.M1_FLAG_DEFAULTS}
    print(f"  m1_flags (effective): {m1_flags}")
    # M2 item flags, effective raw values (QWEN36_M2_<item>, default in tp_common.M2_FLAG_DEFAULTS).
    m2_flags = {item: _tp_common.m2_value(item) for item in _tp_common.M2_FLAG_DEFAULTS}
    print(f"  m2_flags (effective): {m2_flags}")
    # M3 item flags, effective raw values (QWEN36_M3_<item>, default in tp_common.M3_FLAG_DEFAULTS).
    m3_flags = {item: _tp_common.m3_value(item) for item in _tp_common.M3_FLAG_DEFAULTS}
    print(f"  m3_flags (effective): {m3_flags}")
    # C2 item flags, effective raw values (QWEN36_C2_<item>, default in tp_common.C2_FLAG_DEFAULTS).
    c2_flags = {item: _tp_common.c2_value(item) for item in _tp_common.C2_FLAG_DEFAULTS}
    print(f"  c2_flags (effective): {c2_flags}")
    # R3 item flags, effective raw values (QWEN36_R3_<item>, default in tp_common.R3_FLAG_DEFAULTS).
    r3_flags = {item: _tp_common.r3_value(item) for item in _tp_common.R3_FLAG_DEFAULTS}
    print(f"  r3_flags (effective): {r3_flags}")
    # M4 item flags, effective raw values (QWEN36_M4_<item>, default in tp_common.M4_FLAG_DEFAULTS).
    m4_flags = {item: _tp_common.m4_value(item) for item in _tp_common.M4_FLAG_DEFAULTS}
    print(f"  m4_flags (effective): {m4_flags}")
    # M5 item flags, effective raw values (QWEN36_M5_<item>, default in tp_common.M5_FLAG_DEFAULTS).
    m5_flags = {item: _tp_common.m5_value(item) for item in _tp_common.M5_FLAG_DEFAULTS}
    print(f"  m5_flags (effective): {m5_flags}")
    # INT-4 SDPA flags, effective raw values (defaults in ttnn_gated_attention.py) + the flexible q_chunk.
    from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_attention as _ga

    i4_flags = {
        "SDPA_Q64": _ga.i4_sdpa_q64_value(),
        "SDPA_EXP_COMPAT": _ga.i4_sdpa_exp_compat_value(),
        "flexible_q_chunk": _ga.flexible_sdpa_q_chunk(),
        "sdpa_op_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_METAL_SDPA_")},
    }
    print(f"  i4_flags (effective): {i4_flags}")
    # I-3 item flags, effective raw values (QWEN36_I3_<item>, default in tp_common.I3_FLAG_DEFAULTS).
    i3_flags = {item: _tp_common.i3_value(item) for item in _tp_common.I3_FLAG_DEFAULTS}
    print(f"  i3_flags (effective): {i3_flags}")
    # F item flags, effective raw values (QWEN36_F_<item>, default in tp_common.F_FLAG_DEFAULTS).
    f_flags = {item: _tp_common.f_value(item) for item in _tp_common.F_FLAG_DEFAULTS}
    print(f"  f_flags (effective): {f_flags}")
    # R5 item flags, effective raw values (QWEN36_R5_<item>, default in tp_common.R5_FLAG_DEFAULTS).
    r5_flags = {item: _tp_common.r5_value(item) for item in _tp_common.R5_FLAG_DEFAULTS}
    print(f"  r5_flags (effective): {r5_flags}")
    # MM item flags, effective raw values (QWEN36_MM_<item>, default in tp_common.MM_FLAG_DEFAULTS).
    mm_flags = {item: _tp_common.mm_value(item) for item in _tp_common.MM_FLAG_DEFAULTS}
    print(f"  mm_flags (effective): {mm_flags}")

    # --demo-prompt ignores --isl for prompt content (it's always the demo's 2642-token traced_4k
    # prompt) but still needs a KV-cache budget big enough to hold it -- size against
    # max(args.isl, 4096) so a small --isl can't undersize the cache for that fixed-length prompt.
    sizing_isl = max(args.isl, 4096) if args.demo_prompt else args.isl
    num_blocks = _blocks_for(sizing_isl, args.osl)
    max_seq_len = num_blocks * BLOCK_SIZE

    device_params = get_updated_device_params({"l1_small_size": 24576, "num_command_queues": 2})
    device = ttnn.CreateDevice(device_id=0, **device_params)
    ttnn.SetDefaultDevice(device)
    device.enable_program_cache()

    result = {"ok": False}
    try:
        model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len)
        # Before any trace is captured: the token ops compile in the warmup passes.
        model.set_greedy_token_output(ondev_argmax)
        # T8: GDN layers that really run the fused decode op (0 when QWEN36_GDN_DECODE_FUSED=0 or unsupported).
        gdn_fused_layers = sum(
            1 for l in model.layers if not l.is_full_attention and getattr(l.attention, "_decode_fused", False)
        )
        print(f"  gdn_decode_fused_layers={gdn_fused_layers} (QWEN36_GDN_DECODE_FUSED={gdn_decode_fused})")
        tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

        boundary_type = None
        source_desc = None
        doc_included = None
        if args.demo_prompt:
            prompt_mode_used = "demo-prompt"
            token_ids = _demo_prompt_unchanged(tokenizer, _get_prompt)
            source_desc = f"{SAMPLE_PROMPTS_DIR}/input_data_long_4k.json (text_demo._get_prompt(4096, ...), unchanged)"
        elif args.prompt_mode == "corpus":
            prompt_mode_used = "corpus"
            token_ids, doc_included, boundary_type, source_desc = _exact_isl_legitimate_corpus_prompt(
                tokenizer, args.isl, SAMPLE_PROMPTS_DIR
            )
        else:
            prompt_mode_used = "synthetic"
            token_ids = _exact_isl_synthetic_prompt(args.isl)

        actual_len = token_ids.shape[1]
        if not args.demo_prompt:
            assert actual_len == args.isl, f"prompt construction bug: got {actual_len} tokens, expected {args.isl}"
        prompt_sha256 = _sha256_ids(token_ids)
        print(f"  prompt: {actual_len} tokens (mode={prompt_mode_used}) sha256={prompt_sha256}")
        if source_desc:
            print(f"  prompt source: {source_desc}")
        if boundary_type:
            print(f"  prompt cut boundary type: {boundary_type}")
        if doc_included:
            print(f"  document as included, last 200 chars: {doc_included[-200:]!r}")

        chunk_size = args.chunk

        # Legacy (non-chunked) compile-only warmup -- matches text_demo.py exactly: skipped for
        # short prompts, where the masked-bucket path compiles during capture below.
        if actual_len >= chunk_size:
            _warmup_prefill(model, device, token_ids)

        # Paged KV cache + DeltaNet state (allocate AFTER _warmup_prefill, exactly as text_demo.py's
        # _run_traced_generation does: _warmup_prefill runs before any KV cache exists).
        num_kv_heads = model.args.n_kv_heads
        head_dim = model.args.head_dim
        kv_cache_shape = [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
        model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
        page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

        bucket_size = ((actual_len + chunk_size - 1) // chunk_size) * chunk_size
        pad_len = bucket_size - actual_len
        last_token = token_ids[:, -1:].expand(1, pad_len) if pad_len > 0 else token_ids[:, :0]
        padded_token_ids = torch.cat([token_ids, last_token], dim=1)

        assert _should_use_chunked_trace(model), "chunk-seq GDN prefill must be enabled"
        t_cap0 = time.perf_counter()
        gen = Generator([model], [model.args], device)
        state = {"decode_primed": False}
        # Parked-trace-safe order (T7, 2026-09-25): (1) prepare = every prefill program and persistent
        # buffer, incl. the per-request eager programs; (2) prime the decode trace while NO trace is
        # parked, so all decode kernel binaries / constants / trace inputs exist first; (3) capture the
        # prefill trace. The old order (capture prefill, then prime decode inside request 0) put decode
        # kernel binaries in the parked prefill trace's scratch; the next prefill replay corrupted them
        # and the decode trace hung (layout-dependent; see TT_METAL_TRACE_ALLOC_TRACKING).
        model.prepare_prefill_trace_chunked(
            device, page_table, chunk_size=chunk_size, warmup_masked_buckets=True, max_prompt_len=bucket_size
        )
        # Dummy decode input: the last prompt token at position actual_len. prime_decode_trace restores
        # the GDN state it advances, and the first real decode step rewrites this KV slot.
        prime_decode_trace(gen, model, token_ids[:, -1:].to(torch.long), torch.tensor([actual_len]), page_table)
        state["decode_primed"] = True
        model.capture_prefill_trace_chunked(device, page_table, chunk_size=chunk_size, prepared=True)
        capture_s = time.perf_counter() - t_cap0
        print(
            f"  prefill prepare + decode-trace prime + prefill trace capture in {capture_s:.3f}s "
            "(this is NOT counted in TTFT)"
        )

        def run_one_request():
            t0 = time.perf_counter()
            if actual_len < chunk_size:
                logits = model.prefill_masked_bucket(token_ids, page_table, actual_len=actual_len)
            else:
                logits = model.prefill_traced_chunked(padded_token_ids, page_table, actual_len=actual_len)
            if ondev_argmax:
                next_token = int(ttnn.to_torch(logits).reshape(-1)[0])
                assert 0 <= next_token < model.vocab_size, f"prefill token {next_token} out of range"
            else:
                logits_torch = ttnn.to_torch(logits).squeeze()
                assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
                next_token = int(logits_torch.argmax().item())
            ttft_s = time.perf_counter() - t0

            if not state["decode_primed"]:
                prime_decode_trace(
                    gen, model, torch.tensor([[next_token]], dtype=torch.long), torch.tensor([actual_len]), page_table
                )
                state["decode_primed"] = True

            generated = [next_token]
            decode_times_s = []
            current_pos = actual_len
            for i in range(args.osl - 1):
                t_step = time.perf_counter()
                out = gen.decode_forward(
                    torch.tensor([[next_token]], dtype=torch.long),
                    torch.tensor([current_pos]),
                    page_table=page_table,
                    kv_cache=None,
                    enable_trace=True,
                    read_from_device=True,
                )
                v = out[0] if isinstance(out, tuple) else out
                if ondev_argmax:
                    next_token = int(v.reshape(-1)[0])
                    assert 0 <= next_token < model.vocab_size, f"decode token {next_token} out of range at step {i}"
                else:
                    dl = v.squeeze().float()
                    assert not torch.isnan(dl).any(), f"NaN in decode at step {i}"
                    next_token = int(dl.argmax())
                decode_times_s.append(time.perf_counter() - t_step)
                generated.append(next_token)
                current_pos += 1

            e2e_s = time.perf_counter() - t0
            tpot_s = (sum(decode_times_s) / len(decode_times_s)) if decode_times_s else float("nan")
            return {
                "generated": generated,
                "ttft_s": ttft_s,
                "decode_times_s": decode_times_s,
                "tpot_s": tpot_s,
                "e2e_s": e2e_s,
            }

        warmup_results = []
        for i in range(args.warmup):
            r = run_one_request()
            warmup_results.append(r)
            print(
                f"[RUN] phase=warmup idx={i} ttft_s={r['ttft_s']:.4f} tpot_s={r['tpot_s']:.4f} "
                f"e2e_s={r['e2e_s']:.4f} tokens={r['generated']}"
            )

        timed_results = []
        for i in range(args.runs):
            r = run_one_request()
            timed_results.append(r)
            print(
                f"[RUN] phase=timed idx={i} ttft_s={r['ttft_s']:.4f} tpot_s={r['tpot_s']:.4f} "
                f"e2e_s={r['e2e_s']:.4f} tokens={r['generated']}"
            )

        # ---- correctness checks ----
        ok = True
        ref_generated = warmup_results[-1]["generated"] if warmup_results else None
        for i, r in enumerate(timed_results):
            match = r["generated"] == ref_generated
            if not match:
                ok = False
            print(f"[CHECK] timed run {i} vs warmup: {'PASS' if match else 'FAIL'}")

        check_ref_result = None
        if check_ref_abs:
            with open(check_ref_abs) as f:
                ref = json.load(f)
            expected = ref.get("expected_first_tokens") or ref.get("generated") or ref.get("tokens")
            if expected is None:
                print(
                    "[CHECK] --check-ref given but no 'expected_first_tokens'/'generated'/'tokens' "
                    "key found; skipping",
                    file=sys.stderr,
                )
            else:
                n = len(expected)
                check_ref_result = []
                for i, r in enumerate(timed_results):
                    got = r["generated"][:n]
                    match = got == list(expected)
                    if not match:
                        ok = False
                    check_ref_result.append({"run": i, "match": match})
                    print(f"[CHECK] timed run {i} vs --check-ref ({n} tokens): {'PASS' if match else 'FAIL'}")

        # ---- summary stats over TIMED runs ----
        def _stats(key):
            vals = [r[key] for r in timed_results]
            return {
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }

        ttft_stats = _stats("ttft_s")
        tpot_stats = _stats("tpot_s")
        e2e_stats = _stats("e2e_s")
        e2e_model = ttft_stats["median"] + (args.osl - 1) * tpot_stats["median"]

        print("=" * 78)
        print(
            f"TTFT (s): median={ttft_stats['median']:.4f} min={ttft_stats['min']:.4f} "
            f"max={ttft_stats['max']:.4f} stdev={ttft_stats['stdev']:.4f}"
        )
        print(
            f"TPOT (s): median={tpot_stats['median']:.4f} min={tpot_stats['min']:.4f} "
            f"max={tpot_stats['max']:.4f} stdev={tpot_stats['stdev']:.4f}"
        )
        print(
            f"E2E  (s): median={e2e_stats['median']:.4f} min={e2e_stats['min']:.4f} "
            f"max={e2e_stats['max']:.4f} stdev={e2e_stats['stdev']:.4f}"
        )
        print(f"E2E_model (s) = TTFT_median + (osl-1)*TPOT_median = {e2e_model:.4f}")
        print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
        print("=" * 78)

        result = {
            "ok": ok,
            "header": {
                "git": git_info,
                "tt_metal_home": os.environ.get("TT_METAL_HOME"),
                "python_version": py_ver,
                "ttnn_version": ttnn_ver,
                "hf_model": os.environ.get("HF_MODEL"),
                "mesh_device": os.environ.get("MESH_DEVICE"),
                "qwen_env_flags": header_flags,
                "module_defaults": module_defaults,
                "tt_smi": tt_smi,
                "isl": args.isl,
                "osl": args.osl,
                "chunk": args.chunk,
                "warmup": args.warmup,
                "runs": args.runs,
                "prompt_mode": prompt_mode_used,
                "prompt_actual_len": actual_len,
                "prompt_sha256": prompt_sha256,
                "prompt_source": source_desc,
                "prompt_boundary_type": boundary_type,
                "prompt_doc_last_200_chars": doc_included[-200:] if doc_included else None,
                "fused_fla_available": fused_fla_available,
                "ondev_argmax": ondev_argmax,
                "i1_flags": i1_flags,
                "i2_flags": i2_flags,
                "m1_flags": m1_flags,
                "m2_flags": m2_flags,
                "m3_flags": m3_flags,
                "c2_flags": c2_flags,
                "m4_flags": m4_flags,
                "m5_flags": m5_flags,
                "i4_flags": i4_flags,
                "i3_flags": i3_flags,
                "f_flags": f_flags,
                "r5_flags": r5_flags,
                "mm_flags": mm_flags,
                "trace_guard": trace_guard,
                "gdn_decode_fused": gdn_decode_fused,
                "gdn_decode_fused_layers": gdn_fused_layers,
                "gdn_conv_repack": gdn_conv_repack,
                "gdn_conv_kda_tiled": gdn_conv_kda_tiled,
                "gdn_conv_kda_path_counts": _conv_kda_path_counts(),
                "capture_s": capture_s,
            },
            "warmup_results": warmup_results,
            "timed_results": timed_results,
            "checks": {
                "timed_vs_warmup": [r["generated"] == ref_generated for r in timed_results],
                "check_ref": check_ref_result,
            },
            "summary": {
                "ttft_s": ttft_stats,
                "tpot_s": tpot_stats,
                "e2e_s": e2e_stats,
                "e2e_model_s": e2e_model,
            },
        }
    finally:
        ttnn.close_device(device)

    out_path_abs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_abs, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path_abs}")

    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
