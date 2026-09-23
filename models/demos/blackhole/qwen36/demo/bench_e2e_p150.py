#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reproducible end-to-end (prefill + decode) benchmark for Qwen3.5-2B on ONE Blackhole P150.

Plain python (argparse, no pytest) so it can be run directly:

    python3 models/demos/blackhole/qwen36/demo/bench_e2e_p150.py --isl 4096 --osl 8 \
        --out models/demos/blackhole/qwen36/demo/bench_results/run.json

Prefer the wrapper models/demos/blackhole/qwen36/demo/run_bench_e2e_p150.sh, which pins every
QWEN36_*/QWEN_GDN_*/QWEN_* env flag explicitly first (see REPRODUCE_P150_PERF.md).

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

Determinism check: sampling is greedy argmax on host (temperature 0), so every timed run must
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
    "QWEN36_GDN_CONV_LEGACY": ("0", "force the legacy (pre-native) conv1d implementation"),
    "QWEN36_GDN_CONV_SILU_SHARDED": ("0", "run the conv+SiLU on a sharded memory config; 1=enable"),
    "QWEN36_GDN_CONV_T3_MAX": ("0", "cap on the conv kernel's T3 tiling dimension; 0=no cap"),
    "QWEN36_GDN_CONV_TILED_SPLIT": ("0", "split the conv1d input into tiles; 1=enable"),
    "QWEN36_GDN_CONV_XIN_L1_MAX_T": (
        "<unset>",
        "max token count for L1-resident conv input; unset=model-computed default (xin_l1_max_t)",
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
    "QWEN36_GDN_POST_L1": ("1", "keep GDN post-scan tensors in L1; 0=DRAM"),
    "QWEN36_GDN_POST_L1_OUTPROJ": ("1", "keep the GDN output-projection input in L1; 0=DRAM"),
    "QWEN36_GDN_POST_L1_SCAN": ("1", "keep the GDN scan output in L1; 0=DRAM"),
    "QWEN36_GDN_SPLIT_PROJ": ("1", "split the GDN QKVBA projection matmul; 0=single matmul"),
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
    "QWEN_GDN_FP32_STATE": ("0", "GDN recurrent-state dtype for the experimental fused path; 1=fp32"),
    "QWEN_GDN_INV_DOUBLING": ("0", "experimental fused-GDN: use doubling-based matrix inversion; 0=default"),
    "QWEN_GDN_PATH": (
        "<unset>",
        "select the GDN prefill implementation; fused=experimental fused FLA prim (needs the fused-prim build), unset/other=phased chunk-parallel (this branch, F12)",
    ),
    "QWEN_SDPA_BF8": ("0", "experimental gated-attention: store SDPA KV in bfp8; 1=enable"),
}

# Flags that must NEVER be exported with a placeholder value (see the note above the table): a
# bare `if os.environ.get(X):` check, or a fallback computed from other locals rather than a fixed
# literal. run_bench_e2e_p150.sh leaves these truly unset rather than exporting their "default".
UNSAFE_TO_EXPORT_UNSET = {name for name, (default, _meaning) in ALL_QWEN_FLAG_DEFAULTS.items() if default == "<unset>"}


def _header_flag_names():
    """QWEN36_*/QWEN_GDN_* subset reported in the bench header (per spec)."""
    return sorted(n for n in ALL_QWEN_FLAG_DEFAULTS if n.startswith("QWEN36_") or n.startswith("QWEN_GDN_"))


def _fused_fla_available(repo_root):
    """True iff this tree's chunk_gated_delta_rule.cpp mentions QWEN_GDN_PATH (fused prim wired in)."""
    candidates = [
        repo_root / "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp",
        repo_root
        / "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gated_delta_rule.cpp",
    ]
    for p in candidates:
        try:
            if "QWEN_GDN_PATH" in p.read_text():
                return True
        except OSError:
            continue
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
    print(f"  fused_fla_available={fused_fla_available} (QWEN_GDN_PATH=fused can take effect only if True)")
    if tt_smi:
        print(f"  tt-smi: {tt_smi}")
    else:
        print("  tt-smi: not available (binary not found on PATH)")
    print(f"  isl={args.isl} osl={args.osl} chunk={args.chunk} warmup={args.warmup} runs={args.runs}")
    print("=" * 78)

    # ---- heavy / device-related imports (after env is settled) ----
    from transformers import AutoTokenizer

    import ttnn
    from models.demos.blackhole.qwen36.demo.text_demo import (
        BLOCK_SIZE,
        SAMPLE_PROMPTS_DIR,
        _blocks_for,
        _get_prompt,
        _should_use_chunked_trace,
        _warmup_prefill,
    )
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
        model.capture_prefill_trace_chunked(device, page_table, chunk_size=chunk_size, warmup_masked_buckets=True)
        capture_s = time.perf_counter() - t_cap0
        print(f"  prefill trace captured in {capture_s:.3f}s (this is NOT counted in TTFT)")

        gen = Generator([model], [model.args], device)
        state = {"decode_primed": False}

        def run_one_request():
            t0 = time.perf_counter()
            if actual_len < chunk_size:
                logits = model.prefill_masked_bucket(token_ids, page_table, actual_len=actual_len)
            else:
                logits = model.prefill_traced_chunked(padded_token_ids, page_table, actual_len=actual_len)
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
                dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
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
