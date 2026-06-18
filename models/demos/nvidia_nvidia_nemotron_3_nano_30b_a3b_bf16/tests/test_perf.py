# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Performance and full-model correctness test for NemotronH-30B on QB TP=4.

Two tests:
  1. test_full_52layer_pcc — all 52 layers hidden-state PCC vs pure-PyTorch reference
  2. test_decode_latency   — B=1 S=1 decode, 2 warm-up + 5 timed runs; reports ms/tok

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=ttnn:tools:.
    export LD_LIBRARY_PATH=build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
    pytest models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_perf.py -v -s
"""

import os
import sys
import time

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch

import ttnn

PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
N_LAYERS = 52
PCC_THRESHOLD = 0.99
N_WARMUP = 2
N_TIMED = 5


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a -= a.mean()
    b -= b.mean()
    denom = torch.sqrt((a**2).sum() * (b**2).sum())
    return ((a * b).sum() / denom).item() if denom.item() != 0.0 else 1.0


@pytest.fixture(scope="module")
def mesh_device():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    dev = open_device_tp4()
    yield dev
    close_device_tp4(dev)


@pytest.fixture(scope="module")
def weight_cache():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache

    return WeightCache()


def _ref_forward_52(input_ids: torch.Tensor, wc) -> torch.Tensor:
    """Pure-PyTorch reference for all 52 layers (hidden states, no LM head)."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.reference.functional import (
        dense_attention,
        layer_norm,
        mamba2_layer,
        moe_experts,
        moe_gate,
        shared_expert,
    )

    B, S = input_ids.shape
    h = torch.nn.functional.embedding(input_ids, wc["backbone.embeddings.weight"])

    for li in range(N_LAYERS):
        lt = PATTERN[li]
        p = f"backbone.layers.{li}"

        if lt == "M":
            h = mamba2_layer(
                hidden_states=h,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
            )
        elif lt == "E":
            residual = h
            normed = layer_norm(h, wc[f"{p}.norm.weight"])
            flat = normed.reshape(B * S, -1)
            topk_idx, topk_wts = moe_gate(
                flat,
                wc[f"{p}.mixer.gate.weight"],
                wc[f"{p}.mixer.gate.e_score_correction_bias"],
            )
            eu = [wc[f"{p}.mixer.experts.{e}.up_proj.weight"] for e in range(128)]
            ed = [wc[f"{p}.mixer.experts.{e}.down_proj.weight"] for e in range(128)]
            ex_out = moe_experts(flat, topk_idx, topk_wts, eu, ed).reshape(B, S, -1)
            sh_out = shared_expert(
                normed,
                wc[f"{p}.mixer.shared_experts.up_proj.weight"],
                wc[f"{p}.mixer.shared_experts.down_proj.weight"],
            )
            h = (residual + ex_out + sh_out).bfloat16()
        else:  # '*'
            h = dense_attention(
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )
    return h


def test_full_52layer_pcc(mesh_device, weight_cache):
    """Hidden-state PCC for all 52 layers (TTNN TP=4 vs pure-PyTorch reference)."""

    torch.manual_seed(42)
    input_ids = torch.randint(0, 131072, (1, 1), dtype=torch.long)  # S=1: Mamba2 decode kernel

    print(f"\nRunning reference forward (52 layers)...")
    ref_h = _ref_forward_52(input_ids, weight_cache)

    print(f"Running TTNN TP=4 forward (52 layers, hidden states)...")
    wc = weight_cache
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.dense_attention import dense_attention_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.embedding import embedding_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_layer import mamba2_layer_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import _moe_layer_forward

    B, S = input_ids.shape
    h = embedding_forward(mesh_device, input_ids, wc["backbone.embeddings.weight"])
    for li in range(N_LAYERS):
        lt = PATTERN[li]
        p = f"backbone.layers.{li}"
        if lt == "M":
            h, _ = mamba2_layer_forward(
                mesh_device,
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
            )
        elif lt == "E":
            h = _moe_layer_forward(mesh_device, h, li, wc)
        else:
            h = dense_attention_forward(
                mesh_device,
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )

    # h is ttnn.Tensor — bring to CPU for PCC comparison
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    B = input_ids.shape[0]
    h_cpu = _host_rep(h, mesh_device, B)

    score = pcc(h_cpu, ref_h)
    print(f"\nFull-model hidden-state PCC (52 layers): {score:.6f}")
    assert score >= PCC_THRESHOLD, f"PCC {score:.6f} < {PCC_THRESHOLD}"


def test_decode_latency(mesh_device, weight_cache):
    """Decode latency: B=1, S=1 through all 52 layers + LM head on TP=4."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward

    torch.manual_seed(7)
    input_ids = torch.randint(0, 131072, (1, 1), dtype=torch.long)

    print(f"\nWarming up ({N_WARMUP} runs, B=1 S=1)...")
    for _ in range(N_WARMUP):
        _ = nemotron_h_forward(mesh_device, input_ids, wc=weight_cache)

    print(f"Timing {N_TIMED} runs...")
    latencies = []
    for i in range(N_TIMED):
        t0 = time.perf_counter()
        logits = nemotron_h_forward(mesh_device, input_ids, wc=weight_cache)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        print(f"  run {i+1}: {latencies[-1]:.1f} ms  (logits shape {logits.shape})")

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    toks_per_sec = 1000.0 / avg_ms

    print(f"\nDecode latency B=1 S=1 (52 layers + LM head, TP=4):")
    print(f"  avg: {avg_ms:.1f} ms  min: {min_ms:.1f} ms  throughput: {toks_per_sec:.2f} tok/s")

    # Sanity: result is finite, correct shape
    assert logits.shape == (1, 1, 131072)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"


@pytest.mark.timeout(0)
def test_decode_traced(mesh_device, weight_cache):
    """Traced decode: B=1, S=1 through all 52 layers + LM head on TP=4.

    Captures a TTNN trace on a fixed token, then times N_TIMED replay
    executions.  MoE routing is deterministic for a fixed input so the
    trace correctly represents one complete decode step.
    """
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_device
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    N_TRACED = 10

    torch.manual_seed(7)
    token_cpu = torch.randint(0, 131072, (1, 1), dtype=torch.int32)

    # Pre-allocate the persistent device input tensor (trace input).
    ids_tt = ttnn.from_torch(
        token_cpu,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    wc = weight_cache

    print("\nTrace warm-up (1 eager run to compile device kernels)...")
    _ = nemotron_h_forward_device(mesh_device, ids_tt, wc)
    ttnn.synchronize_device(mesh_device)

    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    logits_tt = nemotron_h_forward_device(mesh_device, ids_tt, wc)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    print("Trace captured.")

    # Verify shape before timing loop.
    logits_cpu = _host_rep(logits_tt, mesh_device, 1)
    assert logits_cpu.shape == (1, 1, 131072), f"unexpected logits shape {logits_cpu.shape}"
    assert torch.isfinite(logits_cpu).all(), "logits contain NaN/Inf after trace capture"

    print(f"Timing {N_TRACED} traced executions (same token)...")
    latencies = []
    ids_host = ttnn.from_torch(
        token_cpu,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    for i in range(N_TRACED):
        ttnn.copy_host_to_device_tensor(ids_host, ids_tt)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        print(f"  run {i + 1}: {latencies[-1]:.1f} ms")

    ttnn.release_trace(mesh_device, trace_id)

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    print(f"\nTraced decode B=1 S=1 (52 layers + LM head, TP=4):")
    print(f"  avg: {avg_ms:.1f} ms  min: {min_ms:.1f} ms  throughput: {1000 / avg_ms:.2f} tok/s")


@pytest.mark.timeout(0)
def test_prefill_mamba2_latency(mesh_device, weight_cache):
    """Mamba2 chunked SSD prefill latency sweep across sequence lengths.

    Tests one Mamba2 layer (layer 0, 'M' in PATTERN) for S in SEQ_LENGTHS.
    Reports: total ms, ms/tok, projected 23-layer time, speedup vs S * 45 ms sequential.

    Sequential decode baseline: ~45 ms/tok/layer measured on QB TP=4.
    Prefill wins because S parallel tokens → one kernel launch instead of S launches.
    """
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_layer import mamba2_layer_forward_dispatch
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _upload

    SEQUENTIAL_MS_PER_TOK = 45.0  # per-layer decode baseline, QB TP=4
    N_MAMBA_LAYERS = 23
    SEQ_LENGTHS = [128, 256, 512, 1024, 4096]

    wc = weight_cache
    li = 0  # layer 0 is 'M' (first Mamba2 layer)
    prefix = f"backbone.layers.{li}.mixer"

    nw = wc[f"backbone.layers.{li}.norm.weight"]
    ipw = wc[f"{prefix}.in_proj.weight"]
    cw = wc[f"{prefix}.conv1d.weight"]
    cb = wc[f"{prefix}.conv1d.bias"]
    dtb = wc[f"{prefix}.dt_bias"]
    alog = wc[f"{prefix}.A_log"]
    nmw = wc[f"{prefix}.norm.weight"]
    D = wc[f"{prefix}.D"]
    opw = wc[f"{prefix}.out_proj.weight"]

    print(f"\nMamba2 chunked SSD prefill — single layer, TP=4 QB")
    print(f"{'S':>8}  {'total_ms':>10}  {'ms/tok':>8}  {'speedup vs seq':>16}")
    print(f"{'-'*8}  {'-'*10}  {'-'*8}  {'-'*16}")

    results = {}
    for S in SEQ_LENGTHS:
        torch.manual_seed(42)
        x_cpu = torch.randn(1, S, 2688, dtype=torch.bfloat16)
        x_tt = _upload(x_cpu, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        # 2 warm-up runs (compile kernels, fill caches)
        for _ in range(2):
            _out, _, _ = mamba2_layer_forward_dispatch(mesh_device, x_tt, nw, ipw, cw, cb, dtb, alog, nmw, D, opw)
            ttnn.synchronize_device(mesh_device)

        # 3 timed runs, average
        t0 = time.perf_counter()
        for _ in range(3):
            _out, _, _ = mamba2_layer_forward_dispatch(mesh_device, x_tt, nw, ipw, cw, cb, dtb, alog, nmw, D, opw)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()

        total_ms = (t1 - t0) * 1000.0 / 3.0
        ms_per_tok = total_ms / S
        seq_ms = S * SEQUENTIAL_MS_PER_TOK
        speedup = seq_ms / total_ms

        results[S] = total_ms
        print(f"{S:>8}  {total_ms:>10.1f}  {ms_per_tok:>8.3f}  {speedup:>15.1f}x")

    # Project to all 23 Mamba2 layers (linear scaling assumption)
    print(f"\n  Projected 23-layer prefill (1-layer × 23):")
    print(f"{'S':>8}  {'prefill_s':>10}  {'seq_s':>10}  {'speedup':>10}")
    for S in SEQ_LENGTHS:
        proj_ms = results[S] * N_MAMBA_LAYERS
        seq_s = S * SEQUENTIAL_MS_PER_TOK * N_MAMBA_LAYERS / 1000.0
        print(f"{S:>8}  {proj_ms/1000:>10.2f}  {seq_s:>10.1f}  {seq_s/(proj_ms/1000):>9.1f}x")


@pytest.mark.timeout(0)
def test_isl_sweep_ttft_coherency(mesh_device, weight_cache):
    """Full ISL sweep from 128 to model limit (262K): TTFT, prefill ms/tok, decode tok/s.

    Covers the complete model context range in doubling steps — identical in
    spirit to gpt_oss perf sweeps.  Uses MODEL_MAX_SEQ_LEN=262144 for the KV
    cache and each generate() call so all ISLs share one pre-allocated state.

    Seed token list is built by repeating a short phrase so any ISL up to 256K
    is covered without tokenizing a huge string.

    n_decode is clamped per-ISL so prompt+decode never exceeds 262144.
    Checks output coherency (non-empty, printable text) at every ISL.
    """
    from transformers import AutoTokenizer

    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import SNAP as _SNAP
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import generate
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.kv_cache import (
        DEFAULT_BLOCK_SIZE,
        MODEL_MAX_SEQ_LEN,
        allocate_decoder_state,
    )

    ISL_LIST = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262112]
    N_DECODE = 30

    tokenizer = AutoTokenizer.from_pretrained(_SNAP)

    # Build seed token list by repeating a short phrase.
    # Avoids tokenizing a huge string; round-trip-stable for BPE tokenizers.
    _seed = (
        "Tenstorrent Blackhole is an AI accelerator for hybrid SSM and MoE models. "
        "NemotronH-30B combines Mamba2 and dense-attention layers. "
        "The tensor-parallel TP=4 configuration runs on a four-chip QuietBox. "
    )
    _seed_ids_short = tokenizer.encode(_seed, add_special_tokens=False)
    needed = max(ISL_LIST) + 200
    repeats = (needed + len(_seed_ids_short) - 1) // len(_seed_ids_short)
    _seed_ids = (_seed_ids_short * repeats)[:needed]

    # KV-cache budget strategy:
    # ISLs ≤ 32K share one persistent state (96 MB KV per device); ISLs > 32K get a
    # per-ISL state sized for that context window.  At ISL=65536, pre-allocating
    # MODEL_MAX_SEQ_LEN (262K) costs 1.57 GB/device and leaves no DRAM for MoE
    # intermediates; a per-ISL state costs only 393 MB/device, giving ~1.2 GB back.
    _SMALL_ISL_MAX = 32_768
    print(f"[state] Pre-allocating decoder state (max_seq_len={_SMALL_ISL_MAX})...", flush=True)
    _persistent_state = allocate_decoder_state(
        mesh_device, B=1, max_seq_len=_SMALL_ISL_MAX, block_size=DEFAULT_BLOCK_SIZE
    )
    print("[state] Done.", flush=True)

    # Warmup: compile prefill kernels and decode trace at a short ISL.
    print("\n[warmup] Running warmup generate (ISL=32, 5 decode tokens)...", flush=True)
    _warmup_prompt = tokenizer.decode(_seed_ids[:32], skip_special_tokens=False)
    generate(
        prompt=_warmup_prompt,
        mesh_device=mesh_device,
        wc=weight_cache,
        max_new_tokens=5,
        max_seq_len=_SMALL_ISL_MAX,
        verbose=False,
        cpu_gate=False,
        decoder_state=_persistent_state,
    )
    print("[warmup] Done.", flush=True)

    results = []

    for isl in ISL_LIST:
        prompt_ids = _seed_ids[:isl]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # Clamp decode count: tokenizer round-trip may shift token count by ±few.
        n_decode = max(1, min(N_DECODE, MODEL_MAX_SEQ_LEN - isl - 10))

        # Select / allocate decoder state for this ISL.
        if isl <= _SMALL_ISL_MAX:
            _state = _persistent_state
            _state_max = _SMALL_ISL_MAX
        else:
            # Allocate just enough KV cache for this ISL (rounded to block boundary).
            # actual_tokens < isl always (BPE compression); actual + N_DECODE << isl.
            _isl_max = min(
                ((isl + DEFAULT_BLOCK_SIZE - 1) // DEFAULT_BLOCK_SIZE) * DEFAULT_BLOCK_SIZE,
                MODEL_MAX_SEQ_LEN,
            )
            print(f"[state] Allocating per-ISL decoder state (max_seq_len={_isl_max})...", flush=True)
            _state = allocate_decoder_state(mesh_device, B=1, max_seq_len=_isl_max, block_size=DEFAULT_BLOCK_SIZE)
            _state_max = _isl_max

        print(f"\n{'='*70}")
        print(f"ISL sweep: target ISL={isl}, n_decode={n_decode}")
        print(f"{'='*70}")

        text, metrics = generate(
            prompt=prompt,
            mesh_device=mesh_device,
            wc=weight_cache,
            max_new_tokens=n_decode,
            max_seq_len=_state_max,
            verbose=True,
            cpu_gate=False,
            return_metrics=True,
            decoder_state=_state,
        )

        metrics.print_summary()

        actual_isl = metrics.prompt_tokens
        prefill_s = metrics.prefill_compile_s + metrics.prefill_inference_s
        prefill_ms_per_tok = prefill_s * 1000.0 / max(actual_isl, 1)

        assert len(text) > 0, f"ISL={isl}: generate() returned empty text"
        assert metrics.generated_tokens > 0, f"ISL={isl}: no tokens were generated"
        printable_ratio = sum(1 for c in text if c.isprintable()) / max(len(text), 1)
        assert (
            printable_ratio > 0.9
        ), f"ISL={isl}: text has low printable ratio {printable_ratio:.2f} — possible NaN/garbage output"

        results.append(
            {
                "isl": actual_isl,
                "ttft_ms": metrics.ttft_s * 1000.0,
                "prefill_ms_per_tok": prefill_ms_per_tok,
                "decode_toks_s": metrics.decode_toks_s,
                "gen_tokens": metrics.generated_tokens,
            }
        )
        print(f"\n[ISL={isl}] tail: {repr(text[-120:])}")

        if isl > _SMALL_ISL_MAX:
            _state.free()  # explicitly free DRAM; del alone is non-deterministic (GC fragmentation)
            del _state

    # --- Summary table ---
    print("\n" + "=" * 78)
    print("ISL Sweep — NemotronH-30B QB TP=4 (bulk E-layer prefill, traced decode)")
    print(f"{'ISL':>8}  {'TTFT(ms)':>10}  {'prefill ms/tok':>15}  {'tok/s':>7}  {'gen_toks':>9}")
    print(f"{'-'*8}  {'-'*10}  {'-'*15}  {'-'*7}  {'-'*9}")
    for r in results:
        print(
            f"{r['isl']:>8}  {r['ttft_ms']:>10.0f}  {r['prefill_ms_per_tok']:>15.1f}  "
            f"{r['decode_toks_s']:>7.1f}  {r['gen_tokens']:>9}"
        )
    print("=" * 78)
