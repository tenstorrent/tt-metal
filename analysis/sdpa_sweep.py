# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TEN-4679 single-chip SDPA bottleneck sweep.

Drives the prefill SDPA kernel at multiple sequence lengths under tracy +
perf-counter profiling so we can answer:
  1. FPU% vs SFPU% of kernel cycles
  2. FPU stalled by SFPU% (scoreboard stall, derived) and vice-versa
  3. Single-cycle-exp counterfactual (post-processed analytically)
  4. Both-idle on NoC/L1% (cb_wait_front + L1 bandwidth)
  5. Sequence-length scaling

Run with tracy:
    cd $TT_METAL_HOME
    python -m tracy --profiler-capture-perf-counters=all \\
                    -p -- pytest analysis/sdpa_sweep.py::test_sweep -s

`--perf-counter-groups=all` is equivalent to `fpu,instrn,unpack,pack,l1_0`.
L1 bank 1 (NOC Ring 1) requires a separate run with `=fpu,instrn,unpack,pack,l1_1`
because L1 banks share the same hardware counters via a 3-bit mux selector.
"""

from __future__ import annotations

import os
from typing import List

import pytest


# ------------------------------------------------------------------
# Llama 3.1 8B SDPA shape, Black Hole single-chip, matching Bilal's
# "FPU Util on BH P100" report so we can cross-reference.
#   S × d_m × S, GQA 32 query / 8 KV heads, head_dim = 128
#   BF8 acts and weights, both DRAM-interleaved, causal.
# ------------------------------------------------------------------

LLAMA_8B_HEAD_DIM = 128
LLAMA_8B_QCHUNK = 128
LLAMA_8B_KCHUNK = 128

# Override with env: SDPA_SEQ_LENS="1024" or "1024,2048,...,32768".
_seq_env = os.environ.get("SDPA_SEQ_LENS", "1024,2048,4096,8192,16384,32768")
SEQ_LENS: List[int] = [int(x) for x in _seq_env.split(",") if x.strip()]

# nh / nkv ladder. First entry is the smoke test; full Llama is (32, 8).
# Per-core perf-counter % are intrinsic to the kernel, not the head count,
# so a smaller `nh` still answers the kernel-level questions.
NH_NKV_LADDER = [
    (4, 1),
    (8, 2),
    (16, 4),
    (32, 8),
]


def _run_sdpa_op_only(device, nh, nkv, s):
    """Run only the SDPA prefill op under profiling — no torch reference/PCC.

    Mirrors tests/.../test_sdpa_prefill.py::run_test_sdpa_tt's device-side setup
    exactly (production Llama config: HiFi2, exp_approx, BF8, DRAM-interleaved,
    is_causal=True) so the perf counters match, but skips the expensive CPU
    golden so large-S points are profileable.
    """
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.test_sdpa_prefill import fa_rand

    d = int(os.environ.get("SDPA_HEAD_DIM", "0")) or LLAMA_8B_HEAD_DIM
    # Verification / de-bias knobs (default = production Llama config). Vary ONE axis to
    # prove a doc claim, or vary q_chunk/dtype/head_dim to broaden the calibration grid.
    exp_approx = os.environ.get("SDPA_EXP_APPROX", "1") == "1"
    causal = os.environ.get("SDPA_CAUSAL", "1") == "1"
    qc = int(os.environ.get("SDPA_QCHUNK", str(LLAMA_8B_QCHUNK)))
    kc = int(os.environ.get("SDPA_KCHUNK", str(qc)))
    dtype = {"bfp8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}[os.environ.get("SDPA_DTYPE", "bfp8_b")]
    fidelity = {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi3": ttnn.MathFidelity.HiFi3,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }[os.environ.get("SDPA_FIDELITY", "HiFi2")]
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=exp_approx,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    Q = fa_rand(1, nh, s, d)
    K = fa_rand(1, nkv, s, d)
    V = fa_rand(1, nkv, s, d)
    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device, pad_value=0.0)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=causal, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    ttnn.synchronize_device(device)
    # touch output so the op isn't optimized away; do NOT pull full tensor to host for huge S
    _ = tt_back.shape
    tt_Q.deallocate()
    tt_K.deallocate()
    tt_V.deallocate()
    tt_back.deallocate()


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_sweep(device, seq_len):
    """Run SDPA prefill at one seq_len under whichever (nh, nkv) fits."""
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.test_sdpa_prefill import run_test_sdpa_tt

    if seq_len % LLAMA_8B_QCHUNK != 0 or seq_len % LLAMA_8B_KCHUNK != 0:
        pytest.skip(f"seq_len {seq_len} not divisible by chunk size")

    nh = int(os.environ.get("SDPA_NH", "0")) or None
    nkv = int(os.environ.get("SDPA_NKV", "0")) or None
    # SDPA_SKIP_REF=1 runs the SDPA op under profiling but skips the torch
    # golden + PCC check. The perf counters are independent of the reference,
    # so this lets us profile large S (>=8192) where the O(nh*S^2*d) CPU
    # reference is otherwise the wall-clock bottleneck (hours at nh=32).
    skip_ref = os.environ.get("SDPA_SKIP_REF", "0") == "1"

    candidates = [(nh, nkv)] if nh else NH_NKV_LADDER

    last_err = None
    for cand_nh, cand_nkv in candidates:
        if cand_nh is None or cand_nkv is None:
            continue
        try:
            print(f"\n[sdpa_sweep] S={seq_len}  nh={cand_nh}  nkv={cand_nkv}  skip_ref={skip_ref}", flush=True)
            if skip_ref:
                _run_sdpa_op_only(device, cand_nh, cand_nkv, seq_len)
            else:
                run_test_sdpa_tt(
                    device,
                    b=1,
                    nh=cand_nh,
                    nkv=cand_nkv,
                    s=seq_len,
                    d=LLAMA_8B_HEAD_DIM,
                    q_chunk_size=LLAMA_8B_QCHUNK,
                    k_chunk_size=LLAMA_8B_KCHUNK,
                    dtype=ttnn.bfloat8_b,
                    rmse_threshold=0.0092,
                )
            print(f"[sdpa_sweep] OK S={seq_len} nh={cand_nh} nkv={cand_nkv}", flush=True)
            return
        except Exception as e:
            last_err = e
            msg = str(e)
            print(f"[sdpa_sweep] FAIL S={seq_len} nh={cand_nh} nkv={cand_nkv}: {msg[:200]}", flush=True)
            if "OOM" in msg or "out of memory" in msg.lower() or "Allocator" in msg:
                continue
            raise
    raise AssertionError(f"No (nh,nkv) candidate worked for S={seq_len}; last: {last_err}")
