#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Device verification for HF sliding-layer key retention in denoise (#51080 roadmap item 3).
#
# HF's sliding layers retain only the last ``sliding_window - 1`` = 1023 committed tokens; TT
# currently attends the whole committed prefix on all 30 layers. DG_DENOISE_SLIDING_WINDOW=1
# enforces the HF rule on the 25 sliding layers.
#
# The change has a sharp regime split, and the gate is deliberately ONE-SIDED:
#
#   committed prefix P at block k = cache_len + 256*(k-1)
#     P <= W-1  -> nothing has been evicted yet, the sliding mask is IDENTICAL to today's
#                  -> those blocks MUST be bit-identical (per_block_sha256). HARD GATE.
#     P >= W    -> keys are evicted, so the mask differs. Committed tokens MAY differ, but
#                  equality is NOT a failure: at the real W=1024 only P-(W-1) keys are evicted
#                  (33 of 1312 attended columns at P=1056 = 2.5%), which is a real fidelity
#                  difference yet far too small to reliably flip a 262144-way argmax.
#
# Asserting "bound blocks must differ" would conflate "the mask changed" with "the decisions
# changed" — a mistake this script made on its first run. The mask change itself is verified
# directly and host-side in tests/test_denoise_sliding_window.py.
#
# To prove the plumbing is live END-TO-END (rather than merely not crashing), set
# DG_DENOISE_SLIDING_WINDOW_OVERRIDE to something small: at W=128 and P=1056 the window evicts
# 929 of 1056 committed keys (~88%), so the output MUST move. This script forwards the override
# to both runs and uses it in the regime arithmetic.
#
# The >=W regime is decision-CHANGING. Its ACCEPTANCE gate is a decision-agreement run against
# fp32 HF (NOT against today's TT output, which is the defect being corrected).

# Gumbel source: `device`. `--upfront` needs a MATERIALIZED full-vocabulary Gumbel and `device` is
# now the only one: the `host` mode this script used was DELETED on 2026-07-28 after being measured
# NOT to be the TT language-drift cause (it drifts on exactly the same prompts as `device`, repairs
# 0, and costs 1.40x per request; the real cause was the canvas attending prefill pad keys, fixed in
# d0936d4da4f).
# Consequence for this A/B: committed_sha256 values recorded from earlier `host`-mode runs will NOT
# reproduce here, and absolute latencies are ~1.94x lower than any host-Gumbel figure in
# doc/optimize_perf/winter_borrow_20260727.md. Compare the two arms WITHIN one invocation only.

set -uo pipefail

TT_METAL_ROOT="${TT_METAL_ROOT:-/home/zni/tt-metal}"
MODEL_VENV="${MODEL_VENV:-/home/zni/venvs/tt-diffusion-gemma}"
DG_CKPT="${DG_CKPT:-/home/zni/dg_models/diffusiongemma-26B-A4B-it}"
MESH="${MESH:-P150x4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
NUM_BLOCKS="${NUM_BLOCKS:-5}"
TRACE_REGION_SIZE="${TRACE_REGION_SIZE:-4294967296}" # 4 GiB. Measured: the 48 up-front traces fit at reveal_pmax=4096 (3.04 GiB used, 3 GiB fails) AND in the same 4 GiB at reveal_pmax=16384 (48/48 captured, MeshTraceId 0..47, run local100_trace4g 2026-07-29), so trace memory does NOT scale with reveal_pmax -- the "Scale this WITH reveal_pmax" note this line used to carry was wrong. Over-reserving is not free: the region is DRAM nothing else can allocate, and 6->4 GiB returns exactly 2.0 GiB to the pool, which is what keeps the per-request leak from OOMing the engine at max_model_len 16384. Undersizing fails loudly at capture. See doc/optimize_perf/bisect_trace_region.sh
OUT_DIR="${OUT_DIR:-/tmp/dg_sliding_window}"

mkdir -p "$OUT_DIR"
PY="${MODEL_VENV}/bin/python"

echo "DG_SLIDING_WINDOW_AB_BEGIN blocks=${NUM_BLOCKS} max_seq_len=${MAX_SEQ_LEN}"

for sw in 0 1; do
    echo "--- DG_DENOISE_SLIDING_WINDOW=${sw}"
    env \
        TT_METAL_HOME="${TT_METAL_ROOT}" \
        PYTHONPATH="${TT_METAL_ROOT}" \
        MESH_DEVICE="${MESH}" \
        DG_TRACE_REGION_SIZE="${TRACE_REGION_SIZE}" \
        DG_DENOISE_SLIDING_WINDOW="${sw}" \
        DG_DENOISE_SLIDING_WINDOW_OVERRIDE="${DG_DENOISE_SLIDING_WINDOW_OVERRIDE:-}" \
        "${PY}" -u -m models.experimental.diffusion_gemma.demo.serving_smoke \
        --checkpoint "${DG_CKPT}" --mesh "${MESH}" \
        --max-seq-len "${MAX_SEQ_LEN}" --num-blocks "${NUM_BLOCKS}" \
        --gumbel-mode device --disable-eos-stop --local-files-only \
        --upfront --reveal-pmax "${MAX_SEQ_LEN}" \
        --metrics-json "${OUT_DIR}/sw_${sw}.json" \
        >"${OUT_DIR}/sw_${sw}.log" 2>&1
    echo "DG_DENOISE_SLIDING_WINDOW=${sw} rc=$?"
done

echo
echo "=== verdict ==="
"${PY}" - "$OUT_DIR" <<'PYEOF'
import json, os, pathlib, sys

out = pathlib.Path(sys.argv[1])
WINDOW = 1024
runs = {}
for sw in ("0", "1"):
    meta, log = out / f"sw_{sw}.json", out / f"sw_{sw}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    if "DG_VLLM_SERVING_SMOKE_SUCCESS" not in text or not meta.exists():
        print(f"  !! DG_DENOISE_SLIDING_WINDOW={sw} did not complete")
        print("DG_SLIDING_WINDOW_AB_RESULT FAIL")
        raise SystemExit(0)
    runs[sw] = json.loads(meta.read_text())

off, on = runs["0"], runs["1"]
cache_len = off["cache_len"]
canvas = off["canvas_length"]
sha_off, sha_on = off["per_block_sha256"], on["per_block_sha256"]
print(f"  cache_len={cache_len} canvas={canvas} blocks off/on={len(sha_off)}/{len(sha_on)}")

WINDOW = int(os.environ.get("DG_DENOISE_SLIDING_WINDOW_OVERRIDE", "") or WINDOW)

# The GATE is one-sided on purpose. Blocks whose committed prefix is still inside the window
# MUST be bit-identical: nothing has been evicted, so the mask is unchanged and any difference
# is a bug. Blocks past the window MAY differ — but equality there is NOT a failure. The real
# window evicts only P-(W-1) keys (33 of 1312 attended columns at P=1056, i.e. 2.5%), which is
# a genuine fidelity difference yet far too small to reliably flip a 262144-way argmax. Asserting
# "bound blocks must differ" conflates "the mask changed" with "the decisions changed"; the mask
# change is verified directly (host-side) in tests/test_denoise_sliding_window.py.
ok = True
bound_differ = []
if len(sha_off) != len(sha_on):
    print("  !! different block counts -- not comparable")
    ok = False
else:
    for k, (a, b) in enumerate(zip(sha_off, sha_on), start=1):
        p = cache_len + canvas * (k - 1)          # committed prefix this block denoised at
        binds = p > WINDOW - 1
        same = a == b
        if not binds:
            verdict = "OK (must be identical)" if same else "*** REGRESSION: unbound block changed ***"
            if not same:
                ok = False
        else:
            verdict = "differs (fidelity fix active)" if not same else "identical (decisions stable)"
            bound_differ.append(not same)
        print(f"  block {k}: P={p:5d} window_binds={binds!s:5} identical={same!s:5}  {verdict}")

if not bound_differ:
    print(f"  NOTE: no block reached P>={WINDOW}, so only the bit-exact regime was exercised. "
          "Raise --num-blocks or lower DG_DENOISE_SLIDING_WINDOW_OVERRIDE to cross the window.")
else:
    n = sum(bound_differ)
    print(f"  bound blocks that changed committed tokens: {n}/{len(bound_differ)} "
          f"({'plumbing proven live' if n else 'decisions stable at this window size'})")
print("DG_SLIDING_WINDOW_AB_RESULT " + ("UNBOUND_BLOCKS_BIT_IDENTICAL" if ok else "FAIL"))
PYEOF
