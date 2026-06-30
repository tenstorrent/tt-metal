#!/usr/bin/env bash
# Qwen3.6-27B BH Galaxy text demo (batch-1) — llama-style switch-to-decode path.
# Run from the tt-metal repo root:  bash models/demos/qwen3_6_galaxy_v2/demo/run_text_demo.sh
set -euo pipefail

export TT_METAL_HOME="$(pwd)"
export PYTHONPATH="$(pwd)"
source python_env/bin/activate

# ---- model / mesh ----
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=BH_GLX

# ---- correctness-critical: switch-to-decode (llama-style) path ----
export QWEN36_FORCE_SWITCH_DECODE=1   # decode-mode tt_ccl + decode tail
export QWEN36_DECODE_L1_RESIDUAL=1    # 32-row L1 residual (decode rms_allgather norm requires it)
export QWEN36_RESIDUAL_BUF_BF16=1     # bf16 residual buffer (paired with L1_RESIDUAL)
export QWEN36_LM_HEAD_PLAIN_DECODE=1  # FIX: decode lm_head ring->minimal_matmul (else garbage output)

# ---- prefill correctness at ISL 128 ----
export QWEN36_SEQ_CORES_PER_HEAD=4    # clears the ISL-128 seq-prefill L1 CB clash on HEAD

# ---- perf tuning (standard) ----
export QWEN36_FULLATTN_WO_TUNED=1
export QWEN36_DELTA_OP_TUNED=1
export QWEN36_CCL_NUM_LINKS_DELTA=2

# ---- ISL select (default 128). For long context (>=16k incl 256k) ADD:
#   export QWEN36_PERF_T_PREFILL=262144 QWEN36_PREFILL_CHUNK=4096
#   (256k coherence validated with default GDN chunk_size=128.)
export QWEN36_PERF_T_PREFILL="${QWEN36_PERF_T_PREFILL:-128}"

python -m pytest --noconftest \
    models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py -v -s
