#!/bin/bash
# ND experiment runner. Usage: run_nd.sh <tag> [extra_env...]
#   tag: short name for the log
# Runs the fast prefill repro (5 layers, 1024 tokens, iter5) with the ND probe.
# iter5 gives within-process (iter0..4) AND a clean iter0 baseline for
# across-process comparison. Extra env (e.g. TT_DS_ND_COMBINE_INIT_ZEROS=1) is
# prefixed verbatim.
set -u
TAG="$1"; shift || true
EXTRA="$*"
SB=/data/kgrujcic/tt-metal-sandbox
TS=$(date +%H%M%S)
LOG=/data/kgrujcic/nd_logs/${TAG}_${TS}.log
echo "LOG=$LOG"
echo "EXTRA_ENV=$EXTRA"

# Guard: do not run while anyone else's pytest / soak holds the 32-chip galaxy.
OTHER=$(ps -eo user,cmd | grep -E "pytest|test_prefill|run_iter200" | grep -v grep | grep -v "$USER" | head -1)
if [ -n "$OTHER" ]; then
  echo "ABORT: another user's run holds the device: $OTHER"
  exit 3
fi
# Also abort if a galaxy reset is in flight.
if pgrep -f "glx_reset" >/dev/null 2>&1; then
  echo "ABORT: tt-smi glx_reset in flight."
  exit 3
fi

cd "$SB"
env $EXTRA \
TT_DS_ND_DEBUG=1 \
PYTHONPATH=$SB \
TT_METAL_HOME=$SB \
TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure \
TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden/ \
DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528 \
$SB/python_env/bin/python -m pytest \
  models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py -xvs \
  -k "mesh-8x4 and smoke and pretrained and 5_layers and iter5 and longbook_qa_eng and 1024 and fp32 and right_pad and balanced" \
  > "$LOG" 2>&1
echo "EXIT=$?"
echo "=== NDPROBE summary ($LOG) ==="
grep -aE "NDPROBE|VERDICT|FIRST DIVERGING|PASSED|FAILED|^E |Error" "$LOG" | tail -120
