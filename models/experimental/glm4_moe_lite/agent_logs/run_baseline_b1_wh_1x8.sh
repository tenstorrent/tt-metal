#!/usr/bin/env bash
# Batch-1 ISL baseline on WH LoudBox (T3K) as a 1x8 (1D) mesh, PCC-passing TP=1 config.
#
# This drives debug_run_full_tt_greedy.py directly (not run_sweep_isl_batch.py, whose
# hard-set BH perf flags — FUSE_MLP_MOE_REDUCE etc. — deadlock the 1x8 collectives).
# The config below is the minimal set that RUNS and passes PCC on 1x8; aggressive perf
# flags are intentionally left off (they are the optimization work that comes next).
#
# Fixes baked in:
#   * dedicated wh_1x8 weight cache (the "d8" cache key can't tell 2x4 from 1x8; reusing
#     the 2x4 cache lands expert shards on a nonexistent MeshCoordinate([1,0]))
#   * chunked prefill (MAX_PREFILL_CHUNK_SIZE=128) — unchunked overflows L1 on the MoE gate
#   * head-parallel off (20 attn heads not divisible by tp=8)
#   * DECODE_MLA_CORE_SCALE=0 — pins FlashMLA to 16 cores so decode fits WH's 1.5MB L1 at
#     long context (core-scale jumps to 32 cores at ISL>=2048 and blows L1)
# NB: no `set -e`/`pipefail` — the per-run metric parsing greps may legitimately not match
# (e.g. a decode-OOM run has no "subsequent:" line) and must not abort the whole sweep.
set -u
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate

CACHE="$HOME/.cache/ttnn/models/glm4_moe_lite/wh_1x8"
OUT="models/experimental/glm4_moe_lite/experiments/baseline_b1_wh_1x8"
mkdir -p "$OUT"
CSV="$OUT/baseline_b1.csv"
SCRIPT="models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py"
ISLS=(${ISLS:-512 1024 2048 4096 8192 16384 32768 65536 131072})
MAX_NEW=${MAX_NEW:-16}

# --- baseline runtime config (accuracy/TP=1 + minimal L1-fit flags) ---
export GLM4_MOE_LITE_ENABLE_MOE=1 GLM4_MOE_LITE_EXPERTS_TT_DTYPE=bf8 GLM4_MOE_LITE_MOE_FP32_ACC=1
export GLM4_MOE_LITE_TP=1 GLM4_MOE_LITE_ATTN_DP=0
export GLM4_MOE_LITE_HEAD_PARALLEL_ATTN=0 GLM4_MOE_LITE_HEAD_PARALLEL_KVB2=0
export GLM4_MOE_LITE_CCL_NUM_LINKS=1 GLM4_MOE_LITE_CCL_TOPOLOGY=linear
export GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 TT_METAL_GTEST_ETH_DISPATCH=1
export GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE=128
export GLM4_MOE_LITE_DECODE_MLA_CORE_SCALE=0 GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1

echo "isl,batch,kv_dtype,warmup_prefill_s,prefill_s,first_token_ms,decode_mean_ms,decode_min_ms,decode_max_ms,per_user_tok_s,status" > "$CSV"

for ISL in "${ISLS[@]}"; do
  # bf8 KV at very long context to fit DRAM; bf16 otherwise (batch=1)
  if [ "$ISL" -ge 32768 ]; then KV=bf8; else KV=bf16; fi
  LOG="$OUT/run_isl${ISL}.log"
  echo "[baseline] ISL=$ISL kv=$KV ..." >&2
  set +e
  timeout 2400 python "$SCRIPT" \
    --prompt "Summarize the following document. " \
    --simulate-context-len "$ISL" --min-cache-tokens $((ISL + MAX_NEW)) \
    --max-new-tokens "$MAX_NEW" --batch-size 1 --mesh-rows 1 --mesh-cols 8 \
    --kv-cache-dtype "$KV" --phase both --enable-trace --trace-mode sampling --warmup \
    --cache-dir "$CACHE" > "$LOG" 2>&1
  rc=$?
  # Parse metrics in Python (robust to whitespace / grep-variant quirks).
  row=$(python - "$LOG" "$ISL" "$KV" "$rc" <<'PY'
import re, sys
log, isl, kv, rc = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
t = open(log, errors="replace").read()
def last(pat):
    m = re.findall(pat, t)
    return m[-1] if m else ""
wu   = last(r"warmup_prefill_s=([0-9.]+)")
pf   = last(r"prompt_len=%s prefill_s=([0-9.]+)" % re.escape(isl))
ft   = last(r"first token:\s+([0-9.]+)\s*ms")
sub  = re.findall(r"subsequent:\s+mean=\s*([0-9.]+)\s+min=\s*([0-9.]+)\s+max=\s*([0-9.]+)", t)
dm = dmin = dmax = pu = ""
if sub:
    dm, dmin, dmax = sub[-1]
    try: pu = f"{1000.0/float(dm):.3f}"
    except Exception: pu = ""
if "beyond max L1" in t: status = "decode_L1_oom"
elif re.search(r"out of memory|Allocat.*fail|OOM", t, re.I): status = "oom"
elif "ETH core heartbeat" in t: status = "eth_hang"
elif dm: status = "ok"
elif rc != "0": status = "exit_%s" % rc
else: status = "parse_fail"
print(",".join([isl, "1", kv, wu, pf, ft, dm, dmin, dmax, pu, status]))
PY
)
  echo "$row" >> "$CSV"
  echo "[baseline] ISL=$ISL -> $row" >&2
done

echo "=== baseline CSV ($CSV) ===" >&2
column -t -s, "$CSV" >&2
