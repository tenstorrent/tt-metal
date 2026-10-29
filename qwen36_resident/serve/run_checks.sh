#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The serving checks against a running serve_local.sh server (start it first; this script does not touch the device).
#
# usage: run_checks.sh plain|mtp OUT_DIR
#   plain: parity with OUT_DIR/greedy_ref.json (greedy_ref.py output), then TTFT / TPOT at 128 / 2048 / 8192 / 32000
#          prompt tokens (greedy, Qwen's default sampling and top_k off at 128), chat and document decode rates,
#          and parity again after the long and sampled requests.
#   mtp:   parity with OUT_DIR/plain_parity.json (the plain profile's served greedy text) and, offline, with
#          OUT_DIR/greedy_ref.json and OUT_DIR/mtp_ref.json (mtp_ref.py: the verify-step greedy); then the speeds.
# env: PORT (8000), HF_MODEL (tokenizer), VLLM_VENV (python with transformers and vllm), SERVED_MODEL_NAME
set -euo pipefail
PROFILE=${1:?plain|mtp}
OUT=${2:?OUT_DIR}
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH=$WT${PYTHONPATH:+:$PYTHONPATH}
PY=${VLLM_VENV:+$VLLM_VENV/bin/}python
C="$PY -m qwen36_resident.serve.chat_client"
B=$WT/qwen36_resident/serve/bench_serve.sh
P="--port ${PORT:-8000}"
DOC=$WT/METALIUM_GUIDE.md
mkdir -p "$OUT/bench"

$C speed $P --out "$OUT/${PROFILE}_warm.json" --max-tokens 16 --thinking off --temperature 0
if [ "$PROFILE" = plain ]; then
  $C parity $P --ref "$OUT/greedy_ref.json" --out "$OUT/plain_parity.json"
  NUM_PROMPTS=5 $B "$OUT/bench" plain_greedy 128 2048 8192
  NUM_PROMPTS=3 $B "$OUT/bench" plain_greedy 32000
  NUM_PROMPTS=5 SAMPLING="--temperature 1.0 --top-k 20 --top-p 0.95" $B "$OUT/bench" plain_qwendefault 128
  NUM_PROMPTS=5 SAMPLING="--temperature 1.0 --top-k -1 --top-p 0.95" $B "$OUT/bench" plain_topkoff 128
  $C speed $P --out "$OUT/plain_chat_greedy.json" --max-tokens 256 --temperature 0 --thinking both
  $C speed $P --out "$OUT/plain_chat_qwendefault.json" --max-tokens 256 --temperature 1.0 --top-k 20 --top-p 0.95
  $C speed $P --out "$OUT/plain_chat_topkoff.json" --max-tokens 256 --temperature 1.0 --top-k -1 --top-p 0.95
else
  $C parity $P --ref "$OUT/plain_parity.json" --out "$OUT/mtp_parity.json" || true
  $C compare --served "$OUT/mtp_parity.json" --ref "$OUT/greedy_ref.json" --out "$OUT/mtp_vs_greedy_ref.json" || true
  if [ -f "$OUT/mtp_ref.json" ]; then
    $PY - "$OUT/mtp_parity.json" "$OUT/mtp_ref.json" <<'PY'
import json, sys
served = {r["name"]: r["token_ids"] for r in json.load(open(sys.argv[1]))["prompts"]}
ref = {r["name"]: r["verify_tokens"] for r in json.load(open(sys.argv[2]))["prompts"]}
ok = all(t == ref[n][: len(t)] for n, t in served.items())
for n, t in served.items():
    print(f"{n}: served == verify-step greedy: {t == ref[n][: len(t)]}")
print("MTP VS VERIFY GREEDY " + ("PASS" if ok else "FAIL"))
PY
  fi
  $C speed $P --out "$OUT/mtp_chat_greedy.json" --max-tokens 256 --temperature 0 --thinking both
  NUM_PROMPTS=5 $B "$OUT/bench" mtp_greedy 128 2048 8192
  NUM_PROMPTS=3 $B "$OUT/bench" mtp_greedy 32000
fi
for N in 2048 8192; do
  $C speed $P --out "$OUT/${PROFILE}_doc$N.json" --max-tokens 256 --temperature 0 --thinking off --doc "$DOC" --doc-tokens $N
done
$C parity $P --ref "$OUT/$([ "$PROFILE" = plain ] && echo greedy_ref || echo plain_parity).json" --out "$OUT/${PROFILE}_parity2.json" || true
