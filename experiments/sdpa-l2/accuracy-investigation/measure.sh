#!/usr/bin/env bash
set -euo pipefail
# Use the matching source build; this script does not swap implementations.
# Exact-main controls need all four modified operator files restored and rebuilt.
label=${1:?fresh result label}
variant=${2:?hifi2 or fp32_hifi2}
k_chunk=${3:?512 or 1024}
out=experiments/sdpa-l2/accuracy-investigation/${label}
if [[ -e "${out}.jsonl" ]]; then
    echo "Refusing to overwrite ${out}.jsonl" >&2
    exit 1
fi
if [[ -n "${TT_SDPA_ACCURACY_DIAG:-}" ]]; then
    sha256sum -c experiments/sdpa-l2/accuracy-investigation/DIAGNOSTIC-SHA256.txt
fi
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
    --kv-lens 262144 --full --heads 10 --q-chunk 128 --k-chunks "$k_chunk" \
    --variants "$variant" --seed 1236 --q-len 512 --query-sampling spread \
    --benchmark-warmup 40 --benchmark-iters 10 \
    --label "$label" --output "${out}.jsonl" > "${out}.log" 2>&1
