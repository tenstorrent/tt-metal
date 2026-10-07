#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724): the whole test modules of the dest-reuse opt-ins with the define in place.
set -uo pipefail
cd /work
D='#define ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE true'
for k in ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp \
         ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/csa_compressor/device/kernels/compute_csa_compressor.cpp \
         models/demos/deepseek_v3_b1/micro_ops/dram_streaming_matmul/kernels/dram_streaming_matmul_kernel.cpp \
         models/demos/deepseek_v3_b1/micro_ops/dram_streaming_experts_matmul/kernels/dram_streaming_experts_matmul_kernel.cpp; do
  python3 - "$k" "$D" <<'PY'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read().splitlines(keepends=True)
i = next(k for k, l in enumerate(s) if l.startswith("#include"))
s.insert(i, d + "\n")
open(p, "w").write("".join(s))
PY
  echo "opted in: $k"
done
rc=0
for t in tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py \
         models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_csa_compressor.py \
         models/demos/deepseek_v3_b1/tests/unit_tests/test_dram_streaming_matmul.py; do
  timeout -s INT -k 60 2400 python3 -m pytest -p no:cacheprovider -o timeout_method=thread -q -rfEs "$t" > /tmp/optin_$(basename $t).log 2>&1
  r=$?; [[ $r != 0 ]] && rc=$r
  echo "=== $t rc=$r: $(grep -E 'passed|failed|error' /tmp/optin_$(basename $t).log | tail -1)"
  grep -E "^(FAILED|ERROR)" /tmp/optin_$(basename $t).log | head -20
  grep -E "^SKIPPED" /tmp/optin_$(basename $t).log | sort | uniq -c | head -10
done
exit $rc
