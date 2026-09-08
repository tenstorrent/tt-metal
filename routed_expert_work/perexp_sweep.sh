#!/bin/bash
# Per-expert sweep: expert count x M, at both GU_CHUNKS settings, ND-sharded weights.
# gc=2 is the prefetch-era optimum (multi-expert); gc=3 was the pre-prefetch optimum and is
# what E=1 wants, since at E=1 there is no next expert and the prefetch is inert.
set -u
cd /localdev/mstaletovic/tt-metal
MS=64,256,512,1024,2048
for inter in bf16 bfp8; do
  for gc in 2 3; do
    # bfp8 only needs the gc=3 E=1 line: it exists purely as a regression check vs WORKLOG 7.
    if [ "$inter" = bfp8 ] && [ "$gc" != 3 ]; then continue; fi
    for e in 1 2 4 8 16; do
      if [ "$inter" = bfp8 ] && [ "$e" != 1 ]; then continue; fi
      tag=pe_${inter}_gc${gc}_e${e}
      BENCH_INTERMEDIATE=$inter \
      MOE_FUSED_SWIGLU_GU_CHUNKS=$gc \
      BENCH_M=$MS BENCH_ITERS=5 BENCH_EXPERTS=$e \
      BENCH_DISTINCT_W=1 BENCH_WSHARD=1 BENCH_TAG=$tag \
      timeout 3000 scripts/run_safe_pytest.sh --run-all routed_expert_work/test_bench.py \
        >routed_expert_work/$tag.log 2>&1
      printf '%-22s %s\n' "$tag" "$(grep -aoE '[0-9]+ passed|[0-9]+ failed|Program size \([0-9]+\)' routed_expert_work/$tag.log | sort -u | tr '\n' ' ')"
    done
  done
done
