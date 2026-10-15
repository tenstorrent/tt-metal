#!/usr/bin/env bash
# Scratch job (#60291 topk_xl lever A rest), not for merge: part 1 the tt-llk harness A/B with counters, probes and bits
# (run_llk_ab.sh, plan_la.txt), part 2 the topk_large_indices device-time A/B, bits and tests (run_lao.sh).
set +e +o pipefail
cd /work
echo "== part 1 harness $(date -u +%T)"
FILES=files_h.txt VARIANTS=variants_h GUARD_DIFF="A-B" GUARD_SAME="A-A2 B-B2" bash ci_ab/run_llk_ab.sh plan_la.txt
echo "== part 1 rc=$?"
git -C /work checkout -- $(cat ci_ab/files_h.txt)
echo "== part 2 op level $(date -u +%T)"
bash ci_ab/run_lao.sh
echo "== part 2 rc=$?"
