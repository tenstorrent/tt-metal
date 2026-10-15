#!/usr/bin/env bash
# Scratch job 2 (#60291 topk_xl levers), not for merge: part 1 the tt-llk harness A/B with counters, probes and bits
# (run_llk_ab.sh, plan_la2.txt, variants_h2), part 2 the topk_large_indices device-time A/B, bits and tests (run_lao2.sh).
set +e +o pipefail
cd /work
echo "== part 1 harness $(date -u +%T)"
FILES=files_h.txt VARIANTS=variants_h2 GUARD_DIFF="A-C C-CD" GUARD_SAME="C-C2 CD-CD2" bash ci_ab/run_llk_ab.sh plan_la2.txt
echo "== part 1 rc=$?"
git -C /work checkout -- $(cat ci_ab/files_h.txt)
echo "== part 2 op level $(date -u +%T)"
ARMS="C CD" bash ci_ab/run_lao2.sh
echo "== part 2 rc=$?"
