#!/bin/bash
# Device check of V4.1 prefill-side prefix reuse H2 (the host-RAM chunk store, V41_PREFILL_PREFIX_STORE_GB on top of H1), run FROM
# HOST 29 against the prefill galaxy H (default 30) in a MAINTENANCE WINDOW -- the preconditions, model build and runner harness
# of H1's run_device_check.sh (which this calls): the serving prefill runner on H DOWN, H's chips free, >= 10 GB on H's root disk.
#   R2  references with arm OFF (today's runner; the store neither read nor written), each after X; A twice = determinism
#   S2  the store sequence A, U, A2, U, A, U, AP, U, B, A2, A2 (the store emptied first): every digest (the 7 export tensors, the
#       model rows, the path-A slab's entry rows, the Engram history) == its R2 reference, and every prompt's per-chunk pattern
#       (S skipped / R restored / C computed) == the driver's EXPECT_S2
#   T2  (PHASES=R2,S2,T2) timing, 2 reps: A off vs A reuse (store reads), A2 off vs A2 reuse (3 chunks restored)
# Verdict lines "[driver] VERDICT determinism", "VERDICT reuse", "VERDICT h2 pattern" in $OUT/check.log; results.json. How to
# read them: README.md next to this script, section "H2".
#
#   H=30 bash run_device_check_h2.sh       (~30-45 min: build ~15-25 min, then ~15 min of prompts + digests)
#   env: as run_device_check.sh, plus STORE_GB (default 8: the check stores ~25 chunks = ~0.7 GiB), RESERVE_GB (default 32),
#        SPILL_TIPS (default 2), STORE_TIP_GB (default STORE_GB / 16)
# It does not restart the serving stack: bring it back afterwards (v41ax_stack_up.sh / stack_up.sh).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DST=${DST:-/home/ttuser/sdawle/tt-metal-1017-h2}
export RUN=${RUN:-/home/ttuser/sdawle/prefill_runs/v41_prefix_check_h2}
export OUT=${OUT:-/home/ttuser/sdawle/v41ax_logs/prefix_check_h2/$(date +%m%d_%H%M)}
export PHASES=${PHASES:-R2,S2}
export STORE_GB=${STORE_GB:-8} RESERVE_GB=${RESERVE_GB:-32} SPILL_TIPS=${SPILL_TIPS:-2}
if [ -n "${STORE_TIP_GB:-}" ]; then export STORE_TIP_GB; fi
exec bash "$HERE/run_device_check.sh"
