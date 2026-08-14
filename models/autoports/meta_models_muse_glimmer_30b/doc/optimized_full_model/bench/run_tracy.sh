#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Profile the **reduced** full-model variant with Tracy and render tt-perf-report
# tables/CSVs.  $full-model is explicit that the all-layer stack must not be
# profiled: 2400 ops per decode trace over 52 layers produces multi-GB dumps and
# overflows the profiler's marker buffers.  The reduced variant is the same wrapper
# with one real layer of each kind and the *real* terminal path, which is what the
# profile is for -- attributing the costs the full model adds over the decoder
# stack (embedding + gather, terminal norm, LM head, softcap, sampler).
#
# Same capture rules as the decoder stages' bench/run_tracy.sh: one device job at a
# time, TT_METAL_WATCHER must NOT be set, and ``ITERS`` trace replays per decode window
# -- **1** by default, not 8: the DRAM marker buffer overflows above that and silently
# under-counts, which is what the integrity check at the foot of this file now catches.
#
# On a mesh capture tt-perf-report merges the four devices' rows -- max for an
# ordinary op, mean for a collective -- so a Device Time column is one chip's
# worst-case work.  No --device-id is passed.
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_full_model
B=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/full_model/bench/perf_window.py
mkdir -p "$D/tracy" "$D/logs"

report () {  # $1=ops csv  $2=out prefix  $3=start signpost  $4=end signpost
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --no-summary > "$2.txt"
  tt-perf-report "$1" --start-signpost "$3" --end-signpost "$4" --csv "$2.csv" > "$2.console.log"
}

capture () {  # $1=ops csv destination
  # Newest ops CSV that still has the host+device columns.  Tracy writes a
  # device-only CSV over the same path a moment later and tt-perf-report refuses
  # that one, so taking the newest file unconditionally races away a window.
  local src
  for src in $(ls -t generated/profiler/reports/*/ops_perf_results_*.csv); do
    if head -1 "$src" | grep -q "DEVICE ID"; then cp "$src" "$1"; return 0; fi
  done
  echo "no ops CSV with a DEVICE ID column found" >&2
  return 1
}

python -m tracy -r -p -v "$B" --window prefill --prompt-len 128 2>&1 | tee "$D/logs/tracy_prefill_128.log"
capture "$D/tracy/prefill_128_ops.csv"
report "$D/tracy/prefill_128_ops.csv" "$D/tracy/prefill_128_perf_report" PERF_PREFILL PERF_PREFILL_END

# The decode window is captured **one layer kind at a time**, which the full-model
# stage did not have to do.  The SwiGLU multiply now runs on 80 cores instead of 16
# and is bracketed by two reshards, and the profiler emits markers per *core*: the
# two-layer window overflowed the DRAM marker buffer at ITERS=1 (20 dropped-marker
# lines, ``logs/run_tracy_two_layer_overflow.log``), which silently under-counts.
# One real layer of each kind, in its own capture, is the same coverage with half
# the markers -- and it attributes the terminal path in both.
for pair in "0 sliding" "3 full"; do
  set -- $pair
  layer=$1; kind=$2
  python -m tracy -r -p -v "$B" --window decode --layers "$layer" --iters ${ITERS:-1} 2>&1 \
    | tee "$D/logs/tracy_decode_$kind.log"
  capture "$D/tracy/decode_${kind}_ops.csv"
  report "$D/tracy/decode_${kind}_ops.csv" "$D/tracy/decode_${kind}_perf_report" PERF_DECODE PERF_DECODE_END
done

python -m tracy -r -p -v "$B" --window sampling --iters ${ITERS:-1} 2>&1 | tee "$D/logs/tracy_sampling.log"
capture "$D/tracy/sampling_ops.csv"
report "$D/tracy/sampling_ops.csv" "$D/tracy/sampling_perf_report" PERF_SAMPLING PERF_SAMPLING_END

# Dropped-marker integrity check, and it **fails the run** rather than printing.
# This was `|| true` until the round-5 stage review pointed out that both captures were
# dropping markers -- 460 lines in the decode window, 60 in the sampling one -- while the
# script cheerfully reported them and exited 0. A capture that overflows the profiler's
# DRAM marker buffer silently under-counts, and the give-away is arithmetic: every op
# count in a replay window must be a multiple of the replay count, and they were not
# (PlusOne 2 where 2*ITERS was due, Embeddings 4, AllGather 14). Percentages computed
# over a truncated denominator are worse than no profile, because they look like one.
#
# ITERS defaults to 1, not 8, for the same reason: fewer replays, fewer markers. 2 still
# overflowed the decode window (20 dropped lines); 1 is the value all three windows pass at,
# so it is the default rather than something a caller has to know to pass.
echo "=== dropped-marker integrity check (all must be 0) ==="
dropped=0
for log in "$D"/logs/tracy_*.log; do
  n=$(grep -c "markers were dropped" "$log" || true)
  echo "$(basename "$log"): $n"
  dropped=$((dropped + n))
done
if [ "$dropped" -ne 0 ]; then
  echo "TRACY_INTEGRITY_FAILED: $dropped dropped-marker lines; lower ITERS and re-run" >&2
  exit 1
fi
echo "TRACY_INTEGRITY_OK"

# The repo's check-large-files pre-commit hook rejects anything over 500 KB, and
# gzip -9 was not enough: a raw ops CSV compresses to ~640 KB, which the hook still
# rejects.  xz -9 takes the same file to ~200 KB.
for f in "$D"/tracy/*_ops.csv; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 400000 ] && xz -9 -f "$f"
done
echo "=== done ==="
