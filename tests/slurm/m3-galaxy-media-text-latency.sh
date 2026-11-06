#!/bin/bash
# tt-metal: MiniMax-M3 per-chunk prefill latency, media refs off vs on, no images
#          (--topo galaxy --mode single)
#
# tt-metal#62157 requires text-only latency unchanged within noise once the media refs path is
# compiled in. Every arm runs on the SAME node in the SAME job, so the power cap, the weight cache
# and the fabric state are identical and cancel out of the delta. Report the pair as a delta, never
# as an absolute latency: the node's cap is not production's.
#
#   refs0:  PREFILL_MEDIA_MAX_REFS=0   (refs path off)
#   refs11: PREFILL_MEDIA_MAX_REFS=11  (on, cap at a 512 B H2D page), no PREFILL_PRODUCER_MEDIA
#
# Measured from the runner's own CHUNK_START compute_start timestamps, not from the producer's
# push_ms: push_ms is the host writing the H2D socket (wall=0.0s for 20k tokens) and says nothing
# about prefill. The runner's [prefill timing] line only covers WARMUP compile, which is dominated
# by whether the kernel cache on this node is warm, so it is not comparable between arms either.
#
# Two ordering guards, because the first measurement of a fresh runner is not steady state:
#   - the first chunk of each arm is dropped (one earlier run showed 858 ms for c0 against ~485 ms
#     for the rest, purely because that arm ran first);
#   - the arms run twice in opposite orders (refs0, refs11, refs11, refs0). A real cost appears in
#     both passes; an ordering artefact cancels.
#
# Arm refs0 is the same manifests with one key rewritten, so every other key stays a single source
# of truth; a second hand-maintained pair would be the thing that drifts.
set -euo pipefail

export TT_METAL_HOME="$WORKER_METAL_HOME"
export PYTHONPATH="$WORKER_METAL_HOME"
cd "$WORKER_METAL_HOME"

MANIFESTS=models/demos/minimax_m3/tt/runners/manifests
BINDING_ON=$MANIFESTS/m3_binding_media_1rank.yaml
PRODUCER_ON=$MANIFESTS/m3_producer_media_latency.yaml

# PREFILL_MEDIA_MAX_REFS sizes the H2D socket metadata on both sides and the socket service wants an
# exact match, so arm A rewrites the runner's binding and the producer's manifest together. Rewriting
# one alone dies as metadata.size() == cfg_.metadata_size_bytes.
refs0_copy() {
  local out
  out=$(mktemp "/tmp/m3_$(basename "$1" .yaml)_refs0_XXXX.yaml")
  sed 's/^\([[:space:]]*PREFILL_MEDIA_MAX_REFS: \).*/\1"0"/' "$1" > "$out"
  grep -q 'PREFILL_MEDIA_MAX_REFS: "0"' "$out" || { echo "refs0 rewrite missed the key in $1" >&2; exit 1; }
  echo "$out"
}
BINDING_OFF=$(refs0_copy "$BINDING_ON")
PRODUCER_OFF=$(refs0_copy "$PRODUCER_ON")

HOST="$(echo "$RESOLVED_HOSTS" | cut -d, -f1 | cut -d: -f1)"

run_arm() {
  local label="$1" binding="$2" producer="$3"
  echo "=== arm $label: $binding + $producer ==="
  local r_refs p_refs
  r_refs=$(grep -oE 'PREFILL_MEDIA_MAX_REFS: *"[0-9]+"' "$binding" | grep -oE '[0-9]+')
  p_refs=$(grep -oE 'PREFILL_MEDIA_MAX_REFS: *"[0-9]+"' "$producer" | grep -oE '[0-9]+')
  [ "$r_refs" = "$p_refs" ] || { echo "arm $label: runner refs=$r_refs, producer refs=$p_refs" >&2; exit 1; }
  rm -f /dev/shm/*prefill*; pkill -f prefill_runner || true
  ./models/demos/common/prefill/runners/run_pipeline_prefill.sh "$binding" "$HOST:1" \
    > "$RANKLOG_DIR/runner_$label.log" 2>&1 &
  local runner=$!
  # The runner logs "setup complete, entering request loop" and then blocks; it prints nothing for
  # minutes during mesh bringup and weight load, which is not a hang.
  timeout 2400 bash -c \
    "until grep -q 'entering request loop' '$RANKLOG_DIR/runner_$label.log'; do sleep 5; \
       kill -0 $runner 2>/dev/null || exit 1; done"
  python3 -m models.demos.common.prefill.runners.prefill_producer \
    --manifest "$producer" 2>&1 | tee "$RANKLOG_DIR/producer_$label.log"
  kill "$runner" 2>/dev/null || true
  wait "$runner" 2>/dev/null || true
}

# Pass 1 in one order, pass 2 reversed; see the ordering guards at the top.
PREFILL_PRODUCER_MEDIA="" run_arm refs0_p1 "$BINDING_OFF" "$PRODUCER_OFF"
PREFILL_PRODUCER_MEDIA="" run_arm refs11_p1 "$BINDING_ON" "$PRODUCER_ON"
PREFILL_PRODUCER_MEDIA="" run_arm refs11_p2 "$BINDING_ON" "$PRODUCER_ON"
PREFILL_PRODUCER_MEDIA="" run_arm refs0_p2 "$BINDING_OFF" "$PRODUCER_OFF"

echo "=== per-chunk prefill, steady state (first chunk of each arm dropped) ==="
python3 - "$RANKLOG_DIR" <<'PY'
import re, statistics, sys, pathlib

ranklogs = pathlib.Path(sys.argv[1])
pat = re.compile(r"CHUNK_START c=(\d+) compute_start=([\d.]+)")
rows = {}
for arm in ("refs0_p1", "refs11_p1", "refs11_p2", "refs0_p2"):
    log = ranklogs / f"runner_{arm}.log"
    if not log.exists():
        print(f"{arm}: no runner log")
        continue
    starts = sorted(float(m.group(2)) for m in pat.finditer(log.read_text(errors="replace")))
    gaps = [(b - a) * 1000 for a, b in zip(starts, starts[1:])][1:]  # drop the first gap
    if not gaps:
        print(f"{arm}: {len(starts)} chunk(s), nothing left after dropping the first gap")
        continue
    rows[arm] = gaps
    print(f"{arm:11s} n={len(gaps)} median={statistics.median(gaps):7.1f} ms  "
          f"min={min(gaps):7.1f}  max={max(gaps):7.1f}")

for p in ("p1", "p2"):
    off, on = rows.get(f"refs0_{p}"), rows.get(f"refs11_{p}")
    if off and on:
        d = statistics.median(on) - statistics.median(off)
        print(f"pass {p}: refs11 - refs0 = {d:+.1f} ms "
              f"({d / statistics.median(off) * 100:+.2f}% of refs0)")
print("Both passes must agree in sign and size for the delta to mean anything.")
PY
