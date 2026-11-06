#!/bin/bash
# tt-metal: MiniMax-M3 producer -> runner end to end with images (--topo galaxy --mode single)
#
# The real runner with synthetic images from prefill_producer: PREFILL_MEDIA_MAX_REFS=11,
# PREFILL_PRODUCER_MEDIA="100:50,5000:300" and stub embeddings. The second image starts at 5000 with
# 300 tokens, so it crosses the 5120 chunk boundary -- the case a single-chunk prompt never reaches.
#
# Runner rank 0 and producer rank 0 hand off through /dev/shm/tt_h2d_stream_service_m3_prefill.bin,
# which carries no hostname, so both must be the same machine. One rank on one galaxy makes that
# true without the rank0-mgd directive the quad tests need.
set -euo pipefail

export TT_METAL_HOME="$WORKER_METAL_HOME"
export PYTHONPATH="$WORKER_METAL_HOME"
cd "$WORKER_METAL_HOME"

MANIFESTS=models/demos/minimax_m3/tt/runners/manifests
HOST="$(echo "$RESOLVED_HOSTS" | cut -d, -f1 | cut -d: -f1)"

rm -f /dev/shm/*prefill*; pkill -f prefill_runner || true

./models/demos/common/prefill/runners/run_pipeline_prefill.sh \
  $MANIFESTS/m3_binding_media_1rank.yaml "$HOST:1" > "$RANKLOG_DIR/runner.log" 2>&1 &
RUNNER=$!
timeout 2400 bash -c \
  "until grep -q 'entering request loop' '$RANKLOG_DIR/runner.log'; do sleep 5; \
     kill -0 $RUNNER 2>/dev/null || exit 1; done"

python3 -m models.demos.common.prefill.runners.prefill_producer \
  --manifest $MANIFESTS/m3_producer_media.yaml 2>&1 | tee "$RANKLOG_DIR/producer.log"

kill "$RUNNER" 2>/dev/null || true
wait "$RUNNER" 2>/dev/null || true

# PREFILL_PRODUCER_MEDIA puts an image at 100:50 and one at 5000:300; the second crosses the 5120
# chunk boundary, so refs must land in at least two chunks. Assert it rather than inferring from a
# KV PCC, which a media prompt has no golden for.
echo "=== media refs per chunk ==="
grep -aoE "CHUNK_START c=[0-9]+ .*media_refs=[0-9]+" "$RANKLOG_DIR/runner.log" |
  sed "s/compute_start=[0-9.]* //" | tee /tmp/m3_media_refs.txt
WITH_REFS=$(grep -c "media_refs=[1-9]" /tmp/m3_media_refs.txt || true)
[ "$WITH_REFS" -ge 2 ] || {
  echo "expected media refs in at least 2 chunks (the 5000:300 image spans a boundary), got $WITH_REFS" >&2
  exit 1
}
echo "ok: $WITH_REFS chunk(s) carried media refs"
