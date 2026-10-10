#!/bin/bash
# t334: one Tracy-profiled conv VAE decode on the full 4x8 mesh (1088x1920, 145 frames, t48 defaults),
# then the ops report and the per-layer conv3d table. blx03 serial-runner job; runs at the box's clock.
# Tracy's own capture path is not used: it starts the test with setsid (escapes our process group) and
# launches a WASM server into TT_METAL_HOME. We start tracy-capture ourselves, run the test in-process
# with --no-capture-tool, then build the report with --process-logs-only.
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
B=/home/smarton/fasth3/t315
V=/var/tmp/fasth3/t334
S=$V/src
use=$(df --output=pcent / | tail -1 | tr -dc 0-9)
[ "$use" -le 85 ] || { echo "[t334] / at ${use}%, refusing"; exit 5; }
# blx03: the project footprint under /var/tmp/fasth3 was already 457G before this job (shared weights and caches);
# this job adds a few GB of JIT and profiler logs, so the guard stops it only if that has grown past 480G.
gb=$(timeout 120 du -sxBG /var/tmp/fasth3 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 480 ] || { echo "[t334] /var/tmp/fasth3 at ${gb}G, refusing"; exit 5; }
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt LTX_FUSE_YUV_OUTPUT=1 LTX_PIN_CORES=0
export TT_METAL_CACHE=$V/jit TT_METAL_PROFILER_DIR=$V/prof TT_METAL_PROFILER_CPP_POST_PROCESS=1
export TT_METAL_DEVICE_PROFILER=1
LOGS=$V/prof/.logs
rm -rf "$V/prof" && mkdir -p "$LOGS"
cd "$S" || exit 6
echo "[t334] start $(date -u +%FT%TZ) host $(hostname)"
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
"$B/build/tools/profiler/bin/tracy-capture" -o "$LOGS/tracy_profile_log_host.tracy" -f -p "$PORT" &
CAP=$!
TRACY_PORT=$PORT timeout 480 python -m tracy -p -r -v --no-capture-tool -o "$V/prof" \
  -m pytest -c "$S/pytest.ini" --rootdir="$S" -sv --timeout=460 \
  models/tt_dit/tests/models/ltx/test_vae_ltx_prof_4x8.py
rc=$?
echo "[t334] test rc=$rc $(date -u +%FT%TZ)"
for _ in $(seq 30); do kill -0 $CAP 2>/dev/null || break; sleep 1; done
kill $CAP 2>/dev/null; wait $CAP 2>/dev/null
[ "$rc" -eq 0 ] || exit "$rc"
ls -la "$LOGS"
timeout 240 python -m tracy --process-logs-only -o "$V/prof" || { echo "[t334] report failed"; exit 7; }
CSV=$(ls -t "$V"/prof/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] || { echo "[t334] no ops_perf_results csv"; exit 8; }
echo "[t334] csv $CSV"
gzip -c "$CSV" > "$V/ops_perf_4x8.csv.gz"
python "$V/conv_table.py" "$CSV" "$V/t334_4x8" | tee "$V/t334_4x8_table.txt"
echo "[t334] done $(date -u +%FT%TZ)"
