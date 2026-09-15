#!/bin/bash
# T2.9a indexer_score_dsa on the calibration checkout (analysis/indexer_probe.py), 3 points x 3 iterations, counters at (2048, 8192).
# T2.9b TopK COUNTERS group on the fresh build with the indexer cell excluded.
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"
OUT=${OUT:-$HANDOFF/data/topk}   # in a checkout HANDOFF is DD, so the output lands under the invocation directory
cd $TTM; export TT_METAL_HOME=$TTM ARCH_NAME=blackhole TT_METAL_FORCE_JIT_COMPILE=1; source $PYENV
ix(){ # tag sq t mode
  TAG=$1; MODE=${4:-plain}; T0=$(date -u +%FT%TZ); rm -rf generated/profiler/reports/*; rm -f generated/profiler/.logs/profile_log_device.csv
  if [ "$MODE" = mp ]; then TP="python -m tracy -r --perf-counter-multipass --profiler-capture-perf-counters=all -m pytest"; else TP="python -m tracy -r -m pytest"; fi
  echo "### $TAG start $T0 sha=$(git rev-parse --short HEAD) Sq=$2 T=$3 mode=$MODE"
  IX_SQ=$2 IX_T=$3 IX_ITERS=3 IX_HEADS=64 IX_D=128 $TP analysis/indexer_probe.py::test_indexer_probe -s > $OUT/$TAG.log 2>&1 || true
  R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1); sleep 2
  if [ -n "$R" ] && ls $R/ops_perf_results*.csv >/dev/null 2>&1; then cp $R/ops_perf_results*.csv $OUT/${TAG}_ops_perf_results.csv; cp generated/profiler/.logs/profile_log_device.csv $OUT/${TAG}_profile_log_device.csv; else echo "RUN FAILED $TAG"; grep -E "^E |Error" $OUT/$TAG.log | head -3; fi
  FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $OUT/$TAG.log | sed "s/firmware bundle version: //")
  echo "PROVENANCE: p100a fw_bundle=$FW; tt-metal $(git rev-parse HEAD) branch $(git branch --show-current) (kernels 72620d5, calibration checkout); $T0; cmd: IX_SQ=$2 IX_T=$3 IX_ITERS=3 IX_HEADS=64 IX_D=128 $TP analysis/indexer_probe.py::test_indexer_probe -s; mode=$MODE" > $OUT/${TAG}_PROVENANCE.txt
  echo "### $TAG done $(date -u +%H:%M:%SZ)"
}
ix t29_indexer_Sq2048_T8192 2048 8192
ix t29_indexer_Sq2048_T32768 2048 32768
ix t29_indexer_Sq512_T8192 512 8192
ix t29_indexer_Sq2048_T8192_mp 2048 8192 mp
# T2.9b on the fresh build
TTF=$TTM_FRESH; cd $TTF; export TT_METAL_HOME=$TTF PYTHONPATH=$TTF:$TTF/ttnn:$TTF/tools
SCRIPT=${SCRIPT:-$([ -f "$SD/topk_campaign.py" ] && echo "$SD/topk_campaign.py" || echo "$HANDOFF/topk/topk_campaign.py")}; T0=$(date -u +%FT%TZ); rm -rf generated/profiler/reports/*
echo "### topk group COUNTERS (indexer excluded) mode=mp start $T0 sha=$(git rev-parse --short HEAD)"
python -m tracy -r -p --perf-counter-multipass --profiler-capture-perf-counters=all $SCRIPT --classes COUNTERS --exclude-op indexer --out $OUT > $OUT/run_COUNTERS_noindexer.log 2>&1 || echo "group exited nonzero"
R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1); sleep 2
if [ -n "$R" ] && ls $R/ops_perf_results*.csv >/dev/null 2>&1; then
  cp $R/ops_perf_results*.csv $OUT/ops_perf_results_COUNTERS_noindexer.csv; cp generated/profiler/.logs/profile_log_device.csv $OUT/profile_log_device_COUNTERS_noindexer.csv; cp -r generated/profiler/.logs/perf_counter_passes $OUT/perf_counter_passes_COUNTERS_noindexer 2>/dev/null
  mv $OUT/cells_COUNTERS.csv $OUT/cells_COUNTERS_noindexer.csv 2>/dev/null
  python3 $SCRIPT --postprocess $OUT/ops_perf_results_COUNTERS_noindexer.csv $OUT/cells_COUNTERS_noindexer.csv >> $OUT/run_COUNTERS_noindexer.log 2>&1 && mv $OUT/topk_campaign_results.csv $OUT/results_COUNTERS_noindexer.csv
  FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $OUT/run_COUNTERS_noindexer.log | sed "s/firmware bundle version: //")
  echo "PROVENANCE: p100a fw_bundle=$FW; tt-metal-fresh $(git rev-parse HEAD) (main 2dbd14bf632 merged + BRISC fw-size patch); $T0; cmd: python -m tracy -r -p --perf-counter-multipass --profiler-capture-perf-counters=all topk_campaign.py --classes COUNTERS --exclude-op indexer --out $OUT; report $R" > $OUT/PROVENANCE_COUNTERS_noindexer.txt
  echo "### topk group COUNTERS (indexer excluded) done $(date -u +%H:%M:%SZ)"
else echo "### topk COUNTERS (indexer excluded) FAILED (no ops report)"; grep -E "Error|Traceback|FATAL" $OUT/run_COUNTERS_noindexer.log | head -5; fi
echo "### T2.9 done $(date -u +%H:%M:%SZ)"
