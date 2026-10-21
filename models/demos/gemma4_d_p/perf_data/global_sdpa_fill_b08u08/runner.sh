#!/bin/bash
# Overnight queue runner. queue.txt lines: tag|kind|arg|branch|patch|ENV=a ENV=b
#   kind layer  : tracy layer benchmark, arg = chunk_idx (64, ci, ...), 2K global layer
#        e2e    : traced 256k prefill, arg = chunk size
#        acc    : pytest node/selector of the SDPA nightly test file, arg = -k expression
#        kvpcc  : test_prefill_migration[mock-256k] KV PCC, arg = chunk size
#        script : python file in $W, arg = file name
#   branch: checked out (and built) in the repo before the run ('-' = stay); patch: applied/reverted around the run.
# Stops on STOP or on a timeout (likely device hang).
W=/data/mmanzoor/global_sdpa_fill_b08u08; REPO=/home/mmanzoor/compiler/tt-metal; Q=$W/queue.txt; DONE=$W/done.txt; R=$W/results.txt
touch $Q $DONE; cd $REPO
log() { echo "$(date +%T) $*" >> $R; }
while true; do
  line=$(grep -v -x -F -f <(cat $DONE; echo __none__) $Q | grep -v '^#' | grep -v '^$' | head -1)
  if [ -z "$line" ]; then sleep 20; continue; fi
  [ "$line" = "STOP" ] && { log STOP; exit 0; }
  IFS='|' read -r tag kind arg branch patch envs <<< "$line"
  log "START $tag kind=$kind arg=$arg branch=$branch patch=$patch env=$envs"
  if [ "$branch" != "-" ] && [ "$(git branch --show-current)" != "$branch" ]; then
    if ! git checkout -q "$branch" >> $W/$tag.log 2>&1; then log "END $tag rc=98 CHECKOUT_FAIL"; echo "$line" >> $DONE; continue; fi
    git submodule update --init --recursive >> $W/$tag.build.log 2>&1
    if ! (ninja -C build && ninja -C build install) >> $W/$tag.build.log 2>&1; then log "END $tag rc=97 BUILD_FAIL"; echo "$line" >> $DONE; continue; fi
    log "  built branch $branch at $(git rev-parse --short HEAD)"
  fi
  if [ "$patch" != "-" ] && ! git apply $W/patches/$patch >> $W/$tag.log 2>&1; then log "END $tag rc=96 PATCH_FAIL"; echo "$line" >> $DONE; continue; fi
  set +e
  ( source activate
    export X=1 $envs
    case $kind in
      layer) timeout -k 30 900 python -m tracy -r -p -v -o $W/$tag -m pytest -sv \
               "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk${arg}-global-sz2048-ctx_256k-8x4]" ;;
      e2e)   timeout -k 30 2400 pytest -sv \
               "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk${arg}-text-8x4]" ;;
      acc)   timeout -k 30 3600 pytest -sv tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py -k "$arg" ;;
      kvpcc) export GEMMA4_TEST_CHUNK_SIZE=$arg TT_CACHE_PATH=/mnt/weka/model-cache/scratch/google/gemma-4-31B-it-Cache \
                    PREFILL_TRACE_DIR=/mnt/weka/model-weights/llm/ref-data/gpu_traces/gemma4_d_p/gutenberg-135
             timeout -k 30 2400 pytest -sv "models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-256k]" ;;
      script) timeout -k 30 900 python $W/$arg ;;
    esac ) > $W/$tag.log 2>&1
  rc=$?
  [ "$patch" != "-" ] && git apply -R $W/patches/$patch >> $W/$tag.log 2>&1
  echo "$line" >> $DONE
  log "END $tag rc=$rc"
  case $kind in
    layer) REP=$(ls -td $W/$tag/reports/*/ 2>/dev/null | head -1); [ -n "$REP" ] && python3 -I $W/layer_ops.py $REP > $W/$tag.ops.txt 2>&1; cat $W/$tag.ops.txt >> $R 2>/dev/null ;;
    e2e)   grep -h "\[traced_perf\] DEVICE\|ring-depth cost" $W/$tag.log | sed 's/^.*\[traced_perf\]/  /' >> $R ;;
    acc|kvpcc) grep -hE "passed|failed|PCC|pcc|RRMSE|min per" $W/$tag.log | grep -v "^\s*$" | tail -6 | cut -c1-220 | sed 's/^/  /' >> $R ;;
    script) grep -h "^MM \|Error" $W/$tag.log | tail -10 >> $R ;;
  esac
  grep -h "TT_THROW\|critical" $W/$tag.log | head -2 | cut -c1-220 | sed 's/^/  /' >> $R
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then log "HANG on $tag -> runner stopped; board likely needs reset"; exit 1; fi
done
