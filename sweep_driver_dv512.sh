#!/bin/bash
# d_v=512 (wide value-latent) chunked-prefill perf sweep.
# Matrix: 8 configs (C1-C8) x 5 k-chunks x LATENT-V ONLY x {DM-on, DM-off}.
# Only the 50K+5K last chunk (CHUNKED_ONLY_LAST_CHUNK=1). Perf-only (CHUNKED_SKIP_PCC=1).
# DM-off = bulk NoC primitives physically commented out via dm_toggle.py (no macros).
# OOM-skip: once a config OOMs at some k, larger k for that config are not run (recorded OOMskip).
set -u
cd /localdev/skrstic/tt-metal
source python_env/bin/activate

OUT=/localdev/skrstic/tt-metal/sweep_runs
mkdir -p "$OUT"
RES="$OUT/results_dv512.tsv"
if [[ ! -f "$RES" ]]; then
  printf "dm\tlatent\tlabel\tsp\tper_dev\tq\tk\tchunk_size\td_v\tduration_ms\tutil\tcores\tstatus\n" > "$RES"
fi

configs=(
  "C1 8 640 32 5120"
  "C2 4 1280 64 5120"
  "C3 4 1248 96 4992"
  "C4 8 640 128 5120"
  "C5 8 640 64 5120"
  "C6 8 640 96 5120"
  "C7 8 640 160 5120"
  "C8 8 640 192 5120"
)
ks="256 384 512 640 768"
DV=512
LAST_STATUS=""

append_row() {  # dm label sp per_dev q k cs dur util cores status
  printf "%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$DV" "$8" "$9" "${10}" "${11}" >> "$RES"
}

run_one() {  # dm label sp per_dev q cs k
  local dm=$1 label=$2 sp=$3 per_dev=$4 q=$5 cs=$6 k=$7
  local id="kimi50k-q${q}-k${k}-chunk${cs}"
  local log="$OUT/dv512_${dm}_${label}_k${k}.log"
  CHUNKED_SP_SIZE=$sp CHUNKED_PER_DEVICE_CHUNK=$per_dev CHUNKED_Q_CHUNK=$q CHUNKED_D_V=$DV \
    CHUNKED_LATENT_V=1 CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1 \
    timeout 1000 scripts/run_safe_pytest.sh \
    "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table[${id}]" \
    > "$log" 2>&1
  local rc=$? row dur cores util st
  row=$(grep -E "^\| *10 \|" "$log" | tail -1)
  if [[ -n "$row" ]]; then
    dur=$(echo "$row"  | awk -F'|' '{gsub(/ /,"",$5); print $5}')
    cores=$(echo "$row"| awk -F'|' '{gsub(/ /,"",$6); print $6}')
    util=$(echo "$row" | awk -F'|' '{gsub(/[ %]/,"",$8); print $8}')
    st=OK
  else
    dur=NA; cores=NA; util=NA
    if grep -qiE "beyond max L1|Out of Memory|grow to .* which is beyond|Statically allocated" "$log"; then
      st=OOM
    elif [[ $rc -eq 124 ]]; then st=TIMEOUT; else st=FAIL; fi
  fi
  append_row "$dm" "$label" "$sp" "$per_dev" "$q" "$k" "$cs" "$dur" "$util" "$cores" "$st"
  LAST_STATUS=$st
  echo "[done] dm=$dm $label q=$q k=$k dv=$DV -> dur=$dur util=$util st=$st (rc=$rc)"
}

for dm in on off; do
  python dm_toggle.py "$dm" >/dev/null
  python dm_toggle.py status
  for cfg in "${configs[@]}"; do
    read -r label sp per_dev q cs <<< "$cfg"
    oomed=0
    for k in $ks; do
      # already recorded? seed oomed from its status and skip.
      existing=$(grep -P "^${dm}\t1\t${label}\t${sp}\t${per_dev}\t${q}\t${k}\t" "$RES" 2>/dev/null | head -1)
      if [[ -n "$existing" ]]; then
        est=$(echo "$existing" | awk -F'\t' '{print $NF}')
        [[ "$est" == OOM* ]] && oomed=1
        echo "[skip-done] dm=$dm $label k=$k ($est)"
        continue
      fi
      if [[ $oomed -eq 1 ]]; then
        append_row "$dm" "$label" "$sp" "$per_dev" "$q" "$k" "$cs" NA NA NA OOMskip
        echo "[skip-larger-OOM] dm=$dm $label k=$k -> OOMskip"
        continue
      fi
      run_one "$dm" "$label" "$sp" "$per_dev" "$q" "$cs" "$k"
      [[ "$LAST_STATUS" == OOM ]] && oomed=1
    done
  done
done

python dm_toggle.py on >/dev/null
echo "ALL DONE $(date)"
