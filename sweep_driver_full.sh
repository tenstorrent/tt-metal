#!/bin/bash
# FULL extended sweep, 320 cases:
#   8 configs (C1-C8) x {latent, non-latent} x {DM-on, DM-off} x 10 k-chunks.
#   latent  -> d_v=512 (wide value-latent, V from K)
#   non-lat -> d_v=128 (standard separate V tensor)
# k pool (sorted ascending for OOM-skip): 224 256 384 448 480 512 640 672 768 800
# 50K+5K last chunk, perf-only. DM-off = NoC primitives commented out via dm_toggle.py (no macros).
# OOM-skip: once a (dm,latent,config) OOMs at some k, larger k are recorded OOMskip without running.
# Resumable: re-running skips rows already in the TSV and re-seeds the OOM flag from them.
set -u
cd /localdev/skrstic/tt-metal
source python_env/bin/activate

OUT=/localdev/skrstic/tt-metal/sweep_runs
mkdir -p "$OUT"
RES="$OUT/results_full.tsv"
if [[ ! -f "$RES" ]]; then
  printf "dm\tlatent\tlabel\tsp\tper_dev\tq\tk\tchunk_size\td_v\tduration_ms\tutil\tcores\tstatus\n" > "$RES"
fi

# label sp per_dev q chunk_size
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
ks="224 256 384 448 480 512 640 672 768 800"
LAST_STATUS=""

append_row() {  # dm latent label sp per_dev q k cs dv dur util cores status
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" >> "$RES"
}

run_one() {  # dm latent label sp per_dev q cs dv k
  local dm=$1 latent=$2 label=$3 sp=$4 per_dev=$5 q=$6 cs=$7 dv=$8 k=$9
  local id="kimi50k-q${q}-k${k}-chunk${cs}"
  local log="$OUT/full_${dm}_lat${latent}_${label}_k${k}.log"
  CHUNKED_SP_SIZE=$sp CHUNKED_PER_DEVICE_CHUNK=$per_dev CHUNKED_Q_CHUNK=$q CHUNKED_D_V=$dv \
    CHUNKED_LATENT_V=$latent CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1 \
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
  append_row "$dm" "$latent" "$label" "$sp" "$per_dev" "$q" "$k" "$cs" "$dv" "$dur" "$util" "$cores" "$st"
  LAST_STATUS=$st
  echo "[done] dm=$dm lat=$latent $label q=$q k=$k dv=$dv -> dur=$dur util=$util st=$st (rc=$rc)"
}

for dm in on off; do
  python dm_toggle.py "$dm" >/dev/null
  python dm_toggle.py status | head -1
  for latent in 1 0; do
    if [[ $latent -eq 1 ]]; then dv=512; else dv=128; fi
    for cfg in "${configs[@]}"; do
      read -r label sp per_dev q cs <<< "$cfg"
      oomed=0
      for k in $ks; do
        existing=$(grep -P "^${dm}\t${latent}\t${label}\t${sp}\t${per_dev}\t${q}\t${k}\t" "$RES" 2>/dev/null | head -1)
        if [[ -n "$existing" ]]; then
          est=$(echo "$existing" | awk -F'\t' '{print $NF}')
          [[ "$est" == OOM* ]] && oomed=1
          echo "[skip-done] dm=$dm lat=$latent $label k=$k ($est)"; continue
        fi
        if [[ $oomed -eq 1 ]]; then
          append_row "$dm" "$latent" "$label" "$sp" "$per_dev" "$q" "$k" "$cs" "$dv" NA NA NA OOMskip
          echo "[skip-larger-OOM] dm=$dm lat=$latent $label k=$k -> OOMskip"; continue
        fi
        run_one "$dm" "$latent" "$label" "$sp" "$per_dev" "$q" "$cs" "$dv" "$k"
        [[ "$LAST_STATUS" == OOM ]] && oomed=1
      done
    done
  done
done

python dm_toggle.py on >/dev/null
echo "ALL DONE $(date)"
