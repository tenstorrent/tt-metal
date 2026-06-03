#!/bin/bash
# C1 (sp8 · seq640 · q32 · chunk5120) at d_v=512, FULL 4-way matrix:
# {latent, non-latent} x {DM-on, DM-off} x 5 k-chunks. 50K+5K last chunk, perf-only.
# OOM-skip: once a (latent,dm) config OOMs at some k, larger k are recorded OOMskip without running.
set -u
cd /localdev/skrstic/tt-metal
source python_env/bin/activate

OUT=/localdev/skrstic/tt-metal/sweep_runs
mkdir -p "$OUT"
RES="$OUT/results_C1_dv512.tsv"
if [[ ! -f "$RES" ]]; then
  printf "dm\tlatent\tlabel\tsp\tper_dev\tq\tk\tchunk_size\td_v\tduration_ms\tutil\tcores\tstatus\n" > "$RES"
fi

LABEL=C1 SP=8 PER_DEV=640 Q=32 CS=5120 DV=512
ks="256 384 512 640 768"
LAST_STATUS=""

append_row() {  # dm latent k dur util cores status
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$LABEL" "$SP" "$PER_DEV" "$Q" "$3" "$CS" "$DV" "$4" "$5" "$6" "$7" >> "$RES"
}

run_one() {  # dm latent k
  local dm=$1 latent=$2 k=$3
  local id="kimi50k-q${Q}-k${k}-chunk${CS}"
  local log="$OUT/C1dv512_${dm}_lat${latent}_k${k}.log"
  CHUNKED_SP_SIZE=$SP CHUNKED_PER_DEVICE_CHUNK=$PER_DEV CHUNKED_Q_CHUNK=$Q CHUNKED_D_V=$DV \
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
  append_row "$dm" "$latent" "$k" "$dur" "$util" "$cores" "$st"
  LAST_STATUS=$st
  echo "[done] dm=$dm lat=$latent $LABEL k=$k dv=$DV -> dur=$dur util=$util st=$st (rc=$rc)"
}

for dm in on off; do
  python dm_toggle.py "$dm" >/dev/null
  python dm_toggle.py status | head -1
  for latent in 1 0; do
    oomed=0
    for k in $ks; do
      existing=$(grep -P "^${dm}\t${latent}\t${LABEL}\t${SP}\t${PER_DEV}\t${Q}\t${k}\t" "$RES" 2>/dev/null | head -1)
      if [[ -n "$existing" ]]; then
        est=$(echo "$existing" | awk -F'\t' '{print $NF}')
        [[ "$est" == OOM* ]] && oomed=1
        echo "[skip-done] dm=$dm lat=$latent k=$k ($est)"; continue
      fi
      if [[ $oomed -eq 1 ]]; then
        append_row "$dm" "$latent" "$k" NA NA NA OOMskip
        echo "[skip-larger-OOM] dm=$dm lat=$latent k=$k -> OOMskip"; continue
      fi
      run_one "$dm" "$latent" "$k"
      [[ "$LAST_STATUS" == OOM ]] && oomed=1
    done
  done
done

python dm_toggle.py on >/dev/null
echo "ALL DONE $(date)"
