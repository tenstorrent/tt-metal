#!/bin/bash
# C1 (sp8 · seq640 · q32 · chunk5120), NON-LATENT only, at d_v=128 (true model value dim).
# Pairs with the latent d_v=512 rows already in results_C1_dv512.tsv:
#   latent  -> d_v=512 (wide value-latent, V from K)
#   non-lat -> d_v=128 (standard separate V tensor)
# DM-on/off, 5 k-chunks, OOM-skip. 50K+5K last chunk, perf-only. Appends to the same TSV.
set -u
cd /localdev/skrstic/tt-metal
source python_env/bin/activate

OUT=/localdev/skrstic/tt-metal/sweep_runs
RES="$OUT/results_C1_dv512.tsv"   # same file, append

LABEL=C1 SP=8 PER_DEV=640 Q=32 CS=5120 DV=128 LATENT=0
ks="256 384 512 640 768"
LAST_STATUS=""

append_row() {  # dm k dur util cores status
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$LATENT" "$LABEL" "$SP" "$PER_DEV" "$Q" "$2" "$CS" "$DV" "$3" "$4" "$5" "$6" >> "$RES"
}

run_one() {  # dm k
  local dm=$1 k=$2
  local id="kimi50k-q${Q}-k${k}-chunk${CS}"
  local log="$OUT/C1nl_dv128_${dm}_k${k}.log"
  CHUNKED_SP_SIZE=$SP CHUNKED_PER_DEVICE_CHUNK=$PER_DEV CHUNKED_Q_CHUNK=$Q CHUNKED_D_V=$DV \
    CHUNKED_LATENT_V=$LATENT CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1 \
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
  append_row "$dm" "$k" "$dur" "$util" "$cores" "$st"
  LAST_STATUS=$st
  echo "[done] dm=$dm lat=$LATENT $LABEL k=$k dv=$DV -> dur=$dur util=$util st=$st (rc=$rc)"
}

for dm in on off; do
  python dm_toggle.py "$dm" >/dev/null
  python dm_toggle.py status | head -1
  oomed=0
  for k in $ks; do
    if [[ $oomed -eq 1 ]]; then
      append_row "$dm" "$k" NA NA NA OOMskip
      echo "[skip-larger-OOM] dm=$dm k=$k -> OOMskip"; continue
    fi
    run_one "$dm" "$k"
    [[ "$LAST_STATUS" == OOM ]] && oomed=1
  done
done

python dm_toggle.py on >/dev/null
echo "ALL DONE $(date)"
