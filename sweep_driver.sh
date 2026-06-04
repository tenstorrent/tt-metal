#!/bin/bash
# Autonomous chunked-prefill perf sweep driver.
# Matrix: 4 configs x 5 k-chunks x {latent,non-latent} x {DM-on,DM-off} = 80 runs.
# Only the 50K+5K last chunk is profiled (CHUNKED_ONLY_LAST_CHUNK=1). Perf-only (CHUNKED_SKIP_PCC=1).
# DM-off = bulk NoC primitives physically commented out in the kernels (dm_toggle.py, NO macros).
set -u
cd /localdev/skrstic/tt-metal
source python_env/bin/activate

OUT=/localdev/skrstic/tt-metal/sweep_runs
mkdir -p "$OUT"
RES="$OUT/results.tsv"
if [[ ! -f "$RES" ]]; then
  printf "dm\tlatent\tlabel\tsp\tper_dev\tq\tk\tchunk_size\tduration_ms\tutil\tcores\tstatus\n" > "$RES"
fi

# label sp per_dev q chunk_size
configs=(
  "C5 8 640 64 5120"
  "C6 8 640 96 5120"
  "C7 8 640 160 5120"
  "C8 8 640 192 5120"
)
ks="256 384 512 640 768"

run_one() {
  local dm=$1 latent=$2 label=$3 sp=$4 per_dev=$5 q=$6 cs=$7 k=$8
  local id="kimi50k-q${q}-k${k}-chunk${cs}"
  local log="$OUT/${dm}_lat${latent}_${label}_k${k}.log"
  # Skip if already have a parsed result line for this combo (resume support).
  if grep -qP "^${dm}\t${latent}\t${label}\t${sp}\t${per_dev}\t${q}\t${k}\t" "$RES" 2>/dev/null; then
    echo "[skip-done] dm=$dm lat=$latent $label k=$k"
    return
  fi
  CHUNKED_SP_SIZE=$sp CHUNKED_PER_DEVICE_CHUNK=$per_dev CHUNKED_Q_CHUNK=$q \
    CHUNKED_LATENT_V=$latent CHUNKED_ONLY_LAST_CHUNK=1 CHUNKED_SKIP_PCC=1 \
    timeout 1000 scripts/run_safe_pytest.sh \
    "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_chunked_perf_table[${id}]" \
    > "$log" 2>&1
  local rc=$?
  local row dur cores util st
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
    elif [[ $rc -eq 124 ]]; then
      st=TIMEOUT
    else
      st=FAIL
    fi
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$dm" "$latent" "$label" "$sp" "$per_dev" "$q" "$k" "$cs" "$dur" "$util" "$cores" "$st" >> "$RES"
  echo "[done] dm=$dm lat=$latent $label q=$q k=$k -> dur=$dur util=$util cores=$cores st=$st (rc=$rc)"
}

for dm in on off; do
  python dm_toggle.py "$dm" >/dev/null
  python dm_toggle.py status
  for latent in 1 0; do
    for cfg in "${configs[@]}"; do
      read -r label sp per_dev q cs <<< "$cfg"
      for k in $ks; do
        run_one "$dm" "$latent" "$label" "$sp" "$per_dev" "$q" "$cs" "$k"
      done
    done
  done
done

python dm_toggle.py on >/dev/null
echo "ALL DONE $(date)"
