#!/usr/bin/env bash
# #37 driver, runs detached on blx03: S2 prompt reuse A/B (LTX_S2_PROMPT_REUSE=0 reference, then =1).
# Tree ~/fasth3/tt-metal (t36 16ba9a383d, contains 5d993cd2f7c), conv decoder, 1080p/145f, seed 0,
# gen#0 = capture, gen#1 and gen#2 = pure replays. One project job at a time: each submit waits until no
# other smarton job is running or queued. Writes $D/DONE when both jobs finished (whatever the outcome).
D=/home/smarton/fasth3/t37
OUT=/home/smarton/fasth3/out/t37
# Busy also while the broker holds or recovers the device: never queue behind a hold.
busy() { tt-device-mcp status 1 | sed -n '/^RUNNING/,/^RECENT/p' | grep -qE 'smarton|HELD|recover|reset|power'; }
run_one() {
  local label=$1 reuse=$2 id="" tries=0
  while [ -z "$id" ]; do
    while busy; do sleep 60; done
    id=$(/home/smarton/fasth3/tt-metal/tmp/blx03/submit.sh 600 \
      "OUT=$OUT bash tmp/blx03/run25.sh $label LTX25_DIFFVAE=0 LTX_S2_PROMPT_REUSE=$reuse" \
      "LTX_DUMP_LATENTS=$OUT/$label/lat LTX_E2E_EXTRA_REPLAYS=1 PYTEST_TIMEOUT=500" \
      | awk '/^Job [0-9]+ queued/{print $2; exit}')
    [ -z "$id" ] && { tries=$((tries+1)); [ $tries -ge 240 ] && { echo "$label: submit never accepted" | tee -a $D/jobs; echo DRIVE37_DONE > $D/DONE; exit 1; }; sleep 60; }
  done
  echo "$label job $id" | tee -a $D/jobs
  until ! tt-device-mcp status -j $id | grep -qiE 'Status: +(running|queued|pending)'; do sleep 30; done
  tt-device-mcp status -j $id | head -6
}
mkdir -p $D
[ -f $OUT/s2reuse0/lat.gen1.pt ] || run_one s2reuse0 0
run_one s2reuse1 1
bash $D/compare37.sh > $D/compare.txt 2>&1
echo DRIVE37_DONE > $D/DONE
