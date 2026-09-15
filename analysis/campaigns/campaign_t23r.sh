#!/bin/bash
# T2.3 regimes at q128 k128 (zones off and on, one config each), counters on cross and MLA, plus causal S1024/S16384 walls.
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"; cd $DD   # DD is the data directory in both layouts (campaign_paths.sh)
run_pair(){ # name, target, extra env via caller export
  $SD/set_zone_config.sh 0 0 0 0 0 >/dev/null; $SD/run_regime.sh t23r_$1_zoff plain "$2"
  $SD/set_zone_config.sh 1 0 0 0 0 >/dev/null; $SD/run_regime.sh t23r_$1_zon plain "$2"
  if [ "$3" = mp ]; then $SD/run_regime.sh t23r_$1_zon_mp mp "$2"; fi
}
export P2_NH=16 P2_D=128 P2_QCHUNK=128 P2_ITERS=3
P2_MODE=cross run_pair cross_2048_8192_nh16 "analysis/p2_sweep.py::test_p2_cross -k 2048-8192" mp
P2_MODE=window P2_SEQ=8192 P2_WIN=1024 run_pair window_S8192_W1024_nh16 "analysis/p2_sweep.py::test_p2_window"
P2_MODE=mask P2_SEQ=8192 P2_DENSITIES=0.25 run_pair mask_S8192_d0.25_nh16 "analysis/p2_sweep.py::test_p2_mask"
export MLA_SEQ=2048 MLA_NH=16 MLA_NKV=1 MLA_KVLORA=512 MLA_DROPE=64 MLA_ITERS=3
run_pair mla_nh16_S2048 "analysis/mla_perf_sweep.py::test_mla_sweep" mp
export CK_STARTS=4096 CK_ITERS=3
run_pair chunked_start4096 "analysis/chunked_sweep.py::test_chunked_prefill"
$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=1024 $SD/run_zone.sh t23r_causal_S1024_q128k128_zoff
SDPA_CAUSAL=1 SDPA_QCHUNK=128 SDPA_KCHUNK=128 SDPA_SEQ=16384 $SD/run_zone.sh t23r_causal_S16384_q128k128_zoff
echo "### T2.3 regimes done $(date -u +%H:%M:%SZ)"
