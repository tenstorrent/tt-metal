#!/bin/bash
# Receiver start gate check (RESULTS §98; fabric_exp/streams/port/streams_rxready.patch, plus diag_launch_delay.patch
# for case A). VC1 sanity (vc1_sanity.yaml, two ranks) with the read-only hang capture armed, lead build at -O3.
#   A          deterministic: every rank but 0 launches 200 ms late (diagnostic env var), 2 runs; label p26diag
#   B TAG [N]  repeats with no delay, N runs (default 25); label p26rep<TAG>
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
V=fabric_exp/variants
case "$1" in
A) export TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS=200  # global rank > 0 launches 200 ms late (diag v2 logs it)
   vc1_repeat_capture v3v6 $V/v3v6 p27diag O3 2 ;;
B) vc1_repeat_capture v3v6 $V/v3v6 p27rep$2 O3 ${3:-25} ;;
*) echo "usage: rxready_runs.sh A | B TAG [N]"; exit 1 ;;
esac
rm -f $M
echo ALL_DONE
