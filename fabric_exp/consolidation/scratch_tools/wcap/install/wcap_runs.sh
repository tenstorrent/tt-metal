#!/bin/bash
# Worker hang-capture self-test (HANG_CAPTURE.md): as p1cap_runs.sh selftest (one VC1 sanity run, fresh cache,
# then the read-only capture of the now idle device and its decode), with the current lead v3v2 at -O3 and its own
# label, so results/v3l3_o3_capselftest_hangcap (the router-only reference) is kept.
#   selftest  label wcapv3v2_o3_capselftest -> fabric_exp/results/wcapv3v2_o3_capselftest_hangcap/{index.json,decode.txt}
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
V=fabric_exp/variants
case "$1" in
selftest) vc1_capture_selftest wcapv3v2 $V/v3v2 O3 ;;
*)        echo "usage: wcap_runs.sh selftest"; exit 1 ;;
esac
rm -f $M
echo ALL_DONE
