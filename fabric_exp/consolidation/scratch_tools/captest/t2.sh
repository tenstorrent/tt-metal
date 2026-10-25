cd /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/captest
out=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/captest; CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
trap 'echo ORIGINAL_EXIT_TRAP' EXIT
source /home/aagarwal/tt-metal/fabric_exp/consolidation/vc1_capture_lib.sh
_vc1cap_capture() { echo "STUB capture -> $1 (args: $*)"; }
VC1CAP_SPLIT_TIMEOUT=5
run_vc1_cap t2lab /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/captest/cache2 O3; echo "run_vc1_cap returned $?"
trap -p EXIT
echo "left: $(ps -u $(id -u) -o args= | grep -c '^sleep 30[12]$')"
