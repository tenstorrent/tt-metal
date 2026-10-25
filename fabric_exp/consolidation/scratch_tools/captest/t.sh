cd /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/captest
out=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/captest; CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
source /home/aagarwal/tt-metal/fabric_exp/consolidation/vc1_capture_lib.sh
VC1CAP_RUN_TIMEOUT=6
echo "own sid $(ps -o sid= -p $$ | tr -d ' ')"
_vc1cap_run /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/captest/run.log /tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/captest/cache O3
echo "rc=$VC1CAP_RC hung=$VC1CAP_HUNG sid=$VC1CAP_SID secs=$VC1CAP_SECS"
echo "left: $(ps -u $(id -u) -o pid=,comm=,args= | grep -c 'sleep 30[12]$')"
