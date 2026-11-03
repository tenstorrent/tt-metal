#!/bin/bash
# mvc3_gate.sh LABEL-PREFIX: exit 0 only when vc1vc2_runs.sh's correctness runs under LABEL-PREFIX (e.g. vcvc2g_v4rxg3v)
# all exist, have no error line, no hang capture, and the data-checked vc1_alloc runs finished all 50 tests over both
# ranks. Queued multi-VC jobs (mvc3_runs.sh) run behind it, so a failed correctness job stops them before any device work.
cd /home/aagarwal/tt-metal/fabric_exp/results || exit 1
L=$1; ok=1
for r in alloc inter alloc_1e vc2 vc2_1e; do
    f=${L}_${r}_13129088_vc1.log
    [ -f $f ] || { echo "gate: $f missing"; ok=0; continue; }
    bad=$(sed 's/\x1b\[[0-9;]*m//g' $f | grep -acE 'TT_FATAL|TT_THROW|HUNG|timed out|Aborted|mismatch')
    fin=$(sed 's/\x1b\[[0-9;]*m//g' $f | grep -ac 'Finished\. (test_tt_fabric')
    echo "gate: $r: $fin finishes, $bad error lines"
    [ "$bad" = 0 ] && [ "$fin" -gt 0 ] || ok=0
    case $r in alloc|alloc_1e) [ "$fin" = 50 ] || ok=0 ;; esac
done
ls -d ${L}_*hangcap 2>/dev/null && ok=0
[ $ok = 1 ] && echo "gate: open" && exit 0
echo "gate: CLOSED (correctness runs not clean)"; exit 1
