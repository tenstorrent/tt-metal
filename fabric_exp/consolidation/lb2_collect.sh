#!/bin/bash
# Second-LB harness (LB2_HARNESS.md): bring a finished lb2 job's results into this tree (run on bh-lb-120-a09u32).
#   lb2_collect.sh JOBID
# Copies /data/aagarwal/lb2/jobs/JOBID/results/* into fabric_exp/results/ without overwriting anything: a file whose name
# already exists here is a label collision, left on /data and listed. Writes fabric_exp/results/lb2/JOBID_<node>.tsv
# (path, bytes, sha256/16, node, job, copied|COLLISION) and JOBID_<node>.txt (command, SUMMARY, the job log's lb2 lines).
# Saved builds and the router ELFs/headers of the job's caches stay on /data (jobs/JOBID/builds, caches/<cache>/): a
# PC-sample or offline analysis reads them from there.
set -uo pipefail
JID=${1:?usage: lb2_collect.sh JOBID}
G=/data/aagarwal/lb2
J=$G/jobs/$JID
R=/home/aagarwal/tt-metal/fabric_exp/results
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! run on bh-lb-120-a09u32"; exit 1; }
[ -f $J/SUMMARY ] || { echo "!!! job $JID has no SUMMARY yet (running, or it died before the end step)"; exit 1; }
node=$(grep -oE 'node=[^ ]+' $J/SUMMARY | cut -d= -f2); node=${node:-unknown}
mkdir -p $R/lb2
man=$R/lb2/${JID}_${node#bh-lb-120-}.tsv
[ -e $man ] && { echo "!!! $man exists: job $JID was collected already"; exit 1; }
n=0 c=0
: > $man.tmp
while IFS=$'\t' read -r f sz h nd jb; do
    case $f in results/*) ;; *) continue ;; esac
    rel=${f#results/}
    if [ -e "$R/$rel" ]; then
        st=COLLISION; c=$((c + 1)); echo "   COLLISION (kept on /data): $rel"
    else
        mkdir -p "$(dirname "$R/$rel")" && cp -p "$J/$f" "$R/$rel" && st=copied && n=$((n + 1)) || st=COPY_FAILED
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$sz" "$h" "$nd" "$jb" "$st" >> $man.tmp
done < $J/MANIFEST.tsv
mv $man.tmp $man
{ echo "job $JID on $node"; echo "command: $(cat $J/COMMAND 2>/dev/null)"; echo "summary: $(cat $J/SUMMARY)";
  log=$(ls $G/logs/lb2-*-$JID.log 2>/dev/null | head -1); [ -n "$log" ] && grep -a '^\[lb2 ' $log; } > ${man%.tsv}.txt
echo "job $JID ($node): $n result files copied into fabric_exp/results, $c collisions; manifest $man"
echo "   $(cat $J/SUMMARY)"
[ -d $J/caches ] && echo "   router ELFs/headers: $J/caches/ ($(ls $J/caches | wc -l) caches)"
[ -d $J/builds ] && echo "   saved builds: $J/builds/"
[ $c = 0 ]
