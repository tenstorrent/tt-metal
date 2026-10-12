#!/bin/bash
# t421 post step (detached on g15blx02, no device): waits for the chain's rc file, then VBench-scores
# every seed with both arms (on as candidate, off as reference) with the ltx_eval harness exported to
# evaltree/ from c2d3eda64a1. Writes POST421_DONE <rc> to post.marker.
set -eo pipefail
D=/home/smarton/fasth3/tt-metal/tt-project/state/t421
RC=${1:?chain rc file}
trap 'echo "POST421_DONE $? $(date -u +%FT%TZ)" > $D/post.marker' EXIT
while [ ! -s "$RC" ]; do sleep 300; done
echo "chain rc $(cat $RC)" > $D/vbench.log
rm -rf $D/vb && mkdir -p $D/vb/off $D/vb/on
n=0
for s in 0 1 2 3 4; do
  if [ -s $D/mp4/off-s$s.mp4 ] && [ -s $D/mp4/on-s$s.mp4 ]; then
    ln -s $D/mp4/off-s$s.mp4 $D/vb/off/s$s.mp4; ln -s $D/mp4/on-s$s.mp4 $D/vb/on/s$s.mp4; n=$((n+1))
  fi
done
echo "pairs: $n" >> $D/vbench.log
[ $n -gt 0 ]
cd $D/evaltree
LTX_EVAL_THREADS=8 timeout 5400 nice -n 10 /home/smarton/fasth3/tt-metal/python_env/bin/python -m models.tt_dit.tests.models.ltx.tools.ltx_eval \
  batch --cand-dir $D/vb/on --ref-dir $D/vb/off --out $D/vbench --vbench-ref --jobs 2 >> $D/vbench.log 2>&1
