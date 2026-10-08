#!/bin/bash
# Run 2 (from g15blx02): copy bundle + scripts to blx01 under temp names, mv into place, start the driver detached.
# A/B: base (default t48 path) vs edge (DIFFVAE_NA_EDGE_ORDER=1), one process each, both profiled, -t 450.
set -eo pipefail
H=g15blx01; D=/var/tmp/fasth3/t274/drv; S=$(dirname "$0"); REV=$(sed -n 's/^REV=${REV:-\([0-9a-f]*\)}$/\1/p' $S/run274.sh)
ssh $H "mkdir -p $D"
for f in t274.bundle run274.sh driver274.sh build274.sh probe.sh; do scp -q $S/$f $H:$D/.$f.new; ssh $H "mv $D/.$f.new $D/$f"; done
ssh $H "rm -f $D/driver.marker $D/job.id; setsid nohup bash $D/driver274.sh $REV 'base: edge:DIFFVAE_NA_EDGE_ORDER=1' /var/tmp/fasth3/t274/out_E 450 'base edge' > $D/driver.out 2>&1 < /dev/null &"
echo "started driver for $REV"
