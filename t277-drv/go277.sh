#!/bin/bash
# From g15blx02: copy bundle + scripts to blx01 under temp names, mv into place, start the driver detached.
# A/B pair, one process each, both profiled: default (edge order on) vs off (DIFFVAE_NA_EDGE_ORDER=0).
set -eo pipefail
H=g15blx01; D=/var/tmp/fasth3/t277/drv; S=$(dirname "$0"); REV=a5a774ea17f
ssh $H "mkdir -p $D"
ssh $H "cp /var/tmp/fasth3/t274/drv/decode261.py $D/.decode261.py.new && mv $D/.decode261.py.new $D/decode261.py"
for f in t277.bundle run277.sh driver277.sh build277.sh probe.sh env.yaml; do scp -q $S/$f $H:$D/.$f.new; ssh $H "mv $D/.$f.new $D/$f"; done
ssh $H "rm -f $D/driver.marker $D/job.id; setsid nohup bash $D/driver277.sh $REV 'default: off:DIFFVAE_NA_EDGE_ORDER=0' /var/tmp/fasth3/t277/out 240 'default off' > $D/driver.out 2>&1 < /dev/null &"
echo "started driver for $REV"
