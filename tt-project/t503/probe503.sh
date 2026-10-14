#!/bin/bash
# exit 0 once blx03's runner finished t503-ab-r2 (r1 runs first); 1 while pending; 255 ssh down
timeout 40 ssh -o ConnectTimeout=8 -o BatchMode=yes g14blx03 'test -e /var/tmp/fasth3/runner/done/t503-ab-r2.done'; r=$?
[ $r = 0 ] && exit 0; [ $r = 1 ] && exit 1; exit 255
