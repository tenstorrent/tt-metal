#!/bin/bash
# t251: stage the t220 LTX-2.3 bf8 tree + caches on blx01 under /var/tmp/fasth3/t220 (nothing under blx01 /home).
set -eo pipefail
D=/var/tmp/fasth3/t220; C=$D/cache/dit-ltx23
ssh g15blx01 "mkdir -p $C $D/cache/tt-metal-cache-ltx23 $D/tmp && cp -a /home/sulphur/tt_dit_cache/gemma-3-12b-it-qat-q4_0-unquantized $C/" ; echo "gemma rc=$?"
ssh g14blx03 "tar -C /home/smarton/fasth3 -cf - --exclude=t220/.git t220" | ssh g15blx01 "mkdir -p $D/x && tar -C $D/x -xf - && rm -rf $D/src && mv $D/x/t220 $D/src && rmdir $D/x"; echo "src rc=$?"
ssh g14blx03 "tar -C /var/tmp/fasth3/cache/dit-ltx23 -cf - ltx-2.3-22b-distilled-1.1 ltx-2.3-22b-distilled-1.1.q-all_bf8_lofi ltx-2.3-spatial-upscaler-x2-1.1" | ssh g15blx01 "tar -C $C -xf -"; echo "dit rc=$?"
ssh g14blx03 "tar -C /var/tmp/fasth3/cache/tt-metal-cache-ltx23 -cf - ." | ssh g15blx01 "tar -C $D/cache/tt-metal-cache-ltx23 -xf -"; echo "jit rc=$?"
ssh g15blx01 "du -sh $D/src $C/* $D/cache/tt-metal-cache-ltx23; df -h /var/tmp | tail -1"
echo COPY251_DONE
