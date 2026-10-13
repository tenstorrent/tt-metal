#!/usr/bin/env bash
# Bits of the row max on this merge: H (the previous head's row max) against V7 (this head), exact-reference comparison;
# the partial scaler check on H, V7 and V1 (the zero flag held at preserve for the whole tile).
source $(dirname $0)/common.sh
unset HR_POW2
for side in H V7 V1; do
  use $side
  if [ $side != V1 ]; then
    t0=$(date +%s); timeout -s INT -k 60 1500 python3 $S/tools/bits.py run $side $O/bits_$side.pt > $O/bits_$side.log 2>&1
    echo "== bits $side rc=$? $(( $(date +%s) - t0 )) s $(grep -c '^CASE' $O/bits_$side.log) cases"; grep -E "TT_THROW|TT_FATAL|Error|error:" $O/bits_$side.log | head -5 | cut -c1-300
  fi
  timeout -s INT -k 30 900 python3 $S/tools/partial.py $side 2>&1 | grep -E "^PARTIAL|Error|TT_THROW" | head -10
done
a=$(find /tmp/hr_cache_H -name 'trisc1*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16)
b=$(find /tmp/hr_cache_V7 -name 'trisc1*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16)
echo "== guard H $a V7 $b"; [ "$a" != "$b" ] || { echo "== VOID A/B: H and V7"; exit 1; }
use V7
python3 $S/tools/bits.py cmp H=$O/bits_H.pt V7=$O/bits_V7.pt 2>&1 | grep -v "  EX \|  WORSE " | sed 's/^/BITS /'
exit 0
