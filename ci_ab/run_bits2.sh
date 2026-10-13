#!/usr/bin/env bash
# Bits of the row max: H, V7 (constant flag writes) and V6 with the kernel passing pow2_scaler (V6p), exact-reference
# comparison; then the partial scaler check on H and V6p.
source $(dirname $0)/common.sh
for side in H V7 V6p; do
  base=$side; [ $side = V6p ] && base=V6; use $base; export TT_METAL_CACHE=/tmp/hr_cache_$side; mkdir -p $TT_METAL_CACHE
  [ $side = V6p ] && export HR_POW2=1 || unset HR_POW2
  t0=$(date +%s); timeout -s INT -k 60 1500 python3 $S/tools/bits.py run $side $O/bits_$side.pt > $O/bits_$side.log 2>&1
  echo "== bits $side rc=$? $(( $(date +%s) - t0 )) s $(grep -c '^CASE' $O/bits_$side.log) cases"; grep -E "TT_THROW|TT_FATAL|Error|error:" $O/bits_$side.log | head -5 | cut -c1-300
  timeout -s INT -k 30 900 python3 $S/tools/partial.py $side 2>&1 | grep -E "^PARTIAL|Error|TT_THROW" | head -10
done
unset HR_POW2
a=$(find /tmp/hr_cache_H -name 'trisc1*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16)
for v in V7 V6p; do b=$(find /tmp/hr_cache_$v -name 'trisc1*.elf' | sort | xargs -r sha256sum | awk '{print $1}' | sort | sha256sum | cut -c1-16); echo "== guard H $a $v $b"; [ "$a" != "$b" ] || { echo "== VOID A/B: H and $v"; exit 1; }; done
use H
python3 $S/tools/bits.py cmp H=$O/bits_H.pt V7=$O/bits_V7.pt V6p=$O/bits_V6p.pt 2>&1 | grep -v "  EX \|  WORSE " | sed 's/^/BITS /'
exit 0
