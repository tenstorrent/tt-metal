#!/usr/bin/env bash
# ROW MAX output bits for main's device files (MAIN), the head (H) and the held-flag forms (V1, V2); exact-reference comparison.
source $(dirname $0)/common.sh
for side in H MAIN V1 V2; do
  use $side
  t0=$(date +%s); timeout -s INT -k 60 1500 python3 $S/tools/bits.py run $side $O/bits_$side.pt > $O/bits_$side.log 2>&1
  echo "== bits $side rc=$? $(( $(date +%s) - t0 )) s $(grep -c '^CASE' $O/bits_$side.log) cases"; grep -E "TT_THROW|TT_FATAL|Error|error:" $O/bits_$side.log | head -5 | cut -c1-300
done
use H
guard H V1; guard H V2; guard H MAIN
python3 $S/tools/bits.py cmp H=$O/bits_H.pt MAIN=$O/bits_MAIN.pt V1=$O/bits_V1.pt V2=$O/bits_V2.pt 2>&1 | sed 's/^/BITS /'
exit 0
