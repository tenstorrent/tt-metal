#!/usr/bin/env bash
# Per-kernel ELF identity, H against V6, on this runner's architecture (one run per side, own kernel cache each).
source $(dirname $0)/common.sh
for side in H V6; do
  use $side; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 $S/tools/elf6.py $side 1 > $O/elf_$side.log 2>&1
  echo "== elf $side rc=$? $(( $(date +%s) - t0 )) s $(grep -c 'CASE .* ok' $O/elf_$side.log) ok $(grep -c 'CASE .* FAIL' $O/elf_$side.log) fail"; grep -E "CASE .* FAIL" $O/elf_$side.log | head -4 | cut -c1-300
  grep -E "^CASE .* ok BITS" $O/elf_$side.log | sed 's/^/   /'
done
use H
python3 $S/tools/kelf.py /tmp/hr_cache_H /tmp/hr_cache_V6 H V6 2>&1 | head -80
exit 0
