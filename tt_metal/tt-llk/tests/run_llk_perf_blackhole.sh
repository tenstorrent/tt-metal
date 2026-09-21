#!/usr/bin/env bash
# tt-smi PROBE -- find a spelling that prints or writes AICLK. No device work.
set -uo pipefail
GROUP="${1:?}"
N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then
  echo "probe: only group 1 runs; this group exits."
  exit 0
fi
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/tmp/smi
mkdir -p "$OUT"

echo "===== which tt-smi"
command -v tt-smi || echo "  not on PATH"
tt-smi --version 2>&1 | head -3

echo "===== --help"
tt-smi --help 2>&1 | head -60

echo "===== python module form"
python3 -c "import tt_smi, os; print('tt_smi at', os.path.dirname(tt_smi.__file__))" 2>&1 | head -3

for form in "-s" "-s -f $OUT/s1.json" "--snapshot" "-ls" "-l" "-f $OUT/s2.json"; do
  echo "===== tt-smi $form"
  # shellcheck disable=SC2086
  timeout 60 tt-smi $form >"$OUT/out.txt" 2>&1
  echo "  rc=$?"
  head -25 "$OUT/out.txt" | sed 's/^/  /'
done

echo "===== files tt-smi wrote anywhere obvious"
ls -la "$OUT" 2>&1 | head
find / -maxdepth 4 -name "*tt_smi*snapshot*" -newermt "-10 minutes" 2>/dev/null | head -5
find . -maxdepth 3 -newermt "-10 minutes" -name "*.json" 2>/dev/null | head -5

echo "===== sysfs / driver telemetry as a fallback"
ls /sys/class/tenstorrent 2>&1 | head
for d in /sys/class/tenstorrent/*; do
  echo "--- $d"
  ls "$d" 2>/dev/null | head -20
done
echo "===== probe done ====="
