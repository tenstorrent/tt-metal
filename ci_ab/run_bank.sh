#!/usr/bin/env bash
# Scratch, not for merge: op device time of the DRAM-streaming matmuls against a compute-free twin (same reader).
set -uo pipefail
cd /work
U=models/demos/deepseek_v3_b1/unified_kernels
T=models/demos/deepseek_v3_b1/tests/unit_tests
cp ci_ab/test_cdram_ab.py $T/test_cdram_ab.py
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' 2>/dev/null)"
: > ci_ab/ids.txt
for f in up_proj down_proj; do echo "$T/test_dram_streaming_matmul.py::test_dram_streaming_matmul[not_fp32-m1-$f]" >> ci_ab/ids.txt; done
ls $T/test_cdram_ab.py | sed 's/$//' >> /dev/null
for f in test_cd_mix_n2048 test_cd_mix_n4096 test_cd_bfp4_n2048 test_cd_bfp4_n4096 test_cd_mix024_n2048; do echo "$T/test_cdram_ab.py::$f" >> ci_ab/ids.txt; done
cat ci_ab/ids.txt
md5sum ci_ab/real/*.hpp ci_ab/probe/*.hpp
cmp -s ci_ab/real/dram_streaming_matmul.hpp ci_ab/probe/dram_streaming_matmul.hpp && { echo "VOID: probe equals real"; exit 1; }
for f in dram_streaming_matmul.hpp dram_streaming_matmul_compressed.hpp; do cmp -s ci_ab/real/$f $U/$f || { echo "real differs from tree: $f"; exit 1; }; done
sudo -n true 2>/dev/null; cat /sys/devices/system/clocksource/clocksource0/current_clocksource >/dev/null 2>&1
declare -A H
for step in "real 1" "probe 1" "real2 1" "real2 2" "probe 2" "real 2"; do
  set -- $step; arm=$1; r=$2
  src=$arm; [[ $arm == real2 ]] && src=real
  cp ci_ab/$src/dram_streaming_matmul.hpp ci_ab/$src/dram_streaming_matmul_compressed.hpp $U/
  out=/work/generated/bank/${arm}_$r; rm -rf "$out" /tmp/ttcache_${arm}_$r; mkdir -p "$out"
  t0=$(date +%s)
  TT_METAL_CACHE=/tmp/ttcache_${arm}_$r IDS_FILE=/work/ci_ab/ids.txt PYTHONPATH=/work:/work/tools python3 -m tracy -p -r --no-web-server -o "$out" -m ci_ab.ab_run > "$out.log" 2>&1
  echo "== $arm r$r rc=$? $(( $(date +%s) - t0 )) s: $(grep -E '[0-9]+ (passed|failed)' "$out.log" | tail -1)"
  grep -q -E "passed|failed" "$out.log" || { echo NO_PYTEST_OUTPUT; head -c 3000 "$out.log"; }
  grep -E "Tile counts|per_core_N=|Kt=" "$out.log" | sort -u | head -12
  n=$(find /tmp/ttcache_${arm}_$r -name '*.elf' 2>/dev/null | wc -l); echo "elfs $n"
  h=$(find /tmp/ttcache_${arm}_$r -name '*.elf' -exec md5sum {} + 2>/dev/null | awk '{print $1}' | sort | md5sum | cut -c1-12)
  h0=$(find /tmp/ttcache_${arm}_$r -name '*.elf' -exec md5sum {} + 2>/dev/null | awk '{print $1}' | sort | md5sum | cut -c1-12)
  echo "ELFHASH $arm r$r $h"; H[$arm$r]=$h
  python3 ci_ab/parse.py "$out" $arm $r || tail -5 "$out.log"
done
echo "ELF real1=${H[real1]} real2a=${H[real21]} probe1=${H[probe1]}"
[[ ${H[real1]} != ${H[probe1]} ]] || { echo "VOID: compute ELFs identical in real and probe"; exit 1; }
grep -h "frequency\|AICLK\|MHz" generated/bank/real_1.log | head -3
exit 0
