#!/usr/bin/env bash
# Scratch op-level A/B (not for merge): ttnn.tilize cases of ol/test_tz_ol.py under the device profiler, arms swapped as
# device files into /work and the installed package (ci_ab/files.txt, ci_ab/v/<arm>), fresh JIT cache per arm, order
# <arms> forward then backward, twice. The last arm is the A/A copy of the one before it. Prints RAB OPC lines and ELF hashes.
# usage: run_ol.sh "<arms>"  (e.g. "M B X3 X3b"; an arm ending in b uses the files of the arm without it)
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-6} AB_NODES_FILE=/work/ci_ab/ol/nodes.txt
S=/work/ci_ab; OUT=/work/generated/tz; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-60) package root $SP $(hostname)"
OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); done < $S/files.txt
echo "== package copies found: $(for r in "${!PK[@]}"; do echo ${PK[$r]}; done | wc -w) for $(wc -l < $S/files.txt) files"
ls /work/ci_ab/ol/test_tz_ol.py > /dev/null && cp /work/ci_ab/ol/test_tz_ol.py /work/tests/ttnn/unit_tests/operations/data_movement/test_tz_ol.py
grep -h "^    \"" /work/ci_ab/ol/test_tz_ol.py | sed -n 's/^    "\([^"]*\)".*/tests\/ttnn\/unit_tests\/operations\/data_movement\/test_tz_ol.py::test_tz[case=\1-device_params={\x27l1_small_size\x27: 32768}]/p' > /work/ci_ab/ol/nodes.txt
echo "== nodes $(wc -l < /work/ci_ab/ol/nodes.txt): $(head -1 /work/ci_ab/ol/nodes.txt)"
python3 -c "import ttnn, tracy; print('== imports ok')" 2>&1 | tail -2
use() {  # use <arm>
  local v=${1%b}
  while read -r rel; do
    [[ -f $S/v/$v/$rel ]] || continue
    cp "$S/v/$v/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/v/$v/$rel" "$p"; done
    if [[ -z ${PK[$rel]} && $rel == tt_metal/* && -d $SP/ttnn/tt_metal ]]; then mkdir -p "$SP/ttnn/$(dirname $rel)"; cp "$S/v/$v/$rel" "$SP/ttnn/$rel"; fi
  done < $S/files.txt
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
python3 -m pytest --collect-only -q -p no:cacheprovider -o addopts= tests/ttnn/unit_tests/operations/data_movement/test_tz_ol.py 2>&1 | tail -8 | cut -c1-200
use ${DRY_ARM:-M}; timeout 900 python3 -m pytest -q -p no:cacheprovider -o addopts= "$(head -1 /work/ci_ab/ol/nodes.txt)" > /tmp/dry.txt 2>&1; echo "== dry run ${DRY_ARM:-M} rc=$?"; grep -a -v "^  File" /tmp/dry.txt | grep -a -i -E "error|throw|fatal|what|abort|fail" | head -30 | cut -c1-300
elfsum() { for e in $(ls $1/*/kernels/*/*/trisc*/*.elf 2>/dev/null | grep -i tiliz); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-10; done | sort | md5sum | cut -c1-10; }
RV=$(echo $1 | tr ' ' '\n' | tac | tr '\n' ' ')
declare -A KDONE
r=0
for blk in ${OL_BLOCKS:-1 2}; do for a in ${OL_ARMS:-$1 $RV}; do
  r=$((r+1)); use $a; o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/ol/run_nodes.py > $o.log 2>&1
  echo "== run $r $a rc=$? $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  grep -a -m3 -E '^(FAILED|ERROR)' $o.log | cut -c1-200
  grep -aqE '[0-9]+ passed' $o.log || { echo "== log errors $a r$r"; tail -25 $o.log | cut -c1-300; grep -a -B30 -m1 "Fatal Python error" $o.log | grep -a -v "^  File" | head -24 | cut -c1-300; }
  python3 $S/ol/sum_op19.py $o $a $r
  echo "RAB ELF $a r$r $(elfsum $TT_METAL_CACHE) $(ls -d $TT_METAL_CACHE/*/kernels/*/* 2>/dev/null | grep -ci tiliz) tilize builds"
  if [[ -z ${KDONE[$a]} ]]; then KDONE[$a]=1  # per compute kernel build: md5 of each TRISC's .text
    for e in $(ls $TT_METAL_CACHE/*/kernels/*/*/trisc*/*.elf 2>/dev/null); do
      $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && echo "RAB ELFK $a $(echo $e | awk -F/ '{print $(NF-3)"/"$(NF-2)"/"$(NF-1)}') $(md5sum < /tmp/e.bin | cut -c1-10)"
    done
  fi
done; done
for a in $1; do sort $OUT/cache_$a/sha.txt 2>/dev/null | uniq -c | sed "s/^ */RAB SHA $a /"; done
echo "== DONE $(date -u +%T)"
