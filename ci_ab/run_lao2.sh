#!/usr/bin/env bash
# Scratch job (#60291 topk_xl levers, job 2), not for merge: topk_large_indices at the split's K' 512, 1024 and 2048 rows.
# A = #60291 head 36805d6f18e; each arm of $ARMS a topk_large_indices_split.hpp variant (variants_o2). Indices of every
# case A against each arm, then device time under the device profiler, own fresh JIT cache per arm, A X.. X2.. A2 A2
# X2.. X.. A twice, then test_topk_large_indices.py on each arm, the ELF guard, one table and rule line per arm.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 OPS_NMEAS=${OPS_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/la; ARMS=${ARMS:-C CD}; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); SIZE=${OBJ%objcopy}size
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files_o.txt
use() {  # use <arm>: the arm's variant files (A2 uses A's) into the checkout and the package, the arm's own JIT cache
  local v=${1%2}
  while read -r rel; do
    cp "$S/variants_o2/$v/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants_o2/$v/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files_o.txt
  echo "LA FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files_o.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfs() { find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort; }
elfsum() { for e in $(elfs $1); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
sizes() { for t in 0 1 2; do echo -n "trisc$t max text $(for e in $(elfs $1 | grep "/trisc$t/"); do $SIZE -A "$e" | awk '$1 ~ /^\.text/ {s+=$2} END {print s}'; done | sort -n | tail -1) "; done; }
dump() {  # dump <arm>: every case's indices, no profiler
  use $1; export TT_METAL_CACHE=$OUT/cache_$1_dump; mkdir -p $TT_METAL_CACHE
  TLI_DUMP=$OUT/dump_$1.pt timeout -s INT -k 60 2400 python3 $S/prof_la.py > $OUT/dump_$1.log 2>&1
  echo "== dump $1 rc=$? $(grep -a 'DUMP DONE' $OUT/dump_$1.log) $(grep -ac 'CASE .* FAILED' $OUT/dump_$1.log) failed; $(sizes $TT_METAL_CACHE)"
  grep -a 'CASE .* FAILED' $OUT/dump_$1.log | cut -c1-400
}
fail=0
dump A; grep -aq 'CASE .* FAILED' $OUT/dump_A.log && { echo "LA FIT A fails: the head itself"; fail=1; }
for X in $ARMS; do
  dump $X; grep -aq 'CASE .* FAILED' $OUT/dump_$X.log && { echo "LA FIT $X fails"; fail=1; }
  python3 $S/tli_cmp.py $OUT/dump_A.pt $OUT/dump_$X.pt | sed "s/^TLI CMP/TLI CMP $X/" | tee $OUT/bits_$X.txt
done
declare -A ELF
FWD="A"; for X in $ARMS; do FWD="$FWD $X"; done; for X in $ARMS; do FWD="$FWD ${X}2"; done; FWD="$FWD A2"
REV=$(echo $FWD | tr " " "\n" | tac | tr "\n" " ")
r=0
for blk in 1 2; do for a in $FWD $REV; do
  r=$((r+1)); use $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/prof_la.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-400; }
  grep -a 'CASE .* FAILED' $o.log | head -5 | cut -c1-400
  python3 $S/sum_ops.py $o $a $r | tee -a $OUT/ops.txt
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "LA ELF $a ${ELF[$a]} $(elfs $TT_METAL_CACHE | wc -l) ELFs; $(sizes $TT_METAL_CACHE)"; fi
done; done
n=$(python3 -c "import torch; print(len(torch.load('$OUT/dump_A.pt')))")
for X in $ARMS; do
  use $X; export TT_METAL_CACHE=$OUT/cache_T$X; mkdir -p $TT_METAL_CACHE
  t0=$(date +%s); timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py -k "not production_perf" > $OUT/ut_$X.log 2>&1
  trc=$?
  echo "== test_topk_large_indices.py on $X rc=$trc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut_$X.log | tail -1)"
  grep -aE '^(FAILED|ERROR)' $OUT/ut_$X.log | head -10 | cut -c1-240
  guard=ok
  [[ ${ELF[A]} == "${ELF[$X]}" ]] && { echo "LA GUARD FAIL: A and $X built the same kernels"; guard=fail; }
  [[ ${ELF[A]} != "${ELF[A2]}" || ${ELF[$X]} != "${ELF[${X}2]}" ]] && { echo "LA GUARD FAIL: the A/A arms built different kernels"; guard=fail; }
  echo "LA GUARD $X A=${ELF[A]} A2=${ELF[A2]} $X=${ELF[$X]} ${X}2=${ELF[${X}2]} $guard"
  python3 $S/la_table.py $OUT/ops.txt $X | tee $OUT/table_$X.md
  bits=$(grep -q "TLI CMP $X total $n of $n identical" $OUT/bits_$X.txt && echo ok || echo fail)
  tests=$([[ $trc -eq 0 ]] && grep -aqE '[0-9]+ passed' $OUT/ut_$X.log && ! grep -aqE '[0-9]+ (failed|error)' $OUT/ut_$X.log && echo ok || echo fail)
  perf=$(grep -q 'LA PERF rows [0-9]* slower 0 .*missing 0' $OUT/table_$X.md && echo ok || echo fail)
  gain=$(grep -c '| faster |' $OUT/table_$X.md)
  echo "LA RULE $X bits $bits tests $tests perf $perf guard $guard rows faster $gain runs_ok $([[ $fail -eq 0 ]] && echo yes || echo no)"
done
echo "== done $(date -u +%FT%TZ)"
