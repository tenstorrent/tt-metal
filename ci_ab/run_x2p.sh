#!/usr/bin/env bash
# Scratch job (#60291, topk_xl X2 rotating survivor in the production split), not for merge. topk_large_indices at the K' 2048
# production shapes (and V4 Pro K' 1024 controls): A = #60291 head df27d5a61c9, B = the port (2c051160bf3), C = the port
# only in programs without the segmented body (used only if B does not fit). Fit and bits first (indices of every case,
# A against the chosen arm X), then device time under the device profiler, own fresh JIT cache per arm,
# A X X2 A2 A2 X2 X A twice, then test_topk_large_indices.py on X, the ELF guard and the push rule.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 OPS_NMEAS=${OPS_NMEAS:-8} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/x2p; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); SIZE=${OBJ%objcopy}size
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {  # use <arm>: the arm's variant files (A2 uses A's) into the checkout and the package, the arm's own JIT cache
  local v=${1%2}
  while read -r rel; do
    cp "$S/variants/$v/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants/$v/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "X2P FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfs() { find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort; }
elfsum() { for e in $(elfs $1); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
sizes() { for t in 0 1 2; do echo -n "trisc$t max text $(for e in $(elfs $1 | grep "/trisc$t/"); do $SIZE -A "$e" | awk '$1 ~ /^\.text/ {s+=$2} END {print s}'; done | sort -n | tail -1) "; done; }
dump() {  # dump <arm>: every case's indices, no profiler; the fit check
  use $1; export TT_METAL_CACHE=$OUT/cache_$1_dump; mkdir -p $TT_METAL_CACHE
  TLI_DUMP=$OUT/dump_$1.pt timeout -s INT -k 60 1800 python3 $S/prof_x2p.py > $OUT/dump_$1.log 2>&1
  echo "== dump $1 rc=$? $(grep -a 'DUMP DONE' $OUT/dump_$1.log) $(grep -ac 'CASE .* FAILED' $OUT/dump_$1.log) failed; $(sizes $TT_METAL_CACHE)"
  grep -a 'CASE .* FAILED' $OUT/dump_$1.log | cut -c1-400
}
fail=0
dump A; grep -aq 'CASE .* FAILED' $OUT/dump_A.log && { echo "X2P FIT A fails: the head itself"; fail=1; }
dump B; X=B
if grep -aq 'CASE .* FAILED' $OUT/dump_B.log; then
  echo "X2P FIT B fails: C, the rotation only in programs without the segmented body"
  dump C; X=C
  grep -aq 'CASE .* FAILED' $OUT/dump_C.log && { echo "X2P FIT C fails too, stop (a Program size line means it does not fit)"; X=; }
fi
[[ -z $X ]] && { echo "X2P RULE arm none: push no (B and C fail)"; exit 1; }
echo "X2P FIT arm $X"
python3 $S/tli_cmp.py $OUT/dump_A.pt $OUT/dump_$X.pt | tee $OUT/bits.txt
declare -A ELF
r=0
for blk in 1 2; do for a in A $X ${X}2 A2 A2 ${X}2 $X A; do
  r=$((r+1)); use $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/prof_x2p.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-400; }
  grep -a 'CASE .* FAILED' $o.log | head -5 | cut -c1-400
  python3 $S/sum_ops.py $o $a $r | tee -a $OUT/ops.txt
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "X2P ELF $a ${ELF[$a]} $(elfs $TT_METAL_CACHE | wc -l) ELFs; $(sizes $TT_METAL_CACHE)"; fi
done; done
use $X; export TT_METAL_CACHE=$OUT/cache_T$X; mkdir -p $TT_METAL_CACHE
t0=$(date +%s); timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py -k "not production_perf" > $OUT/ut_$X.log 2>&1
trc=$?
echo "== test_topk_large_indices.py on $X rc=$trc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $OUT/ut_$X.log | tail -1)"
grep -aE '^(FAILED|ERROR)' $OUT/ut_$X.log | head -10 | cut -c1-240
use A
nd=$(comm -3 <(for e in $(elfs $OUT/cache_A); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort) \
            <(for e in $(elfs $OUT/cache_$X); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort) | wc -l)
echo "X2P GUARD A=${ELF[A]} A2=${ELF[A2]} $X=${ELF[$X]} ${X}2=${ELF[${X}2]}; .text images in only one of A and $X: $nd"
guard=ok
[[ ${ELF[A]} == "${ELF[$X]}" ]] && { echo "X2P GUARD FAIL: A and $X built the same kernels"; guard=fail; }
[[ ${ELF[A]} != "${ELF[A2]}" || ${ELF[$X]} != "${ELF[${X}2]}" ]] && { echo "X2P GUARD FAIL: the A/A arms built different kernels"; guard=fail; }
python3 $S/x2p_table.py $OUT/ops.txt $X | tee $OUT/table.md
n=$(python3 -c "import torch; print(len(torch.load('$OUT/dump_A.pt')))")
bits=$(grep -q "TLI CMP total $n of $n identical" $OUT/bits.txt && echo ok || echo fail)
tests=$([[ $trc -eq 0 ]] && grep -aqE '[0-9]+ passed' $OUT/ut_$X.log && ! grep -aqE '[0-9]+ (failed|error)' $OUT/ut_$X.log && echo ok || echo fail)
perf=$(grep -q 'X2P PERF rows [0-9]* slower 0 .*missing 0' $OUT/table.md && echo ok || echo fail)
gain=$(grep -E '^\| (prefill|bounded|decode|glm|dsv3|k2048)' $OUT/table.md | grep -c '| faster |')
push=$([[ $bits == ok && $tests == ok && $perf == ok && $guard == ok && $gain -gt 0 && $fail -eq 0 ]] && echo yes || echo no)
echo "X2P RULE arm $X bits $bits tests $tests perf $perf guard $guard K' 2048 rows faster $gain runs_ok $([[ $fail -eq 0 ]] && echo yes || echo no): push $push"
echo "== done $(date -u +%FT%TZ)"
[[ $push == yes ]] || exit 1
