#!/usr/bin/env bash
# Scratch A/B job (#60072 against main), not for merge. The job's tree is main. Arm M is main's device files, arm H the
# files of the PR's device diff (compute API headers, Blackhole sfpu ckernels, ckernel_defs.h), arm H2 an A/A copy of H;
# each arm has its own fresh JIT cache, its files are copied into the checkout and the installed package before every
# run, M H H2 H2 H M twice. Op level: every case in its own tracy process (prof_bin.py, accurate mode, 2 warm-up and 6
# measured calls), the device kernel durations of the measured ops. ELF guard: the .text of every kernel ELF of an
# arm's cache must be the same in all of the arm's runs, H must equal H2, and M must differ from H.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/sab; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') runtime root $RT $(hostname)"
CASES="add_bf16_hs sub_bf16_hs rsub_bf16_hs mul_bf16_hs mul_f32_hs add_bf16_dram_1024x1024 mul_bf16_dram_1024x1024"
nopkg=0
while read -r rel; do
  [[ -f $RT/$rel ]] || { nopkg=$((nopkg+1)); echo "RAB NOPKG $rel"; }
  cmp -s /work/$rel $S/v/M/$rel || echo "RAB TREE_NOT_MAIN $rel"
done < $S/files.txt
echo "RAB FILES $(wc -l < $S/files.txt) files, $nopkg without a package copy"
use() {  # use <arm> <files>
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"; [[ -f $RT/$rel ]] && cp "$S/v/$2/$rel" "$RT/$rel"
  done < $S/files.txt
  echo "RAB USE $1 $2 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8) $(while read -r rel; do [[ -f $RT/$rel ]] && cat $RT/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
OBJ=/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy; [[ -x $OBJ ]] || OBJ=$(which riscv-tt-elf-objcopy 2>/dev/null)
[[ -x $OBJ ]] || OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
elfsum() {  # name, processor and .text hash of every kernel ELF of a cache, order independent
  for e in $(ls $1/*/kernels/*/*/*/*.elf 2>/dev/null); do
    p=$(basename $(dirname $e)); n=$(basename $(dirname $(dirname $(dirname $e))))
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && echo "$n $p $(md5sum < /tmp/e.bin | cut -c1-12)"
  done | sort | md5sum | cut -c1-12
}
declare -A ES
r=0; fail=0
for blk in 1 2; do for a in M H H2 H2 H M; do
  r=$((r+1)); use $a ${a%2}; t0=$(date +%s)
  for c in $CASES; do
    o=$OUT/${a}_r${r}_$c; mkdir -p $o
    TT_METAL_PROFILER_DIR=$o/pdir R3_NMEAS=6 timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $o $S/prof_bin.py $c > $o.log 2>&1
    python3 $S/sb_sum.py $o $a $r $c
  done
  echo "RAB RUN $a r$r $(cat $OUT/${a}_r${r}_*.log | grep -ac '^CASE .* ok$') of $(echo $CASES | wc -w) cases ok"
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"
  echo "RAB ELF $a r$r $e $(ls -d $TT_METAL_CACHE/*/kernels/*/* 2>/dev/null | wc -l) builds, run $(( $(date +%s) - t0 )) s"
done; done
uniq1() { echo $1 | tr ' ' '\n' | sort -u | wc -l; }
g1=$([[ $(uniq1 "${ES[M]}") == 1 && $(uniq1 "${ES[H]}") == 1 && ${ES[M]# } != ${ES[H]# } && -n $OBJ ]] && echo ok || echo VOID)
g2=$([[ $(uniq1 "${ES[H]} ${ES[H2]}") == 1 && $(uniq1 "${ES[M]}") == 1 ]] && echo ok || echo DIFF)
echo "RAB GUARD M_vs_H=$g1 H_vs_H2_and_stable=$g2"; [[ $g1 == ok && $g2 == ok ]] || fail=1
use M M
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
