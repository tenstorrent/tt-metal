#!/usr/bin/env bash
# Scratch A/B job (#60163 against main), not for merge. The job's tree is main. The DeepSeek sampling micro-op's three
# top-K cases of test_sampling_topk_single_device under the device profiler with main's kernel files (M), the PR head's
# (S) and an A/A copy of S (S2); both arms run the PR's test file (the Blackhole skip dropped). Each arm has its own
# fresh JIT cache, its files are copied into the checkout (and the installed package where it has them) before every
# run, M S S2 S2 S M in two blocks, R3_REPS calls per test (the first a warm-up; main selects the wrong token, so its
# runs report the failed assert after every call was measured). Guard: the sampling kernel's trisc .text hashes of the
# arm's cache must be the same in all of the arm's runs, S equal S2, M different from S.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-6} AB_NODES_FILE=/work/ci_ab/nodes.txt
S=/work/ci_ab; OUT=/work/generated/ab; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') runtime root $RT $(hostname)"
OBJ=/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy; [[ -x $OBJ ]] || OBJ=$(which riscv-tt-elf-objcopy 2>/dev/null)
[[ -x $OBJ ]] || OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
while read -r rel; do echo "RAB FILE $rel package copy: $([[ -f $RT/$rel ]] && echo yes || echo no); tree equals main: $(git show HEAD:$rel | cmp -s - /work/$rel && echo yes || echo no)"; done < $S/files.txt
use() {  # use <arm> <files>
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"; [[ -f $RT/$rel ]] && cp "$S/v/$2/$rel" "$RT/$rel"
  done < $S/files.txt
  echo "RAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {  # .text hash of every sampling_kernel trisc ELF of a cache, order independent
  for e in $(ls $1/*/kernels/sampling_kernel/*/trisc*/*.elf 2>/dev/null); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12
  done | sort | md5sum | cut -c1-12
}
declare -A ES
r=0; fail=0
for blk in 1 2; do for a in M S S2 S2 S M; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ ${a%2} == S && $rc -ne 0 ]] && { fail=1; grep -a -m5 -E '^(FAILED|ERROR)|Error' $o.log | cut -c1-200; }
  python3 $S/sum_op.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"
  echo "RAB ELF $a r$r $e $(ls $TT_METAL_CACHE/*/kernels/sampling_kernel 2>/dev/null | wc -l) builds"
done; done
uniq1() { echo $1 | tr ' ' '\n' | sort -u | wc -l; }
g1=$([[ $(uniq1 "${ES[M]}") == 1 && $(uniq1 "${ES[S]}") == 1 && ${ES[M]# } != ${ES[S]# } && -n $OBJ ]] && echo ok || echo VOID)
g2=$([[ $(uniq1 "${ES[S]} ${ES[S2]}") == 1 ]] && echo ok || echo DIFF)
echo "RAB GUARD M_vs_S=$g1 S_vs_S2=$g2"; [[ $g1 == ok && $g2 == ok ]] || fail=1
use M M
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
