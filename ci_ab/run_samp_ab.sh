#!/usr/bin/env bash
# Scratch A/B job (exp_other regression audit, ra_samp), not for merge. The DeepSeek sampling micro-op's four argmax
# cases under the device profiler with main's files (M), the #60171 head's (R) and an A/A copy of R (R2), each arm with
# its own fresh JIT cache, its files swapped into the checkout and the installed package, M R R2 R2 R M in two blocks,
# R3_REPS calls per test (the first a warm-up). The sampling kernel's trisc .text sums per arm are the swap control.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-4} AB_NODES_FILE=/work/ci_ab/nodes.txt
S=/work/ci_ab; OUT=/work/generated/ab; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {  # use <arm> <files>: the files into the checkout and every copy under the package root, the arm's own JIT cache
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/v/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "RAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(ls $1/*/kernels/sampling_kernel/*/trisc*/*.elf 2>/dev/null | sort); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin; done | md5sum | cut -c1-12; }
r=0; fail=0
for blk in 1 2; do for a in M R R2 R2 R M; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1200 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E '^(FAILED|ERROR)|Error' $o.log | cut -c1-200; }
  python3 $S/sum_op.py $o $a $r
  [[ $r -le 3 ]] && echo "RAB ELF $a $(elfsum $TT_METAL_CACHE) $(ls $TT_METAL_CACHE/*/kernels/sampling_kernel 2>/dev/null | wc -l) builds"
done; done
use M M
echo "== done $(date -u +%FT%TZ)"
exit $fail
