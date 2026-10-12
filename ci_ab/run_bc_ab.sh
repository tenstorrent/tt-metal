#!/usr/bin/env bash
# Scratch A/B job (exp_matmul regression audit, ra_mm), not for merge. #60138: the 39 streaming SDPA calls of
# test_w3_sdpa.py at their single-device shapes with main 3ba039aee20's compute_streaming.hpp (M), #60138's (R) and an
# A/A copy of R (R2), each arm with its own fresh JIT cache, the file swapped into the checkout and the installed
# package, M R R2 R2 R M in two blocks, W3_CALLS calls per case (the first compiles). Each arm's trisc .text sum is the
# swap control.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 W3_CALLS=${W3_CALLS:-6} W3_PROF=1 AB_NODES_FILE=/work/ci_ab/nodes.txt
S=/work/ci_ab; OUT=/work/generated/ab; rm -rf $OUT; mkdir -p $OUT
cp $S/test_w3_sdpa.py /work/models/demos/deepseek_v3_b1/tests/unit_tests/test_w3_sdpa.py
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt "$HOME" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {  # use <arm> <variant>: the variant's files into the checkout and every copy under the package root, the arm's own cache
  while read -r rel; do
    cp "$S/v/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/v/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "RAB FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*trisc*' -name '*.elf' 2>/dev/null | grep -i sdpa | sort); do $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin; done | md5sum | cut -c1-12; }
r=0; fail=0
for blk in 1 2; do for a in M R R2 R2 R M; do
  r=$((r+1)); use $a ${a%2}
  o=$OUT/${a}_r$r; mkdir -p $o; rm -f $o.w3log; t0=$(date +%s)
  W3_LOG=$o.w3log timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/run_nodes.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E '^(FAILED|ERROR)|Error' $o.log | cut -c1-200; }
  python3 $S/rows.py $o $o.w3log $a $r
  [[ $r -le 3 ]] && echo "RAB ELF $a $(elfsum $TT_METAL_CACHE) $(find $TT_METAL_CACHE -path '*trisc*' -name '*.elf' | grep -ic sdpa) sdpa trisc elfs"
done; done
use M M
echo "== done $(date -u +%FT%TZ)"
exit $fail
