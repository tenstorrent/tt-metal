#!/usr/bin/env bash
# Scratch A/B job (#60676 local head c88c1f2407c against main cb51ff54e9c), not for merge. The job's tree is the head with
# a scratch factory whose CI_AB_MAIN_FACTORY env builds main's program (every Blackhole-only choice off). Arms: M = main's
# gate kernels + CI_AB_MAIN_FACTORY=1, H = the head's, H2 = an A/A copy of H. Each arm its own fresh JIT cache, its
# files copied into the checkout and the installed package; M H H2 H2 H M in two blocks. Device kernel duration per
# call under the device profiler (gate_ops2.py, 16 model configurations, GATE_NMEAS calls each), printed as RAB OP lines.
# Guard: the arms' compute ELF .text hashes; the job fails if M equals H or H differs from H2.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 GATE_NMEAS=${GATE_NMEAS:-12} GATE_WARM=2 HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/gate; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-80)"
OBJ=$(find /work/runtime /opt /usr/local -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.abspath(ttnn.__file__)))")
nopkg=0; while read -r rel; do [[ -f $RT/$rel ]] || { nopkg=$((nopkg+1)); echo "GATE NOPKG $rel"; }; cmp -s /work/$rel $S/variants/H/$rel || echo "GATE TREE_NOT_HEAD $rel"; done < $S/files.txt
echo "GATE FILES $(wc -l < $S/files.txt) files, $nopkg without a package copy, runtime root $RT"
use() {
  while read -r rel; do cp "$S/variants/$2/$rel" "/work/$rel"; [[ -f $RT/$rel ]] && cp "$S/variants/$2/$rel" "$RT/$rel"; done < $S/files.txt
  echo "GATE USE $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
  if [[ $2 == M ]]; then export CI_AB_MAIN_FACTORY=1; else unset CI_AB_MAIN_FACTORY; fi
}
elfsum() { for e in $(find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
declare -A ELF; r=0; fail=0
for blk in 1 2; do for a in M H H2 H2 H M; do
  r=$((r+1)); use $a $a; o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  timeout -s INT -k 60 1500 python3 -m tracy -r -p --no-web-server -o $o $S/gate_ops2.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed"
  grep -a 'CASE .* FAILED' $o.log | head -3 | cut -c1-240
  python3 $S/sum_op.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE)
    echo "GATE ELF $a ${ELF[$a]} $(find $TT_METAL_CACHE -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' | wc -l) compute ELFs"; fi
done; done
use H H
echo "GATE GUARD M=${ELF[M]} H=${ELF[H]} H2=${ELF[H2]}"
[[ ${ELF[M]} == "${ELF[H]}" ]] && { echo "GATE GUARD FAIL: M and H built the same kernels"; fail=1; }
[[ ${ELF[H]} != "${ELF[H2]}" ]] && { echo "GATE GUARD FAIL: the A/A arms built different kernels"; fail=1; }
echo "== done $(date -u +%FT%TZ)"
exit $fail
