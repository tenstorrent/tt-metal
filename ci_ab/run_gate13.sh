#!/usr/bin/env bash
# Scratch A/A + A/B job (MoE gate, direct bias add and step2), not for merge. generalized_moe_gate device time under the
# device profiler, identity_indices=True: A = c19f44e28bf's device files, A2 its copy, B = 5069e74e698's, B2 its copy;
# own fresh JIT cache per arm; ABBA blocks (A B B2 A2 A2 B2 B A) three times, 16 calls per case. Bits and tests: G12.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 GATE_NMEAS=${GATE_NMEAS:-16} HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/gate; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP"
OBJ=$(find /work/runtime /opt /usr/local "$SP" -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
while read -r rel; do echo "== $rel -> $(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null | tr '\n' ' ')"; done < $S/files.txt
use() {
  while read -r rel; do
    cp "$S/variants/$2/$rel" "/work/$rel"
    while read -r p; do [[ -n $p && $p != "/work/$rel" ]] && cp "$S/variants/$2/$rel" "$p"; done < <(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  done < $S/files.txt
  echo "OPS FILES $1 $(while read -r rel; do cat /work/$rel; done < $S/files.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/*' -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
declare -A ELF
r=0; fail=0
for blk in 1 2 3; do for a in A B B2 A2 A2 B2 B A; do
  r=$((r+1)); use $a $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  GATE_MODE=prof timeout -s INT -k 60 1200 python3 -m tracy -r -p --no-web-server -o $o $S/gate_ops.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'FAILED|Error' $o.log | cut -c1-240; }
  python3 $S/sum_ops.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE); echo "OPS ELF $a ${ELF[$a]}"; fi
done; done
echo "OPS GUARD A=${ELF[A]} A2=${ELF[A2]} B=${ELF[B]} B2=${ELF[B2]}"
[[ ${ELF[A]} == ${ELF[B]} ]] && echo "GUARD FAIL: A and B compiled the same"
echo "== done $(date -u +%FT%TZ)"
