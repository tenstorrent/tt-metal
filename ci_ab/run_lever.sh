#!/usr/bin/env bash
# Scratch, not for merge. MoE gate (generalized_moe_gate, identity_indices=True) on one P150, PR head build.
# 1. A/B with A/A: P = PR head unified kernel, P2 its copy, R = conditional startup reconfig, R2 its copy; own fresh JIT
#    cache per arm; blocks P R R2 P2 P2 R2 R P three times, 16 calls per case; device kernel duration.
# 2. Outputs of P and R on 18432 tokens, compared.
# 3. Timeline: Z = P with three zones per TRISC (each ends after a tensix_sync), and C = P with perf counters
#    (FPU | PACK | UNPACK | INSTRN), one case per run, two reps.
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
r=0
for blk in 1 2 3; do for a in P R R2 P2 P2 R2 R P; do
  r=$((r+1)); use $a $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  GATE_MODE=prof timeout -s INT -k 60 1200 python3 -m tracy -r -p --no-web-server -o $o $S/gate_ops.py > $o.log 2>&1
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok"
  [[ $rc -ne 0 ]] && grep -a -m5 -E 'FAILED|Error|error:' $o.log | cut -c1-240
  python3 $S/sum_ops.py $o $a $r
  if [[ -z ${ELF[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE); echo "OPS ELF $a ${ELF[$a]}"; fi
done; done
echo "OPS GUARD P=${ELF[P]} P2=${ELF[P2]} R=${ELF[R]} R2=${ELF[R2]}"
[[ ${ELF[P]} == ${ELF[R]} ]] && echo "GUARD FAIL: P and R compiled the same"
for a in P R; do
  use $a $a; GATE_MODE=dump GATE_SEEDS=8 timeout -s INT -k 60 1800 python3 $S/gate_ops.py $OUT/dump_$a > $OUT/dump_$a.log 2>&1
  echo "== dump $a rc=$? $(grep -a 'DUMP DONE' $OUT/dump_$a.log)"
done
python3 $S/gate_cmp.py $OUT/dump_R/results.pt R $OUT/dump_P/results.pt
CASES="gate_b1_sig_k8_lin gate_b1_raw_k8_smax gate_b32_sig_k8_lin gate_b32_raw_k8_smax gate_b64_sig_k8_lin gate_b64_raw_k8_smax"
for rep in 1 2; do
  for a in Z C; do
    if [[ $a == C ]]; then use C P; export TT_METAL_PROFILE_PERF_COUNTERS=39; else use Z Z; unset TT_METAL_PROFILE_PERF_COUNTERS; fi
    for c in $CASES; do
      o=$OUT/${a}_${rep}_$c; mkdir -p $o
      GATE_MODE=prof GATE_NMEAS=8 GATE_CASE=$c timeout -s INT -k 60 900 python3 -m tracy -r -p --no-web-server -o $o $S/gate_ops.py > $o.log 2>&1
      rc=$?; [[ $rc -ne 0 ]] && { echo "== $a $rep $c rc=$rc"; grep -a -m5 -E 'FAILED|Error|error:' $o.log | cut -c1-240; }
      python3 $S/dev_parse.py $o $a $c 2
      GATE_NMEAS=8 python3 $S/sum_ops.py $o $a $rep
    done
  done
done
unset TT_METAL_PROFILE_PERF_COUNTERS
echo "== done $(date -u +%FT%TZ)"
