#!/usr/bin/env bash
# Scratch A/B job (#57395 against main), not for merge. SDPA op-level device time at production Blackhole shapes under
# the device profiler, each side on its own ttnn build: A = main (built in this dispatch, tree /work/side_a), B = the
# #57395 candidate merged with main (tree /work), B2 = an A/A copy of B at the same tree path. Diagnostic arms on B's
# tree, MLA prefill cases only, each with one scratch factory knob: Down = own head chains as before the fix, Doff = no
# causal K/V chains, Ds2 = two K/V slots. Each arm its own fresh JIT cache; A B Down Doff Ds2 B2 B2 Ds2 Doff Down B A in
# two blocks; the first run of A, B and B2 saves outputs for the bits check and the Flux fp64 reference. Guard: each
# arm's host library and SDPA compute ELF hashes (fail if A equals B or B differs from B2) and each knob's effect as
# the factory logged it (fail if a diagnostic arm built B's programs).
set +e +o pipefail
S=/work/ci_ab; OUT=/work/generated/sdpa_ab; rm -rf $OUT; mkdir -p $OUT
export TRACY_NO_WEB_SERVER=1 SDPA_NMEAS=${SDPA_NMEAS:-5} HWLOCK_HELD=ci
MLA="dsv3_mla_prefill_nh16_S1024_bfp8 dsv3_mla_prefill_nh16_S2048_bfp8 dsv3_mla_prefill_nh16_S4096_bfp8"
MLA="$MLA dsv3_mla_prefill_nh16_S8192_bfp8 dsv3_mla_prefill_nh16_S4096_bf16 dsv3_mla_prefill_nh32_S1024_bfp8"
MLA="$MLA dsv3_mla_prefill_nh32_S4096_bfp8 dsv3_mla_prefill_nh32_S4096_bf16"
declare -A KNOB=([Down]="SDPA_AB_OWN=1" [Doff]="SDPA_AB_KVMODE=0" [Ds2]="SDPA_AB_SLOTS=2")
echo "== $(date -u +%FT%TZ) B $(git -C /work log -1 --format='%H %s') A $(git -C /work/side_a log -1 --format='%H %s')"
OBJ=$(find /work/runtime /opt /usr/local -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
echo "== objcopy $OBJ"
for R in /work/side_a /work; do echo "== $R $(ls $R/ttnn/ttnn/*.so 2>/dev/null | tr '\n' ' ') $(md5sum $R/build/lib/libtt_metal.so 2>/dev/null | cut -c1-12)"; done
use() {  # use <arm>: that side's tree, libraries and its own JIT cache
  if [[ $1 == A ]]; then R=/work/side_a; else R=/work; fi
  export TT_METAL_HOME=$R TT_METAL_RUNTIME_ROOT=$R PYTHONPATH=$R/ttnn:$R:$R/tools LD_LIBRARY_PATH=$R/build/lib
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() { for e in $(find $1 -path '*/kernels/*' \( -path '*sdpa*' -o -path '*flash*' -o -path '*mla*' \) -name 'trisc*.elf' ! -name '*.xip.elf' 2>/dev/null | sort); do
  $OBJ -O binary -j .text "$e" $OUT/e.bin 2>/dev/null && md5sum < $OUT/e.bin | cut -c1-12; done | sort | md5sum | cut -c1-12; }
topo() { grep -a -m1 'SDPA_AB chains nqh=16 q_chunks=32 ' $1 | sed 's/.*topo=//'; }
declare -A ELF HOST LOG
r=0; fail=0
for blk in 1 2; do for a in A B Down Doff Ds2 B2 B2 Ds2 Doff Down B A; do
  r=$((r+1)); use $a
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  cases=""; [[ -n ${KNOB[$a]:-} ]] && cases=$MLA
  if [[ -z ${LOG[$a]:-} && -z ${KNOB[$a]:-} ]]; then export SDPA_BITS=$OUT/bits_$a; mkdir -p $SDPA_BITS; else unset SDPA_BITS; fi
  (cd $o && env SDPA_AB_LOG=1 ${KNOB[$a]:-} SDPA_CASES="$cases" timeout -s INT -k 60 2700 python3 -m tracy -r -p --no-web-server -o $o $S/sdpa_ol.py > $o.log 2>&1)
  rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $o.log) ok, $(grep -c 'CASE .* FAILED' $o.log) failed; $(grep -a -m1 '^HOST' $o.log | cut -c1-200) $(grep -a -m1 '^GRID' $o.log)"
  grep -a 'CASE .* FAILED' $o.log | cut -c1-260
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E 'Error|error:|TT_FATAL|TT_THROW' $o.log | cut -c1-240; }
  grep -a '^BITS' $o.log | sed "s/^/TAB $a /"
  python3 $S/sum_ol.py $o $a $r | tee -a $OUT/tab.txt
  if [[ -z ${LOG[$a]:-} ]]; then
    LOG[$a]=$o.log
    grep -a '^SDPA_AB' $o.log | sort -u | sed "s/^/TAB KNOB $a /"
    if [[ -z ${KNOB[$a]:-} ]]; then ELF[$a]=$(elfsum $TT_METAL_CACHE); HOST[$a]=$(grep -a -m1 '^HOST' $o.log | awk '{print $3}')
      echo "TAB ELF $a ${ELF[$a]} $(find $TT_METAL_CACHE -path '*/kernels/*' \( -path '*sdpa*' -o -path '*flash*' -o -path '*mla*' \) -name 'trisc*.elf' ! -name '*.xip.elf' | wc -l) SDPA compute ELFs, ttnn ${HOST[$a]}"; fi
  fi
done; done
echo "TAB GUARD A=${ELF[A]} B=${ELF[B]} B2=${ELF[B2]} hostA=${HOST[A]} hostB=${HOST[B]}"
[[ ${ELF[A]} == "${ELF[B]}" ]] && { echo "TAB GUARD FAIL: A and B built the same SDPA kernels"; fail=1; }
[[ ${ELF[B]} != "${ELF[B2]}" ]] && { echo "TAB GUARD FAIL: the A/A arms built different SDPA kernels"; fail=1; }
[[ ${HOST[A]} != /work/side_a/* || ${HOST[B]} != /work/ttnn/* ]] && { echo "TAB GUARD FAIL: an arm imported the other side's ttnn"; fail=1; }
tb=$(topo ${LOG[B]}); td=$(topo ${LOG[Down]})
echo "TAB GUARD knobs: nh16 S4096 chain topology B=$tb Down=$td"
[[ -z $tb || -z $td || $tb == "$td" ]] && { echo "TAB GUARD FAIL: Down built B's chains"; fail=1; }
grep -a -q 'mode=0 (pr 3)' ${LOG[Doff]} || { echo "TAB GUARD FAIL: Doff kept the chains"; fail=1; }
grep -a -q 'mode=3 (pr 3) slots=2' ${LOG[Ds2]} || { echo "TAB GUARD FAIL: Ds2 kept three slots"; fail=1; }
G=$(grep -a -m1 '^GRID' ${LOG[A]} | awk '{print $2}')
for a in A B; do use $a; (cd $OUT && python3 $S/model_cfg.py ${G:-11x10} 2>&1 | grep -a -E '^MODEL|Error' | sed "s/^/TAB $a /"); done
use B
python3 $S/bits_cmp.py $OUT
python3 $S/flux_ref.py $OUT
python3 $S/ab_table.py $OUT/tab.txt
echo "== done $(date -u +%FT%TZ)"
exit $fail
