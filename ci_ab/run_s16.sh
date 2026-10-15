#!/usr/bin/env bash
# Scratch A/B job (#58819 op rows with no source), not for merge. The job's tree is main. Arms: M main's device files, S the
# #58819 restore head (3625ab8bc46) three-way merged with main (ci_ab/files.txt, ci_ab/v/<arm>/), S2 an A/A copy of S, N S with
# conv3d's reduction tail back to the per tile form. Each arm's files go into the checkout and the installed package before its
# runs, each arm has its own fresh JIT cache, one case per tracy process (prof_ops12.py, 2 warm-up calls, R3_NMEAS measured
# calls), arm order rotated per round. Phase 1: M S S2 on the op cases (topk, transpose, permute, moe gate in 8 L1
# placements, moe_fused_swiglu, conv3d fp32 tails). Phase 2: S N S2 on the two conv3d cases. Guard: S2 builds equal S, M builds
# differ from S in at least one kernel, every run reported its ops.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci R3_NMEAS=${R3_NMEAS:-12} R3_L1_SMALL=16384
S=/work/ci_ab; OUT=/work/generated/s16; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP $(hostname)"
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK PD
nopkg=0
while read -r rel; do
  PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  [[ -z ${PK[$rel]} ]] && { nopkg=$((nopkg+1)); s=$(dirname "$rel" | awk -F/ '{n=NF; print $(n-2)"/"$(n-1)"/"$n}'); PD[$rel]=$(find "$SP" -type d -path "*/$s" 2>/dev/null); }
done < $S/files.txt
echo "== $(wc -l < $S/files.txt) arm files, $nopkg without a package copy on main"
put() {  # put <arm dir> <rel>: the file into the checkout and every package copy
  local a=$1 rel=$2 p d
  if [[ -f $S/v/$a/$rel.absent ]]; then rm -f "/work/$rel"; for p in ${PK[$rel]}; do rm -f "$p"; done; return; fi
  mkdir -p "$(dirname /work/$rel)"; cp "$S/v/$a/$rel" "/work/$rel"
  for p in ${PK[$rel]}; do cp "$S/v/$a/$rel" "$p"; done
  for d in ${PD[$rel]:-}; do cp "$S/v/$a/$rel" "$d/"; done
}
CURF=""
place() {  # place <arm>: S2 is S, N is S with the conv3d compute file of v/N; files are copied only when the set changes
  local a=$1 f=$1; [[ $a == S2 ]] && f=S; [[ $a == N ]] && f=N; [[ $a == S2 || $a == N ]] && a=S
  if [[ $f != $CURF ]]; then
    while read -r rel; do put $a "$rel"; done < $S/files.txt
    [[ $f == N ]] && put N ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp
    CURF=$f
    echo "RAB FILES $f $(while read -r rel; do cat "/work/$rel" 2>/dev/null; done < $S/files.txt | md5sum | cut -c1-8)"
  fi
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
fail=0; r=0
phase() {  # phase <arms> <cases> <rounds>
  local arms=($1) cases=($2) rounds=$3 k i c a v o
  for rr in $(seq 1 $rounds); do
    r=$((r+1)); k=$(( (rr - 1) % ${#arms[@]} )); v=("${arms[@]:$k}" "${arms[@]:0:$k}")
    for c in "${cases[@]}"; do for a in "${v[@]}"; do
      place $a; o=$OUT/run_${c}_${a}_r$r; mkdir -p $o/pd
      TT_METAL_PROFILER_DIR=$o/pd timeout -s INT -k 60 1200 python3 -m tracy -r -p --no-web-server -o $o $S/prof_ops12.py $c > $o.log 2>&1
      rc=$?; ok=$(grep -c "CASE $c ok" $o.log)
      echo "== run $c $a r$r rc=$rc ok=$ok $(date -u +%T)"
      if [[ $rc -ne 0 || $ok -eq 0 ]]; then fail=1; grep -a -m3 -E "FAILED|TT_THROW|Error" $o.log | cut -c1-200; fi
      python3 $S/ol_sum.py $o $a $r $c $R3_NMEAS
      rm -rf $o
    done; done
  done
}
C1="topk_qwen_vocab topk_llama_vocab topk_k512_w65536 transpose_rm_1024x1024 transpose_rm_8x256x128 permute_rm_nchw_nhwc_64x128x128 permute_tile_2301_32x32x64x64"
C2="moe_gate_b32_top8_sigmoid_s0 moe_gate_b32_top8_sigmoid_s1 moe_gate_b32_top8_sigmoid_s2 moe_gate_b32_top8_sigmoid_s3 moe_gate_b64_top8_softmax_s0 moe_gate_b64_top8_softmax_s1 moe_gate_b64_top8_softmax_s2 moe_gate_b64_top8_softmax_s3"
C3="swiglu_dsv3_t3001_xrm swiglu_dsv3_t768_xtile c3d_fp32_amp384k7_t6080 c3d_fp32_ups0_t3054"
phase "M S S2" "$C1 $C2 $C3" ${R16_ROUNDS:-4}
phase "S N S2" "c3d_fp32_amp384k7_t6080 c3d_fp32_ups0_t3054" ${R16_ROUNDS:-4}
for a in M S S2 N; do python3 $S/elfhash.py $a $OUT/cache_$a; done > $OUT/elf.txt; cat $OUT/elf.txt | grep -a "^ELFCOUNT"
python3 $S/elfguard.py $OUT/elf.txt M S S2 | tee $OUT/guard1.txt
python3 $S/elfguard.py $OUT/elf.txt S S2 N | tee $OUT/guard2.txt
g1=$(grep -a "RAB ELFPAIRS M_vs_S builds" $OUT/guard1.txt | awk '$NF>0 && $(NF-2)>0 {print "ok"}' | head -1); [[ $g1 == ok ]] || g1=VOID
g2=$(grep -a "RAB ELFPAIRS S_vs_S2 builds" $OUT/guard2.txt | awk '$NF==0 && $(NF-2)>0 {print "ok"}' | head -1); [[ $g2 == ok ]] || g2=DIFF
g3=$(grep -a "RAB ELFPAIRS S_vs_N builds" $OUT/guard2.txt | awk '$NF>0 && $(NF-2)>0 {print "ok"}' | head -1); [[ $g3 == ok ]] || g3=VOID
echo "RAB GUARD M_vs_S=$g1 S_vs_S2=$g2 S_vs_N=$g3"; [[ $g1 == ok && $g2 == ok && $g3 == ok ]] || fail=1
place M
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
