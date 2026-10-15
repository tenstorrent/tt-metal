#!/usr/bin/env bash
# Scratch A/B job, not for merge: is the #58819 topk_qwen_vocab TopkLargeIndices +0.28 % (run 38713226315) code or placement.
# Tree is main 6a3f10a4242 with the step 16 arm files (ci_ab/files.txt, ci_ab/v/M and v/S). Arms: M main, S the #58819 head
# merged with main, S2 A/A copy of S, MX main and SX the head, each with NOP pads in the route prep compute kernel
# (topk_route_prep_untilize_clamp.cpp, ELF differs M vs S) that make its three TRISC images the same packed size in both
# (found by probe builds of the qwen case, padfit.py), so every program of the call has the same kernel config ring layout.
# Per-arm fresh JIT caches, one case per tracy process, 2 warm-up and R3_NMEAS measured calls, arm order rotated per round.
# Prints every kernel and firmware ELF hash and packed size per arm (elfhash2.py).
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci R3_NMEAS=${R3_NMEAS:-12} R3_L1_SMALL=16384
S=/work/ci_ab; OUT=/work/generated/s16t; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') package root $SP $(hostname)"
RP=ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_route_prep_untilize_clamp.cpp
cp /work/$RP $OUT/rp_orig.cpp
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK PD
nopkg=0
for rel in $(cat $S/files.txt) $RP; do
  PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null)
  [[ -z ${PK[$rel]} ]] && { nopkg=$((nopkg+1)); s=$(dirname "$rel" | awk -F/ '{n=NF; print $(n-2)"/"$(n-1)"/"$n}'); PD[$rel]=$(find "$SP" -type d -path "*/$s" 2>/dev/null); }
done
echo "== $(wc -l < $S/files.txt) arm files + route prep kernel, $nopkg without a package copy on main; route prep copies: ${PK[$RP]}"
putf() {  # putf <source file or "absent"> <rel>: into the checkout and every package copy
  local src=$1 rel=$2 p d
  if [[ $src == absent ]]; then rm -f "/work/$rel"; for p in ${PK[$rel]}; do rm -f "$p"; done; return; fi
  mkdir -p "$(dirname /work/$rel)"; cp "$src" "/work/$rel"
  for p in ${PK[$rel]}; do cp "$src" "$p"; done
  for d in ${PD[$rel]:-}; do cp "$src" "$d/"; done
}
CURB=""; CURR=""
place() {  # place <arm>: base files (M or S) and the route prep variant, copied only when they change; own cache per arm
  local a=$1 b r
  case $a in M|MX) b=M;; *) b=S;; esac
  case $a in MX) r=$OUT/rp_MX.cpp;; SX) r=$OUT/rp_SX.cpp;; *) r=$OUT/rp_orig.cpp;; esac
  if [[ $b != $CURB ]]; then
    while read -r rel; do
      if [[ -f $S/v/$b/$rel.absent ]]; then putf absent "$rel"; else putf "$S/v/$b/$rel" "$rel"; fi
    done < $S/files.txt
    CURB=$b
  fi
  [[ $r != $CURR ]] && { putf "$r" $RP; CURR=$r; }
  echo "RAB FILES $a base $b $(while read -r rel; do cat "/work/$rel" 2>/dev/null; done < $S/files.txt | md5sum | cut -c1-8) rp $(md5sum < /work/$RP | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$a; mkdir -p $TT_METAL_CACHE
}
fail=0
trun() {  # trun <case> <out dir> <log> [measured calls]: one tracy process
  R3_NMEAS=${4:-$R3_NMEAS} TT_METAL_PROFILER_DIR=$2/pd timeout -s INT -k 60 1200 python3 -m tracy -r -p --no-web-server -o $2 $S/prof_ops12.py $1 > $3 2>&1
}
# Probe builds: the qwen case once per padded arm into a throwaway cache, then pads from the route prep packed sizes.
pm=0,0,0; ps=0,0,0; match=0
for it in 1 2 3 4 5 6; do
  python3 $S/padrp.py $OUT/rp_orig.cpp $OUT/rp_MX.cpp ${pm//,/ }
  python3 $S/padrp.py $OUT/rp_orig.cpp $OUT/rp_SX.cpp ${ps//,/ }
  CURR=""
  for a in MX SX; do
    place $a; export TT_METAL_CACHE=$OUT/pc_${a}_$it; mkdir -p $TT_METAL_CACHE; o=$OUT/probe_${a}_$it; mkdir -p $o/pd
    trun topk_qwen_vocab $o $o.log 1; echo "== probe $a it$it rc=$? ok=$(grep -c 'CASE topk_qwen_vocab ok' $o.log) $(date -u +%T)"
    rm -rf $o
  done
  line=$(python3 $S/padfit.py $OUT/pc_MX_$it $OUT/pc_SX_$it $pm $ps); echo "$line it $it pads_m $pm pads_s $ps"
  [[ $line == "RAB PADFIT sizes"* ]] || { echo "RAB PADFIT no sizes"; break; }
  set -- $line; match=$8
  [[ $match == 1 ]] && break
  pm=${10}; ps=${11}
done
echo "RAB PADS MX $pm SX $ps match $match"; [[ $match == 1 ]] || fail=1
grep -H -n "asm volatile\|(void)0" $OUT/rp_MX.cpp $OUT/rp_SX.cpp
CURR=""
r=0
phase() {  # phase <arms> <cases> <rounds>
  local arms=($1) cases=($2) rounds=$3 k c a v o
  for rr in $(seq 1 $rounds); do
    r=$((r+1)); k=$(( (rr - 1) % ${#arms[@]} )); v=("${arms[@]:$k}" "${arms[@]:0:$k}")
    for c in "${cases[@]}"; do for a in "${v[@]}"; do
      place $a; o=$OUT/run_${c}_${a}_r$r; mkdir -p $o/pd
      trun $c $o $o.log; rc=$?; ok=$(grep -c "CASE $c ok" $o.log)
      echo "== run $c $a r$r rc=$rc ok=$ok $(date -u +%T)"
      if [[ $rc -ne 0 || $ok -eq 0 ]]; then fail=1; grep -a -m3 -E "FAILED|TT_THROW|Error" $o.log | cut -c1-200; fi
      python3 $S/ol_sum.py $o $a $r $c $R3_NMEAS
      rm -rf $o
    done; done
  done
}
phase "M S S2 MX SX" "topk_qwen_vocab topk_llama_vocab topk_k512_w65536" ${R16_ROUNDS:-6}
for a in M S S2 MX SX; do python3 $S/elfhash2.py $a $OUT/cache_$a; done > $OUT/elf.txt
cat $OUT/elf.txt
python3 $S/elfguard.py $OUT/elf.txt M S S2 MX SX
python3 $S/elfguard.py $OUT/elf.txt MX SX
python3 $S/elfguard.py $OUT/elf.txt S SX
g1=$(python3 $S/elfguard.py $OUT/elf.txt M S | awk '/ELFPAIRS/ && $NF>0 && $(NF-2)>0 {print "ok"}'); [[ $g1 == ok ]] || g1=VOID
g2=$(python3 $S/elfguard.py $OUT/elf.txt S S2 | awk '/ELFPAIRS/ && $NF==0 && $(NF-2)>0 {print "ok"}'); [[ $g2 == ok ]] || g2=DIFF
echo "RAB GUARD M_vs_S=$g1 S_vs_S2=$g2 PADS=$([[ $match == 1 ]] && echo ok || echo FAIL)"; [[ $g1 == ok && $g2 == ok ]] || fail=1
place M
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
