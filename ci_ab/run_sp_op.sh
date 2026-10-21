#!/usr/bin/env bash
# Scratch op-level job (#61433), not for merge. DRAM group_norm (groupnorm.cpp, the reduce helper's SCALAR statistics),
# device kernel time per op under tracy. Arms H (the base files) and S (the SCALAR pair path), S2 its A/A copy, copied into the
# checkout and the installed package before each run, fresh JIT cache per arm; order H S S2 S2 S H twice, R3_REPS calls
# per test (the first is the warm-up). Guard: the groupnorm compute ELFs, H != S, S == S2, each arm stable across runs;
# outputs hashed per call (bit identity H against S).
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work TRACY_NO_WEB_SERVER=1 R3_REPS=${R3_REPS:-4} AB_NODES_FILE=/work/ci_ab/opl/sp_nodes.txt
S=/work/ci_ab; OUT=/work/generated/sp_ab; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
export PYTHONPATH=$S/opl:${PYTHONPATH:-}
SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s' | cut -c1-60) package root $SP"
OBJ=$(find /work/runtime /opt -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
suffix() { echo "$1" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'; }
declare -A PK
while read -r rel; do PK[$rel]=$(find "$SP" -path "*/$(suffix $rel)" -type f 2>/dev/null); echo "== file $rel package copies: $(echo ${PK[$rel]} | wc -w)"; done < $S/opl/files_sp.txt
use() {
  while read -r rel; do cp "$S/opv/$2/$rel" "/work/$rel"; for p in ${PK[$rel]}; do cp "$S/opv/$2/$rel" "$p"; done; done < $S/opl/files_sp.txt
  echo "== files $1 $(while read -r rel; do cat /work/$rel; done < $S/opl/files_sp.txt | md5sum | cut -c1-8)"
  export TT_METAL_CACHE=$OUT/cache_$1; mkdir -p $TT_METAL_CACHE
}
elfsum() {
  for e in $(ls $1/*/kernels/groupnorm/*/trisc*/*.elf 2>/dev/null); do
    $OBJ -O binary -j .text "$e" /tmp/e.bin 2>/dev/null && md5sum < /tmp/e.bin | cut -c1-12
  done | sort | md5sum | cut -c1-12
}
declare -A ES; r=0; fail=0
for blk in 1 2; do for a in H S S2 S2 S H; do
  r=$((r+1)); src=${a%2}; use $a $src
  o=$OUT/${a}_r$r; mkdir -p $o; t0=$(date +%s)
  OUT_HASH=$OUT/hash_${a}_r$r.tsv timeout -s INT -k 60 2400 python3 -m tracy -r -p --no-web-server -o $o $S/opl/run_nodes.py > $o.log 2>&1; rc=$?
  echo "== run $r $a rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $o.log | tail -1)"
  [[ $rc -ne 0 ]] && { fail=1; grep -a -m5 -E '^(FAILED|ERROR)|Error' $o.log | cut -c1-200; }
  python3 $S/opl/sum_op19.py $o $a $r
  e=$(elfsum $TT_METAL_CACHE); ES[$a]+=" $e"; echo "== elf $a r$r $e"
done; done
uniq1() { echo $1 | tr ' ' '\n' | grep . | sort -u | wc -l; }
one() { echo $1 | tr ' ' '\n' | grep . | head -1; }
gs=ok; for a in H S S2; do [[ $(uniq1 "${ES[$a]}") == 1 ]] || gs=UNSTABLE; done
gx=$([[ $(one "${ES[S]}") == $(one "${ES[S2]}") ]] && echo ok || echo DIFF)
gd=$([[ $(one "${ES[H]}") != $(one "${ES[S]}") && -n $OBJ ]] && echo ok || echo VOID)
echo "== GUARD stable=$gs S_vs_S2=$gx H_vs_S=$gd"
for p in "H_r1 H_r6" "S_r2 S_r5" "H_r1 S_r2" "H_r1 S2_r3"; do set -- $p
  echo "== outputs $1 vs $2: $(paste <(cut -f2 $OUT/hash_$1.tsv) <(cut -f2 $OUT/hash_$2.tsv) | awk '{n++; if ($1==$2) s++} END {print s+0" identical of "n+0}')"
done
use H H
echo "== DONE $(date -u +%FT%TZ) fail=$fail"

