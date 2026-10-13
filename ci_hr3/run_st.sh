#!/usr/bin/env bash
# Stack A/B on one CI Blackhole runner (test-dispatch with tracy). Plan lines "<label> <files> <mode> <tests>": files st (this
# tree, #58816's pack LLK) or mp (main's pack LLK files), mode the binary_ng program (EB_HR_BNG: main, s0 the head before the
# switch, s1 the switch), tests r (replicas and bits) or rm (plus the SDXL modules). Each label its own fresh kernel cache;
# eb_prof_plugin repeats every test EB_REPS times under signposts; eb_bits_plugin and the all-pattern test hash outputs.
set +e +o pipefail
export HWLOCK_HELD=ci TRACY_NO_WEB_SERVER=1
R=$(git rev-parse --show-toplevel); S=$R/ci_hr3; cd $R
PLAN=$S/${1:-plan.txt}
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
export PYTHONPATH="$S/ops:$R${PYTHONPATH:+:$PYTHONPATH}" EB_REPS=${EB_REPS:-4}
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
place() {
  while read -r rel; do
    [[ -z $rel ]] && continue
    cp "$S/opvariants/$1/$rel" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$S/opvariants/$1/$rel" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/opfiles.txt
}
while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | wc -l) copies"; done < $S/opfiles.txt
OUTD=$R/generated/stab; rm -rf $OUTD; mkdir -p $OUTD
i=0
while read -r lab files mode tests; do
  [[ -z ${lab:-} || $lab == \#* ]] && continue
  i=$((i + 1)); place $files
  export TT_METAL_CACHE=$HOME/st_cache_$lab
  if [[ " ${FRESHED:-} " != *" $lab "* ]]; then rm -rf $TT_METAL_CACHE; FRESHED="${FRESHED:-} $lab"; fi
  mkdir -p $TT_METAL_CACHE
  T=("$S/ops/test_eb_st_rep.py" "$S/ops/test_eb_hr_bits.py"); [[ $tests == rm ]] && T+=("$S/ops/test_eb_st_mod.py")
  echo "== st $lab ($i) files=$files mode=$mode $(date -u +%T)"
  EB_HR_BNG=$mode EB_DET_OUT=$OUTD/det_${i}_$lab.jsonl EB_HASH_OUT=$OUTD/hash_${i}_$lab.json timeout -s INT -k 60 ${CI_LIMIT:-1200} python -m tracy -r -p --no-web-server -o $OUTD/out_st_${i}_$lab -m pytest -p eb_prof_plugin -p eb_bits_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE "${T[@]}" > $OUTD/log_${i}_$lab.txt 2>&1
  rc=$?; echo "== st $lab ($i) rc=$rc $(date -u +%T): $(grep -E 'passed|failed' $OUTD/log_${i}_$lab.txt | tail -1)"
  [[ $rc != 0 ]] && grep -a -E "^(FAILED|ERROR)" $OUTD/log_${i}_$lab.txt | head -8 | cut -c1-220
  for e in $(ls $TT_METAL_CACHE/*/kernels/*/*/trisc[012]/trisc[012].elf 2>/dev/null); do
    k=$(echo $e | awk -F/ '{print $(NF-3)"/"$(NF-1)}'); [[ -n $OBJ ]] && $OBJ -O binary -j .text "$e" /tmp/elf1.bin 2>/dev/null && echo "$k $(md5sum < /tmp/elf1.bin | cut -c1-12)"
  done | sort -u > $OUTD/elfeach_$lab.txt
done < $PLAN
for pr in S1-S0 S1-P P-M S0-P; do a=${pr%-*}; b=${pr#*-}
  echo "== per-ELF: $a builds whose .text is not among $b's (kernel/thread: differing of total):"
  awk 'NR==FNR {h[$2]=1; next} {n[$1]++; if (!($2 in h)) d[$1]++} END {for (k in n) if (d[k]) print "   " k, d[k], "of", n[k]}' $OUTD/elfeach_$b.txt $OUTD/elfeach_$a.txt
done
for B in P S0 M; do echo "== reduce (base $B)"; PROF_BASE=$B python $S/ops/prof_reduce_cfg.py $OUTD st 2>&1 | awk -F'|' 'NR<=2 || $3 ~ /BinaryNg/ || $10 ~ /slower/' | sort -u; done
echo "== bits (base P)"; python $S/ops/bits_cmp.py $OUTD P 2>&1 | tail -30
echo "== bits (base M)"; python $S/ops/bits_cmp.py $OUTD M 2>&1 | tail -30
echo "== det (all patterns)"; python $S/ops/det_cmp.py $OUTD P 2>&1 | tail -12
echo "== done $(date -u +%T)"
