#!/usr/bin/env bash
# Bits on the merged head: (1) the harness block-unpack functional cases, block math against per-tile math, LLK asserts on;
# (2) binary_ng on every bf16 pattern, base (the PR head's device files on the same merge) against fin (this tree), with a
# positive control (base with a NOP in the unpack) for the swap. Each variant its own fresh kernel cache.
set +e +o pipefail
export HWLOCK_HELD=ci
R=$(git rev-parse --show-toplevel); S=$R/ci_hr; cd $R
GUARD_BASE=none bash $S/run_llk_ab.sh plan_bits.txt 2>&1 | grep -v "^H4s\|^[A-Za-z0-9+/=]\{150,\}$" | grep -v "@@RESULTS" | tail -12
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
export PYTHONPATH="$S/ops:$R${PYTHONPATH:+:$PYTHONPATH}"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
place() {
  while read -r rel; do
    [[ -z $rel ]] && continue
    cp "$S/opvariants/$1/$rel" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$S/opvariants/$1/$rel" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/opfiles_bits.txt
}
OUTD=$R/generated/bits; rm -rf $OUTD; mkdir -p $OUTD
i=0
for v in base fin base fin ctl; do
  i=$((i + 1)); place $v
  export TT_METAL_CACHE=$HOME/hr_bitscache_$v
  if [[ " ${FRESHED:-} " != *" $v "* ]]; then rm -rf $TT_METAL_CACHE; FRESHED="${FRESHED:-} $v"; fi
  EB_DET_OUT=$OUTD/det_${i}_$v.jsonl timeout -s INT -k 60 1500 python -m pytest -p no:cacheprovider -o timeout_method=thread -q -rfE $S/ops/test_eb_hr_bits.py > $OUTD/log_${i}_$v.txt 2>&1
  echo "== bits $v ($i) rc=$? $(grep -E 'passed|failed' $OUTD/log_${i}_$v.txt | tail -1)"
  for e in $(ls $TT_METAL_CACHE/*/kernels/*/*/trisc[012]/trisc[012].elf 2>/dev/null); do
    k=$(echo $e | awk -F/ '{print $(NF-3)"/"$(NF-1)}'); [[ -n $OBJ ]] && $OBJ -O binary -j .text "$e" /tmp/elf1.bin 2>/dev/null && echo "$k $(md5sum < /tmp/elf1.bin | cut -c1-12)"
  done | sort -u > $OUTD/elfeach_$v.txt
done
for v in fin ctl; do
  echo "== per-ELF: $v ELFs not among base's:"; awk 'NR==FNR {h[$2]=1; next} {n[$1]++; if (!($2 in h)) d[$1]++} END {for (k in n) if (d[k]) print "   " k, d[k], "of", n[k]}' $OUTD/elfeach_base.txt $OUTD/elfeach_$v.txt
done
python $S/ops/det_cmp.py $OUTD base
echo "== done $(date -u +%T)"
