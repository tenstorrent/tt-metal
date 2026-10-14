#!/usr/bin/env bash
# Softmax pack-block opt-in (#58816, softmax.cpp) at every softmax production shape, one Blackhole runner (17_CI_AB.md; scratch).
# Variants (ci_ab/variants/<v>/tree, files.txt, placed in /work and the installed runtime root, each with its own fresh kernel
# cache): M main's device files, H #58816's head, N the head without softmax.cpp's opt-in, Bf the opt-in for outputs that are
# not block float, Ex the opt-in except the exp chain into the exps buffer, H2 an A/A copy of H. Rounds rotate the variants.
# Then one perf counter pass per variant (fpu, pack, unpack, instrn) at the segformer shapes, and the kernels' ELF sums.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
export TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci SM_NMEAS=${SM_NMEAS:-12} PYTHONPATH="$R:$R/tools${PYTHONPATH:+:$PYTHONPATH}"
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R $(git log --oneline -1) package root $SP"
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
G=/tmp/smab; rm -rf $G; mkdir -p $G
src() { case $1 in H2) echo H ;; *) echo $1 ;; esac; }
place() {
  local d=$S/variants/$(src $1)
  while read -r rel; do
    [[ -z $rel ]] && continue
    cp "$d/tree/$rel" "$R/$rel"
    local suf; suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}')
    while read -r p; do [[ -n $p && $p != "$R/$rel" ]] && cp "$d/tree/$rel" "$p"; done < <(find "$SP" -path "*/$suf" -type f 2>/dev/null)
  done < $S/variants/files.txt
}
while read -r rel; do suf=$(echo "$rel" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)"/"$n}'); echo "== $rel -> $(find "$SP" -path "*/$suf" -type f 2>/dev/null | wc -l) package copies"; done < $S/variants/files.txt
elfsum() {
  for k in $(ls -d $1/*/kernels/* 2>/dev/null | xargs -n1 basename | sort -u); do
    for t in 0 1 2; do
      h=$(for e in $(ls $1/*/kernels/$k/*/trisc$t/trisc$t.elf 2>/dev/null | sort); do
            if [[ -n $OBJ ]]; then $OBJ -O binary -j .text "$e" /tmp/elf.bin 2>/dev/null && md5sum < /tmp/elf.bin; else md5sum < "$e"; fi
          done | md5sum | cut -c1-12)
      echo "$k/trisc$t $h"
    done
  done
}
VL=(M H N Bf Ex H2); ROUNDS=${SM_ROUNDS:-4}
for r in $(seq 1 $ROUNDS); do
  k=$(( (r - 1) % ${#VL[@]} )); VS=("${VL[@]:$k}" "${VL[@]:0:$k}")
  (( r % 2 == 0 )) && VS=($(printf '%s\n' "${VS[@]}" | tac))
  for v in "${VS[@]}"; do
    place $v; export TT_METAL_CACHE=$G/cache_$v; mkdir -p $TT_METAL_CACHE
    o=$G/out_${v}_r$r; rm -rf $o; mkdir -p $o
    timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server -o $o $S/tools/prof_sm.py > $o.log 2>&1
    echo "== run $v r$r rc=$? ok $(grep -c 'CASE .* ok' $o.log) failed $(grep -c 'CASE .* FAILED' $o.log) $(date -u +%T)"
    grep -a "CASE .* FAILED" $o.log | head -4 | cut -c1-260
    grep -a "^BITS" $o.log | sed "s/^BITS /BITS $v r$r /"
    python3 $S/tools/sum_sm.py $o $v $r
    rm -rf $o/reports
  done
done
for v in M H N Bf Ex; do
  place $v; export TT_METAL_CACHE=$G/cache_$v
  echo "== compute ELFs of $v:"; elfsum $TT_METAL_CACHE | grep -i "softmax"
  o=$G/ctr_$v; rm -rf $o; mkdir -p $o
  SM_NMEAS=6 timeout -s INT -k 60 1800 python3 -m tracy -r -p --no-web-server --profiler-capture-perf-counters=fpu,pack,unpack,instrn -o $o $S/tools/prof_sm.py seg1_bf8_ns0 seg1_bf16_ns0 seg4_bf8_ns0 seg1_bf8_ns1 > $o.log 2>&1
  echo "== counters $v rc=$? ok $(grep -c 'CASE .* ok' $o.log) $(date -u +%T)"
  grep -a "CASE .* FAILED\|Error" $o.log | head -3 | cut -c1-260
  python3 $S/tools/sum_sm.py $o $v c --counters
done
place H
echo "== DONE $(date -u +%T)"
