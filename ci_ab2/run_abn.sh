#!/usr/bin/env bash
# Scratch CI A/B job, not for merge. remeasure_dc ci_ab2: arms M and D from ci_ab/v, F = D plus the files in ci_ab2/v/F. bash ci_ab/run_abn.sh <batch ids file>
# Arms: ci_ab/arms (first line, e.g. "M A B"; M is the reference) plus M2, an A/A copy of M. Each arm swaps its device files
# (ci_ab/v/<arm>/, ci_ab/files.txt, v/<arm>/.absent = files it lacks) into /work and the installed package before every run, with its own
# fresh JIT cache. Units (#U lines: a test-file chunk) run CIAB_BLOCKS blocks, order M A B M2 | M2 B A M | ..., start rotated per unit,
# one pytest process per run, id order rotated per run. Guard: every arm's ELFs must differ from M's in some unit (the swap reaches the
# compiler); M against M2 is reported.
set +e +o pipefail
B=$1; T0=$(date +%s); DL=${CIAB_DEADLINE_S:-5700}; NB=${CIAB_BLOCKS:-2}
W=${CIAB_W:-/work}; cd $W; export TT_METAL_HOME=$W
S=$W/ci_ab2; OUT=$W/generated/ciab; rm -rf $OUT; mkdir -p $OUT
read -r -a AR < $S/arms; ARMS=("${AR[@]}" M2); NA=${#ARMS[@]}
[[ ${CI_FRESH:-1} == 1 ]] && rm -rf $HOME/.cache/tt-metal-cache /tmp/tt-metal-cache 2>/dev/null
SP=${CIAB_SP:-}; [[ -n $SP ]] || SP=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "CIAB ARMS $(tr '\n' ' ' < $S/ARMS.txt 2>/dev/null)"; echo "CIAB HOST $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%H) batch $B package $SP blocks $NB arms ${ARMS[*]}"
: > $OUT/pkgfiles.txt; [[ -d $SP && $SP != $W ]] && find $SP -type f \( -path '*/tt_metal/hw/*' -o -path '*/tt-llk/*' -o -path '*/kernels/*' \) 2>/dev/null | grep -v "^$W/" > $OUT/pkgfiles.txt
declare -A PK; np=0
while read -r rel; do PK[$rel]=$(grep -F "/$rel" $OUT/pkgfiles.txt | grep -E "/${rel//./\\.}\$"); np=$((np + $(echo ${PK[$rel]} | wc -w))); done < $S/files.txt
echo "CIAB FILES $(wc -l < $S/files.txt) device files, $np package copies"
use() {  # use <variant>
  while read -r rel; do
    if grep -qxF "$rel" $W/ci_ab/v/${1/F/D}/.absent 2>/dev/null; then rm -f "$W/$rel"; for p in ${PK[$rel]}; do rm -f "$p"; done
    else src=$W/ci_ab/v/$1/$rel; [[ $1 == F ]] && { src=$W/ci_ab/v/D/$rel; [[ -f $S/v/F/$rel ]] && src=$S/v/F/$rel; }
      mkdir -p "$(dirname "$W/$rel")"; cp "$src" "$W/$rel"; for p in ${PK[$rel]}; do cp "$src" "$p"; done; fi
  done < $S/files.txt
}
var() { [[ $1 == M2 ]] && echo M || echo $1; }
for v in "${AR[@]}"; do use $v; echo "CIAB VARIANT $v $(while read -r rel; do cat $W/$rel 2>/dev/null; done < $S/files.txt | md5sum | cut -c1-8)"; done
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 PYTHONPATH=$W:$W/ttnn:$W/tools:$S
mapfile -t LINES < <(grep . $B)
U=(); cur=""; for l in "${LINES[@]}" END; do if [[ $l == "#U"* || $l == END ]]; then [[ -n $cur ]] && U+=("$cur"); cur=""; else cur+="$l"$'\n'; fi; done
echo "CIAB UNITS ${#U[@]} ids $(grep -vc '^#' $B)"
est_rate=1.0; done_est=0; done_s=0; fail=0; declare -A GOK; F=$(awk -v na=$NA -v nb=$NB 'BEGIN{print na*nb/6}')  # cost vs the 3-arm 2-block estimate
for k in "${!U[@]}"; do
  mapfile -t G < <(printf '%s' "${U[$k]}")
  # remeasure_dc: drop ids main no longer has (one missing node id makes pytest run none of the unit)
  timeout 600 python3 -m pytest --co -q -p no:cacheprovider "${G[@]}" > $OUT/co_$k.log 2>&1; crc=$?
  if [[ $crc -ne 0 ]]; then KEEP=(); for i in "${G[@]}"; do if timeout 300 python3 -m pytest --co -q -p no:cacheprovider "$i" > /dev/null 2>&1; then KEEP+=("$i"); else echo "CIAB DROPID $i"; fi; done; G=("${KEEP[@]}"); fi
  [[ ${#G[@]} -eq 0 ]] && { echo "CIAB UNIT $k empty after collection"; continue; }
  e=$(sed -n "/^#U/p" $B | sed -n "$((k+1))p" | awk '{print $2}')
  el=$(( $(date +%s) - T0 )); need=$(awk -v e=$e -v r=$est_rate -v f=$F 'BEGIN{printf "%d", e*r*f}')
  if (( el + need > DL )); then echo "CIAB SKIP $k elapsed $el need $need"; printf 'CIAB SKIPID %s\n' "${G[@]}"; continue; fi
  o=$((k % NA)); rot=("${ARMS[@]:$o}" "${ARMS[@]:0:$o}"); rev=(); for ((i = NA - 1; i >= 0; i--)); do rev+=("${rot[$i]}"); done; ord=()
  for ((b = 0; b < NB; b++)); do if ((b % 2 == 0)); then ord+=("${rot[@]}"); else ord+=("${rev[@]}"); fi; done
  n=${#G[@]}; tu=$(date +%s); NR=$((NA * NB))
  for ((r = 0; r < NR; r++)); do a=${ord[$r]}; use $(var $a)
    export TT_METAL_CACHE=$OUT/cache_$a; mkdir -p $TT_METAL_CACHE
    export TT_METAL_PROFILER_DIR=$OUT/prof; rm -rf $TT_METAL_PROFILER_DIR; mkdir -p $TT_METAL_PROFILER_DIR
    s=$(( r * n / NR )); R=("${G[@]:$s}" "${G[@]:0:$s}"); j=$OUT/u${k}_r${r}_$a.jsonl; t1=$(date +%s)
    if [[ -n ${CIAB_MOCK:-} ]]; then for i in "${R[@]}"; do echo "{\"id\": \"$i\", \"passed\": true, \"bits\": \"abc\", \"progs\": {\"1\": [$RANDOM]}}" >> $j; done; echo "1 passed" > $OUT/u${k}_r${r}_$a.log; else
    PROF_OUT=$j timeout -s INT -k 30 ${CIAB_PT_TO:-1500} python3 -m pytest -p prof_plugin -q --no-header -p no:cacheprovider "${R[@]}" > $OUT/u${k}_r${r}_$a.log 2>&1; fi
    rc=$?; echo "CIAB RUN $k $r $a rc=$rc $(( $(date +%s) - t1 )) s $(grep -aE '[0-9]+ (passed|failed|error)' $OUT/u${k}_r${r}_$a.log | tail -1)"
    [[ $rc -ne 0 && $rc -ne 1 ]] && grep -a -m3 -E '^(FAILED|ERROR)|Error' $OUT/u${k}_r${r}_$a.log | cut -c1-200
    python3 $S/rec.py $j $k $r $a
  done
  du=$(( $(date +%s) - tu )); done_s=$((done_s + du)); done_est=$((done_est + e)); est_rate=$(awk -v a=$done_s -v b=$done_est -v f=$F 'BEGIN{r=a/b/f; if(r<0.3)r=0.3; printf "%.3f", r}')
  echo "CIAB UNIT $k ids $n $du s est $e rate $est_rate"
  [[ -n ${CIAB_MOCK:-} ]] && continue
  for x in "${AR[@]:1}"; do [[ -n ${GOK[$x]:-} ]] && continue
    g=$(python3 $S/elfpair.py $OUT/cache_M $OUT/cache_$x M$x); echo "$g"
    if ! { echo "$g" | grep -q "ELFPAIR M$x pairs [0-9]* differ 0 " && echo "$g" | grep -q "ELFNAME M$x names [0-9]* differ 0"; }; then GOK[$x]=1; echo "CIAB GUARD $x ok"; fi
  done
done
for x in "${AR[@]:1}"; do python3 $S/elfpair.py $OUT/cache_M $OUT/cache_$x M${x}_end; [[ -n ${GOK[$x]:-} || -n ${CIAB_MOCK:-} ]] || { echo "CIAB GUARD VOID: $x compiled to M's ELFs in every unit"; fail=3; }; done
python3 $S/elfpair.py $OUT/cache_M $OUT/cache_M2 MM2_end
(( NA > 3 )) && python3 $S/elfpair.py $OUT/cache_${AR[1]} $OUT/cache_${AR[2]} ${AR[1]}${AR[2]}_end
use ${AR[-1]}; echo "CIAB DONE $(date -u +%FT%TZ) $(( $(date +%s) - T0 )) s"; exit $fail
