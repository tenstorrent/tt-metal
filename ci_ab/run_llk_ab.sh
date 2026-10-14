#!/usr/bin/env bash
# tt-llk harness A/B and counter runs on one CI Blackhole runner (knowledge/process/17_CI_AB.md). A fifth column in
# lines.txt sets the ABBA blocks (default 2; 0 runs A once, and B once when it differs). For every label (a side tree: main's
# tree with ci_ab/variants/<label>/tree laid over it) a slot tree under /tmp/ra_slots/<label> is made and its nodes
# are compiled there (fresh build dirs). Then each line of ci_ab/lines.txt ("<tag> <A label> <B label> <run types>")
# runs ABBA twice (A1 B2 B3 A4 A5 B6 B7 A8), the same job and machine for both sides; the A and B repeats are the A/A.
# The perf CSVs, node hash logs and ELF image hashes are printed at the end as one base64 tarball.
set +e +o pipefail
# the harness puts its artefacts under $RUNNER_TEMP when set (GHA), which would put every label in one build dir
unset RUNNER_TEMP
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; SL=${RA_SL:-/tmp/ra_slots}; cd $R
rm -rf $SL; mkdir -p $SL/perf_data $SL/out
echo "== runner $(hostname) $(nproc) cores $(date -u +%FT%TZ)"; ls /dev/tenstorrent 2>&1 | head -3
if [[ -n ${RA_PY:-} ]]; then PY=$RA_PY; else
  python3 -m venv $SL/venv && $SL/venv/bin/pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip.log 2>&1; echo "== pip rc=$? $(tail -1 $SL/pip.log)"
  PY=$SL/venv/bin/python3
  if [[ ! -x $PY ]]; then python3 -m pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip2.log 2>&1; echo "== no venv; pip into $(command -v python3) rc=$?"; PY=$(command -v python3); fi
fi
$PY -c "import tt_exalens, pytest, torch; print('== harness python ok', torch.__version__)"
SF=""
for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]] && { SF=$c; break; }; done
if [[ -z $SF ]]; then (cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $SL/sfpi.log 2>&1); SF=$R/tt_metal/tt-llk/tests/sfpi; fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1)"
WANT=$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" $R/tt_metal/sfpi-version); $SF/compiler/bin/riscv-tt-elf-g++ --version | grep -qF "sfpi:$WANT" || { echo "== wrong sfpi (want $WANT), stopping"; exit 1; }
P="tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/tools/profiler tt_metal/sfpi-version ttnn/cpp/ttnn/operations/experimental"
NP=$(( $(nproc) / 2 )); (( NP > 16 )) && NP=16; (( NP < 4 )) && NP=4; NP=${RA_NP:-$NP}
for L in $(awk '!/^#/ && NF {print $2; print $3}' $S/lines.txt | sort -u); do
  D=$SL/$L; mkdir -p $D $SL/builds/$L
  tar -C $R -cf - $P | tar -C $D -xf -
  (cd $S/variants/$L/tree && find . -type f) | while read -r f; do mkdir -p "$(dirname "$D/$f")"; cp "$S/variants/$L/tree/$f" "$D/$f"; done
  ln -sfn $SF $D/tt_metal/tt-llk/tests/sfpi; rm -rf $D/tt_metal/tt-llk/perf_data; ln -sfn $SL/perf_data $D/tt_metal/tt-llk/perf_data
  rts=$(awk -v l=$L '$2==l || $3==l {print $4}' $S/lines.txt | tr ',' '\n' | sort -u | paste -sd,)
  cat $(awk -v l=$L -v s=$S '$2==l || $3==l {print s"/ids/"$1".txt"}' $S/lines.txt) | sort -u > $SL/ids_$L.txt
  X=(); AO=1; [[ $L == *_wc ]] && X=(--dump-perf-counters); [[ $L == *_fn ]] && AO=0
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $SL/ids_$L.txt
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=$AO TMPDIR=$SL/builds/$L LLK_PERF_L1_MUX_GROUP=0 LLK_PERF_RUN_TYPES=$rts \
      RA_HASHLOG=$SL/out/$L.hash PYTHONPATH=$S/tools timeout -s INT -k 60 3600 $PY -m pytest --compile-producer -n $NP -p no:cacheprovider \
      --override-ini=log_cli=false -q -p ra_hashlog "${X[@]}" "${N[@]}" < /dev/null > $SL/prod_$L.log 2>&1 )
  echo "== built $L ($rts, $(wc -l < $SL/ids_$L.txt) nodes): $(grep -aE '[0-9]+ (passed|failed|error)' $SL/prod_$L.log | tail -1) $(date -u +%T)"
  grep -a -m3 'error:' $SL/prod_$L.log
done
run() { # run <tag> <side> <k> <label> <rts>
  local tag=$1 s=$2 k=$3 L=$4 rts=$5 D=$SL/$4 log=$SL/out/run_$1_$2$3.log
  local X=() AO=1; [[ $L == *_wc ]] && X=(--dump-perf-counters); [[ $L == *_fn ]] && AO=0
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$tag.txt
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=$AO TMPDIR=$SL/builds/$L LLK_PERF_L1_MUX_GROUP=0 LLK_PERF_RUN_TYPES=$rts \
      PERF_RUN_TAG=${tag}_$s$k PERF_KEEP_RUNS=4000 timeout -s INT -k 60 1800 $PY -m pytest --compile-consumer -p no:cacheprovider \
      --override-ini=log_cli=false -q -rfE -o timeout_method=thread "${X[@]}" "${N[@]}" < /dev/null > $log 2>&1 )
  echo "   $tag $s$k rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $log | tail -1) $(date -u +%T)"
  if grep -q "TIMED OUT" $log; then echo "== TENSIX TIMED OUT in $tag $s$k, stopping"; return 1; fi
}
stop=0; [[ -n ${RA_DRY:-} ]] && stop=1
while read -r tag A B rts nb; do
  nb=${nb:-2}
  [[ -z $tag || $tag == \#* || $stop == 1 ]] && continue
  echo "== $tag ($A against $B, $rts) $(date -u +%T)"; k=0
  if [[ $A != $B ]]; then
    ha=$($PY $S/tools/ra_imgs.py $SL/builds/$A $SL/out/$A.hash 2>/dev/null | grep -F -f $S/ids/$tag.txt | cut -f3 | sort | sha256sum | cut -c1-16)
    hb=$($PY $S/tools/ra_imgs.py $SL/builds/$B $SL/out/$B.hash 2>/dev/null | grep -F -f $S/ids/$tag.txt | cut -f3 | sort | sha256sum | cut -c1-16)
    echo "   images $A $ha $B $hb"; [[ $ha == $hb ]] && echo "== VOID $tag: $A and $B images identical"
  fi
  if [[ $A == *_fn || $nb == 0 ]]; then run $tag A 1 $A $rts; [[ $B != $A ]] && run $tag B 2 $B $rts; grep -a -E "^(FAILED|ERROR) " $SL/out/run_${tag}_*.log | cut -c1-300 | head -20; continue; fi
  for b in $(seq 1 $nb); do for s in A B B A; do k=$((k+1)); L=$A; [[ $s == B ]] && L=$B; run $tag $s $k $L $rts || { stop=1; break 2; }; done; done
done < $S/lines.txt
for L in $(awk '!/^#/ && NF {print $2; print $3}' $S/lines.txt | sort -u); do $PY $S/tools/ra_imgs.py $SL/builds/$L $SL/out/$L.hash > $SL/out/img_$L.tsv 2>/dev/null; done
tar -czf $SL/ra_out.tgz -C $SL perf_data/runs out
echo "== results $(stat -c %s $SL/ra_out.tgz) bytes"; echo "=====RA_BEGIN====="; base64 -w 76 $SL/ra_out.tgz; echo "=====RA_END====="
exit 0
