#!/usr/bin/env bash
# tt-llk harness A/B on one CI Blackhole runner (17_CI_AB.md), adapted from regaudit's run_llk_ab.sh. Each label is the
# checked-out tree with ci_ab/variants/<label>/tree laid over it ("<x>_wc" uses x's tree and a counter build). Lines of
# ci_ab/lines.txt ("<tag> <A> <B> <run types>") run ABBA twice; ci_ab/bits.txt ("<label> <ids file>") runs the scratch
# bit dump once per label. Every label's ELF image hashes go out with the CSVs; a line whose two sides build the same
# math images is reported VOID.
set +e +o pipefail
unset RUNNER_TEMP
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; SL=${RA_SL:-/tmp/fl_slots}; cd $R
rm -rf $SL; mkdir -p $SL/perf_data $SL/out
echo "== runner $(hostname) $(nproc) cores $(date -u +%FT%TZ) head $(git rev-parse HEAD)"; ls /dev/tenstorrent 2>&1 | head -3
if [[ -n ${FL_PY:-} ]]; then PY=$FL_PY; else
  python3 -m venv $SL/venv && $SL/venv/bin/pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip.log 2>&1; echo "== pip rc=$?"
  PY=$SL/venv/bin/python3
  [[ -x $PY ]] || { python3 -m pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip2.log 2>&1; PY=$(command -v python3); }
fi
$PY -c "import tt_exalens, pytest, torch; print('== harness python ok', torch.__version__)"
SF=""; for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]] && { SF=$c; break; }; done
if [[ -z $SF ]]; then (cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $SL/sfpi.log 2>&1); SF=$R/tt_metal/tt-llk/tests/sfpi; fi
want=$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" tt_metal/sfpi-version)
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1), pin $want"
$SF/compiler/bin/riscv-tt-elf-g++ --version | head -1 | grep -q ":$want\[" || { echo "== wrong sfpi, stopping"; exit 1; }
P="tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/tools/profiler tt_metal/sfpi-version ttnn/cpp/ttnn/operations/experimental"
NP=$(( $(nproc) / 2 )); (( NP > 16 )) && NP=16; (( NP < 4 )) && NP=4
NP=${FL_NP:-$NP}
labels=$( (awk '!/^#/ && NF {print $2; print $3}' $S/lines.txt; awk '!/^#/ && NF {print $1}' $S/bits.txt) | sort -u)
for L in $labels; do
  V=${L%_wc}; D=$SL/$L; mkdir -p $D $SL/builds/$L
  tar -C $R -cf - $P | tar -C $D -xf -
  if [[ -d $S/variants/$V/tree ]]; then (cd $S/variants/$V/tree && find . -type f) | while read -r f; do mkdir -p "$(dirname "$D/$f")"; cp "$S/variants/$V/tree/$f" "$D/$f"; done; fi
  cp $S/tools/test_*.py $D/tt_metal/tt-llk/tests/python_tests/
  ln -sfn $SF $D/tt_metal/tt-llk/tests/sfpi; rm -rf $D/tt_metal/tt-llk/perf_data; ln -sfn $SL/perf_data $D/tt_metal/tt-llk/perf_data
  echo "== $L tree $( [[ -d $S/variants/$V/tree ]] && (cd $S/variants/$V/tree && find . -type f | sort | xargs md5sum | md5sum | cut -c1-12) || echo base)"
done
build() { # build <label> <ids file> <rts>
  local L=$1 D=$SL/$1 WC=(); [[ $1 == *_wc ]] && WC=(--dump-perf-counters)
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $2
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=1 TMPDIR=$SL/builds/$L LLK_PERF_L1_MUX_GROUP=0 LLK_PERF_RUN_TYPES=$3 \
      RA_HASHLOG=$SL/out/$L.hash PYTHONPATH=$S/tools timeout -s INT -k 60 3600 $PY -m pytest --compile-producer -n $NP "${WC[@]}" -p no:cacheprovider \
      --override-ini=log_cli=false -q -p ra_hashlog "${N[@]}" < /dev/null > $SL/prod_$L.log 2>&1 )
  echo "== built $L ($3, $(wc -l < $2) nodes): $(grep -aE '[0-9]+ (passed|failed|error)' $SL/prod_$L.log | tail -1) $(date -u +%T)"
  grep -a -m3 'error:' $SL/prod_$L.log
}
for L in $(awk '!/^#/ && NF {print $2; print $3}' $S/lines.txt | sort -u); do
  rts=$(awk -v l=$L '!/^#/ && ($2==l || $3==l) {print $4}' $S/lines.txt | tr ',' '\n' | sort -u | paste -sd,)
  cat $(awk -v l=$L -v s=$S '!/^#/ && ($2==l || $3==l) {print s"/ids/"$1".txt"}' $S/lines.txt) | sort -u > $SL/ids_$L.txt
  build $L $SL/ids_$L.txt $rts
done
[[ -n ${FL_DRY:-} ]] && { echo "== dry run, stopping after builds"; exit 0; }
run() { # run <tag> <side> <k> <label> <rts>
  local tag=$1 s=$2 k=$3 L=$4 rts=$5 D=$SL/$4 log=$SL/out/run_$1_$2$3.log WC=(); [[ $4 == *_wc ]] && WC=(--dump-perf-counters)
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$tag.txt
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TT_LLK_DISABLE_ASSERTS=1 TMPDIR=$SL/builds/$L LLK_PERF_L1_MUX_GROUP=0 LLK_PERF_RUN_TYPES=$rts \
      PERF_RUN_TAG=${tag}_$s$k PERF_KEEP_RUNS=4000 timeout -s INT -k 60 900 $PY -m pytest --compile-consumer "${WC[@]}" -p no:cacheprovider \
      --override-ini=log_cli=false -q -o timeout_method=thread "${N[@]}" < /dev/null > $log 2>&1 )
  echo "   $tag $s$k rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $log | tail -1) $(date -u +%T)"
  if grep -q "TIMED OUT" $log; then echo "== TENSIX TIMED OUT in $tag $s$k, stopping"; return 1; fi
}
stop=0
while read -r tag A B rts reps; do
  [[ -z $tag || $tag == \#* || $stop == 1 ]] && continue
  echo "== $tag ($A against $B, $rts) $(date -u +%T)"; k=0; blocks=${reps:-2}
  for ((b = 0; b < blocks; b++)); do for s in A B B A; do k=$((k+1)); L=$A; [[ $s == B ]] && L=$B; run $tag $s $k $L $rts || { stop=1; break 2; }; done; done
done < $S/lines.txt
while read -r L ids; do
  [[ -z $L || $L == \#* || $stop == 1 ]] && continue
  D=$SL/$L; mkdir -p $SL/out/bits_$L
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$ids
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/builds/bits_$L RA_HASHLOG=$SL/out/bits_$L.hash PYTHONPATH=$S/tools \
      timeout -s INT -k 60 1800 $PY -m pytest --compile-producer -n $NP -p no:cacheprovider -p ra_hashlog --override-ini=log_cli=false -q \
      "${N[@]}" < /dev/null > $SL/out/bitsprod_$L.log 2>&1
    env CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/builds/bits_$L FL_BITS_OUT=$SL/out/bits_$L \
      timeout -s INT -k 60 2400 $PY -m pytest --compile-consumer -p no:cacheprovider --override-ini=log_cli=false -q -o timeout_method=thread \
      "${N[@]}" < /dev/null > $SL/out/bits_$L.log 2>&1 )
  echo "== bits $L: $(grep -aE '[0-9]+ (passed|failed|error)' $SL/out/bits_$L.log | tail -1) $(ls $SL/out/bits_$L | wc -l) files $(date -u +%T)"
done < $S/bits.txt
for L in $labels; do
  [[ -s $SL/out/$L.hash ]] && $PY $S/tools/ra_imgs.py $SL/builds/$L $SL/out/$L.hash > $SL/out/img_$L.tsv 2>/dev/null
  [[ -s $SL/out/bits_$L.hash ]] && $PY $S/tools/ra_imgs.py $SL/builds/bits_$L $SL/out/bits_$L.hash > $SL/out/imgbits_$L.tsv 2>/dev/null
done
while read -r tag A B rts reps; do
  [[ -z $tag || $tag == \#* ]] && continue
  a=$(cut -f3 $SL/out/img_$A.tsv 2>/dev/null | tr ' ' '\n' | grep '^math=' | sort | md5sum | cut -c1-8); b=$(cut -f3 $SL/out/img_$B.tsv 2>/dev/null | tr ' ' '\n' | grep '^math=' | sort | md5sum | cut -c1-8)
  [[ $a == $b ]] && echo "== VOID $tag: $A and $B build the same math images" || echo "== $tag sides differ ($a $b)"
done < $S/lines.txt
BL=$(awk '!/^#/ && NF {print $1}' $S/bits.txt | paste -sd' '); [[ -n $BL ]] && $PY $S/tools/bits_tab.py $SL/out $BL
for L in $BL; do $PY $S/tools/bits_pair.py $SL/out/bits_$L; rm -rf $SL/out/bits_$L; done
tar -czf $SL/fl_out.tgz -C $SL perf_data/runs out
echo "== results $(stat -c %s $SL/fl_out.tgz) bytes"; echo "=====RA_BEGIN====="; base64 -w 76 $SL/fl_out.tgz; echo "=====RA_END====="
exit 0
