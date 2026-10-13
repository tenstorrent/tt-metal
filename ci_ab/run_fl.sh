#!/usr/bin/env bash
# #58822 floors round on one CI Blackhole runner (knowledge/process/17_CI_AB.md), path-matched tt-llk harness A/B of the
# reduce header. Slot /tmp/fl/slot is the checkout's device and test dirs plus ci_ab/tests; every variant is that slot with
# its own reduce header (ci_ab/hdr/<v>.h: main, head, n) compiled into its own build dir, so all sides run from one tree
# path. Fresh build dirs and JIT cache per job; both sides in one job, alternated; the A/A pair is main built twice (ma).
# The ELF guard stops the job when the new side's UInt32 and four-tile row kernels equal the head's (a void A/B).
# usage: run_fl.sh f1..f5   sweep (PR module + UInt32 rows) ABBA twice for main, ma, head, n; counters of the targets for
#                       head and n; row and UInt32 bits of main and n; the four reduce modules with LLK asserts on and
#                       test_sfpu_reduce.py without load macros, on n
set +e +o pipefail
JOB=${1:?}
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; SL=/tmp/fl; cd $R
rm -rf $SL; mkdir -p $SL/perf_data $SL/out $SL/builds
echo "== $JOB runner $(hostname) $(nproc) cores $(date -u +%FT%TZ) head $(git rev-parse --short=11 HEAD)"; ls /dev/tenstorrent 2>&1 | head -3
python3 -m venv $SL/venv && $SL/venv/bin/pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip.log 2>&1
echo "== pip rc=$? $(tail -1 $SL/pip.log)"; PY=$SL/venv/bin/python3
WANT=$(grep -o "sfpi_version='[0-9.]*'" tt_metal/sfpi-version | cut -d"'" -f2)
SF=""
for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]] && $c/compiler/bin/riscv-tt-elf-g++ --version | grep -q "$WANT" && { SF=$c; break; }; done
if [[ -z $SF ]]; then (cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $SL/sfpi.log 2>&1); SF=$R/tt_metal/tt-llk/tests/sfpi; fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1) (pin $WANT)"
$SF/compiler/bin/riscv-tt-elf-g++ --version | grep -q "$WANT" || { echo "== wrong sfpi, stopping"; exit 1; }
P="tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/tools/profiler tt_metal/sfpi-version ttnn/cpp/ttnn/operations/experimental"
HDR=tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h
NP=$(( $(nproc) / 2 )); (( NP > 16 )) && NP=16; (( NP < 4 )) && NP=4
md5sum $S/hdr/*.h $HDR
mkslot() {  # mkslot <name>
  local D=$SL/$1; mkdir -p $D; tar -C $R -cf - $P | tar -C $D -xf -
  cp $S/tests/python_tests/* $D/tt_metal/tt-llk/tests/python_tests/
  ln -sfn $SF $D/tt_metal/tt-llk/tests/sfpi; rm -rf $D/tt_metal/tt-llk/perf_data; ln -sfn $SL/perf_data $D/tt_metal/tt-llk/perf_data
}
env_for() { if [[ ${ASSERTS:-0} == 1 ]]; then echo "-u TT_LLK_DISABLE_ASSERTS"; else echo "TT_LLK_DISABLE_ASSERTS=1"; fi; }
build() {  # build <slot> <hdr> <build> <ids> [pytest args]; ASSERTS=1 in the environment for asserts on
  local slot=$1 h=$2 b=$3 ids=$4; shift 4; local D=$SL/$slot
  mkdir -p $SL/builds/$b
  cp $S/hdr/$h.h $D/$HDR || { echo "== no header $h"; return 1; }
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$ids.txt
    env -u RUNNER_TEMP $(env_for) CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/builds/$b LLK_PERF_L1_MUX_GROUP=0 ${RTS:+LLK_PERF_RUN_TYPES=$RTS} \
      timeout -s INT -k 60 2400 $PY -m pytest --compile-producer -n $NP -p no:cacheprovider --override-ini=log_cli=false -q "$@" "${N[@]}" \
      < /dev/null > $SL/out/prod_$b.log 2>&1 )
  echo "== built $b ($slot, $h $(md5sum < $S/hdr/$h.h | cut -c1-8), $ids): $(grep -aE '[0-9]+ (passed|failed|error)' $SL/out/prod_$b.log | tail -1) $(date -u +%T)"
  grep -a -m3 'error:' $SL/out/prod_$b.log
}
run() {  # run <slot> <build> <ids> <tag> [pytest args]; RTS, ASSERTS and EXTRA (K=V ...) from the environment
  local slot=$1 b=$2 ids=$3 tag=$4; shift 4; local D=$SL/$slot log=$SL/out/run_$tag.log
  [[ -d $SL/builds/$b/tt-llk-build ]] || { echo "== no build dir for $b, stopping"; STOP=1; return 1; }
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$ids.txt
    env -u RUNNER_TEMP $(env_for) ${EXTRA:-} CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/builds/$b LLK_PERF_L1_MUX_GROUP=0 ${RTS:+LLK_PERF_RUN_TYPES=$RTS} \
      PERF_RUN_TAG=$tag PERF_KEEP_RUNS=4000 timeout -s INT -k 60 1200 $PY -m pytest --compile-consumer -p no:cacheprovider \
      --override-ini=log_cli=false -q -o timeout_method=thread "$@" "${N[@]}" < /dev/null > $log 2>&1 )
  echo "   $tag rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $log | tail -1) $(date -u +%T)"
  grep -a -m3 -E "^FAILED|AssertionError" $log
  if grep -q "TIMED OUT" $log; then echo "== TENSIX TIMED OUT in $tag, stopping"; STOP=1; return 1; fi
  if ! grep -qaE '[0-9]+ passed' $log; then echo "== nothing passed in $tag, stopping"; grep -a -m3 -E "Error|No such file" $log; STOP=1; return 1; fi
}
STOP=0
mkslot slot
if [[ $JOB == f1 || $JOB == f2 || $JOB == f3 || $JOB == f4 || $JOB == f5 ]]; then
  build slot main ra_m sweep; build slot main ra_ma sweep; build slot head ra_h sweep; build slot n ra_n sweep
  [[ $JOB == f5 ]] && for k in 1 2 3; do build slot m$k ra_m$k sweep; done
  RTS=MATH_ISOLATE build slot head wc_h wc --dump-perf-counters; RTS=MATH_ISOLATE build slot n wc_n wc --dump-perf-counters
  ASSERTS=1 build slot main rb_m rowb; ASSERTS=1 build slot n rb_n rowb
  ASSERTS=1 build slot main ub_m u32b; ASSERTS=1 build slot n ub_n u32b
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot n fn_n fn
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot n fnd_n fnd --disable-sfploadmacro
  $PY $S/tools/elf_guard.py $SF/compiler/bin/riscv-tt-elf-objcopy $SL/builds $SL/out/elf_sha.txt | tee $SL/out/elf_guard.txt
  grep -q "ELF GUARD PASS" $SL/out/elf_guard.txt || { echo "== ELF guard failed, stopping"; STOP=1; }
  echo "== C: sweep, n against main and head, A/A pair $(date -u +%T)"; n=0
  c() { n=$((n+1)); [[ $STOP == 1 ]] && return; run slot ra_$1 sweep f1c_$1_$n; }
  if [[ $JOB == f5 ]]; then
    for blk in 1 2; do c m; c ma; c h; c n; c m1; c m2; c m3; c m3; c m2; c m1; c n; c h; c ma; c m; done
  else
    for blk in 1 2; do c m; c ma; c h; c n; c n; c h; c ma; c m; done
  fi
  echo "== W: counters $(date -u +%T)"; n=0
  w() { n=$((n+1)); [[ $STOP == 1 ]] && return; RTS=MATH_ISOLATE run slot wc_$1 wc f1w_$1_$n --dump-perf-counters; }
  w h; w n; w n; w h
  echo "== B: bits $(date -u +%T)"
  [[ $STOP == 0 ]] && ASSERTS=1 EXTRA="ROWB_OUT=$SL/out/rowb_m" run slot rb_m rowb f1b_rb_m
  [[ $STOP == 0 ]] && ASSERTS=1 EXTRA="ROWB_OUT=$SL/out/rowb_n" run slot rb_n rowb f1b_rb_n
  $PY $S/tools/bits_cmp.py $SL/out/rowb_m $SL/out/rowb_n | tee $SL/out/rowb_cmp.txt
  [[ $STOP == 0 ]] && ASSERTS=1 EXTRA="U32_OUT=$SL/out/u32_m" run slot ub_m u32b f1b_ub_m
  [[ $STOP == 0 ]] && ASSERTS=1 EXTRA="U32_OUT=$SL/out/u32_n" run slot ub_n u32b f1b_ub_n
  $PY $S/tools/bits_cmp.py $SL/out/u32_m $SL/out/u32_n | tee $SL/out/u32_cmp.txt
  echo "== E: functional with asserts on $(date -u +%T)"
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fn_n fn f1e_fn_n
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fnd_n fnd f1e_fnd_n --disable-sfploadmacro
fi
for f in $SL/out/prod_*.log; do grep -aE '[0-9]+ (passed|failed|error)|error:' $f | tail -5 > $f.tail; rm -f $f; done
(cd $SL && find perf_data/runs -name "*.csv" ! -name "*.post.csv"; cd $SL && find out -type f) > $SL/list.txt
tar -czf $SL/fl_out.tgz -C $SL -T $SL/list.txt
echo "== results $(stat -c %s $SL/fl_out.tgz) bytes"; echo "=====FL_BEGIN====="; base64 -w 4000 $SL/fl_out.tgz; echo "=====FL_END====="
exit 0
