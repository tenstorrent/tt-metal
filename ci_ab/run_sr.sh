#!/usr/bin/env bash
# #58822 regression audit phase 2 on one CI Blackhole runner (knowledge/process/17_CI_AB.md), path-matched tt-llk harness
# A/B. Slot /tmp/sr/slot is the checkout's device and test dirs plus ci_ab/tests; every variant is that slot with its own
# reduce header (ci_ab/hdr/<v>.h) compiled into its own build dir, so all sides run from one tree path. Slot /tmp/sr/dc
# adds #58819's device files (ci_ab/dc). Fresh build dirs and JIT cache per job. Both sides in one job, alternated; the
# A/A pair is main built twice into two build dirs of the same slot (ma); the layout-only A/A is main plus 1 to 4 RISC
# nops at the start of init_reduce (m1..m4), in the same job.
# usage: run_sr.sh j1   pacing scan of the Int32 column SUM (main, head, p1..p12), the production shadows for the best
#                       two by tools/sel_k.py's rule, the sweep with SDPA for main, head, g, c and the layout A/A m1..m3
#        run_sr.sh j2   the combined slot (main against c with #58819), c's four reduce modules with LLK asserts on,
#                       test_sfpu_reduce.py without load macros, the float column bits (main, c), ttnn ident.py on c
set +e +o pipefail
JOB=${1:?}
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; SL=/tmp/sr; cd $R
rm -rf $SL; mkdir -p $SL/perf_data $SL/out $SL/builds
echo "== $JOB runner $(hostname) $(nproc) cores $(date -u +%FT%TZ)"; ls /dev/tenstorrent 2>&1 | head -3
python3 -m venv $SL/venv && $SL/venv/bin/pip install -q -r tt_metal/tt-llk/tests/requirements.txt > $SL/pip.log 2>&1
echo "== pip rc=$? $(tail -1 $SL/pip.log)"; PY=$SL/venv/bin/python3
SF=""
for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]] && { SF=$c; break; }; done
if [[ -z $SF ]]; then (cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > $SL/sfpi.log 2>&1); SF=$R/tt_metal/tt-llk/tests/sfpi; fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version | head -1)"
$SF/compiler/bin/riscv-tt-elf-g++ --version | grep -q '7\.86\.0' || { echo "== wrong sfpi, stopping"; exit 1; }
P="tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/tools/profiler tt_metal/sfpi-version ttnn/cpp/ttnn/operations/experimental"
HDR=tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h
NP=$(( $(nproc) / 2 )); (( NP > 16 )) && NP=16; (( NP < 4 )) && NP=4
mkslot() {  # mkslot <name> [overlay dir]
  local D=$SL/$1; mkdir -p $D; tar -C $R -cf - $P | tar -C $D -xf -
  cp $S/tests/sources/* $D/tt_metal/tt-llk/tests/sources/; cp $S/tests/python_tests/* $D/tt_metal/tt-llk/tests/python_tests/
  if [[ -n ${2:-} ]]; then (cd $2 && find . -type f) | while read -r f; do cp "$2/$f" "$D/$f"; done; fi
  ln -sfn $SF $D/tt_metal/tt-llk/tests/sfpi; rm -rf $D/tt_metal/tt-llk/perf_data; ln -sfn $SL/perf_data $D/tt_metal/tt-llk/perf_data
}
env_for() { if [[ ${ASSERTS:-0} == 1 ]]; then echo "-u TT_LLK_DISABLE_ASSERTS"; else echo "TT_LLK_DISABLE_ASSERTS=1"; fi; }
build() {  # build <slot> <hdr> <build> <ids> [pytest args]; ASSERTS=1 in the environment for asserts on
  local slot=$1 h=$2 b=$3 ids=$4; shift 4; local D=$SL/$slot
  # tempfile ignores a TMPDIR that does not exist (every build would land in /tmp/tt-llk-build)
  mkdir -p $SL/builds/$b
  cp $S/hdr/$h.h $D/$HDR || { echo "== no header $h"; return 1; }
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$ids.txt
    env -u RUNNER_TEMP $(env_for) CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/builds/$b LLK_PERF_L1_MUX_GROUP=0 ${RTS:+LLK_PERF_RUN_TYPES=$RTS} \
      timeout -s INT -k 60 2400 $PY -m pytest --compile-producer -n $NP -p no:cacheprovider --override-ini=log_cli=false -q "$@" "${N[@]}" \
      < /dev/null > $SL/out/prod_$b.log 2>&1 )
  echo "== built $b ($slot, $h, $ids): $(grep -aE '[0-9]+ (passed|failed|error)' $SL/out/prod_$b.log | tail -1) $(date -u +%T)"
  grep -a -m3 'error:' $SL/out/prod_$b.log
}
run() {  # run <slot> <build> <ids> <tag> [pytest args]; RTS, ASSERTS and EXTRA (K=V ...) from the environment
  local slot=$1 b=$2 ids=$3 tag=$4; shift 4; local D=$SL/$slot log=$SL/out/run_$tag.log
  [[ -d $SL/builds/$b/tt-llk-build ]] || { echo "== no build dir for $b, stopping"; STOP=1; return 1; }
  ( cd $D/tt_metal/tt-llk/tests/python_tests; mapfile -t N < $S/ids/$ids.txt
    env -u RUNNER_TEMP $(env_for) ${EXTRA:-} CHIP_ARCH=blackhole LLK_HOME=$D/tt_metal/tt-llk TMPDIR=$SL/builds/$b LLK_PERF_L1_MUX_GROUP=0 ${RTS:+LLK_PERF_RUN_TYPES=$RTS} \
      PERF_RUN_TAG=$tag PERF_KEEP_RUNS=4000 timeout -s INT -k 60 900 $PY -m pytest --compile-consumer -p no:cacheprovider \
      --override-ini=log_cli=false -q -o timeout_method=thread "$@" "${N[@]}" < /dev/null > $log 2>&1 )
  echo "   $tag rc=$? $(grep -aE '[0-9]+ (passed|failed|error)' $log | tail -1) $(date -u +%T)"
  if grep -q "TIMED OUT" $log; then echo "== TENSIX TIMED OUT in $tag, stopping"; STOP=1; return 1; fi
  if ! grep -qaE '[0-9]+ passed' $log; then echo "== nothing passed in $tag, stopping"; grep -a -m3 -E "Error|No such file" $log; STOP=1; return 1; fi
}
STOP=0
mkslot slot
if [[ $JOB == j1 ]]; then
  build slot main ra_m b_m; build slot main ra_ma b_s; build slot head ra_h b_h; build slot g2 ra_g b_s; build slot c ra_c b_s
  for k in 1 2 3 4; do build slot m$k ra_m$k b_s; done
  for k in $(seq 12); do build slot p$k ra_p$k b_p; done
  echo "== A: Int32 column SUM pacing scan $(date -u +%T)"; n=0
  a() { n=$((n+1)); [[ $STOP == 1 ]] && return; RTS=L1_TO_L1,MATH_ISOLATE run slot ra_$1 i32cs a1_$1_$n; }
  a m; a h; for k in $(seq 12); do a p$k; done; a h; a m; for k in $(seq 12 -1 1); do a p$k; done; a m; a h
  $PY $S/tools/sel_k.py a1 $SL/perf_data/runs | tee $SL/out/sel_k.txt
  best=$(grep '^BEST' $SL/out/sel_k.txt | cut -d' ' -f2-)
  echo "== B: production shadows ($best) $(date -u +%T)"; n=0
  for v in m h $(echo $best | tr -d none); do
    if [[ $v == m ]]; then cat $S/ids/prod_mlir.txt $S/ids/prod_m.txt > $S/ids/_b_$v.txt; else cat $S/ids/prod_mlir.txt $S/ids/prod_h.txt > $S/ids/_b_$v.txt; fi
  done
  b() { n=$((n+1)); [[ $STOP == 1 ]] && return; RTS=L1_TO_L1 run slot ra_$1 _b_$1 b1_$1_$n; }
  if [[ $best != none && -n $best ]]; then set -- $best; p1=$1; p2=${2:-}
    b m; b h; b $p1; [[ -n $p2 ]] && b $p2; [[ -n $p2 ]] && b $p2; b $p1; b h; b m
  else b m; b h; b h; b m; fi
  echo "== C: sweep with SDPA $(date -u +%T)"; n=0
  c() { n=$((n+1)); [[ $STOP == 1 ]] && return; run slot ra_$1 b_s c1_$1_$n; }
  for blk in 1 2; do c m; c ma; c h; c g; c c; c m1; c m2; c m3; c m4; c m4; c m3; c m2; c m1; c c; c g; c h; c ma; c m; done
elif [[ $JOB == j2 ]]; then
  mkslot dc $S/dc
  build dc main dc_m b_s; build dc main dc_ma b_s; build dc c dc_c b_s
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot c fn_c fn
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot c fnd_c fnd --disable-sfploadmacro
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot main fb_m fbits
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot c fb_c fbits
  echo "== D: combined with #58819 $(date -u +%T)"; n=0
  d() { n=$((n+1)); [[ $STOP == 1 ]] && return; run dc dc_$1 b_s d1_$1_$n; }
  for blk in 1 2; do d m; d ma; d c; d c; d ma; d m; done
  echo "== E: functional and bits $(date -u +%T)"
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fn_c fn fn_c_1
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fnd_c fnd fnd_c_1 --disable-sfploadmacro
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE EXTRA="FCOL_OUT=$SL/out/fcol_m" run slot fb_m fbits fb_m_1
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE EXTRA="FCOL_OUT=$SL/out/fcol_c" run slot fb_c fbits fb_c_1
  $PY $S/tools/fcol_cmp.py $SL/out/fcol_m $SL/out/fcol_c | tee $SL/out/fcol_cmp.txt
  if [[ $STOP == 0 ]]; then
    echo "== F: ttnn ident.py on the checkout (the merged head with c), fresh JIT cache $(date -u +%T)"
    ( export HWLOCK_HELD=ci TT_METAL_CACHE=$SL/ttcache; mkdir -p $TT_METAL_CACHE; cd $R
      timeout -s INT -k 60 1200 python3 $S/tools/ident.py $SL/out/ident_c.pt > $SL/out/ident_c.log 2>&1; echo "   ident rc=$? $(grep -a '^IDENT ' $SL/out/ident_c.log)" )
    grep -a -m5 -E "Error|Traceback" $SL/out/ident_c.log
  fi
elif [[ $JOB == j3 ]]; then
  # The final head f (gate, paired float column groups, six-cycle pacing of the Int32 column SUM): its sweep against
  # main with an A/A pair, the Int32 pipeline and production shadows against main and the previous head, the combined
  # slot with #58819, the four reduce modules with LLK asserts on, the float column bits, ttnn ident.py on the checkout.
  mkslot dc $S/dc
  build slot main ra_m b_m; build slot main ra_ma b_s; build slot head ra_h b_h; build slot f ra_f b_h
  for k in 1 2 3 4; do build slot m$k ra_m$k b_s; done
  build dc main dc_m b_s; build dc main dc_ma b_s; build dc f dc_f b_s
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot f fn_f fn
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot f fnd_f fnd --disable-sfploadmacro
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot main fb_m fbits
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot f fb_f fbits
  for v in m h f; do
    if [[ $v == m ]]; then cat $S/ids/i32cs.txt $S/ids/prod_mlir.txt $S/ids/prod_m.txt > $S/ids/_q_$v.txt; else cat $S/ids/i32cs.txt $S/ids/prod_mlir.txt $S/ids/prod_h.txt > $S/ids/_q_$v.txt; fi
  done
  echo "== Q: Int32 column SUM pipeline and production shadows $(date -u +%T)"; n=0
  q() { n=$((n+1)); [[ $STOP == 1 ]] && return; RTS=L1_TO_L1,MATH_ISOLATE run slot ra_$1 _q_$1 q3_$1_$n; }
  for blk in 1 2; do q m; q h; q f; q f; q h; q m; done
  echo "== C: sweep with SDPA, f against main, A/A pair $(date -u +%T)"; n=0
  c() { n=$((n+1)); [[ $STOP == 1 ]] && return; run slot ra_$1 b_s c3_$1_$n; }
  for blk in 1 2; do c m; c ma; c f; c m1; c m2; c m3; c m4; c m4; c m3; c m2; c m1; c f; c ma; c m; done
  echo "== D: combined with #58819 $(date -u +%T)"; n=0
  d() { n=$((n+1)); [[ $STOP == 1 ]] && return; run dc dc_$1 b_s d3_$1_$n; }
  for blk in 1 2; do d m; d ma; d f; d f; d ma; d m; done
  echo "== E: functional and bits $(date -u +%T)"
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fn_f fn fn_f_1
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fnd_f fnd fnd_f_1 --disable-sfploadmacro
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE EXTRA="FCOL_OUT=$SL/out/fcol_m" run slot fb_m fbits fb_m_1
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE EXTRA="FCOL_OUT=$SL/out/fcol_f" run slot fb_f fbits fb_f_1
  $PY $S/tools/fcol_cmp.py $SL/out/fcol_m $SL/out/fcol_f | tee $SL/out/fcol_cmp.txt
  if [[ $STOP == 0 ]]; then
    echo "== F: ttnn ident.py on the checkout (the final head), fresh JIT cache $(date -u +%T)"
    ( export HWLOCK_HELD=ci TT_METAL_CACHE=$SL/ttcache; mkdir -p $TT_METAL_CACHE; cd $R
      timeout -s INT -k 60 1500 python3 $S/tools/ident.py $SL/out/ident_f.pt > $SL/out/ident_f.log 2>&1; echo "   ident rc=$? $(grep -a '^IDENT ' $SL/out/ident_f.log)" )
    grep -a -m5 -E "Error|Traceback" $SL/out/ident_f.log
  fi
elif [[ $JOB == j4 ]]; then
  # The final head f2 (f plus the init gate for Float16_b AVG in a 16-bit Dest and Int32 AVG): its sweep against main
  # with an A/A pair, the Int32 pipeline and production shadows against main and the previous head, the four reduce
  # modules with LLK asserts on, test_sfpu_reduce.py without load macros, the float column bits.
  build slot main ra_m b_m; build slot main ra_ma b_s; build slot head ra_h b_h; build slot f2 ra_f2 b_h
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot f2 fn_f2 fn
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot f2 fnd_f2 fnd --disable-sfploadmacro
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot main fb_m fbits
  ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE build slot f2 fb_f2 fbits
  for v in m h f2; do
    if [[ $v == m ]]; then cat $S/ids/i32cs.txt $S/ids/prod_mlir.txt $S/ids/prod_m.txt > $S/ids/_q_$v.txt; else cat $S/ids/i32cs.txt $S/ids/prod_mlir.txt $S/ids/prod_h.txt > $S/ids/_q_$v.txt; fi
  done
  echo "== Q: Int32 column SUM pipeline and production shadows $(date -u +%T)"; n=0
  q() { n=$((n+1)); [[ $STOP == 1 ]] && return; RTS=L1_TO_L1,MATH_ISOLATE run slot ra_$1 _q_$1 q4_$1_$n; }
  for blk in 1 2; do q m; q h; q f2; q f2; q h; q m; done
  echo "== C: sweep with SDPA, f2 against main and the previous head, A/A pair $(date -u +%T)"; n=0
  c() { n=$((n+1)); [[ $STOP == 1 ]] && return; run slot ra_$1 b_s c4_$1_$n; }
  for blk in 1 2; do c m; c ma; c h; c f2; c f2; c h; c ma; c m; done
  echo "== E: functional and bits $(date -u +%T)"
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fn_f2 fn fn_f2_1
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE run slot fnd_f2 fnd fnd_f2_1 --disable-sfploadmacro
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE EXTRA="FCOL_OUT=$SL/out/fcol_m" run slot fb_m fbits fb_m_1
  [[ $STOP == 0 ]] && ASSERTS=1 RTS=L1_TO_L1,MATH_ISOLATE EXTRA="FCOL_OUT=$SL/out/fcol_f2" run slot fb_f2 fbits fb_f2_1
  $PY $S/tools/fcol_cmp.py $SL/out/fcol_m $SL/out/fcol_f2 | tee $SL/out/fcol_cmp.txt
elif [[ $JOB == j5 ]]; then
  # Counters (--dump-perf-counters) of the float column SUM and AVG and the Int32 column SUM, MATH_ISOLATE: main against
  # the final head f2, ABBA twice.
  RTS=MATH_ISOLATE build slot main wc_m wc --dump-perf-counters; RTS=MATH_ISOLATE build slot f2 wc_f2 wc --dump-perf-counters
  echo "== W: counters $(date -u +%T)"; n=0
  w() { n=$((n+1)); [[ $STOP == 1 ]] && return; RTS=MATH_ISOLATE run slot wc_$1 wc w5_$1_$n --dump-perf-counters; }
  for blk in 1 2; do w m; w f2; w f2; w m; done
fi
# loadable code only (.init, .text, .ldm_data), as the build path is in the ELF's other sections
OBJ=$SF/compiler/bin/riscv-tt-elf-objcopy
for b in $(ls $SL/builds); do
  find $SL/builds/$b -path "*/elf/*.elf" | while read -r e; do
    h=$(for sec in .init .text .ldm_data; do $OBJ -O binary -j $sec "$e" $SL/sec.bin 2>/dev/null && { echo $sec; cat $SL/sec.bin; }; done | sha256sum | cut -c1-16)
    echo "$b ${e#$SL/builds/$b/} $h"; done
done > $SL/out/elf_sha.txt
for f in $SL/out/prod_*.log; do grep -aE '[0-9]+ (passed|failed|error)|error:' $f | tail -5 > $f.tail; rm -f $f; done
(cd $SL && find perf_data/runs -name "*.csv" ! -name "*.post.csv"; cd $SL && find out -type f) > $SL/list.txt
tar -czf $SL/sr_out.tgz -C $SL -T $SL/list.txt
echo "== results $(stat -c %s $SL/sr_out.tgz) bytes"; echo "=====SR_BEGIN====="; base64 -w 4000 $SL/sr_out.tgz; echo "=====SR_END====="
exit 0
