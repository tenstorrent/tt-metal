#!/usr/bin/env bash
# #58817 regression audit: tt-llk harness A/B on one CI Blackhole machine (17_CI_AB.md). Two side trees, head (this
# checkout's tt_metal/tt-llk and tt_metal/hw) and main (the same with main's 12 device files from llkvar/main/tree), each
# with its own build dir; a side is moved to one tree path ($T/w) and one build path ($T/b) for its builds and runs, so
# both sides compile with the same absolute paths. python_tests are the head's on both sides (main's sources ignore the
# head's defines). Perf runs alternate ABBA (the two runs of one side are the A/A pair); then the functional tests with
# LLK asserts on the head. Each run's perf_data goes to the log as base64 tar.gz lines "LLKAB_DATA <tag> <chunk>".
# usage: bash scratch_r3/run_llk_ab.sh
set +e +o pipefail
R=${LLKAB_ROOT:-/work}; S=$R/scratch_r3; T=${LLKAB_TMP:-/tmp/llkab}; rm -rf $T; mkdir -p $T
unset RUNNER_TEMP
export CHIP_ARCH=blackhole LLK_PERF_L1_MUX_GROUP=0 PERF_KEEP_RUNS=400
echo "== llkab start $(date -u +%T) $(git -C $R rev-parse HEAD 2>/dev/null)"

if [[ -n ${LLKAB_PY:-} ]]; then PY=$LLKAB_PY; else
  python3 -m venv $T/venv > $T/venv.log 2>&1
  $T/venv/bin/pip install -q --disable-pip-version-check -r $R/tt_metal/tt-llk/tests/requirements.txt > $T/pip.log 2>&1
  echo "== pip rc=$? $(tail -n 2 $T/pip.log | tr '\n' ' ')"
  PY=$T/venv/bin/python
fi

want=$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" $R/tt_metal/tt-llk/tests/sfpi-version)
SF=${LLKAB_SFPI:-}
for c in $R/runtime/sfpi /opt/tenstorrent/sfpi; do
  if [[ -x $c/compiler/bin/riscv-tt-elf-g++ ]]; then
    have=$(cat $c/sfpi.version 2>/dev/null)
    echo "== sfpi candidate $c version '$have' (want $want)"
    if [[ -z $SF && ( -z $have || $have == "$want" ) ]]; then SF=$c; fi
  fi
done
if [[ -z $SF ]]; then
  (cd $R/tt_metal/tt-llk/tests && bash ./setup_testing_env.sh > $T/sfpi.log 2>&1); echo "== sfpi download rc=$? $(tail -n 1 $T/sfpi.log)"
  SF=$T/sfpi; mv $R/tt_metal/tt-llk/tests/sfpi $SF
fi
echo "== sfpi $SF: $($SF/compiler/bin/riscv-tt-elf-g++ --version 2>&1 | sed -n 1p)"

mkside() {  # mkside <v>: root symlinks to /work, copies of tt_metal/tt-llk and tt_metal/hw, the variant's files on top
  local v=$1 d=$T/s_$1
  mkdir -p $d/tt_metal
  for e in $R/*; do n=$(basename $e); [[ $n == tt_metal ]] || ln -s $e $d/$n; done
  for e in $R/tt_metal/*; do n=$(basename $e); [[ $n == tt-llk || $n == hw ]] || ln -s $e $d/tt_metal/$n; done
  cp -a $R/tt_metal/tt-llk $R/tt_metal/hw $d/tt_metal/
  [[ -d $S/llkvar/$v/tree ]] && cp -a $S/llkvar/$v/tree/. $d/
  rm -rf $d/tt_metal/tt-llk/tests/sfpi $d/tt_metal/tt-llk/perf_data; ln -s $SF $d/tt_metal/tt-llk/tests/sfpi
  echo "== side $v: $(md5sum $d/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h | cut -c1-12) llk_unpack_AB_matmul.h, $(md5sum $d/tt_metal/tt-llk/tests/sources/matmul_test.cpp | cut -c1-12) matmul_test.cpp"
}
mkside head; mkside main

cat > $T/pyrun.py <<'PYEOF'
import sys, pytest
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
sys.exit(pytest.main(["-p", "no:cacheprovider", "--override-ini=log_cli=false", "-q", *sys.argv[2:], *ids]))
PYEOF

use() {  # use <side> <build>: move that side's tree and build dir to the one tree and build path
  mv $T/s_$1 $T/w; mkdir -p $T/bs_$2; mv $T/bs_$2 $T/b
}
unuse() { mv $T/w $T/s_$1; mv $T/b $T/bs_$2; }
harness() {  # harness <side> <build> <ids> <log> [pytest args...]; env from the caller
  local v=$1 b=$2 ids=$3 log=$4; shift 4
  use $v $b
  (cd $T/w/tt_metal/tt-llk/tests/python_tests && \
   LLK_HOME=$T/w/tt_metal/tt-llk TMPDIR=$T/b timeout -s INT -k 60 ${LIM:-1800} $PY $T/pyrun.py $ids "$@") > $log 2>&1
  local rc=$?; unuse $v $b; return $rc
}
build() {  # build <side> <build> <ids> [asserts]
  if [[ ${4:-} == asserts ]]; then harness $1 $2 $3 $T/prod_$2_$(basename $3).log --compile-producer -n ${LLKAB_NPROD:-16}
  else TT_LLK_DISABLE_ASSERTS=1 harness $1 $2 $3 $T/prod_$2_$(basename $3).log --compile-producer -n ${LLKAB_NPROD:-16}; fi
  echo "== build $2 ($1, $(basename $3)) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/prod_$2_$(basename $3).log | tail -n 1)"
}
run() {  # run <side> <build> <ids> <tag> <run types|-> <limit>
  [[ -n ${LLKAB_DRY:-} ]] && return
  local v=$1 b=$2 ids=$3 tag=$4 rts=$5
  if [[ $rts == - ]]; then unset LLK_PERF_RUN_TYPES; else export LLK_PERF_RUN_TYPES=$rts; fi
  TT_LLK_DISABLE_ASSERTS=1 PERF_RUN_TAG=$tag LIM=$6 harness $v $b $ids $T/cons_$tag.log --compile-consumer -o timeout_method=thread --timeout 120
  echo "== run $tag ($v, $b) rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_$tag.log | tail -n 1)"
  grep -aE '^(FAILED|ERROR) ' $T/cons_$tag.log | cut -c1-300 | sed -n 1,20p
  local pd=$T/s_$v/tt_metal/tt-llk/perf_data/runs
  if [[ -d $pd/$tag ]]; then
    tar czf $T/$tag.tgz -C $pd $tag; echo "== data $tag $(wc -c < $T/$tag.tgz) bytes"
    base64 -w 20000 $T/$tag.tgz | sed "s/^/LLKAB_DATA $tag /"
  fi
}

cat $S/ids_gap.txt $S/ids_full16.txt > $T/ids_perf.txt
build head hp $T/ids_perf.txt; build main mp $T/ids_perf.txt
echo "== ELF identity, main against head builds (loaded PROGBITS sections, per variant dir and thread)"
cat > $T/elfid.py <<'PYEOF'
import hashlib, os, re, subprocess, sys, collections
ro, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
def img(p):
    h = hashlib.sha256()
    for line in subprocess.run([ro + "readelf", "-S", "-W", p], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", line)
        if m and m.group(2) == "PROGBITS" and "A" in m.group(3):
            h.update(subprocess.run([ro + "readelf", "-x", m.group(1), p], capture_output=True, text=True).stdout.encode())
    return h.hexdigest()
cnt = collections.Counter()
for dp, dn, fn in os.walk(a):
    for f in fn:
        if not f.endswith(".elf") or os.path.basename(dp) != "elf": continue
        pa = os.path.join(dp, f); rel = os.path.relpath(pa, a); pb = os.path.join(b, rel)
        bh = os.path.join(os.path.dirname(dp), "build.h")
        k = " ".join(sorted(set(re.findall(r"MATMUL_[A-Z_]+|SyncFull|is_fp32_dest_acc_en = true", open(bh).read())))) if os.path.exists(bh) else ""
        st = "unmatched" if not os.path.exists(pb) else ("same" if img(pa) == img(pb) else "differs")
        cnt[(f, st, k)] += 1
for (f, st, k), n in sorted(cnt.items()): print(f"   {n:4d} {f:10s} {st:9s} {k}")
PYEOF
$PY $T/elfid.py $SF/compiler/bin/riscv-tt-elf- $T/bs_hp $T/bs_mp

G=L1_TO_L1,UNPACK_ISOLATE,MATH_ISOLATE,L1_CONGESTION; F=L1_TO_L1,UNPACK_ISOLATE,L1_CONGESTION
run main mp $S/ids_gap.txt G_m1 $G 900
run head hp $S/ids_gap.txt G_h1 $G 900
run main mp $S/ids_full16.txt F_m1 $F 900
run head hp $S/ids_full16.txt F_h1 $F 900
run head hp $S/ids_full16.txt F_h2 $F 900
run main mp $S/ids_full16.txt F_m2 $F 900
run head hp $S/ids_gap.txt G_h2 $G 900
run main mp $S/ids_gap.txt G_m2 $G 900

build head ha $S/ids_func.txt asserts
[[ -n ${LLKAB_DRY:-} ]] && exit 0
unset LLK_PERF_RUN_TYPES
LIM=1500 harness head ha $S/ids_func.txt $T/cons_func.log --compile-consumer -o timeout_method=thread --timeout 120
echo "== functional, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed|error)' $T/cons_func.log | tail -n 1)"
grep -aE '^(FAILED|ERROR) ' $T/cons_func.log | cut -c1-400 | sed -n 1,40p
echo "== llkab end $(date -u +%T)"
