#!/usr/bin/env bash
# Scratch, not for merge: the MoE DRAM expert matmul at the fused MoE's gate/up and down configurations on one P150.
# Arms (models/.../matmul_expert_compressed_dram.hpp swapped in before each run, each run its own fresh JIT cache):
# H (the head), P (the head with every compute call removed: the reader-only twin), H2 (H again: A/A),
# ZA and ZB (H with sum zones: reader DRAM barrier, CB-full reserve, token and fmt waits; TRISC weight wait, LLK call,
# fmt wait), C (H with the fpu, pack, unpack and instrn counters), S and PS (H and P with six half-block slots in\n# the same CB bytes, up to five in flight; the test halves subblock_k). One JIT cache path, emptied before every run.
set +e +o pipefail
R=$(git rev-parse --show-toplevel 2>/dev/null); R=${R:-/work}; S=$R/ci_ab; cd $R
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=$R:$R/tools
export PYTHONPATH=$R:$R/tools:${PYTHONPATH:-}
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') runtime $RT"
T=models/demos/deepseek_v3_b1/tests/unit_tests/per_core_allocation
cp $S/test_mexp_ab.py $T/test_mexp_ab.py
F=models/demos/deepseek_v3_b1/unified_kernels/matmul_expert_compressed_dram.hpp
cmp -s $S/v/H/$F $F || { echo "VOID: H differs from the tree"; exit 1; }
DEST=$(find "$RT" -path "*/$F" -type f 2>/dev/null); echo "== package copies: ${DEST:-none}"
python3 -m pytest --collect-only -q $T/test_mexp_ab.py 2>/dev/null | grep -oE 'test_mexp\[[^]]*\]' | sort -u | sed "s|^|$T/test_mexp_ab.py::|" > $S/ids.txt
cat $S/ids.txt; [[ -s $S/ids.txt ]] || { python3 -m pytest --collect-only -q $T/test_mexp_ab.py 2>&1 | tail -20; exit 1; }
md5sum $S/v/*/$F
declare -A EH
O=/work/generated/d1; mkdir -p $O
step() {
  local arm=$1 r=$2 extra=$3 src=${1%2}; src=${src%[0-9]}
  [[ $arm == C* ]] && src=H
  cp $S/v/$src/$F $R/$F; for d in $DEST; do cp $S/v/$src/$F $d; done
  local out=$O/${arm}_$r; rm -rf $out /tmp/ttc_cur; mkdir -p $out
  local t0=$(date +%s)
  MEXP_SPLIT=$([[ $src == S || $src == PS ]] && echo 2 || echo 1) TT_METAL_CACHE=/tmp/ttc_cur IDS_FILE=$S/ids.txt timeout -s INT -k 60 600 python3 -m tracy -p -r --no-web-server $extra -o $out -m ci_ab.tools.run_nodes > $out.log 2>&1
  local rc=$?
  echo "== $arm r$r rc=$rc $(( $(date +%s) - t0 )) s: $(grep -aE '[0-9]+ (passed|failed)' $out.log | tail -1)"
  [[ $rc == 124 ]] && { echo "== TIMEOUT in $arm, stopping"; grep -a -m5 -iE "timeout|hang|error" $out.log | cut -c1-200; exit 1; }
  grep -aqE "passed|failed" $out.log || { echo NO_PYTEST_OUTPUT; tail -c 3000 $out.log; }
  [[ $arm == H && $r == 1 ]] && { grep -aE "^(FAILED|ERROR)" $out.log | head -8; grep -a "tile counts" $out.log | sed "s/.*tile counts: //" | sort | uniq -c | head -12; }
  [[ $arm == P && $r == 1 ]] && grep -aE "^(FAILED|ERROR)" $out.log | cut -c1-200 | head -4
  local e=$(python3 $S/tools/elfsum.py $arm /tmp/ttc_cur); echo "$e"; EH[$arm$r]=$e
  if [[ $arm == C* ]]; then python3 $S/tools/ctr.py $out $arm; python3 $S/tools/dlog.py $out $arm $r
  elif [[ $arm == Z* ]]; then python3 $S/tools/dlog.py $out $arm $r detail
  else python3 $S/tools/dlog.py $out $arm $r; fi
  rm -rf /tmp/ttc_cur
}
step H 1; step P 1; step H2 1
h() { echo "${EH[$1]}" | sed 's/^ELF [A-Z0-9]* //'; }
t0of() { echo "${EH[$1]}" | grep -o 'trisc0:[^ ]*'; }
if [[ -z $(t0of H1) || $(t0of H1) == $(t0of P1) || $(h H1) != $(h H21) ]]; then echo "== GUARD FAIL: H $(h H1) P $(h P1) H2 $(h H21)"; exit 1; fi
echo "== GUARD ok: trisc0 H $(t0of H1) P $(t0of P1); H == H2"
step ZA 1 --enable-sum-profiling
step ZB 1 --enable-sum-profiling
step C 1 --profiler-capture-perf-counters=fpu,pack,unpack,instrn
step H 2; step S 1; step PS 1; step S2 1; step H2 2
cp $S/v/H/$F $R/$F; for d in $DEST; do cp $S/v/H/$F $d; done
echo "== DONE $(date -u +%T)"
