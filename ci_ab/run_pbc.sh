#!/usr/bin/env bash
# #60152 regression audit on one CI Blackhole runner (knowledge/process/17_CI_AB.md): the KERNEL L1_TO_L1 rows of
# perf_pack_block_contiguous. Variants (ci_ab/v/<V>, the files of ci_ab/v/files.txt, placed at their tree paths before
# each build and run, so every variant compiles from the same paths): M main's llk_pack_block.h with the head's perf
# module, H the head (364aa763f57), H2 the head again in its own build (identical-code A/A), and the head with one of
# main's three differences back: X1 main's run sequence (mop_sync, outer count written to mop_cfg[0], TTI_MOP), X2
# main's Z = 0 before the run, X3 main's end ops on a full 32 x 32 tile. The job stops if M and H, or X1/X2 and H, build
# the same pack ELF, or H and H2 do not.
# Each variant builds once in a fresh build dir; runs go forward then backward through the list, twice (L1_TO_L1 only).
# The harness builds under $RUNNER_TEMP in GitHub Actions, so each variant sets it to its own dir.
# Per run: the module's CSV and the split of each device run's KERNEL span from the raw zone entries (tools/raw_reduce.py).
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $R, $(git log --oneline -1), $(date -u +%T)"
src() { case $1 in H2) echo H ;; *) echo $1 ;; esac; }
place() { while read -r rel; do [[ -n $rel ]] && cp "$S/v/$1/$rel" "$R/$rel"; done < $S/v/files.txt; }
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$? $(tail -1 /tmp/pip.txt)"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$? $(tail -2 /tmp/setup.txt | tr '\n' ' ')"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk LLK_PERF_L1_MUX_GROUP=0 TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 PERF_KEEP_RUNS=400
cd $R/tt_metal/tt-llk/tests/python_tests
N=perf_pack_block_contiguous.py
VS="M H H2 X1 X2 X3"
EC=""
for v in $VS; do
  place $(src $v); rm -rf $HOME/pbc_$v
  RUNNER_TEMP=$HOME/pbc_$v TMPDIR=$HOME/pbc_$v timeout -s INT -k 60 900 python -m pytest --compile-producer -n 8 -p no:cacheprovider -q $N > /tmp/b_$v.txt 2>&1
  echo "== build $v rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error)' /tmp/b_$v.txt | tail -1)"
  EC="$EC $v=$HOME/pbc_$v"
done
python $S/tools/elfcheck.py $EC | tee /tmp/elf.txt
python - /tmp/elf.txt <<'PY' || { echo "== GUARD FAIL: the variants did not build apart"; exit 1; }
import sys
h = {l.split()[1]: l.split()[2:] for l in open(sys.argv[1]) if l.startswith("ELF ")}
pk = lambda v: [x.split("/")[-1] for x in h[v]]
ok = all(h[v] for v in h) and pk("M") != pk("H") and pk("H") == pk("H2") and pk("X1") != pk("H") and pk("X2") != pk("H")
print("== GUARD", "ok" if ok else "FAIL", {v: len(h[v]) for v in h}); sys.exit(0 if ok else 1)
PY
RV=$(echo $VS | tr ' ' '\n' | tac | tr '\n' ' ')
j=0
for blk in 1 2; do
  for v in $VS $RV; do
    j=$((j + 1)); place $(src $v); tag=pbc_${j}_$v
    RAW_ZONE_DIR=$HOME/raw/$tag RUNNER_TEMP=$HOME/pbc_$v TMPDIR=$HOME/pbc_$v PERF_RUN_TAG=$tag timeout -s INT -k 60 600 \
      python -m pytest --compile-consumer -p no:cacheprovider -q -rfE $N > /tmp/r_$tag.txt 2>&1
    echo "== run $v ($j) rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed)' /tmp/r_$tag.txt | tail -1)"
    [[ $j == 1 ]] && grep -a -m5 -E "FAILED|Error|error" /tmp/r_$tag.txt | cut -c1-200
    f=$(ls $LLK_HOME/perf_data/runs/$tag/*/perf_pack_block_contiguous*.csv 2>/dev/null | grep -v post | head -1)
    [[ -n $f ]] && { echo "== csv $tag"; cat "$f"; echo "== end csv"; }
    python $S/tools/raw_reduce.py $j $v $HOME/raw/$tag
  done
done
cd $R; place H
echo "== DONE $(date -u +%T)"
