#!/usr/bin/env bash
# Scratch job: the tilize functional tests (bits against the goldens) on one or more sides of run.sh's variant sets.
# usage: func.sh "<sides>"
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
echo "== checkout $(git log --oneline -1 | cut -c1-80), $(hostname), $(date -u +%T)"
src() { case $1 in M2) echo M ;; *) echo $1 ;; esac; }
place() {
  local s=$(src $1)
  while read -r f; do [[ -n $f ]] && rm -f "$R/$f"; done < $S/v/$s.absent
  (cd $S/v/$s && find . -type f | while read -r f; do mkdir -p "$R/$(dirname $f)"; cp "$f" "$R/$f"; done)
}
python -m venv $HOME/llkvenv && source $HOME/llkvenv/bin/activate
pip install -q -r tt_metal/tt-llk/tests/requirements.txt > /tmp/pip.txt 2>&1; echo "== pip rc=$?"
(cd tt_metal/tt-llk/tests && ./setup_testing_env.sh > /tmp/setup.txt 2>&1); echo "== setup rc=$?"
export CHIP_ARCH=blackhole LLK_HOME=$R/tt_metal/tt-llk
cd $R/tt_metal/tt-llk/tests/python_tests
MODS="test_unpack_tilize.py test_unpack_tilize_sweep.py test_tilize_calculate_untilize_L1.py test_unpack_tilize_uninit_restore.py test_unpack_tilize_uninit_restore_block.py test_unpack_tilize_uninit_restore_tiny.py test_unpack_tilize_uninit_descriptor.py test_matmul_unpack_tilize.py test_tilize_transition_reconfig.py"
NP=$(nproc); NJ=$(( NP > 16 ? 16 : NP ))
for v in $1; do
  place $v; rm -rf $HOME/tf_$v
  RUNNER_TEMP=$HOME/tf_$v TMPDIR=$HOME/tf_$v timeout -s INT -k 60 3000 python -m pytest -p no:cacheprovider -q --compile-producer -n $NJ $MODS > /tmp/fb_$v.txt 2>&1
  echo "== func build $v rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/fb_$v.txt | tail -1)"
  RUNNER_TEMP=$HOME/tf_$v TMPDIR=$HOME/tf_$v timeout -s INT -k 60 3000 python -m pytest -p no:cacheprovider -q -rfE -o timeout_method=thread --timeout 300 --compile-consumer $MODS > /tmp/fr_$v.txt 2>&1
  echo "== func run $v rc=$? $(date -u +%T) $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/fr_$v.txt | tail -1)"
  grep -a -E "^(FAILED|ERROR)" /tmp/fr_$v.txt | cut -c1-220 | head -40
done
echo "== DONE $(date -u +%T)"
