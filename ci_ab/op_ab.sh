#!/usr/bin/env bash
# Scratch op-level A/B (17_CI_AB.md), not for merge. A = the checked-out tree's device files, B and B2 = ci_ab/variants/B
# laid over them (B2 is the A/A copy of B at the same paths). Every variant has its own fresh kernel cache; rounds rotate
# the order. Device kernel time per signposted case from tracy; each run prints the hash of the trisc1 images of the
# kernels it compiled, so a void swap shows (A must differ from B, B must equal B2).
set +e +o pipefail
cd /work 2>/dev/null || cd "$(git rev-parse --show-toplevel)"; R=$PWD; S=$R/ci_ab
export HWLOCK_HELD=1 R3_NMEAS=${R3_NMEAS:-6}
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=$R:$R/tools:${PYTHONPATH:-}
PKG=$(python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
echo "== $(date -u +%FT%TZ) $(hostname) head $(git rev-parse --short HEAD) ttnn package $PKG"
mapfile -t FILES < <(cd $S/variants/B/tree && find . -type f | sed 's#^\./##')
TARGETS=()
for f in "${FILES[@]}"; do
  TARGETS+=("$R/$f")
  [[ -n $PKG ]] && while read -r t; do TARGETS+=("$t"); done < <(find "$PKG" -path "*/$f" 2>/dev/null)
  [[ -n ${TT_METAL_RUNTIME_ROOT:-} && -e $TT_METAL_RUNTIME_ROOT/$f && $TT_METAL_RUNTIME_ROOT != $R ]] && TARGETS+=("$TT_METAL_RUNTIME_ROOT/$f")
done
printf '== target %s\n' "${TARGETS[@]}"
mkdir -p /tmp/fl_orig; i=0; for t in "${TARGETS[@]}"; do cp "$t" /tmp/fl_orig/$i; i=$((i+1)); done
setv() { # setv A|B
  local i=0 t rel src
  for t in "${TARGETS[@]}"; do
    if [[ $1 == A ]]; then cp /tmp/fl_orig/$i "$t"; else
      for rel in "${FILES[@]}"; do [[ $t == */$rel ]] && { cp "$S/variants/B/tree/$rel" "$t"; break; }; done
    fi
    i=$((i+1))
  done
  echo "   set $1: $(for t in "${TARGETS[@]}"; do md5sum < "$t" | cut -c1-6; done | tr '\n' ' ')"
}
CASES=${FL_CASES:-"softsign_hs_8x8 softsign_dram_1024x1024 softsign_bf16_hs softsign_fp32_hs gelu_bw_tanh_bf16_2048x2048 gelu_bw_tanh_fp32_2048x2048 max_pool2d_indices_9x9_256x256x64 per_token_cast_bf16_2048x7168 per_token_cast_fp32_1024x7168 rand_bf16_4096x2048 rand_fp32_4096x2048 rand_bf16_1024x1024 uniform_fp32_4096x2048 bernoulli_fp32_4096x2048 rand_32x32 sampling_32x256"}
mapfile -t CSA < <(python3 -m pytest --collect-only -q models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_csa_compressor.py::test_csa_compressor_single_device 2>/dev/null | grep "::" | head -3)
printf '== csa node %s\n' "${CSA[@]}"
O=$R/generated/fl; rm -rf $O /tmp/flc_*; mkdir -p $O
ROUNDS=${ROUNDS:-4}; ORDERS=("A B B2" "B B2 A" "B2 A B" "A B2 B")
for r in $(seq $ROUNDS); do
  for v in ${ORDERS[$(( (r - 1) % 4 ))]}; do
    setv ${v:0:1}; c=/tmp/flc_$v
    t0=$(date +%s)
    TT_METAL_CACHE=$c python3 -m tracy -r -p --no-web-server -o $O/${r}_$v $S/prof_cases.py $CASES > $O/${r}_$v.log 2>&1
    echo "== round $r $v rc=$? $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $O/${r}_$v.log) ok, $(grep -c 'CASE .* FAILED' $O/${r}_$v.log) failed"
    grep "CASE .* FAILED" $O/${r}_$v.log | head -3
    if [[ ${#CSA[@]} -gt 0 ]]; then
      TT_METAL_CACHE=$c python3 -m tracy -r -p --no-web-server -o $O/csa_${r}_$v -m pytest -q -p no:cacheprovider -o timeout_method=thread "${CSA[@]}" > $O/csa_${r}_$v.log 2>&1
      echo "   csa $r $v rc=$? $(grep -aE '[0-9]+ (passed|failed)' $O/csa_${r}_$v.log | tail -1)"
    fi
    echo "   images $v: $(find $c -name trisc1.elf 2>/dev/null | sort | xargs -r md5sum | awk '{print $1}' | md5sum | cut -c1-12) ($(find $c -name trisc1.elf 2>/dev/null | wc -l) trisc1)"
  done
done
setv A
cd $O && for d in */reports/*; do mkdir -p /tmp/fl_pack/$(dirname $d); cp $d/ops_perf_results_*.csv /tmp/fl_pack/$(dirname $d)/ 2>/dev/null; done
tar -czf /tmp/fl_op.tgz -C /tmp/fl_pack . ; echo "== results $(stat -c %s /tmp/fl_op.tgz) bytes"
echo "=====RA_BEGIN====="; base64 -w 76 /tmp/fl_op.tgz; echo "=====RA_END====="
exit 0
