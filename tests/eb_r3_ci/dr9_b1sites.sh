#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 fourth review): which call sites the PR's ReduceToOneB1 change (the add's direct LLK calls with
# SrcDvalid::PerTile, reduce_to_one_kernel.cpp's define gone) changes in decoder_block_kernel.cpp, moe_kernel.cpp (dense MLP and
# routed MoE with reduce) and reduce_to_one_kernel.cpp: the tests built with the PR head ec7714f90f8's reduce_to_one_b1.hpp and
# reduce_to_one_kernel.cpp (tests/eb_r3_ci/head) and with the PR's, on a Blackhole Galaxy (slow dispatch, one JIT cache per side,
# the same /work paths), outputs hashed on both sides; every differing build attributed to call sites with the PR's header lines
# mapped to the head's (elfsite/r2o_sites.py); a NOP on the head's ReduceToOneB1 init line is the positive control.
cd /work
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
B1=models/demos/deepseek_v3_b1/tests/unit_tests
S=tests/eb_r3_ci/elfsite
H=models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp
KR=models/demos/deepseek_v3_b1/micro_ops/reduce_to_one_b1/kernels/reduce_to_one_kernel.cpp
KD="rigged_groups1 and t8_seven_picked and sram_bspm_off and random_weights and just_decoder_mla and just_decoder_moe and not mtp and device_params0-0-32768"
KM="(test_mlp_with_reduce and not half and not 7sram and not all-dram) or (test_moe_fused_with_reduce and full_groups and t8_partial)"
SEL="($KD) or ($KM) or test_reduce_to_one_1d"
O=/tmp/eb9; rm -rf $O; mkdir -p $O; cp $H $O/h.pr; cp $KR $O/k.pr
side() {
  local v=$1
  export TT_METAL_CACHE=$O/cache_$v EB_HASH_OUT=$O/hash_$v.json; mkdir -p $TT_METAL_CACHE
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-3600} python3 -m pytest -p eb_bits_plugin -p eb_seed_plugin -p no:cacheprovider -o timeout_method=thread --timeout=0 -q -rfEs \
    $B1/test_decoder_block.py $B1/test_moe_mlp.py $B1/test_reduce_to_one_b1.py -k "$SEL" > $O/log_$v.txt 2>&1
  echo "--- $v rc=$?: $(grep -E 'passed|failed|error' $O/log_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR|SKIPPED)" $O/log_$v.txt | cut -c1-260 | head -12
}
echo "##### head side"; cp tests/eb_r3_ci/head/reduce_to_one_b1.hpp $H; cp tests/eb_r3_ci/head/reduce_to_one_kernel.cpp $KR; side head
echo "##### pr side"; cp $O/h.pr $H; cp $O/k.pr $KR; side pr
echo "##### control side: the head plus a NOP on ReduceToOneB1's init line"
cp tests/eb_r3_ci/head/reduce_to_one_b1.hpp $H; cp tests/eb_r3_ci/head/reduce_to_one_kernel.cpp $KR
python3 - "$H" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::received_cb);\n"
assert s.count(a) == 1
open(p, "w").write(s.replace(a, "            MATH(TTI_NOP); " + a.lstrip()))
PY
side ctl
cp $O/h.pr $H; cp $O/k.pr $KR
echo "##### bits head against pr"
python3 - <<'PY'
import json
a = json.load(open("/tmp/eb9/hash_head.json")); b = json.load(open("/tmp/eb9/hash_pr.json"))
for t in sorted(set(a["outcome"]) | set(b["outcome"])):
    ha, hb = a["hashes"].get(t), b["hashes"].get(t)
    print(f"{t}: outcome {a['outcome'].get(t)} | {b['outcome'].get(t)}; outputs {len(ha or [])} | {len(hb or [])}; bits {'identical' if ha == hb else 'DIFFER'}")
PY
for k in decoder_block_kernel moe_kernel reduce_to_one_kernel; do
  echo "##### keys $k (head | pr)"; python3 $S/elfcmp.py $O/cache_head $O/cache_pr --kernel "^$k$"
  echo "##### sites $k (head | pr)"; python3 $S/r2o_sites.py $O/cache_head $O/cache_pr "^$k$" --a=head --b=pr
  echo "##### keys $k (head | control)"; python3 $S/elfcmp.py $O/cache_head $O/cache_ctl --kernel "^$k$"
  echo "##### sites $k (head | control)"; python3 $S/r2o_sites.py $O/cache_head $O/cache_ctl "^$k$" --a=head --b=head
done
echo "##### done"
