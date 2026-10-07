#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 third review): which call sites ELTWISE_BINARY_PER_TILE_HANDOFF_DEST_REUSE changes in
# decoder_block_kernel.cpp and moe_kernel.cpp: the decoder block (position 0 of the light rigged case) and the dense MLP with reduce
# built with and without the define on a Blackhole Galaxy (slow dispatch, one JIT cache per side, the same /work paths), then every
# differing build's ELF attributed to call sites (objdump line tables) and the Tensix instructions outside the dest-reuse sites
# compared; a NOP planted before ReduceToOneB1's dest-reuse init is the positive control.
cd /work
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
B1=models/demos/deepseek_v3_b1/tests/unit_tests
OPT=tests/eb_r3_ci/optin_b1dr.txt
S=tests/eb_r3_ci/elfsite
KD="rigged_groups1 and t8_seven_picked and sram_bspm_off and random_weights and just_decoder_mla and just_decoder_moe and not mtp and device_params0-0-32768"
KM="test_mlp_with_reduce and not half and not 7sram and not all-dram"
echo "##### build decoder and mlp, both sides"; EB_RUN_LIMIT=3000 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin --timeout=0 $B1/test_decoder_block.py $B1/test_moe_mlp.py -k "($KD) or ($KM)"
rm -rf /tmp/ebsites; mkdir -p /tmp/ebsites; cp -a /tmp/ebbits/cache_main /tmp/ebsites/off; cp -a /tmp/ebbits/cache_optin /tmp/ebsites/on
for k in decoder_block_kernel moe_kernel; do
  echo "##### keys $k"; python3 $S/elfcmp.py /tmp/ebsites/off /tmp/ebsites/on --kernel "^$k$"
  echo "##### sites $k"; python3 $S/elfsites_all.py /tmp/ebsites/off /tmp/ebsites/on "^$k$" --shift=1
  echo "##### outside the dest-reuse sites $k"; python3 $S/nondr_check.py /tmp/ebsites/off /tmp/ebsites/on "^$k$" --shift=1 | tail -12
done
echo "##### control: a NOP before ReduceToOneB1's dest-reuse init"
H=models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp; cp $H /tmp/ebsites/r21.orig
python3 - "$H" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::received_cb);\n"
assert s.count(a) == 1
open(p, "w").write(s.replace(a, "            MATH(TTI_NOP); " + a.lstrip()))
PY
export TT_METAL_CACHE=/tmp/ebsites/ctl; mkdir -p $TT_METAL_CACHE
timeout -s INT -k 60 3000 python3 -m pytest -p no:cacheprovider -o timeout_method=thread --timeout=0 -q $B1/test_decoder_block.py $B1/test_moe_mlp.py -k "($KD) or ($KM)" > /tmp/ebsites/ctl.log 2>&1
echo "--- control rc=$?: $(grep -E 'passed|failed' /tmp/ebsites/ctl.log | tail -1)"
cp /tmp/ebsites/r21.orig $H
for k in decoder_block_kernel moe_kernel; do
  echo "##### control keys $k"; python3 $S/elfcmp.py /tmp/ebsites/off /tmp/ebsites/ctl --kernel "^$k$"
  echo "##### control sites $k"; python3 $S/elfsites_all.py /tmp/ebsites/off /tmp/ebsites/ctl "^$k$"
done
