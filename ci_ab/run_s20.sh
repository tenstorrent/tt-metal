#!/usr/bin/env bash
# Scratch job (#60676 replay scan), not for merge. The job's tree is #60676's head d471087aa84 merged with main 6a3f10a4242
# (a merge commit; its tree equals main plus the four gate files). Builds every kernel of the gate and hash gate model
# configurations (gate_ops.py, 12 cases) on the Blackhole p100a mock, then scans the unpack thread ELFs (trisc0) of the cache
# with replay_scan.py (a replay that plays a different program than the nearest record over its slots, or a replay in a loop
# whose body records another program over its slots). Guard: the scan's selftest passes, all 12 cases built, and the gate
# kernels' ELFs contain replays (else the scan read nothing).
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work
S=/work/ci_ab; OUT=/work/generated/s20; rm -rf $OUT; mkdir -p $OUT
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=/work:/work/tools
echo "== $(date -u +%FT%TZ) $(git log -1 --format='%h %s') $(git log -1 --format='parents %p') $(hostname)"
OBJD=/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump; [[ -x $OBJD ]] || OBJD=$(which riscv-tt-elf-objdump 2>/dev/null)
[[ -x $OBJD ]] || OBJD=$(find /work/runtime /opt -name riscv-tt-elf-objdump -type f 2>/dev/null | head -1)
echo "== objdump $OBJD"; export REPLAY_OBJDUMP=$OBJD
fail=0
python3 $S/replay_scan.py --selftest | tail -4
export TT_METAL_CACHE=$OUT/cache_mock; mkdir -p $TT_METAL_CACHE
GATE_MOCK=1 TT_METAL_MOCK_CLUSTER_DESC_PATH=$S/p100a_cluster.yaml timeout -s INT -k 60 1500 python3 $S/gate_ops.py > $OUT/mock.log 2>&1
ok=$(grep -c ' ok$' $OUT/mock.log); bad=$(grep -c FAILED $OUT/mock.log)
echo "== mock build rc=$? $ok ok, $bad failed"
grep -a -m5 FAILED $OUT/mock.log | cut -c1-300
python3 $S/elfhash.py bh_N $TT_METAL_CACHE moe_grouped_topk moe_hash_gate | grep -a "^ELFCOUNT"
n=$(ls $TT_METAL_CACHE/*/kernels/moe_grouped_topk/*/trisc0/trisc0.elf $TT_METAL_CACHE/*/kernels/moe_hash_gate/*/trisc0/trisc0.elf 2>/dev/null | wc -l)
echo "RAB SCAN gate kernels with a trisc0 ELF: $n"
python3 $S/replay_scan.py $TT_METAL_CACHE/*/kernels/moe_grouped_topk $TT_METAL_CACHE/*/kernels/moe_hash_gate 2>&1 | tee $OUT/scan_gate.txt | cut -c1-400
echo "RAB SCAN all kernels of the cache:"
python3 $S/replay_scan.py $TT_METAL_CACHE 2>&1 | tee $OUT/scan_all.txt | tail -3 | cut -c1-400
w=$(grep -a "^# " $OUT/scan_gate.txt | tail -1 | awk '{print $4}')
g1=$([[ $ok -eq 12 && $bad -eq 0 ]] && echo ok || echo VOID); g2=$([[ ${w:-0} -gt 0 ]] && echo ok || echo NOREPLAY)
echo "RAB GUARD built=$g1 replays_read=$g2"; [[ $g1 == ok ]] || fail=1
echo "== done $(date -u +%FT%TZ) fail=$fail"
exit $fail
