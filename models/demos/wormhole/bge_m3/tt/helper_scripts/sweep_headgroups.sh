#!/bin/bash
# In-model sweep of head_groups for the head split and the concat at B8/S512.
#
# head_groups must divide num_heads=16, so the legal values are 4, 8 and 16.
# B8 inherited 4 for the split and 16 for the concat from the B16/B32 line;
# neither was ever swept at B8. The GenericOp pair costs 1871 us, 10.2% of the
# wall.
#
# The traced wall time from the perf test is the measurement. Isolated
# microbenches were not trusted here: an earlier attempt used the wrong memory
# config, and three later items showed isolated gains that vanished in-model.
set -u
cd /local/ttuser/gtobar/tt-metal
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_MODEL=BAAI/bge-m3 TT_VISIBLE_DEVICES=0
unset TT_METAL_DEVICE_PROFILER
ATT=models/demos/wormhole/bge_m3/tt/attention.py
cp $ATT /tmp/attention_sweep_backup.py
TEST="models/demos/wormhole/bge_m3/tests/perf/perf.py::test_perf[blackhole-nomask-b8_s512-forward-single]"

printf '  %-6s %-7s %9s\n' split concat ms
for SPLIT in 4 8 16; do
  for CONCAT in 4 8 16; do
    cp /tmp/attention_sweep_backup.py $ATT
    python3 - "$ATT" "$SPLIT" "$CONCAT" <<'PYEOF'
import sys
path, split, concat = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
old_split = "head_groups = 4 if self.config.max_batch_size in (8, 16, 32) else self.config.num_heads"
new_split = "head_groups = %s if self.config.max_batch_size in (8, 16, 32) else self.config.num_heads" % split
assert s.count(old_split) == 1, "split anchor"
s = s.replace(old_split, new_split)
old_concat = "concat_head_groups = 16 if self.config.max_batch_size in (8, 16) else 4"
new_concat = "concat_head_groups = %s if self.config.max_batch_size in (8, 16) else 4" % concat
assert s.count(old_concat) == 1, "concat anchor"
s = s.replace(old_concat, new_concat)
open(path, "w").write(s)
PYEOF
    LOG=/local/ttuser/gtobar/hg_${SPLIT}_${CONCAT}.log
    timeout 400 python_env/bin/python -u -m pytest "$TEST" -q -s > $LOG 2>&1
    MS=$(grep -oE 'Best latency:\s+[0-9.]+' $LOG | grep -oE '[0-9.]+$')
    if [ -z "$MS" ]; then
      REASON=$(grep -ioE 'clash with L1|beyond max L1|expected per-shard|Out of Memory|must divide' $LOG | head -1)
      MS="FAIL ${REASON:-see $LOG}"
    fi
    printf '  %-6s %-7s %9s\n' "$SPLIT" "$CONCAT" "$MS"
  done
done
cp /tmp/attention_sweep_backup.py $ATT
echo "  attention.py restored"
