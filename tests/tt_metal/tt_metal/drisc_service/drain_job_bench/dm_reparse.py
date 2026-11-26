# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Usage: python dm_reparse.py <outdir>, from the tt-metal root with TT_METAL_HOME set.

Re-parses each dm_under_load.sh run's saved device CSV with the data movement harness's own collector, without its
bounds check: the harness's PerformanceChecker stops at the first test below its bounds, which under load drops every
test after it. Writes <label>.bw with, per test id and RISC, the lowest and highest bandwidth across the test's runs
(the lowest is the harness's own figure) and the number of runs."""
import glob, os, sys, shutil

sys.path.insert(0, os.environ["TT_METAL_HOME"])
from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG
from tests.tt_metal.tt_metal.data_movement.python.config import DataMovementConfig
from tests.tt_metal.tt_metal.data_movement.python.test_metadata import TestMetadataLoader
from tests.tt_metal.tt_metal.data_movement.python.stats_collector import StatsCollector

out_dir = sys.argv[1]
log = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
config = DataMovementConfig()
id2name, _, bounds, attrs = TestMetadataLoader(config).get_test_mappings()
arch = config.get_arch("blackhole", bounds)
for f in sorted(glob.glob(f"{out_dir}/r*_*.device.csv")):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    shutil.copy(f, log)
    collector = StatsCollector(log, id2name, attrs, verbose=False, arch=arch)
    dm_stats, _ = collector.gather_analysis_stats(collector.gather_stats_from_csv())
    per_test = {}
    for riscv in dm_stats:
        for run in dm_stats[riscv]["analysis"]["series"]:
            a = dm_stats[riscv]["attributes"][run["duration_type"][0]["run_host_id"]]
            bw = a["Number of transactions"] * a["Transaction size in bytes"] / run["duration_cycles"]
            per_test.setdefault((a["Test id"], riscv), []).append(bw)
    label = os.path.basename(f)[: -len(".device.csv")]
    with open(f"{out_dir}/{label}.bw", "w") as w:
        for (tid, riscv), v in sorted(per_test.items()):
            w.write(f"{tid} ({id2name.get(tid, '?')}) {riscv} min {min(v):.4f} max {max(v):.4f} n {len(v)}\n")
    print(label, len(per_test))
