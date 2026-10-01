# FSDP benchmark & profiling kit

Tools to answer "where does an FSDP training step spend its time, and why?" for
`ttml.fsdp.fully_shard`, and to check the collective/compute overlap schedule before changing it.
Everything runs from this directory; results land in `results/` (git-ignored on purpose).

All scripts expect the Blackhole galaxy MGDs under `tt-train/configs/mgd/` and the pre-tokenized
Shakespeare datasets under `tt-train/data/`. The kernel JIT compiler forks subprocesses, so runs
must happen outside any sandbox that blocks `posix_spawn`, and 32-chip runs need `ulimit -u 65536`
(the runners set it).

| Script | What it does |
|---|---|
| `make_config.py [--model tinyllama\|llama8b] [--mode fsdp\|ddp] [--mesh N] [--tp T] [--samples S] [--ga G] [--memeff] [--overlap [columns=1\|rows=1]] [--keep-gathered-gib X]` | Prints a training config for one benchmark point. Two points differ only in what the flags say, so A/B pairs are honest by construction. |
| `run_one.sh <cfg> <mgd> <name> [train.py args]` | End-to-end run; `TTML_NAIVE_PROFILER=1` (default) adds per-phase wall time with a device sync at each boundary (~57 ms/step of overhead, so compare step times with `TTML_NAIVE_PROFILER=0`). |
| `run_guarded.sh <name> <idle_s> <hard_s> -- "<cmd>"` | Runs a device command under an inactivity watchdog and a hard timeout; a hung configuration is recorded in `results/guard/HUNG_RUNS.txt` so it is never re-queued blindly. Exit 125 = hung, 124 = timeout. |
| `summarize_logs.py` | One table (and `results/summary.csv`) over every `results/*.log`: step time, TPS, MFU, per-phase ms. |
| `run_profile.sh <cfg> <mgd> <name>` | Tracy device profiler run; copies `ops_perf_results_*.csv` to `results/<name>_ops.csv`. |
| `analyze_ops_csv.py results/<name>_ops.csv` | Per-step, per-device kernel time by phase and op class, top op codes, op-to-op gaps, per-shape CCL kernel times. |
| `ccl_microbench.py --mesh N --model M --out f.json [--ccl-subdevice rows=1]` | Timing of the exact collectives FSDP issues for every managed weight shape of a model: steady-state device time, single-call latency, host dispatch cost, effective GB/s. With `--ccl-subdevice` they run on the CCL sub-device from the second queue. |
| `overlap_race_harness.py --deps schedule\|slot\|drain\|none` | The overlap schedule reduced to its moving parts (persistent slots, prefetch one block ahead, in-place shard update), every block checked bitwise against a synchronous reference, with the dependency policy pluggable. `schedule` is the shipped design (`ttml.fsdp.SlotSchedule`) and must report 0 mismatches; `none` is the negative control (corrupts every step); `slot` (zero cross-device slack) corrupted on the exploration branch's one-column sub-device and is not proven safe. A few seconds for 20 steps. |

Typical session:

```bash
B=tt-train/tools/profiling/fsdp_bench; M=tt-train/configs/mgd
python $B/make_config.py --mesh 8 --samples 1 > $B/results/fsdp8.yaml
python $B/make_config.py --mesh 8 --samples 1 --overlap > $B/results/fsdp8_ovl.yaml
TTML_NAIVE_PROFILER=0 $B/run_one.sh $PWD/$B/results/fsdp8.yaml $M/bh_galaxy_8_1_ring_ring.textproto fsdp8
TTML_NAIVE_PROFILER=0 $B/run_one.sh $PWD/$B/results/fsdp8_ovl.yaml $M/bh_galaxy_8_1_ring_ring.textproto fsdp8_ovl
python $B/summarize_logs.py
grep '^Step:' $B/results/fsdp8.log | cut -d, -f2 > /tmp/a; grep '^Step:' $B/results/fsdp8_ovl.log | cut -d, -f2 | diff /tmp/a -   # losses must be identical
TT_MESH_GRAPH_DESC_PATH=$M/bh_galaxy_8_1_ring_ring.textproto python $B/overlap_race_harness.py --deps schedule --steps 20
```

`FSDP_PERF_REPORT.md` is the study that motivated the FSDP changes on this branch (throughput
metric, tile-aligned shard dim, keep-gathered blocks, collective overlap). It was written on the
exploration branch and refers to that branch's knob and file names; the current design is in
`docs/FSDP.md`.
