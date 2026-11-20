# Quasar simulator benchmark for tilize, untilize, tiled reshape and slice

`bench_tm_ops_sim.py` runs each manifest case in its own process on a Quasar simulator, reads the device
profiler after every op, checks the output bit for bit against PyTorch, and writes one TSV row per (shape, run).
It exits 2 if any row was not exact, so it doubles as a regression check for the Quasar-only kernels of these ops.

Run it from the tt-metal root with the tree built and its venv active:

    python tests/ttnn/nightly/unit_tests/operations/experimental/quasar/perf/bench_tm_ops_sim.py \
        --sim-dir ~/sim/qsr --manifest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/perf/manifest_tm_ops.json \
        --out results.tsv

`--sim-dir` holds `libttsim.so` and the SoC descriptor as `soc_descriptor.yaml` (see the ttsim README for the
Quasar setup). `manifest_tm_ops.json` carries the 14 shapes the branch was measured with, tensors in DRAM;
change `memcfg` to `L1` for L1-resident tensors.

TSV columns: `span_cycles` is the program span from the first kernel start to the last kernel end anywhere on the
grid, in simulator cycles; `cyc_per_tile_per_cluster` is span times active clusters over output tiles;
`lifetime_by_risc` is each RISC's kernel lifetime as a fraction of the span, waits included, which says which
harts finished early and nothing about how busy one was; `exact` is the bit-exact check. Run 0 carries cold
caches and compiles; report run 1. `zones_report.py` prints per-zone statistics from a profiler CSV.
