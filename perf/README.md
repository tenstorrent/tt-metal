# Quasar spec-as-key — host-dispatch perf harness

Measures **host-side dispatch cost** (µs/op) of the Metal 2.0 spec-as-key migration and each
optimization PR, isolated, on 1-chip and 32-chip, reproducibly.

## Method (why it's trustworthy)

We measure under **graph-capture `NO_DISPATCH`** (`ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)`):
the full host dispatch path runs (create_program_spec build+hash, run-args build, validation,
output-tensor allocation) but **nothing is pushed to the device**, so there is no command-queue
backpressure. This is what lets us measure cleanly on the **simulator** (its ~0.5 KHz device would
otherwise backpressure the queue and turn amortized timing into garbage), and on any chip count.

- Timing: `elapsed / N` over N async enqueues inside the capture, min over reps (`legacy_vs_spec.py`).
- The capture adds a **constant ~50–70µs pedestal** to every absolute number — so the **absolute µs
  are NOT real-dispatch absolutes**. Only **before/after DELTAS are meaningful**: the pedestal is in
  both arms and cancels. Report the µs delta, not a % of the inflated total.
- Repeatability (same build, 1-chip, 3 runs): **±1µs** → 1-chip deltas > ~2µs are real signal.
  32-chip run-noise is ~±7–12µs (only larger deltas resolve there).
- `NO_DISPATCH` cannot measure the **per-device command write** (it's skipped) → it does NOT capture
  the galaxy-parallel-write PR (#48314); that one needs a real multi-chip dispatch run.

## Reproduce

```bash
# build (full coherent build; never partial)
./build_metal.sh

# 1-chip
export TT_METAL_SIMULATOR=<path>/libttsim.so TT_METAL_INSPECTOR=0
export TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
python3 perf/legacy_vs_spec.py single <tag>

# 32-chip galaxy
export TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/umd/tests/cluster_descriptor_examples/6u_cluster_desc.yaml
python3 perf/legacy_vs_spec.py mesh <tag>
```
Output lines: `RESULT <tag> <op> legacy|spec <us>`. `legacy` = `ttnn.<op>`, `spec` =
`ttnn.experimental.quasar.<op>`. For an optimization's contribution, run the bench on the branch tip
(opt ON) and after reverting that opt's commit (opt OFF); the delta on the `spec` column is its effect.

## Optimization commits (revert to get the "without" baseline)

| optimization (PR) | commit | isolatable? |
|---|---|---|
| skip run-arg validation (#48138) | f34b04cf8cf | yes |
| memoize BufferDistributionSpec (#48252) | 236da12b53d | yes |
| CoreRangeSet solid-rect fast path | db55418a0e3 | yes |
| inline RtaName key (#48071) | c2418a173e6 | yes |
| small-vector run-arg Tables (#48060) | 14180506908 | yes |
| run-args builder (#48250) | 76cf114cb5d | foundational (migrations call it) |
| galaxy parallel write (#48314) | 4d88eb5d8a3 | not NO_DISPATCH-measurable (real multi-chip only) |

Base = all quasar ops migrated to spec + galaxy-parallel, with the isolatable opts above toggled off.
