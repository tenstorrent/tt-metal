# KDA optimized cache adapter rebase results

Date: 2026-09-14
Branch: `momcilo/kda/nd-cache-adapter`
Latest-main base: `5dfe136d485aa55a28101acec4faf89830197e0a`
Bead: `tt-metal_tracker-2gb`

## Verdict

The optimized adapter series is correctly rebased onto latest main, builds with
the repository build script, and preserves Kimi-K3 accuracy and cache contents
on all three eight-device Blackhole layouts. The optimized import-plus-export
round trip adds 29.56-56.46 microseconds, or 0.31-0.57% of the corresponding
current-main KDA layer latency.

This remains materially better than the simple adapter: round-trip latency is
reduced by 63.2% at SP1, 74.6% at SP2, and 79.1% at SP4. The optimized adapter
also remains preferable to the existing direct-ND prototype at distributed
layouts: the latter regressed by 238 microseconds at SP2 and 457 microseconds at
SP4 against its main baseline because of its internal staging copy and recurrent
broadcast.

## Current-main comparison

Times are synchronized trace-wall medians on eight Blackhole devices with
firmware 19.5.0, `FABRIC_1D`, real Kimi-K3 layer-1 weights, batch 1, and sequence
5120. Main layer latency is the median of five samples with ten trace replays per
sample. Adapter latency is the median of 20 samples with 100 trace replays per
sample. `Main + optimized` is the measured main latency plus the independently
measured import and export medians.

| Layout | Current main (ms) | Export (us) | Import (us) | Round trip (us) | Main + optimized (ms) | Delta vs main |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | 9.627586 | 14.846 | 14.715 | 29.561 | 9.657146 | +0.307% |
| SP2xTP4 | 9.557060 | 18.552 | 18.096 | 36.648 | 9.593708 | +0.383% |
| SP4xTP2 | 9.980214 | 28.517 | 27.943 | 56.460 | 10.036674 | +0.566% |

The adapter branch's layer-only medians were 9.628063, 9.565427, and 9.983139
ms. Their differences from exact main are +0.477, +8.366, and +2.925
microseconds respectively, all below 0.09%; no material forward-path regression
is visible.

## Comparison with the simple adapter

The simple-adapter values are the existing synchronized measurements preserved
in `specs/kda-cache-adapter-ablation-results.md`. The current optimized values
use the same tensor contract and measurement structure after the rebase.

| Layout | Simple export (us) | Simple import (us) | Simple round trip (us) | Optimized round trip (us) | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | 35.40 | 44.92 | 80.325 | 29.561 | 63.2% |
| SP2xTP4 | 61.72 | 82.58 | 144.301 | 36.648 | 74.6% |
| SP4xTP2 | 114.82 | 154.89 | 269.717 | 56.460 | 79.1% |

The performance benefit comes from the generic aligned-page row-major copy
path. It distributes the small number of logical convolution rows into aligned
page units across the worker grid; KDA kernels and KDA production APIs remain
unchanged.

## Accuracy and identity

| Layout | Output PCC | Minimum recurrent PCC | Minimum convolution PCC | Adapter round trip |
| --- | ---: | ---: | ---: | --- |
| SP1xTP8 | 0.999893 | 0.999795 | 0.999999 | real and patterned bit-identical |
| SP2xTP4 | 0.999894 | 0.999920 | 0.999999 | real and patterned bit-identical |
| SP4xTP2 | 0.999896 | 0.999922 | 0.999999 | real and patterned bit-identical |

The exact-main run also validated the first trace replay against the independent
pure-Torch FP32 reference and produced the same PCC values.

## Validation

Submodules were initialized and updated recursively. The feature branch build
used the requested repository entry point:

```bash
./build_metal.sh --build-type Release --enable-ccache
```

Result: pass, including host compilation, firmware precompile, and install.

Host geometry contract:

```bash
scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/test_cache_adapter_geometry.py -q
```

Result: 6 passed.

Optimized adapter matrix after the authoritative build:

```bash
TT_METAL_HOME=$PWD LD_LIBRARY_PATH=$PWD/build_Release/lib \
KIMI_K3_CKPT=/localdev/mvasilijevic/.cache/Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
KDA_ADAPTER_TIMING_SAMPLES=20 KDA_ADAPTER_TIMING_REPS=100 PERF_REPS=10 \
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_cache_adapter_perf.py \
  --tt-arch blackhole -q -s
```

Result: 3 passed in 19.51 seconds.

Exact-main baseline was built and run in a detached temporary checkout at
`5dfe136d485`:

```bash
PYTHONPATH=$PWD:$PWD/ttnn:$PWD/tools TT_METAL_HOME=$PWD \
LD_LIBRARY_PATH=$PWD/build_Release/lib:$PWD/build_Release/tt_metal:$PWD/build_Release/tt_metal/third_party/umd/lib \
KIMI_K3_CKPT=/localdev/mvasilijevic/.cache/Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
KDA_PERF_SKU=bh_loudbox scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py::test_kimi_k3_layer_1_perf \
  --tt-arch blackhole -q -s
```

Result: 3 passed in 86.10 seconds.

Local raw evidence is retained outside the worktree at
`/tmp/kda-nd-cache-adapter-evidence/` and is not committed.
