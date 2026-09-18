# KDA direct-ND bridge removal results

Date: 2026-09-14

Branch: `momcilo/kda-nd-cache`

Latest-main base: `f98e9b40db3a5fd2cbd9cf81619ae534e49eb264`

Parent bead: `tt-metal_tracker-ef4`

## Verdict

The direct-ND KDA path now reads and replaces its canonical ND-sharded DRAM
caches without either of the layout bridges that caused the original SP2/SP4
regression. It passes the real Kimi-K3 accuracy and performance matrix and is
within 20 microseconds of same-revision main at every layout.

The implementation is measurably faster than wrapping main with either cache
adapter. Against the optimized adapter, direct ND saves 23-37 microseconds per
layer invocation (0.24-0.37% of layer time); against the simple adapter it saves
82-251 microseconds. The optimized-adapter comparison is a projection because
its isolated round-trip measurements were taken at `5dfe136d485`, ten main
commits before this branch's base. None of those commits changes KDA or the
generic copy path, and the adapter cost is added to a freshly measured
same-revision main baseline.

The performance case over the optimized adapter is real but modest. Direct ND
is justified when ND DRAM is the canonical cache contract and avoiding boundary
conversion is a goal. If minimum KDA-specific API and kernel complexity is the
dominant criterion, the optimized adapter remains a reasonable alternative for
roughly 0.3-0.6% total-layer cost.

## End-to-end performance

Times are synchronized trace-wall medians on eight Blackhole devices, firmware
19.5.0, KMD 2.4.1, `FABRIC_1D`, real Kimi-K3 layer-1 weights, batch 1, and
sequence 5120. Each layer value is the median of five samples with ten trace
replays per sample.

| Layout | Same-base main (ms) | Bridge-free direct ND (ms) | Direct delta vs main | Main + optimized adapter (ms) | Direct advantage vs optimized |
| --- | ---: | ---: | ---: | ---: | ---: |
| SP1xTP8 | 9.620299 | 9.618377 | -1.922 us (-0.02%) | 9.649860 | 31.483 us |
| SP2xTP4 | 9.563530 | 9.577202 | +13.672 us (+0.14%) | 9.600178 | 22.976 us |
| SP4xTP2 | 9.981963 | 10.001026 | +19.063 us (+0.19%) | 10.038423 | 37.397 us |

The optimized-adapter round trips are 29.561, 36.648, and 56.460 microseconds
for SP1, SP2, and SP4 respectively. They were independently measured with 20
samples and 100 replays per sample on the same host, firmware, fabric, model,
batch, and sequence. Adding those costs to the current main measurements gives
the projected totals above.

For completeness, the simple adapter's measured round trips are 80.325,
144.301, and 269.717 microseconds. Projected current-main totals and direct-ND
savings are:

| Layout | Main + simple adapter (ms) | Direct advantage vs simple |
| --- | ---: | ---: |
| SP1xTP8 | 9.700624 | 82.247 us |
| SP2xTP4 | 9.707831 | 130.629 us |
| SP4xTP2 | 10.251680 | 250.654 us |

## Where the original slowdown went

| Layout | Same-base main (ms) | Original direct prototype (ms) | After local recurrent copy (ms) | Final direct ND (ms) |
| --- | ---: | ---: | ---: | ---: |
| SP1xTP8 | 9.620299 | 9.612544 | 9.611627 | 9.618377 |
| SP2xTP4 | 9.563530 | 9.795781 | 9.626745 | 9.577202 |
| SP4xTP2 | 9.981963 | 10.443689 | 10.101353 | 10.001026 |

At SP2, removing the recurrent broadcast recovered 169.036 microseconds and
removing initial-history staging recovered another 49.543 microseconds. At SP4
the recoveries were 342.336 and 100.327 microseconds. SP1 changes are within
normal run-to-run noise because no SP bridge is required there.

### Recurrent state bridge

The distributed affine prefix already computes the same final carry on every
SP rank. The old implementation nevertheless invoked `all_broadcast` merely to
obtain the requested ND output placement. It measured 171.919 microseconds at
SP2 and 379.018 microseconds at SP4 in the original profile.

The replacement is a rank-local `ttnn.to_memory_config` at
`models/demos/deepseek_v3_d_p/tt/kda/recurrence.py:301`. The final SP2 profile
contains no broadcast; the replacement copy is 10.021 microseconds. This is
below the plan's 20-microsecond threshold, so a fused recurrent ND writer would
add kernel and output-buffer complexity without a demonstrated payoff.

### Convolution history bridge

The required projected-tail `all_gather` remains. What was removed is the
three-row canonical ND cache conversion that used to precede halo assembly.
The original conversion measured 67.339 microseconds at SP2 and 130.706
microseconds at SP4.

`exchange_convolution_carry` now returns only interleaved predecessor tails and
the final stream tail (`models/demos/deepseek_v3_d_p/tt/kda/convolution.py:10`).
The KDA layer passes the unchanged canonical ND history and that predecessor
source separately (`models/demos/deepseek_v3_d_p/tt/kda/kda.py:258`).

The QKV operation selects a mesh workload only when an SP history axis is
present (`ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_device_operation.cpp:18`). Its factory binds each
mesh coordinate to the layout-correct history tensor at program construction:
rank zero uses the ND cache and later ranks use their interleaved predecessor
tail (`ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_program_factory.cpp:265`). There is no runtime rank
tensor or inner-loop source branch. The reader's existing accessor specialization
handles interleaved versus sharded addressing
(`ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/reader_qkv_causal_conv1d_silu.cpp:66`).

The final SP2 profile has 20 generic data-movement programs instead of 21 after
only the recurrent fix, confirming that the initial-history staging program is
gone. QKV itself measures 1.169 ms, consistent with the 1.167 ms Phase-1 value;
native ND reading did not move the kernel bottleneck.

## Accuracy

The real-weight performance test checks eager output and both replacement
caches against an independent pure-Torch FP32 reference, then repeats the same
check on the first trace replay.

| Layout | Output PCC | Minimum recurrent PCC | Minimum convolution PCC |
| --- | ---: | ---: | ---: |
| SP1xTP8 | 0.999893 | 0.999795 | 0.999999 |
| SP2xTP4 | 0.999894 | 0.999920 | 0.999999 |
| SP4xTP2 | 0.999896 | 0.999922 | 0.999999 |

The full layer suite's production-width synthetic SP2xTP4 case additionally
reports output PCC 0.999952, recurrent PCC 0.999765, convolution PCC 0.999997,
and bit-identical results across three executions. Tests compare exact ND cache
shards on the host because generic `ttnn.ne` attempts an unsupported tile
conversion for the physical three-row row-major shard. This comparison is
outside all timed workloads.

## Validation

Authoritative build after all implementation commits:

```bash
./build_metal.sh --build-type Release --enable-ccache
```

Result: pass; Release host compilation, Python bindings, and install completed.

Full KDA layer suite:

```bash
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/layer \
  --tt-arch blackhole -q -s
```

Result: 23 passed, 5 skipped in 43.71 seconds. Three skips require the optional
checkpoint environment variable; two are hardware/topology exclusions. The
real checkpoint was exercised by the performance matrix below.

Final real-weight accuracy/performance matrix:

```bash
KIMI_K3_CKPT=/localdev/mvasilijevic/.cache/Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
KDA_PERF_SKU=bh_loudbox scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py::test_kimi_k3_layer_1_perf \
  --tt-arch blackhole -q -s
```

Result: 3 passed in 17.96 seconds; all accuracy and calibrated ±3% performance
gates passed. Local raw logs are retained at
`/tmp/kda-bridge-free-final-build.log`,
`/tmp/kda-bridge-free-layer-suite-final.log`, and
`/tmp/kda-bridge-free-final-perf.log`.

## Commit impact map

| Commit | Concern |
| --- | --- |
| `6400e8b0b82` | Replace redundant recurrent SP broadcast with a local ND placement copy. |
| `3fe48575099` | Add the QKV dual-history contract and per-coordinate reader selection. |
| `90a3d38acd2` | Stop staging initial convolution history; pass canonical ND history directly. |
| `4f96cba8b32` | Make determinism checks layout-agnostic for ND cache shards. |

The untracked user-supplied HTML experiment report was not modified or added.
