# HiFi2 / FP32 destination: matched resident-input comparison

September 14, 2026. Neither of the two existing FP32 streaming configurations
tested beats the retained compensated BF16 kernel. The cheaper FP32 path reaches
0.955 TFLOP/s/core versus BF16's 1.591, taking 66.7% longer. This is a measurement
of existing schedules, not a proof that a different FP32 design cannot win.

## Contract and results

Blackhole P100A, yyzo-bh-08, new reservation 219611, logical core (0,0).
Q256/K512/D128, noncausal, original BF16 Q/K/V, no preprocessing. Resident K/V
repeat 512 times; 16 Q repetitions amortize setup. Same resident harness and
frozen v3 compute sources as the previous BF16 comparison. No recurring input
DRAM/NoC transfers or L1 copies; internal score/state traffic remains.
Timing is uninstrumented blocking trace replay: 20 warmups, 10 measurements,
fresh process per case. Useful FLOPs = 4*256*512*128*16*512 = 549,755,813,888.

| Version | Median ms | TFLOP/s/core | HiFi2-peak utilization | Resident normal L2 % | PCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main BF16 streaming, HiFi2 | 275.713 | 1.99394 | 72.12% | 20.4106 | 0.97958956 |
| Retained compensated BF16 streaming, HiFi2 | 345.443 | 1.59145 | 57.56% | 2.44567 | 0.99970087 |
| Cheaper FP32 streaming, HiFi2 | 575.756 | 0.95484 | 34.54% | 0.62010 | 0.99998454 |
| ACCURATE softmax pipeline, matmuls changed to HiFi2 | 597.022 | 0.92083 | 33.31% | 0.90792 | 0.99998652 |
| Retained ACCURATE, HiFi4 / FP32 | 723.556 | 0.75980 | 27.48% | 0.17762 | 0.99999842 |

Common denominator is the nominal 2.7648 TFLOP/s/core HiFi2 peak at 1.35 GHz.
ACCURATE reaches 54.96% of its own HiFi4 peak, not 27.48% of that peak.
These are measured per-core rates, not measured chip throughput.

The error columns are **resident-input diagnostics**, against dense FP64 attention
over the original BF16 inputs. Repeating an identical K/V block leaves the exact
normalized reference unchanged. They are not the full random-256K results
(e.g. 18.65% / 3.16% for main / compensated BF16) or a new qualification suite.

BF16 retains two Q slots and double-buffered K/V; FP32 retains two Q slots and
one K/V slot to fit its wider score/state CBs at Q256. CB allocation is
1,169,408 / 1,333,248 / 1,335,296 bytes for main / compensated / FP32.
Neither reader does recurring input transfer in the timed loop. No chunks or
existing buffering configurations were changed. Both improved production guards
exclude Q256; the wrapper instantiates the compute path explicitly, with no
fallback. This does not expand production support.

## Which FP32 versions?

`fp32_hifi2`: exactly the retained ACCURATE resident configuration except QK/PV
fidelity is HiFi2. It keeps full-FP32 L1 subtraction, the unbiased cubic exp
refiner/load macro, full FP32 online state, and the exact two-phase (0+2)
denominator. Its higher L2 than the cheaper version is not paradoxical: the
denominator sees full P precision while HiFi2 PV sees truncated effective
weights. This mixed-fidelity variant is not qualified ACCURATE mode.

`fp32_hifi2_cheap`: existing earlier FP32 streaming branch, with
`SDPA_FP32_STREAMING`, `SDPA_FP32_STATE`, and `SDPA_HIFI2_ROUND`. It uses the
biased cubic approximate-exp refiner, ordinary FPU score-minus-max subtraction,
and a LoFi denominator consuming the same effective SrcB weights as HiFi2 PV.
It omits the full-FP32 score-subtraction pass. Unlike historical perf-v4 accuracy
numbers, this test does **not** preprocess Q. It is not a test of completely
unmodified main FP32, nor of stock exp with all refinement removed.

Both use FP32 destination, FP32 score/online-state storage, direct-to-DST FP32
state unpack, 1x4 matmul subblocks, and half-DST synchronization. BF16 uses 2x4
subblocks. The wrapper's compile configuration explicitly selects HiFi2 for
both new cases; denominator overrides restore the configured matmul fidelity.

## Repeats and correctness checks

Reverse-order medians: compensated BF16 345.470 ms; higher-precision-softmax
HiFi2/FP32 597.022 ms; cheaper HiFi2/FP32 575.750 ms. Every reverse run has the
same full output hash and numerical metrics as its forward run. All measured
runs check finiteness and exact eager/trace output equality.

| Additional resident diagnostic | Compensated BF16 L2 % | HiFi2/FP32 cheap L2 % | HiFi2/FP32 accurate-softmax L2 % |
| --- | ---: | ---: | ---: |
| Held-out seed1237, 64 K chunks | 2.62952 | 0.67205 | 0.97542 |
| Constant V=1, seed1236, 512 K chunks | 0.69053 | approximately zero | 0.39518 |

Constant-V PCC is not meaningful. These tests are diagnostics, not claims that
either new HiFi2 variant meets a 0.5% qualification gate.

## Interpretation

HiFi2 is insufficient to make the current FP32 schedule competitive with
compensated BF16: the cheaper branch needs about a 40% time reduction just to
match it. Changing only HiFi4 to HiFi2 reduces time 17.5%, not 50%. FP32 has half
the tile capacity per DST half, smaller matmul subblocks, wider internal traffic,
and separate score/state processing and synchronization. This comparison does
not isolate each cost; device counters provide an additional scheduling check.

Separate 8-repetition device profiles confirm 1350 MHz and agree with trace
throughput within 0.05%. Counter intervals were checked against kernel timestamps;
all fit below 2^32 cycles. Raw CSVs and `profile-summary.jsonl` are retained.

| FP32 variant | Device TFLOP/s/core | FPU active | SFPU active | Both active | Neither active |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cheaper HiFi2 | 0.95476 | 38.46% | 39.78% | 16.54% | 38.29% |
| Accurate-softmax HiFi2 | 0.92123 | 38.22% | 31.07% | 16.40% | 47.10% |

These are activity counters, not useful-FLOP utilization. The substantial
neither-unit-active fraction shows this is not purely a matmul-fidelity or exp
arithmetic limit. It does not by itself distinguish unpack/pack, synchronization,
configuration, and internal L1 stalls. Those remain profiling targets for a
future FP32 scheduling redesign, not evidence that such a redesign will win.

The grouped-compensation proposal remains unimplemented and out of this task.
No production C++ or dispatch changes were made. Only a Python harness and
measurement scripts were added; kernels compile/JIT and execute through the
existing build. No host rebuild is required for this Python-only change.

## Reproduce

In the configured remote checkout `/localdev/cglagovich/tt-metal-blackhole-20260908`:

```bash
python_env/bin/python experiments/sdpa-l2/hifi2-fp32-resident-v1/run.py \
  --mode fp32_hifi2_cheap --q-repeats 16 --k-chunks 512 \
  --warmup 20 --iters 10 --label fresh-cheap
# Repeat with --mode fp32_hifi2, fast, accurate and fresh labels.
```

`measure.sh` records the full comparison and holdouts; `profile.sh` records
device counters separately. Use fresh labels to avoid overwriting results.
Each result has CB specs, defines, arguments and source hashes in its provenance
file. Main's raw JSON/provenance reside in `../single-core-resident-v1/` under
`hifi2-fp32-main-control`; its log is here. Frozen measured runner text is
`run-measured.py.txt`; production and frozen compute sources remain unchanged.
