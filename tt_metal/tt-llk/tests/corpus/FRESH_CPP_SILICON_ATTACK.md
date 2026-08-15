# Fresh semantic C++ silicon attack

This lane asks whether compiler-visible SFPI C++ can beat an existing hand schedule. It does not edit production LLKs. `FRESH_CPP_IMPL=0` selects the production implementation and is the default; `1` selects a test-only implementation that contains no fixed LREGs, raw instructions, replay slots, SFPLOADMACRO templates, or manually interleaved schedule.

## Candidate ranking

| Rank | Operation | Why it is useful | Readiness |
|---:|---|---|---|
| 1 | Binary max/min | Production is a 3-cycle/row SFPLOADMACRO schedule; semantic operation is one `max`/`min` expression | measured below |
| 2 | Addcmul | Production manually interleaves two independent rows; semantic expression is `(scale * b) * c + a` | measured below |
| 3 | Addcdiv | Existing correctness and isolated perf; semantic body, but reciprocal expansion adds numerical and scheduling risk | ready after Addcmul |
| 4 | Lerp | Existing correctness and isolated perf | low upside: production is already simple semantic SFPI C++ |
| 5 | Scalar add/sub/mul/rsub | Existing correctness and isolated perf | low upside: production is already one-row semantic SFPI C++ |
| 6 | Reduce row max | Existing correctness and isolated perf | needs reduction/addressing audit before a fresh body |
| 7 | Softmax-K | Correctness harness exists | numerical approximation work; no isolated perf fixture |
| 8 | SDPA exp unclamped | Correctness harness exists | numerical/FP32-destination parity first; no isolated perf fixture |
| 9 | SigmoidAppx | Short typed cubic control with existing unary correctness and isolated perf fixtures | measured below |

Welford, Reduce-SDPA, binary broadcast, TTNNWhere, and MulInt32 are excluded because they were already measured. TopK is excluded from this lane until its typed multi-result/index-tracking conversion is accepted.

## Corpus scorecard

Only correctness-gated, scoped Blackhole device measurements appear here.

| Operation | Production | Fresh semantic C++ | Result |
|---|---:|---:|---:|
| Welford body | 326 | 323 | -0.92% |
| Reduce-SDPA body | 840 | 834 | -0.714% |
| Binary broadcast body | 608 | 608 | parity |
| Reciprocal body, accurate Float16_b | 467 | 459 | **-1.713%** |
| Exp body, accurate Float16_b | 579.7421875 | 989.75 | +70.72% |
| TTNN Where body | 159.25 | 312.50 | +96.23% |
| MulInt32 isolate | 283.9296875 | 562.625 | +98.16% |

The Reciprocal and Exp results and their correctness/evidence provenance are detailed in
`RECIPROCAL_SEMANTIC_SILICON_AB.md` and `EXP_SEMANTIC_SILICON_AB.md`.

## Correctness contract

The A/B variants use identical seeded stimuli and the existing operation golden. For the selected Float16_b lane, `passed_test` requires both:

* every element is within `atol=0.05, rtol=0.05` (paired NaNs accepted); and
* PCC is greater than `0.99` for nontrivial signal.

PCC alone cannot pass a variant. Integer candidates must use the existing exact (`atol=rtol=0`) path instead of this float gate.

## Validation

Using SFPI `858786c` and SFPI-GCC `8f943c2f8`:

| Gate | Binary max/min | Addcmul |
|---|---:|---:|
| Wormhole correctness-source compile | 4/4 | 2/2 |
| Blackhole correctness-source compile | 4/4 | 2/2 |
| Wormhole isolated-perf compile | 4/4 | 2/2 |
| Blackhole isolated-perf compile | 4/4 | 2/2 |
| Blackhole CRAQ functional | 4/4 | 2/2 |

## Blackhole silicon: binary max/min

Each cell is three fresh, serialized device processes. The scoped metric is `TILE_LOOP mean(MATH_ISOLATE)` cycles/tile, not whole-kernel throughput.

| Operation | Production handwritten | Fresh semantic C++ | Delta |
|---|---:|---:|---:|
| Binary max | 140.9296875 / 140.9296875 / 140.9296875 | 198.7578125 / 198.7578125 / 198.7578125 | +41.03% |
| Binary min | 140.9296875 / 140.9296875 / 140.9296875 | 198.7578125 / 198.7578125 / 198.7578125 | +41.03% |

Evidence is archived under `/localdev/nkapre/fresh-cpp-binary-maxmin-bh-silicon-20260815` with per-process logs, raw/post CSVs, and representative disassemblies.

This is a timeboxed negative, but it identifies a general compiler gap. The production body issues configured SFPLOADMACRO words for load + min/max swap + store at three cycles/row. The fresh body does form a four-instruction `ttreplay` capture, but its capture is still an ordinary load/load/`sfpswap`/store sequence followed by `ttincrwc`; it does not form the configured SFPLOADMACRO pipeline. The durable target is therefore general SFPLOADMACRO formation from typed load + `sfpi::min/max` + store dataflow, including safe alternating destination allocation and prologue/epilogue generation. It must not recognize operation names or this test kernel.

## Blackhole silicon: Addcmul

Both selectors passed fresh silicon correctness processes before profiling. Each perf cell is three more fresh, serialized device processes.

| Scope | Production handwritten | Fresh semantic C++ | Delta |
|---|---:|---:|---:|
| `mean(MATH_ISOLATE)` cycles/tile | 292.921875 / 292.921875 / 292.921875 | 357.03125 / 357.03125 / 357.03125 | +21.89% |
| `mean(L1_TO_L1)` cycles/tile | 297.625 / 297.625 / 297.625 | 361.6328125 / 361.6328125 / 361.6328125 | +21.51% |

Evidence is archived under `/localdev/nkapre/fresh-cpp-addcmul-bh-silicon-20260815`. It contains the correctness and perf logs, raw/post CSVs, the MATH_ISOLATE ELFs and disassemblies, per-run hashes, and an aggregate SHA256 manifest.

This negative isolates a second general compiler opportunity. Both variants form replay, so replay formation itself is not the gap. Production unrolls and interleaves two independent rows into one 14-instruction capture: two loads for each input stream, two multiplies, two MADs, two rounding operations, and two stores, then advances RWC by four. Fresh semantic C++ captures one row as seven instructions and advances by two after every replay. It therefore fails to overlap the independent multiply/MAD/round latency chains across adjacent loop iterations. The durable target is target-aware loop unroll-and-jam/software pipelining before replay capture, driven by dependency and SFPU latency/resource information. It must remain operation-agnostic and preserve the source expression's rounding/order semantics.

## Blackhole silicon: SigmoidAppx

Both selectors passed the paired silicon correctness gate. The fresh typed cubic uses the checked-in Float16_b contract (`atol=0.13`, `rtol=0.05`, and PCC greater than `0.99`); its host discriminator measured maximum absolute error `0.060843` and PCC `0.997736`. Wormhole and Blackhole correctness and profiler sources compiled, and both Blackhole selectors passed CRAQ before device execution.

Each cell is three fresh, serialized device processes. The metric is scoped `TILE_LOOP mean(MATH_ISOLATE)` cycles/tile.

| Production | Fresh semantic C++ | Delta |
|---:|---:|---:|
| 222.8515625 / 222.8515625 / 222.8515625 | 446.8515625 / 446.8515625 / 446.8515625 | +100.52% |

Evidence is archived under `/localdev/nkapre/sigmoidappx-bh-silicon-20260815`; its aggregate SHA256 manifest is `b52cdbd14b8b89c62d37d289d5830a5a0c28324f13f55390e73d8a58f0305908`.

The production body uses a replayed load, `SFPLUT`, add, and store. The fresh cubic instead materializes two FP32 constants inside every row (four `sfploadi` half-immediates), then emits square, MAD, multiply, add, and store without replay. This is negative evidence for three generic targets: loop-invariant SFPU constant hoisting, allocation of invariants to special registers, and replay extraction from counted typed loops. The compiler work must be driven by invariant dataflow and loop legality, not by recognizing SigmoidAppx or its coefficients.
