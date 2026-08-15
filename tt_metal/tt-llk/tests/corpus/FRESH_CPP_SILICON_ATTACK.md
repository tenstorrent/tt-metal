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
| Addcmul body with P4 | 292.9296875 | 292.9921875 | parity (+0.02134%) |
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

The initial compiler-visible body lost by 21.89% in `MATH_ISOLATE`: it formed two independent seven-instruction replay groups instead of the handwritten implementation's interleaved 14-instruction group. That result identified adjacent Dst-iteration fusion and cross-row latency hiding as the durable compiler gap; it was not a win.

P4 phases 2 and 3 implement that mechanism generically and default-off in SFPI-GCC `3a5e7c4e4`: a pre-IRA pass proves two typed Dst iterations equivalent and disjoint, rewrites the second row's addresses and RWC advance, then interleaves the two independent dataflow chains before replay formation. It has no operation-name matching. The linked generated performance ELF contains the expected 14-instruction replay group and has SHA256 `ce31a5664442231732afaac37b267da2f810680515f667c450dd5d18093a5fb1`.

Both selectors passed fresh silicon correctness before profiling. Each perf cell below is three additional fresh, serialized Blackhole processes using the same scoped `TILE_LOOP` zone.

| Scope | Production handwritten | Fresh semantic C++ + P4 | Delta of sample means | Classification |
|---|---:|---:|---:|---|
| `mean(MATH_ISOLATE)` cycles/tile | 292.921875 / 292.9453125 / 292.921875 | 292.9921875 / 292.9921875 / 292.9921875 | +0.0625 cycles (+0.02134%) | parity, not a win |
| `mean(L1_TO_L1)` cycles/tile | 297.625 / 297.625 / 297.625 | 297.6640625 / 297.6640625 / 297.6640625 | +0.0390625 cycles (+0.01312%) | parity, not a win |

This closes the measured 21.89% generated-code deficit to reproducible silicon parity; it does not establish a broad compiler win over hand tuning. The paired CRAQ gate also passed 2/2. Raw/post CSVs, per-process logs, and per-sample hashes are archived under `/localdev/nkapre/addcmul-phase3-silicon-v9-archive`. The SHA256 of the lexicographically sorted per-file SHA256 listing is `73854b6d47c34b37fc0e6fadd3cdd292bc1e7f8ee3b4f187f27a5cb815f113bf`.

## Blackhole silicon: SigmoidAppx

Both selectors passed the paired silicon correctness gate. The fresh typed cubic uses the checked-in Float16_b contract (`atol=0.13`, `rtol=0.05`, and PCC greater than `0.99`); its host discriminator measured maximum absolute error `0.060843` and PCC `0.997736`. Wormhole and Blackhole correctness and profiler sources compiled, and both Blackhole selectors passed CRAQ before device execution.

Each cell is three fresh, serialized device processes. The metric is scoped `TILE_LOOP mean(MATH_ISOLATE)` cycles/tile.

| Production | Fresh semantic C++ | Delta |
|---:|---:|---:|
| 222.8515625 / 222.8515625 / 222.8515625 | 446.8515625 / 446.8515625 / 446.8515625 | +100.52% |

Evidence is archived under `/localdev/nkapre/sigmoidappx-bh-silicon-20260815`; its aggregate SHA256 manifest is `b52cdbd14b8b89c62d37d289d5830a5a0c28324f13f55390e73d8a58f0305908`.

The production body uses a replayed load, `SFPLUT`, add, and store. The fresh cubic instead materializes two FP32 constants inside every row (four `sfploadi` half-immediates), then emits square, MAD, multiply, add, and store without replay. This is negative evidence for three generic targets: loop-invariant SFPU constant hoisting, allocation of invariants to special registers, and replay extraction from counted typed loops. The compiler work must be driven by invariant dataflow and loop legality, not by recognizing SigmoidAppx or its coefficients.

### Generic invariant-hoist and counted-replay follow-up

The fresh helper is now kept as a separate `noinline` function so its loop is a
clean compiler input: the function contains only typed Dst loads/stores and the
cubic expression.  Setup, profiler ownership, and RWC setup remain outside the
helper.  There are still no physical LREGs, raw instructions, replay commands,
or copied issue schedules in the semantic body.  The call and return are inside
the measured math scope.

With the operation-agnostic invariant-load and counted-loop replay options, the
linked Blackhole body hoists the four half-immediate loads once and captures a
six-word `SFPLOAD; SFPMUL; SFPMAD; SFPMUL; SFPADDI; SFPSTORE` payload.  The
trailing `TTINCRWC` remains explicit.  Wormhole and Blackhole compilation and
paired CRAQ correctness pass; QSR conservatively refuses these transformations.

The silicon result is an honest loss.  Each value below is a fresh serialized
process, and the scoring field is `TILE_LOOP mean(MATH_ISOLATE)`:

| implementation | three Blackhole samples | median | versus production |
|---|---:|---:|---:|
| production LUT body | 222.8828125 / 222.8828125 / 222.8828125 | 222.8828125 | baseline |
| optimized fresh typed cubic | 361.796875 / 361.796875 / 361.796875 | 361.796875 | +138.9140625 (+62.3261%) |

The two generic passes recover 85.0546875 cycles, or 19.0342%, from the prior
fresh result of 446.8515625 cycles, but do not flip the production comparison.
The six raw/post CSVs and their manifest are archived under
`/localdev/nkapre/sigmoid-silicon-final/cycles`; the manifest hash is
`79a308bdb0e29fce4a341ab315cd960c5354a2d41fcc4239a55498f25c3ae657`.
An earlier run made with performance counters enabled is retained separately in
`counter-only` and is explicitly non-scoring because it has no valid
`mean(MATH_ISOLATE)` field.  Extracted `.text` is invariant across the three
runs: production is 2220 bytes with SHA256
`2b8ed1faa233791dca63ae24031e5a838a9284d1b6900bc14514acfcda65acae`,
and optimized fresh is 2216 bytes with SHA256
`d45c87b7ddc13595fae864e51ba6dd6b5b613605ae8d8f71e0f46a54d365ac8c`.

The remaining mechanism is visible directly in those ELFs.  Production uses a
four-word replay payload (`SFPLOAD; SFPLUT; SFPADDI; SFPSTORE`) and unrolls eight
rows per scalar loop iteration.  The typed cubic uses six payload words and
keeps a scalar branch for every row.  Across 32 rows that is approximately 128
production SFPU payload operations versus 192 typed-cubic payload operations,
plus 28 extra scalar loop backedges; the fresh capture also evaluates one
redundant payload before its first advancing playback.  Constant placement and
replay delivery are therefore no longer the dominant gap.  The next durable
target is semantic cubic-to-LUT lowering under an explicit approximation/error
contract (and generic replay-aware unrolling), not more replay special casing.
