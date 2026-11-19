# Stage 3 operation-topology audit

CPU-only inspection before stage-3 knob tuning. No device experiment or optimization verdict is claimed by this audit. The measured baseline is the completed `tt/fused_decoder.py`, selected d810 source; stage-2 final clean timing is 1.447585 ms linear decode / 0.583101 ms full-attention decode, 478.131568 / 8.937745 ms S256 prefill. Stage 3 must repeat a serial matched baseline and beat the best correct candidate, not the historical functional path.

## Measured operations

Numbers below are recomputed from `../fused_decoder/tracy/layer_{0,3}/{prefill,decode}_perf_report.csv`: sum `Device Time [μs]` by `OP Code`, dividing decode sums/counts by the 16 measured replays. They exclude profiler/host gaps; they are not clean wall latency. Prefill is one warmed S256 invocation. Totals are 465412.496 us linear prefill, 1306.809 us linear decode, 8732.648 us full prefill, 549.995 us full decode.

| Path | Dominant operation or group | Device us per invocation | Interpretation |
|---|---|---:|---|
| Linear decode | 11 GenericOp kernels | 579.590 | Sensitive exact projection, norm, convolution and recurrence helpers; investigate kernel identity before changing arithmetic. |
| Linear decode | 62 BinaryNG kernels | 238.263 | Large recurrent math surface; blanket low precision or fusion is unsafe. |
| Both decode kinds | Sparse expert gate/up, active 8/256 | 113.780 / 113.804 | Shared optimization opportunity, current BF16 HiFi4. |
| Both decode kinds | Sparse expert down, active 8/256 | 59.426 / 60.028 | Shared optimization opportunity, current BF16 HiFi4. |
| Both decode kinds | Shared gate/up/scalar packed projection | 30.196 / 30.141 | K2048 N1056, interleaved input/weights; DRAM-sharded candidate. |
| Full decode | Packed Q/K/V/output gate | 83.370 | K2048 N9216; DRAM sharded already, block width 2. |
| Full decode | Attention output projection | 40.242 | K4096 N2048; DRAM sharded already, block width 2. |
| Full decode | Router | 24.689 | Keep routing accuracy separate from expert precision. |
| Full decode | SDPA decode | 22.454 | Native paged SDPA already present. |
| Full prefill | Four grouped expert gate/up + four down | 6559.657 | 75.1% of device sum; highest-value prefill target. |
| Linear prefill | GenericOp + BinaryNG + accumulation | 397595.452 | 85.4% of device sum; corrected ordered recurrence dominates. |
| Linear prefill | Grouped expert gate/up + down (16 each) | 19400.835 | Same MoE policy applies; only 4.2% of full linear prefill device time. |

## Topology and concrete candidates

| Current sequence | Replacement / search | Constraint and action recommended |
|---|---|---|
| Indexed sparse packed gate/up -> split -> SiLU multiply -> indexed sparse down -> weighted sum | First introduce explicit expert dtype/fidelity and program policy. Cross BFP8 HiFi2/LoFi and BFP4 LoFi with block widths 8/16/32/64 where K permits, core counts and N/subblock widths. | Both kinds benefit. Keep indices-driven active 8-expert execution. `nnz` is unnecessary unless its exact interpretation is proved. BFP4 is candidate-only for stage 8. |
| Prefill four expert-bank routing/gather/grouped gate-up/down/combine sequences | Same expert policy; tune grouped API-supported configs and phase-specific L1 activation storage. | Current grouped ops have a different API from dense DRAM-sharded matmul. Perf-report's generic DRAM advice is not evidence they accept the dense config. Inspect implementation, adapt if legal, and retain exact blocker or measured result. |
| Packed shared gate/up plus padded scalar gate, followed by shared down | Compare packed versus tuned separate gate/up and scalar projection; add DRAM-sharded decode for each legal family. | Current N1056 packs 512+512+32, with only one logical scalar column. Preserve slicing and sigmoid weighting. Same input as router, but merging router would force its accuracy policy onto shared weights and risk tie behavior. |
| Full-attention packed Q/K/V/output gate | Compare current packed family against legal separate projections using equal dtype/fidelity; tune readers 1/2/3 and input-core counts 8/16/32. | Current K2048 input over 32 cores allows only two K tiles per input shard. To test larger block widths, lower activation shard core count rather than rejecting width >2 outright. Preserve Q/gate ordering and partial RoPE. |
| Full attention output DRAM matmul -> DRAM output -> residual -> sharded norm -> DRAM -> MoE | Keep output/residual/norm sharded through next consumers; compare complete boundary cost. | Current output is unconditionally returned to DRAM by `linear`; an isolated layout microbenchmark cannot establish integrated gain. Linear attention has different exact norm implementation and must not receive this rewrite blindly. |
| Linear state projection and output-only Q/Z projection share norm input | Retain sensitive state/terminal scans; evaluate packing only where same arithmetic can be preserved. | The state path uses ordered arithmetic and output path has different precision/output needs. A single ordinary packed matmul is not a legal drop-in numerical equivalent. Separate can be justified by real-weight failure evidence, not op count alone. |
| Native paged SDPA, BF16 KV, sharded q/k/v | Keep baseline cache initially; BFP8 KV candidate only with short prefix, nonaligned tails, and capacity-contract tests. | KV change triggers max-context validation; 32K–64K smoke suffices if dtype/layout stays unchanged. Never lower the advertised 262144 limit speculatively. |
| Linear tiled/row-major conversions around exact norms, projection scans, lookup and recurrence | Inspect inherited stage-2 removal evidence before repeating; prioritize movement inside selected helper boundaries. | There are earned conversion requirements and arithmetic hazards. Generic `reshape()` explicitly tiles to handle padded-row boundaries. Removing every call with that name would corrupt logical tails. |

All production weight uploads in the inspected fused constructor default to **BF16**, not BFP8; dominant profiler rows confirm BF16 x BF16 at HiFi4. Stage-3 BFP8 is therefore a real precision change. Do not claim inherited BFP8 defaults, and do not blanket-convert lookup tables, norms, router/state projections, FP32 recurrent state, or custom-kernel byte contracts. Record explicit sensitive-group exceptions supported by measurements.

## Prior precision evidence to preserve

The fused work log is a navigation aid; inspect linked raw results for selection. Its layer-34 investigation (`endurance/v5_layer34_repair_comparison.json`, `endurance/AUTOFIX_v5_layer34.md`) documents actual-checkpoint failure after seemingly benign native replacements: combined precision + BF16 top-k failed 203 positions, and FP32 softmax/top-k still failed 200. Correct routing alone did not fix upstream recurrent state. Current ordered projections, canonical chunk64 state, exact selector, convolution, terminal norms, and corrected matmul repair that failure.

`endurance/AUTOFIX_output_gate_operand_precision.md` identifies a concrete BF16 binary-activation staging hazard: removing FP32 Z conversion rounds SiLU prematurely. Preserve that conversion until a complete compatible replacement passes. The router storage trial `router_bf16_storage_{0,3}.json` was rejected on speed, not a general low-precision impossibility. There is no evidence here that expert BFP8/BFP4 or HiFi2/LoFi fails; those remain material candidates.

## Harness reuse and acceptance risks

- `tests/run_decoder.py` loads real checkpoint weights, computes fresh HF reference, checks whole-output and state PCC, enforces device-only forward, then measures warmed prefill and traced decode. Its activations are seeded random BF16, not recorded full-model hidden states. Real weights are present, but label the activation source honestly.
- `tests/benchmark_fused_snapshot.py` can load an explicitly named source with `FusedDecoder` and routes `run_decoder.FunctionalDecoder` to that class. Its outputs deliberately say `acceptance_eligible=false`. Reuse its measurement code in an optimized-specific harness, but deliver final tests asserting `OptimizedDecoder` identity and correct production source; merely renaming JSON is insufficient.
- `tests/stress_fused_snapshot.py` similarly injects a candidate into the fresh-HF changing-input stress harness. Required representatives include layer 34 seed154 as well as layers 0 and 3. Preserve allocation tracking including program cache, repeated replay, 33-token continuation and whole-output PCC; report per-row failures diagnostically and investigate actual model-visible deviations.
- Existing `test_fused_decoder.py` explicitly patches the fused class and uses a synthetic boundary battery. Copying that test without changing the injected class would test the wrong path. Synthetic boundary tests supplement real-weight accuracy, not replace it.
- Profile captures are separately implemented by `profile_fused_snapshot.py` and chunk-draining variants, because linear prefill has many ops. Keep current completeness checks and one warmed measured region; do not profile concurrently or under Watcher.
- The previous broad 8192-step endurance battery is useful targeted evidence if sensitive math changes. Stage 3's 32K–64K context smoke and representative repeated/stress runs still need current-source evidence. No max sweep is needed without KV dtype/layout change.

This audit reads existing evidence only. Every candidate above remains proposed until the stage owner records measured action/results in the work log; no candidate is rejected merely from this audit.
