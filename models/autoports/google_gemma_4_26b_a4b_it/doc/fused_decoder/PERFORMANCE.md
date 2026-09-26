# Fused decoder performance

Exactly4096 input tokens,128 subsequent decode steps, batch1 and one concurrent request on one Blackhole ASIC. These are whole single-layer measurements, not full-model token generation.

| Layer kind | Functional prefill us | Fused prefill us | Speedup | Functional traced decode us | Fused traced decode us | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding_attention | 4998536.82 | 2812752.09 | 1.777x | 9811.07 | 5146.70 | 1.906x |
| full_attention | 5009442.54 | 2825397.16 | 1.773x | 10894.37 | 5584.81 | 1.951x |

The denominator spans the first native firmware start through the last firmware end of the complete layer. Every operation and internal device gap is included. Decode is the mean of128 separate replay windows; inter-replay host/input-refresh gaps are excluded. No summed-kernel subtotal or host-wall duration is substituted.

| Layer kind | Useful prefill FLOPs | Useful FLOPs % | Estimated decode DRAM bytes | DRAM % |
| --- | ---: | ---: | ---: | ---: |
| sliding_attention | 882489950208 | 0.189131 | 611427644 | 23.203124 |
| full_attention | 1215408635904 | 0.259315 | 707072292 | 24.727804 |

## Calculation basis

Peak basis: one participating ASIC,120 physical Tensix cores at1.35GHz,4096 FLOP/core/cycle divided by four for BF16 HiFi4:165.888TFLOP/s and512GB/s DRAM. The runtime exposes110 workers; the denominator uses the theoretical ASIC peak. The layer mixes SFPU, FP32/BF16 and fidelities, so this is a useful-work reference rather than homogeneous FPU utilization. [P300c specifications](https://docs.tenstorrent.com/aibs/blackhole/p300.html) and [per-ASIC bandwidth](https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html) support the hardware basis; installed tt-perf-report1.3.0 supplies the FLOP/fidelity convention.

Useful FLOPs include logical QKV/output projections, shared MLP, top8 active experts, router and causal attention. Full tied K/V counts once. Padding, masked work, extra prefill experts, normalization/transcendentals are omitted from the numerator, while their time remains in the denominator. Prefill computes all128 experts per internal batch; this helps explain why useful whole-layer efficiency is much lower than individual sparse-matmul percentages.

DRAM traffic estimates one read/write per padded DRAM operand/output per native op. Sparse weights use recorded nnz/128; embedding/slices count selected inputs; whole-cache untilization counts the full pool. Fused cache operands/outputs are corrected to one32-token page, matching captured INPUT_0/INPUT_2/OUTPUT_0/OUTPUT_1 shapes. Per-core rereads, metadata and profiler traffic are excluded. These are estimates, not controller counters. No percentage is clamped.

Formulae: `100 * useful_flops / (peak_flops_per_s * prefill_device_us / 1e6)` and `100 * estimated_dram_bytes / (peak_dram_bytes_per_s * decode_device_us / 1e6)`.

## Evidence and interpretation

- Final tracy/{sliding,full}_verified/whole_layer.json and .replays.csv contain complete windows and assumptions. provenance.json records original CSV paths, checksums and runtime source hash; verified_validation_commands.json records the same hash per command.
- prefill_perf_report.txt renders the complete prefill table; decode_perf_report.txt renders one complete measured replay. Corresponding *_perf_report.csv files contain all filtered rows. Stacked CSV/PNG summaries and exact report_commands.json are present; large raw captures remain local.
- Sparse matmuls dominate prefill. Gate/up packing and64-row peer batching reduce dispatches. The latter is a small additional gain with bitwise-equal component outputs. Program/fidelity settings remain inherited apart from M sizing for merged batches and the shared decode output-sharding adaptation.
- Decode is dominated by FP32 projection products/reductions, then experts and attention. Broadcast/coalesced projections, packed experts, batched attention and common normalization remove measured work. Producer/consumer sharding removes additional copies, including the concat-heads→output-projection boundary and both MLP outputs into norms. Native SDPA, sliding head norm, sliding prefill rotary and I2S cache dtype conversion fail real accuracy after adaptations; retained boundaries have controls in AUTOFIX.md/PATTERNS.md.
- Native-op counts fall from2501 to1141 prefill and560 to152 per decode replay for sliding; full falls from2473 to1077 and502 to142. These explain topology; latency determines acceptance.
- Runtime rows confirm BF16 weights/cache, HiFi4 expert prefill, inherited LoFi expert sparse decode, HiFi2 shared gate-up/prefill and LoFi shared-down decode, and HiFi4 attention projections. Expert mixing is HiFi4/BF16 output with FP32 accumulation for sliding and BF16 accumulation for full. Shared down writes norm shards with K-block6; the initial K-divisor candidates all inferred LoFi/BF16 while the interleaved control inferred HiFi2. Explicit matched fidelity controls correct that initial attribution; see shared_fidelity_commands.json. No later pipeline stage was performed.

The earlier loop32 full capture is invalid because marker pairing failed during teardown. AUTOFIX_profiler.md records the likely clock-rollover cause; fresh captures pass with unchanged instrumentation and validation enabled. An intermediate summary attempt read a CSV before export finished and was rerun after producer exit. Neither invalid nor incomplete capture supplies final values.

## Final-default reproduction and candidates

- sliding_attention: final unprofiled replay median5067.23us; chosen combined candidate5067.09us; fresh functional control9504.03us.
- full_attention: final unprofiled replay median5521.44us; chosen combined candidate5520.47us; fresh functional control10677.67us.

These host timings are used for candidate selection and are not device/roofline fields. Final defaults reproduce their chosen candidates within0.03%, beat the alternative correct combined configurations, and improve over the preceding best mix-only policies. candidate_summary.json contains66 rederived headline timing rows; passed there denotes the headline check only. Faster early native-normalization sliding candidates fail the real S2049 request-reuse control and are not correct baselines. All accepted/rejected applicable graph patterns and the coherent boundary combinations are in PATTERNS.md and command journals.

### Explicit shared-down fidelity control

The final profiler revealed inferred LoFi for the explicit sharded program, while the original interleaved baseline inferred HiFi2. The initial probe's HiFi2 description was wrong; it did not invalidate the measured outputs or times. Matmul program-config inference is the cause (matmul_device_operation.cpp:2808–2810). Four fresh real-activation controls explicitly set LoFi or HiFi2 with the same BF16 operands, FP32 destination disabled, packer L1 accumulation enabled, norm consumer, layout and K-divisor sweep. Every candidate passes PCC and repeated replay. K6 remains fastest under both policies:

| Kind | Interleaved HiFi2 baseline (us) | Sharded K6 LoFi (us / PCC) | Sharded K6 HiFi2 (us / PCC) |
| --- | ---: | ---: | ---: |
| sliding_attention | 62.031 | 44.718 / 0.9998896404 | 44.880 / 0.9999369429 |
| full_attention | 62.018 | 44.764 / 0.9999019428 | 45.018 / 0.9999455949 |

These are warmed traced host boundary timings, not whole-layer device times. The small fidelity difference is distinguished from the larger sharding improvement. LoFi remains the best measured correct configuration; no runtime change or headline reprofile is needed. Raw controls: shared_fidelity_{sliding,full}_{LoFi,HiFi2}.json; exact commands and exit codes: shared_fidelity_commands.json. Final whole_layer.json summaries and telemetry name the actual mixed-fidelity policy. Historical probe JSON is retained, with its stale attribution superseded here.
