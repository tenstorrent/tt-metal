# Phi-3.5 Mini optimized decoder

`tt/optimized_decoder.py` is a standalone single-device TTNN decoder. It does
not inherit from, call, or fall back to the functional decoder. The optimized
tests import this class directly and assert the selected policy and source-path
contract.

## Selected cumulative path

| Area | Final selection |
| --- | --- |
| Projection precision | BFP4 weights and LoFi for QKV, output, packed gate/up, and down |
| KV cache | BFP8, DRAM-interleaved, paged with page size 32 |
| Decode residual | BF16 width-sharded L1 over 8 cores through norms, residual adds, and matmuls |
| Decode matmuls | DRAM width-sharded weights with role-specific block widths QKV/output/gate/down = 12/12/6/16 |
| Projection topology | Packed QKV and packed gate/up |
| RoPE | Canonical manual rotate-half TTNN composite; fused adjacent-pair RoPE is retained only as a rejected candidate |
| Cache/attention | Fused paged K/V update and TTNN's default paged decode SDPA program |
| Prefill | DRAM-interleaved activations, large-M 8x8 programs, QKV/output block width 2, grouped gate-up/down block width 4, internal long-M chunks and batch groups of eight |

The batch-1 decode activation is padded to one M tile, so `per_core_M=1` is
legal. DRAM-sharded weights are used at both batch 1 and 32.

## Operation-topology audit

| Current operation/opportunity | Candidate | Action | Evidence |
| --- | --- | --- | --- |
| Same normalized input feeds Q, K, V | Three projections vs packed QKV | Packed kept | Final trace has one `32x3072x9216` matmul per replay |
| Same normalized MLP input feeds gate and up | Separate vs packed projection | Packed kept | Isolated attention-topology control: separate `0.392/0.507` ms vs packed `0.358/0.473` ms b1/b32; final packed semantic path passes |
| Packed gate/up split | Direct sharded split vs one interleaved helper boundary | Helper boundary kept | Direct sharded split is finite but wrong: PCC `-0.0278/-0.0021`; artifact `final_context128_sharded_packed_split.log` |
| Residual/norm chain crosses default DRAM layouts | Persistent L1 width sharding | Kept | Trace shows 8-core sharded norms and DRAM-sharded projection inputs |
| Two cache updates | Separate vs fused K/V update | Fused kept | Isolated non-fused `0.395/0.493` ms vs fused `0.358/0.473` ms; paired semantic controls give identical PCC |
| Decode attention program and RoPE basis | 2x2 default/explicit SDPA and manual/fused RoPE | Default SDPA + manual RoPE kept | Only this quadrant passes recorded semantic b32: `0.999963`; the other three give `0.986243`–`0.991249` |
| Head-width-96 rotate-half composite | Manual ops, padding, or fused RoPE | Manual kept | Fused is faster on random/zero-cache controls, but fails the recorded nonzero-cache gate jointly with both SDPA choices |
| Phase-specific prefill RoPE | Fused adjacent-pair prefill plus canonical manual decode | Rejected | Separate pair-basis QKV/tables plus device-only strided slices restore canonical Q/K before SDPA/cache. PCC passes at b1/b32 (`0.998572/0.998578`) and optimized-prefill cache decode passes (`0.998924/0.998949`), but prefill regresses to `1.626/28.358` ms from `1.596/19.670` |
| Large prefill matmuls | Block width 4, 2, inner M/N, and 8x10 grid | Width 2 on 8x8 kept | b32 block-4 `13.216` ms, block-2 `10.064` ms; inner-M/N and 8x10 failed b32 PCC, not merely their first API attempt |
| Long prefill exceeds per-program L1 | Default, 2048-row, 1024-row chunks | 1024-row internal chunks kept | 2048 requests 1,585,920 > 1,572,864 bytes; 1024 passes 131071 and 131072 |
| Batch-32 packed MLP exceeds L1 | Public restriction vs internal grouping | Groups of 8 kept | Groups of 16 request 1,585,920 > 1,572,864 B at logical 127; group-8 preserves public batch 32 |
| Composite attention | Manual attention vs SDPA | SDPA kept | Prefill trace contains one `SDPAOperation`; decode contains one `SdpaDecodeDeviceOperation` per replay |
| CCL/collectives | Audit applicability | Not applicable | This stage is single-device and the final trace contains no collective |

The final trace contains no Torch conversion, host operation, or host fallback.
Remaining layout operations are TTNN contract boundaries for canonical
rotate-half, disjoint K/V grids for fused cache update, head concatenation back
to the residual shard, and the only correct packed gate/up split. The final
b32 profile attributes 3.955 ms to six manual-RoPE permutes. That class was
attacked with the phase-specific fused-prefill adapter above; its required
canonical-cache conversion made total prefill 44.1% slower, so it is retained
only as an explicit sweep policy (`PHI35_OPT_POLICY=phase_split_prefill_rope`).
At the advertised context it adds 116,590,592 persistent bytes per decoder
instance: 100,663,296 B of pair-basis tables, a 15,925,248 B BFP4 QKV copy,
and a 2,048 B transform. This does not fit the selected performance policy and
is not allocated by default.
Direct sharded replacements were also tested; the packed split corrupts values
and the decode RoPE transpose corrupts later batch-32 lanes. The default
manual-RoPE path shares its four row-major tables between prefill and decode,
avoiding the 100,663,296-byte duplicate-table allocation introduced by fused
candidates.

## Correctness and context contract

The PCC bar is 0.995, matching the functional stage. All 16 final tests pass.
Phi-3.5 Mini uses one meaningful dense decoder-layer kind; the suite exercises
its attention, gated MLP, short and long RoPE, prefill/decode, and cache paths.

| Gate | Final result |
| --- | --- |
| Real-weight prefill/decode at logical 33 | PCC `0.9999258 / 0.9999315` |
| Page-boundary prefills 31/32/33/63/64/65 | All PCC >= `0.9999931` |
| Multi-user batch-2 prefill at 33 then decode at 33 | PCC `0.9999931 / 0.9999944`; decode consumes each user's permuted physical pages |
| Nonzero prefill at 32769 | Last-token PCC `0.9969488` |
| Prefill 131071 and exact 131072 | Both pass with exact public shapes |
| Decode at logical context 131072 | PCC `0.9999962` |
| Varied-position b32 decode consuming nonzero cache | PCC `0.9999572` |
| Traced decode b1/b32 | PCC `0.9999962 / 0.9999956`; 10 replays bitwise identical |
| Watcher | `TT_METAL_WATCHER=10`: 5 representative optimized tests pass, no watcher error |

Non-aligned `seq_len` remains public API input; padding and chunking are
internal. BFP8 KV uses less memory than the functional BF16 cache, so the
advertised 131072-token capacity is preserved rather than reduced.

## Like-for-like before and after

Both harnesses use full model shapes, logical position 127, context 128,
warmed prefill, and ten traced decode replays.

| Mode | Functional b1 | Optimized b1 | Change | Functional b32 | Optimized b32 | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill, ms | 2.084286 | 1.608723 | 22.8% faster | 36.949536 | 19.744849 | 46.6% faster |
| Decode mean, ms | 1.009399 | 0.466318 | 53.8% faster | 1.331827 | 0.793861 | 40.4% faster |

Functional prefill uses synthetic values; functional decode and all optimized
signoff use checkpoint weights plus recorded layer-0 activations and matching
prefix caches. Dense runtime is shape/dtype driven. Functional semantic decode
PCC is `0.999822/0.999731`; optimized PCC is `0.999264/0.998993`. The primary
batch-1 target beats the best correct functional baseline, and batch 32 does
not regress.

## Precision and fidelity sweep

The original projection-by-projection sweep used checkpoint weights at both
batches and selected BFP4/LoFi projections plus BFP8 KV. A later independent
review required recorded target activations and their matching nonzero prefix
cache. That semantic control exposed an attention-policy interaction which
random and zero-cache activations masked:

| Policy/control | b1 PCC / ms | b32 PCC / ms | Decision |
| --- | --- | --- | --- |
| BFP8/HiFi2, BF16 KV, fused RoPE + explicit SDPA | `0.999950 / 0.551` | `0.988046` | Reject |
| Same, BFP8 KV | — | `0.988068` | KV dtype refuted |
| Same, identity page table | — | `0.988046` | Page routing refuted |
| BFP8/HiFi2, manual RoPE + default SDPA | `0.999949 / 0.668` | `0.999963 / 1.039` | Accuracy control |
| BFP4/LoFi, BFP8 KV, manual RoPE + default SDPA | `0.999264 / 0.467` | `0.998993 / 0.795` | Selected |
| Selected policy with optimized-prefill-produced cache | `0.998985 / 0.467` | `0.999005 / 0.799` | End-to-end cache control |

BF16/HiFi4 weights were retried but require 2,208,512 bytes of L1 versus
1,572,864 available. The higher-precision runnable BFP8/HiFi2 control and the
final reduced-precision policy both clear PCC 0.995, so the final choice keeps
the faster BFP4/LoFi projection policy.

## Decode geometry sweep

The cumulative BFP4/LoFi+BFP8 matmul policy was swept with real weights at
both batches. Times below isolate the geometry sweep before the attention
policy rollback; the selected geometry was then revalidated in the final
manual-RoPE/default-SDPA path at both batches.

| Geometry | b1 | b32 | Decision |
| --- | ---: | ---: | --- |
| core8 base `4/4/4/4` | 0.379 | 0.494 | Baseline |
| QKV block 6 | 0.361 | 0.474 | Valid |
| output block 6 | 0.360 | 0.476 | Valid |
| gate/up block 4 | 0.361 | 0.475 | Valid |
| down block 8 | 0.363 | 0.477 | Valid |
| down block 32 | 0.357 | 0.474 | Tiny b1 noise win, b32 worse |
| core16 | 0.363 | 0.473 | Valid, no joint win |
| core32 | 0.381 | 0.487 | Rejected |
| selected `12/12/6/16`, core8 | 0.358 | 0.472 | Best geometry; final semantic path is `0.467/0.795` |

Gate/up block 12 was retried through legal block 6 and core-16/32 geometries;
the original candidate needs 1,618,688 bytes versus 1,572,864 available L1.
Decode's DRAM-sharded program type exposes no output-subblock fields
(`matmul_program_config_types.hpp:71` and `matmul_nanobind.cpp:547`), so the
report's generic “no output subblock” advice is inapplicable to this factory.

## Final profiler and roofline

Artifacts in `tracy_final/` contain all four final signpost windows. The raw
capture proves canonical device-only RoPE, fused cache update, default paged
SDPA, BFP4/LoFi material matmuls, and the final batch-32 prefill programs:
block width 2 for QKV/output and block width 4 for grouped gate-up/down.

| Window | Device ops | Op gaps | Device + gaps | Same-run E2E | Modeled DRAM roofline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill b1 | 1.219 ms | 0.682 ms | 1.901 ms | 1.986 ms | 11.1%, 57 GB/s |
| Prefill b32 | 18.974 ms | 0.526 ms | 19.500 ms | 19.794 ms | 5.9%, 30 GB/s |
| Decode b1, per replay | 0.445 ms | 0.073 ms | 0.518 ms | 0.520 ms | 24.9%, 127 GB/s |
| Decode b32, per replay | 0.699 ms | 0.102 ms | 0.801 ms | 0.804 ms | 15.8%, 81 GB/s |

The decode CSVs contain ten replays. Dividing their raw totals gives
0.444738/0.698605 ms of device operations and 0.073456/0.102247 ms of op gaps
at b1/b32. The same Tracy run reports host E2E means of
0.520112/0.804059 ms; device plus gaps gives 0.518194/0.800853 ms. The
remaining 0.0019/0.0032 ms is measurement and host synchronization overhead.
Non-profiled headline means are 0.466318/0.793861 ms, showing expected
profiler overhead without mixing those values into the same-run accounting.

Representative decode rows per replay are QKV 54–55 μs, packed gate/up
92–93 μs, down 47–48 μs, and batch-32 SDPA 192 μs. Gate/up remains the
largest batch-1 matmul and was attacked by packing, precision/fidelity, core
geometry, block width, and split-layout sweeps. Default SDPA becomes the
batch-32 leader after the correctness rollback.

A conservative decode byte roofline counts one read of 113,246,208 BFP4
weight elements (63.701 MB including tile storage) plus context-128 BFP8 K/V
reads. At 512 GB/s this is a 0.126 ms lower bound for b1 and 0.177 ms for b32,
excluding activations and helper traffic. The measured device replay times are
about 3.53x and 3.95x those lower bounds. Against same-run E2E the conservative
ceilings are 24.1% and 22.0%. This is consistent with the final
`tt-perf-report` rooflines, 0.073/0.102 ms replay gaps, and necessary
manual-RoPE/non-matmul work.

## Commands and artifacts

```bash
PHI35_REAL_WEIGHTS=1 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/functional_decoder_perf.py
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
PHI35_REAL_WEIGHTS=1 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py
TT_METAL_WATCHER=10 pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py -k 'real_weight or multi_user or varied_positions or decode_trace_replay'
PHI35_REAL_WEIGHTS=1 python -m tracy -r -p -v -m pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/optimized_decoder_perf.py
```

Primary current-source logs are `final_perf_after_phase_candidate.log`,
`final_optimized_prefill_cache_after_phase_candidate.log`,
`final_correctness_after_phase_candidate.log`,
`final_watcher_after_phase_candidate.log`, and
`final_tracy_after_phase_candidate.log`. Candidate logs and AutoDebug reports are retained
beside them. `tt-smi` was unavailable in this environment; every hardware
command was timeout-bounded and all device-close logs completed.
The four `tracy_final/*_perf_report.txt` files are human-readable
`tt-perf-report` tables with advice; the matching CSV files are machine
readable, and the nonempty `*.console.log` files retain CSV-generation
diagnostics.

## Optimize checklist

- [x] Started with operation-topology and movement audit.
- [x] Measured warmed prefill and traced decode before/after at b1 and b32.
- [x] Swept real-weight precision/fidelity by projection group and KV dtype,
  then reran semantic recorded-target controls with matching nonzero caches.
- [x] Swept DRAM-sharded decode matmul families/geometries at b1 and b32.
- [x] Swept large-prefill program configs at b1 and b32.
- [x] Evaluated sharding, packed same-input projections, fused cache update,
  fused/manual RoPE, SDPA/composites, memory configs, program configs, and
  kernels.
- [x] Retried failures with adapted legal configurations and retained
  before/after correctness and timing evidence.
- [x] Verified the measured runtime has no Torch conversion, host operation,
  fallback, or avoidable layout boundary.
- [x] Passed non-aligned, page-boundary, multi-user, nonzero long-context,
  exact-capacity, cache-consumption, determinism, stress, and watcher gates.
- [x] Reconciled host latency with final `tt-perf-report` device windows and
  recorded roofline accounting.
- [x] Preserved and updated the 131072-token context contract.
- [x] Excluded out-of-scope multichip, full-model, CCL, vLLM, serving,
  sampling, LM-head, and qualitative-generation work.
