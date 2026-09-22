# Qwen3.8-27B optimized decoder

Stage: optimized-decoder, starting from fused checkpoint `ad43d1388fd`.
Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Scope is one real decoder on one Blackhole device; no multichip, full-model,
generator, or serving implementation is introduced.

**Complete: runtime gates passed and independent [stage review](stage_review.md) returned clean-pass.**

## Runtime contract

`tt/optimized_decoder.py` defines an independent `OptimizedDecoder`. Calling
`from_state_dict` without an experimental policy selects `DEFAULT_POLICY`,
also recorded in [final_policy.json](final_policy.json). Tests fail immediately
if either `FunctionalDecoder` or `FusedDecoder` is selected as a fallback.
The runner's explicit `--baseline` switch selects the frozen fused control.

Inputs and outputs retain logical BF16 TILE `[B,S,5120]`; decode uses
`[B,1,5120]`. The caller owns request state, 32-token page mappings, positions,
RoPE and persistent trace inputs. It restores state after warmup/capture and
refreshes those inputs before replay. Only setup uploads weights/constants.
The measured forward rejects Torch operations and host tensor conversions.

Prefill accepts arbitrary valid logical lengths and continuations. Internal
chunks are at most2048 tokens. Full attention processes an initial partial
page one token at a time, then uses paged prefill. SDPA block selection respects
both absolute prefix alignment and the caller's actual mapped capacity.
Decode K blocks divide mapped capacity, bounding rounded reads for every valid
device position without reading that position on the host. No extra virtual
pages or public chunk-alignment restriction is imposed.

Full K/V is BFP8; recurrence stays FP32 and convolution history stays BF16
ROW_MAJOR `[B,3,10240]`. Decode norms, residuals and MLP use packed internal
rows `[1,1,B,W]` and L1 width sharding. Public B>1 tensors are unpacked on
device. For B>=8 the wide attention projection is unpacked into DRAM to leave
L1 space for native recurrent-kernel circular buffers. B1 output remains L1
width-sharded; callers may pass it directly into another optimized decoder.

## Chosen kernels and precision

All five dominant projection roles use real BFP4 weights, LoFi, BF16
activations/output, FP32 destination accumulation and packer L1 accumulation.
Exact norm/recurrent policies retain the fused decoder's numerical safeguards.
Weights have interleaved prefill and DRAM-sharded decode copies; padding is
introduced only in bank storage and never becomes a logical model channel.

| Role | Decode input storage cores | Readers per bank | K block, tiles |
| --- | ---: | ---: | ---: |
| Packed full Q/K/V/gate or linear QKV/Z/B/A | 80, rectangular10x8 | 3 | 2 |
| Attention output | 48 | 3 | 4 |
| MLP gate and up, separate | 80, rectangular10x8 | 3 | 2 |
| MLP down | 32 | 3 | 17 |

Eight DRAM banks therefore supply24 reader/compute workers. Input storage
cores are a separate geometry. The native DRAM-sharded op chooses its output
subblocks; its Python config does not expose independent output block/subblock
overrides. FP32 accumulation limits the destination subblock to four tiles.

The **block2** attention/gate/up choices are deliberate. The precision-locked
search in [projection_results.csv](projection_results.csv) includes storage
cores5/10/20/40/80 for K5120,6/12/24/48/96 for K6144,8/16/32/64 for K17408,
one/two/three readers, and legal divisors from
2/3/4/5/6/7/8/10/12/14/16/17/24/32/34/48/68. Smaller storage grids enable
larger K blocks; they were measured at the selected BFP4/LoFi precision.
Their latency or exact L1/runtime failures are retained. Down's three-reader
32-core geometry beats its isolated two-reader winner in the whole decoder.

Prefill normally uses DRAM-interleaved activations. Large projections use
native `minimal_matmul` with M4/K8/N16 blocks,1x4 output subblocks and11x10
grid. Output/down use it from128 tokens; all projections use it from512.
Inputs totaling at most32 rows use DRAM-sharded projections. Legal explicit
2D multicast configs were tried with grids8x8/11x10/8x4, output blocks and
chunks512/1024/2048/4096; adapted legal candidates lose to the selected kernel.

Full attention retains native head creation, Q/K normalization, partial RoPE,
fused paged updates and SDPA. Prefill starts from Q128/K128, reducing internally
when alignment/capacity requires it. Decode starts from K64 for larger even
mapped capacities and K32 otherwise. Batch-1 decode with fewer than 16 mapped
pages uses an 8x2 SDPA grid; larger contexts and batches retain 11x10.
The smaller grid is faster at short context and slower at 2048/4097, as measured
in `sdpagrid*matrix.json`. Selection depends only on static mapped capacity. Linear attention retains native causal
conv/SiLU, gated delta prep/scan and gated RMS norm. There are no experts,
collectives, LM head or sampling operations in this decoder.

## Measured results

Same harness, B1,128-token real recorded HF inputs; median of five warmed prefills and30 traced decodes. State restore and correctness readback are outside timed windows.

| Layer kind | Fused prefill ms | Optimized prefill ms | Fused traced decode ms | Optimized traced decode ms | Prefill PCC before / after | Decode PCC before / after |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| linear_attention | 3.0881 | 2.2606 | 2.4954 | 0.8221 | 0.9999977 / 0.9997711 | 0.9999996 / 0.9998332 |
| full_attention | 2.7066 | 1.7832 | 2.2624 | 0.6618 | 0.9999910 / 0.9999005 | 0.9999982 / 0.9999039 |

PCC differences are the measured BFP4/BFP8 quantization cost; all remain above0.995. Final default source and effective policy are recorded per run. The 68-case suite reaches minimum per-user prefill0.9995203 and decode0.9993832. Changed-input replay reaches minimum 0.9992942. See [stress_summary.json](stress_summary.json).

Profiler-instrumented accounting is separate from the normal medians above. The following three quantities come from each same `final_l0.json` or `selected_l3.json` / signposted CSV run. Stored-tile bytes include bank padding; this bandwidth bound excludes activation traffic and writes.

| Kind | Read roofline ms | Kernel sum ms | Op gaps ms | Same-run instrumented host ms |
| --- | ---: | ---: | ---: | ---: |
| linear_attention | 0.4391 | 0.7887 | 0.0371 | 0.8495 |
| full_attention | 0.4206 | 0.6257 | 0.0365 | 0.6857 |

The remaining device time includes native attention/recurrence, norms, elementwise work and required layout boundaries. The reports show44 linear/38 full device ops (exact counts in CSV), zero host ops, and actual BF16 x BFP4 LoFi projection rows. [final_matmul_rows.csv](final_matmul_rows.csv) records role shares and configs. The report counts8 matmul cores although native configs use24 reader workers; its resulting utilization above100% is an estimator limitation. `MinimalMatmulConfig` is supplied explicitly even where the advice parser says no `program_config`.

Normal traced prefill is also measured: 1.9110 ms linear and 1.6715 ms full at 128. The caller can capture the same forward to reduce eager dispatch gaps. At2048 tokens, default warmed prefill is11.3059/8.2253 ms (linear/full), with traced-prefill10.9836/7.9993 ms. At the non-aligned length 4097, warmed prefill is 24.0994/19.8779 ms and traced decode 0.8216/0.7051 ms (`final_4097_l{0,3}.json`).

L1-prefill advice was tested at128 and2048 on both kinds (`advice_matrix.json`). Linear worsens2.1607→2.1765 and11.3059→11.6048 ms; full2048 worsens8.2253→8.6750 ms. Full128 eager shows a small1.7932→1.7554 ms change, but traced prefill worsens1.6792→1.7026 ms. The coherent DRAM policy is retained; L1 does not improve decode and increases movement/large-prefill cost.

Compact reports: `tracy/{baseline_l0,baseline_l3,final_l0,final_l3,final_long_l0,final_long_l3,selected_l3}/{prefill,decode}_perf_report.{txt,csv}`. Raw operation rows are retained as `ops_perf_results.csv.gz`; original Tracy/device-event files are archived at paths in `tracy/raw_archive.json`.

## Search evidence and rejected alternatives

The operation-topology audit was recorded before tuning in
[work_log.md](work_log.md). [candidate_results.csv](candidate_results.csv)
contains whole-decoder policies, timings and PCC; projection tables separately
verify quantized matmul arithmetic. Only whole-decoder real-HF checks decide
precision acceptance. Synthetic BFP4 PCC0.99345 was a diagnostic, not a veto;
real checkpoint activations pass above0.9997 in the representative short checks.

| Opportunity | Result and evidence |
| --- | --- |
| Lower projection precision/fidelity | Real-weight BFP4/LoFi wins. Per-role HiFi2, BFP8 attention/gate and reduced activation precision pass but lose (`hifi2_*`, `bfp8_lofi_*`, `activation8_*`). |
| Pack attention | Tuned legal separate projections lose: full0.837/linear1.031 ms versus packed0.671/0.833 ms (`closure_split_attention_*`, `closure_rectangular_*`). |
| Pack gate/up | Tuned separate projections with SiLU fused into multiplication beat packed gate/up0.677/0.839 ms (`closure_packed_rect_*`). Native fused SwiGLU also passes but loses1.299/1.456 ms (`tuned_minimal_l*`). |
| Preserve residual layout | Rectangular working shards and carry through output projection remove avoidable interleaved boundaries (`movement_*`). |
| Large prefill | `prefill_matrix`, `closure_matrix`, `attention_matrix` and `phase_matrix` measure real4097-token inputs. First2D1024 L1 failure was repaired with output-block height1, then rejected on measured latency. |
| DRAM-sharded short prefill | Native M==1 tile limit was adapted with32-row chunks. At128 it passes but loses3.862/3.495 ms; at31 it wins1.365/1.391 ms (`finalprobe_*`). |
| Prefill norms | Sharded norm and coherent2D candidates are legal but slower than selected interleaved prefill (`shortpref_*`). |
| Dispatch gaps | `phase_*` includes five correct traced-prefill repetitions; caller-owned prefill traces reduce gaps. Decode is always traced. |
| Convolution untilize | A ROW_MAJOR boundary is required by the native convolution. The fused stage's adapted untilize epilogue tests failed PCC; optimized native convolution remains. The selected DRAM-sharded kernel cannot fuse untilization: native validation permits `untilize_out` only for Mcast1D (`matmul_device_operation.cpp:435–460`). The adapted Mcast1D control failed PCC; this device boundary is required. |

Intermediate larger-SDPA candidates whose rounded reads exceeded their logical
page tables are **unsafe**, despite passing masked-output PCC. They are excluded
from correct-candidate comparisons. The final capacity-aware implementation
repairs this without changing caller allocation; see
[AUTODEBUG_continuation.md](AUTODEBUG_continuation.md).

## Validation and reproduction

Existing `python_env` and compiled runtime were used on Blackhole device3,
11x10 worker grid, eight DRAM banks, firmware19.8.0/KMD2.8.0. Device commands
are serialized. Watcher and profiler runs are separate. No build is required
for these Python/tests/docs changes; no dependencies were installed.

Record CPU HF activation fixtures with `tests/record_decoder_activations.py`,
then run `bash models/autoports/qwen_qwen3_8_27b/tests/validate_optimized_decoder.sh`
from the repository root, as recorded in [commands.log](commands.log).
[activation_provenance.json](activation_provenance.json) records exact token
IDs, prompts, revision, cache paths and tensor hashes. No tensors are committed.

The suite covers both layer kinds and batches1/2/3/8/16/32, exact output shapes,
per-user PCC, tile/page/chunk boundaries and short reuse after long inputs.
Capacity checks compare complete optimized/fused outputs at262143/262144 and
traced decode at the last valid position. Their long input repeats recorded HF
activations; this is a capacity/parity stress, not a new full-context HF oracle.
Direct HF checks use actual recorded inputs through4097 tokens.

Saved failures and their fixes are part of the evidence: reader storage padding,
batch physical-row packing, batch32 L1 pressure, and SDPA continuation/read
bounds. See the `AUTODEBUG_*` and `AUTOFIX_*` reports and work log. No native
C++ changes, hardware hangs or resets were required.

The complete [optimization checklist](optimization_checklist.md),
[performance summary](performance_summary.json), [watcher summary](watcher_summary.json)
and [stress summary](stress_summary.json) link exact evidence. The final default
suite is `selected_stress.xml` / `selected_stress/*.json` (12 groups, 68 cases).

Full-context checks passed for both kinds: at 262144 tokens optimized/fused
prefill PCC is 0.9997951 linear and 0.9996872 full. Last-position traced decode
PCC is 0.9998990/0.9999097 with bitwise-equal repeats. These are parity checks
on repeated recorded inputs, not direct long-context HF PCC. The largest full
probe recorded 1,077,423,424 allocated bytes per bank out of 4,272,341,376
(with both decoders and comparison buffers live, not a peak allocator trace).
No capability reduction was needed. See the updated [context contract](../context_contract.json).

## Final candidate closure

Headline numbers come from `verified_benchmark_l{0,3}.json`, with the final
default policy. Full-attention default/control/default traced decode medians
are **0.661752 / 0.663265 / 0.661923 ms** (`verified_matrix.json`). The repeated
final default beats the strongest prior correct control, while warmed prefill
is 1.7832/1.7377 versus 2.1397 ms. An older isolated best value, 0.661625 ms,
is within the observed timing spread; it was rerun as that explicit control.
The selected policy reproduces its small-grid candidate within about 1 us;
these sub-percent differences should not be read as precision beyond the run
spread. The material gain against the fused control is about 3.42x full and
3.04x linear traced decode.

The earlier wide-grid final candidate lost about 1 us to the prior control;
`comparison_matrix.json` records the ABBA evidence. The SDPA grid change above
closed that loss. Smaller grids lose at long context, so the large-capacity
path is unchanged. `selected_l3` is the final short-full profile; `final_l0`
and the long profiles remain applicable to unchanged branches. Capacity
probes also exercise unchanged branches. Final watcher short/full coverage
is `watcher_selected_l3.json`; prior watcher B32 and long-tail runs cover the
unchanged boundaries.

A blocking trace API trial did not improve the host remainder consistently:
linear 0.821093 versus 0.821803 ms; full 0.664855 versus 0.663571 ms under the
pre-grid policy. The original nonblocking replay plus synchronization is
retained. The remaining first-norm trace gap is native dispatch/synchronization
after the measured carry-layout and wait-API attempts; no Python forward or
host tensor conversion runs inside the trace.

The initial `selected_matrix` benchmark accidentally merged the sweep's old
BASE policy, exposed by its saved effective policy. Its approximately
1.124/0.959 ms results are historical harness-error evidence, not the final
path. `default_policy: true` now reaches the constructor default explicitly;
`verified_matrix` records the corrected measurement. No such fallback occurs
in the delivered pytest, profile, or watcher paths.

Source snapshots under `sources/<sha256>.py.txt` are immutable experiment
text, not importable runtime code. Large raw profiles and watcher inspector
data stay in the external archive mapped by `tracy/raw_archive.json`;
compact reports, raw-operation CSV gzip files, logs and policies are retained.
Repository hooks normalized generated text whitespace/EOF; exact originals
are preserved in the archive manifest. The oversized candidate profiler log
is losslessly compressed as `candidate_fixed_l3.log.gz`.
No decoder optimization item is deferred to another pipeline stage.
