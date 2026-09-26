# AutoDebug: native paged SDPA decode

## Scope and status

Source-only investigation of `probe_optimized_attention.py --native-sdpa`,
2026-09-26. No hardware was used by this investigator and no implementation
source was changed. The parent agent serializes hardware experiments. The local
AutoDebug CLI had already failed because `bwrap` is unavailable; this fresh
subagent follows the inspection-only prompt and AutoFix hypothesis discipline.

**Confirmed execution-contract repair; actual-input native candidates pass.**
The native SDPA tree correction uses five full-stride DST tile slots while the
original FP32 compute configuration enables half-DST synchronization, which
provides four. A native-only `dst_full_sync_en=True` configuration removes the
catastrophic failure. The remaining failures below were measured with real
checkpoint weights but Gaussian layer inputs. They establish sensitivity and
hard API dtype constraints; they do not veto actual-text candidates.

Later parent-run, actual-text trials passed both attention types with native
full-sync SDPA and BFP8 caches. The artifact minima below were read directly;
these are cumulative policies, not isolated attribution of accuracy or speed to
one attention change. All decode checks in each listed run passed .995, with
repeated traced outputs equal and runtime decode audits enabled.

| Actual-text artifact | Prefill / decode steps | Minimum decoder-output PCC |
| --- | --- | --- |
| `actual_text_native_sdpa_layer0.json` | 4096 / 128 | 0.9951612261 |
| `actual_text_native_sdpa_layer5.json` | 4096 / 128 | 0.9983701394 |
| `actual_text_native_cache8_layer0.json` | 4096 / 128 | 0.9953817041 |
| `actual_text_native_cache8_stress_layer0.json` | 1025 / 512 | 0.9956262163 |
| `actual_text_native_cache8_gate4_layer5.json` | 4096 / 128 | 0.9958367922 |
| `actual_text_native_cache8_gate4_stress_layer5.json` | 1025 / 512 | 0.9969534618 |

The original conclusion to retain precise attention is superseded for those
actual-input policies. The BF16 query/output API constraints and full-DST
synchronization finding remain valid. The historical experiments and reasoning
are retained below with their input scope made explicit.

## Historical Gaussian-input observations

The JSON artifacts were read directly, and the headline log agrees with its
JSON. The cases are not a perfectly controlled length sweep: the long case also
enables the previously selected expert/shared/norm candidates.

| Artifact | Observed result |
| --- | --- |
| `attention_sdpa_layer0_33.json` | Sliding layer 0, positions 33–40: minimum decoder-output PCC 0.9968994944; pass. |
| `attention_sdpa_layer5_33.json` | Full layer 5, positions 33–40: minimum 0.9944092068 at position 39; fail. Other positions exceed 0.9997. |
| `headline_sdpa_layer0.json` | Sliding layer 0, positions 4096–4223: first PCC 0.2152067250, minimum -0.2732382988 at 4183; every decode check fails. Prefill PCC 0.9985633962 passes. Repeated traced output is equal. |
| `headline_sdpa_fullsync_layer0.json` | Identical headline policy with native full-DST synchronization: first PCC 0.9993369481; minimum 0.9715966558. Only four positions fail. Repeated traced output is equal. |
| `headline_sdpa_fullsync_layer5.json` | Full attention, full-DST synchronization, 4096/128: first PCC 0.9997478389; minimum 0.9871193474. Only two positions fail. Repeated traced output is equal. |

The exact headline command and exit code 1 are recorded in
`headline_attention_precision_commands.json`, first entry. The probe preserves
the cache update and output projection and replaces `attn.decode_sdpa` only.
At the time of these initial probes, the ordinary optimized/fused decoder used
its precise attention path.

## Lowered contract

`tests/config.json` supplies 16 Q heads, 8 sliding KV heads, sliding head
dimension 256, global KV heads 2, global head dimension 512, window 1024.
`tests/run_decoder.py:82,117-123` rounds allocation to 1024 tokens and creates
32-token BF16 pages with a reverse INT32 row-major page table. Thus the headline
cache is `[160,8,32,256]`, table `[1,160]`, capacity 5120; the short allocation
has 32 pages and capacity 1024.

`tt/fused_decoder.py:613-642` generates FP32 Q/K/V, applies RoPE, updates BF16
cache rows at the absolute device position, and calls the chosen attention.
`tests/probe_optimized_attention.py:71-106` uses tiled BF16 Q in DRAM, existing
BF16 K/V, scale 1, window 1024, grid 8x8, dynamic chunks (`k_chunk_size=0`),
and `exp_approx_mode=False`. The old branch passes `attn.compute` unchanged.
`tt/decode_attention.py:22-28` makes that HiFi4, exact math, FP32 DST, no packer
L1 accumulation, with omitted `dst_full_sync_en`; its default is **false** in
`ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp:19-24,41-48`.

The wrapper retains dynamic chunks in
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:140-178`.
The program factory assigns eight cores per sliding KV head (64 cores, B=1,
8 KV heads; `device/sdpa_decode_program_factory.cpp:194-208`). Its FP32 DST
limit/dynamic chunk cap is four tiles, or 128 tokens (`:386-389`). It passes
`dst_full_sync_en` directly to the compute descriptor (`:810-821`).

All three kernels share `device/kernels/rt_args_common.hpp:35-107`. Instantiating
that code gives:

| Position | Dynamic chunk | Window start | Absolute chunks | Local assignment / tree |
| --- | --- | --- | --- | --- |
| 33 | 64 tokens | 0 | `[0,1)` | Core 0 alone; no tree correction. |
| 127 | 128 tokens | 0 | `[0,1)` | Core 0 alone; no tree correction. |
| 128 | 128 tokens | 0 | `[0,2)` | Core 0 gets chunk 1, core 1 chunk 0; tree correction begins, before any sliding mask. |
| 4096 | 128 tokens | 3073 | `[24,33)` | Core 0 gets 32, cores 1–6 get 31–26, core 7 gets 24–25; all eight participate. |
| 4223 | 128 tokens | 3200 | `[25,33)` | One chunk each on eight cores; tree correction remains active. |

These calculations were checked with a host Python transcription of the shared
workload helper; they are static checks, not a hardware reproduction.

## Why the DST hypothesis ranks first

`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:545-573`
calls `correction_block` only when an active child contributes partial attention.
The short cases have no active child. In
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:746-763`,
the correction loads `prev_max`, `worker_max`, `prev_sum`, and `worker_sum` into
DST tile slots **0, 1, 3, 4**, using slot 2 for the new maximum. This requires
five full-stride tile slots in one acquisition.

This is not rescued by the SDPA half-tile optimization (16 Q heads select
16x32 CB tiles in `sdpa_decode_program_factory.cpp:445-453`). The datacopy
primitive still addresses each DST index using `DstTileShape::Tile32x32`
(`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h:275-277`).
That index is shifted by six before adding the DST bank base
(`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h:248-268`), even when
only two faces are copied. The correction SFPU itself names five regions at
offsets 0,32,64,96,128
(`tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h:233-269`).
Compact L1 tiles therefore do not make the fifth full-stride DST region local
to the acquired half.

The dedicated low-level regression supplies independent contract evidence:
`tt_metal/tt-llk/tests/python_tests/test_sfpu_sdpa.py:341-347` explicitly lists
“Dest configurations that can hold five tiles” and includes FP32 only with
`DestSync.Full`; FP32 with `DestSync.Half` is absent. The general host capacity
calculation halves capacity for half synchronization and again for FP32
(`compute_kernel_config.cpp:137-159`).

This explains the long/short discontinuity without assuming wrong page data,
an inherent BF16 instability, or a tracing failure. The fifth region can lie in
the other DST section and conflict with pack/math ownership. The exact numerical
manifestation is not derived from source alone. The completed full-sync control
supports this as the cause of the gross failure. Repeated deterministic results
in the old half-sync run did not establish that its ownership contract was safe.

## Competing hypotheses checked

- **Cache extent / rounded read overflow:** not supported here. At 4096 the
  dynamic reader reads tokens 3072–4223, comfortably inside 5120. At 4223 it
  reads 3200–4223. The reader uses absolute chunk indices and the reverse page
  table consistently (`reader_decode_all.cpp:294-344` and
  `dataflow_common.hpp:703-730,783-808`). Extra allocation is not the first
  useful intervention for this repro.
- **Window off-by-one:** the native interval `[pos+1-W,pos]`
  (`rt_args_common.hpp:48-60`) matches the precise attention mask
  (`tt/precise_attention.py:75-78`) and HF harness (`run_decoder.py:217,310`).
  At 4223 the window is already chunk aligned, yet the reported PCC is only
  0.4248811, so the partial sliding mask cannot explain all headline failures.
- **Q/K layout or page permutation:** no contradiction found. The short case
  exercises the same Q/head/page geometry, although it does not exercise the
  later cache pages. A same-input attention probe remains the clean confirmation.
- **BF16 numerical loss:** still relevant to the isolated short full-layer
  failure. FP32 DST does not make this kernel FP32 end-to-end: both intermediate
  and statistics CBs are hard-coded BF16 (`sdpa_decode_program_factory.cpp:439-440`).
  The full-layer position-39 miss occurs without tree correction, so the DST
  finding does **not** explain that separate failure. Earlier same-input
  `doc/fused_decoder/sdpa_exact_sliding.json` reports native attention PCC near
  0.9999 while a batched whole-layer case still fails; downstream sensitivity
  must be measured, not inferred from an aggregate attention PCC.

## Smallest verify/refute experiment

The parent added `--sdpa-full-sync` to the diagnostic probe, constructing a
separate native compute config with only `dst_full_sync_en=True` changed.
Keep query/cache dtypes, dimensions, absolute positions, reverse page table,
scale, chunks, grid, math fidelity, exp mode, and model policy identical.

1. Replay entry 0 of `headline_attention_precision_commands.json`, changing
   only the output/log names and adding `--sdpa-full-sync`. This tests the
   original observed catastrophic failure. Do not accept a performance result
   if its correctness gate fails.
2. If needed, compare positions 127 and 128 on the same Q/K/V cache: the default
   should first become vulnerable at 128, and full synchronization should remove
   that discontinuity. Window masking is inactive at both positions.
3. For localization, wrap `decode_sdpa` to read the same Q/K/V once in diagnostic
   mode and compare precise attention with native half-sync and full-sync,
   before output projection and routing. Compare using the same BF16-rounded Q
   as an additional control. Report per-head max error/PCC and nonfinite counts.
4. A secondary independent control is `max_cores_per_head_batch=1` with unchanged
   precision. It removes cross-core correction, but changes topology and is less
   specific than full synchronization. Do not combine it with the first control.

If full synchronization restores long-context attention, the smallest scoped
Python repair is a dedicated native-SDPA compute config. Adoption still needs
the .995 whole-layer gates, both attention types, headline positions, and the
existing shape/request/trace contracts. The short full-layer miss may continue;
retain the precise fallback until all required gates pass. A general kernel
repair belongs outside this stage's no-C++ scope.

## Follow-up control result and remaining precision localization

The parent completed both full-sync headline runs. The investigator read the
JSON artifacts directly. Sliding misses are positions 4124 (0.9909488352),
4130 (0.9796928271), 4149 (0.9930609120), and 4197 (0.9715966558). Full-attention
misses are 4101 (0.9913724412) and 4105 (0.9871193474). Thus the execution-contract
repair is useful but insufficient for the model gate; no timing improvement is
claimed or accepted here.

**FP32 Q is not a legal native control in this checkout.**
`sdpa_decode_device_operation.cpp:37-44` only admits BF16/BFP8/BFP4 inputs, and
`:412-418` specifically requires BF16 Q for this GQA geometry. Do not run an
FP32 native Q experiment expecting it to isolate accuracy; it should fail
validation before the kernel.

The following separate controls were proposed; their completed results are
recorded in the next section:

1. **Q rounding only:** preserve the existing precise attention and call it with
   `typecast(typecast(q, bfloat16), float32)`; keep K/V and every other operation
   unchanged. This answers whether the compulsory BF16 Q is sufficient to cause
   the failing whole-layer checks. It is legal because the precise path accepts
   FP32 and is still the operation being measured.
2. **Output rounding only:** preserve precise attention, then apply
   `typecast(typecast(result, bfloat16), float32)` before output projection.
   Native output is BF16 before the probe casts it back to FP32
   (`sdpa_decode_device_operation.cpp:501`). This distinguishes the last cast
   from native softmax/internal accumulation.
3. If neither boundary reproduces the failures, compare native full-sync with
   precise attention on identical **BF16 Q** at the recorded failing positions,
   capturing the attention result and projected residual before routing. The
   prior stage's `tests/probe_fused_precision.py:193-194` already implements the
   same-Q comparison. Use per-head errors and router top-k margins, not solely
   aggregate attention PCC. The remaining native path has mandatory BF16 QK,
   statistics, local accumulation and cross-core traffic while precise attention
   retains FP32 scores/probabilities/results.
4. A reduction-only control may set `max_cores_per_head_batch=1` while retaining
   the now-required full-sync setting and dynamic chunks. It changes how BF16
   partials are reduced, so it can test a narrower residual kernel hypothesis;
   it must independently pass the complete gate before being considered a
   usable optimization. It is not a precision guarantee.

The old short probe's small Q-rounding error did not predict the long-context
gate; the completed controls below supersede that prioritization.

## Historical precision controls: native format constraints on Gaussian inputs

The six JSON artifacts and their exact commands in
`headline_attention_boundary_commands.json` were read directly. All use real
checkpoint weights, Gaussian inputs, 4096-token prefill and 128 decode steps
with deterministic repeated traced
outputs. The precision flags operate on **precise attention**, with
`native_sdpa=False`, so they isolate explicit precision boundaries from native
kernel execution. `tests/probe_optimized_attention.py:55-68` wraps the query
or result with BF16 then FP32 casts; it leaves attention arithmetic and cache
unchanged for these two controls.

| Artifact | Minimum whole-layer PCC | Verdict / failed positions |
| --- | --- | --- |
| `headline_precision_query_round_layer0.json` | 0.9919267732 | Fail at 4149. |
| `headline_precision_output_round_layer0.json` | 0.9956135082 | Pass all checks. |
| `headline_precision_query_round_layer5.json` | 0.9963523785 | Pass all checks. |
| `headline_precision_output_round_layer5.json` | 0.9859702036 | Fail at 4105. |
| `headline_precision_cache8_layer0.json` | 0.9684969961 | Fail at 4114, 4124, 4131, 4138, 4147, 4164, 4197. |
| `headline_precision_cache8_layer5.json` | 0.9850933609 | Fail at 4128, 4154, 4183, 4205. |

Precision-policy ledger: QKV uses 16 lane partitions, two terms, grid 8x8,
K-block width 11, HiFi4. Expert gate/up is BFP8 and down is BFP4, LoFi, grid
11x4/K11. Shared projection is BFP8/LoFi with one DRAM reader. Active prefill
uses BFP8/HiFi2/L1 on grid 11x4/K11. Norm placement is `input_common` for
layer 0 and `post_common` for layer 5. Cache remains BF16 except the explicit
BFP8-cache control; page size, reverse mapping, allocation extent and updates
are the original harness contract. This is a **later cumulative policy** than
the earlier native/full-sync pair, which used broadcast QKV and BFP8 expert
down. Therefore these results are not a claim that rounding alone explains
every earlier native miss. The parent validates the unmodified cumulative
baseline separately before final default selection.

The format barriers are concrete:

- **Sliding:** rounding only Q already misses .995 at position 4149. Native
  GQA requires BF16 Q (`sdpa_decode_device_operation.cpp:412-418`), so higher
  compute fidelity/full synchronization cannot restore the input bits removed
  before the kernel. The passing output-only control excludes output rounding
  alone as a sufficient cause for this sliding-policy failure.
- **Full attention:** rounding only the final attention result already misses
  .995 at position 4105; query-only rounding passes. The native output spec
  unconditionally uses `input.dtype()` (`sdpa_decode_device_operation.cpp:491-501`),
  and GQA Q is BF16, so output is BF16. The probe's subsequent FP32 cast cannot
  restore the lost result bits. This is the same failing absolute position as
  the worst earlier full-sync native check, but the policy difference above
  prevents claiming an exact numerical equivalence.
- **Cache:** BFP8 fails while attention arithmetic remains the precise path.
  These are direct model precision-policy failures, not evidence that BFP8
  native-cache kernels are intrinsically incorrect.

There is **no supported FP32 Q/output dtype override** for this native
operation within `optimized_decoder.py`. Its public signature
has no output-dtype argument (`sdpa_decode/sdpa_decode.hpp:29-44`), and
`SDPAProgramConfig` only exposes grid/chunk/exp/core-count fields
(`transformer/sdpa_config.hpp:14-20`). Memory layout, grid, chunks, math fidelity,
and full-DST synchronization cannot change the required Q/output dtypes.
Splitting GQA into MQA calls does not unlock FP32: the common input validator
rejects FP32 for every input before the GQA-specific check (`:37-44`).
Scaling by powers of two does not add BF16 mantissa bits; casting back after
the native call likewise does not recover discarded information. Compensating
attention errors with another approximation would be a new unverified
algorithm, not a dtype configuration fix.

These controls justified deferring native adoption until representative
actual-text inputs were available. They did not establish a universal model
precision barrier. The subsequent actual-input gates listed above pass, so the
Gaussian failures do not veto those candidates. The public API constraints
remain established without further hardware work; final policy claims still
require the matching cumulative-policy checks.
