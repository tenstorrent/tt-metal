# AutoDebug: functional-decoder stage review

Date: 2026-07-28 UTC  
Checkout: `b9e6c242a34011e3daeebab9207fbb5b79750f39`  
Scope: source and existing artifacts only; no TT device command was run.

## Verdict

The stage should not pass yet.

Two source paths are correctness bugs for inputs the public API accepts:

1. prefill above 32,768 tokens always reaches non-chunked SDPA, despite the
   canonical Gemma4 implementation recording that this op silently returns wrong
   results above that boundary;
2. a non-tile-aligned prefill written into a bounded modulo cache writes the
   padded tail as real cache rows and can wrap those rows over live tokens.

The trace suite proves capture/replay only for unchanged payloads. It does not
prove the documented stable-buffer update contract, heterogeneous positions,
permuted page tables, or full-attention shared-HMA replay. The blanket
HiFi4/FP32 policy, batch-1-only prefill, missing synthetic weight-stat fixture,
`TT_METAL_WATCHER=60` rather than `10`, and weak/stale advertised-context
artifacts are stage-gate failures, not additional demonstrated numerical bugs.

The decode DRAM-to-L1 QKV promotion is very likely obsolete: fix commit
`7aa26e4b1f2` is an ancestor of this checkout and is exactly the kernel-side
Blackhole DRAM-interleaved fix that superseded whole-tensor L1 promotion. Remove
it only after the focused A/B below.

## 1. Long prefill uses a known-wrong SDPA path

**Classification: required correctness fix; invalidates the long-prefill
capability evidence.**

Verified facts:

- `prefill_forward` retains the padded sequence and calls
  `_attention_prefill` without any long-sequence dispatch
  (`tt/functional_decoder.py:303-324`).
- `_attention_prefill` always calls
  `ttnn.transformer.scaled_dot_product_attention`
  (`tt/functional_decoder.py:460-469`), including at 65,536 and 262,144.
- Canonical Gemma4 defines `PREFILL_SDPA_MAX_SEQ=32768` and states that the
  non-chunked op silently returns wrong results above it
  (`models/demos/gemma4/tt/attention/operations.py:25-40`).
- Canonical prefill sends long sliding layers through overlapping windowed
  slices and long full layers through paged
  `chunked_scaled_dot_product_attention`
  (`models/demos/gemma4/tt/attention/prefill.py:105-130`;
  `operations.py:210-354`).
- The autoport capacity test checks only shape and finiteness of the last token
  (`tests/test_functional_decoder.py:862-903`). Silent wrong finite output
  therefore generated all `prefill_capacity_*_{65536,131072,262144}.json`
  artifacts without being detected.

Smallest intervention boundary:

- Change attention prefill dispatch only. Reuse or adapt the canonical
  `chunked_prefill_sdpa` and `chunked_prefill_sdpa_sliding` helpers.
- Full attention must use the already-filled paged cache and the correct
  `page_table` row. Sliding attention must use overlapping in-memory Q/K/V
  slices; the paged chunked op is causal-only and cannot express the sliding
  mask.
- Keep the public logical-length padding/slicing contract. Do not treat the
  existing `chunk_page_table` argument as sufficient: the implementation has no
  Q offset/cross-chunk attention contract.

Focused tests:

1. Add a host-only dispatch test with fake tensors at lengths `32768` and
   `32800`; assert non-chunked versus full/sliding chunked call selection.
2. Add an attention-only device test, not a 26B whole-layer test. At `32800`,
   compare selected Q rows (boundary and final rows) against a streaming Torch
   reference so no `S x S` host mask is allocated. Test both layer kinds and a
   permuted page table.
3. Force the threshold low (for example 128) and compare chunked and
   non-chunked outputs at 256 as a cheap regression; retain one real `>32768`
   case for the actual kernel cliff.

Proposed commands after those tests exist:

```bash
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k prefill_attention_dispatch

GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k long_prefill_attention_correctness
```

## 2. Modulo cache fill writes padded rows over valid history

**Classification: conditional required correctness fix.** It triggers when
`cache_position_modulo` is supplied and logical prefill is not tile-aligned,
especially once the prompt crosses the modulo capacity.

Verified facts:

- `prefill_forward` records the logical length, pads hidden states and RoPE
  tensors to 32, then loses the logical length at the attention/cache-fill
  boundary (`tt/functional_decoder.py:303-324`).
- `_attention_prefill` forwards the modulo value while filling all padded K/V
  rows (`tt/functional_decoder.py:452-458`).
- The fill kernel wraps at tile granularity:
  `seq_tile_id %= cache_position_modulo / 32`
  (`paged_fill_cache_device_operation_types.hpp:19-25`;
  `writer_fill_cache_interleaved.cpp:118-180`).
- A concrete host ledger for logical length 1025, physical length 1056, and
  modulo 1024 maps padding positions 1025..1055 to slots 1..31. All 31 slots
  are still part of the live 1024-token history.
- Canonical Gemma4 explicitly caps bounded-cache fill using `valid_seq_len`
  because prompt padding otherwise wraps over real history
  (`models/demos/gemma4/tt/attention/prefill.py:77-100`). Its block-rounded
  slice is useful evidence of the intended contract, but the autoport's
  minimal tile padding exposes a partial-tile case that needs exact handling.
- Existing boundary tests neither pass `cache_position_modulo` nor decode from
  the cache (`tests/test_functional_decoder.py:573-669`), so their 1025-token
  PCC does not cover this bug.

Smallest intervention boundary:

- Preserve `logical_seq_len` as `valid_seq_len` through `_attention_prefill`.
- A simple slice to the preceding full tile is insufficient because it drops
  the final 1..31 real tokens. Either:
  1. extend `paged_fill_cache` with an exact valid-row count/read-modify-write
     contract, or
  2. fill the aligned prefix and write the real tail with
     `paged_update_cache`, preserving the other rows in the wrapped tile.
- Until one exists, reject non-tile-aligned modulo prefill explicitly rather
  than silently corrupting cache state. That rejection is only an interim
  guard; the functional-decoder contract ultimately requires arbitrary logical
  lengths.

Focused test design:

- Use logical `1025`, modulo/window `1024`, a permuted bounded page table, and
  sentinel/nonzero cache contents. Run prefill, then decode position 1025 and
  compare with HF. Independently read the cache and verify every live logical
  position, not only the decoder output.
- Add cases `1023,1024,1025,1055,1056` to distinguish no-wrap, aligned wrap,
  and partial-tile wrap.

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k bounded_modulo_prefill_tail
```

## 3. Trace evidence does not exercise changing stable buffers

**Classification: required evidence/test fix; no trace implementation bug is
yet demonstrated.**

Verified facts:

- Both trace tests capture `decode_forward(**decode_args)` and replay without
  copying new contents into hidden, RoPE, current-position, page-table, or
  cache buffers (`tests/test_functional_decoder.py:335-381,493-567`).
- Every batch row uses position 32 (`:404,497-500`) and batch page tables are
  sequential (`:465-471`).
- Full traced cases use the natural `[blocks,2,128,512]` cache
  (`:407-475`); the shared physical `[blocks,8,64,256]` HMA view is covered
  only by the eager real-weight test (`:181-279`).
- The supplied “nonzero cache” concern is partly refuted: the single sliding
  trace test prefills a real nonzero cache (`:321-334`) and full batch trace
  cases prefill real prefixes (`:423-486`). What is missing is changed cache
  state across replays; sliding batch 1/32 begins with zero history
  (`:440-445,472-476`).
- The advertised-context test permutes its page table, but uses zero history,
  replays unchanged inputs, and checks only finiteness/repeatability
  (`:699-799`).

Smallest test-only intervention:

- Allocate stable device buffers once, capture once, and before each replay use
  `ttnn.copy_host_to_device_tensor` to update hidden, cos, sin, current
  positions, page table, and seeded cache contents in place.
- Run payload sequence A/B/A. Compare each replay to a fresh eager/HF result,
  assert A and B differ, and assert the final A reproduces the first A.
- The minimum matrix is:
  1. sliding batch 32 with heterogeneous positions and independently permuted
     page-table rows;
  2. full batch 32 using the shared-HMA physical cache;
  3. nonzero/sentinel cache history with addressed and non-addressed pages
     distinguishable.

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k trace_mutable_stable_buffers
```

## 4. HiFi4/FP32 was generalized from one marginal failure

**Classification: required precision-isolation evidence and likely narrowing;
not proof that every high-fidelity use is wrong.**

Verified facts:

- The only recorded contrast is default sliding decode PCC `0.993209` versus
  blanket high-fidelity PCC `0.995073`
  (`work_log.md:21-22`; `pcc_layer0_sliding_attention_shared1.json`).
- One `correctness_compute_config` sets HiFi4, exact math, and FP32 destination
  (`tt/functional_decoder.py:819-827`).
- It is used by all RMS norms, prefill/decode SDPA, attention output projection,
  dense MLP projections, router projection/softmax, and all expert matmuls
  (`:407-414,460-478,566-592,612-695,727-759`). QKV projection and pointwise
  ops remain at defaults.
- No component PCC, per-op contrast, or separate HiFi4-versus-FP32-destination
  result is recorded.

Efficient adaptive isolation matrix:

| Run | Norms | SDPA | dense/o-proj | router | experts |
| --- | --- | --- | --- | --- | --- |
| D | default | default | default | default | default |
| H | high | high | high | high | high |
| A | high | high | default | default | default |
| B | default | default | high | high | high |

Use measured PCC deltas, not only pass/fail. Recursively split every half with
a meaningful positive delta (`norm` vs `SDPA`; `dense/o-proj` vs
`router+experts`; then `router` vs `experts`). This is at most eight full
decoder runs and remains valid if multiple groups contribute. For each
identified group, run the 2x2:

`{default fidelity, HiFi4} x {FP32 destination off,on}`.

Implement the matrix as a test-only per-group compute-policy injection; do not
add five production environment branches. Record component outputs after
attention, dense MLP, router, and MoE to localize the first material divergence.
Run this before changing the dense expert topology, otherwise it is no longer an
isolation of the recorded baseline.

## 5. Decode QKV DRAM-to-L1 workaround is probably obsolete

**Classification: probable cleanup/performance fix, hardware A/B required before
removal.**

Verified facts:

- The autoport unconditionally copies the fused BF16 QKV tensor from DRAM to L1
  before `nlp_create_qkv_heads_decode`
  (`tt/functional_decoder.py:493-508`).
- `git merge-base --is-ancestor 7aa26e4b1f2 HEAD` succeeds.
- Commit `7aa26e4b1f2` is titled
  `nlp_create_qkv_heads_decode: fix wrong reads from DRAM input on Blackhole`.
  It adds a kernel-side aligned DRAM path and explicitly states that it
  supersedes whole-tensor L1 promotion.
- The current reader retains the aligned path from that commit
  (`reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:32-120`).
- The generic op suite now covers BH+DRAM, but its interleaved parameters use
  head dimensions 64/96/128, not Gemma4's 256/512
  (`test_nlp_create_qkv_heads_decode.py:74-103`).

Focused experiment:

1. Add DRAM-interleaved op cases `(q=16,kv=8,hd=256,b=32)` and
   `(q=16,kv=2,hd=512,b=32)`.
2. On a frozen baseline, A/B only the line-497 promotion and run real-weight
   traced decode for both layer kinds at batch 32.
3. Require the same PCC and inspect L1/latency before deleting the copy and stale
   issue comment.

```bash
pytest -q tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads_decode.py \
  -k 'test_create_head_interleaved and DRAM and b32'

GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k 'traced_decode_batch_contract and batch32'
```

## 6. Acceptance evidence gaps

**Classification: required stage-gate work; these facts do not independently
prove model math is wrong.**

### Synthetic weights and statistics

Every correctness test calls `_load_layer_state` and skips without a local
snapshot or opt-in range download (`tests/test_functional_decoder.py:39-122`).
There is no committed tensor-stat file or deterministic synthetic state fixture.
The functional-decoder contract requires name, shape, dtype, mean, and std from
real weights, synthetic weights derived from them for normal CI, plus one final
real-weight run.

Smallest boundary: add one stats JSON for layer kinds 0 and 5, a deterministic
full-shape state generator, synthetic PCC tests in the normal suite, and retain
the existing opt-in real-weight gates.

### Batch greater than one prefill

The API explicitly documents only `[1,1,S,H]` and `context_contract.json`
records prefill batch `[1]`. This is not a hidden violation of the currently
written API, but it fails the stage requirement for at least one batch>1
prefill correctness test. Several internals hard-code batch 1, notably router
reshape and MoE shapes (`tt/functional_decoder.py:637-662,698-760`).

Smallest functional implementation is a device-only per-user wrapper around a
single-user prefill core, with distinct `user_id`/page-table rows, followed by
concatenation. Validate batch 2 first and batch 32 if memory permits; if batch 32
does not fit, record the hard limit.

### Watcher

The preserved watcher log exists and its SHA-256 matches the work log, but the
recorded command uses `TT_METAL_WATCHER=60` (`README.md:110-113`), while the
stage gate requires `TT_METAL_WATCHER=10`. Rerun after correctness changes with
a unique logs directory:

```bash
GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=/tmp/gemma4_fd_watcher_10 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q -rA models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k traced_decode_batch_contract
```

### Advertised context

The 262,144 prefill artifacts assert only a finite last token, which cannot
detect the known silent SDPA error. The advertised decode artifacts use
zero-filled history and unchanged replay payloads. Also, the autoport is
untracked, artifacts contain no implementation/test SHA, and current
implementation/test mtimes postdate some context artifacts. Treat the current
context claim as capacity evidence only.

After fixes, rerun context evidence with source SHA, exact command, TTNN build
SHA, hardware ID, correctness metric, and artifact hash. For long prefill use
the sampled streaming reference from finding 1; for full-context decode seed
addressable history so page-table/cache routing affects the result.

## 7. SDPA config and dense expert-path contract

### Explicit decode SDPA config

**Classification: explicit config is justified; derivation/documentation needs
cleanup, not automatic removal.**

Canonical Gemma4 explains that `program_config=None` may use more than the
maximum 64 reduction cores per head on Blackhole and therefore always supplies
an `SDPAProgramConfig` (`models/demos/gemma4/tt/attention/decode.py:229-250`).
The autoport's explicit config is consequently a functional exception allowed
by the stage contract.

The concern is narrower:

- `_make_sdpa_program_config` hard-caps every layer at 8x4 and q/k chunks 32
  (`tt/functional_decoder.py:830-842`);
- canonical uses the device grid for sliding, 8x4 for head-dim 512, and
  k-chunk 64;
- `README.md:86-89` claims there are no per-core forward grids and describes
  them as workload-derived, which is inaccurate.

Add a source/unit test that the derived grid fits the actual device grid and the
reduction-core limit. A/B the canonical sliding/full choices for correctness and
L1. Keep the safest required config in functional code; defer tuning to
optimization.

### Dense all-expert MoE

**Classification: math appears correct, but it is a required functional-stage
topology correction.**

The autoport repeats every token 128 times and uses dense batched matmuls in
prefill and decode (`tt/functional_decoder.py:664-760`). Routing zeros make the
result mathematically equivalent in existing real-weight PCC tests, so this is
not a demonstrated model-math bug. It nevertheless contradicts the
functional-decoder default sparse-expert contract and already caused an
11.81-GB allocation before 1,024-token chunking (`work_log.md:18-30`).

Use the canonical Gemma4/GPT-OSS shape:

- decode: routing weights as top-8 sparsity and `ttnn.sparse_matmul` for
  gate/up/down (`models/demos/gemma4/tt/experts/decode.py:81-159`);
- prefill: canonical Gemma4 currently uses sparse matmul with all experts active,
  then applies routing weights (`experts/prefill.py:26-141`). Thus “all experts”
  is not itself the defect; dense repeat/matmul is.

Keep router semantics and validate router + selected experts end-to-end against
HF before deleting the dense reference path from the experiment branch.

## Experiment ordering and parallelism

| Track | Can run independently | Dependency / merge boundary |
| --- | --- | --- |
| Long-prefill dispatch/reference | Yes | Shares `_attention_prefill` with modulo fix; merge deliberately |
| Modulo partial-tile cache ledger/test | Yes | Shares `_attention_prefill` with long-prefill fix |
| Mutable stable-buffer trace tests | Yes | Test-only; independent of attention implementation |
| QKV DRAM-vs-L1 A/B | Yes | Remove workaround only after op + decoder A/B |
| Synthetic stats/CI fixture | Yes | Regenerate final numbers after implementation settles |
| Precision isolation | Yes on frozen baseline | Must precede sparse-MoE replacement |
| Sparse-MoE topology | Yes after baseline frozen | Rerun precision minimum afterward |
| Batch>1 prefill | Mostly | Likely overlaps MoE shapes and cache-fill API |
| SDPA grid A/B | Yes | Keep separate from precision A/B to avoid confounding |
| Watcher/context/profiler evidence | No, final serial gate | Run only after all retained fixes; hardware commands one at a time |

The two correctness fixes, trace mutation test, and precision isolation should
be resolved before any existing long-context, watcher, or performance artifact
is accepted as final evidence.
