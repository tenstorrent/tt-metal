# Qwen3.5-27B / 4xP150 — Prefill Tracing Port Investigation

**Status:** investigation (no implementation commitment)
**Author:** atupe
**Date:** 2026-04-29
**Source branch (reference):** `qwen9b-p150` @ `05fea69009` (Qwen3.5-9B on 1×P150)
**Target branch:** `atupe/qwen35-27b-4xp150` (Qwen3.5-27B, TP=4, 4×P150)
**Scope:** decision-grade investigation. Output is a recommendation
("proceed / proceed-with-cap / don't proceed") plus a high-level porting
sequence if proceeding. Implementation is out of scope.

---

## 1. Background — the 9b prefill-tracing mechanism

### 1.1 Bucketed trace pattern

The 9b branch captures the **prefill graph itself** in
`begin_trace_capture`/`end_trace_capture` — not just decode. Two methods
implement it (`models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`):

- `capture_prefill_trace_paged(device, page_table, bucket_size, chunk_size)`
  at line 416: pre-allocates persistent input buffers sized for `bucket_size`,
  runs a warm-up `prefill_paged` outside the trace to compile programs and
  prime the program cache, re-pins DeltaNet state pointers to persistent
  external buffers (`_deltanet_external_states`), sets
  `_prefill_trace_inputs` so the next `prefill_paged` reads/writes those
  persistent buffers, and captures.
- `prefill_traced_paged(token_ids, page_table, actual_len)` at line 544:
  asserts `token_ids.shape[1] == bucket`, copies host→device into the
  persistent buffers, calls `execute_trace`, then slices
  `hidden[:, actual_len-1:actual_len, :]` and runs `rms_norm + lm_head`
  **outside** the trace. Releases the trace after one replay.

Trace output is the full last-layer hidden state `[1, bucket, hidden_size]`
(TILE). The norm + LM head are deliberately kept post-trace because
`gather`/`slice` at `actual_len-1` is host-allocating during capture and
would FATAL.

### 1.2 Per-prompt fresh capture; padding strategy

The 9b call-site (`demo/text_demo.py:279-282`) computes
`bucket_size = ceil(T / chunk_size) * chunk_size` and **captures fresh per
prompt**. There is no fixed bucket ladder. The trace is released after one
replay; the next prompt re-captures.

Padding uses the **last real token repeated**, not zero. The DeltaNet
recurrence is sequential and updates state for every input token in the
captured bucket; padding with `<pad>`/`<unk>` corrupts state. Repeating the
last real token yields a smoother (still imperfect) post-prefill state.
The next-token logit is unaffected because it is extracted at
`actual_len-1`; only continued-decode quality is affected.

### 1.3 Why decode-trace-only wasn't enough

Prefill on 9b runs many GDN chunks (sequential delta-rule recurrence is the
hot path). Each chunk launches several ops, each paying Python dispatch
cost. With ~28 GDN layers × dozens of chunks, dispatch alone dominates at
short ISL. Capturing the prefill graph collapses all those launches into a
single device program. Combined with kernel-level wins
(mega-fused QKV+a+b+g, on-device GDN prefill kernel
`5e3cbea445`/`c3a8d6ce8e`/`a4820930f9`, layer-class chunk-size split,
`math_approx_mode`), the 9b TTFT moved from ~10 s to ~3.6 s.

---

## 2. Current 27b prefill state on 4xP150

### 2.1 `prefill_layer_chunked`

`models/demos/qwen35_27b/tt/model.py:405-…` runs prefill layer-by-layer ×
chunk-by-chunk. Chunk size already split by layer class
(`model.py:484-489`):

| Layer type | Chunk size source | Default |
|---|---|---|
| `full_attention` | `ATTN_CHUNK_SIZE` if paged, else full seq | 2048 (env-overridable via `QWEN35_ATTN_CHUNK_SIZE`) |
| GDN | `chunk_size` arg → `GDN_CHUNK_SIZE` if paged, else `args.prefill_len_cutoff` | 2048 (env-overridable via `QWEN35_GDN_CHUNK_SIZE`) |

Constants in `model_config.py:50-51`. Note: 9b uses GDN=2048 (default)
during capture and `attn_chunk_size = max(chunk_size, 4096)` so attention
runs at 4096. 27b's defaults (2048/2048) leave a known-cheap retune on the
table but it is orthogonal to tracing.

### 2.2 GDN prefill state

`gdn.py:570-594` — `_init_prefill_states()` allocates per-layer:

- `_prefill_conv_states[i]` for `i in range(conv_kernel_size=4)`,
  shape `[1, 1, qkv_dim_tp]` bfloat16 TILE.
- `_prefill_rec_states` shape `[Nv_TP, Dk, Dv]` bfloat16 TILE.
- `_prefill_rec_states_f32` shape `[Nv_TP, Dk, Dv]` float32 TILE.
- `_prefill_fused_output` shape `[Nv_TP, 1, Dv]` bfloat16 TILE.

All allocated via `ttnn.from_torch(... mesh_mapper=ReplicateTensorToMesh)`.
**Each `_init_prefill_states` call rebinds these to fresh tensors** —
addresses are not stable across calls. This is the analog of 9b's
`_deltanet_external_states` and is a porting concern: a captured trace
records writes against the addresses present at capture time.

### 2.3 Existing decode trace pipeline

`tests/test_e2e_generate.py::test_e2e_generate_traced` already does decode
tracing with paged KV at `:628-634`. Sequence is: `prefill_layer_chunked`
→ `_replicate_to_batch` → compile run → `begin_trace_capture` around
`ttnn_decode_forward` → replay loop with host-DMA input updates. This is
unchanged by this investigation.

### 2.4 Trace region budget

`tests/test_e2e_generate.py:226, 380` — `trace_region_size: 200_000_000`
(200 MB per device). This is the budget the prefill trace must fit into,
and it is not currently sized against a captured prefill graph.

---

## 3. Gap analysis: what porting requires

### 3.1 New methods on the 27b model

Two new entry points on `models/demos/qwen35_27b/tt/model.py`:

- `capture_prefill_trace_paged(mesh_device, page_table, bucket_size, chunk_size)`
- `prefill_traced_paged(token_ids, page_table, actual_len)`

The capture step's body mirrors 9b's: persistent input allocation →
warmup-outside-trace → re-pin GDN persistent state → set
`_prefill_trace_inputs` → capture. The replay step's body is also a direct
mirror: padded host→device copy → `execute_trace` → post-trace slice +
norm + lm_head → `release_trace`.

### 3.2 Persistent buffers that must be allocated up-front

| Buffer | Shape | Mesh mapping | Notes |
|---|---|---|---|
| `token_ids_buf` | `[1, bucket]` uint32 | Replicate | Host-DMA target on replay |
| `page_table_buf` | `[B, max_blocks]` int32 | Replicate | Host-DMA target on replay |
| `chunk_page_table_bufs[i]` | `[B, blocks_in_chunk_i]` int32 | Replicate | One per full-attn chunk |
| `gdn_output_buf` | `[B*Nv_TP * chunk_size, 1, Dv]` bf16 TILE | per-device shard | Shared across all GDN layers |

`chunk_page_table_bufs` count = `ceil(bucket / attn_chunk_size)`. For
bucket=128k and attn_chunk_size=4096 → **32 chunk-page-table buffers per
prefill trace.**

### 3.3 Address-stable GDN state

27b currently rebinds `_prefill_conv_states` etc. on every
`_init_prefill_states()`. The trace records writes against capture-time
addresses, so we need persistent external state mirroring 9b's
`_deltanet_external_states`. Concretely: after `allocate_paged_kv_caches`,
allocate per-GDN-layer `(rec_state, conv_state)` external buffers once,
copy pointers into each `dn`, and have the GDN op's prefill path write
in-place into them. The "re-pin after warmup" dance from 9b
(`qwen35_model.py:507-519`) must be carried over: the warmup pass leaves
DN state pointers on non-persistent buffers (slices of `x_padded`), so we
explicitly restore the pointers to the external buffers before capture.

### 3.4 Bucket strategy for 128k

Three options, ranked:

- **(i) Per-prompt round-up bucket (9b-style).** Capture per prompt at the
  smallest bucket fitting `T`. Simplest port. Capture cost paid per prompt
  but amortizes against single replay. Best when most prompts are short
  and bucket sizes are coarse.
- **(ii) Fixed bucket ladder (e.g. 4k, 8k, 16k, 32k, 64k, 128k).** Capture
  once per ladder rung, persist trace IDs across prompts, dispatch by
  rounding up to nearest rung. Reduces capture cost amortization for
  repeated prompts but spends 6× the trace memory (or evicts/re-captures).
- **(iii) Single fixed bucket = 128k.** Always pad to 128k. Trivial code,
  but every short prompt eats the 128k trace cost. Almost certainly worse
  than (i) at any sensible prompt distribution. Ruled out.

**Recommendation: start with (i), match 9b.** Reconsider (ii) only after
measurements show capture cost is the dominant TTFT term (unlikely given
the design intent).

### 3.5 Out-of-trace tail on TP=4

After `execute_trace`, the post-trace work is `slice` + `to_layout` +
`to_memory_config` + `rms_norm` + `ttnn.linear(x_last, lm_head_weight)`.
On 4xP150 with TP=4, the LM-head matmul includes a TP all-gather. Each of
these ops pays Python dispatch on the host, but the per-prompt count is
small (≤ 5 ops). Expected post-trace tail: low single-digit ms; not a
material risk.

### 3.6 Padding strategy

Mirror 9b: pad with the last real token repeated. Pull the same comment
into 27b — DeltaNet recurrent state in the padded tail will be slightly
corrupted but the next-token logit at `actual_len-1` is unaffected. The
caller (decode loop) takes over from there using the GDN state at the end
of the *real* prompt range, which means we may also need to **roll back
GDN state to the actual_len point** after replay. **Open question — see
§7.3.**

---

## 4. 4xP150 / TP=4-specific risks

| # | Risk | Likelihood | Impact | Retired by | Fallback |
|---|---|---|---|---|---|
| R1 | Mesh-trace doesn't preserve `ReplicateTensorToMesh` persistent-buffer addresses across replay | Low | Blocking | Spike 1 | None known — would block proceed. Investigate `ShardTensorToMesh` alternative. |
| R2 | TP-sharded GDN buffer (`gdn_output_buf` on `dim=0`) addresses misalign across devices on capture vs replay | Medium | Blocking | Spike 1 + spike 3 | Allocate per-device buffers explicitly via `get_device_tensors` and reassemble. |
| R3 | `trace_region_size=200 MB`/device insufficient for bucket=128k trace | Medium | Caps target ISL | Spike 2 (extrapolate from bucket=4k) | Cap supported bucket at largest power-of-2 that fits. Linearly grows with chunk count. |
| R4 | Per-prompt capture cost on 4 devices wipes out the dispatch savings at short ISL | Low–Medium | Reduces speedup | Spike 2 measures capture time at bucket=4k | Move to fixed-ladder buckets (§3.4 option ii) for short prompts only. |
| R5 | `chunk_page_table_bufs` count (32 at 128k) inflates persistent-input memory footprint | Low | Reduces headroom | Spike 2 (count buffers and sum sizes) | Larger `attn_chunk_size` (e.g. 8192) halves the count; needs SDPA-prefill validation at that size. |
| R6 | DN recurrent-state corruption in padded tail breaks downstream traced decode | Medium | Wrong tokens after first | Paper-only (logic equivalent to 9b's already-validated case) | Add a state rollback after `execute_trace` that re-runs the recurrence on the padding tokens with negation, or re-prefills the last-token-only with the correct state — design later. |
| R7 | `_init_prefill_states` rebinding interacts badly with persistent-state pointers | High | Subtle correctness bug | Spike 3 | Guard `_init_prefill_states` with a "trace mode active" flag that no-ops when set. |
| R8 | Capture pollutes program cache enough to slow down subsequent eager prefill (warmup primes one shape; many prompts re-capture at varying buckets) | Low | TTFT regression on 2nd+ prompt | Paper-only — same risk on 9b, acceptable | Bucket-quantize prompts to reduce shape variety. |
| R9 | Post-trace `lm_head` (TP=4 all-gather) pays measurable dispatch cost | Low | Negligible TTFT impact | Paper | Acceptable. Could be folded into the trace if the slice-by-actual_len is replaced with a fixed-position slice (capture per actual_len, expensive). |

---

## 5. Spike plan

All three spikes live in `models/demos/qwen35_27b/tt/tests/`, marked
`@pytest.mark.spike`, and are **throwaway** (deleted before any production
PR). Each emits structured `[SPIKE-N]` log lines for appendix B.

### 5.1 Spike 1 — mesh-trace + persistent-buffer round-trip

**File:** `test_prefill_trace_spike_1_mesh_persistent.py` (~70 LOC)

**What:** Allocate two persistent buffers on the 4xP150 mesh:
`buf_a = ttnn.from_torch(zeros, mesh_mapper=Replicate)`,
`buf_b = ttnn.from_torch(zeros, mesh_mapper=Shard(dim=0))`. Capture a
trivial trace `c = ttnn.add(buf_a, buf_b)`. Re-replay with three different
host→device updates of `buf_a` and `buf_b`. Check that `c` reflects the
input on each replay (not just the capture-time values).

**Pass criteria:** all three replays produce correct results across all 4
devices; no FATAL or "buffer freed" errors. R1, R2 retired.

### 5.2 Spike 2 — captured prefill trace size at bucket=4k

**File:** `test_prefill_trace_spike_2_size.py` (~80 LOC)

**What:** Skip the persistent-buffer redesign. Take 27b's existing
`prefill_layer_chunked(tokens, ...)` and capture **directly** between
`begin_trace_capture` and `end_trace_capture` at bucket=4k. The captured
trace will be incorrect on replay (buffers were transient), but **size and
capture time are valid** because both are properties of the captured graph
itself, not of replay correctness.

Log:
- captured trace bytes/device (`mesh_device.get_trace_buffer_size(trace_id)` if available, else read trace region usage delta)
- capture wall time
- count of distinct programs in the cache before/after

**Pass criteria:** measured. Extrapolate linearly: `size(128k) ≈
size(4k) × 32 × 1.3` (with chunk-buffer overhead multiplier). If
extrapolated size ≤ 1.5 GB/device, R3 retires green; if 1.5–3 GB, cap
recommendation; if >3 GB, R3 blocks 128k.

### 5.3 Spike 3 — GDN persistent-address stability

**File:** `test_prefill_trace_spike_3_gdn_addr.py` (~50 LOC)

**What:** On a 4-layer 27b stub, instrument
`_prefill_conv_states[0].buffer_address(device_id=…)` for each device
across two `_init_prefill_states()` calls and one
`reset_state()`+`_init_prefill_states()` cycle. Verify whether addresses
change.

**Pass criteria:**
- Addresses stable across `reset_state` + same-pointer rebind: R7 retires green, no refactor needed.
- Addresses stable only across explicit reuse (we hold the tensor): R7 retires yellow — port the 9b external-state pattern verbatim.
- Addresses unstable in any case: R7 retires red — needs deeper investigation, may block.

---

## 6. Cost model

### 6.1 Trace memory budget per bucket

Linear extrapolation from spike 2:

| Bucket | Estimated trace size/device | Within 200 MB? | Within 1.5 GB? |
|---|---|---|---|
| 4 k | `B4` (measured) | TBD | TBD |
| 8 k | `~2.0 × B4 × 1.1` | | |
| 16 k | `~4.0 × B4 × 1.15` | | |
| 32 k | `~8.0 × B4 × 1.2` | | |
| 64 k | `~16.0 × B4 × 1.25` | | |
| 128 k | `~32.0 × B4 × 1.3` | | |

(Filled from spike 2 data in appendix B.) The 1.1–1.3 multiplier reflects
overhead from chunk-page-table buffer count growth and SDPA-prefill
specialization per chunk count.

### 6.2 Expected TTFT speedup

Paper-derived from 9b's measured ratio. 9b TTFT: ~10 s eager → ~3.6 s
with full prefill-tracing + kernel wins. The kernel wins
(mega-fused QKV+a+b+g, on-device GDN kernel, `math_approx_mode`,
core-count bump) account for an estimated ~2× of the speedup; the
remaining ~1.4× is from prefill-tracing eliminating Python dispatch.

For 27b on 4xP150, the dispatch overhead is **higher per op** (4-device
mesh dispatch) and **count is higher** (3× model size, similar layer
count, more chunks at 128k). Therefore the trace-only contribution should
be at least as large as 9b's, plausibly larger. Expected TTFT speedup at
matched bucket size: **1.3–1.6× over current 27b
`prefill_layer_chunked`** for prompts where bucket padding adds < 30 %.

### 6.3 Capture cost vs replay savings — break-even ISL

Capture cost ≈ warmup prefill (one full eager prefill) + a small fixed
cost for `begin/end_trace_capture` and persistent-buffer setup. Therefore
per-prompt capture overhead is **roughly one extra eager prefill**.

So the per-prompt TTFT under 9b-style per-prompt capture is:

```
TTFT_traced = warmup_prefill_time + traced_replay_time
            ≈ eager_prefill_time + traced_replay_time
```

For this to be a TTFT win, `traced_replay_time < 0` — i.e., **per-prompt
fresh capture cannot beat eager prefill on TTFT.** The 9b benefit comes
**only** if the trace is reused across prompts.

This is a critical finding that needs verification against the 9b
measurement methodology. Two possibilities:

1. The 9b 10s→3.6s number is measured **excluding capture time** (only
   replay is counted as TTFT), in which case "per-prompt fresh capture"
   does not actually win on TTFT — it wins on **replay throughput** for
   benchmarks/serving when the same trace is reused.
2. The capture is amortized via `prefill_paged` taking a fast path inside
   the trace that the eager path doesn't have access to (e.g. fewer host
   round-trips because state buffers are persistent).

**This is open question Q3 in §7.3 and must be resolved before
recommending.** If (1) is true, the recommendation must shift to a
**fixed-ladder, capture-once-per-rung** model, not per-prompt fresh
capture.

---

## 7. Recommendation

### 7.1 Headline

**Provisional: proceed with investigation, then proceed-with-cap.** Final
recommendation pending spike 2 (memory cap) and resolution of open
question Q3 (capture-cost accounting in 9b's measurement).

Rationale:

- Mechanism is sound and already proven on 9b.
- 27b GDN already has prefill-state plumbing that closely mirrors 9b's
  pattern; main port surface is `model.py` (two new methods, persistent
  external state) and `gdn.py` (in-place trace-mode write paths). No new
  kernels required.
- The TP=4 / mesh-trace risks (R1, R2, R7) are all empirically settable
  with ≤ 200 LOC of throwaway test code (spikes 1 + 3), which is cheap
  given the potential payoff.
- The 128k target is the largest open risk (R3); spike 2 directly
  addresses it.
- Open question Q3 may shift the strategy from per-prompt to fixed-ladder
  buckets, but does not change the underlying feasibility.

### 7.2 If proceeding — high-level porting sequence (5 milestones)

1. **External GDN state**: add `_deltanet_external_states` analog on the
   27b model; rewire `gdn.py` prefill state init to bind into externally
   provided buffers when present. Validates with existing
   `prefill_layer_chunked` (no trace yet) — must produce bit-identical
   results to current path.
2. **`capture_prefill_trace_paged` skeleton**: persistent input buffers,
   warmup-outside-trace, capture, no replay logic. Validates that capture
   completes and that trace size matches spike-2 prediction.
3. **`prefill_traced_paged`**: replay path with padding, host→device,
   `execute_trace`, post-trace tail. PCC-validate against
   `prefill_layer_chunked` at bucket=4k, 8k, 16k.
4. **Bucket strategy + recapture policy**: per-prompt fresh capture
   (default) or fixed ladder (if Q3 resolves to "must amortize").
5. **128k validation + perf measurement**: end-to-end TTFT on the full
   spectrum (128 → 128k), spike-2-driven trace_region_size update, perf
   regression check on existing decode-trace test.

### 7.3 Open questions for follow-up

- **Q1.** Does `mesh_device.get_trace_buffer_size(trace_id)` exist, or do
  we have to read trace region usage deltas? (Determines how spike 2
  measures size.)
- **Q2.** Does 27b's existing decode trace already implicitly require any
  GDN persistent state that's incompatible with the prefill trace's
  external-state pattern? (Quick read of `gdn.py:241 forward_decode`
  during milestone 1.)
- **Q3 [critical].** How is 9b's 10s→3.6s TTFT actually measured —
  does it include the per-prompt capture time, or only replay? Re-read
  `text_demo.py:280-296` against the perf-print path. Outcome
  determines per-prompt vs fixed-ladder strategy.
- **Q4.** What's the policy for DN recurrent-state corruption in the
  padded tail when the captured prefill is followed by traced decode? Is
  the existing 9b "ignore — only logit at actual_len-1 matters for the
  first decoded token" sufficient at long context, or does decoded-token
  quality degrade visibly?

---

## Appendix A — file/line-level 9b → 27b mapping

| Concept | 9b file:line | 27b file:line (target) |
|---|---|---|
| `capture_prefill_trace_paged` | `qwen3_5_9b/tt/qwen35_model.py:416` | `qwen35_27b/tt/model.py` (new) |
| `prefill_traced_paged` | `qwen3_5_9b/tt/qwen35_model.py:544` | `qwen35_27b/tt/model.py` (new) |
| `_deltanet_external_states` | `qwen3_5_9b/tt/qwen35_model.py` (search) | `qwen35_27b/tt/model.py` (new) |
| `_prefill_trace_inputs` field | `qwen3_5_9b/tt/qwen35_model.py:769` | `qwen35_27b/tt/model.py` (new) |
| GDN prefill state init | `qwen3_5_9b/tt/qwen35_gated_deltanet.py` | `qwen35_27b/tt/gdn.py:570 _init_prefill_states` |
| GDN in-place trace write path | `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` | `qwen35_27b/tt/gdn.py forward_prefill` (modify) |
| Call-site (capture+replay) | `qwen3_5_9b/demo/text_demo.py:279-296` | `qwen35_27b/tt/tests/test_e2e_generate.py` (new test) |
| Trace region size | `qwen3_5_9b/demo/text_demo.py` mesh_device fixture | `qwen35_27b/tt/tests/test_e2e_generate.py:226,380` (revisit per spike 2) |

## Appendix B — raw spike measurements

(Populated after spike runs.)

```
[SPIKE-1]   ...
[SPIKE-2]   ...
[SPIKE-3]   ...
```
