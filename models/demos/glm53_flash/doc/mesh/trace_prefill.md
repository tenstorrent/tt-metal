# Traced prefill for the resident mesh model (2026-10-20, Phase A implemented, Phase B designed)

Code: `tt/trace_prefill.py` (`supported_prefill_lengths`, `PrefillSlot`, `PrefillTraceInputs`),
`GLM53MeshModel.capture_prefill` / `prefill_traced` (`tt/mesh_model.py`), the `trace=` form of
`FunctionalDecoder.prefill_forward`, the `slot=` / `ring_sel=` forms of the DSA fill / index maintenance and
`TtModelHead.selected_logits`. Test: `tests/test_trace_prefill.py`. Knob: `GLM53_TRACE_PREFILL=1` in
`demo/text_demo.py` and `tests/test_mesh_model.py::_ttft`.

## Why

The eager prefill dispatches 45 layers x ~110 ops from the host on every call (919 ms warm for 128 tokens on the
8x4 against ~100-300 ms of device work) and compiles a new program set for every distinct padded length. The
repo's answer (`models/tt_transformers/tt/generator.py` `_capture_trace_prefill` / `_easy_trace_prefill`,
`model_config.trace_prefill_supported_seq_lens`) is one trace per supported length over persistent device
inputs, the prompt padded to the next supported length.

## What is reused from the reference generators, what is new

| piece | reference (`models/tt_transformers/tt/generator.py`, `models/demos/qwen38_27b_qb2/tt/generator.py`) | here |
|---|---|---|
| trace per padded length over persistent inputs refreshed with `copy_host_to_device_tensor`, compile pass before the capture pass with the SAME graph function | `_easy_trace_prefill` / `_prepare_trace_prefill` + `_record_trace_prefill` (`_prefill_trace_forward` shared "verbatim by the compile pass and the capture pass"), `trace_prefill_supported_seq_lens` | `capture_prefill` (allocate all -> compile each -> capture each, `_prefill_graph` shared), `supported_prefill_lengths` |
| every bucket compiled before any recording ("Compile every bucket before recording any of them") | `warmup_model_prefill` with `_defer_prefill_recording` | the same order inside `capture_prefill`; `warmup_decode` before it |
| the last token out of a bucketed trace: the 32-row tile cut by a tensor-argument `ttnn.slice` (persistent `[4]` int32 start / end; the program keys on the bucket, the offset is a runtime tensor), the row `last_token_idx % 32` read on the host | `Transformer.process_logits_after_prefill_trace` (`_tail_slice_start` / `_tail_slice_end`, `extract_user(last_token_idx % 32)`) | `TtModelHead.tile_logits` (the same slice, inside the trace, after the collapse + norm of every row), `PrefillTraceResult.row` |
| trace outputs copied into buffers that predate every capture ("The copy target predates every trace. No capture-created output may remain alive when an earlier decode trace is replayed") | qwen38 `_prefill_trace_step` (`ttnn.copy` into `prefill_prepared["output"]`) | `out_logits` / `out_ids` of `PrefillTraceInputs`, the copies at the end of `_prefill_graph` |
| no persistent allocation after a capture; new signatures -> release and recapture | qwen38 `_release_traces()` before fresh persistent buffers, `prefill_signatures` | the ordering rule above (demo / test); serving should adopt the release-and-recapture |
| the valid prompt length inside a padded bucket | tt_transformers: RoPE / page tables / `last_token_idx` per prompt, the padding rows are never read; qwen38: one trace per EXACT length (no padding) | ttKDA `actual_end` as a persistent device scalar (the kernels clip their chronology there), causal SDPA for the DSA rows |

New here, with the reason:

* the user slot as a device value (`PrefillSlot`): both references trace one fixed slot (qwen38: "physical slot 0, prefix 0" in its trace key; tt_transformers pins `user_id=0` for chunked prefill and routes a traced prefill through the page table of the user) and run other slots eagerly or recapture. Our demo prefills 4-32 distinct slots per row through the same traces, so the slot index became `batch_idx_tensor` of the paged fills (an existing argument of `paged_fill_cache`) and, for the KDA recurrent state that is not paged, the `[B*H*K/32, 1, 32, V]` view + the same paged fill; the tap-major conv history has no paged form, hence the one-hot mask blend (exact in bf16).
* the pooled-key ring tail (`ring_sel`, one HiFi4 matmul per ring + one block fill): the DSA indexer's raw ring is GLM-specific (no reference node); the eager path writes `S % 4` single rows with host-chosen slots, which cannot be traced at a fixed length.
* the KDA zero input state as a persistent tensor instead of `allocate_state` (`ttnn.zeros`) inside the pass: a capture must not allocate a host-written tensor.

## Phase A (implemented): one trace per length in the dense range

Supported lengths: {128, 256, 512, 1024, 2048} rounded up to a multiple of `32 * rows` (sequence-parallel row
slabs), kept when `T <= 2048` (every DSA row attends densely: `TtDSA.sparse_row_start`), `T` fits the caches and
`T % 128 == 0` (whole pool tiles). 8x4: {256, 512, 1024, 2048}; one row: {128, ..., 2048}.

What the eager prefill takes from the host and what replaces it in the captured graph:

| host value in the eager path | traced form (persistent device input, refreshed before the replay) |
|---|---|
| token ids `[1,1,1,T_pad]` uploaded per call | `PrefillTraceInputs.ids`, row-sharded like the decode `ids_dev` |
| `seq_len` -> ttKDA `actual_end` (`TtKDA.scalar(start+seq_len)`, cached per value) | `end` `[1,1,1,1]` uint32 replicated; `actual_start` = the cached `scalar(0)` |
| zero KDA input state (`allocate_state(1)` = `ttnn.zeros` inside the pass) | `PrefillSlot.kda_zero`, allocated once, read-only (ttKDA only reads the input state) |
| KDA carry write-back into slot `user` (`slice` / `concat` / `copy` with the host index) | recurrent: `paged_fill_cache(batch_idx_tensor=user_idx)` on the `[B*H*K/32, 1, 32, V]` view (`ttnn.experimental.view`) of the batched `[B,H,K,V]` fp32 state (identity page table; whole-tile copy, exact); conv history (`[taps, B, W]` bf16 tiles): `dst = dst * keep + new * mask` with one-hot `[taps, B, 1]` masks (every product is `x*1.0` or `x*0.0`: exact in bf16) |
| latent fill `paged_fill_cache(batch_idx=user)` | `batch_idx_tensor=cache_idx` (per column when the cache is user-partitioned, plus the scratch slot) |
| pooled-key index: pools `[0, seq_len//4)` + `seq_len % 4` ring rows written one row per call | every pool of the padded chunk (`T/4`, whole tiles; the pools past the prompt are rewritten from the ring by decode before they become selectable: pool p is selectable only once position 4p+3 was decoded) + the ring block from `ring_sel` `[1,1,32,T]` (rows j < 4 one-hot at the last prompt position with `p % 4 == j`; HiFi4 bf16 matmul: exact) through one `paged_fill_cache` per ring |
| the head on the last row (`slice(streams, S-1)`) | collapse + final norm on every local row, `head_sel` `[1,1,32,T/rows]` (row-sharded; the owner row's block selects its local last row) applied as a bf16 one-hot matmul AFTER the norm (exact), the decode-shaped LM-head matmul on 32 rows, greedy ids; `first_token(ids, owner_row)` reads row 0 of the owner row's chip |

Everything else in the prefill is a per-row computation (dense causal SDPA, mHC, MLP / MoE, norms) where the
padding rows past `S` do not influence the valid rows, so the traced result equals the eager one up to
program-config differences that already exist between different padded lengths of the eager path.

Capture discipline (`capture_prefill`): allocate the inputs of EVERY length first, run each graph once eagerly
(program cache with the exact shapes and argument values), then capture one trace per length. `PrefillSlot` is
allocated in `GLM53MeshModel.__init__`. Rule for callers: no persistent tensor may be allocated after a capture
(it could alias that trace's transient buffers and be overwritten by a replay) -> the demo runs
`warmup_decode` (persistent decode inputs + the eager decode compile) -> `capture_prefill` -> `capture` (decode).
Known hazard left: an eager prefill AFTER the captures with a new `S` creates a new cached `TtKDA.scalar(S)`;
serving should make the eager scalars refreshable persistent tensors (the test keeps every eager pass before
the captures).

Costs per replay: 4 host->device copies of the slot tensors + 4 per-length inputs (ids `4T` bytes, the two
selectors `64T` + `64T/rows` bytes, the end scalar), one `execute_trace`, one `[32,1]` id readback. The trace
region holds one command stream per length (set `GLM53_MODEL_TRACE_MB` accordingly on the 8x4: the decode trace
of 45 layers alone needs ~216 MB at 32 users/row).

Validation (`tests/test_trace_prefill.py`, chip 2 of the Galaxy, (1,1) mesh, layers 0-5 (KDA+MLP 0-2, DSA+MoE 3,
KDA+MoE 4-5), bfp4 experts, B=2, lengths 128 / 256 / 512; `_logs/agents/traceprefill/traced_vs_eager_2.log`,
2026-10-21):

| prompt (slot) | trace T | first token eager / traced | last-token logits pcc, max diff | stream rows [0,S) pcc, max diff | slot state after the prefill (KDA recurrent + conv per layer, latent rows, complete pools, ring rows) |
|---|---|---|---|---|---|
| S=100 (0) | 128 | 45523 / 45523 | 1.000000, 0 | 1.000000, 0 | every tensor bit-identical |
| S=128 (1) | 128 | 1526 / 1526 | 1.000000, 0 | 1.000000, 0 | bit-identical |
| S=103 (1) | 128 | 9276 / 9276 | 1.000000, 0 | 1.000000, 0 | bit-identical, incl. the 3 ring rows of the open pool |
| S=200 (0) | 256 | 31259 / 31259 | 1.000000, 0 | 1.000000, 0 | bit-identical |
| S=384 (1) | 512 | 58387 / 58387 | 0.999880, 0.125 | 0.999887, 0.036 | the eager pass pads to 384, the trace runs 512: the variation between two padded lengths of the eager path (ttKDA chunk scan, MoE block splits) |

Replay-twice: prompt B (S=90, slot 1) replayed after prompt A (S=100, slot 0) through the same trace gives B's eager
first token (74246) and logits (pcc 1.000000; 0.0989 against A's). Decode continuity: the first token + 2 eager
decode steps after the traced prefill equal those after the eager prefill ([13861, 49701, 29452], S=102). The slot
write primitives alone (`test_slot_write_ops`): all exact. TTFT on this one chip (6 layers, S=100): eager 162.9 ms,
traced 145.1 ms: the 6-layer prefill is device-bound here (three MoE layers stream their experts), the saving is the
host dispatch of ~660 ops. The same through `tests/test_mesh_model.py::_ttft` with `GLM53_TRACE_PREFILL=1`
(`ttft_1.log`, the 128-token L45_S128 prompt): eager warm 175 ms (cold 894 ms), traced replays 163 ms, first token
1526 = the host head on the HF rows in both, chain pcc vs HF 0.996515 in both; the three traces captured in 3 s
after the eager prefill had compiled the programs. `demo/text_demo.py` (questions_8, 4 users of 20-27 tokens, the
same chip and layers, `demo_1.log`): `warmup_decode` -> four prefill traces (128..1024) captured in 33 s -> the decode
capture -> traced TTFT 105-109 ms per user -> 61 decode-trace replays at 34 ms/step, i.e. prefill replays before
decode replays and a decode trace captured after the prefill traces corrupt neither. Not validated on a device:
multi-row meshes (the row-sharded ids, the replicated tile bounds, the row CCLs inside the trace), the
user-partitioned cache slot path. The 8x4 / 45-layer measurement (where the eager 128-token TTFT is 919 ms against
~100-300 ms of device work) is the coordinator's run (`GLM53_TRACE_PREFILL=1`, `tests/test_mesh_model.py::_ttft`
`PERF-TTFT-TRACED` line or `demo/text_demo.py`; `GLM53_MODEL_TRACE_MB=1024`).

## Phase B (design): traced chunks for long prompts (`start > 0`, sparse rows)

A chunk at `start > 0` (the chunked contract of `functional_decoder.py`) adds these host dependencies:

| piece | eager today | traceable as-is? | device-side form |
|---|---|---|---|
| KDA chronology | `scalar(start)`, `scalar(start+seq_len)` | yes with persistent `start` / `end` scalars (ttKDA reads them on device; the input state is the user's current carries: `slice` of the batched state with the host user) | input carries through the same device-slot mechanism in reverse: a `paged_fill_cache`-free read = a one-hot matmul `Sel_u [1, B] x state` is NOT exact for fp32; use `ttnn.experimental.view` + `paged_fill_cache` into a persistent single-user scratch state (`[1,H,K,V]`, page table = the user's blocks via `batch_idx_tensor`), conv rows via the mask trick |
| latent fill from block `start // 32` | page-table view `[1, bps - b0]` sliced per call | the fill writes the chunk at the table's column 0, so the offset must be in the table: a persistent `[1, bps]` table whose row is the user's blocks SHIFTED by `b0` (host-built, refreshed per call) with `batch_idx=0` | one trace per chunk length; the shift lives in the refreshed table |
| dense rows of a chunk (`start < 2048`) | `cache.user_rows(user, start)` (start-dependent size) + windowed SDPA (`cu_window_seqlens`, `windowed_q_token_offset=start`) | not at a fixed shape: the prefix length is `start` | bound the prefix to a fixed window: dense chunks only occur for `start < 2048`, so a `[2048, L]` persistent prefix read (`paged_fill_cache`-style gather of the user's first 64 blocks through a device table, zero-padded blocks for `start` smaller than that) and the SDPA in windowed mode with a device `cu_window_seqlens` = `[0, start + T]` (persistent int32, refreshed) and `windowed_q_token_offset` = ... host int (kernel runtime arg): one trace per `(chunk length, start bucket)` for the few dense chunks, or keep the dense chunks eager (at most one 2048-row chunk per prompt) |
| sparse rows (`start >= 2048`) | `cache.user_rows_rm(user, start + T)` (whole-prefix ROW_MAJOR copy, start-dependent size), scorer `select(query_start=start+q0, num_pools_valid=end//4)` (host ints -> kernel args), `sparse_sdpa` over the kv copy | not at a fixed shape: the kv copy grows with the prefix | prefix buckets: one trace per `(chunk length, prefix bucket)` with the kv copy sized to the bucket (`[bucket, L]`, refreshed through a device table gather; rows past `start + T` masked by the index: the scorer's `num_pools_valid` bound must become a device tensor, as the traced decode already does with `SparseDecodeMeta.valid_len`), the query start as a device tensor (the custom prefill scorer `custom_ops/index_score_*` takes it as a runtime arg today -> a device-tensor variant like the decode scorer's `cur_pos`), the top-k / tail ids built from device positions (`_tail_ids(T, query_start)` is a host-built constant per `query_start` today -> `iota + start_t`) |
| index maintenance across a chunk boundary | ring completion of the straddling pool (`start % 4` rows), `write_pools(start_pool)` with a tile-internal offset | the offsets are host control flow | chunk starts are multiples of 32, so `start % 4 == 0`: no straddling pool; `write_pools` at `start_pool = start/4` = whole tiles from block `start/128` -> the shifted-table trick above |
| `next_start` bookkeeping | host | host | unchanged |

Recommended order: (1) KDA chunks at `start > 0` with the persistent start/end scalars and the device-slot
carry read/write (everything is already device-side in ttKDA); (2) the shifted page tables for the latent fill
and the pool writes; (3) sparse chunks with prefix buckets {8k, 32k, 128k, 512k, 1M} (the kv copy of a 1M prefix is
1 GB bf16 per chunk: the bucket gather must be the paged read the sparse_sdpa kernel does itself, i.e. the
`SparseKVFormat` paged variant, before the long buckets are worth tracing); (4) dense chunks stay eager (one per
prompt at most). Expected win: the host dispatch of a 32k-token chunk (~45 x 150 ops) is ~1 s against ~10 s of
device time, so Phase B is a ~10 % TTFT gain for long prompts; Phase A removes ~80 % of the short-prompt TTFT.
