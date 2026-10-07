# Interleaving prefill with decode (vLLM continuous batching)

Goal: new requests arrive while others decode; their prompts are prefilled (in chunks) BETWEEN decode steps of the running requests, without disturbing them
and without recomputing them. Flag: `DSV41_VLLM_INTERLEAVE=1` (default OFF; with it off every existing path is byte-for-byte the previous one). It turns on
`supports_chunked_prefill` in `tt/generator_vllm.py` and sets `DSV41_PF_UMASK=1` for the prefill trace.

## 1. Why a prefill of a subset of users disturbed / recomputed the others (state-by-state)

The traced chunk prefill (`DSV41PrefillModel.run_traced_chunks`) captures ONE trace of a chunk of C tokens for ALL U users of a mesh row (4 rows, U users each, batch
B = 4U) and replays it once per chunk. Everything inside the trace is batched over the U users; the only things that are global to a replay are the position tables
(RoPE, window / latent masks, `lim` / `off`: they depend on the chunk start s0 only). So **one replay = one window [s0, s0 + C) for every user**, and a replay always
costs the compute of U x C tokens per row, whatever the number of users that have real work.

| state | where | per-user? | what a replay did to users that are not prefilled (before) | now |
|---|---|---|---|---|
| paged KV latents (pool pages), window ring rows | `PagedKVPool`, written by `PagedStateSink.write` through per-chunk index tensors (`paged_scatter_rows`) | per user (own pages / own ring rows) | nothing: ids of inactive users are `SKIP` (length 0) | unchanged |
| ratio-2 compressor "previous token" `attn.prev_cs` [1,1,U,1024] | `PagedStateSink.write` (blend with per-user 0/1 mask) | per user | nothing (mask 0) | unchanged |
| decode page table | `pool.sync_page_table` | per user | `admit_users(active=...)` only (re)admits active users; BUT users that were never admitted made `pool.ensure` (decode) raise KeyError | all users stay admitted with >= 1 page (`_admit_idle`, `release_user`) |
| Engram token history (host, `hasher.st.cache`) | `RaggedNgramHash` | per user (row b) | the hash of the whole padded batch wrote the history of every user (old code saved / restored the inactive rows) | hash only the active rows (`rows=`), nobody else is read / written; Engram rows are gathered for the active users only |
| **decode index-key slabs `dec.k_cache` [U,1,n_alloc,128]** (key owners 2/8/14/20) | `Model._export_index_keys` after every prefill | slab shape is per user, but the export was ONE `slice_write` of the WHOLE prefill key slab | **overwrote the keys of every user of the row (live decoders got filler keys)**: this is the reason the adapter re-prefilled all live users from their token history whenever the indexer was on (contexts > 512 / 1024) | per-user, per-mesh-row masked export (`_export_user_keys`): only the finishing users' rows are rewritten (bfp8->bf16->bfp8 of the others is exact) |
| prefill carried state: window halo [U,1,128,512], dense latent FIFO `lat_buf`, sparse kv table `PrefillKV` latent rows, prefill key FIFO `ix.keys` | per layer, persistent, shifted by one chunk per replay | per user (slot u of the row), prefill-private (the decode never reads it) | every replay shifted / overwrote the state of ALL users with filler compute: a prompt could not be resumed after any replay it did not take part in | `DSV41_PF_UMASK=1`: all writes of this state are gated in-trace by a persistent [U,1,1,1] per-row user mask (`ttnn.where`, exact selection), refreshed before each replay |
| decode padding rows | decode feeds every user each step; "idle" users get token 0 at position 0 | per user | position 0 of a user that is mid-prefill would overwrite its ring row 0, its layer-20 latent 0 and its Engram history of the first token; `prev_cs` of any user is rewritten by every decode step | parked position: a user that is being prefilled in chunks is fed token 0 at `end` (its first not-yet-written position; benign: rewritten by the next chunk / the first real decode step). `prev_cs` only matters from the last prefill chunk to the first decode step, and the plugin runs only prefill steps in between |
| decode trace | `Model.trace_id`, released by the adapter before EVERY prefill (shared DRAM, "persistent tensors created after a capture land on the freed intermediates") | global | recaptured by the next decode (seconds) | never released while the prefill trace is stable: every persistent tensor (sink indices, masks, dyn context) is allocated before the first capture; released only when the prefill trace is re-captured (new chunk / S_pad). **A persistent tensor that a captured trace references must never be re-allocated**: `ttnn.experimental.slice_write` is NOT in place (the tensor gets a new buffer address, the old one is freed; measured with `buffer_address()`), which is exactly why the old code had to release the decode trace after every prefill (``export_keys`` rewrites the decode key slab with it). The interleaved export therefore rebuilds the slab from its parts and `ttnn.copy`s it into the existing buffer. |
| prefill trace shape | traced for fixed (C, S_pad); latent / key FIFOs have length S_pad / ratio | global | S_pad bucket growth re-captures (and invalidates every carried state) | S_pad is monotonic per process: set `DSV41_VLLM_S_PAD=<max prompt>` to avoid re-captures |
| position of the replay (s0) | RoPE / masks / FIFO `off` / sink index tensors | global to a replay | users at different offsets cannot share a replay | a call with continuations at different offsets runs one window sequence per distinct start (state is protected by the mask) |

### What the device tests found (and fixed) on the way
1. `slice_write` into the decode index-key slab re-allocated it: with the decode trace kept alive the trace kept using the freed buffer and the new buffer was overwritten by the
   next replay of the prefill trace (symptom: the decode logits of the running users deviated at noise level (PCC 0.99) after the first interleaved window, and the whole
   key slab read back as zeros / NaN). Fixed in `Model._export_user_keys` (in-place `ttnn.copy` of the rebuilt slab). After the fix decode logits are bit-identical.
2. Adapter bug found by the adapter-level device test: a continuation chunk arriving in a logical slot that still belongs to a running request must not be claimed; the test
   now uses only free slots (the plugin never hands out an occupied one).
3. The old adapter admitted only the active users into the page allocator (`admit_users(active=...)`): `pool.ensure` raised KeyError for never-admitted users at the first
   decode. All users now stay admitted with one page.

## 2. Design

* `Model.prefill_interleaved(items, chunk, s_pad_max, want_logits)`: items = `[(user, tokens, start, end)]`. The traced chunk is captured once on dummy, fully masked inputs
  (`DSV41PrefillModel.capture_dyn`), then only REPLAYED (`replay_window`): per window the participants' tokens / Engram rows are uploaded (others: zero filler), the
  sink index tensors and the user mask are refreshed, the trace is replayed, the ragged head reads the first token of the users whose last token is in the window.
  The index keys of the users that finish in the window are exported right after it (the prefill key FIFO is shifted by every later replay).
  `start` must be 0 or the carried position `pf_resume[user]` (the window the user stopped at: valid while no replay overwrote its state; with the mask never).
* The chunk granularity is the model chunk C (multiple of 128). A chunk whose end is not a multiple of C pads its last window and is the END of a prompt (a
  continuation after it is recomputed from position 0 by the adapter: always correct, never fast).
* Adapter (`tt/generator_vllm.py`, plugin contract in `docs/SCHEDULING.md` of vllm-tt-plugin): `prompt_lens` of `prefill_forward` is the END of the scheduled chunk and
  `start_pos` its start; `tokens` hold the prompt from position 0. The plugin re-picks the state slot of a partly prefilled request every step, so a continuation is
  recognised by its token prefix (`vllm_state.PrefillTracker`) and the slot map is re-bound to its model user (`SlotTable.bind`). A continuation that cannot be resumed is recomputed
  from 0. The first token of an intermediate chunk is meaningless (the plugin discards it, and forces host sampling for such a step).
* Cost of a chunk step. With the default `DSV41_PREFILL_UP` = U a replay processes U x C tokens per mesh row whatever the number of new users (the other users are masked fillers,
  not skipped). `DSV41_PREFILL_UP=Up` (< U) builds the prefill trace (attention / sparse tables / hand-off sink / head / Engram inputs) for Up users per mesh row instead:
  a replay then costs Up x C row-tokens; a new request takes a free **prefill slot** of its mesh row (`Model._slot_alloc`; its carried prefill state lives in the slot until it
  finishes or another prompt of that row evicts it, an evicted continuation is recomputed from position 0), the hand-off sink writes slot i of row r into the pool / `prev_cs` /
  decode index keys of the decode user it holds (`PagedStateSink.set_map`: ring / latent index tensors and one-hot prev_cs masks per slot), users of the same mesh row beyond Up
  are processed in further waves, and the adapter spreads concurrent new requests over the mesh rows (`SlotTable.claim_balanced`: it swaps the model users of free logical slots,
  which is invisible to the plugin). The whole-batch `prefill_forward` is routed through the same machinery when Up < U. Up = 1 means 4 concurrent prompts (one per mesh row) per
  replay at no filler cost, independent of the batch size (the trace does not depend on U any more), and a smaller prefill scratch (halo, latent / key FIFOs: x Up / U).

## 2b. Measurements (host .47, random prompts, `tests/test_interleave_device.py`; logs /mnt/tt-data/ssinghal/dsv4-logs/pf_pf_interleave_*.log)

| config | what | result |
|---|---|---|
| 40 layers, B=16 (U=4), ISL 3712, chunk 512, `DSV41_PREFILL_UP=1`, indexer on | 15 users decode, a 16th prompt prefilled in 8 chunk steps between decode steps | decode logits of the 15 users bit-identical to the uninterrupted run (8 steps + 4 joint steps), decode state (pool, rings, prev_cs, index keys) bit-identical before / after each of the 8 windows, first-token logits of the chunked prompt bit-identical to a one-call prefill; adapter run (slot changes between chunks, parking) same first token |
| same | time per interleaved chunk step | 0.70 - 0.79 s (replay 0.62 s = 512 row-tokens at 1.2 ms, host 0.05 s, key export 0.01 - 0.12 s); 8 steps = 5.9 s for the whole prompt vs 5.7 s in one call; decode step of the 15 running users 52 ms (device sampling) |
| same, Up = U = 4 (estimate: replay x4 at fixed row-tokens/replay) | chunk step | about 2.5 s (4 layers: 0.34 s vs 0.08 s measured) |
| baseline (grid, 40 layers, B=16, ISL 3.7k) | one new request = whole-batch prefill | TTFT 20.2 s; with the decode indexer on and other requests live the old adapter additionally re-prefilled them |
| 4 layers, B=128 (U=32), `DSV41_PREFILL_UP=1` | 127 requests in one prefill call / one new prompt in chunks | 22 s / chunk steps 0.4 s (0.13 s after the logits read-back fix); decode bit-identical, state untouched |
| 12 layers, B=16, ISL 3712 | 15 requests in one call (4 waves x 8 windows) | 139 s incl. 64 s compile + capture; chunk steps 0.56 - 0.65 s before the read-back fix (replay 0.19 s) |

## 3. Serving recipe (tt-inference-server / vllm-tt-plugin)

Environment of the server process (the capability is read when the bundle module is imported, so it must be set before the plugin loads it):

    DSV41_VLLM_INTERLEAVE=1        # supports_chunked_prefill + interleaved prefill; default OFF (the older whole-batch / re-prefill behaviour)
    DSV41_PREFILL_UP=1             # prefill slots per mesh row (1: 4 prompts per replay, one per mesh row); default = users per row
    DSV41_VLLM_CHUNK=512           # tokens per user per prefill window (multiple of 128)
    DSV41_VLLM_S_PAD=<longest prompt rounded up to the chunk>   # fixed trace shape; a longer prompt re-captures the traces (and recomputes the prompts in flight)
    vLLM: --max-num-batched-tokens 2048 (= 4 mesh rows x chunk) --long-prefill-token-threshold 512 (= chunk) --enable-chunked-prefill, block_size 128

Every prefill step of the plugin then carries up to 4 prompt chunks (one per mesh row) that share one replay (`Up` x `chunk` row-tokens = chunk x 1.1 ms/row-token x layers
fraction), the plugin's decode-interleave policy (`decode_interleave_prefill_steps` / `decode_interleave_decode_steps`) alternates decode steps in between.

## 4. Limits

* Chunks must end on a multiple of the model chunk (`DSV41_VLLM_CHUNK`) to be resumable; any other continuation is recomputed from position 0 (always correct, never fast).
* A chunk step costs one replay of Up x chunk row-tokens (measured below) plus ~0.1 s of host work (Engram gather, uploads, head read-back, key export) whatever the number of
  decoding users; the decode steps of the running requests wait for it (the plugin never mixes prefill and decode in a step).
* More concurrent prompts in progress than Up per mesh row: the surplus is evicted / recomputed (correct, slow); the adapter spreads new requests over the rows.
* The index keys (decode indexer on, contexts > 512 / 1024) are exported per window end (the last window of every call), so a call that ends mid-prompt exports keys that the
  next call rewrites.
* S_pad is fixed per process (DRAM): a longer prompt than the traced S_pad re-captures the prefill trace (and drops the decode trace).
