# Served GLM-5.3-Flash: slot-level automatic prefix caching (APC)

Design + v1 implementation (2026-10-27, branch `arg/glm53_served_apc`, written without device access: the (2, 4)
block test below is the first device run). Scope: text, image and video prefixes through the vLLM TT plugin.

## 1. What exists today and why APC is off

* `tt/generator_vllm.py` declares `supports_prefix_caching = False`; the catalog passes
  `no-enable-prefix-caching: true`. The plugin (`platform.py` ~2128-2165) enables vLLM's APC only when the model
  declares the capability.
* The device KV is per user **slot**, not per block: `TtLatentCache` holds `blocks_per_seq` latent pages per slot
  behind an identity page table, the pooled-key index contiguous per slot (`LogicalBlockTable`: vLLM's block ids
  never address device memory; `check_row` validates shape / uniqueness only). 1 M profile = 4 slots per mesh
  (4 users per row on 4 columns, the 8 rows replicate), 128 K profile = 28 per row.
* The 34 KDA layers keep ONE recurrent state per slot (`[B, H/cols, 128, 128]` fp32 + conv taps
  `[3, B, 6144]` bf16 per chip, every chip of a row holds every user of the row). There is no per-token KV for
  them, and the traced decode step advances EVERY slot's state in place, idle slots included (parked at
  `(pad, position 0)`: `SlotTable.device_feed`). So a finished request's state exists at one position only, and
  only until the next decode step of anybody.
* The 11 DSA layers are paged per position (latent row + pooled key per `index_kpool = 4` positions + a 4-row
  ring of the open pool). A chunk may start at any 32-multiple (`FunctionalDecoder.prefill_forward`: `start % 32
  == 0`, `start == next_start[user]`, `TtLatentCache.fill` from page `start // 32`, `maintain_prefill` opens a
  fresh pool at `start % 4 == 0` so the ring content is irrelevant there).
* The plugin hands a cached prefix to the model as a plain prefill with `start_pos = num_computed_tokens = N`
  and the FULL token row (`model_runner.py` ~2105-2120: "APC hit: start_pos=cached"), on a freshly allocated
  logical slot (`_alloc_prefill_state_slots`). No block hashes, no mm identifiers reach the model; an mm item
  fully inside `[0, N)` arrives with its pixels stripped (vLLM `strip_covered_mm_data`, `NewRequestData.from_request`).

## 2. Design: content chains + KDA snapshots per physical slot

### (a) Resident content and its KDA position

`tt/vllm_config.py::SlotPrefixCache` keeps, per physical slot `(row, user)`, a `SlotContent`: the token ids the
slot consumed (`tokens[:length]`: prefilled chunks + every decoded token; the latent pages / pools are valid for
`[0, length)`), the multimodal spans `MMSpan(offset, length, kind, identifier)` among them, the positions at
which a KDA snapshot exists (`snapshots: position -> ring entry`), `live` and `last_used` (LRU tick).

The chain is compared on token ids (`torch.equal` over `[0, N)`, 1 M ints in ~1 ms) plus the spans: an mm span
intersecting `[0, N)` matches only with equal `(offset, length, kind)` AND equal, known identifiers. vLLM's block
hashes are mm-aware through exactly those identifiers (`kv_cache_utils._gen_mm_extra_hash_keys`: `(identifier,
offset)` per item), so the adapter's decision is at least as strict as vLLM's; it never hashes pixels.

Why not vLLM's block hashes directly: the plugin does not pass them, and recomputing them would need the mm
identifiers anyway. The token-id chain + identifier spans is the same information at block_size granularity.

### (b) The hit: which slot, which position

A request admitted with `num_computed_tokens = N > 0` lands in `prefill_forward` with `start_pos = N` on a
logical slot the adapter has no state for. `SlotPrefixCache.lookup(tokens, spans, N)` picks a **retained** (not
live) slot whose content equals the first `N` tokens and holds a snapshot at position `s <= N`; the highest `s`
wins, then the most recently used slot. The slot is taken (`SlotTable.allocate(logical, at=(row, user))`), the
snapshot restored into the live KDA state (`GLM53MeshModel.restore_kda(user, entry, s)`, which also re-arms
`next_start[user] = s`), and `[s, end)` is prefilled there through the ordinary eager chunk walk
(`PrefillPlanner.plan(s, end)`: `s` is a 32-multiple, the 16 K device chunks keep their pad-row buckets). The
rows `[s, N)` are recomputed (their latent rows rewritten with the same values); the hit is served from `s`.

KDA position rule: the snapshot IS the state at `s`; the live state of a retained slot is never trusted (the
parked decode steps moved it).

Multi-turn case: a request finishes after consuming `S = p + n - 1` tokens (prompt `p`, `n` outputs, the last
output token is never fed). vLLM caches the full blocks of the computed tokens, so the next turn (`old prompt +
response + new turn`) hits at `N = floor(S / 32) * 32`. The adapter snapshots a slot at every 32-aligned decode
crossing (`decode_forward`: after the step that moves the slot to a 32-multiple), overwriting ring entry 0, so
at finish entry 0 holds exactly position `N` unless the response never crossed a boundary, i.e. `N <= p` (short
answers). For those, the eager prefill can end its last chunk on the prompt's last 32-boundary first
(`GLM53_VLLM_APC_TAIL_SPLIT=1`, `PrefillPlanner.walk(split_tail=True)`: one extra device chunk of < 32 tokens,
padded to the 256 bucket: at 1 M context that chunk still pays the per-chunk whole-prefix latent copy, ~70 ms
over 11 layers, so it is a knob, default off; the traced prefill (prompts <= 2048) is never split: a short
prompt re-prefills faster than any split).

### (c) The miss and where the downgrade lives

When no retained slot serves the hit (the slot was reused; its content is past `N` with no snapshot at or below
it; the identifiers are unknown), the adapter prefills `[0, end)` in a fresh slot. This needs no plugin change:
the plugin gives the model the whole token row and expects the device to be at `end` afterwards, which it is;
vLLM's `num_computed_tokens` bookkeeping stays valid. Costs: the TTFT of a miss, and `usage.prompt_tokens_details.
cached_tokens` over-reports (vLLM counted `N`).

The one case the adapter cannot absorb: an mm item fully inside `[0, N)` arrived without its pixels. v1 raises
`RuntimeError("... cannot be served ...")` with the count in `apc.stats["refused_mm"]`. Nothing generic in today's
plugin expresses "the model refuses this hit": the lane scheduler's downgrade (`lane_scheduler.py` ~632) is the
preemption resume (the request re-prefills from 0 because its KV was freed: `num_computed_tokens` reset), which
is a scheduler-side event the model cannot trigger; `reset_prefix_cache` (`scheduler.py` ~955, patched in
`platform.py` ~1144) drops EVERY cached block and is reachable from the engine layer only; no model hook runs
before `NewRequestData.from_request` strips the payload. Hence the proposal in section 5: the scheduler asks the
model to confirm a hit (the executor is in-process: `distributed_executor_backend = "uni"`, `platform.py` 589),
and forwards the mm identifiers. With it, text misses stop being futile and mm hits are exact.

### (d) Snapshot bytes and how many fit

One snapshot = one user's KDA carries on one chip: 34 x (16 x 128 x 128 x 4 B + 3 x 6144 x 2 B) = 36 904 960 B =
**35.2 MiB per chip** (`memory_plan.kda_state_bytes_per_user`; 1.10 GiB over the 32 chips). Every chip of a row
holds every user of the row, so a ring of `E` entries per slot costs `users_per_row x E x 35.2 MiB` per chip
(`snapshot_bytes_per_chip`; `MemoryPlan.snapshot_entries` / `max_snapshot_entries` / `spare`). Planner (8x4,
partitioned, no scratch slot, bfp8 experts, 512 MiB trace region):

| profile | spare per chip | one entry per slot | entries per slot that fit |
|---|---|---|---|
| 1 M x 4 users per row | 124 MiB | 141 MiB | **0** |
| 512 K x 8 | 145 MiB | 282 MiB | 0 |
| 256 K x 16 | 74 MiB | 563 MiB | 0 |
| 128 K x 28 | 1 464 MiB | 985 MiB | **1** (24 users per row fit 2, 21 fit 4, 16 fit 8) |

Caveat the planner inherits: it counts the LIVE KDA state per `users_per_row / cols` slots per chip, but the
batched state holds every user of the row (4 x 35 MiB = 141 MiB actual vs 35 modelled at the 1 M profile, 985 vs
246 MiB at 128 K x 28). The ring term above is counted correctly (per chip for every user); the live-state
undercount is left as found (the "other" calibration absorbed it at 1 M x 1 slot) and noted here for the planner
owner.

Where no ring fits, the **host store** keeps the snapshots in host memory (`HostKdaSnapshotStore`: 1.10 GiB per
entry per slot on the 8x4; 32 chip reads of 1 MiB per layer per save, i.e. ~34 mesh reads per crossing, and the
mirror uploads per restore). The save runs once per 32 decode steps per live slot (a few per cent of the step
budget at 4 users; measured on the block test), the restore once per hit (~100 ms of PCIe against the re-prefill
of up to 1 M tokens it replaces).

Ring policy (`SlotPrefixCache.plan_snapshot`): entry 0 rolls with the decode (every 32-aligned crossing), entries
1.. hold prefill positions (every eager device chunk end: 16 K multiples; the tail split; an aligned prompt end)
evicting the lowest position when full; with one entry, everything shares it. Partial-prefix hits (shared
document + different questions) resume from the highest chunk end at or below the divergence point and recompute
at most one 16 K chunk; with E = 1 only the multi-turn case is served.

**v1 ships**: the rolling decode entry + prefill chunk-end entries in a device ring sized by the planner
(`GLM53_VLLM_APC_SNAPSHOTS=0` = fit, capped at 8), falling back to one host entry per slot where nothing fits
(the 1 M x 4 profile); the tail split as an off-by-default knob.

### (e) Eviction, affinity, and vLLM's bookkeeping

* A live slot is pinned (never a candidate, never evicted). A finished request's slot is **retained**
  (`release_request` -> `SlotPrefixCache.release`): its content, pages and snapshots stay; it parks at its own
  content end (`device_feed(park=...)`: the idle decode writes its latent row / ring slot / pool past the content
  and never over row 0; the resumed prefill from `s` rewrites everything from `s` on, pools included). Content
  that fills the context cannot park past itself and is dropped.
* Placement of a fresh prompt (`SlotTable.allocate(slot_rank=...)`): column / row balance first, then a slot
  without resident content before the least recently used resident prefix (`slot_rank`). A hit takes its slot
  with `at=`.
* Parked slots cost decode time like a live user at their position (the DSA scorer ranks pools up to the position;
  the KDA step is position-free). Bounded by the profile's designed full load; an inactive-slot mask in the decode
  trace (the adapter's anticipated `set_active_slots` hook) would make parking free.
* vLLM's cache vs the device: vLLM keeps hashes of blocks the device no longer holds (a reused slot). v1 tolerates
  the mismatch (silent re-prefill for text, refusal for stripped mm). The consistent form is section 5's
  confirmation hook, which makes vLLM's `num_computed_tokens` equal what the device serves (and `cached_tokens`
  exact). A model-reported eviction by hash would need the hashes in the model (a larger plugin change); the
  full-paged alternative (vLLM-addressed latent pages + KDA states as `MambaSpec` "align" checkpoints, which this
  vLLM has: `mamba_cache_mode="align"`, `prefill_checkpoint_alignment`) is the v2 direction: it removes the
  slot-level content bookkeeping altogether but needs a device page table in every DSA kernel and a per-block
  snapshot pool (35 MiB per 32 tokens per chip: 1.1 GiB per 1 K tokens, so block-granular states are only
  affordable at the `prefill_checkpoint_alignment` granularity, e.g. 16 K).

Lanes: one engine per mesh, lanes off. The lane coordinator's prefix routing (`lane_prefix_cache_routing`) is
untouched and would compose (each lane's own vLLM cache + the shared adapter records).

## 3. v1 implementation (adapter side)

| piece | where |
|---|---|
| capability `supports_prefix_caching = GLM53_VLLM_PREFIX_CACHING` (default off) | `tt/generator_vllm.py` `model_capabilities` |
| `MMSpan`, `SlotContent`, `PrefixHit`, `SlotPrefixCache`, `mm_spans_of_row`, `mm_payload_missing` | `tt/vllm_config.py` |
| `SlotTable.allocate(at=, slot_rank=)`, `device_feed(park=)`, `PrefillPlanner.walk/plan(split_tail=)` | `tt/vllm_config.py` |
| `kda_state_bytes_per_user`, `snapshot_bytes_per_chip`, `MemoryPlan.snapshot_entries/snapshot_bytes/spare/max_snapshot_entries`, the table column | `tt/memory_plan.py` |
| `DeviceKdaSnapshotRing` (paged fill + conv mask blend), `HostKdaSnapshotStore` | `tt/kda_snapshot.py` (new) |
| `allocate_kda_snapshots`, `snapshot_kda`, `restore_kda`, `rewind_user` | `tt/mesh_model.py` |
| ring sizing at build (`_prefix_cache_plan`), admission in `prefill_forward`, chunk-end snapshots (`_prefill_one(on_chunk=)`), decode-crossing snapshots + parking + content records in `decode_forward`, retention in `release_request`, `confirm_prefix_cache_hit` | `tt/generator_vllm.py` |

Knobs: `GLM53_VLLM_PREFIX_CACHING=1`; `GLM53_VLLM_APC_SNAPSHOTS` (entries per slot, 0 = planner fit, cap 8);
`GLM53_VLLM_APC_SNAPSHOT_STORE=device|host`; `GLM53_VLLM_APC_TAIL_SPLIT=1`. Log lines: `[glm53-apc] hit ...` /
`[glm53-apc] miss ...` with the running `stats` (`hits`, `downgraded`, `refused_mm`, `snapshots`,
`retained_evicted`).

Catalog note (tt-inference-server `workflows/model_specs/dev/llm.yaml`, both `glm53_flash` profiles; dev-only):
replace `no-enable-prefix-caching: true` with `enable-prefix-caching: true`, keep `block_size: 32` (the hash /
hit granularity = the latent page), add `enable-prompt-tokens-details: true` (otherwise
`usage.prompt_tokens_details` is null and hits are invisible to the client), and in the env block
`GLM53_VLLM_PREFIX_CACHING: "1"` (+ `GLM53_VLLM_APC_SNAPSHOT_STORE: host` for the 1 M profile explicitly;
`GLM53_VLLM_APC_SNAPSHOTS: "1"` for the 128 K profile). `prefill_chunk_alignment: 256` stays: vLLM's hit `N` is
a 32-multiple (block aligned), the continuation chunks keep the 256 grid.

## 4. Tests

Host (`tests/test_serving_prefix_cache_host.py`, 12 tests, no ttnn at import): the capability gate in a
subprocess from inside the worktree, chains + mm spans, hit / miss / reused-slot downgrade / LRU / `at=`,
the ring policy, planner sizing (the table above) + the tail split, the adapter end to end on the fake mesh model
(multi-turn hit restores entry 0 at 320 and prefills `[320, 360)`; a miss prefills from 0 in a slot without
content; the tail split serves the short-answer case; chunk-end entries serve a 40 000-token shared prefix from
32 768; a stripped mm hit is refused, identified spans match; retained slots park at their end; APC off keeps
the contiguity error).

Device (`tests/test_serving_prefix_cache_mesh.py`, rows-4-5 block env):

```
GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 \
    pytest models/demos/glm53_flash/tests/test_serving_prefix_cache_mesh.py -s
```

Expected: `2 passed` (device ring, host store), ~10 min warm. Prompt A (1500 tokens, 1024-token chunks ->
snapshot at 1024), 40 greedy decode steps (crossings 1504, 1536 -> entry 0 at 1536) with a second user decoding
while A's slot is parked at 1540, then A + response + 50 new tokens with `start_pos = 1536`, served twice (the
first hit may compile the tail chunk, the second is the warm cost; the PERF line reports both): the adapter
restores entry 0 and prefills `[1536, 1590)`; the from-scratch prefill of the same 1590 tokens in the SAME
physical slot must agree: logits PCC >= 0.999 and equal greedy tokens over the prefill + 5 decode steps (a
differing token is accepted only at a reference top-2 gap < 0.05: the KDA chunk scan and the decode recurrence
are different op sequences, so bit identity is not the bar). Miss: one token altered inside the prefix with
`start_pos = 1536` -> downgraded prefill from 0 in a slot without content, bit-identical to its own from-scratch
run there. The test runs with `GLM53_VLLM_APC_ALLOC_CHECK=1` (section 6): a call site that changes the DRAM
footprint after the capture fails the test by name.

First failure candidates, in order: `paged_fill_cache` / `view` shape refusal in `DeviceKdaSnapshotRing.save`
(the ring view `[E*H*K/32, 1, 32, V]`; compare `_write_user_kda_state_slot`); the conv mask blend on
`[taps, E, W]` with `E` beyond one tile row; `ttnn.slice` of a ring entry; the parked decode of a retained slot
(positions > 2050 take the sparse path: not on this test's lengths); a PCC below 0.999 = the restored state is
wrong (run the host store first: it isolates the paged fill from the copy semantics).

## 6. Allocation discipline after the capture

The rule (tt-metal's own: `allocator.cpp` warns that a buffer allocated while a trace exists "may be corrupted
once a trace is executed if [it is] not released before then"): after the decode trace is captured, **no device
buffer the serving path creates may still be allocated when the next replay runs**, because the replay writes its
transient intermediates at the DRAM addresses they had at capture time, which the allocator has since handed out
again. Persistent buffers predate the capture; everything else is a transient freed before the next replay. The
2026-10-27 served hang (`scalar(49152)` allocated after the captures, clobbered by ~1200 replays, ttKDA wedged the
mesh) and the 2026-10-28 (2, 4) block APC run (correct numerics on the hit, then every NoC of the block wedged in
the first chunk of a prompt whose 512-bucket tail chunk had first been compiled after the capture; the triage input
`_logs/mesh/triage/AUTOTRIAGE_INPUT_apc_hang_20261028.md`) are this class. A program compiled for the first time
after the capture is in it too: tt-metal keeps each program's kernel binaries in a DRAM buffer from the same
allocator (`ProgramImpl::allocate_kernel_bin_buf_on_device`), so the boot warm-up must compile every shape the
served eager path can run.

What the serving path does about it:

| where | discipline |
|---|---|
| KDA snapshot stores (`tt/kda_snapshot.py`) | one set of persistent scratch / selector / page-table tensors per store (`SnapshotDims.plan`: `rec_row`, `conv_row`, `conv_live_rows`, the one-hot `sel_user` / `user_mask` / `user_keep`, `user_idx`, `live_pt`; the ring adds `conv_ring_rows`, `sel_entry`, `ring_mask` / `ring_keep`, `entry_idx`, `ring_pt`), allocated by `allocate_kda_snapshots` before the first capture. Every save / load writes into them (`ttnn.slice(output_tensor=)`, a one-hot HiFi4 `ttnn.matmul(optional_output_tensor=)` for the conv row, `paged_fill_cache` on the state views, `multiply` / `add` with `output_tensor=`); the per-call selectors are rewritten in place with `copy_host_to_device_tensor` (ordered on the command queue ahead of the ops). No `ttnn.slice` / `to_layout` / `permute` / `concat` temporaries remain; the host store reads the scratch rows to the host and refreshes them from the host. |
| a save after a traced replay | `GLM53FlashForCausalLM._snapshot(sync=True)` drains the device (`ttnn.synchronize_device`) before the step's first decode-crossing save and before a traced prefill chunk's save; `GLM53MeshModel.replay` is blocking anyway (`FDMeshCommandQueue::enqueue_trace` finishes the queue), the explicit sync keeps the order independent of that default. Eager chunk-end saves need none (same queue, in order). |
| lazily built device constants on the eager prefill path (`GLMMesh.lazy` / `lazy_const`: the DSA `_cu_window`, Q offsets, dense/sparse row masks, per-user page-table views, the indexer's bounds / tail ids / pool offsets / sentinels, the MoE zeros / iota / pad columns, `mesh.const`) | built before the first capture they are cached for the model's life; built under live traces they are **chunk transients**: handed to the caller, never cached, released at the end of the chunk (`GLM53MeshModel.prefill_streams` / `decode_step` call `release_transients`). The first post-capture build of a key logs once (INFO) which geometry the warm-up missed. The freeze flag and the transient list are per device (`GLMMesh._runtime_state`), so the per-layer meshes `FunctionalDecoder.from_state_dict` builds see the model's capture (before 2026-10-28 the `scalar` / `chrono_scalar` guards in `tt/kda.py` checked a flag the capture never set). |
| the from-0 chunk's zero KDA state | freed with the chunk's temporaries (`FunctionalDecoder._state_for_prefill`); it was never deallocated before. |
| the warm-up ladder | `warmup_model_prefill` walks the listed lengths (`GLM53_VLLM_WARMUP_PREFILL`) and then every eager chunk bucket (`GLM53_PREFILL_BUCKETS`) those lengths do not reach, as a prompt of exactly that length into slot 0, so every padded chunk shape has its programs and its per-length ttKDA prefill layer before any capture. |
| the check | `GLM53_VLLM_APC_ALLOC_CHECK=1` (default off): the adapter records the DRAM allocator's bytes per bank after the decode capture (`GLM53MeshModel.dram_allocated`) and compares after every snapshot save and hit restore (against the footprint just before the call) and after every prefill row and `release_request` (against the baseline); a difference raises with the call site named, equality logs at INFO. `tests/test_serving_prefix_cache_mesh.py` runs with it on. The decode-side persistent buffers (`_top_buf`, `_score_buf`, `_compact`, keyed by the fixed batch) are allocated by the eager decode warm-up, before the capture. |

What stays open: a chunk geometry the warm-up did not cover still compiles its programs after the capture, and
their kernel-binary buffers are persistent; the check names the chunk, the remedy is the warm-up set (lengths and
starts), not the model code. The mixed dense/sparse chunk's row mask is keyed by its dense row count (`n_d`), which
depends on the chunk start: under live traces it is a chunk transient like the rest.

## 5. Plugin side (one generic change: vllm-tt-plugin#261, `mm_identifiers`)

The plugin already reports vLLM's prefix-cache hit to the model: a hit request arrives at ``prefill_forward`` with
``start_pos = num_computed_tokens`` (``docs/MODEL_EXECUTION.md``: "Prefix-cache hits and resumed requests can
also have nonzero start_pos"; the lane coordinator even routes a request to the lane holding its cached prefix,
#249). The adapter serves that hit from a retained slot or re-prefills from 0 inside the same call, so vLLM's
bookkeeping is always honoured and no pre-scheduling confirmation is needed (a "confirms_prefix_cache_hits"
capability was drafted on 2026-10-28 and dropped: it duplicated what the plugin does).

The one piece the adapter cannot derive is vLLM's content identifier of a multimodal item: the token ids of an
image run are all ``image_token_id``, so two images cannot be told apart by the ids, and a prefix-cached item
arrives with its payload stripped. ``mm_identifiers`` (vllm-tt-plugin#261) appends that identifier after the two
placeholder fields; ``mm_spans_of_row`` reads it when present. Until it merges, a hit on a prompt with images is
refused (the mm refusal) and the request is prefilled from 0. Two requests hitting the same slot in one step: the
first takes it, the second's lookup falls back (text: re-prefill; mm: the refusal).
