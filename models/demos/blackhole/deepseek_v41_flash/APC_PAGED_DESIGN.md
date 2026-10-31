# DeepSeek-V4.1-Flash: device-resident, block-paged automatic prefix caching (design note and effort estimate)

Written 2026-10-31 on `ssinghal/dsv4p1-apc-paged` (base: tt-metal `ssinghal/dsv4p1-b8-final` 95a82b1cadf; plugin `ssinghal/dsv4p1-b8-final-plugin` fd7ed0c; catalog 8c3d3d6fc).
Read-only analysis plus one CPU simulation (`tools/apc_paged_sim.py`, logs `/mnt/tt-data/ssinghal/dsv4-logs/apc_paged/`). **No model change, no device run**: all seven hosts were busy (`docker ps`, `fuser /tmp/dsv4_dev.lock`, 2026-10-31).

Path prefixes: `M/` = `models/demos/blackhole/deepseek_v41_flash/tt/`, `D/` = `models/demos/blackhole/deepseek_v41_flash/`, `P/` = plugin `src/vllm_tt_plugin/`.
Evidence tags: **[V]** read in the code at the cited line, **[M]** measured (log cited), **[I]** my inference or estimate (arithmetic shown).

## 0. Summary and the decision this note asks for

1. **Why the first device-resident plan (PAGED_PREFILL_APC_DESIGN.md Part 2, M1) was dropped** [V]: not a technical dead end. (a) 2026-10-19 direction: "if it requires a lot of DRAM, then it should be host, especially if > 16 users are cached" (Part 1, `Direction`); (b) prefill then read a dense per-user bf16 kv table, so pages alone did not carry the state and a bit-exact device hit needed a bf16 pool (+1.25 GiB/chip; 0.84 GiB free at 1M); (c) the cache had to survive in 4 per-row pools. Since then (b) is gone: paged read prefill (`DSV41_PF_PAGED=read`), the single chunk trace with a position-indexed key slab (`M/prefill_single.py:5-18`), per-slot chunk starts (`DSV41_PREFILL_UP_BATCH`) and the fp8 pool serving mode shipped in the B=8 catalog (`DSV41_FP8_SERVING=1`).
2. **What is block-additive and what is not** (section 1): the compressed latents (pool pages) and the index keys are per-block; the **only true tail state is the 40 window halos** (5 MiB per boundary per chip). I recommend (b): a small **tail record per 512-token chunk end** kept in device DRAM, restored by one on-device copy. Replaying the tail (a) is not exact (bounded replay = upstream's approximation), (c) page-addressed halos cost 40 KiB/token.
3. **Hit granularity = the 512 chunk on a global grid** (section 2). Pages stay 128 tokens (= the vLLM block).
4. **DRAM** (section 3): the pool pages cost **nothing extra**: the fp8 pool is built for 2 x 1M tokens per row (16,386 pages x 160 KiB = 2.5 GiB/chip) and a realistic agent load uses 3-12 % of it. The binding resource is the **halo store**: 336 KiB/bank per slot (layers 0-20) + 304 KiB/bank (layers 21-39, only inside the bounded-replay zone). 128 + 48 slots + 2048 key blocks = **about 66 MiB/bank** of the 350-415 MiB/bank free at B=8.
5. **Equivalence** (section 7): bit-equal to cold by construction (the hit reads the bytes the cold prefill wrote); test plan in section 7.
6. **Effort**: first useful milestone **M1 = 13-15 working days** (above the 5-day gate: stop for your go). A 5-6 day sub-step (M1a: CPU index + plugin hook + the 3 device kernels with unit tests) is possible but delivers no user-visible gain by itself.
7. **Finding that should change the order of work (section 9, simulation)**: most of the gain over today comes from the **hit grid** (512 instead of powers of two, tip + divergence points), **not from device residency**. A host-tier tweak "snapshot the last full chunk boundary of every prompt + the divergence point" (M0', 2-3 days, `DSV41_APC_POLICY=tip`) comes within 4 % (multi-turn agents, 4.21 vs 4.06 s) to 20 % (30k shared system prompt, 3.92 vs 3.27 s) of the device-resident TTFT in the simulation (today: 6.13 and 11.29 s), because the measured host restore is cheap (0.10 s at 4096 tokens, 0.27 s at 16k, 0.48 s at 32k). Device residency still wins on: restore 0.02 s vs 0.1-0.5 s, no D2H snapshot traffic and host RAM, one copy of a shared prefix for concurrent users, and it is the base for caching decode output and images later. It **loses its advantage when the home mesh row is busy** unless a cross-row copy (M3) exists.

## 1. The carried state, classified (question 1)

Layers: 0-1 window only; 2-19 ratio 2; 20-39 ratio 1; kv sources 2, 8, 14, 20; index-key owners 2, 8, 14, 20 (layers 24/28/32/36 score against layer 20's keys); `D/PAGED_PREFILL_APC_DESIGN.md` 1.1 [V].
What a prefill chunk at start `s` reads from the past, in the shipped B=8 config (paged read, single trace, fp8 pool, Up=2):

| State | Where (file:line) | Per block or tail | Size per chip | Hit (at boundary s) |
|---|---|---|---|---|
| Compressed latents of the 4 kv sources | paged pool, 320 rows per 128-token page `[src2 64 \| src8 64 \| src14 64 \| src20 128]` (`M/paged_ops.py:15-31`, `M/kv_paged.py:213-237`); written before the attention by `PagedStateSink.write_lat` (`M/prefill_handoff.py:185`) | **block-additive** (function of the prefix up to the block; shareable) | fp8 160 KiB per 128 tokens | **share the page ids** (page-table pointer, refcount); no copy |
| Index keys of the 4 owners (bf16, fp4-simulated) | single-trace slab per prefill slot and column: scratch rows `[0, Cc)` = the current chunk, history entry `j` at row `Cc + j` (`M/prefill_single.py:5-18`, `key_append` `:128`) | **block-additive** | 640 B/token total = 40 KiB per 512 tokens per chip (column-sharded) | copy store -> slab history rows (new kernel `key_load`); written to the store by a twin of `key_append` (`key_save`) |
| Window halo: KV rows of positions `s-128..s-1` of each of the 40 layers | `halo` `[Up,1,128,512]` bf16 tile, replicated over the columns, in-trace update gated by the user mask (`M/prefill_attention.py:111, 401-402, 492-498`) | **tail state**, but *it is one block of KV per layer*: the window equals one 128-token block | 5.0 MiB (128 KiB x 40) | **tail record** per chunk end (b); layers >= 21 only valid inside the replay zone (below) |
| Ring rows, `prev_cs` (ratio-2 compressor), drafter taps | pool tail rings; `prev_cs`; taps (`M/prefill_handoff.py:84-86, 228-252`, `M/prefill_taps.py:4-14`) | tail, rebuilt | - | rewritten by the replay of the last >= 128 positions (rule `s <= e - 128`, `M/prefix_cache.py:473-477`) [V] |
| Compressor partial group (ratio 2) | not carried: pairs are pooled inside the chunk (`M/prefill_attention.py:247-260`) | none at an even boundary | - | nothing |
| Engram n-gram history | host `hasher.st.cache` (`M/prefix_cache.py:384-387`) | pure function of the tokens | host | re-derived |
| Key-FIFO shift position | single trace: none (position-indexed slab: `nvalid`, `kofs` tensors per replay, `SingleCtx.update` `M/prefill_single.py:217`) | - | - | derived from `s` |
| mHC / RoPE / masks | per replay `DynCtx` (`M/prefill_dyn.py`) | none | - | derived from `s` |

**Bounded replay interplay** (`DSV41_PF_REPLAY=2560`, OPTIMIZATIONS.md 8.6, `M/prefill_model.py:34, 569`) [V]: layers 21-39 run only on the chunks whose start is >= `p0 = floor((e - 2560)/512) * 512`. A halo recorded at a chunk end before the zone is **valid for layers 0-20 only** ("partial": `PrefixSnapshot.full`). It is usable at `s <= p0` of the new prompt (the first zone chunk ignores the layers >= 21 halo), and a full halo at any `s`. The tail record therefore has two parts: part A (layers 0-20, 2.625 MiB/chip) always, part B (layers 21-39, 2.375 MiB/chip) only for the <= 5 chunk ends inside a zone. This cuts the halo store by 40 % and must be reproduced exactly.

**Options for the tail state:**
- (a) *replay a short suffix at every hit*: not exact. The replay of `[s-128, s)` needs the halo of `s-128`; clamping the window is upstream GPU vLLM's bounded replay (`vllm/models/deepseek_v41/attention.py:519-546` per PAGED_PREFILL_APC_DESIGN.md Q12) and is approximate. Needs the accuracy gate Q7 and your sign-off; I keep it as an optional extension (M5), not the base.
- **(b) a per-chunk-end tail record in device DRAM, restored by one copy (chosen).** The restore is a device-local copy (no host traffic, no CCL).
- (c) page-addressed halos: a halo for every 128-token block = 40 KiB/token bf16 (16x the latents) and an in-trace dump of all 40 layers every block. Rejected.

Halos exist only at **chunk ends** (the persistent tensor holds the state after the last replay), which is the second reason for the 512 grid.

## 2. Block size, chunk and what aligns (question 2)

- vLLM block = 128 tokens = one pool page (`block_size: "128"` in the catalog; `M/generator_vllm.py:12-16, 824-840` accepts only 128) [V].
- Model chunk C = 512 (`DSV41_VLLM_CHUNK`) = 4 pages = one trace replay per slot.
- Ratio 2: 128 tokens = 64 entries; 512 tokens = 256 entries = **32 rows per column x 8 columns = exactly one 32-row tile** of the column-sharded key slab; ratio 1: 512 entries = 64 rows per column = 2 tiles. This is why `resolve_server_default` requires `chunk % 512 == 0` (`M/prefill_single.py:94`) and why `SingleCtx.values` asserts `nloc % 32 == 0` (`:213`). A 128-token key copy is sub-tile (8 or 16 rows per column) and would need row-granular moves.
- A halo exists only at a chunk end.
- **Decision: hit granularity 512 on a global grid** (a hit continues at `s` = multiple of 512; every chunk end of every prompt is then a multiple of 512; a prompt's final partial chunk is not a boundary). Mean extra recompute at a divergence point = 256 tokens (about 0.15 s), against up to 50 % of the prompt on today's power-of-two grid. Latents and keys are *stored* per 128-token page / per chunk-block, so a finer grid can be added later by recording halos per block (M5) without changing the layout.

## 3. DRAM budget and the page pool (question 3)

**How pages are allocated today** [V]: one `PageAllocator` per mesh row (`M/kv_paged.py:239-304`): a LIFO free list, `pages[user]`, `admit/grow/rollback/release`, **no refcounts, no hashes**; `PagedKVPool.allocs` (`M/paged_ops.py:88`), user `b` lives on row `b // U`. vLLM block ids do not address device memory: `allocate_kv_cache` returns inert placeholders and the `page_table` argument is not read (`M/generator_vllm.py:12-16, 824-840`); vLLM only does admission accounting (`get_max_tokens_all_users` = 8 x 1,048,576 tokens).
**What changes**: the allocator becomes a refcounted, hash-indexed page cache (pages carry `(chunk-block key, refcount)`, an LRU of cached refcount-0 blocks, allocation takes never-cached pages first and then evicts the LRU leaf-first); `release(user)` decrements instead of freeing; the page table of a hitting user is `[shared pages..., fresh pages]`.

**Numbers (B=8 / 1M, fp8 pool)** [V from the code, I for the totals]:
- Pages per row: `users_per_row x ceil((max_len + 128)/128)` = 2 x 8,193 = **16,386** (`default_page_params`, `M/common.py:17`). One page = 320 rows x 512 B = **160 KiB per chip = 20 KiB per bank**; pool = 2.5 GiB/chip = 320 MiB/bank, **already allocated at boot**. B=4/1M: 8,193 pages per row.
- Agent load: 8 users x 8-30k tokens = 64k-240k tokens = 500-1,900 pages over the 4 rows = **3-12 % of the pool**. A finished request's pages simply stay cached (vLLM-style) until a long request needs them; pages therefore cost 0 extra DRAM and hold up to ~2M cached tokens per row.
- Chunk-block (512 tokens) = 4 pages (80 KiB/bank, already allocated) + key slot **40 KiB/chip = 5 KiB/bank** + optional halo record.
- **Halo record** (the DRAM driver): part A 21 x 128 KiB = 2.625 MiB/chip = **336 KiB/bank**; part B 19 x 128 KiB = 2.375 MiB/chip = **304 KiB/bank** (full = 5 MiB/chip = 640 KiB/bank).
- Blocks per MiB per bank: key slots 205 chunk-blocks (105k tokens); halo A 3.0 slots (1.5k boundary tokens); halo B 3.4 slots.
- **Proposed M1 budget per row**: halo A 128 slots (42 MiB/bank), halo B 48 slots (14 MiB/bank), key slots 2048 (10 MiB/bank, 1M tokens) = **66 MiB/bank** = 16-19 % of the 350-415 MiB/bank free at B=8. Env: `DSV41_APC_PAGED_HALO_A/B`, `DSV41_APC_PAGED_KEYBLOCKS`. B=16 (78 MiB/bank free) is out of scope.
- Reductions for M2 (not in M1): halos of layers >= 2 in fp8 (the sparse path reads the window through the fp8 pool scratch, so fp8 is lossless there; layers 0-1 use the bf16 halo in the dense SDPA and stay bf16): about -45 %; column-sharded records (the 128 halo rows are 4 tile rows, so shard the 512 features: 2 of the 16 tile columns per mesh column) -8x but need an all-gather over mesh axis 1 per restore (a CCL op in the restore path: hang-canary item).
- Capacity needs from the simulation (`dsv4-logs/apc_paged/sim_grid.txt`): 8 agent conversations fit in 128 halo slots per row; 16-32 conversations need 256 or thinning (`thin`: interior halos on every 4th chunk end).

**Trace-region/layout hazard**: every new persistent tensor must be allocated at build before any trace capture and changed in place only (section 8).

## 4. Index keys and `DSV41_IDX_KEY_SHARD` (question 4)

- Two key tensors exist: the **prefill slab** per slot (bf16, column-sharded, position-indexed in the single trace) and the **decode slab** per user (bfp8, sharded with `DSV41_IDX_KEY_SHARD=1`, exported from the prefill slab at prompt end by `_export_keys_users`, `M/dsv41_model.py:1571`). The decode slab is **lossy (bfp8) and static per user slot**: it can neither feed a bit-exact hit nor outlive its slot, and a prefill slab is overwritten by the next prompt of the slot. So keys need their own store.
- **Key store**: a bf16 tile tensor per owner `[K, 1, Cc8 x 32 rows..., 128]`-style slots (one slot = one chunk-block: 32 rows (ratio 2) or 64 rows (ratio 1) per column = 8 KiB or 16 KiB per chip per owner), 40 KiB per chip per chunk-block for the 4 owners, K = 2048 -> 80 MiB/chip. Written by `key_save` (a twin of `M/paged_kernels/key_append.cpp`: copy the slab scratch rows `[0, nrb*32)` of slot u to store slot `sid[u]`; **in the trace**, tensor-driven like `kofs`; SKIP for slots not in the replay) and read by `key_load` (store slots `sid[u][0..n)` -> slab history rows at `kofs`, one launch per owner, runtime loop count from a tensor). Both are JIT `generic_op` data-movement kernels (no C++ rebuild), tile-aligned by the 512 grid.
- Sharding: slab and store are both column-sharded identically (global slot p on column p % 8), so no cross-column traffic. Within a mesh row the copy is device-local. Across rows: section 6.
- The decode-slab export after a hit then exports the whole history `[0, e/ratio)` from the (restored + recomputed) prefill slab exactly as today.
- Paged keys for the prefill indexer (PF3, C++) is **not** needed.

## 5. Interplay (question 5)

- **Up=2 and the slots**: at B=8, `Up == U == 2`, so the prefill slot of a user is `b % 2` of its row (`restore_users`: `if model.Up != model.U ... else targets = (b // U, b % U)`, `M/prefix_cache.py:621`). A hit needs a **free user on the owning row**; `_place_row_aware` / `claim_balanced` (`M/generator_vllm.py:989, 1128`, `M/vllm_state.py:129`) get a sibling `claim_affine` (deepest hit, ties by load). `DSV41_PREFILL_ROW_AWARE` is inert at Up=2 (as in the catalog).
- **Per-slot starts**: `DSV41_PREFILL_UP_BATCH=1` already lets the two slots of a row run at different chunk starts in one replay (`M/prefill_single.py:44-55`); a hit slot starts at `s` while its neighbour is mid-prompt.
- **Bounded replay R = 2560**: the two-part halo record and the `s <= p0` rule above; cold == hit stays bit-equal as today.
- **Spec decode / decode writes**: decode and the replay write only positions >= `e`; shared pages are pages `[0, s/128)` with `s <= e - 128`, so **no write ever lands in a shared block** (the last partial page of the prompt is private and never cached; `rollback` frees only pages beyond the look-ahead, `M/kv_paged.py:280`). Copy-on-write is not needed in M1; it becomes necessary only when decode output blocks are cached (M4) or when a hit is allowed with `s > e - 128`.
- **Compression boundaries**: `s` is a multiple of 512, so no ratio-2 group is open and entry `s/2` is the first of a fresh page.
- **Sliding-window ring**: pool tail, per user, rewritten by the replayed tail (`s <= e - 128`).
- **Images**: image prompts keep skipping the cache (`skip_reading_prefix_cache`, plugin fd7ed0c; `PrefixCache._has_image`). **Key extensibility**: the block key is a sha256 chain over the *key tokens* of the prefix (`key_tokens(tokens, spans, image_token_id)` of the other agent's branch `ssinghal/dsv4p1-apc-img2` replaces every image span by a value derived from the image content hash). `BlockIndex.chain(key_tokens, signature)` takes that array as is, so image support later needs only the boundary rule (no block end inside an image span) and the same `vs` plumbing. The signature carries layer ids, ring rows, KV dtype, Up, mode, chunk and `DSV41_PF_REPLAY` as `M/dsv41_model.py:1693` does.

## 6. vLLM / plugin side (question 6)

Today [V]: the model hashes tokens itself (`PrefixStore._chain`), vLLM's block manager (one `BlockPool` of 65,536 blocks at B=8 = 8 x 1M tokens, accounting only) has its own hashes, the plugin **rounds** vLLM's hit to the model's grid (`P/scheduler.py:962-985` `_get_local_prefix_cache_hit`, `P/config.py:806-816` `round_tt_prefix_hit`, policy `pow2`/`all`), and the model ignores `start_pos` for a new request: it recomputes from 0 and takes its own snapshot at `st == 0` (`M/dsv41_model.py:1702-1722`, `M/generator_vllm.py:1100-1121`). The accounting is therefore a guess (the "hit the model cannot resume" stall class).

New: **model-owned cache, scheduler asks the model.** Engine core and worker share one process (`UniProcExecutor`, `P/launcher.py:505-514`, `P/platform.py:652`), so:
1. capability `prefix_cache_hit_policy = "model"`; the plugin registers a process-wide hook object (`vllm_tt_plugin.prefix_hooks`), the model fills it at load. (If scheduler and worker were ever in different processes, the same data would ride on `SchedulerOutput` fields like the reset flag `_TT_PREFIX_CACHE_RESET_ATTR`, `P/scheduler.py:61-66`; not needed with the uniproc executor.)
2. `_get_local_prefix_cache_hit` calls `hook.plan_hit(request_id, key_tokens, num_tokens)` -> `(hit_tokens, row)` and truncates vLLM's computed blocks to `min(hit, vLLM's own hit)` rounded to the 512 grid (vLLM's pool is far larger than the device cache, so its claim is almost always >= the model's). The query **pins** the blocks (refcount) and reserves a free user on the owning row; released at bind, at abort (`release_request`) or by a TTL;
3. `prefill_forward` trusts `start_pos > 0` for a request with a reservation (`hit_bound` set), binds it to the reserved user, and the window loop starts at `start_pos`; a reservation lost to eviction raises the same recompute-from-0 path as today (logged, counted);
4. eviction needs no callback to vLLM (the model is the truth; `reset_prefix_cache` already reaches the model, `P/scheduler.py:61-66, 906-910`); the vLLM prefix cache stays enabled only so that `request.block_hashes` exist;
5. image requests never reach the hook.

**Different mesh rows.** Pages, key store and halo store are per row (each row's chips hold their own copy). Options, measured by the simulation (`p_busy` = probability that the home row has no free user):
- row-affine placement (M1): free home row -> hit; else the deepest hit among rows with a free user, else miss. With a closed loop of <= 8 agents on 8 slots the home row is almost always free (the agent's own previous slot); with more conversations than slots (16-32) a freed slot goes to whoever is next and the home row is free about 25 % of the time: the TTFT gain falls to host-tier level (agents x8, busy 50 %: 5.6 s vs 4.1 s free vs host_pow2 6.1 s);
- **cross-row copy (M3)**: per hit, 4 pages x n chunk-blocks (160 KiB x 4 = 640 KiB/chip per chunk-block, 21 MiB at 16k) + keys + one 5 MiB halo = about 27 MiB/chip, 3 ms at 10 GB/s [I; the row-axis bandwidth is unmeasured]; `ttnn.experimental.send_async/recv_async` or an all_gather of a staging buffer over mesh axis 0; fabric ops are the hang-risk surface (OPTIMIZATIONS.md 8.17: the unified-MoE ring hang), so this is its own milestone;
- scheduler-level affinity (hold a waiting request until its home row frees): changes fairness; not proposed.

## 7. Exact equivalence and the test plan (question 7)

A hit reads the pages, keys and halo that the donor's cold prefill wrote, then runs the same chunks from `s`. It is bit-equal to the donor's own cold prefill; against a cold prefill of the *new* prompt it is equal up to the existing run-to-run noise of the stack (open problem 3 of the B=8 handoff: 2-4 distinct greedy texts per 5-10 identical requests, Up=2 batch composition). The gate is therefore: **hit logits == cold logits in the same slot configuration (bit-equal), and cold-vs-cold in different slots as the noise baseline.**

CPU (no device), `tests/test_apc_paged_index.py` and the plugin tests:
- chain hashing: stable, equal to `PrefixStore._chain` on the same tokens at 512, signature separation, image-style negative key tokens give different chains;
- refcount/pin/evict: shared pages freed only at refcount 0, LRU leaf-first, pinned blocks never evicted, the allocator never double-books a page; halo A/B accounting and the zone rule (`s <= p0` or full);
- hit binding with the fake model of `tests/test_apc_tracker_hit_binding.py`: a hit goes to the reserved user, a continuation never takes a hit, a lost reservation recomputes from 0;
- plugin: `plan_hit` rounding, pin release on abort, `reset_prefix_cache`.

Device, one host under `flock /tmp/dsv4_dev.lock`:
1. kernel unit tests: `key_save`/`key_load` and `halo_copy` round trips bit-identical to the source tensors (all 4 owners, 40 layers, random slots/users, SKIP slots untouched); pool bytes unchanged by a page-table share;
2. hit == cold logits, layers 4 / 21 / 40, prompts 2.6k / 4k / 16k, every hit length `s` in {512, ..., e - 128} on the 512 grid (partial-halo and full-halo cases around `p0`), last-token logits bit-equal, plus the decode state (rings, `prev_cs`, decode key slab, taps) and 16 greedy decode steps; extends `tests/test_apc_m2_snapshot_restore.py`, `tests/test_apc_m3_integration.py`; **Zone-rule exactness (added by the index agent)**: include hits with `b_zone < p0_new` and `s > p0_new` (halo B written by a prompt whose zone started earlier: layers 21-39 had more context before `p0_new` than a cold run); if the logits differ beyond the Up=2 cold-vs-cold noise, switch the index to `strict_zone_equal=True` (`b_zone == p0_new`), see `APC_PAGED_INDEX_INTERFACE.md`.
3. multi-turn growth: turns of +1.6k tokens to 16k (the `apc_suite` `i5b` shape) and a needle in the history, first token equals cold at every turn;
4. concurrency: 8 users with the same 8k system prompt and different tails, concurrently (refcount 8 on the shared pages), outputs equal to solo cold; shared pages byte-identical before and after all users decode (the no-write-into-shared-blocks check);
5. eviction under pressure: `DSV41_APC_PAGED_HALO_A=4`, a small page budget, 3 conversations; evicted block -> miss -> correct recompute; pinned block survives;
6. hang canary: 8 x 8k fresh random prompts x 30 iterations with `DSV41_APC_PAGED=1` (OPTIMIZATIONS.md 8.17 recipe), `DSV41_UNI_TOPO=linear`; then the `apc_suite` client chain (`/mnt/tt-data/ssinghal/dsv4-logs/apc_suite/clients`) as the text regression: repeat == cold, other-suffix hits, multi-turn, c8 identical 32k, aborts, images skip.
Every crash/kill: capture `ls /dev/tenstorrent | wc -l`, `tt-smi -ls`, `sudo dmesg | tail -50`, then `reset_host.sh`, then re-check the cache NVMe mount.

## 8. Milestones, effort, risks (question 8)

Switch: `DSV41_APC_PAGED=1` (default OFF; with it off every code path is the current host-snapshot APC). The host tier stays the fallback (a miss of the device cache may still hit a host snapshot; both can run, the device cache first).

| # | Scope | Effort (days, +-40 %) | Gate |
|---|---|---|---|
| M0 | this note, simulation, worktrees | done | - |
| M0' (optional, independent) | host tier policy `tip` + divergence snapshots at the 512 grid (`PrefixCache.wants` gets the prompt end; a hash trie of seen chunk blocks gives the divergence point) | 2-3 | `apc_suite` rerun; simulation says -31 % mean TTFT on multi-turn agents, -65 % on a 30k shared prompt (host_div column of section 9) |
| **M1a** | `tt/apc_paged.py` (BlockIndex per row, refcounted allocator, halo A/B slots, key slots, pins, zone rule) + CPU tests (2 d); plugin hook `prefix_cache_hit_policy="model"` + tests (1.5 d); kernels `key_save`, `key_load`, `halo_copy` + device unit tests (3 d) | **6.5** | CPU green; kernels bit-exact on a host |
| **M1b** | integration (4 d): stores allocated at build before capture, save hooks at chunk ends (replace `begin_snapshot`, `M/dsv41_model.py:2077`), hit binding replacing `_apc_restore_wave`, refcount in `release_user`, `claim_affine`, `warm_apc` compile-before-capture, MEMREPORT line; validation (3-4 d): section 7 items 2-6 | **7-8** | hit == cold bit-equal, canary clean, `apc_suite` pass |
| **M1 total** | row-local, bf16 halos, 512 grid | **13-15** (11-19) | - |
| M2 | fp8 halos for layers >= 2, thinning policy, per-slot metrics | 3-4 | needle/GSM8K equal |
| M3 | cross-row copy on hit (fabric) | 4-5 | hang canary with copies |
| M4 | cache decode output blocks (multi-turn), copy-on-write at the boundary | 4-6 | - |
| M5 | 128-token grid: bounded replay (accuracy gate Q7) or per-block halos | 3-5 | accuracy |
| M6 | images: key tokens + boundary rule (other agent's branch) | 2-3 | image repeat suite |

**Smallest useful first milestone**: M1 as above. Of the two candidates in the brief, "pool-page sharing + replay of the carried tail" is **not exact** (needs Q7) and "device-resident snapshot ring" has the same kernels and halo/key copies as M1 minus the page dedup (which is the cheap part), so M1 is the minimal exact device-resident design. If the 13-15 days are too much, M0' is the cheap step.

**Risks**
1. *Trace-layout hazard* (B=8 handoff open problem 1: at 40 layers a different DRAM layout changed logits and needles, the tracker raises `N device buffers still alive before trace replay`): the new tensors (stores, `sid`/slot-id tensors for `key_save`) shift the allocator. Mitigation: allocate next to the existing halos at build (`M/prefill_attention.py:401`, `M/prefill_sparse.py:1096 alloc_dyn`), update in place with `h2d` like `kofs`, no outputs from the new `generic_op`s, program warm-up before capture; judge against the trace-hazard branch's outcome.
2. Hangs: M1 uses no CCL (SPMD kernels, same slot id on the 8 columns); M3 adds fabric.
3. Determinism: hit vs cold shares the existing noise; the gate uses the cold-vs-cold baseline.
4. Slot/row availability decides the realised gain (section 6).
5. Reservation races between `plan_hit` and `prefill_forward` under async scheduling: pins plus a counted fallback to recompute-from-0.
6. DRAM: 66 MiB/bank of the 350 MiB/bank free; `MEMREPORT` after boot must show the largest free block still holds the biggest spec/vision allocation.
7. Host availability: all seven hosts were busy at the time of writing; device milestones need a free hold.

## 9. Expected gain (simulation, `tools/apc_paged_sim.py`)

Model: chunk replay 0.30 s (layers 0-20 only) / 0.52 s (all layers, chunk start inside the 2560 zone), fixed 0.12 s per request, fitted to the `apc_suite` colds (1,024: 1.17 s; 1,638: 2.22; 8,192: 5.8; 32k: 20.05; 64k: 39.2; 100k: 59.6; hit at 4,096 of 8,192: 3.64 s). Host restore from the server logs: 0.05 / 0.10 / 0.14 / 0.27 / 0.48 / 0.95 s at 512 / 4k / 8k / 16k / 32k / 64k tokens. Device restore assumed 0.02 s [I]. Sequential requests, unlimited host store, pages never the constraint; halo slots 128 per row; `pb` = probability that the home row has no free user; `x` = cross-row copy available. Mean TTFT in seconds [fraction of prompt tokens recomputed]:

| Workload | no APC | host pow2 (today) | host tip | host tip + divergence | device 512 (home free / pb 0.5 / pb 0.5 + copy) |
|---|---|---|---|---|---|
| 8 agents, 6k system prompt, turns +3.9k to 30k | 11.50 | 6.13 [0.44] | 4.42 [0.26] | 4.21 [0.25] | 4.06 [0.25] / 5.67 / 4.06 |
| 8 agents, 8k-16k, +1.6k per turn | 8.20 | 4.94 [0.49] | 3.47 [0.28] | 3.47 | 3.34 [0.28] / 4.93 / 3.33 |
| shared 6k system prompt + 2k tails x 64 | 6.32 | 4.06 [0.50] | 4.06 | 3.01 [0.32] | 2.91 [0.31] / 3.08 / 2.91 |
| shared 30k system prompt + 2k tails x 32 | 20.32 | 11.29 [0.50] | 11.29 | 3.92 [0.11] | 3.27 [0.10] / 4.39 / 3.27 |
| 16 agents (H = 128) | 10.97 | 6.07 | 4.44 | 4.19 | 7.18 (capacity) / 6.16 / 4.02; H = 256: 4.07 / 5.61 / 3.99 |

Measured examples for the multi-turn shape (`apc_suite/res/i5b.json`, +1,640 tokens per turn, TTFT s): turn 7 (11,478 tokens): cached 8,192, 3.75 s; turn 10 (16,398): cached 8,192, 6.78 s. On the 512 grid the hit would be at 9,728 and 14,336 (the previous prompts were 9,838 and 14,758 tokens): recompute 1,750 / 2,062 tokens = 4 / 5 chunks of 0.52 s, about 2.2 / 2.7 s [I] (1.7x / 2.5x faster). The 8k same-prompt repeat (3.64 s today, hit at 4,096) would recompute the last 128..640 tokens: about 0.8 s [I].
Caveats: single-request model (no queueing, no decode interference), `pb` is a parameter not a measurement, the 0.02 s device restore is an estimate, 16-32 conversations need 256 halo slots or thinning.

## 10. Files of this branch

- `D/APC_PAGED_DESIGN.md` (this note), `D/tools/apc_paged_sim.py` (CPU simulation; run with `python_env/bin/python -I tools/apc_paged_sim.py --H 128 256 --thin 1`).
- Plugin worktree `wt/apc_paged_plugin` (`ssinghal/dsv4p1-apc-paged-plugin`): untouched until M1a.
