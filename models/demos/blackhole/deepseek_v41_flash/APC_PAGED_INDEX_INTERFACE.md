# APC_PAGED: CPU block index and plugin hook, interface (M1a, branch ssinghal/dsv4p1-apc-paged-idx)

Owner: index agent. Spec: `APC_PAGED_DESIGN.md` (sections 2, 3, 6). Code: `models/demos/blackhole/deepseek_v41_flash/tt/apc_paged.py` (pure Python + numpy, no ttnn/torch import, so it runs in the plugin env and in CPU tests).
Switch: `DSV41_APC_PAGED=1` (default off). Off: nothing here is constructed, the plugin policy stays `pow2`, the host-snapshot path is untouched.
Everything below is what the model/device side may rely on. Changes after this doc are announced in the same file (changelog at the end).

## 1. Vocabulary

- chunk-block = 512 tokens = 4 pool pages of 128 = 1 key slot. Boundary `s` = a multiple of 512 (global grid). Block `i` (1-based) covers tokens `[512(i-1), 512 i)`; its key is the chained sha256 over the key tokens of `[0, 512 i)`.
- Per mesh row (`row` 0..3, "rows" is a constructor arg): its own page pool (`pages_per_row`, default 2 x 8193 = 16386 at B=8), key slots (`key_slots`, 2048), halo-A slots (128) and halo-B slots (48). Pages/slots are ids local to the row; all 8 columns use the same id (SPMD).
- halo record at boundary `s` = part A (layers 0-20, always) + part B (layers 21-39, only if the chunk that ended at `s` ran layers >= 21, i.e. its start >= `p0(e) = max(0, floor((e - zone)/512) * 512)`, `e` = prompt length, `zone` = `DSV41_PF_REPLAY` = 2560). A slot id of `-1` = "none / skip" everywhere.
- Hit rule (`lookup`): deepest `s` on the grid with `s <= e - 128` (`margin`), all blocks `1..s/512` published on the row, halo A at `s`, and (`s <= p0(e)` or halo B at `s`). Never a hit with `s == 0`.

## 2. Calls the model makes (model thread unless noted)

```
idx = BlockIndex(rows=4, users_per_row=2, pages_per_row=..., chunk=512, page=128, halo_a=128, halo_b=48, key_slots=2048,
                 zone=2560, margin=128, signature=<str as dsv41_model.py:1693>, free_users=<optional fn(row)->iterable of free in-row users>)
idx.set_signature(sig)                      # != current: drops everything (like PrefixStore.set_signature)
idx.add_listener(fn(event: dict))           # events: {"kind": "block"|"halo_a"|"halo_b", "row", "ids": [...], "why": "lru"|"reset"}
```

### 2.1 lookup / reservation (scheduler thread, via the plugin hook; the model never calls it directly)

```
plan = idx.plan_hit(req_id, key_tokens, num_tokens, max_hit=None, salt=b"", spans=None)   # idempotent per req_id (a re-plan replaces the old reservation)
plan.hit_tokens   # 0 or a multiple of 512, <= min(num_tokens - 128, max_hit)
plan.row          # owning mesh row (hit) or -1
plan.user         # reserved in-row user (hit) or -1  -> the model MUST prefill this request in (row, user)
plan.block_ids    # chunk-block ids (opaque ints, stable while pinned) 1..hit/512
plan.page_ids     # hit_tokens/128 pool page ids (the first entries of the request's page table; shared, refcounted)
plan.key_slots    # hit_tokens/512 key slot ids (history entry j <- key_slots[j]; copy for key_load)
plan.halo_a, plan.halo_b   # halo slot ids at the hit boundary (halo_b = -1 when s <= p0: part B not needed)
idx.cancel(req_id)                          # drop an UNBOUND reservation (abort / TTL); no-op once bound
```
`key_tokens`: 1-D int array over the WHOLE prompt (negative values for image spans from `key_tokens()` of the image agent are hashed as is; length = prompt length). `salt`: bytes folded into the chain head (mm_cache_salt / cache_salt). `spans`: optional `[(start, n, hash)]`; with `strict_spans=True` (constructor arg) no boundary inside a span (start < b < start+n) is a hit point or a commit point. Pins: every block on the hit path, the halo slots and the user stay held until `bind` + `restored` / `release` / TTL (`ttl_s`, default 60 s).

### 2.2 bind (start of `prefill_forward` for a NEW request, i.e. no PrefillTracker entry)

```
b = idx.bind(req_id, start_pos, user=None, prompt_len=None)
```
- planned reservation and `start_pos == plan.hit_tokens`: returns `Binding` = the plan fields above + `src_row` (== `row` in M1). Pins turn into the request's holds (shared pages refcounted for the life of the request; they are freed only at release).
- `start_pos == 0` with no reservation (a miss): `user` is required (the model's own claim; must not be a reserved user: `idx.reserved_users(row)`); returns a Binding with `hit_tokens=0`.
- reservation lost (evicted by `reset`, expired TTL, `start_pos` differs from the plan): returns `None`; counted in `stats()["lost"]`; the model recomputes from 0 exactly as today (and claims a user itself).
- Unknown request id with `start_pos > 0` that is not a tracker continuation: `None`.
`idx.restored(req_id)`: call right after the device restore copies (key_load, halo_copy) are ENQUEUED; releases the halo pins (the slots may be recycled by a later `begin_chunk`; device ops are ordered).

### 2.3 pages (per request)

```
table = idx.page_table(req_id, total_tokens)   # list of pool page ids: [shared hit pages..., private pages...], grows (never shrinks) to ceil(total_tokens/128); OutOfPages if nothing evictable
idx.rollback(req_id, total_tokens)             # frees private pages beyond ceil(total_tokens/128) (spec look-ahead rollback)
```
The model's `PageAllocator` is replaced by these (it still owns `page_table` tensors). Shared pages are never written (rule `s <= e - 128`, design 5).

### 2.4 chunk commit (store the chunk's keys + halo), TWO-PHASE because the saves run inside the trace

```
cp = idx.begin_chunk(req_id, start, end, key_tokens, prompt_len, salt=b"", spans=None)   # BEFORE the replay: ids go into the slot-id tensors
cp.key_slot   # key store slot to write (-1: skip: block already stored / dup / no slot)
cp.halo_a, cp.halo_b   # halo slots to write (-1 skip); halo_b only when the chunk started in the zone (cp.full)
cp.block_idx  # end/512
cp = None for: end % 512 != 0 (a prompt's final partial chunk is not a boundary), a boundary strictly inside a strict span
idx.finish_chunk(cp)   # AFTER the replay was enqueued: the block becomes visible to lookups (device order guarantees the data is there before any later restore)
idx.abort_chunk(cp)    # replay failed / request aborted: slots return to the free lists
```
`finish_chunk` order = chunk order. The request's private pages `[4(i-1), 4i)` become the block's pages (no copy); a duplicate (same key already stored by another request) keeps its pages private and only upgrades a partial halo to full when it brings part B (`cp.halo_b >= 0`, `cp.halo_a == -1`).

### 2.5 release

```
idx.release(req_id)         # request finished/aborted/preempted: private pages freed, holds dropped (idempotent)
idx.release_user(user)      # same, by the model's physical user id (the plugin's release_request(slot) only knows the slot)
idx.reset()                 # model.reset_prefix_cache(): all published blocks dropped; bound requests keep their pages until release; reservations are lost
```

### 2.6 introspection

`idx.stats()` (dict of counters: lookups, hits, hit_tokens, lost, evictions per kind, pages/slots free per row, halo A/B used) for the log line; `idx.peek(key_tokens, num_tokens)` = lookup without pins. `idx.reserved_users(row)`, `idx.free_users(row)`.

## 3. What I expect from the device side (INTERFACE_DEVICE.md is the reference; mismatches go there)

- `key_save(slot ids [Up, nmax])`, `key_load`, `halo_save/halo_copy` take slot ids as tensors uploaded with the per-replay dyn tensors (`h2d` like `kofs`); `-1` = skip for any entry. Slot id spaces: key slot `0..key_slots-1`, halo A `0..halo_a-1`, halo B `0..halo_b-1` (separate id spaces, per row). The id arrays I produce are plain Python ints.
- Restore of a hit (one per request, enqueued in the prefill call before its first window): `key_load` of `plan.key_slots` into slab history rows `[0, hit/512)` (in chunk-block order), halo copy of `plan.halo_a` (layers 0-20) and, when `>= 0`, `plan.halo_b` (layers 21-39) into the user's halo tensors. The page table of the user already starts with `plan.page_ids`.
- Pool pages and the page-table upload are the model's job; I only hand out ids (`page_table`). Page ids are row-local `0..pages_per_row-1`; the model offsets nothing.
- I need from the model at build: `rows`, `users_per_row`, `pages_per_row`, the slot capacities actually allocated (`DSV41_APC_PAGED_HALO_A/B`, `DSV41_APC_PAGED_KEYBLOCKS`), `zone` (`DSV41_PF_REPLAY`), signature string.
- The model registers `ApcPagedHook(idx, key_tokens_fn)` with the plugin (`vllm_tt_plugin.prefix_hooks.set_prefix_hook(hook)`); `key_tokens_fn(prompt_tokens, mm)` builds the image-aware key tokens (default: the tokens themselves; images never reach it today).

## 4. Plugin side (branch ssinghal/dsv4p1-apc-paged-idx-plugin)

- `model_capabilities["prefix_cache_hit_policy"] = "model"` (new value). Scheduler `_get_local_prefix_cache_hit`: runs vLLM's own lookup first (it gives `request.block_hashes` and the vLLM hit `num_local`), then `hook.plan_hit(request_id, prompt_token_ids, num_tokens, max_hit=num_local)` and truncates vLLM's computed blocks to the answer (a multiple of 512). No hook registered / hook returns 0: hit 0 (never fall back to vLLM's own hit in this mode; the host tier is the model's fallback).
- `num_computed_tokens` = hit, so `start_pos` of the new request in `prefill_forward` = hit. The model recognises the reservation by id: the runner passes `request_ids=[...]` (row i = request i of the prefill batch) only when the policy is `model`.
- Abort / finish / preemption of an unbound request: scheduler calls `hook.cancel(request_id)`. Bound requests are released by the model (`release_request(slot)` -> `idx.release_user`).
- `reset_prefix_cache` already reaches `model.reset_prefix_cache()` (flag on SchedulerOutput); the model calls `idx.reset()` there.
- Switch off (`DSV41_APC_PAGED` unset): the model advertises `pow2` as today; the `model` branch is dead code.

Changelog: v1 (initial).
