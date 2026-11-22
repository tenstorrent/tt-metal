# Prefix caching in the disaggregated prefill server, and what the decode server must change

Status (2026-11-21): prefill side written (`demo/prefill_server.py`, `tt/prefill/d_p_bridge.py`,
`tt/prefill/d_p_wire.py` protocol version 2, `demo/prefill_client.py --prefix-check`). Host logic is tested off
device: wire round trip, slot choice and LRU, and the offset readback of the compressed rows. Device validation is
pending. Decode side: not written. Section 3 describes it.

Background: `PREFILL_D_P_INTEGRATION.md` section 6.2 (the server, the protocol, the deployment on hosts 40 / 41).

## 1. What the prefill server does now

### 1.1 Device cache slots

`DPPrefill` holds several cache slots instead of one. Each slot is a one-user metal_2 `KvCaches` of `max_seq`
tokens (`adapter.allocate_kv_cache(num_users=1)`) plus its `DS41DeviceState`: the window rings, the snapshot,
the compressed KV and index-key slabs, and the engram id history. Slots are allocated before the cold-chunk compile.
That way the warm-up runs with the memory that serving will have, and a reserve that is too small fails at startup
rather than on a request.

* `--cache-slots N` (default 256) sets the most slots, and `--cache-reserve-gb G` (default 4) sets the DRAM a chip
  must keep free after them. The server stops adding slots before one would go below the reserve. It measures
  each slot's cost as the free-DRAM drop (`ttnn.get_memory_view(mesh, DRAM)`). The first slot's figure also counts
  the history-gather buffers every slot shares, so the estimate is conservative.
* At the 1M default a slot costs about 0.18 GB a chip: 3 x ratio-2 sources (compressed KV + index keys, ~33 MB),
  layer 20 (ratio 1, ~66 MB), the snapshots, and 21 x 2 window rings. DRAM, not the default of 256, normally sets
  the count. The startup line `N cache slot(s) of T tokens, M MiB a chip each, F GiB a chip of DRAM left` shows it.
* A slot's state stays on the device until a miss needs that slot. A miss takes an empty slot, else the least
  recently used one. Every request (hit or miss) updates its slot's use time.

### 1.2 What counts as a hit

The server keeps, per slot, the ids of the prompt whose state the slot holds (`PrefixCache`, host only). A new
prompt is matched against every slot: it takes the slot that lets it skip the most tokens, and the most recently used
slot on a tie.

A slot can only be continued *exactly* from a position `r` where every chunk from `r` on reads what the slot still
holds:

* compressed KV and index keys of `[0, r)`: always there, because positions are only overwritten by a later prompt
  that runs over them;
* the engram id history: kept per slot (`DS41DeviceState.note_tokens` recomputes the tail from it at any start);
* the window carry `[r - 128, r)` of layers 0-19. metal_2 keeps only the last 192 window rows a layer
  (`WINDOW_KV_KEEP=snapshot128`'s `WindowRing`, `RING = WINDOW + 64`), so the carry is there only when `r` is the
  slot's prompt end or up to 64 positions before it. `DPPrefill.resume_start` asks metal_2 itself
  (`TtWindowCache.carry_ready`) rather than restating this rule.

So the hits are:

| New prompt vs the kept prompt `K` (length `L`) | Result |
| --- | --- |
| starts with all of `K` (next turn of a conversation, a longer document) | hit: resumes at `L`, computes `[L, n)` |
| equals `K` (a retry, a second decode server with the same prompt) | hit: `cached = n`, nothing computed, the reply is read straight from the slot |
| shares only part of `K` (common system prompt, an edited earlier turn) | miss: computed from 0 in another slot (see section 4) |

A hit is exact: the same arithmetic as a cold prefill of the whole prompt, apart from where the chunk boundaries fall.
`prefill_client.py --prefix-check` compares a continued prompt's entries with a cold one's by PCC.

Continuing a slot takes it over: after a hit at `L` the slot holds the new prompt, and its window rings are at `n`.
A later prompt that continues `K` but not the new prompt then misses. This is the right trade for multi-turn chat,
where each turn extends the last, and the wrong one for branching. Section 4 covers copy-on-write.

### 1.3 Running a continued prompt

`DPPrefill.prefill(ids, slot=k, start=r)` keeps the chunks on the 5120 grid. If `r` is off the grid it runs a
partial first chunk `[r, ceil(r / 5120) x 5120)`, and the later chunks are the grid's. Every chunk start then
satisfies metal_2's `start + chunk <= max_seq` (`DS41PrefillRuntime.prefill_chunk`). When `r` itself is within one
chunk of `max_seq`, `resume_start` pulls it back to `max_seq - chunk`, the one start whose carry metal_2 pins, and
re-checks it.

The warm-up now also runs a prompt continued off the grid (`chunk + 128` tokens, then continued to twice that), so
the first real hit doesn't compile the partial-chunk programs. If metal_2 can't continue even that, the server logs
`a slot cannot continue its own ...-token prompt: every request will miss` and keeps serving without hits.

### 1.4 Sending only the new entries

The reply's compressed KV and index keys cover positions `[base, seq_len)` only, where `base` is the `have` the
client sent: the prompt's first `have` tokens whose entries the client already holds. `DPPrefill.host_states(T,
slot=, base=)` reads only the owner-slab blocks those positions reach (`_compressed_rows(c, T, base)`), so the
readback shrinks as well as the TCP transfer.

`have` is the client's statement about its own caches, and the server's cache hit doesn't change it. The two are
reported separately:

* `cached` (reply): tokens the server took from its prefix cache instead of computing. `cached > 0` is a hit, and
  `cached == seq_len` means nothing was computed.
* `base` (reply, always equal to the request's `have`): the first position whose entries are in the reply.

They are kept apart because the server can't know what a decode server still holds. Another decode server may
have filled the slot, or the session may have been rewound or evicted on the decode side. If the server sent
entries from its own hit point, a decode server without them would commit a hole. With `have`, a decode server
that holds nothing gets everything (`have = 0`, the version 1 behaviour), even on a cache hit.

The window tails (`kv_tail`, 128 rows a layer, about 2.7 MB in all) are always sent whole, because decode rewrites
its rings whole.

## 2. Protocol version 2 (`tt/prefill/d_p_wire.py`)

Changes from version 1 (the client refuses a server of another version, so both ends must update together):

* `info` adds `cache_slots`.
* `prefill` request header: `{"op": "prefill", "have": H, "cache": true}`. Both are optional (defaults 0 and true).
  `have` must be a multiple of `align` (128) and at most the prompt length, or the reply is an error and the
  connection stays usable. `cache: false` skips the lookup (computed from 0, and the result is still kept), which
  tests use to get a cold reference.
* `prefill` reply header adds `base` and `cached`. Each KV source's `compressed_kv` / `idx_keys` field has
  `(seq_len - base) / ratio` rows, possibly 0 (a 0-byte blob).
* `PrefillClient.prefill(ids, on_progress=None, have=0, cache=True)` returns the host format with `base` and
  `cached` added.

## 3. Decode server changes

The decode side of the d_p backend isn't written yet (`PREFILL_D_P_INTEGRATION.md` section 6.2, "Not done yet").
This section describes it with the prefix cache included. The changes are listed in the order a request meets them.

### 3.1 The prefiller (`demo/server.py`)

Add a `--prefill-backend d_p --prefill-server HOST:PORT` prefiller, a `_Prefiller` subclass alongside
`_V41Prefiller`, that holds one `PrefillClient` (or a small pool, if prefills of different users should queue on
the server rather than in decode). On connect, check `client.info`: `layers` must cover decode's KV sources,
`window` must equal `config.sliding_window`, `align` must equal `ALIGNMENT`, `max_seq` must be at least the
context, and log `cache_slots`.

### 3.2 What to send: `ids` and `have`

* `ids`: the prompt rounded down to whole 128-token units after leaving the last `sliding_window` (128) tokens to
  decode. That's `_V41Prefiller.aligned(n_tokens)` with `decode_window` set, unchanged. Always send the *whole*
  prompt from position 0, never only the new part: the server needs the full prefix to match its cache and, on a
  miss, to compute it.
* `have`: the number of leading tokens of `ids` whose entries this decode session already holds in its paged pools
  *and will keep*.
  * A new session, or one decode just reset or rewound to 0: `have = 0`.
  * A session continued by this prompt (the scheduler's "continues the slot's conversation" case, where today
    `_V41Prefiller._prefill` runs with `user.pos > 0`): `have = user.pos`. It's already a multiple of 128
    (`seed_prefill` requires it), and it must be at most `len(ids)`. If `user.pos > len(ids)`, the prompt has
    nothing for prefill and decode replays it all, as today.
  * A session that took another slot's prefix (`attach_session_prefix`, `_share_prefix`): its blocks below the
    prefix are shared and must not be rewritten. With `have = P` (the attached prefix, a multiple of 128) the
    commit starts at entry `P / ratio`, a 64-entry block boundary, so it never touches a shared block. Optional: `have = 0` is
    always correct, just slower.
* The server's own cache needs nothing from decode. Successive turns of one conversation hit it on their own,
  because each turn's `ids` starts with the previous turn's.

### 3.3 Reading the reply

```python
got = client.prefill(ids, on_progress=..., have=have)
assert got["base"] == have and got["seq_len"] == len(ids)
hit_tokens = got["cached"]  # > 0: prefix cache hit on the prefill server
```

* `got["cached"]` is the prefill server's hit length. Count it in the request's usage as
  `prompt_cache_hit_tokens` (the decode server already reports its own local hits under that name), log it with
  the prefill time, and add it to the monitor counters.
* A hit doesn't change what decode commits. The entries in the reply are always those of `[base, seq_len)`,
  whether the server computed them or not.
* `got["seconds"]` is 0 when `cached == seq_len`, so guard any tok/s figure against it.

### 3.4 Committing entries that start at `base`

Today `commit_prefill_state(states, sid, start=P)` (`tt/model.py`) already continues a session: it rewrites the
rings whole and writes the paged pools only from the block holding entry `P // ratio`. But it requires every KV
source's `compressed_kv` to hold all `T / ratio` entries:

```python
if source.num_entries != total // ratio:
    raise ValueError(f"kv source {src} holds {source.num_entries} entries for {total} tokens")
```

and it indexes the tensors from entry 0. To commit a reply with `base > 0` without first padding it back to full
length on the host, make the following changes.

1. **`PrefillAttentionState`** (`deepseek_v4_flash/tt/prefill/attention.py`): add `entry_offset: int = 0`, the
   entry index of row 0 of `compressed_kv` / `idx_keys`. `num_entries` stays the row count. Add
   `end_entry = entry_offset + num_entries` if that reads better at the call sites.
2. **`to_prefill_states`** (`tt/prefill/d_p_bridge.py`): accept `host["base"]`. Today it raises a `ValueError`
   for `base > 0` that points here, so nothing commits a partial reply by accident. Instead check
   `entries == (T - base) // ratio`, set `state.entry_offset = base // ratio`, and upload zero-row tensors as
   `None`, because ttnn can't hold a `[1, 1, 0, D]` tensor. `kv_tail` is unchanged.
3. **`commit_prefill_state`**:
   * The check becomes `source.entry_offset + source.num_entries == total // ratio`, and
     `source.entry_offset <= start // ratio`. Decode must hold everything below the reply's first entry, which
     holds when `start == base` (section 3.2 sets `have = user.pos` and the commit uses `start = user.pos`).
   * Local stage: `self._commit_paged_entries(local_scache, group, sid, source.compressed_kv, source.idx_keys,
     start // ratio, entry_offset=source.entry_offset)`. `_commit_paged_entries` already takes `entry_offset`
     (it must be a multiple of `CSA_INDEX_BLOCK_SIZE` = 64 entries, which `base` being a multiple of 128 ensures
     for ratios 1 and 2).
   * Host fallback (`not can_fabric`): pass the same `entry_offset`.
   * Fabric path: iterate entries in absolute indices (`n = entry_offset + num_entries`,
     `lo_block = (start // ratio) // block`). Slice the tensors at `e0 - entry_offset` to `e1 - entry_offset`, and
     keep `_commit_paged_entries(..., e0, entry_offset=e0)` as it is. The `e0 == 0 and e1 == n` no-slice shortcut
     becomes `e0 == entry_offset and e1 == n`.
   * No compressed entries at all (`have == seq_len`, `compressed_kv is None`): skip the source's pool writes. The
     rings are still rewritten. This is the existing "A bounded replay may end at `start` itself, and then
     rewrites only the rings" case.
   * `ensure_session_capacity(total - 1, sid)` is unchanged.
4. **`_commit`** of the d_p prefiller: `commit_prefill_state(states, user.sid, start=user.pos if have else 0)`,
   where `have` is the value sent in section 3.2. Then `free_prefill_states(states)`, as today.

### 3.5 Engram history

Seed the session's engram n-gram history for the committed tokens only: `hash_session(slot, ids[have:], have)`
for `have > 0`, and `hash_session(slot, ids, 0)` for `have == 0`. A continued session already holds the history
of `[0, have)`. This mirrors what the local prefill path does per chunk (`_EngramPrefetch.consume`).

### 3.6 After the commit

Unchanged. Decode replays `[seq_len, prompt_len)` (at least the last 128 tokens) through all 40 layers, which
rebuilds every layer's window (layer 20 and 21-39 included) and gives the first token's logits.

### 3.7 Failure handling

* `PrefillServerError` with `have ...: a multiple of 128 up to the prompt's n tokens` is a decode bug. Log it,
  and retry once with `have = 0`.
* Version mismatch on connect (`speaks protocol 1, this client 2`): refuse to start the d_p backend. Don't fall
  back silently.
* Connection loss mid-request: the server finishes the prefill and keeps the result in its cache, so a retry of the
  same prompt is a full hit (`cached == seq_len`) and costs only the readback.

### 3.8 Tests for the decode side

1. Unit (host): `to_prefill_states` with `base > 0` gives states with `entry_offset = base // ratio`. Zero-row
   sources give `None`.
2. Device, decode host: two-turn conversation through the d_p backend. Turn 2 must report `cached > 0` from the
   server and `base == user.pos`, and its first-token logits must match turn 2 run with `have = 0` (all entries
   sent, the existing path), with exact token equality under greedy sampling.
3. Device: `have == seq_len` (the same prompt re-sent to a session that holds it all) commits rings only and gives
   the same logits.
4. Throughput: `demo/test_server_perf.py` multi-turn traces, comparing prefill time and bytes received for turn 2+
   with and without the cache (`--cache-slots 1` and a prompt mix that evicts gives the "without").

## 4. Limitations and next steps

* **Partial-prefix hits (shared system prompt, edited earlier turn) miss.** The window carry at the divergence
  point is gone from the slot's 192-row rings. Options, cheapest first:
  1. *Bounded replay*: metal_2 already implements DeepSeek's encoder SWA bounded replay under
     `DECODER_SKIP=decode`. A chunk with `global_write_start = g > start` recomputes `[g - 128, g)` with every
     window truncated at its start, and writes no history-cache row below `g`. Wiring this up means passing
     `global_write_start` through `DPPrefill.prefill`, and allowing a `resume_start` at any 128-multiple of the
     common prefix when replay is accepted. It isn't exact. Decode's local prefiller already accepts it
     (`bounded_replay=True`), so it should be a server flag, and the reply should say `exact: false`.
  2. *Window checkpoints*: copy each layer's ring (192 rows) at chunk boundaries into a per-slot device buffer
     (about 4 MB a chip a checkpoint for 21 layers), and restore one before resuming. This is exact at those
     boundaries, but it has to restore metal_2's host bookkeeping of the ring (`WindowRing.pos/vend/own/end`).
  3. *Copy-on-write slots*: copy a slot's slabs and rings to a free slot before continuing it, so the kept prompt
     survives a branch. It needs a device copy of the ND-sharded served tensors and the same ring bookkeeping.
* Each slot is sized for `max_seq`, so short prompts waste most of their slot. Slots of several sizes (a few 1M, many
  64k) would multiply the cache's capacity. Each size is its own `allocate_kv_cache(max_seq=...)`, but the runtime's
  `max_seq` and the history-gather buffers are per model, so this needs checking in metal_2.
* The readback still runs inside the device lock. A hit makes it smaller (only `[have, seq_len)`), but a miss with
  `have = 0` on a 1M prompt still holds the lock for 5-12 s of readback.
* Lookup cost is a host compare of the prompt against every kept prompt: about 1 ms per slot at 1M tokens. A hash of
  each slot's 128-token blocks would make it independent of the slot count, if hundreds of slots ever appear.
