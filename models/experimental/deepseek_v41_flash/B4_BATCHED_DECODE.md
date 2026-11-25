# B=4 batched decode per instance on the DSpark 4-row block

Status: plan, Nov 25 2026. Nothing below is built or measured; the estimates are derived from measurements listed
in "Where the numbers come from" and must be confirmed in phase 0 before the kernel work starts.

## Goal

Decode **four users per instance in one 4-row step** (one row per user, one token each), reusing the 4-row block
step that DSpark's verify runs today (`DeepSeekV41Model.decode_verify_block`, `tt/model.py:5027`). With five
instances that is 20 users decoding at once instead of 5.

Today each instance decodes one user: `--num-users 1` with `--dspark`, and every other request waits in the queue.
The Nov 25 load run on hosts 40/41 showed requests waiting 18-36 s in `wait.queue_for_slot` while 5 replies decoded.

## Summary

* **Same MoE load, every row useful.** Per layer, the persistent MoE serves a 4-row request with independent rows at
  155 us (1 requester) / 123 us (3 requesters). That is what a DSpark verify block costs (162 us at a 23-expert
  union), so the shared TP12 MoE carries the same traffic as now. Under DSpark ~2-3 of the 4 rows become tokens
  (GSM8K ~3, chat ~2); under B=4 all 4 do, and no drafter runs.
* **What carries over:** the persistent MoE's 4-row requests, every row-wise dense op (matmuls, HC, norms, stream
  mix), lm_head, the per-row device sampler, the packet/socket plumbing, block trace capture, and the server's
  instance and queue machinery.
* **What does not:** everything that reads or writes a session's cache state. The block step is one session at
  positions `p .. p+3`: one page-table row, one ring, one compressor window state, causal visibility between its
  rows, and a trace variant chosen from one position. These need kernel and model work (phases 1-2).
* **Trade-off:** total tokens/s per instance should rise about 1.5-2.5x (estimate, falling with context length).
  Per-user tokens/s falls to one token per step: about 40-55 tok/s/user, against DSpark's 145-216 for a lone user.
  B=4 is for load; keep DSpark for 1-2 decoding users (phase 3 switches between them by load).

## Expected gain (estimates)

The terms below are per step on one instance.

| Term | Value | Source |
|---|---|---|
| 1-row decode step | 6.2 ms @0, 8.2 @128k, 9.6 @512k, 10.4 @1M | context-scaling sweep, Nov 16 |
| context-dependent part `g(c)` | ~0, 2.0, 3.4, 4.2 ms | same; almost all `FusedLightningSelectKv` (the indexer's key scan) |
| MoE, 4 independent rows vs 1 row | +~90 us a layer | persistent-MoE rows test, Nov 16 |
| DSpark round (5 instances, 64k-500k context) | ~24 ms verify + ~7 ms draft + ~12 ms host, ~3 tokens | overhead tables, Nov 25 run |

**B=4 step ~= block step + 3 x g(c).** The four rows select over four different sessions' index keys, so the key scan
that the DSpark rows share is done four times. Attention over the ring and the selected top-k rows is already per
row and does not grow.

| Context | B=4 step (est.) | B=4 tok/s/instance | tok/s/user | DSpark tok/s/instance today |
|---|---|---|---|---|
| short (<4k) | ~11 ms | ~360 | ~90 | 145 (chat) - 216 (GSM8K) one user |
| 128k | ~19 ms | ~210 | ~53 | ~70-100 (5 instances, Nov 25 logs) |
| 512k | ~24 ms | ~165 | ~41 | ~60-90 |
| 1M | ~27 ms | ~150 | ~37 | - |

These ignore host time per step (~12 ms a round in the Nov 25 tables, much of it the GIL shared by five instances)
and the MoE contention between instances. **The ceiling is the shared MoE:** five instances x 4 rows per layer at
~123 us per 4-row request make roughly the same aggregate as today's DSpark traffic. So total server throughput is
bounded at about 4 / (accepted tokens per DSpark round) times today's, i.e. ~1.3x (GSM8K) to ~2x (chat). The per-instance
numbers above apply until that bound is reached.

## What the 4-row block step is today

`decode_verify_block(token_ids, pos)` (`tt/model.py:5027`) runs one traced step of `verify_rows = 4` rows for the
resident session at positions `pos .. pos+3`:

* `_block_verify_packet` (`tt/model.py:4635`) writes one `pos` and 4 tokens; `_block_context` (`tt/model.py:4650`)
  derives every per-row integer on device as `pos0 + j`: ring slots, lengths, window slots, entries, index positions,
  RoPE rows.
* `_block_variant_key(pos)` (`tt/model.py:4599`) picks the trace: odd/even start, which ratios are still dense, two-level
  select, which index layers split their keys. A different `pos` can need a different trace.
* Per layer, `DeepSeekV41Attention.decode_block` (`tt/decode/attention.py:2033`) calls:
  * `attention_qkv_decode`: Q, the new KV rows into `staging`, committed into the ring at `ring_slots [R]`.
  * `compress_pool_decode` (kv sources, `:2126`): closes the block's groups from the session's compressor window
    (`win_kv`, `win_gate`, `win_slot`, `prev_slot`, `start_odd`), writing index keys, `comp_kv`, `dense_kv` through
    **one** `page_row`.
  * `indexer.select_kv` (index sources): per-row `index_pos`, but **one** `page_row`, i.e. one session's keys.
  * `flash_mqa_decode` (`:2174`): one ring, one base position (`lengths = [p, 0, 0, 0, n_0..n_3]`); row `j` sees the
    ring plus staging rows `0..j` (causal within the block).
  * `paged_update_cache(scache.kv, ..., update_idxs_tensor=ring_slots)`.
* The MoE request (`_use_server_rows(4)`, `tt/model.py:4477`), HC, norms, lm_head and the device sampler treat the
  rows independently.

The old multi-user batch path is off for V4.1: `prepare_static_decode` raises for `batch != 1` (`tt/model.py:886`).
What does exist is **slotted** state: with `num_sessions > 1` every session owns a slot of each sliding ring and
compressor window, and its page-table row. `prepare_spec_decode` refuses slotted rings with `verify_rows > 1`
(`tt/model.py:3758`).

## Reuse map

| Component | B=4 status | Work |
|---|---|---|
| Persistent MoE 4-row request | ready (independent rows measured) | none |
| Image twins (`image_bias`) | one bias per request | rows of one step must be all-text or all-image; decode rows are text, so batch text only |
| Dense row-wise ops (matmul_decode, HC, norms, stream_mix, all-gathers) | ready | none |
| lm_head, device sampler | ready, per-row draws | per-row seeds/temperatures stay server-wide |
| `attention_qkv_decode` | ring commit at `ring_slots [R]` | slot offsets: `slot_j * window + pos_j % window` into a slotted ring (likely host/step-context only) |
| `paged_update_cache` | per-row indices | same as above |
| `flash_mqa_decode` | one ring, one base, causal rows | **kernel:** per-row ring (slot) and length, rows independent (no staging visibility across rows) |
| `compress_pool_decode` | one session's window state, one page row | **kernel:** per-row window slot set, per-row page row, per-row group closing and `start_odd` |
| `select_kv` / `FusedLightningSelectKv` (+ split keys, `merge_select_kv`, two-level pool) | per-row `index_pos`, one page row | **kernel:** per-row page row (4 key scans); check the split and two-level variants per row |
| `dense_kv` (short contexts) | slotted already (`dense_offset`) | per-row offsets |
| Step context (`decode_step_context`) | per-row values from `pos0 + j` | per-row `pos_j`, `slot_j` packet fields |
| Trace variants | one per start position | per-row runtime flags (below) |
| Engram | per-row row lookup, one session history | per-row session history: `hash_session(slot_j, ...)` per row (host) |
| Drafter / DSpark rings | used | not used by B=4 steps; DSpark stays for low load |
| Host runtime (`_host.write_block`, output read) | 4 tokens | 4 tokens, 4 sessions; per-row slot in the block write |
| Scheduler | one seated decoding turn per instance | up to 4, stepped together (phase 3) |

## Plan

### Phase 0: confirm the gain (1-2 days, device)

1. Time the block step at 0 / 128k / 512k / 1M on one instance: `decode_verify_block`, tracy, `-t 8090`.
2. Time `FusedLightningSelectKv` with 4 rows over 4 sessions' keys. Until the op takes per-row page rows, the
   estimate is 4 rows on one session plus 3 extra single-row selects at the same context.
3. With five instances, measure MoE time per layer at 5 x 4-row requests, to place the shared-MoE ceiling.
4. **Go / no-go:** B=4 tok/s/instance >= 1.5x DSpark's at 128k under 5-instance load.

### Phase 1: kernels (largest part)

Each op gains a **per-row session mode** that leaves today's single-session block unchanged (same inputs, same trace).

1. **`flash_mqa_decode`**: inputs `ring_base [R]` (slot offset) and `lengths` per row; an `independent_rows` flag
   skips the staging-row causal mix, so row `j` attends its own ring and only its own new row. Test against 4
   single-row calls.
2. **`compress_pool_decode`**: `page_row [R]`, `win_slot` and `prev_slot` per row in a slotted window, a per-row
   "closes a group" mask and `start_odd [R]`, instead of `pool_rows` / `start_odd` fixed in the trace. Each row
   closes at most one group per ratio per step, which keeps the loop bounded.
3. **`FusedLightningSelectKv`** (and `merge_select_kv`, `build_candidate_pool`): `page_row [R]`, one key scan per
   row. Check the key-split and two-level paths per row; the host decode path still asserts one row for two-level
   select (`tt/decode/attention.py:2289`).
4. `attention_qkv_decode` / `paged_update_cache`: confirm slotted-ring indices pass through as they are.

Done when each op matches four 1-row calls (bit-exact or PCC >= 0.999) for 4 sessions at different positions,
covering group-closing parities, dense to sparse transitions, key split and two-level select.

### Phase 2: model

1. **Slotted block:** lift the `num_sessions > 1` / `verify_rows > 1` refusal (`tt/model.py:3758`) for B=4 traces.
   Build rings and windows slotted (`_slotted`).
2. **`decode_batch_block(ids[4], sids[4])`:** per-row `pos_j` from each session; `ensure_session_capacity` per
   session; packet fields `token[4]`, `pos[4]`, `slot[4]`; `_block_context` builds every per-row integer from
   `pos_j` / `slot_j` rather than `pos0 + j`.
3. **Variants:** fold the variant key into runtime per-row inputs: parity into `start_odd [R]`, dense versus scored
   per ratio into per-row lengths and offsets, split and two-level per row. If an op cannot take one at runtime,
   build the trace for the union (always-scored, always-split above the crossover) and gate rows by length.
   Fallback: the scheduler batches only rows that share a variant key.
4. **Padding:** with fewer than 4 decoding users, pad rows point at a scratch session slot whose writes nobody
   reads; their tokens are dropped.
5. **Engram:** one history per row (`_engram_lookup.hash_session(slot_j, ...)`), each row's lookup written to its
   row of the Engram input.
6. **Output:** the 4 sampled tokens come back as they do for a verify block; each goes to its own session.

### Phase 3: server

1. `--decode-batch 4` per persistent instance: `--num-users 4` active slots per instance, plus the device-parked
   ones (`_PERSISTENT_SLOTS`, `demo/server.py:345`); the per-instance KV pool (`_PERSISTENT_TOTAL_CONTEXT`, 4M
   tokens, `demo/server.py:335`) is shared by the four. 4 x 1M fills it: cap `--max-context` or evict parked first.
2. **Scheduler** (`_SpecScheduler`, `demo/server.py:2231`): seat up to 4 decoding turns per instance and step them
   together. Turns join and leave between steps. Prompt work (remote KV load, block replay of a prompt tail,
   `_block_replay_prompt`, `demo/server.py:2327`) stays one session at a time between batched steps, as prefill
   rounds are today.
3. **Mode by load:** reuse `_pick_drafts` (`demo/server.py:2456`). One decoding user takes DSpark rounds; two or
   more take B=4 steps. Both send 4-row MoE requests, so the MoE server needs no switch.
4. `_Instances` routing: an instance counts as free while it has a free batch row, not only an empty slot. The
   `_ClaimQueue` stays for overflow beyond 20.

### Phase 4: validation

* **Text equivalence:** greedy B=4 against 4 separate one-row runs on the same prompts gives identical tokens
  (allowing the known ~1-in-512 select near-tie divergence at long context).
* **Load runs:** the GSM8K perf client and the long-context mix from the Nov 25 run. Report total and per-user
  tok/s, TTFT and queue wait, next to DSpark.

## Risks and open questions

* **Variants.** Per-row parity, dense or scored, split and two-level could multiply traces if they cannot become
  runtime inputs. Grouping rows by variant is the fallback, and it costs batching efficiency.
* **`compress_pool_decode`** is the least row-oriented op today (window state, group closing). It is the largest
  kernel item.
* **Per-user speed.** One token per step per user: B=4 users see ~40-55 tok/s instead of 145-216 alone. The mode
  switch in phase 3 keeps the lone-user speed.
* **Host time per step.** ~12 ms per round today, much of it GIL contention across five instances in one process.
  At ~20 ms device steps that would be the next limit; moving instances to separate processes is worth weighing
  before phase 3.
* **KV pool pressure.** 4 long users per instance compete with parked conversations for the 4M-token pool, so
  prefix-cache hits drop.

## Where the numbers come from

* Context scaling of the 1-row step: 161 / 126 / 112 / 104 / 96 tok/s/u at 0 / 4k / 128k / 512k / 1M; after the
  fused-select fix 139 @4k and 122 @128k. The growth is in `FusedLightningSelectKv`; `flash_mqa_decode`,
  `compress_pool_decode`, MoE and HC are flat.
* Persistent MoE, traced, per request: one row 66.4 / 47.3 us (1 / 3 requesters); 4 independent rows 155.0 / 122.7;
  DSpark 4-row blocks 82.9 (6-expert union) to 162 (23-expert union).
* DSpark: one instance ~216 tok/s/u (GSM8K) and ~145 (chat) against ~149 one step at a time; five instances
  635 tok/s total (GSM8K) against 500 one step at a time.
* Nov 25 overhead tables (5 instances, 64k-500k contexts): `compute.verify_read` ~24 ms and `compute.dspark_draft`
  ~7 ms per round, ~12 ms host, ~3 tokens per round; `wait.queue_for_slot` 18-36 s per queued request.
