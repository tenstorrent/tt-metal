# B=4 batched decode per instance on the DSpark 4-row block

Status: phases 1-2 built on branch `smanoj/ds_v4_b4` (Nov 25 2026) and verified on the Galaxy (one persistent instance,
Python host path). The estimates further below predate this and still need phase 0's load runs.

## Measured (Nov 25, Galaxy)

* **Ops** (`tests/decode/test_batch_rows_ops.py`, 20 cases): `compress_pool_decode` independent rows is bit-exact
  against four one-row calls (windows, `idx_key_cache`, `comp_kv`, `dense_kv`, roped rows); `flash_mqa_decode`
  independent rows matches the fp32 torch reference at PCC >= 0.9992.
* **Model** (`tests/decode/test_batch_decode.py`, teacher-forced against one-row decode of each session alone):

  | Run | Variants hit | Picks matching | Differing picks |
  |---|---|---|---|
  | short prompts (positions 3-160) | dense | 193 / 201 | 8, all near-ties (0.25-0.75 logits) |
  | `BATCH_LONG=2100` (to 2260) | scored, both ratios | 1245 / 1251 | 6, all near-ties (<= 0.5) |
  | `BATCH_LONG=32000 BATCH_TAIL=400` (to 32160) | two-level, layer 20 key split | 516 / 521 | 5, all near-ties (<= 0.5); long row 400 / 400 |

  The short rows of the long runs ran the scored / two-level variants (the furthest row picks the variant) with the
  same near-tie positions as in the dense run. Eager reruns reproduce the traced picks exactly.
* **Step time** (traced, Python host, device sampling): 11.8 ms a step at short context (4 rows: ~340 tok/s per
  instance when all four are decoding, vs ~6.5 ms / 1 token for a one-row step); 12.6 ms at 2k and 13.4 ms at 32k
  (one real row and three pads, whose selects still run).
* **L1_SMALL**: the batch programs need more than the demo's 10 KB (the 4-row sampler's all_gather ran out last);
  12 KB is enough, and `demo/server.py` already reserves 16 KB under `--persistent-moe`.
* **Profile, one-row vs batch step** (`tests/decode/test_batch_decode_profile.py`, `tools/batch_profile_report.py`,
  device 27, per step, without the one-row step's H2D receive, which waits for the host's packet):

  | | 256 one-row | 256 batch | 2048 one-row | 2048 batch |
  |---|---|---|---|---|
  | host wall, no profiler | 7.15 ms | 10.72 ms | 7.93 ms | 13.50 ms |
  | device kernel time | 6.84 ms | 10.20 ms | 7.55 ms | 12.78 ms |
  | MoE requests (40) | 2727 us | 4133 (1.52x) | 2724 | 4119 (1.51x) |
  | `fused_lightning_select_kv` | - (dense) | - | 283 (8 calls) | 1148 (32 calls, **4.05x**) |
  | select's per-row setup (repeat, slice, reshard, norm) | 41 | 35 | 118 | 1135 (9.6x) |
  | `flash_mqa_decode` (40) | 458 | 960 (2.09x) | 539 | 1132 (2.10x) |
  | width-sharded all-reduce (42) | 589 | 1065 (1.81x) | 596 | 1063 (1.78x) |
  | `fused_single_user` HC (80) | 1108 | 1487 (1.34x) | 1111 | 1492 (1.34x) |
  | `attention_qkv_decode` (40) | 525 | 781 (1.49x) | 513 | 724 (1.41x) |
  | ring commit (`paged_update_cache`) | 56 | 334 (40 calls) | 48 | 334 |
  | `compress_pool_decode` | 12 | 110 (4 calls) | 12 | 104 |
  | weight matmuls | 981 | 943 (0.96x) | 1209 | 1128 (0.93x) |

  The one op that scales linearly with the rows is the select: one one-row `fused_lightning_select_kv` per row per
  index layer, 35.5 us a call either way. It runs only once a ratio's select is scored (past position 512 / 1025),
  and its key scan grows with the context, so it is the term that takes over the batch step at long context; its
  per-row setup (head replication per row) costs as much again. Everything else is sublinear: the weight matmuls are
  flat, the MoE 1.5x, attention and the all-reduces ~2x.
* **Server, one instance** (`demo/server.py --instances 1 --decode-batch 4 --no-prefill --no-vision --temperature 0`,
  `demo/perf_client.py --decode-tokens 256`, GSM8K prompts; prompts replayed through decode):

  | Prompt | 1 user: total tok/s (one-row steps) | 4 users: total tok/s (batch steps) | 4 users: tok/s/user |
  |---|---|---|---|
  | 128 | 146.9 | 286.9 | 72.2 |
  | 256 | 145.6 | 283.8 | 71.4 |
  | 512 | 137.4 | 251.4 | 63.2 |
  | 1K | 132.6 | 235.9 | 59.3 |

  1.8-2.0x the instance's one-user throughput. A batch step is 13.7 ms at short context and 15.5-16.2 ms at ~1.2K
  (device 11.5-12.1 / 14.5 ms, `compute.batch_step`; the 4 rows' Engram lookup 1.4 ms, `host.batch_engram`, before the
  dispatch; the scheduler ~0.3 ms).
* **Server equivalence** (`demo/test_server_users.py compare --users 8 --workers 4 --stagger 10 --max-tokens 200`,
  greedy; 8 two-turn conversations alone, then 4 at a time joining and leaving mid-reply, over 4 slots so turns park
  and reload): with the default policy (a lone reply on one-row steps) 0 / 16 replies are identical, all diverging
  6-125 tokens in (one-row vs batch-step near-ties, ~1 in 50 tokens, as in the model test), every reply on its own
  topic and tag. With `DEEPSEEK_V41_BATCH_MIN_USERS=1` (a lone reply in padded batch steps too) **16 / 16 are
  identical**: a row's picks do not depend on the other rows, the joins and leaves, or a park and reload. The lone
  replies then ran 25.9 s against 19.6 s on one-row steps.
* **Server, five instances** (`--instances 5`, same flags; users spread evenly, so N users is N/5 an instance):

  | Users (an instance) | One-row steps: total tok/s | Batch steps: total tok/s | Batch: tok/s/user | Batch step |
  |---|---|---|---|---|
  | 5 (1) | 515 | - | - | - |
  | 10 (2) | 479-516 (48-55 tok/s/user, interleaved) | 477-480 | 48 | 20.4 ms, 2 rows |
  | 15 (3) | ~510 | 541-550 | 37 | 26.6 ms, 3 rows |
  | 20 (4) | ~515 | 649-682 | 34-35 | 28.5 ms, 4 rows |

  The shared MoE is the limit, as estimated: a 4-row step's device part grows from ~12 ms alone to 22.4 ms with all
  five instances stepping (5 x 40 layers x ~110 us of 4-row requests), and pad rows load it as real ones do. The 4 rows'
  Engram lookup grows from 1.4 to 4.8 ms (five instances' Python host threads on one GIL). One-row steps cap at ~103
  tok/s an instance whatever its users. So batch steps start at 3 seated replies (`DEEPSEEK_V41_BATCH_MIN_USERS`,
  default 3): at 2 they lose under load and only break even alone (2 rows / 13.7 ms against ~147 tok/s). At 20
  users B=4 serves 4x DSpark's 5 users at about its total (635 tok/s, GSM8K, Nov 17) and 1.3x one-row steps'.

## What is built

* **`flash_mqa_decode(..., independent_rows=True)`** (R = 2 or 4): one row per row group of cores, each with its own
  plan from its own position. `lengths` is `[n_j .., p_j .., ring_first_row_j .., selected_first_row_j ..]` (16
  words at R = 4, the step context's limit). Row j reads its session's ring slot, only its own new kv row (key j of
  the staging chunk) and its own selected rows or its session's `dense_kv` rows; no chunk is shared between rows.
* **`compress_pool_decode(..., independent_rows=True)`**: every row is a group. At ratio 2 row j pools with its own
  window row `prev_slot[j]` (all read before any row is written); `entry[j] = -1` (a row that closes no group)
  skips its comp_kv / index-key writes; `page_row` is `[R]`, one page-table row per row.
* **Select**: `DeepSeekV41Indexer.select_kv_batch` runs the q_b / weights projections once for the four rows, then
  one one-row select per row over its own session's keys (key split + merge and the two-level candidate pool per
  row; `batch_candidates[j]` buffers). The fused multi-row select still scans one session's keys, so this is the
  part with the most per-step overhead left (4 x (2 replications + select [+ all_gather + merge]) per index layer).
* **Model** (`DeepSeekV41Model(batch_block=True, verify_rows=4)`, persistent stage): slotted caches
  (`num_sessions >= 5`, one slot a scratch session that pads short steps), `prepare_batch_decode`,
  `capture_batch_traces` (one-row and batch variants, all compiles before any capture) and
  `decode_batch_block(token_ids, sids, positions=None)`. The packet carries each row's token, position and slot,
  plus the entries / dense rows / select positions the step-context terms cannot derive. Variants are keyed by the
  furthest row (dense while every row is, two-level and key split once any row is), with no parity variants.
  Every batch step writes each new entry below `index_topk` into `dense_kv`, whatever the variant, so a short row's
  dense rows stay complete while a long row keeps the step scored. Python host path only (the Rust host runtime
  owns the Engram history); text tokens only.
* **Tests**: `tests/decode/test_batch_rows_ops.py` (both ops against a torch reference / R one-row calls) and
  `tests/decode/test_batch_decode.py` (B=4 picks vs one-row decode, teacher-forced, rows at different positions,
  padded tail steps; `DEEPSEEK_V41_BATCH_LONG` / `DEEPSEEK_V41_BATCH_TAIL` for the scored, two-level and split
  variants; differing picks are checked as near-ties in the batch step's logits, `debug_batch_logits`).

* **Server** (phase 3, `demo/server.py --decode-batch 4`): each persistent instance is built with
  `batch_block=True` and `--num-users` (default 4) slots plus the scratch one; the shared MoE is launched for 4-row
  requests without the drafter (`launch_instances_server(blocks=True)`). `_BatchScheduler` seats `--max-running` (4)
  replies and steps them together with `decode_batch_block`; replies join and leave between steps; prompts (remote KV
  load, the tail in one-row steps) run between steps, one session at a time, as `_PrefillFirst` runs them. With one or
  two seated replies the instance takes one-row steps (posted ahead), from three on batch steps
  (`DEEPSEEK_V41_BATCH_MIN_USERS`, default 3). A reply's prefix snapshot (needs nothing in flight) is taken by reading
  the pipeline back, not left waiting for a quiet moment that another reply's posted steps never leave (with
  `--no-prefill` two replies on one-row steps otherwise decoded one after the other). `_Instances` routes a new conversation to the instance with the fewest claimed slots while any has a free
  batch row. Forces `--no-dspark` and `--no-rust-host`. Host-only test: `tests/test_batch_scheduler.py` (a fake model
  whose picks hash each session's history: every reply equals decoding it alone, through joins, leaves, a stop and
  prompts between batch steps; two replies below the threshold interleave).

Not built: the mode switch with DSpark by load (a `batch_block` model has no drafter), the Rust host runtime's batch
packet (it would take the Engram lookup, 1.4-4.8 ms a step, off the GIL), and a multi-session fused select. Not run
yet: `--decode-batch` with `--prefill-url` (remote KV load between steps), with the local traced prefill, with
`--vision`, or past ~1.3K context.

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
