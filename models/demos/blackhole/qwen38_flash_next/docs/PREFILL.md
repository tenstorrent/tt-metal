> Historical source-fork guide. Replay-ring and rows-form switches described below belong to the retired pipeline. The maintained port uses the public local-output API; see [the current provenance](../PROVENANCE.md). Measurements here do not qualify the replacement integration.

# Prefill: the chunk bodies and the opt-in slab

> Historical source notes from Samuel Jett (sjettTT, sjett@tenstorrent.com), frozen at
> `cd9a11771107ea2c27da3303a0556ff7343e4af5`. Measurements and implementation details below
> describe that source snapshot. Use the [port README](../README.md) for current commands and
> [PROVENANCE.md](../PROVENANCE.md) for dependencies and the separation of source and port evidence.

The server prefills a prompt through traced chunk bodies on the same device state the decode traces use: 128-row
chunks where the prompt allows and 32-row chunks for the remainder (the served default since 2026-09-29; `--long-chunks`
is accepted and changes nothing, `--prefill-mode teacher_forced` has no chunks), and, with `--prefill-slab ROWS`, slabs
of ROWS rows (a multiple of 128 from 256 to 2048; 2048 is the measured form and the largest that serves) ahead of the 128-row chunks.  The slab and
the 128-row form are tolerance class against each other; on long chat-shaped prompts their greedy streams part early
(token 30 of a 2,160-token document, token 0 of a 20,612-token one; `NUMERICS.md`).  Whatever the
chunk sizes, the last prompt token is teacher-forced through the decode traces, so the decode, the MTP hand-off and the
prompt-end snapshot see the same objects: the committed GDN recurrent state (fp32) and conv history, the QSA KV and
compressed caches, the staging tile and raw-key ring, the PLE history and the device position.

## The slab (`--prefill-slab 2048`)

Inside a slab every layer runs its ROWS rows in one pass:

- every dense linear (GDN in/out projections, the gated-residual down+inject and up, the MoE router and shared experts,
  the QSA index / query-gate / key / value / output projections) runs as one 2D-multicast matmul over the rows on an
  interleaved copy of its resident weight (the decode weights stay DRAM-width-sharded; the copy is made and released
  inside the slab body, 0.04-0.15 ms each);
- the GDN kernel (`chunk_gated_delta_rule`) takes the whole slab in one call (ROWS / 32 sub-chunks, the recurrent
  state carried inside the kernel in fp32) and commits its final state; the FIR taps are row shifts;
- the routed experts run in one `moe_compute` call over the whole slab on the op's local output path (each expert's
  tokens packed into 32-row chunks once per slab, its outputs written straight into the slab's `[10, ROWS, 2560]`
  page), and that page is reduced in 512-row blocks; `QWEN38_MOE_SLAB_ONE_CALL=0` restores the 16 calls of 128 tokens
  through the 128-row instance's own call, tilize and weighted reduce (*The routed experts in one call* below);
- the QSA layer writes one ROWS-row KV slab and ROWS / 128 compressed tiles through one page table, scores and selects
  its blocks in 512-row blocks (the indexer, the score all-reduce, a broadcast causal block mask, `topk_large_indices`),
  and runs `sparse_sdpa` over all ROWS query rows in one call.

The two phases of that GDN kernel are bound to Python as well, `ttnn.prim.chunk_gdn_prep` and
`ttnn.prim.chunk_gdn_scan`, for a caller that lays the per-chunk inputs out itself; the composite
`chunk_gated_delta_rule` the slab calls is unchanged.  The MTP verify rows' GDN runs the same two programs around
those prims since 2026-09-25 (`gdn_rows_wrap`, a default at the 32-row verify tile; `QWEN38_FUSED_OFF=gdn_rows_wrap`
restores the chain): 6 programs where the chain ran 53 per GDN layer, bitwise, the slab keeping `gdn_prefill_rows`
(`docs/NUMERICS.md`).

The remainder of a prompt after the slabs runs through the 128-row chunks, then the 32-row chunks and the padded tail,
then the ordinary hand-off from the 32-row state (the `legacy` plan; the default `cost` plan below covers the remainder
with fewer, padded passes).  `--prefill-slab` implies `--long-chunks`; with `--mtp` (since
2026-09-27) the MTP layer's rows of a slab run inside the slab body through the 128-row twin's slab form -- the slab as
`ROWS / 128` slices of 128 rows, each the 128-row form exactly (its residual rows cut from the slab's layer-47 residual,
its RoPE and QSA chunk inputs at `P + 128 i` from the slab body's own index rows, its MTP tokens one tile per slice), so
the MTP layer's KV rows land where the 128-row chunks would write them and the hand-off stays the 32-row twin's; the
slab's residual is tolerance-class against the chunk bodies, so the MTP stream after a slab prefill is tolerance-class
too, with its own pinned table (`A3-slab2048-mtp4-32k`, whose slab record runs one slab); in the served trace the
slices cost +110 / +99 / +99 ms per 2048-row slab = +0.053 / +0.047 / +0.047 ms per prompt token
of TTFT on the 1x4 line (2026-09-27: 2,100 / 20,860 / 31,714-token prompts, 1 / 10 / 15 slabs: 1.093 / 7.701 / 11.991 s
against 0.983 / 6.713 / 10.504 s without `--mtp`; `docs/NUMERICS.md`).  `--long-chunks` combines with `--mtp`: the MTP layer's rows run
inside the 128-row chunk body as they do inside the 32-row one (a 128-row twin of the chunk extension; its input mixer
projects per 32-row tile, so every row keeps the 32-row form's matmul program), and the hand-off stays the 32-row
state's.  The 32-row and 128-row bodies are unchanged.  Measured 2026-09-26 on the 4-chip p150 line with `--mtp 4`
(server-side, the cold first request of each class): the 560-token chat prompt prefills in 0.896 s (1.60 ms per
prompt token; 1.309 s, 2.34 ms, with the 32-row chunks alone), a 513-token prompt in 0.764 s (1.49; 1.173 s, 2.29,
in 32-row chunks); the served decode after either prefill is the same to 0.2 tok/s, and the MTP DRAM admission adds
the 128-row twin's 25,600 bytes per bank (`docs/SERVER.md`).

## The prefill plan and the padded last pass (`QWEN38_PREFILL_PLAN`)

A pass costs about the same full or nearly empty (QB2 = 2x P300c, blocking replays: 2048-row slab 648 ms, 128-row chunk
156 ms, 32-row chunk 62 ms), so the `legacy` plan's remainder was dear: head 2047 ran 15 x 128 + 4 x 32 rows (2.6 s) where
one slab with 1 row of padding is 0.65 s.  The default `cost` plan (`tools/qwen38_prefill_driver.plan_cover`) covers the
head's rows with the cheapest sequence of passes by the per-form table `PASS_COST_MS` (one place; a slab of another
row count costs in proportion), with the rules the forms need:

- at most one pass is padded, the last, and the hand-off reads its state; a padded 128-row chunk or slab holds at least 9
  real rows (`PADDED_PASS_MIN_ROWS`: the PLE history is nine rows), a smaller remainder stays a padded 32-row chunk;
- a 128-row chunk or slab starts only where the position is a multiple of 128 (their compressed index rows are whole cache
  tiles), so 32-row chunks run up to the boundary (the legacy plan does not check this, it starts them wherever the alignment
  steps left the position; a fresh prompt starts at 0 and meets the rule);
- no pass, padded or not, ends past the allocated context (the pad rows write the KV and index caches); near the end of the
  context the cover falls back to the cheapest one that fits, and none that fits is a ValueError;
- forms are used when the chain has their state (a slab with `--prefill-slab`, the 128-row chunk with `--long-chunks`); a
  chain with MTP chunk extensions keeps the legacy plan (their 128-row and slab forms have no padded pass); the 32-row-only
  chain's cover is the legacy plan.

`QWEN38_PREFILL_PLAN=legacy` (the plan before this one, numerics unchanged) or `cost`; any other value is refused by name
at start-up and logged with the prefill form line (`prefill form: ...; plan: cost`).

What a padded pass changes in the traced bodies (one trace per form, nothing per length: the real-row count reaches the
pass only through host-written device inputs, `Qwen38TTNNTextModel.write_chunk_pad`, restored to a full pass by the hand-off and
by every reset, so a full pass is bitwise what it was):

| state the hand-off reads | how the padded pass keeps it the real rows' |
|---|---|
| GDN recurrent state | beta and the log decay of the pad rows are multiplied by a 0/1 row mask (beta = 0, g = 0: an identity update); the chain bodies mask in `_chunk_rows`, the fused slab body masks the prims' page layout between its two programs |
| GDN conv history | one exact 0/1 selection of the three real rows before the first pad row (the 128-row form's `history_select_full` is the host-written select; the slab picks them out of the projection's q\|k\|v rows) |
| QSA KV and compressed index rows | pad rows write past the prompt: the decode rewrites a row before it reads it, and the indexer masks blocks at or past `(P + 1) // 4`; the slab zeroes a pad row's complete-block count so every block of its score row is hidden and its top-k ids are the same for every pad content (the block-shared kernel builds one block union per tile of queries) |
| QSA staging tile and raw-key ring | the hand-off copies the 32-row tile that holds the last real row from the pass's kept rows into the 32-row state's, by one exact matmul with a select (32 persistent tiles, one per `prefilled % 32`, uploaded once when the chunk state is allocated; no program that changes with the tile) |
| PLE history | the nine normalized rows before the first pad row: the hand-off picks them out of `normalized` (128-row chunk) or out of the slab's per-block snapshot window (`snapshot <- snapshot * keep + window * take`, host-written 1.0 / 0.0 per block) by one exact matmul |

The hand-off's programs have fixed shapes, so a closed program-cache guard sees no new program in a request; the chain's warm
passes run one padded hand-off per form.

Numerics (QB2, `tests/test_prefill_padded_pass_device.py`; one id per process): the same prompt prefilled with two different pad
tokens gives bitwise-identical first-token logits and 32 decode logits and tokens (`pad_long`: 100 real rows in a 128-row
chunk; `pad_slab`: 1000 real rows in a slab).  The K/V and compressed index rows of the 12 QSA layers that a padded 128-row
chunk writes for its real rows equal a full 128-row chunk's bit for bit (`kv_long`).  A padded slab's equal a full slab's
except in the real rows that share a query tile (16 or 32 rows) with pad rows (here rows 992-999 of 1000): `sparse_sdpa_tiled` streams one
block union per tile, so those rows' rounding depends on their neighbors' selections, in a full slab on the next real tokens and
in a padded one on the pad rows (with `QWEN38_FUSED_OFF=sparse_sdpa_tiled` all 24 caches are bitwise equal, `kv_slab`).  Against
the legacy plan the padded plan is tolerance-class, like the slab and the 128-row chunk against the 32-row ones: top-1
agreement 1.000 over 32 teacher-forced steps, KL 8.9e-3 / 7.5e-3 mean (a 256-token and a 1001-token prompt; the 32-row form alone
against the legacy plan: 7.2e-3), identical greedy streams over 32 tokens.  The layer-0 GDN conv history and recurrent state
after the hand-off equal the legacy plan's bit for bit at 127 padded rows (`state_long`).

Measured on QB2 (2026-10-25, the default chain: 2048-row slab, 128-row chunks; median of 3, plain `prefill_forward` wall time
on the batch adapter with 8 lanes, which includes the lane import; the token each plan gives first is the same at every length):

| prompt tokens | legacy ms | cost ms | cost passes (padded last pass) |
|---|---|---|---|
| 32 | 257 | 258 | 1 x 32 (short x 31 real rows) |
| 128 | 439 | 370 | 1 x 128 (long x 127 real rows) |
| 256 | 606 | 520 | 2 x 128 (long x 127 real rows) |
| 512 | 912 | 824 | 4 x 128 (long x 127 real rows) |
| 1024 | 1530 | 921 | 1 slab (slab x 1023 real rows) |
| 2048 | 2775 | 943 | 1 slab (slab x 2047 real rows) |
| 2049 | 933 | 932 | 1 slab |
| 4096 | 3562 | 1591 | 2 slabs (slab x 2047 real rows) |
| 8192 | 4845 | 2899 | 4 slabs (slab x 2047 real rows) |
| 16384 | 7442 | 5538 | 8 slabs (slab x 2047 real rows) |
| 30000 | 11195 | 10174 | 15 slabs (slab x 1327 real rows) |

and `demo/text_demo.py` on the single-slot adapter (greedy, 128 new tokens, the needle prompt found in every run):

| prompt | cost TTFT s | legacy TTFT s | decode ms/token (cost / legacy) | pass key | first divergence of 128 tokens |
|---|---|---|---|---|---|
| 128 | 0.287 | 0.357 | 26.01 / 26.13 | found / found | 15 |
| 1024 | 0.834 | 1.440 | 27.24 / 27.25 | found / found | 13 |
| 2048 | 0.854 | 2.688 | 28.70 / 28.72 | found / found | 25 |
| 4096 | 1.501 | 3.474 | 28.67 / 28.74 | found / found | 30 |
| 8192 | 2.807 | 4.756 | 28.65 / 28.73 | found / found | 39 |
| 32576 | 10.760 | 12.386 | 28.62 / 28.66 | found / found | 43 |

The pad inputs cost a full slab about 8.6 ms (+1.3 %: the history select against the projection, the beta / g masks and the
PLE snapshot, 35 GDN layers each); the 32-row and 128-row passes show no change (P = 32 / 128 / 256 / 512 under the legacy
plan within 1.6 ms of the previous bodies').  The padded plan's hand-off adds the kept-tile and history selections (eager
matmuls, not timed apart: the hand-off's own wait for the last replay hides them in `handoff_ms`).

## The multi-user pack (P3d-2, `ttnn/pack.py`)

A pack prefills 1..G short prompts in ONE replay of the 2048-row slab body.  A bucket is `(S, G)` with S in {128, 256, 512,
1024} rows per user and G = 2048 / S users (16 / 8 / 4 / 2); user u owns the rows `[u*S, (u+1)*S)`: its P_u <= S prompt tokens
(the LAST token included, no forced step) first, the fixed pad token after them; an empty slot is an all-pad segment.  The pack
runs at position 0 of the chain's dense KV and compressed caches (user u at row base `u*S`, compressed row base `u*S/4`), so
the slab's KV write and its compressed-index write need no change.  One trace per bucket on the existing slab chunk state's
buffers.  The body is `Qwen38TTNNTextModel.forward_prefill_pack` (`allocate_pack_state`, `capture_prefill_pack`, `run_pack`);
the layers' `forward_chunk_generic(..., pack=...)` select the pack variants below, the MoE, norms, collectives and dense
linears are the slab's own and are bitwise row independent (P3d-1 probe P1).

- GDN: the chunk prims run with heads = 12 x users per launch (`gdn_source_chunk.chunk_source(heads=, state_output=)`, at most
  8 users per launch: 12 x 9 = 108 is the scan's core cap, so S = 128 runs two launches) from a zero state; q/k/beta/g go
  user-major by `permute` and v / o by slice + concat (about 0.8 ms per layer).  The conv reader and compute kernels take a
  JIT define `SEG_CHUNKS = S / 32` (`gdn_pre_rows.run(seg_chunks=)`): a tile row with `chunk % SEG_CHUNKS == 0` takes the
  history tile (a zero tile) and the `Sel_hist` shift selects like tile row 0.  Undefined (every other caller) the kernels
  compile to today's program.  Pad rows are an identity update (beta = g = 0 by the host-written mask).
- QSA: no indexer query projection, scoring, mask, top-k or id expansion.  For S <= 1024 every complete block of a user is
  selected, so `sparse_sdpa_tiled` reads the bucket's constant block-id rows (global ids `u*S/4 ..`, ASCENDING, then
  sentinels; `pack.block_id_rows`, guarded by `pack.assert_block_ids_ascending`: the kernel misattends silently for any other
  order) with the row index as the (global) query positions.  RoPE and the compressed rows' block-start RoPE use the LOCAL
  positions (host-written rows).  The attention order is ascending where production's is descending score: a tolerance-class
  change (P3d-1 probe P2), deterministic and independent of the other users.
- PLE: the FIR history is zeroed at every segment start and one snapshot window is kept per user.
- Output: the last real row of each user is gathered by an exact 0/1 select (`gather_select`, host-written), the final mixer
  and LM head run on 32 rows (row u = user u), `read_pack_logits` returns the first G rows (fp32, the LM head's padding
  columns unmasked).  The chain's position, GDN state and QSA staging are not touched.

The persistent buffers must exist BEFORE the chain captures its traces (a buffer allocated after a capture can sit on a
trace's intermediates and is overwritten by that trace's replays: a pack replay after any single-user replay then hangs on
a corrupted block-id / zero-state constant).  `allocate_pack_state` and one eager `run_pack` per bucket therefore run in the
chain's warm hook (`chain_warm_hook`, `chain.slab_state` is set at the chain's construction for this); the pack traces are
captured after the open.  Landing sources (what the lane landing reads after a replay; layouts in `ttnn/pack.py`): per GDN layer
the final recurrent state of every user (`states[launch]`, fp32 `[96, 128, 128]`) and its last three conv-input rows
(`conv_rows`, rows `32 u ..`), per QSA layer the user's rows of the dense caches, `kept_kv` and `kept_raw`, and the per-user
PLE snapshot windows.

Device tests (`tests/test_prefill_pack_device.py`, one id per process): `leak_<S>`, `slot_<S>`, `edge_<S>`, `numerics`,
`timing`, `pad_content`, `profile`; host-side tests in `tests/test_pack_no_device.py`.

Measured on QB2 (2x P300c, 4 Blackhole dies, 2026-10-26, context 32768, single slot): user A's logits and every landing source (36 GDN states and
conv rows, 12 QSA layers' K/V, compressed, kept `[v|k]` and raw-key rows over its whole segment, the PLE snapshot; 484 digests
on four devices) are bit-identical when the other users' tokens and lengths change, when the other slots are empty or hold other
tokens, between slots (0 / middle / last, across the two GDN launches of S = 128), and when the pad token changes (real rows only);
the other users' outputs do change.  A user's logits are bit-identical in the 256, 512 and 1024 buckets.  The replay is 546-562 ms
for every bucket against 644 ms for the 2048-row slab (no indexer); host input preparation 51-63 ms per pack (the PLE n-gram rows
are 46-58 of it: the same 2048-row lookup and upload as a slab); DRAM per device: 641 MiB of pack state (548 MiB of it the GDN
landing sources) and 61-74 MiB per bucket trace (259 MiB for the four).  Against the single-user path the first-token logits
are tolerance class: the pack's per-layer K/V, compressed and GDN-state errors equal the cost plan's against the legacy plan
(relative Frobenius error 0.006 at the first QSA layer to 0.15 at the last), top-1 agrees on 16 of 22 prompts and on 11 of 11
with a top-2 margin of 0.5 or more; the six flips have margins of 0.0-0.375.

### Packs in serving (P3d-3 landing, P3d-4 grouping, P3e default; `QWEN38_PREFILL_PACK`)

With the pack on the vLLM batch adapter (`tools/qwen38_vllm.py`, `_prefill_batch`) prefills some rows of a
prefill call as packs and lands each packed user straight into its decode lane (`ttnn/pack_landing.py`,
`Qwen38PlainLaneDecoder.admit_pack`; the vLLM block ids are the lane's pages).  The pack is on by default with 8 or more lanes
(`PACK_DEFAULT_MIN_LANES`, since P3e, 2026-10-26) and off with fewer; `QWEN38_PREFILL_PACK` overrides.  With the pack off the adapter's
prefill is the single path of before, row by row (the code path is unchanged).  The lane decoder alone (`text_demo`, the P1 tests)
keeps the pack off unless the knob is `1` or its `pack` argument is true.

| variable | default | meaning |
|---|---|---|
| `QWEN38_PREFILL_PACK` | unset: on with 8 or more lanes, else off | `1` builds the pack state, the bucket traces and the landing at start (before the chain captures) and plans every prefill call; `0` keeps the single path; any other value is refused |
| `QWEN38_PACK_BUCKETS` | lanes <= 8: `256,512,1024`; more lanes: `128,256,512,1024` | rows per user of the enabled buckets.  With 8 lanes or fewer `S = 128` (16 users) could never fill above half, and dropping it saves its second GDN launch (226.5 MiB) and its 74 MiB trace.  An explicit value wins |
| `QWEN38_PACK_FORCE` | `0` | `1` packs every row of at most the largest bucket, even alone (tests and accuracy runs; the planner's cost rule is off) |

The planner (`tools/qwen38_pack_plan.py`, host only, `plan_prefill_groups(lengths, buckets, lanes, force=)`) returns groups
`("pack", S, [rows])` and `("single", [row])`.  Only a row of at most `max(buckets)` tokens can pack; the bucket of a pack is the
smallest enabled `S` that holds its longest row; a pack holds at most `2048 / S` users.  The rows are sorted by (length, index),
a pack takes a contiguous run of that order, and a dynamic program minimizes the estimated total time; ties go to the single
path, and the same lengths always give the same plan.  The adapter logs `prefill groups for lengths [...]: [...]` per call and
one `prefill pack S=.. users=..` line per pack with its phases.  Cost model (all constants in one place, `qwen38_pack_plan.py`,
from the P3d-3 timings on QB2, 2026-10-26):

- pack: `fixed(S) + users * land(S)` ms; `fixed` = host prep + upload + replay + logits = 657.0 / 648.4 / 652.8 / 661.1 for
  S = 128 / 256 / 512 / 1024, `land` = 5.8 / 6.7 / 8.8 / 12.1 per landed user;
- single: the cost plan's `PASS_COST_MS` of the cover of the `P - 1` head rows, plus 60 ms per request, plus 58 ms per slab pass (its
  host preparation does not overlap): 122 / 216 / 766 ms for P = 32 / 128 / 1024, the measured values.

Consequences: 100-token prompts pack from the fourth user on (3 users: 3 single prefills, 4 users: one pack); a lone prompt of
about 600-1024 tokens already packs (a pack costs about 673 ms, the single slab path 766 ms); a lone prompt under about 500 tokens
stays single.  An S = 256 pack of 8 users with P <= 128 costs 656 ms against 679 ms for the same users in an S = 128 pack (measured,
same prompts, logits bit-identical), which is why the 8-lane default does not need S = 128.

DRAM per device (`pack_dram_bytes(buckets, context)`): the persistent state 187.6 MiB, the landing's selects 10.9 MiB, 226.5 MiB per GDN
launch (S = 128 runs two, the others one), one trace per bucket (74 / 63 / 62 / 61 MiB) and, per bucket, the compressed zero tail
`(context / 4 - S / 4) x 128 x bf16` (7.9 MiB over all four at context 32768; it grows with the allocated context).  All four
buckets: 918.6 MiB at B = 8 or 32, context 32768 (measured); the default 8-lane buckets: 622 MiB measured against 617 MiB estimated.
The pool admission (`batch_pool_token_cap`, `_admit_pool`) counts it: at B = 32 the pool shrinks by 10.2 % (all four buckets); at
B = 8 the 4096-block pool is not DRAM bound and does not change.

Numerics class: a packed request is tolerance class against the same request on the single path (P3d-3 `semantics` test: GDN
state and ring relative error 0.03-0.04 median, up to 0.31-0.37 at the deepest layers of a 1000-token prompt; the single path's own
cost-against-legacy floor is 0.23-0.25 there).  A request's tokens can therefore depend on whether it was packed, which depends on
the co-scheduled load; the pack itself is bitwise independent of its co-users (a user's logits and lane state do not change with the
other users' tokens, lengths or slots) and of the bucket (the same prompt in 256 / 512 / 1024 gives identical logits).  The first
token is drawn from the pack's fp32 logits on the host: a greedy row is `argmax` of the masked row (torch returns the lowest id among equal
maxima, the device greedy's tie rule), any other row the single path's host draw (penalties over the prompt, `position_generator(seed, P - 1)`).
Measured (P3d-4, QB2): the 13 acceptance records through the vLLM server, B = 8, every record packed (`QWEN38_PACK_FORCE=1`):
json 96/96, first divergence against the CPU records chat 39 -> 23, math 61 -> 56, multilingual 9 -> 43, refactor 23 -> 24, every other
record unchanged (code 32, fact 15, list 46, prose 13, sky 19, story 6, summary 1, document 12 is a single-path record).  The contract
test with packs: every first token equals the B = 1 reference (no miss), later tokens diverge on 6 of 8 packed rows (forced run) and 14 of 25 (B = 32 run), first
divergence at token 7-31, the pass key is found in every request.

Accuracy (P3e, 2026-10-26; the reason for the default): GPQA-Diamond (198 questions, one sample each) through the vLLM server at
B = 32, block 64, `--max-model-len 32704`, max 29800 new tokens, concurrency 32, thinking on, the card's sampling (temperature 1.0,
top-p 0.95, top-k 20), each question seeded by its id in both runs.  Pack off: 83.8 % (166), 95 % CI [78.3, 88.9], 33 answers cut at
the length limit.  Every eligible request packed (`QWEN38_PACK_FORCE=1`; 173 packs held 202 requests, the one prompt over 1024 tokens
ran single): 85.4 % (169), [80.3, 89.9], 25 cut.  Paired difference +1.5 points, 95 % CI [-2.0, +5.1] (5 questions right only
without the pack, 8 right only with it); on the 161 questions that neither run cut, -1.2 points, [-3.1, 0.0] (2 and 0).  No
measurable difference.  The model card's 91.7 is outside both intervals; the cut answers and the answers without a final choice
account for most of the gap.

Mixed order: a pack overwrites the chain's dense KV rows 0..2047 and the PLE history; the next single-path prefill resets both.
Measured (4 packs of S = 128 / 256 / 512 / 1024 against single prompts of 40 / 130 / 700 / 1500 tokens, all four devices): a single
prefill after a pack has the same lane state in the rows a decode reads (every family; the compressed rows below `P // 4`, the staging
and raw-ring rows up to the last token), the same KV rows `[0, P)` and the same first-token logits, bit for bit, as without the pack
(the rows past them hold the chain's previous pad-derived data in the single path in any order: a single after a single already differs
there); a pack after a single is bitwise the same pack alone (logits, whole lane state, KV rows).  A pack must not run while a chain
prefill is in progress (the adapter never does).

Gains, `vllm bench serve` against the real server (2026-10-26, `--max-model-len 32704`, block 64, greedy, `--ignore-eos`):

| B | load | knob | output tok/s | TTFT mean / p50 / p99 | TPOT mean |
|---|---|---|---|---|---|
| 32 | burst 32 x 128/128 | off | 235.1 | 6990 / 7207 / 7208 ms | 82.1 ms |
| 32 | burst 32 x 128/128 | on | 343.3 | 1726 / 1773 / 1774 ms | 80.3 ms |
| 32 | c32 128/1024, 64 requests | off | 359.4 | 7002 / 7186 / 7257 ms | 82.3 ms |
| 32 | c32 128/1024, 64 requests | on | 387.8 | 1688 / 1733 / 1739 ms | 80.9 ms |
| 8 | burst 8 x 1024/128 | off | 86.2 | 5711 / 6409 / 6410 ms | 48.6 ms |
| 8 | burst 8 x 1024/128 | on | 115.0 | 3044 / 3383 / 3384 ms | 46.1 ms |
| 8 | c8 128/1024, 16 requests | off | 177.8 | 1586 / 1766 / 1792 ms | 43.5 ms |
| 8 | c8 128/1024, 16 requests | on | 184.3 | 818 / 891 / 910 ms | 42.6 ms |

The adapter's plans: B = 32 burst: the first request alone, then one pack of 16 and one of 15 users (S = 128); B = 8 burst of 1024: the first
request alone in a pack (S = 1024), then packs of 2, 2, 2 and 1 users; B = 8 c8 128-token requests: the first alone, then a pack of 7 (S = 256).

## Numerics class

The slab is **tolerance-class** against the chunk bodies: a 2D-multicast matmul accumulates K in another block order
than the per-tile DRAM-sharded program, so each linear's output differs from the chunk form by up to one bf16 ULP of its
scale (58 % of the elements one step apart, measured on 4x p150 at 2048 rows); the GDN kernel over the slab is bitwise
the chained 128-row calls, and the slab's other ops (the KV writes, `sparse_sdpa`, the block selection, the head sum)
are bitwise per row.  With the block-shared attention (`QWEN38_FUSED=sparse_sdpa_tiled`, below) the slab's attention
leaves that list: the attended set per query is the chain's exactly, the scores are the same bf16 numbers, but the
flash loop visits each query's keys in the tile union's order and in 256-key chunks instead of the query's own top-k
order in 32-key chunks, so the running max / sum / out round at other points (tolerance class against `sparse_sdpa`,
bit-exact run to run).  The acceptance mechanism is the same as for the other prefill modes (`NUMERICS.md`): `json` must
reproduce the CPU record 96/96 and the eleven divergence indices are reported against the pinned table.  The twelve
acceptance prompts are shorter than one slab, so the slab body itself is exercised by prompts of 2048 tokens and more
(the agreement corpus's long items, a long system prompt).

## What it costs and what it saves

Per device, per 2048-row slab (measured on 4x p150, 2026-09-09): a dense linear at 2048 rows runs 10-12x faster than
its 64 per-tile calls (GDN in-proj 6.03 -> 0.49 ms + 0.15 ms for the weight copy; QSA query-gate 4.28 -> 0.48); the GDN
kernel 1.34 ms in one call against 2.47 ms chained; `sparse_sdpa` at S = 2048 10.99 ms against 13.31 for 16 x 128; the
KV slab write 0.056 ms against 0.206.  What did not change with the row count in these measurements is the MoE
expert stream (16 calls of 128 tokens per layer per slab, the most the kernel then admitted per call) and its per-call
combine tilize and weighted reduce, together about 0.4 ms per prompt token: the slab's floor at the time.  The
tables of this section and the next were measured in that form; *The routed experts in one call* below is the
default since 2026-09-25 and lowers the floor.

Measured end to end on 4x p150 (2026-09-09, 32k context; the 6942-token prompt is 3 slabs, 6 long chunks, 2 short
chunks and a 30-row tail, so its rate blends the slab's with the remainder's):

| | 32-row chunks | `--long-chunks` | `--prefill-slab 2048` |
|---|---|---|---|
| ms per prompt token (6942-token prompt) | 3.3 | 1.55 | 1.04 (TTFT 7.27 s) |
| json acceptance | 96/96 | 96/96 | 96/96 |
| divergence table vs the pinned table | pinned | identical | identical |
| decode after the prefill | 19.9 tok/s | 19.9 tok/s | 19.7-20.1 tok/s |
| HF agreement, all 8416 positions (top-1 / clear top-1 / KL) | 0.9518 / 0.9607 / 0.0543 | same | 0.9515 / 0.9605 / 0.0544 |
| HF agreement, the two long windows (160 positions) | 0.9938 / 0.9938 / 0.014 | same | 0.9812 / 0.9812 / 0.0225 |

The agreement column is the server's `--agreement-reference` run against the HF reference; every part except the
two long windows (an 8192- and a 32704-token prompt, the only items longer than a slab) is bitwise the chunked
column, and the long windows moved two of 160 top-1 positions (the tolerance class above).  Ready after 214 s on a
warm cache (the slab's warm pass through the 48 layers takes 31 s and its capture 5.5 s).

Long prompts (one chat completion each, the prompt a body of natural text, `max_tokens` 64; TTFT is the request's
time to its first token, the rate the server's prefill ms per prompt token):

| prompt tokens | `--long-chunks` TTFT / ms per token | `--prefill-slab 2048` TTFT / ms per token (32k context) | the same at a 64k context |
|---|---|---|---|
| 4,179 | 6.36 s / 1.51 | 4.17 s / 0.98 | 4.40 s / 1.04 |
| 8,176 | 12.3 s / 1.50 | 8.85 s / 1.08 | 9.22 s / 1.12 |
| 16,322 | 24.4 s / 1.49 | 15.7 s / 0.96 | 16.5 s / 1.01 |
| 31,988 | 47.4 s / 1.48 | 28.0 s / 0.87 | 29.8 s / 0.93 |
| 64,655 | - | - | 58.1 s / 0.90 (31 slabs) |
| 127,872 | - | - | 151 s / 1.18 at a 128k context (62 slabs; measured before the input pipeline below) |

Where the slab's time goes: the device runs one slab in 1.6-1.8 s (0.78-0.88 ms per token).  The traced body is
19,161 programs; under the device profiler on 32,000 tokens of natural text it is kernel-bound at 1.52-1.60 s per
slab (1.58 s of kernel time at P = 0, 1.51 s at P = 28672): the 768 `moe_compute` calls 0.25-0.31 ms per token (their
time follows the experts the tokens select: 0.20 on a slab of repeated tokens), the collectives 0.11, the element-wise
ops 0.06, the dense matmuls 0.045, `sparse_sdpa` 0.04-0.06, the combine tilize + weighted reduce 0.07; what the server
measures beyond the body (under 0.1 s per slab) is the input copies and the dispatch of the replay.  The host's work
per slab is the preparation of the next slab's inputs (the n-gram lookup of 2048 tokens, read from the table in one
batch, and the row packing: 0.1-0.25 s, against 0.36 s when the rows were read one token at a time) and the enqueue
of their copies, a few ms.  The driver overlaps that preparation with the running replay: the copies are queued behind the
replay on the same command queue (in order, so the device finishes reading the input buffers before they change), and
the driver waits for slab k - 1's event only once slab k's replay is queued, then prepares slab k + 1 while slab k runs.
Only the first slab's preparation is exposed, so a prompt of N slabs costs N device periods plus one preparation; the
served tokens are bitwise those of the un-overlapped driver (the same bytes reach the same buffers before the same
replay).

The slab takes 41 percent less time per prompt token than the 128-row chunks at 32k (0.87 against 1.48 ms: TTFT 28.0 s
against 47.4 s) and 35 percent less at 4k.  The slab body itself runs at about 0.86-0.88 ms per prompt token at 32k
(about 1,150 prompt tokens per second); the shorter prompts pay more for their remainder (the 128- and 32-row chunks
after the last slab), the first slab's preparation and the fixed hand-off.  A 64k context (`--allocated-context 65536`)
costs about 4 percent more per token (the QSA selection scores over twice the blocks) and 56 MB per DRAM bank more; a
128k context about 13 percent more per token (226 MB per bank left); the slab state and trace themselves cost 45 MB per
bank at any context.

Memory: the slab state adds about 200 MB per device at 32k (one set of GDN pass buffers shared by the 35 GDN layers,
the PLE rows, the QSA slab constants and kept rows, the slab trace); the transients inside a layer peak around
150 MB (about 350 MB at 256k, where the score all-reduce per 512-row block is the largest).

## The routed experts in one call (`QWEN38_MOE_SLAB_ONE_CALL`, `QWEN38_MOE_SLAB_RINGS`)

Since 2026-09-25 the slab routes all of its rows through one `moe_compute` call per layer (`QWEN38_MOE_SLAB_ONE_CALL`
unset or `1`) on the op's local output path: each expert's tokens are packed into 32-row chunks once per slab (223
chunks per layer per device on natural text against 745 in 128-row blocks), the op writes each token's expert output
straight into the slab's `[10, 2048, 2560]` page, and the weighted reduce runs over that page in 512-row blocks.  The
rows of the experts a device does not hold are not zero-filled: the page is zero at allocation and only ever holds
finite expert outputs, which the reduce multiplies by an exact 0 where unowned.  The one call is bitwise the 16 x
128-row blocks: on 4x p150 the twelve acceptance records, the 3232 agreement rows and the four long-prompt completions
are identical at three heads.  `QWEN38_MOE_SLAB_ONE_CALL=0` restores the blocks.

Since 2026-09-26 that weighted reduce runs as one program per layer instead of the 21 (four times a
k-strided slice, a tilize, two routing slices and the fused reduce, then a concat): `ttnn/fused/moe_combine` reads the
page's owned rows straight into tiles, weights them with the fused reduce's own multiply-accumulate in its slot order
and packs once, so it is bitwise the blocks by construction; measured on one Blackhole die at 2048 rows, 0 of
5,242,880 output elements differ from the blocks, 2.66 ms -> 0.14 ms of kernel per layer (128 -> 6.5 ms per 2048-row
slab).  On the 4x p150 line (the gate of 2026-09-26, 32k context) the twelve acceptance records, the 3232 agreement
columns and the four probes are identical to the blocks', the 31,716-token prompt reaches its first token in 12.38 s
against 14.30 (0.389 against 0.450 ms per prompt token), and under the device profiler the combine of one 2048-row
slab takes 6.33 ms over its 48 layers against 128.3 ms for the blocks (the slab's kernel time 816.5 -> 695.6 ms,
2,484 -> 2,913 tok/s over the slab; against the chain before the block-shared attention, 899.5 ms / 2,257 tok/s, the stack
is -22.7 percent).
`QWEN38_FUSED_OFF=moe_combine` restores the 512-row blocks; `QWEN38_MOE_COMBINE_COLS` (the column tiles per
work unit, 16 by default; 8 measured equal within 0.5 percent on the 130-core dies of the 4x p150 line, 2026-09-26) is the one
knob worth a sweep on a new die class.

Per device, per layer, per 2048-row slab on one p150 with a captured natural-text routing, the expert stream takes
3.25 ms in one call against 13.45 ms in the 16 calls.  Under the device profiler on the 4-chip line (P = 0, a 2048-row
slab of a 32k record, one chip) the slab's kernel time falls from 1425.3 ms (`moe_compute` 510.1 ms over 768 calls,
19,161 programs) to 1044.9 ms with the one call (`moe_compute` 158.7 ms over 48 calls, 13,401 programs; -26.7 percent,
1,426 -> 1,945 prompt tokens per second of kernel time) and to 971.4 ms with two rings (`moe_compute` 98.3 ms; -31.8
percent, 2,092 tokens per second).  Beside `moe_compute` the one call saves the per-block combine tilize (-25.6 ms,
988 -> 412 calls) and the combine reduce-scatter (-21.9 ms; -35.5 with two rings) and adds 21.3 ms of `slice` (the
reduce's fewer, larger k-strided slices of the one-call page); the page costs 12.2 MB more DRAM per bank.  Served
(4x p150, 2026-09-25, 32k context, natural-text prompts, `max_tokens` 64, TTFT the request's time to its first token
and the rate the server's prefill ms per prompt token):

| prompt tokens | 16 x 128-row blocks | one call, one ring (`QWEN38_MOE_SLAB_RINGS=0`) | one call, two rings (the default) |
|---|---|---|---|
| 2,118 | | 1.59 s / 0.736 | 1.54 s / 0.712 |
| 2,764 | | 2.40 s / 0.855 | 2.31 s / 0.826 |
| 25,546 | | 14.75 s / 0.576 | 13.90 s / 0.543 |
| 31,716 | 24.73 s / 0.779 | 18.09 s / 0.569 | 17.06 s / 0.537 |

At 32k the one call on one ring takes 27 percent less time per prompt token than the blocks and the two rings 31
percent less (the blocks were measured at the longest prompt only in this pass).  The line gate of 2026-09-25 with the
ring exchange's backpressure credit in place (the served kernels of this landing) measured, per 2048-row slab under
the profiler, 1041.7 ms of kernel time on one ring (1,951 prompt tokens per second) and 971.6 ms on two rings
(2,090); served, 31,716 tokens reached their first token in 18.03 s (one ring) and 17.04 s (two rings), 2,118 tokens in
1.58 and 1.53 s; the acceptance records, the agreement rows and the completions were identical across the blocks, the
one ring and the two rings, and the decode after the prefill identical to the lineage's own run.

`QWEN38_MOE_SLAB_RINGS` selects how the one call streams the expert weights; only the one-call slab reads it, the
32- and 128-row chunks and decode never do.  Three rings are the default since 2026-09-26; `QWEN38_MOE_SLAB_RINGS=2`
restores two rings (the default of 2026-09-25) and `=0` one ring:

- `3` (unset): the chunks are split over three rings of cores, each reading the slices of the experts it owns.
  Bitwise on the 4-chip line at 32k (the acceptance records, the 3232 agreement columns and the four completions
  identical to the two-ring form's, two runs) and on one die over about 42,500 device-checked calls (captured and
  adversarial routings, every replay compared); TTFT 12.18 s at 31,716 tokens against 12.45 with two rings
  (-2.1 percent; 10.01 against 10.18 s at 25,546 tokens); per 2048-row slab under the profiler 681.1 ms of kernel
  time (2,974 prompt tokens per second) against 695.5 (2,914), the 48 `moe_compute` calls 1.84 against 2.08 ms each
  (88.2 against 99.6 ms per slab).  Three rings were refused on 2026-09-25, when two runs of
  the same prompts differed: the cause was the ring exchange's missing backpressure (a core's partial could be
  overwritten by its predecessor's next chunk), fixed the same day; the fix is what made two and three rings bitwise;
- `2`: two rings; bitwise on the 4-chip line in three runs; 73.5 ms less kernel time per slab under the profiler than
  one ring (971.4 against 1044.9 ms);
- `0`: the op's one-ring, three-slot weight stream (the form the one call was first gated with);
- `1`: one ring with each expert's weight slice read from DRAM once per slab (a replay ring); bitwise the one-ring
  stream on one die, not measured on the 4-chip line.

## Served rates

The README's prefill row, served through the chat server on 4x p150 at the 2026-09-25 head (the routed experts in one
call on two rings by default): the 32-row chunk trace at 2.4 ms per prompt token (413 / 418 prompt tokens per second in
two runs: about 410), flat from 2k to 261k tokens; `--long-chunks` at 1.27-1.33 ms (750-790); `--prefill-slab 2048` at
0.39 ms (2,562: TTFT 12.38 s for a 31,716-token prompt with the block-shared attention kernel and the one-pass MoE
combine, both defaults since 2026-09-26; 14.25 s with the attention kernel alone, 17.0 s with the chains; 1.228 s for
2,118 tokens; the table above and the rows below).  The release's
long-context figures of 2026-09-04, through the 32-row chunk trace alone at about 3.2-3.5 ms per prompt token: a 40k
prompt reached its first token in 125 s and a 200k prompt in 671 s, and decode stayed at 17-19 tokens/s to 256k (the
decode of that day; the README's decode rows are the current step).

The one-pass combine's served rows (4x p150, 32k context, `--prefill-slab 2048`, 2026-09-26, the block-shared
attention on in both arms; TTFT the request's time to its first token):

| prompt tokens | default (one-pass combine) | `QWEN38_FUSED_OFF=moe_combine` (the 512-row blocks) |
|---|---|---|
| 2,118 | 1.228 s | 1.345 s |
| 2,764 | 2.011 s | 2.130 s |
| 25,546 | 10.16 s | 11.65 s |
| 31,716 | 12.38 s (0.389 ms per prompt token) | 14.30 s (0.450 ms) |

## Block-shared attention (`QWEN38_FUSED=sparse_sdpa_tiled`)

The slab's QSA attention today expands every query's 512 selected blocks to 2048 token ids (five ops over a
`[rows, 2080]` uint32 tensor per layer), pads the six local heads to 32 with a zero V half and runs `sparse_sdpa`,
which gathers each query's own 2048 K/V rows: about 51 GB of cache traffic per 2048-row slab and 81 / 129 ms per slab
at P = 0 / 28672 on the 4x p150 line.  Consecutive queries select overlapping blocks (a tile of 16 queries needs 1.0 /
1.3 / 1.9 / 2.3 / 3.4x one query's blocks in union at P = 0 / 2k / 4k / 6k / 28k, measured on captured slab
selections), so `ttnn/fused/sparse_sdpa_tiled` runs the attention per tile of 16 consecutive queries x the six local
heads on one core: the reader builds the tile's block union (a seed block per row first, then ascending) and one
membership word per block, the union's K/V rows stream once per tile in 64-block chunks through `sparse_sdpa`'s
dual-NoC gather, the writer turns the membership words into an additive bf16 mask band per chunk, and compute runs
`sparse_sdpa`'s streaming flash loop with the band added before the running max.  The selection ends at the block
ids (no expansion, no zero half, no head pad, no output slice); the tile size follows the grid (16 queries where the
tiles fit one per core, else 32).  The kernel is the slab's default since 2026-09-26 at the op family's HiFi2 /
bf16-destination compute config (`QWEN38_SPARSE_SDPA_TILED_FIDELITY=hifi4` restores the slab's HiFi4 / fp32-DEST
form, `QWEN38_FUSED_OFF=sparse_sdpa_tiled` the chain): on the 4-chip line at a 32k context the 31,716-token prompt
reached its first token in 14.25 s against 15.92 with the chain (-10 %, 2026-09-26; under the device profiler the attention of one 2048-row slab takes 16.28 ms over its
12 layers against 89.77 ms for the chain's `sparse_sdpa` and expansion ops, on the prompt's first slab, whose kernel
time went 899.5 -> 816.6 ms, 2,257 -> 2,484 tok/s; the arm table in `NUMERICS.md`).  The chain is the fallback whenever the kernel's shape contract does not
hold.

Measured on one die of a QuietBox 2 (2x p300c, an 11x10 compute grid; a 32k context, the captured layer-3
selections, 2026-09-25) against the chain on the same die: at P = 28672 the chain takes 14.2 ms per QSA layer; the
kernel 4.65 ms with 16-query tiles (18 of the 128 tiles double up on the 110 cores) and 4.8 ms with the 32-query /
32-block-chunk tiling the model picks on this grid (64 tiles, one per core; measured before the mask-band rewrite that
took the 16-query figure from 5.5 to 4.65) -- 3x, 170 against 56 ms per slab; at P = 0 0.89 against 10.3 ms (11.6x).
Against the chain on the same inputs PCC 0.99954 / 0.99988; against an fp32 oracle no farther than the chain (PCC
0.99979 both, row rel_err p50 0.021).  The layer is bound by its dataflow side (the K/V gather plus the union and
band work on the RISCs, 3.8 ms) and its compute side (4.1 ms) alike.  `QWEN38_SPARSE_SDPA_TILED_FIDELITY=hifi2`
runs the kernel at the op family's own compute config (HiFi2, approximate correction exp, bf16 destination): 4.1 ms
with 16-query tiles and 3.6 with 32 on this die, PCC 0.99965 against the chain; the served default since the line
gate of 2026-09-26 (`NUMERICS.md`).  The one-tile-per-core layer time on the 13x10 grid of a p150
is the slowest single tile: the per-core kernel zones on this die read 1.5-2.6 ms per single tile at 28k (median
2.4) and 0.2-0.6 ms at P = 0, so about 2.6-2.9 ms and 0.6 ms per layer there at HiFi4 (less at HiFi2).
The running state's format is a switch (`fp32_state`): fp32 keeps the normalization to 0.4 % (the ones-V column)
against 1.2 % in bf16 but lands no closer to the oracle (the approximate probability exp shapes the row), so bf16 --
the chain's format class -- is the default.

## Glue forms (`QWEN38_PREFILL_GLUE`)

Read program by program, the slab body's non-MoE time is glue around a few kernels: the gated-residual read's
gather, the QSA block selection's masks and score sums, the GDN q/k preparation.  The glue forms are other
arrangements of the same ops for those terms, read only by the slab body (the decode path and the 32/128-row bodies
never see one).  Bitwise forms keep every output element's arithmetic and reduction order; tolerance forms move a
summation order or a rounding point and are judged like the slab itself (`NUMERICS.md`, the long windows against the
references).

The two bitwise forms run by default.  Measured on the 4-chip line (1x4 p150, a 32k context, `--prefill-slab 2048`,
2026-09-25) against the previous slab body in the same process form: the acceptance pins 12/12 identical; the device
column of the agreement corpus's two long windows identical at all 160 scored positions (KL 0.0, top-1 1.0 between
the two columns; both 0.9812 / 0.0225 top-1 / KL against the HF reference); the slab's kernel time 1584.6 -> 1524.3 ms
per 2048-row slab (-60.3 ms, -3.8 %: the gather 34.0 of it, the hoist 26.5, additive to 0.2 ms), 1283 -> 1334 prompt
tokens per second on the body, TTFT at 32k 27.57 -> 26.64 s (-3.4 %), 220 fewer programs per slab.  On the landed
base (BF8 dense weights, the compact expert layout, two DRAM readers per bank; 2026-09-25) the same pair measured in
the served path: TTFT 24.55 -> 23.64 s at 31,716 tokens (0.773 -> 0.744 ms per prompt token, -3.7 %) and 1.979 ->
1.910 s at 2,118 tokens, with the acceptance pins 12/12, all 3,232 scored positions of the agreement corpus and the
four probe completions identical between the two bodies.
`QWEN38_PREFILL_GLUE=today` restores the previous slab body; `QWEN38_PREFILL_GLUE=<name>[,...]` runs exactly the
named forms (a list that wants a default form names it); an unknown name, an exclusive pair or `today` beside a name
refuses to start.

- `gr_gather_generic` (bitwise, default) / `gr_gather_tuned` (bitwise, exclusive with it): the gated-residual read
  gathers its fp32 partials with the generic `all_gather` (0.23 ms per read at 42 GB/s of ingress), or with the async
  op at two workers per link and ten chunks per sync, instead of `today`'s async op at one worker and one chunk per
  sync (the slowest collective of the slab: 0.59 ms per read at 16 GB/s of ingress, 96 reads per slab).  The gathered
  bytes and their page order are the same; the reduce that follows reads the same tensor.
- `qsa_mask_hoist` (bitwise, default): the QSA block selection's causal block masks depend on the position and the row
  index only, so they are derived once per slab and read by all twelve QSA layers instead of derived in each (26.5 ms
  per slab of mask ops).  The masks hold 1 KiB per token of context at 2048 rows: 32 MiB at a 32k context, 64 MiB at
  64k, 128 MiB at 128k, 256 MiB at 256k.  The hoist is admitted once, when the slab's QSA chunk constants are built
  after the model (`hoist_masks` on them, with the numbers; one log line at the build names the decision and
  its numbers in the same words): the masks must be within the 64 MiB cap
  (`HOISTED_MASK_BYTES_MAX`) and the DRAM free per device at that point must hold them plus the slab's working set
  (the dense-linear admission's term: 988.5 MiB at 2048 rows).  So contexts to 64k hoist their masks when that DRAM
  is free; 128k and 256k contexts, or a process short of the headroom, derive the masks per layer as before (the
  bitwise slab body either way; only the mask programs' count differs).
- `qsa_scores_rs_ag` (tolerance, off): each 512-row score block's four device partials are masked on every device,
  summed by a `reduce_scatter` over the block's rows, ranked per device on its 128 rows and the block ids gathered, in
  place of broadcasting every device's scores to every device and summing locally.  A third of the score payload
  crosses the line; the four terms are summed in the collective's hop order, so a near-tie between two blocks can
  rank the other way and change that layer's attention set for that row.
- `gr_partial_rs_ag` (tolerance, off): the gated-residual partials are summed by `reduce_scatter` + `all_gather`
  instead of the gather + device-order reduce; the four fp32 terms are summed in hop order.  It supersedes the gather
  forms when both are named.
- `gdn_qk_flat` (tolerance, off): the GDN chunk kernel takes the raw convolution q/k rows in its flat form and maps
  value heads to key heads and l2-normalizes them itself, in fp32 with the scale folded in, in place of the model's
  0/1 expand, bf16 `rms_norm` and bf16 scale (about 66 ms per slab of the model's and the kernel adapter's glue).  The
  normalized values differ by bf16 rounding and the recurrent state carries the difference into the decode.

`gdn_prefill_rows` (bitwise, on by default; `QWEN38_FUSED_OFF=gdn_prefill_rows` restores the chain and
`QWEN38_FUSED=gdn_prefill_rows` names it explicitly; a fused kernel, not a glue form) replaces the slab's whole GDN glue --
the projection's landing slices, the FIR row shifts, the causal convolution and SiLU, the q/k expand, both `rms_norm`s
and scales, the v row mask, the beta and log-decay gates, the chunk adapter's head-major relayout and q scale, the
gated epilogue's typecast, weighted norm, head fold and sigmoid gate, and the next pass's history tile -- with two
`generic_op` programs, `gdn_pre_rows` before the recurrence and `gdn_post_rows` after it.  Between them the two phases
of the GDN kernel are called directly as `ttnn.prim.chunk_gdn_prep` and `ttnn.prim.chunk_gdn_scan` on the pad-free
per-chunk pages the first program writes, with the arguments the composite passes, so the recurrence itself is
unchanged.  Every arithmetic step is the chain op's own kernel call sequence in the chain op's destination width with
one pack per op, and the row shifts, the head expand, the page maps and the gate column transposes are data movement,
so the gated rows, the recurrent state and the history tile are the chain's bits.  Measured on one p150 die at 2048
rows, traced: 332 us per call for the pre program and 115 for the post pair, 447.6 us per layer for the three
together, against 3,270 us per layer for the ops they replace -- about 102 ms per 2048-row slab over the 36 GDN
layers.  The form is slab-only: the 32-row, 128-row,
verify-lane and decode bodies never reach it, and it allocates its own pass buffers (the prims' `[12, NC, 32, 128]`
q/k pages, the `[12, NC, 32, 1]` gate columns and the four hand-off buffers) in place of the chain's, shared by the
GDN layers as the rest of the slab body is.  On by default since 2026-09-26, measured on the 4x p150 line against `QWEN38_FUSED_OFF=gdn_prefill_rows` in two
independent runs, both bitwise the chain and identical to each other (acceptance pins 12/12, records 12/12, the 3,232
agreement columns including the 160-position long window, the four probes): per 2048-row slab at P = 0 the kernel time
681.24 -> 576.03 ms (-105.2 ms, -15.4 %; 2,973 -> 3,518 prompt tokens per second on the kernel basis), 12,030 -> 9,438
programs (-216 per layer);
the three rows programs are 108 `GenericOp` calls and 15.84 ms per slab where the chain's glue took 110.5 (ReshapeView
35.95, Transpose 13.01, BinaryNg 11.98, Tilize 11.89, Slice 10.50, Matmul 7.80, Concat 7.02, Unary 3.43, Typecast 2.60,
UntilizeCodegen 2.26, FillPad 2.14, Copy 1.89), while `ChunkGdnScan` 19.55, `ChunkGdnPrep` 10.99 and `MoECompute` 88.19 ms
are unchanged;
DRAM per bank unchanged, 2.1 MB more free per bank at READY; the time to the first token at 31,716 tokens 12.15 -> 10.56 and 10.59 s
in the two runs (-13.1 %; 0.382 -> 0.332 ms per prompt token), 25,546 tokens 9.99 -> 8.72 s, 2,764 tokens 1.966 -> 1.873 s, 2,118
tokens 1.201 -> 1.086 s.

## Running it

```
models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile <profile> --devices <nodes> \
    --checkpoint <DIR> --cache-root <DIR> --prefill-slab 2048 [--acceptance --require-json-96]
```

The first start compiles the slab body's programs in the warm pass (an eager slab from the reset state) and captures
its trace after the 128-row chunk trace; `/health` reports `prefill_slab_rows`.  `QWEN38_MOE_SLAB_ONE_CALL=0` in the
server's environment restores the 16 x 128-row expert calls, `QWEN38_MOE_SLAB_RINGS=2` (or `0`) restores two (one)
rings for the one call; both are read when the slab's layer instances are built, so they take effect at the next start.

### The dense-linear switches

Four environment variables, read once when the model is built, move the slab's dense linears (the GDN in/out
projections, the gated-residual down+inject and up, the shared expert's gate/up/down/scalar, the QSA query-gate / K /
V / output projections) and nothing else.  The dtype, fidelity and accumulation defaults keep the slab's arithmetic
(the decode weights copied per slab in their format, `QWEN38_DENSE_WEIGHT_DTYPE`, with the module's compute config
for it; fp32 accumulation); the grid's default is `wide`, and
`QWEN38_PREFILL_DENSE_GRID=today` restores the earlier slab bitwise (the grids of
`decode_matmul.prefill_matmul_program_config`, no resident copy):

| variable | values | default | what it changes |
|---|---|---|---|
| `QWEN38_PREFILL_DENSE_DTYPE` | `bf16`, `bf8` | `bf16` | `bf8`: a resident DRAM-interleaved bfloat8_b copy of each weight, read in place by the slab (no per-slab copy); the decode weights stay bf16 |
| `QWEN38_PREFILL_DENSE_FIDELITY` | `hifi4`, `hifi2`, `lofi` | `hifi4` | `hifi4`: the module's own compute config (the fidelity of its weight format: HiFi4 for bf16, HiFi2 for the bf8 default, LoFi for bf4); `hifi2` / `lofi`: that fidelity for the slab's dense linears instead (`lofi` reads a 5-bit weight mantissa: a quality point to measure, not a free speed knob) |
| `QWEN38_PREFILL_DENSE_GRID` | `today`, `wide` | `wide` | `wide`: more columns for the narrow-N shapes (the query-gate 80 -> 110 cores, the gated-residual down+inject 30 -> 60), and the pair-grouped K/V and the shared expert's gate/up as one linear each on resident copies in the weights' own format (the same tiles the separate linears read: about 58 MB per device as bf8, 110 MB as bf16); `today`: the earlier grids and per-slab copies |
| `QWEN38_PREFILL_DENSE_FP32_ACC` | `1`, `0` | `1` | fp32 accumulation across the K blocks |

The wide grid, measured on the 4-chip line (1x4 p150, a 32k context, `--prefill-slab 2048`, 2026-09-25, with the
bf16 dense weights of record then) against `today` in the same process form: the device column of the agreement corpus's two long windows is identical at all
160 scored positions (KL 0.0, top-1 1.0 between the two columns; both 0.9812 / 0.0225 top-1 / KL against the HF
reference) and the 16-token completions of the long-prompt probe (2k to 32k tokens) are byte-identical; the dense
family (the Matmul class of the slab's device profile) 92.57 -> 75.84 ms per 2048-row slab (-18 %), the slab's
kernel time 1584.3 -> 1567.7 ms (-16.6 ms, -1.05 %: 1284 -> 1297 prompt tokens per second on the body, TTFT at
32k 27.56 -> 27.32 s); 60 fewer programs per slab (the fused siblings); +13.8 MB of DRAM per bank for the two
resident bf16 siblings and no other resident; the acceptance pins 12/12 identical.  The wide grid changes the
block order over the cores and fuses two pairs of sibling linears whose per-column arithmetic is the same (the
siblings are the separate linears' tiles in the same format), so its numerics class is `today`'s under any
`QWEN38_DENSE_WEIGHT_DTYPE`; the grid alone does not touch the fidelity or the accumulation.

The MoE router and the QSA index projections never read the switches: their outputs pick the routed experts and the
attended blocks, so they keep today's config, fidelity and per-slab weight copy under every setting.  The resident
copies (about 950 MB per device as bfloat8_b; the fused siblings alone about 58 MB as bf8) are allocated after the decode
weights and the resident experts and refused, with the numbers, when the DRAM free at that point would not hold
them plus what the slab process allocates after the build: the context-scaled state (417 MiB at 32k), the 2048-row
slab's working set and a 512 MiB margin.  The working set is a measured band (the 4-chip line, 32k, 2026-09-25):
the slab ran with 251.3 MB per bank free after the build and failed in its first layer with 141.0 (the bf8 copies
under the device profiler's default 64,000-program DRAM reservation), and the admission takes the band's upper end,
the only value known to fit, so at 32k it asks for 251.3 MB per bank after the weights.  The bf8 copies fit the
served process at 32k (443 MB per bank after them) and the profiler at a 32,000-program reservation (295 MB), not
the profiler's default (141 MB: refused with the numbers).  The switches act only on a build that runs a slab (the
chain tells the builder the `--prefill-slab` rows before the target is built; a slab above 2048 rows scales the
working-set term with its rows): a build without a slab, such as the single-user 256k server that prefills in
chunks, allocates no resident copy, runs no admission and is unchanged.  The acceptance gate's prompts are
shorter than a slab, so a non-default setting is judged on long prompts: the agreement corpus's long windows and the full-model
gate's 4k / 8k synthetic prompts with `--prefill-slab`.  The design, the arms and their expected gains: the prefill
dense design note (a development document, not shipped).
