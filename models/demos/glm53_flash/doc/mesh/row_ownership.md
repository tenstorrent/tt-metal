# Per-row prefill ownership for GLM-5.3-Flash serving on the 8x4 Galaxy (design study, 2026-10-22)

Scope: design and host-testable code only; no device was opened. Paths are relative to
`models/demos/glm53_flash` (MD) in `/home/ttuser/ashai/glm53/wt_mesh`; `d_p` = `models/demos/deepseek_v3_d_p` in
`/home/ttuser/ashai/glm53/tt-metal`; line numbers are of the files as read on 2026-10-22 17:40.

Goal (DESIGN.md section 2, STATUS.md items 6-7): `max_num_seqs = rows x users_per_row` (32 at 1 M, 256 at 128 K)
with each row prefilling its own prompt, behind `GLM53_VLLM_ROWS_INDEPENDENT`. Today `max_num_seqs = users_per_row`
(4 at 1 M) because the sequence-parallel (SP) prefill replicates one prompt into slot `user` of every row.

## 0. Summary

| item | result |
|---|---|
| why the rows replicate today | the SP prefill splits ONE chunk over the rows and (i) the KDA scan exchanges carries and conv history across the rows and returns the carry replicated, (ii) the DSA slab path all-gathers the chunk's latent / index stems so every row fills its own slot `user`, (iii) every row's slab attends over the full prefix, so the prefix must exist on every row. (iii) is the structural one: SP + chunking needs the whole prefix on every row, i.e. a full-length slot per row for the prompt in flight. |
| cheapest capacity fix (SP + owner-row masks, the README's "next step") | needs a scratch slot on the 7 non-owner rows (the prefix for later chunks): the planner's "partitioned + scratch slot" column gives 0 users at 1 M, 4 / row at 512 K, 12 at 256 K, 28 at 128 K (`doc/mesh/README.md` L700-712). It does not reach the 32 x 1 M cell. Owner-row attention (queries gathered to the owner, no scratch) keeps the cell but costs ~8x the DSA attention on the owner row: ~2x the chunk time at 1 M (section 2.5). |
| recommended mechanism | **(a) row-local waves**: a wave = one chunk per row, each row's chunk from a different prompt; attention and the KDA scan are row-local (as in decode), the per-token expert collectives stay (the hybrid layout's intermediate split over the rows is per token: mixed prompts are what decode already does). Memory per chip unchanged at the default wave width (2 048 rows per row = today's 16 K chunk). The whole-prefix copy, score block and regions buffers are per chip and unchanged. |
| recommended policy | **(c)**: a prompt is prefilled by today's SP path only while that is exact (slot `user` is free on every other row and all its earlier chunks were SP), otherwise by a wave. SP keeps today's lone-prompt TTFT (27 s at 64 K, 460 s at 1 M) at low occupancy; waves give the capacity and the burst throughput. |
| aggregate prefill tok/s (section 2.1 model) | at W = 2 K rows per row: 2.2-2.4 K tok/s (unchanged, 8 prompts advance together); W = 8 K: 2.6-6.5 K; W = 16 K: 2.9-7.3 K. The spread is the unmeasured split of the 7 s chunk between per-wave fixed cost and per-row-token work; the measurement that pins it is one row's chunk time vs W (needs the row-local path, or the (1,4) proxy). |
| per-prompt TTFT | lone prompt under SP: unchanged. A prompt in a wave advances W rows per wave: at W = 2 K a 64 K prompt takes ~224 s alone in a wave (8x worse than SP), 8 prompts together 224 s each (SP: 27..216 s sequential). Waves beat SP on mean TTFT only for bursts of >= 5-13 prompts; hence policy (c). |
| planner envelope | unchanged at W = 2 K; each +2 K of W costs ~0.5 GiB of activations per chip = 1 user per row at 1 M (32 -> 24 at W = 8 K), ~0 at 128 K. `reserve_bytes` gains an activation term (section 3). |
| critical-path dependency | the KDA chunk kernels bind `sp_rank` / `sp_size` from the mesh shape (`chronology_binding.hpp` L16-32, the conv and select-final-carry program factories): row-local KDA on the 8x4 needs an explicit `sequence_parallel_size = 1` attribute (a ttnn op change outside MD; the sp_size == 1 kernel paths are the single-row paths already in CI). |
| phase 1 constraint | every row of a wave uses the same slot index `user` (host ints keyed by `user` in the prefix copy, the index scorer's `cache_batch_idx`, the KDA carry read). Phase 2 lifts it with the tensor-argument `ttnn.slice` (already used by `model_head.tile_logits`), per-chip page-table views and one-hot state reads; it is required for waves to be effective at full occupancy (section 2.6). |
| host slice delivered | `tt/vllm_config.py`: wave-affine placement, `WavePlanner`, SP eligibility; `tt/generator_vllm.py`: the wave path of `prefill_forward` behind `GLM53_VLLM_ROWS_INDEPENDENT=1` (refuses a model without `prefill_wave`); 8 new host tests. Default behaviour unchanged. |

## 1. Cross-row dependencies of today's prefill path

| # | dependency | where (file:line) | what crosses the rows | design (a) treatment |
|---|---|---|---|---|
| 1 | chunk layout: one prompt chunk = `rows` consecutive slabs | `tt/mesh_model.py` `prefill` L187-197 (`unit = 32 * rows`, ids uploaded `rows=3`: each row embeds its slab), `prefill_streams` L204-212 (`T_pad = shape * rows`, `owner_row, local_last = divmod(S-1, T_loc)`), `prefill_chunked` L220-221 (chunk a multiple of `32 * rows`), `tt/trace_prefill.py` `supported_prefill_lengths` L56-66 (unit `TILE * rows`), `PrefillTraceInputs.allocate` L231-233 (`T_loc = T // rows`), `owner_row` / `last_row` L268-274, `tt/functional_decoder.py` L441 and `tt/decoder_layer.py` L233 (`T = streams.shape[-2] * rows`) | the padded chunk is split by row; `S` locates the last token on one row | a wave is `[rows, T_pad]` ids (one prompt per row); `T = streams.shape[-2]`; the last token is per row |
| 2 | KDA sequence-parallel scan | `tt/kda.py` `prefill_layer` L360-397 (`ttKDA(sp_axis=0, tp_axis=1, active_seq_len=global)`, `summary_group_chunks` over `global / rows / 32` L365-371; cap 24 K global: README L386), `prefill_forward` L399-421 (carries "replicated over the rows" L404-406); `d_p/tt/kda/kda.py` L159-167 (`sequence_parallel_size = mesh.shape[sp_axis]`), the conv history exchange L497-512 (`exchange_buffer` / `exchange_semaphores`, `_HistoryExchange` L60-105), `d_p/tt/kda/recurrence.py` `distributed_prefix` L250-301 (row all-gather of the affine summaries + `chain_affine_transforms`), `_run_sp` selection L599-636; kernels: `ttnn/cpp/.../kda/factory/chronology_binding.hpp` L16-32 (`sp_rank` / `sp_size` compile-time args from the mesh coordinate), `qkv_causal_conv1d_silu_program_factory.cpp` L51, L98-99, L131-133, `select_final_carry_program_factory.cpp` L50-52 | the conv carry comes from the predecessor row, the recurrence entry state from the chain of the earlier rows' transforms, the final carry is selected and replicated | row-local scan: ttKDA with an explicit SP size of 1 on the 8-row mesh (op change), local chunk count `W / 32` |
| 3 | KDA carry write into every row's slot | `tt/functional_decoder.py` `_write_user_kda_state` L333-360 (host slice/concat of slot `user`), `_write_user_kda_state_slot` L308-331 (device-slot form: paged fill with `slot.user_idx`, one-hot conv masks), `prefill_forward` L459-471 (comment: "slot `user` of EVERY row ends up holding this prompt"), `next_start` per user L222, L455-458, L471 | the replicated carry lands in slot `user` on all rows | each row writes its own carry; idle rows write a dummy KDA slot (`allocate_state(max_batch + 1)`) through a row-sharded `user_idx` and an all-zero conv mask; `next_start[row][user]` |
| 4 | DSA slab path: chunk stems gathered over the rows, cache filled on every row | `tt/decoder_layer.py` L252-259 (slab when `slot is None`), `tt/dsa.py` `_prefill_slab` L1093-1200: `all_gather_rows(c_loc / k_loc / g_loc, dim=2)` L1131-1133, `cache.fill(c, user, start)` L1137, `indexer.maintain_prefill(kg=...)` L1138-1140; the per-row query geometry is row-sharded already (`_row_offsets`, `row_starts_of` L1171-1183, `tt/indexer.py` `select_pools(row_starts)` L651-668, `_bounds_const(per_row=True)` L817-831) | the chunk's latent and index stems (3 x [1,1,T,512/128/128] bf16) are gathered so the replicated cache / index receive the whole chunk on every row | no gathers: `prefill_forward(slab=False)` (L1024-1090) is already row-local; the only cross-row ops in it are column collectives (`_prefix_rows` L1202-1214, `_sparse_attention(owner=)` L1304-1306). The scorer's per-row superset + bounded top-k (`select_pools` L651-668, 705-706) and the docstring at L839-840 ("mesh rows holding replicated or independent prompts score alike") already cover independent prompts at the same `start` |
| 5 | DSA v1 (gathered hidden) = the traced path | `tt/decoder_layer.py` L260-270 (`all_gather_rows(h)` + `reduce_scatter_rows(y)` x 1/rows), chosen for `slot is not None` (L252-259 comment) | the whole chunk hidden replicated, identical outputs reduce-scattered | the traced wave uses the single-row form directly (dense range, start 0): the same ops minus the two collectives |
| 6 | whole-prefix latent copy per chunk | `tt/dsa.py` `user_rows_rm` L527-544 (`page_table_host[prefill_slot(user)]` host block range, direct untilize for slot 0), `prefix_rows_rm` L593-634 (arena form, `_write_rows` host offsets L646-653), `_prefill_slab` L1168 / `prefill_forward` L1071; `tt/memory_plan.py` `reserve_bytes` L82 (`prefix_copy = isl * 576 * 2`, one per chip) | none (per row, each row copies ITS slot `user` = the replica) | unchanged per chip: the row's prompt. Phase 2 (per-row `user`): the tensor-argument `ttnn.slice` on the block dim (the form `tt/model_head.py` `tile_logits` L213-225 uses) + untilize; `n_rows = max_r(start_r) + T` (the bounded top-k never selects beyond a row's own prefix) |
| 7 | page tables, slots, dummies | `tt/dsa.py` `TtLatentCache` L331-407 (slots per chip L360-375, identity table L389-391, index L395-403), `_prefill_views_for` L448-472 (per-column `[cols, bps]` views replicated over the rows: owner's slot or the dummy row), `fill` L474-499 (`batch_idx_tensor=slot.cache_idx` for the traced form), `tt/indexer.py` `write_pools` L232-270, `write_ring_block` L297-304, `_write_tail_rows` L494-505, `tt/trace_prefill.py` `PrefillSlot` L106-205 (`cache_idx` column-sharded L137-144, L170-174; `user_idx`, conv masks replicated L146-154) | every row's chips write slot `user` (owner column) / dummies (other columns) | the per-chip slot tensor becomes row-and-column sharded (`mesh.shard_2d(0, 0)`, `tt/mesh.py` L113-118): the owner (row, column) names the slot, the other columns the dummy row as today, and idle rows the dummy row too (`heads2users` has `DUMMY_BLOCKS` L62, `dummy_slot` in the index L120); eager fills with `start > 0`: per-chip page-table view rows `[rows * cols, bps]` uploaded per wave |
| 8 | index maintenance host bookkeeping | `tt/indexer.py` `maintain_prefill` L408-470: `r = start % P` L449 (always 0 for 32-aligned starts: P = 4), `lead`, the tail rows `slot0 = r` ..., the traced form writes every padded pool + a selector-matmul ring tail (L434-447, `start == 0` only) | none (host ints equal on every row today) | per-row `seq_len`: the `ring_sel` device form row-sharded (`[rows,1,32,T]`), generalised to `start > 0` with `start_pool` from a per-chip pool page-table view; the straddling-pool branch is dead for 32-aligned starts |
| 9 | routed experts: hybrid intermediate split over the rows | `tt/moe.py` `TtRoutedExperts.forward` L1174-1197 (`all_gather_rows(x, dense_w, idx, w)` L1183-1188, `reduce_scatter_rows(y)` L1196), `_forward_regions_chunked` L1032-1050 (`REGIONS_T` sub-chunks L94-96), `regions_keep_stacked` L90-101 (a second copy of the SAME hybrid slices for the decode block op, not a full-width copy), EP layout `combine_rows` L1503-1525 / `_combine_rows_a2a` L1564-1570; `tt/memory_plan.py` `expert_bytes_per_chip` L53-58 (`I / rows` per chip) | every row's tokens visit every row's expert slice (per token) | kept as is: the gathered `[rows * W]` tokens are 8 prompts' rows instead of 8 slabs of one; identical to decode. Idle / padded rows cost expert work proportional to their share |
| 10 | trace capture | `tt/mesh_model.py` `capture_prefill` L277-309 (inputs allocated before any capture, lengths from `supported_prefill_lengths(rows)`), `_prefill_graph` L256-275 (head `tile_logits` with replicated tile bounds), `prefill_traced` L311-328 (`owner_row(S)`, `next_start[user] = S`), `PrefillTraceInputs.set_prompt` L276-286; `tt/generator_vllm.py` warm-up order L286-339 | the ids are row-sharded (one slab per row); `end`, `ring_sel`, tile bounds replicated | the same graph; `end` `[rows,1,1,1]`, `ring_sel` `[rows,1,32,T]`, tile bounds `[rows,4]` row-sharded; `T_loc = T` (a traced length is per row); `set_prompt` takes `rows` prompts + `S_r` |
| 11 | the head's last token | `tt/mesh_model.py` L209-211 (`last_token_logits(streams, local_last + 1)` host int on the owner row), `tt/model_head.py` `last_token_logits` L205-211, `tile_logits` L213-225 (tensor bounds) | only the owner row's result is the prompt's | eager waves use `tile_logits` with row-sharded bounds; each row's id row `(S_r - 1) % 32` is read on the host |
| 12 | planner accounting | `tt/memory_plan.py` `slots_per_chip` L147-150 (+ scratch), `reserve_bytes` L75-86 (trace, regions, score block, prefix copy, misc; no activation term), `fits` / `max_users_per_row` L158-180; README envelope L700-712 (32 x 1 M, 64 x 512 K, 128 x 256 K, 256 x 128 K = slots, not distinct requests) | the "users on the mesh" column counts replicas as users | unchanged at W = 2 K (today's `[2048, 16384]` fp32 slabs live inside `misc`); add `act(W) = max(0, W - 2048) x 256 KiB` (4 live stream tensors of `W x 16384` fp32) |
| 13 | vLLM side | `tt/vllm_config.py` `ServingGeometry.rows_independent` L48, `request_capacity` L79-81, `chunk_unit` L87-91, `SlotTable.allocate` L194-221, `flat_indices` / `read_index` L177-188, `device_feed` L252-268, `PrefillPlanner` L336-377 (one prompt per plan); `tt/generator_vllm.py` `initialize_vllm_model` L225-226 (`users_per_row = max_batch_size // rows` when independent), `prefill_forward` L352-430 (prompts one at a time, L370-371) | capacity = `users_per_row`; a request's token is fed to its slot on every row | delivered: section 4 |
| 14 | mHC implementation select | `tt/mhc.py` `TtMHCSplit.select` L142-143 (decode path iff the padded row count is 32): a 128-token prompt is 16 rows per row today, so it takes the decode-tile path (README L639-644: ~0.04 PCC on that prompt) | not a dependency; a behaviour change | a 128-token prompt in a wave is 128 rows per row: the prefill wrap path (the long-prompt path) |
| 15 | row determinism | README L817-823: the 8-row reduce-scatter of the expert partials associates differently per destination row, so rows differ in the last ulp at near-ties; users within a row agree | the same prompt on different rows may differ at near-ties | accepted (not required); see risks for the SP-vs-wave case |

Not a cross-row dependency but worth stating: the EP expert layout (`GLM53_MOE_LAYOUT=ep`) and the TP all-reduce
(`tt/mesh.py` `all_reduce_cols` L375-387) are per token / per row and need no change.

## 2. Candidate designs

### 2.1 Cost model and the measured points

Per-row chunk time `t(W) = F + p * W` for W rows per row (the 8 rows run in parallel; a wave's time is the slowest
row's). Measured: today's SP chunk is W = 2 048 rows per row (a 16 K chunk) and takes **7.0-7.6 s** on the 8x4 (45
layers, bfp8, regions experts, README L507-509), i.e. `F + p * 2K = 7 s`. The split is not measured:

* the traced 128-token TTFT (266 ms, README L909) is the hard floor of F (16 rows per row, no dispatch cost);
* the (2,4) block's 8 K chunk (4 K rows per row) takes 3.60 s over 22 layers = 164 ms per layer against the 8x4's
  7.0 s / 45 = 156 ms per layer at 2 K rows per row (`doc/mesh/prefill_profile.md`; the block's expert slices are
  4x wider, its collectives 2-row): doubling the rows per row did not double the layer time, so F is a large share;
* the eager dispatch of ~4 500 ops per chunk and the attributed glue (26-58 ms per layer) put F at >= 1.5-2.5 s.

So **F in [2, 5] s and `p * 2K` in [2, 5] s**; every estimate below is given at both ends. The measurement that
pins it is one row's chunk time at W in {2K, 4K, 8K, 16K} (TP4, the hybrid expert slices), which needs the row-local
path (section 3) or the (1,4) proxy (a Galaxy row via `TT_VISIBLE_DEVICES`; its hybrid experts are full-width, 4x
the per-chip MoE work, so only the KDA / DSA / mHC terms transfer and the MoE term is taken from the regions kernel's
tokens x picks x width model).

Per-token DSA attention at long prefix is known on one chip: 0.43 s (16 K prefix) to 0.80 s (1 M) per layer per
16 K-row chunk (`doc/mesh/prefill_1m_cost.md` section 3, chip 0, TP1); with TP4 and the heads2users owner kernel
(x0.54-0.61, `doc/mesh/sparse_1m.md` section 7) ~0.25-0.5 s per layer per 16 K rows, i.e. ~30-60 ms per 2 K-row
slab, 11 layers ~0.4-0.7 s of the 7 s chunk.

### 2.2 (a) Waves of row-local chunks ("eight independent slabs")

A wave = up to `rows` requests, one per mesh row, each contributing one chunk `[start_r, end_r)` of its own prompt;
the wave's padded width `T_pad = max_r(end_r - start_r)` rounded to 32, every row's chunk padded to it with a per-row
valid end. Rows without a request are idle: their ops run on pad tokens and every state write goes to a dummy target.
Per layer: embedding per row; KDA row-local scan on `T_pad` rows from the row's own carry; mHC / norms / dense MLP /
shared expert per token; routed experts through the unchanged row gather / reduce-scatter (`rows * T_pad` gathered
tokens, the regions path in `REGIONS_T` sub-chunks); DSA through the single-row `prefill_forward` on the row's own
cache slot (the scorer superset over the rows' starts + the bounded top-k, the heads2users column exchanges).

| metric | value | assumptions |
|---|---|---|
| aggregate tok/s, W = 2 K | 2.2-2.4 K (= today) | same tokens per wave as today minus the SP collectives (3 row gathers per DSA layer, the KDA summary gather + conv exchange per KDA layer: ~2-5 ms per layer, 2-3 % of the chunk) |
| aggregate tok/s, W = 8 K | 64 K / (F + 4 x [2, 5] s) = **2.6-6.5 K** (1.1-2.8x) | linear p; the KDA grouped scan at 256 local chunks takes groups of 32 (`summary_group_chunks` L334-358: 8 groups x 16 heads = 128 > 120 owners at 16) |
| aggregate tok/s, W = 16 K | 128 K / (F + 8 x [2, 5] s) = **2.9-7.3 K** (1.2-3.1x) | groups of 128 (serial 128-chunk scans per owner core: the KDA term grows faster than linear; not in the bound) |
| per-prompt TTFT, lone prompt in a wave | W / t(W): 290 tok/s at 2 K (64 K: 224 s; 1 M: 3 600 s), 330-820 at 8 K, 360-890 at 16 K | 3-8x worse than SP's 2.3 K tok/s: one row instead of eight |
| per-prompt TTFT, 8 prompts in flight | every prompt advances at W / t(W); mean over the 8 = the same; SP would serialise them (mean (K+1)/2 x ISL / 2.3K) | waves win on mean TTFT when K > 2 x 2.3K x t(W) / W - 1: never at W = 2 K (K* = 15), K* in [4.6, 13.4] at 8 K, [4.2, 11.8] at 16 K |
| memory per chip | prefix copy: one (the row's prompt), unchanged; score block, regions buffers unchanged; page tables unchanged (+ the dummy row that `heads2users` already has); KDA dummy slot +35 MiB; trace buffers: one command stream per captured length, the same op count (a few MB each); activations `+max(0, W - 2048) x 256 KiB` (0 at 2 K, 1.5 GiB at 8 K, 3.5 GiB at 16 K) | 4 live `[W, 16384]` fp32 stream tensors per layer wrap |
| planner envelope | unchanged at W = 2 K (32 x 1 M, 64 x 512 K, 128 x 256 K, 256 x 128 K as DISTINCT requests); W = 8 K: -1 user per row at 1 M (24), -0 at 128 K; W = 16 K: 1 M cell needs 2 fewer users per row | 11.69 KiB per token per user (`cache_bytes_per_token`), ~2 GiB of headroom at every full cell (the envelope is sized to the chip) |
| vLLM scheduler model | section 2.6 | |

Exactness: every row runs exactly the single-row prefill (the stage-1 kernels and row classes) on its own prompt; the
only shared quantities are the scored width `kv_len` (the longest prefix in the wave; the bounded top-k keeps each
row inside its own prefix) and the padded width. With `W >= 2048` (the dense range) a chunk with dense rows is always
a prompt's first chunk (`start = 0`), so the mixed-chunk form of `_prefill_slab` (dense SDPA over the first
`min(T_pad, 2048)` rows on every row with per-row Q offsets, sparse over all rows, per-row select) applies with no
dense prefix tensor at all; `W < 2048` would need per-row dense prefixes of different lengths (not SPMD) and is
refused by the geometry.

### 2.3 (b) Full-width per-row chunks

The premise "a row runs a 16 K chunk alone with the stacked expert copy" does not hold: the stacked copy
(`GLM53_REGIONS_KEEP_STACKED`, `tt/moe.py` L90-101) is the same `[72 experts x I/8]` hybrid slices in the block /
stream layout for the decode op, not a full-width copy. A row holding all 288 experts at full width would need
42 x 72 x 3 x 4096 x 2048 x 1.0625 B = ~81 GB per chip (31.3 GiB available). So a row can never run the MoE alone:
the per-token row gather / reduce-scatter is inherent to the hybrid layout (and the EP layout's all-to-all alike), and
(b) is (a) at W = 16 K: the second and third rows of the table above, with the activation memory (3.5 GiB per chip:
fits the 256 K cell and below, not 512 K or 1 M at full occupancy) and the KDA serial-scan growth. The PP study's
"2-4x" (`doc/mesh/pp_design.md` section 4) is the same per-row-M argument; its removal of the row collectives applies
only to the SP collectives, which (a) removes at any W.

### 2.4 (c) SP for lone prompts, waves otherwise (the policy)

SP is exact for a request on `(row, user)` iff slot `user` holds no live request on any other row (the replicas land
in free slots, whose contents are garbage by contract: idle slots are parked at position 0 and a `start == 0` prefill
resets them) AND every earlier chunk of the request went through SP (a wave chunk exists only on the owner row, so a
later SP chunk would find no prefix elsewhere). A request may switch from SP to waves between chunks, never back.
Under this rule a lone arrival at low occupancy keeps today's TTFT; beyond `users_per_row` live requests SP is
impossible (every slot index has a live user on some row) and everything is waves. Numbers: the (a) table for waves;
today's 27 s / 460 s for SP.

Measured 2026-10-29 (45 layers, 8x4, 16 K needles): a lone 16 K SP chunk 7.7 s; a two-member wave step of 16 K 56 s
(8 waves of 2048 rows at ~7 s each: one eager 2048-row wave costs one 16 K SP chunk, since both are a 2048-row forward
per row), so an eager wave pays only near full occupancy and `SlotTable.allocate(prefer_sp=True)` (an eager prompt: no
traced wave holds it) takes a slot index live on NO row first (the lowest-load row), keeping up to `users_per_row`
concurrent long prompts SP-eligible; short prompts keep the wave-affine order (one traced wave of `rows` members beats
`rows` traced SP calls). r5's four 64 K needles in pairs of waves averaged 321 s of TTFT; as SP chunks ~75 s (the harness
measured 7.5 s per 16 K SP chunk at the 1 M profile, 45 layers: 16 chunk-steps for 4 x 65514 tokens in 120 s).

### 2.5 (e) SP with owner-row masks (the README's "next step") and owner-row attention

Masks alone (row-sharded slot indices with dummy targets on the non-owner rows, the mechanism of section 1 #7 and
#3) keep the SP compute but every row's slab still attends over the whole prefix, so the non-owner rows need the
prefix of the prompt in flight: a scratch slot per chip. Planner (README L700-712): partitioned + scratch = 0 users
at 1 M, 4 / row at 512 K, 12 at 256 K, 28 at 128 K against 4 / 8 / 16 / 32 without. The 1 M cell is lost. The
no-scratch variant gathers the absorbed queries to the owner row instead (`[1,16,2K,512]` bf16 per column per row,
32 MB, x 11 layers per chunk) and runs the 16 K-query sparse attention on every row over its own slot (SPMD; only the
owner's is kept): the DSA attention term is 8x today's per row, 0.25-0.5 s x 8 per layer at 1 M = 9-12 s per 16 K
chunk on top of ~7 s, i.e. ~1 000 s per 1 M prompt (2.2x today) but 2-4x better than a lone wave at W = 2 K..16 K,
with no scratch slot and 32 x 1 M capacity. It serves one prompt at a time, so for bursts waves are 1.3-2.5x better
on mean TTFT. Recommendation: not now; a follow-up for lone arrivals at full occupancy once waves exist (it reuses
the same row-sharded slot machinery plus one row gather per DSA layer).

#### 2.5.1 (e') SP compute with the owner-row cache through a selected-row push (the lone-prompt follow-up, 2026-10-31)

The variant above gathers the queries to the owner; the cheaper variant pushes the rows: every row keeps the
replicated pooled indexer keys (64 MiB per 1 M user, `TtIndexCache`), scores ITS slab's queries and selects its
top-k as today, then the owner row pushes the UNION of the selected latent rows of the chunk (0.6-5 % of the prefix
per chunk after dedup, the measurement of tt-metal #62997, i.e. 6-50 MiB per DSA layer at 1 M) to the other rows
through the row-axis links, and every row attends locally over its slab's queries (the paged op's gather half,
`custom_ops/paged_sparse_sdpa`, reading from a compact per-chunk region instead of the slot's pages). The fill
(2 MiB of new latent rows per row per layer), the pooled-key write and the KDA carry go to the owner row only; the
non-owner rows hold no copy of the prompt, so the 32 x 1 M cell stays. Cost: tens of ms per DSA layer of link
traffic plus the dedup kernel, an estimated 3-6 % on the 16 K chunk (~520-535 s per lone 1 M prompt against 505 s
today), with SP's full-mesh rate for every arrival, late or lone, and no thin wave ever (the wave path's weak case:
a resident wave user appending alone runs 3-8x slower than SP). Decode is unchanged (it already runs per owner
row); it gains only indirectly: a lone arrival's prefill stalls the decoders for 1/8 of the steps a thin wave costs,
and the 7 replicas an SP user keeps today return as DRAM headroom. Needs: the compact-row push (localize #62997's
mechanism into `custom_ops`), the per-chunk compact region in the paged op's reader, owner-only fill / pool / carry
writes (the row-sharded slot machinery of section 1 #3 and #7), one row gather of the selection per DSA layer.
Sequenced after the capstone (32 resident x 1 M) and the release; measured on the lone 1 M needle (TTFT vs 505 s)
and the "8, then 1 late" shape.

##### 2.5.1.1 What tt-metal #62997 actually provides (survey 2026-10-31 19:15, read-only; branch `ipotkonjak/sparse-kv-push-glm`, open)

Two experimental ttnn ops. `sparse_push_plan(indices, num_devices, stripe_chunk, max_rows, page_bytes, chunk_start)`
builds, on every chip, a bitmap of the selected cache rows (the dedup / union is on device: scatter into per-core
bitmaps, OR-reduce, rank per owner) and rewrites the indices to compact-buffer slots; the model all-gathers the 32
tables itself. `sparse_kv_push(cache, output, table, ..., ready_semaphore, data_valid_semaphore, part, num_links)`
then has every chip push, through the fabric (2D torus, 2 links, per direction one mux core + 2 sender cores, point
to point unicast NoC writes, no multicast), the rows IT owns to every requester's compact row-major buffer, and the
attention op reads that buffer instead of the cache. Both are trace-safe (program hash without addresses /
semaphores / batch index). Measured in the PR (DeepSeek-V3 sparse MLA prefill, the same 8x4 Galaxy, 2 links,
traced): plan 123-130 us, table gather 80 us, push 361-1717 us per layer (0.6-5.3 % of the 1 M rows survive the
dedup, median 25 K rows = 25 MB at 1 KiB rows), attention 0.93-1.0 ms; 1020 K prefill chunk 815 -> 621 ms, TTFT
126 -> 110 s (that model's whole 1 M prefill: 4.6x faster than our 505 s, a gap to profile on its own: section 3 step 4).

Constraints that shape a GLM53 use:

| the op needs | GLM53 today | consequence |
|---|---|---|
| the cache ROW_MAJOR, one row per page (<= the fabric payload, 6144 B in the GLM router config), sharded on its row dim over ALL mesh chips in row-major mesh order: global block `b` lives on chip `b % 32` (block-cyclic, `stripe_chunk` <= 512 rows) | paged bf16 TILE `[max_blocks, 1, 32, 512]`, replicated per chip of the owner row (`tt/dsa.py`) | a second, row-major, 32-way striped latent cache for the prompt being prefilled (1 KiB rows fit the payload), or a per-chunk RM staging copy |
| every one of the 32 chips in every push (requesters = all chips; the done fence counts `(N-1) x links x 2 + senders`) | SP = one prompt over the mesh (fits); waves = 8 prompts per step (does not: one push would couple the rows) | the push belongs to the SP path only; the "owner row pushes to 7 rows" of 2.5.1 becomes "every chip pushes its stripe to 31" |
| `FABRIC_2D_TORUS_XY` with 4 distinct neighbours per chip, mesh >= 3 x 3 | torus is the 8x4 default (`tt/mesh.py`); the (2,4) block runs 1D/2D fabric | no block-level test of the push; mesh only |
| a C++ device op built on `FabricMuxConfig` / `fabric_mux_connection_*` / `tt_fabric_mux.cpp` | `custom_ops/` are Python `generic_op` program descriptors; the nanobind fabric surface has no mux config | not portable into `custom_ops/`; usable only as upstream ttnn once #62997 merges (no code outside the model folder) |

Decode is the open question of this route: with the prompt striped 32 ways, every decode step would need its own
plan + gather + push per DSA layer (~0.3-0.5 ms each: +4-7 ms on the 55 ms step for the DSA layers alone, more if
the selection is wider than the PR's), so the striped cache must be consolidated into the owner row's paged TILE
cache once per chunk (the chunk's new rows: 16 MiB per DSA layer per 16 K chunk, a layout conversion plus a
row-axis gather, well under the chunk's 7 s), and decode stays exactly today's path. That consolidation is also
what keeps the 32 x 1 M cell: the striped copy lives only while the prompt prefills (1/32 of the prompt per chip)
and the owner row holds the only full copy afterwards.

Route, in order, after #62997 merges (track it; nothing to port before then): (1) profile the 1 M SP chunk op by
op against the PR's 621 ms / 5120-token chunk (section 3 step 4); (2) the striped RM cache + per-chunk consolidation on the lone
1 M needle with the upstream ops, SP compute unchanged (measure against 505 s and the DRAM headroom: the 7 replicas
return); (3) only then the owner-row attention over the compact buffer in `custom_ops/paged_sparse_sdpa` (a
"compact source" reader mode). Not pursued: re-implementing the fabric push in `generic_op` (no mux surface), or a
wave-path push (the fence couples the rows).

### 2.6 (d) Mixing prefill rows with decode rows

Every op is dispatched to the whole mesh (one command stream per mesh; a captured trace replays on every chip), so a
row cannot run a decode step while another runs a prefill wave: the idle rows of a partial wave execute the wave's
ops on dummy data. True mixing needs sub-mesh dispatch (per-row `MeshDevice` submeshes with their own command
queues, the model's tensors re-bound per submesh, CCL semaphores and traces per submesh): out of scope. The
practical "mix" remains the interleave policy (prefill waves and decode steps alternate; DESIGN.md section 4): under
waves a decode user's stall per interleaved prefill step is the wave time (7 s at W = 2 K, the same as today's chunk),
and the active-slot mask item is unchanged.

### 2.7 vLLM scheduler model for (a) + (c)

| vLLM quantity | mapping |
|---|---|
| `max_num_seqs` | `rows x users_per_row` (32 at 1 M, 64 at 512 K, 128 at 256 K, 256 at 128 K); the adapter derives `users_per_row = max_num_seqs / rows` and refuses a non-multiple |
| `max_num_batched_tokens` | the prompt tokens the plugin hands `prefill_forward` per step; the adapter forms waves from them, so `rows x W` (16 K at W = 2 K) keeps one wave per step; larger budgets run several waves sequentially in the step |
| chunked prefill / `prefill_chunk_alignment` | per-row chunks need `start % 32 == 0` only (`chunk_unit = 32` when independent; the `32 x rows` unit is SP's); a member's `[start, end)` longer than W is walked in W-row device chunks across consecutive waves |
| slot choice for a new request | `SlotTable.allocate`, wave-affine order (phase 1: fill slot index `u` across the rows before the next `u`, `u` in column-interleaved order so consecutive `u`s sit on different columns; phase 2: rows first, columns balanced) |
| prefill vs decode | unchanged: `decode_interleave_enabled: false` until the active-slot mask; a decode step must not run while any slot is mid-prefill (the adapter's guard) |
| row choice at prefill time | a request's row is its slot's; the wave is whatever pending chunks sit on distinct rows (phase 2) / on distinct rows of the same `u` (phase 1) |
| different prompt lengths in flight | per-row `(start_r, end_r)`; `T_pad = max` rounded to 32; rows past their valid end are padding (KDA `actual_end`, the index ring selector, the head's tile row); the scorer width is the longest prefix in the wave; idle rows are pad tokens into dummy targets |
| SP fallback | the eligibility rule of 2.4, evaluated per request per step; an SP prompt writes replicas into free slots only |
| outputs | per row of the call: the first greedy token (device) or the last token's logits (host) from the owner row's tile row, as today |

Steady state at full occupancy: a free slot appears wherever a request finished, so pending prefills sit on random
`(row, u)`. With phase 1's uniform-`u` waves, 8 pending prompts at 1 M (4 slot indices) form ~3.6 waves per round
(80 tok/s per prompt, 640 aggregate), at 128 K (32 indices) ~8 waves of 1 (36 tok/s per prompt). Phase 1 is a
bring-up step; **phase 2 (per-row `u`) is what makes waves effective at full occupancy** (one wave per round: 290
tok/s per prompt at W = 2 K, 2.3 K aggregate, and the W > 2 K gains).

## 3. Recommended design and implementation steps

Mechanism (a) with phase 1 (uniform slot index per wave) then phase 2 (per-row slot index); policy (c); W = 2 048
rows per row by default (memory-neutral), raised per cell after the F / p measurement.

| step | file | change |
|---|---|---|
| 0 (ttnn, outside MD) | `ttnn/cpp/ttnn/operations/experimental/kda/factory/chronology_binding.hpp` L16-32, `qkv_causal_conv1d_silu/device/*_program_factory.cpp` L51, L98-133, `select_final_carry/device/*_program_factory.cpp` L50-52, the nanobind signatures | an explicit `sequence_parallel_size` attribute (default: the mesh extent on the axis); with 1 on a multi-row mesh every coordinate binds `sp_rank 0 / sp_size 1` and takes the existing single-rank paths (no exchange, no chain). `d_p/tt/kda/kda.py` L159-167 and `recurrence.py` L599-636: the override selects `_run_grouped` / `_run_direct` instead of `_run_sp` |
| 1 | `tt/kda.py` | `prefill_layer(active_seq_len, local=False)`: `local` builds ttKDA with SP size 1 and `summary_group_chunks` over `W / 32`; `prefill_forward(..., local=True)`; `scalar_rows(values)` = row-sharded `[rows,1,1,1]` uint32 `actual_start` / `actual_end` (the cached replicated `scalar()` stays for SP) |
| 2 | `tt/functional_decoder.py` | `next_start[row][user]`; `allocate_state(max_batch + 1)` (the dummy KDA slot, decode addresses `[0, B)`); `prefill_forward(..., wave=WaveCall)`: per-row `(start, end, active)` as device tensors (the `PrefillTraceCall` form generalised: `end` row-sharded, `slot` row-and-column sharded with the dummy slot on idle rows, zero conv masks there); the continuation carry read stays `ttnn.slice(state, [user])` in phase 1, a one-hot `[1,B] x [B, H*K*V]` fp32 select in phase 2 |
| 3 (dsa.py, another agent's file: design only) | `tt/dsa.py`, `tt/indexer.py` | eager wave fill with `start > 0`: per-chip page-table view rows `[rows*cols, bps]` (`_prefill_views_for` generalised with the wave's per-row slot / start and the dummy row for idle rows); `maintain_prefill`: the `ring_sel` form for `start > 0` (pools from `start // P` through a per-chip pool page-table view, the tail by the row-sharded selector; the `r = start % P` branch is dead for 32-aligned starts); `_sparse_rows(row_starts_of=per-row true starts)` (the slab hook with `start_r + q0`); `prefix_rows_rm(user, max_r(start_r), T)`; the traced wave = `prefill_forward(slot=, ring_sel=)` without the v1 gather. Phase 2: the tensor-argument `ttnn.slice` on the cache's block dim (row-sharded starts, uniform length) before the untilize in `user_rows_rm`; the pooled-key cache sliced to the row's slot the same way and scored with `cache_batch_idx = 0` (64 MiB transient per layer at 1 M) |
| 4 | `tt/moe.py` | none (per-token path). Note the pad / idle rows in the gathered tokens |
| 5 | `tt/decoder_layer.py` | `prefill_forward(..., wave=...)`: `T = streams.shape[-2]`; KDA `local=True` with the row-sharded scalars; DSA `self.attn.prefill_forward(h, cache, user, seq_len, start=, slot=, ring_sel=, wave=)` with no row gather / reduce-scatter; MoE unchanged |
| 6 | `tt/trace_prefill.py` | `supported_prefill_lengths(rows=1 unit)` for waves (`T % 128 == 0`, `T <= 2048`); `PrefillSlot`: `cache_idx` / `user_idx` `[rows*cols]` via `shard_2d(0, 0)` (owner (row, col) = slot, else dummy), conv masks `[rows*taps, B, 1]` row-sharded (all-zero on idle rows); `PrefillTraceInputs`: `T_loc = T`, `end` `[rows,1,1,1]`, `ring_sel` `[rows,1,32,T]`, tile bounds `[rows,4]` row-sharded; `set_wave(mesh, ids [rows,T], S_r)`; `last_row(S_r)` per row |
| 7 | `tt/mesh_model.py` | `prefill_wave(wave)` (eager: ids `[rows, T_pad]` row-sharded upload, per-row scalars, every layer with `wave=`, the head's `tile_logits` with row-sharded bounds, per-row `(first token, logits)`), `capture_prefill(waves=True)`, `prefill_traced_wave`, `has_prefill_trace` per form; `prefill` (SP) kept unchanged; `wave_per_row_users = False` (phase 1) / `True` (phase 2) class attribute read by the adapter |
| 8 | `tt/memory_plan.py` | `reserve_bytes(..., wave_rows=2048)`: `+ max(0, wave_rows - 2048) * 256 KiB`; `MemoryPlan.max_wave_rows(users_per_row, isl)`; the host test pins W = 2 K neutrality and the 1 M cell's W bound |
| 9 (delivered) | `tt/vllm_config.py`, `tt/generator_vllm.py`, `tests/test_generator_vllm_host.py` | section 4 |

Tests, in order:

1. Host (now): `tests/test_generator_vllm_host.py` (wave-affine placement, `WavePlanner` grouping and chunk walking,
   SP eligibility and the SP -> wave one-way switch, the adapter's wave path on the fake model, the per-row decode
   feed, the refusal of a model without `prefill_wave`); `tests/test_memory_plan.py` for step 8.
2. (2,4) block, `tests/test_mesh_wave_prefill.py` (new, after steps 0-7): (i) the same prompt through SP and through a
   wave on row 0 and on row 1: first token equal, 64 greedy tokens equal modulo near-ties, the owner row's slot state
   vs the SP replica (latent pages, pools, ring, KDA carry) within bf16 ulp; (ii) two different prompts in one wave vs
   each alone in a wave: bit-identical per row (the wave must not mix); (iii) idle-row mask: a live user on `(1, u)`
   decodes identically before and after a wave on `u` with row 0 only; (iv) per-row starts: row 0 at chunk 0 (mixed
   dense / sparse) with row 1 at 48 K (sparse) in one wave vs separately, bit-identical; (v) the traced wave at
   256..2048 vs the eager wave (first token, logits tile); (vi) `GLM53_LONG_USERS=all` at 16 K with a different
   needle per row.
3. 8x4 e2e: the 128-token first token vs HF (102647 on 45 layers, README L867-880) through a wave on each of the 8
   rows and through SP; the 64 K needle with 8 distinct needles per wave (`test_mesh_long_context`, 32 users);
   32 distinct prompts at 1 M (4 waves of 8 at W = 2 K: TTFT per wave, needles 32/32, users within a row agree when
   prefilled by the same path, rows agree modulo near-ties); then the vLLM server at `--max-num-seqs 32
   --max-model-len 1048576` with the 128/128/32 and 100/100/32 benchmarks (TTFT + decode tok/s from one run).

4. (next device slot, before any push work) the 1 M SP chunk profiled op by op: one 16 K chunk at the 1 M position
   under the device profiler (one DSA layer and one KDA layer isolated, then the whole chunk), split into indexer
   scoring, top-k, the sparse latent gather, attention, the KV all-gather across rows, KDA, MoE and the CCLs.
   Reference point: DeepSeek-V3's sparse MLA prefill on the same Galaxy runs a 5120-token chunk at 1020 K in 621 ms
   (#62997), 4.6x our per-token rate at 1 M (16 K in ~7.4 s); the profile decides whether the replicated-cache
   all-gather or the attention over the replicated prefix is the term to attack first, and the push of 2.5.1.1
   only matters if it is the former.

Risks:

| risk | note |
|---|---|
| step 0 is a ttnn kernel-attribute change in the KDA ops (7 files, outside MD) | the single-rank code paths exist (the (1,N) meshes); the change is where `sp_size` comes from. Without it there is no row-local KDA prefill on a multi-row mesh (a composite conv would still leave the scan kernels binding the mesh rank) |
| SP and wave numerics differ at the last ulp (different scan grouping, no reduce-scatter of identical copies) | rows already differ; two users on one row prefilled by different paths may diverge at near-ties. The rule "users within a row agree" holds per path; test 2(i) bounds the difference |
| phase 1's uniform slot index makes waves of ~1 at full occupancy | section 2.6; phase 2 is part of the plan, not optional |
| per-prompt TTFT regression when a wave holds one prompt | policy (c) is mandatory in the first version; beyond `users_per_row` live requests a lone arrival is 3-8x slower than SP (section 2.5 is the later mitigation) |
| wave width W > 2 K at a full cell | activations exceed the ~2 GiB headroom; the planner bound (step 8) refuses it; W is per deployment, not per request |
| idle rows' dummy writes | the heads2users dummies are "racing DRAM writes of garbage that nothing reads" (sparse_1m.md section 7); the KDA dummy slot is never read by decode (`[0, B)`) |
| pad / idle rows through the routed experts | finite (pad-token embeddings; the traced warm-up already runs dummy prompts); wasted expert work proportional to the idle share |
| the scorer width is the longest prefix in the wave | the score block is the same shape as today's worst case (`QBLOCK x pools` at the chip's longest live prefix) |
| KDA grouped scan at W >= 8 K | serial scans of 32-128 chunks per owner core; measure before raising W |
| trace inputs per row | 8x the host->device refresh bytes per replay (ids `4T` per row, selectors `64T`): still KB-MB |

## 4. Host-testable slice (delivered)

Files changed (none owned by another agent): `tt/vllm_config.py`, `tt/generator_vllm.py`,
`tests/test_generator_vllm_host.py`. Default behaviour (`GLM53_VLLM_ROWS_INDEPENDENT` unset) is unchanged: the
replicated geometry, the column-first placement and the sequential SP prefill path are what they were; the existing
tests pass unmodified except `test_slot_table_rows_independent_placement`, which pins the (previously unwired)
independent-rows placement and now covers both affinities.

`tt/vllm_config.py`:

* `ServingGeometry`: `wave_rows` (W, default 2048, `>= DENSE_PREFILL_ROWS`, a multiple of 32), `wave_affinity`
  (`"user"` = phase 1 uniform slot index, `"row"` = phase 2), `sp_fallback` (policy (c)); `chunk_unit` is 128 when
  independent (`WAVE_CHUNK_UNIT`: a wave member's pools are written per 32-pool tile, so every member start sits on
  the 128-token grid; vLLM's `prefill_chunk_alignment` 256 is a multiple, and the planner refuses an off-grid start
  with the cause: the plugin's KV-block fallback for a running request short of the grid, r3's `(2, 0, 224)` after a
  128-token prefix-cache hit, fixed plugin-side by waiting for a step that reaches the grid); `device_prefill_chunk`
  keeps SP's `32 x rows` unit.
* `SlotTable.allocate`: the wave-affine order under `rows_independent` + `"user"` (slot index `u` filled across the
  rows before the next `u`, `u` in column-interleaved order), preceded for an eager prompt (`prefer_sp`) by a slot
  index live on no row (section 2.4's measurement); the previous rule (column load, then row load) stays for
  `"row"`. `SlotState.sp_only` tracks the one-way SP -> wave switch; `sp_eligible(logical)` is the rule of 2.4.
* `WavePlanner.plan(pending)`: `pending = [(logical, start, end)]` -> an ordered list of `SPStep` (one request, its
  SP device chunks from `PrefillPlanner`) and `WaveStep` (`user` or None, `t_pad`, `members`: `(logical, row, user,
  start, end)`, at most one per row), walking every member's range in `wave_rows` device chunks; `WaveCall` is the
  plain-data form handed to the model.
* `PrefillPlanner.plan(..., wave=True)` chunks a member at `wave_rows`.

`tt/generator_vllm.py`:

* `initialize_vllm_model`: `max_batch_size % rows == 0` when independent; `wave_rows` from `GLM53_VLLM_WAVE_ROWS`;
  `wave_affinity` from the model's `wave_per_row_users`; refuses (clear message) a mesh model without
  `prefill_wave` when the knob is on.
* `prefill_forward`: under `rows_independent` the step's rows go through `WavePlanner`; SP steps take `_prefill_one`
  (today's path) and mark the slot `sp_only`; wave steps call `model.prefill_traced_wave(call)` (all members at
  `start == 0` within a captured length, tracing on) or `model.prefill_wave(call)` and clear `sp_only`; outputs are
  assembled per row of the call. `warmup_model_prefill` also warms the wave path when present.

Model API the device side has to provide (`tt/mesh_model.py`, steps 6-7): `prefill_wave(call: WaveCall) ->
dict[row, (first_token, last_logits | None)]`, `prefill_traced_wave(call) -> the same`, `has_prefill_trace(T)`
(unchanged), class attribute `wave_per_row_users: bool`.

Verification: `PYTHONPATH=$WT:$W/ttnn:$W $W/python_env/bin/pytest
models/demos/glm53_flash/tests/test_generator_vllm_host.py --noconftest -p no:cacheprovider -q` and
`$W/python_env/bin/pre-commit run --files <the three files>` (see STATUS.md for the run record).

## 5. Open questions

1. Who owns the ttnn KDA op change (step 0) and whether the d_p team wants the SP-size override as a public
   `ttKDA` argument (it is the natural home: `sequence_parallel_size=1` on a multi-row mesh).
2. The F / p split of the 7 s chunk: one row's chunk time vs W (the measurement the PP study also asks for); it sets
   W per envelope cell and decides whether waves at W > 2 K are worth their activation memory.
3. Phase 2's device-indexed reads: the tensor-argument `ttnn.slice` on the paged cache's block dim and on the
   pooled-key cache (shape support for dim 0 of a 4-D TILE tensor), or a small paged-gather op; without it waves stay
   uniform-slot and ineffective at full occupancy.
4. The plugin's prefill batching: whether one `prefill_forward` call carries every pending chunk of the step (the
   wave needs them together) and whether `max_num_batched_tokens = rows x W` is honoured per step under the TT
   scheduler's chunked prefill.
5. Decode starvation during waves is the interleave / active-slot item (DESIGN.md section 4), unchanged by this
   design: a 7 s wave between decode steps is the same stall as today's 7 s chunk.
6. Section 2.5's owner-row attention variant for lone arrivals at full occupancy: worth it only if lone arrivals
   dominate the 1 M workload (at 32 x 1 M the system is prefill-bound by 5-15x and bursts are the norm).
