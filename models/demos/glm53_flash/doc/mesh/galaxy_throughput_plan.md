# GLM-5.3-Flash on the Blackhole Galaxy: throughput plan (2026-10-28)

The 8-GPU reference node (8 x RTX Pro 6000, TP8, NVFP4, vLLM) publishes 55 tok/s per user at batch 1, 410 aggregate
tok/s at batch 32 (1K context) and sub-2 s TTFT up to 65K tokens. The 8x4 Galaxy holds 1 TB of DRAM, runs 32 x 1 M
contexts and already beats that node on aggregate decode (520 tok/s at 32 short users, 1287 at 256 x 128K), but is
3x behind per user at low batch, 15-17x behind on prefill, and the served engine admits 4 distinct 1 M requests. This
plan scopes the work that turns the memory into users and throughput, where each piece sits in the existing stack,
what it depends on, and how it is measured. Numbers are the README rows (one run per row, today's defaults) and the
2026-10-28 served smokes (`doc/mesh/README.md`).

## 0. Targets

| point | 8x RTX Pro 6000 | Galaxy today | target | workstream |
|---|---|---|---|---|
| decode, 1 user, tok/s per user | 55 | 17.5 (57 ms step) | 30 (35 ms step) | B |
| decode, 32 users (4 per row) | 410 agg, 12.8 per user, 1K ctx | 16.3 per user, 520 agg (short); 14.4 per user at 1 M | 25 per user, 800 agg | B |
| decode, 256 users x 128K | not published | 5.0 per user, 1287 agg | 8 per user, 2000 agg | B |
| TTFT 1K, batch 1 | 40 ms | 118 ms (128 tokens, traced) | < 100 ms | C |
| TTFT 32K, batch 1 | 0.86 s | ~15 s (two 16K eager chunks) | 4 s | C, A |
| TTFT 64K, batch 1 | 1.9 s | 28-34 s | 10 s | C, A |
| TTFT 1 M, batch 1 | n/a | 485 s | 150 s | C, A |
| served distinct requests at 1 M | n/a | 4 per engine (rows replicate) | 32 | A |
| served aggregate prefill | ~35K tok/s | 2.2K tok/s (one prompt at a time) | 8 prompts in flight, 2.6-7K tok/s | A |
| served decode step vs harness | the same engine | 73 vs 61 ms | within 5 ms | D |

Release targets stay measured post-optimization points (one e2e run per row reporting TTFT and decode tok/s).

## 1. The stack today, and where each workstream lands

| layer | component | files (models/demos/glm53_flash) | what it does today |
|---|---|---|---|
| mesh orchestration | `GLMMesh`, `GLM53MeshModel` | `tt/mesh.py`, `tt/mesh_model.py` | TP4 over the columns x 8 rows; decode = 8 independent row batches (4 users per row); prefill = ONE prompt chunk split into 8 sequence-parallel slabs (`unit = 32 * rows`), results gathered over the rows so every row holds the prompt |
| decoder | `FunctionalDecoder`, `DecoderLayer` | `tt/functional_decoder.py`, `tt/decoder_layer.py` | per-slot chunk cursors `next_start[user]`, KDA carries `[B, H/cols, K, V]` per chip, the per-user state writes |
| attention | KDA (34 layers), DSA (11 layers) | `tt/kda.py` (d_p's ttKDA, `sp_axis=0`), `tt/dsa.py`, `tt/indexer.py`, `tt/trace_prefill.py` | KDA scan sequence-parallel over the rows; DSA dense rows (absolute position < 2048) over the exact prefix, sparse rows via the pooled index; prefix read bucketed on the sparse path only |
| experts | routed MoE | `tt/moe.py`, `custom_ops/expert_stream`, `custom_ops/router_topk`, regions lists | EP over the 32 chips, per-token dispatch / combine (row-agnostic: decode already mixes users per row) |
| residual | mHC | `tt/mhc.py`, `custom_ops/mhc_pre_block` | fused pre-block + Sinkhorn on decode; the composite wrap on prefill rows |
| traces | prefill / decode | `tt/trace_prefill.py`, `GLM53MeshModel.capture*` | decode trace + prefill traces at {256, 512, 1024, 2048}; the eager chunked path above 2048; the program cache sealed after the captures |
| serving adapter | `GLM53FlashForCausalLM`, `GLM53FlashForConditionalGeneration` | `tt/generator_vllm.py`, `tt/vllm_config.py`, `tt/vllm_spec.py`, `tt/multimodal.py`, `tt/mm_inputs.py`, `tt/kda_snapshot.py` | slot table, prefill planner, warm-up ladder, APC (chunk-aligned hits), spec contract (P/V/C/D/D0), image + video; the wave planner exists, the device side (`prefill_wave`) does not |
| server | vLLM 0.30 + plugin main, TIS catalog | `~/ashai/glm53/serve_env`, `~/ashai/tt-inference-server-glm` | the 1m (4 users) and 128k (28 users) profiles; smoke stages a-e |

Workstream A changes the orchestration and attention layers (prefill only). B changes the decode custom ops and
the expert / residual glue. C changes the prefill chunk's kernels and launch pattern. D changes the adapter and
the catalog. Nothing changes outside `models/demos/glm53_flash` except step A0 (one ttnn attribute), flagged below.

## 2. Workstream A: per-row prefill ownership (waves) = shape-static prefill

Design: `_agents/vllm_glm53/ROW_OWNERSHIP.md` (2026-10-22) and `doc/mesh/row_ownership.md`. Mechanism (a): a wave
= one chunk per row, each row's chunk from a different prompt; attention and the KDA scan row-local (as in decode);
the per-token expert collectives unchanged; policy (c): the sequence-parallel path only for a lone prompt whose slot
is free on every other row, waves otherwise. The adapter half is delivered (`WavePlanner`, wave-affine `SlotTable`,
the `prefill_forward` wave path behind `GLM53_VLLM_ROWS_INDEPENDENT=1`, 22 host tests).

The unifying rule, learned twice this week (the sealed program cache, the APC hit alignment): every start-dependent
host integer of the prefill becomes a per-row device tensor, so one SPMD op sequence serves eight different
`(user, start, length)` tuples. That is the same change the prefix cache needs to restore at any 32-aligned snapshot
(the 21:40 experiment: `_prefix_rows` -> `user_rows(user, start)` is an exact-`start` slice). A and the APC's
"option 2" are one piece of work.

| step | file | change | status |
|---|---|---|---|
| A0 | ttnn `operations/experimental/kda/factory/chronology_binding.hpp` (+ the two program factories and their python bindings) | `sp_size` comes from `device.shape()[sequence_parallel_axis]`; add an optional `sequence_parallel_size` override (1 = every coordinate is rank 0 of a one-row scan). Backwards compatible | OUTSIDE the model dir: needs the go (and the d_p team as reviewers); prepared as a separate branch |
| A1 | `tt/kda.py` | `prefill_layer(active_seq_len, local=True)`: ttKDA with SP size 1, `summary_group_chunks` over `W / 32`; `actual_start` / `actual_end` as row-sharded scalars | not started |
| A2 | `tt/functional_decoder.py` | `next_start[row][user]`; a dummy KDA slot for idle rows; `prefill_forward(..., wave=WaveCall)` with per-row `(start, end)` tensors | not started |
| A3 | `tt/dsa.py`, `tt/indexer.py` | the dense prefix read over the whole dense range [0, 2048) with a host-built mask over [start, 2048); a fixed dense query-row count per slab (superset rows, `_select_rows`); row-sharded page-table / pool views for `fill` at per-row starts (`_prefill_views_for(user, b0_per_row)`); the index maintenance's `start`-derived ints as tensors | not started (option 2 of the APC design is this step) |
| A4 | `tt/decoder_layer.py`, `tt/mesh_model.py` | `prefill_wave(call)`: ids `[rows, T_pad]` row-sharded, per-row scalars through every layer, the head's last-token pick per row; `prefill_traced_wave` at 256..2048 | not started |
| A5 | `tt/memory_plan.py` | `reserve_bytes(..., wave_rows)`: W = 2048 rows per row is memory-neutral; the planner refuses larger W at a full cell | not started |
| A6 | `tt/generator_vllm.py`, `tt/vllm_config.py` | phase 2: per-row slot index (`wave_affinity="row"`) so waves fill at full occupancy | after A3 (the per-chip views) |
| A7 | TIS catalog | 1 M x `max_num_seqs` 32 and 128K x 256 with `GLM53_VLLM_ROWS_INDEPENDENT=1` | after the 8x4 validation |

Tests, in order (the design's section 3): host (`test_generator_vllm_host.py`, `test_memory_plan.py`); the 8x4 with
4 layers (`tests/test_mesh_wave_prefill.py`, new): the same prompt via SP and via a wave on each row (first token
equal, the owner slot's state within bf16 ulp), two prompts in one wave bit-identical to each alone, idle-row mask,
per-row starts (row 0 at chunk 0, row 1 at 48K), the traced wave; the 8x4 at 45 layers: the 128-token first token vs
HF per row, 64K needles with 8 distinct needles per wave, 32 distinct prompts at 1 M, then the served engine at
`--max-num-seqs 32` with the README benchmark points. The decisive measurement before raising W above 2048: one
row's chunk time vs W (the F / p split of the 7 s chunk), on the 8x4 with seven idle rows padded.

Expected: served distinct requests 4 -> 32 at 1 M (the metal row already runs 32 x 1 M), eight prompts prefilling
concurrently (aggregate 2.2K -> 2.6-7K tok/s depending on p(W)), APC hits at any 32-aligned snapshot. Risks: the SP
and wave numerics differ at the last ulp (rows already differ); a lone arrival at full occupancy is 3-8x slower in a
wave than under SP (policy (c) keeps SP for it); idle rows cost wasted expert work proportional to the idle share.

## 3. Workstream B: the decode step to the roofline floor

`tt/roofline.py` (`tests/test_roofline_host.py`) models the step from measured per-op rows; floor 28.3 ms at 4
users per row (the fused pre-block default), 51.7 ms at 32 per row. Measured: 61.5 ms (32 users), 57 ms (1 user),
117 ms (256 users). The ranked actions (`ranked_actions`, 2026-10-28):

| id | saving at 4 / 32 per row (ms) | change | where | status |
|---|---|---|---|---|
| E0 | 13.9 / - | the expert_stream op as the decode path (341 vs 683 us at 28 active) | `tt/moe.py`, `custom_ops/expert_stream` | default on the regions-sourced streaming decode; the roofline's "measured" column predates it: refresh the table from a layer-4 tracy of today's step first |
| D13 | 5.5 / 5.5 | column-shard the residual streams (distributed-norm style): elementwise work / 4 | `tt/mhc.py`, `tt/decoder_layer.py` | design: the two collectives it adds per wrap must not outweigh the /4 (STATUS.md D13) |
| H2 | 4.7 / - | T < 32: compact expert output `[rows x T, H]` + all_reduce_rows instead of the padded reduce-scatter | `tt/moe.py` | not started |
| E1 | 4.2 / 17.6 | expert_stream fixed cost (launch x2, x copy, routing scans, DST fill): fold the selector matmul + x copy into the op; at 32 per row the stream dispatch form (weights read once) | `custom_ops/expert_stream` | not started |
| H1 | 2.9 / - | T < 32: one all_gather_async on the compact row-major `[T, H]` | `tt/moe.py` | not started |
| C12 | 2.9 / 7.5 | bfp4 gate/up experts (-1/3 of the expert bytes) | `tt/moe.py` dtypes | NOT planned unless it retains accuracy and PCC against the bf16 reference (user rule 2026-10-28): the gate is PCC vs bf16 on the 2K prose prompts + the release evals equal to bfp8's, not a token check |
| B5 | 1.6 / 2.1 | one `dsa_decode_state` program (29 -> 1 op) | `custom_ops/dsa_decode_state` | knob exists (`GLM53_DSA_DECODE_STATE=1`), accuracy gate open |
| B8 | 1.5 / 14.0 | dense regime: attend the paged cache directly; pool-major cache pages | `tt/dsa.py`, `custom_ops/sparse_bound` | not started; the big item at 256 users |
| P1, K3, S1, K2 | ~3 / ~3 | generic-op access pattern, KDA fixed cost, matmul epilogues | `custom_ops/kda_decode`, `custom_ops/expert_stream` | not started |

Order: refresh the measured column (one tracy of the 4-user and the 32-per-row steps), then E1 + H2 + H1 (the
expert glue: ~12 ms at 4 per row), D13, B8 (for the 256-user profile); C12 only if its PCC against bf16 holds. Every change lands
behind a knob, with the chip-0 / block A/B and the 2K prose logits gate (PCC >= 0.98 / top-1 >= 85 %), then the
default flips and the README row is re-measured from one run. Expected: 61.5 -> ~36 ms at 32 users (27 tok/s per
user, 870 agg), 117 -> ~75 ms at 256 users (13 per user, 3400 agg).

## 4. Workstream C: the prefill chunk

A 16K chunk takes 7.1-7.4 s on the sequence-parallel path (2.2K tok/s per prompt, 2023-2530 tok/s on the regions
expert path), 58-94 s at 1 M positions on the 2026-10-21 run (the growing term is the whole-prefix copy and the
multi-row scorer, `doc/mesh/prefill_1m_cost.md`). The per-op profile (`doc/mesh/prefill_profile.md`) puts the chunk at
~0.3 % of the compute roofline: per-token work in narrow kernels (each row's slab is 2K rows), not the prefix. The
eager launch floor is visible on short prompts: 128 tokens take 858 ms eager vs 119 ms traced.

| item | change | where | expected |
|---|---|---|---|
| C1 | waves (A): each row runs a 16K chunk instead of a 2K slab: 8x the rows per op at the same launch cost | A | the F / p measurement decides; 2.6-7K tok/s aggregate |
| C2 | the regions expert path as the prefill default (2530 vs 1957-2330 tok/s) | `tt/moe.py` (`GLM53_PREFILL_EXPERTS`) | +10-25 % per chunk |
| C3 | fused prefill glue: the per-layer host round trips of the eager chunk (the dispatch guard's readback is already off; profile one chunk with cProfile for the rest, e.g. the mHC wrap's per-wrap ops) | `tt/mhc.py` (the prefill wrap), `tt/decoder_layer.py` | the 858 -> 119 ms short-prompt ratio says the fixed cost per chunk is ~0.7 s per 45 layers |
| C4 | traced chunks above 2048 (the trace_prefill design's phase B: shape-static sparse chunks at bucketed starts) | `tt/trace_prefill.py` | removes the launch floor from every long chunk; depends on A3 |
| C5 | the prefill expert kernels (gate|up / down at 38-54 % of DRAM rate on 2K-row slabs, `doc/mesh/expert_kernel_sweep.md`) | `custom_ops/expert_stream` prefill form | the largest per-token term once the slab is 16K rows |

Expected: 32K TTFT 15 -> 4 s, 64K 30 -> 10 s, 1 M 485 -> ~150 s, with the per-prompt numbers measured from one
run each (`tests/test_mesh_long_context.py`, the README rows).

## 5. Workstream D: the served path

| item | change | where | status |
|---|---|---|---|
| D1 | the step gap (73 served vs 61 harness ms): device-side token feedback + position increment inside the traced step, ids-only read, then `supports_async_decode` | `tt/generator_vllm.py`, `tt/mesh_model.py`, `tt/packed_inputs.py` | packed inputs landed; the rest not started |
| D2 | APC at any 32-aligned snapshot | A3 | after A3 the default `GLM53_VLLM_APC_HIT_ALIGN` drops from the chunk to 32 |
| D3 | APC with the plugin: `mm_identifiers` (images in the content chain) and `confirms_prefix_cache_hits` (the scheduler trims a hit to the adapter's resume position) | vllm-tt-plugin (two generic PRs) | written, pending the go |
| D4 | spec v2: partitioned-cache verify (spec at 1 M), sparse-range verify (lift the 2050 bound), `logits` accept mode, async decode | `tt/vllm_spec.py`, `tt/spec_decode.py` | branch `arg/glm53_served_spec_long` holds the sparse-range verify draft |
| D5 | catalog: 1 M x 32, 128K x 256, the multimodal class, the smoke's stages a-e as the release pre-check | TIS `arg/glm53_flash_galaxy` | after A7 |
| D6 | release flow: GPQA, Terminal-Bench, LiveCodeBench, MMMU-Pro, benchmarks vs the measured targets | TIS workflows | not started; needs D5 |

## 6. Sequencing and the device plan

The device is the scarce resource: the queue runner owns it, one stage at a time, every stage from a frozen snapshot.

1. Now: the 45-layer served smokes of the merged tree (text a-e, spec at 4 per row, APC with repeated 64K needles)
   finish the integration branch's validation. Device-free in parallel: A1, A2, A4 and the A3 host side (the masks,
   the row-sharded views), the A0 patch on its own branch, the B roofline refresh harness.
2. A0 decision; then the 4-layer wave tests on the 8x4, the F / p measurement, the 45-layer wave e2e rows, the served
   engine at 32 x 1 M (A7 + D5).
3. B in knob-gated A/B stages between the A stages (each a 2-10 minute block or chip-0 stage), the README rows
   re-measured from one run after each default flip.
4. C2-C5 and D1 behind A; D3 / D4 when approved; D6 last.

Validation rules (unchanged): one run per README row with TTFT and decode tok/s; the 2K prose logits gate before any
default flip; every served check at 256K and 128K ISL first; the program-cache seal on in every served run; no
runs outside the queue; nothing pushed without the go.

## 7. Decisions needed

| decision | why | default if unanswered |
|---|---|---|
| A0: the ttnn KDA `sequence_parallel_size` override (outside the model dir) | without it there is no row-local KDA scan on a multi-row mesh, so no waves and no A3 benefit for the KDA layers | the patch is prepared on its own branch and not applied; A3 proceeds for the DSA / page-table side (it also serves the APC) |
| D3: the two plugin PRs | APC with images and exact hit accounting | APC serves text hits with the plugin's own block hashes; images miss |
| C12: bfp4 gate/up experts | 1/3 of the expert bytes on every decode step | stays bfp8; bfp4 is taken only if accuracy and PCC against bf16 are retained (user rule), never for the speed alone |
