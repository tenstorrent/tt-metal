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
| A0 | ttnn `operations/experimental/kda/factory/chronology_binding.hpp` (+ the two program factories and their python bindings) | `sp_size` comes from `device.shape()[sequence_parallel_axis]`; add an optional `sequence_parallel_size` override (1 = every coordinate is rank 0 of a one-row scan). Backwards compatible | validated 2026-10-29 01:00 on the 8x4 (row-local KDA vs HF per row, SP vs local PCC 1.0); ttnn branch `arg/kda_sequence_parallel_size` (PR pending the push go), base port `_base` for this box |
| A1 | `tt/kda.py` | `prefill_layer(active_seq_len, local=True)`: ttKDA with SP size 1, `summary_group_chunks` over `W / 32`; `actual_start` / `actual_end` as row-sharded scalars | validated with A0 (`test_mesh_local_kda`, 2026-10-29) |
| A2 | `tt/functional_decoder.py` | `next_start[row][user]`; a dummy KDA slot for idle rows; `prefill_forward(..., wave=WaveCall)` with per-row `(start, end)` tensors | done at the block level 2026-10-29 02:20 (wt_mesh 033017bba8b): per-row cursors, the eager wave forward with the on-device carry select; validated with layers 0-2 and 0-3 (heads2users, 4 users/row) |
| A3 | `tt/dsa.py`, `tt/indexer.py` | the dense prefix read over the whole dense range [0, 2048) with a host-built mask over [start, 2048); a fixed dense query-row count per slab (superset rows, `_select_rows`); row-sharded page-table / pool views for `fill` at per-row starts (`_prefill_views_for(user, b0_per_row)`); the index maintenance's `start`-derived ints as tensors | dense half validated (chip 0, static vs dynamic PCC 1.0); the wave form (per-row views, per-row index maintenance, static attention with per-row starts) VALIDATED at the layer level on the (2,4) block 2026-10-29 01:55: wave vs slab bit-exact in the dense regime (both row orders) and in the sparse regime with per-row starts (446/452 continued rows bit-exact, the rest 1 ulp); the index half for 32-aligned (non-128) starts not started |
| A4 | `tt/decoder_layer.py`, `tt/mesh_model.py` | `prefill_wave(call)`: ids `[rows, T_pad]` row-sharded, per-row scalars through every layer, the head's last-token pick per row; `prefill_traced_wave` at 256..2048 | VALIDATED on the 8x4 at 45 layers 2026-10-29 02:49: the dump's first token = HF (102647) through SP, a wave on all 8 rows, row 0 alone, and the TRACED wave (bit-identical to the eager wave; capture 6.4 s at 256 rows per row); block tests eager + traced on layers 0-3; the adapter's per-row trace semantic (has_wave_trace / traced_length) still to wire; the throughput point: a wave of 8 x 2048 = 7.1 s vs one 16 K SP chunk 12.2 s cold (the MoE's gathered tokens dominate at 2 K per row; the launch-bound gain is for short prompts: warm numbers next) |
| A5 | `tt/memory_plan.py` | `reserve_bytes(..., wave_rows)`: W = 2048 rows per row is memory-neutral; the planner refuses larger W at a full cell | done 2026-10-29 02:03 (wt_mesh 39cbffce958): `MemoryPlan.wave_rows` (the chunk working set at rows x W gathered tokens: neutral at W = 2048 on the 8x4), `max_wave_rows` = 2304 at 1 M x 4 per row, 4736 at 128 K x 28 (host pins) |
| A6 | `tt/generator_vllm.py`, `tt/vllm_config.py` | phase 2: per-row slot index (`wave_affinity="row"`) so waves fill at full occupancy | after A3 (the per-chip views); the served adapter under per-row ownership PASSES on the block (served_waves_block_v6, override runtime, 2026-10-29 05:06: traced wave of two prompts + joint decode, the wave beyond the width + its eager continuation, vLLM's chunked member); phase 2 (per-row slot index) still open; the 45-layer served run with 32 distinct requests (served_waves_8x4) running |
| A7 | TIS catalog | 1 M x `max_num_seqs` 32 and 128K x 224 with `GLM53_VLLM_ROWS_INDEPENDENT=1` | 1 M DONE 2026-10-29 13:48 (tt-inference-server arg/glm53_flash_galaxy 36205629c, local): 32 requests, rows independent, prefix cache off (its host-store snapshots cost 725-1194 ms per step at 32 slots); served as catalogued: a/b PASS, bench 128x32 TTFT 5.2 s / 11.0 tok/s/u / 244 aggregate (adapter step 61.2 ms = 16.3 / 522), 64K x 4 TTFT 88 s; the 128 K profile at 224 (7 slots per chip) OOMs at boot under per-row ownership (14:15: 162 MiB per bank after the prefix walk); 192 (6 per chip) DONE 14:58: 4 x 64 K needles 4/4 (TTFT 49-113 s), bench 128x192 TTFT 27 s / 6.3 tok/s/u / 522 aggregate (adapter step 116 ms = 8.6 / 1651 over 192 live), 0 cache misses, boot 29 min -> catalogued at 192 |
| A8 | `tt/vllm_config.py`, `tt/generator_vllm.py` | SP-first placement for eager prompts (`SlotTable.allocate(prefer_sp)`: a slot index live on no row): a lone 16 K SP chunk 7.7 s vs a two-member wave step 56-64 s (2026-10-29, 16 K needles), so up to `users_per_row` concurrent long prompts stay sequence-parallel; short prompts keep the wave-affine order | DONE: served 2026-10-29 10:51 (served_8x4_needle64k_spfirst_r2): 4 x 65514 needles 4/4, 0 waves, 16 SP chunks at 7.4-7.6 s, TTFT 60/60/119/119 s (mean 90 s vs r5's 321 s in waves) |
| A9 | `tt/vllm_config.py` + plugin `arg/prefill_align_wait` | wave member starts on the 128 grid (`WAVE_CHUNK_UNIT`; the planner refuses an off-grid start): r3's engine death = a 96-token chunk after a 128-token prefix-cache hit ending on the 224 KV block (the plugin's running-request fallback under alignment 256); the plugin fix makes the request wait for a step that reaches the grid | model side DONE (804c8e80da7); plugin commit a94ec5b local, 13 host tests, NOT pushed (needs the go; generic) |
| A11 | `tt/generator_vllm.py` wave ladder | a wave with a member continued at 256 beside one at 2048 misses the SEALED program cache in the scorer crop (block case (f), 2026-10-29 12:03: SliceDeviceOperation, topk_bounds limit 1792): a served engine death for >= 5 concurrent long prompts; the ladder must cover it | DIAGNOSED 13:15 (block_v14): the lone member after a 256-token first chunk gives op_start 256 and a crop width (1280 pools at T=2048) between the bucket-multiple starts' widths, reachable only when the prefix bucket (block: 256 pools) is smaller than the superset slack R*T/P (served buckets >= 4096 pools = the slack at W=2048: safe today, re-opened by a wider wave); FIX d70cc3d41c9: the ladder enumerates every (sub-block rows, width) a wave can reach (`TtDSAIndexer.scored_width`) and runs one lone row-0 wave per width the starts missed; VALIDATED 13:49 (block_v16: cases (a)-(d), (f), (h) PASS; the 1 M profile's boot needed 1 such wave) |
| A12 | `tt/generator_vllm.py` decode contract, `tt/mesh_model.py` traced step | served parity: at 32 users the device step is 48 ms (harness) but the adapter's host-wall step is 61 ms (the packed token / position upload, the per-row ids readback and the sync every step: the version-1 reload contract, `supports_async_decode` False) and the vLLM bench's inter-token 91 ms (the engine loop with no-async-scheduling, decode held while a burst's later arrivals prefill). Resident decode = the plugin's `plan_decode_reload` contract: the model feeds its own device-sampled ids and advances positions on device, the host reloads only on transitions (new rows, layout / sampling changes, verify -> plain), which also unlocks vLLM's async scheduling | step 1 DONE 2026-10-29 19:45 (`tt/resident_decode.py`, `GLM53_RESIDENT_DECODE=1`; served_8x4_resident_r1 on the 1 M profile: ids identical to the reload path, bench 128x32 14.13 tok/s/u / 288 aggregate vs 11.0 / 244, TTFT unchanged); next: the knob on the 1 M catalog profile, the 128 K profile smoke under the knob, step 3 async scheduling (the readback made non-blocking), step 4 the interleave (design in resident_decode.md) |
| A13 | `tt/generator_vllm.py` prefill_forward, `tt/vllm_config.py` WavePlanner | multimodal under per-row ownership: the wave path has no vision scatter, so an image / video request is `sp_required` (SP-first placement on a slot index live on no row; the planner refuses only when the index is shared, naming the cause) and its SP step runs the multimodal eager prefill (the rows-independent branch ran the text prefill and answered without the image: r2, TTFT 2 s) | DONE 2026-10-29 19:16 (1cb68898184 + fd833862c8d; served_8x4_mm_32_r3 stage e PASS on the 1 M profile: image TTFT 5.4 s, video 5.7 s, both answers correct) |
| A10 | `tt/functional_decoder.py` | the rows-2/3 NaN under per-row ownership: KDA carry blends x 0.0 keep NaN (an idle row's non-finite scan poisons a slot for good); `_finite` = where(isfinite) on every blend operand, served-waves case (h) as the regression | VALIDATED 2026-10-29 12:45: the 45-layer replays r6-r8 have no NaN (rows 2-3 first-token pcc 0.982 / 0.978 like rows 0-1), the KDA carries drift uniformly over the rows (0.963-0.973), the lone-wave control (g2) drifts the same (0.984) -> the residual 0.98 is the wave scan's own drift; long profile criterion pcc >= 0.97 + SP top-1 in the wave's top-3 (73798ae1b13); block (a)-(d) pass; SERVED CLOSURE 2026-10-29 13:10: r5's stage c with waves on the fix tree = 4/4 needles (rows 2-3 included), 64 waves, 0 cache misses; the regression case (h) PASSES on the block (13:49) |
| S1 | `tt/vllm_spec.py`, `tt/spec_decode.py`, `tt/generator_vllm.py` | K by the LIVE batch (user directive 2026-10-29 03:38): a `k_by_rows` table measured on the harness (users per row x K, the B (K + 1) <= 32 tile per row), the step's K chosen from the live users per row (<= the plugin's num_speculative_tokens = the largest K of the table), V/C/D graphs captured per table entry (the live users compacted into the entry's B_sub slots through device-side slot indices, as the wave slot does), plain decode above the cliff | step 1 DONE 2026-10-29 07:30 (92a7375c797): `vllm_spec.k_by_rows` = the sweep's table under the catalog's ceiling (3 up to 8 users per row, 1 at 16; the acceptance plateaus at ~2.5 drafts whatever K: 4/row K=3 30.1 tok/s/u, 1/row K=7 25.5, K=15 16.9); the remaining points refine it; steps 2-4 (per-step K from the live users, per-entry graphs + compaction, the partition-aware verify) open |
| S2 | `tt/moe.py` decode_glue_modes, `tt/vllm_spec.py` | the full-tile speculative verify (B (K + 1) = 32) broke losslessness on mesh row 5: the 32-row shape took the padded gather + reduce-scatter glue (per-row drift 0.3-0.75 logits at 45 layers) while the compact gather + all-reduce is row-identical; the compact path covers T = 32 by default, the policy cap is the full tile | DONE 2026-10-29 17:50 (3f595f1ae47): lossless + row-identical through the compact path (spec_fulltile_glue_B8_K3); the table keeps 8 -> 1 (compact at 32 rows: 69 ms per iteration vs 50); 16 -> K=1 being measured |
| V1 | `tt/ttkda/` | the ttKDA prefill driver vendored from `deepseek_v3_d_p/tt/kda` (@ f8fabeb9b9c; the ttnn ops stay shared), tuned here at the user's direction (2026-10-29): grouping geometry, L1 budgets, the chronology contract | DONE 2026-10-29: tweak 1 = the chunk terms' L1 budget 256 KiB (the 1024-1088-local-row L1 clash); the contract that ttKDA's SP placement is keyed on the absolute start is met by chunk-relative bounds in `decoder_layer` (the served round-1 artifacts' root cause) |

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

## 2b. Server completeness (user definition 2026-10-29 20:02; the order of work: completeness, then parity, then metal)

| item | status 2026-10-29 20:25 | gate |
|---|---|---|
| multimodal (image + video) on the 1 M profile | DONE 19:16 (served_8x4_mm_32_r3: both answers correct, TTFT 5.4 / 5.7 s) | stage e of every later smoke |
| 1 M context through the API | DONE 2026-10-30 01:34 (needle1m_r6: the needle 104 K tokens before the question answered exactly, TTFT 495 s; r4/r5: 56-66 ms per token at 1 M; the depth-0.56 loop = the model's ~460 K retrieval limit) | README served 1 M row booked |
| APC on (both switches) at 32 slots | DONE 2026-10-30 02:01 (async_r4: every stage PASS with async + APC host store 2 entries + resident + confirmer: 4/4 hits, repeats identical, bench 14.64 tok/s/u, repeated image served); catalog flipped | the served smoke a-e is the regression gate | the multi-turn hit served from the snapshot, the bench within noise of APC off, stage e twice (the image hit) |
| 32 users x 1 M | QUEUED 2026-10-30 02:08: the capstone waiter restarts the runner with BUDGET=25200 once the 4 async A/B arms and the step-profile arm are done, then queues `served_8x4_needle1m_32_r1` (32 distinct 1 040 000-token prompts, depth 0.9, 256-token answers, SMOKE_CLIENT_TIMEOUT_S=20000); ~4.3 h to the last first token at ~2.2 K tok/s aggregate prefill | 32 needles found under APC + resident; TTFT per arrival and the decode at 32 x ~1 M booked |
| decode steps between a prompt's chunks (the interleave) | model side DONE 2026-10-30 04:20 (`GLM53_DECODE_SLOT_MASK=1`: the KDA gated twin leaves parked slots alone; `slot_mask_block_v2` pass F: carry byte-identical, chain == uninterleaved); the served step cost (`ab_slotmask_32_r2`) and the interleave arm (`ab_interleave_32`, after the capstone) decide the catalog flip (`decode_interleave_enabled: true`, one decode step per prefill step): the served cost is nil (`ab_slotmask_32_r2` 59.4 ms, `ab_interleave_32` 59.7 ms = `nb_32`), the capstone runs with the interleave on and its PASS flips the catalog | a 1 M prompt's prefill no longer stalls the running decoders |

## 2c. A/B method for decode-path changes (user rule 2026-10-30 01:40)

Async scheduling, the resident chain, speculative decode and the decode glue are A/B'd on short-ISL bench points at
batch 32 (the 1 M profile) and batch 128 (the 128 K profile, 192 slots), never on the 1 M prompt (that one is a
completeness check: `needle1m_r6`). One served smoke per arm, bench stage only, minimal warm-ups (the prefix-start
walks, the multimodal warm-up and the wave ladder beyond the 256 trace are not needed for 128-token prompts):

    served_8x4_ab_<arm>|PROFILE=1m STAGES=d BENCH_POINTS=128:128:32 SMOKE_EXTRA_VLLM_ARGS=max-num-seqs=32;<async-scheduling=true | no-async-scheduling=true> SMOKE_EXTRA_ENV=GLM53_RESIDENT_DECODE=1;GLM53_VLLM_PREFIX_CACHING=1;GLM53_VLLM_APC_SNAPSHOTS=2;GLM53_VLLM_WARMUP_PREFIX_STARTS=0;GLM53_VLLM_WARMUP_MM=0;GLM53_VLLM_WAVE_TRACE_LENGTHS=256|.../test_serve_smoke_glm53.py
    served_8x4_ab_<arm>_128|PROFILE=128k STAGES=d BENCH_POINTS=128:128:128 SMOKE_EXTRA_VLLM_ARGS=max-num-seqs=192;<...> SMOKE_EXTRA_ENV=GLM53_RESIDENT_DECODE=1;GLM53_VLLM_WARMUP_PREFIX_STARTS=0;GLM53_VLLM_WARMUP_MM=0;GLM53_VLLM_WAVE_TRACE_LENGTHS=256|...

The model build (~8.5 min from the tensorbin cache) is the floor of every boot; the warm-ups above take ~10 min on
the full profile and ~3 min minimal (to be measured by the first pair). The verdict of an arm = the bench's tok/s per
user and TTFT next to the adapter's perf line (device step, chain count). Spec decode adds one long-ISL point at
batch >= 8 when the acceptance depends on the context, still never the 1 M point.

Arms queued 2026-10-30 (runner budget 5400 s each, ~14 min per bench arm): `ab_async_on_32` (ITL 68.2 ms, TTFT
5.57 s), `ab_async_off_32`, `ab_async_on_128`, `ab_async_off_128`, `profile_step_32` (`GLM53_VLLM_STEP_PROFILE=60:120`),
`ab_async_nb_32` / `ab_async_nb_128` (`GLM53_DECODE_REPLAY_BLOCKING=0`: the replay no longer blocks the engine
thread, doc/mesh/resident_decode.md), `nb_gate_32` (the a-e gate under the knob); the capstone follows.

Results: batch 32 async ON 68.2 / 69.1 ms ITL (mean / median), TTFT 5.57 s; async OFF 76.3 / 73.7 ms, TTFT 6.11 s
(async +10 %; the remaining gap to the 48 ms device step is the blocking replay). `slot_mask_block_v1` (pass F of
`tests/test_decode_packed_inputs_mesh.py`: `GLM53_DECODE_SLOT_MASK=1`, the interleave's model side) runs on the
rows-4-5 block after the gate arm.
Batch 128 (the 128 K profile, 192 slots) async ON: ITL 129.1 / 127.5 ms (P99 174), TTFT 18.65 s. The burst TTFT is
uniform in both: one engine step prefills every arrival, and a traced wave replays the smallest captured per-row
bucket (256 rows x 8 rows = 2048 padded tokens at the model's ~2 K tok/s prefill rate = ~1 s per wave, whether it
carries 1 or 8 prompts); `ab_wave128_32` adds the 128-row bucket. The prefill rate itself is workstream C.
`ab_async_nb_32` (the non-blocking replay): ITL 59.5 / 60.0 ms (P99 66.2), TTFT 5.54 s = -13 % vs the blocking
replay; the adapter's wall is 0.3 ms, so the remaining 11 ms over the device step is the engine's host path per step.
`ab_async_nb_128` (batch 128, 192 slots): ITL 108.4 / 109.8 ms (P99 115.9) vs 129.1 / 127.5 (P99 174) blocking:
-16 % mean, -34 % P99; TTFT 18.54 s. The non-blocking replay wins at both batches; the default flips once the a-e
gate `nb_gate_32` is green. The harness pair `e2e_8x4_B4_ctx1m` / `ctx256` then says how much of the remaining gap
is the decode step's own context-size cost (the 48 ms reference ran at `GLM53_MODEL_MAX_SEQ_LEN=256`).
Measured: the traced step is 48.36 ms at max_seq_len 256 and 54.92 ms at 1 M with the user-partitioned cache
(`e2e_8x4_B4_ctx256` / `ctx1m_r2`, same first token and chain PCC): +6.6 ms is the step's own context-size cost (a
metal item: bound the DSA decode's work by the live length), and the served 59.5 ms ITL is 4.6 ms (8 %) over the
true device step. Served decode parity at 32 users is reached up to the engine's host path; the slot mask costs
nothing (`ab_slotmask_32_r2`: 59.4 ms); the 128-row wave bucket would cut the burst TTFT 5.5 -> 3.8 s but its trace
is wrong at 45 layers (`wave128_gate` FAIL: a wrong first token for the dump's prompt), so the default stays 256,512
until the T=128 wave is fixed (the wave gate now captures 128 and 256).

The async / replay A/B at 128 x 128 tokens (one served run per cell; ITL mean / median / P99 in ms, TTFT):

| batch | async OFF, blocking replay | async ON, blocking replay | async ON, non-blocking replay |
|---|---|---|---|
| 32 (1 M profile, APC on) | 76.3 / 73.7 / - , 6.11 s | 68.2 / 69.1 / 75.7, 5.57 s | **59.5 / 60.0 / 66.2, 5.54 s** |
| 128 (128 K profile, 192 slots) | 129.6 / 127.7 / 146.5, 19.9 s | 129.1 / 127.5 / 174.4, 18.65 s | **108.4 / 109.8 / 115.9, 18.54 s** |

Async scheduling pays only once the model's submit returns before the device finishes (the non-blocking replay):
with the blocking replay it hides the read alone, nothing at batch 128.

## 3. Workstream B: the decode step to the roofline floor

`tt/roofline.py` (`tests/test_roofline_host.py`) models the step from measured per-op rows (refreshed 2026-10-29 from the per-layer decode profiles today_8x4_dsa3 / kda4: the stream expert op 264 us, the gather glue 175 + 67 us, the padded reduce-scatter 269 us, the mHC pre-block 82 us; model 64.9 ms vs the 61.5 ms e2e step); floor 29.0 ms at 4
users per row (the fused pre-block default), 51.7 ms at 32 per row. Measured: 61.5 ms (32 users), 57 ms (1 user),
117 ms (256 users). The ranked actions (`ranked_actions`, 2026-10-28):

| id | saving at 4 / 32 per row (ms) | change | where | status |
|---|---|---|---|---|
| E0 | 13.9 / - | the expert_stream op as the decode path (341 vs 683 us at 28 active) | `tt/moe.py`, `custom_ops/expert_stream` | default on the regions-sourced streaming decode; the roofline's "measured" column predates it: refresh the table from a layer-4 tracy of today's step first |
| D13 | 5.5 / 5.5 | column-shard the residual streams (distributed-norm style): elementwise work / 4 | `tt/mhc.py`, `tt/decoder_layer.py` | design: the two collectives it adds per wrap must not outweigh the /4 (STATUS.md D13) |
| H2 | 4.7 / - | T < 32: compact expert output `[rows x T, H]` + all_reduce_rows instead of the padded reduce-scatter | `tt/moe.py` | MEASURED 2026-10-29 with H1 (`GLM53_HYBRID_COMBINE=allreduce` 48.70 ms = 20.5 tok/s/u, 657 agg; `allreduce_ccl` 47.86 ms = 20.9 tok/s/u, 669 agg; first token = HF both): H1+H2 = -13.6 ms per step, the 20 tok/s/u target at 32 users; DEFAULT since 2026-10-29 05:05 with H1 (the same gate run); GLM53_HYBRID_COMBINE=rs keeps the padded reduce-scatter |
| E1 | 4.2 / 17.6 | expert_stream fixed cost (launch x2, x copy, routing scans, DST fill): fold the selector matmul + x copy into the op; at 32 per row the stream dispatch form (weights read once) | `custom_ops/expert_stream` | not started |
| H1 | 2.9 / - | T < 32: one all_gather_async on the compact row-major `[T, H]` | `tt/moe.py` | MEASURED 2026-10-29 (`GLM53_HYBRID_GATHER=compact`, e2e 45 layers x 32 users on the 8x4, f103): traced step 61.5 -> 55.80 ms = 17.9 tok/s per user, 573 agg; first token = HF; DEFAULT since 2026-10-29 05:05 (abac4d0f93b): the 45-layer accuracy gate logits_8x4_h1h2 = the base's numbers exactly (pcc 0.9952, top-1 93.0 %, top-5 95.5 %); e2e_8x4_B4_h1h2_default queued for the README row |
| C12 | 2.9 / 7.5 | bfp4 gate/up experts (-1/3 of the expert bytes) | `tt/moe.py` dtypes | NOT planned unless it retains accuracy and PCC against the bf16 reference (user rule 2026-10-28): the gate is PCC vs bf16 on the 2K prose prompts + the release evals equal to bfp8's, not a token check MEASURED 2026-10-29 03:18 (8x4, 45 layers, the S=128 logits gate): gate/up bfp4 = logits PCC 0.9838 (bar 0.98) but top-1 83.6 % < 85 % and top-5 88.1 % vs bfp8's 93.0 % / 95.5 % -> REJECTED (the perf point was not taken) |
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
| D3 | APC with images: `mm_identifiers` next to the placeholder fields (vLLM's content identifier of an item, present when the payload was stripped). The plugin already reports the hit itself as `start_pos`, so no confirmation hook | vllm-tt-plugin#261 (one generic PR) | open |
| D4 | spec v2: partitioned-cache verify (spec at 1 M), sparse-range verify (lift the 2050 bound), `logits` accept mode, async decode | `tt/vllm_spec.py`, `tt/spec_decode.py` | branch `arg/glm53_served_spec_long` holds the sparse-range verify draft |
| D5 | catalog: 1 M x 32, 128K x 256, the multimodal class, the smoke's stages a-e as the release pre-check | TIS `arg/glm53_flash_galaxy` | after A7 |
| D6 | release flow: GPQA, Terminal-Bench, LiveCodeBench, MMMU-Pro, benchmarks vs the measured targets | TIS workflows | not started; needs D5 |
| D7 | the served APC switch: vLLM's `enable-prefix-caching` must follow the adapter's `GLM53_VLLM_PREFIX_CACHING` (the catalog pins `no-enable-prefix-caching: true` from before APC existed and the plugin's capability gate only ever switches APC off), else vLLM never hashes blocks and no request arrives with `num_computed_tokens > 0` (the 2026-10-28 APC smoke: hit rate 0.0 %, round-2 TTFT = round-1) | `_logs/mesh/serve_smoke_glm53.sh` (flips the flag with the knob, 2026-10-29), TIS catalog | smoke fixed; `serve_smoke_apc_f102_r3` queued; the catalog sets both with the APC default (D5) UNBLOCKED 2026-10-29 04:22: the served round-1 artifacts were ttKDA's absolute-start placement (fixed, chunk-relative bounds); the 45-layer APC smoke passes a/b/c; the catalog flips next; DONE 2026-10-29 04:55: the catalog flipped on both Galaxy profiles (TIS arg/glm53_flash_galaxy 39d72e996) and the catalog-default smoke passes a/b/c (served_apc_catalog, wt_frozen120); push pending the go |

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
| D3: the `mm_identifiers` plugin PR (#261) | APC with images | APC serves text hits from the plugin's `start_pos`; prompts with images are prefilled from 0 |
| C12: bfp4 gate/up experts | 1/3 of the expert bytes on every decode step | REJECTED 2026-10-29: top-1 83.6 % (bar 85 %, bfp8 93.0 %) on the 45-layer 8x4 logits gate although the PCC bar passed (0.9838); stays bfp8 |
