# GLM-5.3-Flash functional decoder (stage 1, single Blackhole chip)

Design of record: `~/ashai/GLM_5.3_DESIGN.md` (Phase 1 = this stage). Code: `models/demos/glm53_flash/`.
Host: `sjc3-glx-a04u07`, one chip of the 32-chip Blackhole Galaxy (`TT_VISIBLE_DEVICES=0`, `MeshShape(1,1)`),
tt-metal main `8cfcaf754d3` + the commits on `arg/glm53_flash`.

## Contract

`tt/functional_decoder.py::FunctionalDecoder` is one decoder layer of any kind with device-owned state.

| Call | Input | Output | State effect |
|---|---|---|---|
| `from_state_dict(state_dict, hf_config=cfg_or_config_json, layer_idx, mesh_device, max_batch, max_seq_len)` | raw checkpoint tensors of `model.language_model.layers.{i}.` (prefix stripped, fp8 groups dequantized by `reference/checkpoint.load_layer_raw`) | decoder | allocates the KDA user slots / the DSA paged latent + pooled-key index caches |
| `prefill_forward(streams, user, seq_len, start=0)` | packed mHC streams `[1,1,T_pad,4*4096]` fp32 of one chunk (row i = position `start + i`), `T_pad % 32 == 0`, `start % 32 == 0`, rows past `seq_len` padding | `[1,1,T_pad,4*4096]` fp32 | KDA: `start == 0` resets slot `user`, else continues from its carries and writes them back; DSA: fills the user's cache pages `[start, start + T_pad)`, pooled keys, tail ring |
| `decode_forward(streams, positions)` | `[1,1,B,4*4096]` fp32 (B == max_batch user slots); `positions` = `DecodePositions` from `make_decode_positions(host_positions)` / `update_decode_positions` (persistent `[B]` int32: position, position // 4, position % 4, min(position, 2050) = the dense flash-MLA bound, + host values) | `[1,1,B,4*4096]` | KDA state updated in place; DSA latent appended and the pooled-key index maintained; dense flash-MLA for every slot (HF-exact through position 2050 = index_topk + index_kpool - 2), then the rows of the users whose host position is > 2050 are overwritten by the per-user indexer + `sparse_sdpa` result (eager, host chunk geometry) |
| `capture_decode_trace(streams, positions)` / `replay_decode(trace_id)` | persistent device inputs | `(trace_id, output tensor, warm-up output)` | the dense decode graph is trace-replayable (state and pooled-key index updated inside the captured graph, HF-exact through position 2050); refused for DSA decoders with max_seq_len > 2051 unless `allow_sparse_range` (the sparse rows are never captured) |

Residual convention: HF `[B, S, 4, 4096]` <-> packed `[1, 1, B*S, 16384]` (`host_streams_to_device` /
`device_streams_to_host`, test boundary only).

### Chunked prefill

A prompt can be prefilled in consecutive chunks: `prefill_forward(chunk, user=u, seq_len=n, start=s)` with
`s == 0` for the first chunk (new prompt: the KDA slot is reset / the DSA pages are refilled from 0) and every
later chunk starting exactly where the previous one ended (`start == next_start[u]`, asserted), on a 32-row
boundary (`start % 32 == 0`, so only the last chunk of a prompt may have a non-32-aligned `seq_len`) and ending
within `max_seq_len`. The chunk continues from the device-owned state and the result equals the single-shot
prefill of the whole prompt (same kernels, same row classes):

- KDA: the user's recurrent + conv carries are read out of the batched state (`[u]` slices; the batched tensors
  themselves at `max_batch == 1`), the chunk runs through `ttKDA.forward(actual_start=s, actual_end=s+n)` (both
  global positions, cached UINT32 scalars) and the new carries are written back (`_write_user_kda_state`). The conv
  history (3 rows) and the recurrent state come from the previous chunk, so the result is bit-identical to the
  single-shot run.
- DSA: the chunk's latent goes into the user's pages from block `s // 32` (`paged_fill_cache` with a device slice
  of the page table as the view); the pooled-key index is maintained from position `s` (`maintain_prefill(start=s)`
  writes the complete pools from `s // 4` through `write_pools(start_pool)`, which re-reads the earlier pools of a
  partially written 32-pool tile, and leaves the incomplete last pool's rows in the ring; a pool straddling a
  chunk boundary is completed from the ring rows plus the new rows, which the indexer-level test exercises at
  arbitrary offsets). Attention per chunk row: rows at absolute positions `< 2048` run the dense causal SDPA over
  the FULL prefix (K/V materialized from the cached latent rows `[0, s)` + the chunk's; the SDPA runs in windowed
  mode, `cu_window_seqlens = [0, s + T_d]`, `windowed_q_token_offset = s`, whose on-device causal mask at global
  positions is bit-identical to the plain causal kernel's rows); rows from 2048 on use
  `select(query_start = s + split)` + `sparse_sdpa` over `user_rows_rm(u, s + T_pad)` exactly as the single-shot
  sparse path (so only queries that see >= 2048 pooled tokens reach `sparse_sdpa`). The split row inside a chunk
  is `max(0, 2048 - s)`.
- No host tensor is created inside a chunk pass: the KDA scalars, the window bounds and the tail-slot ids are
  cached per value / geometry; the page-table views are device slices.
- `decode_forward` advances `next_start` (position + 1 per slot), so a prefill chunk may also follow decode
  steps when its start is 32-aligned.

## Modules

| File | Content | Reuses |
|---|---|---|
| `tt/config.py` | frozen config from `config.json` | |
| `tt/mhc.py` | mHC hyper-connection (fused Sinkhorn parametrization, fp32) | `deepseek_v3_d_p/tt/mhc/TtMHCWrap` |
| `tt/kda.py` | KDA: prefill via `ttKDA` (chunked kernels); decode = the 6-op fused block with `GLM53_KDA_FUSED=1` (qkv matmul, 2 gate matmuls, ONE `ttnn.transformer.kda_decode` program, tilize, o_proj; state + conv history in place) or the default ~75-op composite recurrence on `[B*H,128,128]` fp32 (see the KDA decode block section) | `deepseek_v3_d_p/tt/kda`, `ttnn/.../transformer/kda_decode` |
| `tt/dsa.py` | MLA (NoPE): dense causal SDPA for prefill rows [0, 2048), paged latent cache, absorbed `paged_flash_multi_latent_attention_decode` for every decode slot (decode = the 7-op fused block with `GLM53_DSA_FUSED=1`, see the DSA decode block section); prefill rows >= 2048 and decode users at position > 2050 via the indexer + `sparse_sdpa` | paged cache ops, flash MLA decode, sparse SDPA |
| `tt/indexer.py` | GLM k-pool indexer: pooled-key cache, `indexer_score_dsa(key_compression_ratio=4)`, `topk_large_indices` over 4x-expanded pool scores, tail ids | |
| `tt/moe.py` | router (exact fp32 composite top-8) + `sparse_matmul` experts per 32-token block + shared expert | gpt_oss sparse-matmul pattern |
| `tt/mlp.py` | clamped SwiGLU (dense layers, shared expert) | |
| `tt/decoder_layer.py` | mHC(attention) -> mHC(FFN) composition | |
| `tt/embedding.py` | `TtEmbedding`: `embed_tokens` gather (uint32 ids `[1,1,1,T]`, bf16 table widened to fp32) + `ttnn.repeat` to the 4 packed streams | `tt_mhc.mhc_expand` |
| `tt/model_head.py` | `TtModelHead`: unweighted stream mean -> final RMSNorm (fp32) -> LM head (bfp8, 5 vocab column chunks) -> `greedy` (multicore untilize + argmax); `last_token_logits` slices the stream row first | |
| `tt/model.py` | `GLM53Model`: embedding + all 45 layers + head on ONE chip by weight streaming (build -> run -> release per layer, persistent per-layer state, expert tensorbin cache); `prefill` / `decode` / greedy `generate` (`doc/full_model/README.md`) | `FunctionalDecoder.release_weights` / `attach_layer`, `GLMMesh.upload_cached` |
| `reference/` | checkpoint access + fp8 dequant, HF layer reference runner (dumps, `--generate N` greedy streaming), chain sensitivity probe, KDA reference equivalence check | transformers 5.20 `glm5_next` |

## Evidence (real weights, PCC vs HF)

| Test | What | Result |
|---|---|---|
| `test_kda.py` | layer 0 KDA, T=96 / 1000 + 3 decode | prefill 0.99994 / 0.99995, state 0.9999, decode 0.9999 |
| `test_moe.py` | layer 4 MoE, T=32 / 256 | router sets identical 32/32, 255/256; output 0.99993 / 0.99995 |
| `test_dsa.py` | layer 3 MLA, T=1, 2, 3, 31, 33, 96, 1000 + 3 decode | prefill 0.99989 (T=1) .. 0.99961 (T=1000), latent cache 0.99997, decode 0.9991-0.9999 |
| `test_decoder_layer.py` | full layers 0 / 3 / 4 vs HF dump, T=256 + 4 decode; T=200 non-aligned live HF | 0.99984 / 0.99992 / 0.99997 prefill, 0.9999 decode; 0.99980 / 0.99993 |
| `test_decode_trace.py` | traced decode replay vs the HF dump (layers 0/3/4); layer 3: the pooled-key index written inside the captured graph (65 complete pools + the 4 ring rows) vs a host recomputation over the 256 prefill + 4 replayed tokens; two users (100 + 37 tokens, slots) | replay PCC 0.9999 (same as the eager run of the same steps); index after the replays 0.999997 (worst pool row 0.99999), ring rows k 0.999994 / gate 0.999995; per-user 0.9998-0.99998 |
| `test_dsa_sparse.py` | layer 3 at 4097 tokens (repeated prompt): dense rows [0,2048) + sparse rows, decode 4097..4100 | prefill 0.99992 (rows<2048 0.99994, rows>=2048 0.9999), decode 0.9999 |
| `test_dsa_long.py` | random-token prompt: 4097 + 12 sparse decode steps with TT/HF selected-id sets (contiguous-prefix contract asserted) and the pooled-key cache vs a host recomputation; 2040-token dense prefill (slot 1, permuted page table) then decode 2040..2063 with the short user 0 checked every step; 16385-token prefill + 2 decode; prompts 2042/2043 (tails 2, 3) + 12 decode steps with the pool-cache check | prefill 0.99994 (rows<2048 0.99988, >=2048 0.99986), decode 0.9997+, 98-99 % ids common, pool cache 0.99999; transition >= 0.9996 both users; 16K 0.99983 (0.99988 / 0.99980); tails 2/3: prefill 0.99987 / 0.99990, decode >= 0.99978, pool cache 0.99999 |
| `test_batch8.py` | B = 8 and B = 32 user slots (prompt lengths 1..200 mixed, permuted page table), layers 3 and 4, 3 / 2 batched decode steps; every per-user sample held to 0.995, assertion deferred to the end (`_logs/fix3_batch8_B8.log`, `fix3_batch8_B32-3.log`, `fix3_batch8_B32-4.log`) | B=8: prefill >= 0.9968 (1-token user), decode worst per user 0.999377 (layer 3) / 0.999363 (layer 4). B=32 (flash-MLA decode on 120 cores = 3 cores/user): layer 4 PASSED, prefill worst 0.999937, decode worst per user 0.999698 / 0.999944 (steps 0 / 1); layer 3: all 64 decode samples > 0.995 (worst 0.997380 / 0.999489), but the case FAILS the per-user prefill gate on user 18 (S=5, offset 1620): 0.994797, one of its five rows at 0.98025 (see Known limits) |
| `test_embedding_head.py` | embedding of the 4097 R4097 prompt ids + 8 decode ids vs HF `embed_tokens` expanded; head on the layer-4 streams (stand-in for the layer-44 residual) 256 / 4097 rows and decode B=8 vs `lm_head(norm(mean))` in fp32 on the host; traced head+argmax replayed on new rows | embedding exact (max abs diff 0.0); head pcc 0.999966 / 0.999663 / 0.999980, top-1 agreement 0.973 / 0.978 / 8 of 8, top-5 overlap 0.977 / 0.983 / 0.975; replay bit-identical to eager, 2.36 ms per B=8 step; warm: embedding 1.8 ms (4128 rows), head 3.9 ms (256 rows) / 50.4 ms (4128 rows) |
| `test_determinism.py` | layers 0, 3: repeated prefill and traced-decode replay with `throw_exception_on_fallback`; sparse prefill + sparse decode repeated | bit-identical outputs |
| `test_chunked_prefill.py` | chunked prefill (`start=`): layers 0 / 3 on R4097 in chunks 1024x3+1025 (layer 3 also 1536+1024x2+513, a chunk straddling the 2048 split) vs the dump and vs the single-shot run, then 4 decode steps; layer 3 R2042 / R2040 in 512x3+506 / 504 chunks + 12 decode steps + pool cache vs host; both layers at T=2040 as 1024+1016 vs single-shot; indexer maintenance at arbitrary (non-pool-aligned) chunk offsets | layer 0 R4097: chunks 0.99967-0.99978 vs HF, 0.99999 vs single-shot (chunk 0, carry-free, already 0.999996: ttKDA geometry at 4128 vs 1024 rows), decode 0.9998; layer 3 R4097: 0.999937 / 0.999938 vs HF (single-shot 0.999938), 0.999999 / 1.000000 vs single-shot, chunks 0.99992-0.99998, row classes >= 0.99986, the straddling chunk [1536,2560) 0.99996, decode 0.99996; T=2040: layer 0 bit-identical to single-shot (max diff 0, 100 % equal), layer 3 PCC 1.000000 (99.83 % equal); R2042 / R2040 512-chunks 0.99987 / 0.99988 vs HF (row classes >= 0.99987), decode >= 0.99978, pool cache worst row 0.99999; indexer offsets: all pools and ring rows 1.00000 |
| watcher | `TT_METAL_WATCHER=10` on the layer-3 dump test (`~/ashai/glm53/_logs/watcher_layer3`) | PASSED, 10 dumps, no watcher exceptions |
| `test_expert_dtype.py` | expert dtype gate (design section 9): routed experts bfp8 / bfp4 gate-up / bfp4 all on layers 3 and 4, MoE-only T=256 + R4097 prefill (4097 rows, per-row distribution) + 4 decode steps; `doc/functional_decoder/dtype_gate.md` | bfp4 gate/up: MoE sublayer 0.99968 (L3) / 0.99504 (L4), full layer >= 0.99945 prefill, >= 0.99747 decode; bfp8 default kept, eval-set gate pending |
| `test_all_layers.py` | every layer 0..44 on its own HF input (`L45_S128` dump: 128 random tokens + 2 decode steps through the full HF stack, produced layer by layer with `hf_layer_reference.py --stream`), prefill S=128 + 2 decode vs the live HF layer, router top-8 sets with near-tie margins, attention / FFN sub-block PCCs | 45/45 pass: prefill 0.99886 (layer 23) .. 1.0, decode >= 0.99786 (layer 27, one 4e-4-margin expert flip), per-row prefill min 0.9755 (layer 23 row 125); router sets 109-126/128 identical, every flipped prefill row below the median 8th-9th margin except 3 rows of layer 43; sub-blocks: attention >= 0.99915, FFN >= 0.99589 (prefill); HF live == dump (1.000000) for all 45 layers; 4 chips, ~2 min/layer |
| `test_full_model.py` | all 45 layers + embedding + head chained on one chip (weights streamed, `tt/model.py`): the L45_S128 random prompt vs the dump layer by layer, logits of the 128 rows vs the host head, the dump's 2 decode steps; 16-token greedy generation vs HF greedy (`doc/full_model/README.md`) | chain PCC vs dump 0.9998 (layer 0) -> 0.9966 (11) -> 0.967 (19) -> 0.857 (23) -> 0.69 (29-33) -> 0.858 (44), top-1 0.23 over the random rows (head alone 0.99997 / 0.97); the HF bf16 chain itself with a 1 % per-layer perturbation (PCC 0.99995 per layer) decays the same way (0.907 at 23, 0.75 at 29-33: `reference/hf_chain_sensitivity.py`), so the random-prompt chain is the model's own sensitivity, not a stack defect; real-prompt greedy generation matches HF on the leading tokens (see the full-model doc); cached MoE layer build 5-10 s, pass over 45 layers 4-8 min on a loaded host |
| `test_perf_layer.py` | warmed prefill S=1024 (wall), traced decode B=1 (one replay, tt-perf-report `Device Time`) | prefill: layer 3 325.7 ms wall (re-measured 2026-10-20 at 5c5122ccf3a + the review-3 fixes, `GLM53_PERF_PHASE=prefill`, no tracy, `_logs/fix3_perf_layer3_prefill.log`; host load average 60-110 on 64 cores from other sessions' CPU jobs during the run; the earlier 314 ms predates eca757b2ffc and 4cd5f8fe829, which added the prefill index maintenance), layer 4 297.8 ms wall from the tt-perf-report run that predates the cached-scalars fix 4cd5f8fe829 (its device-time table, 293.4 ms over 1002 ops, is unaffected: the fix removed host-side uploads, not device ops); decode: layer 3 10.87 ms/step device (435 ops, wall 11.31), layer 4 10.76 ms/step device (471 ops, wall 11.2) |

| `test_cache_1m.py` | 1 M-position cache test (one user): layer 3 `FunctionalDecoder` at `max_seq_len = 1 048 640`, context seeded synthetically through `TtDSA.prefill_state` in 32 K chunks (realistic-statistics random attention inputs; device latent / pooled-key rows checked against the HF module's own stems per chunk), decode steps at N, N+1 (dense op unbounded) and N+2, N+3 (bounded) for N = 16384 / 131072 / 1048576 with real R4097 decode rows vs the genuine HF `Glm5NextTextAttention` forward on a `DynamicCache` holding the host rows (130 GiB of host K/V at 1 M) and its restricted form (same ops over the HF-selected keys; proven identical by `test_host_reference_matches_hf`), selected-id overlap, per-stage eager wall (`StageTimer`), DRAM view (`_logs/cache1m_final.log`) | builds in 61-77 s: 8.42 GiB resident (7.35 weights + 1.000 GiB latent pages + 64 MiB pooled keys; x 11 DSA layers = 11.69 GiB), 23.4 GiB free; seeding: latent rows PCC >= 0.99983, pooled keys >= 0.99999 on all 33 chunks (state-only prefill 1.5 M tok/s per 32 K chunk warm); decode vs HF-full 0.999820 / 0.999788 / 0.999694 at 16 K / 131 K / 1 M (restricted 0.9996-0.9998 at every step, full == restricted to 1 bf16 ulp, HF ids identical); id overlap 0.92-0.94 / 0.85-0.92 / 0.84-0.88 (synthetic Gaussian context: dense near-ties at the top-512 boundary, see Known limits); eager sparse attention step at 1 M: 26-44 ms warm with the dense op bounded vs 120-131 ms unbounded (the discarded dense flash-MLA row over the 1 M prefix = 94.5 ms; 131 K: 13-15 vs 46 ms; 16 K: 7-9 ms); whole layer eager step at 1 M 26.5 ms bounded vs 119.9 ms (131 K 20.0 vs 26.7, 16 K 25 ms); DRAM at stage boundaries peaks at 9.44 GiB (+1 GiB inside the kv untilize) |

Context contract: `doc/context_contract.json`. Full log: `doc/functional_decoder/work_log.md`.
Expert dtype gate evidence (knobs `expert_dtype` / `expert_down_dtype`, default bfp8): `doc/functional_decoder/dtype_gate.md`.
Profiler note: Tracy device zones of the SDPA prefill kernels (dense and sparse share `compute_streaming.hpp`: `:632` vs
`:1769`, hash 0x6835) collide on main 8cfcaf754d3, so prefill tables exist for layer 4 only; decode tables for both layers
come from one-replay runs (`GLM53_PERF_DECODE_STEPS=1`, layer 3 with `GLM53_PERF_SEED_ONLY=1`), tables under `doc/functional_decoder/tracy/`.

## KDA decode block (fused, `GLM53_KDA_FUSED=1`)

Design 8.1 / the 2026-10-20 directive (every data-movement op of the block merged or removed). One chip (chip 3 of
the Galaxy, 64 heads), real layer-4 weights, traced decode of the KDA layer alone (`tests/test_kda_perf.py`), ms per step:

| users | composite (ops) | fused block (ops) | `kda_decode` device us | block device us |
|---|---|---|---|---|
| 1 | 1.378 (72) | 0.720 (6) | 44 (as 64 one-user items) | - |
| 4 | 2.366 (75) | 0.794 (6) | 209 | 751 |
| 8 | 3.546 (75) | 0.898 (6) | - | - |
| 32 | 11.425 (75) | 1.491 (6) | 906 | 1453 |

Op-level (`tests/ttnn/unit_tests/operations/transformers/test_kda_decode.py::test_kda_decode_perf`, eager): 241 us at
B=4 / 930 us at B=32 with 64 heads; 197 us / 341 us with the TP4 shard's 16 heads (the users of a head are split over
cores when heads x tile rows leave cores idle). Full layer 4 (KDA+MoE, `test_perf_layer.py`, `GLM53_PERF_B`): B=1
8.23 -> 7.55 ms/step, B=32 18.15 -> 8.21 ms/step. The remaining block time is the weight-streaming matmuls (qkv 291 us
on 110 cores, o_proj 149 us on 64 cores, the two gate matmuls 50 + 40 us -- the first runs on 10 cores) and the
`kda_decode` program (64 items of 32 serial users at B=32 on this chip).

The block: `qkv = x W_qkv` (bf16) -> `g1 = x [W_fa | W_ga | W_b]` (bf16, as HF) -> `g2 = g1 blockdiag(W_fb, W_gb, I)`
(fp32: `[f | og | b_logits]`) -> `ttnn.transformer.kda_decode(qkv, g2, conv_state, recurrent, taps, [dt_bias; exp(A_log)],
o_norm_w, selectors, H)` -> tilize -> `o_proj`. The program (one Tensix core per (head, 32-user tile row, user part))
does the causal conv + SiLU on the device-owned bf16 history (read, shifted in place, raw rows appended), the per-head
L2 norms, `g = -5 sigmoid(exp(A_log) (f + dt_bias))`, `beta = sigmoid(b)`, the per-key-channel delta-rule step per user
(the user's rows are lifted into tile row 0 by one-hot selector matmuls; the state is read and written once, in
place) and the gated RMSNorm, and writes the bf16 row-major `[B, H*V]` output. The per-key-channel decay was first
added to `ttnn.transformer.fused_recurrent_gated_delta_rule` (rank-4 `g`, heads beyond the core count dealt
round-robin, `GLM53_KDA_FUSED=2` keeps that path: composite conv/gates + one fused recurrence op, 63 ops).

Correctness (`tests/test_kda.py`, real weights): block vs HF decode output PCC 0.99993 / 0.99991 (prefill 96 / 1000
then 3 steps; the composite 0.99991 / 0.99991), state PCC 0.99993 / 0.99886 (composite 0.99993 / 0.99885);
`test_kda_block_vs_composite` (layers 0 and 4, B=4 and 32, 10 steps on one evolving state): worst output PCC vs the
composite 0.99996, state 0.99999, conv history 1.0, and vs HF (user 0) the block is at or above the composite every
step. `test_decoder_layer.py` and `test_decode_trace.py` pass on the block path (5/5 each, worst decode PCC 0.99973,
trace-safe: the program updates the persistent state and history in place and allocates only its output). The op's
own tests: `tests/ttnn/unit_tests/operations/transformers/test_kda_decode.py` (torch reference of the block, B=1/4/32/40,
H=16/64, multi-step) and `test_fused_recurrent_gdn_per_key.py` (the per-key decay / round-robin heads / ring mode of
the extended recurrence op). Logs: `_logs/optkda_*.log`, `_logs/optkda_ops_*.txt`.

Limits: K = V = 128 and conv kernel 4 are hard-wired in the program; the TP-mesh placement of the block path
(`w_g1` per-chip columns) is written but not run on a mesh (single chip only here); padding rows of `qkv`/`gates`
must be finite (ttnn matmuls on zero-padded inputs keep them zero).

## DSA decode block (fused, `GLM53_DSA_FUSED=1`)

Same directive as the KDA block (every data-movement op merged or removed). One chip (chip 3, 64 heads), real
layer-3 weights, traced decode of the DSA layer alone at position 1024 (`tests/test_dsa_perf.py`), ms per step:

| users | composite (ops) | fused block (ops) | fused device us: matmuls / MLA decode / state op / transposes |
|---|---|---|---|
| 4 | 1.327 (41) | **0.859 (7)** | 614 / 158 / 42 / 27 (840 total) |
| 32 | 1.892 (41) | **0.971 (7)** | 621 / 185 / 62 / 96 (964 total) |

Full layer 3 (DSA+MoE, `test_perf_layer.py`, `GLM53_PERF_B`): B=4 7.05 -> 6.51 ms/step, B=32 10.94 -> 9.94 ms/step.
The composite's 41 ops were 5 stem slices, 3 single-core norms, 4 x (InterleavedToSharded + PagedUpdateCache),
2 ring slices, the 7-op pooling, 5 ReshapeViews (455 us at B=32) and 4 transposes around the absorb bmms.

The block: `stem = x W_stem` (one [4096 x 2336] matmul) -> **one `ttnn.transformer.dsa_decode_state` program**
(role-A cores: the q_a RMSNorm of each 32-user tile row, returned as q_resid; one role-B core per user: its
RMSNorm'd kv_a row placed at row a % 32 of its latent page, its LayerNorm'd indexer k row and raw gate row placed at
ring slot a % 4, the pooled key of pool a // 4 = softmax-weighted mean over the ring (ape added, unused ring rows
masked, shift by the first in-pool row) placed in the pool cache -- all read-modify-written in place, rows moved by
one-hot selector matmuls) -> `q_lat = q_resid W_qb1` with `W_qb1[h] = W_qb[h] W_kb1[h]` (the absorb folded into the
projection: one [1536 x H*512] matmul) -> `transpose(1, 2)` of the free head-major view (for B <= 32 the packed
tiled rows are the [1, H, B, 512] tile sequence; a logical reshape would shuffle rows at ~5 GB/s, 448 us at B=32)
-> `paged_flash_multi_latent_attention_decode` -> `transpose` -> `y = o_lat W_o2` with
`W_o2[h] = W_kb2[h]^T W_o[h]` stacked over heads (one [H*512 x 4096] matmul, the head sum inside the K dim)
(+ the TP all-reduce on a mesh). Users at positions > 2050 keep the per-user indexer + `sparse_sdpa` row
replacement on the head-major view (`test_dsa_sparse.py`, `test_dsa_long.py` pass on the block).

Correctness (real layer-3 weights): block vs HF decode PCC 0.99991 / 0.99991 / 0.99989 (3 steps after a 96-token
prefill; the composite 0.99991 / 0.99991 / 0.99989); `test_dsa_fused_vs_composite` (B=4 and 32, 10 steps on one
evolving cache set): latent cache, pools and both rings bit-exact between the paths, worst output PCC 0.99983
(the state program alone, `GLM53_DSA_FOLD=none`: 0.99990; q fold only 0.99989; o fold only 0.99985) -- the bfp8
products round differently from the composite's two bfp8 matmuls with a bf16 intermediate; vs HF the folded block
is at or above the composite (worst step 0.99954 vs 0.99946 at B=4). `test_decoder_layer.py` / `test_decode_trace.py`
(layers 0, 3, 4; traced; two users) pass on the block path. The program's own test:
`tests/ttnn/unit_tests/operations/transformers/test_dsa_decode_state.py` (torch reference, B=1/4/32/40, permuted
page tables, 6-step evolution). On the Galaxy (TP4, 16 local heads) the folded weights read 46 MB per layer-step
instead of the composite's 27 MB (+50 us) and remove the per-head bmms (160 us) and all relayouts; on this 64-head
chip they read 184 MB (the matmul share above).

Limits: K/V-lora widths 1536 / 512 and index dim 128 are wired into the program (D = 128, L and Lq multiples of
128); the latent cache must be bf16 (the program read-modify-writes bf16 tiles); B + ceil(B / 32) cores; the
TP-mesh placement of the folded weights (`cols=-1` / `cols=-2`, head-major) is written but not run on a mesh here;
`GLM53_DSA_FUSED` defaults to off. Logs: `_logs/optkda_dsa*.log`, `_logs/optkda_dsaops_*.txt`.

## Running

```bash
source ~/ashai/glm53/env.sh
TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests -s           # all device tests (one chip)
python models/demos/glm53_flash/reference/hf_layer_reference.py --layers 0,1,2,3,4 --seq-len 256 --num-decode 4 \
    --out bringup/references/glm53_flash/T256                            # regenerate the HF dumps (CPU)
python models/demos/glm53_flash/reference/hf_layer_reference.py --layers 0-44 --seq-len 128 --random-ids 45 \
    --num-decode 2 --stream --out bringup/references/glm53_flash/L45_S128   # all-layer dump (layer by layer, ~25 min)
TT_VISIBLE_DEVICES=0 GLM53_LAYERS=0-11 pytest models/demos/glm53_flash/tests/test_all_layers.py -s --timeout 1800
                                                                         # every layer vs HF (shard GLM53_LAYERS over chips)
```

## Known limits (single chip, v1)

- Chunked prefill needs 32-aligned chunk starts (a non-aligned logical length only in the last chunk); the
  dense rows of a chunk re-materialize K/V for the whole prefix (<= 2048 rows, bounded) and the sparse rows read
  the user's latent rows `[0, start + T_pad)` once per chunk. The sparse prefill materializes a
  `[rows >= 2048, 4 * pools]` bf16 score matrix per DSA layer per chunk (~0.5 GB for a 16K single shot), so the
  practical single-chip prefill bound is that quadratic intermediate, not DRAM; chunking bounds it per chunk.
- Single-chip capacity (31.83 GiB allocatable DRAM = 8 banks x 3.979 GiB; 7.35 GiB resident after the layer-3 build,
  `_logs/fix3_perf_layer3_prefill.log`), two different answers:
  (i) decode is not capacity-bound at the advertised 1 M positions for one user: 11 DSA layers x 1,048,576 x 512 x 2 B
  = 11 GiB of latent pages + pooled keys (one [128] bf16 row per 4 tokens = 64 B/token: 0.69 GiB) + 2 x 32 x 128 x 2 B
  ring rows per user and layer. Validated for one layer at 1 048 576 positions (`test_cache_1m.py`, synthetic context,
  one user): no kernel limit is hit (`topk_large_indices` rows up to 2^30 elements, `indexer_score_dsa` kv_len 262 176
  pools, the whole-prefix slice + untilize of 1 GiB, `sparse_sdpa` over 1 048 608 row-major rows, the flash-MLA decode
  page table of 32 770 blocks). Per-step costs of the eager sparse decode that grow with the user's context (one layer,
  warm, `_logs/cache1m_final.log`): the user's whole latent prefix is copied and untilized for `sparse_sdpa`'s row-major
  kv contract (kv_slice + kv_untilize: 12-21 ms at 1 M, 1.8 ms at 131 K, 0.5 ms at 16 K; +2 GiB of transient DRAM at
  1 M), the pooled-score pass (index scoring 1.6-2.2 ms at 1 M, top-512 pools over 262 176 scores 2.5 ms + 1 ms id expansion; before the pool-level top-k the 4x token-level expansion + top-2048 cost 13 + 9 ms); the dense
  flash-MLA row computed for every slot reads the whole prefix (94.5 ms at 1 M, 12 ms at 131 K) unless its position is
  bounded with `DecodePositions.dense_positions` = min(position, 2050) (0.4 ms; the row of a sparse user is discarded,
  so this is exact). A paged / tiled kv read in `sparse_sdpa` (Phase 3 fusion) removes the untilize.
  (ii) single-shot prefill IS capacity-bound: for a prompt of T tokens (S = T - 2048 sparse query rows,
  W = 32 * ceil(max_seq_len / 128) pooled-key columns, about T/4) the peak is inside `TtDSAIndexer.select`. Until
  the pool-level top-k (selection of the top-512 pools over `[S, W]` with the token ids expanded on device) it sat at
  the 4x `repeat_interleave` of the scores (ttnn implements it along the last dim as transpose, 4x concat, transpose
  back): the TILE scores `[S, W]` bf16, their ROW_MAJOR copy, the transposed copy, the repeated `[4W, S]` and the result
  `[S, 4W]` were live together = 22 * S * W bytes (about 5.5 * T^2); now the quadratic term is the three `[S, W]`
  copies (TILE, the kv_len slice, ROW_MAJOR) = 6 * S * W bytes, and the figures below (kept as measured before the
  change) are upper bounds, plus the tensors held across the call (fp32 streams
  64 KiB/row, pre-norm + normalized input 16 KiB/row, q 32 KiB/row, q_resid 3 KiB/row, the sparse-row slices and
  indexer queries 19 KiB/row, the dense-row output 64 MiB, the caches 1 KiB + 64 B per token). The next-highest point,
  `sparse_sdpa` (q_rm + o_rm `[64, S, 512]` bf16 = 128 KiB/row, kv_rm, indices `[S, 2176]` u32), is linear in T.
  Evaluated: T = 16385 -> 1.21 GiB quadratic, 3.35 GiB peak (the validated point); 32768 -> 9.4 GiB; 49152 ->
  18.3 GiB; the largest single-shot prompt that fits the 24.48 GiB free next to the layer-3 weights is about
  57 K tokens (58368, peak 24.4 GiB; computed, not validated). Chunked prefill (`prefill_forward(..., start=)`, delivered) scores `[T_chunk, W]` per chunk and removes this bound.
- Runtime audit method: `grep -nE 'from_torch|to_torch|torch\.'` over `tt/*.py` forward paths (hits are
  construction-time or first-use-cached constants only) plus `throw_exception_on_fallback=True` runs
  (`test_determinism.py`); the ring tail rows and the KDA scalars use tensors created once per geometry/value.

- Performance is the design's Phase 2/3 work: the KDA decode block is fused behind `GLM53_KDA_FUSED=1` (default off until the integrator flips it; see below), the per-block MoE is a composite,
  the sparse decode untilizes the user's latent prefix per step, and the decode SDPA runs on 4 cores per user.
- The sparse DSA decode path takes host positions (chunk geometry) and is therefore not captured in the trace;
  the pooled-key index is nevertheless maintained inside the traced dense graph (checked against a host
  recomputation in `test_decode_trace.py`), so a user can cross 2050 and continue on the eager sparse rows.
- Indexer scores are ranked at bf16 resolution (`indexer_score_dsa` + `topk_large_indices`): 98-99 % of the
  HF-selected ids per step on a random-token prompt (4 K), layer PCC >= 0.9997. On the synthetic Gaussian context of
  `test_cache_1m.py` (no structure, so the top-512 pool boundary is dense with near-ties) the overlap falls to 92-94 %
  at 16 K, 86-92 % at 131 K and 85-88 % at 1 M while the attention output stays at PCC >= 0.9996 vs HF: the swapped
  boundary pools carry little attention weight. An fp32 scoring path is the Phase 2/3 option if token-level accuracy
  at long context shows a gap.
- `moe_grouped_topk` is not used (bf16 ranking swaps near-tied experts); the exact fp32 composite costs ~8x5 ops.
- Per-row PCC tail of the MoE layers = near-tie routed-expert swaps: when the 8th and 9th biased router scores are
  within ~3e-3 (18 % of tokens are < 1e-3 apart, 39 % < 3e-3), the exact fp32 router on the device's bf16
  post-attention stream can pick HF's 9th expert; such rows score 0.98-0.999 (layer 3, 32 prompts of the B=32 case:
  21/2542 rows < 0.997, min 0.98025; every low row checked in `_logs/fix3_router_flip_user18.log` is such a swap,
  rows with identical expert sets are at 0.9999). A 1..5-row prompt is a handful of draws of this distribution,
  which is why `test_batch8.py` B=32 layer 3 fails the 0.995 per-user gate on its S=5 user 18 (0.994797) while all
  64 decode samples pass; the bar is kept and the case is red on this draw. Phase 2/3 option: fp32 router input.
