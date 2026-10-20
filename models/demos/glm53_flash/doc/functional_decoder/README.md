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
| `decode_forward(streams, positions)` | `[1,1,B,4*4096]` fp32 (B == max_batch user slots); `positions` = `DecodePositions` from `make_decode_positions(host_positions)` / `update_decode_positions` (persistent `[B]` int32: position, position // 4, position % 4, + host values) | `[1,1,B,4*4096]` | KDA state updated in place; DSA latent appended and the pooled-key index maintained; dense flash-MLA for every slot (HF-exact through position 2050 = index_topk + index_kpool - 2), then the rows of the users whose host position is > 2050 are overwritten by the per-user indexer + `sparse_sdpa` result (eager, host chunk geometry) |
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
| `tt/kda.py` | KDA: prefill via `ttKDA` (chunked kernels), decode = composite recurrence on `[B*H,128,128]` fp32 with in-place state writeback | `deepseek_v3_d_p/tt/kda` |
| `tt/dsa.py` | MLA (NoPE): dense causal SDPA for prefill rows [0, 2048), paged latent cache, absorbed `paged_flash_multi_latent_attention_decode` for every decode slot; prefill rows >= 2048 and decode users at position > 2050 via the indexer + `sparse_sdpa` | paged cache ops, flash MLA decode, sparse SDPA |
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

Context contract: `doc/context_contract.json`. Full log: `doc/functional_decoder/work_log.md`.
Expert dtype gate evidence (knobs `expert_dtype` / `expert_down_dtype`, default bfp8): `doc/functional_decoder/dtype_gate.md`.
Profiler note: Tracy device zones of the SDPA prefill kernels (dense and sparse share `compute_streaming.hpp`: `:632` vs
`:1769`, hash 0x6835) collide on main 8cfcaf754d3, so prefill tables exist for layer 4 only; decode tables for both layers
come from one-replay runs (`GLM53_PERF_DECODE_STEPS=1`, layer 3 with `GLM53_PERF_SEED_ONLY=1`), tables under `doc/functional_decoder/tracy/`.

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
  ring rows per user and layer; the single-chip decode bound is validation time of the v1 composite paths.
  (ii) single-shot prefill IS capacity-bound: for a prompt of T tokens (S = T - 2048 sparse query rows,
  W = 32 * ceil(max_seq_len / 128) pooled-key columns, about T/4) the peak is inside `TtDSAIndexer.select` at the 4x
  `repeat_interleave` of the scores (ttnn implements it along the last dim as transpose, 4x concat, transpose back):
  the TILE scores `[S, W]` bf16, their ROW_MAJOR copy, the transposed copy, the repeated `[4W, S]` and the result
  `[S, 4W]` are live together = 22 * S * W bytes (about 5.5 * T^2), plus the tensors held across the call (fp32 streams
  64 KiB/row, pre-norm + normalized input 16 KiB/row, q 32 KiB/row, q_resid 3 KiB/row, the sparse-row slices and
  indexer queries 19 KiB/row, the dense-row output 64 MiB, the caches 1 KiB + 64 B per token). The next-highest point,
  `sparse_sdpa` (q_rm + o_rm `[64, S, 512]` bf16 = 128 KiB/row, kv_rm, indices `[S, 2176]` u32), is linear in T.
  Evaluated: T = 16385 -> 1.21 GiB quadratic, 3.35 GiB peak (the validated point); 32768 -> 9.4 GiB; 49152 ->
  18.3 GiB; the largest single-shot prompt that fits the 24.48 GiB free next to the layer-3 weights is about
  57 K tokens (58368, peak 24.4 GiB; computed, not validated). Chunked prefill (`prefill_forward(..., start=)`, delivered) scores `[T_chunk, W]` per chunk and removes this bound.
- Runtime audit method: `grep -nE 'from_torch|to_torch|torch\.'` over `tt/*.py` forward paths (hits are
  construction-time or first-use-cached constants only) plus `throw_exception_on_fallback=True` runs
  (`test_determinism.py`); the ring tail rows and the KDA scalars use tensors created once per geometry/value.

- Performance is the design's Phase 2/3 work: the KDA decode recurrence and the per-block MoE are composites,
  the sparse decode untilizes the user's latent prefix per step, and the decode SDPA runs on 4 cores per user.
- The sparse DSA decode path takes host positions (chunk geometry) and is therefore not captured in the trace;
  the pooled-key index is nevertheless maintained inside the traced dense graph (checked against a host
  recomputation in `test_decode_trace.py`), so a user can cross 2050 and continue on the eager sparse rows.
- Indexer scores are ranked at bf16 resolution (`indexer_score_dsa` + `topk_large_indices`): 98-99 % of the
  HF-selected ids per step on a random prompt, layer PCC >= 0.9997.
- `moe_grouped_topk` is not used (bf16 ranking swaps near-tied experts); the exact fp32 composite costs ~8x5 ops.
- Per-row PCC tail of the MoE layers = near-tie routed-expert swaps: when the 8th and 9th biased router scores are
  within ~3e-3 (18 % of tokens are < 1e-3 apart, 39 % < 3e-3), the exact fp32 router on the device's bf16
  post-attention stream can pick HF's 9th expert; such rows score 0.98-0.999 (layer 3, 32 prompts of the B=32 case:
  21/2542 rows < 0.997, min 0.98025; every low row checked in `_logs/fix3_router_flip_user18.log` is such a swap,
  rows with identical expert sets are at 0.9999). A 1..5-row prompt is a handful of draws of this distribution,
  which is why `test_batch8.py` B=32 layer 3 fails the 0.995 per-user gate on its S=5 user 18 (0.994797) while all
  64 decode samples pass; the bar is kept and the case is red on this draw. Phase 2/3 option: fp32 router input.
