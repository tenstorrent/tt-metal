# DeepSeek-V4.1-Flash device prefill on the 4x8 Blackhole Galaxy - design (M0)

Status: draft for review. Package `models/demos/blackhole/deepseek_v41_flash`. Author: prefill agent, 2026-10-02.
Scope (updated by the user): batched prefill of many users, prompts from 128 tokens up to 64k, 1M-token context as stretch
goal, PAGED KV cache like GPT-OSS. Layout of the paged cache is owned by the KV-capacity agent
(`2026-10-02-dsv41-kv-paged-capacity-design.md`, not yet published); this document is layout-agnostic and lists what it needs.

## 1. What has to be produced

After a prompt of S tokens per user (16..128 users) the device must hold exactly the state decode consumes today:

| decode state (per attention layer) | where it lives now | prefill must write |
|---|---|---|
| window KV ring, position p at slot p % 128 | `attn.cache[:, :, 0:128, :]` | K (=V) of the last 128 prompt positions, post kv_norm + RoPE |
| compressed latents (kv-source layers 2, 8, 14, 20; readers copy them) | `attn.cache[:, :, 128+j, :]` | latent j = pool of tokens [j*r, (j+1)*r), RMSNorm, RoPE at position j*r |
| ratio-2 compressor "previous token" `prev_cs` | `attn.prev_cs` fp32 `[1,1,T,1024]` = [kv\|score] of token S-1 | same |
| Engram n-gram hash history | host `NgramHashState.cache` | computed on the host by `hashes(prompt, 0)` (nothing on device) |
| (new, long context) indexer keys of index-source layers | not implemented in decode | `index_k` (ratio-compressed, 128-wide) |

plus the first token logits (last prompt token -> final hc_pre, norm, LM head). Reference semantics (checkpoint `model.py`): ring slot
for prefill is `p` when S <= 128 and `p % 128` otherwise; ratio 2 pools NON-overlapping groups with softmax over the pair and keeps the
tail token(s) of an incomplete group in `kv_state/score_state`; ratio 1 has no pooling (`norm(wkv(x))`, one latent per token). A query at
position t sees compressed entries j < (t+1)//ratio (ratio 1: j <= t) PLUS its 128-token window; with <= 512 entries the indexer selects
everything, beyond that it picks the top-512 per query.

## 2. Reuse analysis

### 2.1 Decode graph blocks that are token-count agnostic (reuse as is, T = 32 "tokens" per mesh row)

mHC (`mixes`, `collapse_norm[_rm]`, `expand`: fused kernels, verified T = 4/8/16/32), `DSV41MoEBlock` (exact fp32 router + `moe_compute`
with the decode-format bfp8 expert ring, up to 32 tokens per device per call), the shared expert, device Engram (`DSV41DeviceEngram.forward`
with rows `[32,1,1,6144]`), embedding, `DSV41DeviceHead`. None of them know about positions, so a chunk of 32 consecutive prompt
positions (of one or several users) is just "32 users" for them. The MoE is the constraint that decides the design:

* routed experts exist ONLY in the packed bfp8 `moe_compute` ring (~450 MB/chip/layer, all 40 layers resident = most of the chip's DRAM). A second
  prefill-format copy (~18 GB/chip) does not fit (~6 GB free per chip).
* `moe_compute` takes <= 32 tokens per device and the tokens are REPLICATED over the 8 mesh columns (each column owns a different
  expert slice and the output is reduce-scattered over the columns), dispatch runs along the 4 mesh rows. So one MoE call handles
  4 x 32 = 128 tokens of the whole mesh, independent of how many columns exist. At 128 tokens x top-6 over 384 experts nearly every
  expert is active, so a call costs ~ the full DRAM sweep of the layer's experts (~60 us + ~83 us x 12 experts of the busiest device
  ~ 1.0-1.1 ms). Prefill therefore runs at about 128 tokens / (40 layers x ~2 ms) ~ 1.5-1.6k tokens/s, whatever S is, until/unless the
  MoE kernel is generalised (out of scope; listed in section 9).

### 2.2 deepseek_v3_d_p (repo prefill machinery)

* `tt/moe/tt_moe.py` (+ dispatch/combine/routed expert): prefill-style MoE with dense per-expert matmul weights ([experts_per_chip, ...] bf8/bf4), several GB
  per layer; format differs from the moe_compute ring -> NOT reusable under the DRAM budget.
* `tt/mhc/tt_mhc.py`: already used by decode (consts, `build_consts`) - the fused kernels in `tt/mhc_*.py` are what we use.
* `tt/mla/sliding_window_attention.py` (`TtSWA`, V4 sliding layers, chunked prefill with halo + carry): confirms on this build that
  `ttnn.transformer.scaled_dot_product_attention` runs with head_dim 512, an explicit additive mask (baked sliding band, `is_causal=False`),
  `attention_sink` (pass `sink / scale`, same convention as our decode) and `q_chunk_size = k_chunk_size = 128` ("with a mask the reader streams
  a [q_chunk,k_chunk] tile, wider sizes exhaust L1"). Its structure (sequence-parallel across rows, batch 1, TP row-parallel q/kv with reduce-scatter) does
  not match our layout (users across rows, heads across columns, replicated activations), so we copy PATTERNS, not the module.
* `tt/mla/compressor.py`, `heavily_compressed_attention.py`, `indexer.py` (1123 lines): V4-HF compressors (overlapping windows, ratios 4/128), written for
  sequence-parallel; useful for the indexer scoring pipeline (needed beyond ~1k tokens), not for our ratio-2/ratio-1 compressor.
* `ttnn.transformer.sparse_sdpa` (sdpa/sparse_sdpa.hpp): single-chip Blackhole sparse prefill SDPA: q `[1,H,S,K_DIM]`, kv `[1,1,T,K_DIM]`, `indices [1,1,S,TOPK]`
  (0xFFFFFFFF sentinels, contiguous tail), `v_dim`, `attention_sink` (V4 convention = `sink/scale`), `cache_batch_idx` into a shared `[B,1,T,K_DIM]` cache. H must be a
  multiple of 32 -> our 8 heads/column must be padded to 32 (4x waste) or the layout switched to 32 heads per device for the sparse layers. This is the
  candidate for compressed layers beyond 512 entries (section 6).

### 2.3 GPT-OSS prefill (what to mimic)

| aspect | GPT-OSS (`tt/model.py ttnn_prefill_forward`, `tt/attention/prefill.py`, `tt_transformers/tt/generator.py`) | DSV4.1 plan |
|---|---|---|
| batching | short prompts: all users of the call in the batch dim, `[1,1,B*S,H]` activations, QKV matmul on `B*S`, reshape `[B,1,S,*]` for SDPA; allowed while `B*S <= max_prefill_chunk_size`; padded to power-of-2 `prefill_seq_len` buckets (`SUPPORTED_PREFILL_BATCH_SIZES`) | same: per mesh row `users_per_row x S_pad` tokens, S_pad multiple of 32 (power-of-2 buckets for trace reuse) |
| long prompts | one user at a time, chunks of `max_prefill_chunk_size`; per chunk `chunk_page_table` (blocks of that chunk), `chunk_start_idx`; attention via chunked SDPA reading K/V from the paged cache | same driver structure: `prefill_chunk(tokens, chunk_start, chunk_page_table)`; window layers read a 127-token halo, compressed layers read the paged compressed cache |
| KV write | `ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx)`; per-user call for batch > 1 (flattened batch is wrong) | `paged_fill_cache` of K for the window ring/halo and of latents for the compressed cache, per user |
| last token | `get_last_token` = tile of `last_token_idx`, slice 32 rows then norm+lm_head only on that tile (`process_logits_after_prefill_trace`, slice offsets as device tensors so one trace serves all prompt lengths) | same: only the tile with the last real token goes through hc_pre/norm/head |
| trace | one prefill trace per (bucket length, batch) | one trace per (bucket, chunk) once eager is correct |
| MoE | GPT-OSS has dense-prefill-capable expert weights (`moe_compute` made default on BH) | decode-format experts at T=32 chunks (above) |

Key difference: GPT-OSS K/V are 2 tensors x 8 KV heads; here K == V, ONE KV head of 512 dims, ring-shaped window (128) + compressed sources only, so the
cache per user is tiny (section 5) and the long-context constraint is the compressed cache + indexer, not the window.

## 3. Prefill graph (per layer)

Notation: per mesh row R tokens (= users_per_row x chunk_len, the SAME for every row), chunks of 32 for the token-wise blocks.

```
x  streams for the R tokens (compact store, section 5.2)             [R,4,D] fp32, replicated over columns
for c in chunks of 32 tokens:   x_c [32,1,4,D] -> mHC mixes(attn), collapse_norm        (decode kernels, T=32)
h = concat_c(h_c)  [1,1,R,D] bf16
a = PrefillAttention(h)  # new: S-wide projections, RoPE, SDPA, comp. state, paged writes, o-proj, column all-reduce
for c: expand(a_c, x_c) -> mixes(ffn) -> collapse_norm_rm -> MoE(T=32) + shared -> expand      (decode blocks, T=32)
```

Engram layers (1, 14): before the layer, per chunk `DSV41DeviceEngram.forward(x_c, rows_c)` (rows from the host, section 7).
After the last layer: gather the tile of the last real token of each user, final hc_pre + norm + LM head (existing `DSV41DeviceHead` at T=32 rows).

### 3.1 PrefillAttention (new, `tt/prefill_attention.py`), wraps an existing decode attention object

It REUSES the decode weights already resident (`wqkv` [wq_a|wkv], `q_norm`, `kv_norm`, `wq_b` column shard, `wo_a`/`wo_b`, `Pf` rope matrix, compressor `c_wcat`/`c_wkv`/`c_norm`), so
prefill costs no extra weight DRAM. Per device (row r, column c), users u = 4 per row:

1. `y = h @ wqkv` -> `[1,1,R,1792]`; `qr = rms_norm(y[:, :1280])`; `q = qr @ wq_b_local` -> `[R, 8*512]`; `kv = rms_norm(y[:, 1280:])`. (large-M matmuls, default
   multicore configs are fine at R >= 512.)
2. reshape to `[U, S, ...]`, heads: `q [U,8,S,512]`, `kv [U,1,S,512]` (`nlp_create_qkv_heads`-style permutes; K == V).
3. RoPE on the last 64 dims with the full-width tables `x*C + (x@Pf)*S` for positions `chunk_start .. +S` (tables `[S,512]`, broadcast over users/heads).
4. SDPA: `ttnn.transformer.scaled_dot_product_attention(q, kv, kv, ...)`, GQA 8 q heads : 1 kv head, head_dim 512, sink = `attn_sink/scale` as a `[1,8,1,1]` tile tensor,
   * window layers (0, 1): `is_causal=True, sliding_window_size=128`;
   * compressed layers with <= 512 entries: K = `[kv (S_pad) | latents (Sc_pad)]`, `is_causal=False`, additive mask `[1,1,S,S_pad+Sc_pad]` (causal band 128 on the first block, `j < (t+1)//r`
     on the second, pad keys -inf), q/k chunk 128 (L1 rule from `TtSWA`); mask is a host constant per (S, ratio) shared by all layers.
   (To be verified on device in M1: sliding_window off-by-one, sink scale, chunk sizes - probe `tests/test_prefill_sdpa_probe.py`.)
5. inverse RoPE on o's rope dims, zero "head 0" row prepended (decode `wo_a` has 512 zero rows for the kv-in-q-tile trick), `o @ wo_a_local`, `@ wo_b_local`, all-reduce over the 8 columns
   (`mesh_config.allreduce`, same as decode `_finish`) -> `[1,1,R,D]`.
6. State outputs (written once per user): ring `[U,1,128,512]` (slice/rotate: for S > 128 `ring = concat(last[128-s:], last[:128-s])`, s = S % 128), compressed latents, `prev_cs`.
   With the paged cache: `paged_fill_cache` into the pages named by `page_table[u]`; with today's dense decode cache: build the `[U,1,128+max_comp,512]` tensor and `ttnn.copy` it into
   `attn.cache` (keeps the buffer address, so a decode trace captured earlier stays valid).
7. Compressor (kv-source layers): ratio 2: `cs = h @ c_wcat` fp32 `[R,1024]` -> `[U, S/2, 2, 1024]`; `pooled = kv_b + (kv_a - kv_b)*sigmoid(score_a - score_b)` (the decode formula, exact softmax over 2);
   `rms_norm(c_norm)`; RoPE at positions `j*2`; `prev_cs = cs[token S-1]`. ratio 1: `rms_norm(h @ c_wkv)`, RoPE at `j`. Complete groups only (S odd leaves one token in `prev_cs`).
   Readers (layers 3-7, 9-13, 15-19, 21-39) attend over the owner's latents (kept alive by the owner until its last reader ran).

Why not the decode attention for prefill: it processes one token per user, writes via `paged_update_cache`, and the whole kernel is built for T <= 32.

### 3.2 Alternatives compared (effort / benefit)

| option | TTFT S=128 x 16 users (estimate) | new code | risk |
|---|---|---|---|
| (b0) literally run S decode steps with the prompt tokens (teacher forced) | S x 65 ms = 8.3 s; 64k tokens: 70 min | none | none; also a state ORACLE for the device (decode-built state vs prefill-built state) |
| (b) THIS DESIGN: S-wide attention + decode blocks at T=32 chunks | 16 passes x 40 layers x ~2.5 ms ~ 1.5-2.5 s; 64k tokens/user x 4 users ~ 3 min | PrefillAttention + driver | medium (SDPA d=512 numerics, L1) |
| (a) deepseek_v3_d_p prefill machinery | n/a | large; needs a prefill-format expert copy (~18 GB/chip) | does not fit |

Recommendation: build (b); keep (b0) as the first end-to-end fallback and as an equality oracle (cheap: `tests/test_prefill_vs_decode_state.py`).

## 4. Chunking, batching and the memory model

Free DRAM per chip after all 40 layers are resident: ~6-7 GB (decode needs ~2.9 GB/bank... measured +72 MiB/bank/layer). Everything below is PER CHIP; R = tokens per mesh row in
flight; all 8 chips of a row hold the same R tokens (TP over heads).

Per-token resident / transient bytes (D = 5120, bf16 = 2 B):

| item | bytes per token per chip |
|---|---|
| residual stream, current + next layer, decode layout `[R,1,4,D]` fp32 TILE (4 rows padded to 32!) | 2 x 655 KB = 1.31 MB  <- unusable beyond a few k tokens |
| residual stream, COMPACT store `[R, 4*D]` fp32 tile (no padding), current + next | 2 x 82 KB = 164 KB |
| h (attention input) bf16 | 10 KB |
| attention transients (y 3.6 KB, q 8 KB x ~3 live temporaries for rope, kv, sdpa out 8 KB, o-proj partial + all-reduce out 20 KB) | ~70-80 KB |
| compressor fp32 `[R,1024]` | 4 KB |
| per-32-token chunk pipeline (padded stream chunk 21 MB, mixes, router, MoE buffers) | ~0.1-0.2 GB total, independent of R |
| Engram rows (layers 1, 14) bf16 `[R, 6144]` | 12 KB per Engram layer, only while that layer runs |

=> ~255-270 KB per token with the compact store -> R_max ~ (6 GB - 0.5 GB chunk pipeline - KV pages) / 265 KB ~ 18-20k tokens per row (~75k tokens in flight over the 4 rows). With the decode
fp32 `[T,1,4,D]` layout it would be only ~4k per row, so the first implementation task after M1 is the compact store plus a cheap conversion to the `[32,1,4,D]` chunk layout (the conversion is
row-major reshape: to_layout RM -> reshape -> to_layout TILE; to be measured, ~tens of us per chunk). Dropping stream precision to bf16 would halve it again if needed.

Chunk-size / batch trade-off: `R = users_per_row x chunk_len` (<= ~16k now). Examples (4 rows): 16 users x 128 tokens: R = 512; 16 users x 1k: R = 4k; 4 users x 16k chunk (1 user per row, 64k prompt = 4 chunks): R = 16k;
1M context single user per row: chunk 16k, 64 chunks, KV pages growing every chunk. The MoE processes 32-token sub-chunks, so R only changes memory and attention efficiency, never MoE cost per token.
SDPA with an explicit mask additionally needs a `[chunk, keys]` mask in DRAM: 16k x (16k+640) x 2 B = 0.5 GB for a chunk against 16k window-halo keys - better to use the causal+sliding-window
(no mask) kernel path for window layers (halo of 127 previous tokens prepended, 1-tile alignment padding) and keep the mask only for the < 512-entry compressed layers.

Time model (per chip-row, dominated by MoE ~ 2 ms/layer/128 tokens): prefill throughput ~1.5k tokens/s (all users together). 16 x 128 = 2k tokens ~ 1.5-2.5 s (incl. attention); 64k x 4 users = 256k tokens
~ 3 min; 1M x 4 users = 4M tokens ~ 45 min (+ indexer quadratic term, section 6).

## 5. Decode state writes and the paged cache

### 5.1 What I need from the paged-cache layout (KV-capacity agent)
1. Per attention layer: which tensors are paged: (a) window ring (probably NOT paged: 128 slots x 4 users - keep dense `[U,1,128,512]`), (b) compressed latents (paged, shared by all readers of one
   source: only layers 2, 8, 14, 20 own pages), (c) `index_k` for index sources.
2. Block size and dtype of the pages (bf16 vs bfp8 - affects the SDPA-readable form), page table shape `[users, blocks]` int32 per source, its mesh placement (row-sharded users like decode).
3. API for a prefill chunk: `paged_fill_cache(cache, latents[1,1,C/r,512], page_table_u, batch_idx=0)` per user (GPT-OSS rule: batch>1 flattening is wrong) and the chunk-local page table;
   how `max_comp` (currently 128 slots/layer, decode hard-wired) changes; decode must read latent j from page j // block (the existing `paged_update_cache`/SDPA decode paged variants).
4. Positions beyond 256: `DSV41StepState` is limited to positions <= 256 because integer tables are bf16 (flag for the decode owner).

Until then the prefill writes into today's dense `attn.cache` (section 3.1 step 6) so M1/M2 can finish without the paged layout.

### 5.2 Compact residual store
`DSV41PrefillStream` in `tt/prefill_stream.py`: `[1,1,R,4D]` fp32 TILE, `chunk(c) -> [32,1,4,D]` and `store(c, x)` via RM reshape. Used between layers.

## 6. Reaching 64k and 1M

* Window layers (0, 1) and the window part of every layer: causal sliding band, chunk c attends to `[last 127 keys of chunk c-1 | chunk c]` (halo = the ring written by the previous chunk, or kept as a tail tensor);
  the ring after the last chunk is the decode state. Cost linear in the context.
* Compressed latents grow with the context (ratio 2: ctx/2 entries per source, ratio 1: ctx). Beyond ~1000 tokens (> 512 entries) the INDEXER is mandatory:
  * index sources 2, 8, 14, 20, 24, 28, 32, 36; owners of `index_k` are the kv sources; level 1 (candidate blocks of 8, top-2048 blocks) only at layer 20.
  * per layer, per query token: `q_idx = wq_b_idx(qr)` (32 heads x 128, RoPE on 64), `weights = h @ weights_proj`, `score[t,j] = sum_h relu(q_idx[t,h]·k_idx[j]) w[t,h]` over all visible j,
    masked by `j < compress_len(t)`, top-512, re-sorted; the later layers of a group re-use the layer's top-k indices (`shared_attn.topk_idxs`, kept per index group as `[U,S,512]` uint32 = 2 KB/token).
  * cost is quadratic: ctx x (ctx/r) x 32 x 128 x 2 flop per index layer: 64k, r=1: 3.4e13 per layer (~10-30 ms/chip-set), 1M, r=1: 8.8e15 per layer x 8 layers ~ 7e16 -> ~25 s on the full mesh at ~3 PFLOPs effective (not the bottleneck vs ~45 min MoE).
    Scoring is a plain batched matmul + relu + weighted sum; top-k via `ttnn.topk` on blocks (k=512 needs a two-stage topk or the sparse op's own candidate path; to be probed in M-long).
  * attention over the selected entries: `ttnn.transformer.sparse_sdpa` (q `[1,H=32 padded,S,512]`, kv = `[window halo | chunk kv | compressed pages]` in one `[1,1,T,512]` cache, indices = window positions +
    selected compressed positions + `offset`), sink = `sink/scale`. Needs: H padded to 32 (waste 4x on 8 real heads) and a ROW_MAJOR bf16 kv; to be benchmarked, alternative = gather the
    512 selected latents per query block and dense SDPA.
* 1M: chunk 16k x 64 chunks; per chunk the compressed pages grow by C/r entries (written with `paged_fill_cache` at chunk offset `chunk_start/r`); index keys likewise; nothing else grows (rings are 128).
  Capacity (not time) is the limit: per user ~ (3 ratio-2 sources x 0.5M + 1 ratio-1 source x 1M) entries x 512 B (bfp8) = ~1.3 GB + index keys; handled by the KV-capacity design.

## 7. Engram prefill

`HostEngramRows.hashes(prompt_ids, 0)` already returns hashes for any `[B, L]` (`NgramHashState` cache sized `max_batch x max_seq_len` int64: 128 MB for 16 users x 1M) and `rows(layer, hashes)`
gathers 24 rows x 256 bf16 = 12 KB per token and Engram layer from the in-RAM fp8 tables (`load_ram`, ~215 s once, ~190 GB RSS). S = 128, 16 users: 2048 tokens x 12 KB = 25 MB per layer (50 MB for both), gather
~0.3 us/row x 98k rows ~ 30 ms. Upload per chunk as `[R_chunk,1,1,6144]` bf16 row-major (decode uses the packed row-major trick), tile on device; the device Engram runs per 32-token chunk
(`forward`: wkv matmul [32,6144]x[6144,3200 per column] + all-gather, ~0.8 ms at T=32). Hash history continues into decode because the host `NgramHashState` already contains the prompt (call
`hashes(prompt, 0)` then decode tokens at `start_pos = S`). At 64k per user the host work scales linearly (64k x 16 users = 1M tokens = 25 GB of rows/layer): streamed per chunk, never materialised.

## 8. Files (new only) and test oracle

* `reference/ref_prefill_dump.py` (DONE; CPU oracle: per layer prefill input/output streams, attn/ffn in/out, routing, final decode state, first-token logits, plus the same keys as `ref_chain` so `test_decode_steps`
  can run on the dirs). Dumps for S = 128 / 32 / 9, 16 users, 3 teacher-forced decode steps, Engram on, head on, in `/mnt/tt-data/ssinghal/dsv4-prefill-s{128,32,9}` (running in the
  background on .44, ~50-70 s/layer at S=128 under contention, ~45 min total; layer 0 alone S=128 measured 50 s).
* `tt/prefill_attention.py` (PrefillAttention + compressor prefill + state writers), `tt/prefill_masks.py` (host mask / rope table builders, cached per (S, ratio)),
  `tt/prefill_stream.py` (compact residual store), `tt/prefill_layer.py` (T=32 chunk pipeline around a built `DSV41Layer`), `tt/prefill_model.py` (embedding, Engram feed, layer loop, last-token head, state hand-off),
  later `tt/prefill_indexer.py`, `tt/prefill_paged.py`.
* `tests/test_prefill_sdpa_probe.py` (SDPA d=512 / sink / window / mask / chunk sizes vs torch), `tests/test_prefill_attention_device.py` (layers 0, 2, 20; S = 9, 128: attention out PCC, state PCC),
  `tests/test_prefill_layer_device.py` (M1: hidden PCC >= 0.999, state equality, decode continuation), `tests/test_prefill_model_device.py` (M2/M3).
* No edits to the other agents' files (attention.py, layer.py, decoder.py, mhc*, moe_block, router, shared_expert, engram, step_state, model.py, reference/).

Oracle assertions (M1): hidden-state PCC >= 0.999 vs `prefill.h_out` (first S positions only; pads dropped); ring/compressed cache PCC >= 0.999; `prev_cs` PCC >= 0.9999 (bit-exactness is impossible with bfp8 weights; for even S the reference holds the
zero/-inf init in that slot, so the device value is compared with the CPU projection of token S-1); then the existing decode layer run on the new state against `dec_out`.

## 9. Risks / open items

1. SDPA prefill at d=512 with sink + mask + GQA 8:1 on this build: `TtSWA` suggests yes (chunks 128); probe first (M1 step 1).
2. Padded `[T,1,4,D]` tile layout of the stream makes memory 8x larger: compact store required for anything beyond ~S=256 x 16 users.
3. MoE cap of 128 tokens/pass (tokens replicated over columns): a column-split dispatch would give up to 8x; needs a kernel/module change (not mine).
4. Positions > 256 in `DSV41StepState` (bf16 integer tables) and `max_comp` = 128 slots in the decode cache (S = 128 ratio 1 already needs 128 + decode growth): need larger `max_comp` or the paged layout.
5. Indexer + sparse attention for > 512 compressed entries: not in M1-M3; needs the KV agent's layout.
6. Router calibration `gate_cutoff` was calibrated on embedding-level activations; prefill uses the same router (exact fp32 path), no new issue.
