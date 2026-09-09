# VSA v0 requirements (functional; 4×8 Galaxy, TP=4/SP=8)

Sparse attention VSA function required for Fast H3.

## R1 — Tile geometry & packing (host)

Video rows are reordered cube-major into (4,4,4) 64-token tiles; text/cond/audio segments are
chopped into 64-token chunks that never cross a segment boundary. Ragged tiles are zero-padded to
64 slots. Per-device sequence length is a multiple of 64 and no tile straddles an SP shard
boundary. Each tile carries metadata: global tile id, valid-token count, and tags {1D | 3D} and
{exempt | candidate}. Tile placement across SP shards is a pluggable map with two implementations:
identity (FastVideo order, default) and work-striped (exempt tiles spread across shards). RoPE
tables, AdaLN row indices, and output unpack maps follow the placement permutation.

**Check:** tile order, valid counts, and tags match a torch port of FastVideo's geometry builders
exactly; packed→unpacked round trip is the identity.

## R2 — KV gather

After QK-norm and RoPE, K and V are all-gathered on the SP axis using the existing
persistent-buffer all-gather, so every device holds full K/V ([1, 14, N, 128]). No ring/compute
overlap.

**Check:** gathered K/V equal the concatenation of all shards' local K/V.

## R3 — Coarse stage (device, unfused ttnn ops)

(a) Pool local Q, K, V per tile via a matmul with a host-built block-diagonal averaging matrix
(entries = 1/valid_count). (b) All-gather pooled K_c, V_c on the SP axis. (c) Per head:
scores = Q_c @ K_cᵀ / √128; O_c = softmax(scores) @ V_c, broadcast tile→64 tokens. (d) Gate: new
`to_gate_compress` linear (ColParallel, 5376→7168, bias-free, zero-init when absent from
checkpoint); final output = out_fine + gate ⊙ O_c; an all-zero gate weight may skip the branch but
must give identical output. (e) Selection: top-k over candidate columns only,
k = max(1, ceil((1−sparsity) · n_candidate_tiles)); every row's index list =
[all exempt tile ids] + [its top-k candidate ids]; exempt-query rows list all tile ids.
Output: uint32 index tensor (shape per R4), sentinel 0xFFFFFFFF. All steps run on device and are
trace-compatible.

**Check:** pooled values, scores, O_c, and index sets match the torch oracle (index sets compared
as sets per row; ties may differ).

**Spike, before dependent work starts:** ttnn.topk at k≈200 over ~1,802 columns; if out of
envelope, fall back to threshold bisection and record the decision.

## R4 — New fine-stage op `vsa_sdpa` (forked from `sparse_sdpa_msa`; MSA, ring, and exp_ring ops unmodified)

**Interface:** q [1, H, S_local, d] and k/v [1, H, T, d], TILE layout, natural head-major order
(no permutes, no row-major); indices [1, H, S_local/64, T/64] uint32 global block ids with
sentinel tails (last dim is the global tile count — required because exempt-query rows are fully
dense and the shape must be static for tracing); block_counts [T/64] uint32 valid tokens per
block; block_size = 64 (the cube size); non-causal only.

**Chunking:** q_chunk_size fixed at 64 tokens (2 tiles, one q-tile = one index row).
k_chunk_size = m × 64 with m ≥ 1 a host-tunable op parameter: the reader gathers the row's next m
listed blocks into one contiguous L1 chunk; compute does one QK matmul and one softmax-rescale per
chunk. A row whose valid block count is not a multiple of m ends with a partial chunk. Per-block
valid-count masking (columns ≥ count → −inf) applies within chunks.

**Semantics, per (head, q-tile):** online-softmax attention of the tile's 64 queries over exactly
the listed blocks; output [1, H, S_local, d], numerically equivalent to dense SDPA under the same
block mask. The kernel contains no SP/mesh logic and no notion of "exempt" — the index list is the
whole contract.

**Check:** matches a torch reference for m ∈ {1, and at least one m > 1}, on synthetic cases
covering ragged blocks, non-uniform row counts (including fully-dense rows), single-block rows,
counts not divisible by m; identical results across m values.

## R5 — Model integration

VSA is a fourth attention path in MiniMaxH3Attention, selected by config (sparsity, tile
placement, k_chunk multiplier); dense paths unchanged and default. Runs traced and untraced at
production shapes on 4×8.

**Check:** config off ⇒ bit-identical to today's model; config on ⇒ traced forward completes at
15 s/768p.

## R6 — Acceptance gates

(a) sparsity = 0 ⇒ output matches the dense ring path (PCC ≥ 0.9995, mirroring existing component
gates). (b) sparsity = 0.9, zero gate ⇒ PCC vs the torch VSA oracle at 15 s/768p (ragged video
tiles) and 1280×768/39 frames (no ragged video tiles). (c) Same with a random nonzero gate weight.
(d) Striped placement ⇒ same outputs as identity placement after unpacking.

## Non-goals (v0)

Ring/AG–compute overlap (in-flight frame-granular ring sparsity in PR #50937 noted; v1 should
evaluate it); fused coarse stage; fp8 KV; intra-device work balancing (dense q-tiles as per-core
stragglers accepted); SP=32/quad; fl2va & ref2va policies (tags exist, nothing more); perf
targets; FastH3 LoRA + 4-step scheduler (separate ticket — prerequisite for quality evaluation;
v0's bar is numerical parity, not video quality).
