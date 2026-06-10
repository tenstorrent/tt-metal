# DeepSeek-V3.2 Indexer Score Op — Proposal

## Background (brief)

DeepSeek-V3.2 = V3.1-Terminus + **DeepSeek Sparse Attention (DSA)**. A cheap
"lightning indexer" scores every cached token per query; attention (MLA) then
attends only to the **top-2048** keys. Main attention drops from O(L²) to
O(L·2048); the indexer remains O(L²) but tiny (64 heads, dim 128, single-head
key, ReLU instead of softmax).

Indexer score, per query `s` and key `t`, summed over 64 indexer heads:

```
score[s, t] = Σ_h  relu(q[h, s, :] · k[t, :]) * w[h, s]
```

`q` comes from MLA's query latent, `k` is a single shared 128-dim key per
token, `w` are per-head gates (scalars `Hi^-0.5 · d^-0.5` pre-folded). After
the score, top-k(2048) indices feed the sparse MLA.

## The op we want

The projections, norm, RoPE are covered by existing ttnn ops. The new op is
the scorer:

```
ttnn.experimental.indexer_score(q, k, weights, is_causal, chunk_start_idx) -> score
```

Weighted-ReLU MQA scoring with head-sum accumulation (analogue of DeepGEMM's
`fp8_mqa_logits` on the CUDA side). SDPA-like kernel skeleton, minus
softmax / V / output path.

A **new top-k op is being written alongside** (k=2048, row-major input, uint32
indices) — top-k is not part of this op, but this op's output layout is
designed for it.

## Main case: chunked prefill on Galaxy

5K query chunk attending to 55K context. SP=8 shards queries (640/device),
TP=4 within an SP group, K all-gathered to all devices. Indexer heads are
**replicated** across TP ranks (cheap; avoids ~70 MB score all-reduce; all
ranks deterministically agree on indices, which MLA's head shards require).

Per-device dims (T padded 55K → 55296):

| Tensor | Shape | Layout / dtype |
|---|---|---|
| in `q` | `[1, 64, 640, 128]` | TILE, bf16 |
| in `k` | `[1, 1, 55296, 128]` | TILE, bf16 |
| in `weights` | `[1, 64, 640, 1]` | TILE, bf16 |
| out `score` | `[1, 1, 640, 55296]` | **ROW_MAJOR**, bf16 (~71 MB) |

In tiles: q = 64 heads × 20×4, k = 1728×4, score = 20×1728 before untilize.

## Compute pipeline

Per output tile `[s_tile, t_tile]`, loop over the 64 heads:

1. **`matmul_tiles`** — `q_h[s_tile, 0:4] × kᵀ[0:4, t_tile]`, the 4 inner
   tiles (D=128) accumulate in DEST.
2. **`relu_tile`** — SFPU, in place on DEST.
3. **`mul_tiles_bcast_col`** — per-row scale by `w[h, s]` (column-vector
   broadcast along T; the reason weights are laid out `[1, 64, 640, 1]`).
4. **Accumulate** into an L1 CB across heads (packer L1-accumulate or
   `add_tiles`).

After the last head: `copy_tile` accumulator → DEST, **`pack_untilize`** to
emit row-major. No standalone untilize over 71 MB; downstream top-k consumes
640 independent contiguous rows (~108 KB each — parallelizes trivially,
streams without holding a row in L1).

Loop order: k-chunk outer, heads inner — `k` is the dominant read traffic
(~0.9 GB/device, 64 head passes) and should be reused in L1; q is only ~10 MB.

**Note: pack-ReLU does not fit.** ReLU sits *between* matmul and the weight
multiply, and `relu(x)·w ≠ relu(x·w)` since `weights_proj` outputs can be
negative; the packer can't apply a per-row scalar after its ReLU. Packing
after matmul would add a 64×-per-tile L1 round trip. ReLU stays on SFPU in
DEST.

## Formats / fidelity

- CBs / inputs: bf16 (later: `k` as bfp8_b — biggest bandwidth lever).
- DEST: fp32 (`fp32_dest_acc_en`) — accumulating 64 heads in bf16 loses
  bits, and the scores feed a discrete top-k cut.
- Matmul fidelity: HiFi2 (reference scores in fp8; selection-only output).
  HiFi4 only for first PCC bring-up.
- Output: bf16 (fp32 packed to bf16 at the end; bf16 ±inf survives packing).

## Causality and masking — the `-inf` invariant

Causality from `chunk_start_idx = 50176 + sp_rank·640` — no mask tensor.
Three regions along T per device:

- keys `[0, 50176)`: fully visible, no masking, full compute (~97% of T);
- trailing `[640, 5120]` band: ~20 diagonal tiles get an additive `{0,-inf}`
  mask tile (`add_tiles`); fully-future tiles skip compute entirely;
- pad columns `[55040, 55296)`: skip compute.

**Hazard:** skipped tiles never pass through DEST or the packer, so nothing
writes them — they must be explicitly `-inf`-filled (writer memset of RM row
tails / fill pass), or the RM top-k will select garbage. Zeros are NOT safe:
real scores can be negative (negative gates). Test this with negative
weights so unmasked padding would win top-k.

## Validation plan

Single-chip tests, parametrized by `sp_rank` (SP enters the op only through
`chunk_start_idx`, so each ring position is just a different scalar; no
multi-device needed to validate the op — Galaxy only matters for what
surrounds it: all-gather before, MLA after).

- torch reference + comparison contract: -inf maps must match exactly
  (anything ≤ bf16 lowest counts as masked), visible values by PCC ≥ 0.999
  (element-exact is wrong: head-sum reduction order alone moves values ~1e-5).
- ranks 0 and 7 are the boundary cases: rank 0 maximizes skipped future
  tiles, rank 7 puts the causal diagonal flush against T.
- ttnn bring-up on tile-aligned mini case first (Sq=64, T=256), then GLX.

## Status

Branch: `skrstic/dsa_indexer_score_op`.

- [x] Test + torch reference: `tests/nightly/blackhole/sdpa/test_indexer_score.py`
  (with ring-joint SDPA tests). One test, GLX dims, `sp_rank ∈ {0, 7}`;
  test drives the ttnn op. Reference verified vs einsum and brute force
  per element.
- [x] Skeleton op `ttnn.experimental.deepseek.indexer_score` at
  `ttnn/cpp/ttnn/operations/experimental/deepseek/indexer_score/`: host API,
  device op (`is_causal` + `chunk_start_idx` attrs, input validation, RM bf16
  output spec), empty program factory (no kernels), nanobind + CMake
  registration. Compiles, dispatches at GLX dims, output is garbage — test
  fails PCC as expected and is the bring-up gate.
- [ ] real program factory (reader / compute / writer per pipeline above).
- [ ] new row-major top-k (separate effort).
- [ ] negative-weights topk-safety test once both exist (padding must not win).

## Out of scope (for now)

Decode / paged k-cache variant, fp8/bfp8 inputs, fused score+topk — perf
follow-ups, not needed for functionality.
