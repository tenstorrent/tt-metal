# DeepSeek-V3.2 Indexer Score Op — Design

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
token, `w` are per-head gates (scalars `Hi^-0.5 · d^-0.5` pre-folded, can be
negative). After the score, top-k(2048) indices feed the sparse MLA.

## The op

```
ttnn.experimental.deepseek.indexer_score(q, k, weights, is_causal, chunk_start_idx) -> score
```

Weighted-ReLU MQA scoring with head-sum accumulation (analogue of DeepGEMM's
`fp8_mqa_logits`). Causality from the scalar `chunk_start_idx` — key `t` is
visible to query `s` iff `t ≤ chunk_start + s`; no mask tensor. Output is
ROW_MAJOR bf16, designed for a row-major top-k written alongside.

## Main case: chunked prefill on Galaxy

5K query chunk attending to 50K history + itself: T = 51200 + 5120 = **56320**
(55K, exactly 1760 tiles — no pad). SP=8 shards queries (640/device), K
all-gathered; indexer heads replicated across TP (all ranks agree on indices).

Per-device dims:

| Tensor | Shape | Tiles | Layout / dtype |
|---|---|---|---|
| in `q` | `[1, 64, 640, 128]` | 64 heads × 20×4 | TILE, bf16 |
| in `k` | `[1, 1, 56320, 128]` | 1760×4 | TILE, bf16 (later bfp8_b) |
| in `weights` | `[1, 64, 640, 1]` | 64 heads × 20×1 | TILE, bf16 |
| out `score` | `[1, 1, 640, 56320]` | 20×1760 | **ROW_MAJOR**, bf16 (~72 MB) |

`chunk_start_idx = 51200 + sp_rank·640`.

## Parallelization: flat deal of causal-valid output tiles

The op is **output-stationary**. Reasons:

- the output (~72 MB) is the only large tensor — write each tile once, never
  spill partials; q (10 MB) and k (14 MB) are cheap to re-read per core;
- **heads are a reduction**, not a parallel dim: any input-side head split
  needs a ~72 MB cross-core score reduction; an output split needs none;
- work per valid output tile is uniform (H·Dt matmuls + relu + ×w), so dealing
  equal tile counts is balanced compute, with no causal pathologies.

The host computes the valid tiles per row in closed form,

```
valid(s)  = min(Tt, chunk_start/32 + s + 1)        # per q-tile-row
V         = Σ_s valid(s)
```

and deals V evenly across cores **in row-major order** — the same flat split
SDPA's program factory uses for global Q chunks. Per-core runtime args reduce
to buffer addresses + `flat_unit_start/count` + `chunk_start`; kernels invert
`valid(s)` locally. Fully-future and pad tiles are never assigned.

Each core's flat span lands as 1–2 **segments**: contiguous (row, t-range)
runs. GLX rank 0: V = 32,210 of 35,200 (91.5%), ~248 tiles per core /130;
rank 7: V = 35,010 (99.5%), ~269. Cores compute their segments left→right;
imbalance ≤ ±1 tile by construction (early-prefill triangles included).

## Per-core loop (one Tensix)

```
for seg {row s, t_first..t_last}:
    READER  q[h-block, s] + w[h-block, s] resident; k chunk (Kt×Dt) double-buffered
    for kc in chunks of Kt:
        zero acc (q_chunk×Kt fp32 CB)
        for hb in head blocks:
            for h: matmul q 1×Dt ⊗ k Dt×Kt → DEST (dest blocks of 8 fp32)
                   SFPU fused relu·w[h]
                   pack l1-acc → acc CB
        if diagonal chunk: add −inf mask tile
        pack_untilize acc → out CB → WRITER streams RM fragments
    WRITER memsets row tail [chunk_start+s+1, T) to −inf if last valid owner
```

DRAM is read once per core (q ~0.5 MB/segment, k ~2 MB, out written once).
Head re-reads are L1/unpack only: k unpacked once per chunk into srcB and
reused across heads; per head q is 4 tiles. Roofline at HiFi2: math ~1472
cyc/head/chunk vs unpack ~704 — **matmul-bound** (~60–76% FPU). Total ~2–2.6
ms on BH 130 cores; tweaks: keep pack off the math path (half-sync 4-tile
blocks or DEST-resident accumulation), fuse relu+scale into one SFPU pass.

Heads may also be stacked into one matmul: 8 heads as tile rows ⊗ same k →
8 DEST tiles, separate per head (sum still at pack-acc). Same matmul count;
just `rt/ct` subblock shape.

## Knobs

| knob | meaning | default | range / notes |
|---|---|---|---|
| `q_chunk` | tile rows per work-unit row | 1 | DEST loops dest blocks, so any size; acc/out CBs scale ×Kt fp32; big values reduce parallel rows (Sqt/q_chunk × colgroups ≥ cores) |
| `k_chunk` (Kt) | k tiles per streamed chunk | 16 bf16 / 32 bfp8 | only acc width + L1; bigger = fewer chunks, less q re-read |
| `head_block` | heads resident at once | H if fits | floor 1 always fits; <H streams q per chunk (traffic = blocks×chunks×slice; grow Kt to amortize) |
| grid / fidelity | cores; HiFi2→LoFi with bfp8 k | full, HiFi2 | |

L1 (1464 KB): `2·Dt·Kt (k) + HB·QC·(Dt+1) (q+w) + QC·Kt fp32 acc + 2·QC·Kt out`.
Defaults: 64h×4 q resident 512K + 128K w + 256K k + 64K acc + 64K out ≈ 1.0 MB.
512-head model: head_block=64 streamed, Kt=64 — same kernel, no scheme change.

## Formats / fidelity

- inputs bf16; `k` as bfp8_b is the biggest lever (halves k BW, Kt→32, LoFi).
- DEST fp32 (`fp32_dest_acc_en`): 64-head sum, feeds discrete top-k cut.
- HiFi2 (reference scores fp8; selection-only). HiFi4 first PCC bring-up only.
- Output bf16; w stays bf16 (scales fp32-side in DEST).

## −inf invariant

Skipped tiles never pass through the packer — the writer **memsets row tails
to −inf** (rank0: 2,990 tiles ≈ 6 MB, <3% runtime; pattern:
`moe_ungroup_rmw_writer.cpp`). Zeros are NOT safe: gates negative. RM partial-
page writes from multiple cores are aligned (32-elem fragment = 64 B ≥ NoC
align). Later: `valid_len` contract with top-k to skip the fill entirely.

## Implementation (functional, in tree)

Files under `ttnn/cpp/ttnn/operations/experimental/deepseek/indexer_score/device/`:
factory `indexer_score_program_factory.cpp`, kernels
`kernels/{reader,compute,writer}_indexer_score.cpp`.

### Program factory

Validates tile alignment (`Sq, T, D, chunk_start % 32 == 0`,
`chunk_start + Sq ≤ T`, `Hi % 8 == 0`, B=1, causal only), computes
`valid(s) = min(Tt, chunk_t + s + 1)` and `V = Σ valid(s)` on host, and deals V
flat across the full grid (`min(V, cores)` workers, base = V/N, remainder one
extra). Per-core runtime args are just `flat_start, count` + buffer addresses
(replaced in `override_runtime_arguments`); all three kernels invert the flat
index locally with the same loop, so there are no segment tables. Compile args
common to all kernels: `Hi, Sqt, Tt, Dt, chunk_t`; TensorAccessor args
appended for reader (q/k/w) and writer (out, page = T·2 bytes).

ComputeConfig: HiFi4, `fp32_dest_acc_en`, `dst_full_sync_en` (8 fp32 DEST
tiles → 8-head passes).

### CBs

| CB | what | size |
|---|---|---|
| c_0 q | resident q row, `Hi·Dt` tiles bf16 | 512 KB @64h |
| c_1 k | k column, double buffered | `2·Dt` bf16 |
| c_2 w | resident w row | `Hi` bf16 |
| c_3 mask | diag strict-upper −inf tile, persistent | 1 bf16 |
| c_24 qk | per-head relu(q·kᵀ) | `Hi` fp32 |
| c_25 mul | relu·w contributions | 8 fp32 |
| c_26 acc | head accumulator ping-pong | 2 fp32 |
| c_16 out | untilized output | 2 bf16 |
| c_17 scratch | writer-only −inf source | 1 bf16 |

### Kernels (all walk the same flat span)

**Reader** — on a new q-tile-row pushes the resident q row (`Hi·Dt` tiles,
id `h·Sqt·Dt + s·Dt + d`) and w row (`h·Sqt + s`); per output tile pushes the
k column (`t·Dt + d`). Builds the mask tile once in L1 (strict upper = 0xFF80,
else 0, face-ordered).

**Compute** — per output tile: phase 1, 8 heads per DEST acquire: `Dt`
`matmul_tiles` accumulate q·kᵀ (transpose set at `mm_init`), `relu_tile`,
pack 8 → cb_qk. Phase 2 per 8: `mul_tiles_bcast_cols(qk, w)` (per-row gate in
col 0), then ping-pong `add_tiles` into cb_acc (first contribution primes via
`copy_tile`). Diagonal (`t == chunk_t + s`): `add_tiles(acc, mask)`. Then
`pack_untilize<1,1>` → cb_out bf16. Pops q/w on row change. Heads are fully
core-local; sum order is the same on every device, so SP ranks agree exactly.

**Writer** — scatters each tile as 32 row fragments of 64 B at
`page = s·32 + r`, offset `t·64` (64 B aligned, NoC-safe). The owner of a
row's last valid tile (`t == valid(s) − 1`, `valid(s) < Tt`) fills the tail
`[valid·64, Tt·64)` of all 32 rows from the −inf scratch — the within-tile
masked part comes from the compute mask, the tail from the writer; no other
core touches those rows.

Same scheme for every device in the ring; SP rank enters only via
`chunk_start`. Kernel includes are `api/compute/...` / `api/dataflow/...`;
compute entrypoint `kernel_main()`. Perf knobs from the design (head_block,
k_chunk, l1-acc, bfp8 k) are not wired yet.

## Validation

Single-chip, `sp_rank ∈ {0,7}` boundary cases via `chunk_start`; −inf maps
exact, visible PCC ≥ 0.999 (0.99 with bfp8 k); negative weights so padding
can't hide. Mini case Sq=64/T=256 first.

## Status

Branch: `skrstic/dsa_indexer_score_op`.

- [x] torch reference + test (`tests/nightly/blackhole/sdpa/test_indexer_score.py`)
- [x] skeleton op, validation, RM out spec, nanobind, CMake; empty factory
- [x] design (this doc); visualizations `indexer_score_viz.html`, `indexer_core0_viz.html`
- [x] test dims 55296 → 56320
- [x] program factory: flat valid-tile deal + runtime args
- [x] kernels: reader / compute (8-head DEST passes, fp32 ping-pong acc, diag
      mask) / writer (RM 64 B fragments, −inf tails); mini + GLX rank 0/7 pass
      (PCC ≥ 0.999, exact −inf map)
- [ ] row-major top-k (separate); negative-weights topk-safety test
- [ ] perf: bfp8 k + LoFi, fused relu·w, pack overlap

Out of scope: decode/paged variant, fused score+topk.
