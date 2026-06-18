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
ttnn.experimental.indexer_score(q, k, weights, chunk_start_idx) -> score
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
| in `k` | `[1, 1, 56320, 128]` | 1760×4 | TILE, bf16 or bfp8_b |
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
    WRITER (owner of the group's last unit) memsets every group row tail
        [group_valid·32, T) to −inf — disjoint from the rectangle compute emits
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

Implemented as `IndexerScoreProgramConfig` (`ttnn.IndexerScoreProgramConfig`,
SDPAProgramConfig analogue), sizes in elements:

| knob | meaning | default | range / notes |
|---|---|---|---|
| `q_chunk_size` (QC) | tile rows per work unit | 32 (1 tile) | must divide Sq; rows above the diagonal masked with the full −inf tile |
| `k_chunk_size` (KC) | k tiles per unit | 32 (1 tile) | edge units partial (kw < KC); bigger = fewer chunks, less q re-read; perf target 16 bf16 / 32 bfp8 |
| `head_group_size` (HB) | heads resident at once | 1 (always fits L1) | must divide Hi; <Hi streams q blocks per tile — slow but safe; set higher (0 = all) for perf |
| grid / fidelity | cores; HiFi2→LoFi with bfp8 k | full, HiFi4 | host-side only for now |

L1 (1464 KB): `2·Dt·Kt (k) + HB·QC·(Dt+1) (q+w) + QC·Kt fp32 acc + 2·QC·Kt out`.
Defaults (HB=1) stay under 0.5 MB for any model; GLX perf config HB=64
resident: 512K q + 128K w + 256K k + 64K acc + 64K out ≈ 1.0 MB.
512-head model: head_block=64 streamed, Kt=64 — same kernel, no scheme change.

### Head residency in the full model (HB=16, not HB=0)

The V3.2 indexer has `H_idx=64` heads, and `mla.py::_indexer_topk` runs the op
**replicated across TP** — each device builds all 64 heads (`nlp_create_qkv_heads(num_heads=64)`
on the SP-gathered, TP-replicated q) and scores against the full key prefix. At 64 resident
heads, the q+w residency term `HB·QC·(Dt+1)` dominates L1: `HB=0` (all 64) overflows the
1464 KB budget for any `KC≥4` (measured: ~1.89 MB at KC=8). So the full-model default
(`tt/ops.py::INDEXER_FULL_MODEL_CONFIG`) is **QC=2 / KC=8 / HB=16** — heads streamed in 4
groups, which fits. `KC=8` (not the GLM5 `KC=16`) matches the op's matmul-bound DSv32 preset
and is always valid in the model because DSA only runs when `end_pos > index_topk (2048) ≫ 8`
tiles. (Shape unit tests with tiny `Skv` cap `KC` to `Skv/32` via `ops.indexer_program_config(Skv)`.)

#### Alternative (not implemented): TP-shard the heads + all-reduce the logits

The logit is a per-head sum, `logits[s,t] = Σ_h w[s,h]·relu(⟨q[s,h],k[t]⟩)`, and `relu` is
applied per head **before** the sum — so the head-sum is separable across the TP axis. The
op's intended DSv32 deployment exploits this: shard the 64 heads across TP=4 → **16
heads/device**, each device's `indexer_score` emits a *partial* logit (sum over its 16 heads),
and a **TP all-reduce(SUM)** reconstructs the full logit before top-k. The causal `−inf` mask
is head-independent, so it is identical on every partial and survives the sum. At 16
heads/device, `HB=0 / KC=8` fits L1 and per-device indexer compute drops 4× (no replication).

Concretely this is a model-parallelism change to `_indexer_topk`: make `wq_b` column-parallel
(shard its `H_idx·D_idx` output → 16 heads/chip, no reduce since the q-latent is already
replicated), slice this TP-rank's 16 head-weights out of `weights_proj`, keep the **key**
SP-gather (MQA keys are head-independent — every query still needs the full prefix), then
all-reduce the partial logits over `tp_axis`. The trade-off vs. the chosen HB=16 path: it
removes the 4× redundant indexer matmul but adds an all-reduce of the **full logits**
(`[1,1,glob,end_pos]`, ~580 MB at the §7 50k+5k scale) — a large, non-overlapped collective
that may cost more than the matmul it saves; measure before adopting. Caveat: at **TP=1** there
is no head split (still 64 heads), so that path falls back to HB=16 streaming regardless.

## Formats / fidelity

- q/w bf16; `k` accepts bfp8_b (matmul srcA only, never packed). bfp8 k halves
  k BW and lets the factory select LoFi: measured PCC ≥ 0.9997, ~1.18x faster
  than the bf16/HiFi2 default and ~1.66x vs the original HiFi4 (rank0/7
  5.34/5.62 vs 6.32/6.65 vs 8.85/9.31 ms). It is the fastest config.
- DEST bf16 (default): 8-head subblocks in half-sync, PCC ≥ 0.999 holds for
  the 64-head sum; fp32 DEST (`fp32_dest_acc_en`, 4-head subblocks) kept as a
  compile-time fallback if top-k cuts ever need it.
- Fidelity follows k dtype: bf16 k → HiFi2 (PCC ≥ 0.9998, ~1.4x faster than the
  HiFi4 bring-up default); bfp8 k → LoFi (the format carries ~fp8 mantissa, so
  the extra HiFi passes would be wasted). HiFi4 was first-PCC bring-up only.
- Output bf16; w stays bf16 (scales fp32-side in DEST).

## −inf invariant

Skipped tiles never pass through the packer — the writer **memsets the row
tails to −inf** (rank0: 2,990 tiles ≈ 6 MB, <3% runtime; pattern:
`moe_ungroup_rmw_writer.cpp`). Zeros are NOT safe: gates negative. RM partial-
page writes from multiple cores are aligned (32-elem fragment = 64 B ≥ NoC
align). The tail starts at the q-row-group's valid width `group_valid` (not
each row's own width): compute emits the full `[0, group_valid)` rectangle for
every row (future cells of upper rows are full −inf tiles), so the writer fill
is **disjoint** from compute's output and two cores never write the same bytes
when a group's k-units split across cores. Later: `valid_len` contract with
top-k to skip the fill entirely.

## Implementation (functional, in tree)

Files under `ttnn/cpp/ttnn/operations/experimental/indexer_score/device/`:
factory `indexer_score_program_factory.cpp`, kernels
`kernels/{reader,compute,writer}_indexer_score.cpp`.

### Program factory

Validation lives in `IndexerScoreDeviceOperation::validate_on_program_cache_miss`
(shapes, TILE layout, bf16 q/w and bf16-or-bfp8_b k, tile alignment of `Sq, T, D, chunk_start`,
`chunk_start + Sq ≤ T`, knob divisibility, B=1, causal only); the factory only
derives tile dims and the one build-specific subblock constraint. The work unit is
QC q-tile-rows × up-to-KC k-tiles: per q-row-group g,
`valid_max(g) = min(Tt, chunk_t + (g+1)·QC)` and `units(g) = ceil(valid_max/KC)`;
`V = Σ units(g)` is dealt flat across the full grid (`min(V, cores)` workers,
base = V/N, remainder one extra). Per-core runtime args are just
`flat_start, count` + buffer addresses (replaced in
`override_runtime_arguments`); all three kernels invert the flat index locally
with the same loop, so there are no segment tables. Compile args common to all
kernels: `Hi, Sqt, Tt, Dt, chunk_t, QC, KC, HB`; TensorAccessor args appended
for reader (q/k/w) and writer (out, page = T·2 bytes).

ComputeConfig: bf16 DEST, half-sync, math fidelity from k dtype (bf16 → HiFi2,
bfp8_b → LoFi; HiFi4 was bring-up only). The qk subblock height HP comes
from SDPA's `determine_largest_subblock_size(HB, 1, dst_size)`
(`sdpa_subblock_utils.hpp`) with SDPA's `dst_size = fp32 ? 4 : 8` → `{8, 1}`
at 64 heads, passed to compute as CT arg 8. Half-sync lets pack drain one
DEST half while math fills the other; flipping `fp32_dest_acc_en` back on
(factory constant) drops HP to 4 and the qk/mul/acc CBs go fp32.

### CBs

| CB | what | size |
|---|---|---|
| c_0 q | q head-group block `[QC][HB][Dt]` bf16 | ×2 when streaming (HB < Hi) |
| c_1 k | k chunk, double buffered | `2·KC·Dt` bf16/bfp8_b |
| c_2 w | resident w group `[QC][Hi]` | `Hi·QC` bf16 |
| c_3 mask | [diag strict-upper, full] −inf tiles, persistent | 2 bf16 |
| c_24 qk | per-subblock relu(q·kᵀ) | `2·HP` DEST fmt |
| c_25 mul | relu·w contributions | `2·HP` DEST fmt |
| c_26 acc | accumulator ping-pong | `2·QC·KC` DEST fmt |
| c_16 out | untilized output | `2·KC` bf16 |
| c_17 scratch | writer-only −inf source | 1 bf16 |

CB pushes must divide capacity: the k ring pushes a full `KC·Dt` even for
partial edge chunks (kw < KC) — a kw-sized push wraps the ring mid-block and
later linear writes overflow the CB into neighbor L1 (debugged on GLX, where
every group ends in a partial chunk; small T never trips it).

### Kernels (all walk the same flat span)

Kernels share `kernels/indexer_score_common.hpp` (dims at CT args 0–4,
`valid(s)`, `ValidTileSpan` flat-span cursor) and are split into named phase
functions in SDPA style: dataflow uses `Noc`/`CircularBuffer` wrappers; the
diagonal mask and the writer scratch reuse SDPA's
`fill_causal_diagonal_tile_bf16` / `fill_neginf_tile`.

**Reader** — on a new q-row-group pushes the resident w group (`[QC][Hi]`,
id `h·Sqt + s`) and, when all heads fit, the q group (`[QC][Hi][Dt]`, heads
contiguous per row so HP head rows stride Dt for `matmul_block`); per unit
pushes the k chunk (`(c0+c)·Dt + d`). With HB < Hi the q head-group blocks
stream per tile instead. Builds both mask tiles once at start.

**Compute** — per unit, each tile (r, c) completes all head groups before the
next so cb_acc holds one in-flight tile. Phase 1 is a subblock matmul
mirroring SDPA's `matmul_blocks` (heads as rows, M=HB, N=1, K=Dt): one
`matmul_block(rt=HP, ct=1)` per inner-dim step, relu fused before pack →
cb_qk; packs stay subblock-relative (reserve HP / pack / push HP). The
pre-matmul `reconfig_data_format(k, q)` follows the matmul's operand→register
map — `matmul_block(in0=q, in1=k)` sends in1→srcA, in0→srcB — so srcA is set
from k; the reverse order is invisible for bf16 k but reads bfp8 k as bf16. Phase 2 per
HP: `mul_tiles_bcast_cols(qk, w)`, then ping-pong `add_tiles` into cb_acc
(first contribution primes via `copy_tile`) — the L1 round-trip per head is
the known perf TODO. Then the diag mask on `t == chunk_t + s`, full −inf on
`t > diag` (rows above the diagonal in QC>1 groups), `pack_untilize<1,1>` →
cb_out. Pops q/w on group change. Heads are fully core-local; sum order is
the same on every device, so SP ranks agree exactly.

**Writer** — pops unit tiles in (r, c) order; scatters each as 32 row
fragments of 64 B at `page = s·32 + rr`, offset `t·64` (64 B aligned,
NoC-safe). The owner of a group's last unit fills every row tail
`[group_valid·64, Tt·64)` from the −inf scratch (`fill_group_tails`); in-unit
masked tiles come from compute, the tail from the writer, and the two regions
are disjoint so no byte is written twice even when a group splits across cores.

Same scheme for every device in the ring; SP rank enters only via
`chunk_start`. Kernel includes are `api/compute/...` / `api/dataflow/...`;
compute entrypoint `kernel_main()`. Remaining perf knob not wired: packer
l1-acc.

## Validation

Single-chip, `sp_rank ∈ {0,7}` boundary cases via `chunk_start`; −inf maps
exact, visible PCC ≥ 0.999 (measured ≥ 0.9997 even with bfp8 k); negative
weights so padding can't hide. Mini case Sq=64/T=256 first. The test also
sweeps the knobs (QC/KC/HB, incl. the diagonal-mid-group corner), a multi-core
QC=2 shape (so a q-row-group splits across cores), head-count/head-dim
generality, bfp8 k at the production shape, the production absolute-time perf
(both k formats), and the host validation paths (bad dtype/layout,
unaligned/overflow/non-causal/knob rejects).

## Status

Branch: `skrstic/dsa_indexer_score_op_2` (cleanup) on `skrstic/dsa_indexer_score_op`.

- [x] torch reference + test (`tests/nightly/blackhole/sdpa/test_indexer_score.py`)
- [x] skeleton op, validation, RM out spec, nanobind, CMake; empty factory
- [x] design (this doc); visualizations `indexer_score_viz.html`, `indexer_core0_viz.html`
- [x] test dims 55296 → 56320
- [x] program factory: flat valid-tile deal + runtime args
- [x] kernels: reader / compute (8-head DEST passes, fp32 ping-pong acc, diag
      mask) / writer (RM 64 B fragments, −inf tails); mini + GLX rank 0/7 pass
      (PCC ≥ 0.999, exact −inf map)
- [x] kernel refactor: shared `indexer_score_common.hpp` + phase functions,
      SDPA dataflow/compute idioms, mask/scratch from SDPA helpers
- [x] subblock qk matmul: HP from `determine_largest_subblock_size`, half-sync
      pack overlap, relu fused in DEST
- [x] knobs as `IndexerScoreProgramConfig` (q_chunk/k_chunk/head_group), work
      units QC×KC with head streaming; knob tests + GLX KC=16 pass
- [x] perf: HiFi2 default (bf16 k) and bfp8 k + LoFi (fastest, PCC ≥ 0.9997);
      production perf tests (absolute ms, both k formats) alongside accuracy
- [ ] row-major top-k (separate); negative-weights topk-safety test
- [ ] perf: DEST-resident acc, fused relu·w, knob sweep for best GLX config

Out of scope: decode/paged variant, fused score+topk.
