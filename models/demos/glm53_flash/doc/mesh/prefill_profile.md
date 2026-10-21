# Long-context prefill chunk: per-op profile and reference swaps on a (2,4) block (2026-10-21, in progress)

The 1 M-token run on the 8x4 (`_logs/mesh/long_8x4_N1048576.log`, 45 layers bfp8, 16 K chunks, composite hybrid
experts, composite mHC, one cache slot) spends 58-72 s per 16 K chunk (229-286 tok/s on 32 chips, ~0.3 % of the compute
roofline), almost all of it prefix-independent per-token work. This document attributes that work by op and by layer
kind on ONE chunk of the same kind of run on the rows-6-7 (2,4) block, and records the swaps of the top items for the
sibling models' reference implementations (kernels and program configs only; the mesh orchestration of
`doc/mesh/README.md` is unchanged: TP4 over the columns, the chunk's rows sequence-parallel over the mesh rows, the
hybrid experts (column c = experts `[72c, 72c+72)`, row r = intermediate slice r), row gather / reduce-scatter and the
column all-reduce around the experts, the replicated single-slot DSA cache).

## Method

`tests/test_mesh_prefill_profile.py` on `TT_VISIBLE_DEVICES=25,24,29,28,21,20,17,16` (`MESH_DEVICE=P150x8`,
`GLM53_MESH_SHAPE=2,4`, the p150_x8 mesh graph descriptor, FABRIC_2D): the resident model of layers 0-21 (3 KDA+dense,
5 DSA+MoE, 14 KDA+MoE; bfp8 experts, ~21 GB per chip), `max_batch=1`, `GLM53_PARTITION_USERS=0` (the 1 M run's
single cache slot). The DSA layers' latent pages + pooled-key index are seeded synthetically to 65 536 positions
(`TtDSA.prefill_state` on realistic-statistics rows, the `test_cache_1m` method; the KDA carries start from zeros,
which the chunk's cost does not depend on), then ONE chunk of 8192 real prompt tokens (the needle prompt of
`test_mesh_long_context`) is prefilled at `start = 65536` through every layer and the head: a compile pass, a free
pass (one `synchronize_device` per layer: the per-layer wall table and the chunk time), and an attributed pass
(every module call bracketed by a device sync; exclusive wall per (layer kind, module)). The per-op device table comes
from a Tracy run of one layer of each kind (`GLM53_PROF_LAYERS=0,3,4`, `GLM53_PROF_TRACY=1`: signposts per layer and
per module, `ReadDeviceProfiler` drains after every layer and every 16 routed-expert blocks) through
`tests/perf_prefill_report.py` (tt-perf-report on the window, every op labelled with its innermost signpost); every
layer of a kind issues the same op sequence, so the per-kind tables compose to the 22-layer chunk.

Per-chip shapes on this block (what a kernel reference must match): KDA 16 heads x K = V = 128, 4096 local rows per
row of the chunk; DSA 16 MLA heads (q 256, latent 512), 8192 gathered rows, 32 index heads; MoE 72 experts per chip
with the 1024-wide intermediate half (`[8192, 4096] x [4096, 2 x 1024]` gate|up, `[.., 1024] x [1024, 4096]` down,
bfp8), top-8 of 288, routed scale 2.5; mHC 4 streams x 4096 fp32 on 4096 local rows.

## Swaps: provenance (per the coordinator's filter: kernels + program configs from Blackhole models at our shape)

| node | our implementation | reference taken | source model / device tuned for | per-chip shape match | status |
|---|---|---|---|---|---|
| KDA prefill | `tt/kda.py` on `deepseek_v3_d_p/tt/kda::ttKDA` (`ttnn.experimental.kda.*` chunk kernels), HiFi2 projections, bf16 gated-norm output, SP grouped scan | the same (reuse map row 1); the qwen38 `chunk_gated_delta_rule` is rejected (scalar per-head decay: PCC 0.96-0.99 vs the per-channel rule) | Kimi-K3 / DSv4 prefill on the 8x4 BH Galaxy, SP8 x TP4, 128-dim heads | 16 heads x 128 per chip: identical | kept; `summary_group_chunks` now sized to the worker grid (a 4 K-row slab = 128 chunks x 16 heads needs 32 chunks per group = 64 owner cores; the old rule asked for 128 of 120) |
| MoE prefill (routed experts) | `tt/moe.py::TtRoutedExperts._block`: one `sparse_matmul` pair per 32-token block (256 blocks x 13 ops per layer at 8 K tokens; the active experts' weights re-streamed per block, the inactive tokens of a block computed) | `deepseek_v3_d_p/tt/moe`: `masked_bincount` -> region offsets -> `deepseek_prefill.dispatch` -> `unified_routed_expert_moe` (ClampedSiluGlu = `swiglu_limit` 10) -> `combine` -> `post_combine_reduce` (reuse map row 11), on a ONE-CHIP dispatch group per column so the hybrid orchestration stays | DeepSeek-V3/V4 and GLM-5.3 (big) prefill on the 8x4 BH torus (EP32 full-width 7168 x 2048 experts, SP x TP) | the kernels take any `[H, I]` per-expert weights; ours are 72 x `[4096, 1024]` (+ `[1024, 4096]`) per chip (half-width slices of the hybrid layout), the dispatch/combine run on the NOC only (group of one) | `GLM53_PREFILL_EXPERTS=dispatch`; device validation pending |
| MoE prefill (fallback) | same | `deepseek_v3` Galaxy prefill form of `sparse_matmul`: all blocks of a group in one call (`[1, G, 32, K] x [1, E, K, N]` under a `[1, G, 1, E]` mask, then `[G, E, 32, K]` under `[1, 1, G, E]`) | DeepSeek-V3 prefill on the Wormhole Galaxy (kernel only; same kernel on Blackhole) | same per-chip shapes as today | `GLM53_PREFILL_EXPERTS=batched`; pending |
| mHC prefill | `deepseek_v3_d_p/tt/mhc::TtMHCWrap` (fused `mhc_split_sinkhorn`, composite apply on packed streams) | the same composite is the reference (row 13); d_p's TP-sharded-stream configuration is orchestration and is not imported | DSv4 prefill on the Galaxy | 4 streams x 4096 fp32 per row: identical | measured first |
| DSA stems / absorb / o_proj | `ttnn.linear` default dispatch (q_a, q_b, kv_a, o_proj), per-head `matmul` absorb | pending the profile: `deepseek_v3/tt/mla/mla1d.py::build_prefill_matmul_program_config` (shape-driven 2D mcast configs) if the stems show up | DeepSeek-V3 MLA prefill (Galaxy) | re-derived for the 12x10 grid and our K/N | pending |

## Baseline: one 8 K chunk at the 64 K prefix, 22 layers + head (rows 6-7 block, 2026-10-21 02:18)

`_logs/agents/prefillprof/J4_profile_22L_baseline.log` (`GLM53_PREFILL_EXPERTS=blocks`, the stage-1 path of the 1 M run).

| pass | chunk wall | tok/s (8 chips) | KDA+dense x3 | KDA+MoE x14 | DSA+MoE x5 | embedding + head |
|---|---|---|---|---|---|---|
| compile (first call) | 110.2 s | 74 | 23.7 s (L0 compiles) | 566 ms / layer | 13.4 s / layer (L3 63.6 s compiles) | - |
| **free** (one sync per layer) | **12.50 s** | **656** | 54 ms / layer | 564 ms / layer (470-707) | 853 ms / layer (822-917) | ~25 ms |
| attributed (sync at every module boundary) | 15.13 s | 542 | 150 ms / layer | 587 ms / layer | 1236 ms / layer | - |

DRAM 22.0 GiB per chip (weights ~21 GB + the 64 K + 8 K cache slot of 5 DSA layers + activations). The free pass is the
chunk time; the attributed pass serializes dispatch at every label (its totals are inflated) and gives the shares.

### Exclusive wall per layer of each kind (attributed pass; `_ModuleTimer`, innermost label)

| kind | routed experts (`MOE_EXPERTS`) | DSA index scorer | DSA other stages | mHC pre + post (2 wraps) | attention (KDA / DSA stems, absorb, sdpa, o_proj) | router | shared expert | collectives | norms | glue |
|---|---|---|---|---|---|---|---|---|---|---|
| KDA+MoE (587 ms) | 473 ms (80 %) | - | - | 44 ms (7.5 %) | KDA 10 ms (1.7 %) | 3 ms | 4 ms | 13 ms | 5 ms | 26 ms |
| DSA+MoE (1236 ms) | 509 ms (41 %) | 246 ms (20 %: 4 sub-blocks of 2048 queries x 18.4 K pools, custom `index_score_decode` rows form) | select / top-k 39, sparse_sdpa 47 (16 heads padded to 32), absorb q 20 + o 18, index queries 19, stems 15, cache fill 5, index maintain 7, kv copy 2.6 (75 MB at 72 K rows), 127 ms unlabelled inside `prefill_forward` (slices, q/o layout conversions, concat) | 51 ms (4.1 %) | o_proj 11 ms | 7 ms | 7 ms | 29 ms | 8 ms | 58 ms |
| KDA+dense (150 ms attributed, 54 ms free) | - | - | - | 48 ms | KDA 16 ms | - | dense MLP 10 ms | 11 ms | 12 ms | 64 ms (sync overhead dominates this kind) |

Reading: the routed experts are **9.2 s of the 12.5 s chunk (73 %)**: 256 x 32-token `sparse_matmul` pairs per MoE
layer (the active experts' weights re-streamed per block, the inactive tokens of a block computed; the stage-1 block
path). The DSA index scorer is 1.2 s (10 %; prefix-dependent, and it runs on both rows over the whole chunk: the
sequence-parallel redundancy slabattn owns), the two mHC wraps 0.9 s (7 %), the KDA sublayer 0.2 s (2 %: the ttKDA
recurrence is not a target, as the reuse map says), the collectives 0.4 s (3 %). The prefix-DEPENDENT part at 64 K
(kv copy + scorer + top-k) is 1.44 s per chunk; the rest is flat in the prefix.

Projection, 1 M tokens in 128 chunks of 8 K on this block (prefix-independent part flat, the prefix-dependent DSA
stages linear in the prefix from the 64 K point): ~2900 s with 22 layers, ~5900 s scaled to 45 layers (the 8x4 run
measured 58-72 s per 16 K chunk with 45 layers on 32 chips = ~4400 s per 1 M; the block numbers are per 8 chips).

## Results of the swaps

### MoE prefill, step 1: the kernels' A/B on real layer-4 weights (`tests/test_moe_prefill_paths.py`)

| path | where | tokens | warmed wall | tok/s | PCC vs HF | note |
|---|---|---|---|---|---|---|
| blocks (stage 1: one sparse_matmul pair per 32-token block) | chip 25 (1x1) | 2048 | 891.6 ms | 2297 | 0.999615 | 64 blocks x 13 ops |
| batched (deepseek_v3 Galaxy form: all blocks of a group in one sparse_matmul call) | chip 25 | 2048 | 890.4 ms | 2300 | 0.999615 (identical) | same device time: the block-sparse kernel re-streams the active experts per block whatever the batching; op-launch gaps are not the cost |
| blocks | rows 6-7 block (2,4) | 4096 (2048 per row, gathered) | 296.8 ms | 13.8 K | 0.999741 | the hybrid layout's per-chip share (72 experts x 1024-wide) |
| dispatch (d_p `deepseek_prefill.dispatch` -> `unified_routed_expert_moe` -> `combine`, every chip its own source along the row axis) | block | 4096 | HUNG | - | - | the fabric dispatch op never returned on any of the 8 chips (FABRIC_2D, cluster_axis 0, group 2, Linear); the SIGTERM at the 1500 s bound left stuck Ethernet cores on chips 25 / 29 (board reset needed; `_logs/agents/prefillprof/J6b_moe_paths_block.log`, `J0_sanity_after_J6b.log`) |

Reading: the `sparse_matmul` block path cannot be batched into efficiency (a 32-token block activates ~64 of the
72 local experts, so every block streams ~8 MB x 64 of weights and computes 32 x 64 expert-rows for ~2 x 32 useful
ones: bandwidth- and compute-wasteful by ~30x). The gain is in the per-expert region form: tokens gathered per
expert, one dense FFN per expert (weights read once, only the routed rows computed), which is what the d_p
`unified_routed_expert_moe` kernel does. The d_p fabric dispatch/combine is the transport the Galaxy prefill
engine uses for its EP32 layout; on this block it hung (not root-caused: no tt-triage capture of the live process).
The region buffer the kernel needs is also buildable WITHOUT fabric inside the hybrid layout (every chip already
holds the gathered tokens): see step 2.

### MoE prefill, step 2: the d_p expert kernel on a fabric-free region buffer (`GLM53_PREFILL_EXPERTS=regions`)

Inside the hybrid orchestration every chip already holds the chunk's gathered tokens, so the region buffer the
`unified_routed_expert_moe` kernel consumes (each local expert's routed rows contiguous, tile-aligned regions,
device-resident counts and offsets) is built with plain ttnn ops and no fabric transport
(`tt/moe.py::TtRoutedExperts._forward_regions`):

| step | op | shape |
|---|---|---|
| local one-hot | `ne(dense_w, 0) x mask` (the chip's 72 experts) | `[T, E]` fp32 |
| counts / ranks | `sum` over T; `cumsum` over T | `[1, E]`, `[T, E]` |
| region offsets | tile-aligned exclusive `cumsum` over E (fp32 integers, exact) | `[1, E]` |
| slot of every pick | `gather` at the router's top-8 ids (non-local picks -> the dummy slot `max_buf - 1`) | `[T, 8]` |
| slot -> token | one `scatter` of the token ids at their slots | `[1, max_buf]` uint32 |
| the regions | `ttnn.embedding(slots, x_rows)` (a row gather) | `[max_buf, 4096]` bf16 ROW_MAJOR |
| experts | `deepseek_prefill.unified_routed_expert_moe` (ClampedSiluGlu, 72 per-expert `[4096,1024]` / `[1024,4096]` bfp8 tensors) | `[max_buf, 4096]` bf8 |
| back to tokens | `ttnn.embedding(pick slots, y_rows)` | `[T, 8, 4096]` |
| weighted sum | `deepseek_prefill.post_combine_reduce` with the routing weights zeroed at the non-local picks (its zero-weight skip) | `[T, 4096]` |

`max_buf = T * 8 + 72 * 32` rows (512 MB at 8 K tokens); the kernel reads only the counted rows of each region.
Weights: the per-expert tensors are sliced once at build from the stacked hybrid tensors, which are then released
(prefill-only for now; a model that also decodes through the block path keeps both: a follow-up). The fabric variant
(`dispatch`, the d_p `dispatch` / `combine` ops along the row axis) stays in the code behind its knob for the EP
layout work but is not used.

**Status: written and unit-tested host-side only** — the single-chip A/B (`J8`) could not open a device (the block's
Ethernet cores are stuck since the hung fabric op, pending the board reset); the chunk-level before/after (J7) and the
Tracy per-op tables (J5) are queued behind the reset. Expected from the kernel's own numbers (d_p: every expert
weight read once, only routed rows computed): the routed-expert share of a MoE layer should fall from ~470-510 ms to
tens of ms per 8 K chunk on this block (useful work 413 GFLOP + 0.96 GB of weights per chip per layer).

## Not done (2026-10-21 03:05)

- Tracy per-op device tables of the chunk (deliverable 1's per-op part): the capture needs the block; the per-module
  attribution above is from the host-synchronized pass. `tests/test_mesh_prefill_profile.py` (`GLM53_PROF_TRACY=1`) and
  `tests/perf_prefill_report.py` are ready to produce them.
- MoE swap: device validation (PCC vs HF and vs the block path) and the chunk re-measure.
- mHC prefill: measured only (0.9 s of the 12.5 s chunk; the composite `TtMHCWrap` IS the reference); no swap.
- DSA stems / absorb (`deepseek_v3` MLA prefill matmul configs): not started; they are ~80 ms of the 1236 ms DSA layer
  (absorb q/o 38 ms, stems 15 ms, o_proj 11 ms), below the index scorer (246 ms) which belongs to the slab-attention
  work.
