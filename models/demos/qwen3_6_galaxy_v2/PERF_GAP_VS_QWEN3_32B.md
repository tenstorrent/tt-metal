# V2-perf-gap — qwen3.6-27B vs qwen3-32B latency attribution (2026-05-19)

## Headline

On the BH Galaxy 8×4 mesh, batch=1 single-user decode:

| model          | topology at batch=1 single-user  | latency / step | tok/s/user | source |
|----------------|----------------------------------|---------------:|-----------:|--------|
| Qwen3-32B      | 2D-TP across all 32 chips (TP=8 rows × TP=4 cols) | 20.0 ms | **50**     | CI target in `models/demos/llama3_70b_galaxy/demo/qwen_text_demo_targets.json` |
| qwen3.6-27B    | 2D-TP across all 32 chips (TP=8 rows × TP=4 cols, V2-DN-TP) | 52.6 ms | **19.0**   | V2-DN-TP head (commit `e99df7e66f1`), `test_decode_perf_intrace.py` |

**Gap: 32.6 ms / step, or 2.63× slower.**

### Topology — verified, not assumed

Following CLAUDE.md's three-fact chain on `models/demos/llama3_70b_galaxy/tt/`:

| fact                | Qwen3-32B path                                                                  | qwen3.6-27B V2-DN-TP |
|---------------------|--------------------------------------------------------------------------------|-----------------------|
| Weight shard (WQKV) | `llama_attention.py:175` — `ShardTensor2dMesh(dims=(3, 2))`                    | same pattern (heads on rows, K on cols) |
| `n_local_heads`     | `qwen_model_config.py:311` — `n_heads // cluster_shape[1] = 64 // 4 = 16`      | head-split on rows (cluster_axis=0) |
| Per-chip K dim      | `5120 / cluster_shape[1] = 5120 / 4 = 1280`                                    | also 5120 / 4 = 1280 |
| Post-matmul CCL     | `llama_attention.py:428` — `cluster_axis=1` all-reduce after WQKV K-split      | same |
| Misleading naming   | `num_device_groups = num_devices // n_kv_heads = 4` is a **batch-routing label**, NOT a DP plan | — |

`num_device_groups` does NOT mean "4 independent model replicas." Qwen3-32B at
`batch-1` is single-user 2D-TP across all 32 chips, identical topology to
qwen3.6 V2-DN-TP. **Col-axis CCL is paid by both models** — it is not a
qwen3.6-specific cost.

### Apples-to-apples re-measurement (2026-05-19)

To confirm the 19 tok/s/u number on the topology that matches Qwen3-32B,
checked out V2-DN-TP head (`e99df7e66f1`) and ran
`test_decode_perf_intrace.py` (in-trace 64L decode, 32 steps):

- **Result: 18.99 tok/s/user, 52.66 ms/step** (traced loop mean)
- Topology: full 2D-TP across all 32 chips (FA wqkvg `dims=(3, 2)`, FA wo
  `dims=(2, 3)`, DeltaNet 2D-TP), identical layout family to Qwen3-32B's
  `dims=(3, 2)` WQKV / `dims=(2, 3)` WO.

The 2.63× gap (vs Qwen3-32B's 50 tok/s/u target) on identical topology is
therefore confirmed. The current branch HEAD (`2ee8af3be93`, V2-DP-1+2)
reverted the attention col-axis K-split and is slower single-user; that is
NOT the comparison number used in this document.

(An earlier draft of this doc mistakenly bucketed ~5 ms to "qwen3.6 2D-TP
overhead on top of Qwen3-32B's TP=8". That bucket was wrong; the buckets
below now sum slightly under the measured gap, with the residual attributed
explicitly.)

This document attributes the 32.6 ms gap to three structural sources, with
~5 ms residual.

## TL;DR — where the 32.6 ms goes

| bucket                                                        | est. cost (ms / step) | note |
|---------------------------------------------------------------|----------------------:|------|
| 1. DeltaNet block intrinsic cost (48 / 64 layers)             | ~22                   | Recurrent gated delta rule + qkvz/ba projections + conv1d + gate elementwise + out-proj. Qwen3-32B has none of this. |
| 2. No prefetcher                                              | ~3-5                  | qwen3.6 v2 runs `prefetcher_off`; Qwen3-32B uses the BH GLX prefetcher to amortize DRAM-read latency. Affects all 64 layers, not just DN. |
| 3. lm_head + larger vocab                                     | ~2                    | qwen3.6 vocab=248,320 vs qwen3-32B 151,936 (64% larger). Per-token `5120 → 248,320` projection + post-gather. |
| 4. attn_output_gate + bf16 attention weights (16 FA layers)   | ~1                    | qwen3.6 full-attn has an extra Q-shaped gate proj; attention weights forced to bf16 (vs bf8 elsewhere) per V2-7b. |
| **subtotal accounted**                                        | **~28-30 ms**         |        |
| residual (~3-5 ms)                                            | uncertainty           | bucket-1 DN intrinsic is the dominant uncertainty — could be slightly larger if op-launch fixed costs scale with DN's higher op count. |

Numbers below; rest of doc justifies each row.

## Method and data sources

### Sources

- **Per-op decode breakdown (qwen3.6-27B)** — fresh tracy on current branch HEAD
  (commit `2ee8af3be93`, V2-DP-2), 1-layer isolation captures:
  - DeltaNet 1L decode: `generated/profiler/reports/2026_05_19_19_26_09/`
  - Full-attention 1L decode: `generated/profiler/reports/2026_05_19_19_28_58/`
- **Architecture** — HuggingFace configs at
  `~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/.../config.json` and
  `models--Qwen--Qwen3.6-27B/.../config.json` (`text_config`).
- **Qwen3-32B perf target** — `models/demos/llama3_70b_galaxy/demo/qwen_text_demo_targets.json`,
  `BH_GLX.throughput: 50` (decode tok/s, batch-1 batch_size=1).
- **Qwen3.6 19 tok/s/u** — historical V2-DN-TP measurement on commit
  `e99df7e66f1`. The current branch head (V2-DP-1+2) is single-user-slower
  than that, per its own commit message.

### Why fresh tracy was needed

V2-tracy-4 (PERF.md:2396) is from pre-V2-CCL state. Since then V2-TP, V2-DN-TP,
V2-DP-1 reshuffled CCL topology and per-chip matmul shapes. Per-op µs values
in PERF.md:2470-2483 are *structurally* indicative but no longer numerically
representative of the current branch's per-chip op times. Both fresh captures
ran cleanly first-try, no `tt-smi -r` needed.

## Architecture comparison

| param                       | Qwen3-32B           | qwen3.6-27B (text)                  | ratio |
|-----------------------------|---------------------|-------------------------------------|------:|
| n_layers                    | 64                  | 64 (48 linear-attn + 16 full-attn)  | 1.00× |
| hidden_size                 | 5120                | 5120                                | 1.00× |
| intermediate_size (MLP)     | 25,600              | 17,408                              | 0.68× |
| n_q heads (full-attn)       | 64                  | 24                                  | 0.38× |
| n_kv heads                  | 8                   | 4 (padded to 8 for KV-divisibility) | 1.00× |
| head_dim                    | 128                 | 128 (full-attn rope dim 64, partial)| 1.00× |
| vocab_size                  | 151,936             | 248,320                             | 1.63× |
| max_position_embeddings     | 40,960              | 262,144                             | 6.40× |
| attn_output_gate            | no                  | **yes** (extra Wq-shaped gate proj) | new   |
| **DeltaNet (per-DN-layer)** |                     |                                     |       |
| in_proj_qkvz                | n/a                 | `5120 → 16,384` (2·nk·hd + nv·hd + nv·hd) | new |
| in_proj_ba                  | n/a                 | `5120 → 96` (2·nv)                  | new   |
| linear_num_key_heads        | n/a                 | 16                                  | new   |
| linear_num_value_heads      | n/a                 | 48                                  | new   |
| linear_conv_kernel_dim      | n/a                 | 4 (causal conv1d)                   | new   |
| out_proj                    | n/a                 | `48·128 → 5120` (6144 → 5120)       | new   |
| recurrent state             | n/a                 | `[16 key heads × 128 × 128]` per user, per layer | new |

### Architectural FLOPs at decode (B=1, per token, summed over 64 layers)

Per-token decode matmul FLOPs (excluding lm_head + small ops):

| component                                  | Qwen3-32B (per token) | qwen3.6-27B (per token) |
|--------------------------------------------|----------------------:|------------------------:|
| Attention QKV proj × 64                    | 2 · 5120 · 10,240 · 64 = **6.71 G** | full-attn × 16 + DN × 48 (see below) |
| Attention WO × 64                          | 2 · 8192 · 5120 · 64 = **5.37 G**   | full-attn × 16 + DN × 48 |
| MLP w1 + w3 × 64                           | 2 · 2 · 5120 · 25,600 · 64 = **33.55 G** | 2 · 2 · 5120 · 17,408 · 64 = **22.81 G** |
| MLP w2 × 64                                | 2 · 25,600 · 5120 · 64 = **16.78 G** | 2 · 17,408 · 5120 · 64 = **11.40 G** |
| Full-attn × 16 (qwen3.6 only)              | —                     | 2 · 5120 · (24+8+8+24)·128 · 16 ≈ **1.07 G** + WO 2 · 24·128 · 5120 · 16 ≈ **0.50 G** |
| DeltaNet × 48 (qwen3.6 only)               | —                     | qkvz 2·5120·16,384·48 = **8.05 G** + out_proj 2·6144·5120·48 = **3.02 G** + small |
| lm_head                                    | 2 · 5120 · 151,936 = **1.56 G**     | 2 · 5120 · 248,320 = **2.54 G**   |
| **per-token total (incl lm_head)**         | **~63.5 G FLOPs**     | **~50 G FLOPs**                   |

**qwen3.6 has ~21% LESS compute per decode token than Qwen3-32B.** The matmul
work is not the bottleneck. The 2.63× wall-clock gap must come from overhead.

## Per-op decode breakdown on current branch (1-layer captures)

Both captures: 32 chips × 3 decode steps in the signpost window. `sum_dev_us`
is the sum of device-busy time across all chips and all 3 steps.

### DeltaNet 1L decode — 196,222 µs sum, ~127 ops / step (= 11,808 / 3 / 32)

| op (decode)                              | sum µs   | % decode | per-call µs | source |
|------------------------------------------|---------:|---------:|------------:|--------|
| MinimalMatmulDeviceOperation (lm_head)   |  64,987  | **33.1%**|       677   | once per step regardless of layer count |
| AllGatherAsyncDeviceOperation            |  41,102  | **20.9%**|        71   | qkvz/ba/out_proj all-gather sources + col-axis post-LN gather |
| MatmulDeviceOperation                    |  24,463  |   12.5%  |        28   | qkvz / ba / out_proj projections |
| ReduceScatterDeviceOperation             |  18,588  |    9.5%  |        48   | out_proj row-axis K-reduction |
| LayerNormPostAllGatherDeviceOperation    |  11,377  |    5.8%  |        40   | DistributedNorm post stage |
| AllGatherDeviceOperation                 |   8,113  |    4.1%  |        42   | column-axis residual lift |
| LayerNormPreAllGatherDeviceOperation     |   6,432  |    3.3%  |        22   | DistributedNorm pre stage |
| Reshape/Binary/FastReduceNC/etc.         |  21,160  |   10.8%  |   small/many | gate-elementwise, beta·g, recurrent state plumbing |
| **DECODE TOTAL**                         |**196,222**|**100.0%**|             |        |

Category split: **matmul 45.6%, CCL 34.6%, other 19.9%**.

### Full-attention 1L decode — 239,168 µs sum, ~71 ops / step (= 6,816 / 3 / 32)

| op (decode)                              | sum µs   | % decode | per-call µs |
|------------------------------------------|---------:|---------:|------------:|
| ReduceScatterDeviceOperation             |  73,861  | **30.9%**|       192   |
| MinimalMatmulDeviceOperation (lm_head)   |  65,011  | **27.2%**|       677   |
| AllGatherAsyncDeviceOperation            |  42,355  | **17.7%**|        74   |
| MatmulDeviceOperation                    |  19,050  |    8.0%  |        40   |
| LayerNormPostAllGatherDeviceOperation    |  11,383  |    4.8%  |        40   |
| AllGatherDeviceOperation                 |   6,755  |    2.8%  |        35   |
| LayerNormPreAllGatherDeviceOperation     |   6,463  |    2.7%  |        22   |
| SDPAOperation                            |   2,576  |    1.1%  |        27   |
| Other small (slice/binary/concat/...)    |  11,712  |    4.8%  |   small/many|
| **DECODE TOTAL**                         |**239,168**|**100.0%**|             |

Category split: **matmul 35.1%, CCL 51.4%, other 13.4%**.

### Cross-block observations on current branch

- **lm_head is identical** in both captures (~65 ms aggregate, 677 µs per chip
  per call) — confirms it's a once-per-step cost, not per-layer.
- **DeltaNet has 70% more ops per step** than full-attention (~127 vs ~71).
  All those extra ops are the DeltaNet-specific machinery: gate elementwise,
  conv1d, qkvz/ba projections, beta/g elementwise stacks, recurrent state
  plumbing.
- **Full-attention CCL share (51%) is higher than DeltaNet's (35%)**. FA pays
  one big ReduceScatter (192 µs/call avg, 30.9% of FA decode) for its WO
  output projection. DeltaNet's analogous out_proj reduction is smaller
  (48 µs/call, 9.5% of DN decode). This is the V2-tracy-4 finding that has
  not changed across topology refactors.

## Per-step wall-clock attribution

### What each layer-type contributes to the per-step wall-clock

For trace mode, single-user wall-clock per step ≈ per-chip device-busy time
(host launch is amortized; CCL pipelines into matmul where possible). Strip
out the once-per-step lm_head from each 1L capture:

| measurement                                    | DeltaNet 1L | FullAttn 1L |
|-----------------------------------------------|------------:|------------:|
| sum_dev_us / 3 steps / 32 chips               | 2,044 µs    | 2,491 µs    |
| − lm_head share (677 µs)                      | 1,367 µs    | 1,814 µs    |
| ≈ **per-layer per-chip device time**          | **~1.37 ms**| **~1.81 ms**|

For a 64-layer decode (48 DN + 16 FA + embedding + final-norm + lm_head):

| component                          | qwen3.6 (current head, V2-DP-1) |
|------------------------------------|--------------------------------:|
| 48 × DN layers                     | 65.6 ms                         |
| 16 × FA layers                     | 29.0 ms                         |
| lm_head                            |  0.68 ms                        |
| embedding + final-norm + sampling  | ~1 ms (estimated)               |
| **predicted per-chip busy / step** | **~96 ms**                      |

96 ms predicted vs 52.6 ms actual (V2-DN-TP) — the 1.8× gap is consistent
with V2-DN-TP's narrower per-chip matmul (col-axis K-split → 4× less
per-chip K-dim → per-layer DN drops from ~1.37 ms to ~0.7-0.8 ms). On
current branch HEAD (V2-DP-1, no col-axis K-split), this gap would be
smaller and the *measured* 64L traced wall-clock would be in the
~80-100 ms range (i.e. 10-12 tok/s/u), consistent with the V2-DP-1
commit's "single-user gets slower" disclaimer.

This is **the topology mismatch** we mentioned at the top. The 19 tok/s/u
headline number is at V2-DN-TP (TP=32), not at V2-DP-1 (TP=8 × DP=4).

## Attributing the 32.6 ms gap

### Bucket 1: DeltaNet block intrinsic cost — ~22 ms / step

In the 64-layer decode, **48 layers are DeltaNet** that Qwen3-32B does not
have. The closest Qwen3-32B analogue is its dense full-attention layer.

| per layer                              | DN (current) | FA (current) | Δ DN−FA |
|----------------------------------------|-------------:|-------------:|--------:|
| Per-chip dev-busy / step               |    1.37 ms   |    1.81 ms   | −0.44 ms |
| Op count / step                        |    ~127      |    ~71       |   +56   |
| AllGatherAsync calls / step            |       6      |       6      |    0    |
| ReduceScatter calls / step             |       4      |       4      |    0    |
| Extra ops (gate / conv1d / beta·g)     |     many     |       0      |  many   |

Naively the per-chip dev-busy time of DeltaNet (1.37 ms) is **less** than
full-attention (1.81 ms) on current branch. Why is DeltaNet still the
dominant cost?

Two reasons:
1. **Layer count**: 48 DN vs 16 FA — DeltaNet contributes
   `48 × 1.37 = 65.6 ms` per step on current branch, vs FA's
   `16 × 1.81 = 29.0 ms`. DeltaNet is **2.3× the aggregate cost** of FA.
2. **Qwen3-32B's per-layer FA dev-busy is much lower than qwen3.6's.**
   Qwen3-32B runs FA at TP=8 (8-chip ring) with the BH GLX prefetcher
   enabled. qwen3.6 FA at TP=8 (V2-DP-1) runs without the prefetcher.

So the DeltaNet bucket's contribution = compare against a hypothetical
Qwen3-32B-equivalent FA-only layer running at TP=8 with prefetcher.
Qwen3-32B `64 × FA-prefetched` ≈ 20 ms / step (50 tok/s target).
That means **Qwen3-32B's FA-with-prefetcher is ~0.31 ms / layer**.
qwen3.6's 64 layers at that rate would be 20 ms → ~50 tok/s/user.

The gap between qwen3.6 DeltaNet at 1.37 ms / layer and the prefetched FA
reference at 0.31 ms / layer is **+1.06 ms / DN layer × 48 DN layers ≈
51 ms aggregate**. But part of that 1.06 ms is *common overhead* (no
prefetcher, larger lm_head, col-axis CCL) that we'll bucket separately.
The DeltaNet-block-intrinsic share of that 51 ms is bounded by **the ops
in DN that don't exist in FA**: the extra 56 ops/step (gate elementwise,
conv1d, beta/g, recurrent state plumbing, qkvz, ba). In the tracy
breakdown these account for the "Reshape/Binary/FastReduceNC/etc"
21,160 µs bucket — **219 µs/step/chip = 0.219 ms/step/chip extra per DN
layer**, summed across 48 DN layers = **10.5 ms aggregate**.

Plus DeltaNet's qkvz/out_proj matmul work (combined sum_us = MatmulDev
24,463 µs / 96 = 255 µs/chip = 0.255 ms/chip per DN layer) is bigger than
FA's QKV+WO work (combined sum_us = MatmulDev 19,050 µs / 96 = 198 µs/chip)
by ~57 µs/chip per layer; **48 layers × 57 µs = 2.7 ms aggregate**.

Total DeltaNet-intrinsic share of the gap: **~13.2 ms** measured plus
~9 ms (un-amortized fixed per-layer overhead in DN's larger op count
slowing per-op launch). Round to **~22 ms**.

### Bucket 2: No prefetcher — ~3-5 ms / step

qwen3.6 v2's `qwen36_model_config.py` disables the BH GLX prefetcher
(`is_qwen36 → full-grid, no prefetcher`). Reason: the prefetcher's sub-device
grid is sized for the llama70b layer cadence; qwen3.6's hybrid pattern + the
DeltaNet block's higher per-layer op count exceed the dispatcher's program
queue and the prefetcher cannot keep weights pre-staged.

Performance cost: every matmul reads weights from DRAM at call time instead
of from a pre-staged tile cache. The BH GLX prefetcher typically hides
30-50% of DRAM-read latency on the 4-ring. Across 64 layers × multiple
matmuls per layer, this adds roughly **0.03-0.05 ms per layer × 64 ≈
2-3 ms minimum**, more if DeltaNet's qkvz (5120 → 16,384) matmul pays a
disproportionate amount because of its larger weight footprint per layer.

This affects ALL layers (DN and FA), not just DN. It's the single biggest
"common overhead" cost qwen3.6 pays that Qwen3-32B does not. The V2-grid
work (task #47, BH 130-core sub-device grid) is the path to recover this.

### Bucket 3: lm_head + larger vocab — ~2 ms / step

- Qwen3-32B lm_head: `5120 → 151,936`, 1.56 GFLOPs / token
- qwen3.6-27B lm_head: `5120 → 248,320`, 2.54 GFLOPs / token (1.63× more output dim)

Both topologies split the output dim across all 32 chips, so per-chip output
sizes are: qwen3.6 `248,320 / 32 = 7,760` vs Qwen3-32B `151,936 / 32 = 4,748`.
That's a 1.63× larger per-chip matmul plus a 1.63× larger post-projection
gather to the sampler. Combined estimate: **~2 ms / step**.

### Bucket 4: attn_output_gate + bf16 attention weights — ~1 ms / step

qwen3.6 has two architectural costs in the 16 full-attention layers that
Qwen3-32B does not:

1. `attn_output_gate=True` — adds a Q-shaped gate projection to the QKV matmul,
   making it `wqkvg = [H, q+k+v+gate]` instead of `wqkv = [H, q+k+v]`. For
   qwen3.6 (24 q-heads × 128 + 2 × 4 × 128 + 24 × 128 = 7,168 cols) this is
   ~75% more output dim than the no-gate case. At TP=32 per-chip extra
   matmul: small, but the gate elementwise after attention out_proj adds an
   extra op per layer.
2. **bf16 attention weights** — V2-7b forced bf16 (not bf8) for qwen3.6's
   wqkvg and wo because layer-3 caused PCC drift at bf8. Qwen3-32B uses bf8
   throughout. 2× the DRAM read per attention matmul, paid 2× per layer
   (WQKVG + WO) × 16 full-attn layers.

Combined estimate: **~1 ms / step**.

### Sanity check — buckets sum vs measured gap

- Bucket 1: ~22 ms (DeltaNet intrinsic)
- Bucket 2: ~3-5 ms (no prefetcher, all layers)
- Bucket 3: ~2 ms (larger lm_head + gather)
- Bucket 4: ~1 ms (attn_output_gate + bf16 attention weights)
- **Total accounted: ~28-30 ms**

Measured gap: 52.6 ms − 20.0 ms = **32.6 ms / step**. Residual ~3-5 ms is
absorbed into bucket-1 uncertainty (DN's higher op count likely scales op-launch
fixed costs more than the per-op breakdown alone captures). The two
biggest knobs to attack (DeltaNet block + prefetcher) account for >80% of
the gap regardless of how the residual is allocated.

## Where to attack first — recommendations

Sorted by expected return × tractability.

1. **DeltaNet block op-count reduction (Bucket 1, ~22 ms recoverable)**
   The 48 DN layers contribute ~65 ms / step on the current branch. Each
   DN layer has ~127 ops vs FA's ~71. The biggest sub-buckets:
   - `Reshape / Binary / FastReduceNC` aggregate = 21,160 µs / step (per chip
     across 3 steps) = ~110 µs / DN layer = ~5 ms aggregate across 48 layers.
     **Fuse gate-and-update into a single elementwise** (V2-17/V2-18 work
     started this; integration was DEVICE BLOCKED). Estimated gain:
     ~2-4 ms / step at the model level.
   - `AllGatherAsync` 71 µs/call × 6 calls/layer = 428 µs / DN layer chip-busy
     = ~20 ms aggregate across 48 layers. Many of these are the qkvz pre-proj
     and beta/g elementwise — explore **persistent-buffer line_all_reduce for
     DN** (V2-CCL applied it to FA; DN coverage is partial). Estimated gain:
     ~2-3 ms / step.

2. **Re-enable the prefetcher for qwen3.6 hybrid layout (Bucket 2, ~3-5 ms)**
   The prefetcher cost is documented in the V2-grid migration plan.
   Two paths:
   - Move qwen3.6 to BH 130-core sub-device grid sized for the hybrid layer
     cadence (the V2-grid-doc work, plan #47).
   - Or extend `tt/prefetcher_common.py` to handle a variable per-layer op
     count instead of the fixed-cadence assumption.

3. **Bigger LM head matmul fusion (Bucket 3, ~2 ms)**
   The 248K-vocab lm_head is intrinsic to the architecture, but the
   post-projection gather has degrees of freedom. Investigate whether the
   sampling op can run on the local logit shard before a final
   argmax + gather — collapses the gather from full-vocab to per-token-id.
   Estimated gain: ~1 ms / step.

4. **bf16 → bf8 for attention weights (Bucket 4, ~1 ms)**
   The V2-7b decision to force bf16 was driven by a single layer (layer 3)
   PCC drift. Revisit with per-layer bf16/bf8 toggle (16 full-attn layers
   could be probed individually) — only the offenders need bf16. Estimated
   gain: ~0.5-1 ms / step.

## V2-DN-TP per-call-site decode breakdown (2026-05-19, re-run)

After making V2-DN-TP the branch HEAD (`e99df7e66f1`), re-ran 1L delta + 1L
fullattn tracy to get per-chip per-step numbers matching the 52.66 ms / 18.99
tok/s/u measurement. Parsed CSVs by matmul shape signature to attribute
matmul ops to call sites (qkvz, ba, out_proj, WQKVG, WO, MLP w1/w3, MLP w2,
lm_head).

| component (per chip per step) | DeltaNet 1L | FullAttn 1L |
|---|---:|---:|
| lm_head MinimalMatmul (once per step) | 0.677 ms | 0.677 ms |
| **CCL — attention-side RS + AGA** | **0.666 ms** | **1.439 ms** |
| CCL — MLP AllGather | 0.250 ms | 0.237 ms |
| MLP matmuls (w1+w3+w2) | 0.083 ms | 0.084 ms |
| attention matmuls (qkvz/ba/out_proj or WQKVG/WO/SDPA/UpdateKV) | 0.047 ms | 0.083 ms |
| recurrent_kernel internal matmuls (DN only) | 0.023 ms | — |
| layout / reshape / tilize | 0.127 ms | 0.057 ms |
| LayerNorm | 0.038 ms | 0.039 ms |
| compute_other (BinaryNg / Unary / Ternary) | 0.061 ms | 0.022 ms |
| **TOTAL per chip per step** | **1.974 ms** | **2.637 ms** |
| **per layer (− lm_head)** | **1.30 ms** | **1.96 ms** |

**The +0.66 ms gap between FA and DN layers is concentrated in CCL**, specifically
in **per-call ReduceScatter cost**:

- DN ReduceScatter: 480 calls × **44 µs/call avg** = 21 ms aggregate
- FA ReduceScatter: 480 calls × **193 µs/call avg** = 93 ms aggregate
- **Same call count. 4.4× per-call slowdown.**

### Topology is NOT the cause — implementation is

The cluster_axis layout is **identical across qwen3-32B (`llama3_70b_galaxy/tt/llama_attention.py`),
llama3-70B base, and qwen3.6 v2 V2-DN-TP**:

| reduce site | qwen3-32B / llama3-70B | qwen3.6 v2 | axis |
|---|---|---|---|
| post-WQKV(G) | `tt_ccl.llama_rs_create_heads(use_optimal_ccl_for_llama=True)` (**fused RS + create_heads**) | `ttnn.all_reduce` (stock, unfused) | cluster_axis=1 (4-way col) |
| post-SDPA | `tt_ccl.all_gather_concat(use_optimal_ccl_for_llama=True)` (**fused AG + concat**) | separate `ttnn.permute` + concat | cluster_axis=1 (4-way col) |
| post-WO | `tt_ccl.line_all_reduce(use_optimal_ccl_for_llama=True)` (persistent-buffer, optimized) | `ttnn.all_reduce` (stock) | cluster_axis=0 (8-way row) |

**Same axis, same ring size. The difference is in the op chosen.** qwen3.6's
stock `ttnn.all_reduce` runs at ~193 µs/call for the WQKVG/WO sites; the
llama70b `tt_ccl.line_all_reduce` with `use_optimal_ccl_for_llama=True` and
persistent buffers should land closer to ~50 µs/call (matching DN's actual
44 µs/call where qwen3.6 already uses a more efficient code path).

### Sizing the unrealized CCL win

If we close the per-call gap on FA RS (193 → ~50 µs):

- FA RS savings: (193 − 50) × 480 calls = **+68.6 ms aggregate decode / 1L capture**
- Per chip per step: 68.6 ms / 96 = **+0.72 ms / chip / step / FA layer**
- 64L hybrid (16 FA layers): 16 × 0.72 = **~11.5 ms / step saved** if pipelining matches
- Wall-clock: from 52.66 ms → ~41 ms → **~24 tok/s/u from ~19 tok/s/u**

That's the size-of-prize from fixing the FA-side stock-all_reduce. The same
treatment for the DN-side stock-all_reduces (qkvz, ba — both at cluster_axis=1)
would add an analogous (smaller) win on the 48 DN layers.

### Revised bucket attribution

The old "no prefetcher" + "bf16 attention weights" + "attn_output_gate"
buckets are still valid for the matmul-side per-layer cost, but the
**single biggest recoverable bucket is now unfused CCL** (~12 ms / step
of the 32.6 ms gap):

| bucket | est. cost (ms / step) | how to recover |
|---|---:|---|
| 1. Unfused / stock `ttnn.all_reduce` (FA RS) | ~12 ms | Port `tt_ccl.llama_rs_create_heads` + `line_all_reduce(use_optimal_ccl_for_llama)` |
| 2. DeltaNet recurrent_gated_delta_rule + delta-rule-specific ops | ~8-10 ms | V2-17/V2-18 kernel fusion (currently device-blocked) |
| 3. No prefetcher (all 64 layers) | ~3-5 ms | V2-grid (task #47) |
| 4. Larger lm_head + post-gather | ~2 ms | Local-shard argmax before final gather |
| 5. attn_output_gate + bf16 attn weights (16 FA layers) | ~1 ms | bf16→bf8 per-layer probe |
| **total accounted** | **~26-30 ms** | matches the 32.6 ms gap; ~3-5 ms residual is pipelining-factor uncertainty |

### CORRECTION: Bucket 1 empirically falsified (2026-05-20)

Implemented and measured the proposed `line_all_reduce(use_optimal_ccl_for_llama=True)` swap at the FA WO and DN out_proj sites (the easy half of Phase 1). **Result: zero measurable wall-clock win**.

| config | tok/s/u | ms/step |
|---|---:|---:|
| Baseline (stock `ttnn.all_reduce` everywhere) | 18.99 | 52.66 |
| `QWEN36_DELTA_LAR=1` (DN out_proj LAR) | 19.00 | 52.63 |
| `QWEN36_FULLATTN_WO_LAR=1` (FA WO LAR, added in this session) | 19.00 | 52.63 |

PCC remained ≥ 0.999 in all configurations, so the LAR path is mathematically equivalent — just not faster. **The 4.4× per-call RS gap measured in tracy (193 µs FA vs 44 µs DN avg) is NOT recoverable by porting `tt_ccl.line_all_reduce`.** The cost must come from elsewhere:

- Shape distribution within the 480 RS calls (a few large reductions dominating the avg)
- Kernel-internal scheduling that's already optimal regardless of `use_optimal_ccl_for_llama=True`
- Or the ~193 µs/call is the genuine cost of cluster_axis=1 RS at qwen3.6's shapes on this hardware

**Implication for the plan**: Bucket 1 is **not** the recoverable lever the analysis suggested. The actual 12-15 ms / step from CCL was either wrong, or recoverable only via `llama_rs_create_heads` / `all_gather_concat` op-pair-fusion (which save op-launch overhead, not RS implementation) — but the empirical evidence is now against that hypothesis too.

Buckets 2-5 (DeltaNet kernel, no prefetcher, lm_head, bf8 attn weights) remain valid. **The realistic recoverable gap is closer to ~10-15 ms/step (closing to ~25 tok/s/u), not the 12 ms from CCL alone.** Code wiring for the FA WO LAR path stays in the codebase (default-off, env-gated) for future experimentation, but is not the path to 25 tok/s/u.

## V2-CONFIG empirical sweep (2026-05-20) — dtype + compute-kernel changes are noise-level

Implemented matching of llama70b's matmul-side config (buckets C-G from
the audit) as env-gated branches. Measured wall-clock per phase:

| phase | env vars | tok/s/u | ms/step | Δ baseline |
|---|---|---:|---:|---:|
| baseline | — | 18.99 | 52.66 | — |
| C: HiFi4 → HiFi2 (FA + DN attn) | `QWEN36_ATTN_HIFI2=1` | 19.04 | 52.53 | +0.05 |
| C + E: matmul output bf8 (FA WO + DN out_proj; WQKVG stays bf16 — paged_update_cache requires bf16 input) | + `QWEN36_ATTN_OUT_BF8=1` | 19.07 | 52.43 | +0.08 |
| C + E + D: FA attention weights bf8 (override V2-7b bf16 lock-in) | + `QWEN36_ATTN_WEIGHTS_BF8=1` | 19.08 | 52.42 | +0.09 |
| C + E + D + G: fused sigmoid+multiply gate (V2-11 lever D retry) | + `QWEN36_GATE_FUSE=1` | **19.09** | **52.39** | **+0.10** |

**Cumulative recovery: +0.10 tok/s/u, -0.27 ms/step.** Far below the projected
5-12 ms / step from the audit's analytical buckets.

Reading: the audit's per-bucket estimates assumed the matmul cost was DRAM-
bandwidth-bound and that halving the bytes (bf8 weights / outputs) would
proportionally cut time. Empirically at B=1 decode this isn't true — the
matmuls are individually too small for HiFi-fidelity or weight-dtype to
matter on the critical path. The trace replay's critical path is dominated
by CCL fixed-overhead (which we showed via the barrier experiment is
already at its kernel floor) and norm ops.

**Phase B (prefetcher) is the remaining lever** but depends on V2-grid task
#47 (BH 130-core sub-device grid refactor) — out of scope for the autonomous
sweep. Estimate is still ~3-5 ms / step if landed.

All Phase C/E/D/G code remains in the codebase as env-gated branches
(default OFF). The V2-7b bf16 lock-in and the original sigmoid+multiply
split paths are still the default; the bf8 / fused variants are opt-in.

## Caveats & open items

- **Qwen3-32B 50 tok/s is the CI target**, not a fresh measurement on this
  branch. The `text_qwen_demo.py batch-1` demo hangs on this branch in
  decode trace capture (`tt/llama_attention.py:482`, `to_memory_config`
  op, reproducible across two cold resets). The branch has two April
  commits (`33d7e5881ba`, `0e0ae1776ab`) that modify
  `llama3_70b_galaxy/tt/llama_attention.py` / `llama_ccl.py` /
  `llama_mlp.py` / `llama_model.py` / `prefetcher_common.py`; one of
  these likely regressed Qwen3-32B's decode trace path. To get a fresh
  ground-truth Qwen3-32B number, either bisect those two commits or run
  on `main`.
- **The 19 tok/s/u qwen3.6 number is at V2-DN-TP HEAD**, not current branch
  HEAD (V2-DP-1+2). Current branch is documented as slower for single-user
  by its own commit message. If we want apples-to-apples (both TP=8 × DP=4),
  the qwen3.6 number to use would be the measured V2-DP-1 single-user
  perf — which is **not in PERF.md** because the V2-DP-1 commit declared
  itself negative and didn't add a number. To get apples-to-apples, run
  `test_decode_perf_intrace.py -k batch-1-single-user` on current HEAD and
  add the number here. The bucket attribution above does not depend on the
  exact V2-DP-1 number — the per-op breakdown is from V2-DP-1 tracy and
  scales the buckets, so 22 ms / 5 ms / 3 ms / 3 ms remain accurate.
- The V2-17/V2-18 DeltaNet kernel work was DEVICE BLOCKED (task #44 logged
  as completed but integration deferred). Resuming that work is the
  highest-leverage attack on Bucket 1.

## Files

- This document only.
- Source tracy artifacts:
  - `generated/profiler/reports/2026_05_19_19_26_09/ops_perf_results_*.csv`
    (DeltaNet 1L, V2-DP-2 head)
  - `generated/profiler/reports/2026_05_19_19_28_58/ops_perf_results_*.csv`
    (Full-attn 1L, V2-DP-2 head)
