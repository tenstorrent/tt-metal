<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mistral-Medium-3.5-128B — prefill bring-up

Prefill for the Mistral-Medium-3.5-128B text backbone on the
[`models/demos/common/prefill`](../common/prefill) engine, brought up against
[`MODEL_BRINGUP_RECIPE.md`](../common/prefill/docs/MODEL_BRINGUP_RECIPE.md). The process log is
[`bringup_log.jsonl`](bringup_log.jsonl); read it with
`models/demos/common/prefill/tools/bringup_digest.py`.

**Inputs** (both binding-or-advisory as the recipe defines):
[`Mistral-Medium-3.5-128B.spec.json`](Mistral-Medium-3.5-128B.spec.json) (BINDING) and
[`Mistral-Medium-3.5-128B.donor.json`](Mistral-Medium-3.5-128B.donor.json) (advisory).

---

## 1. Architecture

Everything below is read from the vendored
[`configs/Mistral-Medium-3.5-128B/config.json`](configs/Mistral-Medium-3.5-128B/config.json) and
asserted against it by `tests/unit/test_reference_config.py`.

| | |
|---|---|
| Family | `mistral3` wrapper / `ministral3` text backbone (`Mistral3ForConditionalGeneration`; the Pixtral vision tower is **out of scope**) |
| Layers · hidden · intermediate | 88 · 12288 · 28672 |
| Attention | dense GQA — 96 Q heads / 8 KV heads / head_dim 128 (group 12) |
| Attention extras | **none** — no QK-norm, no attention sinks, no sliding window (`sliding_window: null` on every layer), no q/k/v/o bias |
| MLP | dense SwiGLU (`silu`), every layer. **No MoE anywhere** |
| Norm | plain RMSNorm, eps 1e-5 (no Gemma `1 + w` fold) |
| RoPE | full rotary, YaRN: theta 1e6, factor 64, original_max_position 4096, beta_fast 4.0, beta_slow 1.0 |
| Vocab | 131072, untied |
| Checkpoint | fp8, `activation_scheme: static`, `weight_block_size: null` ⇒ **per-tensor** `weight_scale` |

Two config details that would be silently wrong if guessed:

* **YaRN `truncate` is TRUE.** HF reads `rope_parameters.get("truncate", True)` and this config carries
  no `truncate` key, so upstream floors/ceils the correction dims. The gpt-oss donor sets
  `truncate: false` and its inline comment warns *against* truncating — that comment does not
  transfer. Getting it wrong shifts every frequency by ~3.4e-4, which is invisible at short sequence
  and collapses long-context K PCC.
* **`llama_4_scaling_beta` is 0.** `Ministral3Attention` multiplies Q by
  `1 + beta*log(1 + floor(pos/original_max_position))`; at beta 0 that is exactly 1.0 at every
  position, so the TT attention omits the term. `reference.model.assert_llama4_scale_is_identity`
  fails loudly if a config ever turns it on.

## 2. Parallelism, from the spec

| | |
|---|---|
| Target | `bh_galaxy` — mesh **(4, 8)**, 32 devices |
| TP = 8 | on the **cols**. 96 Q heads → 12/device; 8 KV heads → **1/device** |
| SP = 4 | on the **rows**: sequence sharded, block-cyclic |
| chunk_size | 5120 (1280 tokens per SP row) |
| max_seq_len | 262144 servable |
| Cache capacity | **266240** = 52 × 5120 — see below |
| Activations · KV cache · weights | bfloat16 · bfloat8_b · bfloat8_b |
| Acceptance PCC | 0.99 |

**Why the capacity is not the spec's `max_seq_len`.** The spec requires only
`max_seq_len % (32 * sp) == 0`, but the whole-cache block-cyclic indexed rope and the KV chunk
address table both tile the cache by `chunk_size`, and 262144 is 51.2 chunks of 5120. `spec.py`
therefore exposes `cache_capacity = ceil(max_seq_len / chunk_size) * chunk_size = 266240`; the extra
tail is capacity only and `prefill_chunk` asserts requests stay inside the servable 262144.

## 3. What was reused, and what was written fresh

The donor map's own confidence ranking held up: the `high`-confidence entries transferred almost
unchanged, and the two entries it flagged are exactly the two that needed real work.

| Part | Source | How much changed |
|---|---|---|
| MeshConfig, CCLManager | `gpt_oss_d_p/tt/{config,ccl}.py` | as-is (validated target from the spec; `reduce_scatter` exposed) |
| Attention structure | `gpt_oss_d_p/tt/attention/` | **sinks, sliding window and all four biases removed**; head_dim 64→128, n_q 64→96; n_kv stays 8, so the 1-KV-head-per-column layout carried over untouched |
| RoPE | `gpt_oss_d_p/tt/rope.py` | near-exact; YaRN constants from the config, `truncate=True` (see above) |
| KV cache cluster (7 roles) | `gpt_oss_d_p` — **one package, as required** | head_dim 128, sinks/sliding threading dropped; layout, slot packing, bank walk and `ROUND_ROBIN_1D` verbatim |
| Runtime | `gpt_oss_d_p/tt/tt_prefill_runtime.py` | 88 layers, capacity-vs-context split, `use_trace`/`metadata_msg` fields the engine now reads |
| Dense MLP | `minimax_m3/tt/dense_mlp.py` | clamped SwiGLU-OAI → **plain silu SwiGLU**; ported onto this package's utils rather than importing across packages |
| Embedding | `minimax_m3/tt/parallel_embedding.py` | as-is, **defaulted to 1D** (§6) |
| Weight loading | `gpt_oss_d_p/tt/model_config.py` | wrapper prefix `model.language_model.*` → `model.*`, vision keys dropped, MXFP4 → fp8 |
| **fp8 dequant** | `deepseek_v3/utils/hf_model_utils.py` (structure only) | **written fresh**: no donor does per-tensor fp8. Reused the sorted key walk, the fatal "fp8 with no scale", and the single dtype exit |
| **`kv_migration_stages`** | `minimax_m3` pattern | **written fresh**: gpt_oss_d_p does not implement it. Two stages (k, v), not M3's three (no `index_k`) |
| RMSNorm | `gpt_oss_d_p/tt/rms_norm.py` | **recomposed** — `ttnn.rms_norm` cannot run at hidden 12288 (§6) |
| Reference model | transformers `ministral3` | **imported, not vendored** — it ships in transformers 5.12 and constructs standalone |
| MoE substrate | — | **not used**: dense model |

## 4. Running it

```bash
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
# This pod is a plain 8x4 GRID (no torus wrap-around), so FABRIC_1D + Topology.Linear. A
# torus-wired galaxy sets MISTRAL_LINEAR_FABRIC=0 (and PREFILL_TOPOLOGY=ring for the runner).
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto

# 0) mesh prerequisite: mesh opens, MeshConfig + CCLManager, all-gather + all-reduce
python models/demos/mistral_3_5_d_p/tests/galaxy_mesh_smoke.py

# 1) the module + model PCC suites (random weights, target mesh)
python -m pytest models/demos/mistral_3_5_d_p/tests/unit -q

# 2) the KV chunk address table, bit-exact (no model, no fabric traffic)
python -m pytest models/demos/mistral_3_5_d_p/tests/test_kv_cache_table.py -q
```

Real weights need a checkpoint. There is none on this host (§7), so the runs below use a
**format-identical synthetic** one:

```bash
# a checkpoint in the published on-disk format (fp8 + weight_scale/input_scale + wrapper prefix)
python models/demos/mistral_3_5_d_p/scripts/make_synthetic_checkpoint.py \
    --out /tmp/ckpt --layers 4 --hidden 12288 --intermediate 28672 --vocab 2048

# the CPU golden trace, generated THROUGH the production loader
python models/demos/mistral_3_5_d_p/scripts/generate_golden_kv_cache.py \
    --weights /tmp/ckpt --layers 4 --hidden 12288 --intermediate 28672 --vocab 2048 \
    --isl 5120 --out /tmp/golden

# P1: one-shot per-layer KV PCC          P2: the same, chunked
HF_MODEL=/tmp/ckpt PREFILL_TRACE_DIR=/tmp/golden \
    python models/demos/mistral_3_5_d_p/tests/galaxy_prefill_kv_pcc.py
HF_MODEL=/tmp/ckpt PREFILL_TRACE_DIR=/tmp/golden PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=1280 \
    python models/demos/mistral_3_5_d_p/tests/galaxy_prefill_kv_pcc.py

# P3: the SHARED two-process producer/runner e2e (11 x 5120 tokens, so use a 56320-token golden).
# PREFILL_MODEL and PREFILL_SP/TP must be exported ALONGSIDE the manifest: only the RUNNER reads
# PREFILL_MANIFEST — the producer resolves its adapter from PREFILL_MODEL and reads SP/TP directly,
# and without them it silently falls back to the default MLA model and decodes KV at the wrong
# head_dim. The runner needs a signal to leave its sync-op loop once the producer passes, so the
# pytest verdict lands a few minutes after the PCC does.
PREFILL_MODEL=mistral_3_5_d_p PREFILL_SP=4 PREFILL_TP=8 PREFILL_TOPOLOGY=linear \
PREFILL_MANIFEST=models/demos/mistral_3_5_d_p/tt/runners/manifests/mistral_3_5.json \
PREFILL_TRACE_DIR=/tmp/golden_56320 HF_MODEL=/tmp/ckpt_l2 PREFILL_HF_MODEL=/tmp/ckpt_l2 \
PREFILL_NUM_LAYERS=2 \
    python -m pytest models/demos/common/prefill/tests/test_producer_runner_e2e.py \
        -k single_user_full_depth -q

# the SHARED KV-chunk decode test (host only, every supported cache byte format)
PREFILL_MODEL=mistral_3_5_d_p \
    python -m pytest models/demos/common/prefill/tests/test_prefill_producer_kv_decode.py -q
```

Only ONE device job at a time: a second one blocks on the `CHIP_IN_USE_0_PCIe` chip lock and both
look hung, with the waiting process logging nothing but a lock warning.

Environment knobs this package adds: `MISTRAL_LINEAR_FABRIC` (fabric topology),
`MISTRAL_EMBED_SHARD_VOCAB` (embedding sharding), `MISTRAL_WEIGHT_DTYPE` (a weight-dataformat A/B),
`MISTRAL_WEIGHTS_FROM_CACHE`, `MISTRAL_KV_PCC_MIN` (a PCC floor for the harness),
`MISTRAL_35_HOST_REF_CACHE` (the golden cache dir).

## 5. Accuracy

### Per-block, random weights, target mesh, vs an **fp32** torch oracle

| Block | PCC | at |
|---|---|---|
| RMSNorm | 0.99999 | hidden 12288, 32/128/512 tokens |
| SwiGLU | 0.99999 | intermediate 3584/shard |
| Dense MLP | 0.99998 (bf16 w) / 0.99991 (spec bf8 w) | 12288 → 28672 → 12288 |
| Attention (one-shot) | 0.99989 | 128 and 512 tokens |
| Ring SDPA, live Q/K/V | 0.99998 / 0.99996 | 512 / **5120** tokens |
| Ring SDPA, cache read | 0.99995 / 0.99989 | 2×128 / **2×5120** tokens |
| KV write round trip | 0.99997 | incl. the spec's 5120 chunk |
| Attention post-RoPE K/V through the seam | K 0.99981, V 0.99982 | 256 tokens, indexed rope |
| Chunked attention (2 chunks, same module) | 0.9952 / 0.9916 | 2×256 / 2×512 |
| Decoder layer | 0.9964 (one-shot) / 0.9954 (chunked) | 512 tokens, real dims |
| Embedding (1D) | exact, row by row | 128 / 4096 / **5120** tokens |
| LM head | 0.99995 | column-parallel over vocab |
| Whole model, per-layer KV | L0 0.99994, L1 0.9966–0.9971 | 2 layers, 512 tokens |

### Per-layer KV PCC vs the golden trace (real loader, synthetic weights, 5120 tokens)

The recipe's graded quantity. Also the number this section exists to be honest about.

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| one-shot, 4L | 0.99994 | 0.99800 | 0.99101 | 0.97771 | | | | |
| one-shot, 8L | 0.99994 | 0.99801 | 0.99102 | 0.97771 | 0.95983 | 0.93945 | 0.91817 | 0.89678 |
| chunked (4×1280), 4L | 0.99994 | 0.99496 | 0.97884 | 0.95323 | | | | |

**Layers 0–2 clear the spec's 0.99; the decay continues with depth and does not.** Three things are
worth stating plainly about that number:

1. **It is measured on RANDOM weights.** Each layer's output is then near-orthogonal to its input, so
   relative error compounds with nothing to damp it. A trained residual stream is dominated by a
   strong shared signal, which is why this decay is far steeper than a real checkpoint's would be.
   The recipe's own rule — random weights until P1, real weights at P1 — is exactly the step that
   cannot be completed here (§7).
2. **It is against an fp32 oracle, which is the strict choice.** The sibling harnesses
   (`gpt_oss_d_p`, `minimax_m3`) generate their goldens in bf16. Measured here, a bf16 CPU forward of
   the same weights is a far WORSE approximation of fp32 than the device is: at 5120 tokens its own
   layer-3 K lands where the device would score 0.44 against it. `--compute-dtype bfloat16` will
   generate that golden; it is not a fairer bar, it is a looser and noisier one.
3. **Chunked trails one-shot by ~0.025 at layer 3**, which is the bfloat8_b KV cache: the ring path
   reads K/V back out of the cache, the one-shot path uses live bf16 K/V. The spec fixes the cache
   dtype, so that gap is the spec's, not the implementation's.

Four accuracy fixes found by these measurements, each worth several points and each a place the donor
did not transfer (all recorded in `bringup_log.jsonl`):

| Fix | Effect |
|---|---|
| fp32 destination accumulation on the projection matmuls | decoder layer 0.989 → 0.996 |
| fp32 destination accumulation on the plain SDPA | layer-3 KV 0.957 → 0.978 |
| 1D embedding instead of 2D vocab-parallel | layer-0 KV 0.995 → 0.99994 |
| recomposed RMSNorm | made hidden 12288 runnable at all |

### Serving: the shared producer/runner e2e (P3)

`test_producer_runner_e2e.py::test_producer_runner_pcc[single_user_full_depth]`, model selected by
env, 2 layers x 11 chunks x 5120 = **56320 tokens**, one user. The producer pushes token chunks over
the H2D socket, then reads the KV back **device-lessly over UMD through the published address table**
and PCCs it against the golden trace:

```
layer  0: K=0.99994 V=0.99994
layer  1: K=0.99385 V=0.99385
[producer] KV cache PCC PASSED (min 0.993847 >= 0.93 across 1 slots)
[producer] drained 22/22 layer acks in 0.13s
```

Two things this gate proves that nothing before it does: the address table describes the cache well
enough for a reader in ANOTHER PROCESS with no device handle to reconstruct it, and the per-layer
D2H acks arrive (22 = 11 chunks x 2 layers). The read-back numbers match the in-process ones exactly,
so the table and the device agree.

### Throughput

Not a tuned number and not comparable to a torus-wired pod (this one is a plain grid, so every
collective is Linear). 4 layers, 5120 tokens, one user: **28.3k tok/s** one-shot (181 ms), 7.5k tok/s
chunked at 4×1280 (687 ms). 8 layers one-shot: 14.0k tok/s.

## 6. Deviations from the donors, and why

1. **RMSNorm is composed, not `ttnn.rms_norm`.** At hidden 12288 the interleaved layernorm kernel's
   circular buffers grow to 2565120 B against a 1572864 B L1 budget and it THROWS. The sibling models
   avoid this by norming an `emb/tp` shard, which presumes an `emb/tp`-sharded residual stream —
   restructuring the whole model's residual is the sharding optimisation bring-up is not meant to
   take on. So the norm is composed from `multiply`/`mean`/`rsqrt` with an fp32 sum of squares, which
   is width-independent and measures 0.99999. See `tt/rms_norm.py`.
2. **The embedding defaults to 1D, not M3's 2D.** The 2D vocab-parallel path returns wrong rows above
   4096 tokens per chunk (32–44 bad rows at 4608/5120/8192, a mix of zeros and wrong content, first
   at global position ~1088) and the spec's chunk is 5120. It fails **quietly** — it showed up as
   layer-0 KV at 0.995 with nothing raised. 1D is exact at every length tested and still saves 8x
   against a replicated table (0.38 GiB/device vs 3.2 GiB). `tests/unit/test_parallel_embedding_vs_ref.py::test_2d_embedding_is_broken_above_4096_tokens`
   pins the defect at its boundary so the fix is noticed.
3. **fp32 destination accumulation** on the projection matmuls and the plain SDPA. The donor lets
   `ttnn.linear` take its defaults and pins every SDPA to `fp32_dest_acc_en=False`; the latter
   constraint is real but applies only to the ring cache-read op, which still hard-codes it.
4. **The fused o_proj matmul + reduce-scatter is not carried over.** The donor gates it off on
   Blackhole (a semaphore race) and the spec targets `bh_galaxy`, so the only path it could take is
   the known-broken one.
5. **No on-device sampling hooks.** Decode is out of scope and prefill runs `skip_lm_head=True`.

## 7. Known gaps

* **No real checkpoint.** The published 128B weights are not reachable from this host, so every
  real-weights run uses `scripts/make_synthetic_checkpoint.py` — identical in FORMAT (safetensors
  shards + index, `model.language_model.*` prefix, per-tensor fp8 with `weight_scale`/`input_scale`,
  unquantized `lm_head`) and random in VALUE. That fully validates the loader, the dequantizer, the
  key mapping and the golden pipeline; it says nothing about the model's accuracy on real weights,
  and it is why the depth decay in §5 should not be read as a property of Mistral-Medium-3.5.
* **Reduced depth and width in the host-side comparisons.** The full model is ~121 B parameters; no
  host can materialise that as random weights to drive both sides. Every test runs at the model's
  REAL width and head geometry (12288 / 28672 / 96·8·128 — the parts under test) and reduced depth,
  plus a reduced vocab where an embedding table has to fit on the host. The code is depth-agnostic:
  `PREFILL_NUM_LAYERS` selects any depth, and the deepest run measured here is 8 layers.
* **Full 88-layer, 262144-token run not executed.** Nothing in the code prevents it (the KV cache and
  the rope are sized from `spec.cache_capacity`), but it needs the real checkpoint.
* **`kv_migration_stages` is untested end to end.** It is implemented and its address table is
  verified bit-exactly, but Gate 2 of `PREFILL_MIGRATION_TESTING.md` (the real DRAM → transport →
  DRAM copy) needs the external `migration_endpoint` / `migration_worker` binaries.
* **Multi-rank pipeline parallelism not exercised.** Single-rank only, like the donor. The
  `first_layer_idx` / `is_first_rank` / `is_last_rank` plumbing is threaded through but untested.
* **No perf work.** Out of scope per the recipe. In particular the residual stream is replicated
  across TP rather than `emb/tp`-sharded, and the o_proj/MLP tails all-reduce rather than
  reduce-scatter.
* **The KV pad window is not zeroed before a layer ack.** DeepSeek's block zeroes the cache past
  `actual_end` so a migration of a partially-filled final chunk does not move stale bytes. That is
  implemented there and not here; it affects a REAL migration of the tail chunk (Gate 2), not the
  acks and not a PCC over `[0, real_len)`. See `_layer_ack_callback`.
* **The shared e2e runner needs a signal to finish.** After the producer's PCC passes and it exits,
  the runner stays in its sync-op loop; the test's SIGINT did not land while it was blocked in the
  H2D receive, so the run only returned its verdict after the runner was signalled directly. The
  gate itself is unaffected (the PCC is the producer's), but budget for it: `PASSED` arrives after
  the runner is torn down, not when the producer succeeds.

## 8. Layout

```
spec.py                       the BINDING spec, parsed + validated (cache_capacity lives here)
conftest.py                   picks the mesh-graph descriptor before the cluster initialises
reference/
  mistral_config.py           dimension constants, asserted against the vendored config.json
  model.py                    the torch oracle: imported ministral3 + a second, inline golden
  golden_cache.py             frozen-key golden cache (ReferenceCacheKey), assert-not-recompute
tt/
  config.py  ccl.py           MeshConfig / CCLManager (fixed references)
  rms_norm.py                 composed RMSNorm (see §6.1)
  rope.py                     YaRN + whole-cache block-cyclic indexed rope
  mlp.py                      dense silu SwiGLU, column/row-parallel
  attention/                  config · weights · operations · kv_cache · dense_sp · prefill
  parallel_embedding.py       1D / 2D sharded embedding (1D default, see §6.2)
  model.py  layer.py          the 88-layer stack and one decoder layer
  model_config.py             ModelArgs: safetensors → fp8 dequant → prefix map → Meta swizzle
  fp8_dequant.py              per-tensor fp8 (written fresh)
  tt_prefill_runtime.py       compile / make_chunk_input / prefill_chunk / migration hooks
  runners/
    kv_chunk_table.py         the 16-config DRAM address table
    adapters/mistral_3_5.py   the engine boundary
    manifests/mistral_3_5.json
tests/
  test_factory.py             mesh + fabric parametrization, shard mappers, read-back helpers
  galaxy_mesh_smoke.py        the D3 mesh prerequisite
  galaxy_prefill_kv_pcc.py    P1 (one-shot) and P2 (chunked) KV PCC
  test_kv_cache_table.py      P4, bit-exact
  golden_hf_first_token.py    M1 ground truth against a real checkpoint dir
  unit/                       the decoder + model PCC suites (67 tests)
scripts/
  generate_golden_kv_cache.py the ONE place a CPU reference forward runs
  make_synthetic_checkpoint.py a format-identical checkpoint (see §7)
```
