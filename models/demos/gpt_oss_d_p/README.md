<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# GPT-OSS-120B Prefill (`gpt_oss_d_p`)

Long-context, chunked **prefill** for GPT-OSS-120B on a **4×8 Blackhole Galaxy** (TP=8, SP=4,
EP=32). The package builds the model in TTNN and fills a sequence-parallel KV cache one chunk at a
time. It is built to hand that cache to a decode engine through the disaggregated-prefill machinery
in `models/demos/common/prefill`. It reuses the DeepSeek expert-parallel MoE substrate and follows
the structure of `models/demos/minimax_m3`. Everything GPT-OSS-specific is written here: GQA with
attention sinks, sliding/full layer alternation, YaRN RoPE, MXFP4 experts and a plain top-k router.

Owner: Maciek Dragula (`@mdragulaTT`), models team.

**Terms used below.** TP / SP / EP: tensor, sequence and expert parallelism (which mesh axis shards
features, the sequence, and the experts). GQA: grouped-query attention, several Q heads sharing one
KV head. PCC: Pearson correlation coefficient between a device tensor and its CPU reference. CCL:
the collective communication ops (all-gather, reduce-scatter) between chips. Halo: the few rows of
K/V a chip needs from its neighbour so a sliding window can cross a chip boundary. One-shot: a
prefill of the whole prompt in a single chunk. Indexed RoPE: rotary embeddings looked up by cache
position instead of recomputed per chunk. Tilized cache: weights pre-converted to the device tile
layout and stored on disk. D2H: device-to-host. TTFT: time to first token.

## Status (September 2026)

GPT-OSS prefill and prefill/decode disaggregation are **paused**. What is here is complete and
validated for single-user chunked prefill. The plan for the next steps is recorded in issues so the
work can resume as-is.

| Area | State |
|---|---|
| Chunked prefill, full 36-layer model, real weights | merged; every chunk goes through the cache-backed ring SDPA |
| Variable chunk sizes (one runtime serving several sizes) | merged |
| Bounded sliding-window KV cache (circular window for the 18 sliding layers) | merged, opt-in (`GPT_OSS_BOUNDED_SLIDING_KV=1`) |
| Per-layer KV-PCC validation vs a CPU golden; two galaxy CI accuracy stages and a smoke stage | merged |
| GPT-OSS through the common prefill runner (producer/runner e2e), with its galaxy CI stage | in review, [#56519](https://github.com/tenstorrent/tt-metal/pull/56519) |
| Trace capture of the prefill chunk | planned, [#56661](https://github.com/tenstorrent/tt-metal/issues/56661) (stage one [#56660](https://github.com/tenstorrent/tt-metal/issues/56660), stage two [#56115](https://github.com/tenstorrent/tt-metal/issues/56115)) |
| Bounded sliding cache as the default, KV migration of a bounded cache | planned, [#55646](https://github.com/tenstorrent/tt-metal/issues/55646) |
| Hoisting shared prefill scaffolding into `common/prefill` | open, [#55647](https://github.com/tenstorrent/tt-metal/issues/55647) |

## The machine

- **Mesh**: 4 rows × 8 columns of Blackhole chips. `tt/config.py::MeshConfig` puts **TP=8 on the
  columns**, so every feature-sharded tensor spans a whole row. The rest follows from the mesh
  shape: the **4 rows carry SP** (each row holds a quarter of the sequence) and the **32 chips carry
  EP** for the 128 routed experts, 4 experts per chip.
- **Topology**: `ttnn.Topology.Ring` by default (torus wraparound links, faster CCLs);
  `PREFILL_TOPOLOGY=linear` for pods without wraparound. The ring SDPA cache read needs ring
  connectivity along the SP axis, which the CI stages get from the `torus_xy` fabric profile.
  Under the common runner the GPT-OSS manifest (`tt/runners/manifests/gpt_oss_d_p.json`) sets
  `PREFILL_FABRIC_MODE=1d_ring`; the runner-path CI stage in [#56519](https://github.com/tenstorrent/tt-metal/pull/56519) exports `2d_torus_xy` instead.
  Both provide that ring. Accepted values live in `models/demos/common/prefill/runners/runner_utils.py`.
- **Precision**: activations bf16. MoE expert weights `bfloat4_b` by default or `bfloat8_b` with
  `EXPERT_DTYPE=bf8`, which the accuracy stages and every number quoted here use; the common-runner
  path honours the same knob from [#56519](https://github.com/tenstorrent/tt-metal/pull/56519) on. KV cache `bfloat8_b`; the chunked
  ring path requires it (`KV_CACHE_DTYPE=bf16` is accepted only by one-shot prefill). The checkpoint
  ships the experts MXFP4-quantized; `ModelArgs.load_state_dict` dequantizes them to bf16 on the
  host (`tests/unit/test_mxfp4_loader.py` pins that path).

## The model, as it runs here

36 decoder layers. Layer types alternate `sliding_attention` / `full_attention` from
`hf_config.layer_types`, 18 of each; the sliding window is 128 tokens. Attention is GQA with 64 Q
heads over 8 KV heads, head_dim 64, full rotary (YaRN) and a learned per-head **attention sink**.
The MLP is a 128-expert MoE, top-4 per token, SwiGLU-OAI with biases, no shared expert and no dense
layers.

Per chip, per layer, one prefill chunk (`S_loc = S / SP`):

| tensor | shape | dtype | notes |
|---|---|---|---|
| Q | `[1, 8, S_loc, 64]` | bf16 | 8 of the 64 Q heads (TP=8) |
| K, V | `[1, 1, S_loc, 64]` | bf16, cache bf8_b | 1 of the 8 KV heads; the 8 local Q heads share it (no on-chip KV repeat) |
| sinks | `[1, 8, 1, 1]` | bf16 | stored pre-divided by `config.scaling`, so the SDPA kernel's own scaling recovers the raw HF sink logit |
| out | `[1, 8, S_loc, 64]` | bf16 | |

Every chunk, including chunk 0, writes its K/V into the cache and reads back through
`ttnn.transformer.ring_joint_scaled_dot_product_attention` (`tt/attention/dense_sp.py`). Sinks, the
sliding or causal mask and the cross-chip halo are handled inside the op. Only an equal-sized
one-shot prefill (`PREFILL_CHUNKED=0`, chunk == `max_seq_len`) keeps the all-gather path instead,
because the sliding ring path needs a short Q against a longer cache.

## Two entry points, one lifecycle

Both paths drive the same `TtPrefillRuntime` (`tt/tt_prefill_runtime.py`):

```
ModelArgs(hf config + weights)                       tt/model_config.py
  -> TtPrefillRuntime(mesh, hf_config, state_dict, TtPrefillRuntimeConfig)
       constructor: build the 36 decoder layers, EP MoE and CCL manager; allocate the
                    block-cyclic SP KV cache (unless the engine owns it); build the
                    whole-cache indexed RoPE once, reused by every chunk
       .compile()                                    warm-up per supported chunk size: two chunks
                                                     (the second exercises the ring cache read),
                                                     one for equal-sized one-shot
       .make_chunk_input(token_ids, chunk_size)      SP-sharded uint32 tokens for one chunk
       .prefill_chunk(tokens, slot_id=, actual_start=, actual_end=, chunk_size=)   x N chunks
       .gather_layer() / .kv_cache_pcc_check()       read the cache back, compare to the golden
```

`TtPrefillRuntimeConfig` holds the geometry: `num_layers`, `max_seq_len` (per-user cache length),
`default_chunk_size` (8192) plus `additional_chunk_sizes`, `num_users` (cache slots),
`cache_dtype`, `expert_weight_dtype`, `weight_cache_path`, the pipeline-rank fields the engine
fills in, and `bounded_sliding_kv_cache`. The runtime has no trace path.

**1. Standalone harness** — `tests/galaxy_prefill_kv_pcc.py`, a plain `python3` script run under
`mpirun`, not pytest. It loads real weights from the tilized TTNN cache and prefills the golden
prompt in chunks. It prints throughput two ways, over the real prompt tokens and over the processed
(padded) tokens. Then it reads every layer's K and V back, PCCs them against the golden, and fails
below `GPT_OSS_KV_PCC_MIN`. The two CI accuracy stages run exactly this; it is the quickest way to
reproduce a number.

```bash
cd <tt-metal>; source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
export HF_MODEL=/mnt/models/blaze/openai/gpt-oss-120b            # config.json; weights come from the cache below
export TT_CACHE_PATH=/mnt/models/blaze/openai/gpt-oss-120b       # holds tensor_cache_bfp8_MeshShape([4, 8])
export PREFILL_TRACE_DIR=/mnt/models/blaze/openai/gpt-oss-120b/golden/longbook_qa_eng_prefill_5000
export GPT_OSS_WEIGHTS_FROM_CACHE=1 EXPERT_DTYPE=bf8
export PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=1024 GPT_OSS_KV_PCC_MIN=0.85
mpirun --pernode --tag-output bash -lc 'python3 models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py'
```

Expect about 12 minutes warm: the tilized cache is roughly 180 GB read over NFS, then kernel JIT,
prefill and the PCC readback. Without the cache (`GPT_OSS_WEIGHTS_FROM_CACHE` unset) the run
tilizes from safetensors first, which takes far longer than the CI budget.

**2. Common prefill runner** — the production path (`models/demos/common/prefill`: runner process,
request queue, producer, KV migration to decode). GPT-OSS plugs in through
`tt/runners/adapters/gpt_oss.py` (`PREFILL_MODEL=gpt_oss_d_p`); `tt/runners/manifests/gpt_oss_d_p.json`
carries the mesh defaults. The adapter allocates the cache, builds the runtime and points the engine
at the model's weight-cache layout. Layer completion is reported to the migration layer through the
host `on_layer_complete` callback (`set_layer_completion_sink`), not through D2H device records.
`tt/runners/kv_chunk_table.py` builds the KV chunk address table for migration from the packed
cache. The producer/runner e2e scenario `single_user_full_depth`
(`models/demos/common/prefill/tests/test_producer_runner_e2e.py`, 11 × 5120-token chunks) runs
GPT-OSS end to end and checks the producer's KV PCC over the first 5000 positions, which is all the
golden holds. The runner-path CI stage that lands with [#56519](https://github.com/tenstorrent/tt-metal/pull/56519) runs that scenario; its `cmd` block in
the matrix is the reference launch command. General runner usage: `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`
and `models/demos/common/prefill/runners/RUN_COMMANDS.md`.

## The KV cache

`tt/attention/kv_cache.py`. By default one K tensor and one V tensor hold every layer of every
user: per chip `[num_users * num_layers, 1, seq_local, head_dim]`, batch slot
`user * num_layers + layer`. This is the layout KV migration reads.

Along SP the cache is **block-cyclic**. Each chunk is cut into SP equal pieces, and SP row `r`
stores piece `r` of every chunk, one chunk after another. Every row therefore holds an interleaved
quarter of the sequence, and the ring SDPA reads the neighbouring rows' pieces through the halo.
`blockcyclic_positions` (from the DeepSeek MLA utilities) is the inverse map back to natural
positions, and `gather_layer` uses it to return a layer in token order for validation.

**Variable chunk sizes.** One runtime serves `default_chunk_size` plus `additional_chunk_sizes`.
Every size gets its own indexed RoPE and the MoE buffers are sized for the largest. Each size must
divide `max_seq_len`, and a given sequence uses one size for all its chunks.

**Bounded sliding-window cache** (`bounded_sliding_kv_cache`, env `GPT_OSS_BOUNDED_SLIDING_KV=1`).
A sliding layer only ever attends to the last 128 tokens, so its cache slot does not need the full
sequence. With the flag on, full-attention and sliding layers move into separate K/V tensors.
`build_layer_map` gives each layer of this rank `(is_sliding, ordinal, n_type)`, so its slot is
`user * n_type + ordinal`. A sliding slot holds `min(2 × largest chunk size, max_seq_len)` tokens
(`sliding_capacity_tokens`): the chunk being written plus the previous one, which always contains the
whole window and halo. Two more rules apply with the flag on: every served chunk size must divide
that capacity, so each chunk lands on a whole slab, and the 128-token window must fit the smallest
chunk. `write_kv_chunk` writes each chunk at its position modulo that capacity, the
ring SDPA reads the slot circularly (`circular_kv_cache=True`, an op-level feature of RingJointSDPA),
and `bounded_blockcyclic_positions` is the readback inverse. At 128k context with 8k chunks the 18
sliding layers shrink from full-length to two-chunk slots: **153 → 86 MiB of KV per user per chip**,
with unchanged PCC. Two limits are recorded: the circular read is not available on the trace
metadata path ([#56115](https://github.com/tenstorrent/tt-metal/issues/56115)), and KV migration
cannot consume a bounded cache yet, which is why the flag is off by default
([#55646](https://github.com/tenstorrent/tt-metal/issues/55646)).

## MoE

`tt/mlp.py` → `tt/moe/router.py` (linear + bias, top-4, softmax over the 4) → `tt/moe/tt_gpt_oss_moe.py`,
a thin composition of the DeepSeek EP modules: `TtDispatchModule`, `TtRoutedExpert` (the fused
`unified_routed_expert_moe` kernel in SwiGLU-OAI mode), `TtCombineModule`, `TtReduceModule` and
`TtMoERoutingSetup`. `tt/moe/weights.py` de-interleaves the packed `gate_up_proj` into plain torch
gate and up tensors; the per-chip expert placement and the device dtype are applied downstream by
`ExpertMapping` and `TtRoutedExpert`.

## Validation

Accuracy is gated on **per-layer KV-cache PCC against a CPU golden**. The HF model
(`reference/model.py`, with `reference/tiled_ops.py` so a long prompt fits in memory) captures each
layer's post-RoPE K and raw V before any sliding-window truncation. `scripts/generate_golden_kv_cache.py`
writes the trace and `scripts/verify_golden_kv.py` checks its structure (see
`scripts/README_golden_kv.md`). The staged golden is a 5000-token "longbook" prompt. The gate is
**0.85**; measured minima across the 36 layers are K ≈ 0.97–0.98 and V ≈ 0.89–0.92 with bf8
experts (the V minimum sits on a full-attention layer, bounded cache on or off).

Unit tests: `pytest models/demos/gpt_oss_d_p/tests/unit` (the CPU-only ones run anywhere).

| test | needs | covers |
|---|---|---|
| `tests/unit/test_attention_vs_ref.py` | 1 card | QKV + bias, GQA split, YaRN RoPE, sinks, sliding/full masks vs a torch reference |
| `tests/unit/test_kv_cache_vs_ref.py` | 1 card | cache write + read-back round trip, slot and layout |
| `tests/unit/test_indexed_rope_vs_ref.py` | 1 card | whole-cache indexed RoPE vs the reference |
| `tests/unit/test_moe_vs_ref.py` | 1 card for router and expert; the 4×8 galaxy for the EP path | router, routed expert, dispatch/combine |
| `tests/unit/test_bounded_kv_math.py` | CPU | layer map, sliding capacity, circular positions |
| `tests/unit/test_variable_chunk_config.py` | CPU | chunk-size resolution |
| `tests/unit/test_mxfp4_loader.py` | CPU | MXFP4 expert dequantization |
| `tests/test_kv_cache_table.py` | 2×4 Blackhole mesh | KV chunk address table vs the allocated caches |
| `tests/galaxy_prefill_kv_pcc.py` | galaxy | full model, real weights, throughput + per-layer PCC |
| `tests/variable_chunk_smoke.py` | galaxy | one runtime prefilling the same tokens as a 1k and an 8k chunk (no PCC) |

Galaxy CI (`tests/pipeline_reorg/blaze_models_prefill_tests.yaml`, "Blaze Models Prefill tests"):
`(GPT-OSS-120B) chunked prefill KV accuracy longbook 5k@2.5k` and `... 5k@1k` run the harness with
the PCC gate and `(GPT-OSS-120B) variable-chunk prefill smoke 1k vs 8k` runs the smoke.
`(GPT-OSS-120B) prefill runner accuracy longbook 55k@5k vs 5k golden prefix`, the e2e scenario
through the common runner, lands with [#56519](https://github.com/tenstorrent/tt-metal/pull/56519). Op-level coverage for the sliding ring read lives in
`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` (production and circular-cache accuracy,
metadata-path rejection) and the header gtest
`tests/ttnn/unit_tests/gtests/sdpa/test_sliding_window_work_plan.cpp`.

## Performance (measured September 2026, single user, 36 layers, bf8 experts)

| prompt @ chunk | throughput | TTFT (derived) |
|---|---|---|
| 5k @ 1k | ~2.5k tok/s | ~2 s |
| 5k @ 2.5k | ~6.3k tok/s | ~0.8 s |
| 128k @ 8k | ~16k tok/s | ~8 s |

Device time is about 410 ms per 8k chunk, close to 20k tok/s if the host kept up, and 60–70 % of
it is communication (MoE combine, reduce and dispatch dominate). The spread between 1k and 8k
chunks is host-side dispatch overhead, not device work. Trace capture of the chunk program is the
fix ([#56661](https://github.com/tenstorrent/tt-metal/issues/56661)); an earlier 4-layer
eager-vs-replay measurement, recorded in
[#56660](https://github.com/tenstorrent/tt-metal/issues/56660), put ~35 % of the per-layer wall
time in host dispatch.

## File map

```
tt/tt_prefill_runtime.py     TtPrefillRuntimeConfig + TtPrefillRuntime (lifecycle, prefill_chunk, cache readback, PCC)
tt/model.py, tt/layer.py     embedding -> decoder layers (final norm and lm_head skipped for prefill)
tt/attention/                prefill.py (forward), dense_sp.py (ring SDPA cache read), kv_cache.py (allocation,
                             block-cyclic + bounded write, layer map), operations.py, config.py, weights.py
tt/moe/                      router.py, tt_gpt_oss_moe.py (EP composition), weights.py (gate/up de-interleave)
tt/mlp.py, tt/rms_norm.py, tt/rope.py, tt/ccl.py
tt/config.py                 MeshConfig (TP axis -> SP/EP axes)
tt/model_config.py           ModelArgs: HF config, weight loading incl. MXFP4 dequantization, tensor-cache naming
tt/runners/adapters/gpt_oss.py   common/prefill adapter;  tt/runners/kv_chunk_table.py  migration table
reference/                   CPU golden (HF model + memory tiling)
scripts/                     golden generation and verification
tests/                       unit tests, KV table device test, galaxy harness, variable-chunk smoke
utils/                       state-dict helpers
```

`GptOss120BConfig` (dimensions) lives in `models/demos/deepseek_v3_d_p/reference/gpt_oss_120b_config.py`,
shared with the DeepSeek MoE op tests that already exercise gpt-oss shapes. The default HF config
directory is `models/demos/gpt_oss/configs/gpt-oss-120b`, borrowed from the Wormhole demo as data.

## Environment variables

| variable | read by | meaning |
|---|---|---|
| `HF_MODEL`, `PREFILL_HF_MODEL` | model config, adapter, golden script | directory with `config.json` (and weights, unless loading from the cache) |
| `TT_CACHE_PATH`, `PREFILL_TTNN_CACHE` | model config, adapter | TTNN tilized weight cache root; the model layout is `tensor_cache_{dtype}_MeshShape([4, 8])` |
| `GPT_OSS_WEIGHTS_FROM_CACHE=1` | harness, smoke, adapter | skip the safetensors load and take every weight from the tilized cache |
| `EXPERT_DTYPE` (`bf4`, `bf8`) | harness, smoke; adapter from [#56519](https://github.com/tenstorrent/tt-metal/pull/56519) | routed-expert weight dtype |
| `KV_CACHE_DTYPE` (`bf8`, `bf16`) | harness | KV cache dtype; bf16 only with one-shot prefill |
| `PREFILL_NUM_LAYERS` | harness, smoke, common runner | layers to build (bring-up on a subset) |
| `PREFILL_CHUNKED`, `PREFILL_CHUNK_SIZE` | harness | chunked vs one-shot; chunk size (default 8192) |
| `PREFILL_NUM_USERS` | harness | cache slots; every slot prefills the prompt and is checked |
| `PREFILL_TPS_ITERS` | harness | repetitions for the throughput number |
| `PREFILL_TRACE_DIR` | harness, runtime, runner and producer | golden trace directory (prompt + per-layer K/V) |
| `GPT_OSS_KV_PCC_MIN` | harness | fail below this min per-layer PCC (0.85 in CI) |
| `GPT_OSS_BOUNDED_SLIDING_KV=1` | harness, adapter | bounded circular cache for the sliding layers |
| `PREFILL_TOPOLOGY` (`ring`, `linear`) | harness, adapter | CCL topology |
| `GPT_OSS_KV_DUMP`, `GPT_OSS_KV_DUMP_DIR` | runtime | bring-up diagnostics: per-layer tensor dumps and a per-position K RoPE probe |
| `GPT_OSS_DELTA_PROBE` | decoder layer | bring-up diagnostic: per-layer norms of the attention and MoE residual deltas |
| `VARCHUNK_SMALL`, `VARCHUNK_LARGE` | smoke | the two chunk sizes of the variable-chunk smoke |
| `REF_ATTN_Q_CHUNK`, `REF_FFN_TOKEN_CHUNK` | reference | memory tiling of the CPU golden |

The common runner's own knobs (`PREFILL_MODEL`, `PREFILL_SP`/`PREFILL_TP`, `PREFILL_FABRIC_MODE`,
`PREFILL_LAYER_ACK_D2H`, migration and producer settings) are documented in
`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md` and
`models/demos/common/prefill/runners/runner_utils.py`.

## Reused vs written here

Reused from `models/demos/deepseek_v3_d_p`: the EP MoE modules listed above and their init
helpers (`ExpertMapping`, mesh mappers), the block-cyclic helpers (`block_cyclic_reorder`,
`blockcyclic_positions`) and `GptOss120BConfig`. Reused from `ttnn`: the chunked-KV update op
(`update_padded_kv_cache`), indexed RoPE (`rotary_embedding_indexed`), and
`ring_joint_scaled_dot_product_attention` with sinks, sliding window and the circular cache read.
Written here: attention (GQA, sinks, sliding/full, YaRN), the router, MXFP4 weight loading and
expert weight preparation, the KV cache with its bounded sliding variant, the runtime and the
`common/prefill` adapter. The Wormhole `models/demos/gpt_oss` demo was a code-lineage source only;
nothing is imported from it except the default config directory named above.
