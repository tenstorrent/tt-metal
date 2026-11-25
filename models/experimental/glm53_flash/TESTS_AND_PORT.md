# GLM-5.3-Flash: unit tests + Blackhole → Wormhole port

Target hardware: **Wormhole Galaxy**, 32 chips (`tt-galaxy-wh L`), host `j11glx01`.
Source: `aperezvicente-TT/glm-5.3-flash` @ `18457222c2c` (title: *"GLM-5.3-Flash on Blackhole"*),
installed at `models/experimental/glm53_flash/`.

## Result

```
PYTHONPATH=$PWD ./python_env/bin/python -m pytest \
    models/experimental/glm53_flash/tests/ -q
→ 254 passed, 2 skipped, 7 xfailed   (~4 min; test_full_layer.py is ~3 min of it)
```

| test file | module(s) | tests | status |
|---|---|---|---|
| `test_config_precision.py` | `config`, `precision`, `indexer` guard | 14 | pass |
| `test_mlp_moe.py` | `mlp`, `moe` | 10 | pass |
| `test_hyperconnection.py` | `hyperconnection` | 7 | pass |
| `test_mla.py` | `mla` | 5 | pass |
| `test_kda.py` | `kda` | 6 pass / 3 xfail | see blocker |
| `test_decoder_layer.py` | `decoder_layer` | 12 | pass |
| `test_kda_block.py` | `kda_block/*` | 14 pass / 2 xfail | see blocker (block prefill) |
| `test_terminal.py` | `kda_block/terminal`, `model.Terminal` | 14 pass / 1 xfail | lm_head via DSLinear blocked |
| `test_functional_decoder.py` | `functional_decoder` (full width, all 3 layer kinds) | 21 pass / 1 xfail | see blocker (KDA prefill) |
| `test_weights.py` | `weights` | 13 | pass |
| `test_sampling.py` | `sampling` | 69 | pass (host) |
| `test_host_modules.py` | `generator` planning, `prefix_cache` | 38 | pass (host) |
| `test_tokenizer_chat.py` | `tokenizer_chat` | 24 | pass (host, needs the snapshot) |
| `test_full_layer.py` | real layers 0 / 3 / 4, real weights, real prompts | 7 pass / 2 skip | KDA prefill skipped (blocker); needs the checkpoint |

## Full layers on real weights (`test_full_layer.py`)

Nothing scaled down: `FunctionalDecoder.from_state_dict` on the checkpoint's fp8-dequantized weights
(`reference/hf_layer_reference.py::build_layer`) with the checkpoint config — 288 experts top-8
(bfloat4_b), shared expert, 64×128 KDA, 64-head MLA, dense I = 12288. Inputs are the real residual
streams: two chat prompts embedded and run through HF layers 0…4 on the host (fp32 math), so layer
L's input is HF layer L−1's output. Decode for both users batched; MLA also prefill 20 + decode 4.

| layer | vs HF with bf4-rounded experts (implementation) | vs exact HF (deployment) |
|---|---|---|
| 0 `kda_dense` | output PCC median 0.99991, min 0.99986 | same (no experts) |
| 3 `mla_moe` | 0.999987, min 0.99994 | 0.9979, min 0.9944 |
| 4 `kda_moe` | 0.999999, min 0.99997 | 0.99990, min 0.9969 |

Layer 3's deployment gap is entirely the 4-bit expert storage: HF's own MoE with bf4-rounded
experts scores the same 0.996 against the exact one, the device MoE matches that bf4 HF MoE at
0.99999, the device MLA alone matches HF at 0.99987, and HF's DSA mask at this length is exactly
causal (dense-exact holds). HF's bf16 path and bf8 projection rounding each cost < 1e-4.

The module tests from the first round grade against a torch reference written from the formula in
each module's own docstring, so a failure points at one module rather than at "the model is wrong".
Everything from `test_kda_block.py` on grades against transformers' own GLM-5.3-Flash code
(`Glm5NextTextDecoderLayer` / `Glm5NextTextModel`, the logits warpers, `AutoTokenizer` + the
checkpoint's chat template).

## Remaining modules (`test_terminal` … `test_tokenizer_chat`)

- **Terminal**: `GLMEmbedding` + `GLMHead` (mean of streams → RMSNorm → lm_head) match HF exactly /
  at PCC > 0.999. Embedding → 2 KDA blocks → head decodes the same tokens as `Glm5NextTextModel`.
  The upstream `Terminal.embed_streams` / `hidden` work on WH; its lm_head imports DSLinear (xfail).
- **FunctionalDecoder** at the real width (hidden 4096, 64×128 KDA heads, 64-head MLA, only the FFN
  shrunk): decode for `kda_dense` / `kda_moe` / `mla_moe` × 1–2 users, inactive slots (`cur_pos = -1`),
  `reset_slot`, `snapshot_state`/`restore_state` (bit-exact), decode trace replay == eager, MLA
  prefill (5 / 64 / 70 tokens) then decode, cache shapes, page-table and position validation.
- **weights**: every HF tensor is accounted for, and each converted layout is checked by its meaning
  (`x @ W == Linear(x)`, conv taps reproduce the causal conv, `W_uk`/`W_uv` reproduce `kv_b_proj`).
- **sampling**: `warp` equals transformers' Temperature/TopK/TopP warpers bit-exactly over 36
  combinations; seeded sampler reproducibility, state restore and empirical distribution.
- **generator / prefix_cache**: chunk planning, page-table widths, the slice-boundary rule (every
  branch of its docstring), and the prefix store's LRU / host tier / pins / trailing snapshots.
- **tokenizer_chat**: encode/decode and chat rendering (reasoning effort, clear_thinking, tools,
  continue_final_message) equal transformers'; tool calls survive a render → parse round trip.

Findings from these (tests encode the behaviour; no source changed):

- **bf4 routed experts cost ~0.02 PCC at D = 4096 on random weights.** Every MoE test
  (`test_mlp_moe`, `test_kda_block`, `test_terminal`, `test_functional_decoder`) loads the routed
  experts as **bfloat4_b**, as deployed, asserts that dtype on the device tensors, and grades against
  reference experts rounded to their bf4 values (`test_kda_block.py::bf4_round_experts`; exact to
  0.99996 then). `test_bf4_expert_storage_budget` keeps the unrounded comparison (> 0.98) so the
  storage cost stays visible.
- **Router near-ties flip experts.** A token whose 2nd/3rd corrected router scores differ by 0.002
  picks the other expert under bf16 inputs (PCC 0.86 on that token, 0.9997 on the rest). Prefill
  checks skip tokens within `ROUTER_TIE_MARGIN` of a tie.
- **The device sampler's candidate path is not exact under ties at the k-th logit** — transformers
  keeps all tied tokens, the candidate path keeps exactly k. `SAMPLER_PLAN` says "exact for top_k ≤
  32"; that holds only without a tie (`test_candidate_truncates_ties_at_the_kth_logit`).
- **`resolve_snapshot()` only finds revision `eb9eb208…`.** The local hub snapshot is `46a050be…`,
  so the default lookup fails; the tests pass the path explicitly (or set `$GLM53_HF_SNAPSHOT`).

## Blocked on this build (reported, not tested)

| module(s) | blocker |
|---|---|
| `optimized_decoder` (DSLinear), `mk_*`, `pmk_*`, `ds_mm`, `mk_readers`, `model.Terminal` lm_head | `ttnn._ttnn.operations.matmul.DramReaderPlacement` (DRAM prefetcher) not in this build / not on WH |
| `spec_mtp` (MTP layer) | TP megakernel — same prefetcher dependency |
| KDA prefill (`kda`, `kda_block`, `functional_decoder`) | `prepare_chunk_recurrence` needs 33 DFBs/core, Gen1 allows 32 |
| sparse DSA indexer | not implemented upstream (only the dense-exact guard, kv ≤ 2051) |
| vision tower | no device implementation |

## KDA decoder block (`tt/kda_block/`)

`GLMKDADecoderLayer` is the `kda_dense` / `kda_moe` block in the shape of the DeepSeek-V4-Flash decoder
blocks from metal_2 (`models/experimental/deepseek_v4_flash/tt/{decode,prefill}/decoder_layer.py`):
HF-named weight dict, optional `WeightCache`, `[B, S, hc, D]` residual streams,
`post, comb, collapsed = hc(streams)` / `mix_streams(post, comb, out, streams)`.

| piece | source |
|---|---|
| `tt/ds_v4/common.py`, `tt/ds_v4/weight_cache.py` | metal_2 DSv4, verbatim |
| `tt/ds_v4/layers.py::Linear`, `tt/ds_v4/streams.py` | metal_2 DSv4 (`Linear`, `_strip_prefix`, `flatten_streams`, `load_norm_gamma`, `rows_from_tokens` generalised to `[B, S]`) |
| `kda_block/hyperconnection.py` | DSv4 prefill hyper-connection structure; `fused_hyperconnection` → `mhc_split_sinkhorn` + matmuls, `deepseek.mix_streams` → composite (neither DSv4 op is in this build) |
| `kda_block/ffn.py` | dense MLP = `DeepSeekV4MLP`'s non-prefetch path with the clamp; MoE = this repo's sparse_matmul `MoE` (DSv4's needs `moe.fused_experts`) |
| `kda_block/attention.py` | HF KDA weights → this repo's `KDA` |

No weight goes through the DRAM prefetcher (`decode_prefetch`, `LinearDecode*` are not used).
Decode runs on Wormhole; block prefill hits the same `prepare_chunk_recurrence` blocker as `test_kda.py`.

## The port: what actually needed changing

The Python is nearly arch-clean — it calls `ttnn.init_device_compute_kernel_config(device.arch(), …)`
rather than hard-coding a config, so it produces a `WormholeComputeKernelConfig` on this box
(`test_compute_kernel_configs_are_arch_typed` pins that). Four things did need work:

### 1. `kda.py` used a deleted op (fixed)

`kda.py` imported `models.demos.deepseek_v3_d_p.tt.kda.chronological_selections`, which was **removed
from `main`** by *"[KDA] Fuse the QKV split, convolution and history exchange into one op"* (#60450).
The replacement, `qkv_causal_conv1d_silu_with_carry`, does the sequence-parallel history exchange
itself and returns the replacement carry, so it subsumes both the old explicit selection and the
plain conv call. Rewrote the prefill block to use it, including the persistent
`[1, 12, Q+K+V]` ROW_MAJOR exchange buffer and three `L1_SMALL` global semaphores on the op's
exchange core.

Verified by `test_conv_with_carry_matches_plain_conv`: the new path is **bit-identical**
(`torch.equal`, not PCC) to the plain `qkv_causal_conv1d_silu`, and its carry matches the history the
removed selection used to produce. This is the regression test for the port.

### 2. `prepare_chunk_recurrence` was Blackhole-gated (gate removed, then a hardware wall)

`prepare_chunk_recurrence_device_operation.cpp` had

```cpp
TT_FATAL(arch == tt::ARCH::BLACKHOLE, "{} is only supported on Blackhole, got {}", …)
```

Its kernels contain no Blackhole-specific instructions (the TF32 mentions are precision notes about
the FP32 DST path, not a required opcode), so I widened the guard to accept `WORMHOLE_B0` and
rebuilt `ttnn` incrementally (`ninja _ttnn.so`, one file recompiled).

Removing the gate exposed **two latent bugs in the module's own call**, both of which had been
unreachable behind the arch check. Both fixed:

| bug | fix |
|---|---|
| `beta` passed as **fp32**; the op requires BF16 (`…device_operation.cpp:59`) | cast at the boundary |
| `beta` laid out `[H, n_chunks, CHUNK, 1]`; the op wants rank-3 `[1, n_chunks*32, num_heads]` (`:112`) — a leftover from the older op signature | reshape to `(1, c, h)` |

After both, the call reaches **kernel placement**, where it hits a real hardware limit:

```
TT_THROW: ProgramSpec 'prepare_chunk_recurrence' places 33 DataflowBufferSpecs
          on node (3, 0), but Gen1 supports at most 32 device slots per core
```

Wormhole is Gen1: **32 DFB slots per core, hard limit**. The op needs 33 distinct per-core streams.
Adding cores does not help — every stream is placed on every core that has work, so the *per-core*
count is what overflows. **This is unresolved** and is why KDA *prefill* remains Blackhole-only.
KDA *decode* is a different op path and works on Wormhole (tested).

Lifting it is kernel surgery, not porting. Two viable directions are recorded at
`tests/test_kda.py::PREP_BLOCKER`:
- alias buffers whose lifetimes are disjoint (`eye`/`tril`/`ones`/`block_masks` are constants consumed
  early; `workspace_0`/`workspace_1` are disjoint in time);
- split the op so the `T_inv` construction runs as its own program, keeping only its output live —
  that removes the workspaces and masks from the prep program entirely.

### 3. Nothing else

`mlp`, `moe`, `mla`, `hyperconnection`, `weights`, `config`, `precision`, `indexer` all imported and
ran unmodified. The `mhc_split_sinkhorn`, `fast_reduce_nc`, `paged_*`, `chunked_sdpa`,
`paged_flash_multi_latent_attention_decode`, `indexer_score_dsa`, `qkv_causal_conv1d_silu`,
`sigmoid_gated_rms_norm` and `fused_recurrent_gated_delta_rule` ops all work on Wormhole.

## What the tests pin

Beyond "it runs", these check properties that a wrong-but-plausible implementation would violate:

- **Swiglu clamp is real** — a gate and up both at 100 with `limit=2` must yield `silu(2)*2`, not
  `silu(100)*100` (`test_swiglu_clamped_actually_clamps`).
- **Router selects the same expert set as fp32** — with GLM's real bias shape (offset ~34.5, spread
  ~0.14), which is the case where a bf16 ranking row silently flips the top-8
  (`test_router_matches_reference` asserts the *set*, not just PCC).
- **Sinkhorn output is doubly stochastic** — rows and columns sum to 1; that is the constraint's
  entire purpose (`test_sinkhorn_is_doubly_stochastic`).
- **The MLA absorber identity holds** — `W_uk^T q · c == q · (W_uk c)`; this is only exact because
  `qk_rope_head_dim == 0`, so the test is really asserting the NoPE premise.
- **`pre` doesn't mutate its input** — the layer passes streams by reference across the step.
- **KDA state keeps its buffer addresses** — a captured decode trace depends on in-place updates.
- **The residual order is attn-then-ffn** — a swapped pair would still run and still produce correct
  shapes, so the order is asserted explicitly.

## Notes for whoever picks this up

- The xfails are `strict=True`, so if someone lifts the DFB limit they will flip to
  "unexpectedly passed" and force the marker off. That is intentional.
- `ttnn` in this checkout loads `build/lib/_ttnncpp.so`, not `ttnn/ttnn/_ttnn.so`; after a C++ change
  both need installing or the old behaviour persists (this cost me one debugging cycle).
- `models/demos/deepseek_v3_d_p/reference/glm_5_3/` on `main` is a *different* GLM-5.3 (78 layers, no
  KDA, no mHC). This tree is the real 45-layer Flash checkpoint.
- `MLA.decode` hard-codes `decode_heads = 32`; a test instance must therefore have a head count that
  is a multiple of 32. `MLA.prefill` requires chunk length and page-table width to be multiples of 64
  and 32 respectively.
- The upstream repo has **no tests, no README and no docs** — these tests are the first executable
  specification of that code. Everything above was found by running it, not reading it.
