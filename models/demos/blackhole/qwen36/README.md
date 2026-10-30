# Qwen3.5 / Qwen3.6 / Qwen3.8 on Blackhole

This directory implements Tenstorrent Blackhole inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6/3.8 family. The same code
path serves every checkpoint below:

| Model           | `HF_MODEL`             | Mesh / `MESH_DEVICE` | Parallelism           |
| --------------- | ---------------------- | -------------------- | --------------------- |
| Qwen3.5-9B      | `Qwen/Qwen3.5-9B`      | single P150 — `P150` | single device         |
| Qwen3.5-27B     | `Qwen/Qwen3.5-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel |
| Qwen3.6-27B     | `Qwen/Qwen3.6-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel |
| Qwen3.6-27B     | `Qwen/Qwen3.6-27B`     | P150x8 — `P150x8`    | 8-way tensor parallel |
| Qwen3.8-27B     | `Qwen/Qwen3.8-27B`     | P150x4 — `P150x4`    | 4-way tensor parallel |
| Qwen3.8-27B     | `Qwen/Qwen3.8-27B`     | P150x8 — `P150x8`    | 8-way tensor parallel |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | P150x4 — `P150x4`    | 4-way TP + sparse MoE |

The **35B-A3B** is the sparse Mixture-of-Experts member of the family (`qwen3_5_moe`:
256 routed experts, top-8, plus a gated shared expert on every layer). Every layer's
dense SwiGLU MLP is replaced by the sparse MoE block in `tt/moe/`; dispatch is
config-driven (`args.is_moe_layer`), so on the dense 9B/27B `num_experts == 0` and the
dense MLP path is byte-for-byte unchanged.

- The **9B** runs on a **single Blackhole P150** device. It uses the validated
  single-device forward path (no collectives).
- The **27B** variants (Qwen3.5-27B, Qwen3.6-27B and Qwen3.8-27B) run on a
  **P150x4** (a `(1, 4)` Blackhole mesh) using **4-way tensor parallelism (TP)**.
  The TP path needs `FABRIC_1D` for the cross-device collectives (all-reduce /
  reduce-scatter) and a trace region for the captured chunk-outer prefill trace.
- **Qwen3.6-27B and Qwen3.8-27B also run at TP=8** on a `(1, 8)` mesh (`P150x8`),
  the Galaxy configuration. Because these checkpoints have only **4 KV heads**,
  TP=8 cannot give each device its own head: each head is instead **replicated
  across the device pair holding its GQA query group** (devices 0-1 share KV
  head 0, 2-3 head 1, and so on), so `n_local_kv_heads` is 1 at both TP=4 and
  TP=8 and the whole runtime KV path is unchanged. See
  `tp_common.replicate_kv_weight` and `ModelArgs.SUPPORTS_KV_REPLICATION`.

Everything model-specific (hybrid layer dispatch, DeltaNet head/conv dims,
partial rotary factor, vocab, layer count) is read from the parsed HF config, so
the single code base adapts to each checkpoint. The device count alone
(`num_devices > 1`) switches between the single-device and TP code paths — see
`tt/model_config.py` and `tt/tp_common.py`.

## Architecture

Assembly: `tok_embeddings → N × Qwen36DecoderLayer → RMSNorm → LM Head`.

Each model interleaves two attention block types (read from the HF
`layer_types`): **Gated DeltaNet** (linear-attention, recurrent + causal conv
state) layers and **Gated Full Attention** (paged KV cache) layers. The 9B has
32 layers (24 DeltaNet + 8 full-attention). Qwen3.5 uses zero-centered RMSNorm
everywhere and **partial** RoPE (only a fraction of each head is rotated).

## Environment setup

Before running **any** test, export the two environment variables that select
the checkpoint and the device mesh.

**Checkpoint — Qwen3.8-27B:**

```bash
export HF_MODEL=Qwen/Qwen3.8-27B
```

Then pick the mesh that matches your hardware.

**QB2 or P150x4 (LB) — 4-way TP:**

```bash
export MESH_DEVICE=P150x4
```

**Galaxy (Glx) — 8-way TP:**

```bash
export MESH_DEVICE=P150x8
```

On Galaxy, also prefix each pytest command with `TT_VISIBLE_DEVICES` to pin the
run to the eight devices that form the mesh:

```bash
TT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"
```

**Galaxy (Glx) — 8-way TP x 4-way DP:**

`test_demo_text_dp` opens the whole 32-chip Galaxy as a (4, 8) mesh
(`FABRIC_1D_RING`), splits it into four (1, 8) TP=8 submeshes, and runs one model
replica per submesh (one user each); decode trace replays are enqueued across the
groups in lockstep.

```bash
export MESH_DEVICE=BHGLX
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "dp4_traced_128"      # same prompt on all 4 groups (outputs must match)
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "dp4_distinct_128"    # a different prompt per group
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "tp8_glx_traced_128"  # a single TP=8 replica on one (1, 8) row
```

Measured on a Blackhole Galaxy (Qwen3.8-27B, 128-token prompt): 41.7 tok/s/user at
TP=8, 166.6 tok/s aggregate at TP=8 x DP=4, TTFT ~0.06 s.

**Other checkpoints:**

```bash
# Qwen3.6-27B — P150x4 (QB2 / LB) or P150x8 (Galaxy)
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=P150x4

# Qwen3.5-27B — P150x4
export HF_MODEL=Qwen/Qwen3.5-27B
export MESH_DEVICE=P150x4

# Qwen3.5-9B — single P150
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=P150
```

**35B-A3B (P150x4, sparse MoE):**

```bash
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
export MESH_DEVICE=P150x4
```

`HF_MODEL` is the single source of truth for the checkpoint — it may be a Hugging
Face hub id (resolved via `snapshot_download`) or a local checkpoint directory.
`MESH_DEVICE` selects the mesh shape (`P150` → `(1,1)`, `P150x4` → `(1,4)`,
`P150x8` → `(1,8)`).

Optional flags:

```bash
# Also cast the prefill SDPA Q to BF8 (the KV cache is BF8 by default; slightly lower precision).
export QWEN_SDPA_BF8=1
```

Default decode settings (TP paths, 27B), each with an env var that reverts it:

| Default | Revert / alternatives |
|---|---|
| GDN decode through `ttnn.experimental.kda.gdn_decode_step` (conv + gating + recurrence + gated norm in one op) for every `max_batch_size` from 1 to 32 | `QWEN36_GDN_DECODE_STEP_OP=0` (single-user fused kernels at B=1, the older batched paths above) |
| exact-fp32 recurrent-state update in that op | `QWEN36_GDN_STEP_FP32_STATE=0` (FPU path) |
| dense-MLP gate/up weights in BFP8 | `QWEN36_MLP_GATEUP_BFP8=0` (BFP4) |
| dense-MLP matmuls at HiFi2 | `QWEN36_MLP_FIDELITY=lofi\|hifi3\|hifi4` |
| 8 KB fabric packet payload on multi-device meshes (demo and vLLM) | `QWEN36_FABRIC_PAYLOAD_BYTES=0` (fabric default, 4352 B) or another size |
| BF8 paged KV cache (demo and vLLM; prefill Q stays BF16) | `QWEN36_KV_BF8=0` (BF16 cache) |

With all of the defaults above, Qwen3.8-27B served through vLLM on a P150x8 ring (`max_num_seqs` 32, thinking on,
temperature 1.0 / top_p 0.95 / top_k 20, `qwen38_handoff/eval/task_eval.py`) passes the task evals (reference score
inside the question-clustered 95% CI):

| Eval | Score [95% CI] | Reference |
|---|---|---|
| AIME 2026, 4 repeats, 65k max tokens | 94.17 [86.67, 99.17] | 98.13 |
| GPQA-Diamond, 2 repeats, 65k max tokens | 89.14 [85.10, 92.93] | 89.2 |
| MMLU-Pro, 500-question subset | 85.20 [82.00, 88.20] | 85.54 |

`accuracy_512` passes as well. BFP4 MLP-down + attention on top of these defaults failed AIME.


## End-to-end demo test (`demo/text_demo.py`)

The e2e text-generation test lives in `demo/text_demo.py`. It is a single
parametrized test (`test_demo_text`) covering a range of input sequence lengths
(ISLs): 128, 4k, 8k, 16k, 32k, 64k, 128k, and 256k tokens. Each ISL runs prefill
+ decode and validates output (non-degenerate generation) and per-ISL
performance gates (TTFT and decode tok/s).

Two execution variants exist per ISL, identified by the test id prefix:

- **`traced_*`** — captures the prefill (chunk-outer) and decode forward passes
  as device traces and replays them. This is the **preferred** path and the one
  vLLM serves; run these by default.
- **`paged_*`** — non-traced paged path, useful as an eager reference/fallback.

Single-user TP decode runs **MTP speculative decode by default**. Set
`QWEN36_SPEC=0` to turn it off and fall back to plain single-token decode.

Multi-user TP decode with up to 8 users also runs MTP speculative decode by
default. All users' verify rows share one 32-row tile, so the draft length K
shrinks as the batch grows (B=2: K=11, B=4: K=7, B=8: K=3); batches above 8 run
plain batched decode (the automatic policy keeps K >= 3). The traced draft chain and the traced reseed run for every
batch size. Knobs:

- `QWEN36_SPEC=0` — plain decode.
- `QWEN36_SPEC_DRAFT_LEN=<K>` — override K (must keep B x (K+1) <= 32).
- `QWEN36_SPEC_EAGER_DRAFTER=1` — run the drafter eagerly instead of as a recorded
  trace (default 0; for A/B, same tokens).
- `QWEN36_SPEC_TIMING=1` — per-phase timing lines.
- `QWEN36_SPEC_BATCH_DISTINCT=1` — distinct prompts per user in the batched demo
  cases.

```bash
# Batched spec decode demo cases (B=2/4/8 at 128 tokens, B=4/8 at 4k, B=8 at 8k/16k)
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "spec_128_b8"
# Batched spec decode unit tests (lossless vs plain greedy, determinism, traced == eager)
pytest models/demos/blackhole/qwen36/tests/test_spec_batched.py
```

Run the preferred traced cases (the env vars above must already be exported):

```bash
# All traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"

# A single ISL, e.g. the short 128-token traced case
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128 and not traced_128k"

# Medium / long traced ISLs
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_4k"
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_64k"
```

The **same command works for every checkpoint** — only the exported `HF_MODEL` /
`MESH_DEVICE` differ. On a single device the test takes the validated 9B path; on
the `(1,4)` or `(1,8)` mesh it routes through the TP chunk-outer traced prefill +
paged traced decode path automatically (the sparse MoE block is selected per layer
from the config, transparent to the demo). On Galaxy, prefix each command with
`TT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8`.

> Long-context cases (64k+) download a public-domain corpus (Frankenstein, War
> and Peace) on first run and cache it under `demo/sample_prompts/.context_cache`.

### Reference accuracy (`accuracy_512`)

Scores top-1/top-5 against HuggingFace by teacher-forcing a committed reference;
the floors live in `models/model_targets.yaml`.

```bash
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "accuracy_512"
```

When a checkpoint changes, regenerate its reference on CPU (no device needed):

```bash
python models/tt_transformers/tests/generate_reference_hf.py --model Qwen/Qwen3.6-27B \
    --output_file models/tt_transformers/tests/reference_outputs/Qwen3.6-27B.refpt
```

## Automatic prefix caching (vLLM serving, batched TP path)

vLLM's APC reuses the paged attention KV blocks; the 48 Gated-DeltaNet layers carry a non-paged recurrent state, so the
model keeps its own LRU of GDN-state snapshots (`tt/prefix_cache.py`, `GdnPrefixStateCache`: `rec_state` + `conv_carry`
of every GDN layer, device-to-device copies of the B=1 prefill scratch). Serve with `--enable-prefix-caching
--block-size 64`. On a hit vLLM passes the full prompt, the full page-table row and `start_pos` (cached tokens, a
multiple of 64); `prefill_paged_slots` resumes at the largest snapshot position `s <= floor(start_pos/128)*128` whose
token digest matches, restores the GDN state, and prefills only `[s, L)` at absolute positions (chunk trace with an
absolute chunk start, then the short bucket trace with `chunk_start`). Without a matching snapshot the prompt is
recomputed from 0 (correct, slower).

Snapshot points (all multiples of 128, because the SDPA chunk offset uses q_chunk=128, and always `<= L-3` so every
segment keeps >= 3 tokens for the conv new-state selection): the vLLM-hit position when we have no snapshot there
("learn"), the end of long prompts (`((L-3)//128)*128`, prompts >= `QWEN36_PREFIX_SNAPSHOT_MIN_LEN`, which splits the prefill
there), for shorter prompts with no other save point the last full 2048-token chunk boundary `p = s + num_full*2048`
when `p > s` and `p <= L-3` (saved between two chunk-trace replays, no split, about 3 ms), and every
`QWEN36_PREFIX_SNAPSHOT_STRIDE` tokens after the resume point. A prompt longer than 2048 tokens whose last chunk would leave a 1-2 token remainder gets an extra split 128 tokens before the last full-chunk boundary so the final segment runs on the short trace (not eagerly). With the cache off, or no hit and no snapshot to take,
the prefill is the unchanged path. Only the batched slot path (`max_num_seqs > 1`, text) uses snapshots; the B=1 TP and
single-device paths ignore `start_pos` and recompute from 0.

| Env var | Default | Meaning |
|---|---|---|
| `QWEN36_PREFIX_CACHE` | `1` | advertise `supports_prefix_caching` and build the snapshot cache (`0` disables) |
| `QWEN36_PREFIX_CACHE_SLOTS` | `32` | snapshot slots, allocated before trace capture: about 19 MB of DRAM per device per slot at TP=8, about 38 MB at TP=4. 32 fits next to the default BFP8 gate/up weights and the 1.05M-token KV pool on P150x4; 64 runs out of DRAM there. On the BH Galaxy (TP8 rows, 31.9 GiB per device) one slot is 21.75 MiB per device (48 x (384 KiB fp32 state + 80 KiB tile-padded conv carry)); the production configuration (525312-token KV pool per rank, every prefill and verify trace) allocates 13.1 GiB per device with 64 slots, so 256 slots (+4.1 GiB) leave about 13 GiB free |
| `QWEN36_PREFIX_CACHE_FAST_ALLOC` | `0` | `1`: allocate the snapshot slots (and the host-tier staging set) on device, uninitialized (`ttnn.allocate_tensor_on_device`), instead of uploading host zeros tensor by tensor (measured on a TP=8 Galaxy row: 25 ms per slot, so 1.6 s per 64 slots and 6.5 s for 256, against 1.3 ms per slot on device). A slot is never read before a save writes it |
| `QWEN36_PREFIX_SNAPSHOT_MIN_LEN` | `8192` | minimum prompt length for the end-of-prompt snapshot (a prefill split); shorter prompts save at the last full-chunk boundary instead |
| `QWEN36_PREFIX_END_AT_CAP` | `0` | `1` (MTP spec path): the end-of-prompt snapshot sits at the resume cap of the next request that extends the prompt, `((floor(L/64)*64 - 1)//128)*128`, instead of `((L-3)//128)*128`. The default position is one 128-block above that cap whenever `L mod 128` is in `[3, 63]`, so about half of the next turns miss it and resume from an older snapshot. Never more save points than today (no extra split; a stride point at the old position goes with it); a next turn that matched further resumes at most 128 tokens below its cap |
| `QWEN36_PREFIX_SNAPSHOT_STRIDE` | `8192` | snapshot every N tokens (chunk boundaries, relative to the resume point) |
| `QWEN36_PREFIX_CACHE_DEBUG` | `0` | `1` logs one line per prefix-cached prefill (hit position, saves, stats) |
| `QWEN36_PREFIX_CACHE_VALUE_EVICT` | `0` | `1`: value-aware eviction. End-of-prompt checkpoints (and the last-full-chunk point of shorter prompts) and snapshots hit >= `QWEN36_PREFIX_CACHE_PROTECT_HITS` (`2`) times are evicted only after every other unpinned snapshot; at most `QWEN36_PREFIX_CACHE_PROTECT_FRAC` (`0.5`) of the slots are protected (the most recently used ones). Pure stride snapshots still go first |
| `QWEN36_PREFIX_HOST_SLOTS` | `0` | `N > 0`: host-RAM spill tier of N snapshots (about 150 MB of host RAM each at TP=8, per DP rank). An evicted snapshot is read to host (exact bytes of every device) before its slot is reused; a later hit writes it back through one extra device staging slot (allocated before trace capture). A spill is a blocking device read inside the prefill |
| `QWEN36_PREFIX_HOST_SPILL_MIN_POS` | `2048` | only snapshots at positions >= this are spilled (shorter ones are cheaper to recompute) |
| `QWEN36_PREFIX_HOST_MIN_GAIN` | `1024` | a host snapshot is used only when it resumes at least this many tokens later than the best device snapshot (or than 0) |
| `QWEN36_PREFIX_HOST_SPILL_LOW` | `0` | `1` also spills pure stride snapshots |
| `QWEN36_PREFIX_SUMMARY_EVERY` | `1000` | spec engine: one INFO summary line every N prefills, with the true resume metrics: `vllm_cached_tok` (what vLLM counts as cached), `gdn_resumed_tok` (where the GDN state really resumed), `hidden_recompute_tok` (cached by vLLM but recomputed), `below_cap_tok` (recomputed because no snapshot existed at the MTP-safe cap), and the tier (`from_device` / `from_host` / `from_0`) |

`QWEN36_PREFIX_SNAPSHOT_MIN_LEN` also controls end-of-prompt checkpoints for chat-sized prompts: e.g. `512` takes the
`((L-3)//128)*128` snapshot for every prompt >= 512 tokens (one extra short-trace segment per prefill), so the next turn
of the conversation resumes within its own new tokens instead of at the last 2048-token chunk boundary or at 0.

## Chunked prefill (vLLM serving, batched TP path)

Opt-in: with `QWEN36_CHUNKED_PREFILL=1` the model declares `supports_chunked_prefill` (serve it with
`--no-enable-prefix-caching`, see below). With `--max-num-batched-tokens` below `max_model_len`, vLLM prefills a
longer prompt over several engine steps, and the plugin runs decode steps for the other requests in between (its
decode interleave: at most `decode_interleave_prefill_steps` (2) prefill steps, then `decode_interleave_decode_steps` (1)
decode step, both in `additional_config['tt']`). Without it, one prefill step holds every waiting prompt that fits the
budget, which tt-inference-server sets to `max_context`, and running requests get no tokens until the step ends.

Each chunk's row holds the prompt from position 0, `prompt_lens` is the chunk end and `start_pos` the tokens already
prefilled; non-final chunk ends are 128-aligned (the plugin's `prefill_chunk_alignment`). The model is not told which
chunk is the last. A row that may be non-final (its end is 128-aligned and the step used its whole token budget, or
`long_prefill_token_threshold` cut it) snapshots its GDN state at its end into a small continuation store
(`model._chunk_state_cache`, the `GdnPrefixStateCache` class with its own slots, independent of APC), and the next chunk
resumes there through the APC planner. A 1-2 token final chunk resumes from the previous chunk's snapshot; a missed
snapshot is recomputed from an earlier one or from 0 (correct, slower). Multimodal requests recompute from 0 on every
chunk (vLLM never splits an image across chunks). The B=1 TP path recomputes from 0 on every chunk: keep
`--max-num-batched-tokens >= max_model_len` at `max_num_seqs 1`. The store is allocated (before trace capture) only when
vLLM can split a prompt: chunked prefill on and the budget below `max_model_len`. With chunked prefill on, vLLM refuses a
budget below the largest image/video item (16384 tokens at the default processor settings: "Chunked MM input disabled
but max_tokens_per_mm_item ... is larger than max_num_batched_tokens"); go lower only for text-only serving
(`--limit-mm-per-prompt '{"image": 0, "video": 0}'`).

Chunked prefill and vLLM prefix caching do not start together on the TT plugin today (vllm-tt-plugin d84049e and main
6afed54): with chunked prefill on, the plugin sets `disable_chunked_mm_input=True`, and vLLM's hybrid-model prefix
caching (`mamba_cache_mode='align'`) rejects that in `validate_block_size` ("Chunked MM input is required because we need
the flexibility to schedule a multiple of block_size tokens ..."). Serve chunked prefill with `--no-enable-prefix-caching`
until the plugin relaxes that for TT-owned recurrent state; the continuation store does not need APC.

| Env var | Default | Meaning |
|---|---|---|
| `QWEN36_CHUNKED_PREFILL` | `0` | `1` advertises `supports_chunked_prefill` and builds the continuation store (needs `--no-enable-prefix-caching`) |
| `QWEN36_CHUNK_STATE_SLOTS` | `4` | continuation snapshot slots (about 19 MB of DRAM per device per slot at TP=8) |

## Multimodal serving: vision buckets and `max_pixels` (vLLM)

The vision tower runs on padded patch buckets (ceil-to-2048 of the patch count + 1 step) that must be compiled
**before** prefill/decode traces are captured; a program first dispatched after capture can be clobbered by a trace
replay and hangs the device (seen: `tt_all_reduce`). `warmup_vision` therefore warms `QWEN36_VISION_WARMUP_GRIDS`
(default `1x32x32;1x48x64;1x64x64;1x80x80`, buckets up to 8192 patches) in the eager warmup phase.

- **Always serve with `--mm-processor-kwargs '{"max_pixels": 1048576}'`** (or your limit). `initialize_vllm_model` reads
  it from the vLLM config (`model_config.mm_processor_kwargs`) and `warmup_vision` also warms the bucket covering
  `max_pixels / patch_size^2` patches (1 MP -> 4096 patches -> bucket 6144, already in the defaults; e.g. 3 MP appends
  bucket 12288), so every input vLLM can accept is pre-compiled. Video is bounded by the same figure (conservatively).
- If `max_pixels` is unset (vLLM default ~16 MP, ~65k patches) a WARNING is logged at warmup, and any item whose bucket
  exceeds the largest warmed one is **rejected** with a `ValueError` (patch count, warmed max, advice) instead of being
  run. This still fails that request's step (the engine reports the error) but avoids the device hang.
- Test: `tests/test_vllm_vision_guard.py`.

## Tests

There are two tiers of tests under `tests/`.

### Single-device unit / component tests — **9B**

These run on a single P150 against the 9B checkpoint (the conftests
`setdefault HF_MODEL=Qwen/Qwen3.5-9B`). They validate each component's forward
pass against a torch reference (PCC) — see `tests/pcc_thresholds.json` for the
per-test thresholds.

`tests/unit/` (component PCC vs torch):

| Test                       | Validates                                            |
| -------------------------- | ---------------------------------------------------- |
| `test_embedding.py`        | token embedding                                      |
| `test_rms_norm.py`         | zero-centered RMSNorm (the "+1" fold)                |
| `test_rope.py`             | partial-rotary RoPE (host freqs + on-device lookup)  |
| `test_mlp.py`              | single-device SwiGLU MLP (layer 0)                   |
| `test_moe.py`              | single-device sparse MoE MLP (MoE checkpoint only; decode + prefill) |
| `test_attention.py`        | single-device gated full attention (layer 3)         |
| `test_gdn.py`              | single-device Gated DeltaNet (layer 0)               |
| `test_lm_head.py`          | LM head logits (bf8 vs bf16)                          |
| `test_layer.py`            | full decoder-block sanity (no NaN/Inf, non-constant) |
| `test_model.py`            | Generator decode contract: traced vs paged decode    |
| `test_substate.py`         | weight `substate` helper (pure CPU, no device)       |

`tests/` (single-device, also 9B):

| Test                     | Validates                                                  |
| ------------------------ | ---------------------------------------------------------- |
| `test_prefill.py`        | masked fixed-bucket + chunk-outer prefill vs `prefill_paged` |
| `test_weight_mapping.py` | HF → internal weight key remapping (pure CPU)              |

Run the 9B unit suite (with `HF_MODEL=Qwen/Qwen3.5-9B`, `MESH_DEVICE=P150`):

```bash
pytest models/demos/blackhole/qwen36/tests/unit/ -v -s
pytest models/demos/blackhole/qwen36/tests/test_prefill.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_weight_mapping.py -v -s
```

> `test_prefill.py` auto-skips cases longer than `--max-prefill` (default 8192).
> Raise it to exercise long-context prefill, e.g. `--max-prefill 131072`.

### Tensor-parallel tests — **27B (P150x4 / P150x8)**

The `*_tp` tests exercise the multi-device TP path and default to the 27B
checkpoint. They must run on the `(1,4)` mesh — or the `(1,8)` Galaxy mesh — with
`FABRIC_1D` (the `parametrize_mesh_tp` helper wires this from `MESH_DEVICE`). PCC
thresholds are in `tests/pcc_thresholds.json`.

| Test                  | Validates                                                            |
| --------------------- | ------------------------------------------------------------------- |
| `test_mlp_tp.py`      | TP SwiGLU MLP (column/row-parallel + reduce-scatter)                |
| `test_moe_tp.py`      | TP sparse MoE MLP (router + experts + shared; MoE checkpoint only; decode + prefill) |
| `test_attention_tp.py`| TP gated full attention: decode / prefill / paged-KV contract       |
| `test_gdn_tp.py`      | TP Gated DeltaNet: decode + chunk-prefill                           |
| `test_model_tp.py`    | full-model TP contract: paged+traced path matches the bespoke oracle |
| `test_generate_tp.py` | full-model bespoke `generate_tp` on a real prompt (answer oracle)   |

Run the 27B TP suite (with `HF_MODEL=Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B` or
`Qwen/Qwen3.5-27B`, and `MESH_DEVICE=P150x4`):

```bash
pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s
```

On Galaxy (`MESH_DEVICE=P150x8`), prefix each command with `TT_VISIBLE_DEVICES`:

```bash
TT_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq
```

> The MoE-specific tests (`test_moe_tp.py`, and the MoE path in `test_model_tp.py` /
> `test_generate_tp.py`) require the sparse checkpoint — run them with
> `HF_MODEL=Qwen/Qwen3.6-35B-A3B MESH_DEVICE=P150x4`. On the dense 27B they are inert
> (`num_experts == 0`), and the dense/MoE checkpoints must not be mixed in one run
> (each test file `setdefault`s or expects a single `HF_MODEL`).

> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.
