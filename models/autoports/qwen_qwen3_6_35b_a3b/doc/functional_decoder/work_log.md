# Functional Decoder Work Log — Qwen/Qwen3.6-35B-A3B

Stage: 1/11 `functional-decoder`
Autoport dir: `models/autoports/qwen_qwen3_6_35b_a3b`
Branch: `agentic-research/qwen36-35b-a3b-claude`

## 1. Environment / hardware

```
$ timeout 60 tt-smi -ls --local
4 x Blackhole p300c (UMD chip ids 0..3), all resettable
$ python -c "ttnn.open_mesh_device(ttnn.MeshShape(1,1))"
arch = Arch.BLACKHOLE, compute_with_storage_grid = 11x10, dram_grid = 8x1
MESH_SMOKE_OK
```
transformers 5.10.2, torch 2.11.0+cpu, ttnn = source build in `/localdev/vkovacevic/tt-metal`.
HF weights already in the local hub cache:
`~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`
(26 safetensors shards, 66.97 GiB total).

This stage runs on a **1x1 mesh** (single Blackhole chip) per the skill contract.

Usable device DRAM, probed by allocating 512 MiB DRAM tensors until the bank manager refused
(`tests/probe_dram_capacity.py` -> `logs/probe_dram_capacity.log`):

```
CAP allocated 63 x 512 MiB = 33822867456 bytes = 31.50 GiB
CAP refused at chunk 64: TT_FATAL @ .../tt_metal/impl/allocator/bank_manager.cpp:462: false
```

That 33822867456 B / 31.50 GiB is the number the capability contract's byte accounting is measured
against (`doc/context_contract.json` -> `device_capacity_evidence.usable_dram_bytes`, which quotes
the log path). It began as an ad-hoc probe; round 2 of the review pointed out that a contract field
should not rest on prose, so it is a committed script with a committed log now.

## 2. HF architecture analysis (source of truth)

`config.json` → `architectures: ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: qwen3_5_moe`.
Text decoder config is nested under `text_config`. Implementation read line-by-line from
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (2294 lines).

### 2.1 Decoder layer kinds

`text_config.layer_types` has 40 entries, pattern `[linear, linear, linear, full] * 10`
(`full_attention_interval = 4`), so:

| kind | count | layer indices | token mixer |
|---|---|---|---|
| `linear_attention` | 30 | all i where (i+1) % 4 != 0 | `Qwen3_5MoeGatedDeltaNet` |
| `full_attention` | 10 | 3, 7, 11, ..., 39 | `Qwen3_5MoeAttention` |

Both kinds share the identical MoE MLP and the identical residual/norm structure
(`Qwen3_5MoeDecoderLayer.forward`):

```
h = x + mixer(input_layernorm(x))
out = h + moe_mlp(post_attention_layernorm(h))
```

### 2.2 Norms

`Qwen3_5MoeRMSNorm` (used for input_layernorm, post_attention_layernorm, q_norm, k_norm):
`x_f32 * rsqrt(mean(x_f32^2)+eps) * (1.0 + w)` — note the **`1 + w`** (weights are stored
zero-centred). Handled by folding `1 + w` at weight-load time.

`Qwen3_5MoeRMSNormGated` (gated-delta-net output norm) uses plain `w` (no `1 +`) and then
multiplies by `silu(z)`.

### 2.3 `full_attention` (Qwen3_5MoeAttention)

* `hidden=2048`, `head_dim=256`, `num_attention_heads=16`, `num_key_value_heads=2` (GQA 8:1)
* `q_proj: 2048 -> 16*256*2 = 8192`. Output is viewed as `(..., 16, 512)` and `chunk(2, -1)`:
  first 256 per head = Q, second 256 per head = **output gate** (`attn_output_gate: true`).
  So the 8192-wide q_proj output is *head-interleaved* `[h0_q, h0_gate, h1_q, h1_gate, ...]`.
* `k_proj/v_proj: 2048 -> 2*256 = 512`, no biases (`attention_bias: false`).
* `q_norm`/`k_norm` are RMSNorm over `head_dim=256`.
* RoPE: `partial_rotary_factor = 0.25` → `rotary_dim = 64`; rotation applied to the **first 64**
  dims of each 256-wide head, dims 64..255 pass through unrotated. `rotate_half` pairs
  `i` with `i+32` inside that 64-wide block (NeoX/HF convention).
* mRoPE: `mrope_interleaved: true`, `mrope_section: [11,11,10]`, `rope_theta = 1e7`.
  `apply_interleaved_mrope` overwrites the T-row with H/W rows at strided indices; for
  **text-only** input all three position rows are identical, so the result collapses exactly
  to standard 1-D RoPE. (Verified in `tests/test_reference_math.py::test_mrope_text_only_is_standard_rope`.)
* `scaling = head_dim^-0.5 = 0.0625`
* `attn_output = concat_heads(sdpa(...)) * sigmoid(gate)`, then `o_proj: 4096 -> 2048`.

### 2.4 `linear_attention` (Qwen3_5MoeGatedDeltaNet)

* `linear_num_key_heads=16`, `linear_key_head_dim=128` → `key_dim=2048`
* `linear_num_value_heads=32`, `linear_value_head_dim=128` → `value_dim=4096`
* HF `conv_dim = 2*key_dim + value_dim = 8192`, depthwise causal `conv1d` with
  `linear_conv_kernel_dim = 4`, no bias, followed by SiLU. (The TTNN layer's `conv_dim` is
  16384: q/k are pre-duplicated 16->32 heads and the `z` block rides the same conv with an
  identity tap — see §3.6.)
* `in_proj_qkv: 2048->8192`, `in_proj_z: 2048->4096`, `in_proj_b: 2048->32`, `in_proj_a: 2048->32`
* `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)` (both `[.., 32]`, fp32)
* Q/K are `repeat_interleave(2)` from 16 to 32 heads, then L2-normalised (eps 1e-6) inside the
  delta-rule kernel, and Q is scaled by `1/sqrt(128)`.
* Recurrent state `[batch, 32, 128, 128]`; conv state = last `kernel-1 = 3` pre-conv inputs
  (HF stores 4 columns but the oldest is provably dead — see §3.3).
* Output: `RMSNormGated(core_attn_out, z)` per 128-wide value head, then `out_proj: 4096->2048`.

### 2.5 MoE MLP (identical in both layer kinds)

* `num_experts=256`, `num_experts_per_tok=8`, `moe_intermediate_size=512`
* router: `gate.weight [256, 2048]`, `softmax(logits, fp32)` over all 256 →
  `topk(8)` → **renormalised** by the top-8 sum.
* experts: fused `gate_up_proj [256, 1024, 2048]` (gate = first 512 rows, up = last 512 per
  the `.chunk(2, -1)` on the *output*, i.e. rows 0..511 = gate, 512..1023 = up) and
  `down_proj [256, 2048, 512]`; `silu(gate) * up`, no biases.
* shared expert: `Qwen3_5MoeMLP(intermediate=512)` gated by `sigmoid(shared_expert_gate(x))`
  (a `[1, 2048]` linear).
* `out = sum_k w_k * expert_k(x) + sigmoid(gate_s(x)) * shared(x)`

### 2.6 Advertised context

`text_config.max_position_embeddings = 262144` (256K). This is the context the stage must
prove (see `doc/context_contract.json`).

## 3. Algebraic restructurings needed for TTNN

Recorded here because they are the non-obvious correctness-critical parts. All three are
covered by CPU unit tests in `tests/test_reference_math.py`.

### 3.1 UT transform without a 64-step Python loop

`torch_chunk_gated_delta_rule` builds `attn` by 63 sequential row updates. That loop is exactly
forward substitution for `T = (I - A)^-1` where `A` is the strictly-lower-triangular matrix
`-(k_beta @ k^T) * decay_mask`. Since `A^C = 0` for a strictly lower triangular `C x C` matrix,

```
(I - A)^-1 = (I + A)(I + A^2)(I + A^4) ... (I + A^(C/2))
```

which is `2*log2(C) = 12` batched matmuls for `C = 64` instead of 63 serial slice-updates.

### 3.2 Chunk-scan state recurrence

The inter-chunk loop is a first-order linear recurrence in the `[128,128]` state; it stays a
Python loop over `seq/64` chunks, but every step is fully batched over `(batch, 32 heads)`.

### 3.3 Depthwise causal conv1d as shifted MACs

`silu(conv1d(x, w, groups=8192, padding=3)[..., :S])` with left context `c` (3 rows) equals

```
sum_{j=0..3} w[:, j] * concat([c, x])[j : j+S]
```

so prefill needs 4 device slices + 4 muls + 3 adds, and decode needs 4 muls + 3 adds over
3 persistent single-row state buffers. No `ttnn.conv1d` dependency, no groups=8192 kernel.

Also proved: HF's 4-column conv state has a dead oldest column, so a 3-column TTNN state is
exactly equivalent.

### 3.4 L2 norm as a manual fp32 reduction (not `ttnn.rms_norm`)

`l2norm(x, eps) = x * rsqrt(sum(x^2) + eps)` is algebraically `ttnn.rms_norm(x,
weight=1/sqrt(D), epsilon=eps/D)`, and folding the delta rule's extra `1/sqrt(D)` q-scale
into the same weight makes it one fused op. **Measured and rejected**: `ttnn.rms_norm` is
~4e-3 off in relative terms even at HiFi4 + `fp32_dest_acc_en` (see §4 probe R2), and these
vectors' inner products drive the recurrence. The manual fp32 form (`mul`, `sum`, `rsqrt`,
`mul`) is exact (`maxabs 5.6e-5`) for 3 extra ops and is what the code does. The algebraic
identity itself is still unit-tested (`test_l2norm_as_rms_norm`) because it documents why the
epsilon rescale is what it is.


### 3.5 Two foldings that remove runtime ops

* **q/k head duplication into the weights.** HF does
  `repeat_interleave(q, 2, dim=heads)` after the conv. The conv is depthwise, so duplicating
  channels commutes with it provided the per-channel taps are duplicated identically. Both the
  input projection and the conv taps are duplicated at load time, so the runtime path never
  needs a `repeat_interleave` and `nlp_create_qkv_heads(num_heads=32, num_kv_heads=32,
  head_dim=128)` splits q/k/v directly. Cost: the projection grows 8192 -> 12288 columns.
* **`silu(z)` from the same conv.** `z` is appended to the conv input with the identity tap
  `[0,0,0,1]`, so `silu(conv(...))` emits post-conv q/k/v *and* `silu(z)` in one op, and no
  layout gymnastics are needed to get `z` into the head-major layout the gated norm wants.
  Cost: conv width 12288 -> 16384.

## 4. TTNN op probes (device)

`tests/probe_ttnn_ops.py` checks, on device, every op behaviour the design depends on before
any of it was written. Final state: **21/21 ok**. It is kept as a script (not a pytest) so a
failure prints the exact op and shapes.

The committed log (`logs/probe_ttnn_ops.log`) is the **final** state of the script: 21 probes
that all pass, each printing its shapes and PCC. The table below is the *design* record — several
rows are conclusions from running variants of these probes during development (a rejected core
grid, a dtype the op refuses, a precision comparison between two spellings). Those variants are
not in the final script, because a probe file that keeps deliberately-failing calls cannot report
`21/21 ok`. The `evidence in final log` column says which rows a reader can re-derive from the
committed log and which are development findings recorded here so a later stage does not
re-discover them.

| probe | result | consequence | evidence in final log |
|---|---|---|
| `ttnn.slice` non-aligned on dim -2 | ok, pcc 0.999999 | conv shift-MAC uses plain slices | yes (`slice non-tile-aligned on dim -2`) |
| `ttnn.sparse_matmul` | `program_config` is **mandatory** (`MatmulMultiCoreReuseMultiCast1DProgramConfig`, `mcast_in0=True`) | `_sparse_program_config` builder; `Kt % in0_block_w == 0` snap | partly — the log shows the three `sparse_matmul` modes passing *with* a program config; that one is required, and the `in0_block_w` divisibility rule, are development findings |
| `ttnn.sparse_matmul` core grid | the full 11x10 grid is rejected; 8x8 works | `_SPARSE_CORE_GRID = (8, 8)` | no — development finding; the final probes use 8x8 only |
| `rotary_embedding_hf` decode mode | requires a **sharded** input | decode uses a manual partial RoPE on interleaved tensors instead | partly — the log has both the prefill `rotary_embedding_hf` probe and the `manual partial rope decode` probe that replaced it |
| `paged_update_cache` | requires a **height-sharded** update tensor | decode keeps `nlp_create_qkv_heads_decode`'s sharded K/V for the cache write and only moves Q to DRAM for SDPA | partly — the log's `paged_update_cache decode with page table` probe is annotated `(height-sharded input)` and passes; the rejection of interleaved input is a development finding |
| `ttnn.topk` | **bfloat16 only** (fp32/bf8 rejected/worse) | router logits are cast to bf16 for top-k; see §5 | partly — the log shows bf16 top-k passing; the fp32/bf8 rejection is a development finding |
| `ttnn.rms_norm` precision | relmax 1.9e-2 (bf16), 3.7e-3 (fp32+HiFi4) | manual fp32 l2-norm for delta-rule q/k (§3.4) | partly — the log has both spellings (`rms_norm with weight + epsilon` pcc 0.999988 and `rms_norm as l2norm` maxabs 3.243e-03); the bf16-vs-fp32 relmax pair is a development finding |
| `ttnn.softplus` | fp32 relmax 3.4e-4, bf16 relmax 3.2e-2 | the `a + dt_bias` path is forced to fp32 (`dt_bias` reaches +15.6) | no — development finding |
| `ttnn.exp` with a `-1e30` additive mask | -> 0, no NaN | decay masks are added **before** `exp`; cumulative gates reach ~-1e5 so `exp` of the unmasked upper triangle would overflow to `inf` and produce `0*inf = NaN` | no — development finding; the behaviour it justifies is exercised by every `linear` test |
| `ttnn.permute` `(0,3,2,1)` / `(2,3,0,1)` | ok | gets `beta`/`g` from `[1,1,T,32]` into the `[.., heads, .., 1]` broadcast layout without a relayout | no — development finding; exercised by every `linear` test |
| `chunked_scaled_dot_product_attention` with `chunk_start_idx_tensor` | ok, pcc 0.999790 | the device-tensor offset form **works** on this build; not adopted because feeding it without a host write needs a setup-time offsets table plus a per-chunk device slice, and its only benefit is one program instead of 128 for a full-context prefill (README §6 limitation 6, handed to `optimize`) | yes |

Three further aliasing facts were **not** established by `probe_ttnn_ops.py` (which only checks
buffer addresses for the in-place case, `p_inplace`) but by ad-hoc buffer-address comparisons while
debugging the crashes in README §7 item 1. They are recorded here because they shaped the code:

| behaviour | finding | consequence |
|---|---|---|
| whole-tensor `ttnn.slice` | returns an **aliasing view** (same buffer address) | `_subview` / `_owned_slice` / `_move`; see README §7 |
| `ttnn.reshape` | **always** an aliasing view | `_view` helper documents that the input stays the owner; deallocating both double-frees, deallocating the view early is a use-after-free |
| `ttnn.split(t, 1, dim)` with one output | may return the input | per-piece buffer-address check before deallocating in the delta-rule scan |

## 5. Precision analysis

### 5.1 bf16 top-k router selection

`ttnn.topk` only accepts bf16, so expert selection runs on bf16 logits. Reproducible
measurement (`tests/measure_expert_union.py`, CPU only, synthetic layer-3 weights, the 2048
tokens the perf capture uses):

```
BF16_TOPK token expert-set agreement = 94.0%  (123 of 2048 tokens differ)
```

i.e. ~6% of tokens swap their 8th expert for the 9th. Quantified end-to-end on CPU with **real**
layer-3 weights by emulating exactly that (bf16 logits for selection, fp32 logits for the
weights) — a one-off measurement, not scripted, which is why the agreement figure differs
slightly from the 94.0% above (different weights and therefore different logits):

```
full-decoder-output PCC (fp32 router vs bf16-logit router): 0.99999963   rel-rms 0.086%
token expert-set agreement: 95.7%
```

Negligible, because the swapped expert is the *lowest-weighted* of the eight and the residual
dominates the layer output. Recorded here so a later stage does not re-investigate it.

The same script measures the other routing fact the perf analysis needs — how many experts a
32-token `sparse_matmul` group activates, since that, not `top_k`, is the real expert-matmul
work:

```
GROUP_UNION uniform  = 163.3
GROUP_UNION measured = mean 162.3  min 152  max 174     (64 groups of a 2048-token prefill)
GROUP_UNION work multiplier vs ideal gather-by-expert = 20.3x
```

README §6 limitation 1 divides by that 162.3.

### 5.2 What the reference itself is worth

HF's own bf16-vs-fp32 divergence on these layers, real weights, seq 1024:

```
layer 0 (linear): PCC 0.9999950  rel_rms 0.316%  maxabs 0.0706
layer 3 (full):   PCC 0.9999949  rel_rms 0.319%  maxabs 0.0409
```

The TTNN bf16 implementation lands at 1.06% (linear) / 0.64% (full) rel-RMS against the
**fp32** reference, i.e. 1.1-3.4x HF's own bf16 sensitivity, with PCC >= 0.99994. That is the
expected cost of bf16 activations plus bf16 sparse matmuls, and it is why the default 0.995 bar
needs no model-specific exception.

### 5.3 Skipped experts must contribute exactly zero

`_experts` sums the sparse-matmul output over all 256 experts and relies on the routing weight
being exactly 0 for the experts a token did not select, so a skipped expert's output tile has to be
either zero or at worst finite: `0 * NaN` and `0 * Inf` are `NaN`, which would poison the sum for
the whole token. The 276 passing PCC rows are strong empirical evidence (any poisoned tile would
destroy the layer output, not perturb it), but nothing asserts it directly — recorded as a
hard-check gap rather than claimed as verified. A later stage adding an `nnz`-aware or
gather-by-expert path should assert it explicitly, because that path changes which tiles the op
writes.

### 5.4 Router weight computation

HF softmaxes over all 256 experts, takes the top 8 and renormalises by their sum. Softmax is
monotonic, so top-k of the raw logits picks the same experts, and a softmax over just those 8
values *is* the renormalised weight. The implementation therefore does top-8 then an explicit
fp32 max-subtract / exp / sum / div (measured `maxabs 1.7e-3` vs 3.6e-2 for a plain
`ttnn.softmax`), and keeps a separate binary mask as the sparse-matmul sparsity so the pattern
is exactly 8 experts per token even when a routing weight rounds to zero in bf16.

### 5.5 Long-context decode-SDPA investigation

The one visibly-off number in the stage (`test_longest_decode_context[full]`, PCC 0.9986 at
position 262143 vs ~0.99999 everywhere else) was root-caused to a specific
`ttnn.SDPAProgramConfig` field and **fixed**, not annotated. Three diagnostics, all kept as
scripts with their output in `logs/`:

| script | log | what it settles |
|---|---|---|
| `tests/diag_long_decode.py` | `logs/diag_long_decode.txt` | localises the error to the attention branch and **rules out operand quantisation**: an exact bf16-operand HF control matches fp32 to 2.4e-6 at every context, while TTNN diverges from both identically |
| `tests/diag_sdpa_decode.py` | `logs/diag_sdpa_decode.txt` | drives `paged_scaled_dot_product_attention_decode` **alone** and sweeps grid x `k_chunk_size` x `exp_approx_mode` x **`max_cores_per_head_batch`** x context, plus a warmed timing pass at the advertised context |
| `tests/diag_decode_sdpa_onmodel.py` | `logs/diag_decode_sdpa_onmodel.txt` | the on-model decision: the three candidate settings measured on the **whole layer** against HF, off one real prefilled cache per context, at 258 / 1024 / 4096 / 32768 / 262144 |

**Root cause: `SDPAProgramConfig::max_cores_per_head_batch`** — how many cores the op splits each
KV head's keys across. It is a *struct default* (`sdpa_config.hpp:18`, value 16) and the factory
reads `program_config.has_value() ? program_config->max_cores_per_head_batch : num_cores_available`
(`sdpa_decode_program_factory.cpp:192-193`), so on this 11x10 part with batch 1 and 2 KV heads,
**no config = 55 cores/head and any explicit config = 16** unless the field is set on purpose.
Accuracy rises with cores/head at long context and collapses below a context that grows with it
(2-4 cores break at 257 keys, 8-16 at 1024, 32 at 4096, 55 everywhere above 128). **1 core/head is
the only value correct at every context.**

Ruled out along the way, each by holding the other axes fixed: `exp_approx_mode` (approx and exact
rows are **bit-identical**), the core grid (`8x8` == `11x10` at equal cores/head), and the chunk
policy (`k_chunk_size` 0 == 128 at equal cores/head; 512 only *moves* which context breaks).

**The fix ships.** `DecoderConfig.decode_sdpa_max_cores_per_head = 1`, built into an
`SDPAProgramConfig` once at construction (`_decode_sdpa_program_config`). On-model, worst case over
five contexts: **0.9997685** (shipped) vs 0.9985674 (no config) vs 0.0304429 (struct default). And
it is *faster*: 11.5 ms vs 28.1 ms per op call at 262144 keys, so there was no trade-off to make —
55 cores/head was both the slowest and the least accurate setting measured.

**Scope of the bug: small batch.** The field is `max_cores_per_head_*batch*` and the factory divides
by the batch, so the old default's cores/head was `min(110, 110*B*2)/B/2` — 55 at batch 1, 27 at 2,
13 at 4, 6 at 8, 3 at 16 and **1 at 32**. At batch 32 the old and new settings are therefore
identical, which is why 82 batch-32 `decode[full]` rows never caught it and why the batch-32 perf
table barely moved. It was a small-batch, long-context defect: one or a few long requests, i.e. the
serving case, and what `test_longest_decode_context` (batch 1) measures. The shipped config makes the
decomposition batch-independent at 1 core per (slot, KV head).

**Two things reported rather than explained**, both worth an upstream issue:

* an explicit config with `max_cores_per_head_batch=110` derives the same 55 cores/head as passing
  no config — identical grid, chunk policy and `exp` as far as the factory reads them — yet scores
  0.0534 against 0.7664. This stage cannot account for that from the source.
* every setting above 1 core/head returns PCC ~0.16-0.37 at some short context instead of failing,
  i.e. a silent wrong answer from a device op.

**Two earlier root causes of mine were wrong** and are recorded here so the correction is legible:
first "the dynamic `k_chunk_size` path", then "whether a program config is passed at all". The
second survived a review round because the confound-free sweep it rested on was missing this
fourth axis; the independent review caught it. The handoff it produced ("`optimize` must select a
config per position") was wrong too — a single static config is correct at every context.

## 6. Hardware incidents

**2026-08-17 ~13:26 — stale process holding sysmem (infrastructure, not a model result).**

* Signature: `RuntimeError: Sysmem mapped at unexpected NOC address (likely a stale process
  holding sysmem)` and `UMD | Waiting for lock 'CHIP_IN_USE_0_PCIe' which is currently held by
  ... PID: 3321146`, raised at `open_mesh_device`.
* Cause: an earlier 32K long-context smoke run hit `TT_FATAL: context_id ... is out of range
  (max 32)` because `test_max_batch_full_context_capacity` opened a *second* mesh device while
  the module-scoped fixture still held one. The process then hung inside device teardown and
  kept the PCIe/sysmem lock.
* Fix to the cause: the capacity test now uses the same fixture, and the long-context fixture
  was changed to **function scope** so each of these (largest-in-stage) allocations starts
  from empty device DRAM.
* Recovery: `kill -TERM 3321146 3321145` (only processes from this run), `pkill -f
  test_long_context`, then `timeout 60 tt-smi -ls --local` → all 4 Blackhole p300c chips
  present, then a mesh-open/close smoke → `MESH_SMOKE_OK`. **No `tt-smi -r` reset was
  needed** (listing was complete and the mesh opened cleanly), no locks had to be cleared by
  hand, and the long-context runs were restarted from the same stage state.
* The rerun script (`tests/run_long_context.sh`) now runs one pytest process per case so a hung
  teardown cannot poison the next one.

**Classified warning (not an incident):** `test_traced_decode_pcc[linear]` emits
`Allocating device buffers is unsafe due to the existence of an active trace. These buffers may be
corrupted once a trace is executed.` once per run (`logs/test_suite_main.log`,
`watcher/pytest.log`). Source is the **test harness**, not the layer:
`harness.restore_state` stages a host tensor with `ttnn.from_torch(..., device=...)` after
`end_trace_capture` in order to rewind the state before the measured replay
(`tests/harness.py` `restore_state`). The staged buffer is deallocated before `execute_trace`, so
no trace buffer is aliased, and `test_traced_decode_matches_eager` shows replay is bit-identical to
eager from the same state. The functional-decoder skill calls allocation after trace capture a red
flag, so it is classified here rather than left in the log: the *model* allocates nothing after
capture, only the rewind helper does.

**Classified warning (not an incident):** every Python process in this stage ends with
`nanobind: leaked N instances! / leaked N types! / leaked N functions!` — all 13 files in
`logs/`, three of them (`summarize_perf.log`, `write_context_contract.log`, `test_contract.log`)
in processes that **never opened a device at all** (`grep -c 'Opening user mode device driver'`
= 0). That alone rules out a device-resource leak: it is nanobind reporting binding objects that
outlive module teardown at interpreter shutdown, emitted *after* `ttnn.close_mesh_device` has
already logged `Cluster destructor completed` in the runs that do open a device. The named
objects are pure binding types (`CoreRangeSet`, `DispatchCoreConfig`, `MathFidelity`,
program-config classes), never buffers or tensors, and every following command in the serialized
sequence opened the mesh cleanly. Recorded so it is not mistaken for a leak.

**2026-08-17 ~14:09-14:13 — two concurrent Tracy captures (self-inflicted, infrastructure).**

* Signature: a second `run_perf.sh` launched while the first was still capturing; they contended
  for the Tracy capture port and the device, and the loser left `python -m tracy` and
  `tracy-capture` processes alive.
* Recovery: `pkill -f run_perf.sh`, then `kill -9` the remaining `tracy` / `tracy-capture` /
  `serve_wasm.py` pids, `tt-smi -ls --local` (all 4 chips present), mesh-open/close smoke →
  `MESH_SMOKE_OK`. No reset needed.
* Two real fixes came out of it, both now in `tests/run_perf.sh`: it kills the Tracy WASM GUI
  server subprocess after every case, and §7 documents that perf cases must be run one at a
  time. Three unrelated `run_perf.sh` bugs were also fixed while getting the first capture
  through: tracy's own flags (`-r -p -v -n`) must precede `-m pytest`; tracy re-splits the
  forwarded argv on whitespace so a quoted `-k "a and b"` selector arrives as separate
  arguments (node ids are used instead); and the profiler's default 1000-op budget
  (`tools/tracy/common.py`) is far below one warmed prefill of this layer, which made
  post-processing abort with "Device data missing" — `--op-support-count 20000` fixes it.
* A fourth: closing a device that still held a layer's ~1.5 GiB of weights *and* a full
  profiler buffer segfaulted inside `close_mesh_device`. `test_perf.py` now calls
  `ttnn.ReadDeviceProfiler` + `synchronize_device` before close and releases the layer
  (`FunctionalDecoder.release()`) when a measurement finishes.

**2026-08-17 ~15:34-15:40 — device-profiler marker mismatch aborted one Tracy case
(infrastructure, recovered without a reset).**

* Signature, in the 4th of 4 perf cases (`decode` / `full`), *after* the test itself printed
  `PASSED`: `TT_FATAL: Start and end marker IDs do not match.` ->
  `TT_FATAL @ tt_metal/impl/profiler/profiler.cpp:2104: false` -> `Fatal Python error: Aborted`.
  The pytest process died by `SIGABRT` while post-processing device markers; `tracy-capture` and
  the `python -m tracy` wrappers then hung for ~6 minutes waiting for a client that was gone.
* Where it comes from: `profiler.cpp:2081-2108` pairs device zone markers on a stack and only
  tolerates a mismatch when `had_dropped_markers` is set. The abort is that tolerance not
  applying — a profiler post-processing failure, in no way a model result. The same case,
  same code, had completed 10 minutes earlier in the previous evidence pass.
* Triage: `tools/tt-triage.py --llm-output` was started first, as `$tt-device-usage` requires,
  but produced a zero-byte report and made no progress within its 180 s bound (it waits on the
  device lock the aborted process still held), so it was killed. `triage/tt-triage.txt` is
  therefore the empty file it left; recorded rather than deleted, so the sequence is auditable.
* Recovery: killed the runner parents, then `kill -9` on the 6 surviving
  `python -m tracy` / `tracy-capture` / `tee` pids **by explicit pid**. (`pkill -f <pattern>`
  is a trap here: the pattern appears in the killing shell's own command line, so `pkill` kills
  it before it kills the target. That cost one attempt.) Then `timeout 60 tt-smi -ls --local`
  (all 4 Blackhole p300c chips present) and a 1x1 mesh open/close -> `MESH_SMOKE_OK`.
  **No `tt-smi -r` reset was needed.**
* Likely contributor, and the cheap mitigation: `generated/profiler/` had grown to 11 GB across
  the stage's perf runs, including `.logs/zone_src_locations.log` and
  `new_zone_src_locations.log`, which accumulate across runs. It was cleared (untracked,
  regenerable — `.gitignore:73`) before retrying, and all four cases were then re-captured in
  one run. `run_perf.sh` deletes `perf_host_summary.jsonl` on start, so a single case cannot be
  re-run in isolation without dropping the other three rows: retry is always all four.

**2026-08-17 ~16:50 — killed an in-flight long-context run on purpose (not a fault).**

* Why: a logging fix landed mid-run that changed which file `record()` writes to, and two of the
  five advertised-context cases had already written to the wrong one. Letting the run finish would
  have produced evidence that did not correspond to any single code state.
* Handling, per `$tt-device-usage`: killed the runner and its pytest child **by explicit pid**,
  then `timeout 60 tt-smi -ls --local` (4 Blackhole p300c chips present, 8 table rows) and a 1x1
  mesh open/close smoke -> `MESH_SMOKE_OK`. **No reset needed**, no locks cleared, no stale
  processes left. The run was then restarted from the top with the fixed code.
* Recorded because a killed device job is worth an audit trail even when it was self-inflicted and
  recovered cleanly.

## 7. Commands

```bash
# 0. weight stats (writes doc/functional_decoder/weight_stats/layer_{00,03}.json)
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py        -k test_weight_stats_match_real_checkpoint

# 1. CPU-only algebra unit tests (no device, no checkpoint)
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_reference_math.py

# 2. device op probes
python models/autoports/qwen_qwen3_6_35b_a3b/tests/probe_ttnn_ops.py

# 3. main correctness suite (CPU algebra + device, incl. real weights) -> logs/test_suite_main.log
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py

# 4. advertised-context evidence  -> logs/long_*.log, long_context.jsonl
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_long_context.py -m slow

# 5. performance (Tracy, one run per mode/kind)  -> tracy/<kind>_<mode>/
python -m pip install tt-perf-report            # 1.2.8
./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh                  # all four
./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh prefill linear   # one case
# Run one case at a time: two concurrent tracy captures fight over the capture port and the
# device, and the loser leaves a process holding sysmem (see section 6).

# 5b. reduce the filtered CSVs into doc/functional_decoder/perf_summary.json
python models/autoports/qwen_qwen3_6_35b_a3b/tests/summarize_perf.py

# 6. watcher-clean run (never combined with the profiler)  -> watcher/
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/watcher \
  pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py -k "<selector>"
# in practice always via the runner, which also greps the log and gzips it:
./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_watcher.sh

# 7. the machine-readable capability contract  -> doc/context_contract.json
python models/autoports/qwen_qwen3_6_35b_a3b/tests/write_context_contract.py

# 8. diagnostics behind README section 3.8 (long-context decode SDPA)
python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_long_decode.py         # localise
python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_sdpa_decode.py         # op-only sweep
python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_decode_sdpa_onmodel.py # on-model control

# 9. CPU-only: experts activated per 32-token sparse-matmul group (README section 6 limitation 1)
python models/autoports/qwen_qwen3_6_35b_a3b/tests/measure_expert_union.py
```

Hardware-facing commands above are always run **one at a time**, per `$tt-device-usage`; the
evidence in `doc/functional_decoder/` was produced by one serialized pass in the order 4 -> 7 ->
2 -> 6 -> 5 -> 5b -> 3 -> 8, after the last change to `tt/functional_decoder.py`.

## 8. Result summary

| requirement | evidence |
|---|---|
| `tt/functional_decoder.py` with documented prefill/decode contract for both layer kinds | module docstring + README §2; `test_config_matches_hf`, `test_layer_kinds_cover_the_whole_model` |
| decode runs fully under traced execution | `test_traced_decode_pcc` (PCC from replay), `test_traced_decode_matches_eager` (bit-identical) |
| every layer kind, real config shapes, paged prefill/decode, page table, current position | README §3.1/§3.4; 276 PCC rows in `pcc.jsonl` |
| longest feasible seq/context | 262143-token prefill and position-262143 decode for both kinds (`long_context.jsonl`) |
| non-aligned lengths around chunk/page/tile boundaries | 1/32/33/64/65/128/129/1024/1025/2048/2049/3000/4096 + 262143 per kind |
| `doc/context_contract.json` | derived from evidence by `tests/write_context_contract.py`, re-checked by `test_context_contract_file_is_consistent`; **no capability reduction** |
| real-weight test passing | `test_real_weights_prefill_and_decode[linear,full]`, `pcc_real_weights.jsonl` |
| PCC >= 0.995 prefill and decode | worst in main suite 0.9999450; worst at 262144 context 0.9997685 (§5.5 — root-caused to one `SDPAProgramConfig` field and **fixed**, not waived) |
| warmed prefill + traced warmed decode perf with tt-perf-report tables + CSV/provenance | `tracy/<kind>_<mode>/`, `perf_summary.json`, README §5 |
| runtime fallback audit clean | `test_no_runtime_host_fallback` (static: 26 runtime methods plus every module-level helper, the helper list derived from the module rather than hand-written) + `test_no_host_ops_during_forward` (dynamic: ttnn host bridges monkeypatched to raise) |
| determinism / repeated input | `test_prefill_determinism`, `test_decode_determinism` (bit-identical over 3 repeats) |
| watcher-clean run | `watcher/` — 8 passed, 5404-line log (10 dumps), `watcher_hits.txt` empty (README §3.9) |
| README + work log with commands, PCC, perf, limitations, artifacts | this file + `README.md` |

## 9. Independent stage review (round 1) and the work it produced

`$stage-review` returned **more-work-needed** on the first pass. Every finding was worked, not
argued down:

| finding | outcome |
|---|---|
| P1 — the "dynamic `k_chunk_size`" root cause was confounded: omitting the program config also flips the grid and `exp_approx_mode`, one label was wrong, and `k_chunk_size=0` (dynamic chunk path + explicit config) was never measured | **Superseded by round 2 — the conclusion recorded below was still wrong**; see §10 and §5.5. What round 1 produced: the sweep shows `exp_approx_mode` is *bit-identically irrelevant* and the chunk policy is not the discriminator, and it concluded the presence of a program config was. Added `tests/diag_decode_sdpa_onmodel.py`, the on-model control the reviewer asked for: shipped setting 0.9999954 @ ctx 1024 / 0.9985674 @ 262144 vs explicit config 0.0304429 / 0.9999958. Decision unchanged but now measured on the shipped path; README §3.8 rewritten; the fabricated `8x8` label and the "same cache" claim are gone. |
| P1 — unguarded whole-tensor `ttnn.slice` of the persistent RoPE tables frees the layer's weights when `supported_context <= prefill_chunk_size` | **Real bug, fixed.** Both slices go through `_subview`; `_valid_mask` likewise. Regression test `test_prefill_covering_whole_context_does_not_free_weights` reproduces the exact trigger and then runs two more forwards on the same layer. |
| P2 — `current_pos = -1` reached `ttnn.embedding` as an unsigned index | **Real bug, fixed.** Clamped on device with `ttnn.maximum(idx, 0)` before the lookup; the `-1` test now also asserts the inactive rows are finite. |
| P2 — the root `.gitignore` (`*.log`, `*.csv`, `generated`) silently kept the test logs, watcher log and filtered perf CSVs out of the commit, while `tracy/README.md` claimed they were committed | **Fixed.** `doc/.gitignore` re-includes them (22 evidence files now commit); the oversized raw ops CSVs stay excluded by the repo's own 500 KB rule, documented in `tracy/README.md`. |
| the `q*8` control was inert (RMSNorm makes token scaling a no-op for the attention branch) and was described as a softmax-peaking control | Relabelled for what it actually isolates (residual dilution) in both the diagnostic and README §3.8. The softmax question is now answered by the step-2/step-3 controls instead. |
| `record()` appended, so the committed jsonl mixed runs (542 lines for "271 rows") | Fixed: `harness.reset_log()` clears a log once **per run** — the main suite from an autouse session fixture, the long-context and perf runners by deleting theirs before looping. Deliberately not per *process*: the advertised-context evidence is five separate pytest processes accumulating into one file, so truncating per process (the first attempt, caught immediately) would have left only the last case. `pcc.jsonl` is now 274 lines for 274 rows and `pcc_real_weights.jsonl` 6 for 6. |
| unclassified `Allocating device buffers is unsafe due to the existence of an active trace` warning | Classified in §6: it comes from the test harness's state-rewind helper, not the layer; the staged buffer is freed before `execute_trace`. |
| no negative/error-path tests | Added `test_decode_forward_rejects_out_of_contract_inputs` covering all six documented raise paths. |
| probe attribution: three aliasing facts were credited to `probe_ttnn_ops.py`, which does not check them | §4 corrected to say where they actually came from. |
| numeric mismatches (94 vs 95 passed, op-to-op gap bounds, "~900 ops", `8192 -> 16384`, `moe_weight_bytes`, env-var names) | All corrected against the artifacts. |

Round 2 of the review returned **more-work-needed** as well; §11 records that round.

Two things were fixed in round 1 that the reviewer did not ask for, found while re-deriving
the numbers:

| found | outcome |
|---|---|
| `start_pos`'s 128-alignment was documented as a local choice ("so offsets stay legal"), which reads like a self-imposed restriction on a public path | Derived the actual op contract: chunked SDPA converts the offset with `chunk_start_idx / q_chunk_size` **integer division** in both entry points this checkout has (`sdpa_program_factory.cpp:133` for the scalar offset; `kernels/dataflow/reader_interleaved.cpp:260` for the `chunk_start_idx_tensor` offset), and validates only `>= 0` (`sdpa_device_operation.cpp:187`), so a misaligned offset is *silently* wrong. Recorded in the source, README §2/§9 and the `sdpa_chunk` knob row, together with the only lever (lower `sdpa_chunk`, floor 32) and the fact that it never limits prompt length. |
| the nanobind teardown message in every log was unclassified | Classified in §6 with the decisive evidence: it also appears in three processes that never opened a device. |

All evidence in `doc/functional_decoder/` was then regenerated against the fixed code (see §10),
including the two SDPA diagnostics, which reproduced the round-1 on-model numbers exactly
(0.9999954 / 0.0304429 / 0.9985674 / 0.9999958). Round 2 then replaced that comparison with a
three-setting, five-context one and changed the shipped configuration; §10 records it.


## 10. Independent stage review (round 2) and the work it produced

`$stage-review` returned **more-work-needed** again. It re-derived every headline number
successfully, and then found one thing that mattered a great deal:

| finding | outcome |
|---|---|
| **P1 — the §3.8 root cause was still confounded**: `SDPAProgramConfig::max_cores_per_head_batch` is a struct default (16) that the factory replaces with `num_cores_available` when no config is passed, so "program config present" and "16 vs 55 cores per head" were the same event, and the axis was never swept | **The reviewer was right, and the layer changed because of it.** The 4th axis is now swept (`diag_sdpa_decode.py`), the root cause is cores-per-head, and 1 core/head is the only value correct at every context. It **ships** (`decode_sdpa_max_cores_per_head = 1`): advertised-context decode PCC 0.9985674 -> **0.9997685**, and the op is 2.4x *faster* there (11.5 ms vs 28.1 ms), so there was no trade-off. The old claims "A is the only shippable setting" and "`optimize` must select per position" were both wrong and are gone. Two residual oddities are reported for upstream rather than explained (§5.5). |
| P2 — README §5 said the two expert sparse matmuls were 81% of prefill and "the whole attention path is the remainder", overstating attention ~14x and hiding the MoE elementwise cost | Replaced with the measured three-way split (token mixer / expert matmuls / MoE dense-intermediate elementwise) for all four cases. `summarize_perf.py` now *derives* it into `perf_summary.json` — the mixer/MoE boundary is found structurally (last `LayerNormDeviceOperation` before the first sparse matmul in an iteration) rather than hand-counted, so it cannot drift from the CSVs. |
| Other concern — decode perf measured only at `cur_pos=4095`, so the advertised-context latency is unknown | Measured: the decode SDPA alone is 11.5 ms/call at 262144 keys vs ~1 ms at the profiled shape, so the advertised-context step is ~10 ms slower than the §5 table. Recorded in §5, in `test_perf.py`'s own comment, and in README §3.8's timing table. |
| P2 — `ttnn.embedding` untilizes a TILE-layout weight on **every** call, so each decode step untilized both full RoPE tables (2 x 32 MiB at the advertised context); and decode perf was measured at `supported_context=8192` without saying so | **Real inefficiency, fixed.** The tables are now stored ROW_MAJOR, so decode gathers `max_batch_size` rows instead of untilizing the whole table, and prefill tilizes only the chunk it slices. The first attempt kept *both* layouts and hit `Out of Memory: Not enough space to allocate 2147483648 B DRAM buffer` in the two real-weight tests — 64 MiB per layer of duplicated tables is real pressure once the session's cached layers are live — which is why one table plus a per-chunk tilize is the right shape. `supported_context` is stated in the perf section. |
| P2 — three refuted pre-round-1 explanations still lived in committed code and in a regenerated artifact (`test_long_context.py`'s "input-conditioning artefact of random K/V", `diag_sdpa_decode.py`'s printed "specific to the dynamic chunk path" note, `diag_long_decode.py`'s one-sided docstring) | All three corrected in place; the diagnostics were re-run so the logs no longer print the refuted story. |
| P2 — README claimed *every* artifact postdated the last source change; mtimes refuted both that and the narrated run order | Replaced with what is actually true and checkable, including how to tell whether a late source edit could matter (`git diff`; the fallback audit strips comments). |
| P2 — `measure_expert_union.py` was untracked and had no log, yet README/work_log cited its numbers; the DRAM capacity probe behind `context_contract.json`'s `usable_dram_bytes` was likewise prose-only | Script tracked, `logs/measure_expert_union.log` committed. Added `tests/probe_dram_capacity.py` + `logs/probe_dram_capacity.log` so the capacity number is a recorded measurement. |
| P2 — five staged `watcher/generated/inspector/*.yaml.gz` blobs no longer existed on disk, and 3 MB of uncompressed replacements were unignored, so no clean checkpoint commit was possible | `doc/.gitignore` excludes the inspector side-output (it is not evidence — the watcher log is), and the stale blobs are unstaged. |
| linear-attention state was not self-healing: a slot reused for a new sequence inherited the previous occupant's conv/recurrent state, hidden by every test calling `reset_state()` first | **Real trap, fixed.** A prefill at `start_pos == 0` now zeroes that slot's carry; `test_prefill_resets_linear_state_for_new_sequence` reproduces it without any reset. The `current_pos = -1` case is documented as self-healing (junk stays in-slot and is cleared by the next fresh prefill). Cost a second lesson: the new test first used a `layer_pairs` key no other test had, and since that cache is session-scoped and never evicts, two extra ~1.5 GiB layers pushed the real-weight tests into `Out of Memory`. It now reuses an existing key, and `conftest.py` logs every new key with the live count so a later OOM is traceable. |
| found while re-running: a **filtered** pytest session (`-k context_contract`) truncated `pcc.jsonl` through the same session fixture that gives the full run one file per run, so a subset run could silently replace 274 rows of committed evidence with nothing | Fixed: filtered sessions write `*_partial.jsonl` (gitignored) and never touch the committed logs. This is the third bug in this logging code: the first attempt truncated per *process* (would have destroyed the five-process long-context accumulation), and the fix for *this* one first diverted **every** log in a filtered session — which silently sent the long-context rows to `long_context_partial.jsonl`, caught two cases into the rerun. The redirect is now a set of log *names* the main suite owns, and all three failure modes are spelled out in `reset_log`'s docstring. |
| the static fallback audit's helper list was hand-written and missed `_view` | The list is derived from the module now, with an explicit setup-only exemption set, so a new helper cannot drift out of the audit. |
| `test_real_weights_prefill_and_decode`'s docstring said "traced decode" but the test is eager; `current_pos` had no documented upper bound; README's watcher size and suite command were imprecise; no hard-check for skipped-expert tiles | All corrected or recorded (`work_log.md` §5.3 for the skipped-expert gap). |

## 11. Commits

Local only; nothing pushed.

| SHA | contents |
|---|---|
| `12c947d9147670eb0b3a9b23136635b89de709f3` (`12c947d9147`) | the whole stage: `models/autoports/qwen_qwen3_6_35b_a3b/**` plus the `conftest.py` guarded-import fix (§7 item 4 of the README) |
| `b2bb054161fcde8a1664f848ce0f35ad3f58aeea` (`b2bb054161f`) | this work log's commit table (the SHA above could not be recorded in the commit it names) |
| `ea58fe8fa7ae1138dbc35a363b6b817faeeed605` (`ea58fe8fa7a`) | **review rounds 1 and 2**: the fixes in §9 and §10, the shipped `decode_sdpa_max_cores_per_head = 1`, the ROW_MAJOR RoPE tables, the per-slot state reset, and every artifact regenerated against this code |

All evidence was regenerated after the last code change, one hardware command at a time. The
round-2 pass ran: main suite (gate) -> long-context (5 pytest processes) ->
`write_context_contract.py` -> contract test -> op probes -> watcher -> Tracy perf (4 cases) ->
`summarize_perf.py` -> on-model SDPA control -> long-decode diagnostic -> DRAM capacity probe ->
main suite again (so `pcc.jsonl` is from the final code) -> the op sweep. Check it rather than
trust it:

```bash
find models/autoports/qwen_qwen3_6_35b_a3b/doc -type f \
  ! -newer models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py
```

should list only `weight_stats/*.json` (checkpoint-derived), the `.gitignore` / `README.md` files
that describe the artifact policy, and `triage/` (a record of the §6 incident when it happened).

Three numbers reproduced **bit-identically** across independent re-runs, which is worth recording
because it makes the determinism claim concrete: the five advertised-context PCCs, the on-model
three-setting comparison (0.9999950 / 0.4791771 / 0.9999949 at context 258 and so on), and the
whole 20-row op sweep. Only wall-clock and per-op device times moved, by <0.2%.

Pre-commit reformatted the Python sources (black/isort/autoflake — formatting only, no import
or semantic changes) and rejected two >500 KB artifacts; the full suite was re-run after the
reformat (**95 passed**) before committing, and the artifact policy is documented in
`tracy/README.md`.
