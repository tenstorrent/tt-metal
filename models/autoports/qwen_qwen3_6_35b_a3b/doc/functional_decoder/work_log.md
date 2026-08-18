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
any of it was written. Final state: **24/24 ok**. It is kept as a script (not a pytest) so a
failure prints the exact op and shapes.

The committed log (`logs/probe_ttnn_ops.log`) is the **final** state of the script: 24 probes
that all pass, each printing its shapes and PCC. The table below is the *design* record — several
rows are conclusions from running variants of these probes during development (a rejected core
grid, a dtype the op refuses, a precision comparison between two spellings). Those variants are
not in the final script, because a probe file that keeps deliberately-failing calls cannot report
`24/24 ok`. The `evidence in final log` column says which rows a reader can re-derive from the
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
| `chunked_scaled_dot_product_attention` with `chunk_start_idx_tensor` | ok, pcc 0.999773 | the device-tensor offset form **works** on this build; not adopted because feeding it without a host write needs a setup-time offsets table plus a per-chunk device slice, and its only benefit is one program instead of 128 for a full-context prefill (README §6 limitation 6, handed to `optimize`) | yes |

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

### 5.3 The real-weight max-abs outlier is recurrence accumulation

`pcc_real_weights.jsonl`'s `prefill[linear] seq=1024` row carries `maxabs = 1.2687` against 0.0449
for the same case with synthetic weights and 0.0706 for HF's own bf16 divergence (§5.2). Round 3 of
the review pointed out that quoting only PCC and rel-RMS there selects the flattering metric, which
was fair. Measured with `tests/diag_real_weight_maxabs.py`
(`logs/diag_real_weight_maxabs.txt`):

* **Not a large-magnitude element.** The worst element is at `|want| = 1.08`, 9.5% of the tensor max
  (11.38), with a 118% relative error — the sign flips. So the absolute number is not "small
  relatively".
* **Not the bf16 top-k swap** of §5.1. 4.4% of tokens get a different expert set, but the worst
  token does not, and **none of the worst eight** do.
* **It is the recurrent state.** Sweeping sequence length with the same weights and seed:
  `maxabs` 0.0630 / 0.1350 / 0.1910 / 0.4420 / 1.2687 at 1 / 2 / 4 / 8 / 16 delta chunks, i.e.
  roughly 2x per doubling; and the no-recurrence control (`full`, same weights, same length) stays
  at 0.1367. At one chunk — no carry at all — the error is the same order as both controls.
* **Bounded, not divergent — now measured on one curve.** Extending the same sweep: 32 chunks give
  1.9178 and 64 chunks give **1.9178**, identical to four decimals, so the growth flattens rather
  than compounding. Note what is bounded: the **max-abs**. PCC still declines slowly across the same
  doubling (0.9999036 -> 0.9998920), so the worst element stops getting worse while more elements
  pick up small error — round 5 was right to want that distinction stated. The mechanism is the
  decay: `g <= 0` so `exp(g) <= 1`, and old error ages out. This is why the 262143-token prefill
  (4096 chunks) is *not* worse at 0.9999742. Round 4 was right that the earlier version of this
  bullet did not support its own claim: it compared a synthetic-weight 262143-token *tail* maxabs
  against this real-weight 1024-token *full-window* one — different weights, different window, ~16000x
  fewer compared elements. The 2048/4096 rows are the comparison that actually settles it. The error
  also stays localized: 145 of 2.1M elements above 0.2 abs, across 43 of 1024 tokens.

Consistent with the rest of the stage: `linear recurrent_state` is the worst PCC anywhere
(0.9999450) and `longest-prefill state recurrent` is 0.9998960. Recorded as **controlled** rather
than fixed — it is the expected cost of a bf16 linear-attention recurrence, and `delta_dtype` is
already fp32 for the state itself (§5.2 / README §6 limitation 5).

### 5.4 Skipped experts must contribute exactly zero

`_experts` sums the sparse-matmul output over all 256 experts and relies on the routing weight
being exactly 0 for the experts a token did not select, so a skipped expert's output tile has to be
either zero or at worst finite: in IEEE arithmetic `0 * NaN` and `0 * Inf` are `NaN`, which would
poison the sum for the whole token.

**Now measured** (round 4, `probe_ttnn_ops.py`): with 4 rows x 8-of-256 experts selected, all 507904
values in the 248 unselected experts' output tiles are finite **and exactly 0.0**. So the weighted
sum is exact, not merely well-conditioned, and the property no longer rests on the 276 passing PCC
rows alone. A later stage adding an `nnz`-aware or gather-by-expert path should keep this probe, since
that path changes which tiles the op writes.

Worth noting alongside it: the same round measured that this build's `ttnn.mul(t, 0.0)` clears a NaN
rather than propagating it, so the IEEE rule above does not describe the kernel. `_zero_` uses
`ttnn.fill` anyway — a write cannot depend on kernel details either way.

### 5.5 Router weight computation

HF softmaxes over all 256 experts, takes the top 8 and renormalises by their sum. Softmax is
monotonic, so top-k of the raw logits picks the same experts, and a softmax over just those 8
values *is* the renormalised weight. The implementation therefore does top-8 then an explicit
fp32 max-subtract / exp / sum / div (measured `maxabs 1.7e-3` vs 3.6e-2 for a plain
`ttnn.softmax`), and keeps a separate binary mask as the sparse-matmul sparsity so the pattern
is exactly 8 experts per token even when a routing weight rounds to zero in bf16.

### 5.6 Long-context decode-SDPA investigation

The one visibly-off number in the stage (`test_longest_decode_context[full]`, PCC 0.9986 at
position 262143 vs ~0.99999 everywhere else) was root-caused to a specific
`ttnn.SDPAProgramConfig` field and **fixed**, not annotated. Three diagnostics, all kept as
scripts with their output in `logs/`:

| script | log | what it settles |
|---|---|---|
| `tests/diag_long_decode.py` | `logs/diag_long_decode.txt` | localises the error to the attention branch and **rules out operand quantisation**: an exact bf16-operand HF control matches fp32 at every context, while TTNN diverges from both identically |
| `tests/diag_sdpa_decode.py` | `logs/diag_sdpa_decode.txt` | drives `paged_scaled_dot_product_attention_decode` **alone**: an identity control for what "no config" resolves to, then a 2-D **`k_chunk_size` x `max_cores_per_head_batch`** grid over 11 contexts, the k-chunks-per-core table, held-axis checks, and a warmed timing pass |
| `tests/diag_decode_sdpa_onmodel.py` | `logs/diag_decode_sdpa_onmodel.txt` | the on-model decision: four candidate settings measured on the **whole layer** against HF, off one real prefilled cache per context, at 258 / 1024 / 4096 / 32768 / 262144 |

**Root cause: `SDPAProgramConfig::k_chunk_size`** — how many keys the op accumulates per chunk, i.e.
the depth of the sequential bf16 accumulation. At 262144 keys and one core per head the op scores
0.7664 at `k_chunk_size=32`, 0.9179 at 64, 0.9704 at 128, 0.9809 at 256 and 0.9825 at 512:
monotone, which is what an accumulation-depth mechanism predicts and nothing else here does.

**What made this hard, and what I got wrong twice.** The paged entry point does not pass an empty
config to the device op; it substitutes one (`sdpa_decode.cpp:122-129`) with the device grid,
`q_chunk_size=32`, **`k_chunk_size=32`**, `exp_approx_mode` unset and
**`max_cores_per_head_batch=1`**. Three things follow, and each of them broke an earlier conclusion
of mine:

* the factory branch I had cited — `program_config.has_value() ? max_cores_per_head_batch :
  num_cores_available` (`sdpa_decode_program_factory.cpp:192-193`) — is **unreachable from this op**;
* so is the struct default of 16 (`sdpa_config.hpp:18`);
* "no config" therefore means `k_chunk_size=32` at 1 core/head, which is the **worst** setting in the
  whole grid at long context.

That block has been in the tree since 2026-05-20, months before any measurement here. The op sweep
now opens with an identity control — no config versus an explicit config spelling out that
substitution — and they are **bit-identical at all 11 contexts**, which is the evidence the previous
two attributions lacked.

**`max_cores_per_head_batch` still matters, but as a constraint rather than the lever.** Every value
above 1 returns a silently wrong answer below some context, and the boundary moves with the chunk
size, so no value above 1 is correct everywhere. 1 is both the op's own default and the only safe
choice; the config pins it so a later stage has to read why before changing it.

Ruled out by holding axes fixed: `exp_approx_mode` (approx and exact rows are **bit-identical**) and
the core grid (`8x8` == `11x10` at equal cores/head).

**The fix ships.** `DecoderConfig.decode_sdpa_k_chunk_size = 512` with
`decode_sdpa_max_cores_per_head = 1`, built into an `SDPAProgramConfig` once at construction
(`_decode_sdpa_program_config`). 512 is the **largest legal chunk**: 1024 fails to build with
`circular buffers ... grow to 2371456 B which is beyond max L1 size of 1572864 B`, which is an
op-contract blocker rather than a choice. On-model, worst case over five contexts: **0.9999939**
(shipped) vs 0.9997685 (round 2's `k_chunk_size=0`, which resolves to 128) vs 0.9985674 (no config).
And it is *faster*: 7.45 ms vs 11.53 vs 28.2 per op call at 262144 keys, so within the safe family
bigger chunks are better on both axes and there is no trade-off to make.

**What is left on the table is latency, not accuracy.** The fastest setting in the grid is
`k_chunk 256` at 16 cores/head: 1.39 ms/call, 5.4x quicker than shipped, and *more* accurate at the
advertised context (0.9997 op / 0.9999958 on-model). It is unshippable as a static choice because it
is silently wrong at 257, 1024 and 4096 keys — confirmed **on the real layer**, where it scores
0.0383 at 4096. Taking it needs per-context trace bucketing, which is `optimize`'s call.

**One thing reported rather than explained**, and worth an upstream issue: every
`max_cores_per_head_batch` above 1 makes this op return a silently wrong result below some context —
as low as PCC 0.0000 (`k32`, 16 cores/head, 262143 keys) — instead of refusing to run.
`diag_sdpa_decode.py` reproduces it with no model code, and the k-chunks-per-core table it prints is
the starting point for narrowing it.

**Three earlier root causes of mine were wrong** and are recorded here so the correction is legible:
"the dynamic `k_chunk_size` path", then "whether a program config is passed at all", then
"`max_cores_per_head_batch`". The third survived a review round and shipped. Two lessons worth more
than the fix:

* **A control that fails is evidence about your own model first.** The round-2 sweep declared its own
  control in the script ("the `max_cores=110` row should reproduce the no-config row exactly"). It did
  not — 0.0534 against 0.7664 — and I recorded the mismatch as an upstream reproducer instead of
  letting it falsify the hypothesis. It is fully explained by the substitution above: no config is
  `k32` at **1** core/head while explicit `max_cores=110` derives 55, so the rows differ in cores.
  That reproducer is withdrawn.
* **Sweeping one axis at a time cannot separate interacting axes.** Round 2 varied cores at
  `k_chunk_size=0` only and never measured the op's actual default of 32, so the axis that did all
  the work was held constant at a non-default value. The 2-D grid is what settled it.

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

**2026-08-17 ~20:30 — I edited a shell script while bash was executing it (self-inflicted, no
hardware fault).** The round-5 evidence pass was running from `/tmp/rev16.sh` when I appended two
steps to that same file. Bash reads a script incrementally and remembers a byte offset, so shifting
the offsets underneath a running interpreter can make it resume mid-statement and execute garbage.
Nothing had visibly gone wrong yet, but the run could not be trusted, so I stopped it rather than
finish it:

* killed by **explicit pid**, innermost child first (`245286` pytest -> `245285` timeout ->
  `245279` runner -> `238598` shell), `TERM` then `KILL` after 20 s. Never `pkill -f` — earlier in
  this stage a `pkill -f <pattern>` matched its own invoking command line and killed my shell;
* `timeout 60 tt-smi -ls --local` -> all 4 p300c chips present; mesh smoke -> `MESH_SMOKE_OK`.
  **No reset needed**, so none was run;
* relaunched from a copy made read-only (`chmod a-w`) so the same mistake cannot repeat.

Cost: ~20 minutes of redone gate and long-context work. Worth recording because the failure mode is
silent — a corrupted script does not announce itself, it just runs the wrong thing.

**2026-08-17 ~20:47 — device-profiler abort plus a hung `tracy-capture`, for the second time
(infrastructure, not a model result).** The round-5 perf pass crashed on its third Tracy case:

* Signature: `TT_FATAL: End marker found without a corresponding start marker`
  (`tt_metal/impl/profiler/profiler.cpp:2089`) -> `Fatal Python error: Aborted`, raised **after** the
  test itself reported `PASSED`, so it is a profiler-teardown failure rather than a model failure.
  `tracy-capture` then sat holding the device for 40 minutes.
* `tools/tt-triage.py` hung again (`timeout 180` -> rc 124, 0 bytes written), exactly as in the
  first occurrence, so there is still no triage capture for this failure mode. Recorded as a gap
  rather than glossed over: while a hung `tracy-capture` holds the device, triage cannot run.
* Recovery, in this order: killed the 8-process tree by **explicit pid**, innermost first
  (`tracy --no-capture-tool` -> `tracy-capture` -> its `sh` wrappers -> `tee` -> the `tracy` parent ->
  `run_perf.sh` -> the pass shell), `TERM` then `KILL` after 20 s; `tt-smi -ls --local` -> all 4
  p300c chips present; **`generated/profiler/` had grown to 20 GB and was cleared** (it is scratch —
  the committed evidence is the copied ops CSVs under `tracy/`); mesh smoke -> `MESH_SMOKE_OK`.
  **No reset was needed, so none was run.**
* Then re-ran **all four** Tracy cases rather than only the two that were missing, so the §5 table
  comes from one coherent run instead of being spliced across a crash.

The first occurrence (same signature, same 11 GB-scale profiler tree, same triage hang) is the reason
the profiler tree is now cleared as part of recovery. Two data points is enough to say the pattern is
a large accumulated `generated/profiler/` tree plus repeated Tracy runs in one session, not anything
this stage's kernels do.

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
| every layer kind, real config shapes, paged prefill/decode, page table, current position | README §3.1/§3.4; 276 PCC rows in `pcc.jsonl` (108 tests: 32 CPU-only + 77 device) |
| longest feasible seq/context | 262143-token prefill and position-262143 decode for both kinds (`long_context.jsonl`) |
| non-aligned lengths around chunk/page/tile boundaries | 1/32/33/64/65/128/129/1024/1025/2048/2049/3000/4096 + 262143 per kind |
| `doc/context_contract.json` | derived from evidence by `tests/write_context_contract.py`, re-checked by `test_context_contract_file_is_consistent`; **no capability reduction** |
| real-weight test passing | `test_real_weights_prefill_and_decode[linear,full]`, `pcc_real_weights.jsonl` |
| PCC >= 0.995 prefill and decode | worst in main suite 0.9999450; worst at 262144 context 0.9998960 (`longest-prefill state recurrent`), advertised-context decode 0.9999939 (§5.6 — root-caused to one `SDPAProgramConfig` field and **fixed**, not waived) |
| warmed prefill + traced warmed decode perf with tt-perf-report tables + CSV/provenance | `tracy/<kind>_<mode>/`, `perf_summary.json`, README §5 |
| runtime fallback audit clean | `test_no_runtime_host_fallback` (static: 26 runtime methods plus every module-level helper, the helper list derived from the module rather than hand-written) + `test_no_host_ops_during_forward` (dynamic: ttnn host bridges monkeypatched to raise) |
| determinism / repeated input | `test_prefill_determinism`, `test_decode_determinism` (bit-identical over 3 repeats) |
| watcher-clean run | `watcher/` — 8 passed, 5405-line log (10 dumps), `watcher_hits.txt` empty (README §3.9) |
| README + work log with commands, PCC, perf, limitations, artifacts | this file + `README.md` |

## 9. Independent stage review (round 1) and the work it produced

`$stage-review` returned **more-work-needed** on the first pass. Every finding was worked, not
argued down:

| finding | outcome |
|---|---|
| P1 — the "dynamic `k_chunk_size`" root cause was confounded: omitting the program config also flips the grid and `exp_approx_mode`, one label was wrong, and `k_chunk_size=0` (dynamic chunk path + explicit config) was never measured | **Superseded by round 2 — the conclusion recorded below was still wrong**; see §10 and §5.6. What round 1 produced: the sweep shows `exp_approx_mode` is *bit-identically irrelevant* and the chunk policy is not the discriminator, and it concluded the presence of a program config was. Added `tests/diag_decode_sdpa_onmodel.py`, the on-model control the reviewer asked for: shipped setting 0.9999954 @ ctx 1024 / 0.9985674 @ 262144 vs explicit config 0.0304429 / 0.9999958. Decision unchanged but now measured on the shipped path; README §3.8 rewritten; the fabricated `8x8` label and the "same cache" claim are gone. |
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
| **P1 — the §3.8 root cause was still confounded**: `SDPAProgramConfig::max_cores_per_head_batch` is a struct default (16) that the factory replaces with `num_cores_available` when no config is passed, so "program config present" and "16 vs 55 cores per head" were the same event, and the axis was never swept | The reviewer was right that the attribution was confounded, and the layer changed because of it — but **the conclusion this round reached was also wrong**, and round 4 overturned it. `max_cores_per_head_batch` is not the variable: the *paged* entry point substitutes its own config (`sdpa_decode.cpp:122-129`) so the `num_cores_available` branch is unreachable and the op already defaults to 1 core/head. What this round actually changed was `k_chunk_size` (0 instead of the op's 32), which is why its numbers improved. See §12; §5.6 now carries the corrected account. |
| P2 — README §5 said the two expert sparse matmuls were 81% of prefill and "the whole attention path is the remainder", overstating attention ~14x and hiding the MoE elementwise cost | Replaced with the measured three-way split (token mixer / expert matmuls / MoE dense-intermediate elementwise) for all four cases. `summarize_perf.py` now *derives* it into `perf_summary.json` — the mixer/MoE boundary is found structurally (last `LayerNormDeviceOperation` before the first sparse matmul in an iteration) rather than hand-counted, so it cannot drift from the CSVs. |
| Other concern — decode perf measured only at `cur_pos=4095`, so the advertised-context latency is unknown | Measured: the decode SDPA alone is 11.5 ms/call at 262144 keys vs ~1 ms at the profiled shape, so the advertised-context step is ~10 ms slower than the §5 table. Recorded in §5, in `test_perf.py`'s own comment, and in README §3.8's timing table. (Round 4's setting brings that op to 7.45 ms/call; the current numbers are in §3.8.) |
| P2 — `ttnn.embedding` untilizes a TILE-layout weight on **every** call, so each decode step untilized both full RoPE tables (2 x 32 MiB at the advertised context); and decode perf was measured at `supported_context=8192` without saying so | **Real inefficiency, fixed.** The tables are now stored ROW_MAJOR, so decode gathers `max_batch_size` rows instead of untilizing the whole table, and prefill tilizes only the chunk it slices. The first attempt kept *both* layouts and hit `Out of Memory: Not enough space to allocate 2147483648 B DRAM buffer` in the two real-weight tests — 64 MiB per layer of duplicated tables is real pressure once the session's cached layers are live — which is why one table plus a per-chunk tilize is the right shape. `supported_context` is stated in the perf section. |
| P2 — three refuted pre-round-1 explanations still lived in committed code and in a regenerated artifact (`test_long_context.py`'s "input-conditioning artefact of random K/V", `diag_sdpa_decode.py`'s printed "specific to the dynamic chunk path" note, `diag_long_decode.py`'s one-sided docstring) | All three corrected in place; the diagnostics were re-run so the logs no longer print the refuted story. |
| P2 — README claimed *every* artifact postdated the last source change; mtimes refuted both that and the narrated run order | Replaced with what is actually true and checkable, including how to tell whether a late source edit could matter (`git diff`; the fallback audit strips comments). |
| P2 — `measure_expert_union.py` was untracked and had no log, yet README/work_log cited its numbers; the DRAM capacity probe behind `context_contract.json`'s `usable_dram_bytes` was likewise prose-only | Script tracked, `logs/measure_expert_union.log` committed. Added `tests/probe_dram_capacity.py` + `logs/probe_dram_capacity.log` so the capacity number is a recorded measurement. |
| P2 — five staged `watcher/generated/inspector/*.yaml.gz` blobs no longer existed on disk, and 3 MB of uncompressed replacements were unignored, so no clean checkpoint commit was possible | `doc/.gitignore` excludes the inspector side-output (it is not evidence — the watcher log is), and the stale blobs are unstaged. |
| linear-attention state was not self-healing: a slot reused for a new sequence inherited the previous occupant's conv/recurrent state, hidden by every test calling `reset_state()` first | **Real trap, fixed.** A prefill at `start_pos == 0` now zeroes that slot's carry; `test_prefill_resets_linear_state_for_new_sequence` reproduces it without any reset. The `current_pos = -1` case is documented as self-healing (junk stays in-slot and is cleared by the next fresh prefill). Cost a second lesson: the new test first used a `layer_pairs` key no other test had, and since that cache is session-scoped and never evicts, two extra ~1.5 GiB layers pushed the real-weight tests into `Out of Memory`. It now reuses an existing key, and `conftest.py` logs every new key with the live count so a later OOM is traceable. |
| found while re-running: a **filtered** pytest session (`-k context_contract`) truncated `pcc.jsonl` through the same session fixture that gives the full run one file per run, so a subset run could silently replace every row of committed evidence with nothing | Fixed: filtered sessions write `*_partial.jsonl` (gitignored) and never touch the committed logs. This logging code took four attempts, all recorded because the failure mode is always "evidence quietly disappears": (1) truncating per *process* would have destroyed the five-process long-context accumulation; (2) diverting **every** log in a filtered session sent the long-context rows to `long_context_partial.jsonl`; (3) detecting subsets by `-k`/`-m` alone missed **node-id** selection, so `run_perf.sh` (which passes `test_perf.py::test_perf_prefill[linear]`) still deleted `pcc.jsonl` — visible as a `D` in `git status` after a perf run. The rule is now stated as the question that actually matters, in `pytest_collection_modifyitems`: a session may replace an owned log only if it selected nothing *and* collected the file that owns it. Verified against all six invocation shapes the stage uses. |
| the static fallback audit's helper list was hand-written and missed `_view` | The list is derived from the module now, with an explicit setup-only exemption set, so a new helper cannot drift out of the audit. |
| `test_real_weights_prefill_and_decode`'s docstring said "traced decode" but the test is eager; `current_pos` had no documented upper bound; README's watcher size and suite command were imprecise; no hard-check for skipped-expert tiles | All corrected or recorded (`work_log.md` §5.3 for the skipped-expert gap). |

## 11. Independent stage review (round 3) and the work it produced

`$stage-review` returned **more-work-needed** a third time. No P1, and it found **no correctness
defect in the shipped layer** — all five of its targeted checks passed and every headline number it
re-derived reproduced exactly. The three findings were evidence-level, and one of them was a real
unclassified anomaly:

| finding | outcome |
|---|---|
| P2 — `test_long_context.py`'s docstring still quoted **0.9986** as the current advertised-context decode PCC; that is now the *rejected* setting's number, the shipped one is 0.9997685 | Fixed at the time; superseded by round 4, which changed the shipped setting again (0.9999939) and removed the small-batch framing that this round's fix introduced. |
| P2 — four README numbers disagreed with the artifacts they cite: "97 items" (log says 99), "71 device cases" (73, contradicting the same paragraph), "6.5x" (the ratio is 6.19x), and prefill "7270/5963 tok/s" (contradicting `perf_host_summary.jsonl`) | All four corrected against their artifacts. |
| P2 — the real-weight `prefill[linear]` **max abs error of 1.2687** is 28x the synthetic-weight equivalent and 18x HF's own bf16 divergence, and neither write-up compared the maxabs column at all | **Real anomaly, now measured and classified** — see §5.3. `tests/diag_real_weight_maxabs.py` refutes both candidate mechanisms (it is not a large-magnitude element: `\|want\|` is 9.5% of the tensor max with a 118% relative error; it is not the §5.1 bf16 top-k swap: none of the worst eight tokens has a differing expert set) and identifies it as **bf16 accumulation in the gated-delta-rule recurrent state**: maxabs grows 0.063 -> 1.269 monotonically over 1 -> 16 delta chunks, and the no-recurrence `full` control at the same weights and length stays at 0.137. Bounded, not divergent — the decay ages error out, which is why the 4096-chunk advertised-context prefill is *better* (0.9999742). |

Acted on from the same review's "Other Concerns" and "Hard-Check Gaps", none of which were required:

| concern | outcome |
|---|---|
| `ttnn.mul(x, 0.0, output_tensor=x)` cannot clear a NaN/Inf (`0 * NaN = NaN`), so the README's unconditional "a reused slot starts clean" claim did not hold for a poisoned buffer | Fixed in code: both `reset_state()` and the `start_pos == 0` per-slot reset go through `_zero_`. Superseded in round 4, which replaced the implementation with `ttnn.fill(..., output_tensor=t)` and **measured** the premise: on this build `ttnn.mul(t, 0.0)` *does* clear a NaN, so this round's stated reason was wrong even though its change was harmless (§12). |
| **no control validated the long-context tail references**, which the entire advertised-context claim is measured against | Closed with `test_tail_reference_matches_full_prefill`: at seq 512 / tail 128 — a length where the *full* `hf_prefill` is affordable — both `hf_prefill_tail` and `hf_linear_prefill_tail` (with a deliberately small 128-token chunk) match it to 2e-4. CPU-only, so it runs in the main suite. |
| **the shipped SDPA setting was only ever measured at batch 1**, so its batch-independence was a source-level argument | Closed with `test_longest_decode_context_batched`: the advertised context decoded at batch 2, comparing the slot that holds it while the other sits at `current_pos = -1`. Added as a sixth case to `run_long_context.sh`. |
| the third perf bucket was labelled "elementwise" but also holds the router matmul, the shared-expert matmuls, `topk` and the scatters | README §5 now says what the bucket actually is ("the MoE minus its expert matmuls"), and `summarize_perf.py`'s docstring enumerates the contents. |
| `block_split` assumed every iteration in the window has an identical op sequence without checking | Now checked: the window must be a whole number of iterations with identical `OP Code` sequences, or the split is not emitted. |
| `write_context_contract.py` hardcoded `usable_dram_bytes` while claiming the contract is derived from evidence | It parses `logs/probe_dram_capacity.log` now, and `test_context_contract_file_is_consistent` re-checks the parsed value plus every recorded PCC against the evidence rows. |
| `to_layout` + unconditional deallocate in the prefill RoPE path would double-free if the tables were ever stored TILE — the same class as §7 items 1 and 5 | Routed through a new `_tilized` helper that returns an ownership flag, like `_subview` / `_move`. |
| README §5 treated artifact mtimes as the run-order record, but pre-commit's whitespace hooks rewrite committed text artifacts | §5 now says so, and points out that content is unaffected (each `.txt` table's totals match its own `.csv`). |
| `tile_aligned` in the contract actually tested `% 128` (PREFILL_ALIGN, not TILE) | Renamed to `aligned_to_prefill_align_128`. |

## 12. Independent stage review (round 4) and the work it produced

`$stage-review` returned **more-work-needed** a fourth time, with one P1 that changed the shipped
layer again — and this time the reviewer was right about something three previous rounds, including
two independent reviews, had let through.

| finding | outcome |
|---|---|
| **P1 — the §3.8 root cause was *still* wrong.** `paged_scaled_dot_product_attention_decode` substitutes its own program config when none is passed (`sdpa_decode.cpp:122-129`, in the tree since 2026-05-20), so the factory branch the write-up quoted is unreachable, the op already runs 1 core/head by default, and `max_cores_per_head_batch` never differed between the settings being compared. The axis that did differ was `k_chunk_size`. The round-2 sweep's own declared control row had failed and been filed as an upstream bug instead of falsifying the hypothesis | **Confirmed in source, then measured — and the layer changed again.** The op sweep is now a 2-D `k_chunk_size` x `max_cores_per_head_batch` grid opening with an identity control: no config vs an explicit config spelling out the substitution, **bit-identical at all 11 contexts**, which is the evidence the previous attribution lacked. Within the safe (1 core/head) family accuracy is monotone in chunk size, and the largest legal chunk **ships** (`decode_sdpa_k_chunk_size = 512`; 1024 exceeds L1 at 2371456 B vs 1572864 B). Advertised-context decode PCC **0.9997685 -> 0.9999939** on-model and the op is **1.55x faster** than round 2's setting and 3.8x faster than the op's own default. The withdrawn upstream reproducer is recorded as withdrawn in §5.6. |
| P2 — the advertised-context tail reference was only validated at aligned boundaries (`seq=512, tail=128, chunk=128`, every piece a multiple of the 64-token delta chunk), while the real run is `seq=262143, tail=128, chunk=2048`: a 1919-token final piece and a tail offset 63 from the global chunk grid | Closed by measurement rather than argument. `test_tail_reference_matches_full_prefill` is now parameterized over `aligned`, `ragged-head` (`seq=575, tail=128, chunk=150` — head 447 = 6*64+63) and `ragged-head-and-tail` (`tail=129`), both kinds, 6 cases. All pass against a full `hf_prefill`, so the split is exact for the non-aligned shape the contract actually uses. CPU-only. |
| P2 — eight doc numbers still disagreed with their artifacts, the fourth consecutive round for that class | Fixed, and the recurring cause addressed: §3.8 is no longer hand-written. `/tmp/write_sdpa_section.py` regenerates the entire section from the two diagnostic logs, the same way `summarize_perf.py` + `postprocess_docs.py` already own §5 and §4. Numbers that cannot be derived were replaced by pointers to the artifact. |

Acted on from the same review's "Other Concerns" and "Hard-Check Gaps", none of which were required:

| concern | outcome |
|---|---|
| the "bounded, not divergent" half of the §5.3 maxabs classification compared a *synthetic*-weight 262143-token tail number against a *real*-weight 1024-token full-window one — different weights, different window | Measured properly: the real-weight sequence sweep now runs to 4096 (64 delta chunks), so boundedness is read off one curve at fixed weights and seed. §5.3 records the result. |
| `_zero_` (`ttnn.zeros_like` + `ttnn.copy`) transiently allocates a full-size peer, so `reset_state()` at the shape the capability contract certifies — batch 32 at 262144 context, 8 GiB per K/V cache — would need a transient extra 8 GiB out of the ~15 GiB the contract leaves spare. Not exercised by any test | **Changed in code**, because it is a latent OOM in a *certified* configuration: `_zero_` is now `ttnn.fill(t, 0.0, output_tensor=t)`, which writes into the existing buffer. A new probe measures the two properties it has to have — the buffer address survives (the state tensors are baked into a trace) and NaN/Inf are cleared. The same probe **refuted the round-3 justification for `_zero_`**: on this build `ttnn.mul(t, 0.0)` clears a NaN rather than propagating it, so "`0 * NaN = NaN`" did not describe the kernel. A write is still preferable, but on the weaker ground that it cannot depend on kernel details. |
| `ttnn.scatter` was the one alias-prone call in the runtime path with no ownership check — `_router` scatters twice over one `zeros` and then deallocates it | `probe_ttnn_ops.py` asserts the three buffer addresses are distinct and that the source is still all-zero, so out-of-placeness is a checked fact rather than an inference from passing PCC. |
| nothing asserted that a *skipped* expert's `sparse_matmul` output tile is finite, even though the MoE relies on `0 * garbage` | New probe reads the 248 unselected experts' tiles and asserts all finite, recording whether they are exactly zero: they are **all exactly 0.0**, so the weighted sum is exact rather than merely well-conditioned. A second probe pins `_zero_`'s primitive (in-place, address-preserving, NaN-clearing). |
| the narrated evidence order in §7/§13 disagreed with artifact mtimes again | §13 now states the order the runner script actually ran and says which reference the freshness check uses and why — including that test-file edits after the layer is frozen are expected, so the honest claim is about the shipped layer, not about every file. |

## 13. Independent stage review (round 5) and the work it produced

`$stage-review` returned **more-work-needed** a fifth time. No P1, and it independently corroborated
round 4's mechanism from a direction this stage had not looked at: `sdpa_decode_program_factory.cpp`
hard-codes `im_df = stats_df = Float16_b`, so the flash-decode running output and running max/sum are
bf16 CBs regardless of `fp32_dest_acc_en` — which is *why* the k-chunk count is the only user-visible
lever. Seven P2s, one of them a latent correctness defect in a configuration the docs advertise.

| finding | outcome |
|---|---|
| **P2 — `sdpa_chunk` was disconnected from the alignment check it is documented to control.** `prefill_forward` rejected `start_pos % PREFILL_ALIGN` (a module constant, 128) while the op received `cfg.sdpa_chunk` as `q_chunk_size`. So `sdpa_chunk=256` accepted `start_pos=128` and the op computed chunk index `128 // 256 == 0` — the silently-wrong causal-mask placement the check exists to prevent — and `sdpa_chunk=32`, documented as "the lever", changed nothing | **Fixed in code.** The runtime check now divides by `cfg.sdpa_chunk` (the value the op actually gets), and `__post_init__` requires `sdpa_chunk` to be a tile multiple that divides `PREFILL_ALIGN`, so the 256 case is rejected at construction instead of being silently wrong. `test_sdpa_chunk_and_start_pos_alignment_agree` pins both halves plus the divisor the runtime path reads. CPU-only. |
| **P2 — README §5's "attention is rounding error" is true only of the profiled shape**, and it is the stage's main handoff to `optimize`. Chunked SDPA's key length is `chunk_start_idx + Sq`, so `full`'s per-chunk mixer cost grows with position while the MoE's does not; the table is one chunk at `abs_pos = 0` | **Quantified from artifacts already committed, no new hardware.** Extrapolating the per-chunk cost to the 128 chunks of a 262143-token prefill: `linear`, whose mixer is position-independent, lands within 0.09 s (0.2%) of its measured run — which is the control that makes the `full` row readable, and `full` is **12.8 s / 26% short**. So the `full` attention path is ~27% of an advertised-context prefill, not 1.3%. §5 now carries that table, the bullet is scoped to the shape it was measured at, and the derivation is generated so it cannot drift. |
| P2 — three rows of README §3.1, the headline PCC table, disagreed with `pcc.jsonl` (`decode[full]` 0.9999849/pos=149 vs 0.9999871/pos=140; `decode-ragged`; `traced-decode[full]`). The table was generated against the gate run, not the committed final one | **Closed as a class, not an instance** — see below. §3.1 is now generated, and a test fails on drift. |
| P2 — four more counts stale after round 4 grew the suite: "99 items"/"101 passed"/"28 CPU-only" (105/105/32), "21 device op-behaviour probes" (24), the chunked-SDPA probe PCC (0.999790 vs 0.999780) | All derived from the committed logs now. |
| P2 — the committed SDPA diagnostic **asserted a mechanism its own grid refutes**: "values < 1 chunk per core are where the op goes silently wrong". Counter-examples in the same log run both ways — `k32`/2 cores at context 128 has 2.00 chunks per core and is wrong (0.7250), `k512`/2 cores at 257 has 0.50 and is fine | Removed. The table is now labelled raw data for narrowing, with the counter-examples named in the docstring. This is the same class round 2 caught ("refuted explanations still living in committed code and in a regenerated artifact") and I reintroduced it, which is worth recording. |
| P2 — two committed derivations claimed `exp_approx_mode = nullopt` resolves to **false**; `sdpa_decode_program_factory.cpp:211-213` resolves it to **true**. So the identity control was not literally "the substituted default spelled out", even though it came out bit-identical | Both corrected. `OP_DEFAULT` now passes `exp_approx_mode=None`, so the control *is* the substituted config, and a new held-axis row measures the unset case. The shipped config keeps `False` deliberately, which the docstring now says is a difference from the default rather than a match. |
| P2 — the freshness command in README §5 / work_log §13 did not produce the result it claimed, and `diag_decode_sdpa_onmodel.txt` — the artifact that *selected* the shipped config — predated the final layer source | Removed the exception rather than documenting it: the round-4 evidence pass now re-runs **both** SDPA sweeps inside the pass, after the last source edit. The freshness wording no longer names a file that changes every round. |

Acted on from the same review's "Other Concerns" and "Hard-Check Gaps":

| concern | outcome |
|---|---|
| **the renderers lived in `/tmp`**, so "generated, cannot drift" was not reproducible from a checkout | The real fix for five rounds of number drift: `tests/render_docs.py` is committed, and it owns §3.1, §3.8, §5 and the scattered counts. It also refuses to run when a provenance log is missing — rendering during an in-flight suite reads a half-written `pcc.jsonl`, which happened once while fixing this round. |
| nothing *checked* the hand-written doc numbers | `test_docs_match_artifacts` (CPU-only, in the main suite) re-derives §3.1's per-family `n` and worst PCC, the three evidence row counts, the suite and probe counts, and every PCC quoted in §4, and fails on any mismatch. Negative-tested: reverting `decode[full]` to the stale 0.9999849 makes it fail with exactly that diff. |
| "3.8x faster than the op default" used the `no config` timing row, not the `op default` row | The ratio now comes from the row it names, and both are quoted so the run-to-run spread is visible. |
| `p_fill_in_place` covered one small bf16 tensor, while `_zero_` is also applied to the fp32 recurrent state and to 8 GiB paged caches | Extended to bf16 **and** fp32 over a multi-tile shape. |
| `logs/test_suite_gate.log` was committed but unlisted; §7 item 6's unary group total was quoted as an exact figure; the 7.45 ms decode-SDPA figure is batch 1 against a batch-32 profiled row | All three fixed: the gate log is described (and marked as not the authoritative run), the unary total is derived, and both batches are now named where the comparison is made. |
| `_record_contract` is a one-way `max()` ratchet in a file the stage treats as derived | Documented as a convenience writer, with the two things that stop it going stale: the file is regenerated from `long_context.jsonl`, and the contract test asserts *equality* against the evidence rows, so an inflated field fails. |

## 14. Independent stage review (round 6) and the work it produced

`$stage-review` returned **more-work-needed** a sixth time. No P1. It re-derived essentially the whole
stage successfully — §3.1, §4, §5, §3.8's four tables, the watcher run, the sparse-matmul derivation,
the non-aligned coverage and both operational incidents from round 5 — and then found three things,
one of which was a **correctness defect this stage introduced in round 5**.

| finding | outcome |
|---|---|
| **P2 — round 5's `sdpa_chunk` fix widened the `start_pos` contract into a silent page-write bug.** Dividing by `cfg.sdpa_chunk` closed the `sdpa_chunk=256` hole but opened one downwards: at `sdpa_chunk=32` a `start_pos` of 32 passed validation, and the paged fill computes its block offset as `abs_pos // block_size = 32 // 64 = 0` — writing the chunk into the wrong page while the SDPA masks as if it were elsewhere. Same for the padded end, which is `PREFILL_ALIGN`-rounded and could overrun the RoPE and page-table rows | **Real defect, fixed** (README §7 item 10). `start_pos` has **three** consumers with independent alignment requirements — the SDPA chunk index, the paged block offset, and the padding bound — and the accepted value is the *maximum*, not any one of them. The runtime bound is `PREFILL_ALIGN` again, `__post_init__` now asserts it is a multiple of **both** `sdpa_chunk` and `block_size` so that single bound is sufficient, and the padded end is bounds-checked explicitly. The lesson worth keeping: "widening a contract" is a claim about every consumer, and I checked one. The test now asserts the block-alignment property instead of the "lowering can only widen" claim that *was* the bug, and README §2/§9 no longer advertise `sdpa_chunk` as a lever it never was. |
| **P2 — the committed PCC evidence was written by the suite run the README calls *not* authoritative, after the docs were rendered and checked.** Round 5's pass ended with a gate re-run, so `pcc.jsonl` postdated `render_docs.py` and `test_docs_match_artifacts` by three minutes; inside that last run the docs test stood down by its own newer-artifacts rule. The numbers happened to agree — `--check` passes — but the pass had not established that | Ordering fixed at the source: the provenance-log suite now runs **exactly once and last** among the writers, then the docs are rendered from those files, then `render_docs.py --check` proves the match and its output is committed as `logs/render_docs_check.log`. No suite run happens after the render. |
| **P2 — doc numbers disagreed with artifacts for the sixth round running, in the three places the new machinery does not reach**: README §3.2's `decode[full]` row still carried round 3's numbers (0.9999856/0.9999908, 0.54%/0.44% vs the artifact's 0.9999864/0.9999911, 0.52%/0.43%); work_log §4 said "21/21 ok" in three places where the log says 24/24 (the README's copy had been fixed, but nothing read the work log); and the freshness command listed eight files while the prose claimed six | Coverage extended rather than the instances patched: `render_docs.py` now generates §3.2 from `pcc_real_weights.jsonl`, rewrites the work log's probe and suite counts, and **enumerates the freshness command's actual output** into both documents. `test_docs_match_artifacts` now checks §3.2 row by row and reads `work_log.md`. The pattern across six rounds is unambiguous — a number that is written by hand goes stale — so the response each time is to move it into the renderer, not to correct it. |

Acted on from the same review's "Other Concerns":

| concern | outcome |
|---|---|
| `tracy/README.md` offered `run_perf.sh decode linear` for regenerating one artifact, while `run_perf.sh` unconditionally deletes `perf_host_summary.jsonl` — so a single-case rerun silently left a 1-row file against four quoted cases | Fixed in the script, not just the doc: the reset now happens **only** on a full run, and a partial run warns on stderr. The doc now says to regenerate all four and explains why. |
| the two remaining raw `ttnn.to_layout` + unconditional-deallocate sites in the MoE are the exact aliasing shape round 3 routed through `_tilized` everywhere else | Both go through a new `_row_major` now; `_tilized` and it share a `_relayout` core, so **every** `to_layout` on the runtime path answers the ownership question instead of assuming it. |
| §3.8's `token*8` row claimed the attention branch is "bit-identical" under input scaling; one of the three columns moves in its last digit (0.9874978 -> 0.9874977) | Generated text corrected to "unchanged to within 1e-7", naming which columns repeat exactly. |
| `triage/README.md` explained only the first empty triage capture; README §8 described the directory in the singular | Both cover the second occurrence now. |
| README §3's count sentence read "106 passed ... 32 CPU-only + 75 device" without saying 107 were collected | The generated line now states the collected total and names the one skip. |

## 15. Independent stage review (round 7) and the work it produced

`$stage-review` returned **more-work-needed** a seventh time. No P1, and it re-derived the substance
of the stage independently — §3.1, §3.2, §4, §5's four perf rows and the sparse derivation, §3.8's
identity control and all four tables, the watcher log, 24/24 probes — and confirmed the `start_pos`
contract is correct against every consumer. Four findings, two of which were mechanisms this stage
had added and which were not actually working.

| finding | outcome |
|---|---|
| **P2 — a prefill writes zero-valued K/V for its padded tail into the paged cache, and a continuation would attend over it.** The fill works in whole `block_size` pages, so a `seq_len` of 1000 writes rows 1000..1023 as exact zeros (`rms_norm(0) = 0`, bias-free projections, `RoPE(0) = 0`). Harmless inside the call — causal masking and the output slice discard them — but a later chunk scores `q . 0 = 0` on those slots, takes `exp(0) = 1` of softmax weight each and returns a silently diluted result. And since `start_pos` must itself be `PREFILL_ALIGN`-aligned, a non-aligned chunk has **no** legal continuation at all | **Real hazard, now guarded.** `prefill_forward` tracks a per-slot logical high-water mark and requires a continued prefill to resume exactly where the previous one ended; `start_pos == 0` always starts fresh, and `reset_state()` clears it. `test_prefill_continuation_must_be_contiguous` pins the rejection and the legal contiguous case, per slot, for both kinds. The bookkeeping is a host-side `dict[int, int]` — never a tensor, never read by the device path — so it does not touch the fallback audit or trace safety. The first version of the check sat *before* the context-bound checks and changed an existing contract message; it now sits after them. |
| **P2 — the freshness-list generator was self-disabling.** Its anchor required the block to end on a specific prose line, but the block it writes continues past that line — so it could rewrite the block exactly once and never match again. It reported `WARN`, which (unlike `MISS`) does not fail the run, while both documents asserted the list was generated and therefore could not disagree with the command | **Fixed and negative-tested.** The block is delimited by explicit HTML-comment markers (see `render_freshness_list`), a missing marker is now **`MISS`** like every other anchor, and `--check` exits 1 when one is removed. Verified idempotent. Fixing it surfaced the same class once more: `render_counts`' regex had stopped matching the tally sentence its own previous run wrote, so the suite count silently stayed at 106 while the log said 108. That pattern now matches on structure and reports `MISS`. |
| **P2 — the committed `render_docs*.log` / `test_docs.log` did not reproduce from the committed tree**, because I re-ran the renderer by hand after the pass had captured them. Same ordering class round 6 recorded as fixed at the source | The closing sequence is re-run as the last step and its logs committed, so `diff <(render_docs.py --check) logs/render_docs_check.log` is empty. Round 6's fix was to the *pass*; what it missed is that a manual render afterwards silently invalidates the captured proof. |
| **P2 — "chunked SDPA is silently wrong on a misaligned `chunk_start_idx`" is false for the path this layer takes.** With an explicit `program_config` and the scalar offset — exactly what `_full_attention_prefill` passes — `sdpa_device_operation.cpp:277-292` `TT_FATAL`s on `chunk_start_idx % q_chunk_size` and `% k_chunk_size`. Only the device-tensor form skips those checks, and this layer does not use it | Corrected in README §7 and §9, the `prefill_forward` comment and the test docstring. The `PREFILL_ALIGN` bound is unchanged and still necessary, but for the **right reason**: the genuinely silent consumer is the paged fill's `abs_pos // block_size`, which truncates. Worth recording that this stage cited the `>= 0` check as if it were the only validation for three rounds, in a section whose whole subject is reading op contracts carefully. |

Acted on from the same review's "Other Concerns" and "Hard-Check Gaps":

| concern | outcome |
|---|---|
| §5's 27% figure mixes device attention growth with host program creation, and the split is derivable from the committed CSV | Split, and generated so it stays honest: chunked SDPA scales with `Sk = chunk_start_idx + Sq`, so the first chunk's `SDPAOperation` summed over 128 chunks is **~7.2 s of device attention** (15% of the prefill), leaving ~5.6 s of program creation. They have different fixes — an SDPA configuration versus the device-tensor offset form — so quoting only the combined number would hand `optimize` one problem where there are two. |
| no traced-replay correctness case crossed a change in the decode SDPA's **k-chunk count** — `test_traced_decode_pcc` replayed at 256/257, both inside the first 512-key chunk | Replay positions moved to **511 and 512**, which straddle the boundary (1 chunk -> 2) through one captured program, at no extra memory since the `layer_pairs` key is unchanged. 511 also makes the prefill non-aligned, so the padded-tail path is in the mix. |
| `__post_init__` asserted `PREFILL_ALIGN % sdpa_chunk` and `% block_size` but not `% delta_chunk_size`, the one leg of the triangle `PREFILL_ALIGN`'s own docstring names | Added for linear layers. |
| README §2 justified not bounding `current_pos` above by "would need a host read, which the runtime path forbids" — but `_decode_rope` already clamps the other end on device, so a symmetric `ttnn.minimum` needs no host read either | Re-justified honestly: an exact *rejection* would need a host read; a clamp would not. The bound is unchecked because nothing has needed it, not because it is impossible, and the note says what a serving stage should add. |
| §9 said "the accuracy question is closed" next to an attention branch measured at 0.987 | Scoped to "closed **at the layer contract**", with the branch number and an explicit instruction that stage 5 measure it end to end rather than inherit the word. |

**A postscript that belongs in the record.** While writing the row above about the self-disabling
sentinel, I quoted the sentinel *literally* in this table. `render_freshness_list` matches
`BEGIN .*? END` non-greedily, found that quote first, and rewrote everything from it to the real end
marker — destroying the rest of this section, which was then committed. It reported `same`, because
from its point of view it had rewritten a block successfully. The repair added a cardinality guard
(exactly one marker pair or refuse, negative-tested), and round 8 found the guard was satisfied by
the *stray* marker while the real block sat elsewhere, so the section stayed broken through another
round. Written down because three separate self-checks failed quietly in two rounds, and the shape
was the same every time: the mechanism reported success for something it had not done.


## 16. Independent stage review (round 8) and the work it produced

`$stage-review` returned **more-work-needed** an eighth time — but with a materially different
shape: **no P1 and no correctness defect in the shipped layer.** The reviewer re-derived §3.1, §3.2,
§4, §5 (all four perf rows, the three-way split, the sparse derivation, the 27% extrapolation),
§3.8's identity control and all four tables, the maxabs sweep, the watcher run and the probe count
from the committed artifacts, and re-checked the whole `start_pos` alignment contract against every
consumer in source. All of it reproduced. Both findings were in the documentation / self-check layer.

| finding | outcome |
|---|---|
| **P2 — this work log was still broken, and every guard passed.** The round-7 write-up quoted the freshness sentinel literally inside a table cell; that stray marker was the *only* pair in the file, so round 7's new "exactly one begin / one end or refuse" guard was satisfied and the renderer kept writing its block **into §15's finding table**. §15 was left with one truncated row, 43 lines duplicated from the next section, and its other three findings gone; §17 carried a stale hand-maintained copy of the list, one entry short of what the command returns. `render_docs --check`, `test_docs_match_artifacts` and the committed check log all reported success | §15 rebuilt from the review record with the marker escaped, the duplicate block removed, the stale hand copy deleted, and the generated block returned to the section whose prose introduces it. The guard was cardinal but not positional — it counted markers without asking whether they were in the right place — which is why a *misplaced single* marker sailed through. §15 now carries a postscript describing the whole sequence, because three separate self-checks failed quietly across two rounds and the shape was identical every time: **the mechanism reported success for something it had not done.** |
| **P2 — the round-7 continuation guard was an undocumented public rejection path**, and three documented statements contradicted it: README §2's `start_pos` block described streaming with no contiguity condition, §4's continuation row listed alignment as the only requirement, §9 enumerated the consequences for later stages as alignment only, and `context_contract.json` recorded `start_pos_alignment` and nothing else | Documented in all five places plus the module docstring, with the consequence stated plainly rather than implied: a chunk whose `seq_len` is not a multiple of 128 **cannot be continued at all**, and a KV cache restored outside this layer object cannot be continued into — which is exactly the prefix-cache handoff §9 exists to warn a serving stage about. The hazard itself is now README §7 item 11. The goal contract asks for a *documented* prefill/decode contract; a guard that only exists in code and a commit message does not satisfy it. |

Acted on from the same review's "Other Concerns":

| concern | outcome |
|---|---|
| the continuity check ran before the `page_table is None` and `user_id >= max_batch_size` checks, so an out-of-range `user_id` reported "must continue from 0" | Moved after them. The round-7 fix had moved it after the *context-bound* checks only, which was two of the four. |
| `self._prefill_end[user_id]` was written **before** the chunk loop, so an exception mid-prefill left a mark claiming work that never completed | Recorded at the end of `prefill_forward`, after every chunk has run. |
| `decode_forward` does not advance the high-water mark, so prefilling a slot that has since decoded is accepted | Left as is and **documented** at the record site: decode is replayed from a captured trace, so host bookkeeping inside it would differ between eager and traced execution. That pattern is out of contract rather than rejected — `reset_state()` and start from `start_pos = 0`. |
| §3.8's "`1` is the only `max_cores_per_head_batch` value with no bad cell **anywhere in the grid**" overstates it — the `(k_chunk 32, 1 core)` row is 0.7664 at 262143 keys | Scoped to the cores axis, which is the claim actually being made. |

## 17. Independent stage review (round 9) and the work it produced

`$stage-review` returned **more-work-needed** a ninth time. No P1 and, for the second round running,
**no correctness defect in the shipped layer** — the reviewer re-derived every table in §3.1, §3.2,
§3.8, §4 and §5 from the committed artifacts, plus the watcher run, the probe count, the freshness
list and the evidence-pass ordering, and could not break any substantive claim. Three findings: one
documentation-integrity defect, one refuted claim still standing in the deliverable's own API
docstring, and one derived performance number whose model the op source contradicts.

| finding | outcome |
|---|---|
| **P2 — the work log still contained the duplicated block §16 records as removed.** 49 lines were byte-identical between §15 and §17, created by the manual repair in `3d40b51f646` and left in place by round 8, whose finding table nonetheless says "the duplicate block removed". Round 8's marker guard is *cardinal* — one begin, one end — so it cannot see prose that was copied | Duplicate deleted, the generated freshness block returned to the section whose prose introduces it (§17, the evidence-order discussion), and §16's remedy row corrected to say what actually happened. Added the structural check that was missing: `render_docs.py` now fails if either document repeats a paragraph of 400+ characters. Negative-tested — re-injecting a duplicate reports `MISS ... repeats 1 paragraph(s)` and exits 1. That is the third distinct guard added after a doc mechanism failed quietly, and the first one that looks at the *shape* of the document rather than at its own markers. |
| **P2 — the module docstring still carried the claim round 7 refuted**, four lines above the paragraph round 8 added: `start_pos` "must be a multiple of `PREFILL_ALIGN` (128) == `sdpa_chunk`, because ... a misaligned offset is silently wrong rather than an error". Both halves are contradicted by the same file 700 lines later, and equating `PREFILL_ALIGN` with `sdpa_chunk` is exactly the reasoning that produced the round-5 silent page-write bug | Rewritten to match `prefill_forward`: `PREFILL_ALIGN` is the **maximum** of three independent constraints, the SDPA offset is validated *loudly* by the op, and the consumer that fails silently is the paged fill's `abs_pos // block_size`. The goal contract's first bullet is a **documented** prefill/decode contract, and the module docstring is that document — it is what a later stage reads first. Round 7 fixed four copies of this claim and missed the fifth. |
| **P2 — §5's device-attention / program-creation split rested on a scaling model the op contradicts.** It assumed cost ~ `Sk = chunk_start_idx + Sq`, i.e. chunk *i* costs `(i+1)x` chunk 0. The chunked entry point is **always causal**, so q-chunk *j* of call *i* walks `16i + j + 1` k-chunks, not all of them — the growth is ~1.9x steeper, which would put the device term above the entire excess being split | **Measured instead of modelled.** New `diag_prefill_sdpa_scaling.py` times `chunked_scaled_dot_product_attention` at the real prefill shapes across chunk indices: growth is linear at **1.69x per chunk**, and chunk 15 costs **26.4x** chunk 0 — against 29.2x for the causal model and 16x for the rectangular one the doc assumed. Summed over 128 chunks that is ~12.1 s, i.e. **94% of the 12.8 s excess is the attention op itself**, leaving ~0.7 s for program creation rather than the 5.6 s the old model implied. §5 and §6 limitation 6 now say so: the handoff to `optimize` is **one** problem, not two, and the device-tensor `chunk_start_idx_tensor` form is a tidy-up rather than the lever. |

Also from round 9's "Other Concerns": §4 now says why the batch-2 advertised-context row is
*expected* to be bit-identical to the batch-1 row (at 1 core per (slot, KV head) adding a slot cannot
perturb the other slot's accumulation order), so an identical triple reads as the result it is rather
than as a copied number.

## 18. Commits

Local only; nothing pushed.

| SHA | contents |
|---|---|
| `12c947d9147670eb0b3a9b23136635b89de709f3` (`12c947d9147`) | the whole stage: `models/autoports/qwen_qwen3_6_35b_a3b/**` plus the `conftest.py` guarded-import fix (README §7 item 4) |
| `b2bb054161fcde8a1664f848ce0f35ad3f58aeea` (`b2bb054161f`) | records the SHA above |
| `ea58fe8fa7ae1138dbc35a363b6b817faeeed605` (`ea58fe8fa7a`) | **review rounds 1 and 2**: the fixes in §9 and §10 — ROW_MAJOR RoPE tables, the per-slot state reset, and the `decode_sdpa_max_cores_per_head = 1` config whose *rationale* round 4 later overturned — and every artifact regenerated against that code |
| `b5c71c62624f984353960c1d6c266dc2fbd428d2` (`b5c71c62624`) | **review round 3** (§11): the classified real-weight maxabs anomaly, `_zero_`, `_tilized`, the provenance-log reset rule, and two new tests |
| `60e2a90711448a9fd48366919a17f61a37026153` (`60e2a907114`) | **review round 4** (§12): the corrected decode-SDPA root cause and the shipped `decode_sdpa_k_chunk_size = 512`, the 2-D sweep with its identity control, `_zero_` via `ttnn.fill`, the ragged tail-reference cases, three new op probes, the extended maxabs sweep, and every artifact regenerated against that code |
| `a08c264e60011b92fadb6f88d6b46660a9645888` (`a08c264e600`) | **review round 5** (§13): the `sdpa_chunk` alignment fix, the position-dependence correction to §5, the committed `tests/render_docs.py` + `test_docs_match_artifacts`, the corrected `exp_approx_mode` derivations, and every artifact regenerated against that code |
| `e1c0248979bb56d154a3c2a8bdea99873266e501` (`e1c0248979b`) | **review round 6** (§14): the `start_pos` alignment fix (§7 item 10), the corrected evidence-pass ordering, §3.2 + the work-log counts + the freshness list moved into `render_docs.py`, `_row_major`, the `run_perf.sh` partial-run guard, and every artifact regenerated against that code |
| `051a5da981ce42d0840014e802da6f1613d33139` (`051a5da981c`) | **review round 7** (§15): the prefill-continuation guard, the sentinel-anchored (and now fatal) freshness renderer plus the tally-regex fix, the corrected chunked-SDPA validation claim, the §5 device-vs-program-creation split, the k-chunk-straddling traced replay |
| `3d40b51f646ea340ff56201d7799861286420325` (`3d40b51f646`) | restores the Commits section this renderer deleted, and makes a duplicated marker fatal (§15's last row) |
| `1c05b286efa8065b4647467d3ff3f2352ba74336` (`1c05b286efa`) | **review round 8** (§16): the rebuilt §15 and the freshness block returned to its section, the continuation contract documented in five places plus the contract generator, the check-ordering and high-water-mark fixes, and every artifact regenerated |
| `4a953aa155d013f5e8862037d7417bc718a61df3` (`4a953aa155d`) | **review round 9** (§17): the de-duplicated work log and its new duplicate-paragraph guard, the corrected module docstring, and the **measured** prefill-SDPA scaling (`diag_prefill_sdpa_scaling.py`) that replaced §5's modelled split |
| later commits | documentation only: SHA records and analysis notes that could not be written before the commits they describe |

A table cannot contain its own SHA, so the last rows are deliberately open-ended rather than
chased with one more commit each time. The authoritative list is:

```bash
git log --oneline 12c947d9147^..HEAD -- models/autoports/qwen_qwen3_6_35b_a3b
```

Every commit is local; **nothing was pushed**. Only one file outside the autoport directory was ever
touched (`conftest.py`, in the first commit).

All evidence was regenerated after the last change to the shipped layer, one hardware command at a
time. The round-4 pass ran in the order `/tmp/rev14.sh` records: main suite (gate) -> long-context
(6 pytest processes) -> `write_context_contract.py` -> contract test -> op probes -> long-decode
diagnostic -> real-weight maxabs diagnostic -> watcher -> Tracy perf (4 cases) ->
`summarize_perf.py` -> main suite again, so `pcc.jsonl` is from the shipped code. The two SDPA
sweeps ran *before* that pass, because their result is what selected the shipped configuration.

Check it rather than trust it — but check it against the right reference. The shipped layer is
`tt/functional_decoder.py`:

```bash
find models/autoports/qwen_qwen3_6_35b_a3b/doc -type f \
  ! -newer models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py
```

<!-- render_docs: freshness list (generated) -->
It currently lists exactly these, and each is explained:

* `.gitignore`
* `functional_decoder/logs/measure_expert_union.log`
* `functional_decoder/logs/probe_dram_capacity.log`
* `functional_decoder/tracy/.gitignore`
* `functional_decoder/triage/README.md`
* `functional_decoder/triage/tt-triage-perf-hang.txt`
* `functional_decoder/triage/tt-triage.txt`
* `functional_decoder/weight_stats/layer_00.json`
* `functional_decoder/weight_stats/layer_03.json`

None of them can have been produced by a different version of the shipped layer: the
`weight_stats/*.json` are checkpoint-derived, the `.gitignore` files describe the artifact policy,
`triage/` records the two incidents in work_log section 6, and the `logs/` entries are the
expert-union and DRAM-capacity probes, neither of which imports `functional_decoder`. The list is
generated by `tests/render_docs.py`, which excludes only the three logs the pass writes *after*
rendering (`render_docs.log`, `render_docs_check.log`, `test_docs.log`) -- running the raw command
may show those, and their age says nothing about the layer.
<!-- /render_docs -->

**That command deliberately does not use the newest file under `tests/`, and round 4 was right to
flag the earlier wording for glossing over it.** Test and harness files are edited after the layer
is frozen — a docstring correction, a new parameter, the provenance-log reset rule — so the honest
claim is narrower: no artifact here was produced by a *different version of the shipped layer*. To
audit the rest, list what predates the newest source file of any kind and check each hit against
`git log -p`:

```bash
find models/autoports/qwen_qwen3_6_35b_a3b/{tt,tests} -name '*.py' -o -name '*.sh' | xargs ls -t | head -1
```

Deliberately not naming the file here, because it changes every round and the last three write-ups
went stale doing exactly that. What matters is the rule for reading the result: a hit under `tests/`
is expected and benign when the change is a docstring, a new test parameter, or the provenance-log
reset rule (which governs only which of `pcc.jsonl` / `pcc_real_weights.jsonl` a session may replace
— both rewritten by the final main-suite run — and which the standalone `diag_*.py` scripts do not
load at all). A hit that a change to `tt/functional_decoder.py` could explain is a real staleness
finding, and there are none: the round-4 pass re-ran both SDPA sweeps *after* the last edit to the
shipped layer, so no decision artifact predates it.

**Cross-process reproducibility, measured.** Rounds 5 and 6 both noticed that two nominally
identical suite runs had produced different worst-case rows in README §3.1, and round 6 recorded it as
unexplained residual risk ("cross-process PCC values are not perfectly reproducible"). They are. The
round-6 pass runs the whole suite twice — once as the gate, once as the authoritative writer — so
snapshotting `pcc.jsonl` between them measures it directly: **all 276 rows are bit-identical**, same
label sequence, and `pcc`, `maxabs` and `rel_rms` all match exactly. The earlier discrepancy was not
run-to-run variance; it was §3.1 being rendered from a *partially written* `pcc.jsonl` (the renderer
had been run while a suite was mid-flight — the incident that added the missing-artifact guard to
`render_docs.py`). Two separate reviews reached for non-determinism as the explanation, and the actual
cause was an ordering mistake of mine, which is why the pass now writes the provenance logs exactly
once and last.

Three numbers reproduced **bit-identically** across independent re-runs, which is worth recording
because it makes the determinism claim concrete: the advertised-context PCCs, the on-model
comparison, and the op sweep's identity control (no-config vs the substituted default, max abs diff
exactly 0.0 at all 11 contexts). Only wall-clock and per-op device times moved, by <0.2%.

Pre-commit reformatted the Python sources (black/isort/autoflake — formatting only, no import
or semantic changes) and rejected two >500 KB artifacts; the full suite was re-run after the
reformat (**95 passed** — the suite as it stood at that commit; it has grown since, and §8 records
the current count) before committing, and the artifact policy is documented in `tracy/README.md`.
