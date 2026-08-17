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

Usable device DRAM, probed by allocating 512 MiB DRAM tensors until the bank manager refused:

```
CAP stop at 31.5 GiB: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:462
CAP allocated_GiB 31.5
```

That 31.5 GiB is the number the capability contract's byte accounting is measured against
(`doc/context_contract.json` -> `device_capacity_evidence`).

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

Findings that changed the design:

| probe | result | consequence |
|---|---|---|
| `ttnn.slice` non-aligned on dim -2 | ok, pcc 0.999999 | conv shift-MAC uses plain slices |
| `ttnn.sparse_matmul` | `program_config` is **mandatory** (`MatmulMultiCoreReuseMultiCast1DProgramConfig`, `mcast_in0=True`) | `_sparse_program_config` builder; `Kt % in0_block_w == 0` snap |
| `ttnn.sparse_matmul` core grid | the full 11x10 grid is rejected; 8x8 works | `_SPARSE_CORE_GRID = (8, 8)` |
| `rotary_embedding_hf` decode mode | requires a **sharded** input | decode uses a manual partial RoPE on interleaved tensors instead |
| `paged_update_cache` | requires a **height-sharded** update tensor | decode keeps `nlp_create_qkv_heads_decode`'s sharded K/V for the cache write and only moves Q to DRAM for SDPA |
| `ttnn.topk` | **bfloat16 only** (fp32/bf8 rejected/worse) | router logits are cast to bf16 for top-k; see §5 |
| `ttnn.rms_norm` precision | relmax 1.9e-2 (bf16), 3.7e-3 (fp32+HiFi4) | manual fp32 l2-norm for delta-rule q/k (§3.4) |
| `ttnn.softplus` | fp32 relmax 3.4e-4, bf16 relmax 3.2e-2 | the `a + dt_bias` path is forced to fp32 (`dt_bias` reaches +15.6) |
| `ttnn.exp` with a `-1e30` additive mask | -> 0, no NaN | decay masks are added **before** `exp`; cumulative gates reach ~-1e5 so `exp` of the unmasked upper triangle would overflow to `inf` and produce `0*inf = NaN` |
| `ttnn.permute` `(0,3,2,1)` / `(2,3,0,1)` | ok | gets `beta`/`g` from `[1,1,T,32]` into the `[.., heads, .., 1]` broadcast layout without a relayout |
| whole-tensor `ttnn.slice` | returns an **aliasing view** (same buffer address) | `_subview` / `_owned_slice` / `_move`; see README §7 |
| `ttnn.reshape` | **always** an aliasing view | `_view` helper documents that the input stays the owner; deallocating both double-frees, deallocating the view early is a use-after-free |
| `ttnn.split(t, 1, dim)` with one output | may return the input | per-piece buffer-address check before deallocating in the delta-rule scan |

## 5. Precision analysis

### 5.1 bf16 top-k router selection

`ttnn.topk` only accepts bf16, so expert selection runs on bf16 logits. Measured
**set-agreement with fp32 torch top-8 of 256: 94.2%** — i.e. ~6% of tokens swap their 8th
expert for the 9th. Quantified end-to-end on CPU with **real** layer-3 weights by emulating
exactly that (bf16 logits for selection, fp32 logits for the weights):

```
full-decoder-output PCC (fp32 router vs bf16-logit router): 0.99999963   rel-rms 0.086%
token expert-set agreement: 95.7%
```

Negligible, because the swapped expert is the *lowest-weighted* of the eight and the residual
dominates the layer output. Recorded here so a later stage does not re-investigate it.

### 5.2 What the reference itself is worth

HF's own bf16-vs-fp32 divergence on these layers, real weights, seq 1024:

```
layer 0 (linear): PCC 0.9999950  rel_rms 0.316%  maxabs 0.0706
layer 3 (full):   PCC 0.9999949  rel_rms 0.319%  maxabs 0.0409
```

The TTNN bf16 implementation lands at 1.06% (linear) / 0.64% (full) rel-RMS against the
**fp32** reference, i.e. 2-3.3x HF's own bf16 sensitivity, with PCC >= 0.99994. That is the
expected cost of bf16 activations plus bf16 sparse matmuls, and it is why the default 0.995 bar
needs no model-specific exception.

### 5.3 Router weight computation

HF softmaxes over all 256 experts, takes the top 8 and renormalises by their sum. Softmax is
monotonic, so top-k of the raw logits picks the same experts, and a softmax over just those 8
values *is* the renormalised weight. The implementation therefore does top-8 then an explicit
fp32 max-subtract / exp / sum / div (measured `maxabs 1.7e-3` vs 3.6e-2 for a plain
`ttnn.softmax`), and keeps a separate binary mask as the sparse-matmul sparsity so the pattern
is exactly 8 experts per token even when a routing weight rounds to zero in bf16.

### 5.4 Long-context decode-SDPA investigation

The one visibly-off number in the stage (`test_longest_decode_context[full]`, PCC 0.9986 at
position 262143 vs ~0.99999 everywhere else) was root-caused, not annotated. Two diagnostics,
both kept as scripts with their output in `logs/`:

| script | log | what it settles |
|---|---|---|
| `tests/diag_long_decode.py` | `logs/diag_long_decode.txt` | localises the error to the attention branch and **rules out operand quantisation**: an exact bf16-operand HF control matches fp32 to 2.4e-6 at every context, while TTNN diverges from both identically |
| `tests/diag_sdpa_decode.py` | `logs/diag_sdpa_decode.txt` | reproduces the identical degradation (0.9998 -> 0.7664 over 1024 -> 262143 keys) driving `paged_scaled_dot_product_attention_decode` **alone**, with no projections/RoPE/gate/o_proj/MoE, and sweeps grid x `k_chunk_size` x context |

Root cause: the op's **dynamic** `k_chunk_size` path (`get_dynamic_Sk_chunk_t`,
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:96`), which
picks `nearest_pow_of_2_up_to_8(seq_len_in_tiles)` from `cur_pos` and already carries an in-source
*"seeing PCC issues"* caveat. Second independent control: the *prefill* op computes the same
262144-key attention at 0.9999891 because the layer gives it an explicit `q=k=128` config.

Why the dynamic path is kept anyway: every explicit config that helps at long context is
structurally wrong at some shorter context (`8x8/128` -> 0.3721 at 1024, `8x1/128` -> 0.1621 at
257, `8x8/512` -> 0.2514 at 4096), the only universally-correct explicit configs run on 1-2 cores,
and `cur_pos` is a runtime *device* tensor so no per-call selection is possible without a host read
the fallback audit forbids. Full analysis and the handoff to `optimize` are in README section 3.8.

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
* A fourth: closing a device that still held a layer's ~2.2 GiB of weights *and* a full
  profiler buffer segfaulted inside `close_mesh_device`. `test_perf.py` now calls
  `ttnn.ReadDeviceProfiler` + `synchronize_device` before close and releases the layer
  (`FunctionalDecoder.release()`) when a measurement finishes.

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
```

## 8. Result summary

| requirement | evidence |
|---|---|
| `tt/functional_decoder.py` with documented prefill/decode contract for both layer kinds | module docstring + README §2; `test_config_matches_hf`, `test_layer_kinds_cover_the_whole_model` |
| decode runs fully under traced execution | `test_traced_decode_pcc` (PCC from replay), `test_traced_decode_matches_eager` (bit-identical) |
| every layer kind, real config shapes, paged prefill/decode, page table, current position | README §3.1/§3.4; 271 PCC rows in `pcc.jsonl` |
| longest feasible seq/context | 262143-token prefill and position-262143 decode for both kinds (`long_context.jsonl`) |
| non-aligned lengths around chunk/page/tile boundaries | 1/32/33/64/65/128/129/1024/1025/2048/2049/3000/4096 + 262143 per kind |
| `doc/context_contract.json` | derived from evidence by `tests/write_context_contract.py`, re-checked by `test_context_contract_file_is_consistent`; **no capability reduction** |
| real-weight test passing | `test_real_weights_prefill_and_decode[linear,full]`, `pcc_real_weights.jsonl` |
| PCC >= 0.995 prefill and decode | worst in main suite 0.9999450; worst at 262144 context 0.9985674 (§5.4) |
| warmed prefill + traced warmed decode perf with tt-perf-report tables + CSV/provenance | `tracy/<kind>_<mode>/`, `perf_summary.json`, README §5 |
| runtime fallback audit clean | `test_no_runtime_host_fallback` (static, 26 methods) + `test_no_host_ops_during_forward` (dynamic) |
| determinism / repeated input | `test_prefill_determinism`, `test_decode_determinism` (bit-identical over 3 repeats) |
| watcher-clean run | `watcher/` — 8 passed, 17262-line log, `watcher_hits.txt` empty (README §3.9) |
| README + work log with commands, PCC, perf, limitations, artifacts | this file + `README.md` |
