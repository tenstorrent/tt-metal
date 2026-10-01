# Qwen3.8-27B — prefill bring-up (BH Galaxy, SP=8 × TP=4)

TTNN prefill for **Qwen/Qwen3.8-27B** (`qwen3_5_text`) on one Blackhole Galaxy (8×4 mesh), validated
per layer against a full-depth CPU golden trace (upstream `transformers` Qwen3_5TextModel, fp32 compute).
Bring-up ends at recipe step P2 (real weights + chunked prefill). Serving, KV migration and perf tuning
are out of scope.

Binding spec: `prepare/spec.json` (tp 4, sp 8, `max_seq_len` 262144, `chunk_size` 5120, activations bf16,
KV cache bf8, weights bf8, `pcc_target` 0.99, `pcc_lower_bound` 0.87). Dimensions come from the checkpoint
`config.json`, vendored in `configs/Qwen3.8-27B/config.json` and asserted in `tests/torch/test_config_and_reference.py`.

## Architecture

| | |
|---|---|
| Layers | 64, hybrid: `linear_attention` (Gated DeltaNet) ×48, `full_attention` every 4th layer (3, 7, …, 63) ×16 |
| Hidden / MLP | 5120 / 17408 dense SwiGLU (SiLU), every layer |
| Full attention | GQA 24 q / 4 kv heads, head_dim 256, QK-norm (zero-centred), partial RoPE (64 of 256, θ=1e7; interleaved M-RoPE == 1D RoPE for text), **sigmoid output gate** (`q_proj` emits `[q | gate]`) |
| Gated DeltaNet | 16 key / 48 value heads × 128, causal depthwise conv (k=4) + SiLU, gates β=σ(b), g=−exp(A_log)·softplus(a+dt_bias), L2-normalised q/k, gated RMSNorm (plain weight) · SiLU(z) |
| Norms | zero-centred RMSNorm `x·(1+w)`, eps 1e-6 |
| Vocab | 248320 (untied LM head) |
| Carried state ("KV cache") | attention layers: K (post-RoPE) / V `[1, 4, T, 256]`; GDN layers: recurrent state `[1, 48, 128, 128]` fp32 + conv state (last 3 pre-conv projection columns) `[1, 10240, 3]` |

### Mesh layout (rows = SP, columns = TP)

* **Residual stream** `[1, 1, s_local, 5120]`: sequence SP-sharded (contiguous slice per row within a chunk,
  block-cyclic across chunks), replicated across TP; **kept fp32** (see PCC section). Every block ends in a TP all-reduce.
* **MLP**: gate/up column-parallel (4352 per column), down row-parallel, TP all-reduce (fp32 partial sums).
* **Attention**: fused per-column `[q(6 heads) | k(1) | v(1) | gate(6)]` projection, `nlp_create_qkv_heads`,
  fp32 QK-norm, partial RoPE (host cos/sin per chunk, `rotate_half` as a ±1 matmul), K/V written through
  `update_padded_kv_cache`; causal SDPA over SP with `ring_joint_scaled_dot_product_attention` (first chunk / one-shot,
  fp32 accumulation) or the composed cache read (later chunks); sigmoid gate; row-parallel `o_proj`.
* **Gated DeltaNet**: heads TP-sharded (column c owns key heads 4c…4c+3 and their value heads 12c…12c+11, so the GVA
  groups stay local). The recurrence spans the whole sequence but SP splits it over 8 rows, so after the local projections
  the chunk's q/k/v + gate streams are **all-gathered over SP and every row runs the full chunk's recurrence** (identical
  on all rows); `ttnn.mesh_partition` then keeps each row's own tokens for the gated norm and out_proj. Exact, redundant
  (×8) compute in the core; the cross-row state hand-off is follow-on perf work.
* **KV cache** (`tt/kv_cache.py`): canonical chunked-KV layout (per chip `[users·16, 1, cap/8, 256]`, NdShard
  `[1,1,32,256]` ROUND_ROBIN_1D over `get_num_dram_banks`, slot = user·16 + attention-layer ordinal, bf8). Only the 16
  attention layers get slots. GDN state lives beside it, per user and layer, TP-sharded, identical across SP rows.
  Capacity is `max_seq_len` rounded up to whole chunks (266240) — `update_padded_kv_cache` requires it (spec_gap, logged).
* **Embedding**: table sharded on hidden across TP (vocab replicated), TP all-gather. **LM head**: vocab column-parallel
  (only when logits are requested; prefill's product is the cache).
* **Runtime** (`tt/runtime.py`): `compile`, `make_chunk_input`, `prefill_chunk(input, caches, slot, actual_start, actual_end)`
  with loud asserts on out-of-contract ranges (slot range, chunk-aligned start, `start < end ≤ start+chunk`, capacity,
  `max_seq_len`, tile-aligned end, and — because GDN state is sequential — in-order chunks per user), plus `prefill_one_shot`.

### Where each part came from (exploration, recipe §2)

Gate: packages in `ADAPTER_PATHS` that ran on a BH galaxy 8×4 → **minimax_m3** (GQA, SP8×TP4, hidden 6144,
head_dim 128, chunk 5120). gpt_oss_d_p ran on (4,8) with sliding/sinks. No registered package has a GDN layer;
`models/demos/blackhole/qwen36` implements this model family but only on P150x4/x8 (TP only, no SP) → **math only**.

| Part | Source | Envelope of the source | Notes |
|---|---|---|---|
| Mesh / CCL manager | minimax_m3 `tt/ccl.py` `CCLManager` (imported) | 6144, 128, 5120, sp8×tp4 | collectives otherwise stateless `ttnn.all_gather/all_reduce` |
| KV cache (all 5 roles) | minimax_m3 `tt/attention/kv_cache.py` (layout copied) | same | K/V only (no `index_k`); `get_num_dram_banks` instead of hardcoded 8 |
| Ring SDPA call | minimax_m3 `tt/attention/dense_sp.py` | same | re-derived: k_chunk 256 for hd 256; fp32 acc on the live ring |
| MLP / attention structure | minimax_m3 `dense_mlp.py`, `attention/` | same | SiLU (not swigluoai); `[q|gate]` split, sigmoid gate from qwen36 math |
| Norm | minimax_m3 `rms_norm.py` structure | same | re-derived: fp32 gain, composed in fp32 (measured, below) |
| Runtime contract | minimax_m3 `tt_prefill_runtime.py` | same | + sequential-GDN assert |
| GDN math | qwen36 `tt/gdn/*`, upstream HF `modeling_qwen3_5.py` | 5120, 128, 2048, tp4 on P150x4 | core **composed fresh** from ttnn primitives (`tt/gdn_core.py`) |
| RoPE | write-fresh (HF half-split, partial) | — | keeps HF layout, so cached K compares with the golden without permutation |
| Weight loading | write-fresh `reference/checkpoint.py` | — | plain bf16 checkpoint: no dequant |
| Golden cache | `deepseek_v3_d_p/utils/transformer_helpers.py` `ReferenceCacheKey` (imported) | — | |

Fallbacks to torch CPU: **none** (every block runs in ttnn).

## Real weights and golden trace

* Weights: `PREFILL_HF_MODEL=/home/aleksajovanovic/models/qwen_3_8_27b_prefill` (symlinks to
  `/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2…`), bf16 safetensors. `visual.*` / `mtp.*` are not read.
* Trace: `PREFILL_TRACE_DIR=/mnt/models/Qwen/Qwen-3_8-27B-Cache/golden/synthetic_10240L_hf_ref` — 10240 tokens, 64 layers,
  upstream HF fp32 compute, stored bf16. The CPU reference in `reference/` reproduces it at PCC 0.999999 (first 128 tokens,
  final hidden and K/V of layers 3/23/43/63: `tests/torch/test_golden_hf_first_token.py`). This trace was removed from
  `/mnt/models` by 2026-10-01; the results below that use it are kept as recorded.
* Second trace: `PREFILL_TRACE_DIR=/mnt/models/agentic-prefill-goldens/Qwen/Qwen3.8-27B/isl10240` — first 10240 tokens of
  wikitext-2 (train), HF fp32 on CPU, stored bf16; token ids in the shared `token_cache` file. Results in
  [PCC vs the wikitext golden](#pcc-vs-the-wikitext-golden-agentic-prefill-goldens).
* Tilized weight cache: `~/.cache/tt_qwen_3_8_27b/tensor_cache_8x4/real_v2` (outside the package; `QWEN38_TT_CACHE` moves
  it, `=off` disables). Cold build ≈ 4 min (checkpoint already page-cached; first NFS read ≈ 7 min more), warm build ≈ 31 s.

## PCC status (full depth 64 layers, full width, real weights, 10240 tokens, fabric **linear** / FABRIC_1D)

Synthetic trace (`synthetic_10240L_hf_ref`). For the wikitext trace see the next section.

Measured by `tests/test_prefill_acceptance.py` (the verifier's interface). GDN rows grade recurrent state / conv state.

| Run | min per-layer PCC | layers below target (0.99) | e2e final-hidden PCC | wall time |
|---|---|---|---|---|
| one-shot (`PREFILL_CHUNKED=0`) | **0.9456** (layer 47 V) | 7: attention layers 39, 43, 47, 51, 55, 59, 63 | 0.9908 | 32 s |
| chunked 2×5120 (`PREFILL_CHUNKED=1`) | **0.9438** (layer 47 V) | 16: the same 7 + GDN conv states of 38, 41–46, 48–50 | 0.9903 | 34 s |

Chunked matches one-shot to within 0.002 on every layer (P2 goal). All values are above `pcc_lower_bound` 0.87.

**Why the listed layers sit between the bound and the target.** This model amplifies small errors with depth. A CPU
forward in *pure bf16 with the exact bf16 checkpoint weights* drifts from the fp32 trace to final-hidden PCC ≈ 0.92 at
layer 50 (2048-token diagnostic) — the trace README names bf16 compounding as the dominant error term. Every device
layer fed the *reference's own input* is ≥ 0.99996 (`scripts/diag_layer_drift.py`, `QWEN38_DIAG_ISOLATED=1`), so no
single layer is wrong; the residual drift enters the late attention layers' **V/K** (un-normalised projections of the
drifted hidden) and, in chunked mode, the GDN **conv states** (the last 3 tokens only, computed in chunk 2, whose attention
reads the prefix from the spec's **bf8** KV cache — bf8 K/V alone is 2.1% relative attention error, measured on CPU).
Weights in bf16 instead of the spec's bf8 change the drift by < 0.003 (measured, diagnostic spec override), so the
spec's weight dtype is not the cause.

**Correctness knobs tried, in order, with measured effect** (one-shot min per-layer PCC, 10k):

| Knob | min PCC | Kept |
|---|---|---|
| Baseline (bf16 residual, bf16 norm gains, ring SDPA fp32-acc off, fused GDN op) | 0.824 (fails bound) | — |
| Residual stream + TP partial sums fp32 (all matmul/SDPA/GDN inputs stay bf16) | 0.853 | yes (`QWEN38_FP32_RESIDUAL`, default on) |
| Norm gains fp32, norm composed in fp32 (bf16 `1+w` lost 87% of `w`, +0.1% scale bias per norm) | — (norm now matches CPU; no drift change) | yes |
| `fp32_dest_acc_en=True` on the live-KV ring SDPA (4.1% → 1.2% rel. error; cache-read ring refuses it) | 0.895 | yes |
| Composed cache read for chunks ≥ 2 (masked plain SDPA, fp32 acc) instead of the cache-read ring (5.4% error) | chunked 0.849 → 0.891 | yes (`QWEN38_CACHE_ATTN`, default `masked`) |
| GDN core composed in fp32 from ttnn primitives instead of the fused `chunk_gated_delta_rule` (2.6%/6.8% → 0.3%/0.4% on layer 28) | **0.946** | yes (`QWEN38_GDN_CORE`, default `composed`) |
| Weights bf16 (diagnostic only — violates the spec) | no change (< 0.003) | no |
| Other: HiFi4 + fp32 accumulation on every matmul; conv, gates and GDN state in fp32 | — | yes (from the start) |

Remaining levers (follow-on, not tried): bf16 activations are the spec's dtype — computing attention/MLP intermediates in
fp32 would be the next step if the spec allowed it; a bf16 KV cache would lift the chunked conv states.

### Per-layer table (one-shot | chunked; ¹ = below `pcc_target`)

| layer | type | graded tensors | one-shot | chunked |
|---|---|---|---|---|
| 0 | linear_attention | recurrent / conv | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| 1 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 2 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 3 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 4 | linear_attention | recurrent / conv | 0.9997 / 0.9999 | 0.9997 / 0.9999 |
| 5 | linear_attention | recurrent / conv | 0.9997 / 0.9999 | 0.9997 / 0.9999 |
| 6 | linear_attention | recurrent / conv | 0.9996 / 0.9999 | 0.9996 / 0.9999 |
| 7 | full_attention | K / V | 0.9999 / 0.9998 | 0.9999 / 0.9998 |
| 8 | linear_attention | recurrent / conv | 0.9996 / 0.9999 | 0.9996 / 0.9999 |
| 9 | linear_attention | recurrent / conv | 0.9995 / 0.9999 | 0.9995 / 0.9999 |
| 10 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 11 | full_attention | K / V | 0.9999 / 0.9998 | 0.9999 / 0.9998 |
| 12 | linear_attention | recurrent / conv | 0.9997 / 0.9999 | 0.9997 / 0.9999 |
| 13 | linear_attention | recurrent / conv | 0.9997 / 0.9999 | 0.9997 / 0.9999 |
| 14 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 15 | full_attention | K / V | 0.9999 / 0.9998 | 0.9999 / 0.9998 |
| 16 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9997 / 0.9999 |
| 17 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9998 / 0.9999 |
| 18 | linear_attention | recurrent / conv | 0.9997 / 0.9999 | 0.9997 / 0.9999 |
| 19 | full_attention | K / V | 0.9999 / 0.9998 | 0.9999 / 0.9998 |
| 20 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9998 / 0.9999 |
| 21 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 22 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9998 / 0.9999 |
| 23 | full_attention | K / V | 0.9999 / 0.9998 | 0.9999 / 0.9998 |
| 24 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 25 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 26 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9998 / 0.9999 |
| 27 | full_attention | K / V | 0.9998 / 0.9997 | 0.9998 / 0.9997 |
| 28 | linear_attention | recurrent / conv | 0.9998 / 0.9998 | 0.9998 / 0.9997 |
| 29 | linear_attention | recurrent / conv | 0.9998 / 0.9998 | 0.9997 / 0.9996 |
| 30 | linear_attention | recurrent / conv | 0.9998 / 0.9998 | 0.9997 / 0.9994 |
| 31 | full_attention | K / V | 0.9992 / 0.9987 | 0.9991 / 0.9985 |
| 32 | linear_attention | recurrent / conv | 0.9996 / 0.9991 | 0.9995 / 0.9983 |
| 33 | linear_attention | recurrent / conv | 0.9994 / 0.9983 | 0.9992 / 0.9972 |
| 34 | linear_attention | recurrent / conv | 0.9995 / 0.9976 | 0.9994 / 0.9968 |
| 35 | full_attention | K / V | 0.9968 / 0.9951 | 0.9963 / 0.9943 |
| 36 | linear_attention | recurrent / conv | 0.9994 / 0.9979 | 0.9994 / 0.9946 |
| 37 | linear_attention | recurrent / conv | 0.9988 / 0.9984 | 0.9976 / 0.9947 |
| 38 | linear_attention | recurrent / conv | 0.9979 / 0.9969 | 0.9956 / 0.9889 ¹ |
| 39 | full_attention | K / V | 0.9914 / 0.9823 ¹ | 0.9907 / 0.9809 ¹ |
| 40 | linear_attention | recurrent / conv | 0.9997 / 0.9970 | 0.9996 / 0.9905 |
| 41 | linear_attention | recurrent / conv | 0.9988 / 0.9959 | 0.9985 / 0.9866 ¹ |
| 42 | linear_attention | recurrent / conv | 0.9983 / 0.9947 | 0.9973 / 0.9843 ¹ |
| 43 | full_attention | K / V | 0.9806 / 0.9608 ¹ | 0.9795 / 0.9609 ¹ |
| 44 | linear_attention | recurrent / conv | 0.9978 / 0.9931 | 0.9976 / 0.9818 ¹ |
| 45 | linear_attention | recurrent / conv | 0.9983 / 0.9909 | 0.9982 / 0.9758 ¹ |
| 46 | linear_attention | recurrent / conv | 0.9984 / 0.9922 | 0.9969 / 0.9797 ¹ |
| 47 | full_attention | K / V | 0.9750 / 0.9456 ¹ | 0.9735 / 0.9438 ¹ |
| 48 | linear_attention | recurrent / conv | 0.9971 / 0.9937 | 0.9970 / 0.9819 ¹ |
| 49 | linear_attention | recurrent / conv | 0.9969 / 0.9942 | 0.9961 / 0.9765 ¹ |
| 50 | linear_attention | recurrent / conv | 0.9980 / 0.9964 | 0.9974 / 0.9859 ¹ |
| 51 | full_attention | K / V | 0.9675 / 0.9533 ¹ | 0.9661 / 0.9516 ¹ |
| 52 | linear_attention | recurrent / conv | 0.9984 / 0.9992 | 0.9982 / 0.9955 |
| 53 | linear_attention | recurrent / conv | 0.9978 / 0.9993 | 0.9975 / 0.9962 |
| 54 | linear_attention | recurrent / conv | 0.9960 / 0.9992 | 0.9955 / 0.9956 |
| 55 | full_attention | K / V | 0.9662 / 0.9595 ¹ | 0.9645 / 0.9566 ¹ |
| 56 | linear_attention | recurrent / conv | 0.9940 / 0.9997 | 0.9939 / 0.9980 |
| 57 | linear_attention | recurrent / conv | 0.9967 / 0.9997 | 0.9957 / 0.9984 |
| 58 | linear_attention | recurrent / conv | 0.9978 / 0.9997 | 0.9974 / 0.9981 |
| 59 | full_attention | K / V | 0.9710 / 0.9791 ¹ | 0.9696 / 0.9777 ¹ |
| 60 | linear_attention | recurrent / conv | 0.9936 / 0.9999 | 0.9941 / 0.9993 |
| 61 | linear_attention | recurrent / conv | 0.9972 / 0.9998 | 0.9972 / 0.9990 |
| 62 | linear_attention | recurrent / conv | 0.9954 / 0.9999 | 0.9960 / 0.9994 |
| 63 | full_attention | K / V | 0.9850 / 0.9969 ¹ | 0.9844 / 0.9967 ¹ |

## PCC vs the wikitext golden (agentic-prefill-goldens)

Same build, weights and test as above, graded against a second, independent trace:
`PREFILL_TRACE_DIR=/mnt/models/agentic-prefill-goldens/Qwen/Qwen3.8-27B/isl10240`, the first 10240 tokens of
wikitext-2-raw-v1 (train) under the Qwen3.8 tokenizer, HF forward on CPU in fp32 (sdpa), stored bf16. Token ids come from
the shared `token_cache` the trace's `metadata.json` points to (`n_tokens` = 10240); the acceptance test reads both formats.
Measured 2026-10-01 on the 8×4 Galaxy, fabric linear.

| Run | min per-layer PCC | layers below target (0.99) | e2e final-hidden PCC | wall time |
|---|---|---|---|---|
| one-shot (`PREFILL_CHUNKED=0`) | **0.9987** (layer 43 V) | 0 | 0.9997 | 30 s |
| chunked 2×5120 (`PREFILL_CHUNKED=1`) | **0.9987** (layer 43 V) | 0 | 0.9997 | 30 s |

Every layer is above `pcc_target` in both modes; chunked matches one-shot to within 0.0003 on every layer.

**Reading this against the synthetic trace.** The same code scores 0.9456 / 0.9438 (min) on the synthetic prompt, with
the deficit concentrated in the late attention layers' K/V and the chunked GDN conv states (see above). On real text none of
that drift appears: the worst layer is 0.9987 and the final hidden state 0.9997. The depth-amplification explanation above
was measured on the synthetic prompt only; whether that prompt is unusually sensitive to bf16 compounding, or real text is
unusually benign, is not established by these two traces.

### Per-layer table, wikitext trace (one-shot | chunked)

| layer | type | graded tensors | one-shot | chunked |
|---|---|---|---|---|
| 0 | linear_attention | recurrent / conv | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| 1 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 2 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 3 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 4 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 5 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 6 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 7 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 8 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 9 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 10 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 11 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 12 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 13 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 14 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 15 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 16 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 17 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 18 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 19 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 20 | linear_attention | recurrent / conv | 0.9999 / 1.0000 | 0.9999 / 1.0000 |
| 21 | linear_attention | recurrent / conv | 0.9999 / 1.0000 | 0.9999 / 1.0000 |
| 22 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 23 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 24 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 25 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 26 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 27 | full_attention | K / V | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 28 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 29 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 30 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 31 | full_attention | K / V | 0.9999 / 0.9998 | 0.9999 / 0.9998 |
| 32 | linear_attention | recurrent / conv | 0.9998 / 0.9998 | 0.9998 / 0.9999 |
| 33 | linear_attention | recurrent / conv | 0.9998 / 0.9998 | 0.9998 / 0.9999 |
| 34 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 35 | full_attention | K / V | 0.9998 / 0.9998 | 0.9998 / 0.9998 |
| 36 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 37 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 38 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 39 | full_attention | K / V | 0.9998 / 0.9997 | 0.9998 / 0.9997 |
| 40 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 41 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9999 / 0.9999 |
| 42 | linear_attention | recurrent / conv | 0.9995 / 0.9999 | 0.9998 / 0.9999 |
| 43 | full_attention | K / V | 0.9996 / 0.9987 | 0.9996 / 0.9987 |
| 44 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 45 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 46 | linear_attention | recurrent / conv | 0.9999 / 0.9999 | 0.9998 / 0.9999 |
| 47 | full_attention | K / V | 0.9996 / 0.9989 | 0.9996 / 0.9988 |
| 48 | linear_attention | recurrent / conv | 0.9996 / 0.9999 | 0.9995 / 0.9999 |
| 49 | linear_attention | recurrent / conv | 0.9997 / 0.9999 | 0.9997 / 0.9999 |
| 50 | linear_attention | recurrent / conv | 0.9996 / 0.9999 | 0.9997 / 0.9999 |
| 51 | full_attention | K / V | 0.9996 / 0.9992 | 0.9996 / 0.9992 |
| 52 | linear_attention | recurrent / conv | 0.9999 / 1.0000 | 0.9999 / 0.9999 |
| 53 | linear_attention | recurrent / conv | 0.9999 / 1.0000 | 0.9999 / 0.9999 |
| 54 | linear_attention | recurrent / conv | 0.9998 / 0.9999 | 0.9998 / 0.9999 |
| 55 | full_attention | K / V | 0.9995 / 0.9993 | 0.9995 / 0.9994 |
| 56 | linear_attention | recurrent / conv | 0.9998 / 1.0000 | 0.9998 / 0.9999 |
| 57 | linear_attention | recurrent / conv | 0.9999 / 1.0000 | 0.9998 / 1.0000 |
| 58 | linear_attention | recurrent / conv | 0.9999 / 1.0000 | 0.9998 / 0.9999 |
| 59 | full_attention | K / V | 0.9995 / 0.9994 | 0.9995 / 0.9996 |
| 60 | linear_attention | recurrent / conv | 0.9996 / 1.0000 | 0.9994 / 0.9999 |
| 61 | linear_attention | recurrent / conv | 0.9998 / 1.0000 | 0.9997 / 0.9999 |
| 62 | linear_attention | recurrent / conv | 0.9996 / 1.0000 | 0.9995 / 1.0000 |
| 63 | full_attention | K / V | 0.9996 / 0.9998 | 0.9996 / 0.9998 |

## Component tests (random weights, full width, 8×4 mesh, fabric linear)

All assert `pcc_lower_bound` (0.87) from `PREFILL_SPEC`; all are ≥ `pcc_target`.

| Test (pattern) | What | PCC |
|---|---|---|
| `unit/test_mesh_smoke.py` | 8×4 opens, all-gather + all-reduce on both axes | pass (linear **and** torus) |
| `unit/test_blocks_vs_ref.py` | RMSNorm (input/final), QK-norm, SwiGLU, partial RoPE, dense MLP 2k/10k | 1.0000, 1.0000, 0.99999, 1.0000, 0.99991 |
| `unit/test_ring_joint_sp_vs_ref.py` | ring SDPA live 2k/10k (hd 256, 24q/4kv); cache-read 2×2048 / 2×5120 | 0.99997; 0.99976 / 0.99975 |
| `unit/test_attention_vs_ref.py` | gated attention 2k/10k; KV-cache write (K/V read back); 2-chunk (masked / ring cache read); GQA cache write+read 3×5120 and one-shot 10240, 2 users × 2 layers | 0.99983; 0.99994 / 0.99995; 0.99980 / 0.99959; 0.99997 |
| `unit/test_gdn_vs_ref.py` | GDN block 2k/10k + recurrent/conv state, 2-chunk carry; composed and fused core | out 0.99987, rec 0.99991, conv 0.99997 (composed) |
| `unit/test_decoder_layer_vs_ref.py` | decoder layer 0 (GDN), 3 (attention), 5120 tokens | 0.99976, 0.99990 |
| `unit/test_model_vs_ref.py` | embedding, LM head; **8-layer (reduced depth)** model one-shot and 2×2560 chunked, every layer's state | 1.0000, 0.99997; hidden 0.99801, worst state 0.9982 |
| `torch/*` (host) | config vs config.json; reference vs upstream HF (rope, delta rule, whole model incl. cache); vs inline golden (8 and 64 layers, reduced width); chunked == one-shot; golden cache round trip; reference vs real checkpoint (upstream trace, PCC 0.999999); loader shapes; runtime chunk contract | all pass |

Rows not applicable to this model (logged as `skip`): fused MoE gate, EP MoE (dense model), mxfp4 loader (bf16
checkpoint), vocab-sharded-on-SP embedding mode (not implemented; emb-on-TP only).

**Reduced runs** (diagnostics only, never the result): the 8-layer model test above; `scripts/diag_*.py` on the first
2048 trace tokens (per-layer drift, isolated-layer error, per-block error vs CPU bf16, GDN core accuracy per column).

## Topology

Acceptance and component PCCs were measured on the plain galaxy mesh descriptor with `FABRIC_1D` + `Topology.Linear`
(`QWEN38_FABRIC=linear`, default). The mesh smoke test also passes on the torus (`QWEN38_FABRIC=torus`,
`single_bh_galaxy_torus_xy_graph_descriptor`, `FABRIC_1D_RING`); the model was not measured on the torus.

## Reproduce

```bash
cd $TT_METAL_HOME   # the tt-metal checkout, branch prefill/qwen_3_8_27b
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD LD_LIBRARY_PATH=$PWD/build/lib
export PATH=$PWD/python_env/bin:$PATH
bash <method>/check_runtime_env.sh python_env/bin/python
export PREFILL_SPEC=<prepare>/spec.json
export PREFILL_HF_MODEL=/home/aleksajovanovic/models/qwen_3_8_27b_prefill HF_MODEL=$PREFILL_HF_MODEL
export PREFILL_TRACE_DIR=/mnt/models/agentic-prefill-goldens/Qwen/Qwen3.8-27B/isl10240  # or a trace with inline token_ids

# host tests (no device)
python -m pytest models/demos/qwen_3_8_27b/tests/torch -q
# device component suite (8x4 mesh), ~6 min
scripts/run_safe_pytest.sh --run-all models/demos/qwen_3_8_27b/tests/unit
# acceptance, full depth/width, real weights (prints "PCC REPORT {...}" before the asserts; writes the JSON report on pass)
PREFILL_CHUNKED=0 PREFILL_ACCEPTANCE_OUT=/tmp/oneshot.json scripts/run_safe_pytest.sh models/demos/qwen_3_8_27b/tests/test_prefill_acceptance.py -s
PREFILL_CHUNKED=1 PREFILL_ACCEPTANCE_OUT=/tmp/chunked.json scripts/run_safe_pytest.sh models/demos/qwen_3_8_27b/tests/test_prefill_acceptance.py -s
# diagnostics (reduced, 2048 tokens)
PYTHONPATH=$PWD python models/demos/qwen_3_8_27b/scripts/diag_layer_drift.py 2048
```

### Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `QWEN38_FABRIC` | `linear` | `linear` (FABRIC_1D, plain mesh descriptor) or `torus` (FABRIC_1D_RING, torus descriptor); read before cluster init |
| `QWEN38_FP32_RESIDUAL` | `1` | `0` = bf16 residual stream / partial sums |
| `QWEN38_GDN_CORE` | `composed` | `fused` = `ttnn.transformer.chunk_gated_delta_rule` (faster, ~10× less accurate on real inputs) |
| `QWEN38_GDN_NEWTON` | `3` | Newton refinement steps of the chunk inverse in the composed core |
| `QWEN38_CACHE_ATTN` | `masked` | `ring` = the ring cache-read op for chunks ≥ 2 (bf16 accumulation only) |
| `QWEN38_TT_CACHE` | `~/.cache/tt_qwen_3_8_27b` | tilized weight cache root; `off` disables |
| `QWEN38_REF_CACHE` | `/tmp/qwen_3_8_27b_transformer_ref_cache` | `ReferenceCacheKey` golden cache dir; `QWEN38_REQUIRE_REF_CACHE=1` fails on a miss |

## Known gaps

* On the synthetic trace, 7 attention layers (one-shot) / 16 layers (chunked) are between `pcc_lower_bound` and
  `pcc_target` — explained above; all knobs within the spec's dtypes were tried. On the wikitext trace every layer is
  ≥ 0.9987.
* Throughput is not tuned: the composed GDN core runs 320 sequential chunk steps per GDN layer at 10k tokens and every SP
  row recomputes the whole chunk's recurrence (×8 redundancy) — ~32 s per 10k one-shot (the fused core: ~12 s).
  Cross-row GDN state hand-off, trace capture and the fused core's accuracy are follow-ons.
* The masked cache read builds a host mask `[8, 1, chunk/8, prefix+chunk]` per chunk and all-gathers the prefix K/V;
  memory grows with the prefix (≈ 335 MB/device at 262k) — fine for bring-up, a scaling limit for long contexts.
* One-shot prefill writes the KV table with the one-shot length as block-cyclic period; continuing a one-shot sequence
  with chunked prefill is not supported (the runtime ends the sequence after `prefill_one_shot` / a padded chunk).
* A padded (short) final chunk is supported for GDN (β, g zeroed past `actual_end`, conv state taken at `actual_end`) but
  was only exercised by the host contract test, not on device with real weights.
* The acceptance run and component tests were measured on the linear fabric only.
* Not done (out of scope): serving adapter / `ADAPTER_PATHS` registration, KV migration, decode, logits/top-1 vs HF.
* `bringup_digest.py --lint` is not present in this checkout; the log was checked against the §7 schema by hand
  (two over-length `fix` fields were shortened in place right after writing).
