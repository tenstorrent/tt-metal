# DSpark speculative decoding for V4.1-Flash: design

Status: implemented (`tt/decode/dspark.py`, `tt/spec.py`, the spec section of `tt/model.py`); device
validation in progress. It covers bringing the checkpoint's DSpark drafter (`mtp.0..2`) into
`deepseek_v41_flash` decode as lossless speculative decoding. Sections 3.2, 3.4 and 3.5 describe what
was built; where they differ from the original plan, the code wins.

Prior art: `ssinghal/dsv4p1` (`~/smanoj/metal_2`, `models/demos/blackhole/deepseek_v41_flash`,
`SPEC_NOTES.md`, `SPEC_ADAPT_NOTES.md`). That implementation targets a different deployment: one
4x8 mesh, TP32, several users per round, a paged pool. Its *algorithms*, test methodology and
measurements carry over here. Its *code* mostly does not, because this package decodes one user on
an 8-stage x TP4 pipeline.

## 1. What DSpark is

From the checkpoint's own `inference/model.py` (`DSparkBlock`, `DSparkAttention`,
`DSparkMarkovHead`, `DSparkConfidenceHead`, `Transformer.forward_spec`) and `config.json`
(`dspark_block_size=5`, `dspark_noise_token_id=128799`, `dspark_target_layer_ids=[37,38,39]`,
`dspark_markov_rank=256`, `dspark_n_routed_experts=128`, `dspark_num_experts_per_tok=3`):

* The three `mtp.N` modules are three stages of **one block drafter**, always run together. They
  are not three alternative one-token MTP heads. One draft pass produces 5 draft tokens.
* `main_hidden` [15360] is the concatenation, over target layers 37/38/39, of the **mean of the
  4 mHC streams at the layer's input**.
* Stage 0: `main_x = main_norm(main_proj(main_hidden))`. Each stage writes
  `main_kv = kv_norm(wkv_s(main_x))`, with RoPE at the position, into its own 128-position window
  ring, for **every verified position**.
* The block input is `[t, noise x4]`, embedded with the backbone embedding (tied weights) and
  repeated over 4 streams, with `pre_mix` one-hot.
* Each stage is a normal V4.1 block:
  * mHC on attention and FFN.
  * Window-only MQA attention (low-rank q, 1 KV head, head_dim 512, sink, grouped `wo_a`/`wo_b`)
    over `[ring | 5 block keys]`, **non-causal inside the block**.
  * A MoE with **128 routed experts, top-3**, plus a shared expert.
  * No compressor, no indexer, no Engram.
* Head: the backbone LM head (tied weights) with stage 2's own `norm`. Then 5 sequential Markov
  steps: `logits[i] += markov_embed(tok_i) @ markov_head^T`, and `tok_{i+1} = argmax`.
* Confidence head: `proj([pre-norm x | markov_embed])`, which gives P(draft j accepted | drafts
  before j accepted). It is only needed for adaptive k.
* The accept rule is ours to define. We use greedy: draft d_{j+1} is accepted iff it equals the
  argmax of verify row j. The output is then the plain greedy stream, exactly (up to bf16 near-ties).

Tunable: k = how many of the 5 drafts to verify (1..5). The drafter's cost and weights are fixed.

## 2. Why the metal_2 design does not transplant

| | metal_2 | this package |
|---|---|---|
| Deployment | one 4x8 mesh, TP32, U users per mesh row | 8 pipeline stages x TP4 (1x4 submeshes), **batch 1** |
| Verify | k+1 rows per user in **one** step (virtual users, paged pool, new mHC / `paged_kv_step` work) | `fused_experts` takes exactly one token row (`assert t == 1` in `DeepSeekV4PreloadedExperts.forward`), so a wide verify step needs new kernels |
| Idle hardware | none | in batch-1 decode **7 of 8 stages are idle** at any moment |
| Existing code | full DSpark | V4's `tt/decode/dspark.py` is a standalone stand-in (dense SwiGLU, MHA q/k/v/o), not the checkpoint's mHC / MoE / MQA. V4's tap link (`_dspark_tap_ids` 40-42, `mtp_submesh`) needs an idle row, and V4.1 uses all 32 chips |

## 3. Design

### 3.1 Verify = pipelined single-token steps

The verify block `[t, d1..dk]` at positions `p..p+k` runs as **k+1 ordinary decode steps sent back
to back**. `DeepSeekV4Model.decode_prompt_traced` already does this: it queues every step's traces,
a writer thread pushes the packets, the caller reads the outputs, and the pipeline holds one step
per stage.

* **No new verify kernels.** Every existing trace and variant (`_variant_key(pos)`: causal, r=2
  phase, r=2 select) is reused per step.
* **Causality is free**: rows run in position order.
* **Cost**: about `(8 + k) * s`, where `s` = per-stage time (~3.7 ms at the current 34 tok/s/user).
  A plain token costs `8 * s`. So k=3 costs ~1.4x a plain token, not 4x.

Estimated throughput (to be replaced by Phase 0 measurements). Assumptions: drafter ~8 ms,
host ~1.5 ms per round, acceptance from metal_2's CPU reference (GSM8K: 0.905 / 0.796 / 0.693 / ...).

| k | round ms | tokens/round (GSM8K) | tok/s/user | tokens/round (long doc) | tok/s/user |
|---|---|---|---|---|---|
| 0 (plain) | 29.4 | 1 | 34 | 1 | 34 |
| 3 | ~50 | ~3.3 | ~66 (1.9x) | ~2.5 | ~50 |
| 5 | ~58 | ~4.2 | ~73 (2.1x) | ~2.6 | ~45 |

Speculative decoding fits this deployment better than metal_2's batched one: there, extra verify
rows cost real compute; here, they fill stages that would otherwise sit idle.

### 3.2 Rollback: what a rejected row leaves behind

A rejected position q (`p+m+1 <= q <= p+k`) has already run through every layer. State it touched:

| State | Keyed by | Needs rollback? |
|---|---|---|
| Ratio-2 pooled `comp_kv` entry, index key | position | No. Re-running q overwrites them, and no accepted row reads entry `q // r` before q is re-run (rows run in order) |
| **Ratio-2 `win_kv` / `win_gate`** (slot `pos % 2`) | slot | **Yes**. With the last verified position f even, a rejected row at f+2 overwrites slot 0, which still holds row f; the pool at f+1 then reads the rejected row |
| Ratio-1 `comp_kv` / index key (entry = pos) | position | No |
| `sel_tile` | recomputed each step (`fill_cache(sel_tile, kv, 0)` + select) | No |
| Cross-stage `new_kv_row` / `new_key_row` | per step | No |
| Engram host hash (`NgramHashState.cache[:, start_pos:...]`) | position | No (overwritten when q is re-fed) |
| **Every layer's sliding ring `scache.kv`** `[1,1,128,512]`, slot `pos % 128` | slot | **Yes**. Writing q overwrites position q-128, which the next round's rows still attend once `p + k >= 128` |
| Drafter rings (3.4) | slot `pos % 128` | No: only accepted rows are written (`write_main` after the verify, rejected rows masked out) |

**Fix: shadows, window histories and a per-round fixup trace on every stage** (`prepare_spec_decode`,
`_spec_fixup`, `spec_fixup`).

* Each layer's ring gets a persistent DRAM shadow (5 layers x 128 x 512 bf16 per stage, 640 KB).
* Each ratio-2 kv source also writes every step's window rows into `win_hist_kv` / `win_hist_gate`
  `[1, 1, 32, 512]` at slot `pos % 32` (`hist_slot`, in `_latent`).
* Before each verify block, every stage replays a small fixup trace:
  1. `ring = ring * keep + shadow * mask`: restore the previous round's rejected slots.
  2. `win = win * slot_keep + slot * (pick @ hist)`: put the last verified position's window rows back
     in their slot.
  3. `copy(ring, shadow)`: back up the state for this block.
  Every product is with 0 or 1, so the restore is exact.
* `mask` / `keep` `[128, 1]`, `pick` `[1, 32]` and `slot` / `slot_keep` `[2, 1]` are persistent device
  tensors per stage, refreshed with `copy_host_to_device_tensor`.
* Phase 5 folds the restore range into the step packet, which removes the extra trace.

The fixup is enqueued on each submesh's cq before that block's step traces, so per-submesh FIFO
order guarantees it runs first.

### 3.3 Taps (layers 37-39, stage 7)

With the default pipeline group size 1, stage 7 owns layers 35-39, so all three tap layers are
local to it.

* Override V4's `_dspark_tap_ids` with `config.dspark_target_layer_ids` and add a **local tap**
  mode that needs no idle row and no D2D socket.
* In the stage-7 trace, at the input of layers 37 / 38 / 39, write the 4-stream mean (fp32
  accumulate, as in `_collapse_streams` with weights 1/4) into a persistent stash `[RING_D, 15360]`
  bf16 at slot `pos % RING_D` (index from the packet's position).
* V4.1 prefills by replaying decode, so the stash fills during prefill as well. Seeding the drafter
  needs no prompt replay, which is metal_2's `prefill_taps` lesson.
* If V4.1 gets a dedicated prefill model later (as V4 has `build_prefill`), that prefill must write
  the stash for the last 128 prompt positions.

### 3.4 Drafter (stage 7 submesh, TP4)

`tt/decode/dspark.py`: `DeepSeekV41DSparkStage` x 3 and `DeepSeekV41DSpark`. The 5 block rows run
together as `[1, 1, 5, D]` TILE tensors (`[1, 5, hc, D]` streams), the layout of the V4 prefill blocks:

| Piece | Built from | Notes |
|---|---|---|
| mHC attn / ffn | `DeepSeekV4PrefillHyperConnection` + V4.1's carried `pre_mix` | multi-token `fused_hyperconnection` |
| Attention | plain `ttnn.matmul` / `softmax` | q heads column-sharded (16 per rank). The 5 rows' queries share the keys and the mask, so they fold into one `[1, 1, 80, 512]` query against `[ring 128 | block 5 | sink row | padding]` (160 keys); the sink is a per-head bias on the zero key row. Inverse RoPE, batched local-group `wo_a`, row-parallel `wo_b`, all-reduce |
| MoE | `fused_experts` once per row (ROW_MAJOR, the decode path), 128 experts, top-3, bfp4 experts (`dspark_moe_config`) | sqrt-softplus scores, bias ranking; shared expert with the SwiGLU clamp. A multi-row TILE call pads the rows to 32 and its CBs do not fit. bfp8 experts are more faithful but their double-buffered full-K weight slice (~700 KB/core) does not fit below stage 7's prefetch rings |

Stage 7 L1 is the binding constraint (the decode prefetch rings sit above ~719 KB), so every drafter op uses
its single-row program: `fused_hyperconnection` per token (8-core single-user program), `fused_experts`
per row, all-reduces on ROW_MAJOR rows (a 32-row tile makes the direct reduce-scatter stage 640 KB/core),
and the stash split into 1024-wide column pieces (`paged_update_cache` double-buffers whole rows on one core).
| `write_main` | one-hot gather from the stash, `main_proj`, `main_norm`, per stage `wkv` / `kv_norm` / RoPE | `ring = ring * keep + scatter @ kv`: up to `main_rows` (8) rows per call, rejected rows not written |
| Head | the decode `lm_head` (tied) + `mtp.2.norm` | 5 rows |
| Markov | `embed` (ROW_MAJOR, `ttnn.embedding`), `head` as `Linear` | 5 sequential steps, `ttnn.argmax` on device |
| Confidence | `proj` split into its hidden / Markov halves, fp32 | returned as logits |

Host inputs (one-hots, RoPE rows, block embeddings, the frontier bias row) are persistent tensors
refreshed with `copy_host_to_device_tensor`; `write_main` and `draft` are two traces on the output
stage. `spec_draft` reads the 5 ids and confidences back with `to_torch`.

DRAM on stage 7 at TP4 is ~2.3 GB/chip (routed experts bfp4 ~1.9 GB; attention, shared expert,
`main_proj`, Markov tables), beside ~9.6 GB/chip of main-model experts.

Standalone vs the reference (random taps, bfp8 experts, `test_dspark.py`): stage 0 attention PCC 0.9999,
the MoE on the device's own input 0.9999 (median); end to end the logits PCC averages 0.987 (first lap) /
0.946 (wrapped), with 7/8 and 4/8 first proposals matching. The drift is routing near-ties (bf16 scores)
flipping a stage's top-3, which random taps make common. bfp4 experts: 0.91 mean, 5/8.

Full model (`test_dspark_spec.py`, k = 4, 64 tokens, chat prompt, bfp4 drafter): the oracle and DSpark
streams both equal plain greedy token for token.

| Run | tokens / round | tok/s/user |
|---|---|---|
| Plain greedy | 1 | 33.1 |
| Oracle drafts (1/3 corrupted) | 3.10 | 49.0 |
| DSpark | 2.50 (rounds with 0..4 accepted: 8, 8, 3, 3, 4) | 34.0 |

Measured round split with DSpark (ms; `SpecStats.phases` plus isolated timings): verify 54 (1 token 30.7,
5 tokens 55.2: each extra token costs ~6.1 ms, the pipeline's slowest stage), draft 17 (trace 13.0, `set_block`
uploads 4.3, readback 0.2), write 5 (trace 1.0, uploads 0.5, the rest waiting for the verify replays to
dispatch), fixup 2.5 (4.8 isolated, uploads to all 8 stages). At 2.5 tokens/round that only breaks even. Phase 5 (in-trace argmax, fixup folded into the packet)
and cheaper drafter ops are what turn acceptance into speed.

### 3.5 One round (host, `tt/spec.py`)

```
state: p (position of t), X = [t, d1..dk], rejected (positions of the previous round's rejected rows)
1. spec_fixup(rejected, frontier=p-1)            # every stage: restore, then snapshot
2. decode_prompt_traced(X, p)                    # k+1 pipelined steps; a_j = host argmax of row j
3. m = leading j with X[j+1] == a_j;  emit a_0..a_m
4. spec_write_main(p..p+k, written=[j <= m])     # drafter rings take the verified rows
5. spec_draft(a_m, frontier=p+m) -> d'1..d'5 (+conf)
6. rejected <- p+m+1..p+k;  p <- p+m+1;  X <- [a_m, d'1..d'k]
```

Prefill feeds the prompt with `decode_prompt_traced`, then `spec_seed` writes the last 128 prompt
positions into the drafter rings from the stash (256 slots deep, so they are still there).

Phase 5 moves steps 3-4 on device (in-trace argmax on stage 7) and, later, the token loop
(stage 7 -> stage 0 D2D) off the host.

## 4. Files

Follows `PORTING.md`: V4.1 files hold only what changed, V4 hooks are class attributes with V4
defaults, no thin wrappers.

| File | Change |
|---|---|
| `tt/config.py` | `dspark_weights(loader, stage)` (`mtp.N.*` thunks, fp8 dequant), `dspark_expert_provider` |
| `tt/decode/dspark.py` (new) | `DeepSeekV41DSparkStage`, `DeepSeekV41DSpark` (`write_main`, `draft`), `dspark_moe_config`, `host_to_device` |
| `tt/decode/attention.py` | `win_hist_kv` / `win_hist_gate` on `DeepSeekV41LayerCache`; `hist_slot` writes in `_latent` |
| `tt/model.py` | taps + stash on the output stage; ring shadows, window histories, fixup traces; `prepare_spec_decode`, `capture_spec_traces`, `spec_fixup` / `spec_write_main` / `spec_seed` / `spec_draft` |
| `tt/spec.py` (new) | `DSparkSpeculator`: host round loop, accept, stats (later: adaptive k) |
| `tools/fill_expert_cache.py` | `--dspark` converts the drafter's 3 x 128 experts |
| `tests/reference.py` | `reference_dspark(...)` via the checkpoint's `DSparkBlock` / `forward_spec` (greedy) |
| `tests/decode/test_dspark.py` (new) | standalone drafter vs the reference: stage 0 attention, MoE on the device's input, logits PCC, first proposals |
| `tests/decode/test_dspark_spec.py` (new) | full model: greedy baseline, corrupted-oracle exactness, DSpark exactness + acceptance |

## 5. Phases and exit criteria

**Phase 0: de-risk.** Host work, plus short device probes (approval before each).
* Time `decode_prompt_traced` on 4- and 6-token bursts vs single steps. Exit: the measured per-round
  verify cost confirms (or replaces) the 3.1 cost model.
* Probe `fused_experts` with 128 experts, `top_k=3` at TP4; `fused_hyperconnection` at 5 rows; SDPA
  decode with 80 folded query heads and one KV head.
* `reference_dspark` runs on CPU.

**Phase 1: verify + rollback, no drafter.** This is the main correctness gate.
* Ring shadows, fixup traces, `tt/spec.py` loop.
* Test with **corrupted oracle drafts**: run plain greedy, then spec where the drafts are the plain
  stream with random corruptions. Exit: the spec stream equals the plain stream token for token over
  >= 256 generated tokens (crosses the ring wrap and both r=2 phases), for k = 1..5.

**Phase 2: taps.** Exit: the stash rows match the reference's `main_hidden` (PCC >= 0.99) for prompt
and generated positions.

**Phase 3: device drafter.** Exit:
* Per-stage PCC vs the reference >= 0.95 (bfp4 experts).
* Draft tokens equal the CPU drafter's on >= 90% of positions given identical taps (metal_2's M2
  test reached 0.90 / 0.75 / ... teacher-forced).
* Drafter trace time measured.

**Phase 4: end to end.** Add a spec mode in `tests/decode/test_full_model_decode_demo.py`, fixed
k = 3 and 5. Report:
* tok/s/user, accepted drafts per round, P(m >= j).
* The split between verify, draft and host time.
* Exactness vs plain greedy, with the top1-top2 logit gap at each first divergence (near-ties are
  expected under bf16 / bfp8).

**Phase 5: optimizations** (rough value order):
1. In-trace argmax for verify rows (send token + top-2, not 258 KB of logits per row).
2. Ring restore folded into the packet (no fixup trace).
3. Adaptive k from the confidence head (metal_2 `AdaptiveSpec`: argmax over k of E[tokens](k) / t(k)).
4. Device-side accept + stage 7 -> stage 0 token loop (host off the critical path).

## 6. Risks

* **Pipeline overlap**: if back-to-back steps do not overlap per stage as assumed (single-page D2H
  FIFO, host packet / Engram rate), the verify cost and the speedup shrink. Phase 0 measures this.
* **`fused_experts` at top-3 / 128 experts**: the zero-padding fallback is exact but doubles drafter
  expert compute.
* **Drafter time at TP4** (15 sequential MoE calls): if it is well above 8 ms, split the 3 stages
  over pipeline stages 5-7, or batch the 5 rows once the op supports it.
* **Context limit**: positions run up to `p + k`, so `ensure_session_capacity` / `max_seq` must
  leave k positions of slack.
* **Lessons from metal_2 to keep**:
  * Compile every trace (decode variants, fixup, drafter) **before the first capture**. A tensor
    allocated after a capture can land in that trace's freed scratch memory and get overwritten by
    its replays.
  * Allocate persistent buffers (shadows, stash, masks) before any trace.
  * Gate on "first generated token equals the CPU reference".
