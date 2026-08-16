# DFlash speculative decoding — work log

Goal: implement the DFlash drafter Muse-Glimmer-30B ships with, and win decode
t/s/u at batch 1.  The model card advertises **3.1× on an RTX 5090** (74.9 →
233.4 tok/s) from this feature, and no stage of the original bring-up ported it.

Status: **drafter implemented and numerically validated on CPU; device PCC
queued behind another job holding the chips.**

---

## What DFlash actually is, on this checkpoint

The drafter is a *separate published artifact*,
[`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)
— public, ungated, 5.11 GB, `MuseGlimmerAssistantModel`.  The reference
implementation ships in the installed `transformers` 5.15.0, so the port is
graded against real HF math rather than a re-implementation:

| | |
|---|---|
| drafter | 5 layers, all `sliding_attention`, window 2048 |
| dims | hidden 6656, FFN 19968, 32 Q / 8 KV heads, head_dim 128 |
| block | `block_size` 16 → 1 anchor + 15 drafted tokens |
| context tap | target hidden states at layers `[1, 13, 25, 37, 49]`, concatenated → 33280 |
| cache | `DFlashCache` — context K/V persists, the diffusion window is appended then cropped |
| driver | `DFlashTokenCandidateGenerator`, selected by `speculation_type="dflash"` |

**The drafter has no `embed_tokens` and no `lm_head`.**  Its 58 tensors are 5
decoder layers + `encoder.fc` + `encoder.output_norm_enc` + `norm`.  Confirmed
two ways: read from the safetensors header, and by arithmetic — 5 × 467 M +
221.5 M ≈ 2.56 B params × 2 bytes = **5,111,976,608 bytes**, which is the file
size exactly.  Input embeddings come from the *target's* table via a plain
lookup; candidate logits come from the *target's* `lm_head`.

---

## Four ways this differs from the target decoder

Each of these produces a silent accuracy loss rather than a crash if ported by
analogy from the existing `functional_decoder.py`:

1. **Plain RMSNorm, not centered.**  The target's text layers use
   `MuseGlimmerTextCenteredRMSNorm` (`x * (1 + w)`), which the target port
   pre-folds the `+1` into at setup.  The drafter uses ordinary `x * w`.
   Reusing `_MuseGlimmerNorm` here adds a spurious `+1` to every weight.

2. **QKV cannot be fused.**  Q is projected from the 16-token window; K/V are
   projected from `concat(context, window)`.  The target port fuses QKV into one
   matmul — correct there, wrong here.

3. **Attention is bidirectional.**  `is_causal = False`; the mask is
   `bidirectional_mask_function AND sliding_window_overlay(w)`, and since the
   former is unconditionally true the whole condition collapses to
   `kv_idx > q_idx - 2048` — **a lower bound only, no causal upper bound**.  The
   16 window positions see each other in both directions.  A causal mask still
   runs and still emits plausible tokens; it just lowers the acceptance rate,
   which is invisible unless measured.

4. **The context half of K/V is not re-normalised per layer.**  Each layer
   applies `input_layernorm` to the window only; the context entering K/V is the
   encoder output, shared unchanged by all five layers.

---

## Findings

### F1 — The speedup is real: 4.80× fewer target forwards, measured

`tests/dflash_cpu_oracle.py` runs the genuine end-to-end loop on CPU with real
target + real drafter weights, and counts target forwards via a hook.

| | baseline greedy | DFlash |
|---|---|---|
| tokens | 48 | 48 |
| target forwards | 48 | **10** |
| wall clock | 68.9 s | **16.6 s** |

**5.33 accepted tokens per target forward** (ceiling 16), **4.80×** fewer
forwards, **4.14×** wall-clock. Above the card's 3.1× claim — CPU flatters the
drafter, since it is 11.7× smaller than the target and the ratio is
bandwidth-bound, but the forward-reduction figure is hardware-independent and is
the one that transfers.

### F2 — DFlash is *not* output-lossless in bf16, and that is the target's fault

The oracle found DFlash diverging from plain greedy at token 39 of 48.  That
should be impossible: a token is only accepted when it equals the target's own
argmax, so the accepted sequence should *be* the greedy sequence.

`tests/dflash_divergence_probe.py` settles it **without the drafter involved at
all** — take the greedy sequence, re-score it with one wide teacher-forced
forward, compare per-position argmax:

```
positions compared: 48
argmax mismatches between incremental and teacher-forced: 1
  idx 39: incremental 1574 vs wide-forward 20694 | top2 gap 0.06250 | incr rank 1
VERDICT (B): the target model's own argmax depends on forward width in bf16.
```

A top-2 gap of **0.0625** is one bf16 ulp at that magnitude.  Baseline greedy
computes logits with `query_len == 1`; DFlash verifies with `query_len == 16`;
those are different reduction orders through identical weights, so near-tied
logits argmax differently.

**Consequence for the port:** "token-identical to greedy" is not an achievable
correctness gate and must not be used as one.  Gate instead on (a) accepted
tokens being the target's argmax from the verify forward *by construction*,
(b) divergence rate staying at the bf16 near-tie noise floor, (c) eval parity.
The card's "producing identical output quality" is true only in exact arithmetic.

### F3 — `to_empty()` + `assign=True` silently produced a garbage RoPE table

Cost roughly an hour, and is worth recording because the failure is invisible.

The first reference harness built the model on `meta`, then `to_empty(device=
"cpu")` and `load_state_dict(..., assign=True)`.
`MuseGlimmerAssistantRotaryEmbedding.inv_freq` is a **non-persistent buffer**,
so it is absent from the state dict, `to_empty` gave it uninitialised memory,
and nothing ever filled it.  HF's `inv_freq` came out as
`[0.0, 1.9e-19, 0.0, 4.7e-18, 0.0, 0.0]` — garbage — instead of
`[1.0, 0.815, 0.664, ...]`.

The model ran without complaint and produced plausible activations, so the
goldens were quietly wrong and the port was graded against noise: end-to-end PCC
0.73–0.92, *degrading with context length*, which looks exactly like a real
attention bug.

Localised by reimplementing the layer in **pure torch** first: it failed
identically (0.92 vs the port's 0.916), which proved the fault was in the shared
understanding rather than in any ttnn op.  Hooking HF's layer-0 internals then
showed everything matching to ≥0.999996 up to `q_norm`, and the RoPE tables
disagreeing at PCC 0.011 with `max|Δ| = 2.0`.

Fixed by loading through `from_pretrained`, plus `_assert_rope_initialised()`,
which checks `inv_freq` against the analytic default table so this can never
recur silently.

**After the fix, the pure-torch reimplementation matches the HF drafter at
0.99994 on every layer** — confirming the bidirectional mask, plain RMSNorm,
unfused QKV, context-as-KV, QK-norm and RoPE slicing are all right.

### F4 — Device contention is not handled by anything in this repo

Device PCC could not run: another job's `VLLM::EngineCore` (from
`/home/ttuser/dev/laguna/.../poolside_laguna_xs_2_1`, orphaned to init) holds
`CHIP_IN_USE_0_PCIe`.  tt-metal blocks on the lock with a bare warning and no
timeout, so a test run simply hangs until the 300 s pytest timeout fires and
reports as a test failure rather than as "hardware busy".

Note also that `pgrep -x 'VLLM::EngineCore'` cannot match it — the name exceeds
the 15-char comm limit — and `pgrep -f` matches the polling script's own argv.
Both traps were hit while writing the waiter.

---

## Artifacts

| file | what |
|---|---|
| `tests/reference_dflash.py` | HF reference + golden generator; asserts the 58-tensor contract, absence of `embed_tokens`/`lm_head`, pinned config, initialised RoPE |
| `tests/dflash_goldens.pt` | goldens at context 1 / 16 / 128 / 2048 / 4096 (4096 is the only one exceeding the window) |
| `tests/dflash_cpu_oracle.py` + `.json` | end-to-end CPU oracle: acceptance rate, forward reduction, losslessness check |
| `tests/dflash_divergence_probe.py` + `.json` | isolates F2 to target-model numerics |
| `tt/dflash_drafter.py` | the TTNN drafter |
| `tests/test_dflash_drafter.py` | PCC parity + mask-semantics unit tests |

## Next

1. Device PCC (queued; fires when the chips free).
2. Target hidden-state taps at layers 1/13/25/37/49.
3. The draft/verify/accept loop in the generator.
4. Batch-1 t/s/u sweep against the 43.4 t/s/u baseline.
