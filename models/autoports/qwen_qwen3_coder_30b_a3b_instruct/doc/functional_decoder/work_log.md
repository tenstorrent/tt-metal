# Work log — functional decoder (stage 01)

Chronological. Records what was decided, what broke, and how each conclusion
was reached — including the wrong turns, because the reasoning that corrected
them is the part worth keeping.

---

## 01a — HF layer mapping

Read `Qwen3MoeDecoderLayer`. Identified three conversions that fail *silently*
(the layer still runs and returns plausible numbers):

1. RoPE channel ordering (Meta interleaved vs HF split-half)
2. QK-norm weights, which must follow whatever conversion (1) does
3. Expert gate/up fusion order — `Qwen3MoeExperts.forward` chunks the matmul
   output and treats the **first** half as gate

## 01b — Layer-only reference

Loading the 30.5B causal LM to test one layer costs ~57 GB of host RAM. Instead
`tests/reference.py` reads only `model.layers.0.*` out of the safetensors
shards and populates one `Qwen3MoeDecoderLayer` (0.62 B params, ~1 s).

`test_moe_matches_unfused_reimplementation` recomputes the MoE from the raw
per-expert tensors and requires PCC ≥ 0.9999. Everything downstream is compared
against this reference, so an error here would be invisible forever after.

## 01c-1 — Weight mapping, and a reversed decision

Initially followed `tt_transformers`: `reverse_permute` on q/k plus
`reverse_permute_1d` on the QK-norm weights (Meta convention).

**Reversed after reading the exemplar.** `models/demos/gemma4` uses
`ttnn.experimental.rotary_embedding` — HF-style, *"no transformation matrix
needed"*. The two conventions are interchangeable only as a pair; HF-style
weights through a Meta-style op run fine and produce garbage.

Switched to HF-style, which deletes both permutations — and with them trap (2)
entirely, since it only existed as a consequence of trap (1).

Vindicated later: attention landed at 0.9994+ on the first run. A crossed
convention scores 0.3–0.7, not four nines.

## 01c-2 — RMSNorm

First on-device module, 4/4 at ~0.99999. Chosen first deliberately: it proves
the whole harness (mesh open, real weights uploaded, PCC compared) before
anything harder is layered on. When attention later misbehaves, normalisation
is already ruled out.

## 01c-3 — Attention

Exemplar: gemma4 (has Qwen3-shaped per-head QK-norm; gpt_oss does not).
Confirmed ordering matches HF: split heads → per-head norm → RoPE.

**Divergence caught before it bit:** gemma4 also norms V, because Gemma has a
`v_norm`. Qwen3 does not. Copying that line would silently corrupt V.

Config trap: `rope_theta` is not a top-level field on this checkpoint — it is
nested at `rope_scaling.rope_theta = 1e7`. The cos/sin cache is built by
`Qwen3MoeRotaryEmbedding` itself so the value cannot drift.

PCC 0.9997 / 0.9996 / 0.9994 at S = 32 / 128 / 512.

## 01d-1 — Router: the most instructive failure

First attempt scored **0.88** dense-routing PCC, with **83/128 tokens routed to
a different expert set**.

Diagnosis, host-side simulation:

| configuration | tokens differing |
|---|---|
| fp32 logits, bf16 softmax | 71/128 |
| bf16 logits, bf16 softmax | 75/128 |
| bf16 logits, **fp32 softmax** | **5/128** |

The bf16 *matmul* was blameless (logit PCC 0.9999973). The softmax was the
whole story: over 128 experts the 8th-place probability sits near 0.008 with
only ~1.4e-5 separating it from the 9th, while bf16 resolution there is
~3.2e-5. The two candidates are the same number in bf16.

**First fix was insufficient.** Moving the softmax to fp32 only reached 0.963 /
34 tokens, because `ttnn.softmax` carries ~3.3e-4 of error *regardless of tensor
dtype*. Device measurements isolating each op:

```
ttnn.topk   on fp32 softmax probs ... 0/128  (topk is exact)
ttnn.topk   on fp32 logits ........... 0/128
ttnn.softmax fp32 vs torch fp32 ...... max abs error 3.3e-4   <-- the culprit
```

**Actual fix: delete the op.** Softmax is monotonic, so it cannot change *which*
experts win; and with `norm_topk_prob` the denominator cancels during
renormalisation:

```
w_i = [exp(x_i)/Z] / Σ_{j∈top8}[exp(x_j)/Z] = exp(x_i) / Σ_{j∈top8} exp(x_j)
```

So the 128-wide softmax is unnecessary. Selecting on raw logits and softmaxing
over the 8 survivors is identical by algebra and removes the error rather than
mitigating it. **0.88 → 0.9999987**, 5/128 — exactly the predicted bf16
projection floor.

Guarded by an assert on `norm_topk_prob`, since the cancellation depends on it.

Also hit: `ttnn.scatter` has no fp32-tiled support (`scatter.cpp:109`), forcing
a bf16 cast. Harmless — representing a weight to 0.4% is a different problem
from deciding which weights exist.

## 01d-2 — Experts

Correct on the first run: 0.99981 isolated, 0.99811 end-to-end.

Two more gemma4 divergences: **SwiGLU not GeGLU** (`hidden_act="silu"`), and no
router pre-norm / per-expert scale.

`fp32_dest_acc_en` left **off** despite looking like the obvious accuracy knob —
it halves the matmul dest and corrupts expert output on Blackhole
(tt-metal #49068). HiFi4 supplies the accuracy instead.

Testing the experts against the *reference's own* routing weights, separately
from end-to-end, is what makes the end-to-end number interpretable: it
separates expert arithmetic from router selection.

## 01e — Composition

0.9991 / 0.9995 / 0.9994 at 32 / 128 / 512.

The composed layer scores *above* its MoE component (0.9991 vs 0.9981), which
is the expected signature of a correct pre-norm residual — the residual stream
dominates, so sublayer error is diluted rather than compounded.

One bug, mine: `torch.allclose` on a bf16 device output vs an fp32 reference —
`RuntimeError: BFloat16 did not match Float`.

## 01f — Decode, tracing, non-aligned lengths

**Sharded-layout mismatch.** `paged_update_cache` requires sharded K/V
(`paged_update_cache_device_operation.cpp:255`) but `rms_norm` needs
interleaved DRAM. Fix: capture the layout the head-split produced, restore it
after RoPE.

**RoPE sequence padding.** `K: 64, V: 33` — the rotary op pads dim 2 to a tile
multiple, and V never passes through RoPE. Worth remembering because dim 2
means *different things in the two modes*: heads in decode, sequence in
prefill. The same op is a hazard twice, for unrelated reasons.

**Blackhole #16667.** `nlp_create_qkv_heads_decode` zeroes odd-indexed Q rows
when the fused QKV is read from DRAM. Staged through L1 first. Guarded by a
zero-fraction test, because with half of Q zeroed the output is still finite
and still scores a deceptively high PCC.

**A non-bug worth recording.** Token 32 of seq=33 scored 0.99465 — right at the
pad boundary, exactly where a padding bug should appear. The per-token profile
showed tokens 6 and 32 low at *every* length, including seq=32 where no padding
exists, median 0.9998. They are router near-ties, not padding damage.

That changed the *test*, not the code: the tail check now asserts the tail is no
worse than the sequence's own worst body token. An absolute per-token bar cannot
separate "padding corrupted the tail" from "this token was always noisy".

**Tracing.** Bit-exact replay (PCC 1.0). The second test — overwrite the input
buffer and confirm the output changes (Δ 0.396) — is the one that matters: a
trace that captured its input by value replays stale results forever and passes
every equality check.

## 01f (completion pass)

Audited against the stage-01 goal prompt and found real gaps. Closed:

- **Paged KV cache + page table.** Was non-paged (`page_table=None`). Added a
  `KVCache` abstraction supporting both; paged(32) and paged(64) match
  contiguous PCC exactly. Multi-step decode now runs paged, since that is where
  a block-table mapping error would surface.
- **Context contract.** Probed rather than assumed. Both prefill and decode
  reach the **full 262144** HF context on one die — no capability reduction.
  Validated by `check_context_contract.py`.
- **Determinism.** Bit-identical across repeated prefill and across two
  independent prefill+decode sequences from fresh caches; 64-step rollout
  stable (activation std ratio 1.42).
- **Performance + profiling.** Baselines recorded; `SparseMatmul` is 96.8% of
  prefill and 92.6% of decode.
- **Fallback audit.** AST scan of all 11 forward functions: zero
  `torch`/`from_torch`/`to_torch`. Corroborated by trace capture succeeding,
  which is impossible if the graph contains host ops.

## Handoff to stage 02

Evidence-backed optimisation targets, in order of expected value:

1. **`active=128/128` in prefill.** All 128 experts computed per chunk when 8
   are needed. Decode already passes real sparsity; prefill does not.
2. **~5.4% of peak FLOPs**, DRAM 12.5–24.9% on expert matmuls.
3. **24 of 110 worker cores** on gate/up (64 on down) — `_sparse_matmul_config`
   is unturned.
4. **bf16 → `bfloat8_b`/BFP4 + fidelity policy** on expert weights. Note the
   #49068 constraint: `fp32_dest_acc_en` must stay off.

Baselines to beat: prefill 536 µs/token; traced decode 1.565 ms @ ctx128,
1.993 ms @ ctx4096.
