# DeepSeek-V3 prefill non-determinism investigation (branch kgrujcic/deepseek_nd)

**Owner:** automated debug session (Claude) for kgrujcic
**Started:** 2026-05-31
**Repo under test:** `/data/kgrujcic/tt-metal-sandbox` (Blackhole galaxy, mesh-8x4 = 32 chips)
**Symptom:** `test_prefill_transformer.py` produces a different first token across
repeated runs on identical input.

> This investigation was started **from scratch** at the user's request; prior
> `drift_analysis/*` docs (different branch) are treated as untrusted and are not
> relied upon. Conclusions here are only what this session measured directly.

## Method / measurement discipline

The sampler uses a **Gumbel-softmax trick at `temperature=0.5`**
(`tt_prefill_transformer._sample_token`), drawing from torch's global RNG. The
RNG advances every iteration, so the **sampled token is NOT a valid determinism
signal** within a process. We therefore fingerprint the **logits** and
**per-layer hidden states** (deterministic functions of the model), never the
sampled token.

Instrumentation added (all gated, pure measurement, no device-compute change):
- `utils/nd_debug.py` — bit-exact SHA1 + f64 norm/sum + non-finite count.
- `TT_DS_ND_DEBUG=1` — test fingerprints per-layer hidden states + logits each
  iteration; `nd_compare_log` reports the first stage that diverges between
  consecutive iterations (within-process).
- `TtMoe.forward` per-stage probes (`[NDPROBE-MOE]`): gate scores/indices,
  shared_output, dispatched_buf, metadata, expert_outputs, combined_output,
  routed_output, final_output. Device-0-shard fingerprints (cheap).
  Target layer via `TT_DS_ND_DEBUG_LAYER` (default 3 = first MoE layer).
- `nd_logs/nd_compare_runs.py` — compares two run logs at the same iter index
  (across-process) or consecutive iters (within-process); prints first divergence.
- `TT_DS_ND_COMBINE_INIT_ZEROS=1` — flips the hardcoded `init_zeros=False` of the
  combine module to `True` (no rebuild) for A/B.

## Evidence collected so far

### 1. Non-determinism is real and large (across-process)
From a colleague's 20× soak (`nostojic`, same 61-layer/iter25/25600 test, separate
processes, each `torch.manual_seed(42)`), the final first token across 14 runs:
`312, 14085, 17926, 31, 3244, 260, 31, 260, 31, 260, 260, 31, 20390, 31` with
sampled-token probabilities ranging **0.005 – 0.77**. Identical input + fixed seed
⇒ fixed Gumbel ⇒ the **logits themselves diverge run-to-run**. Not sampling noise.

## Leading hypothesis (H1): uninitialized combine-output DRAM → NaN poisoning

`tt_moe.py:337` constructs the combine module with **`init_zeros=False`** (default
is `True`). The combine op allocates its output via `create_device_tensor`
(uninitialized DRAM) and, with `init_zeros=False`, does **not** pre-zero it.

The reduce kernel
(`post_combine_reduce/.../deepseek_moe_post_combine_reduce_compute.cpp`) streams
`cb_combine_input` for **every** expert slot of every token (waits/pops all
`num_experts`). For the accumulator-initializing slot (`must_zero_init` on the
DeepSeek dispatch-table path, or `first_active` on the GPT-OSS path) it executes
`mul_tiles_bcast(combine_input, weight=0)` — i.e. it **reads the combine buffer
and multiplies by zero**, relying on `garbage * 0 == 0`.

But for an **unwritten** combine slot with `init_zeros=False`, the buffer holds
uninitialized DRAM. If those bits decode to **NaN/Inf** (entirely possible for
bf16/bf8 garbage), then `NaN * 0 = NaN`, which poisons that token's reduced
output. The *set* of poisoned tokens is deterministic (the skip decision reads
deterministic weights / dispatch table), but the garbage *value* is
non-deterministic per process → **non-deterministic logits run-to-run**, matching
the symptom. Over 61 layers any poisoned token compounds through the residual
stream into a totally different argmax/sampling outcome.

**Predicted A/B result:** `TT_DS_ND_COMBINE_INIT_ZEROS=1` should remove the NaN
poisoning. If it removes *all* run-to-run divergence → H1 is the root cause and
the fix is to zero-init the combine buffer (or have reduce not read zero-weight
slots). If divergence shrinks but persists → there is also a finite-garbage path
(a genuinely missing combine write with non-zero weight) and/or a separate cause
(kernel L1 race) to chase.

**Confirmed (static):** combine `init_zeros=True` zeros the *entire* output tensor
(`combine_program_factory.cpp:746` splits `get_num_pages(output_tensor)` across all
cores), so it covers the no-local-expert slots the reduce reads. The A/B is valid.
The MoE probe also fingerprints gate scores/indices, so it distinguishes a
routing-level cause (gate non-determinism) from the combine cause.

### IMPORTANT caveat on H1 (avoid over-anchoring)
`finite_garbage * 0 == 0` (the *correct* result). H1 only corrupts when the
unwritten slot's DRAM decodes to **non-finite** (NaN/Inf). The observed symptom
(nostojic soak) is **clean but different** tokens — valid IDs, finite probs
0.005–0.77 — i.e. **dense drift, not NaN poisoning**. So H1 most likely explains
only the *intermittent catastrophic* case, not the dense run-to-run drift. The
dense drift is more consistent with:
- **Non-deterministic FP accumulation order** in a multi-core matmul or a CCL
  reduce/all-reduce (core scheduling varies per process → different fp rounding),
  and/or **finite L1 leftovers** read with non-zero weight, then **amplified by
  gate top-k flips** over 61 layers (tiny perturbation → different expert routing
  in deep layers → large output change).

The probe is built to tell these apart: bit-exact SHA finds the FIRST diverging
stage even for sub-ULP drift; `nonfinite` count flags NaN/Inf (H1); gate-index
fingerprints reveal routing flips (amplification). **Conclusion deferred to the
on-device measurement — do not commit to H1 until the first-diverging stage and
its finiteness are measured.**

## Backup hypotheses (if H1 A/B does not fully fix determinism)
- **H2 — dispatch-buffer padding (FFN input).** `dispatch` output is also
  `create_device_tensor` (unzeroed); padding rows within each expert region are
  garbage and the FFN matmul computes on them. But combine only routes *valid*
  token rows back (by metadata), and matmul M-rows are independent, so this garbage
  should stay in padding rows and not reach valid outputs. Lower priority.
- **H3 — L1 leftovers / races in compute kernels.** A kernel reading an
  un-/under-initialized CB or a multicast handshake race would also be per-process
  non-deterministic. Hard to localize statically; pursue only if H1/H2 fail. The
  per-stage probe will point at the offending stage.

## MEASURED RESULTS (on-device, 2026-05-31, this branch)

### Run 1: 5 layers, 1024 tokens, iter5 (baseA), then a second process (baseB)
- **Within-process: FULLY deterministic.** All 5 iterations bit-identical
  (per-layer + lm_head SHA match). ⇒ the non-determinism is **not** iter-to-iter
  leaked state; device buffers are reused at the same address with the same
  leftover within one process.
- **The combine/dispatch buffers DO carry non-finite garbage** (device-0 shard):
  `02_dispatched_buf` nonfinite=8263, `04_combined_output` nonfinite=1697, with
  ~1e40 magnitudes — exactly the uninitialized-DRAM effect of `init_zeros=False`.
- **Across-process (baseA vs baseB @ iter0): the model output is DETERMINISTIC.**
  Every per-layer hidden state, `norm`, and `lm_head` is **bit-identical** across
  the two processes; `05_routed_output` and `06_final_output` are **bit-identical**
  too. Only the garbage-bearing buffers (`02_dispatched_buf`, `02b_disp_tiled`,
  `03_expert_outputs`, `04_combined_output`) differ — i.e. **reduce correctly
  masks the garbage out; it never reaches valid output at this scale.**

**⇒ H1 is real but HARMLESS at 5L/1024 — the combine garbage is masked.** The
run-to-run non-determinism the symptom shows therefore needs **scale** (25600
tokens and/or 61 layers). Hypotheses re-ranked: the trigger is something that only
manifests at higher token-count / depth — candidates: (a) the garbage masking
breaks at scale (e.g. an `Inf` that survives `must_zero_init`, or a dispatch-buffer
**capacity overflow** that drops tokens), or (b) a genuinely non-deterministic
CCL/matmul accumulation under higher load. Next: reproduce at 25600 tokens.

### Run 2: 5 layers, 25600 tokens, iter1 — REPRODUCED + ROOT-CAUSE LOCALIZED
Across two separate processes (C1/C2, then D1/D2 with all-shard GLOBAL probe),
identical input + fixed seed:
- First diverging **layer = layer_3** (first MoE layer); dense finite drift
  (layer_3 dnorm ~1e-4..1e-3, **nonfinite=0**), amplifying through norm → lm_head.
- Within layer 3 (GLOBAL all-shard fingerprints):
  - `00_moe_input`, gate scores/indices, **`01_shared_output`**, `02_dispatched_buf`
    (FFN **input**), `02_metadata`, `02b_disp_tiled` → all **SAME** (bit-identical).
  - `03_expert_outputs` (FFN **output**) → **DIFF**; `05_routed_output` (GLOBAL) →
    **DIFF** (norm 54.3025 vs 54.3057, dnorm 3.1e-3, nonfinite=0); `06_final_output`
    → DIFF.

**ROOT CAUSE: the routed-expert FFN compute is non-deterministic given identical
input.** Same FFN input buffer → different FFN output across processes. The shared
expert (also matmuls) is GLOBALLY bit-deterministic, so this is **not** generic
matmul non-determinism — it is specific to the **routed-expert FFN** (the unified
fused kernel on Blackhole). The drift is **finite** (no NaN), so it is an
L1-leftover / multicast-handshake race in the kernel, **not** the combine-buffer
garbage (H1) — confirmed harmless at every scale here.

Also confirmed NOT the cause: KV-cache state (within-process fully deterministic),
gate/routing (indices bit-identical), dispatch buffer (FFN input identical),
capacity overflow (per-chip 12356 ≪ capacity 204800), shared expert.

### Run 3: A/B unified vs naive routed-expert path (TT_REXPERT_FORCE_NAIVE), 5L/25600
Across two processes each, identical input, GLOBAL all-shard fingerprints:

| stage (layer 3)            | UNIFIED (D1 vs D2) | NAIVE (N1 vs N2) |
|----------------------------|--------------------|------------------|
| `03_expert_outputs` (FFN)  | **DIFF**           | **SAME**         |
| `05_routed_output` GLOBAL  | **DIFF** (dnorm 3.1e-3) | **SAME**    |
| `04_combined_output`       | DIFF (garbage)     | DIFF (garbage)   |
| `layer_3` hidden           | **DIFF**           | **SAME**         |
| `lm_head`                  | **DIFF** (dnorm 0.24) | **SAME**      |

**CONFIRMED ROOT CAUSE: the unified fused routed-expert FFN kernel
(`ttnn/.../unified_routed_expert_ffn`) is non-deterministic across processes given
identical input.** The naive per-expert path (extract → `routed_expert_ffn` default
matmuls → insert) is deterministic and yields a bit-identical `lm_head`. The combine
buffer garbage (`04_combined_output` DIFF, from `init_zeros=False` / H1) is present
in BOTH paths and is harmlessly masked in both — so H1 is NOT the cause.

Residual: NAIVE shows a tiny layer_4 drift (dnorm 5.9e-5, `lm_head` still SAME) —
negligible vs the unified kernel's dominant effect, but may compound over 61 layers
(to investigate separately; likely a minor combine-garbage or CCL-order effect).

## FIX
Two options:
1. **Low-risk config fix:** route the routed expert through the deterministic naive
   path (gate the unified kernel off). Restores determinism; costs the unified
   kernel's perf gain.
2. **Proper fix:** repair the non-determinism inside the unified FFN kernel
   (likely a per-K-block multicast-handshake ordering / L1-reuse race — the valid
   semaphore may be observable before the mcast data fully lands, or an L1 buffer is
   reused before all receivers consumed it). Kernels are JIT-compiled, so this can be
   iterated and verified on-device with the 5L/25600 fingerprint A/B (no full rebuild).

### Status of experiments
- [ ] Within-process drift (iter-to-iter) — pending device (blocked by soak)
- [ ] Across-process first-diverging stage — pending device
- [ ] A/B `init_zeros` True vs False — pending device

> Device note: shared 32-chip galaxy was occupied ~2h by a colleague's soak that
> `pkill -9 -f pytest` + `tt-smi -glx_reset` between iterations, so concurrent runs
> are impossible. Experiments queued for when it frees.
