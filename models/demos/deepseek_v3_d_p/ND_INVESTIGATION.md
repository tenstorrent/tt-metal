# DeepSeek-V3 prefill non-determinism investigation (branch kgrujcic/deepseek_nd)

**Owner:** automated debug session (Claude) for kgrujcic
**Started:** 2026-05-31
**Repo under test:** `/data/kgrujcic/tt-metal-sandbox` (Blackhole galaxy, mesh-8x4 = 32 chips)
**Symptom:** `test_prefill_transformer.py` produces a different first token across
repeated runs on identical input.

---
## ★ RESOLUTION SUMMARY (2026-06-02) — supersedes everything below

**The earlier "unified FFN kernel" conclusion was WRONG** (it was based on 2 across-process
samples that agreed by chance; two samples never prove determinism for a probabilistic
across-process race). At full scale the **naive** routed-expert path is ALSO non-deterministic
(61L/iter25: tokens 2845 vs 6029), so the seed is NOT in the experts — it is **upstream, in
the attention backbone, shared by both FFN paths.**

**Method:** cross-process *allshards* SHA fingerprints at every sub-stage (probes added in
`tt_prefill_block.py` BLK_*, `mla/mla.py` MLA_*, `moe/tt_shared_expert.py` SE_*, all gated by
`TT_DS_ND_SUBSTAGE=1`; MoE-stage probes by `TT_DS_ND_DEBUG=1`). Always compare **logits /
hidden-state fingerprints**, never the Gumbel-sampled token (temperature=0.5 → sampled token
is RNG-confounded). Reproduces only at scale (61L + 25600 tok).

**PRIMARY ROOT CAUSE (found + fixed):** `ttnn.transformer.ring_joint_scaled_dot_product_attention`
(`mla.py:679`, the ring/distributed flash attention). Its inputs (Q, V-embedding) are
bit-identical across processes, but its **output differs by ~1e-6** — the bf16 *streaming*
online-softmax merge combines cross-chip partial results in a timing-dependent way. The sub-ULP
seed compounds through 61 residual layers and flips gate top-k routing → different token.
First divergence: `BLK_2_mla_out` / `MLA_2_attn_out_postSDPA` (attn_norm is deterministic).
**FIX (validated):** point the ring SDPA at an `fp32_dest_acc_en=True` compute-kernel config
(the non-streaming fp32 merge path) — env-gated `TT_DS_ND_SDPA_FP32=1`. With it the **prefill
forward is bit-deterministic across processes** (iter0 layer-60 `06_final_output` SHA identical
across ~7 processes; iter1 ×4 all token 2845). **Trade-off:** fp32 non-streaming SDPA is slower.

**FIX (2) — persistent-buffer reset (default-on, `mla.py`):** reset
`persistent_k/v_output_buffer = ttnn.zeros_like(...)` before each ring SDPA call
(`TT_DS_ND_RESET_PBUF=0` to disable). This moved the **reported 61L/iter25 token from
divergent (260/14) to deterministic + correct (2845) across 7 processes.** Both fixes are
now DEFAULT-ON (fix (1) via `TT_DS_SDPA_FAST=1` escape; fix (2) via `TT_DS_ND_RESET_PBUF=0`).

**STATUS: user-observable symptom RESOLVED.** A fresh process / single prefill (= real usage,
= iter0) is **bit-deterministic and correct (2845)** across processes; the reported iter25
token is **2845 deterministically across 7 processes.** Clean first-token-logit signal
(`TT_DS_ND_LOGIT_FP=1`): argmax = 2845 for 24/25 iters across 3 processes; iter0 bit-identical.

**REMAINING bit-level residual (does NOT affect the reported token; needs kernel-level work):**
when the SAME process re-runs the prefill (iter1+ of the soak loop — e.g. a server reusing a
process across requests), some iters drift at the logit level and one (iter1) shows non-finite
garbage that flips that iter's argmax. Ruled OUT as causes: KV cache (resetting it per iter did
NOT help; `ttnn.mul_` on the paged cache segfaults — use realloc), and it is NOT a per-op
reduction-order issue (iter0 is bit-perfect, so all reductions are deterministic given equal
inputs). The garbage originates in the **ROUTED-EXPERT dispatch/combine buffers**, which are
allocated uninitialized (`create_device_tensor`) inside the C++ `dispatch`/`combine` ops — the
long-standing "masked H1/H2 garbage" surfacing on DRAM re-use. `TT_DS_ND_COMBINE_INIT_ZEROS=1`
is NOT a clean fix (it removed the non-finite values but introduced iter0 drift). A proper fix
must zero-init the dispatch/combine output (or its padding) inside the C++ op, plus make the
reused per-call workspace deterministic. Out of scope of the attention (ring-SDPA) fix.

--- (original "persistent buffer" framing of the secondary residual was partially right:
the buffer reset helped the token; the deeper bit-level residual is the dispatch/combine
uninitialized DRAM in the routed-expert path.) ---

---

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

## ⚠️ CORRECTION (2026-06-01, later): the CB-buffering theory below is DISPROVEN at scale

The 5L/25600 GLOBAL `routed_output` oracle is **UNRELIABLE for validating kernel
fixes**: this race is timing-dependent, and ANY timing perturbation (DPRINT, or
single-buffering CBs) *masks* it at 5L (2 MoE layers) — a **false negative** — while
it survives at 61L. Verified at 61L/iter25/pie960 (two runs each):
- unified baseline: tokens 14 / 260 → ND (control).
- single-buffer `cb_in0_down_full`: tokens **260 / 1086** → still ND.
- single-buffer ALL input CBs: tokens **2919 / 260** → still ND.

So **CB double-buffering is NOT the root cause** (it only changed 5L timing). The
section below is kept for the record but its "fix" does not hold at 61L; all kernel
edits were reverted to pristine. What the evidence robustly says about the race:
- It is in the **unified FFN compute** (naive path is deterministic at 61L).
- **Within a process it is fully deterministic** (iter-to-iter bit-identical);
  **across processes it differs.** ⇒ the race resolves the same way for a given
  process but differently across processes — consistent with a **fixed-per-process
  scheduling/ordering** dependency (e.g. cross-core accumulation/arbitration order or
  a once-per-process initial-state read), NOT a within-iteration nondeterminism.
- Added **synchronization masks it** (DPRINT global throttle → deterministic).
- It **scales with depth/token-count** (5L rarely manifests; 61L reliably does).

**Reliable fix remains the shipped naive default** (validated deterministic at
61L/iter25). Pinning the unified-kernel race needs a **non-Heisenbug** observation at
61L — a DRAM-scratch per-core checksum tensor the op returns (no host-sync, unlike
DPRINT) — then a structural fix. This is kernel-owner-level work with slow (61L)
validation; the 5L fast oracle cannot be trusted for it.

### Fix levers ruled out at 61L (all reverted to pristine)
| lever | 61L/iter25/pie960 (two runs) | verdict |
|-------|------------------------------|---------|
| unified baseline | 14 / 260 | ND (control) |
| reader sender `flushed`→`barrier` + atomic barriers | ND already at 5L (reliable) | not it |
| single-buffer `cb_in0_down_full` | 260 / 1086 | not it |
| single-buffer ALL input CBs | 2919 / 260 | not it |
| `fp32_dest_acc_en=true` | 14 / 469 (both p=1.0) | not it |
| **naive path (shipped fix)** | **2845 / 2845** | **deterministic** |

⇒ Not CB double-buffering, not sender-side mcast completion ordering, not
fp-accumulation precision. The defect is a **cross-core data race** (read-before-write)
in the unified FFN compute path, whose outcome is fixed-per-process — i.e. it depends
on per-process core scheduling/arbitration, which global synchronization (DPRINT)
forces into a canonical order (hence DPRINT masks it). Each process confidently
computes a *different* wrong result (fp32 runs both gave p=1.0 on different tokens),
so it is corrupting real data, not just last-bit rounding.

### Honest status
The observable non-determinism is **fixed** (deterministic naive default, validated at
full 61L/iter25). The **unified kernel's internal race is NOT structurally fixed** —
it is precisely characterized and all accessible fix levers are ruled out, but pinning
the exact racing access requires DRAM-scratch 61L instrumentation + likely NoC-level
analysis (kernel-owner work). Recommended next step is exactly that; the reliable
determinism is available now via the shipped default (perf cost) until then.

---
## (superseded) earlier hypothesis — CB double-buffering of `cb_in0_down_full`

How it was pinned (all via the 5L/25600 GLOBAL `routed_output` A/B oracle, which is
bit-stable when deterministic):
- DPRINT confirmed it's a **timing race** but *masks* it (Heisenbug): with DPRINT on,
  `routed_output` GLOBAL is identical across runs; off, it differs. So DPRINT can't
  localize it.
- The bug reproduces with token count (more M), and is killed by **synchronization**.
- **CB double-buffer bisection** (env `TT_DS_ND_SB_*`, single- vs double-buffering CB
  groups), each config run twice:
  - single-buffer ALL input CBs → deterministic (correct value `e0fa3578…`).
  - single-buffer gate/up only (down double) → still ND (`ccd48e64` vs `9d7a964a`).
  - single-buffer down only (gate/up double) → deterministic (`e0fa3578`).
  - single-buffer **only `cb_in0_down_full`** (activated) → deterministic (`e0fa3578`).
  - single-buffer only `cb_in1_down` (weights) → (control).

**Mechanism:** phase 4 mcasts each K-block's activated from a **rotating sender**
(`gx == kb`) into every M-row core's `cb_in0_down_full`, coordinated by a **single
per-core `act_ready`/`act_valid` semaphore**. With `cb_in0_down_full` double-buffered,
the reader prefetches K-block `kb+1` (whose sender is a *different* core) while compute
is still consuming `kb` — so two different sender cores race on the receiver's single
`act_valid`/`act_ready` sem, intermittently signalling the wrong slot. The gate/up
phases use *fixed* senders, so their double-buffering is race-free. Single-buffering
`cb_in0_down_full` forces the reader to wait until compute pops `kb` before starting
`kb+1`'s handshake, serialising the rotating-sender sem accesses — structurally
removing the race (deterministic AND correct value).

**FIX (shipped): single-buffer `cb_in0_down_full`.** Low-risk, verified deterministic.
The costly DRAM weight prefetch (`cb_in1_down`) stays double-buffered, so the perf cost
is only the activated L1-mcast/down-matmul overlap (modest). **Perf-preserving
follow-up:** give the activated mcast **per-slot (kb%2) `act_ready`/`act_valid`
semaphores** so consecutive rotating senders use different sems — then
`cb_in0_down_full` can stay double-buffered. (More intricate; not yet implemented.)

## FULL-SCALE VALIDATION + SHIPPED FIX (2026-06-01)
Validated at the real scale (61 layers, iter25, 25600 tokens) using the
drift-sensitive `pie960` input and the Gumbel-safe token oracle (two separate
processes, same `manual_seed(42)` → identical Gumbel → the final token differs
ONLY if the logits are non-deterministic):

| routed-expert path | run 1 | run 2 | verdict |
|--------------------|-------|-------|---------|
| **unified** (BH fused kernel) | token 14, p=0.9210 | token 260, p=0.0274 | **NON-DETERMINISTIC** |
| **naive** (per-expert) | token 2845, p=1.0000 | token 2845, p=1.0000 | **DETERMINISTIC** |

(longbook_qa_eng, which predicts a near-certain token, also gave a stable 1009/1009
under naive but is a weak discriminator — pie960 is the discriminating input.)

**SHIPPED FIX:** `tt_routed_expert.py` now defaults the Blackhole routed expert to the
deterministic **naive** path. `TT_REXPERT_FORCE_UNIFIED=1` re-enables the faster
non-deterministic unified kernel for perf work; `TT_REXPERT_FORCE_NAIVE=1` forces naive
everywhere. WH 2880 path unchanged. This makes the prefill **deterministic out of the
box** at the cost of the unified kernel's Blackhole FFN speedup, pending the kernel-level
race fix (still open — see compute-kernel section; needs Trisc DPRINT/NoC tracing).

Fix attempts on the unified kernel that did NOT work (kernels reverted to pristine):
sender `flushed`→`barrier` (mcast writes are non-posted, so barrier already waits —
data lands before valid), max-sync reader (8 barriers), `dst_full_sync_en=true`. These
narrow the defect to the compute kernel `fused_swiglu` (single-core compute producing
non-deterministic output from deterministic inputs).

## FINAL SUMMARY (what is proven)

**Root cause:** the **unified fused routed-expert FFN kernel**
(`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/`)
produces **different output for bit-identical input across separate processes**.
This is the DeepSeek-V3 prefill non-determinism (different first token run-to-run).

**Chain of proof (all on this branch, bit-exact fingerprints):**
1. Within one process the model is fully deterministic (iter0..4 bit-identical) ⇒
   not iter-to-iter leaked state; the issue is across-process.
2. At 5L/1024 the whole model is deterministic across processes (combine/dispatch
   garbage present but masked) ⇒ needs scale.
3. At 5L/25600, first diverging layer = layer_3 (first MoE). GLOBAL fingerprints:
   `moe_input`, gate scores/indices, `shared_output`, `dispatched_buf` (FFN INPUT) =
   bit-identical; `expert_outputs` (FFN OUTPUT) and `routed_output` = DIFFER
   (finite, nonfinite=0). Same input → different output = non-deterministic FFN.
4. A/B `TT_REXPERT_FORCE_NAIVE=1` (per-expert extract→`routed_expert_ffn`→insert,
   standard matmuls): `expert_outputs`, `routed_output`, `layer_3`, `lm_head` all
   bit-identical across processes ⇒ the NAIVE routed FFN is deterministic; only the
   UNIFIED fused kernel is not.

**Ruled OUT as the cause** (each measured): KV-cache state, gate/routing
(indices identical), dispatch buffer (FFN input identical), shared expert (GLOBAL
identical), combine `init_zeros=False` garbage / H1 (present + harmlessly masked in
BOTH unified and naive), dispatch-buffer capacity overflow (12356 ≪ 204800).

**Fix attempt tried & ruled out:** replacing the unified reader's pre-valid-semaphore
`noc_async_writes_flushed()` with `noc_async_write_barrier()` (all 4 mcast sites) did
NOT restore determinism (kernel confirmed JIT-recompiled). So the race is **not**
sender-side write-completion ordering — it is deeper (NoC-VC ordering of the data
mcast vs the posted valid-semaphore atomic, a receiver-side visibility gap, or the
concurrent dual-sender / activated-L1-mcast optimization). The drift is finite (no
NaN), in valid rows, and timing-dependent — characteristic of a multicast/L1
visibility race, not an accumulation-order or uninitialized-NaN bug.

## RECOMMENDED FIX
- **Immediate / low-risk (restores determinism now):** route the routed expert
  through the deterministic naive path — set `TT_REXPERT_FORCE_NAIVE=1`, or flip the
  `use_unified` default in `tt_routed_expert.py`. Cost: loses the unified kernel's
  Blackhole perf gain. Verified deterministic (lm_head bit-identical) at 5L/25600.
- **Proper fix (kernel owner):** repair the multicast/L1-visibility race inside the
  unified FFN kernel. Reproduce + verify with the 5L/25600 GLOBAL `routed_output`
  fingerprint A/B (kernels are JIT — no full rebuild). Likely areas, in priority:
  (a) NoC/VC used by the activated-L1 mcast (phase 4) vs its `flushed`/valid-sem
  (the comment says it runs on "NoC 1" while the flush is default-NoC);
  (b) the receiver-side guarantee that mcast data is L1-visible before the valid
  semaphore is observed (posted-atomic vs posted-write ordering);
  (c) the concurrent dual-sender "ack both upfront" optimization.

## Max-sync reader experiment (reader exonerated)
Applied maximal synchronization to the unified **reader**: replaced all 4
pre-valid-sem `noc_async_writes_flushed()` with `noc_async_write_barrier()` AND
added `noc_async_atomic_barrier()` after every valid-semaphore multicast (8 barriers,
JIT-recompiled, confirmed). Result at 5L/25600 across two processes: `routed_output`
GLOBAL **still DIFFERS** (dnorm 3.45e-2, nonfinite=0). So the non-determinism is
**NOT** the reader's explicit mcast synchronization. Two possibilities remain:
1. **compute (`fused_swiglu`, PACKER_L1_ACC matmul) or writer** is the source; or
2. the **multicast data write is posted** (no ack), so `noc_async_write_barrier()` is
   a no-op for it and the data-vs-valid ordering relies on same-NoC delivery, which
   the concurrent dual-sender / loopback optimization may violate — i.e. a reader race
   that *these particular barriers cannot fix* (would need non-posted mcast writes or a
   receiver-side data fence).

Next experiments (when device free; all JIT, ~3 min/run, 5L/25600 GLOBAL A/B oracle):
- Probe `expert_outputs` masked to valid rows to confirm FFN valid output drifts
  (already implied: `routed_output` GLOBAL drifts).
- Serialize the dual-sender path; replace the activated loopback mcast with local
  copy + plain mcast (tests possibility 2 / canonical-deviation).
- If reader variants all fail → instrument/inspect `fused_swiglu` PACKER_L1_ACC and the
  writer (possibility 1); needs DPRINT/NoC tracing — kernel-owner tooling.

## CB L1 allocation overlap — ruled out
Checked the program factory CB allocations: all CBs use distinct indices
(`c_0`..`c_13`) created via plain `CreateCircularBuffer`, so the tt-metal L1
allocator places them non-overlapping. The compute partials CBs (`c_7`/`c_8`/`c_13`)
do not overlap the reader-written input CBs (`c_0`..`c_3`, `c_12`). So a structural
CB-overlap race is not the cause.

## Compute packer-reconfig sequencing — checked, correct
`matmul_phase_fused_gu` places `llk_pack_reconfig_l1_acc(1)` only AFTER block 0's
full subblock loop (both gate and up packs), so block-0 packs correctly overwrite and
blocks 1+ accumulate — matches the canonical matmul kernel. Not the bug. So within the
compute kernel the defect is a subtler LLK/hardware hazard (SFPU silu pass, dst-register
management across gate→up→copy phases, or a Trisc pipeline hazard) — needs Trisc DPRINT
of intermediate tiles across two runs to pin.

## Reader DEFINITIVELY exonerated → the compute kernel is the source
`noc_async_write_multicast` (and the loopback variant) default to **`posted=false`**
(non-posted, `dataflow_api.h`) — multicast writes are acked, so the max-sync
`noc_async_write_barrier()` genuinely waited for the data to LAND at receivers before
the valid semaphore was set. Max-sync still didn't fix it ⇒ the receivers got correct,
fully-delivered input data, yet the FFN output is non-deterministic. Combined with the
writer being trivially correct (writes unique per-core valid rows, barriered), the
non-determinism is in the **compute kernel `fused_swiglu`** — a timing-dependent
hazard that turns deterministic inputs into non-deterministic output. Most likely:
a packer hazard around `llk_pack_reconfig_l1_acc(0/1)` not synchronized with the
adjacent pack, the shared-packer-state dual-partials fused gate/up matmul, or an
SFPU/dst-register management hazard. Pinning the exact line needs Trisc DPRINT of
intermediate tiles across two runs (kernel-owner tooling).

## Hypotheses eliminated (testable ones) — what remains needs hardware tracing
Eliminated by direct measurement/inspection: KV-cache, gate/routing, dispatch,
shared expert, combine `init_zeros`/H1, capacity overflow, reader mcast
synchronization (max-sync), CB L1 overlap, matmul precision config (deterministic
given inputs). What remains — a finite, timing-dependent race producing
non-deterministic FFN output from deterministic inputs — is most likely (a) the
compute kernel `fused_swiglu` (PACKER_L1_ACC RMW / dual-partials / SFPU silu) or the
writer, or (b) a posted-multicast NoC-ordering issue in the concurrent dual-sender /
activated-loopback path that explicit barriers can't fix. Pinning it requires
**per-core DPRINT or NoC tracing** comparing intermediate values across two runs
(e.g. checksum `cb_out`/`cb_activated` per core) — kernel-owner tooling. The 5L/25600
GLOBAL `routed_output` A/B remains the determinism oracle for any candidate fix.

## Comparison vs the canonical (known-deterministic) matmul mcast
The standard tt-metal block-sharded matmul in0 sender
(`ttnn/.../matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`)
does, per block, simply: `async_write_multicast(data)` then `receiver_sem.set_multicast(VALID)`
on the **same NoC, with no flush/barrier between** — correctness comes from same-NoC
write ordering, and there is a **single** mcast sender per block.

The unified FFN reader deviates structurally in ways that are the most likely race sites:
1. **Concurrent dual-sender "ack-both-upfront" optimization** — receivers ack BOTH the
   in0 (M-row) and in1 (N-col) senders before either starts, then both senders mcast in
   parallel. Two concurrent mcasts into different CBs on overlapping receiver cores.
2. **Activated-L1 loopback mcast** (phase 4) — `noc_async_write_multicast_loopback_src`
   writes the activated tiles to all M-row cores *including the sender's own L1*, then a
   loopback valid-sem. Loopback + concurrent in1_down mcast on another NoC.

These are absent from the canonical pattern. The fact that swapping `flushed`→`barrier`
on the sender did nothing is consistent with the race being in this added concurrency /
loopback rather than in simple data→valid ordering. **Recommended for the kernel owner:**
first try serializing the dual-sender path (do in0 mcast fully, then in1) and/or replacing
the activated loopback mcast with a local copy + plain mcast, testing each via the
5L/25600 GLOBAL `routed_output` A/B.

## How to reproduce / verify (fast)
```
TT_DS_ND_DEBUG=1 [TT_DS_ND_DEBUG_LAYER=-1] [TT_REXPERT_FORCE_NAIVE=1] \
  <env...> pytest test_prefill_transformer.py -xvs \
  -k "mesh-8x4 and smoke and pretrained and 5_layers and iter1 and longbook_qa_eng and 25600 and fp32 and right_pad and balanced"
```
Run twice (two processes); compare with `nd_tools/nd_compare_runs.py <logA> <logB> --iter 0`.
`routed_output` GLOBAL bit-identical across the two ⇒ deterministic. The 5L/25600
config reproduces in ~3 min/run; 1024 tokens does NOT reproduce (too little scale).

### Status of experiments
- [x] Within-process drift — measured: fully deterministic (iter0..4 bit-identical).
- [x] Across-process first-diverging stage — layer_3 → routed-expert FFN output.
- [x] A/B unified vs naive — unified non-deterministic, naive deterministic.
- [x] A/B combine `init_zeros` relevance — garbage present+masked in both; not the cause.
- [x] Kernel fix attempt: sender flush→barrier — did NOT fix (race is deeper).
- [~] Kernel fix attempt: max-sync reader — could not complete (device re-taken by a
      second colleague soak); the reader/compute/writer were reverted to pristine.
- [ ] Proper unified-kernel fix — handed off to kernel owner (see RECOMMENDED FIX).

### Deliverables on this branch
- Instrumentation: `utils/nd_debug.py` + test hooks (`TT_DS_ND_DEBUG`,
  `TT_DS_ND_DEBUG_LAYER`, `TT_DS_ND_FULL_INTERMEDIATES`, `TT_DS_ND_COMBINE_INIT_ZEROS`).
- Test knobs: `TT_REXPERT_FORCE_NAIVE` (python), `TT_REXPERT_FORCE_DEFAULT` (C++).
- Tooling: `nd_tools/nd_compare_runs.py`, `nd_tools/run_nd.sh`.
- All device kernels left UNMODIFIED (fix attempts reverted; they did not work).

> Device note: the shared 32-chip galaxy was occupied by colleague `nostojic`'s
> 20× soaks (which `pkill -9 -f pytest` + `tt-smi -glx_reset` between iterations),
> so experiments had to be serialized into free windows. The kernel-fix iteration
> is unfinished only because the device was re-taken, not because of a dead end —
> it can resume via the JIT A/B in "How to reproduce / verify".
