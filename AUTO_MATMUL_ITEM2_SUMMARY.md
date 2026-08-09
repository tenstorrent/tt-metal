# Item #2 — distributed correctness gate (+ the two accuracy notes) — done

Reviewer item #2: single-device candidates were rel-L2-checked vs the bf16 default, but **distributed
recipes were picked on timing alone** — validate distributed winners against a real reference the way
`place_weight(measure=True)` does. Plus two accuracy notes (#1 gate vs bf16 default not fp32 golden;
#2 `_estimate_l1_kb` assumes bf16). All three closed in `ttnn/ttnn/_experimental/auto_config/_selector.py`.

## What changed

### C — dtype-aware L1 estimate (accuracy note #2)
`_dtype_bytes(dtype)` (FLOAT32→4, BFLOAT8→1.0625, BFLOAT4→0.5625, INT8→1, bf16→2) feeds `_estimate_l1_kb`,
which now takes real in0/in1/out byte widths (interm stays fp32=4) instead of hard-coding bf16. Threaded from
both builder call sites, so a bf8 config is correctly seen as ~half the L1 (more configs valid) and fp32 as
double (fewer).

### A — fp32 golden for single-device (accuracy note #1)
`_fp32_reference(prepared, signature)` builds an independent **float32** `a @ b` (+bias, honoring
transpose_a/b) from the host operands read off the device — never from the device matmul result, so it can't
validate the device against itself.

**The threshold is gated RELATIVE to the default's own error, not a flat 0.05.** Against an fp32 golden every
bf16 output carries quantization error, and on large-K / low-fidelity / bf8 shapes the *default itself* is
already >0.05 off. So: `err_budget = max(default_err × 1.10, 0.05)` and a candidate is rejected iff its rel-L2
vs the golden exceeds it. The default becomes the numerical baseline, not a gate-exempt free pass; a config is
rejected only when it is genuinely **worse than the stock op**. (Reused the bf16-default reference only as a
fallback when the golden can't be built.)

### B — distributed correctness gate (item #2 core)
For a distributed signature (`_infer_distributed_plan` kind ∈ gather / reduce-scatter), **every** collective
recipe is now verified against an fp32 golden reconstructed across the mesh (`_fp32_reference_distributed` →
`_reconstruct_operand_f32` via `ConcatMeshToTensor`) **before it can win**:
- reduce-scatter outputs are concatenated back on the collective dim;
- all-gather (replicated) outputs are checked with an **all-shards-agree** test (`_replicated_shards_agree`) —
  a recipe that diverges per device is caught even if shard 0 matches.
- A recipe that is numerically wrong or throws is recorded `incorrect` and **excluded** (never cached). If the
  golden can't be built we **fail closed** (base-op fallback) rather than trust timing blindly.

**Failure taxonomy (stated honestly):** the runtime gate closes the *ran-but-numerically-wrong / catchable-
exception* class — which is exactly the Group A/B class (those recipes were *cached*, i.e. they produced a
timing). It cannot stop a hard-hang / device-assert (running it to check *is* the hang); those are handled by
the existing static pre-filtering in `_infer_distributed_plan` (e.g. the K-mismatch → `unsupported` bypass).
Principle: **static validity for un-catchable, runtime golden for numerical.**

## Tests

**Host-only** (`tests/.../test_auto_config_helpers.py`, 12 new, mocked — no device):
- fp32 golden built correctly (incl. transpose_b + bias); None for a distributed signature.
- `_rel_l2_error` size-independent; rejects NaN/Inf and shape mismatch (never a silent pass).
- **the fix-A backfire guard**: with a default 7% off the golden, a 6%-off config the old flat 0.05 gate would
  wrongly reject is **accepted** (no worse than default), a 20%-off config is still rejected; the floor still
  protects a near-exact default.
- distributed golden reconstructs full operands from mesh shards; all-shards-agree catches per-device
  divergence; a wrong / divergent / crashing distributed candidate is rejected; reduce-scatter output is
  reconstructed on the collective dim; end-to-end a wrong recipe is gated out and selection **fails closed**.

**Hardware** (`AUTO_MATMUL_ITEM2_VALIDATE.py`, N300 1×2 mesh — run on the Tracy build):
reproduces the **real GPT-OSS-20b** shapes (o_proj K=4096 reduce-scatter, qkv K=2880 all-gather) and asserts
the **two positive directions** — winner is a genuine distributed recipe AND verified vs the fp32 golden (a
base-op fallback on a should-work path is a FAILURE) — plus a negative case (mismatched shard axis → clean
bypass, no crash).

## Hardware verification (N300 Tracy build, done)

Overlaid the selector onto the installed ttnn and ran end-to-end:

- **Host suite: 67/67 pass** against the item-#2 selector. End-to-end testing found (and this
  fix closes) a real bug: `_fp32_reference`/`_fp32_reference_distributed` accessed
  `prepared.input_tensor_b` unguarded and `AttributeError`d on a minimal mock — now `getattr`-guarded
  so the golden degrades to None (falls back) instead of raising. One pre-existing test
  (`..._cache_hit_returns_fallback_without_retuning`) was updated: the item-#2 gate now excludes a
  throwing distributed candidate *at the gate* (running it), not at benchmarking, so the test asserts
  the surviving invariant (fail-closed + selection runs once + second call served from cache).
- **Single-device fix A, on device:** square/nonpow2/wide all select a real tuned `program_config`,
  **0 candidates falsely rejected** by the fp32 gate (the backfire guard, live), output rel-L2 ≈ 0.024
  vs a float32 torch golden.
- **Distributed fix B, 1×2 mesh:** the mesh-reconstructed fp32 golden matches a pure-torch full matmul
  **exactly (rel-L2 = 0.000000)**. The real GPT-OSS o_proj reduce-scatter recipe **genuinely crashes**
  on this 2-chip build (`reduce_scatter_minimal_async` TT_FATAL — the real Group-B crash); the gate
  catches it, excludes it, and selection **fails closed to the base op** (crashing recipe not cached).

**Honest coverage limit:** on this 2-chip build the distributed collectives
(`reduce_scatter_minimal_async` / `all_gather`) do not execute for any shape tried, so a "distributed
recipe *wins* and verifies" positive is not demonstrable on this hardware — which is itself why the
reject-broken-recipe / fail-closed gate is needed. The numerical-gating and all-shards-agree paths are
covered by the host-only tests. Reproduce with `python3 AUTO_MATMUL_ITEM2_VALIDATE.py`.

## Honest scope
- N300 = 2 chips → only a 1×2 mesh is validated here; T3K/TG larger meshes are a stated coverage limit.
- fp32 golden host cost: computed once per signature, reused across candidates.
- Reconstruction reuses the exact `place_weight` composer logic to avoid a new bug.
