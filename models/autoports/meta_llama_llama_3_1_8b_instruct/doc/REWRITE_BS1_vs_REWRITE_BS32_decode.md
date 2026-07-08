# Rewrite-bs1 vs Rewrite-bs32 — Decode Optimization Differences

Two `forge-functional-decoder` → `optimize` runs over the **same** tt-forge emit for
`meta-llama/Llama-3.1-8B-Instruct`, differing only in the version of the
`forge-functional-decoder` skill used.

- **bs1** = the first rewrite (branch `mvasiljevic/agentic-research-llama-with-forge-fd-skill`).
  Skill retargeted the module to **batch 1**.
- **bs32** = the rerun (branch `mvasiljevic/llama-bs32-rerun`). Updated skill
  **preserves the emitted batch 32** in the functional module.

Both ran only pipeline stages 01–02 (functional decoder + optimized decoder), so both
are a single bare decoder layer with no embedding / LM head / 32-layer stack.

| | bs1 | bs32 |
| --- | --- | --- |
| `forge-functional-decoder` skill | pre-fix (retargets batch 1) | batch-preserving |
| Functional layer batch | 1 | **32** (default + tested; real-weight PCC 0.99999805) |
| Optimize perf target | batch 1 | **batch 1** (unchanged) |
| Final traced decode | ≈1.283–1.289 ms | **1.059 ms** |
| Decode device ops | 37 | 38 |

---

## 1. What is identical

The two runs used the **same optimization method and the same core recipe** — the
difference is not in *how* they optimized:

- **Method:** `tt-perf-report`-driven, one-shot A/B sweeps, `$stage-review` gating.
- **Perf target:** **batch-1 single-user latency** in *both* — driven by
  `optimize/SKILL.md` line 14, which neither run changed.
- **Kept levers (identical):** BFP4/BFP4 weights + LoFi, packed decode QKV, separate
  gate/up, BF8_B paged KV cache, DRAM-sharded `down_proj` (`in0_block_w=14`), tuned
  prefill K/V geometry.
- **Rejected (identical):** HiFi2 fidelity, BF16 cache, L1 input movement, packed
  gate/up, separate decode QKV.

So the optimization *philosophy* — attack decode's weight-bandwidth bottleneck with
BFP4, keep QKV packed/interleaved, tune at batch 1 — is the same in both.

---

## 2. What differs

### 2a. Batch of the functional module (the intended difference)

The skill change did what it was meant to: bs32's functional layer runs at the emitted
batch 32 (`from_state_dict(..., batch=32)`, input `[1, 32, seq, 4096]`, tested at batch
32), where bs1 collapsed it to batch 1.

**But this did not change the optimize stage** — both still tune and measure decode at
batch 1 (`optimize/SKILL.md` line 14). bs32's decode residual is sharded for a `[32,128]`
shard, i.e. the 32-row *tile padding of one token*, not 32 real users. So the emitted
batch survives into the **module** but not into the **optimization target** in either
run.

### 2b. Two extra decode optimizations in bs32 (the perf difference)

bs32's `$stage-review` flagged DRAM-interleaved decode ops that bs1's review never
caught, driving two optimizations that **do not exist in bs1's code at all**:

| Optimization | bs1 | bs32 |
| --- | --- | --- |
| Sharded residual + post-attn RMSNorm (32-core) | not done — left DRAM-interleaved | **done** — post-attn RMSNorm 94 μs → 10 μs (≈−12% decode) |
| Gate/up matmul **geometry** | only chose packed-vs-separate | **additionally** tuned 64-core `in0_block_w=4`, output subblock `1x7` → gate/up 258/270 μs → 202/201 μs |

Verified by diffing the branches: bs1's `optimized_decoder.py` has **no**
`_decode_residual_memory_config` / `_decode_residual_norm_program_config` /
`LayerNormShardedMultiCore…` helpers; bs1's work-log stage-review cleared after only
"decode signpost + gate/up projection choice + prefill K/V evidence." bs32's work-log
adds a sharded-residual/norm repair and a gate/up-geometry autofix.

### 2c. Result

| | bs1 | bs32 |
| --- | ---: | ---: |
| First correct BFP8/BFP8 decode | 1.845 ms | 1.845 ms |
| Final traced decode | ≈1.283–1.289 ms | **1.059 ms** |
| Δ from BFP8/BFP8 baseline | ≈−30% | **−42.6%** |
| Decode device ops | 37 | 38 |

---

## 3. Why it differs — and why that's *not* about batch

- The **batch-32 preservation** is the deliberate, skill-driven difference — but it only
  touched the functional module, not the optimize approach.
- The **1.059 vs 1.289 ms** gap is **stage-review variance**, not a batch effect: the
  two extra wins (sharded residual/norm, gate/up geometry) are unrelated to batch size.
  bs1's run could have found them and simply didn't — its stage-review declared
  clean-pass earlier.

In other words: had bs1's stage-review been as thorough, it would likely have reached a
near-identical ≈1.06 ms at batch 1. The batch change did not make bs32 faster; a more
thorough second review did.

---

## 4. Takeaways

- **Same recipe, same batch-1 optimize target.** The skill edit changed batch handling
  in the functional layer only.
- **bs32 is faster (1.059 vs 1.289 ms) purely because it landed two more decode
  optimizations** (sharded residual/norm + gate/up geometry) — a difference in
  stage-review thoroughness, not in method or in batch.
- **The batch problem is still half-open:** to make the *optimized/measured* decode a
  batch-32 number, the `optimize` skill's batch-1 directive must also change (see
  `REWRITE_BS32_vs_INPLACE_EMIT_decode.md` §2).

---

*Evidence: bs1 — `origin/mvasiljevic/agentic-research-llama-with-forge-fd-skill`
`optimized_decoder/{work_log,README}.md`; bs32 — this branch's
`optimized_decoder/{work_log,README}.md` and `functional_decoder/work_log.md`.
Companions: `REWRITE_vs_INPLACE_EMIT_decode.md` (bs1 vs GRAPH),
`REWRITE_BS32_vs_INPLACE_EMIT_decode.md` (bs32 vs GRAPH).*
