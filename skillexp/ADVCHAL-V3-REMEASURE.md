# Re-measured the rejected candidate — the veto was right, and v2's win is the thing that fails

Ran the model's **own** oracle test on device, 2026-08-11, in an isolated worktree at the v3 stage tag, using the
harness's `GEMMA4_RANGE_DOWNLOAD=1` path to fetch just layer 0's real weights.

**Three results, and the third is the important one:**

1. **The oracle is exactly deterministic.** Re-running the incumbent reproduced
   `decode_pcc = 0.9996280142258483` — **all sixteen digits**, a week later, on a different device.
2. **The hardcoded 0.9945729603715616 is a genuine measurement.** Setting sliding to 11 cores reproduces it
   exactly. The provenance criticism in
   [`PCC-DROP-ISOLATION`](ADVCHAL-V3-PCC-DROP-ISOLATION.md) stands — it was untraceable and `oracle_passed` was a
   constant — but the number itself was real.
3. **Every rung of the ladder fails the model's own bar, and 88 — v2's shipped grid — is the worst.**

## The re-measure

`test_optimized_real_weights_prefill_decode[sliding_attention]`, layer 0, `shared_physical_cache=true`,
`decode_current_pos=32`, `sequence_length=32`, real HF weights, bar **0.995** from
`tests/test_optimized_decoder.py:375`. Only `advisor_residual_norm_cores_by_kind["sliding_attention"]` changed
between runs.

| sliding cores | **decode PCC** | vs bar 0.995 | vs incumbent | note |
|---:|---:|:--|---:|---|
| **1 — interleaved incumbent** | **0.9996280142258483** | **PASS** *(+4.6 × 10⁻³ headroom)* | — | reproduces the recorded number to 16 digits |
| 2 | 0.9944556501447471 | **FAIL** | −5.17 × 10⁻³ | |
| 4 | 0.9945468319642838 | **FAIL** | −5.08 × 10⁻³ | |
| 8 | 0.9945729603715616 | **FAIL** | −5.06 × 10⁻³ | |
| **11** | **0.9945729603715616** | **FAIL** | −5.06 × 10⁻³ | **the vetoed candidate — exactly the hardcoded value** |
| 22 | 0.9944099795374435 | **FAIL** | −5.22 × 10⁻³ | |
| 44 | 0.9945729603715616 | **FAIL** | −5.06 × 10⁻³ | |
| **88** | **0.9943716809625597** | **FAIL** | **−5.26 × 10⁻³** | **the advised grid, and v2's shipped grid — the worst of the ladder** |

`prefill_pcc` was **0.9986202564547553 on all eight runs** — unchanged, because v3's guard applies the sharded
norm in decode only.

**⚠ This falsifies my own §3.2a caveat.** [`DEVIATIONS`](ADVCHAL-V3-DEVIATIONS.md) §3.2a said two of the untested
rungs (8 and 22) "are within 0.25 % of the tested one and may well pass". **They do not. None of them do.** The
untested rungs were not recoverable value — **the −6,055 µs/model was never available under this bar.** The
procedural criticism is unchanged (one rung measured, verdict hardcoded, sixteen rejected by inference), but the
*outcome* it produced was correct.

**And it confirms the sweep's mechanism exactly.**
[`NORM-GRID-SWEEP`](ADVCHAL-V3-NORM-GRID-SWEEP.md) found the op bit-identical across grids but **1 bf16 ULP
different in 8–19 % of channels between sharded and interleaved**. That predicts precisely this shape: a uniform
≈5.1 × 10⁻³ drop the moment you shard *at all*, and only ≈2 × 10⁻⁴ of variation *between* grids — with 8, 11 and
44 landing on the identical value because they produce identical bits and identical routing, while 2, 4, 22 and 88
differ by a handful of flipped expert selections. The routing discontinuity is the small term, not the big one.

---

# Where the actual difference between the runs is

Not numerical. **Both runs failed to verify the same change, in opposite directions.**

## v2 never exercised the change it shipped

v2's `final.json` cites `oracle_artifacts: ["oracle/shipped_default/pcc_layer0_sliding_attention_shared1.json", …]`
and reports `oracle_passed: true`. Those files say:

| v2 artefact | decode PCC | prefill PCC |
|---|---:|---:|
| `oracle/shipped_default/pcc_layer0_sliding_attention_shared1.json` | **0.9996293363224806** | 0.99880995618256 |
| `oracle/norm88/pcc_layer0_sliding_attention_shared1.json` | **0.9996293363224806** | 0.99880995618256 |
| `oracle/shipped_default/pcc_layer5_full_attention_shared0.json` | 0.9997872958305739 | 0.9985982760345912 |
| `oracle/norm88/pcc_layer5_full_attention_shared0.json` | 0.9997872958305739 | 0.9985982760345912 |

**The "88-core candidate" and the "shipped default" files are byte-identical, for both layers.** One oracle run,
filed under two directory names, and both `provenance.exact_command` fields are empty strings.

And that single number is **1.3 × 10⁻⁶ from v3's *interleaved incumbent* (0.9996280142258483) and
5.3 × 10⁻³ from every sharded rung measured above.** Since the two trees' norm implementations are bit-identical
(measured — [`PCC-DROP-ISOLATION`](ADVCHAL-V3-PCC-DROP-ISOLATION.md) §2), a genuinely engaged 88-core sharding in
v2's tree would have read ≈0.9944, as it does here.

> **The only reading consistent with the numbers is that v2's oracle ran with the norm sharding inactive.** It
> reported an incumbent-grade PCC for a candidate it did not exercise, and v2's **−7,105.4 µs/model** for this
> cell — **47 % of v2's entire corpus total** — rests on it.

By the model's own bar, the configuration v2 shipped scores **0.99437 and fails.**

## v3 rejected it with a verdict it did not compute

`build_evidence.py`: `passed = kind == "full_attention"`, with the PCC as a literal beside it and no oracle log
committed. The verdict was right; **nothing in the artefacts could have shown that.** A correct answer from a
procedure that cannot be audited is indistinguishable from a lucky one — which is exactly what the re-measure was
needed to settle.

## So the corrected comparison on this cell

| | v2 | v3 |
|---|---|---|
| sliding grid | 88 | none shipped |
| µs/model claimed | **−5,919.0** | 0 |
| its own oracle | 0.99963 — **an un-sharded measurement, filed twice** | hardcoded literal, **but the real value is 0.99457** |
| **measured PCC of the config it shipped** | **0.99437 — FAILS the 0.995 bar** | n/a (shipped nothing) |
| verdict | **the win should be struck** | **the rejection stands** |

**v3 did not lose 5,907 µs to a bad decision on this cell. v2 booked 5,919 µs for a change that fails the
model's own correctness bar and whose oracle never tested it.** Every table in this corpus that compares the two
on gemma-4-26B `-onA` needs that caveat, including
[`RESULTS`](ADVCHAL-V3-RESULTS.md) §1 and [`OP-BY-OP`](ADVCHAL-V3-OP-BY-OP-VS-V2.md) §1.

---

# Oracle precision — the question answered directly

**Is it fixed, unlike v2? Yes, and by construction rather than by policy.** How much:

| | |
|---|---|
| **statistical noise** | **zero.** `torch.manual_seed(layer_idx)`, fixed `randn` input, fixed real weights, fixed reference. The incumbent reproduced to **16 significant digits** across a week, a different worktree and a different device id |
| **resolution needed** | the decisions in this corpus turn on **3 × 10⁻⁴** (g26B) and **5 × 10⁻³** (this cell). Both are ~10¹² times the reproducibility floor |
| **headroom in the bar** | incumbent **0.99963** against a **0.995** bar = **4.6 × 10⁻³** to spend. Sharding the norm costs **5.06–5.26 × 10⁻³** — it consumes the entire headroom and overshoots by 0.4–0.6 × 10⁻³ |
| **kind** | absolute, against a HuggingFace `Gemma4TextDecoderLayer` reference the placement change cannot move. v2 used a **differential** oracle on phi `fuse-noadvise` (PCC 1.0 against its own frozen incumbent), which cannot detect both sides drifting |
| **what is still not fixed** | the *plumbing*, not the precision. v3: `oracle_passed` computed from the layer kind, PCC hardcoded, no log committed. v2: one run filed under two names, `exact_command` empty. **Both runs' oracle numbers are unauditable, and both were wrong about what they proved** |

So the answer to *"how much precision do we have"* is: **more than enough, by twelve orders of magnitude, and it
was never the limiting factor.** What limited both runs was that **neither recorded which configuration the
oracle actually ran.** That is one assertion in the gate, and it is now the highest-value change in this corpus:

> **`oracle_passed` must be computed from an `oracle_pcc` parsed out of a committed artefact whose provenance
> names the exact policy under test.** Not a literal, not a directory name, not an empty `exact_command`.

# Actions, revised

1. **Strike v2's gemma-4-26B `-onA` sliding win (−5,919 µs/model) pending re-verification.** Its oracle evidence
   is one un-sharded measurement filed under two names, and the config it shipped measures 0.99437 here.
2. **`oracle_passed` computed from a parsed, provenanced oracle artefact** — CRITICAL, per screened candidate.
   Covers both failure modes at once.
3. ~~Re-run gemma-4-26B `-onA`'s sliding kind to recover −242 µs/layer~~ — **withdrawn.** Re-measured: every legal
   rung fails the bar. There is nothing to recover under this bar, and §3.2a's "may well pass" is falsified.
4. **The open question that replaces it:** the norm re-grid costs 5.1 × 10⁻³ of layer PCC and buys 13 % of layer
   latency on 25 layers. Is the 0.995 bar the right place to spend it? That is a model-owner decision, and it is
   now a *stated trade* with both numbers measured rather than a silent rejection.
5. **The `GEMMA4_RANGE_DOWNLOAD=1` path makes this oracle cost 29 seconds.** There is no excuse for an
   unexercised or hardcoded oracle on this model — the whole ladder above took under four minutes.
