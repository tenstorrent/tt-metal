# Datatype sweep — work log

Stage 8 of the Muse-Glimmer-30B bringup: choose the fastest weight / activation /
CCL / KV-cache / compute-fidelity policy that still meets a full-model accuracy bar,
starting from the completed optimized full model at `c593334a898`. Skills:
`$datatype-sweep`, `$tt-device-usage`, plus `$qualitative-check` and `$stage-review`.

The headline is in [`README.md`](README.md). This log is the order things were done
in, what was measured, and what was rejected and why.

## 1. What the previous stage handed over, and what was missing

The optimized full model ships a policy that is already most of the way down the
precision ladder: BF16 activations and residual, **BFP8** attention weights, **BFP4**
MLP weights, **BFP8** KV cache, **BFP8** prefill CCL payload, BF16 decode CCL payload,
**LoFi** on every projection, and a **BFP4/LoFi** LM head. It states plainly that "no
broad datatype frontier search was run here; `$datatype-sweep` owns Pareto selection".

Two things had to be built before a sweep could be honest about it.

**The policy could not express half the candidate matrix.** `PrecisionPolicy` carried
one `decode_math_fidelity` and one `prefill_math_fidelity` for *all six* projection
roles, and no notion of a layer exception. So the two comparisons `$datatype-sweep`
requires by name — BFP8+LoFi against BFP8+HiFi2 on the dominant decode projections, and
a BFP4+LoFi arm for every material BFP4 group — could only have been run by moving the
MLP and the attention projections together, which conflates them. And the restore ladder's
"keep the first and last layer at the safer dtype" step had no representation at all.

**There was no artifact for a selected policy to live in.** Every precision field was a
module constant or a default argument: `DEFAULT_PRECISION` in `tt/optimized_decoder.py`,
`LM_HEAD_*` in `tt/model.py`, `DEFAULT_*_CCL_DTYPE` in `tt/multichip_decoder.py`. A
selection written to JSON would have been a document about the code rather than an input
to it, which is exactly the failure mode the skill calls out: *"If a field appears in
`selected_precision_config.json` but the code path ignores it or hard-codes a different
value, the sweep is incomplete."*

## 2. Plumbing, before any measurement

### 2.1 Per-role compute fidelity

`PrecisionPolicy` gained `decode_math_fidelity_by_role` and
`prefill_math_fidelity_by_role`, both `((role, fidelity), ...)` tuples so the dataclass
stays frozen and hashable (it is part of the generator cache key). `decode_fidelity(role)`
and `prefill_fidelity(role)` fall back to the scalar default, so a policy that names no
role behaves exactly as before.

The decoder builds one compute-kernel config **per role** —
`decode_compute_kernel_config_by_role` / `prefill_compute_kernel_config_by_role` — and
`_decode_projection` / `_prefill_projection` index it by the role they were already
passed, in both `tt/optimized_decoder.py` (single chip) and `tt/multichip_decoder.py`
(the shipped TP=4 path). The old scalar `decode_compute_kernel_config` is still built
and is what non-projection ops use, so nothing else moved.

### 2.2 Layer exceptions

`PrecisionPolicy.layer_exceptions` is `((layer_indices, ((field, value), ...)), ...)`.
`for_layer(idx)` returns `self` when no exception names that index — so the common case
allocates nothing and the cache key does not churn — and otherwise returns a `replace`d
copy with `layer_exceptions` cleared, so applying it twice cannot re-apply.
`MuseGlimmerModel.from_pretrained` calls `precision.for_layer(layer_idx)` per layer.

The geometry tables key on `(role, weight dtype)` already, so a per-layer dtype change
picks up that dtype's measured `(cores, in0_block_w)` without any further work.

### 2.3 The artifact, and making it required

`tt/precision_config.py` defines the schema and the two directions:
`config_from_policy` writes one, `build_kwargs_from_config` turns one into the exact
keyword arguments `build_generator` passes down.

`build_generator` now **reads
`doc/datatype_sweep/selected_precision_config.json` on every build**. It is a required
input, not an optional override: a missing or malformed file raises with a message that
says why, rather than falling back to a module constant — a fallback is precisely how a
selected policy stops being the one that runs. A caller that passes `lm_head_dtype=` or
`decoder_kwargs={"kv_cache_dtype": ...}` overrides that one field and the generator
records `precision_config_id = "<id>+override(<fields>)"`, so an evidence file can never
claim "selected policy" about an overridden build.

Three kinds of field, consumed three different ways, and all three are checked:

| kind | fields | how it is consumed |
| --- | --- | --- |
| plumbed | weight dtypes per group, layer exceptions, per-role decode/prefill fidelity, activation dtype, KV-cache dtype, CCL payload dtypes, LM-head dtype/fidelity/fp32-acc/output dtype **and geometry** | constructor arguments |
| structural | embedding table dtype, norm weight dtype, residual dtype | **validated**; a wrong value raises. There is no knob (the embedding table must be BF16 ROW\_MAJOR for `ttnn.embedding`; the residual stream *is* the activation tensor) |
| provenance | measured numbers, run ids | not consumed |

The LM-head **geometry** is in the artifact because it has to be. The head's static
circular-buffer budget is dtype-scaled, and the shipped `in0_block_w=2` overflows L1 at
BFP8 — see §3.2. A dtype field without the geometry that makes it legal is not a
configuration.

### 2.4 Reading the policy back off the build

`OptimizedDecoder.precision_report()` reports, per role, the **packed weight tensor's**
dtype and the **compute-kernel config's** `math_fidelity` — i.e. what the matmul was
handed, not what was asked for. `MultichipDecoder` adds the collective payload dtypes by
calling `_row_parallel_dtype` rather than restating it. `MuseGlimmerModel.precision_report()`
folds the 52 per-layer reports to the distinct ones with the layer indices that produced
each (so a first/last-layer exception shows as three groups, not 52 copies) and adds the
embedding, terminal norms and LM head. `capability_report()` carries the whole thing plus
the logits/sampling dtype assumptions.

`precision_config.check_propagation(requested, realised)` diffs the two and returns the
mismatches. `sweep.py` **refuses to record a measurement** when that list is non-empty:
a candidate whose policy did not propagate is not a measurement. The check is only worth
running if it can fail, so `test_check_propagation_catches_a_field_the_build_ignored`
feeds it a realised report where the build ignored the BFP4 request and asserts it
reports exactly the three attention roles.

### 2.5 Tests

`tests/test_precision_config.py`, **22 cases, all passing**. Seventeen are host-only —
schema, every rejection, `for_layer` idempotence, per-role fidelity isolation, and the
propagation check's own ability to fail. Five are device cases on the reduced two-layer
build: the shipped artifact is the policy the build runs; the **readiness factory path**
(`tt/generator.py` loaded by path under a synthetic module name, as the runners and the
vLLM adapter load it) builds the selected config; a per-role HiFi2 request reaches only
those roles *and only the decode phase*; a layer exception reaches only the layer it names
(layer 0 excepted, layer 3 not, both in the same build); and a caller override is recorded
in `precision_config_id` rather than silently applied.

## 3. The smoketest, before the 52-layer sweep

`$datatype-sweep` is explicit that a dtype change can need a semantic code change and
that the place to find that out is a one-decoder smoketest, not a Pareto sweep. So every
candidate artifact was first run on a **two-layer** build (one sliding, one
full-attention, real weights, real terminal path):
`doc/datatype_sweep/bench/smoketest.py`, output
[`smoketest.json`](smoketest.json), log [`logs/smoketest.log`](logs/smoketest.log).

Each candidate prefills a **200-token** prompt — not a multiple of the 32-row tile, the
64-token page or the 8192-token chunk, so `paged_fill_cache` runs at the candidate's
cache dtype on the pad/slice path — then takes six traced decode steps, so
`paged_update_cache` and the paged decode SDPA run against that cache.

### 3.1 Two findings that changed the candidate matrix

**The prompt had to be a real one.** The first pass used random token ids. On a
two-layer stack those produce near-degenerate logits: four candidates with completely
different perturbations returned *identical* six-token sequences, because every
perturbation fell into the same attractor. The smoketest now takes the first 200 tokens
of the AIME24 chat reference's own prompt, and the token comparison became informative
immediately. First pass kept at
[`logs/smoketest_first_pass.log`](logs/smoketest_first_pass.log).

**The BFP8 LM head does not fit L1 at the shipped geometry.**

```
TT_THROW: Statically allocated circular buffers on core range [0-0 - 10-9] grow to
1821824 B which is beyond max L1 size of 1572864 B
```

from `ttnn::prim::matmul` during trace capture. `per_core_K = 208/52 = 4`, so
`in0_block_w` may be 1, 2 or 4, and only 1 fits at BFP8. That is a geometry constraint on
the candidate, not a reason to drop it: `c10` carries `in0_block_w=1`, and `c11` was
added as its **control** — the shipped BFP4 dtype at the same `in0_block_w=1` — so the
BFP8 arm's result is not confounded with its geometry.

### 3.2 What the smoketest established

Every artifact builds, propagates with **zero** mismatches, prefills a non-aligned prompt and
takes traced decode steps. (The first pass of this smoketest covered the twelve candidates
that existed when it ran and was not re-run when the matrix grew to twenty-one — round 1 of
the stage review caught that, the smoketest was re-run over the whole matrix, and the figure
gate now asserts that the smoketested set *is* the candidate set.) In particular the **BFP4 KV cache** needs no
semantic change: `paged_fill_cache` already typecasts K/V to the cache dtype
(`tt/optimized_decoder.py`), and `paged_update_cache` owns the repack into the cache
dtype itself, so K/V stay BF16 on the way in whatever the cache is. Both were written
that way by earlier stages; the smoketest is what proves it holds at BFP4.

One result is worth naming because it shapes the fidelity conclusions:
**at BFP4 weights, LoFi and HiFi2 are bit-identical.** `c01` and `c02` differ only in the
attention projections' math fidelity and returned the same six tokens and the same
prefill PCC to every digit; likewise `c04` against `c00` for the MLP. At BFP8 they are
not: `c03` moves the prefill PCC against `c00` from 1.000 to 0.99929. This is the
mechanism the canonical guidance describes from the other side — *"Use HiFi2 for BFP8
weights to drop the least-significant bit of a BF16 @ BFP8 matmul... Use LoFi for BFP4
weights"* (`tech_reports/LLMs/llms.md`) — and it means the BFP4 fidelity comparisons the
skill requires are answered exactly rather than statistically.

## 4. The sweep, in two passes

### 4.1 Pass 1: five rounds, twelve candidates

`bench/sweep.py`, one process and one 52-layer build per candidate
(`bench/run_sweep.sh`). Each candidate's artifact is installed as
`tt.precision_config.SELECTED_PRECISION_CONFIG_PATH` for the life of the process, so
`build_generator(model_dir, mesh_device)` **with no knobs** — which is what the readiness
runners call — constructs it. That is not a convenience: the runners load `tt/generator.py`
by path and build their own generator, so a `precision_config=` argument passed to one call
would not have reached them, and the accuracy numbers would have come from whatever
`selected_precision_config.json` happened to hold. The generator cache then hands the runners
the build made in the driver, so the 52-layer stack is packed once.

**Two defects the first pass exposed, both fixed before the numbers were kept.**

*`c01` was measured twice over.* `tt/generator.py` was edited (a cache-key field) while `c01`
was mid-run. The driver process had imported the old module; the readiness runner loads the
file **by path** and got the new one, so the two disagreed on the cache key, the runner built
a *second* 52-layer model, and the counters read from the driver's generator showed
`trace_replays = 0` for a teacher-forcing run that happened on the other one. Nothing about
this is subtle in hindsight and it is the reason the second pass re-ran every candidate on a
frozen tree. The failure was visible rather than silent, because `sweep.py` records the
trace-replay counter per round and the gate reads it.

*Five rounds could not order the leading candidates.* `c00` came out at 38.060 and `c01` at
38.046 — inverted against the traced logits-only cross-check, which had them at 44.139 and
44.382. Teacher forcing spends ~3.7 ms of its ~26 ms step on host restaging, sampling and
token readback, none of which a dtype change touches, so a device-side win is diluted; and
round 0 of every candidate is a warm-up 2–3 % low. At five rounds the ~0.4 % spread is the
same size as the effect.

### 4.2 Pass 2: eleven rounds, uniform, the whole matrix

Every candidate re-run at `ROUNDS=11` on a frozen tree, so the comparison is like-for-like.
The spreads collapse to ~0.15 % and the order stabilises: `c00` 38.037 [37.986–38.159]
against `c01` 38.204 [38.145–38.266] — non-overlapping, and in the order the cross-check
independently gives. Pass 1 is kept at `runs_pass1_rounds5/`; every number in the README is
from `runs/`.

Four candidates were added between the passes because pass 1 pointed at them:

* **`c14`** — BFP4 attention weights *plus* a BFP8 decode CCL payload, with the BFP8 KV cache
  kept. Pass 1 showed the three fast switches to be BFP4 attention (+0.55 % on the
  cross-check), BFP8 decode CCL (+0.46 %), and BFP4 KV (+0.0 %, and the only one that costs
  top-1). `c08` had stacked all three; nothing in the matrix had stacked the two that are free.
  It is the selection.
* **`c15`** — `c14` with layers 0 and 51 restored to BFP8, so the ladder has a rung rather
  than a cliff.
* **`c16`/`c17`** — BFP4 decode CCL payload, alone and stacked. Untried, and the obvious next
  rung below `c06`.
* **`c18`/`c19`** — the adapted retry for `c13`'s op-contract failure, and its layout control.

A twenty-first was added after round 1 of the stage review named it as the obvious legal
candidate the matrix did not contain:

* **`c20`** — the **layer-scoped** KV cache: BFP4 on the 39 sliding-window layers, BFP8 on the
  13 full-attention ones, on top of `c14`. The per-layer A/B had already shown the two effects
  separate — the cache's decode win lives entirely in the 13 full-attention layers (0.5149 →
  0.5027 ms at context 131071) while its capacity saving is per layer and therefore mostly in
  the 39 — so a policy that takes the capacity without the layers whose reads the accuracy loss
  most plausibly comes from is legal, expressible in the artifact schema, and had not been
  measured.

## 5. What was rejected, and on what

### 5.1 Compute fidelity: HiFi2 is expensive and, at BFP4, free of any benefit

| group | dtype | LoFi | HiFi2 | delta | numerically |
| --- | --- | --- | --- | --- | --- |
| attention | BFP4 | `c01` 44.382 | `c02` 41.393 | **−6.7 %** | bit-identical |
| MLP | BFP4 | `c00` 44.139 | `c04` 32.453 | **−26.5 %** | bit-identical |
| attention | BFP8 | `c00` 44.139 | `c03` 41.124 | **−6.8 %** | prefill PCC 1.000 → 0.99929 |

The bit-identity at BFP4 is from the two-layer smoketest, where `c01`/`c02` and `c00`/`c04`
return the same six greedy tokens and the same prefill PCC to every digit. It is the expected
consequence of a 4-bit mantissa fitting inside what LoFi already multiplies, and it settles
the BFP4 fidelity question exactly rather than statistically. BFP8 does not fit, so HiFi2
does change the numbers — by an amount that moves no full-model accuracy digit, for 6.8 %.

The −26.5 % on the MLP is worth naming separately: the MLP is three of the six projections
and the two widest, so its fidelity is the single most expensive knob in the policy.

### 5.2 The LM head: BFP4 is right, and the geometry is part of the dtype

`c10` restores the head to BFP8 and measures **43.302** against the baseline's 44.139
(−1.9 %) with top-1 unchanged at 0.990 — so the previous stage's BFP4 choice is priced rather
than assumed. `c11` is its control: the shipped BFP4 dtype at `c10`'s `in0_block_w=1`, which
measures 43.355. So the geometry accounts for essentially all of the difference between the
baseline and `c10`, and BFP8 weights themselves cost ~0.1 %.

The head could not run BFP8 at the shipped `in0_block_w=2` at all:

```
TT_THROW: Statically allocated circular buffers on core range [0-0 - 10-9] grow to
1821824 B which is beyond max L1 size of 1572864 B
```

`per_core_K = 208/52 = 4`, so the legal widths are 1, 2 and 4 and only 1 fits at BFP8. That is
why the LM-head matmul geometry is a field of the precision artifact: a dtype without the
geometry that makes it legal is not a configuration.

### 5.3 The KV cache: measured at both ends of the context range

`c05` (BFP4 cache alone) is **44.103** against the baseline's 44.139 at the benchmark's
128–256 decode positions — worth nothing — and costs top-1 0.990 → 0.970. Rejecting it on
that alone would have been hiding the decoder stage's finding that the *same* lever is worth
10 % at context 131071 once the SDPA chunking is fixed, so `bench/layer_ab.py` measured it
there:

| context | layer kind | KV BFP8 | KV BFP4 | delta |
| --- | --- | --- | --- | --- |
| 131071 | sliding ×39 | 0.4416 ms | 0.4416 ms | 0.0 % |
| 131071 | full ×13 | 0.5149 ms | 0.5027 ms | **−2.4 %** |

Only the 13 full-attention layers read the whole cache; the sliding layers read a bounded
window whatever the context. Over the stack that is 0.159 ms of a ~22.4 ms step — **0.71 %**
at the advertised context against ~0 % at the benchmark's, for two top-1 points at both. The
selection is made on the full-model metric the skill mandates, which is measured at the
reference's own context; the long-context number is on the record so a serving stage that
wants the concurrency (15 → 28 full-context sequences) can make that trade knowingly.

### 5.4 The blocked candidates, and the adaptations tried

Five candidates produced no *decode* number, every one on an exact op contract rather than a
first API error, and every one reproduced at the two-layer smoketest as well as at 52 layers.
Three of them — `c12`, `c16`, `c17` — got through a full 100-token AIME24 **prefill** accuracy
pass first, all at 0.990 / 1.000 / 1.000, and failed at decode-trace capture; `c13` and `c18`
fail inside prefill and have no accuracy number. The distinction is not bookkeeping: it is what
makes `c12`'s un-run prefill-only adaptation a measured-accuracy proposition rather than a
speculative one.

**BFP8 activations (`c12`)** — `nlp_create_qkv_heads_decode_device_operation.cpp:41` takes
FLOAT32 or BFLOAT16 only. Reproduced first-hand rather than inherited from the decoder stage.
The payload narrowing it would have bought is available independently as `decode_ccl_dtype`,
which is `c06`, which passes and is half the selection.

**BFP4 on any collective payload (`c13`, `c16`, `c17`, `c18`)** — two norm ops reject it
depending on which the payload reaches: `c13` lands in the fractured prefill norm's
`layernorm_pre_all_gather_device_operation.cpp:44` (BFLOAT16, BFLOAT8_B or FLOAT32) and the
other three in `layernorm_device_operation.cpp:52` (FLOAT32, BFLOAT16 or BFLOAT8_B). Every collective in
this model is consumed by an RMSNorm, and no norm op in TTNN accepts BFP4. The adaptation was
run rather than argued: `c18` disables the fractured prefill norm, which moves the payload
from `layernorm_pre_all_gather` to `layernorm`, and hits the identical restriction; `c19` is
its layout control so "the fractured norm is off" is not confounded with "the payload is
BFP4", and it measures 38.098 — i.e. the layout change alone is neutral.

The remaining workaround is a typecast between the collective and the norm, and this model's
own profile prices it out. The decode step runs 104 collectives (2 per layer × 52), and
`../optimized_full_model/tracy/decode_sliding_perf_report.csv` puts a 6656-wide elementwise
op on this grid at ~5.05 µs, so 104 typecasts is ~0.5 ms of a 22.4 ms step — over 2 % —
against the ≤ 0.45 % the entire BF16 → BFP8 payload change was worth. The reason the reduced
payload is free today is that the row-parallel matmul is asked for the payload dtype
directly; a typecast hands that back with interest. **BFP8 is the floor for a collective
payload in this model.**

## 6. The selection, and why it is not the fastest row

Ranked by the mandated metric, the top of the passing set is:

| config | top-1 | teacher-forcing median | 11-round range | logits-only |
| --- | --- | --- | --- | --- |
| `c08` | **0.970** | 38.293 | 38.186–38.351 | 44.661 |
| `c09` | **0.970** | 38.269 | 38.232–38.342 | 44.421 |
| `c15` | 0.990 | 38.244 | 38.173–38.329 | 44.564 |
| **`c14`** | **0.990** | **38.227** | **38.151–38.303** | **44.578** |
| `c01` | 0.990 | 38.204 | 38.145–38.266 | 44.382 |

`c08` and `c14` differ by 0.17 % on the ranking metric and their 11-round ranges overlap
across most of their width, so the ranking metric does not separate them. `$datatype-sweep`'s
tie rule then applies — *"prefer the simpler and safer one"* — and `c08` is not the safer one:
its BFP4 cache is the only change in the sweep that moves full-model accuracy, reproducibly,
on real weights, on both references and on all 11 rounds.

The rule is in `bench/analyse.py::select` and is applied by the script, not by hand. It reads:
within the set that the ranking metric does not separate from the fastest, prefer no
regression in measured full-model top-1; then take the traced logits-only cross-check when it
separates the survivors; then the simplest policy. `sweep_results.json:selected` records the
tied set, every member's round range, the cross-check values, and the sentence that decided
it, so the decision is auditable without re-running anything.

`c15` sits inside `c14`'s spread on both metrics and is one field more complex, so `c14` is
selected and `c15` is recorded as a one-file switch if a later stage finds a first/last-layer
sensitivity this reference does not.

**The per-layer PCC that would have rejected the winner.** `layer_ab.py --real-weights` puts
the selected policy at prefill PCC **0.977068** and decode PCC **0.985285** on the sliding
layer, against the baseline's 0.997450 / 0.997222. 0.977 is the number the decoder stage
declined BFP4 attention weights on. On the full model it costs nothing: top-1 0.990, top-5
1.000, top-100 1.000, on both references, with a clean qualitative suite. Deciding this on
full-model accuracy rather than layer PCC is the reason this stage exists.

## 7. Post-selection, and one infrastructure recovery

The artifact was installed as `selected_precision_config.json` with a provenance block, and
everything after that point was measured through the **default** construction path — no
precision knobs anywhere. `evidence_accuracy.json`'s `build_kwargs` is `{}` and its
`capacity.precision_config_id` is `c14-attn4-cclbfp8-kv8`.

* accuracy on both references, prefill misses, prompt shapes, sampling contract, fallback
  audit → `evidence_accuracy.json`;
* the **same warmed token-out benchmark** the optimized full model reported →
  `evidence_perf.json`: **23.078 ms/token · 43.33 t/s/u**, against 23.298 · 42.92, i.e.
  **−0.94 %**;
* the autoregressive readiness run → `evidence_autoregress.json`;
* the shared qualitative suite against the full-model stage's HF control → `qualitative/`;
* `doc/context_contract.json` rebuilt, with a new `kv_cache_dtype_capacity` block pricing
  both cache dtypes from the sweep's own capability reports;
* `tests/test_full_model.py` 59 passed, `tests/test_precision_config.py` 22 passed;
* watcher, in two processes because the two test modules each own a module-scoped `mesh`
  fixture.

**Infrastructure recovery.** The first watcher attempt failed at mesh open with
`Timed out while waiting for active ethernet core 29-25 to become active again` — the
recoverable ERISC fault `$tt-device-usage` lists. No stale process from this run held the
device. `doc/full_model/bench/tt_reset.py` returned `failures=0` and a mesh smoke opened and
closed a 1x4 mesh, so the run was retried. The precision watcher set came back
`WATCHER_CLEAN` with all four detach lines; the ten-case set tripped
`subordinate_erisc detected invalid NOC command buffer state ... fabric_erisc_router.cpp` on
acteth core 29-25 after its first test, which is the same fabric signature. A second reset
left the mesh undiscoverable entirely (`Graph specified in MGD could not fit in the discovered
physical topology`) and a third recovered it — which is exactly the "a first reset can leave
part of the mesh or Ethernet fabric missing, run the bounded sequence once more" case. The
ten-case set was then re-run and came back **10 passed, `WATCHER_CLEAN`**, with all four
attach and all four detach lines present in its own log.

Recorded as infrastructure recovery, not a model correctness or performance result: no
accuracy or performance number in this stage comes from a run that tripped, and the two
watcher artifacts are re-derived from their own logs by `check_watcher.py` rather than
asserted.

## 8. Round 1 of the stage review, and what it changed

The reviewer returned `more-work-needed` on two P2 findings and several concerns. What each
one was, and what was done:

**P2 — the smoketest covered twelve candidates, not twenty, and the README said otherwise.**
Correct, and the mechanism is the boring one: the smoketest was run when the matrix had twelve
candidates in it and was not re-run when `c12`–`c19` were added, so the selected config had
never been through it and five candidates were rejected on 52-layer op-contract failures
without the one-decoder check `$datatype-sweep` asks for first. The smoketest was re-run over
the whole matrix, and — because the failure was a *coverage* claim that a numeric figure gate
by construction cannot catch — `bench/check_reported_figures.py` gained coverage assertions:
the candidate/measured/blocked counts are re-derived and must appear in the README, every
config artifact must have a run artifact and vice versa, every candidate must appear in
`smoketest.json`, and every measured or blocked candidate must be named in the README. The
same review found six count errors elsewhere in the two documents (fourteen for sixteen
measured, six for five blocked, sixteen for fourteen in pass 1, 18+4 for 17+5 tests); all are
corrected and the first three are now gated.

**P2 — `context_contract.json` carried a stale byte figure in prose.**
`byte_budget_at_full_context.per_device_total_bytes` was correctly recomputed to 6,603,027,712,
while the `note` in the same object still said "needs **7.18 GB**/device" — the pre-sweep
number the README itself reports as superseded. The parent builder restates that total in
prose and this stage moves it, so `refresh_context_contract.py` now rewrites the figure from
the measured value and **raises** if the note does not contain a figure it recognises, rather
than leaving a contract that contradicts itself.

**Concern — the selection prose named the wrong tie-break rule.** The README said `c14` was
selected over `c15` "because it is the simpler policy"; `analyse.py` in fact picked it as the
argmax of the logits-only cross-check, by 0.03 %, which is at or below that metric's
resolution. Simplicity gives the same answer for a better reason, but the document now says
what the code did and reads the margin honestly (README limitation 3).

**Concern — `sweep_results` could not distinguish `c19` from `c00`.** `policy_summary` omitted
`decoder_overrides`, so the layout control and the baseline printed identical policy rows.
Added to both the JSON and the CSV.

**Concern — no layer-scoped KV-cache candidate.** Added as `c20` and measured. It lands
exactly between the two cache policies on every axis — cache 1.200 GB/device against 1.854 and
0.981, full-context sequences 24 against 15 and 28, **top-1 0.980 against 0.990 and 0.970** —
which makes it the cleanest evidence in the stage that the accuracy loss is the cache dtype
itself, scaling with the number of layers that hold one, rather than an artifact of one layer
kind. It is still a top-1 regression against `c14`, so the same rule rejects it. Its per-layer
readback (39 sliding layers at BFP4 and each of the 13 full-attention layers at BFP8, as 14
distinct groups in `runs/c20-*.json:realised_precision.layer_groups`) is also the strongest
propagation evidence in the stage for layer exceptions.

**Concern — the README named the selected config as a Pareto frontier point.** It is not one:
the top-1 frontier is `c08` and `c15`, and `c14` sits 0.017 t/s/u below `c15` at the same
accuracy, inside what either metric resolves. The document now says so, and the figure gate
re-derives the frontier and fails if the README's claim about the selected point's membership
disagrees with it.

**Re-running what the fixes invalidated.** Adding `decoder_overrides` to the artifact schema
touched `precision_config.py`, `model.precision_report` and `multichip_decoder.precision_report`
*after* the acceptance tests, the watcher runs and `c19` had already run. The model those
changes produce is identical — `decoder_overrides` defaults to `{}` and the rest is reporting —
but "identical by inspection" is the claim this stage exists to avoid making, so `c19` (the one
candidate that uses a companion setting), both test suites and both watcher sets were re-run on
the final tree.

**Concern — `build_generator` now hard-fails without a file under `doc/`.** Deliberate, and
recorded as a packaging note for the vLLM stage: `tt/` has a hard runtime dependency on
`doc/datatype_sweep/selected_precision_config.json`, which is the price of the artifact being
genuinely consumed rather than advisory.

**Anomaly the reviewer classified and the stage had not.** `evidence_accuracy.json`'s
split-sampling two-step probe returns the same token three times (`45116` → `45116` →
`45116`), where the previous stage recorded `45116 → 25 → 1102`. Its prompt is 128 *random*
token ids, so a copy-the-last-token continuation is ordinary model behaviour on garbage input,
and four independent signals in the same file show device-side feedback is live: eight
non-aligned prompt shapes each return two *different* tokens, the 33-token counter audit shows
32 trace replays against 1 token refresh, and both autoregressive completions are coherent
128-token device-fed generations. The consequence worth carrying forward is that the probe's
`token_feedback_is_device_side` assertion is computed as `mid_tokens == sampled_step1` and is
therefore **vacuously true** in this run: it has lost its discriminating power on a degenerate
prompt. That is the *parent* harness (`doc/full_model/bench/evidence.py`), unchanged by this
stage, so it is recorded here rather than patched under a datatype sweep — the assertion needs
a non-degenerate prompt, and the four controls above are what actually carry the claim today.

## 9. Round 2 of the stage review

Round 2 returned `more-work-needed` on four P2 findings. Two of them are the same class of
defect as round 1's — a document claiming something the artifacts do not support — and two are
measurement errors that changed a number.

**P2 — the two qualitative arms did not run the same prompt.** The suite's system message
embeds the current date. The parent harness re-renders the prompts unconditionally, so reusing
the full-model stage's HF control (rendered 2026-08-14) while re-rendering the TT prompt
(2026-08-15) put the two arms one token apart at index 31 — which guarantees a divergence
regardless of precision and made `first_divergence_from_hf` a measure of the calendar. This is
the most consequential finding of either round, because that number is this stage's only
quantitative HF comparison and it had been compared against the previous stage's, which *was*
prompt-matched.

`--reuse-hf-control` now **pins** the prompt token ids to the control's own and logs how many
it pinned, and the whole TT arm was re-run: `QUAL pinned 6 prompt(s) to the reused control's
token ids`. The figure gate asserts the identity, so the two cannot drift apart again. The
confounded numbers are withdrawn and the README carries the pinned ones.

**And the pinned re-run surfaced a real anomaly the confounded one had hidden.** `p1` now
diverges from the control at **token 1** — the position an early-divergence rule flags as a
wrapper bug. It is the chat template's recipient slot: the control emits ` to=self` (the
internal reasoning channel) and the selected config emits ` to=user` (a direct answer), and
both continuations are coherent and on-topic. `bench/channel_margin_probe.py` scores that exact
position on the pinned prompt plus the control's own first token and finds the margin between
the two tokens is **1.500 logits under the baseline and 0.0625 under the selected policy** —
0.45 % of a ~13.9 logit, i.e. a tie. At that margin the branch is decided by which numeric path
evaluates it: the probe's prefill path takes `self` under both configs, the qualitative run's
traced-decode path takes `user` under `c14` and `self` under `c00`. Classified as controlled,
with the probe as the control rather than the prose.

**P2 — `selected_precision_config.json`'s provenance described an older sweep.** It was
hand-assembled at selection time and still quoted a nine-config tied set after `c20` had joined
it. `analyse.py` now writes the artifact itself, provenance included, from the same `selection`
it just computed, and the figure gate asserts that its `selection_reason` and tied set match
`sweep_results.json`. Hand-assembly is what made it possible for the file to describe a
different sweep from the one that chose it.

**P2 — the README understated the cross-check's variance by two orders of magnitude.** It said
"~0.01 % spread"; the real figure is that the **third** of the three 64-replay rounds is
1.51–2.11 % slower than the first two in *every* measured candidate, and the first two agree to
0.024–0.038 %. The metric takes the min, so the systematic cannot bias the ranking — but the
`c14`-versus-`c15` margin the tie-break turned on is 0.033 %, which is that resolution, not
something well inside it. The regimes table now carries the measured spreads, the systematic is
named, and the figure gate re-derives it.

**P2 — count errors that round 1 reported as fixed were still in `work_log.md` and the bench
scripts.** Round 1's fixes went into the README; the gate only ever read the README, so the
work log kept the old host-only test count and the old blocked-candidate count, and
`run_watcher.sh` still described one fewer precision device case than it runs. Fixed, and the
gate now reads `work_log.md` too and rejects a spelled-out count that contradicts the
artifacts — including one quoted as a historical error, which is why the wrong figures are
described here rather than reproduced.

**Concerns acted on.** Two more candidates were added and measured, both of which the review
named as the obvious legal ones the matrix lacked:

* **`c21`** — the BFP4 LM head at HiFi2, which is the BFP4+LoFi-versus-BFP4+HiFi2 comparison
  `$datatype-sweep` requires for every material BFP4 group and which had only been run for the
  two decoder groups. The head is 190 MB/device and the largest matmul in the step, so it
  qualifies. Same answer as the decoder groups: numerically identical, **2.2 % slower**.
* **`c22`** — BFP4 KV cache on the 13 full-attention layers, the converse of `c20`. Together
  they show the accuracy cost is almost entirely in those 13 layers (`c22` alone reaches
  0.970, the same as putting BFP4 on all 52, while `c20`'s 39 sliding layers only reach 0.980)
  while three quarters of the capacity saving is in the other 39. That makes `c20`, not `c08`,
  the efficient trade for a serving stage that wants cache headroom — which is a real result
  for the next stage rather than a box ticked.

The `c13` blocker's quoted code block named the wrong op (`layernorm` rather than
`layernorm_pre_all_gather`); both are quoted now. The "no prompt echo, no control-token
leakage" line was replaced with the upstream classification, since the completions *do* open
with ` to=self<|message|>` and the HF control does the same. The final watcher run's exit-134
teardown abort is now named in the README, so `WATCHER_CLEAN` is not read as "exit 0".

## 10. Round 3 of the stage review

Round 3 returned `more-work-needed` on four P2 findings and confirmed the measurements
themselves: it independently re-derived the candidate matrix, the round ranges, the
propagation, the Pareto frontier, the fidelity bit-identity, the miss position and the layer
floors, and all of them checked out. Every finding was a document or a provenance string
disagreeing with an artifact — and three of the four were *recurrences* of the class the
previous two rounds had reported fixed, which is the useful thing this round found.

**P2 — the tie-break measured the wrong pair.** `analyse.py::select` computed the cross-check's
"separation" as the spread across the whole surviving set, so a slow-but-accurate candidate at
the bottom of it (`c19`) made the rule announce a decisive 0.98 % margin while the two
candidates actually being chosen between were 0.033 % apart. The rule now works the way it
reads: drop everything the cross-check separates from its best **by more than that metric's own
measured resolution**, then take the simplest of what remains. The resolution is measured from
the metric rather than assumed — `cross_check_resolution()` returns the worst agreement between
any candidate's two good rounds, 0.038 % — because a fixed 0.5 % band would swallow the real
0.4 % gap between `c14` and `c01`. The selection is unchanged and now rests on the rule it
claims to: step 1 drops the five KV-BFP4 candidates on top-1, step 2 leaves `c14` and `c15`,
step 3 takes `c14` as the simpler policy. Fixing it in the wrong direction first was
instructive: applying simplicity across the whole surviving set picks `c01`, which is 0.44 %
slower on the cross-check — real, separated, and exactly what the resolution-based step exists
to protect.

**P2 — `work_log.md` carried figures the artifacts contradict, and the round-2 gate extension
did not cover them.** Round 2 taught the gate two spelled-out-count phrases; round 3 found a
superseded `c19` throughput, a device-case count, a pass-1 candidate count and the wrong op in
the `c13` blocker. The gate now re-derives per-candidate throughput figures quoted in the work
log against `sweep_results.json`, and the device/host/pass-1 counts against the junit XML and
`runs_pass1_rounds5/`. The lesson each round has repeated is the same one: a gate that reads one
document guards one document.

**P2 — the README's qualitative verdict asserted the opposite of its own artifact.** The line
claiming no prompt echo and no control-token leakage survived round 2's fix (which had been
written against a slightly different sentence and silently matched nothing). Every completion
does open with a channel header and five of six restate the prompt — and the HF control does the
same on the same prompts, which under `$qualitative-check` makes it model behaviour. The README
now records that classification and cites the control.

**P2 — the third-round penalty figure went stale when `c22` was added.** 1.51–2.11 % became
1.50–2.31 %. Now re-derived by the gate.

**Concerns acted on.** The `p1` channel flip attribution was extended over four configs. Round
4 then showed the extension was half-answered: the probe scored the *prefill* path only, and
`decode_ccl_dtype` is consumed exclusively on the decode path, so `c06` matching the baseline
there was structurally forced rather than measured — and the flip itself happens on a path the
probe never scored. The probe now scores both, and the decode arm reproduces the flip exactly
(`c14` decode argmax is ` to=user`, which is what the qualitative run generated). The two
changes turn out to do different halves of it: BFP4 attention weights collapse the margin from
1.750 to **0.0000** — an exact tie, both logits landing on the same bf16 value — and the BFP8
decode CCL payload, worth 0.19 logits on its own and flipping nothing, tips that tie by 0.125.
Neither flips this token alone. The contract's inherited
layer-stack floor is no longer emitted as if it applied: `refresh_context_contract.py` withholds
it and records the selected policy's own floor instead, because `evidence_perf.json` flags the
inherited figure as measured under a different policy and the contract was dropping that flag.
The KV capacity table's labels now name the *scoped* dtype (`c22` was filed under a label naming
its majority dtype, the mirror of `c20`'s), and its note carries both halves of the result. The
chart's label separation was widened.

**One concern not acted on, and why — with the reason corrected in round 4.** Round 3 noted
that `c12` (BFP8 activations) was rejected on an exact op contract without the adapted retry
`c13` got, and named two adaptations: keep the wqkv output BF16 while the residual stream is
BFP8, or run BFP8 activations in prefill only. This log first answered that both need a
per-tensor dtype exception the schema cannot express. Round 4 showed that is wrong for the
second one, and the artifact says so: `c12`'s blocker is a **decode** op firing inside
`_capture_decode_trace`, and `runs/c12-activations-bfp8.json` records a completed 100-token
prefill accuracy pass at **0.990 / 1.000 / 1.000** before it. A prefill-only arm therefore needs
a *phase-scoped* activation dtype — the pattern this stage already built twice, for the CCL
payload and for math fidelity — and its accuracy half is already measured.

It is still not run, for a reason about the win rather than the cost. The only metric a
prefill-only change can move is TTFT, and prefill here is host-dispatch bound: 4122 ttnn calls
issuing in 54.91 ms against 55.08 ms to drain, with 33 % of the wall time in 209 collective
*calls* rather than in their payloads (previous stage). Narrower activations remove no dispatch.
TTFT's process-to-process spread on identical code is ~61–70 ms, so the lever could not clear
the variance of the metric it moves even if it worked perfectly. That is a quantified reason to
decline it, and it is recorded as one — where the earlier version recorded a wrong mechanism.

**Two observations worth keeping.** The two-layer smoketest's greedy-token comparison is
knife-edge: `c11` differs from `c00` only in the LM head's `in0_block_w` and returns a different
prefill top-1 and all six different decode tokens at a prefill PCC of 0.99976. That is a
weakness of the token comparison taken alone — and it is why the BFP4 bit-identity conclusions
rest on identical prefill PCC *to sixteen digits* as well as identical tokens. And the selected
policy demonstrably moves individual logits by ~1 at contended positions (the `p1` pair, and the
one readiness miss's top1-top2 margin narrowing from 2.0 to 1.6875); top-1, top-5 and top-100 do
not move, but the evidence resolving that is 100 reference tokens plus 6x128 qualitative ones,
which README limitation 1 states.

## 11. Round 4 of the stage review

Round 4 confirmed the round-3 fixes and the measurement layer — it re-derived every median,
round range, third-round penalty, trace-replay count, propagation report and the Pareto frontier
independently, re-executed `analyse.py::select` and reproduced `sweep_results.json:selected`
field for field, and ran a mutation negative control against `check_propagation` — and returned
two P2 findings, both in the evidence-to-claim layer.

**P2 — the margin probe's `c06` arm could not measure what it was cited for.** The probe scored
the prefill path only; `decode_ccl_dtype` reaches the model exclusively through
`_row_parallel_dtype(role, prefill=False)`, so `c06` matching the baseline on a prefill probe is
forced by the code rather than observed. Worse, the flip being explained happens on the decode
path, which the probe never scored — so the attribution rested half on a tautology and half on a
path substitution. The probe now takes `--paths prefill,decode`, and the decode arm both
reproduces the flip and changes the conclusion: it is not that the CCL payload contributes
nothing, it is that the BFP4 attention weights collapse the margin to an **exact tie** (1.750 →
0.0000) and the CCL payload then tips it (0.125). The earlier, tidier claim was wrong.

**P2 — `c12`'s rejection rationale was contradicted by `c12`'s own run.** See §10's corrected
paragraph: the blocker is a decode op, the prefill half completed at 0.990/1.000/1.000, and the
un-run adaptation needs a phase-scoped activation dtype rather than a per-tensor exception. Both
documents now carry the per-candidate blocked table (which of the five fail in prefill and which
at decode-trace capture, with the three prefill accuracy triples) instead of describing all five
as producing no number.

**Concerns acted on.** `selected.reason` opened by describing the tied set as "within 0.5 % of
the fastest" when `c05` is in it via the round-range-overlap clause; the sentence now states the
rule it applies. `c10`'s rejection percentage in the README table was quoted against `c14` while
every other row is against the arm it modifies; it is now against `c00`, matching §5.2. The
before/after table's sourcing line claimed both `evidence_perf.json` files for a teacher-forcing
row neither contains; it now names the sweep rows. `accuracy_stable` was computed over
(top-1, top-5) while the README claims the triple; `sweep.py` now covers top-100 and, because
the stored artifacts predate that, the figure gate re-derives the triple from the raw rounds.

**Concerns recorded rather than acted on.** `o_proj`'s decode geometry (16 cores,
`in0_block_w=2`) is carried across the attention group's move to BFP4. The value is the maximum
legal divisor at that core count — per-device K is 1024, i.e. 2 K-tiles per core — so the
"try larger legal divisors" criterion is satisfied by exact divisibility rather than by a sweep.
The alternative the decoder stage measured and declined (8 cores / `in0_block_w=4`, +0.11 %) was
measured at BFP8, and BFP4 halves that row's DRAM traffic, which moves a DRAM-bound row's
optimum toward *fewer* bytes per core rather than more — so the declined candidate is less
attractive at BFP4, not more. Recorded rather than re-measured. And the ~1248
`Mismatch between computed MemoryConfig ... Using computed config` warnings per run
(`matmul_device_operation.cpp:239`, computed 14 cores of `[32, 96]` against a provided 16-core
grid) are present identically in the previous stage's logs, so they are inherited and not a
precision regression; they are noted here because no document in the port had classified them.
