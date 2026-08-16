# Stage 07 — precision-config plumbing, and the baseline refreshed through it

This is the *preparation* for the datatype sweep, not the sweep. It does one
thing: turn the model's precision policy from module-level constants into a
value that construction consumes, and then re-measure the shipped baseline
through the new default path to show that nothing moved.

Nothing here changes a shipped precision value.

## 1. Why the constants had to go

The stage-07 goal requires the selected precision config to be "either the model
default or a required config artifact that later full-model/vLLM construction
paths actually consume by default", and says in as many words that **a JSON
field ignored by hard-coded model code does not satisfy this requirement**.

Before this stage the policy was five module-level constants in
`tt/optimized_decoder.py` (`EXPERT_WEIGHT_DTYPE`, the two `EXPERT_IN0_BLOCK_W_*`,
`EXPERT_MATH_FIDELITY`, `ATTENTION_WEIGHT_DTYPE`), two more in `tt/model.py`
(`LM_HEAD_WEIGHT_DTYPE`, `EMBED_WEIGHT_DTYPE`), and a scatter of literal
`dtype=ttnn.bfloat16` arguments in the forward paths. Read directly at their call
sites, they are bound at import: a sweep could not vary them without editing
source between runs, and a JSON file written next to unchanged source would have
been exactly the thing the goal rules out.

## 2. What was built

`tt/precision.py` — a frozen `PrecisionConfig` dataclass, `DEFAULT_PRECISION`,
and lossless JSON serialisation. Twenty fields:

| group | fields |
| --- | --- |
| expert weights | `experts_gate_up_dtype`, `experts_down_dtype` |
| expert matmul shape | `experts_gate_up_in0_block_w`, `experts_down_in0_block_w` |
| attention weights | `attention_qkv_dtype`, `attention_wo_dtype` |
| other weights | `lm_head_dtype`, `router_dtype`, `embedding_dtype`, `norm_weight_dtype` |
| compute fidelity | `experts_fidelity`, `attention_fidelity`, `router_window_fidelity`, `lm_head_fidelity`, `norm_fidelity` |
| activations | `activation_dtype`, `ccl_dtype` |
| kv cache | `kv_cache_dtype` |
| terminal | `logits_dtype`, `sampling_dtype` |

Four design decisions worth stating.

**gate_up and down are separate fields, and so are qkv and wo.** Both pairs ship
at one dtype today, so one field each would have reproduced the default
perfectly — and would have made half the sweep unaskable. `down` feeds the
residual directly and `wo` feeds the attention all-reduce; those are the two the
sweep most wants to price apart from their siblings.

**The block widths live in the precision config even though they are not
dtypes.** `EXPERT_IN0_BLOCK_W_GATE_UP`'s own comment records that 16 wins at LoFi
and 32 at HiFi4 — the knobs interact, and stage 02 first drew the *wrong*
conclusion from sweeping dtype at `in0_block_w=1`. A sweep that could move dtype
but not block width would be measuring a mis-tuned point and would rediscover
that error.

**`attention_fidelity` and `ccl_dtype` default to `None`, meaning "unchanged".**
The attention projections pass no compute-kernel config today and the
collectives run at whatever dtype the partial arrives in. Encoding either as an
explicit value would have been a guess about what the op picks; `None` reproduces
today exactly (no config object built, no cast op emitted) and any other value
takes a real path. This is the only way the default could be *behaviourally*
identical rather than merely numerically similar.

**`fp32_dest_acc_en` is deliberately not a field.** It corrupts expert output on
Blackhole (tt-metal #49068). A sweep must not be able to turn it on.

### Migration, not duplication

The seven old constants **still exist and still resolve to the same values**, but
they are now *derived* from `DEFAULT_PRECISION` rather than being the source of
truth. Probes under `doc/` and several stage-02/04 tests import them by name and
none of them had to change. Anything that needs to *vary* the policy takes a
`PrecisionConfig`; the constants are bound at import and cannot follow a
non-default model, which their comments now say.

### Threading

`Qwen3CoderModel.__init__` and `from_checkpoint` take `precision=`, resolved once
into `self.precision` and read by every builder and every forward. It accepts a
`PrecisionConfig`, a dict, or a **path to JSON** — the last so a sweep runner can
hand over `selected_precision_config.json` without importing the dataclass.

`tt/generator.py:build_generator` — the factory the readiness runners, the
qualitative suite and (later) vLLM all arrive at — takes the same kwarg and also
reads `QWEN3_PRECISION_CONFIG` from the environment as a path. Unset, which is
every run to date, means `DEFAULT_PRECISION`.

Below that, `precision` is a defaulted parameter on `upload_multichip_weights`,
`create_mesh_kv_cache`, `all_reduce`, `router_forward_multichip`,
`moe_decode_multichip`, both `decoder_layer_*_multichip`, `fallback_audit`, and
their single-chip counterparts in `optimized_decoder.py`.

## 3. Proof that it is consumed

`tests/test_precision_config.py` — 11 tests, 9 host-only and 2 on the mesh.

`fallback_audit` now also reports what the precision config *actually put on the
device*: dtypes read back off the uploaded tensors, the block widths
`_tuned_sparse_matmul_config` resolved to, the fidelities in the compute configs,
and the per-die expert allocation in bytes. `test_non_default_precision_reaches_the_device`
builds a 2-layer model twice on the same mesh — once at the default, once at a
`NON_DEFAULT` that moves five fields — and asserts on the difference. Nothing in
it reads `model.precision`; every assertion is against device state.

Observed, default → non-default:

| observable | default | non-default |
| --- | --- | --- |
| gate/up weight dtype on device | `BFLOAT4_B` | `BFLOAT8_B` |
| down weight dtype on device | `BFLOAT4_B` | `BFLOAT4_B` (not overridden — the two are separate fields) |
| `wo` weight dtype on device | `BFLOAT8_B` | `BFLOAT16` |
| `lm_head` weight dtype on device | `BFLOAT8_B` | `BFLOAT4_B` |
| resolved `gate_up_in0_block_w` | 16 | 32 |
| resolved `down_in0_block_w` | 12 | 12 (unchanged) |
| expert math fidelity in the compute config | `LoFi` | `HiFi4` |
| expert weight bytes per die, one layer | 84.9 MB | 135.3 MB |

and the non-default model then generates four tokens, so a config that reaches
the device without wedging it is what was demonstrated.

`test_default_construction_audit_matches_the_shipped_values` is the same
assertion in the other direction: the no-argument path puts the shipped dtypes,
widths and fidelities on the device.

## 4. Serialisation

`PrecisionConfig` ↔ JSON round-trips losslessly, twice over, for the default and
for two non-default configs (`test_json_round_trip_is_lossless`). The JSON emits
exactly the dataclass fields — asserted set-equal, so a field added without a
serialisation rule fails the test rather than silently vanishing from the
artifact — and `test_json_carries_every_field_the_goal_lists` maps each item the
goal enumerates onto the field(s) that carry it.

Names are plain (`"bfloat4_b"`, `"LoFi"`), not `repr` (`DataType.BFLOAT4_B`), so
the artifact diffs cleanly across sweep rows. The default is archived here as
`default_precision_config.json`; it becomes `selected_precision_config.json` when
the sweep picks a point.

## 5. Baseline refreshed through the new default path

All three runs on the shipped tree with **no precision argument passed**, i.e.
through `DEFAULT_PRECISION`. Reference is `../../readiness_aime24_chat.refpt`
(AIME24, chat template, 158 prompt tokens, 100 generated tokens). Logs in
`logs/`, each with its command, git state and date in the first three lines.

| gate | stage 06 (committed) | stage 07, default path | log |
| --- | --- | --- | --- |
| `run_prefill_check` top-1 / top-5 / top-100 | 0.980 / 1.000 / 1.000 | **0.980 / 1.000 / 1.000** | `logs/run_prefill_check.log` |
| `run_teacher_forcing` top-1 / top-5 / top-100 | 0.990 / 1.000 / 1.000 | **0.990 / 1.000 / 1.000** | `logs/run_teacher_forcing.log` |
| `run_teacher_forcing` traced decode | 42.25 t/s/u | **42.33 t/s/u** (+0.19%) | same |
| `run_teacher_forcing` TTFT (158-token gate prompt) | — | 3387.46 ms | same |
| warmed TTFT, prompt 128 | 125.431 ms | **126.114 ms** (+0.54%) | `logs/perf_full_model.log`, `probes/perf_full_model.json` |
| warmed `token_out` | 19.693 ms / 50.78 t/s/u | **19.686 ms / 50.80 t/s/u** (-0.03%) | same |
| warmed `model_trace` (logits only) | 19.567 ms / 51.11 t/s/u | 19.561 ms / 51.12 t/s/u (-0.03%) | same |

**No drift.** Every accuracy figure is bit-identical to the committed one. The
two decode figures move by 0.03% and 0.19%, which is inside the run-to-run
spread these probes already document. TTFT is +0.54%: its five warm samples are
125.64-126.40 ms against stage 06's 125.28-126.01, i.e. two overlapping bands
about 0.7 ms wide, and TTFT is the noisiest of the three (it is one prefill, not
a median over 128 traced tokens). Nothing in the default path emits an op that
did not exist before, so there is no mechanism for a real 0.5% prefill cost:
`attention_fidelity=None` builds no compute-kernel config, `ccl_dtype=None`
emits no cast, and `sampling_dtype == logits_dtype` emits no typecast.

The generated tokens agree too, to the extent these gates observe them: both
readiness runners score identically against the same reference (98/100 and
99/100), and the perf probe's two sampler legs both return token 16, as they did
at stage 06.

The traced teacher-forcing figure is the goal's Pareto ranking metric and is
taken from `run_teacher_forcing`, which runs the traced generate path; no eager
number is quoted here.

`run_prefill_check` and the pytest suite were re-run once more at the very end,
after the last source edit (threading `activation_dtype` into
`attention_prefill`), so all archived logs come from the same tree. The
teacher-forcing and perf logs already did.

### Non-aligned prompt lengths

Nothing touched here changes chunking: `moe_prefill_optimized`'s pad-to-chunk and
slice-back is untouched, the collectives still scatter on dim 3, and
`attention_prefill` gained only a dtype argument. The non-aligned gates in
`tests/test_full_model.py` and `tests/test_multichip_decoder.py` are part of the
suite below and pass.

### Test suite

`pytest tests/ -m "not models_performance_bare_metal" -q` — **157 passed**,
16 deselected (`logs/pytest_full_suite.log`). Stage 06's bar was 146; the 11 new
tests are `tests/test_precision_config.py`. The perf tests were **not** run —
they rewrite committed CSVs. (This is the *plumbing* phase's count. The final
suite is 158 — see §11 — after a twelfth precision test was added for the
`norm_fidelity` regression.)

## 6. What makes a clean sweep awkward

These are the things a sweep will trip over, written down now rather than
discovered mid-sweep.

**1. Two module copies of `tt.precision`, and `isinstance` lies.**
`tt/generator.py` imports `tt.model` by *absolute* path
(`models.autoports.qwen_...tt.model`) while tests and probes usually import it
*relatively*. There is no `models/__init__.py`, so under this repo's
`--import-mode=importlib` pytest roots the package at `models/autoports` and the
two spellings produce two distinct module objects — and therefore two distinct
`PrecisionConfig` classes, for which `isinstance` is `False` on an object that is
by every meaning the right one. This cost a debugging cycle here. Two mitigations
are in place: `tests/test_precision_config.py` imports absolutely and says why,
and `_resolve_precision` falls back to rebuilding through `to_dict()` for an
object whose class is *named* `PrecisionConfig`. **Sweep scripts should import
absolutely**, or pass the JSON path, which sidesteps the problem entirely.

**2. `_tuned_sparse_matmul_config` silently lowers an illegal block width.**
It clamps `in0_block_w` to the largest divisor of K in tiles at or below the
target, without raising. A sweep row that asks for a width that does not divide K
will therefore *run*, and will be recorded under the width it asked for rather
than the width it got. `fallback_audit` now reports the **resolved** widths
(`gate_up_in0_block_w`, `down_in0_block_w`) — sweep rows should record those,
not the config's own fields.

**3. Not every literal dtype is configurable, and the gaps are load-bearing.**
`activation_dtype` reaches the embeddings, both expert matmul families, the
attention projections on both prefill and decode, and the LM head. It does *not*
reach: the router projection (deliberately `float32` — selection on raw logits is
what makes the top-k renormalisation valid), the router tail's `bfloat16` casts,
the rotary tables, or the SDPA output. Those are either correctness constraints
or shapes no one wants to sweep, but a sweep that sets `activation_dtype` to
something exotic will find the model still bf16 in places.

**4. `bfloat4_b` and `bfloat8_b` have no `element_size()`.** The binding raises
for block-float types, so the per-die byte figures come from a tile-bytes table
in `multichip_decoder._TILE_BYTES` (1088 B/tile for bfloat8_b, 576 for
bfloat4_b — a shared exponent per 16-element face row on top of the payload). A
sweep that adds a dtype must add a row there or `device_expert_bytes_per_die`
comes back `None`.

**5. `ccl_dtype` changes the persistent CCL buffer cache.** `_decode_ccl_buffers`
keys on `(logical shape, padded shape, dtype)`, so a non-default wire dtype
allocates a *second* set of buffers rather than reusing the first. That is
correct — and it means a sweep that varies `ccl_dtype` inside one process pays
DRAM for every value it visits. Sweep one value per process.

**6. `attention_fidelity` has no measured default.** It is `None`, meaning "the
op picks". When a sweep sets it, the flags around it (`math_approx_mode`,
`fp32_dest_acc_en`, `packer_l1_acc`) are copied from
`_expert_compute_kernel_config`, which is the closest *measured* neighbour but
is not itself an attention measurement. The first non-`None` row should be
compared against the `None` row before anything is concluded from it.

**7. Anything varying an expert dtype pays a full checkpoint reload.** The
weights are quantised at upload, so a sweep point cannot be reached by mutating
a live model — it is ~3 minutes of 48-layer load per row. The 2-layer tier
(`QWEN3_FULL_MODEL_LAYERS=2`, or `override_num_layers=2`) is ~10 s and is enough
for every *structural* assertion; only accuracy and t/s/u need 48.

**8. The old module constants are still importable and still bound at import.**
`O.EXPERT_WEIGHT_DTYPE` and friends read the *default*, always, whatever model is
loaded. Probes under `doc/` still use them and are correct to, because they
measure the default; a sweep probe that copies one of those files and expects it
to follow a non-default model will get the default silently. Take the value from
`model.precision` or from the audit.

---

# Part 2 — the sweep itself

Part 1 above built the machinery and re-measured the baseline through it.
This part uses it. The conclusions and the full results table live in
`README.md`; this is the narrative of how they were reached, including the
things that went wrong.

## 1. Sequencing, and what it cost

The device is single-tenant, so every measurement below is strictly
sequential. The budget shaped the design more than anything else:

| phase | rows | wall time |
| --- | --- | --- |
| tier A structural (2 layers) | 23 | ~6 min |
| tier B sweep (48 layers) | 20 measured + 3 blocked | ~62 min |
| stacked rows (48 layers) | 3 | ~9 min |
| noise band (2 configs × 3) | 6 | ~18 min |
| post-selection validation | proof + perf + 2 gates + suite | ~25 min |

**The two-tier split paid for itself immediately.** Tier A found three
candidates that cannot be constructed at all. At 48 layers that would have been
three wasted 3-minute runs; at 2 layers it was 30 seconds and produced a better
artifact — the runtime's own `TT_FATAL` text, which is exactly what the goal
asks for in place of a measurement.

Tier A also confirmed the thing that would have quietly invalidated the winning
rows: **no requested `in0_block_w` was clamped**, on any row, including the
full-K 64. `_tuned_sparse_matmul_config` clamps on divisibility alone
(`k_tiles % blk`) with no L1-capacity check, so it cannot silently rescue an
oversized block — meaning a slow full-K row would have been a real spill rather
than an invalid config. Every width in the results is a *resolved* width read
from `fallback_audit`, never a requested one.

## 2. A bug found in the audit, before it could lie

`runtime_fallback_audit` reported `"kv_cache_dtype": "bfloat16"` as a **string
literal**. It would have reported `bfloat16` for the bfp8 KV row too, and the
sweep would have recorded a KV-dtype experiment under the wrong dtype.

Fixed to read the dtype off the allocated cache tensor, with a
`kv_cache_dtype_source` field distinguishing `device_readback` from
`config_not_yet_allocated`.

**That fix then broke a test, and the fix to the fix is the interesting part.**
Reading back gives `str(dtype)` → `"DataType.BFLOAT16"`, but
`test_full_model.py` asserts the plain `"bfloat16"`, and so do
`doc/optimized_full_model/probes/check_published_figures.py` and its committed
`runtime_fallback_audit.json` — **stage evidence this stage must not modify**.
So the audit emits the plain name via `dtype_to_name()`: still a genuine device
readback, still compatible with the committed contract. The sibling `device_*`
fields keep `str(dtype)` and were left alone. Changing the test instead would
have been the wrong direction — it would have silently invalidated a committed
stage's checker.

## 3. What the sweep expected, and what it found

The candidate set was designed around per-token weight bytes, on the reasoning
that batch-1 decode is bandwidth-bound. The **lm_head** led the set: 2048×37984
per die, read in full every token, the biggest non-expert read in the model.
Halving it to `bfloat4_b` should have been the sweep's headline.

It bought **+0.12%** — inside the noise band.

The CCL was actively harmful (−2.06%), the terminal path flat-to-negative
(−0.31%), embeddings blocked and norms flat. **Twenty dtype and fidelity levers,
and exactly one of them survived contact with the band**: `R06_attn_bfp4`, both
attention projections to `bfloat4_b`, at +0.45% for one top-1 point — beyond the
band, at the declared floor, and therefore eligible.

That one survivor then failed the *second* test, which is the one that matters.
`R06` and the block-width winner touch disjoint ops, so the two were stacked and
measured as `R26_attn_bfp4_bw64_24` rather than assumed to add. Three repeats
each put `R26` at 43.46 / 43.52 / 43.55 against `R25`'s 43.46 / 43.38 / 43.54 —
**overlapping ranges**, means 0.12% apart, a third of the band. The attention
dtype buys nothing on top of the widths and costs a top-1 point every time.

So the model is evidently not weight-bandwidth-bound in the way the candidate
set assumed — with only 8 of 128 experts active per token, expert weights are
already sparse, and the remaining matmuls are latency- or schedule-bound rather
than byte-bound.

What did win was the pair of fields in the precision config that **are not
dtypes at all**: the expert matmul inner block widths. That is a slightly
uncomfortable result for a document called "the datatype sweep", and it is the
honest one. It also only surfaced because stage 07's plumbing put the block
widths *in* the precision config, on the explicit grounds that dtype and width
interact and sweeping either alone finds the wrong optimum. That decision was
made to avoid mis-tuning the dtype rows; it turned out to be where the entire
result lived.

## 4. The bracket, and the mistake not made

The first pass measured `gate_up` at 8 / 16 / 32 and `down` at 6 / 12 / 24, and
both came back **monotonic upward**. The shipped 16 and 12 were not near an
optimum; they were on a slope.

Two things could have gone wrong here and did not:

**Stopping at 32.** The first pass's best `gate_up` was 32, and it would have
been easy to select it. Extending to full-K (64) found another +0.29 t/s/u. The
brackets are asymmetric because K differs — `gate_up`'s K is `hidden_size`
(2048 = 64 tiles) so 16 was a *quarter* of full-K with two rungs above it, while
`down`'s K is `moe_intermediate_size` (768 = 24 tiles), so the first pass's 24
was **already** its ceiling. There is no `down` analogue of the bw64 row and the
README says so explicitly, so a reader does not wonder why one got a wider
sweep.

**Assuming the gains add.** They do not:

```
gate_up 64 alone  +0.89      down 24 alone  +0.65
naive sum         +1.54  ->  would predict 43.88
measured together +1.20  ->  actual        43.54
```

Inferring the combination would have overstated it by 0.34 t/s/u — about the
width of the whole noise band. Stage 02's lesson, reproduced exactly.

The 32→64 step (+0.29) being smaller than 16→32 (+0.60) is the expected shape:
at full-K there is no inner blocking left to remove. A *large* jump at full-K
would have been more suspicious than a small one.

**Why the shipped values were stale.** They came from single-chip stage-02
tuning. Expert parallelism has since cut per-die N four-fold, which changes
which blocking the matmul wants — so `EXPERT_IN0_BLOCK_W_GATE_UP`'s comment that
"16 wins at LoFi" was true when written and stale by stage 06. The comment is
now updated with the 48-layer numbers.

## 5. Two methodological saves

**The noise band.** Early rows differed by tenths of a percent and it was
tempting to read them as results. `probes/repeats.py` re-ran three identical
configs three times each: the band is **0.368%**. Nine rows sit inside it and
are reported as *indistinguishable from the default* rather than as small wins.
The most important casualty is `R04_qkv_bfp4`, whose +0.33% — the row a literal
"fastest passing config" rule would have selected — is **inside the band**, so
it was never a demonstrated win at all, quite apart from costing three top-1
points.

**Top-1's own resolution.** Three reduced-precision configs (`R05`, `R13`,
`R14`) scored top-1 **1.000**, *above* the 0.990 baseline. On a 100-token
reference one point is one token, so ±0.01 is noise. This is why the top-1 floor
is set at exactly one point rather than tighter, and why top-1 is reported as a
first-class column but not ranked on.

## 6. The KV row: a conclusion nearly recorded wrong, then a bug fixed

`R19_kv_bfp8` scored **0.010 on top-1, top-5 and top-100** and was 32% slower
with 2.5x worse TTFT. The obvious write-up — "bfp8 KV is too imprecise for this
model" — would have been **wrong**, and was nearly written.

Three things do not fit that story: top-100 at chance is not graceful
degradation (bfp4 expert weights are far more aggressive and hold top-5 at
1.000); a pure dtype reduction should be *faster*, not slower; and `bfloat8_b`
KV is used widely on Tenstorrent.

`probes/kv_bfp8_diagnosis.py` answered it at the op level in seconds, with no
model at all: a paged cache allocated `bfloat8_b` and filled from a `bfloat16`
input — exactly what this model does — reads back as **NaN**, while the same
cache filled from a `bfloat8_b` input round-trips at PCC 1.0.

**The first version of this stage stopped there**, recorded the row as
`unevaluated_integration_defect`, named the fix site, and declared the fix out of
scope. That was the wrong call twice over. First, the stage's own goal says to
attempt the fix and stop only if it fails, and no attempt was recorded. Second
and worse, leaving it meant shipping a **public, documented field of the
precision config this stage introduced that accepts a legal value and silently
fills the KV cache with NaN** — validated by nothing, warned about by nothing.

### The fix, and the thing the diagnosis had not yet asked

The probe's own docstring promised a `paged_update_cache` half that `main()`
never ran. Implementing it changed the answer:

| op | cache | input | round-trip |
| --- | --- | --- | --- |
| `paged_fill_cache` | bfloat8_b | bfloat16 | **NaN** |
| `paged_fill_cache` | bfloat8_b | bfloat8_b | PCC 1.0 |
| `paged_update_cache` | bfloat8_b | bfloat16 | PCC 0.999969 |
| `paged_update_cache` | bfloat8_b | bfloat8_b | **rejected by the op** |

**The two writers have opposite contracts.** The fill writer wants the input cast
*to* the cache dtype; the update writer wants it left as bfloat16, converts
internally, and hard-rejects a block-float update at
`paged_update_cache_device_operation.cpp:296`. The obvious symmetric fix — "cast
K/V to `precision.kv_cache_dtype` at all six call sites" — would have replaced
silent corruption with a hard crash on the decode path. It was the *decode*
half of the probe, the half the original stage promised and skipped, that said
so.

`tt/functional_decoder.match_cache_dtype` therefore casts at the four fill sites
only, takes the dtype off the **allocated cache tensor** rather than off the
config so the two cannot drift, and is a no-op in the shipped configuration. The
alternative the review offered — reject the mismatch in
`PrecisionConfig.__post_init__` — was not taken: it closes the hole by making a
working configuration illegal, and the cast makes it work.

### What bfp8 KV actually scores

Re-run post-fix, twice: as a delta from the stage-06 baseline (`R19`, 42.29
t/s/u, −0.12%) and on top of the selected widths (`R28`, 43.45 t/s/u, −0.21%
against `R25`). Both clear the gate at top-1 0.980 / top-5 1.000 / top-100
1.000. **bfp8 KV works, is decode-neutral, and is rejected for the ordinary
reason.** It costs 2.4x TTFT — 7.8 s against 3.3 s — which is *not* the cast:
the pre-fix run showed the same 8.1 s while the cache was full of NaN. That is
unexplained and is recorded as limitation 9 rather than guessed at.

This changes the context-contract conclusion's *reason* again, and this time to
something better than either previous version: 262144 stands not because the KV
candidate failed and not because it could not be evaluated, but because it was
evaluated, buys 3.020 GB/die of headroom nobody currently needs, and buys no
throughput. `doc/context_contract.json` carries the arithmetic as
`stage07_kv_bfp8_candidate` so a future capacity stage does not have to re-derive
it.

## 7. Selection, and the deviation declared

The rule is stated in `README.md` §1 **before** the results table, including the
two clauses that go beyond the formal gate (a top-1 floor of one point, and a
requirement to beat the measured band). Both are judgment calls and both are
labelled as such.

`R25` costs zero top-1 and beats the default by ~7x the band, so clauses (a)
and (b) are moot *for it*. They matter for what they exclude: `R04`, and the
nine band-bound rows.

A third clause was needed and is declared as such. The literal rule — fastest
config clearing the formal gate — selects `R26_attn_bfp4_bw64_24` at 43.58,
which leads `R25` by 0.09%: a quarter of the band, bought with a top-1 point.
**Clause (c): among the eligible, rows within one band of the fastest are tied,
and a tie breaks on the simpler and safer config — fewest dtype/fidelity fields
moved off the default — then on top-1, then decode, then TTFT.** The tiebreak is
the governing `datatype-sweep` skill's own rule, *"if two configs are within
measurement noise, prefer the simpler and safer one"*, and it settles this pair
outright: `R25` moves **no** dtype and no fidelity, only two block widths, which
are a scheduling choice and bit-identical; `R26` takes attention QKV and W_O to
bfp4 across 48 layers.

Leading on simplicity rather than on top-1 was a correction made during the
re-review, and it matters. Teacher forcing is deterministic per config, so
`R25`'s `0.990 ×3` against `R26`'s `0.980 ×3` is one token re-observed, not
three observations — and §1(a) and limitation 3 both hold that a one-point top-1
difference here is not signal. A tiebreak resting on top-1 while limitation 3
cited the tiebreak as its reason was circular; the skill's rule is independent
of that judgment and reaches the same answer. Top-1 is kept as the secondary
ordering and agrees.

Clause (c) was written after seeing the row it decides, which is stated in the
README rather than smoothed over, and it was checked rather than argued — `R26`
got the same three-repeat treatment `R00` and `R25` got, and the ranges overlap.
Without clause (c) the rule would rank on a difference it has already called
unmeasurable; applying the band against the default but not between candidates
was the actual inconsistency.

`R13_experts_bfp8_cotuned` deserves its mention: co-tuned at 32/24 so the
comparison is fair, it prices the shipped bfp4 expert choice at 0.93 t/s/u
slower than bfp4 at the same widths, for one token of top-1 and double the
per-die expert bytes. **The shipped bfp4 experts are confirmed correct** — which
is a real result even though nothing changed.

## 8. Proof, not assertion

`DEFAULT_PRECISION` now carries the selected widths, so
`default_precision_config.json` and `selected_precision_config.json` are
byte-identical. `probes/selection_proof.py` clears `QWEN3_PRECISION_CONFIG`,
builds the real 48-layer model with **no precision argument**, and diffs the
selected config against **device readback** — resolved block widths and dtypes
off uploaded tensors, never `model.precision`. **All 21 checks match**; it
exits non-zero otherwise.

One gap was found and closed while writing this: the audit's `kv_cache_dtype`
reads `model.kv_cache`, but the generator keeps its cache in
`Qwen3CoderGenerator._kv_cache`, so through this path the audit honestly
reported `kv_cache_dtype_source == "config_not_yet_allocated"` and fell back to
the configured value. The `source` field meant this was disclosed rather than
silently wrong -- which is exactly why it was added -- but a proof should not
lean on a fallback, so the probe now also inspects the allocated cache tensor
directly.

Updating the default required updating the tests that pinned 16/12
(`test_full_model.py`, `test_precision_config.py`). Those are now pinned to
64/24 with a pointer to this stage.

## 9. The prefill cost

The selection is **not free**: warmed TTFT regresses 126.114 → 129.941 ms
(+3.03%), outside TTFT's ~0.55% warm spread. Wider expert blocks help the
batch-1 decode matmul and cost a little in the prefill matmul, which runs at a
different M.

That rests on the warm benchmark alone. The teacher-forcing TTFT was briefly
cited as corroboration and **does not corroborate it**: `repeats.json` has `R00`
at 3277.95 / 3496.98 / 3463.80 ms and `R25` at 3248.42 / 3500.34 / 3439.81 ms,
fully overlapping, with the mean moving *down*. One cold prefill per run cannot
resolve 3%. The claim stands on the measurement that can.

Breakeven is **8.1 generated tokens**; at the 128-token profile the net is
−2.15% total latency and at 1024 tokens −2.37%. Reported in the README rather
than buried, because a reader tuning for a prefill-dominated workload needs to
know it.

## 10. What I would do next

1. **Explain bfp8 KV's 2.4x TTFT.** The dtype now works and is decode-neutral,
   so the only thing between the model and 3.020 GB/die of KV headroom is a
   prefill cost nothing in this stage profiles. Largest open question here.
2. **`R17`/`R18` (`activation_dtype = bfloat8_b`) — the barrier moved, it did
   not lift.** They were originally blocked at
   `paged_fill_cache_device_operation.cpp:36`. `match_cache_dtype` cleared that,
   both rows were re-probed on the fixed tree, and they now fail further in at
   `nlp_create_qkv_heads_decode_device_operation.cpp:41`
   (`input_tensor.dtype() == FLOAT32 || BFLOAT16`, *"Unsupported data format"*)
   — a different op with an unrelated rule, the decode head-split, nothing to do
   with the KV cache. `activation_dtype` still cannot leave `bfloat16` on this
   path. Getting past it means either a BF16 cast right before the head split,
   which gives back the bytes the row was trying to save, or a TTNN change.
3. **Re-tune `in0_block_w` for prefill separately.** The config has one width
   per expert matmul shared by both phases, and this stage showed they want
   different things. A per-phase width would take the decode win without the
   TTFT cost.
4. **A bfp8-expert row at 64/24.** `R13`/`R14`/`R15` co-tune at 32/24 because
   the bw64 result arrived after them; the dtype×width interaction is unmeasured
   at the new ceiling.
5. **Thread `norm_fidelity` into the prefill norms too**, or rename it. It
   reaches only `decode_residual_norm` today (§11).

## 11. What the independent review changed

The review returned `more-work-needed` on two P1s and several P2s. Everything
below is what those turned into. Three of the findings uncovered defects larger
than the finding itself, which is recorded here because that is the useful part.

**P1-A — "not one dtype survived" was false, and the stacked row was never
measured.** `R06_attn_bfp4` cleared every clause (+0.45%, beyond the band, top-1
exactly on the 0.980 floor) and `selection_reasons.json` had it under `eligible`
with no rejection reason — while the README's headline said no dtype survived,
its rejection table listed `R06` among the band-bound rows (making that row nine
entries against §1's stated eight), and §1 claimed Pareto-optimality against
everything evaluable. All three were prose contradicting data that was itself
correct. `R26` (`R06 ∘ R25`) and `R27` (its LoFi pair) were measured as full
48-layer rows, `R26` was repeated three times, and the headline now says what the
data says.

`R26` also forced clause (c) — see §7. It is the only rule change this review
produced, and it changes no eligibility verdict.

**P1-B — `kv_cache_dtype = bfloat8_b` silently wrote NaN.** Fixed, measured, and
described in §6. The review proposed casting at all six call sites; implementing
the `paged_update_cache` half of the probe (which the original stage promised in
a docstring and never ran) showed that would have crashed the decode path. Four
sites cast, two deliberately do not.

**A dead config field, found by closing an audit gap.** The review noted that
`selection_proof.py` verified 16 of 20 fields, and that `R03`, `R21` and `R22`
therefore had `device_audit` blocks byte-identical to `R00`'s — so "no effect"
and "lever never engaged" were indistinguishable. Adding
`lm_head_math_fidelity`, `norm_math_fidelity` and the two observed terminal
dtypes to `runtime_fallback_audit` found that for `norm_fidelity` it was the
second: `decode_residual_norm` built its compute config from the module default
and never saw `self.precision`. **`R21_norm_hifi2` had measured nothing.** The
threading is fixed and `R21` was re-measured for the first time (42.44 t/s/u,
+0.24%, still inside the band). The proof now checks 21 fields; `ccl_dtype` is
labelled a resolved config value rather than counted as a readback.

**A baseline that moved under its own candidate set.** Not in the review, but
found while acting on it. `candidates.py` built every row as a delta from
`DEFAULT_PRECISION` — and this stage *moves* `DEFAULT_PRECISION`. Re-running tier
A after the selection landed therefore reported `R00_default` at the **selected**
widths, contradicting the 48-layer row every gain in this document is quoted
against. `candidates.BASELINE_PRECISION` now pins the stage-06 policy, and
`_assert_baseline_is_stage06` cross-checks it against `configs/R00_default.json`,
which was written at sweep time and is a byte-level record of what `R00` ran.
Tier A was re-run against the pinned baseline and now matches tier B row for row.

**The charts.** `top5_perf_pareto.png` drew no frontier and no frontier legend
entry: the plotting code drew the frontier as a line and skipped it below two
points, and on the top-5 axis every config sits at 1.000 so the frontier is a
single point. It is now drawn as a ringed marker with its own legend entry — the
single-point frontier *is* the finding. The top-1 chart's dotted line used
`MIN_TOP5` labelled "shown for reference: the gate binds on top-5, not top-1",
while `selection_reasons.json` enforces `top1 >= 0.980`; the two numbers are
equal, so the line was right by coincidence and captioned as if it were not. It
is now the top-1 floor, labelled as such.

**Smaller corrections.** §1 said a literal reading of the rule selects `R04`; it
does not — `R04` is twelfth fastest and was never in contention. What the clauses
stop is `R04` *outranking the default*. §9 cited teacher-forcing TTFT as
corroborating the +3.03% warm-TTFT regression; `repeats.json` shows fully
overlapping ranges with the mean moving down, so the corroboration is deleted and
the primary claim, which is sound, stands alone. `tt/precision.py` said the
default was the shipped policy unchanged and that stages 02-06 were measured at
exactly it, which stopped being true when the widths moved; both places now say
which two fields differ. `sweep_results.json` no longer contains bare `NaN`
(RFC 8259 has no such literal) — the probe records a measured NaN as an explicit
flag, and both writers pass `allow_nan=False` so it cannot come back.
`test_multichip_decoder.py`'s width assertion compared the audit against
`O.EXPERT_IN0_BLOCK_W_*`, which is derived from `DEFAULT_PRECISION` — the same
source the audit resolves from, so it was an identity that could not fail. It
pins the literals now, as `test_precision_config.py` already did.

**And a checker, so this class of drift is mechanical next time.**
`probes/check_published_figures.py` re-derives every figure in `README.md` and
this file from `sweep_results.json`, `repeats.json`, `selection_reasons.json` and
the perf JSONs, including the specific defect above: **no config listed as
`eligible` may appear in a rejection row**. Every stage before this one had such
a checker; stage 07 did not, and the review found exactly the kind of drift it
would have caught.

### Final state after the review

Everything below was re-run on the post-review tree, all through the default
construction path with `QWEN3_PRECISION_CONFIG` unset:

| | result | log |
| --- | --- | --- |
| tier A, all 29 rows, against the pinned stage-06 baseline | every row builds; widths resolve as requested; the four new audit fields discriminate | `logs/structural_probe.log` |
| `R19`, `R26`, `R27`, `R28` at 48 layers | measured | `logs/sweep_review_rows.log`, `logs/sweep_review_r27.log` |
| `R21_norm_hifi2`, first real measurement | 42.44 t/s/u, +0.24%, top-1 0.990 | `logs/sweep_review_r21.log` |
| `R26` × 3 repeats | 43.46 / 43.52 / 43.55, top-1 0.980 ×3 | `logs/repeats_r26.log` |
| `selection_proof.py`, 21 checks | **PASS** | `logs/selection_proof.log` |
| prefill gate | top-1 0.980 / top-5 1.000 / top-100 1.000 | `logs/run_prefill_check_selected.log` |
| decode gate | top-1 0.990 / top-5 1.000 / top-100 1.000, 43.48 t/s/u | `logs/run_teacher_forcing_selected.log` |
| full test suite | **158 passed, 16 deselected** | `logs/pytest_selected.log` |
| `check_published_figures.py` | **PASS** | — |

The selection is **unchanged**: `R25_gateup64_down24`. Every `tt/` change made
during the review is a no-op in the shipped configuration — the KV cast fires
only when the cache dtype differs from the K/V dtype, and `norm_fidelity`'s
threading passes the same `HiFi4` the module default already produced — so the
post-selection performance figures in `README.md` §9 stand, and the
teacher-forcing confirmation re-ran at 43.48 against the original 43.63, inside
the band.

## 12. What the re-review changed

Four items, none of which moved a selection, a chart or a gate.

**1. A blocker that had gone stale, and our own fix caused it.** `README` §4 and
`sweep_results.json` published `R17`/`R18` as blocked at
`paged_fill_cache_device_operation.cpp:36`. That was true when they were first
probed. `match_cache_dtype` then cleared it, the tier-A probe was re-run on the
fixed tree at 09:46, and `probes/structural_probe.json` recorded a **different**
failure for both rows —
`nlp_create_qkv_heads_decode_device_operation.cpp:41`, *"Unsupported data
format"* — but the published rows were never regenerated from it.

The three `blocker_*` fields were re-derived from `structural_probe.json`
through `sweep_runner.blocked_row`, the same parser that wrote them originally,
so nothing was retyped. §4's table and limitation 1 were rewritten: the old text
argued that this blocker and §6's NaN were "the same op's dtype rule seen from
two sides", which is no longer true — `nlp_create_qkv_heads_decode` is a
decode-path head-split with its own hard BF16/FP32 rule and nothing to do with
the KV cache. The **conclusion** does not move: `activation_dtype` still cannot
leave `bfloat16` on this path, so no number, chart or selection changes.

**2. The checker gained the assertion that would have caught it.**
`check_published_figures.py` now re-derives every blocked row's `blocker_op`,
`blocker_assertion`, `blocker_info` and `blocker_raw` from
`probes/structural_probe.json` and requires the README's blocker table to quote
the same op and info. This is a distinct failure mode from the ones the checker
was built for: the README and `sweep_results.json` agreed with each other
*perfectly*, and both disagreed with the probe underneath them, so every
existing cross-check passed. Verified in both directions — injecting the old op
into `sweep_results.json` fails 2 checks, injecting it into the README fails 4.

**3. The post-selection token-out benchmark was re-run.** Its artifact was
timestamped 07:50 and the review fixes landed in `tt/` from 09:23, so §9's
headline described a tree that no longer existed. The no-op argument was sound,
but the goal names this benchmark specifically and it is a three-minute run, so
it was measured rather than reasoned about (`logs/run_review_fix5.sh`, same
command as `run_final_stage07.sh`). It moved by 0.02% on `token_out` and 0.03%
on TTFT — 19.217 → 19.213 ms and 129.910 → 129.941 ms — which is the no-op
claim confirmed on silicon. §9 now carries the new numbers and says which run
they came from.

**4. Clause (c) re-argued on the citation that was available all along.** The
governing `datatype-sweep` skill says: *"If two configs are within measurement
noise, prefer the simpler and safer one."* That settles `R25` vs `R26` outright
and without appeal to top-1, which matters because clause (c) previously broke
its tie on top-1 while limitation 3 cited clause (c) as the reason not to trade
a top-1 point — circular, and resting on an axis §1(a) had already called
unresolvable. Simplicity is now computed mechanically in `analyze_sweep.select`
(`numerical_changes`: dtype and fidelity fields moved off the default; block
widths excluded, being a scheduling choice and bit-identical), top-1 is demoted
to the secondary ordering, and limitation 3 was rewritten to stop citing the
clause it justifies. `R25` moves 0 such fields, `R26` moves 2, `R28` moves 1 —
so the selection is unchanged and now rests on the skill's own rule.
