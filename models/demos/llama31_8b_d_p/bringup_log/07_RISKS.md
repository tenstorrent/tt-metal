# 07 — Risks and open questions

The register and the sections must agree. Re-checked at every phase boundary (last: end of P5.6,
2026-09-04).

| Id | Severity | Phase found | Summary | Status | Owner |
|---|---|---|---|---|---|
| R-001 | medium | P0 | `(1,1)` gates test a KV-head count the deployment mesh never produces | open — scoped to P8 | P8 (`G-KV-TP8`) |
| R-002 | low | P0 | Checkpoint identity established against the in-repo config, not against the live gated HF repo | mitigated | P0 (`DEC-001`) |
| R-003 | medium | P0 | Pre-existing tilized weight caches inside `$HF_MODEL` (`ttnn_cache/`, `P150/`) | open | P6 (`G-WEIGHTS`) |
| R-004 | low | P0 | `CHUNK_SIZE` / `MAX_SEQ_LEN` not yet chosen | open — deferred by `DEC-004` | P7 (`G-CHUNK`) |
| R-005 | high | P1 | `rope_theta` is absent from the `transformers` 5.12.1 config object; `getattr` with a default silently substitutes a wrong theta | **mitigated and enforced** as of P5.3 | closed by `tt/rope.py` + `G-ROPE` |
| R-006 | medium | P1 | The hand-written oracle and HF could share a misreading of the architecture | open — inherent | P0 card / P6 (`G-MODEL`) |
| R-007 | medium | P2 | `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` and `rotary_embedding_indexed` are consumed from `deepseek_v3_d_p`'s substrate; no Llama-specific test exists upstream | **write path covered** as of P5.6 (`G-KV`, bit-exact at `head_dim = 128`); the indexed RoPE is still untested numerically | P7 (`G-CHUNK`), P8 |
| R-008 | low | P2 | The kit ships fewer example files than its own README and `WHY_THESE_EXAMPLES.md` advertise | open — affects the kit, not the model | kit maintainer |
| R-009 | medium | P2 | No in-repo template implements a **dense, bias-free, full-RoPE** attention block; `tt/attention/` is an adaptation with three features deleted | **mitigated** as of P5.5 (`G-ATTN`: every hand-written stage 1.00-2.50x of its floor, block 0.999+) | closed by `G-ATTN`; P6 re-checks at layer level |
| R-010 | low | P2 | `compute_llama3_parameters` hard-codes low/high frequency factors instead of reading the config | mitigated in-package (P5.3 assert); open upstream | P5.3 (`G-ROPE`) |
| R-011 | low | P5.2 | `gpt_oss_d_p`'s dormant distributed-RMSNorm branch passes `stats` twice and would raise `TypeError` if enabled | open upstream; not carried into this package (`DEC-031`) | upstream `gpt_oss_d_p`, plus P8 if scheme B is taken |
| R-012 | medium | P5.1 | The repo root ignores `*.log`, so every gate raw log — the run's whole evidence base — was untracked through P0-P4 | mitigated by a nested `.gitignore` (`DEC-037`); **the kit half is now closed** — see the entry | kit maintainer / repo maintainer |
| R-013 | medium | P5.6 | Recipe §2.5's "keep probe values <= 256" is a **bf16** rule; at `bfloat8_b` — the dtype the recipe mandates for the KV cache — the exact-integer ceiling is **128**, so a probe built to the stated rule fails on a correct cache | mitigated in-package (`DEC-044`); the kit's rule is still wrong | kit maintainer; P8/P10 probe authors |
| R-014 | low | P5.6 | The repo-root `expect_error` fixture matches `message` as a **regex** while its docstring describes a substring, so any refusal message containing `*`, `(`, `)`, `.` or `+` silently never matches | worked around (`DEC-045`: metachar-free substrings) | repo maintainer (`conftest.py:948`) |
| R-015 | medium | P5.5 | `G-ATTN`'s **8x block budget** (Appendix A) is arithmetically unreachable at bf16 given the fused SDPA kernel's own slack, which §2.3 of the same recipe measures at 71x — the two numbers cannot both hold for any correct implementation | mitigated (`DEC-042`: gate on the SDPA-attributed residual, raw 8x asserted at bf8_b) | kit maintainer; **P6 owns the same arithmetic for `G-LAYER` (8x) and `G-MODEL` (8x / 4x step)** |
| R-016 | low | P5.5 | `verify_citations.py`'s doc-ref pass only checks that a cited line is **in range**, not that it contains anything relevant, so a wrong-but-in-range `path:line` is reported as `resolved` | mitigated by promoting the load-bearing refs into `CITES` (content-checked); the pass itself is unchanged | kit maintainer |

---

## R-001 — A `(1,1)` gate can test a configuration the model never produces
**Fact.** The packed KV cache holds exactly one KV head per chip
(`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:95`, `:99`), which forces `TP == num_key_value_heads == 8`
(`00_MODEL_CARD.md` §4.1). On a `(1,1)` mesh TP is 1, so `nkv_local` is 8, not 1 — a shape the
deployment mesh never emits, and one `update_padded_kv_cache` rejects outright with
`TT_FATAL(cache_shape[1] == input_shape[1], ...)`
(`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
**Impact.** `G-KV` at `(1,1)` can only prove the cache **primitive** (write correctness, no
collateral writes, `head_dim=128` geometry) with a single synthetic head. It proves nothing about
the model → cache path. Unaffected: every non-KV `(1,1)` gate, whose per-chip shapes at TP=1 are the
full-width shapes and are legitimate.
**Status.** open, by design. The recipe scopes the missing coverage to P8's `G-KV-TP8`.
**How to close.** Run `G-KV-TP8` on a `(1,8)` submesh (head→column mapping gated on **bit-equality**,
not PCC — recipe §2.5) and state in `G-KV`'s own gate block what it does not cover. Owner: P8.

## R-002 — Identity rests on the in-repo config, not the live HF repo
**Fact.** `DEC-001` establishes `meta-llama/Llama-3.1-8B-Instruct` by byte-identity between
`$HF_MODEL/config.json` and `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`
(`md5 3cd5831d379b509d53afade0e24c36e9`). The recipe asserts (§The machine) that the live gated repo
also byte-matches; this run did not re-derive that, and no network call was made in P0–P2.
**Impact.** If the vendored copy were itself wrong, the card would be confidently wrong. That would
require the repo's own Llama-3.1-8B config to be wrong, which `models/tt_transformers` exercises
continuously.
**Status.** mitigated — two independent in-tree sources agree byte-for-byte, and the safetensors
index is consistent with those dims (checked in P1's `G-REF` transcript).
**How to close.** One `huggingface_hub` fetch of the repo's `config.json` and a diff, if a reviewer
wants the third source. Owner: whoever needs it; not on the critical path.

## R-003 — Stale tilized weight caches inside the checkpoint directory
**Fact.** `$HF_MODEL` contains `ttnn_cache/` and `P150/` — tilized-weight caches written by an
earlier, unrelated run (`ls /home/mstojkovic/models/Llama-3.1-8B-Instruct`).
**Impact.** A tilized cache is **already sharded**, so one written at a different mesh shape or
dtype is wrong at ours; the recipe's symptom for that is "one layer runs on garbage, three phases
later" (P3 conventions; Appendix B). Nothing in P0–P2 reads them.
**Status.** open.
**How to close.** This package must put the **mesh shape and dtype in its own cache path** (the
`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75` pattern) and must not write into
`$HF_MODEL`. `G-WEIGHTS` (P6, and its P8 TP=8 extension) is the gate that would catch a stale hit.
Owner: P6.

## R-004 — `CHUNK_SIZE` / `MAX_SEQ_LEN` still unchosen
**Fact.** `DEC-004` defers the values; the constraints are recorded in `00_MODEL_CARD.md` §4:
`CHUNK_SIZE % (SP*32) == 0` → `% 128` at SP=4, `MAX_SEQ_LEN % CHUNK_SIZE == 0`, and
`MAX_SEQ_LEN > CHUNK_SIZE` so the cache-read path is actually exercised
(`models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md:62`; `BRINGUP_RECIPE.md:1811`).
**Impact.** None before P7. If P7 picks values violating the arithmetic the symptom is a chunk-write
assert, which is loud.
**Status.** open, deferred deliberately.
**How to close.** P7 picks both, logs a `DEC`, and `G-CHUNK` measures them. Owner: P7.

## R-005 — `rope_theta` substitution is silent on transformers 5.12.1
**Fact.** Measured on this box (`raw/G-REF_20260904T035140Z.log`, and `01_REFERENCE.md` §4): `cfg.rope_theta`
raises `AttributeError`; `getattr(cfg, "rope_theta", 10000.0)` returns **10000.0**; the true value
500000.0 lives in the JSON and inside `cfg.rope_scaling`/`cfg.rope_parameters`. Two live call sites
in the nearest template do exactly this: `models/demos/gpt_oss_d_p/tt/model_config.py:76` and
`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:185`.
**Impact.** A RoPE wrong at every position, with no exception anywhere — the highest-severity silent
trap in the bring-up. Copying either line into `tt/model_config.py` or `tt/tt_prefill_runtime.py`
reproduces it exactly.
**Status.** mitigated by design (`01_REFERENCE.md` §4 pins the rule: theta and scaling are read from
the **raw `config.json` dict** through `models/tt_transformers/tt/common.py:165` / `:183`, in exactly
one place, asserted non-`None`), **regression-tested** in P1 by
`test_rope_theta_is_not_an_attribute`, and **enforced in device code as of P5.3**:
`models/demos/llama31_8b_d_p/tt/rope.py`'s `rope_params` is the only reader, it goes through
`get_rope_theta` / `get_rope_scaling` on the raw dict, and it asserts each of the three values
non-`None`. Measured through it at `G-ROPE`: `(theta, factor, orig_context_len) ==
(500000.0, 8.0, 8192)` (`raw/G-ROPE_20260904T091040Z.log`).
**How to close.** Closed for `tt/rope.py`. Re-checks owed where the same values could re-enter:
P6.2's `ModelArgs` and P7's `tt/tt_prefill_runtime.py` must call `rope_params` rather than read the
config themselves, and `G-CLEAN` greps for `getattr(.*rope_theta` across the package. Owner: P6.2,
P7, P9.

## R-006 — Two agreeing oracles can share one misreading
**Fact.** `G-REF` shows the hand-written torch reference and HF `LlamaDecoderLayer` agree to
PCC 1.0 / `max|Δ| = 0.0`. That proves the transcription is faithful; it does not prove either is
right about Llama-3.1's architecture (recipe P1, "Read the bit-exactness honestly").
**Impact.** An architectural misreading (e.g. wrong GQA repeat order, wrong norm placement) would be
invisible to every module gate and would surface only as a bad top-1 at `G-MODEL`.
**Status.** open — inherent to the method; two controls exist.
**How to close.** (a) The P0 card's per-row provenance, every row traced to a `config.json` key;
(b) `G-MODEL`'s 100% top-1 against HF with **real weights**, which no shared transcription error
survives. Owner: P6.3.

## R-007 — The chunked-KV and indexed-RoPE ops have no Llama-shaped upstream test
**Fact.** `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` and
`ttnn.experimental.rotary_embedding_indexed` come from the DeepSeek prefill substrate and are
consumed by `models/demos/gpt_oss_d_p` and `models/demos/minimax_m3` at *their* head geometries
(head_dim 64 / 128 with partial RoPE). Llama's geometry — head_dim 128, **full** rotary, GQA 32/8 —
is a combination neither package exercises.
**Impact.** An op-level constraint that happens to hold at head_dim 64 could bite at 128; the
failure mode would be a loud `TT_FATAL`, not a silent one, so severity is bounded.
**Status.** open.
**How to close.** `G-KV` (P5.6) exercises the write op at head_dim 128 on one card;
`G-CHUNK` (P7) exercises indexed RoPE at full rotary. If either refuses, `07_RISKS` gets an upstream
issue rather than a local workaround. Owner: P5.6 / P7.

## R-008 — The kit advertises example files it does not ship
**Fact.** `models/demos/common/bringup/README.md:22` lists `examples/module_test_vs_ref.py`, and
`models/demos/common/bringup/WHY_THESE_EXAMPLES.md:17-24` additionally lists `examples/mesh_config.py`,
`examples/ccl_manager.py`, `examples/dense_mlp.py` and `scripts/new_bringup.sh`. The kit directory
contains only `examples/noise_floor.py` and `examples/verify_citations.py`.
**Impact.** None on this model — the recipe's in-repo templates (`models/demos/minimax_m3`,
`models/demos/gpt_oss_d_p`) cover every missing example, and `WHY_THESE_EXAMPLES.md` says as much
itself. It cost this run a few minutes of looking for `module_test_vs_ref.py`, which is the canonical
gate-test shape the recipe repeatedly points at; the substitute used is
`models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py`, which the recipe also names.
**Status.** open — a kit defect, reported upward, not fixable from inside this package.
**How to close.** Ship the four files, or delete the rows. Owner: kit maintainer.

## R-009 — No in-repo template is a dense, bias-free, full-RoPE attention block
**Fact.** `02_SURVEY.md` §3: `models/demos/gpt_oss_d_p/tt/attention/` is the structural template but
carries sinks, sliding-window alternation, YaRN and biases; `models/demos/minimax_m3` carries partial
RoPE, QK-norm and MSA. Llama has none of them (`00_MODEL_CARD.md` §3). `models/tt_transformers`
*does* have the right shape but is a whole-model framework with no `models/demos/common/prefill`
adapter and no SP.
**Impact.** `tt/attention/*` is the one part of the package that is an adaptation-by-deletion rather
than an import. Deletions are cheap to get wrong in the safe direction (dead code) and expensive in
the unsafe one (a sink tensor left at zeros changes the softmax denominator).
**Status.** open until `G-ATTN`.
**How to close.** P5.5 writes the block against the P0 "does NOT have" list, and `G-ATTN` measures it
against a hand-written reference with **no** sinks and **no** window — so a leftover feature shows up
as a PCC failure rather than as a plausible number. Owner: P5.5.

## R-010 — `compute_llama3_parameters` hard-codes the low/high frequency factors
**Fact.** `models/tt_transformers/tt/common.py:407-408` sets `low_freq_factor = 1` and
`high_freq_factor = 4` as literals inside the function, and `apply_scaling`
(`models/tt_transformers/tt/common.py:437`) passes only `scale_factor` and `orig_context_len`
through. HF's `_compute_llama3_parameters` reads all four from the config.
**Impact.** **None for this model** — Llama-3.1-8B-Instruct's `config.json` says
`low_freq_factor: 1.0`, `high_freq_factor: 4.0`, so the literals agree and `G-REF` measured the
helper bit-identical to a config-driven transcription (`max|Δ| = 0.0`). Any *other* llama3-scaled
checkpoint with different factors would be silently wrong — the failure mode is a RoPE that is
correct below `original_max_position_embeddings` and wrong above it, which short-sequence gates
cannot see.
**Status.** **mitigated in this package as of P5.3**; still open upstream.
`models/demos/llama31_8b_d_p/tt/rope.py`'s `assert_llama3_factors` compares the config's
`low_freq_factor` / `high_freq_factor` against the helper's literals and raises on a mismatch; it
runs on every call to `rope_params`, so no table can be built past it.
`tests/unit/test_rope_vs_ref.py::test_llama3_scaling_is_active` executes it
(`raw/G-ROPE_20260904T091040Z.log`).
**How to close.** Upstream: pass the factors through `apply_scaling`. Owner: upstream note; the
in-package half is done.

## R-011 — The template's dormant distributed RMSNorm would raise `TypeError` if switched on
**Fact.** `models/demos/gpt_oss_d_p/tt/rms_norm.py:82` passes `tt_gathered_stats` positionally to
`ttnn.rms_norm_post_all_gather` and `:89` passes the same tensor again as `stats=`. The op's second
positional parameter **is** `stats`, so the call cannot bind. Measured on this box:
`ttnn.rms_norm_post_all_gather(x, s, stats=s)` raises
`TypeError: ttnn.rms_norm_post_all_gather(): incompatible function arguments`. The branch is
unreachable there (`models/demos/gpt_oss_d_p/tt/rms_norm.py:33` pins `is_distributed = False` with
its condition commented out), so nothing has ever executed it.
**Impact.** **None for this package**: `tt/rms_norm.py` passes `stats` once (`DEC-031`), and the
branch is dormant here too under residual scheme A (`DEC-025`). The risk is to whoever enables the
distributed norm in `gpt_oss_d_p` — or to a future package that copies the same lines, which is
exactly what nearly happened here. It is also the general case of the dormant-branch hazard
`DEC-028` names: a branch that has never run is a claim, not a fact.
**Status.** open upstream; not carried into this package.
**How to close.** File the one-line fix against `models/demos/gpt_oss_d_p/tt/rms_norm.py` (drop the
`stats=` keyword). Owner: upstream `gpt_oss_d_p`; this package's P8 re-checks it if residual scheme
B is ever taken.

## R-012 — The repo's `.gitignore` silently excluded every gate raw log
**Fact.** `.gitignore:7` is `*.log`. `git check-ignore -v` confirms it matches
`bringup_log/raw/*.log`, and `git ls-files models/demos/llama31_8b_d_p/bringup_log/raw/` was
**empty** after three committed phases — so the raw logs for `G-CARD`, `G-REF`, `G-SURVEY`,
`G-OUTLINE` and `G-CCL-PLAN` were never in a commit. The gate ledger cites them by filename
throughout.
**Impact.** The recipe's central evidence rule — "A gate with no raw log did not happen"
(`BRINGUP_RECIPE.md:199`) — and Appendix C item 2 (`BRINGUP_RECIPE.md:1817-1819`) were both
unsatisfiable by construction: on a fresh clone the ledger would cite 15 files that do not exist.
Nothing about the *numbers* is affected; what was at risk is their auditability, which is the
stated point of the whole logging protocol. The logs were never lost here only because the session
ran on the same disk.
**Status.** mitigated in-package by `bringup_log/raw/.gitignore` containing `!*.log`
(`DEC-037`), which re-includes them retroactively — P0-P4's logs are still on disk and are now
trackable.
**How to close.** Two things the kit and the repo owe, neither of which this session may write:
`models/demos/common/bringup/BRINGUP_RECIPE.md` §1.2 should say that a raw-log directory needs a
`.gitignore` exception (it currently discusses only the `check-large-files` hook, which implies
committing without saying how), and `scripts/new_bringup.sh` should scaffold that file alongside
`bringup_log/raw/`. Owner: kit maintainer.

**Re-checked at the end of P5.6: the kit half is closed.** `BRINGUP_RECIPE.md:197-215` now carries
the whole thing — the blanket `*.log` diagnosis, the exact `!*.log` file to scaffold, the
"verify `git ls-files <pkg>/bringup_log/raw/ | wc -l` is non-zero" step, and the note that
`scripts/new_bringup.sh` writes it for you. The in-package `.gitignore` from `DEC-037` is what that
guidance asks for, so nothing in the package changes. What remains open is only the repo-root
`*.log` rule itself, which is a repo-wide decision and not this package's to make.

## R-013 — `bfloat8_b` is exact only to 128, so §2.5's "<= 256" probe rule is wrong for the KV cache
**Fact.** `BRINGUP_RECIPE.md:1260-1262` and §2.5 (`:477-482`) both give the ceiling for an
integer-valued probe payload as **256**, justified by `bfloat16`. Measured on this box, quantising
the integers 0..511 through `quantize_like_device` with each 16-lane block already held **constant**
(the ideal case for a shared exponent): the first inexact integer is **129** at `bfloat8_b` and
**257** at `bfloat16`. The recipe mandates `bfloat8_b` for the KV cache (P5.6, "every threshold in
Appendix A assumes it").
**Impact.** A `G-KV` positional probe written exactly as the recipe specifies **fails on a correct
cache**. It did, here: 4 chunks of 64 covering positions 0..255, `bfloat8_b`, chunk 2, odd rows,
`max|delta| = 1.0` — bf8_b's 7-bit block magnitude gives a resolution of 2 above 128, so odd
integers land on their even neighbour. This is §2.5's own trap ("a failing probe is not evidence of
a failing module until the probe's own numerics are checked") firing against §2.5's own rule, and it
costs a debugging session to anyone who trusts the stated number.
**Status.** mitigated in-package (`DEC-044`): the probe is 4 chunks of 32 = 128 positions, and the
per-dtype ceiling is a named, measured constant asserted against the payload, so the next author is
stopped by an assertion rather than by a mystery.
**How to close.** The kit should state the ceiling **per dtype** — 128 at bf8_b, 256 at bf16 — and
note that a bf8_b probe additionally needs each 16-lane block held constant. Owner: kit maintainer.
Downstream: `G-KV-TP8` (P8) and `G-KV-TABLE` (P10) both probe positions well past 128 and must use
§2.5's split-lane encoding rather than raising the ceiling.

## R-014 — `expect_error`'s `message` is a regex, not the substring its docstring describes
**Fact.** `conftest.py:948`'s docstring says "`message` must appear in the real device error text
(the TT_FATAL line), since that's what the triager matches"; `conftest.py:962` implements it as
`pytest.raises(error, match=message)`, i.e. a regex search.
**Impact.** Any refusal message containing a regex metacharacter silently fails to match, and the
test fails with `Regex pattern did not match` while the code under test behaved correctly. Hit here
on `"must be a multiple of TILE_SIZE*sp"` — the `E*` made it unmatchable. Assertion and `TT_FATAL`
text is full of parenthesised values (`kv_actual (16) must be tile-aligned`), so this is a standing
trap for every refusal test, and the four gates whose whole content is a refusal (`G-MESH`,
`G-RUNTIME`, `G-SP-RING`, plus the `scatter_output` seam) are the most exposed.
**Status.** worked around (`DEC-045`): this package matches on a metachar-free substring and says
why at the call site.
**How to close.** Either fix the docstring or `re.escape` the argument in the fixture — the latter
would match the documented behaviour and break nothing, since a literal is a valid regex.
Owner: repo maintainer. A kit note under `LANDMINES.md`'s existing `prefer-expect-error` row would
also have saved the time.

## R-015 — `G-ATTN`'s 8x block budget contradicts §2.3's own measurement of the fused SDPA kernel
**Fact.** Appendix A sets `G-ATTN` at "block <= 8x" its noise floor
(`BRINGUP_RECIPE.md:1770`). §2.3 (`:396-412`) separately measures
`ttnn.transformer.scaled_dot_product_attention` **alone** at **71x** its modelled floor and states
that the floor model "does not describe a fused kernel's interior". Both cannot hold: measured here,
the block sits at **2.17-2.22x** at bf8_b and **11.82-12.32x** at bf16, and the excess is
**entirely** the kernel — floor error plus the kernel's own excess predicts the block PCC to 5-6
decimals, leaving a residual of **0.70-1.10x**. For the raw bf16 ratio to reach 8x the kernel would
have to sit under ~17x its own floor.
**Impact.** A correct implementation fails a stated Appendix A threshold, which under §0 rule 1
("no forward progress on a `FAIL`") stops the bring-up. Note the counter-intuitive direction: the
**better** absolute PCC (bf16, 0.9998) has the **worse** ratio, because a smaller floor error
divides the same fixed slack — so the metric penalises the more accurate configuration.
**Status.** mitigated (`DEC-042`): the raw 8x is asserted at bf8_b, the package's weight dtype, and
both dtypes are gated on the **SDPA-attributed residual** — which is tighter than the raw budget
(0.70-1.10x measured against 8x) and still catches a regression in any stage this package wrote.
**How to close.** The kit should either state the block budgets **per weight dtype**, or define the
block budget on the fused-kernel-attributed residual as `DEC-042` does, or state a separate
allowance for blocks containing a fused kernel. Owner: kit maintainer. **P6 owns the same
arithmetic**: `G-LAYER` (8x) and `G-MODEL` (8x, plus a 4x per-layer step) both contain this kernel,
and both will meet this wall — `G-MODEL`'s 4x depth step is the tighter of the two.

## R-016 — The citation verifier's doc-ref pass range-checks rather than content-checks
**Fact.** `scripts/verify_citations.py`'s `CITES` list checks that a *substring* appears on the
cited line; its second pass, over every backtick-quoted `path:line` in the logs and docstrings,
checks only that the line number is **within the file**.
**Impact.** A wrong-but-in-range citation is reported as `resolved`, which is exactly the state
§1.6 calls "worse than no citation, because it reads as authoritative". **21 of this session's own
citations were wrong**, in two distinct ways:

* **A multi-file `cat -n`.** Eight refs into `models/demos/gpt_oss_d_p/tt/attention/operations.py`
  carried a **+209 line offset**, because they were read out of a `cat -n weights.py operations.py`
  whose numbering ran straight across both files. Four overshot the file and were caught as
  `DOC OUT OF RANGE`; the other four — `operations.py:223`, `:250-256`, `:340`, `:351` — were in
  range, wrong, and reported clean.
* **Interpolated, not read.** Seventeen refs into `BRINGUP_RECIPE.md` were estimated from the
  section headings a TOC grep had produced, rather than read from the cited line. Every one of them
  landed in the right *section* and the wrong *line*: Appendix A's `G-ATTN` row cited as `:1731`
  when it is at `:1770`, the `scatter_output` refusal as `:971-973` when it is at `:992`, the
  positional-probe rule as `:1250-1254` when it is at `:1260`. All in range; all reported
  `resolved`.

**This is not confined to this session.** Spot-checking the recipe refs written in P0-P5.3 shows the
same pattern — `:1728` cited for `G-RMS`'s Appendix A row (that row is at `:1767`; `:1728` is a P9
README checklist item), `:1720-1753` cited for "the 32 gate rows in Appendix A" (that range is P9's
cleanliness checklist), `:900-902` cited for the semaphore-reuse warning (which is at `:921-923`),
`:1817-1819` cited for Appendix C item 2 (which is at `:1858-1860`). Those entries are in
append-only logs written by other phases and are **not** rewritten here; they are recorded so a
maintainer can fix them in one pass.
**Status.** mitigated for this package's own refs by promoting every load-bearing P5.4-P5.6
citation into `CITES` — **359** content-checked entries, up from 279 at the end of P5.3, including
one per `BRINGUP_RECIPE.md` reference the package makes (the `RCP` block), so a recipe edit that
shifts a section now produces a `MISMATCH`. Prior phases' recipe refs remain unverified.
**How to close.** Two cheap options for the kit: have pass 2 warn when a cited line is blank or
consists only of a closing bracket, and/or resolve a doc ref by checking that the *citing sentence's*
backticked identifier appears within a few lines of the target. Owner: kit maintainer. A discipline
note is worth as much: **never read line numbers out of a multi-file `cat -n`.**
