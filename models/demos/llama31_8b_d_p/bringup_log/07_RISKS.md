# 07 — Risks and open questions

The register and the sections must agree. Re-checked at every phase boundary (last: end of P2,
2026-09-04).

| Id | Severity | Phase found | Summary | Status | Owner |
|---|---|---|---|---|---|
| R-001 | medium | P0 | `(1,1)` gates test a KV-head count the deployment mesh never produces | open — scoped to P8 | P8 (`G-KV-TP8`) |
| R-002 | low | P0 | Checkpoint identity established against the in-repo config, not against the live gated HF repo | mitigated | P0 (`DEC-001`) |
| R-003 | medium | P0 | Pre-existing tilized weight caches inside `$HF_MODEL` (`ttnn_cache/`, `P150/`) | open | P6 (`G-WEIGHTS`) |
| R-004 | low | P0 | `CHUNK_SIZE` / `MAX_SEQ_LEN` not yet chosen | open — deferred by `DEC-004` | P7 (`G-CHUNK`) |
| R-005 | high | P1 | `rope_theta` is absent from the `transformers` 5.12.1 config object; `getattr` with a default silently substitutes a wrong theta | **mitigated and enforced** as of P5.3 | closed by `tt/rope.py` + `G-ROPE` |
| R-006 | medium | P1 | The hand-written oracle and HF could share a misreading of the architecture | open — inherent | P0 card / P6 (`G-MODEL`) |
| R-007 | medium | P2 | `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` and `rotary_embedding_indexed` are consumed from `deepseek_v3_d_p`'s substrate; no Llama-specific test exists upstream | open | P5.6 (`G-KV`), P7 |
| R-008 | low | P2 | The kit ships fewer example files than its own README and `WHY_THESE_EXAMPLES.md` advertise | open — affects the kit, not the model | kit maintainer |
| R-009 | medium | P2 | No in-repo template implements a **dense, bias-free, full-RoPE** attention block; `tt/attention/` is an adaptation with three features deleted | open | P5.5 (`G-ATTN`) |
| R-010 | low | P2 | `compute_llama3_parameters` hard-codes low/high frequency factors instead of reading the config | mitigated in-package (P5.3 assert); open upstream | P5.3 (`G-ROPE`) |
| R-011 | low | P5.2 | `gpt_oss_d_p`'s dormant distributed-RMSNorm branch passes `stats` twice and would raise `TypeError` if enabled | open upstream; not carried into this package (`DEC-031`) | upstream `gpt_oss_d_p`, plus P8 if scheme B is taken |
| R-012 | medium | P5.1 | The repo root ignores `*.log`, so every gate raw log — the run's whole evidence base — was untracked through P0-P4 | mitigated by a nested `.gitignore` (`DEC-037`); the kit does not warn about it | kit maintainer / repo maintainer |

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
