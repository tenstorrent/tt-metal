# 07 — Risks and open questions

The register and the sections must agree. Re-checked at every phase boundary (last: end of P8,
2026-09-04).

| Id | Severity | Phase found | Summary | Status | Owner |
|---|---|---|---|---|---|
| R-001 | medium | P0 | `(1,1)` gates test a KV-head count the deployment mesh never produces | **closed as of P8** — `G-KV-TP8` proved head `c` -> mesh column `c` **bit-exactly** at TP=8 (8/8 columns, both RoPE states) and scored 32 layers of model-produced K/V against the fp32 golden (min K 0.9986432 / V 0.9942853) | closed (`G-KV-TP8`) |
| R-002 | low | P0 | Checkpoint identity established against the in-repo config, not against the live gated HF repo | mitigated | P0 (`DEC-001`) |
| R-003 | medium | P0 | Pre-existing tilized weight caches inside `$HF_MODEL` (`ttnn_cache/`, `P150/`) | **closed** as of P6.2 — `weight_cache_path` refuses to fall back to the checkpoint dir (`DEC-048`), gated by `G-WEIGHTS` | closed (`DEC-048`) |
| R-004 | low | P0 | `CHUNK_SIZE` / `MAX_SEQ_LEN` not yet chosen | open — deferred by `DEC-004` | P7 (`G-CHUNK`) |
| R-005 | high | P1 | `rope_theta` is absent from the `transformers` 5.12.1 config object; `getattr` with a default silently substitutes a wrong theta | **mitigated and enforced** as of P5.3 | closed by `tt/rope.py` + `G-ROPE` |
| R-006 | medium | P1 | The hand-written oracle and HF could share a misreading of the architecture | open — inherent | P0 card / P6 (`G-MODEL`) |
| R-007 | medium | P2 | `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` and `rotary_embedding_indexed` are consumed from `deepseek_v3_d_p`'s substrate; no Llama-specific test exists upstream | **closed as of P7** — the write path was bit-exact at P5.6 (`G-KV`) and the indexed RoPE is now **bit-identical** to the contiguous builder at `sp = 1`, four chunk offsets, `torch.equal` (`G-CHUNK`). The `sp > 1` block-cyclic path is still untested and is `R-001`'s, not this one's | **closed as of P8** — `G-MESH-KV` scores the block-cyclic `sp = 4` cache at three different periods (chunk_local 256/128/64) against the fp32 golden, min K 0.9967119 / V 0.9866232 | closed (`G-MESH-KV`) |
| R-008 | low | P2 | The kit ships fewer example files than its own README and `WHY_THESE_EXAMPLES.md` advertise | open — affects the kit, not the model | kit maintainer |
| R-009 | medium | P2 | No in-repo template implements a **dense, bias-free, full-RoPE** attention block; `tt/attention/` is an adaptation with three features deleted | **mitigated** as of P5.5 (`G-ATTN`: every hand-written stage 1.00-2.50x of its floor, block 0.999+) | closed by `G-ATTN`; P6 re-checks at layer level |
| R-010 | low | P2 | `compute_llama3_parameters` hard-codes low/high frequency factors instead of reading the config | mitigated in-package (P5.3 assert); open upstream | P5.3 (`G-ROPE`) |
| R-011 | low | P5.2 | `gpt_oss_d_p`'s dormant distributed-RMSNorm branch passes `stats` twice and would raise `TypeError` if enabled | open upstream; not carried into this package (`DEC-031`) | upstream `gpt_oss_d_p`, plus P8 if scheme B is taken |
| R-012 | medium | P5.1 | The repo root ignores `*.log`, so every gate raw log — the run's whole evidence base — was untracked through P0-P4 | mitigated by a nested `.gitignore` (`DEC-037`); **the kit half is now closed** — see the entry | kit maintainer / repo maintainer |
| R-013 | medium | P5.6 | Recipe §2.5's "keep probe values <= 256" is a **bf16** rule; at `bfloat8_b` — the dtype the recipe mandates for the KV cache — the exact-integer ceiling is **128**, so a probe built to the stated rule fails on a correct cache | mitigated in-package (`DEC-044`); the kit's rule is still wrong | kit maintainer; P8/P10 probe authors |
| R-014 | low | P5.6 | The repo-root `expect_error` fixture matches `message` as a **regex** while its docstring describes a substring, so any refusal message containing `*`, `(`, `)`, `.` or `+` silently never matches | worked around (`DEC-045`: metachar-free substrings) | repo maintainer (`conftest.py:948`) |
| R-015 | medium | P5.5 | `G-ATTN`'s **8x block budget** (Appendix A) is arithmetically unreachable at bf16 given the fused SDPA kernel's own slack, which §2.3 of the same recipe measures at 71x — the two numbers cannot both hold for any correct implementation | mitigated (`DEC-042`); **P6's half is measured and the wall does NOT bite at layer or model level** — `G-LAYER` raw 4.51-7.05x at bf16 and 1.47-1.81x at bf8_b, `G-MODEL` 2.23x, all inside 8x (`DEC-051`) | kit maintainer (`G-ATTN` only) |
| R-016 | low | P5.5 | `verify_citations.py`'s doc-ref pass only checks that a cited line is **in range**, not that it contains anything relevant, so a wrong-but-in-range `path:line` is reported as `resolved` | mitigated by promoting the load-bearing refs into `CITES` (content-checked); the pass itself is unchanged | kit maintainer |
| R-017 | medium | P6 | The kit's recipe grew 1986 -> 2017 lines **after** P5 was gated, so every prose `BRINGUP_RECIPE.md:NNNN` ref written in P0-P5 shifted; only the content-checked `CITES` half was updated, and pass 2 range-checks the rest | open — **and it recurred in P10**: the recipe grew another **23 lines while the P10 session was live**, turning 38 content-checked citations red at once (all re-pointed, `06_GATES.md`'s closing note). The mitigation works — pass 1 found all 38 in seconds — but §0.2's "do not mutate while a session is live" rule covers the worktree and not the **specification** the session is citing | kit maintainer; P9 |
| R-018 | medium | P6.1 | A negative control on a residual block is only as strong as the **input scale**: the same norm-swap control measures 0.99993 on a `randn` input and 0.66830 on real embedding-scale input | mitigated in P6.1 (`DEC-058`); open as guidance | P7/P8/P10 control authors; kit maintainer (§1.4) |
| R-019 | medium | P6.3 | `transformers` 5.12.1's `output_hidden_states` tuple ends with the **post-final-norm** stream, not the last layer's output, and `CausalLMOutputWithPast` has no `last_hidden_state` — norming it again is nearly idempotent and reads as a plausible wrong PCC (0.9916 vs the true 0.9997) | mitigated in-package (`DEC-052`: forward hooks only) | P7 (golden-KV generator); kit maintainer (a sixth P1 trap) |
| R-020 | medium | P6.3 | `G-MODEL`'s absolute PCC threshold is scoped to the **reduced-depth** runs by the phase text (`:1420-1422`) and to all depths by the Appendix A row (`:1856`); at 32 layers the measured post-norm PCC is 0.9984849 | resolved in-package (`DEC-053`); the kit's wording is still ambiguous | kit maintainer |
| R-021 | **high** | P6.3 | `G-MODEL`'s floor omitted the bf16 rounding of the RoPE tables the device stores while `G-LAYER`'s floor included it — worth **45%** of the floor error at 32 layers, and the whole difference between a reported 2.79x and **1.53x** | **fixed in-package** (the floor now quantises cos/sin); both numbers recorded (`DEC-053`) | kit maintainer (§2.2 wording); P7/P8 for their own floors |
| R-022 | medium | P6.3 | §2.3.1's additive kernel attribution **over-subtracts at 32-layer depth**: the substituted chain scores 0.9981153, worse than the device's 0.9984849, giving an attributed residual of **0.63x** (< 1.0) | open as a method limit; worked around (gate on the raw ratio against a complete floor) | kit maintainer; P7/P8 |
| R-023 | medium | P7 | **Delta 3** — chunk *k*'s queries attending the prefix read back out of the cache — cannot run in P7: it needs the ring path and TP=8. `G-CHUNK-ATTN` is recorded `BLOCKED` | **closed as of P8** — `G-CHUNK-ATTN` measured it: L1 mutual K **0.9999505** (>= 0.999), worst gated per-layer step 2.14x against 4x, both arms inside `G-CHUNK`'s golden thresholds | closed (`G-CHUNK-ATTN`) |
| R-024 | medium | P7 | Six engine-called runtime hooks (migration, layer-ack, trace) are present and **raise**; a migration or trace run therefore fails rather than silently publishing nothing | **narrowed as of P10** — `build_kv_chunk_table`, `kv_migration_base_address` and `set_layer_ack_channel` are implemented and gated (`G-KV-TABLE`, `G-MOCK-MIG`); `set_layer_completion_sink` and `set_d2h_ack_service` still raise (multi-rank, trace — both explicit non-goals), and `build_kv_chunk_table` still refuses a pipeline-rank layer slice (`R-032`). P10 also found the reverse defect: a **seventh** parameter, `metadata_msg`, was refused and had to be *accepted* (`DEC-108`) | P10 for the three closed; the two remaining are non-goals |
| R-025 | medium | P7 | The recipe describes `verify_golden_kv.py` as a **device**-vs-golden scorer (`:1548-1549`) and, six lines later, as importing **no ttnn** (`:1588-1590`) | resolved in-package by following the gate (`DEC-060`); the kit's two passages still contradict | kit maintainer |
| R-026 | medium | P7 | The 128 MB fp32 golden trace is **not in the repo**; `G-CHUNK`'s evidence depends on regenerating it from `$HF_MODEL` | mitigated: one command, ~100 s, and `metadata.json`'s `token_ids` pin the inputs (`DEC-066`) | P8/P10 (they score against the same trace) |
| R-027 | low | P7 | The recipe's delta-1 negative control quotes **two** numbers (0.706 / 0.655) for a control that in the gate's own decomposition can move only **K** — V is never rotated | resolved in-package: two controls, one per delta (`DEC-065`); K measured 0.72466 against the quoted 0.706 | kit maintainer |
| R-028 | low | P7 | Recipe P7 step 4 names `models/demos/minimax_m3/tests/unit/test_attention_chunked_vs_ref.py` as the template for the P7 test of the same name — but that file is precisely the **delta-3** cache-read test the same phase forbids P7 to run | worked around: the P7 file implements deltas 1-2 as the `G-CHUNK` gate text specifies, and the named template becomes P8's | kit maintainer |
| R-029 | **high** | P7 | `TtPrefillRuntime` has **never been instantiated**: `G-RUNTIME` is a static audit and the `tp == num_key_value_heads` equality forbids `(1,1)`, so no line of its happy path has executed | **closed as of P8** — `G-MESH-KV` drives the deployment path through it, `compile()` included (`DEC-084`); the six engine hooks still raise, which is `R-024`'s, not this one's | closed (`G-MESH-KV`, `G-CHUNK-ATTN`, `G-RACE`) |
| R-030 | **high** | P8 | **There is no ring fabric on this Blackhole Galaxy.** `FABRIC_1D_RING` cannot be initialised at all, and `Topology.Ring` is unserviceable for the ring SDPA | open — worked around by running the whole phase on `FABRIC_1D` + `Topology.Linear` (`DEC-079`, `DEC-081`) | machine owner / kit maintainer |
| R-031 | medium | P8 | The SP path has been measured under `Topology.Linear` **only**; nothing in P8 has run the deployment's collectives under `Ring` end to end | open — `Ring` is untestable here (`R-030`) | whoever runs a torus-cabled galaxy |
| R-032 | medium | P8 | **Multi-galaxy is out of scope by user instruction**, so multi-rank pipelined prefill, the KV-chunk-table merge and `G-LOOPBACK`'s two-rank half are unrun and the runtime **raises** on all of them | open — scoped out, not deferred. **P10 made the refusal explicit and tested — at the second attempt.** `tt/runners/kv_chunk_table.py::assert_single_rank_stage` refuses a non-zero `first_layer_idx`, a mismatched `num_my_layers`, a gathered stage list carrying more than one rank, a single stage that is not the whole model, a bare dict and an empty list — six `G-RUNTIME` cases, plus three positive ones. The **first** version of that guard was vacuous on every engine path and would additionally have crashed every real migration run: `DEC-111`. So this row's mitigation was not real until `DEC-111` | user / a multi-galaxy phase |
| R-033 | low | P8 | Both mesh-graph descriptors the recipe names in §The machine are **multi-galaxy** (a 4-galaxy super-pod and a quad galaxy) and unusable on one galaxy | open — affects the kit, not the model (`DEC-071`) | kit maintainer |
| R-034 | medium | P8 | The block-cyclic cache read-back in `tests/galaxy_prefill_kv_pcc.py` **re-derives** the position map instead of importing one, so it and `models/demos/deepseek_v3_d_p/tt/mla/utils.py`'s statement of the same formula can drift | open — mitigated by reading at three different periods and by asserting the map covers every position exactly once | P9 / P10 |
| R-035 | low | P8 | `meta_head_index` is duplicated between the `G-MESH-KV` script and the unit tests, because a script must not import a pytest module | open — pinned by an equality test (`DEC-085`), but the duplication is a wart | P9 |
| R-036 | medium | P8 | No **bit-exact** check exists for K's *post-RoPE* head->column placement; it is impossible by construction and is covered only numerically | open — stated in `G-KV-TP8`'s gate block (`DEC-078`) | accepted |
| R-037 | low | P8 | `G-WEIGHTS`'s P8 arm does not hash the replicated embedding table on devices 1-30 (33.6 GB of D2H per pass) | open — scoped, with the replication itself asserted on the first and last device (`DEC-087`) | accepted |
| R-038 | medium | P8 | `G-RACE`'s pass covers a few hundred collectives on **one** user slot with the barrier ping-pong only 2 deep; it says nothing about a long-running multi-user server | open — the recipe requires this scope statement rather than a fix | perf/serving phase |
| R-040 | low | P8 | Six P8 raw gate logs exceed the repo's own 500 KB `check-large-files` limit, inflated by ~5,000 tt-metal `Pinned source memory start address ... must be aligned` info lines each — not by progress-bar output as `LANDMINES.md` assumes | worked around: gzipped, losslessly, as `LANDMINES.md` itself prescribes | kit maintainer / tt-metal |
| R-041 | medium | P8 | A per-phase regression run **overwrites a previous phase's evidence file**: `G-CHUNK`'s per-layer JSON is rewritten with whatever `$PREFILL_TRACE_DIR` currently holds | worked around (`DEC-091`): restored from the P7 commit, and `G-CHUNK-ATTN` now skips rather than fails on a short trace | P9; kit maintainer (§1.2) |
| R-042 | medium | P8 | Two P5.4-P5.6 ledger citations point at raw logs that **do not exist** and never did; found by the raw-artefact pass this phase added to `verify_citations.py`, which four previous doc gates had no equivalent of | open — deliberately not re-attributed, because guessing which run was meant is what §1.6 warns against | P9 |
| R-039 | **high** | P8 | The `sp_bootstrap` attention core is reachable **only** when `max_seq_len == chunk_global`, which the deployment config (chunk 8192, cache 131072) never satisfies — it has a gate but no deployment use; and the deployment pair itself has never been run | **half closed as of P10** — `G-REQUEST`'s deployment arm serves the real pair (chunk 8192, cache 131072) through the engine end to end, so the pair is no longer unrun. The `sp_bootstrap` half stands: it still has a gate and no deployment use | perf phase (the unused core) |
| R-043 | medium | P10 | `G-LOOPBACK` (the real DRAM->transport->DRAM copy) needs the tt-llm-engine binaries and verifies the **engine's** byte copy rather than this model | **out of scope by `DEC-103`**, not blocked; the residual gap is enumerated in the entry, and `G-KV-TABLE` proves the table the copy reads, bit-exactly | tt-llm-engine build owner (`H7`) |
| R-044 | medium | P10 | Slot cross-talk is invisible: every serving gate ran **one** prompt, so all slots' KV would be byte-identical | open — the table's slot axis *is* proved (`G-KV-TABLE`'s labelled probe + its `next_slot` control); the serving path's is not. Needs a second golden trace | P9 to list; a follow-up phase to close |
| R-045 | medium | P10 | `tt/runners/kv_chunk_table.py` **imports** the address walk from `models/demos/gpt_oss_d_p`, a cross-package dependency on another model's runners module | deliberate (`DEC-099`), guarded by a layout assertion and by `G-KV-TABLE`'s bit-exact read-back | `common/prefill` maintainer (promote it to `common/`) |
| R-046 | low | P10 | The engine's Gate-1 doc gives that gate **one** hook, but it also needs `set_layer_ack_channel`, and the binding it prints omits `PREFILL_ENABLE_LAYER_ACK=1` — so the documented Gate-1 configuration cannot pass Gate 1 | open — logged, nothing filed (`H7`) | `common/prefill` maintainer |
| R-047 | low | P10 | The shared packed-GQA read-back imports `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` from **minimax_m3** and walks every model's cache with it | latent (all three are 32); our side asserts ours against the template's. Fix is one line: read `table.config(id).chunk_n_tokens` | `common/prefill` maintainer |
| R-048 | low | P10 | An exception inside the engine's request loop leaves the runner spinning and **immune to SIGTERM**; it needs `kill -9` and leaves `/dev/shm/tt_h2d_*` behind | open — measured once (`DEC-108`'s failing run); our part is not to raise there | `common/prefill` maintainer; P9 to note the cleanup |
| R-049 | low | P10 | The engine builds and publishes the mock-migration KV chunk table **twice** on the Gate-1 path: the block at `prefill_runner.py:567` and the non-exclusive `elif` at `:691` both run | open — **measured** (two of every publish line in the `G-MOCK-MIG` runner log). Idempotent and atomic, so harmless; it doubles the build and reads like a retry | `common/prefill` maintainer |
| R-050 | low | P10 | The runtime ignores `params.num_links` (`prefill_runner.py:489`) and derives its own from `get_default_num_links`; the two agree at `(4,8)` on Blackhole and diverge on Wormhole and on any single-row mesh | open — no model in this tree consumes the engine's field, so the wart is the field's existence. A `build_runtime` assertion would catch a divergence but would fire on an otherwise-correct Wormhole run | `common/prefill` maintainer; P9 to carry the note |
| R-051 | low | P10 | The runner defaults `PREFILL_NUM_USERS` to **2** and the producer to **1**, for a value the engine's own doc lists among those that must agree | open — every P10 gate set it explicitly, so no measurement is affected; unset, the runner silently allocates and tables a 2-user cache the producer drives one slot of | `common/prefill` maintainer; P9 to note it must be set |
| R-052 | low | P10 | `build_kv_chunk_table` has never run at the deployment capacity: 2,097,152 entries vs the 45,056 `G-MOCK-MIG` built, and `R-049` doubles it | open — a cost, not a correctness gap (`G-KV-TABLE` gates the arithmetic bit-exactly at two periods). One timing run closes it | perf phase |

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
(`models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md:62`; `BRINGUP_RECIPE.md:1834`).
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
(`BRINGUP_RECIPE.md:199`) — and Appendix C item 2 (`BRINGUP_RECIPE.md:1840-1842`) were both
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
(`BRINGUP_RECIPE.md:1793`). §2.3 (`:396-412`) separately measures
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

## R-017 — An out-of-band recipe edit silently invalidated every prose `path:line` into it
**Fact.** `models/demos/common/bringup/BRINGUP_RECIPE.md` grew from **1986 to 2017 lines** in
`cbb38d0aa7a` ("bringup kit: make the seven human-in-the-loop points explicit HUMAN GATES"), a
commit landed **after** P5 was gated and committed (`e3604f24811`). That commit updated the 56
`RCP` entries in this package's `scripts/verify_citations.py` — so the **content-checked** half of
the citation surface is current — and did **not** touch the `BRINGUP_RECIPE.md:NNNN` references
written in prose inside the package's modules, tests and logs. Measured in this package right now:
**146 distinct prose refs** into the recipe, of which **20 land on a blank or purely structural
line** (a fence, a table rule) and are therefore certainly wrong; the rest land on prose and are
individually unverified. `verify_citations.py` reports all of them `resolved`, because pass 2
**range**-checks a doc ref (`R-016`) and a longer file is still long enough.

Two of them can be dated precisely: `tests/unit/test_attention_vs_ref.py` cites
`BRINGUP_RECIPE.md:1793` for `G-ATTN`'s Appendix A threshold, and line 1770 was **already blank** in
P5's own committed tree — while the same commit's `CITES` entry for that row said `1265`/`1818`.
So the prose refs were interpolated at write time (exactly what `R-016` records) *and* then shifted
by the later kit edit.
**Impact.** Bounded but corrosive: no number and no verdict changes, and every load-bearing claim is
also in `CITES`. What is damaged is the thing §1.6 says an unverified citation costs — "worse than
no citation, because it reads as authoritative". A reader following
`BRINGUP_RECIPE.md:1259-1274` from `tt/layer.py` lands 116 lines from the paragraph being cited.
**Status.** open. P6's own refs are correct and **all** content-checked: 427 `CITES` entries, up
from 359, including one per recipe reference P6 makes. P5's and earlier phases' prose refs are
untouched — rewriting a gated phase's files to fix a comment would put its raw-log evidence out of
step with the tree for a cosmetic gain (§0.2 rule 4). The single exception, fixed because it was
wrong rather than stale: `tt/config.py:265` cited
`models/demos/gpt_oss_d_p/tt/model.py:65` for `hf_config.head_dim`, which is at `:64`; it is now in
`CITES` so it cannot drift again.
**How to close.** Three things, none of which this package can do alone:
1. the kit should treat the recipe's line numbers as an interface — either stop citing it by line
   from downstream packages (cite `§2.3.1` / `Gate G-LAYER` instead), or ship an anchor map;
2. `verify_citations.py`'s pass 2 should **content**-check refs into the recipe the way `CITES` does,
   by requiring the citing sentence's backticked identifier to appear near the target (`R-016`'s
   own suggested fix);
3. a kit edit that shifts line numbers should say so in its commit message, since every executed
   run's logs cite it.
Owner: kit maintainer; P9 for the in-package sweep.

## R-018 — A negative control on a residual block is only as strong as the input **scale**
**Fact.** Measured in P6.1 (`DEC-058`), three arms of the same norm-swap control:
`randn` input + random weights -> **0.99864**; `randn` input + **real** layer-0 weights ->
**0.99993**; **real `embed_tokens` rows** + real layer-0 weights -> **0.66830**. The mechanism is
the recipe's own (`BRINGUP_RECIPE.md:1388-1389`): a perturbation of `s` in `y = r + s` is attenuated
by `||y||/||s||`. A norm removes its input's scale, so a sublayer's output magnitude is nearly
independent of the input's while the residual's *is* the input's — and `randn` is ~100x larger than
what layer 0 actually receives (`embed_tokens` rows have an RMS of **0.0106**). The measured
attenuation is ~65x with `randn` and 3.40x with real embeddings.
**Impact.** A control built on a residual block with an out-of-scale input **cannot fail**, and it
reads as a passing control. This is not hypothetical: P6.1's first two attempts both "passed" while
discriminating by 1.4e-3 and 7e-5 respectively. Recipe §2.1(b)'s "input distribution is a red
herring" is correct about the **floor** and does not transfer to a control's power, which is a
different quantity.
**Status.** mitigated in P6.1 (`DEC-058`), open as guidance for later phases.
**How to close.** Every later gate whose control perturbs a residual block must state the input's
**scale** next to its distribution, and drive at least one arm at the scale the model presents to
that block. Owed by: **P7** (`G-CHUNK`'s mutual-PCC controls), **P8** (`G-KV-TP8`'s rotated-column
control at model level, `G-TP-PARITY`), **P10** (`G-KV-TABLE`). The kit should add the scale to
§1.4's four mandatory fields — "input distribution" is currently satisfiable by naming a
distribution alone. Owner: P7/P8/P10 authors; kit maintainer for §1.4.

## R-019 — `transformers` 5.12.1's `output_hidden_states` ends with the **post-norm** stream
**Fact.** For an `n`-layer Llama the tuple is `(embeddings, L0, ..., L[n-2], POST-FINAL-NORM)`:
length `n+1`, and the **last** element is `model.norm`'s output, not the last layer's — which
appears nowhere in it. `LlamaForCausalLM`'s output object also exposes no `last_hidden_state`
(`LlamaModel.forward` builds one at
`python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:421`,
`LlamaForCausalLM` consumes it at `:484`). Verified by hooking every module (`DEC-052`).
**Impact.** Taking `hidden_states[-1]` as a pre-norm layer output and norming it computes RMSNorm
**twice**, which is nearly idempotent and therefore produces a *plausible* number rather than
garbage: it cost P6.3 a debugging pass on a reported hidden-state PCC of **0.9916270** where the
device was correct and measures 0.9997314. The symptom that gave it away is worth recording: the
hidden-state PCC was 20x **worse** than the logits PCC computed from those same hidden states, which
is arithmetically impossible.
**Status.** mitigated in-package: `tests/unit/test_model_vs_ref.py` reads every HF tensor through
explicit forward hooks and passes `output_hidden_states` nowhere (`DEC-052`).
**How to close.** P7's `scripts/generate_golden_kv_cache.py` reads the same oracle and must use
hooks too (it needs per-layer post-RoPE K and raw V, which are *inside* the attention module and not
in that tuple at all, so this is a fortunate constraint rather than a burden). The kit's P1 trap
list should carry it as a sixth trap: it is the same class as trap 3 (`attention_mask=None` is
silently non-causal) — an HF default that produces a plausible wrong answer. Owner: P7; kit
maintainer.

## R-020 — `G-MODEL`'s absolute PCC threshold is scoped to reduced depth by the phase text and to all depths by Appendix A
**Fact.** `BRINGUP_RECIPE.md:1420-1422` attaches "hidden-state PCC >= 0.999, <= 8x the floor, and
top-1 token agreement = 100%" to the **reduced layer count** runs (`n_layers=2`, then 4), and
`:1424-1425` states the full 32-layer run's own gate as "record the per-layer hidden-state PCC curve
into `bringup_log/raw/` and gate the **step** between consecutive layers at **<= 4x** from layer 3
onward". Appendix A's single row (`:1856`) compresses all of it into
">= 0.999, <= 8x floor, per-layer step <= 4x from L3; 100% top-1", which reads as if the absolute
threshold also applied at depth 32.
**Impact.** It decides a verdict. Measured at 32 layers, seq 512, bf8_b weights: the post-final-norm
hidden PCC is **0.9984849** — below 0.999 — while the reduced-depth runs the phase text gates are
0.9997314 (L2/s128) and pass comfortably, the per-layer step never exceeds **1.27x** against a
budget of 4x, top-1 agrees with HF, and the curve is smooth with no step anywhere. Reading the
Appendix row literally makes this a `FAIL` that stops the bring-up (§0 rule 1); reading the phase
text makes it a `PASS` with the absolute number recorded.
**Status.** resolved in-package by `DEC-053`, which follows the phase text, asserts the step and
top-1 as stated, **adds** a measured full-depth floor so the absolute number has a reference rather
than a bare comparison, and records everything either way.
**How to close.** The kit should make Appendix A's `G-MODEL` row say which threshold applies at which
depth — e.g. ">= 0.999 and <= 8x floor **at 2 and 4 layers**; step <= 4x from L3 **at full depth**;
100% top-1". More generally, an absolute PCC threshold that is not annotated with a depth is the
trap `LANDMINES.md` already records one row above it ("a mutual-PCC gate with no stated depth ...
measures depth, not the op") — the same applies to an absolute one. Owner: kit maintainer.

## R-021 — Two gates disagreed about what a floor is, and it was worth 45% of the floor error at depth
**Fact.** `G-LAYER`'s floor quantises the RoPE cos/sin tables to bf16 (`cos_q`, `sin_q` in
`tests/unit/test_decoder_layer_vs_ref.py`), because that is what the device stores —
`tt/rope.py::build_prefill_rope` delegates to `models/tt_transformers/tt/common.py:534`, which builds
**bf16** tensors and takes no dtype argument. `G-MODEL`'s floor was the HF `LlamaForCausalLM` class
itself with device-valued weights, and HF computes its rotary tables **internally in fp32**. So the
model-level floor omitted one rounding the device pays on every layer, while the layer-level floor
did not.

**Measured, at 32 layers / seq 512 / bf8_b weights** (`raw/G-MODEL-H5_20260904T114724Z.log`):

| floor | `1 - floor` | raw ratio of the same measurement |
|---|---|---|
| fp32 RoPE tables (HF's internal, what the gate first used) | 5.4300e-04 | **2.79x** |
| bf16 RoPE tables (what the device holds) | 9.9160e-04 | **1.53x** |

The omitted rounding is **4.4860e-04**, i.e. **45% of the correct floor error** — because the same
tables are applied in all 32 layers, so their rounding accumulates with depth while a
single-layer measurement barely sees it. **This one omission is the entire difference between a
reported 2.79x and 1.53x**, and 2.79x was the number that triggered `HUMAN GATE H5`.
**Impact.** No verdict changes: both 2.79x and 1.53x clear the 8x model budget, and every other
`G-MODEL` assertion (top-1, per-layer step, the control) is untouched. What was wrong is the
**interpretation**: an incomplete floor makes a correct implementation look further from the
arithmetic limit than it is, which is the failure mode §2.2's "keep one definition of the floor
helpers" exists to prevent — and the two definitions here were not two *copies* of a helper, they
were two different **models** of what the device stores, which the shared-helper rule does not catch.
**Status.** **fixed in-package.** `_build_hf_model(quantise=True)` now hooks
`model.model.rotary_emb` and quantises its output, so the model floor and the layer floor agree.
Both numbers are recorded in `06_GATES.md`'s `G-MODEL` block and in `DEC-053`; the corrected floor is
the one the ratio is asserted against, and the correction is a floor *completion* mandated by §2.2,
not a threshold change (no threshold in this package moved).
**How to close.** Two things for the kit. (1) §2.2 should say that "inputs" includes **every constant
the device stores** — RoPE tables, masks, scales — not just activations and weights, and that a
reference implementation which computes such constants internally at higher precision produces an
optimistic floor. (2) The floor's *definition* deserves the same "one definition" treatment as the
helpers: a per-package list of what gets quantised, asserted identical across gates. A cheap
in-package check that would have caught it: the same module's floor computed two ways must agree —
which is exactly the cross-check `test_model_full_depth_attribution` now performs
(`abs(floor_fp32 - 0.9994570) < 1e-6` proves the staged chain reproduces the HF-built floor, and the
bf16/fp32 pair exposes the gap). Owner: kit maintainer; P7/P8 must apply the corrected floor
definition to `G-CHUNK`, `G-MESH-KV` and `G-KV-TP8`.

## R-022 — §2.3.1's additive attribution over-subtracts at 32-layer depth
**Fact.** At layer level the additive model is excellent: floor error plus the fused kernel's excess
predicts the layer PCC to 5-6 decimals, and the attributed residual is 1.13-1.15x (bf8_b) /
2.01-2.14x (bf16) — `DEC-051`. Applied at **model** scale over 32 layers it breaks: substituting the
device's real SDPA into the (otherwise fp32) floor chain at every layer gives a predicted PCC of
**0.9981153**, which is **worse than the device's own 0.9984849**. The implied kernel excess is
8.931e-04 (58.9% of the total measured error), and subtracting it leaves an attributed residual of
**0.63x** — below 1.0, which recipe §2.3 itself calls "a broken floor, not a kernel beating
arithmetic".
**Impact.** The attributed residual is **not** a usable gate quantity at depth, so the number
`G-MODEL` asserts is the raw ratio against a *correct* floor (1.53x), with the residual recorded as
a diagnostic. It would be a real error to read 0.63x as "the model is better than its floor".
**The mechanism, as far as this run can establish it.** The additive model assumes the error sources
are independent. In the real device path the fused kernel receives inputs that are **already**
bf16-rounded and carry 1..k-1 layers of accumulated error, and its own error is partly correlated
with — and partly cancels against — that upstream error. In the substituted chain the kernel
receives fp32-precision inputs, so its error is measured in isolation and then added on top of a
floor that never interacted with it. Over one layer the difference is negligible; over 32 it is
larger than the term being subtracted.
**Status.** open as a method limitation; worked around in-package (assert the raw ratio against the
corrected floor; record the residual and log a warning when it falls below 1.0).
**How to close.** The kit should scope §2.3.1 the way its own amended text scopes the ratio wall —
by measurement, not by presence of the kernel — and add that the attributed residual is validated
**at block and layer scale** and is not known to be additive across many layers. A depth at which it
demonstrably holds would be worth stating. Owner: kit maintainer. Downstream: P7's `G-CHUNK` and P8's
`G-MESH-KV` both compare accumulated 32-layer quantities and should gate on the raw ratio against a
complete floor rather than on an attributed residual.

## R-023 — Delta 3 cannot run in P7, and the runtime refuses it rather than approximating it
**Fact.** A chunked prefill differs from a one-shot in exactly three places
(`BRINGUP_RECIPE.md:1562`): the indexed RoPE and its per-chunk offset (delta 1), the advancing
cache-write offset (delta 2), and **chunk *k*'s queries attending the prefix read back out of the
cache** (delta 3). P7 owns the first two and measured both exactly. Delta 3 needs the ring-joint
SDPA over the block-cyclic cache (`tt/attention/dense_sp.py`, which `raise`s) at TP=8 on the `(4,8)`
mesh — neither of which P7 has.
**Impact.** `G-CHUNK` proves that the chunked KV *producer* is bit-identical to the one-shot
producer, and it proves nothing about whether a chunk-2 query attends chunks 0-1 correctly. The
consequence is bounded and named: after `G-CHUNK`, the KV cache written by a multi-chunk prefill is
verified; the *attention output* of a multi-chunk prefill is not verified at all.
**Why it is not weakened into `G-CHUNK`.** `BRINGUP_RECIPE.md:1656-1660` forbids both available
shortcuts: "Do **not** weaken `G-CHUNK` to cover it, do not move P7 to a multi-device mesh to make
it run". The third option — running a cache-backed chunk on the dense path anyway — is the dangerous
one, because plain `is_causal` SDPA assumes Q row 0 aligns with K row 0 and so produces a mask off
by `actual_start`: a *plausible* wrong answer, not a crash.
**Status.** open by construction. Two refusals stand between this gap and a wrong number:
`tt/attention/prefill.py::attention_forward` refuses `cached_len > 0`, and
`tt/tt_prefill_runtime.py::prefill_chunk` refuses `actual_start > 0` on the dense path with a message
naming P8, `G-CHUNK-ATTN` and this risk id. Both are asserted (`G-RUNTIME`'s
`test_prefill_chunk_refuses_a_cache_backed_chunk_on_the_dense_path`, `G-ATTN`'s own refusal test).
**How to close.** **P8 owns it**, as gate `G-CHUNK-ATTN`: ">= 0.999 at layer 1; deep layers gated by
step <= 4x; both vs golden" (`BRINGUP_RECIPE.md:1981`). It should score against the same trace this
phase generated, so the chunked-attention numbers are comparable to `G-CHUNK`'s producer numbers.
Owner: P8.

## R-024 — Six engine-called runtime hooks are deliberate stubs that raise
**Fact.** The `G-RUNTIME` AST walk over `models/demos/common/prefill/runners/prefill_runner.py`
found nine methods the engine calls on a runtime handle. Three are implemented (`compile`,
`prefill_chunk`, and `make_chunk_input` which the engine does not call but the contract requires).
The other six are present and raise `NotImplementedError`:

| hook | engine call site | guarded? | owner |
|---|---|---|---|
| `build_kv_chunk_table` | `prefill_runner.py:570`, `:644`, `:699` | no | P10 (`G-KV-TABLE`) |
| `kv_migration_base_address` | `:616` | `hasattr` | P10 (`G-LOOPBACK`) |
| `set_layer_ack_channel` | `:768` | no | P10 (`G-REQUEST`) |
| `set_layer_completion_sink` | `:752` | no | not in this iteration (multi-rank) |
| `set_d2h_ack_service` | `:746` | no | not in this iteration (trace) |
| `kv_migration_stages` | `:613`, `:615` | `hasattr` | not implemented (single cache) |

**Impact.** Each of the unguarded four is reachable only when the corresponding engine env var is
set (`PREFILL_ENABLE_MIGRATION`, `PREFILL_ENABLE_LAYER_ACK`, `PREFILL_USE_TRACE`), so a default
single-rank serving run touches none of them. With them set, the run **fails loudly at setup** with
a message naming P10 — which is the intended behaviour and is strictly better than the alternative
that was considered and rejected (`DEC-063`): a `build_kv_chunk_table` that returned its `path`
argument would let a migration run publish an empty table and report success.
**One cost is real and recorded.** Defining `kv_migration_base_address` makes the engine's
`hasattr` at `:616` true, so the engine's own diagnostic at `:619-623` is no longer reachable. The
replacement message is more specific (it names P10 and this risk), so the trade was taken.
**Status.** open — deliberate. All six refusals are asserted on their messages by
`G-RUNTIME`'s `test_unimplemented_engine_hooks_refuse_loudly` and the module's 25 `raise`
statements are counted by `test_every_raise_in_the_module_is_covered`.
**How to close.** P10 implements `build_kv_chunk_table`, `kv_migration_base_address` and
`set_layer_ack_channel` (the last is one line — `tt/model.py`'s `on_layer_complete` seam already
exists, `DEC-050`). The two remaining hooks stay refusals: multi-rank pipeline parallel and
trace/2CQ are explicit non-goals (`BRINGUP_RECIPE.md:16`). Owner: P10.

## R-025 — The recipe describes `verify_golden_kv.py` two incompatible ways, six lines apart
**Fact, quoted.** `BRINGUP_RECIPE.md:1548-1549`: "`scripts/verify_golden_kv.py` — compare a device KV
read-back against the golden, per layer, reporting min/mean PCC per layer for K and V."
`BRINGUP_RECIPE.md:1588-1590`: "**Gate `G-GOLDEN`:** `verify_golden_kv.py` runs clean over all 32
layers and prints a per-layer table ... **It imports no ttnn** — the device-vs-golden scoring lives
in `G-CHUNK`." A file that compares a device read-back must import ttnn.
**Impact.** It decides what the file *is*, and therefore what `G-GOLDEN` measures. Followed
literally, step 2 makes `G-GOLDEN` a second device gate that duplicates `G-CHUNK` and gives two
gates two ways to disagree about the same PCC; followed as the gate states, it is a host-only
structural check with its own negative controls. Half an hour was spent establishing which, and the
tiebreakers were outside the P7 section: Appendix A (`:1953`) gives `G-GOLDEN` device "host (imports
no ttnn)", and both in-repo templates
(`models/demos/minimax_m3/scripts/verify_golden_kv.py:26`,
`models/demos/gpt_oss_d_p/scripts/verify_golden_kv.py:111`) are host-only checkers.
**Status.** resolved in-package (`DEC-060`): the gate text wins. The kit's two passages still
contradict each other.
**How to close.** Rewrite P7 step 2 to match the gate — something like "`scripts/verify_golden_kv.py`
— check the golden trace's structure **and content** over every layer, per-layer table, no ttnn;
the device-vs-golden PCC is `G-CHUNK`'s." Owner: kit maintainer.

## R-026 — The golden trace is 128 MB and lives outside the repo
**Fact.** 32 layers x 512 tokens x 8 KV heads x 128 head_dim x 2 tensors x 4 bytes = **128 MB**, and
the repo's `pre-commit` `check-large-files` hook rejects anything over 500 KB. The trace therefore
lives at `$PREFILL_TRACE_DIR` (this run: `/home/mstojkovic/prefill_traces/llama31_8b_d_p/s512`) and
is not committed.
**Impact.** `G-CHUNK`'s numbers cannot be re-derived from the repo alone. They *can* be re-derived
from the repo plus `$HF_MODEL` by one command in ~100 s, and the inputs are pinned: `metadata.json`
records the 512 `token_ids`, the prompt, the reference dtype policy, and the `rtol=atol=0`
`LlamaModel` cross-check. The generator's own run is in `raw/G-GOLDEN-GEN_20260904T123642Z.log`.
**Why it is not worked around.** The only gitignored directory inside the package is `generated/`,
which is where ttnn writes its inspector, watcher and fabric artifacts; putting a 128 MB reference
trace there would mix evidence with scratch. Compressing does not help — fp32 K/V does not compress.
**Status.** mitigated, not closed (`DEC-066`).
**How to close.** Either accept it (the trace is a derived artifact with a pinned recipe, like a
weight cache), or commit a **small** structural fixture (2 layers, 64 tokens, ~1 MB) so CI can
exercise the plumbing without the reference. The second is not a substitute for the reference.
Owner: P8/P10, which score against the same trace and will feel it first.

## R-027 — The recipe's delta-1 control quotes a V number no correct implementation can produce
**Fact.** `BRINGUP_RECIPE.md:1585-1586` specifies one negative control for `G-CHUNK`: "rope every
chunk at `kv_actual_global = 0` and the mutual PCC must collapse (measured 0.706 / 0.655)". Two
numbers implies two quantities. But `G-CHUNK`'s decomposition — mandated four lines earlier
(`:1571-1574`, "feed **the same hidden states** ... to both KV producers") — feeds both producers
identical hidden states, and **V is never rotated**. So freezing the RoPE offset cannot move V by
any amount.
**Measured.** delta-1 control: mutual **K = 0.72466**, mutual **V = 1.00000**. The K figure agrees
with the recipe's 0.706 to within 3%, which is good evidence this is the same experiment; the second
quoted number is presumably from a variant that re-ran the layer stack per chunk, where a wrong K
corrupts attention and therefore the *next* layer's V.
**Impact.** Small but real: an implementer who takes 0.655 as a target will believe the control is
broken and go looking. Worse, a single control for a two-delta gate leaves delta 2 ungated — and a
dropped `kv_actual` is the most likely chunked-prefill bug there is.
**Status.** resolved in-package (`DEC-065`): two controls, one per delta. delta 2 (write every chunk
at `kv_actual = 0`) measures **K = 0.22048, V = 0.04473**, and the delta-1 control now *asserts* V is
unchanged, turning the "V is never rotated" invariant into a check.
**How to close.** The kit should quote the delta-1 control as a **K-only** discriminator, and require
a second control for delta 2. Owner: kit maintainer.

## R-028 — P7 step 4's named template is the delta-3 test P7 is forbidden to run
**Fact.** `BRINGUP_RECIPE.md:1559-1560` (P7 step 4) says: "`tests/unit/test_attention_chunked_vs_ref.py`
— the chunked-vs-one-shot equivalence test (template:
`models/demos/minimax_m3/tests/unit/test_attention_chunked_vs_ref.py`)". That template runs chunk 0
then chunk 1 **with `cached_len=chunk`**, i.e. it is exactly the cache-read path — delta 3 — on an
`(8,4)` mesh. Twenty-five lines later the same phase says delta 3 "cannot run here", must be recorded
`BLOCKED`, and that P7 must not move to a multi-device mesh to make it run.
**Impact.** An agent following step 4's template will write the delta-3 test, discover
`attention_forward` refuses `cached_len > 0`, and then have to decide whether the refusal or the
recipe is wrong. That is a genuine fork in the road at the point where the phase is most expensive to
back out of.
**Status.** worked around. `tests/unit/test_attention_chunked_vs_ref.py` here implements the
**`G-CHUNK` gate text** (two KV producers on identical hidden states, deltas 1-2), and the minimax
file becomes the template for P8's `tests/unit/test_sp_attention_chunked.py`, which is what
`bringup_log/03_OUTLINE.md` §2.15's gate table already assigned it to.
**How to close.** Step 4 should name the template as P8's and describe the P7 file as the two-producer
comparison the gate text specifies. Owner: kit maintainer.

## R-029 — `TtPrefillRuntime` has never been instantiated
**Fact.** `G-RUNTIME` is a static gate — Appendix A gives it device "none"
(`BRINGUP_RECIPE.md:1977`) — and the runtime refuses construction unless
`tp == num_key_value_heads == 8`, which no `(1,1)` mesh satisfies. So `__init__` past its two
refusals, `_build_indexed_rope`, `make_chunk_input`, `compile` past its rank check, and the whole
happy path of `prefill_chunk` have **not executed**.
**Impact.** This is the largest untested surface P7 leaves. What *is* proved is that the class cannot
fail with a `TypeError` on the engine's call (every unguarded name present, every parameter bindable
— the failure mode recipe P10 warning 1 records, which costs a mesh open and a 15 GB weight load to
discover), that all 25 refusals fire with their stated messages, and that the deployment chunk
arithmetic is right. What is not proved is that any of it *runs*.
**Mitigation that exists.** The parts of the runtime that are pure delegation are covered elsewhere:
the indexed RoPE builder is proved bit-identical to the contiguous one on device (P7's own probe and
`G-CHUNK`, mutual PCC exactly 1.0 on all 32 layers), the cache write is `G-KV`'s, and the model
forward is `G-MODEL`'s. The runtime's own contribution is the wiring between them.
**Status.** open, and stated in the `G-RUNTIME` gate block rather than left implicit.
**How to close.** P8's first act on the `(4,8)` mesh should be to build a `TtPrefillRuntime` and run
`compile()` — which is also the cheapest possible smoke test of the ring path, since `compile` warms
a second chunk at `actual_start = chunk` and therefore exercises delta 3 immediately. Owner: P8.

---

# P8 additions and closures

## R-001 — CLOSED at P8
**How it closed.** `G-KV-TP8` on a `(1,8)` submesh: global KV head `c` lands on mesh column `c`,
asserted `torch.equal` with `rtol=atol=0`, **8/8 columns** on both a V probe with RoPE enabled and a
K probe with RoPE disabled (`DEC-078`); the advancing write offset `{0, 128, 256}` **24/24** blocks
bit-identical; and 32 layers of model-produced K/V scored against the fp32 golden at min K
**0.9986432** / min V **0.9942853**, with layer 0 at **1.10x** / **1.02x** its complete floor. The
rotated-column control scored **PCC 0.99887** while failing bit-equality, reproducing recipe §2.5's
0.99890 on this package's own probe — which is the measurement behind the rule, not a quotation of it.
**Residual:** `R-036` (no bit-exact post-RoPE K placement check).

## R-007 — CLOSED at P8
**How it closed.** `G-MESH-KV` runs the deployment `(4,8)` mesh at `sp = 4`, where the block-cyclic
sequence layout is live rather than the identity, and reads the cache back at **three different
periods** — `chunk_local` 256 (one-shot), 128 and 64 (chunked) — scoring against the fp32 golden every
time (min K 0.9987994 / 0.9967119 / 0.9967844). A read-back with the wrong period cannot score at more
than one. **Residual:** `R-034` (the map is re-derived rather than imported).

## R-023 — CLOSED at P8
**How it closed.** `G-CHUNK-ATTN`. Delta 3 measured on the deployment mesh: one runtime, one
`CCLManager`, two arms (`sp_bootstrap` at chunk 1024, `sp_ring` at chunk 512), 1024 real tokens, real
weights. Mutual K **1.0000000** at layer 0, **0.9999505** at layer 1 against a >= 0.999 threshold,
worst gated per-layer step **2.14x** against 4x, both arms inside `G-CHUNK`'s golden thresholds. The
control — chunk 1 served against a never-written prefix — leaves layer 0 at **1.0000000** and
collapses to **0.87279** by layer 5, which is also the demonstration that a layer-0-only check proves
nothing about delta 3.

## R-029 — CLOSED at P8
**How it closed.** `G-MESH-KV`, `G-CHUNK-ATTN`, `G-RACE` and `G-SEMAPHORE`'s P8 arm all drive
`TtPrefillRuntime` for real (`DEC-084`): `TtPrefillRuntimeConfig.__post_init__`,
`resolve_chunk_sizes`, `_build_indexed_rope`, `make_chunk_input`, `compile()` (including its
second-chunk warm-up at `actual_start = chunk`), `prefill_chunk`'s argument checks and `_resolve_kv`
have now executed on the target mesh. **Residual:** `R-024` — the six engine hooks (migration, ack,
trace) still raise, so nothing on those paths has run. That is P10's, and it is the reason this
closure is about the *prefill* path only.

## R-030 — There is no ring fabric on this Blackhole Galaxy
**Fact.** Three independent measurements, all in `G-FABRIC-MATRIX`'s logs:
1. `ttnn.set_fabric_config(FABRIC_1D_RING)` with the only single-galaxy RING/RING descriptor,
   `single_bh_galaxy_torus_xy_graph_descriptor.textproto`, aborts at
   `tt_metal/fabric/topology_mapper.cpp:544`: "Graph specified in MGD could not fit in the discovered
   physical topology ... Intra-mesh mapping failure for logical mesh 0 -> physical mesh 0: Mapping
   validation failed: **32 target node(s) are not mapped to any global node** ... Either relax
   pinnings or modify the MGD."
2. It is **not** the channel policy: a copy of that descriptor with `policy: STRICT` -> `RELAXED`
   fails identically, so the torus wrap links are absent rather than merely under-provisioned.
3. Asking `FABRIC_1D_RING` of a LINE/LINE or LINE/RING descriptor is refused a step earlier, at
   `tt_metal/fabric/mesh_graph.cpp:447-453`: "FabricConfig can only restrict topology (e.g.,
   torus->mesh), not create new connections."
And the consequence that actually bites: `ttnn.transformer.ring_joint_scaled_dot_product_attention`
under `Topology.Ring` aborts with
`tt_metal/fabric/fabric.cpp:174: forwarding_direction.has_value()` — "Could not find any forwarding
direction from src (M0, D0) to dst (M0, D3)", D0 -> D3 being the 4-device SP ring closing on itself.
**Impact.** The entire phase runs on `FABRIC_1D` + `Topology.Linear`. `ttnn.Topology.Ring`
*collectives* do work on `FABRIC_1D` (bit-exact at every P8 shape and both axes), so the choice is
driven by the ring SDPA alone.
**What the recipe says, and it is wrong here.** `BRINGUP_RECIPE.md:82-84`: "The Ring topology P8
needs the torus descriptor; a Ring topology on a plain `FABRIC_1D` fabric **hangs** rather than
erroring." On this machine the torus descriptor is unusable and Ring-on-`FABRIC_1D` neither hangs nor
errors — it returns bit-exact results.
**Status.** open, worked around (`DEC-079`, `DEC-081`). `PREFILL_FABRIC=1d_ring` is kept as an
override.
**How to close.** Run `./build/test/tt_metal/tt_fabric/test_system_health` (the mapper's own
suggestion) to establish whether the wrap links are physically absent or merely untrained. If absent,
this is a cabling fact and the recipe's §The machine bullet needs a single-galaxy qualifier; if
untrained, it is a machine-health issue with an owner. **Owner:** machine owner, then kit maintainer.

## R-031 — The SP path has been measured under `Linear` only
**Fact.** Every P8 number in this ledger was measured with `CCLManager.topology = Topology.Linear`,
forced by `R-030`.
**Impact.** If the deployment ever runs on a torus-cabled galaxy it will use `Topology.Ring`, and
nothing here has exercised that: the reduce-scatter/all-gather route, the ring SDPA's halo exchange
and the barrier ping-pong all change. The numbers would need re-measuring, not merely re-checking.
**Status.** open; untestable on this machine.
**How to close.** Re-run the P8 ladder with `PREFILL_TOPOLOGY=ring PREFILL_FABRIC=1d_ring` and the
torus descriptor on a galaxy where it maps. Everything needed is already env-driven. **Owner:**
whoever runs a torus-cabled galaxy.

## R-032 — Multi-galaxy is out of scope by user instruction
**Fact.** The scope for this session is **one** Blackhole Galaxy. Multi-rank pipelined prefill needs
two, so anything requiring it was not configured, attempted or planned.
**Impact.** Unrun, and the runtime raises rather than guessing on each:
`set_layer_completion_sink` (the multi-rank layer-completion sink), the multi-rank half of
`build_kv_chunk_table`'s merge, and `G-LOOPBACK`'s cross-rank copy. `TtPrefillRuntimeConfig` pins
`is_first_rank = is_last_rank = True` and `first_layer_idx = 0`, and `compile()` refuses a non-first
rank outright.
**Status.** open — **scoped out, not deferred.** No gate was weakened to accommodate it and no gate
was recorded `BLOCKED` for it, because no P8 gate needs it: every P8 gate is single-rank by
construction.
**How to close.** Two galaxies. **Owner:** user / P10.

## R-033 — The recipe's named mesh descriptors are multi-galaxy
**Fact.** `BRINGUP_RECIPE.md:80-83` names `bh_galaxy_sp4_torus_xy_graph_descriptor.textproto` (4
meshes of `[32,4]`, `host_topology [4,1]` — a 512-device super-pod) and
`32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto` (`[32,4]`, `host_topology [4,1]` — a
128-device quad galaxy) as the descriptors for a machine it describes as "Blackhole Galaxy, 32
devices". The single-galaxy pair is `single_bh_galaxy_mesh_graph_descriptor.textproto` and
`single_bh_galaxy_torus_xy_graph_descriptor.textproto`, both `[8,4]` with `host_topology [1,1]`.
**Impact.** An agent following the bullet literally would try a descriptor that cannot map, and the
resulting `topology_mapper.cpp:544` message names neither the descriptor nor the galaxy count.
**Status.** open — affects the kit, not the model (`DEC-071`). **Owner:** kit maintainer.

## R-034 — The cache read-back re-derives the block-cyclic map
**Fact.** `tests/galaxy_prefill_kv_pcc.py::cache_row_to_global_position` implements
`(lr // chunk_local) * chunk_global + row * chunk_local + (lr % chunk_local)`. The same formula is
*stated* in `models/demos/deepseek_v3_d_p/tt/mla/utils.py:88-92` and *implemented* differently, for
the forward direction, in `block_cyclic_reorder:65-80` — which is what `tt/rope.py::build_indexed_rope`
uses. There is no shared inverse.
**Impact.** If the writer kernel's layout ever changed, the RoPE tables and the read-back would have
to be fixed in two places, and a read-back fixed alone would produce a plausible wrong PCC.
**Status.** open, mitigated twice: the map is asserted to cover every global position **exactly
once** before any scoring, and `G-MESH-KV` reads at three different periods.
**How to close.** Export an inverse from the deepseek util and import it. **Owner:** P9 / P10.

## R-035 — `meta_head_index` is duplicated
**Fact.** `tests/galaxy_prefill_kv_pcc.py::meta_head_index` duplicates
`tests/unit/test_decoder_layer_vs_ref.py::_meta_head_index`, because the former is a script and must
not import a pytest module and the natural shared home (`tests/test_factory.py`) imports `pytest` at
module scope.
**Impact.** A divergence would permute the golden one way in `G-MESH-KV` and another in every unit
gate, and both would look plausible — the same failure mode recipe §2.2 warns about for floor helpers.
**Status.** open, pinned by `test_meta_head_index_does_not_drift_between_the_script_and_the_tests`
(`DEC-085`).
**How to close.** A `pytest`-free helpers module. **Owner:** P9.

## R-036 — No bit-exact check of K's post-RoPE head->column placement
**Fact.** `G-KV-TP8`'s arm A is bit-exact on V through the full path (V is never rotated) and on K
with RoPE disabled. Predicting post-RoPE K bit-exactly would require re-implementing the device's
bf16 RoPE arithmetic on the host — a second implementation of the thing under test.
**Impact.** A hypothetical bug that placed post-RoPE K on the wrong column *without* affecting V or
un-rotated K would be caught only numerically (arm B's min K 0.9986432 over 32 layers, and
`G-CHUNK-ATTN`).
**Status.** open, accepted and stated in the gate block (`DEC-078`). No plausible mechanism for such
a bug exists: the same `column_parallel` mapper places both weights and RoPE is applied after the
head split, per-head.

## R-037 — The replicated embedding table is not hashed on all 32 devices
**Fact.** `G-WEIGHTS`'s P8 arm hashes 32 shards of 11 tensors and the **first and last** device of
`model.embed_tokens.weight`, which is replicated (`DEC-024`) at 1.05 GB per device.
**Impact.** A rebuild that corrupted the embedding on one of devices 1-30 only would not be caught by
this arm. The `(1,1)` arm covers the tensor's contents; the first-and-last pair covers the
replication.
**Status.** open, scoped (`DEC-087`). **How to close.** Hash it in native dtype without the fp32
upcast, or accept.

## R-038 — `G-RACE`'s scope
**Fact.** 3 runs x 2 chunks x 32 layers x 2 collectives per layer on **one** user slot, with the
barrier ping-pong **2 deep** (`DEC-026`) and `reset_global_semaphores` deliberately not resetting the
barrier or ring-attention sets.
**Impact.** A pass says the ping-pong cycling is correct at this scale. It does not cover a
long-running server, hundreds of thousands of collectives, or multiple user slots interleaving.
**Status.** open by construction; the recipe requires the scope statement rather than a fix. The
documented first move on failure — deepening the barrier ring 2 -> 4 — was **not** taken
pre-emptively, so the gate could measure. **Owner:** perf/serving phase.

## R-039 — `sp_bootstrap` has a gate but no deployment use, and the deployment shape has never run
**Fact.** `select_attention_core` selects `sp_bootstrap` only when `max_seq_len == chunk_global`. The
deployment config is `chunk_size = 8192`, `max_seq_len = 131072` (`DEC-061`), so
`max_seq_len > chunk_global` always and the deployment **always** selects `sp_ring`. Separately, that
deployment pair itself has never been run: every P8 measurement used a 1024-token cache.
**Impact.** Two distinct gaps. (a) `sp_bootstrap` is exercised only by test configurations, so a
regression in it would be invisible to a deployment run — the inverse of the usual worry, and the
reason `G-MESH-KV`'s one-shot arm is worth keeping. (b) Nothing has validated 8192-token chunks or a
131072-token cache: the DRAM footprint, the `ttnn.move` guard past 32k tokens
(`tt/layer.py::_MOVE_GUARD_SEQ_LEN`), the `bfloat8_b` activation switch past 32k
(`tt/attention/prefill.py::_BF8_ACTIVATION_SEQ_LEN`) and the 16-chunk `build_indexed_rope` table are
all unrun.
**Status.** open.
**How to close.** (a) decide whether `sp_bootstrap` is a supported configuration or test-only, and
say so in the README's "not implemented" section; (b) run `G-MESH-KV` once at the deployment pair
against a long golden trace. Both are cheap and neither is a P8 gate. **Owner:** P10 / perf phase.

## R-040 — Six P8 raw logs exceed the repo's own 500 KB commit limit, and it is not progress-bar output
**Fact.** `check-large-files` rejects files over 500 KB (`LANDMINES.md`, "Repo hooks that will block
your commit"). Six P8 gate logs are over it: `P8-REGRESSION` (2.1 MB), `G-CHUNK-ATTN` (1.0 MB),
`G-RACE` (876 KB), `G-MESH-KV-chunked256` (863 KB), `G-MESH-KV-chunked512` (851 KB), `G-SEMAPHORE`
(840 KB).
**Cause.** Not the tests' own output and not a progress bar, which is what `LANDMINES.md` blames.
It is one tt-metal `info` line per host<->device transfer:
`Metal | Pinned source memory start address 0x... must be aligned 64 B (dispatch.cpp:...)`. In
`G-SEMAPHORE`'s log that pattern accounts for **5,120 of its lines**, and it scales with the device
count — so every 32-device gate log is inflated by ~800 KB of it regardless of what the gate does.
**Impact.** Cosmetic for this session (nothing is committed) but a commit blocker for whoever lands
this, and it makes the logs slow to read.
**Status.** worked around exactly as `LANDMINES.md` prescribes: the six are **gzipped**, which is
lossless, so the evidence stays byte-exact, and the ledger cites the `.log.gz` names.
**How to close.** Either raise the metal log level for gate runs (which would make the raw log a
*filtered* record and is therefore worse), or fix the `dispatch.cpp` info line upstream — it reads
like a warning that fires on the normal path. **Owner:** kit maintainer (the `LANDMINES.md` note
naming progress bars as the cause), then a tt-metal owner.

## R-041 — A per-phase regression run can overwrite a previous phase's evidence file
**Fact.** `tests/unit/test_attention_chunked_vs_ref.py` (P7, `G-CHUNK`) writes
`bringup_log/raw/G-CHUNK_per_layer_pcc.json` on every run, with content derived from whatever
`$PREFILL_TRACE_DIR` points at. `G-CHUNK-ATTN` (P8) needs a **1024**-token trace where `G-CHUNK` was
recorded against a **512**-token one, so a P8 regression run at `s1024` rewrote P7's file with
1024-token content — 211 changed lines — while P7's ledger row still cites the 512-token
measurement. `raw/G-MODEL_per_layer_pcc.json` was also touched (a trailing newline).
**Impact.** A `PASS` recorded in an earlier phase can have its evidence file silently replaced by a
later phase's run. The recipe's §1.2 raw-output rule covers `tee`d `.log` files, which are
timestamped and therefore collision-free; it says nothing about the derived `.json` tables gates
write, which are not.
**Status.** worked around: both files restored from the P7 commit after the final P8 regression, and
`G-CHUNK-ATTN` now **skips** rather than fails on a short trace (`DEC-091`).
**How to close.** Put the discriminating parameters in the filename — `G-CHUNK_s512_c128_per_layer_pcc.json`
— and update the citing ledger row in the same change. That is a two-file edit spanning two phases'
records, so it belongs to P9's cleanliness pass rather than to P8. **Owner:** P9; kit maintainer for
the §1.2 rule.

## R-042 — Two P5 ledger citations point at raw logs that do not exist
**Fact.** `06_GATES.md:894` cites `P5-REGRESSION_20260904T100738Z.log` for "93 passed, 0 failed"
and `:896` cites `G-CITE_20260904T101355Z.log` for "330/330 verified ... 629/629 doc refs
resolved". Neither file is on disk and neither is in git history. `raw/` holds a P5 regression pair
at `T092212Z` / `T102256Z` and a `G-CITE` pair at `T092334Z` / `T102606Z`.
**How it was found.** `verify_citations.py`'s **pass 3**, added in P8 (`G-CITE (P8)`): passes 1 and 2
both key on `path:line`, so a bare a bare backticked `raw/`-relative log name reference — the ledger's own evidence citation
format — was scanned by neither, and four previous doc gates passed clean over these two lines.
**Impact.** Two claims in the P5.4-P5.6 status block have no retrievable evidence. The recipe's rule
is unambiguous: "A gate with no raw log did not happen" (`BRINGUP_RECIPE.md:199`). The *gates*
themselves (`G-MLP`, `G-ATTN`, `G-KV`) each cite their own logs and those exist — what is
unevidenced is the **per-phase regression count** and the **citation count** at that point, not a
numeric gate.
**Status.** open, and deliberately **not** re-attributed. Pointing the two claims at the surviving
`T102256Z` / `T102606Z` logs would assert that those carry the same numbers, which this session
cannot know; §1.6's whole argument is that a wrong-but-plausible citation is worse than none.
**How to close.** Whoever owns the P5 record either re-runs the two commands and appends a new status
entry (append-only, §1.1), or appends a note saying the two logs were lost. Either closes it; editing
the old line does not. **Owner:** P9.

---

## R-043 — `G-LOOPBACK` is out of scope, and these properties are therefore unproven
- **Severity:** medium.
- **Phase opened:** P10. **Status:** open — **scoped out by `DEC-103`, not deferred**.
- **What it is.** The real DRAM -> transport -> DRAM migration copy (the doc's Gate 2) needs the
  tt-llm-engine binaries `migration_endpoint` and `migration_worker`, a `_migration_client*.so`, and
  an MPI launcher able to place two worker slots on this host
  (`models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md:456-461`, `:587-596`). None is in
  this repository. `HUMAN GATE H4` ("ask whose bug a red gate would be") answers *the engine's*, and
  the doc agrees: the gate "verifies the *engine's* model-agnostic byte copy, not this model".
- **What stays unproven, itemised** — this is the residual gap `BRINGUP_RECIPE.md:1968-1975` requires
  a scoped-out gate to enumerate:
  1. the `MigrationLayerClient` attach and the `WORKER_READY` handshake
     (`models/demos/common/prefill/runners/migration.py:270-280`); this package calls neither — the
     engine does — so nothing here has exercised the publish path end to end;
  2. that a real migration worker can read the addresses this table publishes. `G-KV-TABLE` proves
     the addresses are right *and readable over the same UMD path the worker uses*, which is the
     strongest available substitute, but not the worker itself;
  3. ~~`kv_migration_base_address`'s return value in anger~~ — **closed** by `G-MOCK-MIG` arm 2
     (`PREFILL_ENABLE_MIGRATION=1` + `PREFILL_MOCK_MIGRATION=1`), which calls it at
     `prefill_runner.py:617`, all-gathers the stage layout at `:626` and builds the merged table at
     `:644`, with no worker binaries. What stays unproven is only whether a **worker** is happy with
     the anchor; the stage layout it feeds describes K only (`DEC-107`);
  4. the destination read-back (`--verify-migration dst-bytes`), which the doc says needs **no**
     model-specific surface (`PREFILL_MIGRATION_TESTING.md:544`) — so a failure there would not be
     ours;
  5. cross-endpoint prefill->decode migration, which the driver skips even when run
     (`PREFILL_MIGRATION_TESTING.md:298-300`).
- **Mitigation in place.** `G-MOCK-MIG` proves the KV bytes and the table together, in a second
  process, with a second reader — and its **arm 2** now drives the engine's real stage-gather branch,
  so `allgather_kv_stage_layouts`, `kv_migration_base_address` and the merged-table build at
  `prefill_runner.py:644` have all executed. `G-KV-TABLE` proves the table alone, bit-exactly, at two
  block-cyclic periods, with five discriminating controls. The unimplemented multi-rank merge
  **raises** (`R-032`).
- **How much of Gate 2 is left, after arm 2.** Only the transport: `publish_serialized_table_and_wait_ready`
  (`migration.py:270-280`), the two worker processes, and the destination read-back. Everything the
  runner does on the real path short of the publish has now run.
- **How to close.** Build the two binaries against this tt-metal tree, export
  `PREFILL_MIGRATION_CLIENT_DIR`, and run `PREFILL_MIGRATION_TESTING.md` Gate 2 with
  `--verify-migration both`. Expect the first failure to be the `wait_ready` timeout at `:599-622`,
  which is a launcher problem.
- **Owner:** whoever owns the tt-llm-engine build on this box (`HUMAN GATE H7`).

## R-044 — Slot cross-talk is invisible: every gate ran one prompt
- **Severity:** medium.
- **Phase opened:** P10. **Status:** open.
- **What it is.** With one prompt every slot's KV is byte-identical, so a copy — or a table lookup —
  landing in the wrong slot is indistinguishable from a correct one
  (`PREFILL_MIGRATION_TESTING.md:301-304`). `G-MOCK-MIG` ran `PREFILL_NUM_USERS=1` (`DEC-106`), so
  the question does not even arise there.
- **What partly covers it.** `G-KV-TABLE`'s probe labels the slot index in its own 32-lane field and
  its `next_slot` control confirms slot 0 and slot 1 read different bytes — so the *address table's*
  slot axis is proved. What is not proved is the **serving** path's slot handling: that a chunk
  pushed for slot 1 lands in slot 1's rows.
- **How to close.** Generate a second golden trace from a different prompt
  (`scripts/generate_golden_kv_cache.py`) and run the producer with
  `PREFILL_PRODUCER_SLOT_TRACES=<dirA>,<dirB>`, `PREFILL_NUM_USERS=2`,
  `PREFILL_PRODUCER_MAX_REQUESTS=2` — each slot is then PCC'd against its own golden
  (`PREFILL_MIGRATION_TESTING.md:215-232`). Cheap: one generator run (~100 s) plus one gate run.
- **Owner:** P9 (to list it) / a follow-up phase (to close it).

## R-045 — `tt/runners/kv_chunk_table.py` imports another model package's address walk
- **Severity:** medium.
- **Phase opened:** P10. **Status:** open — deliberate (`DEC-099`), with two guards.
- **What it is.** The block-cyclic bank walk is imported from
  `models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:66` rather than copied, because P5.6 kept
  this cache structurally identical to gpt-oss's for exactly that purpose. It is a dependency from
  one model package into another model package's *runners* module, which is a layering smell, and an
  upstream change to that walk lands here unannounced.
- **Guards.** `_assert_layout_still_shared()` raises if the two packages'
  `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` ever diverge; `G-KV-TABLE` reads every address back over UMD
  and compares bit-exactly, so a drifted walk fails as a wrong address rather than a worse PCC.
- **How to close.** Promote the builder to `models/demos/common/prefill/runners/` — where
  `serialize_prebuilt_kv_chunk_table` already lives — and have both packages import it from there.
  That is a shared-code change with an owner outside this package (`HUMAN GATE H7`).
- **Owner:** the `common/prefill` maintainer.

## R-046 — The engine's Gate-1 documentation omits a hook the gate cannot run without
- **Severity:** low (documentation), but it costs a run.
- **Phase opened:** P10. **Status:** open — **filed nothing** (`HUMAN GATE H7`: "log them; file nothing").
- **What it is.** `PREFILL_MIGRATION_TESTING.md:539-545` tabulates the runtime hooks each gate needs
  and gives Gate 1 exactly one: `build_kv_chunk_table`. Gate 1 also needs `set_layer_ack_channel`:
  with `PREFILL_PRODUCER_CHECK_PCC=1` the producer **exits 1** if the LayerAck channel is absent
  (`prefill_producer.py:1065-1071`, "UMD read would race the runner's prefill"), and that channel
  exists only when the runner calls the hook (`prefill_runner.py:767-768`). Compounding it, the
  documented Gate-1 binding (`:421-425`) does not set `PREFILL_ENABLE_LAYER_ACK`, which defaults to
  `PREFILL_ENABLE_MIGRATION` — i.e. **0** on the mock path (`prefill_runner.py:552-554`). So the
  configuration the doc gives for Gate 1 cannot pass Gate 1.
- **How to close.** Two lines in that document: add `set_layer_ack_channel` to the Gate-1 row, and
  `PREFILL_ENABLE_LAYER_ACK: "1"` to the Gate-1 binding block.
- **Owner:** the `common/prefill` maintainer.

## R-047 — The shared packed-GQA reader takes one model's DRAM block constant to read every model's cache
- **Severity:** low today, latent.
- **Phase opened:** P10. **Status:** open — not changed by this session.
- **What it is.** `_read_kv_slice` (`models/demos/common/prefill/runners/prefill_producer.py:531`)
  imports `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` from `models.demos.minimax_m3`, and every layout
  branch — including the packed-GQA one this model now shares (`DEC-104`) — walks positions with it.
  All three packages set it to 32, so nothing is wrong today; if MiniMax-M3 ever changes it, the
  reader silently steps through *our* cache at the wrong stride and the PCC degrades rather than
  failing.
- **Mitigation on our side.** `tt/runners/kv_chunk_table.py::_assert_layout_still_shared` raises if
  our constant and the template's diverge, and the table itself writes `chunk_n_tokens` into every
  config — so a reader that honoured the table rather than an import would be immune.
- **How to close.** Read `table.config(config_id).chunk_n_tokens` instead of importing a constant.
  One line in shared code, and it makes the reader self-describing.
- **Owner:** the `common/prefill` maintainer.

## R-048 — An exception inside the engine's request loop leaves the runner unkillable by SIGTERM
- **Severity:** low (developer ergonomics), measured.
- **Phase opened:** P10. **Status:** open.
- **What it is.** When `prefill_chunk` raised on the first served chunk (`DEC-108`), the traceback
  reached the top of `main()` and the process then sat at ~125% CPU indefinitely. `pkill` (SIGTERM)
  did not end it — the engine installs a SIGTERM handler that only sets `_shutdown`
  (`prefill_runner.py:99-101`), and the loop it guards has already exited. It needed `kill -9`, and
  it left `/dev/shm/tt_h2d_*` and `tt_socket_manifest_*` behind for manual cleanup.
- **Why it matters here:** on a shared box that is an orphaned 32-device holder; on this one it cost
  a few minutes and a manual `rm -f /dev/shm/tt_h2d_*`. Any session driving these gates should check
  for a stray `prefill_runner` before opening the mesh again.
- **How to close.** Upstream: teardown in `_serve_request`'s `finally` should not depend on the loop
  exiting cleanly. Locally: nothing — our part is not to raise there, which `DEC-108` fixed.
- **Owner:** the `common/prefill` maintainer; **and P9**, which should note the cleanup command.

## R-049 — The engine builds and publishes the mock-migration table **twice**
- **Severity:** low, and it is a real defect rather than a cosmetic one.
- **Phase opened:** P10. **Status:** open — **measured**, not inferred from reading.
- **What it is.** On the `PREFILL_MOCK_MIGRATION=1, PREFILL_ENABLE_MIGRATION=0` path — which is the
  doc's Gate 1, i.e. the configuration `G-MOCK-MIG` runs — the runner builds the KV chunk table and
  serializes the device map **twice**. The block at `prefill_runner.py:567`
  (`if _mock_migration and not _migration_enabled:`) does it, and the `elif` at `:691` belongs to
  `if _migration_enabled:` rather than to that first block, so nothing skips it and it does the same
  work again.
- **Evidence.** `raw/G-MOCK-MIG-runner_20260904T173939Z.log.gz` carries two
  `[gpt-oss-d-p-kv-table] multi-config table built ... entries=45056` lines, two
  `[migration] KV chunk address table serialized to ...` lines, two
  `[migration] device map (32 chips) serialized to ...` lines, and two `[mock-migration]` summaries —
  the second set from `_serve_request:702` rather than `:573`.
- **Why it is harmless here.** Both builds produce the same table and
  `serialize_prebuilt_kv_chunk_table` writes atomically (`migration.py:39-42`), so the file a polling
  reader imports is correct either way, and the device map is overwritten with identical content.
- **Why it is worth recording anyway.** It doubles a build that is `O(configs x layers x slots x
  positions)` — 45056 entries here, and ~5.9 M at the deployment capacity — and the duplicated log
  lines read like a retry, which is exactly the wrong impression to give a reader debugging a
  migration run. It also means the mock path's timing is not the real path's.
- **How to close.** One line upstream: make `:691`'s `elif` exclusive of the `:567` block (or delete
  the `:567` block, whose only extra behaviour is `remove_stale_device_map_sidecars`).
- **Owner:** the `common/prefill` maintainer (`HUMAN GATE H7`: logged, nothing filed).

## R-050 — The runtime ignores `params.num_links` and derives its own; the two agree only here
- **Severity:** low on this machine, latent elsewhere.
- **Phase opened:** P10. **Status:** open.
- **What it is.** The engine computes `num_links = 2 if is_blackhole() else 1`
  (`prefill_runner.py:489`) and hands it to the adapter on `PrefillRunParams`. This adapter does not
  pass it on: `TtPrefillRuntimeConfig` has no such field, and `TtPrefillRuntime.__init__` builds its
  `CCLManager` with `get_default_num_links(mesh_device)`
  (`models/demos/gpt_oss_d_p/utils/general_utils.py:27`), which returns **1** for a single-row mesh
  and otherwise **2** on Blackhole / **4** on Wormhole.
- **Where they agree and where they do not.** On the deployment `(4,8)` Blackhole mesh both say
  **2**, which is why nothing is wrong today and every P8 and P10 number stands. They diverge on
  **Wormhole** (engine 1, helper 4) and on any **single-row** mesh (engine 2 on Blackhole, helper 1).
- **Why it is not simply a bug.** The reference adapter ignores the field too
  (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:128-145` passes no `num_links`), so the
  engine's value is not consumed by any model in this tree — the collectives' link count is the
  runtime's business and `get_default_num_links` is the repo's converged answer to it. What the
  engine's field buys is unclear, and that is the wart.
- **How to close.** Either thread `params.num_links` into `TtPrefillRuntimeConfig` and let the engine
  own it, or delete the field from `PrefillRunParams`. Both are shared-code decisions.
  Meanwhile a one-line assertion in `build_runtime` — `params.num_links == get_default_num_links(...)`
  — would turn a silent divergence into a startup failure; it is not added here because it would
  fire on a Wormhole run that is otherwise correct, and this iteration has never run on Wormhole.
- **Owner:** the `common/prefill` maintainer; P9 to carry the note.

## R-051 — The runner and the producer disagree on the default `PREFILL_NUM_USERS`
- **Severity:** low, and it wastes DRAM rather than producing a wrong answer.
- **Phase opened:** P10. **Status:** open.
- **What it is.** `PREFILL_MIGRATION_TESTING.md:552-568` lists the slot count among the values that
  **must agree** across processes, and the two processes default it differently: the runner to **2**
  (`prefill_runner.py:77`) and the producer to **1** (`prefill_producer.py:361`). Left unset, the
  runner allocates and tables a two-user cache that the producer only ever drives slot 0 of —
  double the KV DRAM and double the address table, silently.
- **Why the model manifest does not pin it.** `PREFILL_NUM_USERS` is the **caller's**, not the
  model's: a deployment chooses how many concurrent users to serve, and recipe P10 step 2 says the
  manifest must pin "nothing that belongs to the caller". Every gate in this phase set it explicitly
  (`PREFILL_NUM_USERS=1`), so no measurement here is affected.
- **How to close.** One default, in the engine, in one place. Until then: set it explicitly on both
  sides, which the doc's own "must agree" table already tells you to do.
- **Owner:** the `common/prefill` maintainer; P9 to note that it must be set explicitly.

## R-052 — `build_kv_chunk_table` at the deployment capacity has never been run
- **Severity:** low (a cost, not a correctness gap), but it is unmeasured.
- **Phase opened:** P10. **Status:** open.
- **What it is.** The table has one entry per `(config, layer, slot, 32-token position)`, so at the
  deployment geometry (16 configs, 32 layers, 1 user, 131072 tokens => 4096 positions) it is
  **2,097,152** entries, each set by a Python-level `table.set()` call inside the imported walk
  (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:160-169`) — and `R-049` means the
  mock path builds it **twice**. `G-MOCK-MIG` ran at capacity 2816, i.e. **45,056** entries, which
  was instant; `G-REQUEST`'s deployment arm ran at the real capacity but with migration **off**, so
  no table was built. Nothing has measured the build at 46x that size.
- **Why it is not a correctness gap.** The walk is the same arithmetic at any size, and
  `G-KV-TABLE` gates it bit-exactly at two block-cyclic periods. What is unknown is the wall time
  and whether a migration-enabled runner start at deployment capacity is tolerable.
- **How to close.** One run: `PREFILL_ENABLE_MIGRATION=1 PREFILL_MOCK_MIGRATION=1` at
  `PREFILL_CHUNK_SIZE=8192 PREFILL_MAX_SEQ_LEN=131072`, timing the two `[llama31-8b-d-p-kv-table]`
  log lines. No PCC is possible there (no golden that deep — `R-026`), so it is a timing measurement
  only, which is why it was not made a gate.
- **Owner:** perf phase.
