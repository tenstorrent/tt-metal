<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 07 — Risks, open questions, `UNVERIFIED` facts

One row per risk. `Owner` is a slot for a human name. `Status` ∈ open / mitigated / closed / blocked.

| Id | Severity | Phase found | Summary | Status | Owner |
|---|---|---|---|---|---|
| `R-001` | **high** | P0 | Model identity is an assumption: `llama31_8b` ⇒ Llama-3.1-8B-Instruct | open — **needs user confirmation** | _(unassigned)_ |
| `R-002` | medium | P0/P1 | Installed `transformers` is 5.12.1; `config.rope_theta` is `None` (moved into `rope_parameters`) | mitigated (use `get_rope_theta`) | _(unassigned)_ |
| `R-003` | **high** | P0 | No checkpoint anywhere on this machine → all real-weight gates are `BLOCKED` | blocked | _(unassigned)_ |
| `R-004` | medium | P0 | TP=8 ⇒ **1 KV head per chip**; no op in the SDPA/KV-write path checked for a `num_kv_heads > 1` assumption | partially mitigated | _(unassigned)_ |
| `R-005` | medium | P1 | `tt_transformers.ModelArgs` **raises** without `HF_MODEL`, so the recipe's preferred `reference_*` oracles are unreachable here | mitigated (`DEC-004`) | _(unassigned)_ |
| `R-006` | medium | P2 | `compute_llama3_parameters` **hard-codes** `low_freq_factor=1` / `high_freq_factor=4` — it does *not* read them from `config.json` | mitigated for this model; latent for others | _(unassigned)_ |
| `R-007` | medium | P2 | `gpt_oss_d_p`'s distributed-RMSNorm branch is **hard-disabled upstream**, i.e. never exercised — Llama's scheme-B path would be first use | open | _(unassigned)_ |
| `R-008` | low | P2 | `ProgramConfig.get_prefill_sdpa_config` hard-codes an 8×8 core grid; Blackhole is wider | open | _(unassigned)_ |
| `R-009` | low | P2 | `MeshConfig` is duplicated in two packages with **divergent** feature sets; neither is importable-as-is | open | _(unassigned)_ |
| `R-010` | medium | P2 | `get_rot_transformation_mat(dhead)` **ignores its argument** and hard-codes `dhead = 32` | mitigated (informational) | _(unassigned)_ |
| `R-011` | low | P0/P1 | `test_factory.TestFactory.setup_test` cannot run until P5 creates `tt/config.py` + `tt/ccl.py` | open by design | _(unassigned)_ |
| `R-012` | medium | P4 | `G-TP-PARITY` runs on `(1,N)` meshes, which use `num_links=1` + `Topology.Linear`; the 2-link `Ring` path is exercised **only** by `G-MESH-KV`/`G-RACE` on `(4,8)` | open | _(unassigned)_ |
| `R-013` | medium | P4 | The barrier-semaphore ping-pong is only **2** deep (a one-op gap), and `reset_global_semaphores` deliberately skips the barrier and ring-attention semaphores that chunked prefill now reuses across chunks | **deferred to P8 by `DEC-052`** — unchanged in P7 and *not tested* there (at `(1,1)` no collective runs at all); if `G-RACE` fails, first move is deepening the ping-pong 2→4 | _(unassigned)_ |
| `R-014` | medium | P3 | **`R-002` is factually wrong** as measured: `cfg.rope_theta` does not exist (raises `AttributeError`), `cfg.rope_scaling` is a full dict, and `getattr(cfg, "rope_theta", D)` returns `D` — so the hazard is a *silent default*, not a silent `None` | mitigated (`DEC-010`) | _(unassigned)_ |
| `R-015` | low | P3 | **`DEC-006`'s stated premise is false**: `gpt_oss_d_p` and `minimax_m3` both cross-import `models.demos.deepseek_v3_d_p.tt.*` extensively, so "no demo package imports another demo's `tt/`" is not the tree's convention | open (informational) | _(unassigned)_ |
| `R-017` | medium | P3 | The tilized **weight cache is mesh-shape dependent** but no gate checks a cache-only rebuild at TP>1: `G-WEIGHTS` runs on 1 card, and `TestFactory.setup_test` takes a raw `tensor_cache_path` with no mesh in it | open | _(unassigned)_ |
| `R-016` | medium | P3 | **`R-008`'s proposed fix would break P8**: deriving the SDPA program grid from `compute_with_storage_grid_size()` (measured 12×10) violates the ring-joint assert `ccl_core_grid_offset.x >= sdpa_grid.x` at offset 11 | mitigated (`DEC-012`) | _(unassigned)_ |
| `R-026` | medium | P7 | `tokenizer.apply_chat_template(..., tokenize=True)` returns a **`BatchEncoding`** on transformers 5.12.1 (`return_dict` now defaults to `True`), so `list(...)` yields the dict KEYS — a plausible 2-element "token list" of strings | mitigated (`return_dict=False` + an int assert); latent in `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:180` | _(unassigned)_ |
| `R-027` | **high** | P7 | The packed KV cache is **one KV head per chip**, so `TP` must equal `num_key_value_heads` (8). No model-level KV write is possible at `(1,1)` — it dies in a C++ `TT_FATAL`. `G-KV`'s `(1,1)` coverage used `nkv = tp = 1`, a head count the model never produces there | mitigated by a loud runtime assert; **the coverage hole is open and is P8's** | _(unassigned)_ |
| `R-028` | **high** | P7 | Chunked cache-read attention (`cached_len > 0`) is unimplemented on a single device, so `G-CHUNK`'s attention-core third is `BLOCKED`; it needs `tt/attention/dense_sp.py` (P8) or a paged chunked SDPA | open — blocks `G-CHUNK-ATTN` | _(unassigned)_ |
| `R-029` | medium | P7 | `TtPrefillRuntime.gather_layer` / `dump_slot_kv` / `kv_cache_pcc_check` are **never executed on device** in P7 (they need TP=8); only their format contract is asserted | open — P8's `G-MESH-KV` is the first real exercise | _(unassigned)_ |
| `R-030` | medium | P7 | `TtPrefillRuntime.build_kv_chunk_table` **raises**; the KV chunk-address table is P10's `tt/runners/kv_chunk_table.py`, so `PREFILL_ENABLE_MIGRATION=1` cannot work yet | open by design — P10 | _(unassigned)_ |

---

## R-001 — Model identity is an assumption

**Fact.** No public "Llama-3.2 8B" exists. The in-tree Llama-3.2 family is 1B / 3B (text) and
11B / 90B (Vision) — `ls models/tt_transformers/model_params/` shows no `Llama-3.2-8B*`. The only 8B
Llama in the tree is `Llama-3.1-8B-Instruct`.

**Decision taken.** `DEC-001`: proceed on `meta-llama/Llama-3.1-8B-Instruct` dims. The recipe
explicitly instructs not to stall (`BRINGUP_RECIPE.md:233-234`).

**What the user must confirm.** Whether the intended target is Llama-3.1-8B-Instruct. If it is a
Llama-3.2 *text* model instead, the delta is exactly three config keys
(`00_MODEL_CARD.md` §1): `rope_scaling.factor` (8.0 → 32.0), `tie_word_embeddings` (false → true),
and an explicit `head_dim`. No structural code changes.

**Mitigation now in place.** All three of those values are read from the config at runtime and none
is hard-coded, so retargeting is a `configs/<Name>/config.json` swap. **P3+ must preserve this**:
never inline `8.0`, never assume `lm_head.weight` exists as a distinct tensor without checking
`tie_word_embeddings`, never inline `128` for `head_dim`.

---

## R-002 — `transformers` 5.12.1 moved `rope_theta` into `rope_parameters`

**Fact, measured.** With `transformers 5.12.1` installed
(`python_env/lib/python3.12/site-packages/transformers/`):

```
LlamaConfig.from_pretrained('models/tt_transformers/model_params/Llama-3.1-8B-Instruct')
  .rope_theta      -> None                      # <-- attribute EXISTS and is None
  .rope_parameters -> {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
                       'original_max_position_embeddings': 8192, 'rope_type': 'llama3',
                       'rope_theta': 500000.0}
  .rope_scaling    -> (aliased to rope_parameters, same dict)
```

The bundled `config.json` was authored for `transformers_version: 4.42.3`
(`Llama-3.1-8B-Instruct/config.json:35`) where `rope_theta` was top-level.

**Why this is dangerous.** `getattr(cfg, "rope_theta", DEFAULT)` returns **`None`**, not `DEFAULT`,
because the attribute exists. A silent `None` θ produces a RoPE that is wrong at every position —
exactly the Appendix B "Attention PCC ~0.5–0.9, norms fine" failure mode, but with no exception.
`models/demos/gpt_oss_d_p/tt/model_config.py:76` and
`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:185` both use that `getattr(..., default)` shape
and are therefore latently affected under transformers 5.x. **Do not copy that pattern into
`llama31_8b_d_p`.**

**Mitigation.** The repo already has the correct helper, added for exactly this:
`models/tt_transformers/tt/common.py:165` `def get_rope_theta(config: dict, default=None)` — it
checks top-level `rope_theta`, then `rope_parameters["rope_theta"]` (flat, Qwen/Llama), then
`rope_parameters["full_attention"]["rope_theta"]` (Gemma-style). Note it takes a **dict**, not a
config object. The explanatory comment is at `common.py:160-163`. Sibling helper `get_rope_scaling`
at `common.py:183`. **P3+ must route every θ read through `get_rope_theta`** and must assert the
result is not `None`. This package reads dims from the raw JSON dict, which keeps `rope_theta`
top-level and sidesteps the issue for dimension-only paths.

---

## R-003 — No checkpoint on this machine: real-weight gates are `BLOCKED`

**Fact.** `HF_MODEL` is empty; the orchestrating session verified nothing under `/proj_sw`,
`/mnt/MLPerf`, or `~/.cache/huggingface`. No safetensors for any Llama exist here.

**What is unaffected.** Every module-level PCC gate (`G-RMS`, `G-ROPE`, `G-MLP`, `G-ATTN`, `G-KV`,
`G-LAYER`), because the recipe's own pattern drives the torch reference and the TT module from
*identical random weights* (`BRINGUP_RECIPE.md:305-308`, `:588`). `G-REF` is unaffected for the same
reason. Key-*name* work is unaffected: `LlamaForCausalLM(config)` / `from_config` yields a randomly
initialised model with the real state-dict key names.

**What is `BLOCKED`, and must be recorded as `BLOCKED` rather than `PASS`:**

| Gate | Why blocked |
|---|---|
| `G-WEIGHTS` (real-checkpoint key consumption) | needs actual safetensors to prove no key is silently unused/missing. The *structural* half (cache-only rebuild) can run on random weights. |
| `G-GOLDEN` | `scripts/generate_golden_kv_cache.py` must run the real reference on real weights |
| `G-CHUNK` / `G-MESH-KV` vs golden | depend on `G-GOLDEN` |
| `G-MODEL` top-1 token agreement | random weights give meaningless tokens; hidden-state PCC on random weights is still meaningful |
| `G-REQUEST`, `G-MOCK-MIG` | need a populated weight cache + staged golden trace |

**Do not attempt a multi-GB download.** Instruction from the orchestrating session.

**Mitigation.** Structure every test so the real-weight case is a `pytest.mark.skipif` on `HF_MODEL`
(the `requires_hf_reference` marker, `tests/test_factory.py`), not a hard failure — so the same test
file goes green on this machine and covers more on a machine with weights.

---

## R-004 — TP = 8 gives exactly one KV head per chip

`8 num_key_value_heads / TP 8 = 1`. Nothing in the SDPA or KV-write path has been *executed* at
`num_kv_heads = 1`; only read.

**Partial mitigation — the SDPA op is confirmed safe.** `ttnn.transformer.scaled_dot_product_attention`
supports GQA natively and its only head-count constraint is
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:97-101` (non-paged) and
`:325-329` (paged/chunked), both:

```cpp
TT_FATAL(nqh >= nkv && nqh % nkv == 0,
         "Q num_heads must be >= K num_heads and divisible by K num_heads. Got Q: {}, K: {}",
         nqh, nkv);
```

At TP=8 that is `4 >= 1 && 4 % 1 == 0` ✓. Head counts are read at `:61-62`; K and V head counts must
match each other (`:89`, `:317`). This also settles the recipe's open question at
`BRINGUP_RECIPE.md:680-681` ("verify this against the op's signature and log it"): **no on-chip KV
repeat is needed** — see `02_SURVEY.md`.

**Still unverified:** `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` and
`ttnn.transformer.ring_joint_scaled_dot_product_attention` at `n_kv = 1`. First real exercise is
`G-KV` (single card, TP=1 → 8 KV heads, so it does *not* cover this) and then `G-TP-PARITY` /
`G-MESH-KV` in P8. **P8 owns closing this.** If it bites, `(8, 4)` / TP=4 (2 KV heads per chip) is
the pre-costed fallback — see `DEC-002`.

---

## R-005 — `tt_transformers.ModelArgs` requires `HF_MODEL`, so the recipe's preferred oracle is unreachable

The recipe's P1 option 1 (`BRINGUP_RECIPE.md:294-300`) is "HF `transformers` `LlamaForCausalLM`
directly … `models/tt_transformers/tt/model_config.py` already exposes per-module accessors …
`reference_transformer` (`:4037`), `reference_decoder` (`:4393`), `reference_attention` (`:4410`),
`reference_mlp` (`:4365`), `reference_rms_norm` (`:4167`), `reference_embedding` (`:4379`),
`reference_lm_head` (`:4027`)". **All seven line numbers are correct.**

But those are methods on `ModelArgs`, and `ModelArgs.__init__` **raises** without a checkpoint:
`models/tt_transformers/tt/model_config.py:702` —
`raise ValueError("Please set HF_MODEL to a HuggingFace name ...")` (`HF_MODEL` read at `:683`,
`self.CKPT_DIR = HF_MODEL` at `:687`). Every `reference_*` accessor funnels through
`reference_transformer` (`:4037`), which calls `model_cls.from_pretrained(self.CKPT_DIR, ...)`
(`:4126-4144`) and therefore wants real weights on disk.

There is an escape hatch — `ModelArgs(dummy_weights=True, ...)` (ctor kwarg at `:617`) plus
`load_checkpoint=False` takes an `AutoConfig.from_pretrained(self.LOCAL_HF_PARAMS[...])` /
`from_config` path at `:4044-4076` — but `self.CKPT_DIR` is still consulted first at `:4064`, and the
whole object still demands `HF_MODEL` at construction.

**Mitigation.** `DEC-004`: use `transformers` directly (`LlamaConfig` + `LlamaDecoderLayer` /
`LlamaForCausalLM`) rather than through `ModelArgs`. This is *more* faithful to the recipe's stated
first preference ("nothing to vendor, nothing to keep in sync") than the accessor route is, and it
removes `HF_MODEL` from the P1–P6 critical path entirely. **P3+ note:** the `reference_*` accessors
become usable and worth switching to the moment a checkpoint is staged — they handle the Meta↔HF
weight conversion for you (`reference_mlp` monkey-patches `load_state_dict`, `:4368-4376`).

---

## R-006 — `compute_llama3_parameters` hard-codes `low_freq_factor` and `high_freq_factor`

The recipe states (`BRINGUP_RECIPE.md:620-624`) that `rope_type="llama3"` uses
`compute_llama3_parameters:405` "with `factor`, `low_freq_factor`, `high_freq_factor`,
`original_max_position_embeddings` straight from `config.json:rope_scaling`".

**That is not what the code does.** `models/tt_transformers/tt/common.py:405` is
`def compute_llama3_parameters(freqs, scale_factor, orig_context_len)` — three arguments. Lines
`:407-408` are:

```python
low_freq_factor = 1
high_freq_factor = 4
```

They are **local constants, not parameters**. Only `factor` (as `scale_factor`) and
`original_max_position_embeddings` (as `orig_context_len`) are threaded through from the config.

**Impact on this model: none.** Llama-3.1-8B's config has exactly `low_freq_factor: 1.0` and
`high_freq_factor: 4.0` (`Llama-3.1-8B-Instruct/config.json:28-29`), so the hard-coded values
coincide. Llama-3.2-1B/3B also use 1.0/4.0. So the function is correct for the whole Llama-3.x
family.

**Impact if the identity assumption changes or another model is added: silent wrongness.** A config
with different low/high factors would be silently ignored. `tt/rope.py` must therefore **assert** the
config's `low_freq_factor == 1.0` and `high_freq_factor == 4.0` before delegating, rather than
passing them and trusting they are used. This turns a silent-wrong into a loud-fail.
Recipe correction filed.

---

## R-007 — `gpt_oss_d_p`'s distributed-RMSNorm branch is dead code upstream

`models/demos/gpt_oss_d_p/tt/rms_norm.py:33` reads:

```python
self.is_distributed = False  # self.mesh_config.tp > 1
```

The condition is commented out and the flag pinned `False`. So the 3-op distributed path
(`ttnn.rms_norm_pre_all_gather` `:67` → `ttnn.all_gather` `:70-78` → `ttnn.rms_norm_post_all_gather`
`:82-90`) is **present but never executed** in the package it is being borrowed from; the plain
`ttnn.rms_norm` else-branch (`:93-99`) is what is actually validated.

**Impact.** The recipe (`BRINGUP_RECIPE.md:613`, `:786`) says to keep the branch and leave it `False`
until P8 — which is right — but it should not be assumed working. If P4 selects residual scheme B
(TP-sharded residual), that branch becomes load-bearing and Llama would be its **first real user**.
`04_CCL_PLAN.md` must state this when it records the residual decision. Recipe recommends scheme A
(replicated residual, `:561`), which keeps the branch dead for this iteration — the right call given
this risk.

---

## R-008 — SDPA program config hard-codes an 8×8 core grid

`models/demos/gpt_oss_d_p/tt/attention/config.py:90` `get_prefill_sdpa_config(self, mesh_device, seq_len)`
hard-codes `ttnn.CoreCoord(8, 8)` at `:96`, and `get_compute_kernel_config()` at `:102` returns a
`ttnn.WormholeComputeKernelConfig` (`:103`) — on Blackhole hardware.

This is the same class of bug the recipe warns about for the *CCL* grid
(`BRINGUP_RECIPE.md:511-512`, Appendix B row 4: "Ring-SDPA assert `ccl_core_grid_offset.x >=
sdpa_grid.x`"), except here it is the SDPA compute grid rather than the CCL grid. `CCLManager` does
derive its grid correctly — `models/demos/gpt_oss_d_p/tt/ccl.py:44` calls
`compute_with_storage_grid_size()` and `:61` sets
`ring_attention_ccl_core_grid_offset = (compute_grid_size.x - 1, 0)`. So the CCL side adapts to
Blackhole's wider grid while the SDPA side does not, and the two must stay consistent for the
ring-SDPA assert to hold.

**Action for P3/P5:** derive the SDPA grid from `mesh_device.compute_with_storage_grid_size()` rather
than copying the literal, and pick the compute-kernel-config class by arch
(`models/common/utility_functions.py:1043` `is_blackhole()`).

---

## R-009 — `MeshConfig` is duplicated with divergent feature sets

Two near-identical `MeshConfig` classes exist and **neither is a superset**:

| | `models/demos/minimax_m3/config.py:21` | `models/demos/gpt_oss_d_p/tt/config.py:19` |
|---|---|---|
| `reduce_scatter` | **yes**, `:155` | **no** |
| `sp` property | no (reads `mesh_shape[sp_axis]` inline, `:175`) | **yes**, `:55-56` |
| `_validate` | `:40`, lenient — mismatch is a `logger.warning` (`:45-49`) | `:38`, strict — hard `raise` (`:44-48`) |
| `allreduce` / `allgather` | `:77` / `:135` | `:85` / `:138` |
| `_VALIDATED_MESH_SHAPE` / `_VALIDATED_TP` | `(8, 4)` / `4` (`:17-18`) | `(4, 8)` / `8` (`:15-16`) |

The recipe's P4 text (`BRINGUP_RECIPE.md:518-523`) describes `MeshConfig` as owning "the three
collectives `allreduce`, `allgather`, `reduce_scatter`" and cites minimax `config.py:21`
(`allreduce:77`, `allgather:135`, `reduce_scatter:155`) — **all four line numbers correct** — but does
not mention that `gpt_oss_d_p` has its own copy lacking `reduce_scatter`, nor that the recipe's own
`(4,8)`/TP=8 target already lives in `gpt_oss_d_p/tt/config.py:15-16`.

**Consequence.** "Reuse means import, not copy-paste" (`BRINGUP_RECIPE.md:72-73`) cannot be honoured
for `MeshConfig` without importing across sibling demo packages, which the templates themselves do
not do (`gpt_oss_d_p/README.md:46` states the Wormhole gpt_oss demo was "a code-lineage source only
and is **not** imported"). P5 will need a `DEC` for the copy, and it should take the **union**:
minimax's `reduce_scatter` + gpt_oss's `sp` property and strict `_validate`. Consolidating the three
copies into `models/demos/common/` is the right long-term fix and is out of scope here.

---

## R-010 — `get_rot_transformation_mat` ignores its argument

`models/tt_transformers/tt/common.py:562` is `def get_rot_transformation_mat(dhead=32)`, but `:564`
hard-codes `dhead = 32`, overwriting the parameter, before delegating to
`get_rot_transformation_mat_v2(dhead)`.

That is *correct* for `ttnn.experimental.rotary_embedding_llama`, whose transformation matrix is a
per-tile 32×32 construct independent of `head_dim` — but the signature invites the reader to pass
`head_dim` (128) and believe it mattered. Informational: **call it with no argument** so no one later
"fixes" the call site by passing 128 and concludes something changed. Recipe cites the line correctly
(`BRINGUP_RECIPE.md:624`, `:630`) without noting this.

---

## R-011 — `TestFactory.setup_test` is inert until P5

`tests/test_factory.py::TestFactory.setup_test` is specified by the recipe
(`BRINGUP_RECIPE.md:323-324`) to build `MeshConfig` + `CCLManager`, which are P5 deliverables
(`tt/config.py`, `tt/ccl.py`). The function is written now with those imports **inside the function
body** so that importing `test_factory` — and therefore every dimension-only test — works today, and
`setup_test` starts working the moment P5 lands, with no edit. Calling it before then raises
`ModuleNotFoundError` with the module path in the message. Open by design; closes in P5.

---

## R-012 — `G-TP-PARITY` does not cover the 2-link ring path

`get_default_num_links` returns **1** for any single-row mesh
(`models/demos/gpt_oss_d_p/utils/general_utils.py:33`) and 2 on Blackhole otherwise (`:35`). The P8
parity meshes are `(1,2)`, `(1,4)`, `(1,8)` — all single-row — so `G-TP-PARITY` runs at
`num_links = 1` with `Topology.Linear` (`DEC-020`).

**Consequence.** `G-TP-PARITY` proves the *sharding math* (each chip holds the right slice and the
collective sums the right partials). It proves nothing about the 2-link, ring-routed fabric path that
the deployment target uses. The first and only exercise of `num_links=2` + `Ring` is
`G-MESH-KV`/`G-RACE` on `(4,8)`.

**How to read a failure.** A `(4,8)`-only failure after a green `G-TP-PARITY` is a fabric/topology
problem (link count, ring route, torus descriptor), not a sharding one — check
`TT_MESH_GRAPH_DESC_PATH`, `FABRIC_1D_RING`, and `PREFILL_TOPOLOGY=linear` as a bisect, before
touching any module. **Mitigation to consider in P8:** add a `(2,8)` or `(4,8)`-with-TP-only
parametrisation to `test_tp_parity.py` so at least one parity run uses `num_links=2`.

---

## R-013 — Barrier ping-pong depth is 2, and the reset skips barrier/ring semaphores

Two inherited properties of `CCLManager`, both fine for one-shot prefill and both worth stating
before P7/P8 rely on them:

1. **Depth 2.** `get_barrier_semaphore()` cycles over exactly two handles
   (`models/demos/gpt_oss_d_p/tt/ccl.py:78`, index at `:105`). Inside one `allreduce` the
   reduce-scatter takes `barrier[0]` and the all-gather takes `barrier[1]`
   (`models/demos/minimax_m3/config.py:102` then `:124`), so the *next* `allreduce`'s
   reduce-scatter takes `barrier[0]` again — a one-op gap. At 64 all-reduces per chunk
   (`04_CCL_PLAN.md` §7) that is 128 acquisitions cycling over 2 handles. This is the template's
   design and `G-RACE` (3 runs bit-identical) is what validates it. **If `G-RACE` fails, deepening
   the barrier ping-pong from 2 to 4 is the first thing to try**, before suspecting a module.
2. **The reset is partial.** `reset_global_semaphores()` resets only the RS and AG ping-pong sets
   (`models/demos/gpt_oss_d_p/tt/ccl.py:129`) and deliberately not the barrier or ring-attention
   semaphores — the upstream comment (`:132`) says one-shot prefill never reuses a `CCLManager`
   across runs and marks extending it as an open TODO. **Chunked prefill does reuse one across
   `prefill_chunk` calls.** P7 must decide whether to extend the reset; if it does, that is a `DEC`
   with `G-RACE` as its evidence, and if it does not, that is also a `DEC`.

---

## R-014 — `R-002` is wrong: `rope_theta` does not exist on the config object

`R-002` states "`.rope_theta -> None   # <-- attribute EXISTS and is None`". Measured on this
machine (transformers 5.12.1, four construction paths — `LlamaConfig.from_pretrained`,
`AutoConfig.from_pretrained`, `LlamaConfig()`, `LlamaConfig(**raw_json)`), raw log
`raw/G-OUTLINE_20260903T170527Z.log`:

```
cfg.rope_theta                        -> AttributeError: 'LlamaConfig' object has no attribute 'rope_theta'
hasattr(cfg, "rope_theta")            -> False
getattr(cfg, "rope_theta", 500000.0)  -> 500000.0          # the DEFAULT, not None
cfg.rope_scaling                      -> {..., 'rope_theta': 500000.0}      # a full dict, not None
[k for k in cfg.to_dict() if "rope" in k] -> ['rope_parameters']
```

**Why the correction matters rather than being pedantry.** `R-002` and Appendix F.2 describe the
hazard as a silent `None` propagating into the RoPE. The actual behaviour is the **inverse and
worse**: attribute access fails *loudly*, and the `getattr(..., DEFAULT)` pattern
(`models/demos/gpt_oss_d_p/tt/model_config.py:76`, `tt_prefill_runtime.py:185`) *succeeds* while
substituting a hard-coded θ. A default of `10000.0` against Llama-3.1-8B's `500000.0` produces a
RoPE wrong at every position with no exception — the Appendix B "attention PCC 0.5–0.9, norms fine"
signature. The mitigation is unchanged (`get_rope_theta`, assert non-`None`, `DEC-010`); only the
reason is. Also: `cfg.rope_scaling` is **not** `None`, so any code written on the assumption that
it is will read a dict where it expected a fallback.

---

## R-015 — `DEC-006`'s premise ("no demo imports another demo's `tt/`") is false

`DEC-006` justifies copying `MeshConfig`/`CCLManager`/`utils` partly on the claim that "No demo
package in this tree imports another demo package's `tt/`", citing
`models/demos/gpt_oss_d_p/README.md:46`. Measured by grep, both templates cross-import
`deepseek_v3_d_p` heavily:

| importer | line | imports |
|---|---|---|
| `models/demos/gpt_oss_d_p/tt/rope.py` | `:25` | `models.demos.deepseek_v3_d_p.tt.mla.utils.block_cyclic_reorder` |
| `models/demos/gpt_oss_d_p/tt/mlp.py` | `:21` | `deepseek_v3_d_p.tt.moe.init_helpers` |
| `models/demos/gpt_oss_d_p/tt/moe/tt_gpt_oss_moe.py` | `:30-35` | six `deepseek_v3_d_p.tt.moe.*` modules |
| `models/demos/minimax_m3/tt/moe/tt_minimax_moe.py` | `:24-27` | four `deepseek_v3_d_p.tt.moe.*` modules |

**What this does and does not change.** It does not overturn `DEC-006` — the recipe explicitly
instructs the copy (`BRINGUP_RECIPE.md:600-603`), the union requirement stands (`R-009`), and
`README.md:46`'s statement is about the *Wormhole* `models/demos/gpt_oss` demo specifically, not a
tree-wide rule. But `DEC-006`'s generalisation from it is unsupported, and a reviewer who checks
will find that out. It also has a concrete P3 consequence: importing `block_cyclic_reorder` from
`deepseek_v3_d_p` in `tt/rope.py` (rather than copying it) is **precedented by gpt-oss's own
`rope.py:25`**, and `03_OUTLINE.md` §3.5 takes that route.

---

## R-016 — `R-008`'s proposed fix would break the ring-joint SDPA assert

`R-008` (and `02_SURVEY.md` row 11) instruct P3/P5 to "derive the SDPA grid from
`mesh_device.compute_with_storage_grid_size()` rather than copying the literal
`ttnn.CoreCoord(8, 8)`". Measured: that grid is **(12, 10)** on this Blackhole. The ring-joint SDPA
op asserts, in the column-major branch that gpt-oss selects
(`use_column_major_ccl=True`, `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:134`):

```cpp
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421
TT_FATAL(args.ccl_core_grid_offset.x >= args.program_config.value().compute_with_storage_grid_size.x,
         "SDPA coregrid overlaps with AllGather coregrid (column-major)");
```

with the offset fixed at `(grid.x - 1, 0) = (11, 0)`
(`models/demos/gpt_oss_d_p/tt/ccl.py:61`). `11 >= 8` ✓; `11 >= 12` ✗. So the "fix" turns a working
configuration into a `TT_FATAL` the moment SP > 1 — and it would pass every P5 single-card gate
first, surfacing only in P8.

**Resolution:** `DEC-012` keeps 8×8 as an explicit, named `ProgramConfig` field (the real defect is
that it was a buried literal) and adds `assert sdpa_core_grid[0] <= grid.x - 1` for SP > 1.
Appendix D was right; `R-008` and survey row 11 are hereby corrected. The related survey claim that
one should "pick the compute-kernel-config class by arch" is also void, measured two ways:
`hasattr(ttnn, "BlackholeComputeKernelConfig")` is **False** (the name is not exported —
`ttnn/ttnn/__init__.py:305` exports only the Wormhole one), and where it is defined it is the same
object (`ttnn/ttnn/types.py:61`, `BlackholeComputeKernelConfig = WormholeComputeKernelConfig`).
Use `ttnn.init_device_compute_kernel_config(mesh_device.arch(), ...)` (`DEC-013`).

---

## Corrections to earlier risk entries (P3/P4)

Earlier entries are left intact above; these are the amendments.

| Entry | Amendment |
|---|---|
| `R-002` | **Superseded by `R-014`.** The attribute does not exist; the hazard is a silent *default*, not a silent `None`. |
| `R-003` | **Void.** Appendix F.1: real weights are staged at `/home/mstojkovic/models/Llama-3.1-8B-Instruct`. `G-WEIGHTS` (real half), `G-GOLDEN`, `G-CHUNK`/`G-MESH-KV`-vs-golden, `G-MODEL` top-1, `G-REQUEST` and `G-MOCK-MIG` are **runnable, not `BLOCKED`**. Keep the `requires_hf_reference` marker so the suite still runs on a weightless machine. `06_GATES.md`'s P6–P8 checklist rows that say `BLOCKED, R-003` are stale. |
| `R-007` | **Narrowed.** The dormant distributed-RMSNorm branch is only load-bearing for residual scheme B in its `distributed` norm mode. Minimax ships scheme B by **default** with `norm_mode = "gather_first"` (`models/demos/minimax_m3/tt/residual.py:26`, `:32`), which never enters that branch. So "scheme B is unproven" is false as stated; "B-with-distributed-norm is unproven" is true. See `DEC-018`. |
| `R-008` | **Corrected by `R-016` / `DEC-012`.** The recommended fix is wrong; the literal is kept and made explicit. |
| `R-011` | **Unchanged, closes in P5.1** — `tt/config.py` + `tt/ccl.py` are P5.1 deliverables, and `setup_test` must additionally wrap its dict in `llama_hf_config()` per `DEC-009`. |

---

## R-017 — the weight cache is mesh-shape dependent, and no gate covers cache-only at TP>1

Every module loads weights through
`ttnn.as_tensor(..., mesh_mapper=..., cache_file_name=get_cache_file_name(path, name))`
(`models/demos/minimax_m3/tt/dense_mlp.py:64`, `:70`). The cached file therefore holds the
**already-sharded, already-tilized** per-device tensor, so a cache written on one mesh shape is not
interchangeable with one written on another — and cache-only mode (an empty `state_dict` plus a
populated cache) is a load-bearing path: the disaggregated runner relies on it
(`GPT_OSS_WEIGHTS_FROM_CACHE=1` in `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:121`
region).

**The gap.** `G-WEIGHTS` (Appendix A) runs on **1 card**, so the cache-only half is only ever proven
at TP=1. Nothing in the ladder rebuilds from cache on `(4,8)`/TP=8 and compares. A stale or
wrong-shape cache would then present as garbage weights in one or more layers — the
Appendix B "one layer runs on garbage" signature — first observed at `G-MESH-KV`, three phases
downstream.

**Two mitigations, both cheap, both owed by P6.2:**
1. **Put the mesh shape in the path.** `ModelArgs.weight_cache_path(dtype)` must include it, exactly
   as the adapter does — `<cache>/llama31_8b_d_p_bh_<N>dev/<sp>x<tp>`
   (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:75`). Then a `(1,1)` cache and a
   `(4,8)` cache cannot collide, and `TestFactory.setup_test`'s raw `tensor_cache_path` should be
   derived from it rather than passed in bare.
2. **Extend `G-WEIGHTS`'s cache-only assertion to the target mesh in P8**, alongside
   `G-TP-PARITY`: build once with weights, build again from `{}` + cache, and assert the device
   tensors are bit-identical on `(4,8)`.

---

## R-018 — Appendix E's threshold method is input-distribution-sensitive (measured, P5.2)

**Status:** measured and mitigated for `G-RMS`; **open for `G-MLP` / `G-ATTN` / `G-KV`.**

Appendix E's method is: *measure what the existing implementation gets on the same op set, dtype and
silicon, then set the new module's threshold from that measurement.* P5.2 found the method has an
unstated fourth variable — the **input distribution**.

The oracle (`models/tt_transformers/tests/test_rms_norm.py:80`) drives its input with
`torch.rand(1, 1, 32, dim)`: uniform on `[0, 1)`, strictly positive, mean large relative to spread.
Appendix E reports **0.9999867 / 0.9999886** from it and revises `G-RMS` to **>= 0.9999**.

Measured on this box, `tt/rms_norm.py` on `(1,1)`, identical seed and weights, **only the input
distribution changed**:

| input | seq 32 | seq 512 | vs the 0.9999 gate |
|---|---|---|---|
| `randn` (what `G-RMS` uses) | 0.9999637 | 0.9999629 | passes |
| `rand[0,1)` (the oracle's own) | **0.9998979** | **0.9998413** | **FAILS** |

So reproducing the oracle's input distribution would have failed the gate that distribution's own
measurement produced. PCC on a positive-mean signal is dominated by the mean, so bf16 activation
rounding costs more correlation there than on a zero-mean signal.

**Why this is a risk and not just a note:** the same reasoning applies to every remaining
Appendix-E-derived threshold (`G-MLP` >= 0.999 @bf8_b, `G-ATTN` >= 0.999, `G-LAYER` >= 0.999). A
P5.4-P5.6 gate that lands at, say, 0.9985 must not be assumed broken before its input distribution
is compared with the oracle's.

**Mitigation, owed by P5.4-P5.6 and P6:**
1. State the input distribution in every gate detail block (`DEC-026`).
2. When a gate lands in the band between the Appendix E number and the recipe's original guess,
   **re-run it under the oracle's distribution before suspecting the module** — that is a two-minute
   check that distinguishes "the threshold's provenance is distribution-dependent" from "the module
   is wrong".
3. Recipe change worth folding in: Appendix E should record the *input distribution* of each oracle
   alongside its PCC, because the number is not portable without it.

---

## R-019 — This package is untracked in git, so P0-P4 never ran the repo's `pre-commit` hooks

**Status:** partially resolved in P5.1 (all 32 files are now hook-clean); the **process** gap remains.

`git ls-files -o --exclude-standard` lists every file in `models/demos/llama31_8b_d_p/` — the
directory has never been committed, and the recipe forbids committing (`BRINGUP_RECIPE.md:91`). The
consequence is that no P0-P4 deliverable had been through `black`, `isort`, `autoflake`, or the
repo's local hooks, even though P9's `G-CLEAN` requires them.

Two concrete effects, both hit in P5.1:

1. **`black` moved a cited line.** It collapsed `UPSTREAM_CONFIG_JSON` in `tests/test_factory.py`
   from three lines to one, moving `llama_config_dims` from `:49` to `:47` — breaking the citation
   that `03_OUTLINE.md` §3.23 and `DEC-009` both record. Corrected in `DEC-027`.
2. **A local hook rejects `pytest.raises` in any `tests/` file.** `.pre-commit-config.yaml:51-56`
   defines `prefer-expect-error`, a `pygrep` hook matching
   `(?<!allow-)pytest\.raises(?!.*allow-pytest\.raises)` over `(^|/)tests/.*\.py$`. The repo-root
   `expect_error` fixture (`conftest.py:948`) is the required form; it wraps `pytest.raises` and logs
   `[EXPECTED_ERROR BEGIN/END]` so CI log triage can tell an expected error from a real one. Its
   signature is `expect_error(error, message)` and **`message` is mandatory** (it becomes `match=`),
   so a bare `pytest.raises(Exception)` has to be given a real error class and substring. Neither
   the recipe nor `03_OUTLINE.md` mentions this hook.

**Mitigation:** run `pre-commit run --files <the files you touched>` **before** writing any
`path:line` into a log, and before declaring a doc gate. P5.4-P5.6 and P6 own this for their own
files; it is stated as `DEC-027`.

---

## Corrections to earlier risk entries (P5)

- **`R-002` is superseded by `R-014`, and `tests/test_factory.py` still carried the wrong wording.**
  `R-002`'s claim that `rope_theta` "EXISTS and is `None`" was already corrected by `R-014` and
  `03_OUTLINE.md` §1.1 (it raises `AttributeError`; `getattr` returns the *default*). But
  `llama_config_dims`' docstring in `tests/test_factory.py` still repeated the `R-002` version. P5.1
  rewrote that docstring to the measured behaviour and pointed it at Appendix F.2. Re-measured this
  phase on `transformers` 5.12.1, and recorded as a positive test rather than a comment
  (`test_mesh_config.py::test_llama_hf_config_from_transformers_object`):
  `cfg.rope_theta` → `AttributeError`; `getattr(cfg, "rope_theta", 10000.0)` → **10000.0**;
  `cfg.to_dict()` → neither key; `get_rope_theta(cfg.to_dict())` → **500000.0**.
- **`R-006` is now enforced in two places, not one.** The hard-coded `low_freq_factor = 1` /
  `high_freq_factor = 4` are asserted in `llama_hf_config()` (the single dict-read point) **and**
  re-asserted in `tt/rope.py::_assert_llama3_scaling` from the normalised object, so a config that
  never went through the normaliser also cannot reach `compute_llama3_parameters` (`DEC-025`).
- **`R-007` (the dormant distributed-RMSNorm branch) is unchanged but now reachable by
  configuration.** `is_distributed` is a constructor argument defaulting to `False` rather than a
  pinned literal with the condition commented out, so P8 enables scheme B without editing a module
  (`DEC-024`). It is still **unexercised** — no PCC number exists for it.
- **`R-008` / `R-016` are now a build-time assert.** `test_ccl_semaphores.py` asserts
  `ring_attention_ccl_core_grid_offset.x >= 8` (the pinned SDPA program grid) on the real device, so
  the `11 >= 8` vs `11 >= 12` distinction Appendix F.8 warns about is checked in P5 instead of
  surfacing at SP > 1 in P8. Measured: grid **(12, 10)**, offset **(11, 0)**.
- **`R-010` confirmed on device.** `get_rot_transformation_mat()` called with no argument yields the
  `(1, 1, 32, 32)` tile `rotary_embedding_llama` expects; asserted in `G-ROPE`.
- **`R-011` is closed.** `TestFactory.setup_test` is live: `tt/config.py`, `tt/ccl.py` and
  `tt/model_config.py` all exist, and it now returns `hf_config` as a **`LlamaHFConfig` object**
  (`DEC-009`), not the raw dict.
- **One P4 citation error found and corrected.** `04_CCL_PLAN.md` §3 cites
  `models/demos/gpt_oss_d_p/tt/config.py:55` for the `sp` property; `:55` is the bare `@property`
  decorator and `def sp` is on `:56`. It slipped through because that reference was never in
  `CITES` and pass 2 only checked in-range — the identical hole `03_OUTLINE.md` §8 flagged for
  `02_SURVEY.md:76`. `DEC-030` closes it by making pass 2 resolve abbreviated references and by
  scanning `05_DECISIONS.md` and `06_GATES.md` too (**333/333** doc references now resolve).

---

## R-020 — `build_prefill_rope` cannot serve a chunked prefill past chunk 1 (found P5.3, owed by P7)

**Status:** guarded by an assert; the real fix is P7 using the indexed path.

`models/tt_transformers/tt/common.py:534` `get_prefill_rot_mat` precomputes a frequency table of
exactly `seq_len * 2` positions (`:536`) and then gathers `[start_pos, start_pos + seq_len)` from it
(`:538`). Measured:
`gather_cos_sin(torch.arange(1024, 1536), *precompute_freqs(128, 1024, ...))` →
`RuntimeError: index 1024 is out of bounds for dimension 0 with size 1024`.

So the bound is `start_pos <= seq_len`. A chunked prefill at `chunk = 512` works for chunk 0
(`start_pos = 0`) and chunk 1 (`start_pos = 512`) and **breaks on chunk 2** (`start_pos = 1024`) —
with an `IndexError`-class message from inside a delegate that mentions neither RoPE nor chunking.
P7 is the phase that would hit it, two phases after `tt/rope.py` looks settled.

**Mitigation in place:** `tt/rope.py:108` asserts the bound with a message naming
`build_indexed_rope()` as the correct chunked path, and `G-ROPE` covers both the last legal offset
and the refusal (`DEC-029`).

**Residual risk:** if P7 ever wants a one-shot (non-indexed) RoPE for a mid-stream chunk, the fix is
upstream — `get_prefill_rot_mat` would have to size its table from `start_pos + seq_len` rather than
`seq_len * 2`.

---

## R-021 — `ttnn` compute-kernel defaults differ per op, so "copy the template" is unsafe (measured, P5.4-P5.5)

**Status:** measured and mitigated for `G-MLP` / `G-ATTN`; **open for every op P6-P8 adds.**

`DEC-031` established that `ttnn.rms_norm` with no `compute_kernel_config` is ~25x worse than
HiFi4 + `fp32_dest_acc_en=True`. P5.4-P5.5 measured the same A/B on the matmul and attention paths
and found the polarity is **not consistent across ops**:

| op | no config passed | `fp32_dest_acc_en=True` | `fp32_dest_acc_en=False` |
|---|---|---|---|
| `ttnn.rms_norm` (`DEC-031`) | 0.9999652 | **0.9999971** | 0.9999607 |
| `ttnn.linear` / MLP, bf8_b weights | 0.9999143 | **0.9999143** (identical) | 0.9925392 |
| `ttnn.linear` / MLP, bf16 weights | 0.9999852 | **0.9999852** (identical) | 0.9917529 |
| attention block (proj + SDPA + `o_proj`), bf8_b | — | **0.9997449** | 0.9963324 |
| attention block, bf16 | — | **0.9998033** | 0.9959098 |

So the matmul's own default *already* enables fp32 destination accumulation (bit-identical PCC),
while the norm's does not. Two consequences:

1. **Explicitly passing the flag is right for both ops, for different reasons** — it fixes the norm
   and it pins the matmul against a future default change.
2. **The real hazard is the opposite of an omission: it is copying the template's explicit `False`.**
   `models/demos/gpt_oss_d_p/tt/attention/config.py:71` ships `fp32_dest_acc_en: bool = False`. Carried
   forward unexamined that costs **14x** (bf8_b) to **21x** (bf16) of measured attention error — and
   at bf8_b it still scores 0.9963, i.e. it **still clears the recipe's 0.999 gate** and would have
   been recorded as a clean PASS. This is exactly the failure `DEC-032`'s noise-floor gating exists
   to catch, and it is the first time it caught something.

**Owed by P6-P8:** every op that accepts a `compute_kernel_config` gets one explicitly, and any op
where `fp32_dest_acc_en=True` is refused or measurably worse needs a `DEC` recording **both**
numbers. The ring-joint SDPA is the one known `False` (it is required —
`models/demos/gpt_oss_d_p/tt/attention/prefill.py:200`), which is why `dense_sp.py` must build its own
config rather than reuse `ProgramConfig.get_compute_kernel_config`.

---

## R-022 — a storage-dtype noise floor does not model a kernel's interior (measured, P5.5)

**Status:** measured, attributed and fenced for `G-ATTN`; **the method needs the same treatment
wherever a fused kernel appears** (P7's chunked SDPA, P8's ring SDPA).

`DEC-032`'s noise floor rounds every *stored* tensor to its device dtype and does the arithmetic in
fp32. That is exactly right for a norm or a matmul chain, and **wrong for a flash-attention kernel**,
whose online softmax and chunked accumulation are internal to the op and have no stored intermediate
to round.

Measured: `ttnn.transformer.scaled_dot_product_attention` fed bf16 Q/K/V directly (GQA 32/8,
head_dim 128, seq 128, no projections) scores **0.9999204** against a bf16-input fp32-math reference
whose own floor is **0.9999989** — **71x**. `q_chunk`/`k_chunk` ∈ {32, 128, 256} moves it <4%;
`exp_approx_mode` not at all. It is the kernel, not the configuration.

That one term is the entire reason `G-ATTN`'s block-level ratio is 2.6x (bf8_b) / 5.1x (bf16) while
the stages this package implements — projections, GQA split, RoPE — sit at **1.00-1.47x**.

**Why this is a risk and not a footnote:** the naive reading of `DEC-032` ("anything over ~3x is a
finding") would have condemned a correct attention module. The equally bad reading ("just widen the
budget") would have hidden a real bug. The method that works, and that `DEC-034` records, is:

1. measure the fused kernel **alone** against its own floor, and keep that probe in the suite;
2. gate the stages you implement at a **tight** ratio, excluding the kernel;
3. let the block-level budget be the sum, with the kernel term named in the gate block.

**Owed by P7/P8:** the paged chunked SDPA and the ring-joint SDPA need their own standalone
kernel-vs-floor probes before their block gates are interpreted. Without one, a genuine regression
in the ring path is indistinguishable from the 71x that is already there.

---

## Corrections to earlier risk entries (P5.4-P5.6)

**`R-018` is superseded, not just closed.** It framed Appendix E's threshold method as
*input-distribution-sensitive* and its "mitigation" was to re-run under the oracle's distribution
before suspecting the module. `DEC-032` (raised by the orchestrator during P5.4) shows the framing
was wrong: the bf16 torch floor is 0.9999986 under **both** `rand[0,1)` and `randn`, so distribution
does not move the floor — the oracle looked better because its *reference* loads HF weights at
`torch_dtype: bfloat16` and shares the device's own rounding. P5.4-P5.6 therefore did **not** perform
`R-018`'s prescribed re-run; it computed torch noise floors instead. `R-018`'s surviving requirement
— *state the input distribution in every gate detail block* — is honoured by `G-MLP`, `G-ATTN` and
`G-KV`, each of which also states the reference's dtype policy. Its item 3 (fold input distributions
into Appendix E) is subsumed by `DEC-032`.

**`R-016` is confirmed on device.** It predicted that deriving the SDPA program grid from
`compute_with_storage_grid_size()` would break the ring-joint assert. Measured in P5.5: the grid is
**(12, 10)**, the pinned 8x8 passes `assert_sdpa_grid_fits`, and a `(12, 10)` `ProgramConfig` is
refused at `Attention.__init__` (`raw/G-ATTN_20260903T180817Z.log`). The check is deliberately
**unconditional** rather than gated on SP > 1 (`DEC-036` item 4) — gating it would have reproduced
the very Appendix F.8 failure mode of passing every single-card gate and failing in P8.

**`R-019`'s process gap is still open, and it bit again.** `black` reformatted three of the eleven
files written in P5.4-P5.6 on their first `pre-commit` run; `pre-commit` was run **before** any
`path:line` was recorded, per `DEC-027`, so no citation was harmed. But the package remains untracked
in git, so nothing enforces that ordering for the next session.

---

## R-023 — Appendix E's masking caveat is a cross-test comparison, and the recipe builds a rule on it (measured, P6.1)

**Status:** measured and corrected in `DEC-040`; **the recipe text is still wrong** and P7+ will read it.

`BRINGUP_RECIPE.md:1131-1141` and `03_OUTLINE.md` §5.1 assert that a decoder-layer PCC comes out
*higher* than either of its sublayers' "because the residual stream dominates the correlation", from
the `tt_transformers` oracles 0.9999985 (decoder) vs 0.9996099 (attention) / 0.9995823 (MLP).
Measured on this box against **one** fp32 reference, one input distribution and one dtype ladder:

| | @bf8_b | @bf16 |
|---|---|---|
| `G-ATTN` attention block, seq 128 | 0.9997554 | 0.9998129 |
| `G-LAYER` whole layer, seq 128 | **0.9995864** | **0.9997674** |

The layer is **worse** than its own attention block at both dtypes, because the layer's noise floor is
itself lower (0.9997390 vs 0.9999067) — the MLP's bf8_b weights add quantisation the attention block
never sees. And the masking mechanism, measured directly, is 1.06-1.73x, not orders of magnitude:
for `y = r + s` the attenuation is exactly `||y||/||s||`, which is **1.06x** on the gate's random
weights and **1.73x (attn) / 1.23x (mlp)** on real layer-0 weights with real embedding inputs. A
1.1-1.7x attenuation cannot turn 0.9996 into 0.9999985.

The 0.9999985-vs-0.9996099 comparison is therefore a **cross-test** comparison between two
`tt_transformers` test files with different reference constructions, different input distributions
and different dtype ladders — the exact error Appendix **E.1** identifies and forbids, not applied to
E's own caveat section.

**What is NOT affected:** the *rule* ("`G-LAYER`/`G-MODEL` are integration checks and never sublayer
evidence") stands, on two grounds that are measured rather than assumed — an aggregate PCC cannot
localise which sublayer is wrong, and a layer's floor is looser than its sublayers', so a layer
threshold a sublayer would fail is arithmetically normal. No threshold was changed.

**Owed by whoever edits the recipe:** replace Appendix E's "Caveat that matters more than the
numbers" and `03_OUTLINE.md` §5.1's justification with `DEC-040`'s. Falsifier for the remaining
open question (why the oracle scores 0.9999985): instrument `tt_transformers`'
`test_decoder_prefill` and `test_attention_prefill` against one fp32 reference on one input.

---

## R-024 — `models/tt_transformers` HF->Meta key conversion is prescribed for a package that consumes HF keys (measured, P6.2)

**Status:** closed for this package by `DEC-039`; **the recipe and `03_OUTLINE.md` §3.3 still
prescribe the harmful version**, and P7/P10 load weights.

`BRINGUP_RECIPE.md:762-764` and `03_OUTLINE.md` §3.3 tell P6.2 to run the checkpoint through
`map_hf_to_meta_keys` / `convert_hf_qkv_to_meta_format` (via `convert_hf_to_meta`) and to expose
`load_state_dict(weights_path, convert_to_meta_format=True)`. Both halves are actively harmful here:

1. `models/demos/llama31_8b_d_p/tt/attention/weights.py:71` already applies the Q/K
   `reverse_permute` (`DEC-033`), so a second permute at load is the transform applied twice —
   neither HF nor Meta layout.
2. Every module in this package strips **HF** sub-dicts (`substate(sd, "mlp")`), so Meta renaming
   makes every `substate()` return `{}`. With a populated `tensor_cache_path` that is **not an
   error**: `ttnn.as_tensor` silently loads whatever the cache holds. Measured: the rename produces
   **291 missing and 291 unused of 291** keys (`raw/G-WEIGHTS_*.log`).

**Owed by P7/P10:** load weights through
`models/demos/llama31_8b_d_p/tt/model_config.py:298` `ModelArgs.load_state_dict`, which keeps HF
naming and layout and refuses a Meta-keyed dict via
`models/demos/llama31_8b_d_p/tt/model_config.py:245` `state_dict_uses_meta_keys`. Do **not**
reintroduce a `convert_to_meta_format` flag.

---

## R-025 — the per-layer PCC step threshold is calibrated at one sequence length (open, P6.3)

**Status:** open, owned by P7.

`DEC-047` sets `MAX_LAYER_ERROR_STEP = 4.0` from the measured 32-layer curve at **seq 128**, where
the consecutive per-layer error ratio stays in 0.99x-1.38x from layer 3 onward
(`raw/G-MODEL-CURVE_20260903T195712Z.log`). Two limits:

* it is one sequence length, one input, one dtype. A longer sequence changes the SDPA chunk sizes
  (`ProgramConfig.prefill_threshold = 2048`) and therefore the per-layer error, so the curve's shape
  at 4K-16K is unmeasured;
* a *small* wrong weight in one layer produces a step below 4.0 and would pass. The defences that do
  not depend on this threshold are `G-WEIGHTS`'s per-key value check
  (`models/demos/llama31_8b_d_p/tests/unit/test_weight_loading.py:197`) and the delta probe
  (`DEC-041`).

**Owed by P7:** record the curve at the real chunk size and re-derive the threshold there before
relying on 4.0 for a long-context run.

---

## Corrections to earlier risk entries (P6)

* **`R-003` (no checkpoint) is void and was already marked so by P3.** P6 confirms it operationally:
  the full 291-tensor checkpoint loads, and `G-WEIGHTS` / `G-MODEL` both ran against real weights
  with `requires_hf_reference` still in place so the suite skips rather than fails on a weightless
  machine.
* **`R-005` is confirmed and routed around, not inherited.** `models/demos/llama31_8b_d_p/tt/model_config.py:257`
  `ModelArgs` is a fresh class, not a subclass of `models/tt_transformers/tt/model_config.py:539`.
  P6 did however use that module's `:4393` `reference_decoder` docstring as a *rejected* option, for
  the Appendix E.1 reason (its HF weights load at the checkpoint's `torch_dtype: bfloat16`, so its
  reference shares the device's own rounding), not for the `HF_MODEL` reason.
* **Appendix F.2's "always pass an explicit causal mask" does not apply to `LlamaModel.forward`.**
  Measured (`raw/G-MODEL_*.log`, `test_hf_reference_is_causal`): with `attention_mask=None` and
  `attn_implementation="eager"`, changing only the **last** token id leaves every earlier row of
  `last_hidden_state` **bit-identical** (`max|delta| = 0.0`), i.e. `create_causal_mask`
  (`python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:399`) does
  build the mask. F.2's warning is correct and load-bearing for hand-written
  `eager_attention_forward` calls (which is how P5's sublayer references are written); it is not a
  reason to distrust the full-model reference. The probe stays in the gate so a transformers upgrade
  that changes this is caught rather than assumed.
* **`R-019` (this package never ran `pre-commit`) — P6 ran it on every file it created or changed.**
  `pre-commit run --files ...` clean (black, autoflake, isort, prefer-expect-error). No
  `pytest.raises` anywhere in the new tests; the root `expect_error(ErrorClass, "substring")` fixture
  is used, message mandatory.

### R-040 — The multi-rank KV-chunk-table merge is untested (consequence of DEC-070)
> Numbered R-040 out of the P7 session's sequential range to avoid an id collision.

**Status:** OPEN, accepted for this bring-up, **and now loud rather than latent (P10).**
**Owner:** whoever first runs pipelined (multi-rank) prefill for this model.

**P10 update.** `G-MOCK-MIG` ran and passed at one rank, so everything below still holds exactly as
written. What changed is the failure mode: `TtPrefillRuntime.build_kv_chunk_table`
(`tt/tt_prefill_runtime.py:583`) no longer *discards* the runner's `first_layer_idx` /
`num_my_layers` / `stage_layout(s)` arguments the way the template does — it **raises
`NotImplementedError` naming this risk** when they describe a merge that is not implemented
(`DEC-109`, covered by `tests/unit/test_kv_chunk_table.py::test_runtime_hook_refuses_a_multi_rank_merge`).
A first pipelined run therefore gets an error at table-publish time instead of a table that addresses
rank 0's DRAM under every rank's layer ids. Those three refusals are also the precise checklist of
what has to start working when someone implements the merge.

One more piece of surface joined this risk in P10: `pipeline_activation_emb_tp_sharded = True` on the
adapter (the emb-axis sharding of the cross-rank D2D hidden state). Single-rank runs build no D2D
socket, so it is an assumption carried from `DEC-018`, not a measurement.

Skipping Gate 2 (`DEC-070`) leaves exactly one piece of *our own* surface unexercised, and it is worth
stating precisely rather than hiding behind "migration is out of scope":

- **`PREFILL_MOCK_MIGRATION=1` is single-rank only.** The runner rejects it for `num_ranks > 1`,
  because each rank would publish a table covering only its own layer slice and a merged mock table is
  not implemented (`models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`, *Shared setup*).
- So `G-MOCK-MIG` validates `build_kv_chunk_table` for **one rank's** layer slice. The real path
  merges per-rank stage layouts through the worker
  (`deliver_device_map_and_gather_stage_layouts`) to publish one table spanning every rank's layers —
  and **only Gate 2 exercises that merge**.
- Consequently `kv_migration_base_address` / `kv_migration_stages` (the anchors the engine all-gathers
  to build the merged table) are implemented against the documented contract but never executed.

**What this does NOT affect:** single-rank prefill correctness, the KV cache contents, the chunk
table for one rank, the adapter/runtime contract, or any PCC gate. All of those are covered.

**How to close it:** build the `tt-llm-engine` migration component (see `DEC-070` for the exact
products and constraints) and run Gate 2 on a 2- or 4-rank binding — note the doc's warning that a
multi-host driver verifies only its own host's layers unless launched via `run_migration_driver.sh`
with the full rank-ordered host list, so a single-process `PASSED` there would be telling the truth
about a fraction of the model.


---

## R-026 — `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, not `list[int]` (measured, P7.1)

**Status:** mitigated here; **latent in `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:180`.**

Measured on transformers 5.12.1 while writing `scripts/generate_golden_kv_cache.py`. Three distinct
return types from one method, and only the third is what a caller wants:

| call | actual return |
|---|---|
| `apply_chat_template(msgs, add_generation_prompt=True)` | the rendered chat **string** (`tokenize` defaults `True`, but `return_dict` also defaults `True`... see below) |
| `apply_chat_template(msgs, ..., tokenize=True)` | a **`BatchEncoding`** — `{'input_ids': [...], 'attention_mask': [...]}` |
| `apply_chat_template(msgs, ..., tokenize=True, return_dict=False)` | **`list[int]`** |

The signature is
`(conversation, ..., tokenize: bool = True, ..., return_dict: bool = True, ...)`.

**Why it matters more than an argument mistake.** `list(BatchEncoding)` returns
`['input_ids', 'attention_mask']` — a 2-element list of *strings*. Downstream that is either a loud
`ValueError: too many dimensions 'str'` (what happened here) or, in a script that indexes or
re-tokenizes it, a **plausible short token sequence** that generates a valid-looking golden for the
wrong prompt. Since the golden's `token_ids` are what the device is then required to prefill, a wrong
`token_ids` produces a KV comparison that is internally consistent and meaningless.

**Mitigation:** `return_dict=False` plus
`assert all(isinstance(i, int) for i in ids)` in `tokenize_prompt`, so a future default change fails
at tokenization rather than 32 layers later.

**Latent elsewhere:** `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:180` passes
`tokenize=True` and not `return_dict=False`. On this transformers version that script's `ids` would
be a `BatchEncoding`. Not filed upstream by P7 (different model package, and M3 may pin an older
transformers), but it is the same shape of bug and it is what `scripts/verify_citations.py` now pins.

---

## R-027 — The packed KV cache needs `TP == num_key_value_heads`, so no model-level KV write is possible on one card (measured, P7)

**Status:** mitigated by a loud assert; the **coverage hole it exposes is open and belongs to P8.**

`allocate_kv_cache` allocates a per-chip cache of
`[num_users*num_layers, 1, seq_local, head_dim]` — one KV head per chip, hard-coded and commented as
such (`models/demos/llama31_8b_d_p/tt/attention/kv_cache.py:130`). The model's attention produces
`mesh_config.shard_size(num_kv_heads)` local KV heads. Those agree only when `TP == 8`.

Measured at `(1,1)`, running a 1-layer `Model.prefill_forward` with a cache attached:

```
TT_FATAL: cache and input num-heads dim must match
  ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/
  update_padded_kv_cache_device_operation.cpp:230
RuntimeError ... cache_shape[1] == input_shape[1]
```

The write fires for **both** `cached_len = 0` and `cached_len = 128`, i.e. this is not the chunked
blocker (`R-028`) — it stops the *first* chunk too.

**Two consequences, and the second is the important one.**

1. `TtPrefillRuntime._resolve_kv` asserts `tp_factor == num_key_value_heads` with a message naming
   the mesh, the head counts and the C++ assert, so the failure is a sentence instead of a
   `TT_FATAL` (`DEC-056`).
2. **`G-KV`'s single-card coverage does not exercise the head count the model actually produces.**
   `tests/unit/test_kv_cache_vs_ref.py:101` sets `nkv = tp`, which is **1** at `(1,1)` — correct for
   what a chip holds at TP=8, and therefore a configuration the model never produces on the mesh the
   test runs on. `G-KV` is not wrong (the layout it pins is the deployment layout), but the
   *model→cache* interface is unproven until TP=8. `G-CHUNK` narrows the hole by driving the real
   device projections and RoPE into the real cache one head at a time (head `h` → slot `h`), which
   is exactly the per-chip write; what remains unproven is the mesh-mapper step that puts head `c`
   on column `c`.

**How to close it:** P8's `G-MESH-KV` on `(4,8)`, or a `(1,8)`/TP=8 parametrisation of `G-CHUNK`.
The latter is cheap and would close it without the SP ring — recommended as P8's first KV step,
before `dense_sp_attention`.

---

## R-028 — Chunked cache-read attention is unimplemented, so `G-CHUNK`'s attention-core third is blocked (P7)

**Status:** open. Blocks `G-CHUNK-ATTN`. Owner: P8.

A chunked prefill differs from a one-shot in three places. P7 measured two of them
(`DEC-058`); the third cannot run:

`tt/attention/prefill.py:218` raises `NotImplementedError` for `cached_len > 0` outside the SP
branch, because a plain `is_causal` SDPA assumes Q row 0 aligns with K row 0 — with a non-empty
cache it is off by `cached_len` and **silently wrong** (a correctly-shaped, plausible tensor). The
SP branch at `:195` needs `sequence_parallel=True` **and** `mesh_config.sp > 1` **and** a real
`tt/attention/dense_sp.dense_sp_attention`, which is still the P5 stub
(`tt/attention/dense_sp.py:43`).

**Scope this honestly:** enabling it is not a flag flip. It requires porting
`models/demos/gpt_oss_d_p/tt/attention/dense_sp.py`'s ring-joint SDPA over the block-cyclic cache
(dropping `attention_sink` / `sliding_window_size`, keeping the cache at bf8_b, the SDPA grid at
8x8, and `fp32_dest_acc_en=False` for the ring op) **or** adding a paged
`chunked_scaled_dot_product_attention` path with a page table. Both touch
`tt/attention/`, which P7 does not own. `ttnn.transformer.chunked_scaled_dot_product_attention`
does exist on this build (verified by `dir(ttnn.transformer)`), so the paged route is available —
but it wants a paged cache, and this cache is a DRAM `NdShard`, not pages.

**What P8 must change, precisely:**
1. Implement `tt/attention/dense_sp.dense_sp_attention` (the ring-joint port).
2. Then set `TtPrefillRuntimeConfig(sequence_parallel=True)` on a mesh with `sp > 1`.
   `TtPrefillRuntime._chunked_read_supported()` **probes** the stub, so it becomes `True`
   automatically and both the `compile()` two-chunk warm-up and `prefill_chunk(actual_start>0)`
   start working with no edit to `tt/tt_prefill_runtime.py` (`DEC-056`).
3. Re-run `tests/unit/test_attention_chunked_vs_ref.py` with the *model* driving the cache, and
   promote `G-CHUNK-ATTN` from `BLOCKED`.

**What is already known to work, so P8 does not re-debug it:** the indexed RoPE at every non-zero
`kv_actual_global` (`raw/G-CHUNK_20260903T204519Z.log`, chunked-vs-one-shot K/V PCC **1.00000** over
32 layers at two chunk sizes) and the chunked cache write offsets (same log; `G-KV`'s bit-exact
positional read-back).

---

## R-029 — The runtime's KV read-back helpers are unexercised on device (P7)

**Status:** open. First real exercise is P8's `G-MESH-KV`.

`TtPrefillRuntime.gather_layer`, `dump_slot_kv` and `kv_cache_pcc_check` all route through
`_resolve_kv`, which asserts `TP == num_key_value_heads` (`R-027`), so at `(1,1)` they refuse rather
than run. P7 narrows the gap two ways rather than leaving it silent:

* `G-CHUNK` writes a device dump in **the same format** `dump_slot_kv` writes and scores it with the
  same `compare_device_dump`, so the format and the golden comparison are both exercised end to end;
* `tests/unit/test_attention_chunked_vs_ref.py::test_device_dump_metadata_contract_matches_the_verifier`
  asserts, from the source text, that every metadata key `dump_slot_kv` writes is a key
  `compare_device_dump` reads — the two live in different files and would otherwise be free to drift
  on the only mesh this phase can run.

What is **not** covered: the `blockcyclic_positions` inverse at `sp > 1` on a real device (it is
proved host-only by `G-KV`'s `test_blockcyclic_positions_are_an_exact_inverse`), and the
`r * cols + col` device-tensor indexing that maps KV head `c` to column `c`.

---

## R-030 — `build_kv_chunk_table` raises; migration is not wireable until P10 (P7)

**Status: CLOSED by P10.** `tt/runners/kv_chunk_table.py` implements the table and
`tt/tt_prefill_runtime.py:583` forwards to it. Two independent gates cover it:
`G-KV-TABLE` reads a labelled pattern back out of DRAM through the table **bit-exactly**
(`rtol = atol = 0`, 2 users x 2 layers x 8 heads x K/V x 512 tokens, after a protobuf export/import
round trip, with a negative control), and `G-MOCK-MIG` PCCs the real 32-layer KV against the fp32
golden through it — reproducing P8's on-device numbers (min K 0.99646 / V 0.98445) to five decimal
places from a different process. The geometry recorded below is exactly what got encoded, and
`mesh_device.dram_grid_size().x` measured **8** as predicted. The successor risk is `R-040`
(the multi-rank merge), which P10 turned from silent into loud — see `DEC-109`.

_Original entry (P7):_ open by design. Owner: P10.

`TtPrefillRuntime.build_kv_chunk_table` raises `NotImplementedError` naming
`tt/runners/kv_chunk_table.py` (P10's deliverable per `03_OUTLINE.md` §3.21), so
`PREFILL_ENABLE_MIGRATION=1` cannot work. Raising rather than returning an empty table is the point:
the engine publishes whatever it returns to the migration worker, and a structurally valid but wrong
table migrates the wrong DRAM ranges — which surfaces as a corrupted decode long after prefill, with
nothing pointing at the table.

`kv_migration_base_address` **is** implemented (`int(kv.k.buffer_address())`), and the geometry the
table must encode is already fixed and gated by `G-KV`:
`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`, shard row `[1, 1, 32, 128]`, `ROUND_ROBIN_1D` over
`mesh_device.dram_grid_size().x` (measured **8**) banks.

---

## Corrections and status updates from P7

* **`R-020` is closed, and by measurement rather than by argument.** The mitigation was an assert
  naming `build_indexed_rope`; P7 ran that path. The indexed RoPE at `(1,1)` over four chunk offsets
  (0, 128, 256, 384) reproduces a torch absolute-position Meta rotation at **PCC 1.0000008**, and
  chunked-vs-one-shot K is **1.00000** over all 32 layers at both (512,128) and (2048,512)
  (`raw/G-CHUNK_20260903T204519Z.log`). The residual noted in `R-020` — "if P7 ever wants a one-shot
  RoPE for a mid-stream chunk" — did not arise: the indexed path serves every chunk.
* **`R-025` is answered for the KV product and still open for the hidden state.** `DEC-060` carried
  `MAX_LAYER_ERROR_STEP = 4.0` **and** `STEP_CHECK_FROM_LAYER = 3` over unchanged and re-measured
  them on the per-layer KV curve at chunk 128 and chunk 512: max K step **1.95x** / **1.81x**, max V
  step **1.48x** / **1.60x** — half the ceiling, at two chunk sizes instead of one. The excluded
  layer-2 step (**4.49x** / **4.18x**) is real and is exactly what `STEP_CHECK_FROM_LAYER` exists
  for: layer 1's K error is 3.34e-5, ~1/55th of the deepest layer's 1.82e-3. **Still open:** the
  *hidden-state* curve at the 8192-token default chunk, which P7 did not measure (a 32-layer fp32 HF
  reference at 8192 tokens is a much larger host run, and the SDPA program config crosses
  `ProgramConfig.prefill_threshold = 2048` again above it). Re-assigned to P8/P9.
* **`R-004` (n_kv=1 at TP=8) is superseded by `R-027`, in the opposite direction.** `R-004` and
  Appendix F.6 worried that TP=8 leaving 1 KV head per chip was under-exercised. Measured: `nkv = 1`
  per chip is the **only** configuration the cache supports, and the untested case is every *smaller*
  TP — including the `(1,1)` mesh five gates ran on. F.6's "no topology change is needed" stands;
  its framing ("the residual delta is `head_dim`") missed that the head *count* constrains the mesh.
* **Numbering irregularity in the logs, not a code fault.** `05_DECISIONS.md` contains a lone
  `DEC-070` and `07_RISKS.md` an `R-040` (with `06_GATES.md`'s `G-LOOPBACK` block), with **no**
  `DEC-062`-`DEC-069` and no `R-026`-`R-039`. §1.3 specifies monotonic numbering, and P7 was told to
  continue from `DEC-052`, so P7 used `DEC-052`-`DEC-061` and `R-026`-`R-030`. There is no collision,
  but a reviewer should know the gap is a reservation someone made, not a lost block.

---

## Status updates and closures from P8

Every risk P8 owned was settled by a measurement on the real `(4,8)` galaxy, not by argument. The
raw logs are named per item.

### `R-027` — **CLOSED.** The model -> cache path is proven at TP=8
The coverage hole was that `G-KV` and `G-CHUNK` ran at `(1,1)` with `nkv = tp = 1`, a head count the
model never produces on that mesh, so the mesh-mapper step (KV head `c` -> mesh column `c`) was
untested. `G-KV-TP8` (new, `tests/unit/test_kv_cache_tp8.py`) closes it two ways:

* **bit-exactly, without the model** — a position/head-labelled tensor written through the model's own
  mesh mapper reads back with column `c` holding head `c` at `rtol=atol=0`, and the per-chip shape is
  `(1, 1, 128, 128)`, i.e. exactly the one KV head `kv_cache.py:130` allocates;
* **through the real model** — `Model.prefill_forward` writing the real cache at TP=8 scores min K
  **0.99789** / V **0.99134** over 32 layers against the fp32 golden, versus `G-CHUNK`'s hand-written
  per-head **0.99818 / 0.99206** at `(1,1)`. The mesh mapper and the TP all-reduce cost 1.16x / 1.09x
  of the error and nothing else.

The assert `_resolve_kv` raises at `TP != num_key_value_heads` stays: it is still true that the packed
cache holds one KV head per chip and that TP must equal 8. What is no longer open is whether the model
feeds it correctly. `raw/G-KV-TP8_20260903T222825Z.log`.

### `R-028` — **CLOSED.** `dense_sp_attention` is implemented and `G-CHUNK-ATTN` ran
`tt/attention/dense_sp.py` is the ring-joint port (`DEC-083`, `DEC-084`), and
`tt/attention/prefill.py` gained the SP bootstrap that `DEC-021` owed, selected by the same
cache-capacity rule as upstream. `G-CHUNK-ATTN` is promoted from `BLOCKED` to
**PASS-WITH-DEVIATION** (`DEC-085`): one attention layer of ring-vs-bootstrap difference is
**0.99996**, and the deep-layer divergence is measured accumulation (max step 1.90x against a 4.0x
ceiling), attributed to the ring's mandatory `fp32_dest_acc_en=False`.

R-028 predicted "enabling it is not a flag flip", and that was right: it needed the port, the
bootstrap, a ring program config and a ring compute-kernel config. It also predicted that
`TtPrefillRuntimeConfig(sequence_parallel=True)` would then work **with no edit** to
`tt/tt_prefill_runtime.py` because `_chunked_read_supported` *probes* the stub — and that was right
too. The runtime's only P8 change is a bring-up-only logging helper (`_log_layer_error_steps`).

### `R-029` — **CLOSED.** The read-back helpers ran on device
`gather_layer`, `dump_slot_kv`, `compare_device_dump` and `kv_cache_pcc_check` all executed against a
real device cache for the first time, at TP=8 (`G-KV-TP8`) and at SP=4 x TP=8 (`G-MESH-KV`,
`G-RACE`, `G-CHUNK-ATTN`). Both previously-unexercised pieces are now covered: the
`blockcyclic_positions` inverse at `sp > 1` (every `(4,8)` read-back inverts it, and the numbers
would collapse if it were wrong) and the `r * cols + col` indexing that maps KV head `c` to column
`c` (asserted bit-exactly, with a rotated-column negative control that scores **-0.038** against the
golden). The metadata contract test that stood in for them can stay as a cheap guard.

### `R-013` — **SETTLED by measurement. Nothing changed.**
`G-RACE` ran the 32-layer prefill three times in **one process on one `CCLManager`** — 384
all-reduces over a 2-deep barrier ping-pong plus 192 ring-attention invocations over one semaphore
pair — and produced one KV digest, `ec96afaa…`, three times. Two other processes produced the same
digest. So the depth-2 barrier ping-pong is safe for this workload and
`reset_global_semaphores` stays partial (`DEC-052` upheld, `DEC-086`). The 2 -> 4 deepening remains
the documented first move **if `G-RACE` ever goes red**; making it now, with a green gate, would be
an unfalsifiable edit. **Still open for P10:** 384 all-reduces is not 384,000, and `num_users > 1`
and a post-migration reuse of the manager are untested.

### `R-012` — **CLOSED, and its premise no longer holds on this machine.**
R-012 said the `(1,N)` parity meshes would run `num_links = 1` + `Topology.Linear` and so prove
nothing about the ring transport. On this box they cannot run as top-level meshes at all
(`DEC-080`), so every parity shape is a submesh of the galaxy running `Topology.Ring` (`DEC-081`).
The `(2,8)` parametrisation R-012 asked for is added anyway and does run `num_links = 2` with
`sp > 1`. `G-TP-PARITY` is green on all five shapes, worst cell **0.999972**.

### `R-017` — **CLOSED.** Cache-only loading proven where the cache is sharded
Two independent checks at `(4,8)`: 21 device tensors SHA-256-identical between a checkpoint build and
a `{}` + cache rebuild, each spanning all **32** device shards; and a full 32-layer cache-only prefill
producing a **byte-identical KV digest** to the checkpoint-loaded run. The `4x8` segment of
`weight_cache_path` (`DEC-048`) was exercised by a real write and read for the first time.

### `R-025` — **CLOSED for the KV product at the real chunk size; the hidden-state curve stays open.**
`MAX_LAYER_ERROR_STEP = 4.0` and `STEP_CHECK_FROM_LAYER = 3` were carried over **unchanged** and
re-measured on the `(4,8)` mesh at **2048 tokens with chunk 512** — 4x the sequence length and 4x the
chunk size P7 measured them at, and past `ProgramConfig.prefill_threshold = 2048`: max consecutive
error step **K 2.17x**, **V 1.76x**, both at layer 8, against the 4.0x ceiling. The excluded layer-2
step (K 5.35x) is the same near-exact-baseline artefact P7 recorded. The step curve is now logged by
`TtPrefillRuntime._log_layer_error_steps` on every `kv_cache_pcc_check`, so it is re-measured by
every future harness run rather than by a one-off script.

**Still open (re-assigned to P9/P10):** the *hidden-state* curve at the 8192-token default chunk. P8
measured the KV product at 2048, not the hidden state at 8192, and a 32-layer fp32 HF reference at
8192 tokens is still the large host run P7 declined.

---

## R-031 — Fabric bring-up fails on any top-level partial mesh, so every sub-shape must be a submesh (measured, P8)

**Status:** mitigated by `DEC-080` + `TestFactory.setup_submesh`; **the underlying limitation is
open** and belongs to whoever runs this package on other hardware.

`ttnn.open_mesh_device(MeshShape(1, 8))` and `(2, 8)` both open and then die in fabric bring-up with
`Fabric Router Sync: Timeout after 10000 ms … furthest-behind stage: STARTED`
(`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:200`), with and without
`TT_MESH_GRAPH_DESC_PATH`, under `STRICT_INIT` and `RELAXED_INIT`. The routers on the opened devices
wait for an ethernet handshake with partners outside the mesh, which have no kernel running.

**Consequence for the tests:** `parametrize_galaxy_submeshes` always opens the full `(4,8)` and carves
the shape, which is *not* what `models/demos/minimax_m3/tests/test_factory.py:89` does. On a LoudBox
or T3K, where `(1,8)` **is** the whole machine, the minimax form is the correct one and ours would
over-allocate. Whoever ports these tests to smaller hardware must switch back.

Evidence: `raw/G-FABRIC-MATRIX_20260903T221822Z.log`, cases `1x8:linear:1:1:toplevel` and
`2x8:ring:2:1:toplevel`.

---

## R-032 — Two overlapping submeshes hang the machine unless `quiesce_devices()` separates them (measured, P8)

**Status:** mitigated in `tests/unit/test_tp_parity.py`; **the trap is open for every future test.**

`mesh_device.hpp:296` documents that a barrier is required "between phases that use overlapping
submeshes on the same physical devices" and names `quiesce_devices()` (`:305`). Nothing enforces it.
Measured, one variable at a time on a freshly reset box:

| case | result |
|---|---|
| `(1,2)` collective, then `(1,8)` collective, both submeshes live, **no barrier** | **HANG** |
| the same two phases with `parent.quiesce_devices()` between | **ok** |
| `(1,8)` alone in its own process, same topology and link count | **ok** |

**Why this is worse than an error.** The hang is not contained: after it, *every* later collective on
the machine hangs too — including a `(4,8)` all-reduce that had passed forty seconds earlier — until
`tt-smi -r`. A pytest session that hits it turns every remaining gate into a false FAIL, and the first
diagnosis is wrong: the P8 draft blamed `Topology.Linear` on the 8-wide logical row and had a tidy
physical-mapping story for it (`DEC-081` keeps the whole wrong argument on the record). Only running
the shape **alone** falsified it.

**What a future test must do:** carve one submesh per test where possible; where two shapes are needed
(`G-TP-PARITY` compares `(1,1)` against `(1,TP)`), call `parent_mesh.quiesce_devices()` between the
phases. And any harness that *can* hang should run its cases in subprocesses with a timeout
(`DEC-082`), so a hang is a recorded measurement instead of an outage.

Evidence: `raw/G-FABRIC-MATRIX_20260903T221822Z.log`, cases `overlap-nobarrier` / `overlap-quiesce`.

---

## R-033 — The ring path can never use the fp32 accumulator, and that cost grows with depth (measured, P8)

**Status:** open by construction — not fixable in this package.

`use_streaming_compute = !fp32_dest_acc_en`
(`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:1304`), and
passing `kv_actual_isl` — the KV-pad rotation **every** chunk of this package's prefill passes —
requires the streaming path (`:1306`). So for chunked prefill the two are mutually exclusive by
construction; `fp32_dest_acc_en=True` is refused with a `TT_FATAL`, not merely discouraged.

Measured cost (`DEC-084`): the ring op alone scores **0.999784** against fp32 torch on identical
values (floor 0.999973, `err_ratio` 7.98x). End to end over 32 layers, against the fp32 golden, the
ring carries **1.45x** the error of the SP bootstrap, which keeps the accumulator: min K **0.99695**
vs **0.99789**. The gap grows smoothly with depth (max consecutive step 1.90x, ceiling 4.0x).

**Why it is filed rather than fixed:** the alternatives are all outside this package — a paged
`chunked_scaled_dot_product_attention` (needs a paged cache; this one is a DRAM `NdShard`), or an
upstream change giving the streaming compute path fp32 dest accumulation. **What it means for
P9/P10:** the deployment path's KV is ~1.45x further from fp32 than a one-shot request's, permanently,
and any future KV threshold must be set against the *chunked* number, not the one-shot one.
