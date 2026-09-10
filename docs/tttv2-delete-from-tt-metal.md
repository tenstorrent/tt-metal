# Delete TTTv2 from tt-metal

Status: **implemented** on branch `gwang/delete-tttv2-from-tt-metal`. Q1–Q10 decided.
Decided: Q1 = one-release `ImportError` stub (§7). Not a re-export.
Decided: Q2 = defer the e2e retarget. This delete does **not** install
`tt_transformers` in CI. Comment out the six TTTv2 pytest lines so
nightlies do not import stubbed code. A later PR re-enables them
against the new repo (§4.3).
Decided: Q3 = keep `LazyWeight` as `models.common.lazy_weight`.
Decided: Q4 = break Quasar’s TTTv2 import; TODO points at
`tenstorrent/tt_transformers`.
Decided: Q5 = defer vLLM. Separate PR later; not in this work.
Decided: Q7 = keep `tt_ccl.py`; move it to `models.common.tt_ccl`.
Do not fold into TTTv1 `ccl.py`.
Decided: Q8 = one mega-PR. Do not split relocate vs delete.
Decided: Q9 = do not audit or cherry-pick post-extract TTTv2
commits onto `tt_transformers`. This work only deletes TTTv2 from
tt-metal without breaking remaining tt-metal callers.
Decided: Q10 = one line in `models/common/README.md`. No
`models/docs/*` changelog.
Date: 2026-09-10
Scope: tt-metal deletion / leftover-consumer retarget. The code already lives in
[tenstorrent/tt_transformers](https://github.com/tenstorrent/tt_transformers)
(`main`, extracted from tt-metal `00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0`).
MoE was intentionally excluded from that extraction.

This is **not** a copy/move PR. It is a surgical delete plus the minimum
tt-metal surgery required so TTTv1, Galaxy, DeepSeek, gpt_oss, bge_m3, MoE
ops, and remaining `models/common` helpers still build and CI after the
package is gone.

---

## 1. Goal

Remove the in-tree TTTv2 product from tt-metal so there is one source of
truth: the `tt_transformers` package (`tt_transformers.modules`,
`.llm_runtime`, `.models`, `.sampling`).

Success:

- No TTTv2 *implementations* remain under `models.common.modules`,
  `models.common.llm_runtime`, or `models.common.models`. Those three
  packages become one-release `ImportError` stubs that name the
  `tt_transformers` import (see §7). Relocated keepers are not behind
  the stub.
- In-tree TTTv2 *module / runtime / demo* CI jobs and sweeps are gone.
  The six product e2e nightlies that currently pytest
  `models/common/tests/demos` are **commented out** in this work. A
  later PR re-enables them against `tt_transformers` (§4.3).
- TTTv1 (`models/tt_transformers`), Galaxy, DeepSeek, gpt_oss, bge_m3,
  MoE gate/decode, and shared `models/common` helpers still work.
- tt-metal `pyproject` does **not** depend on `tt_transformers`. No CI
  clone/install of that repo in this work.

Non-goals:

- Do not copy more code into `tt_transformers`.
- Do not cherry-pick or reconcile post-extract TTTv2 drift onto
  `tt_transformers`. New-repo completeness is out of scope.
- Do not delete TTTv1 (`models/tt_transformers/**`).
- Do not delete `models/common/sampling` (on-device sampling used by
  TTTv1 / Galaxy / gpt_oss / VL / train).
- Do not delete MoE (`models/common/modules/moe` was never TTTv2).
- Do not “improve” leftover consumers beyond making them compile.
- Do not re-export `tt_transformers` from the old paths (that is a new
  dep). The stub raises; it does not forward.

---

## 2. What TTTv2 is (in this repo)

Two layers, plus the models that compose them:

| Layer | tt-metal path | New-repo path |
| --- | --- | --- |
| Reusable modules | `models/common/modules/{attention,embedding,lm_head,mlp,rmsnorm,rope,sampling,lazy_*,tt_ccl}` | `src/tt_transformers/modules/` |
| LLM runtime | `models/common/llm_runtime/` | `src/tt_transformers/llm_runtime/` |
| Shared model executor | `models/common/models/executor.py`, `llama3_executor.py`, `qwen2_executor.py` | `src/tt_transformers/models/` |
| 12 concrete models | `models/common/models/{llama32_1b,llama32_3b,llama3_8b,llama33_70b,qwen2_7b,qwen25_7b,qwen25_72b,qwen25_coder_32b,qwen3_32b,mistral_7b,phi4,deepseek_r1_distill_qwen_14b}/` | same names under `src/tt_transformers/models/` |
| Demos | `models/common/tests/demos/` | `examples/` + `tests/hardware/models/` |
| Module / runtime / model tests | `models/common/tests/{modules,llm_runtime,models}/` | `tests/{modules,llm_runtime,models}/` |
| Qualification | `models/tttv2_*`, `models/test_tttv2_validate_vllm_matrix.py` | `qualification/` + `tests/qualification/` + `tests/hardware/capabilities/` |

Canonical import remap (for leftover consumers / docs only):

```
models.common.modules.X          ->  tt_transformers.modules.X
models.common.llm_runtime.X      ->  tt_transformers.llm_runtime.X
models.common.models.X           ->  tt_transformers.models.X
models.common.modules.lazy_weight ->  tt_transformers.modules.lazy_weight
```

`models.common.sampling` is **not** this remap. That package stays in
tt-metal. The new repo has its own copy under `tt_transformers.sampling`.

---

## 3. Hard keep (do not delete)

These live next to TTTv2 but are used by the rest of tt-metal.

### 3.1 Shared `models/common` production code

| Path | Why it stays |
| --- | --- |
| `lightweightmodule.py` | TTTv1, Galaxy, DeepSeek, SDXL, VL, DiT, experimental models |
| `rmsnorm.py` | TTTv1 / older stacks (not `modules/rmsnorm`) |
| `sampling/` | TTTv1, Galaxy, gpt_oss, qwen36, VL, gemma4, minimax, train GRPO |
| `warmup/` | on-device sampling / TTTv1 warmup |
| `auto_compose.py` | bge_m3, experimental, validation |
| `validation_tools.py`, `metrics.py`, `distribute_as.py` | repo-wide validation |
| `tensor_utils.py` | bge_m3 (`TILE_SIZE`), gemma4_d_p (`get_rot_transformation_mat`) |
| `device_utils.py` | used outside TTTv2; keep unless proven unused after delete |
| `utility_functions.py`, `utils.py`, `helper_funcs.py`, `generation_utils.py`, `reference_rope.py` | shared host helpers |
| `README.md` | validation-tools guide, not the TTTv2 modules guide |

### 3.2 TTTv2-adjacent but **not** TTTv2 — must relocate, not delete

| Path | Consumers after delete | Proposed new home |
| --- | --- | --- |
| `models/common/modules/moe/**` | MoE gate/decode + `ops_unit_tests.yaml` + `run_python_model_tests.sh` | `models/common/moe/` (keep `models.common.moe` import) |
| `models/common/tests/modules/moe/**` | same | `models/common/tests/moe/` |
| `models/common/modules/tt_ccl.py` | **TTTv1** `models/tt_transformers/tt/ccl.py` (`get_num_links`), **tt_dit** `models/tt_dit/utils/test.py` (`get_num_links` via `skip_if_unsupported_num_links`), `tests/test_mesh_fixture_policy.py` (`default_topology`) | **`models/common/tt_ccl.py`** (decided). Move the file; retarget those three. Do not fold into TTTv1. |
| `models/common/modules/lazy_weight.py` | **bge_m3** (`tt/{norm,mlp,attention,embeddings,tiny_model,weight_adapter}.py` + tests) | **`models/common/lazy_weight.py`** (decided). Update bge_m3 imports. |

`tt_ccl` and `lazy_weight` were copied into the new repo. Leaving the
tt-metal copies is **not** a dual product — they are now shared infra that
happens to have been invented for TTTv2.

### 3.3 Shared tests that must stay

The job `TT-Transformers common unit tests` currently does:

```bash
pytest --ignore=models/common/tests/modules models/common/tests
```

That mixes TTTv2 (`llm_runtime/`, `models/`, `demos/`) with keepers:

| Keep | Notes |
| --- | --- |
| `tests/test_sampling.py` | CI job `On-device sampling unit tests`; also gpt_oss `run_logprobs_tests.sh` |
| `tests/test_tt_sampling.py`, `test_tt_log_probs_state.py`, `test_sampling_vocab_padding.py` | sampling package |
| `tests/test_auto_compose.py` | CI job currently tagged `model_family: TTTv2` — **retag**, do not delete |
| `tests/test_metrics.py`, `tests/host/test_metrics_pytorch_only.py` | |
| `tests/test_validation_tools.py`, `test_distribute_as.py`, `test_utils.py` | |
| `tests/host/test_utility_functions_imports.py` | |
| `tests/modules/moe/**` | after relocate |
| slimmer `tests/conftest.py` | device-lock / mesh fixtures still used by remaining device tests |
| `tests/test_lazy_weight.py` | stays; retarget import to `models.common.lazy_weight` |
| `tests/test_mesh_fixture_policy.py` | stays; retarget `default_topology` to `models.common.tt_ccl` |

Delete the TTTv2-only tests listed in §4. Rewrite the “common unit tests”
job so it no longer crawls `llm_runtime/` / `models/` / `demos/`.

### 3.4 Entire trees that are **not** this work

- `models/tt_transformers/**` — TTTv1. Stays. Only change: import
  `get_num_links` from `models.common.tt_ccl` instead of
  `models.common.modules.tt_ccl`.
- `models/demos/**` except leftover TTTv2 imports (bge_m3 LazyWeight).
- `models/experimental/tt_transformers_v2/` — not TTTv2 product; only uses
  `auto_compose` / `validation_tools`. Leave it.

---

## 4. Hard delete

Delete these once leftover consumers in §5 are retargeted.

### 4.1 Production

Delete the implementations. Leave the three package roots as stubs (§7).
In this PR, relocate moe / `tt_ccl` / `lazy_weight` *before* writing
the `modules/` stub — a raising `modules/__init__.py` would also
kill `from models.common.modules.lazy_weight import ...`. Same-PR
ordering, not a second review.

```
models/common/llm_runtime/**        # implementations + READMEs + vLLM skill
                                    # keep only llm_runtime/__init__.py stub
models/common/models/**             # 12 families + executor.py + *executor.py + READMEs
                                    # keep only models/__init__.py stub
models/common/modules/README.md
models/common/modules/attention/
models/common/modules/embedding/
models/common/modules/lm_head/
models/common/modules/mlp/
models/common/modules/rmsnorm/
models/common/modules/rope/
models/common/modules/sampling/     # Sampling1D / Penalties1D — NOT models/common/sampling
models/common/modules/lazy_buffer.py
# lazy_weight.py and tt_ccl.py: relocate first (see §3.2), then delete old path
# moe/: relocate first, then delete old path
# modules/__init__.py: stub, not deleted
```

### 4.2 Tests / demos / qualification still in tt-metal

```
models/common/tests/llm_runtime/
models/common/tests/models/
models/common/tests/demos/
models/common/tests/modules/attention/
models/common/tests/modules/embedding/
models/common/tests/modules/lm_head/
models/common/tests/modules/mlp/
models/common/tests/modules/rmsnorm/
models/common/tests/modules/rope/
models/common/tests/modules/sampling/
models/common/tests/modules/test_lazy_buffer.py
models/common/tests/modules/test_tensor_utils.py   # tests TTTv2 tensor_utils usage; keepers have their own
models/common/tests/test_lazy_weight.py            # KEEP — retarget to models.common.lazy_weight
models/common/tests/test_llama3_8b_hf_adaptor.py
models/common/tests/test_bh_required_capabilities.py
models/common/tests/test_device_lock.py            # only if no remaining consumer after conftest slim-down
models/common/tests/test_mesh_fixture_policy.py    # KEEP — retarget default_topology to models.common.tt_ccl
models/tttv2_bh_required_capabilities.schema.json
models/tttv2_llama3_8b_bh_required_capabilities.json
models/tttv2_llama33_70b_bh_required_capabilities.json
models/tttv2_qwen3_32b_bh_required_capabilities.json
models/tttv2_validate_bh_required_capabilities.py
models/tttv2_validate_vllm_matrix.py
models/tttv2_vllm_hardware_gate_runner.sh
models/test_tttv2_validate_vllm_matrix.py
```

`models/common/readiness_check/` is already gone locally (moved to the new
repo’s `qualification/readiness/`).

### 4.3 CI — delete or rewrite

**Delete** (`model_family: TTTv2` module/runtime jobs):

- `tests/pipeline_reorg/models_unit_tests.yaml`
  - `TT-Transformers common unit tests` — **rewrite**, not delete: it also
    runs keeper tests. After delete it must list keeper paths only.
  - `TT-Transformers {MLP,RMSNorm,RoPE,LM head,attention,embedding,sampling,RMSNorm 2D,MLP 2D} module unit tests`
  - `TT-Transformers auto-compose 2D unit tests` — **keep the pytest**, drop
    `model_family: TTTv2`, point at `models/common/tests/test_auto_compose.py`
- `tests/pipeline_reorg/models_sweep_tests.yaml`
  - `TT-Transformers {MLP,RoPE,RMSNorm,attention,embedding,LM head} module sweep`
- `.github/workflows/t3000-unit-tests.yaml` extra `tttv2 modules`
- `tests/scripts/t3000/run_t3000_unit_tests.sh` TTTv2 module coverage blocks
  (lines that pytest `models/common/tests/modules/{mlp,rmsnorm,...}`)
- generated locks that enumerate `"tttv2 modules"`:
  `.github/workflows/{test-command,silencer}.lock.yml` — regenerate, do not
  hand-edit

**Six TTTv2-substituted e2e legs — decided: do nothing in this PR
except park them.** Do not install `tt_transformers`. Do not revert to
TTTv1. Do not leave cmds pointing at files we delete (those demos
import stubbed `models.common.models` / `llm_runtime` and would go red).

This delete PR: comment out the TTTv2 pytest lines. Leave a `# TODO`
pointing at a follow-up that mikadoes the retarget. Job entries can
stay so the later PR is a small diff.

| Job | This PR | Later PR (not this work) |
| --- | --- | --- |
| Llama 3.2-1B / 3B e2e | comment out the `models/common/tests/demos/...` pytest | re-enable against `tt_transformers` `tests/hardware/models/*/test_demo.py` |
| Qwen3-32B e2e (T3K) | comment out both TTTv2 pytests | same |
| Qwen2.5-Coder-32B / 7B e2e | comment out | same |
| Mistral-7B e2e | comment out the TTTv2 pytest only; **keep** the TTTv1 `simple_text_demo.py` eval-32 sibling | re-enable the TTTv2 line |
| Qwen2.5-72B e2e (already commented) | leave commented | retarget when/if re-enabled |

BH qwen3-32B / qwen25-coder-32B stay on TTTv1 `simple_text_demo.py`.

**Later PR sketch** (so we do not forget the landmines): clone a pinned
SHA of the private `tt_transformers` repo, `pip install --no-deps -e`
(their `pyproject` pins `ttnn==0.77.0` — must not fight CI’s in-tree
wheel), put the checkout on `PYTHONPATH` (wheel does not ship
`examples/` / `tests/`; wrappers do `from examples.<model> import demo`),
pytest `tests/hardware/models/<model>/test_demo.py`. Needs a SHA owner
and a private-clone token. Not this work.

**Keep** (not TTTv2):

- `On-device sampling unit tests` → `models/common/tests/test_sampling.py`
- MoE gate: `ops_unit_tests.yaml` + `tests/scripts/run_python_model_tests.sh`
  (update path after moe relocate)
- All TTTv1 `models/tt_transformers/demo/simple_text_demo.py` jobs

### 4.4 Docs / ownership / debug

- `.github/CODEOWNERS`: `models/common` stays (keepers). No TTTv2-specific
  path exists today; after relocate add `models/common/moe/`.
- `models/model_targets.yaml` / `models/model_trace_region_sizes.yaml`:
  strip TTTv2-only rows if any remain after the e2e revert. Do not delete
  the files (TTTv1 / other models use them).
- Release notes / skills that point at `models/common/llm_runtime` or
  `models/common/modules` as the product path: one-line “moved to
  tt_transformers”.
- `.vscode/launch.json`: no TTTv2 demo entries found; no change expected.

---

## 5. Leftover consumers that will break a naive `rm -rf`

These are the only in-tree imports of TTTv2 production code from **outside**
`models/common/{modules,llm_runtime,models,tests}`.

### 5.1 Must fix in the deletion PR (or a strict prerequisite)

| Consumer | Import | Fix |
| --- | --- | --- |
| `models/tt_transformers/tt/ccl.py` | `from models.common.modules.tt_ccl import get_num_links` | `from models.common.tt_ccl import get_num_links` |
| `models/tt_transformers/tests/test_ccl_utils.py` | `from models.common.modules import tt_ccl` | `from models.common import tt_ccl` (keep the dual-module parametrize: common + TTTv1 wrapper) |
| `models/tt_dit/utils/test.py` | `from models.common.modules.tt_ccl import get_num_links` | `from models.common.tt_ccl import get_num_links` |
| `models/common/tests/test_mesh_fixture_policy.py` | `from models.common.modules.tt_ccl import default_topology` | `from models.common.tt_ccl import default_topology` |
| `models/demos/wormhole/bge_m3/tt/{norm,mlp,attention,embeddings,tiny_model,weight_adapter}.py` + tests | `LazyWeight` / `resolve_lazy_weight` | **decided:** `from models.common.lazy_weight import ...` |
| `models/experimental/llama32_1b_quasar/models/generator.py` | `from models.common.models.llama3_8b.model import EagerLlamaExecutor, Llama3Transformer1D, TracedLlamaExecutor` | **decided:** delete that import. Leave a `TODO` that Quasar must move onto `tenstorrent/tt_transformers`. The old path hits the stub until they do. |

### 5.2 External (not in this checkout) — coordinate, do not ignore

| Consumer | Why it matters |
| --- | --- |
| TT vLLM plugin | `TT_*_TEXT_VER=tt_transformers_v2` still maps to `models.common.models.*.generator`. Default / current vLLM CI is TTTv1 and stays up. **Out of scope for this work** — follow-up PR in the vLLM plugin later. Until then the v2 opt-in path hits the stub `ImportError`. |
| Any out-of-tree notebook / internal fork still doing `from models.common.modules...` | gets the stub `ImportError` with the new import path (one release), then `ModuleNotFoundError` after the stub is deleted |

### 5.3 Do **not** treat as TTTv2 leftover

- Galaxy / gpt_oss / VL / `models/tt_transformers/demo` importing
  `models.common.sampling` — keeper.
- Anything importing `LightweightModule`, `auto_compose`, `validation_tools`.
- Quasar’s vendored `models/experimental/llama32_1b_quasar/modules/**` —
  already a fork; leave it. Only the `llama3_8b.model` import is broken
  with a TODO (Q4).

---

## 6. Recommended execution (after decisions)

**One mega-PR.** Relocate keepers and delete TTTv2 in the same review.
Do not split. Commit order inside the PR still matters: relocate
first, then write the raising stubs / delete the rest. A raising
`modules/__init__.py` that lands while keepers still import
`models.common.modules.X` will break bge_m3 and TTTv1 CCL.

### This PR — relocate keepers, then delete TTTv2 + CI

1. Move `modules/moe/` → `models/common/moe/` and update imports + CI paths
   (`ops_unit_tests.yaml`, `run_python_model_tests.sh`,
   `tests/ttnn/.../test_generalized_moe_gate_program_cache.py` comment).
2. Move `modules/tt_ccl.py` → `models/common/tt_ccl.py` (whole file).
   Update TTTv1 `ccl.py` + `test_ccl_utils.py`, tt_dit
   `utils/test.py`, and `tests/test_mesh_fixture_policy.py`.
   After the delete, in-tree leftovers only call `get_num_links` /
   `default_topology`; `TT_CCL` / `get_tt_ccl` have no remaining
   callers. Still move the whole file — do not slim it.
3. Move `modules/lazy_weight.py` → `models/common/lazy_weight.py`.
   Update bge_m3 + `tests/test_lazy_weight.py` to
   `from models.common.lazy_weight import ...`. Do **not** re-export
   from `models.common.modules.lazy_weight` — that path dies with the
   stub in the same PR.
4. Do **not** slim `models/common/tests/conftest.py` unless a remaining
   keeper test requires it.
5. Delete the implementations in §4.1–§4.2.
6. Write the three `ImportError` stubs in §7. Inline the message.
   No helper module. No stub test.
7. Remove in-tree TTTv2 module/sweep jobs. Comment out the six TTTv2
   e2e pytest lines per §4.3. Do not add a `tt_transformers` setup
   script.
8. Quasar `models/experimental/llama32_1b_quasar/models/generator.py`:
   remove the `models.common.models.llama3_8b.model` import. Add a
   `TODO` that this experimental path must transition to
   `https://github.com/tenstorrent/tt_transformers`. Do not port it
   in this work.
9. CODEOWNERS + one line at the top of `models/common/README.md`:
   TTTv2 modules/runtime/models live in tenstorrent/tt_transformers;
   old imports raise. Do not touch `models/docs/*`.
10. Grep gate — implementations gone; stubs + this plan + comments OK:

```bash
rg -n "from models\.common\.(modules|llm_runtime|models)" --glob '*.py'
rg -n "models/common/(tests/demos|tests/llm_runtime|tests/models)" \
  tests/pipeline_reorg .github/workflows tests/scripts
```

Allowed remaining hits: the three stub `__init__.py` files,
`models.common.moe` / `.tt_ccl` / `.lazy_weight`, this plan,
and comments that say “moved”.

Verify: host pytest on keeper tests; no configure/build needed (Python
only). Device: MoE + sampling unit + one TTTv1 e2e + bge_m3 PCC. Cannot
re-run TTTv2 device tests in tt-metal after this PR — that is the point.

### Later PRs (not this work)

- **Mikado e2e:** re-enable the six parked nightlies against a pinned
  `tt_transformers` checkout (`--no-deps`, private clone, hardware
  `test_demo.py`). See §4.3 later-PR sketch.
- Delete the three stubs after one release (they become
  `ModuleNotFoundError`).
- vLLM plugin: remap `TT_*_TEXT_VER=tt_transformers_v2` strings to
  `tt_transformers.models.*.generator`.
- Strip dead TTTv2 rows from `model_targets.yaml` /
  `model_trace_region_sizes.yaml`.
- Delete this plan file.
- Quasar: actually port onto `tt_transformers` (the TODO from this PR).

No `tt_transformers` package pin in tt-metal `pyproject`. AGENTS.md:
new external dependency needs infra review. Default is **no**.

---

## 7. Compatibility policy — **decided: ImportError stub, one release**

No implementations. No re-export. No `tt_transformers` dependency.
Old package imports raise with the new path.

Leave exactly these three files after this PR (nothing else in those trees):

- `models/common/modules/__init__.py`
- `models/common/llm_runtime/__init__.py`
- `models/common/models/__init__.py`

A raising parent `__init__.py` also catches deep imports
(`from models.common.modules.mlp.mlp_1d import MLP1D`,
`from models.common.models.llama3_8b.generator import Llama3Generator`)
because Python loads the parent package first. No nested stub packages.

Inline `raise ImportError(...)` in each of the three `__init__.py`
files. No shared helper module. No host test for the stub.

Do **not** use `__getattr__` forwarding. Do **not** `import tt_transformers`.

Duration: one tt-metal release. A later PR deletes the three files.
After that, the same import is a plain `ModuleNotFoundError`.

---

## 8. Drift / completeness check against the new repo

Use
[`docs/provenance/source_inventory.csv`](https://github.com/tenstorrent/tt_transformers/blob/main/docs/provenance/source_inventory.csv)
as the delete checklist. Disposition meanings for **this** work:

| Disposition in inventory | Action in tt-metal |
| --- | --- |
| `renamed` / `split` / `replaced` of `llm_runtime`, `modules` (non-MoE), `models/*`, TTTv2 tests/demos, `models/tttv2_*` | delete |
| `excluded` (`modules/moe/**`, moe tests) | keep, relocate |
| `renamed` of `models/common/sampling/**` | **keep the tt-metal copy** — new repo forked it; Galaxy/TTTv1 still own this tree |
| `renamed` of `lightweightmodule`, `tensor_utils`, `device_utils`, `auto_compose`, `validation_tools`, `metrics` | **keep the tt-metal copy** |
| `archived` CI YAML snapshots | delete the live TTTv2 jobs (the archive already exists in the new repo’s audit tag) |
| `split` of `models/tt_transformers/tt/{common,generator,model_config,rope}.py` | **do not delete TTTv1 files** — only the TTTv2 bridge usage goes away with the modules |

Post-extract drift is **accepted**. This work does **not** cherry-pick
tt-metal TTTv2 commits after `00748e6` onto `tt_transformers`. Job is
delete the in-tree product without breaking remaining tt-metal callers
(TTTv1, tt_dit, bge_m3, MoE, sampling). What the new repo is missing
is out of scope.

---

## 9. Open decisions (block implementation)

Answer these. Defaults in **bold**.

1. **Compatibility?** **Decided: `ImportError` stub for one release**
   (§7). Not a hard cut (`ModuleNotFoundError` with no hint). Not a
   re-export (no `tt_transformers` dep).
2. **tt-metal e2e after delete?** **Decided: park the six TTTv2 pytest
   lines in this PR. Retarget to `tt_transformers` in a later PR.**
   Do not install the package now. Do not revert to TTTv1. Comment out
   so nightlies do not import stubbed code. Keep mistral’s TTTv1 sibling.
3. **`LazyWeight` in tt-metal?** **Decided: keep as
   `models.common.lazy_weight`.** Relocate the file, retarget bge_m3.
   Do not vendor. Do not import from `tt_transformers`.
4. **Quasar `llama32_1b_quasar`?** **Decided: break the
   `models.common.models.llama3_8b.model` import.** Leave a `TODO` that
   this experimental code should transition to
   `tenstorrent/tt_transformers`. No port in this work.
5. **vLLM cutover?** **Decided: out of scope.** Default / current vLLM
   CI is TTTv1 and stays up. The v2 opt-in path will hit the stub until
   a later vLLM-plugin PR remaps `platform.py`. Do not block this delete
   on that PR.
6. **May tt-metal depend on the `tt_transformers` package?** **No in
   this work** (no `pyproject` dep, no CI clone). The later e2e mikado
   PR may add a git SHA + `--no-deps` install; that PR owns the pin
   fight (`ttnn==0.77.0`) and the private-clone token.
7. **`tt_ccl` owner?** **Decided: keep the file, move it to
   `models.common.tt_ccl`.** Do not fold into TTTv1 `ccl.py` (that
   would make Flux/Wan/Mochi tests import the LLM stack for a
   device-name lookup). Retarget TTTv1 + tt_dit +
   `test_mesh_fixture_policy.py`.
8. **PR shape?** **Decided: one mega-PR.** Relocate + delete + stubs +
   CI in the same review. Do not split. Commit order inside the PR:
   relocate keepers, then stub/delete. Later mikado / stub-removal /
   vLLM stay out of this PR.
9. **Post-extract cherry-picks?** **Decided: skip.** Do not audit
   `00748e6..HEAD` or land missing commits on `tt_transformers`.
   This PR only deletes TTTv2 from tt-metal and retargets leftover
   tt-metal callers. New-repo completeness is someone else’s problem.
10. **Docs?** **Decided: one line in `models/common/README.md`.**
    Point at `tenstorrent/tt_transformers`; mention the stub. Do not
    edit `models/docs/MODEL_UPDATES.md` or bring-up docs. The stub
    `__init__.py` is the import-time message.

---

## 10. Risks

- **Post-extract TTTv2 fixes stay only in the deleted tree.** Commits
  after `00748e6` on TTTv2 paths (notably #53575 BH runtime) are not
  audited or ported. Accepted. New-repo owners own that.
- **Coverage hole until the mikado PR.** Commenting out the six TTTv2
  e2e lines drops those HF models from tt-metal T2/T3 nightlies until
  the later retarget lands. Accepted. BH / TTTv1 siblings stay.
- **vLLM v2 opt-in (deferred).** Default serving and
  `vllm_model_tests.yaml` stay on TTTv1. `TT_*_TEXT_VER=tt_transformers_v2`
  hits the stub until a later plugin PR. Accepted; not a default-nightly
  outage.
- **Stub vs keeper collision.** If the `modules/` stub is written
  before moe / `tt_ccl` / `lazy_weight` are relocated, bge_m3 and
  TTTv1 CCL die on the parent-package `ImportError`. Relocate first
  in the same PR; do not `rm -rf models/common/modules` and then
  remember the keepers.
- **`tt_ccl` / `LazyWeight` accidental delete.** Both sit under
  `modules/`. A glob `rm -rf models/common/modules` without moving
  them first breaks TTTv1 CCL and bge_m3.
- **`models/common/sampling` vs `modules/sampling`.** Easy to delete the
  wrong one. The keeper is `models/common/sampling/`. The delete is
  `models/common/modules/sampling/`.
- **“common unit tests” job is a landmine.** It pytest’s the whole
  `models/common/tests` tree. After delete it must be rewritten to an
  explicit keeper list or it will either vanish needed sampling tests or
  keep failing on missing TTTv2 paths.
- **Generated workflow locks.** Hand-editing `*.lock.yml` will be
  overwritten. Regenerating them is part of this PR.
- **Later e2e mikado landmines** (not this PR): `ttnn==0.77.0` pin
  fight if someone `pip install`s with deps; private clone; examples
  not on the wheel; SHA drift. Documented in §4.3 so the follow-up
  does not rediscover them.

---

## 11. Verification (when implementing)

Python-only change. No `copilot-build.sh`.

```bash
# no implementation imports (stubs + this plan OK)
rg -n "from models\.common\.(modules|llm_runtime|models)" --glob '*.py'

# keeper tests still collected
pytest models/common/tests/test_sampling.py \
       models/common/tests/test_auto_compose.py \
       models/common/tests/test_validation_tools.py \
       models/common/tests/modules/moe \
       models/tt_transformers/tests/test_ccl_utils.py \
       --collect-only

# device (human / CI, not this runner)
# - MoE gate
# - on-device sampling unit
# - one TTTv1 simple_text_demo
# - bge_m3 PCC
```

State in the PR: unverified on silicon in the agent environment; device
results are not claimed.

---

## 12. File-level checklist (this PR)

Use this as the reviewer’s punch list. Relocations and deletes land
together.

- [ ] `models/common/llm_runtime/` is only the stub `__init__.py`
- [ ] `models/common/models/` is only the stub `__init__.py`
- [ ] `models/common/modules/` is only the stub `__init__.py` (moe / tt_ccl / lazy_weight already moved)
- [ ] `models/common/modules/{attention,embedding,lm_head,mlp,rmsnorm,rope,sampling,lazy_buffer.py,README.md}` gone
- [ ] TTTv2 test trees in §4.2 gone
- [ ] `models/tttv2_*` + `models/test_tttv2_validate_vllm_matrix.py` gone
- [ ] pipeline YAML TTTv2 module/sweep jobs gone
- [ ] six TTTv2 e2e pytest lines commented out with a TODO to the later mikado PR
- [ ] mistral TTTv1 `simple_text_demo.py` sibling still runs
- [ ] no `setup_tt_transformers.sh` / no CI clone of `tt_transformers`
- [ ] t3000 unit-test extra + script blocks gone
- [ ] workflow locks regenerated
- [ ] CODEOWNERS updated for `models/common/moe/`
- [ ] `models/common/lazy_weight.py` exists; bge_m3 + `test_lazy_weight.py` retargeted
- [ ] `models/common/tt_ccl.py` exists; TTTv1 `ccl.py` + `test_ccl_utils.py`, tt_dit `utils/test.py`, and `test_mesh_fixture_policy.py` retargeted
- [ ] Quasar `generator.py` no longer imports `models.common.models.llama3_8b`; TODO names `tenstorrent/tt_transformers`
- [ ] `models/common/README.md` has one line pointing at tt_transformers
- [ ] `models/docs/*` untouched
- [ ] grep gate in §6 is clean (stubs allowed)
- [ ] no `pyproject` / `find_package` / `CPMAddPackage` / CI pip dep on `tt_transformers`
