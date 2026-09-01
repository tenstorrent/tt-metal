# HANDOFF — AttnRes 4/4: the device model, the walk, and its tests

**PR** [#53318](https://github.com/tenstorrent/tt-metal/pull/53318) ·
**branch** `nmilicevic/bringup/kimi-k3-attnres-3-model` ·
**head** `a047480ee93` · **base** `origin/main` at `65a1138e26b`

> **Not for merge.** This file is context for picking the work up on another machine.
> Drop it before the PR lands: `git rm HANDOFF.md && git commit --amend` (it is the tip commit).

---

## 0. The series

Kimi K3 attention residuals (AttnRes) on Blackhole, split out of omnibus PR
[#52676](https://github.com/tenstorrent/tt-metal/pull/52676) into four merge-able PRs.

| PR | what | branch suffix | number | state |
|---|---|---|---|---|
| 1/4 | torch reference | `-0-reference` | #53114 | **MERGED** 2026-08-25 (**squash**) |
| 2/4 | `attn_res_weighted_reduce_nc` | `-1-weighted-reduce` | #53116 | open, **APPROVED, MERGEABLE** |
| 3/4 | `attn_res_gather_softmax` | `-2-gather-softmax` | #53126 | open, review required, conflicts |
| **4/4** | **device model + walk + tests** | `-3-model` | **#53318** | **open, review required, conflicts** |

**This branch stacks on all three.** It contains 2/4's op, 3/4's op, and 1/4's reference in
full, because it was branched before any of them merged. 39 commits, 62 files, +7817/-1.
Its **own** contribution is 23 files / +3543 under `models/demos/deepseek_v3_d_p/`, plus the
`.github/time_budget.yaml` bump and one CI row.

**Read 3/4's HANDOFF first** if you want the op internals; this one covers the model layer.

---

## 1. Goal and current state

**Goal.** The Kimi K3 attention-residual stream as a `LightweightModule` the prefill block can
call: `TtAttnRes` (the math), `TtAttnResStream` / `TtAttnResWalk` (the schedule over a 93-layer
stack), `AttnResWeights` (the folded queries + their tensorbin cache), and a test suite that
gates all of it against the merged torch reference.

**State: the module and its tests are DONE and green. Two integration items are NOT started.**

| item | state |
|---|---|
| `TtAttnRes` — `to_query`, `inter_block`, `merge` | done, PCC-gated |
| `TtAttnResStream` / `TtAttnResWalk` | done, gated single-rank |
| `AttnResWeights` + tensorbin cache | done, gated |
| Test suite (36 tests) | done, green: **36 passed, 121.90 s** on LoudBox |
| CI row on `bh_loudbox` | added; last green e2e run `32035344341` |
| **Pipeline support in `TtAttnResWalk`** | **not started** |
| **K3 block integration (`tt_prefill_block.py`)** | **not started** |

**CI at head is unproven.** The last green LoudBox e2e (`32035344341`) was at `fddab85b1e0`,
**one commit before** the tip `a047480ee93` (a clang-tidy `std::min` clamp in 3/4's program
factory). The clamp is numerically inert — for `uint32_t`, `if (a > b) a = b;` and
`a = std::min(a, b)` are the same clamp — and the suite was re-run locally after it
(36 passed / 121.90 s vs 125.06 s before). Re-dispatch anyway before asking for review.

**`mergeable: CONFLICTING`.** See §6 — the rebase here is the non-trivial one in the series.

**Production shape:** 2×4 Blackhole LoudBox, `HIDDEN_SIZE = 7168` global (1792 per chip),
`PER_CHIP_TOKENS = 640`, `TP_AXIS = 1` (factor 4, Galaxy's), FABRIC_2D, `LAYERS = 93`,
`READ_SITES = 24`.

---

## 2. Every file, and why

### `models/demos/deepseek_v3_d_p/tt/attn_res/` — the module (1069 lines)

| file | why |
|---|---|
| `attn_res.py` (612) | `TtAttnRes(LightweightModule)`. Three public methods: **`to_query`** (:222) folds a host query into the device layout; **`inter_block`** (:511) is the mix — batches R sites and calls `attn_res_weighted_reduce_nc`; **`merge`** (:571) is the read — one `attn_res_gather_softmax` dispatch. The constructor docstring is the module's spec; read it before changing any argument. |
| `attn_res_stream.py` (223) | `TtAttnResStream` (:22) — one block's rolling state: `merge`/`seal`/`accumulate`/`_flush`. `TtAttnResWalk` (:135) — the whole-stack schedule: `read`/`write`/`open_layer`/`finish`, with `_block_sites` (:121) laying out which sites a block owns. |
| `weights.py` (223) | `AttnResWeights` (folded pre/post/output queries), `load_attn_res_weights`, `walk_sites`, the tensorbin cache (`build_ttnn_cache`, `from_cache`, `check_cache_complete`) and its artifact naming. |
| `__init__.py` (11) | Re-exports. |

### `models/demos/deepseek_v3_d_p/reference/kimi_k3/attn_res/` — 1/4's reference, carried

`attn_res.py`, `hf_attn_res.py`, `hf_walk.py`, `weights.py`, `__init__.py`, `LICENSE-Kimi-K3`,
`tests/test_attn_res_reference.py`, `tests/test_weights.py`. **These are already on main** via
the squash-merge of #53114 — they are duplicated here only because this branch predates it, and
they are the bulk of the rebase conflict (§6).

### `models/demos/deepseek_v3_d_p/tests/attn_res/` — the suite (36 tests)

| file | why |
|---|---|
| `model/harness.py` (107) | The one place shape, fabric config and `blackhole_only` live. Fixes `HIDDEN_SIZE = 7168`, `PER_CHIP_TOKENS = 640`, `PROJ_STD = 0.02`, `FABRIC = FABRIC_2D`; provides `generator`/`random_case`/`place_case`/`read_block`/`compose`. Shape is **fixed, not parametrized** — the collective picks its algorithm from payload size, so a 64-token gate exercises a reduction the model never issues. |
| `model/test_attn_res.py` (233) | The PCC gate. `test_read_matches_reference[S1,S8]` walks a 12-layer block's 24 reads and scores each site against its own oracle — `PCC_GATE = 0.9999`, `REL_ERR_GATE = 2e-2`. `test_sequence_axis_communicates_nothing` is the **exact** gate: the same tokens under two SP placements must be bit-identical. |
| `model/test_attn_res_contract.py` (173) | `test_read_repeats_exactly_without_disturbing_its_inputs` — a read is a pure function of its operands, twice over. `test_trace_replay_matches_eager` — trace capture/replay must equal eager, which is what makes the op usable inside prefill's traced loop. |
| `model/test_forward_loop.py` (163) | `test_transformer_loop_matches_reference` — the seal cadence across a whole stack, the thing a single block's test cannot see. |
| `model/test_integration_example.py` (192) | `test_example_matches_reference` — the copy-pasteable caller. Doubles as documentation of how a block wires the walk in. |
| `assertions.py` (106) | `assert_accurate` (PCC + non-constant + finite), `assert_equal`, `assert_bit_identical`. |
| `test_assertions.py` (94) | 12 tests on the assertions themselves — a PCC helper that silently passes a constant tensor makes every gate above vacuous. Keeps `assert_bit_identical` honest about signed zero and 1 ULP. |
| `test_weight_cache.py` (80) | 6 host-only tests: cache stem stays inside the caller-owned namespace, dtypes/layouts are separated in the path, artifacts cover every query exactly once, an incomplete cache is not reported complete, `TT_KIMI_K3_PREFILL_TTNN_CACHE` wins over a checkpoint, and the walk order skips layer 0's pre-read. |
| `checkpoint_utils.py` (88) | `attn_res_tensor_cache_path`, `load_attn_res_state_dict`. |
| `conftest.py` (22) | `kimi_k3_checkpoint_dir` — returns **`None`**, not a skip, when `KIMI_K3_CKPT` is unset. |
| `fetch_query_weights.py` (93) | Stages a query-weights-only checkpoint subset. **By hand, never in CI.** |

### CI

`tests/pipeline_reorg/blackhole_e2e_tests.yaml:298`:

```yaml
- id: bh-lb-kimi-k3-attn-res
  name: bh_lb_Kimi_K3_ATTN_RES
  model-name: kimi_k3
  cmd: pytest models/demos/deepseek_v3_d_p/tests/attn_res/ -vvv --tb=short
  skus:
    bh_loudbox:
      timeout: 10
  owner_id: U09LK84G4TF # Nikola Milicevic
  team: models
```

`.github/time_budget.yaml`: `models` / `bh_loudbox` **193 → 200 minutes** (+7 for a row that
measures ~2 minutes; `verify_time_budget.py` sums declared timeouts, not measured time, and it
hard-fails if the sum exceeds the budget).

---

## 3. Design decisions, and why they are not obvious

### `hidden_size` is the **global** `d`, not the shard

The single place a sharded AttnRes returns quietly wrong numbers. The RMS normalization divides
by the full `d = 7168`; passing 1792 (this chip's shard) gives a plausible tensor with a wrong
temperature and a PCC that degrades smoothly rather than failing. The constructor docstring says
so first, in those words. Keep it there.

### `tp_axis` and `sp_axis` are separate arguments, and `tp_factor == 1` is rejected outright

A query sharded on one axis and reduced on another is a mismatch **nothing downstream can
detect** — the shapes agree. So both axes are named explicitly rather than inferred.
`tp_factor == 1` raises instead of degrading: the read's cross-shard exchange is what its one
dispatch is built around, and a degenerate identity path is code the model never executes.
This is also why there is **no single-device test arm** and there should not be one.

### `topology` is one entry **per mesh axis**

Galaxy prefill is `[LINE, RING]`. A scalar `Ring` applied to a linear axis points a collective at
a wrap link with no physical fabric edge. `697baa3d28c` passes the configured topology through to
the fused read rather than letting it default.

### The site axis is **dim 0**, and that was a 7.1× fix

Originally `inter_block` batched R = 24 read sites on the **last** dim of the softmax tensors
(`[1, C, N, R]`), and the per-site loop extracted each with a 1-wide last-dim `ttnn.slice` — off
the 32×32 tile boundary, so every extraction became untilize → slice → tilize. Measured live:
**26.8 µs per extraction vs 1.67 µs** for a tile-plane-aligned slice, 2243.1 µs per 24-site block.

Fix: permute to site-major `[R, C, N, 1]` after the scalar chain and slice **dim 0**. Standalone
probe: 2238.1 → **358.0 µs, 7.1×**, bit-identical (`max|A−B| = 0.000e+00`). Over the 186-read
schedule that was ~11.6 ms off 68.43 ms.

Two traps that survive in the code:
* The **aliasing guard** stays. A slice spanning its whole input aliases the parent buffer, so
  `R == 1` must still clone — the guard key just moved from `shape[-1]` to `shape[0]`.
* At `R == 1` the permute is skipped entirely.

### Three perf knobs on the constructor, all measured, all defaulted to the fast setting

| knob | effect |
|---|---|
| `fold_stats` | Folds the statistics across the tensor axis. **348 µs → 47 µs at C = 18.** |
| `one_pass_stats` | One pass instead of two: squares **782 → 232 µs**, dots **793 → 450 µs**. |
| `stats_dtype` | fp32 gives **0.9999500** vs bf16's **0.9999401** over 186 chained reads — both clear the 0.9999 gate, and the margin is thin enough that this is a knob, not a constant. |

### CI runs on **random** weights, deliberately

`conftest.py`'s `kimi_k3_checkpoint_dir` returns `None` rather than skipping. The assertions do
not depend on weight values, so the random arm is the one CI takes and the checkpoint is opt-in
via `KIMI_K3_CKPT`. **CI must never download real weights.** `fetch_query_weights.py` exists to
stage a query-only subset by hand.

`PROJ_STD = 0.02` puts `‖q‖₂ ≈ 1.7`; K3's own query weights run 0.07–0.23 over a block, which is
a near-uniform softmax and a *milder* shift for the online rescale. **The random arm is the
harder of the two** — that is why it is the default and not a fallback.

### Two cache entry points, both gated

A complete tensorbin cache reads with no checkpoint present — that is the state a brought-up
model ships in. A checkpoint without one builds the cache first. `TT_KIMI_K3_PREFILL_TTNN_CACHE`
wins over a checkpoint (`test_cache_root_takes_the_env_var_over_a_checkpoint`). When weights are
cached, `_device_queries` places them a **second** time rather than copying the host tensors
across, which is what puts the cache path itself inside the PCC gate.

### 93 layers, 187 queries, **186** reads

Layer 0 has no sealed snapshot, so its pre-attention read is skipped: 92 pre + 93 post + 1
model-level read after the stack. `test_walk_order_skips_layer_zeros_pre_read` is the gate; an
off-by-one here shifts every query by one layer and still produces plausible numbers.

### `test_sequence_axis_communicates_nothing` is an **exact** gate, not a PCC gate

Same tokens under two SP placements → **bit-identical** outputs. The sequence axis must carry no
information; a PCC gate would pass a version that leaked a little. This is also the only gate
that would catch an accidental all-reduce over the wrong axis.

### `test_trace_replay_matches_eager` exists because of `site`

`site` patches runtime args on a cached program (see 3/4's handoff). Trace capture freezes those
args. If the per-site patch ever stopped surviving capture, every traced read would silently
return site 0's plane. Eager-vs-replay equality is the only thing that catches it at this level.

### No Galaxy `(8, 4)` arm

Same TP factor over a wider sequence axis, which the op is indifferent to. It would cost 32
chips to re-run the reduction `mesh-2x4` already covers. `mesh_device` skips a placement asking
for more chips than the host has, so single-card Blackhole SKUs **collect and skip** this file
rather than failing.

---

## 4. Dead ends — measured, and not worth repeating

| tried | result |
|---|---|
| Site axis on the last dim with 1-wide slices | 16× slower per extraction (26.8 vs 1.67 µs). The permute to site-major is not optional. |
| Two-pass statistics | 782/793 µs vs 232/450 µs one-pass. |
| Unfolded statistics across the tensor axis | 348 µs vs 47 µs at C = 18. |
| Fusing the `hc_post` composite into one kernel (**adjacent mHC work, same lesson**) | Bandwidth-bound; ~1.2× ceiling. Rejected — the composite stands. |
| bf16 statistics as the default | Works (0.9999401 over 186 chained reads) but the margin over the 0.9999 gate is ~4e-6. Kept as a knob, not the default. |
| A single-device (`1x1`) arm | The reduce becomes an identity. `TtAttnRes` rejects `tp_factor == 1`, so the arm cannot exist. |
| Parametrizing the test shape down to 64 tokens | The collective picks its algorithm from payload size — it would gate a reduction the model never issues. |

**Also see 3/4's handoff §4** for the op-level dead ends (per-packet fabric law, the second
fabric link, wide folds, faces-only sends). None of them are worth re-testing from this layer.

---

## 5. Build, test, repro

**Hardware:** Blackhole **LoudBox (8× P150)**, 2×4 mesh, FABRIC_2D. Single-card Blackhole SKUs
collect and skip. Wormhole is excluded by `blackhole_only`.

**Landmines:**

* Bare `python` is `/opt/venv` and imports `ttnn` as a **hollow namespace package with no error**.
  Always `./python_env/bin/python`.
* After any branch switch: `git submodule update --init --recursive tt_metal/third_party/umd`.
* Build **only** through `bash build_metal.sh`.
* A rebase invalidates the local `_ttnn.so`; the symptom is `import ttnn` raising `AttributeError`
  on a `_ttnn` symbol from inside `conftest.py`. Stale build, not a test failure.
* Do **not** wipe `~/.cache/tt-metal-cache` (18 GB, shared); use `TT_METAL_CACHE=<scratch>`.
* `gh` is no longer on `PATH`; use `/proj_sw/user_dev/moconnor/bin/gh`. `gh pr edit` fails on this
  repo — use `gh api -X PATCH repos/tenstorrent/tt-metal/pulls/<N> -F body=@<file>`.
* `git rebase -i` is not supported in this environment.

```bash
cd /localdev/nmilicevic/tt-metal
git checkout nmilicevic/bringup/kimi-k3-attnres-3-model
git submodule update --init --recursive tt_metal/third_party/umd
bash build_metal.sh

# the exact CI command — expect 36 passed, ~122 s
./python_env/bin/python -m pytest models/demos/deepseek_v3_d_p/tests/attn_res/ -vvv --tb=short

# just the PCC gate
./python_env/bin/python -m pytest \
  models/demos/deepseek_v3_d_p/tests/attn_res/model/test_attn_res.py -v

# host-only, no device needed
./python_env/bin/python -m pytest \
  models/demos/deepseek_v3_d_p/tests/attn_res/test_assertions.py \
  models/demos/deepseek_v3_d_p/tests/attn_res/test_weight_cache.py -v

# the two ops this stacks on
./python_env/bin/python -m pytest \
  tests/ttnn/unit_tests/operations/experimental/test_attn_res_weighted_reduce_nc.py -xv --timeout=60
./python_env/bin/python -m pytest \
  tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py -k "mesh-2x4"

# real weights — OPT-IN, by hand, never in CI
./python_env/bin/python models/demos/deepseek_v3_d_p/tests/attn_res/fetch_query_weights.py
KIMI_K3_CKPT=<dir> ./python_env/bin/python -m pytest .../model/test_attn_res.py -v
# or, with a prebuilt cache and no checkpoint at all:
TT_KIMI_K3_PREFILL_TTNN_CACHE=<dir> ./python_env/bin/python -m pytest .../model/test_attn_res.py -v

# CI budget validation — third arg is the workflow KEY, literally `e2e`
./python_env/bin/python .github/scripts/utils/verify_time_budget.py \
  tests/pipeline_reorg/blackhole_e2e_tests.yaml .github/time_budget.yaml e2e
# and for 2/4's row, which this branch also carries:
./python_env/bin/python .github/scripts/utils/verify_time_budget.py \
  tests/pipeline_reorg/ops_unit_tests.yaml .github/time_budget.yaml unit

# dispatch CI
gh workflow run "(Blackhole) e2e tests" \
  --ref nmilicevic/bringup/kimi-k3-attnres-3-model \
  -f system-type="LoudBox (8xP150)" -f model=kimi_k3
```

---

## 6. Open questions and the next things to do

### 1. The rebase — read this before starting it

This is the messy one. **1/4 was squash-merged**, so the five reference commits on this branch
are *not* ancestors of main even though their content is. A plain `git rebase origin/main` will
try to re-apply them on top of themselves.

Conflict surface (`git diff --name-only $(git merge-base HEAD origin/main) HEAD` ∩ same for main):

* `.github/CODEOWNERS`, `.github/workflows/blackhole-e2e-tests.yaml`,
  `tests/pipeline_reorg/blackhole_e2e_tests.yaml` — shared with 3/4, keep both sides' rows.
* `tests/pipeline_reorg/ops_unit_tests.yaml`, `.github/time_budget.yaml` — shared with 2/4.
* `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp` — both ops register here.
* **All nine 1/4 files** under `reference/kimi_k3/attn_res/` plus `tests/attn_res/assertions.py`
  and `tests/attn_res/test_assertions.py` — **drop this branch's copies, take main's.**

Order matters: land 2/4 (approved) and 3/4 first, then rebase this one onto a main that already
has both ops. That collapses the diff to the 23 model files + the CI rows. Rebuild after,
re-run the 36, then force-push with `--force-with-lease`. **Any PCC movement after a rebase is a
finding, not noise.**

### 2. Pipeline support in `TtAttnResWalk`

Not started. Today the walk is single-rank: `read`/`write`/`open_layer`/`finish` assume one
layer's block is resident. Prefill's pipelined schedule overlaps layers, so `_block_sites`
(`attn_res_stream.py:121`) and `_free_batches` (:217) need to become aware of more than one live
block at a time. This is the larger of the two open items.

### 3. K3 block integration in `tt_prefill_block.py`

Not started. `test_integration_example.py` is deliberately the shape the block should copy —
start from it. The block currently does not call AttnRes at all.

### 4. PR hygiene, pending your go-ahead

Rewrites drafted but **not applied** (do not post without saying so):

* title → *"[Kimi K3] AttnRes 4/4: add the device model, the walk, and its tests"*
* `## What is gated` → `## What the tests cover`
* *"The gates are meant to hold on weights the model never saw"* → *"The assertions don't depend
  on weight values"*

### Open questions, honestly unresolved

* **`fold_stats` and `one_pass_stats` are constructor knobs with no test arm on the slow path.**
  Both default to fast. Nothing would catch a regression that only shows up with them off.
* **`stats_dtype=bf16` clears the PCC gate by ~4e-6 over 186 chained reads.** If the walk ever
  gets longer, that margin goes. No test pins it.
* **`READ_SITES = 24` is a 12-layer block.** The 93-layer walk is only covered by
  `test_forward_loop.py`'s cadence check, not end-to-end at production depth.
* `test_attn_res_gather_softmax.py`'s inline `_oracle()` is a third transcription of the score
  formula outside `reference/`. It has not drifted, but nothing stops it.

### Also outstanding, outside this branch

* Port both padded-vs-logical validation fixes (`94c33f52660`, `6d2b5f1c4ae`) back to omnibus
  #52676 before closing it, and push the local-only docstring commit `3835eacbb40`.
* Post the closing comment on issue #51887.
