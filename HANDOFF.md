# HANDOFF — AttnRes 2/4: `attn_res_weighted_reduce_nc`

**PR** [#53116](https://github.com/tenstorrent/tt-metal/pull/53116) ·
**branch** `nmilicevic/bringup/kimi-k3-attnres-1-weighted-reduce` ·
**head** `1a59bc2c112` · **base** `origin/main` at `240a20cc367`

> **Not for merge.** This file is context for picking the work up on another machine.
> Drop it before the PR lands: `git rm HANDOFF.md && git commit --amend` (it is the tip commit).

---

## 0. The series

Kimi K3 attention residuals (AttnRes) on Blackhole, split out of omnibus PR
[#52676](https://github.com/tenstorrent/tt-metal/pull/52676) into four merge-able PRs.

| PR | what | branch suffix | number | state |
|---|---|---|---|---|
| 1/4 | torch reference | `-0-reference` | #53114 | **MERGED** 2026-08-25 |
| **2/4** | **`attn_res_weighted_reduce_nc`** | `-1-weighted-reduce` | **#53116** | **open, APPROVED, MERGEABLE** |
| 3/4 | `attn_res_gather_softmax` | `-2-gather-softmax` | #53126 | open, review required, conflicts with main |
| 4/4 | device model + walk + tests | `-3-model` | #53318 | open, review required, conflicts with main |

Merge order: reference → ops → model. 2/4 and 3/4 are independent of each other.
Each branch carries its own `HANDOFF.md`; 4/4's is the superset.

**Mapping to the reference.** The torch reference has two halves and each became one op:
`attn_res_weighted_reduce_nc` (this PR) is the **inter-block / mix** half — it produces the
unnormalized numerator plus the shift and mass for a whole sealed set. `attn_res_gather_softmax`
(3/4) is the **merge** half — it takes those, scores the live candidate, and folds everything into
one blended residual.

---

## 1. Goal and current state

**Goal.** One dispatch that reduces a candidate axis with a per-row weight, for **R** weight sets
at once:

```
out[r][0][h][w] = sum_c input[0][c][h][w] * weight[r][c][h][0]
```

`input` is `[1, C, H, W]`, `weight` is `[R, C, H, 1]`, output is `[R, 1, H, W]`.
The weight broadcasts along the last dim, which `BroadcastType::COL` does natively — no transpose,
no padding beyond what the tile layout already imposes. The product is MAC'd into the accumulator,
so the input tensor is read **once** instead of three times (mul → materialize → sum).

The `R` batching is the point: a caller walking R read sites gets them in one dispatch and pays
one input read per group of sets, not one per set. That is the difference between the op being
bound by its own arithmetic and being bound by re-reading a tensor it already has.

**State: DONE.** Nothing in flight, nothing broken.

* Op, kernels, nanobind binding, CMake registration, 31 unit tests, one CI row — all landed.
* PR is **approved and mergeable**. It is waiting only on someone pressing merge.
* Kernels were written Device-2.0-native from the start (`TensorAccessorArgs`/`TensorAccessor`,
  `CircularBuffer` objects, `Noc` object, `compute_kernel_hw_startup`), so unlike 3/4 there is no
  2.0 migration commit here.
* Nightly L2 green at `5286eb7d851` (run `31849875000`). `1a59bc2c112` is a
  timeouts-only yaml change on top; PR Gate + static checks green at head.

**Constraints, stated so nobody widens them by accident:** Blackhole only, bf16 input,
`dim == 1` only, rank-4 interleaved TILE operands.

---

## 2. Every file, and why

18 files, +1298, pure addition apart from three shared lines.

### The op — `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/`

| file | why |
|---|---|
| `attn_res_weighted_reduce_nc.hpp` | Public signature + the long comment that pre-empts the "why not `fast_reduce_nc` with a weight" review question. **Do not let review trim that comment** — see §3. |
| `attn_res_weighted_reduce_nc.cpp` | Host entry; normalizes a negative `dim` before the device op sees it. |
| `attn_res_weighted_reduce_nc_nanobind.{hpp,cpp}` | Python binding + docstring. `bind_function` sets the Python name, so the C++ namespace move did not rename anything user-visible. |
| `device/..._device_operation.{hpp,cpp}` | Validation and output specs. All shape checks compare **logical and padded** — see §3, the real bug. |
| `device/..._device_operation_types.hpp` | Operand/attribute structs. |
| `device/..._program_factory.{hpp,cpp}` | Work split over **input positions** (not output tiles), and `sites_per_group` — the site axis grouped by DEST capacity. |
| `device/kernels/reader_weighted_reduce_nc.cpp` | Streams input + weight tiles. `TensorAccessorArgs<8>()`, `TensorAccessor`, `CircularBuffer`, `Noc` — already 2.0. |
| `device/kernels/weighted_reduce_nc.cpp` | The MAC loop: `bcast_init<ELWMUL, COL>` + `mul_tiles_bcast_cols` under a hand-overridden MATH `acc_to_dest`. `compute_kernel_hw_startup` at the top. |
| `device/kernels/writer_weighted_reduce_nc.cpp` | Writes the R output planes. `TensorAccessorArgs<4>()` + `CircularBuffer`. |
| `CMakeLists.txt`, `sources.cmake` | Per-op registration, the current convention (not the root list). |

### Shared files (the only pre-existing ones touched)

| file | why |
|---|---|
| `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp` | +2 lines (include + call). **3/4 touches the same two lines** — whichever merges second conflicts trivially; resolve by keeping both. |
| `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/CMakeLists.txt` | +3 lines, adds the op subdirectory. |
| `tests/pipeline_reorg/ops_unit_tests.yaml` | +1 CI row (below). |

### Test

| file | why |
|---|---|
| `tests/ttnn/unit_tests/operations/experimental/test_attn_res_weighted_reduce_nc.py` | 31 tests. **The directory is load-bearing** — see §3. |

### The CI row (`ops_unit_tests.yaml:104`, inline comments elided)

```yaml
- name: ttnn attn_res_weighted_reduce_nc unit tests
  cmd: 'pytest tests/ttnn/unit_tests/operations/experimental/test_attn_res_weighted_reduce_nc.py -xv --timeout=60'
  skus:
    bh_p150b_civ2:
      timeout: 3
  owner_id: U09LK84G4TF # Nikola Milicevic
  team: ttnn
  category: experimental
```

* SKU `bh_p150b_civ2` — a single Blackhole P150b on a CI v2 runner. One card is enough.
* `timeout: 3` is the **step** budget; `--timeout=60` is per-test and fires *inside* the step, so a
  hang produces a pytest stack dump with 2 minutes left to write the report. The old `600` could
  never fire before the step was killed.
* Sized from a **cold-cache** run (`TT_METAL_CACHE` at an empty dir): 38 s pytest / 51 s wall,
  slowest single test 9 s (8 s of it device open). 60 s is 6.7× the slowest test.
* `ttnn.unit.bh_p150b_civ2` budget is 1038 of 1045 min after this row. **Do not inflate it.**
* The row runs in `tt-metal-l2-nightly.yaml` ("Nightly tt-metal L2 tests", cron 06:00 UTC), whose
  schedule leg includes category `experimental` and SKU `bh_p150b_civ2`. It does **not** run in
  `pipeline-select.yaml` (wh_n300_civ2 + cpp categories only).

**No CODEOWNERS row.** Deliberate — see §3.

---

## 3. Design decisions, and why they are not obvious

### The name, and the question it exists to pre-empt

Called `fast_weighted_reduce_nc` until 2026-08-14. That name promised "`fast_reduce_nc` plus a
weight operand", which is false in two structural ways:

* `fast_reduce_nc` sums into **one** output plane and splits work over **output tiles** — each
  output tile owns a private set of input tiles. This op produces **R** planes from a single pass
  and splits over **input positions**, because every plane reads the same input and reading it once
  is the entire point. No shared work-split, no shared output-shape contract.
* `fast_reduce_nc`'s second CB operand is a **zero tile** (`prepare_zero_tile<cb_id_in1>()` in its
  reader); it sums via `add_tiles` against zero with `acc_to_dest`. There is no free operand to
  fill — the weighted op replaces the whole compute body.

Blast radius rules out merging them anyway: `fast_reduce_nc` backs `ttnn.sum`
(`reduction/generic/generic_reductions.cpp`), `argmax`, `quasar/reduction/generic` and ~30 model
files.

The `attn_res_` prefix follows the folder's own convention — the argument to make in review is that
`experimental/reduction/` already holds `deepseek_grouped_gate`, `deepseek_moe_fast_reduce_nc` and
`deepseek_moe_fast_reduce_nc_fused`, and the middle one has a **fully generic** signature and still
carries a model prefix. `batched_` was considered and rejected: zero ttnn ops use it, so it would
invent a convention rather than follow one.

Both the C++ header and the nanobind docstring say this. Keep them.

### Where the op lives, and why there is no CODEOWNERS row

Under `experimental/deepseek_prefill/`, one directory per op, because the prefill team builds and
maintains it (mbezuljTT's ask in review: prefill-team ops stay under `experimental/deepseek_prefill`).
The device op stays in `ttnn::experimental::prim`; host and nanobind take
`ttnn::operations::experimental::deepseek_prefill::<op>`.

**CODEOWNERS is last-match-wins, so a narrower row REPLACES the owners it inherits rather than
adding to them.** The `deepseek_prefill/` rows already name `metalium-developers-ds-prefill` (plus
`metalium-developers-infra` on `**/CMakeLists.txt`). Adding a per-op row would *drop* the infra team.
The old row under `experimental/reduction/` was deleted by the move.

### Where the test file lives — this is not incidental

`tests/ttnn/unit_tests/operations/experimental/` is globbed by **no** group suite; it is reached
only by explicit `::test` rows. The obvious home, `operations/reduce/`, is collected wholesale by
the `ttnn reduce group` (`ttnn_sanity_tests.yaml:249`) on `wh_n300_civ2, bh_p100, bh_p150b_civ2,
sim_wormhole_b0, sim_blackhole` under `-x`. **P100 is Blackhole**, so `is_blackhole()` does not skip
there — a P150b-only op would run on hardware it has never seen, inside another team's `-x` suite,
where one failure aborts the rest. `nightly/operations/reduction/` has the same `bh_p100` hazard.

Do not confuse `tests/ttnn/unit_tests/operations/experimental/` (ours, un-globbed) with
`tests/ttnn/nightly/unit_tests/operations/experimental/` (globbed by three rows in
`ops_unit_tests.yaml`). Different trees. Consequence: no `ttsim-skip-list.yaml` entry is needed
either, because no simulator suite collects it.

### The real bug this PR fixed (`94c33f52660`)

Validation compared `padded_shape()` only. **Dim 2 is tile-padded**, so a 120-row input against a
100-row weight both padded to 128 and passed — and since the output's logical shape comes from
`input.logical_shape()`, 20 output rows were built from the **weight's padding** and returned as
data. Padding is not guaranteed zero, so that is arbitrary values, not zeros.

Now requires logical equality too. Negative arm: `test_..._rejects[bad=rows_same_tile_bucket]`,
which necessarily fails without the fix.

**3/4 had the same defect** in five places plus a worse variant, fixed there as `6d2b5f1c4ae`.
**Both fixes still need porting back to omnibus #52676** before it is closed, or they are lost with it.

### `sites_per_group` is not "granularity"

`sites_per_group` groups the **site** axis by DEST register capacity. The candidate loop always
runs the whole axis one tile at a time — this op has **no candidate-side granularity**
(`grep -rn granularity` in the op dir returns nothing). That concept belongs to the neighbouring
*fused* op (3/4), which has `input_granularity`. An earlier round of test IDs claimed otherwise and
was wrong. Do not reintroduce it.

---

## 4. Dead ends — do not redo these

| tried | outcome |
|---|---|
| Making this a flag on `fast_reduce_nc` | Rejected: no shared work-split, no shared output contract, and ~30 model call sites of blast radius. |
| Naming it `batched_weighted_reduce_nc` | Rejected: zero ttnn ops use `batched_`; inventing a convention loses the review argument that the prefix follows the folder. |
| Test under `operations/reduce/` or `nightly/operations/reduction/` | Rejected: both are globbed onto `bh_p100` inside another team's `-x` suite. |
| `--timeout=600` in the CI row | Useless — larger than the 3-minute step budget, so it could never fire. Now 60. |
| Validating on `padded_shape()` alone | The bug above. Logical **and** padded, at every site. |

---

## 5. Build, test, repro

**Hardware:** one Blackhole card (the `device` fixture). No mesh, no fabric.

**Landmines, both real, both cost time before:**

* Bare `python` is `/opt/venv` and imports `ttnn` as a **hollow namespace package with no error**.
  Always `./python_env/bin/python`.
* After any branch switch: `git submodule update --init --recursive tt_metal/third_party/umd`.
* Build **only** through `bash build_metal.sh`. Never cmake/ninja/make directly.
* A rebase onto a newer main invalidates the local `_ttnn.so`. The symptom is `conftest.py` failing
  at `import ttnn` with an `AttributeError` on a `_ttnn` symbol — that reads like a test failure and
  is not one. Rebuild.
* Do **not** wipe `~/.cache/tt-metal-cache` (18 GB, shared). For a cold-cache run point
  `TT_METAL_CACHE` at a scratch directory instead.

```bash
cd /localdev/nmilicevic/tt-metal
git checkout nmilicevic/bringup/kimi-k3-attnres-1-weighted-reduce
git submodule update --init --recursive tt_metal/third_party/umd
bash build_metal.sh

# the exact CI command
./python_env/bin/python -m pytest \
  tests/ttnn/unit_tests/operations/experimental/test_attn_res_weighted_reduce_nc.py \
  -xv --timeout=60
# expect: 31 passed, ~20 s warm cache / ~38 s cold

# blast-radius check on the shared experimental_nanobind.cpp / sources.cmake edits
./python_env/bin/python -m pytest tests/ttnn/unit_tests/operations/reduce/test_fast_reduce_nc.py
# expect: 63 passed

# cold-cache timing, the way the CI number was measured
TT_METAL_CACHE=$(mktemp -d) ./python_env/bin/python -m pytest \
  tests/ttnn/unit_tests/operations/experimental/test_attn_res_weighted_reduce_nc.py -xv --timeout=60

# CI budget validation (third arg is a workflow KEY, not a filename)
./python_env/bin/python .github/scripts/utils/verify_time_budget.py \
  tests/pipeline_reorg/ops_unit_tests.yaml .github/time_budget.yaml unit
```

**Tooling note:** `gh` is not on `PATH` on this box any more; a working binary is at
`/proj_sw/user_dev/moconnor/bin/gh` and picks up your own `~/.config/gh/hosts.yml`.
`gh pr edit` fails on this repo — update a body with
`gh api -X PATCH repos/tenstorrent/tt-metal/pulls/<N> -F body=@<file>`.

**What the 31 tests cover:** production shapes (`[1,9,256,1792]`, `[1,9,2560,1792]`), candidate
counts 1/5/8/12/13, degenerate `Wt=1` and `Ht=1`, the R batch at `num_sites` 1/3/5/8/24 plus an
identity check against per-site calls, negative `dim`, unaligned rows, fp32 weight, program cache,
a composed-`mul`+`sum` equivalence, and 11 rejection arms (dim, weight width, leading dims,
`rows_same_tile_bucket`, batched input, rank, both dtypes, host weight, row-major, sharded).

---

## 6. Open questions and the next things to do

1. **Merge it.** Approved and mergeable; the only reason it is still open is that nobody pressed the
   button. Whoever merges second between this and 3/4 resolves `experimental_nanobind.cpp` by
   keeping both lines.
2. **Then rebase 3/4 and 4/4** onto main and force-push with `--force-with-lease`. Expect the
   nanobind conflict only.
3. **Port `94c33f52660` (and 3/4's `6d2b5f1c4ae`) back to omnibus #52676** before that PR is closed,
   or both validation fixes die with it.

**Open question, unresolved:** whether the `sites_per_group` DEST grouping is worth a tuning knob at
all — it has never been swept. Nothing depends on the answer; the op is not the bottleneck in the
read (3/4's fused op is), so this is only worth doing if a profile points here.
