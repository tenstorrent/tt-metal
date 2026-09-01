# HANDOFF — AttnRes 3/4: `attn_res_gather_softmax`, the fused read

**PR** [#53126](https://github.com/tenstorrent/tt-metal/pull/53126) ·
**branch** `nmilicevic/bringup/kimi-k3-attnres-2-gather-softmax` ·
**head** `5b287af315d` · **base** `origin/main` at `65a1138e26b`

> **Not for merge.** This file is context for picking the work up on another machine.
> Drop it before the PR lands: `git rm HANDOFF.md && git commit --amend` (it is the tip commit).

---

## 0. The series

Kimi K3 attention residuals (AttnRes) on Blackhole, split out of omnibus PR
[#52676](https://github.com/tenstorrent/tt-metal/pull/52676) into four merge-able PRs.

| PR | what | branch suffix | number | state |
|---|---|---|---|---|
| 1/4 | torch reference | `-0-reference` | #53114 | **MERGED** 2026-08-25 |
| 2/4 | `attn_res_weighted_reduce_nc` | `-1-weighted-reduce` | #53116 | open, **APPROVED, MERGEABLE** |
| **3/4** | **`attn_res_gather_softmax`** | `-2-gather-softmax` | **#53126** | **open, review required, conflicts with main** |
| 4/4 | device model + walk + tests | `-3-model` | #53318 | open, review required, conflicts with main |

**Mapping to the reference.** `attn_res_weighted_reduce_nc` (2/4) is the **inter-block / mix** half:
it emits the unnormalized numerator plus the shift and mass for a sealed set. **This op is the
merge half**: it takes those, scores the live candidate against the sealed ones, and folds
everything into one blended residual.

One-line summary for a group chat: *K3 attention residuals — the read's whole path (per-rank
statistics, the cross-shard gather, and the online-softmax merge) in one dispatch.*

---

## 1. Goal and current state

**Goal.** One read site's whole path from a TP-sharded residual stream to the mixed hidden state,
in **one dispatch**. Three stages inside one program:

```
stats    per-rank sum of squares and dots over this rank's shard of d
gather   completes them across the shard axis (fabric)
fold     the online-softmax combine
```

and the fold is

```
live_scores = sum_p dots_p * rsqrt(sum_p sum_squares_p * inv_hidden_size + eps)
m           = max(shift, live_scores)
h           = (partial * exp(shift - m) + running_sum * exp(live_scores - m))
              / (mass * exp(shift - m) + exp(live_scores - m))
```

API (`attn_res_gather_softmax.hpp`):

```cpp
std::vector<ttnn::Tensor> attn_res_gather_softmax(
    partial, running_sum, shift, mass, q, stats, semaphore,
    cluster_axis, site, inv_hidden_size, eps,
    pending, num_links, topology, subdevice_id, memory_config, compute_kernel_config);
```

**State: code DONE, PR not yet reviewed.**

* Op, 4 kernels, nanobind binding, CMake, 6 unit tests, one CI row, one CODEOWNERS row — all landed.
* `code-analysis / 🤖 Clang Tidy (Full)` **success** at head.
* **CI at head is unproven.** The last green LoudBox e2e was at `85236dcd8fa` (run `32028140482`),
  one commit before the clang-tidy clamp. The run at head (`32038503561`) failed in **`Set up job`**
  — GitHub returned 429/503 downloading `slackapi/slack-github-action`. Pure infra flake; the test
  never executed. **Re-dispatched: run `33481086131`, in flight.**
* **`mergeable: CONFLICTING`** against main. Rebase needed (see §6).
* The clamp itself is numerically inert: for `uint32_t`, `if (a > b) a = b;` and `a = std::min(a, b)`
  are the same clamp. A/B locally anyway: op tests 6 passed 13.91 s (was 14.23 s), model suite
  36 passed 121.90 s (was 125.06 s).

**Production shape this op was built and measured for:** 2×4 Blackhole mesh, `HIDDEN_SIZE = 7168`
global / **1792 per chip**, `PER_CHIP_TOKENS = 640` → `Ht = 20`, `Wt = 56`, `ring_size = 4`,
`kPeers = 3`, FABRIC_2D.

---

## 2. Every file, and why

22 files, +2959, pure addition apart from four shared lines.

### The op — `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/`

| file | why |
|---|---|
| `attn_res_gather_softmax.hpp` | Signature + the contract comment: what `stats` is, what `site` does, what `pending` buys. Reviewers read this first; keep it. |
| `attn_res_gather_softmax.cpp` | Host entry. |
| `attn_res_gather_softmax_nanobind.{hpp,cpp}` | Python binding + docstring. Docstring states `stats` is scratch — that is what licenses the op to own its interior layout (§3). |
| `device/..._device_operation.{hpp,cpp}` | Validation (logical **and** padded at five sites), cache-hit checks for `site` bounds and operand device affinity, 2D-fabric requirement. |
| `device/..._device_operation_types.hpp` | Operand/attribute structs. |
| `device/..._program_factory.{hpp,cpp}` | The core plan: 1 gather core + `Ht` stat cores + `kFoldCoresPerRow * Ht` fold cores, CB sizing, fabric connections, and the per-site runtime-arg patch on a cache hit. Constants at :36-52. |
| `device/kernels/compute/attn_res_gather_softmax.cpp` | 359 lines: the scalar chain and the fold. Device 2.0 init API. |
| `device/kernels/dataflow/reader_attn_res_gather_softmax.cpp` | Streams `partial` / `running_sum` / `q` planes. |
| `device/kernels/dataflow/gather_attn_res_gather_softmax.cpp` | **The unusual one.** One core; opens the fabric connections, sends this rank's packed statistics to every peer and waits on the arrival counter. |
| `device/kernels/dataflow/writer_attn_res_gather_softmax.cpp` | Packs the local statistics into the wire layout, publishes them, un-packs peers' on arrival, writes the output. |
| `device/kernels/dataflow/attn_res_stats_layout.hpp` | The wire/interior layout contract, **shared by both dataflow kernels so they cannot disagree**. Installed by CMake (`85236dcd8fa`) — without that a `.cpp`-only package cannot JIT the kernels. |
| `CMakeLists.txt`, `sources.cmake` | Per-op registration. |

### Shared files

| file | why |
|---|---|
| `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp` | +2 lines. **2/4 touches the same two lines**; whichever merges second keeps both. |
| `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/CMakeLists.txt` | +3 lines. |
| `.github/CODEOWNERS` | +1 line (464) routing the op's test files to ds-prefill — §3. |
| `.github/workflows/blackhole-e2e-tests.yaml` | +1 `kimi_k3` dispatch option, so the row can be selected. |
| `tests/pipeline_reorg/blackhole_e2e_tests.yaml` | +1 CI row. |

### Test

`tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py` — 6 tests, all on a
2×4 mesh with FABRIC_2D. Same un-globbed directory as 2/4's test, for the same reason (a
LoudBox-only test must not be dragged onto a SKU it has never run on inside someone else's `-x` suite).

The 6:

| test | what it gates |
|---|---|
| `test_matches_torch[...-plain-fabric2d-mesh-2x4]` | The fold against a torch oracle, no `pending`. |
| `test_matches_torch[...-settle-fabric2d-mesh-2x4]` | Same with `pending` — the second output must equal `running_sum + pending`. |
| `test_every_site_reads_its_own_plane_from_one_cached_program` | R sites walked through **one** cached program; each must read its own plane. This is what catches a per-site runtime-arg patch that stops taking effect. |
| `test_rejects_a_site_past_the_batch_on_a_cache_hit` | The bounds check must survive the cache hit, not just the first dispatch. |
| `test_rejects_shapes_that_only_agree_once_padded[rows_same_tile_bucket]` | Logical-vs-padded validation, row count. |
| `test_rejects_shapes_that_only_agree_once_padded[unaligned_width]` | Logical-vs-padded validation, tile alignment. |

### The CI row (`blackhole_e2e_tests.yaml:78`, inline comments elided)

```yaml
- id: bh-lb-ttnn-attn-res-gather-softmax-unit
  name: ttnn attn_res_gather_softmax unit tests
  model-name: kimi_k3
  cmd: pytest tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py -k "mesh-2x4"
  skus:
    bh_loudbox:
      timeout: 10
  owner_id: U09LK84G4TF # Nikola Milicevic
  team: ttnn
```

Measured: 6 passed in 22.35 s inside CI, 35 s step (~17× headroom on the 10-minute budget).
`model-name: kimi_k3` is **required**: the workflow's `model` input is a strict equality filter, so
an untagged row is dropped by a `model=kimi_k3` dispatch (the docs claim otherwise; they are wrong).

---

## 3. Design decisions, and why they are not obvious

### The dataflow is not the usual reader/compute/writer

One program, four kernels, one `EnqueueProgram`, one Tracy row. (Kernel count is unrelated to
dispatch count — worth stating because "fused" reads as "one kernel".)

* **Stat cores**, one per token row (`Ht` of them). Each owns one row-tile of `v`, computes this
  rank's `sum_squares` and `dots`, and writes them into the statistics scratch.
* **The gather core**, exactly one. It owns the fabric: it publishes this rank's packed statistics
  to every peer's slot and waits until all `kPeers` have arrived.
* **Fold cores**, `kFoldCoresPerRow = 2` per token row. They complete the cross-shard sum, run the
  scalar chain (max/exp/rsqrt/divide) and multiply the result into the two stream planes.

At the production shape that is `1 + 20 + 40 = 41` of 110 cores. **That is not the problem** —
see §4, three independent sweeps say more cores is worse.

### The statistics are packed by column, and that is the single biggest perf decision

`20493658d88`. A tile of statistics carries **one logical column**: 32 useful values inside a
32×32 tile. The fabric charges **~173 ns per packet, essentially payload-independent** (measured:
packet count fixed, payload varied 80×, cost moved by noise). So the plane was 20 packets carrying
128 B of information each.

Packed dense, a plane is `Ht * 128 B` = one packet. **120 packets → 6.**

Measured, one interleaved Tracy run, program cache cleared between arms, 3 rounds × 20 warm reps:

| arm | packets | median | delta |
|---|---|---|---|
| tile-shaped | 120 | 90.94 µs | — |
| packed | 6 | **66.91 µs** | **−24.03 µs, −26.4 %** |

Output **bit-identical** (`max|Δ| = 0.000e+00`). Re-profiled after the A/B scaffolding was removed:
**66.80 µs** (66.45 / 66.86 / 66.80 across three arms).

It beat the pure packet ceiling (22.8 %) because packing cuts the DRAM side ~32× too: the gather
core reads 8 kB instead of 164 kB; each fold core reads 8×128 B instead of 8×4096 B; each stat core
writes 2×128 B instead of 2×4096 B.

**Why the op is allowed to do this:** `stats` is caller-supplied scratch — the nanobind docstring
already says it is not read on entry and holds nothing meaningful on exit — so the op owns its
interior. **This is not an API change.** A packed token row is one value per token of a row-tile
(`32 * elem_bytes`) and a page holds 32 such rows, so a plane never needs more pages than the
tile-shaped layout it replaces and the caller's tensor is always big enough. The contract lives in
`attn_res_stats_layout.hpp`, included by both dataflow kernels so they cannot drift apart.

**Pack and un-pack sit on the workers**, one core per token row — 20-way parallel, and the tiles are
already in L1 there. Both alternatives were priced and rejected: packing on the gather core is 1280
strided loads serialized on one core (~10-15 µs on the critical path); un-packing on arrival is
491 kB of DRAM writes from one core (~25 µs).

**Garbage lanes are harmless, and this was checked rather than assumed.** Un-packing writes only
column zero, leaving the rest of each scalar tile as stale L1. The whole scalar chain is lane-local
(`copy_tile`, `add_binary_tile`, `mul_unary_tile`, `add_unary_tile`, `rsqrt_tile`, `mul_binary_tile`,
`binary_max_tile`, `sub_binary_tile`, `exp_tile`) and only `mul_tiles_bcast_cols` reads column zero.
Zeroing the tiles would have cost 32 kB of stores per fold core and eaten most of the win. The
bit-exact A/B is the proof it is unnecessary.

### The arrival semaphore, and why one site stayed on the pre-2.0 API

The collective needs a counter every peer can increment and this rank can wait on. That is a
**global semaphore** — an allocator buffer at a mesh-wide address, passed in as the `semaphore`
operand — not a per-program `CreateSemaphore` id.

`ea86b5fd7ed` changed the consume from "wait then set 0" to **wait then subtract `kPeers`**
(`noc_semaphore_inc(addr, 0u - kPeers)`). Setting it to zero races: a fast peer's increment for the
*next* site can land between the wait and the store and be erased. Subtraction cannot lose one.

That is also why this one site is still raw NOC calls after the Device 2.0 migration: `Semaphore<>`
is constructed from a compile-time semaphore **id**, and this counter is named by a runtime address
it cannot take — and the decrement has to be atomic. The reasoning is a code comment, not just a
commit message.

### Device 2.0 migration — two commits, deliberately scoped

`binary_op_init_common`, `init_bcast` and `mul_bcast_rows_init_short` are deprecated for removal
after 2026-09-15 (#49924).

* `1b4eb7245f6` — compute init: `binary_op_init_common(a,b,c)` → `compute_kernel_hw_startup(a,b,c)`
  at the top of `kernel_main`; `init_bcast<ELWMUL, COL>(a,b,c)` → `bcast_init<...>(a,b)` (the
  output-CB third arg is gone); `mul_bcast_rows_init_short` → `mul_bcast_rows_init`.
  Output bit-identical, device time 66.46 µs against a 66.45–66.86 µs band.
* `be64cebb7e8` — dataflow objects: `cb_reserve_back`/`get_write_ptr`/`cb_push_back(id, …)` →
  `CircularBuffer obj(id)` with `.reserve_back()/.get_write_ptr()/.push_back()`;
  `get_semaphore(id)` + raw `volatile tt_l1_ptr uint32_t*` + `noc_semaphore_wait/set/inc/inc_multicast`
  → `Semaphore<> sem(id)` with `.wait()/.set()/.up(noc,x,y,1)/.inc_multicast(noc,…)`;
  `noc_async_atomic_barrier()` → `noc.async_atomic_barrier()`.

2/4's kernels needed no migration — they were 2.0-native from the start.

### `site`, and the cached program

`partial`, `shift` and `mass` may carry R read sites on dim 0; `site` picks the plane. **`site`
shapes no kernel**, so walking R sites reuses one cached program and only patches runtime args.
Two things that follow, both with their own test:

* `2d2881245f0` — the per-site patch must be *gated*: applying it unconditionally on a cache hit
  wrote args for a program that had not been built for this operand set.
* `dfa68e66b03` — operand **device affinity** must be re-checked on a cache hit, not only on the
  first dispatch.

### FABRIC_2D only

`3d338cf4f6d`. The 1D branch was dropped on request: everything else in this model runs FABRIC_2D,
and a second transport path that nothing exercises is a liability. The op now requires a 2D fabric
and routes by node. `topology` is one entry **per mesh axis**, not a scalar — Galaxy prefill is
`[LINE, RING]`, and applying a scalar `Ring` to a linear axis points a collective at a wrap link
with no physical fabric edge.

### The same padded-vs-logical defect as 2/4, plus a worse variant (`6d2b5f1c4ae`)

Five sites compared `padded_shape()` only (running_sum vs partial, pending vs running_sum,
shift/mass dims 1-2, stats dim 2, q's last dim) while `compute_output_specs` builds the output from
`partial.logical_shape()`. Two row counts in one tile bucket compared equal and the shorter
operand's padding came back as data.

The variant 2/4 does not have: `partial_shape[-1] % TILE_WIDTH == 0` on the **padded** shape is true
by construction, so the tile-alignment check enforced nothing. It matters more here — `Wt` comes
from `partial.padded_shape()` and the statistics reduce runs the whole padded row, so a logically
narrower `partial` folds its own padding into `sum_squares` and `dots`, setting the live score for
**every** row.

### CODEOWNERS line 464

```
tests/ttnn/unit_tests/operations/experimental/test_attn_res_*.py @tenstorrent/metalium-developers-ds-prefill @tenstorrent/codeowner-bypass
```

Executes nothing — it is review routing. CODEOWNERS is last-match-wins and no line after 464 matches
this pattern, so without it the match falls back to line 424 (`tests/ttnn/**/operations/` →
`metalium-developers-ops-leads`) and the op's own tests would be reviewed by a team that does not own
the op. Op sources are already routed the same way at lines 349-350. Rows 460/461/463 use the
identical per-file pattern for the sdpa team.

### `readability-use-std-min-max` is warnings-as-errors

`5b287af315d`. The reviewdog inline comment looked advisory; `code-analysis / 🤖 Clang Tidy (Full)`
was actually **failing**. The check flags `if (a > b) { a = b; }` but not `if (a > b) { a = c; }`,
which is why only the second of two adjacent clamps tripped it.

---

## 4. Dead ends — measured, and not worth repeating

**The mechanism everyone reaches for first is wrong.** The exchange is **per-packet overhead, not
link bandwidth**. A dedicated packet-size sweep held packet count fixed and varied payload 80×; cost
moved by noise. Baseline 120 packets = 90.6 µs, 3 packets = 69.9 µs → ~173 ns/packet. The earlier
"the gather core is link-throughput saturated at 20.9 GB/s" figure is an artifact of dividing padded
bytes by a time that padding did not cause. Anything derived from packet count stands; anything
derived from "link throughput" does not.

| tried | result |
|---|---|
| Faces-only sends (half the bytes, twice the packets) | **22.6 % slower.** Direct confirmation of the per-packet law. |
| Dropping the per-packet blocking flush | 0.9 %. Noise. |
| **A second gather core on the second fabric link** | Worth ~12 µs *before* packing, worth ~nothing after (the whole exchange is now ~1 µs). Both links exist (`get_forwarding_link_indices` returns `[0,1]` for every hop) and the factory uses `link_idx=0` for both directions. **Do not build it.** |
| Merging the two planes into one page | ~3 packets ≈ 0.5 µs. Below the stated 1-3 % rejection bar. |
| "Coalesce the 20 row-tiles per peer into one transfer" | Impossible as stated: `get_tt_fabric_max_payload_size_bytes()` = 4352 B, and a 4096 B stat tile is already 94 % of a max packet. The lever was **density**, not coalescing. |
| Widening the fold across all ~110 cores | 41 cores 98.6 µs, 61 → 113.4, 81 → **111.4**, 101 → 119.5. Wider is *worse*; the fold is not core-starved. `kFoldCoresPerRow = 2` confirmed optimal. The curve sawtooths on whether `v` divides `Wt = 56`. |
| Grid widening on the older `attn_res_stats` op | Throughput peaks at 55 cores and is 2.6× worse at 110, because the q CB on `all_cores` makes DRAM traffic `cores × 114.7 kB`. |
| Packing on the gather core | ~10-15 µs of serialized strided loads on the critical path. |
| Un-packing on arrival on the gather core | ~25 µs of DRAM writes from one core. |
| Zeroing the un-packed scalar tiles | Unnecessary — the chain is lane-local; bit-exact A/B proves it. Would have cost 32 kB of stores per fold core. |
| Adding an `ops_unit_tests.yaml` row for this op | Needs a **new allocation on a scarce 8-chip runner**: `ttnn.unit` has no `bh_loudbox` key in `.github/time_budget.yaml` at all, and `verify_time_budget.py` hard-fails on a missing `budgets[team][workflow][sku]`. That is why the op reaches CI through the e2e workflow instead. `bh_quietbox_2` is a **4-chip** box and cannot host a 2×4 mesh. |

**Cost model at the production point** (`t = F + c·Wt + k·Ht + d·Ht·Wt`, from two width sweeps at
two heights): `F = 27.74 µs` fixed (launch, fabric open/close, two semaphore barriers, cross-chip
skew), `c = 0.3353 µs/Wt`, `k = 1.3556 µs/Ht`, `d = 0.01583 µs/(Ht·Wt)`. Predicts 64.2 µs against
66.8 measured. **What is left is the 27.7 µs fixed term (now ~41 % of the op) and a 36.5 µs
compute + stream block.** Treat compute and stream DRAM as one block — `d` implies 646.9 GB/s,
126 % of peak, so the `c`/`d` boundary is soft.

---

## 5. Build, test, repro

**Hardware:** Blackhole **LoudBox (8× P150)**, 2×4 mesh, FABRIC_2D. There is no single-device arm and
there should not be — `tp_factor == 1` makes the gather an identity.

**Landmines:**

* Bare `python` is `/opt/venv` and imports `ttnn` as a **hollow namespace package with no error**.
  Always `./python_env/bin/python`.
* After any branch switch: `git submodule update --init --recursive tt_metal/third_party/umd`.
* Build **only** through `bash build_metal.sh`.
* A rebase invalidates the local `_ttnn.so`; the symptom is `import ttnn` raising `AttributeError`
  on a `_ttnn` symbol from inside `conftest.py`. That is a stale build, not a test failure.
* Do **not** wipe `~/.cache/tt-metal-cache` (18 GB, shared); use `TT_METAL_CACHE=<scratch>` instead.
* `gh` is no longer on `PATH` here; `/proj_sw/user_dev/moconnor/bin/gh` works with your own
  `~/.config/gh/hosts.yml`. `gh pr edit` fails on this repo — use
  `gh api -X PATCH repos/tenstorrent/tt-metal/pulls/<N> -F body=@<file>`.
* `git rebase -i` is not supported in this environment.

```bash
cd /localdev/nmilicevic/tt-metal
git checkout nmilicevic/bringup/kimi-k3-attnres-2-gather-softmax
git submodule update --init --recursive tt_metal/third_party/umd
bash build_metal.sh

# the exact CI command
./python_env/bin/python -m pytest \
  tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py -k "mesh-2x4"
# expect: 6 passed, ~14 s warm

# CI budget validation — the third arg is a workflow KEY. For blackhole e2e it is
# literally `e2e` (.github/workflows/blackhole-e2e-tests-impl.yaml:50-53), NOT a filename.
./python_env/bin/python .github/scripts/utils/verify_time_budget.py \
  tests/pipeline_reorg/blackhole_e2e_tests.yaml .github/time_budget.yaml e2e

# dispatch CI
gh workflow run "(Blackhole) e2e tests" \
  --ref nmilicevic/bringup/kimi-k3-attnres-2-gather-softmax \
  -f system-type="LoudBox (8xP150)" -f model=kimi_k3

# per-step timings and the pytest summary line from a finished run
gh api repos/tenstorrent/tt-metal/actions/runs/<id>/jobs --paginate \
  -q '.jobs[] | "\(.conclusion)\t\(.name)"'
gh api repos/tenstorrent/tt-metal/actions/jobs/<jobid>/logs | grep -a passed
```

**Profiling** (how the 66.8 µs number was produced): Tracy build, isolated vehicle so only this op
dispatches, 3 rounds × 20 warm reps, per-rep **slowest-device device-FW**, median across reps.
Clear the program cache between arms or the factory will not re-read a compile-time mode.

---

## 6. Open questions and the next things to do

1. **Watch run `33481086131`** (LoudBox e2e at head). The previous attempt died in `Set up job` on a
   GitHub 429, not on anything in this branch. If it is green, head is fully proven.
2. **Rebase onto main.** The PR is `CONFLICTING`. 1/4 was **squash-merged**, so its original commits
   are not ancestors of main; expect the conflict surface to be `.github/CODEOWNERS`,
   `.github/workflows/blackhole-e2e-tests.yaml`, `tests/pipeline_reorg/blackhole_e2e_tests.yaml` and
   `experimental_nanobind.cpp`. Rebase, **rebuild**, re-run the 6 tests, then force-push with
   `--force-with-lease`. Any PCC or timing movement after a rebase is a **finding**, not noise.
3. **Get it reviewed.** It is out of draft and CODEOWNERS routes it to ds-prefill. All Copilot rounds
   are addressed and resolved.

**Do not re-add** the paragraph beginning *"The row names its SKU rather than letting a shared
command skip itself on a narrower box…"* to the PR body. It was deleted on purpose as uninformative.

**Open questions, honestly unresolved:**

* **The 27.7 µs fixed term is now 41 % of the op and nothing has been measured about what dominates
  it.** Launch, fabric open/close, the two semaphore barriers and cross-chip skew are the candidates.
  This is the only remaining lever worth real effort; everything else on the list is dead.
* `test_attn_res_gather_softmax.py:48`'s inline `_oracle()` is a **third** transcription of the score
  formula outside `reference/`. It has not drifted, but nothing stops it.
* A `4x2` LoudBox arm would restore `ring_size == 2` coverage, which no test currently has.
