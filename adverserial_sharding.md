# Adversarial Sharding Checklist (op-agnostic)

A working reference for an **adversarial tester** hunting correctness bugs (wrong
numbers + hangs) in the *sharded* paths of any TTNN op. It is deliberately op-agnostic:
substitute your op's name, its reduced/parallel axes, and its reference formula.

**Why this exists.** Shard *geometry* is not a declared registry axis, so the golden
grader structurally cannot enumerate it. `auto_shard_config` emits only *canonical*
splits (full device, based at (0,0), ROW_MAJOR sharding orientation). `ttnn.from_torch(memory_config=…)`
silently accepts illegal shards. And PCC is blind to some common sharded-reduction
failures like a uniform scale error. So an entire class of bugs lives in a space that is
**green on the grader and latent in the field**. Your job is that space.

Scope: single-device sharding correctness. Not OOM (the allocator reports that — out of
scope), not multi-mesh, not perf.

---

## 0. The one-paragraph method

Read the op's `SUPPORTED[memory_layout]` and its **program descriptor creation for sharded tensors*
to find every place geometry is *assumed* rather than *read* (counts from
`bounding_box`, coords from `(0,0)`, uniform `per_core`, sub-tile-only masks). For each
assumption, construct a **custom** shard spec that violates it (NOT `auto_shard_config`).
Run with **scale-sensitive metrics** (rel-RMS + norm ratio, all_close checks with rtol and
atol similar to those that the canonical correct paths of the op do satify, never PCC alone),
**deterministic inputs** (all-ones), and **L1 poisoning** to make padding bugs reproducible.
Run suspected hangs under `--dev` and read the triage callstack. Report each confirmed bug
with the descriptor/kernel `file:line` and a fix direction; mark bugs `xfail`/`skip` so the
suite is green but the defect is documented.

---

## 1. How to read the op — where the weak spots hide

Read three things, in this order, extracting the listed signals.

### 1a. `SUPPORTED` + `EXCLUSIONS` + `validate()` (the op's `*.py`)
- **Which sharded layouts are claimed?** `SUPPORTED["memory_layout"]` — HEIGHT / WIDTH /
  BLOCK. Each is a different code path; WIDTH/BLOCK usually imply *cross-core* work.
- **Is shard geometry validated at all?** `validate()` almost never checks the shard
  *shape/grid* — only that the layout enum is supported. That gap is your opening: an
  unsupported geometry sails past `validate()` straight into the kernel → hang or garbage
  instead of a clean `ValueError`. **A norm/reduce op that can silently miscompute or hang
  is worse than one that rejects — flag any geometry that isn't validated *and* isn't
  handled.**
- **What does the output memory_config do?** If the op inherits the input's `shard_spec`
  but *changes shape* (a reduction → `(N,1)`), the inherited spec may be illegal.

### 1b. The program descriptor (host-side geometry) — the richest source
Grep the sharded builder(s) for these and interrogate each:

```
grep -nE "bounding_box|grid_size\(\)|shard_spec|shard.shape|\.grid\b|// *per_|per_h|per_w|K =|num_cores|split_work_to_cores|virt\(|CoreRange|orientation|start\.x|start\.y" <descriptor>.py
```

Red-flag questions:
- **Count from width, not cores?** `K = bb.grid_size().x` (or `.y`) assumes the reduction
  group is exactly one grid row/column. Breaks when #shards > that extent (multi-row) →
  **under-counts the reduction** (scale error) or, on a partial row, **hangs**.
- **Absolute vs relative coords?** `virt(0, y)`, `shard_idx = x`, rectangles from `(0,0)`.
  Break the moment the grid doesn't start at column/row 0 → **wrong mcast target → hang**.
- **Uniform `per_core` with no clamp?** `per_w = shard_w // 32` applied to every core,
  including the last, when `Wt` doesn't divide evenly → the last shard carries **whole
  padding tiles** that the op processes as real data.
- **Does it walk `shard_spec.grid` verbatim** (correct for sharded — data placement *is*
  the work assignment) **vs `split_work_to_cores`** (interleaved only)? If a sharded builder
  used `split_work_to_cores` it would ignore residency — a bug. If an interleaved path is
  reused for sharded without switching, likewise.
- **Rectangle assumption for mcast?** Any multicast needs a *rectangle*; a ragged
  `CoreRangeSet` (union of ranges) can't be one → the rectangle spans phantom cores → hang.
- **Orientation read?** Does it honor `shard_spec.orientation` (ROW_MAJOR vs COL_MAJOR),
  or assume ROW_MAJOR when mapping `shard_idx → (x,y)`?
- **Grid-shape / architecture assumptions?** Any literal `8`/`64`, any `x == y` or pow-2
  reasoning, any use of `compute_with_storage_grid_size()` that feeds index/count math. These
  pass on Wormhole's fortunate 8×8 but break on Blackhole's 10×13 (non-square, non-pow-2) or a
  grid. Ask literally: *"would this line hold on a 10×13 grid, or an 8×7 one?"* (see §3 → Grid
  portability).
- **Leading-dim collapse:** `rows = prod(leading) * ceil(H/32)` (correct, per-image tile
  padding) vs `ceil(prod * H / 32)` (wrong for multi-image non-aligned H).

### 1c. The kernels (device-side)
- **Cross-core sync:** mcast rectangle coords, participant count `K`, semaphore ids reused
  across disjoint groups, loopback/self-send handling, wait-before-send ordering.
- **Byte-offset math:** per-shard reads at `shard_idx * shard_w * elem` — sub-32B offsets
  get **floored to the alignment granule**, so odd shards read the neighbor's data.
- **Tile-count loops:** `for t in range(per_core)` reading past the valid region on the last
  shard; `read_sticks_for_tilize` / accessor reads past `tensor_volume()` (over-cover).
- **`element_size()` on block floats** (bf8b/bf4b) — invalid; guard behind a layout flag.
- **SPSC:** is a resident shard buffer used as *both* a compute output and a reader's mcast
  source? Two consumers on one CB → race (PCC-invisible scale error).

### 1d. Cross-reference the golden `feature_spec.py` INPUTS
List the sharded shapes the golden suite actually uses. They will be **roundish, even,
powers of two, built by `auto_shard_config`**. Everything *not* in that set — uneven splits,
multi-row grids, off-origin/ragged placements, sub-tile shards, COL_MAJOR, prime dims — is
ungraded. That delta is your target list.

---

## 2. Building adversarial shard specs (the toolkit)

Do **not** rely on `auto_shard_config` — it only produces the canonical geometry the grader
already covers. Build specs explicitly. Two entry points from `eval.sharding`:

```python
from eval.sharding import shard_config, auto_shard_config, Split, assert_legal_shard
import ttnn

WIDTH  = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK  = ttnn.TensorMemoryLayout.BLOCK_SHARDED
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
CRS, CR, CC = ttnn.CoreRangeSet, ttnn.CoreRange, ttnn.CoreCoord
```

`shard_config(shard_shape, grid, tensor_memory_layout, *, layout, dtype, orientation, device)`
gives exact control (and asserts the result is HW-legal). `grid` accepts an `(x,y)` tuple
(rectangle from origin) **or a `CoreRangeSet`** (arbitrary placement).

### Canonical control (must pass — anchors the ladder)
```python
mc = auto_shard_config(list(shape), WIDTH, layout=TILE, dtype=dtype, device=device)  # even, 1 row
```

### Uneven tile split (last shard gets a padding tile) — reachable via the DEFAULT helper
```python
# W = 352 = 11 W-tiles; PAD split -> 6 cores x 2 tiles = 12 -> 1 padding tile in the last shard.
mc = auto_shard_config([1,1,32,352], WIDTH, layout=TILE, dtype=dtype, device=device)  # Split.PAD default
# The safe workaround the op SHOULD steer users to:
mc_ok = auto_shard_config([1,1,32,352], WIDTH, layout=TILE, dtype=dtype, device=device, split=Split.EVEN)
```

### Multi-row WIDTH grid (more shards than the grid is wide)
```python
# 16 W-shards of 1 tile each on an 8x2 grid. auto_shard_config NEVER does this.
mc = shard_config([32, 32], (8, 2), WIDTH, layout=TILE, dtype=dtype, device=device)  # W = 16*32 = 512
```

### Off-origin placement (grid not based at (0,0))
```python
grid = CRS({CR(CC(1, 2), CC(4, 2))})           # 4 cores at y=2, x=1..4
mc = shard_config([32, 32], grid, WIDTH, layout=TILE, dtype=dtype, device=device)
```

### Ragged / non-rectangular grid
```python
grid = CRS({CR(CC(0,0), CC(7,0)), CR(CC(0,1), CC(3,1))})   # 8 + 4 = 12 cores, not a rectangle
mc = shard_config([32, 32], grid, WIDTH, layout=TILE, dtype=dtype, device=device)  # W = 12*32 = 384
```

### Non-square / Blackhole-shaped / sub-grid (emulate on Wormhole)
```python
# Force the op onto a grid that ISN'T the fortunate 8x8. `grid=` caps the split.
mc = auto_shard_config(list(shape), BLOCK, layout=TILE, dtype=dtype, device=device, grid=(10, 1))
mc = auto_shard_config(list(shape), BLOCK, layout=TILE, dtype=dtype, device=device, grid=(7, 3))   # non-square
# Or place explicitly on a non-square sub-grid (also off-origin-capable):
grid = CRS({CR(CC(0, 0), CC(9, 0))})  # 10 wide, 1 tall — a Blackhole-ish row that 8x8 can't be
mc = shard_config([32, 32], grid, WIDTH, layout=TILE, dtype=dtype, device=device)  # W = 10*32
```

### Sub-tile shards (ROW_MAJOR only — TILE requires 32-aligned edges)
```python
# RM width granule is 8 (bf16) / 4 (fp32); a shard can be 8/16/24 wide -> per-tile < 32.
mc = auto_shard_config([1,1,64,128], WIDTH, layout=RM, dtype=ttnn.bfloat16, device=device)
```

### COL_MAJOR orientation (transposed shard→core map)
```python
mc = shard_config([128, 64], (8, 2), BLOCK, layout=TILE, dtype=dtype,
                  orientation=ttnn.ShardOrientation.COL_MAJOR, device=device)
# NOTE: ttnn's tensor-spec layer rejects some COL_MAJOR configs itself (a TT_FATAL, not a bug).
# If it's accepted but the op indexes shard_idx=y*nx+x, the data is transposed -> wrong.
```

### Both dims non-aligned + leading-dim / rank variation
```python
for shape in [(1,1,47,100), (3,47,100), (47,100), (4,8,47,100), (1,1,33,33), (2,3,65,129)]:
    ...   # exercise partial_h AND partial_w together, ranks 2/3/4, big batch*channel
```

**Legality gate.** Before blaming the op, confirm the spec is HW-legal (else you're testing a
config a real op would reject): `assert_legal_shard(mc, layout=layout, dtype=dtype, device=device)`.
`from_torch` accepting it is NOT proof of legality.

---

## 3. Metrics & determinism — how to actually see the bug

PCC hides scale/structural errors (a uniform rescale has PCC ≈ 1.0). Always gate on **all** of:

```python
def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

def _rel_rms(got, exp):                       # scale + shape sensitive
    return ((got.float() - exp.float()).pow(2).mean().sqrt() / (exp.float().std() or 1)).item()

def _norm_ratio(got, exp):                    # THE scale detector: correct ~= 1.0
    return (got.float().norm() / (exp.float().norm() or 1)).item()

def _assert_correct(got, exp, ctx, pcc_gate, rms_gate, nr_band=(0.97, 1.03)):
    assert torch.isfinite(got.float()).all(), f"{ctx}: NaN/Inf"          # catches the RM-multirow NaN
    assert _pcc(got, exp) >= pcc_gate,   f"{ctx}: PCC low"
    assert _rel_rms(got, exp) <= rms_gate, f"{ctx}: rel-RMS high"
    lo, hi = nr_band
    assert lo <= _norm_ratio(got, exp) <= hi, f"{ctx}: SCALE error ||got||/||ref||={_norm_ratio(got,exp):.3f}"
```

**Deterministic inputs expose the mechanism.** All-ones turns a norm into a known constant:
for RMSNorm, all-ones → output should be exactly `1.0`; the multi-row reduction bug yields
exactly `sqrt(K_total / group_size)` (`√2` on an 8×2 grid) — an unambiguous fingerprint that
random+PCC would blur. Use all-ones / monotonic / per-position-marker inputs to read *what*
the op did, not just *that* it's off.

**L1 poisoning makes padding bugs reproducible.** Padding-tile bugs fold *stale L1* into the
result, so they flake green when L1 happens to be zero. Dirty it first:
```python
def _poison_l1(shape, mc, dtype, layout):
    p = ttnn.from_torch(torch.full(shape, 50.0, dtype=torch_dtype(dtype)),
                        dtype=dtype, layout=layout, device=device, memory_config=mc)
    ttnn.deallocate(p)   # the op's shard reuses this L1; the unmasked padding tile now reads 50.0
```

**Establish a baseline first.** Run a *fair* shape (aligned, pow-2, interleaved) to fix the
per-dtype rel-RMS band, then judge every adversarial case against it. A "wrong" result is one
that leaves the fair-shape band, not one that fails an arbitrary threshold.

**Grid portability & architecture (ask this on every op).** The single most important
portability question: **"would this work on *any* grid, and does the op make *any* assumption
about the grid itself?"** Grid dimensions are arch-specific, and Wormhole is a *fortunate*
square that hides bugs which Blackhole exposes:

| Arch | Compute grid | Notes |
|------|--------------|-------|
| Wormhole | **8×8** (64), or **8×7** (56) under WORKER dispatch | square, powers of two — flatters `x==y`, pow-2, and full-grid assumptions |
| Blackhole | **10×13** (130), or **10×11** (110) | non-square, non-pow-2 — breaks anything that assumed 8, assumed square, or assumed a pow-2 core count |

These are the grids the device *reports* (`compute_with_storage_grid_size()`) — **harvesting is
already reflected in them**, so don't treat it as a separate axis; just never hard-code the
dimensions and always read the live grid. The 8×7 / 10×11 variants come from dispatch config
(WORKER dispatch reserving a row), not from harvesting.

Concrete failure modes this surfaces:
- **Hard-coded 8 / 64 / pow-2**, or a `bounding_box.width`-as-count that "happens to divide" on
  8×8 but not on 10×13.
- **`x == y` / square assumptions** — index math that transposes cleanly only on a square grid.
- **Full-grid-from-(0,0) assumptions** — fine on the harness that hands the whole 8×8, wrong when
  the shard occupies a sub-grid or an off-origin region (category A2).
- **Regime/grid-dependent selection** — an op that branches on `compute_with_storage_grid_size()`
  can route a shape to a *different code path* per arch (the 56-vs-64 disparity already flipped
  rms_norm's regime and hid a bug). Log which path a shape actually takes; don't assume.

How to test it without the other silicon: (a) **cap the grid** — pass an explicit `grid=(gx,gy)`
to `auto_shard_config`/`shard_config` with **non-square, non-pow-2** extents (e.g. `(7,3)`,
`(10,1)`, `(5,11)`) to emulate Blackhole-shaped and narrower grids on Wormhole; (b) run the
**dual dispatch grid** (56-core `open_device` vs 64-core harness) where the op self-selects a
grid; (c) explicitly construct sub-grids and off-origin `CoreRangeSet`s (§2) so the op can't lean
on "the whole device starts at (0,0)". A green run on 8×8 alone is **not** evidence of grid
portability.

**Hangs → `--dev` + triage.** A suspected hang must be run with `scripts/tt-probe.sh --dev`
(or `run_safe_pytest.sh --dev`). Read `generated/tt-triage/triage.txt`: filter to the user
kernels, look at `Kernel Callstack` / waypoints — `NSW`/`cb_wait_front` on a reader in the
allgather, compute stuck on the external/partial CB = a cross-core deadlock (wrong mcast
target or participant count).

---

## 4. Test structure — escalation ladder & marking

Structure tests so the corner under test is obvious, and so the suite stays green while
documenting real bugs.

```
L0  precision baseline    fair shapes, every layout          -> PASS (defines the band)
L1  canonical sharding    even/auto, incl. large + rank      -> PASS (what the grader covers)
L2  interleaved corners   extreme partials, both-dims, ranks -> PASS (usually robust)
L3  uneven splits         padding-tile family                -> BUG?  (poison L1)
L4  grid geometry         multi-row / off-origin / ragged    -> BUG?  (deterministic + hang)
L5  local-axis robustness HEIGHT/uneven-H (the safe mirror)  -> PASS (isolates the bug)
```

Marking conventions:
- **Deterministic wrong-number bug** → `@pytest.mark.xfail(strict=True, reason="BUG …: <root cause + file:line>")`.
  Strict = it flips red when fixed, prompting marker removal.
- **Nondeterministic bug** (stale-L1 padding) → `xfail(strict=False)` + `_poison_l1` to maximize
  detection. A strict marker on a flaky outcome is itself flaky.
- **Hang** → `@pytest.mark.skip(reason="BUG …: HANGS — <root cause>")`. A hang is unrecoverable
  and aborts the whole session; never let it run in the normal suite. Keep a *passing* control
  nearby (e.g. the same geometry on a local-axis layout) to isolate the cause.
- **OOM / L1 fragmentation** → skip, not fail (out of scope):
```python
_ALLOC = ("bank_manager", "program.cpp:1741", "Out of Memory", "Statically allocated", "circular buffer", "clash")
def _run_or_skip(run, ctx):
    try: return run()
    except RuntimeError as ex:
        if any(k in repr(ex) for k in _ALLOC): pytest.skip(f"{ctx}: L1 alloc/fragmentation (OOM, out of scope)")
        raise
```
- **Never weaken the gate to make a case pass** (don't loosen tolerances, don't drop the
  norm-ratio/finiteness check, don't remove poisoning). Do **not** edit `eval/golden_tests/`
  (owned by the user); genuine gaps become INVALID/EXCLUSION escalations, not test edits.

Report per bug: reproduction, observed metrics, root cause with descriptor/kernel `file:line`,
and a fix direction. Mirror `ttnn/ttnn/operations/rms_norm/rms_norm_stress_test_report.md`.

---

## 5. The bug catalog (comprehensive, expandable)

Legend: **✓conf** = confirmed in an op (rms_norm here) · **~hyp** = plausible, hunt for it.
Add rows freely; keep the columns.

### A. Grid mapping — host assumed a shape the CoreRangeSet doesn't have
| ID | Trigger | Symptom | Detection recipe | Status |
|----|---------|---------|------------------|--------|
| A1 | WIDTH shards > grid width (multi-row grid) | reduce sums only one row's shards → scale by `√(K_total/group)` (or NaN) | `shard_config([32,32],(8,2),WIDTH,…)`; all-ones → `√2` | ✓conf (RMS BUG 1) |
| A2 | grid not based at (0,0) | absolute mcast coords hit wrong cores → **hang** | off-origin `CoreRangeSet`; `--dev` triage | ✓conf (RMS BUG 3) |
| A3 | ragged / non-rectangular grid | mcast rectangle spans phantom cores → **hang** | `CRS({CR(0,0→7,0), CR(0,1→3,1)})` | ✓conf (RMS 3b) |
| A4 | COL_MAJOR orientation | `shard_idx→(x,y)` transposed → wrong data | build COL_MAJOR; compare to torch ref | ~hyp (infra rejects some) |
| A5 | grid holes / non-contiguous worker rows (dispatch & eth cores) | a "start..end" rectangle walk spans cores that aren't workers | inspect the live worker grid; sub-grid + off-origin specs | ~hyp |
| A6 | logical vs virtual coord confusion | NoC targets wrong physical core | audit `worker_core_from_logical_core` usage | ~hyp |
| A7 | arch grid-shape assumption (8 / 64 / square / pow-2) | works on WH 8×8, breaks on BH 10×13 (non-square, non-pow-2) | cap grid to non-square non-pow-2 `(7,3)`,`(10,1)`,`(5,11)`; ask "holds on 10×13?" | ~hyp |

### B. Padding & remainders — shard covers more than the logical tensor
| ID | Trigger | Symptom | Detection recipe | Status |
|----|---------|---------|------------------|--------|
| B1 | `Wt % ncores != 0` (uneven tile split) | last shard's padding tile folded into reduce → collapse/scale, **nondeterministic** | default `auto_shard_config` on non-pow2 W + `_poison_l1` | ✓conf (RMS BUG 2) |
| B2 | sub-tile shard read as a full tile | reads neighbor / OOB | RM sub-tile width (8/16/24); all-ones + per-position markers | ✓conf (RMS 5c) |
| B3 | ceil-split over-cover (shard rows > tensor rows) | accessor reads past `tensor_volume()` → assert/**hang** | tall shape, tile-count coprime to grid; `--dev` | ✓conf (RMS 4b) |
| B4 | tile-padding vs shard-padding conflated | mask handles sub-tile cols but not whole padding tiles | both-non-aligned + uneven split together | ✓conf (RMS B1 root) |
| B5 | zero-work core (all-padding shard) | kernel launched, waits on CBs that never fill → **hang** | craft a spec giving one core an empty shard | ~hyp |

### C. Cross-core communication / reduction
| ID | Trigger | Symptom | Detection recipe | Status |
|----|---------|---------|------------------|--------|
| C1 | wrong participant count `K` | hang (too many) or scale error (too few) | multi-row / partial-row grids | ✓conf (RMS A1/A3) |
| C2 | semaphore id reuse across overlapping groups | cross-talk / missed signal | BLOCK with adjacent groups; `--dev` | ~hyp |
| C3 | mcast loopback / self-send off-by-one | partial double-counted or dropped | all-ones exactness (should be integer) | ~hyp |
| C4 | wait-before-send ordering | deadlock under some grid | vary group size; `--dev` | ~hyp |
| C5 | SPSC violation: shard buffer is compute-output AND mcast-source | PCC-invisible scale race, dispatch-timing dependent | all-ones exactness across repeated runs | ✓conf (RMS Regime-B orig) |

### D. CB / L1 placement (sharded-specific)
| ID | Trigger | Symptom | Detection recipe | Status |
|----|---------|---------|------------------|--------|
| D1 | CB page size ≠ shard page | reads walk off the shard | mismatched dtype/tiling vs `cb_descriptor_from_sharded_tensor` | ~hyp |
| D2 | in-place alias when out-shape ≠ in-shape | output clobbers input mid-flight | shape-changing op with inherited shard spec | ~hyp |

### E. Addressing / stride math
| ID | Trigger | Symptom | Detection recipe | Status |
|----|---------|---------|------------------|--------|
| E1 | sub-32B / non-L1-aligned byte offset | offset floored → reads aligned neighbor | sub-tile shard, per-shard byte gather; distinct per-shard values | ✓conf (RMS 5c gamma, PCC 0.52) |
| E2 | shard-width vs tensor-width stride confusion | diagonal / shifted reads | RM shard where `shard_w != W`; monotonic input | ~hyp |
| E3 | `element_size()` on bf8b/bf4b | invalid size → wrong geometry, regresses block-float | bf8b sharded after an RM/byte-geometry change | ✓conf (RMS R3) |

### F. Shape / rank / batch × sharding
| ID | Trigger | Symptom | Detection recipe | Status |
|----|---------|---------|------------------|--------|
| F1 | multi-image non-aligned H, flat row formula | wrong tile-index stride → `nan`/garbage | `(4,8,47,W)` etc.; compare to torch | ✓conf (RMS R2) |
| F2 | sharding the axis the op reduces/operates on | local op becomes cross-core (or wrong) | shard the reduced dim; all-ones | ~hyp (op-dependent) |
| F3 | sharded dim = 1, or rank < 2 | degenerate spec mishandled | craft the degenerate shape | ~hyp |

### G. Output memory-config propagation
| ID | Trigger | Symptom | Detection recipe | Status |
|----|---------|---------|------------------|--------|
| G1 | inherit input shard spec when out-shape differs | illegal/oversized output config | reduction/shape-changing op, no explicit out `memory_config` | ~hyp |
| G2 | caller `memory_config` grid ≠ computed grid | work/alloc mismatch | pass a mismatched output config | ~hyp |

**To add an entry:** give it an ID in the right category, a one-line *trigger* (the geometry),
the *symptom* (hang / scale / NaN / garbage), a concrete *detection recipe* (the spec + the
metric/input that reveals it), and mark ✓conf once reproduced with `file:line`.

---

## 6. Prioritization (where to spend first)

1. **Any axis the op reduces over or communicates across, that can be sharded.** Cross-core
   group logic (category A/C) is the highest-yield and the worst failures (silent scale + hangs).
2. **Any per-shard byte-offset or tile-count arithmetic** (categories B/E). Padding/stride/
   alignment bugs are subtle, nondeterministic, and reachable through the *sanctioned* helper.
3. **Local-axis sharding (e.g. HEIGHT for a W-reduction) is the safe mirror** — use it as a
   *control*: if the same adversarial geometry passes on the local-axis layout but fails on the
   cross-core layout, you've isolated the bug to the cross-core machinery. (In rms_norm, every
   confirmed bug was WIDTH/BLOCK; HEIGHT was immune because W stays local — no cross-core coords
   or shard-counts to get wrong.)

## 7. Common false positives — don't report these as op bugs
- **OOM / L1 fragmentation** (`program.cpp:1741`, `bank_manager`) — allocator-reported, out of
  scope; skip. (But spot-check the "passes in isolation" claim — re-run the failing cell in a
  fresh device before accepting the fragmentation narrative.)
- **Legitimate `ExcludedCell` / `ValueError` refusals** — the op correctly declining an
  unsupported combination is not a bug.
- **Infra-rejected configs** — a `TT_FATAL` from the tensor-spec layer (e.g. some COL_MAJOR
  shapes) is ttnn guarding an illegal config, not the op.
- **Honest low-bit rounding** — bf8b/bf16 near a tolerance edge on a genuinely hard shape.
  Judge against the fair-shape baseline band, not an arbitrary gate.
