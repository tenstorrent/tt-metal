# indexer_score — fast-untilize (SOLVED)

Goal-oriented notes for landing the **fast `pack_untilize` path** in the
`indexer_score` compute kernel. Focus case is **8 heads, bfp8 k**.

## STATUS: DONE ✅ (2026-06-11)

The `W>=2` fast-untilize path is landed and correct multi-core. Two distinct
bugs had to be fixed (the handoff doc conflated them as one "corruption"):

1. **Output layout.** The BH fast strided untilize does NOT emit `W` separate
   32×32 tiles — it emits ONE wide row-major strip: 32 rows × `W·32` bf16, row
   pitch `W·32` elements (confirmed in `llk_pack_fast_untilize.h` and the conv3d
   consumer). The committed per-tile writer reads that as garbage, so a naive
   `untilize<KC>(1)` into `cb_out` fails *even single-core* (PCC ~0.05). The
   ruled-out variants below passed single-core only because they had already
   adapted the writer; this was the half that was easy to miss.

2. **The multi-core sync drift = the LLK uninit asymmetry.** The BH fast
   untilize runs a *private* `SyncHalf` DEST contract
   (`FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE`, `llk_fast_untilize_common.h:37`).
   `fast_untilize_init` re-seeds the math↔pack semaphore and `reset_dest_offset_id()`
   (`fast_untilize.h:64-65`), but `fast_untilize_uninit`'s matching re-sync is
   guarded by `if constexpr (DST_SYNC_MODE != FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE)`
   (`fast_untilize.h:167`). This kernel is built `dst_full_sync_en=false` →
   `DST_SYNC_MODE == SyncHalf == FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE`, so **that
   re-sync is compiled out**. Across the many units a multi-core run packs, the
   `dest_offset_id` parity drifts, the packer reads the wrong DEST half, and the
   diagonal strict-upper `-inf` mask tile bleeds into valid cells of full tiles —
   exactly the "per-row-varying `-inf` count within a single tile" signature.
   (conv3d uses fast untilize + half-sync and is fine because its per-iteration
   section count stays balanced; the indexer's per-unit count varies.)

### The fix (no shared-LLK change)

- **Layout:** dedicated `cb_acc_strip` (c_27) + `cb_out_strip` (c_18) with
  uniform `KC` push/pop, so the fast packer's KC-tile reads never wrap the ring
  (mixing KC and 1-tile pushes on one ring straddles it — the doc's
  push-must-divide-capacity hazard). `accumulate_heads` is slot-indexed so a
  full-width row's KC output tiles land contiguously in `cb_acc_strip`, then one
  `untilize<KC, cb_acc_strip, cb_out_strip>(1)` produces the strip. A new
  `write_strip` in the writer scatters each of the 32 strip rows as a single
  contiguous `KC·64`-byte run (one `async_write`/row vs `KC` per-tile 64 B
  fragments). Compute and writer pick the full-strip-vs-per-tile path
  identically via the shared `valid_prefix_tiles` helper
  (`indexer_score_work_split.hpp`). Partial-prefix and masked-suffix rows stay on
  the per-tile `W=1` path into `cb_acc`/`cb_out`.
- **Sync:** a full `mm_block_init(cb_q, cb_k, cb_qk, ...)` after each fast
  untilize re-establishes the kernel's sync contract (`llk_math_pack_sync_init`
  + `llk_pack_dest_init`, `matmul.h:264,272`) before the next unit's matmul. This
  is the same reset `fast_untilize_init` does on entry; doing it kernel-side
  avoids touching the shared LLK (which conv3d et al. depend on). Cost ≈ 0.6% of
  runtime — dwarfed by the untilize win.

### Measured result (sp7, BH 110 cores, this board)

- `IDX_UNTILIZE` zone: **43.7 ns/tile** (was ~148 ns/tile at `W=1`). Target met.
- Whole-kernel device duration (committed `W=1` → fast `W>=2`):
  - heads8  bfp8: 1.322 → **0.827 ms** (1.60×, math_util 18.3 → 29.2%)
  - heads16 bf16: 1.683 → **1.326 ms** (1.27×, 28.7 → 36.4%)
  - heads16 bfp8: 1.507 → **0.827 ms** (1.82×, 32.0 → 58.4%)
  - heads64 bf16: 3.055 → 3.011 ms (1.01×) — matmul-bound, untilize <2%, no regression
  - heads64 bfp8: 3.012 → 2.975 ms (1.01×) — same
- All 33 accuracy/validation tests + 5 sp7 perf cases pass; the multi-core
  `t=16384` repro that failed every prior variant now passes (PCC 0.99997, exact
  `-inf` map). NOTE: the handoff doc's old "committed 0.488 ms" 8-head baseline
  was stale — the true current-board committed baseline is 1.322 ms.

Files touched: `compute_indexer_score.cpp` (strip path, slot-indexed accumulate,
`produce_full_strip` + sync reset), `writer_indexer_score.cpp` (`write_strip`),
`indexer_score_common.hpp` (`use_fast_strip`, `row_valid_prefix`),
`indexer_score_work_split.hpp` (`valid_prefix_tiles`),
`indexer_score_program_factory.cpp` (the two strip CBs).

---

## Original TL;DR (for reference)

- The kernel ends every output tile with a `pack_untilize` (TILE → row-major).
- Today it uses `compute_kernel_lib::untilize<1, cb_acc, cb_out>(n)` — block
  width `W=1`, which **does not hit the BH fast-untilize LLK** (that needs
  `W>=2`). Measured untilize cost: **~148 ns/tile**.
- A `W>=2` block untilize is **~3× faster: ~45 ns/tile** (measured via zones).
- That's worth ~**7% of the 8-head compute ceiling** (~1% at 64 heads — it only
  matters for the low-head-count TP shard, because at high head counts the
  matmul dominates and untilize is already <2%).
- **Blocker (now fixed, see above):** every `W>=2` variant tried was correct
  single-core but **corrupted output deterministically at multi-core + large `t`**.

## Current state (baseline to build on)

- Branch `skrstic/dsa_indexer_score_op_2`, commit **`47452b9a4a2`**
  ("indexer_score batch untilize per work unit").
- That commit already did the *safe* untilize win: per-unit `num_blocks=n`
  batching amortizes the `pack_untilize` init/uninit bracket (one per work unit
  instead of per tile). 8-head bfp8 sp7 ceiling went **0.523 → 0.488 ms
  (46.2% → 49.4% math util)**. Correctness verified on production + bfp8.
- The compute untilize call lives in `untilize_acc_strip(...)` in
  `ttnn/cpp/ttnn/operations/experimental/deepseek/indexer_score/device/kernels/compute_indexer_score.cpp`.
- Per-output-tile compute breakdown at 8 heads (zones, DMA off):
  `IDX_MATMUL ~46% / IDX_MULACC ~45% / IDX_UNTILIZE ~10%`. Untilize is the
  remaining fixed-overhead lever at low head count.

## What "hit the fast untilize path" means

`compute_kernel_lib::untilize<W, in_cb, out_cb>(num_blocks)` dispatches in
`ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl`:

- `can_use_fast_untilize<W, in, out>()` (search the `.inl`) requires, on
  Blackhole: **`W >= 2`** AND 32×32 tiles AND input `Float16_b`/`Bfp8_b` +
  output `Float16_b`. Our `cb_acc`/`cb_out` are `Float16_b` (bf16 DEST), so the
  only thing stopping the fast path is **`W==1`**.
- `FAST_UNTILIZE_MAX_UNIT_DIM = 4`
  (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_fast_untilize_common.h`):
  - `W` in {2,3,4} → **non-strided** fast path (`full_ct_dim <= MAX_UNIT_DIM`).
  - `W > 4` → **strided** fast path (chunks of 4). Produces a row-major strip
    too, but via `llk_pack_fast_untilize_block_strided`.
- LLK source: `tt_metal/hw/inc/api/compute/experimental/fast_untilize.h` and
  `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_*fast_untilize*.h`.

Production work-unit knobs (from `production_config()` in the test): **QC=1,
KC=16** (`k_chunk_size=512`), all heads resident. So a full unmasked unit is a
**16-tile row-strip** — the natural thing to fast-untilize as `W=16` (strided)
or `W=4 × 4 sub-strips` (non-strided). `cb_acc` is sized `2*QC*KC` and `cb_out`
`2*KC` (factory `indexer_score_program_factory.cpp`), so they already hold a
full strip.

### Where the masking interacts (important)

Masked (`-inf`) tiles are a **contiguous suffix** of each unit's column range
(`k_tile` rises with `c`). The committed kernel already splits each row into:
- **unmasked prefix** `[0, valid)` → `untilize_acc_strip(valid)` (the thing to
  make fast); only `valid == KC` units are full-width strips.
- **masked suffix** `[valid, k)` → per-tile `untilize<1>(1)` + `add_mask`
  (must stay per-tile: `add_mask` reorders `cb_acc`, and masked tiles are rare).

So the fast path only needs to cover the **full-width prefix** (`valid == KC`).
A correct implementation also needs the **writer** to consume whatever layout
the strip untilize produces, in lockstep (`writer_indexer_score.cpp`).

## What was tried and RULED OUT (don't repeat blindly)

All of these are **correct single-core** but **corrupt deterministically at
multi-core + large `t`** (`-inf` appearing in valid cells, at full-unit strip
tiles):

1. `untilize<KC=16>(1)` strip into the shared `cb_out`. → fails multi-core.
2. **Separate strip CB** `cb_out_strip` (c_18, uniform KC pushes, so no
   ring-wrap straddle). → still fails. ⇒ not a CB-wrap / push-divides-capacity
   issue.
3. **Width-4 non-strided sub-strips** `untilize<4>(KC/4)` (avoids the strided
   pack path entirely). → fewer mismatches but still fails. ⇒ not the strided
   pack.
4. **64 B-fragment writer** reading from the strip layout (preserves the
   documented 64 B partial-page disjointness invariant; writer becomes
   byte-identical to the safe per-tile writer). → still fails. ⇒ not the write
   granularity / DRAM RMW race.

Also confirmed: compute and writer get the **same per-core flat range**
(factory sets identical `{flat, count}` for compute and writer), so it is not a
flat-range desync. The full↔partial decision (`valid == k_tiles_per_unit`) is
computed identically in both kernels.

**Common factor across all failures:** a `W>=2` `pack_untilize` of full units,
only at multi-core scale. `W=1` (committed) is always correct. Leading
hypothesis: an LLK **packer/DEST state interaction** in the fast untilize that
drifts across many full↔partial untilize transitions (a core processing many
units interleaves `W>=2` full-unit untilizes with `W=1` partial-unit untilizes
and the matmul/mul packs). Next investigation should instrument that.

## Bisection signal (use to reproduce fast)

- **Single-core** (small `sq`): all `W>=2` variants PASS.
- **Multi-core, small `t`** (`t<=8192`): PASS.
- **Multi-core, large `t`** (`t>=16384`): FAIL, deterministically (same mismatch
  count across runs). Failures land on **full-unit strip tiles**, with a
  per-row-varying number of `-inf` columns within a single tile.

So the cheapest failing repro is multi-core (sq=640) with `t≈16384, cs≈12288`,
not the full 56320 production shape.

## How to measure

`source python_env/bin/activate` first. Always go through
`scripts/run_safe_pytest.sh` (flock + auto device reset; never `tt-smi -r`).
Kernel edits JIT-rebuild (no `build_metal.sh`); **factory/CB edits need
`./build_metal.sh`** (host C++).

### Accuracy (must pass before claiming a win)

Production shape (sq=640, t=56320), multi-core — this is where the strip bug shows:

```bash
scripts/run_safe_pytest.sh --run-all \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_production \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_bfp8_k
# expect 8 passed; the test checks the -inf map EXACTLY and visible PCC >= 0.999
```

Before merge also run the knob/corner coverage (QC=2, KC=1, diagonal-mid-group):

```bash
scripts/run_safe_pytest.sh --run-all \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_knobs \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_corner_shapes \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_glx_chunked
```

Fast iteration repro (single-core PASS vs multi-core FAIL), 8 heads — save and run:

```python
# /tmp/idx_repro.py  — run:  flock /tmp/tt-device.lock python /tmp/idx_repro.py
import torch, ttnn, numpy as np
dev = ttnn.open_device(device_id=0)
def run(sq, t, cs, label):
    heads, dim = 8, 128
    g = torch.Generator().manual_seed(7)
    q = torch.randn(1, heads, sq, dim, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, t, dim, generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, heads, sq, 1, generator=g, dtype=torch.bfloat16)
    dt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=512, head_group_size=0)  # KC=16
    o = ttnn.to_torch(ttnn.experimental.deepseek.indexer_score(
        dt(q), dt(k), dt(w), is_causal=True, chunk_start_idx=cs, program_config=cfg)).float()[0, 0]
    ref = torch.zeros(sq, t)
    for h in range(heads):
        ref += torch.relu(q[0, h].float() @ k[0, 0].float().T) * w[0, h].float()
    fut = torch.arange(t).unsqueeze(0) > cs + torch.arange(sq).unsqueeze(1)
    om = o <= torch.finfo(torch.bfloat16).min
    inf_mismatch = int((om != fut).sum())
    valid = ~fut
    a, b = o[valid][o[valid] > torch.finfo(torch.bfloat16).min].numpy(), ref[valid][o[valid] > torch.finfo(torch.bfloat16).min].numpy()
    pcc = float(np.corrcoef(a, b)[0, 1]) if len(a) else float('nan')
    print(f"{label}: -inf mismatches={inf_mismatch}  PCC(valid)={pcc:.5f}")
try:
    run(64,  2048, 512,   "single-core  (expect PASS)")
    run(640, 8192, 4096,  "multicore t=8192  (expect PASS)")
    run(640, 16384,12288, "multicore t=16384 (expect FAIL on the strip path)")
finally:
    ttnn.close_device(dev)
```

### Perf — the untilize region (the number to move)

Add device zones to the compute kernel to isolate the untilize region, then run
under tracy. In `compute_indexer_score.cpp`:

```cpp
#include "tools/profiler/kernel_profiler.hpp"   // top of file
// wrap the prefix untilize call:
{ DeviceZoneScopedN("IDX_UNTILIZE"); untilize_acc_strip<...>(valid); }
// (optionally wrap the matmul phase as IDX_MATMUL and mul+accum as IDX_MULACC
//  inside accumulate_heads() to see the full split)
```

Profile the 8-head bfp8 sp7 case (hold the device lock so it's cooperative):

```bash
flock /tmp/tt-device.lock bash -c '
  source python_env/bin/activate
  python -m tracy -r -p -o generated/profiler/idx_zone \
    -m "pytest tests/nightly/blackhole/sdpa/test_indexer_score.py::test_indexer_score_sp7_perf_impl[heads8_k_bfp8]"
'
```

Read the zone (BH clock 1.35 GHz; ZONE name col 11, type col 12, cycles col 6):

```bash
awk -F, '$11=="IDX_UNTILIZE"{
    if($12=="ZONE_START"){t[$1"_"$2]=$6}
    else if($12=="ZONE_END"){k=$1"_"$2; if(k in t){d=$6-t[k]; s+=d; n++; delete t[k]}}
} END{printf "IDX_UNTILIZE: pairs=%d avg=%.1f cyc (%.1f ns) per call\n", n, s/n, (s/n)/1.35}' \
  generated/profiler/idx_zone/.logs/profile_log_device.csv
```

Reference numbers (8-head bfp8 sp7): one `IDX_UNTILIZE` call covers ~16 tiles
(a full unit). num_blocks `W=1`: ~2381 ns/call ≈ **148 ns/tile**. Fast `W>=2`:
~718 ns/call ≈ **45 ns/tile**. Target: get the fast number *with accuracy
passing*.

### Perf — whole-kernel ceiling / math util

```bash
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_indexer_score.py::"test_indexer_score_sp7_math_util[heads8_k_bfp8]"
# logs: device=<ms>, math_util=<%>.  Baseline (committed): 0.488 ms, 49.4%.
```

For a **pure compute ceiling** (untilize is fixed overhead that hides behind
matmul once DMA is on), isolate compute by commenting out the `noc.async_read`/
`noc.async_write` calls in `reader_indexer_score.cpp` / `writer_indexer_score.cpp`
(keep the CB `reserve/push/wait/pop` so the pipeline still flows) — mark them
`// [compute-ceiling]` and restore after. With DMA off the 8-head ceiling was
0.488 ms / 49.4%. Verify accuracy with DMA restored.

## Suggested plan for next session

1. Reproduce the FAIL cheaply with `/tmp/idx_repro.py` (multicore t=16384).
2. Re-add the `W=4` non-strided sub-strip untilize (simplest fast path) +
   separate `cb_out_strip` + 64 B writer (all already worked single-core).
3. Instrument the LLK: confirm whether the fast untilize's pack/DEST sync state
   (`FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE`, `_llk_pack_dest_init_`,
   `pack_untilize_uninit`) is fully restored for the subsequent matmul/mul packs
   and the next per-tile `W=1` untilize. Suspect a missing uninit/reinit when
   full (`W>=2`) and partial (`W=1`) untilizes interleave many times on one core.
4. If it's a state-restore gap: force a clean `pack_untilize_uninit` /
   `mm_block_init` boundary between the strip untilize and the next unit, or keep
   the packer state explicit. Re-measure `IDX_UNTILIZE`; expect ~45 ns/tile.
5. Gate: accuracy (production + bfp8 + knobs) green, untilize zone down,
   `math_util[heads8_k_bfp8]` improved.

## File map

| file | what |
|---|---|
| `.../indexer_score/device/kernels/compute_indexer_score.cpp` | `untilize_acc_strip()`, `accumulate_heads()`, masked-suffix per-tile path |
| `.../indexer_score/device/kernels/writer_indexer_score.cpp`  | output scatter (`write_tile`; add `write_strip` for strips) |
| `.../indexer_score/device/kernels/indexer_score_common.hpp`  | dims/knobs, `WorkUnitSpan`; put a compile-time `strip_w` here |
| `.../indexer_score/device/indexer_score_program_factory.cpp` | CB allocation (`make_cb`); add `cb_out_strip` here (needs build) |
| `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.{hpp,inl}`        | `untilize<W>()`, `can_use_fast_untilize`, dispatch |
| `tt_metal/hw/inc/api/compute/experimental/fast_untilize.h`   | fast untilize entry; `FAST_UNTILIZE_MAX_UNIT_DIM=4` |
| `tests/nightly/blackhole/sdpa/test_indexer_score.py`         | accuracy + `sp7_math_util` perf; `heads8_k_bfp8` case |
