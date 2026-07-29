# pi0.5 tiny-tile (16×32) integration plan

Branch: `pi05_tiny_tile_integration` (from `pi05_16_chip_rebase_main`)
Source: `origin/smanoj/pi0_tiny_tile` @ `6d86b8a1812`

## Topology

`models/experimental/pi0_5`, `matmul_decode` and `kv_sdpa` exist on **neither** main base —
both branches add them independently, so there is no shared git ancestor for them.

```
main@a206c6afcfd (Jul 14) ── 3 commits ──► HEAD   pi0_5 (181 files), fused matmul_decode, kv_sdpa
   └── …main… ──► main@f1f4ff75579 ── 215 ──► smanoj/pi0_tiny_tile
                                              tiny-tile ttnn infra (238 files / 27k lines)
                                              + pi0_5 lean subset (50 files) + tiny-tile denoise
```

Tiny-tile model delta is only **11 files / +770** (`git diff 061501a4bed..tip -- models/experimental/pi0_5`),
all in the denoise path. `tt_bh_glx/pipeline_16_decode.py:22-27` imports `denoise_block` /
`denoise_pipeline` directly, so it lands exactly where the production 16-chip pipeline runs.

## Design rule

```
32×32 : prefix KV, matmul weights (inputB), adaRMS mod-Dense weights
16×32 : suffix activations, q/k/v, RoPE tables, norm weights/biases, mod triples, x_t
```

`matmul_decode` rides the tiny tile on **inputA only**. Mixed-tile `kv_sdpa` absorbs the
32-prefix / 16-suffix mismatch (own CB pair per phase, shared online-softmax state), so the
prefill→denoise handoff needs no retile.

Switch: `tt/tile_config.py` `TILE_HEIGHT`, overridable via **`PI05_TILE_HEIGHT`** (16 or 32) so the
A/B runs against one build.

The win: `perf_suffix_len(ah=10, 16) = 16` instead of 32. A 16×32 tile has 2 faces not 4, so
denoise matmul M-work roughly halves. Pushback: SDPA forces `subblock_w == 1` on partial-face
tiles (LLK `partial_face` unpack/math mismatch for `ct_dim > 1`); a retile lands in the hot path
(`_to_tile16_bf8` on the SDPA output); the unfused MLP adds a `sharded_to_interleaved`.
**Net effect must be measured, not assumed.**

## Decisions

- **Sequencing: correctness first, then re-fuse.** Their tiny-tile block unfuses
  `matmul_decode`/`gate_up`/`concat_heads` and drops LoFi + the residual/gate epilogues because it
  targets a leaner `matmul_decode`. Stage 1 lands it as-is for a PCC-verified reference; Stage 7
  re-fuses onto our ops with per-step A/B.
- **Keep both `matmul_decode` copies.** `ttnn.matmul_decode` (ours, fused) keeps the production
  16-chip pipeline running; `ttnn.experimental.matmul_decode` (theirs, tiny-tile inputA) serves the
  tiny-tile denoise path. Unify only once the numbers pick a direction.

## Stages

| # | Stage | Verify |
|---|---|---|
| 1 | Merge + resolve 18 conflicts | no markers; `git diff --check`; all pi0_5 modules import |
| 2 | Fix the 5 verification-blocking defects | targeted |
| 3 | One full rebuild (~30 min, no ccache; touches `tt_metal/impl/data_format`) | build green |
| 4 | ttnn-level tiny-tile unit tests | `test_tiny_tile.py`, `test_tilize_retile`, `test_sdpa_tiny_tile`, `test_matmul_decode.py` |
| 5 | Control: `TILE_HEIGHT=32` | merge did not regress the 32×32 path |
| 6 | Target: `TILE_HEIGHT=16` | `test_l1_single_layer_pcc` ≥ 0.99; walltime; `compare_profiles.py` 32-vs-16 |
| 7 | Production 16-chip path | `test_perf_tt_bh_glx_16_e2e_trace_2cq.py` + PCC suite |
| 8 | Re-fuse onto our fused ops | A/B each step against Stage 6 |

## Results

### Stage 3 — build

Two duplicate-registration merge artifacts had to be fixed first. Both sides added registration
lines in **different files**, so git auto-merged them with no conflict:
`kv_sdpa` was `add_subdirectory`'d from both `ttnn/CMakeLists.txt` and
`operations/CMakeLists.txt` (hard CMake error), and `ttnn-nanobind/__init__.cpp` both included
`kv_sdpa_nanobind.hpp` and defined the `m_kv_sdpa` submodule twice. **Lesson: after a merge like
this, always grep for duplicated CMake/nanobind registrations — conflict markers will not show
them.** Also note `cmake --build` alone is not enough; `build_metal.sh` uses `--target install`,
which is what stages `_ttnn.so` into `ttnn/ttnn/`.

### Stage 4 — ttnn tiny-tile verification

Found and fixed a real bug **in the source branch**: `QK_COL_VECTOR_MODE` was defined at namespace
scope in *both* `compute_common.hpp` and `compute_streaming.hpp`, and five kernels include both in
one TU (`sdpa.cpp`, `{exp_,}ring_joint_sdpa.cpp`, `sparse_sdpa{,_msa}_compute.cpp`), so none of them
JIT-compiled — `ttnn.transformer.scaled_dot_product_attention` and the joint/sparse variants were
entirely dead at **both** tile heights. Hidden on their branch because `kv_sdpa/flash_fused.cpp`
includes only `compute_common.hpp` and the denoise path uses `ttnn.kv_sdpa`; only the SDPA fallback
and their own new nightly `test_sdpa_tiny_tile` reach it. **Report upstream to Sankar.**

| Test | tile32 | tile16 |
|---|---|---|
| `test_sdpa_tiny_tile_numerics` | PCC 0.99993 | PCC 0.99993 |
| `test_pad_tiny_tile_corrupts_data` | 0.99999 | 0.99999 |
| `test_addcmul_tiny_tile_promotes_tile` | pass | pass, `out_tile==(16,32)` |
| `test_sdpa_bf8_mask_corrupts` qkv_bf16 | pass | pass |
| `test_sdpa_bf8_mask_corrupts` qkv_bf8 | pass | **FAIL, PCC −0.99983** |
| `experimental/test_matmul_decode.py` | — | 12 passed |
| `test_tilize_retile` | 40 passed / 10 skipped (bf8 at height < 16), 0 failed |

These files are *bug reproducers* whose docstrings say tile16 is expected to fail — they now pass,
so the branch really did fix the tiny-tile SDPA numerics and the addcmul tile-promotion. That
retroactively justifies `_gated_residual` keeping `ttnn.addcmul`.

One real bug remains, narrowly scoped: **bf8 q/k/v + dense `attn_mask` + 16-row tile inverts the
sign**. bf16 passes at both heights and bf8 passes at 32×32. Does not block the denoise path, which
passes `attn_mask=None` (the mask is an all-zero no-op) — but that workaround is load-bearing.

Deferred: the full 713-item `test_tilize.py` sweep (~5 h at ~25 s/param since each re-opens a
32-chip mesh). 35/35 passed before it was stopped; re-run it and the `i2s`/`s2i` regressions
out-of-band.

### Stage 5/6 — model correctness + perf

`test_l1_single_layer_pcc` (the reference gate) **passes at both tile heights, PCC 0.9999** vs the
torch oracle. At `TILE_HEIGHT=32` all three tests pass, so the merge did not regress the existing
32×32 path.

Three more defects had to be fixed to get there — see the commit log. The notable one:
**`concat_heads_matmul` was never bound to Python.** It is built and in `sources.cmake`, and
`bind_concat_heads_matmul` is declared *and defined*, but `experimental_nanobind.cpp` only called
`bind_concat_heads_matmul_decode`. `ttnn_gemma.py` called it directly and unguarded, so the DRAM
ExpertChunkSlice denoise path was broken — **pre-existing on our branch, not a merge regression**
(pre-merge made the same call with the same missing bind). Fixing the bind restored that path.

Perf, L1 single-layer traced replay (`test_walltime_l1_single_layer`, added because the existing
walltime test needs the DRAM path and so is 32×32-only):

| Configuration | median | note |
|---|---|---|
| tile-32, **fused** rope | 0.1825 ms | current baseline |
| tile-32, unfused rope | 0.2130 ms (n=3, 0.212–0.214) | |
| tile-16, unfused rope | 0.2010 ms (n=3, 0.198–0.201) | |

Two independent deltas:
- **The tile geometry is a real ~5.6% win** (tile-16 beats tile-32 on identical code; ranges do not
  overlap across 4 paired runs).
- **Losing the fused rope costs ~0.030 ms / +17%** — the `nlp_create_qkv_heads_rope` fallback turns
  1 dispatch into 3.

So tiny tile as integrated is ~10% slower than baseline *entirely because of the defusion*, not the
geometry. Making the rope op tiny-tile aware should land tile-16 near ~0.171 ms vs the 0.1825 ms
baseline (~6% better) — consistent with the measured geometry win.

The gain is ~5%, not the ~2× that "half the M-work" would suggest, because SDPA forces
`subblock_w == 1` on partial-face tiles and the hot path pays a retile (`ttnn.tilize` + `ttnn.slice`
on the SDPA output). Both are documented LLK/kernel limitations, not tuning knobs.

### Stage 7 — production 16-chip path: BLOCKED, and why

Measured with the user's commands (`PI0_NUM_CAMERAS={2,3} PERF_ITERS=20`):

| 3-cam 16-chip e2e | result |
|---|---|
| **pre-merge**, DECODE_ALL=1 (our **fused** block) | 28.52 ms (2-cam 25.31 ms) |
| post-merge, DECODE_ALL=1 (their **unfused** block) | **L1 clash** |
| post-merge, DECODE_ALL=0 (plain linear) | 31.03 ms |

`Statically allocated circular buffers ... clash with L1 buffers on core range [0-0 - 7-0]. L1
buffer allocated at 592576 and static circular buffer region ends at 800512`

**Root cause: the defusion, not the tile geometry and not main drift.** Ruled out by measurement:
bf8 32×32 is 1088 bytes both pre- and post-merge (only 16×32 grows, 544→576); the
`interleaved_to_sharded` CB rewrite is byte-identical at 32×32; and the 1×8 path still passes
post-merge. The unfused path materializes L1 intermediates our fused ops folded away —
`_matmul_decode_pws` does an explicit `to_memory_config(a, width_sharded_l1_config(...))` per matmul
(5×/layer) where `matmul_decode` resharded inside its reader; `nlp_concat_heads` produces a real
tensor where `concat_heads_matmul_decode` used a free view; plus separate `gate`/`up`/`hid`. With
2–3 layers pinned per chip, L1 overflows. **Re-fusion is a hard requirement for 16-chip L1 to fit,
not a perf nicety.**

NOTE: `TILE_HEIGHT=32` is NOT a pre-merge control — it runs *their* block at 32×32. Only the
pre-merge tree is a true control, which is why it had to be built.

#### Tiny-tile gaps found in the multi-stage path (all fixed here)

The source branch's only model test builds ONE stage with `suffix=None`, `is_last=False`, via
`build_single_stage_reference`. **The multi-stage `build_n_stage_pipeline` path — the one the 16-chip
pipeline uses — was never run at 16×32 upstream**, so everything outside that subset was unexercised:

1. `_bind_prefix_kv` swept to `from_torch_pi05` (would put prefix KV on the tiny tile); their own
   test hand-injects 32×32 to work around it. Kept prefix KV at 32×32.
2. `_build_stages`: `assert suffix_len % 32 == 0` → `% TILE_HEIGHT` (2 sites).
3. `_linear_weight_to_tt` defaulted to the model tile, so 5 weights feeding plain `ttnn.linear` got
   16×32 (`_tt_final_mod_w` + all 4 suffix-embedding weights) → `in1_tile.get_height() == TILE_WIDTH`.
   Inverted the default to 32×32.
4. `pipeline_16_decode._action_horizon_padded` hardcoded `((ah+31)//32)*32` = 32 while the stages
   were told `perf_suffix_len` = 16 → `kv_sdpa: query length must be exactly one tile (16); got 32`.
5. The 4 suffix-embedding linears passed no program_config/core_grid → generic MatmulMultiCore, the
   only factory that rejects a tiny outer tile. Supply `core_grid` when the activation is tiny.
6. Still open: `The last two dimensions of the first tensor and the last dimension of the second
   tensor must be a multiple of tile size`.

Stage 7 is deferred until after stage 8 (re-fusion), since re-fusion is required for L1 anyway and
removes most of the unfused/generic-matmul call sites generating these gaps.

#### Environment hazards worth recording

- `PYTHONPATH` cannot override an editable install: the venv's `__editable__.ttnn-*.pth` registers a
  `sys.meta_path` finder, which wins over `sys.path`. A second build tree needs the finder stripped
  (see `run_isolated.py`) or it silently tests the wrong build.
- A second build tree must pin system Python — `Python3_EXECUTABLE=/usr/bin/python3`,
  `Python3_INCLUDE_DIR=/usr/include/python3.10`,
  `Python3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so` — or `_ttnn.so` gets
  `undefined symbol: PyObject_Vectorcall`.
- Worktrees share `.git/modules`: running `git submodule update --init` in one **rewinds the other's**
  submodules (it moved `tt-cluster-descriptors` in the main tree, which governs Galaxy cluster
  descriptors). Re-check `git submodule status --recursive` afterwards; plain `git status` hid it.

### Stage 8 — re-fusion onto our ops: DONE, and tiny tile is a win

`test_l1_single_layer_pcc` **PCC 0.9999 at both tile heights** with everything fused.

Most of the expected work was already done: **our `matmul_decode` family was already tiny-tile
aware** — all three factories derive inputA/inputB/output tile heights from the tensors, compute
`M_tiles = div_up(M, inputA_tile_height)`, and require only that inputB stays 32. So re-fusion was
mostly Python rewiring.

L1 single-layer traced replay, median of 3 paired runs (`test_walltime_l1_single_layer`):

| configuration | tile-32 | tile-16 |
|---|---|---|
| unfused (their block) | 0.2130 ms | 0.2010 ms |
| fused matmuls, unfused rope | 0.1530 ms | 0.1690 ms |
| **fully fused (incl. rope)** | **0.1510 ms** | **0.1380 ms** |

Two conclusions, in order of size:
1. **Fusion is the big win: 0.213 → 0.151 ms at tile-32 (−29%).**
2. **Tiny tile is a genuine −8.6% on top of that** (0.1380 vs 0.1510; ranges 0.132–0.139 vs
   0.149–0.152 do not overlap) — **but only once everything is fused.** On the fused-matmul/unfused-rope
   path tile-16 was *slower*, purely because it paid 3 rope dispatches to tile-32's 1.

Beware two earlier mis-measurements in this file's history: a "0.1825 ms baseline" that was really
tile-32 with fused rope + UNFUSED matmuls (not the pre-merge config), and a "tile-16 is 5.6% faster"
result measured on the fully-unfused path, which does not carry to the fused path. **Any tile-height
A/B must hold the fusion state identical on both sides.**

#### Changes

- `_matmul_decode_pws` prefers `ttnn.matmul_decode(reshard_input=True, compute_kernel_config=_LOFI,
  residual=, gate=)`; the explicit `to_memory_config` pre-reshard of `hidden_states` is skipped on that
  path (that materialized L1 copy is the 16-chip L1 pressure).
- `gate_up_matmul_decode` fusion (also drops a `sharded_to_interleaved`),
  `concat_heads_matmul_decode` free-view concat, and `_fuse_mlp_residual`/`_fuse_attn_residual`
  re-enabled — all gated on `_FUSED_MD` so the experimental-op fallback still works.
- `concat_heads_matmul{,_decode}`: validate `seq` against the operand's own tile height (and round the
  output spec to it). This also clears one of the two DRAM-leg blockers.
- `nlp_create_qkv_heads_rope`: full tiny-tile checklist — tile-aware `Ht == 1` validation, CBs sized
  from each tensor's own tile plus `set_tile_dims`, tile dims added to `compute_program_hash`, and
  **`PageConfig(qkv.layout())` → `qkv.tensor_spec().page_config()`** (the bare form drops the tile and
  undersizes the outputs against the CBs — their doc's checklist item 2).

#### New ttnn gaps found

- **`ttnn.repeat` does not preserve a tiny tile**: a `(16,32)` input yields a `(32,32)` output
  (verified directly). Not in the tiny-tile op list; same class as the addcmul promotion. Worked
  around in `_build_fused_gate_ws` with a precompute-time retile (zero per-replay cost).
  **Report upstream** with the `QK_COL_VECTOR_MODE` and bf8-mask bugs.
- `_build_fused_gate_ws` itself hardcoded a 32-row replicate; now uses `TILE_HEIGHT`.

### Stage 7 (revisited after re-fusion) — 16-chip path GREEN at both tile heights

`test_perf_16_socket_traced_2cq`, `PERF_ITERS=20`:

| 16-chip e2e | pre-merge (fused) | tile-32 | **tile-16** |
|---|---|---|---|
| 2 cams | 25.31 ms | 27.21 ms | **25.80 ms** |
| 3 cams | 28.52 ms | 30.33 ms | **28.79 ms** |

- **Tiny tile is −5.1%/−5.2% vs tile-32 on the same build** (consistent with the −8.6% single-layer
  result, diluted by the unchanged vision/prefill stages).
- **tile-16 reaches rough parity with pre-merge** (+1.9% / +0.9%), i.e. tiny tile offsets an
  as-yet-unexplained ~7% regression that the tile-32 build carries vs pre-merge.

**RESOLVED (mostly): the ~7% tile-32 vs pre-merge regression was kv_sdpa math fidelity.** The re-fused
block kept the tiny-tile branch's `get_sdpa_compute_kernel_config()` (BAKED **HiFi4**) where pre-merge
used `_KV_SDPA_HIFI2` (**HiFi2**). kv_sdpa's fp32 dest accumulation is un-gated, so fidelity is the only
effective knob. Restoring HiFi2 on both kv_sdpa call sites (the general-SDPA fallback stays HiFi4, as
pre-merge) recovered 1.05 ms / 1.30 ms — the 3-cam figure matching pre-merge's "~1.3 ms" note exactly.
PCC unchanged at 0.9999.

| 16-chip e2e | pre-merge | tile-32 HiFi4 | tile-32 HiFi2 | **tile-16 HiFi2** |
|---|---|---|---|---|
| 2 cams | 25.31 ms | 27.21 | 26.16 | **25.10** |
| 3 cams | 28.52 ms | 30.33 | 29.03 | **27.92** |

**tile-16 now beats pre-merge by 0.8% / 2.1%**, and is 3.8–4.1% faster than tile-32 on the same build.

Ruled out: the kv_sdpa chunk cap (`_KV_SDPA_MAX_CHUNK_TILES = 32` reproduces pre-merge's effective
4-tile prefix chunk exactly) and `ttnn.repeat` (`_build_fused_gate_ws` is reached only from
`precompute_mods`, which runs at BUILD time, so repeat and its retile cost zero per replay).

**Still open: a residual +1.8%/+3.4% on tile-32 vs pre-merge.** Deprioritized because the shipping
config (tile-16) is now ahead of pre-merge. Suspects remain main drift or a subtler wiring difference.

Two more blockers had to be cleared beyond re-fusion:

1. **kv_sdpa L1 footprint.** Their chunk picker maximizes chunk size to minimize per-chunk overhead;
   at `DH=256` (`DHt=8`) `max_chunk_tiles = 128/8 = 16` gives 256-tile prefix K/V CBs, ~272 KB EACH
   (~544 KB together) vs ~136 KB for our pre-merge `{4,3,2,…}` picker. That **+408 KB** is exactly the
   `Statically allocated circular buffers ... clash with L1 buffers` failure (observed CB region
   638 KB, and it grew with camera count as the longer prefix changed the divisor). Added a
   `max_kv_chunk_tiles` parameter (public API → prim → attributes → factory → nanobind), **default 128
   so single-chip behaviour is unchanged**; the denoise block passes 32. The attributes struct is
   hashed by default, so cap variants cannot collide in the program cache.
2. **The suffix-embedding linears.** An earlier fix here was wrong: passing only `core_grid` routes to
   the 1D-systolic AUTO config generator, which computes
   `m_tiles = (batch * M) / ttnn::TILE_SIZE` against the global 32 — so M=16 fails its
   "must be a multiple of tile size" check and would yield `m_tiles == 0`. Trading the generic
   MatmulMultiCore factory for that generator swapped one non-tile-aware path for another. Now builds
   the 1D-mcast `program_config` explicitly with `m_tiles = div_up(M, tile_h)` via `matmul_pcfg`; the
   matmul FACTORY is tile-aware, only the auto-generator is not. Returns `{}` at tile-32 so that path
   stays byte-identical.

Eight tiny-tile gaps were found in the multi-stage path in total, none of them reachable from the
source branch's single-stage reference test.

## Conflict resolution notes

All 18 resolved as follows.

- **`kv_sdpa` (6 files)** — took **theirs** wholesale. Theirs is strictly ahead: it already carries
  the improved `Sk_chunk_t` picker (largest divisor of `Kt` with `Sk_chunk_t*DHt <= 128`, lifted into
  a `pick_chunk` lambda applied separately to `suffix_Kt` and `prefix_Kt` for the two-source path),
  plus tiny-tile `QK_NUM_FACES`, bf8, and the newer-main `sources.cmake` convention. Our HEAD had the
  *older* `{4,3,2,5,6,7,8}` list.
- **`nlp_concat_heads` / `nlp_create_qkv_heads` factories** — combined: their tile-aware
  `input_tile_height` divisor **plus** our `head_split` / `mqa_split` block multipliers.
- **`ttnn_gemma.py`** — took **ours**. All 8 hunks were our unclamped-grid tuning and direct
  `ttnn.experimental.*` / `ttnn.kv_sdpa` calls vs their `_ttnn_compat` wrappers; the tiny-tile branch
  added no tile awareness to this file.
- **`modeling/{gemma,bs,common,suffix}.py`** — took **theirs**; we had never diverged from their
  baseline in these, so their tiny-tile delta applies cleanly. Deduped their doubled `tile_config`
  import in `common.py`.
- **`modeling/pcfg.py`** — took **ours**. The tiny-tile branch never touched this file; the
  `min(grid_y, ...)` clamp is the original and we removed it deliberately. Taking "theirs" would have
  silently reverted our own tuning change.
- **`tt_pipeline/__init__.py`**, **`_d2d_pipeline.py`** — took **ours**. Their lazy `__getattr__`
  shim and missing `prologue_fn` were both artifacts of `stage_denoise.py` being absent on their
  branch; the 16-chip socket path needs `prologue_fn` for the in-trace KV recv.
- **`denoise_block.py`** — took theirs as the base (per the correctness-first decision), then
  re-applied our orthogonal changes: unclamped tuned-pcfg grid; `DECODE_ALL = False` module default
  (every caller sets it); restored the real `_decode_all_active()` probe over their unconditional
  `return True`; removed the four hot-path `print()`s; kept `_KV_SDPA_HIFI2` defined but unwired
  pending the Stage-6 fidelity A/B.
- **`denoise_pipeline.py`** — took **ours** as the base (ours is the superset: `refresh_prefix_kv`,
  `_layer_lo`, and the `build_eager`/`step`/`capture(prologue_fn)`/`reseed_noise` single-root capture
  the 16-chip pipeline depends on), then ported their two changes in: `perf_suffix_len(ah,
  tile_height=TILE_HEIGHT)` and the `from_torch_pi05` uploads. **Deliberately kept prefix KV on the
  default 32×32 tile** — their mechanical sweep had converted `_bind_prefix_kv` to `from_torch_pi05`,
  which their own reference test then works around by hand-injecting 32×32 tensors.
- **`_ttnn_compat.py`** — extended `decode_all_supported()` to also accept
  `ttnn.experimental.matmul_decode`, since that is the op the tiny-tile block actually dispatches
  (the old probe checked `ttnn.matmul_decode` + `gate_up_matmul_decode`, which the tiny-tile path no
  longer calls).

## Defects to fix (block verification)

1. `tt/ttnn_gemma.py:932` — `assert q_rope.shape[-2] == 32` hard-fails at `Sq=16`, breaking the DRAM
   leg of the reference test. Make tile-aware.
2. `tt/tt_pipeline/denoise_block.py:110` — `_decode_all_active()` has an unconditional `return True`
   above the real logic, dead-coding the `decode_all_supported()` probe.
3. `denoise_block.py:329-343` — four `print()`s inside the traced hot path.
4. `tests/ttnn/utils_for_testing.py::select_tile` — returns `Tile((16,32))` for every TILE case
   regardless of dtype; `None` for row-major; leftover `print()`s. Diverges from its own design doc.
5. `tests/test_tiny_tile_ttnn_bugs.py:148` — passes `dtype=` to a `_to_dev` with no such param.

## Noted, NOT fixed here (pre-existing / out of scope)

- Dead code: `_to_tile32_bf8`, `_build_fused_gate_ws`, `_ttnn_compat.concat_heads_matmul_decode`.
- `compute_padded_shape()` always pads to 32×32; worked around by a duplicated
  `compute_padded_shape_for_tile()` in both `pad.cpp` and `reshape.cpp`.
- `fill_implicit_tile_padding` assumes 32×32 when deciding whether implicit pad exists.
- Contradictory comments: `test_denoise_single_layer_l1_vs_dram.py:158-161` (prefix KV tile),
  `test_tiny_tile_ttnn_bugs.py:97-98` (claims addcmul avoided; `_gated_residual` uses it),
  `test_tiny_tile_ttnn_bugs.py:10` vs `tile_config.py:13` (blocked dtypes at tiny tile).
- Duplicated import at `modeling/common.py:19-20`.

---

# Post-integration optimization rounds (R6-R9)

Baseline for this section: tile-16 single-layer denoise **0.116-0.117 ms** (K=4 / 16-16-16,
`_KV_SDPA_MAX_CHUNK_TILES=64`, LoFi kv_sdpa, block-sharded adaRMS), i.e. -42% from the 0.201 ms the
merge originally landed at. 16-chip e2e 24.25 ms @2 cams / 26.76 ms @3 cams.

## R6 - config space converged (no change)

`_KV_SDPA_MAX_CHUNK_TILES` re-swept at the K=4 operating point: bf8 prefix cap 64 -> 0.117,
96 -> 0.120, 128 -> 0.118; bf4 prefix cap 64 -> 0.116, 128 -> 0.118, 256 -> 0.126. bf4's 0.116 vs
bf8's 0.117 is inside the +-0.002-0.003 run-to-run band, so bf4 stays available but unused. Cap 64
remains optimal. **The matmul_decode / kv_sdpa scalar config space is exhausted** -- further gains
need structural change.

## R7 - per-matmul K-split for the MLP: NEGATIVE, do not retry

Profiling the tuned tile-16 config against tile-32 showed exactly ONE op regressed while every other
improved: `GateUpMatmulDecode` **+17%** (10.67 -> 12.49 us). Cause is real: a global `_K_BLOCKS=4`
caps the MLP at `n_blocks=16`, because `down`'s `N_tiles=32` has no divisor in (16, 30] and
`k_blocks*n_blocks <= 120`. So the MLP loses half its output-block parallelism.

Threaded a per-weight `k_blocks` through `_pws_B` and swept the MLP alone:

| MLP (K, n) | PCC | single-layer |
|---|---|---|
| (4, 16) = current | 0.9999 | **0.116 ms** |
| (2, 32) = tile-32's layout | 0.9999 | 0.121 ms (+4%) |
| (1, 32) | **FAILS** | 0.118 ms |

**Lesson: a per-op regression in a profile diff is not independently actionable when the config is
globally coupled.** K=4 trades gate_up slower for QKV/O/kv_sdpa faster and nets -3.3% overall. The
`k_blocks` plumbing was reverted (dead flexibility); the finding is recorded in `denoise_block.py`.

## R8/R9 - kv_sdpa split-KV (flash decode): the real structural headroom

`kv_sdpa` is **33% of denoise device time** (28.4 us/call) and its program factory assigns
`num_cores_to_corerangeset(NQH)` = **one core per Q head = 8 of ~120 cores**. Because `Sq` is a
single tile there is NO Q-parallelism to exploit, so the KV axis is the only one left. Per-core rate
is ~0.59 TFLOP/s, a few percent of Blackhole bf8 peak.

Sizing the prize from the existing cap sweep (which varied chunk count at constant work): 8 chunks
0.131 vs 2 chunks 0.127 => ~0.7 us fixed cost per chunk, so at 4 chunks only ~2.8 of the 28.4 us is
per-chunk overhead and **~88% is parallelizable work**.

Implemented `kv_splits=S` (`PI05_KV_SPLITS` env knob): `NQH*S` cores, split `s` of head `h` takes
prefix tiles `[s*prefix_Kt/S, (s+1)*prefix_Kt/S)`; split 0 also takes the (much shorter) suffix and
acts as reducer, merging the S-1 partial `(max, sum, out)` states with the standard online-softmax
correction. Reuses `correction_block()` from `compute_common.hpp`. **Flat** reduction, not the
`sdpa_decode` tree: S is small and `Sq_chunk_t == 1`, so one semaphore + per-child slots replaces the
nibble-encoded tree topology. Each core `matmul_reduce`s its partial sum to a true column vector
before merging (the flash loop leaves it as a partial row-sum).

Two costs found that are specific to this pipeline:

1. **The reduction CBs are not free on the S=1 path.** Declaring them at full size unconditionally
   added ~31 bf16 tiles (~62 KB) of L1 per core and moved S=1 from 0.116 to **0.129 ms**. They are
   now a 1-tile placeholder when `num_children == 0`.
2. **kv_sdpa's cores are the resident-weight cores.** Both use
   `num_cores_to_corerangeset(n, grid, row_wise=true)`, so kv_sdpa's 8 cores are exactly the first 8
   of the 64 `_pws_B` weight cores. Raising S spreads the prefix K/V CBs (~278 KB/core at cap 64)
   across MORE weight-holding cores, so **S is gated on the 16-chip L1 budget, not just on speed** --
   the same coupling that made K=4 unusable at tile-32. Expect to lower `max_kv_chunk_tiles` as S
   rises (each split has fewer prefix tiles anyway).

First measurements (tile-16, single-layer, machinery present at every S):

| kv_splits | cores | single-layer |
|---|---|---|
| 1 | 8 | 0.129 ms |
| 2 | 16 | 0.121 ms |

**SUPERSEDED -- do not read this table as a result.** It looked like "the split works, -6% from S=1 to
S=2, and CB overhead was masking it". Both readings were wrong: the CB guard changed nothing, and the
apparent gain was an artifact of a ~12% runtime-arg penalty that BOTH arms were paying. See "VERDICT"
and "THE KEY FINDING" below for what actually happened.

## Harness note: the standalone kv_sdpa microbenchmark is NOT a valid oracle

`scratchpad/kvs_bench.py` compares the op against a torch SDPA reference at the pi0.5 shape. It
reports PCC ~0.38 for **stock** kv_sdpa across all four dtype/tile combinations (bf8/bf16 x
tile16/tile32), so the torch reference does not match the op's contract -- do not read correctness or
timing from it. (Separately, bf8 + tile16 there produces `absmax 3.4e38` garbage, while bf16 + tile16
is sane; the model uses bf8+tile16 successfully, so that is harness-specific and unexplained.)
**Validate kv_sdpa changes in-situ** with `test_l1_single_layer_pcc` + `test_walltime_l1_single_layer`.

## The governing structural fact: half the denoise runs on <10% of the cores

Because the denoise Q is a **single tile** (16 rows), every per-head op degenerates to one core per
head, and they are allocated by head count, not by work:

| op | core count | how it is derived | share of device time |
|---|---|---|---|
| `kv_sdpa` | **8** | `num_cores_to_corerangeset(NQH)` | 33.2% |
| `LayerNorm` (adaRMS) | **8** | `sharded_norm_pcfg(..., max_grid_y=min(8, m_tiles))`, `m_tiles == 1` | 9.4% |
| `NlpCreateQkvHeadsRope` | **10** | `total = nq + 2*nkv` = 8+1+1 | 6.8% |
| `matmul_decode` / `gate_up` | 64 | `_K_BLOCKS * n_blocks` | 43.2% |

So **~49% of denoise device time runs on <= 10 of ~120 cores.** Only the projection matmuls are
actually parallel. This is why the scalar config sweeps converged: they were tuning the 43% that is
already spread over 64 cores, while the 49% on 10 cores was untouched.

The remaining headroom is therefore all of one kind -- **find a second parallel axis for the
single-Q-tile ops**:

* `kv_sdpa` -> split the KV sequence (implemented as `kv_splits`, R8/R9 above). Only axis available.
* `NlpCreateQkvHeadsRope` -> split along `head_dim` (256 = 8 tiles), giving 10*8 = 80 cores. RoPE is
  a per-dim-pair rotation, so it partitions cleanly; the reader would need a dim-tile offset.
* `LayerNorm` -> cannot split width without a cross-core RMS reduction (measured slower, see
  `sharded_rms_norm`). The real win is fusing the reshard + norm into the consumer matmul's reader,
  which would also remove the 2 `InterleavedToSharded` per layer (64 calls, 39.6 us, 1.4%) that come
  from the norm's own `to_memory_config(x, memcfg)`.

Note all three small ops use the SAME `num_cores_to_corerangeset(n, grid, row_wise=true)` allocator as
the resident weights, so they always land on the first N weight-holding cores. Any core-count increase
is therefore also an L1 decision, not just a scheduling one.

### R9/R11 - the dominant cost was runtime args, not L1

The first split-KV implementation regressed the **inert** `kv_splits == 1` path from 0.116 to
0.129-0.131 ms (+12%), while PCC stayed 0.9999 at every S. Two hypotheses, tested in order:

1. **Extra CBs (wrong).** The reduction CBs were declared full-size unconditionally (~31 bf16 tiles,
   ~31 KB/core). Shrinking them to a 1-tile placeholder when `num_children == 0` changed nothing
   (0.129 -> 0.131, inside noise). Not the cause -- but the guard is correct and was kept.
2. **Runtime args (right).** Passing the per-core values (`suffix_num_chunks`, `is_reducer`,
   `num_children`) as *runtime* args made the suffix flash loop runtime-bounded, so it lost
   constexpr bounds and constant propagation into `flash_accumulate_chunk`. Fixed by emitting **two
   kernel variants** on two core sets -- reducer cores (split 0) and worker cores -- so every
   role-dependent value is a compile-time arg again. Only truly per-core values (prefix slice offset,
   output row, reducer NOC coords, slot index) remain runtime args. At `kv_splits == 1` only the
   reducer variant is emitted, with args identical to the pre-split program. This is the same
   structure `nlp_create_qkv_heads_rope` uses for `qk_cores` / `v_cores`.

**Lesson: on a kernel this small, moving a loop bound from compile-time to runtime cost more (12%)
than anything else attempted in this round gained.** Prefer multiple kernel variants over runtime
predication whenever the variants are known at program-build time.

Also note the barrier batching (3 per-child NOC round trips -> 1) made **no measurable difference**
(S=2 was 0.121 both before and after), so the cross-core hop is not the reduction's bottleneck; the
serial merge chain is.

Split gain, measured consistently across two rounds at equal handicap: **S=2 is 0.008-0.010 ms faster
than S=1**, i.e. kv_sdpa ~28.4 -> ~20 us. S=4 and S=8 are progressively worse (0.127, 0.134), an
interior optimum at S=2 -- consistent with the merge chain lengthening and the prefix CBs spreading
onto more weight-holding cores.

### VERDICT: split-KV does NOT pay at the pi0.5 denoise shape

With the machinery implemented correctly (compile-time per-core role, two kernel variants), the
single-layer denoise at tile-16, PCC 0.9999 at every S:

| kv_splits | cores | single-layer | vs S=1 |
|---|---|---|---|
| 1 | 8 | **0.117 ms** | baseline (== the pre-split 0.116-0.117, so the machinery is FREE when off) |
| 2 | 16 | 0.133 ms | +14% |
| 4 | 32 | 0.125 ms | +7% |

**The earlier "S=2 is 8-10 us faster" result was an artifact.** In the runtime-arg version BOTH arms
paid a ~12% penalty that scaled with the reducer's own chunk count, so halving that work looked like a
win. Once the penalty was removed the split is uniformly a loss. This is the second time this round
that an A/B was invalidated by the two arms not being otherwise identical (the first was the tile-16
vs tile-32 fusion-state mismatch) -- **when a change moves a shared cost, re-measure the baseline in
the SAME build before believing a delta.**

Why it loses: halving each core's prefix work should save ~12 us of the ~25 us of chunk work, but S=2
is 16 us *slower*, implying the reduction costs ~28 us. The merge math is only ~5-6 us of tile ops
(one `correction_block` on 1 tile, two 8-tile bcast-muls, an 8-tile add, ~11 tile copies), and the
walltime harness is TRACED (`begin_trace_capture`/`execute_trace`), so this is real device time, not
host dispatch. The remainder is the reducer *waiting* -- it cannot start merging until its slowest
child has finished its whole flash loop, staged its partials, and completed a NOC hop plus a semaphore
round trip. At this size the barrier costs more than the parallelism buys. (Batching the three
per-child NOC reads behind one barrier changed nothing, confirming the hop itself is not the cost.)

**Implication for the other single-Q-tile ops:** the "spread it over more cores" thesis is NOT
automatically a win here. A split only pays if the per-core work saved exceeds a cross-core barrier,
and at Sq == 1 tile the per-core work is already tiny. The rope split (10 -> 40 cores) needs no
cross-core reduction at all (RoPE is elementwise per dim-pair, so each core just writes its own output
slice) -- that is the one worth trying, precisely because it has no merge step. LayerNorm's width split
DOES need a cross-core RMS reduction and was already measured slower, which is the same lesson.

`kv_splits` is retained at its default of 1, where it is measurably free, so the capability is
available for shapes with much longer prefixes (where the per-core saving would dominate the barrier).

## THE KEY FINDING: the denoise block is DISPATCH-bound, not compute-bound

Profiling `kv_splits=1` vs `2` (device kernel time, tile-16, same build) settles what the walltime
number alone could not:

| | S=1 | S=2 | delta |
|---|---|---|---|
| `KvSdpa` device kernel | 28.47 us/call | **20.75 us/call** | **-27.1%** |
| total device kernel | 2741.5 us | **2456.5 us** | **-10.4%** |
| traced wall-clock (single layer) | 0.117 ms | 0.133 ms | **+14%** |

Every other op is unchanged (all within +-0.5%), so the split did exactly what it was designed to do.
**Device kernel time went DOWN 10% while wall-clock went UP 14%**, which means the cost is not kernel
execution.

Do the arithmetic: 2741.5 us of kernel time over 31 block iterations = **88.4 us/iteration of kernel
time against 117 us of measured traced wall-clock** -- so **~29 us (25%) per iteration is already
non-kernel overhead**, roughly 2.6 us across the ~11 ops in the block. At S=2 that overhead grows to
~54 us/iteration. Adding 8 cores to ONE op cost ~25 us of launch/sync, swamping the 7.7 us of kernel
time it saved.

**The ~3.7 us/op is not a mesh-fanout artifact.** `test_walltime_l1_single_layer` is parametrized
`@pytest.mark.parametrize("mesh_device", [1])`, i.e. a SINGLE-device mesh, so per-op launch cost is not
being multiplied across chips. That is also exactly the per-chip situation in the 16-chip pipeline
(each chip runs this same denoise block on its own submesh), so the figure transfers.

**Do not over-model the overhead as "proportional to core count"** -- S=4 (32 cores, 0.125 ms) beats
S=2 (16 cores, 0.133 ms). At S=4 each core gets exactly ONE prefix chunk, so `processed == 0` on its
only chunk and the entire online-softmax rescale block is skipped, which is a real kernel-side saving
pulling the other way. The robust claim is narrower and still decisive: **a 10% cut in device kernel
time produced a 14% INCREASE in traced wall-clock, and ~25% of wall-clock is not kernel time at all.**

This retroactively explains three earlier results that were recorded as bare facts with no mechanism:

* "64 cores beats 80 (QKV=16 over QKV=20)" -- fewer cores, less launch overhead.
* "`_RESHARD_CORES=2` beats 4 and 8" -- same reason.
* Every scalar config sweep converging (R6) -- they were all tuning kernel time, which is only 75%
  of the cost, and the knobs that mattered were the ones that quietly reduced core count.

**Consequences for what to optimize next.** The lever is FEWER OPS AND FEWER CORES, not faster
kernels:

1. **Op fusion is now the top priority**, not parallelization. Each op removed is worth ~2.6 us of
   pure overhead on top of whatever kernel time it held. The queued candidates are exactly right:
   fuse the adaRMS norm + its reshard into the consumer matmul's reader (removes 2 `LayerNorm` +
   2 `InterleavedToSharded` per layer = 4 of ~11 ops, ~10 us/iteration of overhead alone, plus the
   258 us of LayerNorm kernel time).
2. **The rope head_dim split (task 21) should NOT be pursued.** It would take 10 -> 40 cores on an op
   whose kernel time is only 5.76 us. Even a perfect 4x kernel speedup saves ~4 us while adding ~30
   cores of launch overhead. The barrier-free argument for it is irrelevant if the cost is launch,
   not synchronization.
3. `kv_splits` stays default 1. It is a real -27% on `kv_sdpa`'s kernel time and will pay on a
   platform or pipeline where per-op launch overhead is amortized better (or at a much longer prefix,
   where the kernel saving scales while the launch cost does not -- see the prefix crossover sweep).

**Methodological note:** kernel-time profiling and traced wall-clock disagreed in SIGN here. Neither
alone was sufficient: the walltime said "the split fails", the profile said "the split works". Both
are true, and only together do they identify dispatch as the bottleneck. Always check both.

## Next optimization, scoped: fuse the adaRMS norm into the consumer matmul_decode

This is the right target *because* the block is dispatch-bound. It REMOVES ops instead of adding cores.

**Why it is feasible (and why it does not hit the barrier that killed `kv_splits`):**
`reader_partial_width_sharded.cpp` calls `gather_full_a()`, so the ENTIRE A row is already gathered
onto every compute core. `sum(x^2)` over `hidden` is therefore computable **locally per core with no
cross-core reduction** -- the exact cost that made split-KV lose does not exist here.

**Insertion point** is unambiguous: in `compute_partial_width_sharded.cpp`, between
`cb_wait_front(full_in0_cb_id, full_in0_num_tiles)` and `phase1_partial(...)`. The kernel already has a
compile-time-gated epilogue framework (`fused_gelu`, `fused_residual`, `residual_cb_id`, `gate_cb_id`,
`mmg_cb_id` at CTAs 6-12) to model a prologue on, and `phase1_partial` is shared with the gate_up
compute so the prologue is written once.

**No redundant normalization at the op level:** each norm feeds exactly ONE consumer -- attention's
`normed` -> the wqkv `matmul_decode`, the MLP's `normed` -> `gate_up_matmul_decode`. So only two ops
need the prologue (the o-proj and down-proj do not).

**But note the per-core redundancy.** `full_in0_num_tiles = M_tiles * K_tiles` = 32 tiles, and all
`K_blocks * n_blocks` = 64 cores hold the full A, so every core recomputes the same norm -- ~32 tile-ops
each, which is MORE than the 16 tile-matmuls a core does for its own partial. That is acceptable only
because the cores run in parallel: the op's *duration* grows by roughly one core's share (~1 us), not
64x. This is the main thing to verify empirically rather than assume.

**Sizing it from the per-op profile.** Per block iteration (31 iterations in the profile) the ops and
their kernel time are:

| op | calls/iter | us/call | us/iter |
|---|---|---|---|
| `KvSdpa` | 1 | 28.47 | 28.5 |
| `MatmulDecode` | 3 | 8.31 | 24.9 |
| `GateUpMatmulDecode` | 1 | 12.12 | 12.1 |
| `LayerNorm` | 2 | 4.01 | 8.0 |
| `NlpCreateQkvHeadsRope` | 1 | 5.76 | 5.8 |
| `InterleavedToSharded` | 2 | ~0.62 | 1.2 |
| **sum of kernel time** | **10** | | **~80.5** |
| **measured traced wall-clock** | | | **117** |

So **~36.5 us (31%) of wall-clock is inter-op overhead, ~3.7 us per op.** That is the real currency.

**Expected for this fusion:** removing 4 of the 10 ops saves 4 x 3.7 = ~15 us of overhead plus 8.0 us
(LayerNorm) + 1.2 us (i2s) of kernel time = **~24 us on a 117 us block, ~20%** -- against ~2 us of added
prologue. That is the largest single win identified, and it is larger than everything this round
achieved combined. Gate it on BOTH device kernel time and traced wall-clock.

**Rejected as insufficient:** simply keeping the residual stream sharded to kill the 2 i2s (~8.6 us,
7%) does not work on its own -- `matmul_decode`'s natural output is width-sharded across its `n_blocks`
base cores while the norm wants an 8-core block-shard, so a reshard is required either way. The
layouts only reconcile inside the fused op.

### CAUTION on the fusion: the prologue is REDUNDANT across cores, and that may sink it

Correcting the estimate above before anyone writes the kernel. Two things make the prologue more
expensive than "one extra pass":

1. `compute_kernel_lib::reduce<SUM, REDUCE_ROW, ...>` sums `x`, not `x^2`, so RMS needs a **square pass
   into scratch** and then the reduce -- the production `layernorm_sharded.cpp` does exactly this.
2. **It is recomputed on every core.** The standalone `LayerNorm` op normalizes the 32 A-tiles ONCE
   spread over 8 cores (4 tiles/core). A fused prologue runs on all `K_blocks * n_blocks` = 64 cores,
   and each must square-and-reduce all 32 *gathered* tiles to get `sum(x^2)` over the full hidden dim.
   Each core then only needs to SCALE its own `Kc_tiles` = 8-tile slice (`phase1_partial` takes a
   `k_offset` and matmuls only its slice), so it is ~32 square + ~32 reduce + ~24 scale/weight/bias
   ~= 56 tile-ops per core versus the 4 tiles/core the standalone norm does.

So the fused version does roughly **14x the per-core normalization work** of the op it replaces. Whether
it still wins depends entirely on the marginal per-tile cost of eltwise/SFPU work inside
`matmul_decode`, which is NOT known -- the honest range is "+2 us (fusion wins big)" to "+15 us per op
(fusion loses)".

**Do the cheap probe before building it.** Add a throwaway prologue to
`compute_partial_width_sharded.cpp` that squares `full_in0` into scratch and reduces it (no
correctness, no weight/bias, output discarded), then profile `MatmulDecodeDeviceOperation`'s
device-kernel time against the 8.31 us baseline. ~30 lines and one profile run decides a multi-hour
implementation:

* delta < ~2 us/op -> build the real fusion; expected net ~ -16 to -24 us on a 117 us block.
* delta > ~6 us/op -> abandon; removing 4 ops (~15 us of dispatch + 9 us of kernel) will not cover it.

If it is abandoned, the fallback for the dispatch-bound problem is to look for op removals that add NO
per-core work -- e.g. folding the two `InterleavedToSharded` into an existing reader, or reducing the
op count in the adaRMS modulation path -- rather than moving compute into a 64-core op.

#### Grounding the prologue cost from measured per-tile rates (before doing the probe)

The profile already contains two clean anchors for per-tile cost on this device at this shape:

* `InterleavedToSharded`: 0.62 us for ~32 tiles => **~0.02 us/tile** of pure data movement.
* `BinaryNgDeviceOperation`: 1.43 us for ~32 tiles => **~0.045 us/tile** of eltwise compute.

At ~56 tile-ops per core the prologue is therefore ~56 x 0.045 = **~2.5 us per fused op**, not the
~15 us worst case feared above. Two fused ops (wqkv + gate_up) => **+5 us**.

Against that, the removals are 8.0 us (`LayerNorm` kernel) + 1.2 us (`i2s` kernel) + 4 ops x 3.7 us
(dispatch overhead) = **~24 us**. Net **~ -19 us on a 117 us block, ~ -16%**.

So the fusion is very likely a win, and the probe is now optional rather than gating -- the residual
uncertainty is only whether the `reduce` and the single `rsqrt` are materially pricier per tile than a
plain binary eltwise. Recommend implementing it directly, in this order (each step independently
verifiable with `test_l1_single_layer_pcc` + walltime):

1. wqkv only (`matmul_decode`): removes 1 `LayerNorm` + 1 `i2s`, expect ~ -10 us / -8%.
2. then the MLP (`gate_up_matmul_decode`), which shares `partial_phases.hpp`: the other ~ -10 us.

Keep the fused-norm path behind a compile-time flag defaulting OFF so the existing path is byte
-identical until the A/B passes -- and remember runtime args are NOT free here (R9/R11: they cost 12%).

---

# RETRACTION: the "dispatch-bound, ~3.7 us per op" conclusion is FALSIFIED

The section above concluded, from the `kv_splits` experiment, that this block is dispatch-bound at
~3.7 us per op and therefore that **removing** an op should be worth ~3.7 us on top of its kernel time.
That prediction was tested directly by implementing the norm fusion it recommended, and it **failed**.

The fusion (branch `pi05_fuse_norm_matmul`, commit `b3999234ea0`, `PI05_FUSE_NORM=1`) is correct --
`test_l1_single_layer_pcc` passes at PCC 0.9999 -- and it removes exactly the ops it was designed to
remove, confirmed by profile op counts per block iteration:

| | ops/iter | LayerNorm calls | InterleavedToSharded calls |
|---|---|---|---|
| fusion off | 322 | 64 | 66 |
| fusion on | 258 | 32 | 34 |

Two ops per iteration gone. And yet:

| per iteration | kernel time | traced wall-clock | non-kernel |
|---|---|---|---|
| fusion off | 88.2 us | 116.5 us | 28.3 us |
| fusion on | 91.3 us | 123.0 us | **31.7 us** |

Non-kernel time went **UP** 28.3 -> 31.7 us despite two fewer dispatches. The predicted ~7.4 us saving
did not appear at all. (The prologue itself costs ~7.5 us on the wqkv call; batching it over DST -- one
`tile_regs` cycle per 8 tiles instead of per tile -- changed nothing, so that part is real work: the
full-row square+reduce on every core plus the per-core weight/bias fetch.)

**So both levers have now failed to move wall-clock:** kv_sdpa's kernel time fell 27% and wall rose;
op count fell by 2/iteration and wall rose. What is actually established is only this:

* ~24% of traced single-layer wall-clock (28.3 of 116.5 us) is NOT accounted for by summed device
  kernel durations.
* That gap responds to neither less kernel time nor fewer ops.

Its true nature is **unknown**. Candidates worth testing before any further optimization:
1. The profiler's `DEVICE KERNEL DURATION` excludes per-op device-side cost (program setup, CB drain,
   semaphore waits), so "kernel time" is simply not the right denominator.
2. A critical-path / serialization effect: total kernel time is a SUM, but wall-clock follows the
   longest dependency chain, so shrinking a non-critical op changes nothing.
3. Fixed per-program cost in trace replay that scales with something other than op count.

Hypothesis 2 deserves the first look: it would explain BOTH failures at once, and would mean the whole
"sum the kernel times" framing was wrong -- one should instead find which ops are on the critical path
(e.g. from per-op start/end timestamps in `profile_log_device.csv`, which the current tooling
aggregates away).

**Do not treat "fewer ops" or "less kernel time" as a proxy for this pipeline's wall-clock.** The one
result that has held up across every experiment is the opposite direction: adding CORES to an op
reliably hurts (kv_splits S=2/4/8, "64 cores beats 80", `_RESHARD_CORES=2` beats 4 and 8).
