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

Switch: `tt/tile_config.py:11 TILE_HEIGHT = 16`. No env var; set to `32` to revert.

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
