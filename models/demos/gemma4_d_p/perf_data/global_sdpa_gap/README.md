# Gemma4 global ring SDPA gap: experiments (bh-glx-120-c04u02, 2026-10-15)

Goal: close the gap between global attention (measured ~3.56 s of the 10.3 s 256k/2K prefill) and its roofline
(~1.06 s at 120 cores LoFi). Background: /data/mmanzoor/global_sdpa/GLOBAL_SDPA_FINDINGS.md (dig-base stack,
b09u08) and the llm_perf roofline study (`llm_perf_study_copy/`, also at tt-metal/llm_perf_study on c04u02).

## Code

- Base: origin/main `91d77faccd3` (see BASE_COMMIT).
- Branch `mmanzoor/gemma4-global-sdpa-gap-edit` (3 commits), as a bundle and as patches:
  ```bash
  git fetch /data/mmanzoor/global_sdpa_gap_c04u02/gemma4-global-sdpa-gap.bundle \
      mmanzoor/gemma4-global-sdpa-gap-edit:mmanzoor/gemma4-global-sdpa-gap-edit
  # or: git checkout -b <br> 91d77faccd3 && git am /data/mmanzoor/global_sdpa_gap_c04u02/patches/*.patch
  ninja -C build && ninja -C build install     # factory change is host C++
  ```
  1. `WIP: port ring SDPA partial readiness` (env `TT_RING_SDPA_PARTIAL_READINESS=1`; default off = inert).
     Ported from /data/mmanzoor/overnight/gemma4_ring_partial_readiness.patch; one include conflict resolved.
     **HANGS on main** (see results). Needs debugging before use.
  2. `TEMP: env override` `GEMMA4_GLOBAL_SDPA_CFG="q,k,splits,segmented"` for the global ring SDPA chunk sizes.
  3. `TEMP: allocator dump` `GEMMA4_DUMP_L1_BEFORE_GLOBAL_SDPA=<prefix>` dumps L1/DRAM allocator state once,
     before the first global SDPA (to find the persistent L1 buffer at 1440576). Not run yet.
- On c04u02 the main checkout (/home/mmanzoor/compiler/tt-metal, branch `mmanzoor/gemma4-global-sdpa-gap`) has
  commits 1+2 applied uncommitted and built; the edit worktree is /home/mmanzoor/compiler/tt-metal-edit.

## Benchmark

`run_layer.sh <tag> [chunk_idx=64]`: tracy run of one traced global layer, 2K chunk, chunk 64 (133k prefix):
`text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk64-global-sz2048-ctx_256k-8x4]`, then
`layer_ops.py` prints the median-over-devices SDPA op time and the layer sum into `<tag>.ops.txt`.
Env vars pass through (`env FOO=1 run_layer.sh tag`). `batch2.sh` = sequential runs, each under `timeout 900`,
stopping at the first timeout. Run inside tmux. ~7 min per run (3 min test + 4 min tracy post-processing).
Paths assume the repo at /home/mmanzoor/compiler/tt-metal and `source activate`; edit `run_layer.sh` on a new box.

## Results (chunk 64, 133k prefix, median over 32 devices)

| Tag | Config | Global SDPA (us) | Layer sum (us) | Note |
|---|---|---:|---:|---|
| baseline | main default: q64, k256, 3 K-split bands, segmented (96 busy cores) | **2652.8** (2542-2719) | 3475.4 | host-timed layer 3.71 ms |
| patched_off | commits 1+2 applied, knobs off | 2651.9 | 3476.5 | patch inert when off |
| q128b5 | `GEMMA4_GLOBAL_SDPA_CFG=128,256,5,1` (dig-base layout, 80 busy cores) | n/a | n/a | **L1 clash**: CBs end at 1475840 > persistent L1 buffer at 1440576 (~35 KB over) |
| pr | `TT_RING_SDPA_PARTIAL_READINESS=1` | n/a | n/a | **hung** after kernel compile (needed board reset) |
| q64k512 | `GEMMA4_GLOBAL_SDPA_CFG=64,512,3,1` | n/a | n/a | **Does not fit**: CBs grow to 1733888 B > 1572864 B L1 (needs single-buffered K/V or smaller CBs, not just the persistent buffer moved) |

Key finding: main's short-slab default (q64/3 bands, from #59195) is ~13% slower than dig-base's q128/5 bands
(2343 us on b09u08), and matches the dig-base q64/3-band experiment (2654 us). q128/5 no longer fits on main,
so the first lever is freeing ~35-130 KB of L1 (the buffer at 1440576 -> top of L1, ~129 KB per core).

## Next steps

1. Run `env GEMMA4_DUMP_L1_BEFORE_GLOBAL_SDPA=/data/.../l1dump_ run_layer.sh l1dump` and read
   `*detailed_memory_usage.csv` to find what owns 1440576..top of L1 at the global SDPA (live L1 activations at
   the call are small: normed V and packed_kv ~2 KB/core; suspect the residual stream or a persistent buffer).
   Moving it unblocks q128/5-bands (expect ~2343 us, -12%) and maybe k512.
2. Debug the partial readiness hang with DPRINT of the SDPA reader's semaphore values vs
   partial_half/full_threshold; main's all-gather reader/writer counting likely differs from dig-base.
   Run under `timeout`. Worth ~-4%.
3. Bigger levers (multi-day): lazy rescale (ceiling -9.9%), single read of the packed 640-wide K/V row
   (reader is near saturation), non-row-aligned bands to use the last 14 idle cores.

Note: the batch was stopped on purpose after q64k512 (pr killed 1 s after start, before device open) (tmux session `gsdpa_guard` kills `gsdpa` before the `pr`
run), so a hang can't wedge the board after hand-off. The q64k512 result is appended to batch2.out.
