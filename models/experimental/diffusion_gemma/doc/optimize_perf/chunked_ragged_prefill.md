# Chunked ragged prefill — killing the >4096 MoE cliff

Date: 2026-07-13 · Device target: QB2 / P150x4 / 4× Blackhole / TP=4 · Model: `diffusiongemma-26B-A4B-it`
Throughput artifact source: `233b88276ab` · Current HEAD has not rerun this sweep.

## The cliff

Pure-prefill sweep (`context_window_prefill_only_20260713_msl65536.md`), one 65536-context build:

| context | prefill tok/s | MoE path |
|---:|---:|---|
| 1,024 | 1,473.9 | ragged (top-8 experts) |
| 4,096 | 3,213.2 | ragged (top-8 experts) |
| **16,384** | **379.5** | **dense (all 128 experts)** |
| 32,768 | 339.4 | dense + long-context chunked SDPA |
| 65,536 | 278.4 | dense + long-context chunked SDPA |

The 4K→16K drop is an **8.5× MoE cliff**, not attention: the attention SDPA only chunks above 32768
(and internally — it still hands full-`S` `[1,1,S,H]` to the MoE). The cliff is entirely the MoE
gate.

## Root cause

`tt/prefill_moe.py` dispatched the fast zero-drop **ragged** sparse prefill (only the routed top-8
experts/token) through two coupled hooks gated at `1 < S <= 4096`:

- `_contextual_router_forward` → `ragged_router_forward` (returns a `RaggedRouting` object)
- `_contextual_prefill_forward` → `ragged_sparse_prefill_forward`

Above 4096 both fell back to the shared Gemma4 **dense** prefill, which computes all 128 experts per
32-token tile and zeros ~120/128 via the routing weights — ~16× wasted compute.

The 4096 was a **conservative verification ceiling**, not a hard limit: bit-identity was verified to
S=2048, the gate set at 4096 (the serving context), and the ragged packer's multi-segment design
(`max_segments = ceil(S/128)`) already scales to arbitrary `S`. Nothing in the ragged path is
dimensioned to 4096 (confirmed across the packer, `_ragged_prefill_program_config`, `sparse_matmul`
and `embedding` device-op validators).

## The fix — token-dim chunking (`DG_PREFILL_RAGGED_LONG`, default ON)

MoE FFN is **per-token** (position-independent), so a long prefill is processed in
`DG_PREFILL_RAGGED_CHUNK`-token slices (default 4096, TILE-aligned) through the *unchanged*,
validated ragged path, concatenating the per-chunk `[1,1,chunk,H]` outputs on the token dim.

`tt/sparse_moe.py :: chunked_ragged_sparse_prefill_forward` — drop-in for
`ragged_sparse_prefill_forward`:

- Router runs **once at full `S`** (`ragged_router_forward`), producing the `RaggedRouting`. The
  wrapper slices `RaggedRouting.values`/`.indices` **and** `hidden_states` by the same boundaries and
  calls `ragged_sparse_prefill_forward` per chunk (its per-chunk TP all-reduce preserved). The shared
  `per_expert_scale` is passed by reference (cached by `id()`, so only chunk-0 does the host readback).
- `S <= chunk` (or a non-`RaggedRouting` argument) delegates straight through → the single-chunk path
  is byte-for-byte today's behavior.

`tt/prefill_moe.py`: `ragged_long_prefill_enabled()` + `_use_ragged_for()` move **both** gates
together — `1 < S <= 4096` when OFF (unchanged), any `S > 1` when ON. (The gates must stay coupled:
the ragged router emits a `RaggedRouting` only the ragged prefill can consume.)

### Why per-chunk routing is *not* used

`MoEBlock.__call__` computes routing from `router_input` (raw residual) but runs experts on a
*different* `expert_input` (normed); the prefill hook only sees `expert_input`. Recomputing routing
per chunk would also run the router `ttnn.linear` at `M=chunk` instead of `M=S` — a different matmul
accumulation order that could break `max_abs==0` vs the dense baseline (which routes at `M=S`).
Slicing the full-`S` routing keeps routing byte-identical and the experts' per-row sparse-matmul is
`m_blocks`-invariant, so **chunked-ragged == full-`S` ragged == dense** per token.

## Why chunking is *required* past ~128K (not just faster)

A single full-`S` ragged call materializes `selected/gathered` with logical volume `top_k*S*H`
(=1.48e9 at 64K). That **overflows int32 at ~128K and uint32 near 256K** — so unchunked ragged
physically cannot reach the 256K context target regardless of DRAM. It is also DRAM-fragile at 64K
(~8.8 GiB transient over a 17.3 GiB resident baseline). 4K chunking caps every intermediate at the
S=4096 footprint (~0.55 GiB peak, ~9.2e7 element volumes) — flat at any context. L1 is
`S`-independent (fixed by the program config), so this is purely a DRAM/index-range story.

## Verification

- **Host (device-free, run):** `tests/test_prefill_moe.py` — flag default-off/toggle, gating window,
  coupled router+prefill dispatch (S=128/4096 → ragged, 16384 → dense when OFF; all multi-token →
  chunked when ON), and the chunk-loop plumbing (correct chunk-aligned slice boundaries incl. a
  32-row tail, N-way concat on dim 2, single-chunk fast-path, dense-routing passthrough, parent
  routing freed). The host suite passed.
- **Device (QB2, PASSED):** `doc/optimize_perf/verify_chunked_ragged_prefill.py` (30 layers, full
  `diffusiongemma-26B-A4B-it`) — chunked-ragged == dense, logits **and** full KV cache
  `max_abs == 0`, at every case (`chunked_ragged_prefill_bitident.json`):

  | seq_len | chunks | dense prefill | chunked prefill | speedup | logits max_abs | KV max_abs |
  |---:|---:|---:|---:|---:|---:|---:|
  | 4,096 | 1 (fast path) | 10.40 s | 3.19 s | 3.26× | 0 | 0 |
  | 6,144 | 2 (+2048 tail) | 15.53 s | 3.28 s | 4.74× | 0 | 0 |
  | 8,192 | 2 | 22.70 s | 4.82 s | 4.71× | 0 | 0 |

  The single-chunk (4096) case confirms the fast-path is a byte-exact no-op; 6144/8192 confirm the
  tail and multi-chunk seams. (QB2 needed a `tt-smi -r` first — the prior vLLM server had left eth
  core 29-25 hung, the known recurring reset.)

### Commands

Device bit-identity (re-run to re-confirm):
```
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export TT_METAL_HOME=/home/zni/tt-metal PYTHONPATH=/home/zni/tt-metal
python models/experimental/diffusion_gemma/doc/optimize_perf/verify_chunked_ragged_prefill.py \
  --seq-lens 4096,6144,8192 \
  --output-json models/experimental/diffusion_gemma/doc/optimize_perf/chunked_ragged_prefill_bitident.json
```

Confirm the cliff is gone (rerun the pure-prefill sweep with the extension ON):
```
DG_PREFILL_RAGGED_LONG=1 DG_PREFILL_MOE_TUNED=1 \
python models/experimental/diffusion_gemma/doc/optimize_perf/context_window_sweep.py \
  --prefill-only --num-blocks 0 --max-seq-len 65536 \
  --prompt-lengths 1024,4096,8192,16384,32768,65536 --mesh P150x4 \
  --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it \
  --label chunked-ragged-long --output <...>.json
```
Expect 8192/16384/32768/65536 to climb out of the ~279–380 tok/s dense regime toward the ~3213 tok/s
ragged regime (minus the full-`S` attention cost, which grows separately and dominates at 32K+).

## Throughput recovery (QB2, pure-prefill sweep, 65536 build)

`context_window_sweep.py --prefill-only` with `DG_PREFILL_RAGGED_LONG=1`
(`context_window_prefill_only_chunkedlong_20260713_msl65536.json`) vs the original dense-fallback
cliff:

| context | prefill | tok/s (fix on) | tok/s (was) | speedup | DRAM after |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.78 s | 1,309 | 1,474 | ~1× (var) | 17.34 GiB |
| 4,096 | 1.37 s | 2,987 | 3,213 | ~1× (var) | 17.34 GiB |
| 8,192 | 4.15 s | 1,973 | — | — | 17.35 GiB |
| 16,384 | 5.55 s | **2,950** | 379 | **7.8×** | 17.35 GiB |
| 32,768 | 10.84 s | **3,021** | 339 | **8.9×** | 17.36 GiB |
| 65,536 | 35.58 s | **1,842** | 278 | **6.6×** | 17.36 GiB |

The cliff is gone: 16K/32K now run at full ragged throughput (~3000 tok/s, matching the 4096 anchor).
DRAM after prefill is **flat (~17.35 GiB) at every context** — the chunked path adds no resident
growth with S. Notes: 1024/4096 match the pre-fix ragged regime (same code path; the ~1× deltas are
first-execution timing variance). 8192 is the first long row so it pays the multi-chunk program
compile; the S-independent ragged program configs are then warm for 16K/32K. 65536 drops to 1,842
because at S>32768 the attention SDPA itself chunks (`chunked_prefill_sdpa`) — that residual is the
long-context *attention* cost, a separate lever from this MoE fix.

## Status

Implemented, host-verified, **QB2 device bit-identical (`max_abs == 0`) and throughput-verified,
default ON**. Set `DG_PREFILL_RAGGED_LONG=0` to force the shared dense fallback. The remaining 64K
gap is attention (chunked SDPA at >32768), not MoE.
