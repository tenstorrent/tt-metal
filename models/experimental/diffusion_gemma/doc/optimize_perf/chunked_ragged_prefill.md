# Chunked ragged prefill — killing the >4096 MoE cliff

Status: current — this is the prefill authority for DiffusionGemma. These numbers describe the
**prefill** path only (unchanged since `233b88276ab`, which has not been re-run at HEAD); they are
not evidence about the denoise path, whose MoE was replaced on 2026-07-29.
Owns: the >4096 dense all-128-expert prefill cliff, the chunked-ragged fix, its bit-identity, and
the current 65536-build pure-prefill throughput table.
See also: [refuted list](../REFUTED.md), [optimize-perf hub](README.md),
[vLLM serving](../vllm_integration/README.md).

**Artifact warning:** any `context_window_prefill_only_*` artifact **without** `chunkedlong` in its
name is the superseded pre-fix dense-fallback control (`ec5b64b4891`) and must never be quoted as
current performance. **MEASUREMENT TRAP:** pure prefill (this file), serving prefill
([context speed sweep](context_speed_sweep_20260722.md)) and DiffusionGemma "TTFT" are three
different metrics, never quoted for one another — see the three-metrics rule in the
[optimize-perf hub](README.md). Device: QB2 / P150x4 / 4× Blackhole / TP=4, model
`diffusiongemma-26B-A4B-it`, 2026-07-13.

## Current shipped default

`DG_PREFILL_RAGGED_LONG=1` in `tt/prefill_moe.py`: every multi-token prefill uses the ragged top-8
path, and sequences above 4096 are split into `DG_PREFILL_RAGGED_CHUNK`-token slices (default 4096,
TILE-aligned). `DG_PREFILL_RAGGED_LONG=0` is the diagnostic control that restores the dense
all-128-expert fallback.

## The cliff and its root cause

Pre-fix pure-prefill on one 65536 build at `ec5b64b4891` (build 21.37 s, 17.33 GiB DRAM/chip
resident; each row one synchronized first execution, so first-use compile is included):

| context | prefill | tok/s | MoE path |
|---:|---:|---:|---|
| 1,024 | 0.69 s | 1,473.9 | ragged (top-8 experts) |
| 4,096 | 1.27 s | 3,213.2 | ragged (top-8 experts) |
| **16,384** | **43.17 s** | **379.5** | **dense (all 128 experts)** |
| 32,768 | 96.54 s | 339.4 | dense + long-context chunked SDPA |
| 65,536 | 235.44 s | 278.4 | dense + long-context chunked SDPA |

The 4K→16K drop is an **8.5× MoE cliff, not attention** — the SDPA only chunks above 32768.
`tt/prefill_moe.py` gated **both** `_contextual_router_forward` and `_contextual_prefill_forward` at
`1 < S <= 4096`; above that the shared Gemma4 dense prefill computed all 128 experts per 32-token
tile and zeroed ~120/128 through the routing weights — roughly 16× wasted compute.

The 4096 bound was a **conservative verification ceiling, not a hard limit**: bit-identity had been
verified to S=2048, the gate was set at the serving context, and the ragged packer's
`max_segments = ceil(S/128)` design already scales to arbitrary `S` (confirmed across the packer,
`_ragged_prefill_program_config`, `sparse_matmul` and the `embedding` device-op validators). The two
gates **must stay coupled**: the ragged router emits a `RaggedRouting` object that only the ragged
prefill can consume.

## The fix — token-dim chunking

MoE FFN is per-token, so a long prefill runs in chunk-sized slices through the *unchanged* ragged
path, concatenating the per-chunk `[1,1,chunk,H]` outputs on the token dim
(`tt/sparse_moe.py :: chunked_ragged_sparse_prefill_forward`). The router runs **once at full `S`**
and the wrapper slices `RaggedRouting.values`/`.indices` and `hidden_states` on the same boundaries;
`S <= chunk` delegates straight through, so the single-chunk path is byte-for-byte the old
behaviour. Per-chunk routing recompute was refuted — see [refuted list](../REFUTED.md).

Chunking is **required, not merely faster:** a single full-`S` ragged call materializes
`selected/gathered` with logical volume `top_k*S*H` (1.48e9 at 64K), which overflows int32 at ~128K
and uint32 near 256K, so unchunked ragged physically cannot reach the 256K context target regardless
of DRAM (it is also DRAM-fragile at 64K: ~8.8 GiB transient over a 17.3 GiB resident baseline). 4K
chunking caps every intermediate at the S=4096 footprint (~0.55 GiB peak, ~9.2e7 element volumes),
flat at any context; L1 is `S`-independent because the program config fixes it.

## Device bit-identity (QB2, 30 layers, full checkpoint)

Chunked-ragged == dense with logits **and** full KV cache `max_abs == 0`
(`chunked_ragged_prefill_bitident.json`):

| seq_len | chunks | dense prefill | chunked prefill | speedup | logits max_abs | KV max_abs |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1 (fast path) | 10.40 s | 3.19 s | 3.26× | 0 | 0 |
| 6,144 | 2 (+2048 tail) | 15.53 s | 3.28 s | 4.74× | 0 | 0 |
| 8,192 | 2 | 22.70 s | 4.82 s | 4.71× | 0 | 0 |

The single-chunk 4096 case is deliberately in the matrix to prove the fast path is a byte-exact
no-op; 6144 and 8192 prove the tail and the multi-chunk seams. **ENVIRONMENT TRAP:** QB2 needed a
`tt-smi -r` first — the prior vLLM server had left eth core 29-25 hung, the known recurring reset.

Host coverage is `tests/test_prefill_moe.py`: gating window, coupled router+prefill dispatch
(S=128/4096 ragged, 16384 dense when OFF, all multi-token chunked when ON), chunk-aligned slice
boundaries including a 32-row tail, N-way concat on dim 2, single-chunk fast path, parent routing
freed.

## Throughput recovery — the current 65536-build pure-prefill table

`context_window_sweep.py --prefill-only` with `DG_PREFILL_RAGGED_LONG=1` at `233b88276ab`
(`context_window_prefill_only_chunkedlong_20260713_msl65536.json`), build 21.48 s, 17.33 GiB/chip
after build. Each row is one synchronized **first** execution of that prompt shape, so shape-specific
first-use compilation is included; 8192 is the first multi-chunk shape and pays that path's
first-use cost, which is why it dips below 16K and 32K.

| context | prefill | tok/s (fix on) | tok/s (was) | speedup | DRAM after |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.78 s | 1,309 | 1,474 | ~1× (var) | 17.34 GiB |
| 4,096 | 1.37 s | 2,987 | 3,213 | ~1× (var) | 17.34 GiB |
| 8,192 | 4.15 s | 1,973 | — | — | 17.35 GiB |
| 16,384 | 5.55 s | **2,950** | 379 | **7.8×** | 17.35 GiB |
| 32,768 | 10.84 s | **3,021** | 339 | **8.9×** | 17.36 GiB |
| 65,536 | 35.58 s | **1,842** | 278 | **6.6×** | 17.36 GiB |

DRAM after prefill is **flat (~17.35 GiB) at every context** — the chunked path adds no resident
growth with `S`. (32,768 is 3,021.5 tok/s raw, quoted as 3,021 here and rounded to 3,022 in the hub:
one measurement, two roundings.) The 1024/4096 rows match the pre-fix ragged regime — same code
path, and the ~1× deltas are first-execution timing variance. The residual 64K slowdown to 1,842
tok/s is the separate long-context **attention**-chunking path above 32768 (`chunked_prefill_sdpa`),
not a dense-MoE fallback — a different lever from this MoE fix.

## Repro

Env: see [plan](../../plan.md). Device bit-identity:

```
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export TT_METAL_HOME=/home/zni/tt-metal PYTHONPATH=/home/zni/tt-metal
python models/experimental/diffusion_gemma/doc/optimize_perf/verify_chunked_ragged_prefill.py \
  --seq-lens 4096,6144,8192 \
  --output-json models/experimental/diffusion_gemma/doc/optimize_perf/chunked_ragged_prefill_bitident.json
```

Throughput sweep:

```
DG_PREFILL_RAGGED_LONG=1 DG_PREFILL_MOE_TUNED=1 \
python models/experimental/diffusion_gemma/doc/optimize_perf/context_window_sweep.py \
  --prefill-only --num-blocks 0 --max-seq-len 65536 \
  --prompt-lengths 1024,4096,8192,16384,32768,65536 --mesh P150x4 \
  --checkpoint /home/zni/dg_models/diffusiongemma-26B-A4B-it \
  --label chunked-ragged-long --output <...>.json
```

Artifacts: `context_window_prefill_only_chunkedlong_20260713_msl65536.json` +
`prefill_speed_64k_build_chunkedlong_20260713.png` (current);
`context_window_prefill_only_20260713_msl65536.json` +
`prefill_speed_64k_build_20260713.png` (pre-fix control);
`chunked_ragged_prefill_bitident.json`.
