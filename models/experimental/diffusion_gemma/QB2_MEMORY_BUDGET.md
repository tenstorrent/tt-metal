# QB2 memory budget and batch ceiling (#47487)

Status: current — the measured QB2 fit for the causal text backbone plus the diffusion-path additions.
Owns: measured per-chip DRAM, weights, KV at 256K, the static batch ceiling, and the probes' traps.
See also: [plan (env recipe, box identity)](plan.md#5-qb2-environment-and-recovery-recipe-bh-qbge-06-the-one-copy) · [refuted list](doc/REFUTED.md) · [serving hub](doc/vllm_integration/README.md)

Slightly over the 60-line target: the three probe traps and the repro command are not cut for length.

## Measured on QB2 (2026-06-24, `ttnn.get_memory_view(mesh_device, BufferType.DRAM)`)

`P150x4`, (1,4) mesh, TP=4, **bf16 weights**, KV allocated eagerly at build so the weights+KV budget
is captured without a prefill (`tests/test_qb2_memory_budget.py`).

| quantity | GiB/chip | note |
|---|---:|---|
| usable DRAM | **31.87** | 8 banks × 3.984, only ~0.13 reserved (`tech_reports/memory/allocator.md:21`, telemetry `ENABLED_GDDR=0xff`). An earlier "~4 GB/chip" read per-bank as per-chip; the "28–30 GB" estimate is superseded |
| weights (bf16, TP=4-sharded) | **13.25** | = 52 GB / 4; **corrects** the bf8-derived "~6.5–7 GB/chip" estimate — the real run loads bf16 |
| + paged KV @256K, batch 1 | **17.25** total (Δ 4.0 KV) | 54% of usable; **headroom 14.6** |
| + paged KV @256K, batch 2 | **19.80** total | ⇒ +2.55 GiB/chip per extra batch |

**Static weights+KV ceiling** = `(31.87 − 13.25 − 4.0)/2.55 + 1 ≈ batch 6`. This is a weights+KV
bound, **not** a validated end-to-end generation ceiling, and batch > 1 is out of current scope
(#47557) — do not read it as an operational limit.

**Experts are TP-SHARDED, resolved empirically 2026-06-22:** the 26B-A4B full causal backbone ran on
QB2 `P150x4` TP=4 in 110 s with no OOM, via `MeshConfig.column_parallel`/`row_parallel` (~5.7 GB/chip
at bf8), so `test_full_model`'s `if is_moe and tp<8: pytest.skip` guard is conservative/stale for this
fit. Backbone logits PCC vs HF = **0.8665** on "The capital of France is", above the Blackhole
`test_full_model[blackhole-1x8]` baseline of 0.83; the 12B dense backbone on QB2 = 0.9595. Had the
experts turned out replicated, the fix would have been Expert Parallelism — the HF config ships
`base_model_ep_plan` (`configuration_diffusion_gemma.py:68-77`) and gemma4 `tt/` does not wire EP.

Reproduce (env: see [plan.md](plan.md); requires the `tp<8` MoE skip removed/relaxed):

```bash
MESH_DEVICE=P150x4 HF_MODEL=/home/zni/dg_models/gemma-4-26B-A4B-it \
  pytest models/demos/gemma4/tests/unit/test_model.py::test_full_model -k "1x4"
```

- **TRAP:** the 0.83 threshold applies only when `HF_MODEL`'s **basename** is `gemma-4-26B-A4B-it`.
  Use the symlink above — a bare HF-cache snapshot basename is a hash, and the lookup then falls back
  to the 0.99 default.

## The real limiter is the prefill-activation regime, not weights/KV

The generator forces a **single prefill chunk** (`tt/generator.py:76-85` — Gemma4 ignores
`chunk_start_idx` and rounds `max_prefill_chunk_size` up to a power of 2 ≥ the prompt), materializing
the full `[1, L, 2816]` activation ∝ L on top of the 17.25 GiB. gemma4's own demo notes single-chunk
prefill runs at 128k without OOM, consistent with the 14.6 GiB headroom.

- **TRAP:** a raw single `ttnn_prefill_forward` of L > `sliding_window` is **not** a valid probe — it
  writes the whole sequence into the bounded-sliding 1024-token KV pool and trips
  `update_cache_device_operation.cpp:106`. Long prefill must go through the generator's chunked +
  bounded-sliding path (`operations.py:210-353`).
- **TRAP:** the demo's `trace_region_size=200_000_000` (`text_demo_v2.py:143`) is too small for
  long-context prefill traces — a 64k trace needs ~445 MB (`TT_FATAL` at `mesh_trace.cpp:78`) — and
  the trace region is carved from DRAM, so subtract it from the headroom.

## KV geometry and the diffusion-path additions (not in the static budget)

- **Sliding layers (25)** bound KV to `sliding_window=1024`: `2·8·1024·256·2 B ≈ 8.4 MB/layer`,
  ~210 MB total. **Full-attention layers (5)** have `num_global_key_value_heads=2`,
  `global_head_dim=512`, `attention_k_eq_v=True`, and are unbounded but paged/right-sized
  (`page_block_size=64`, `page_max_num_blocks=4096`).
- The denoise forward returns logits for all 256 canvas positions every step, so it must **disable
  gemma4's §2.8 last-tile LM-head slice**. Full-canvas logits `[256, vocab]` cost ~34 MiB/chip
  column-parallel (`vocab/4`) or ~137 MiB/chip if all-gathered, plus an equal-size softmax/probs
  buffer, recomputed every step.
- Per-step canvas K/V scratch (#47474 storage class ii) is statically estimated by `memory_budget.py`
  at ~15 MiB/chip (QB2 TP=4, bf16, batch 1: 25 sliding ≈ 12.5 MiB, 5 full-attn ≈ 2.5 MiB) — the TT
  path materializes separate K and V even for K=V-tied full layers.
- Per-token contiguous KV in the served configuration, and its unresolved derivation, live in
  [traced_serving](doc/vllm_integration/traced_serving.md).
