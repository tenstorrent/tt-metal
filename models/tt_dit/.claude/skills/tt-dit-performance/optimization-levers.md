# Lever catalogue

Multipliers are from real TT-DiT campaigns, some on branches not yet merged.
They are **order-of-magnitude calibration for choosing a lever, not reproducible
benchmarks** — measure your own shape.
Match every lever to the **measured bound class** from
`../tt-dit-benchmark-profile/reading-profiles.md`; compute-bound levers on an
overhead-bound op is the most common way a loop stalls.

## Invariant: dtype matches the reference

**Dtype is a correctness contract, not a tuning knob.** Check the `diffusers`
(or upstream) implementation per component and match it. The two directions are
not symmetric:

| Situation | Action | Note |
|---|---|---|
| Running fp32 where the reference runs bf16 | **Fix it — it's a bug.** The speedup is a side effect | Measured **2.83×** on a video-VAE encoder. VAEs are often fp32 by convention, not necessity |
| Wanting to drop *below* the reference | Last resort only | Needs a perceptual gate, not just PCC. Legitimate when the kernel is bf16-only anyway |

After **any** dtype change, re-sweep blockings — tables tuned at fp32 miss at
bf16 and fall back to a conservative default. Grep the log for
blocking-fallback warnings.

## Order

| # | Lever | Typical | Applies when | Limits / traps |
|---|---|---|---|---|
| 1 | **Parallelism you don't have** | 10–30× | Chips idle, or one work unit owns the whole mesh | Largest lever, routinely skipped because it feels like architecture |
| 2 | **Kernel research** | — | Any hot spot, before writing code | Research step; most hot spots are a fast path that isn't engaged |
| 3 | **Layout round-trips** | 1.5–2× | `Untilize`/`Tilize` bracketing one op | Frequently present around GroupNorm |
| 4 | **Math fidelity** | up to 2× | HiFi4 on ops that don't need it | Distinct from dtype — doesn't change storage type |
| 5 | **Fusion / folding** | 1.2–1.5× | Adjacent ops in the warm profile | Only after 1 and 2 |
| 6 | **Blocking / config sweeps** | 1.5–2× | Tuning surface exists | **Last** — every lever above invalidates the tuning |
| 7 | **Trace** | large | Many small ops, predictable shapes | **Absolutely last.** Fails silently — see below |

---

## 1. Parallelism — 10–30×

Decision order in `../shared/parallelism.md`.

| Kind | Buys | Cost | Gate |
|---|---|---|---|
| **Data parallel over work units** | ~30× near-linear on a 4×8 | None — no collectives, no halo | **PCC = 1.0**, bit-exact. Anything less means the split is unclean |
| **Spatial H/W sharding** | Latency on a *single* unit, which DP cannot buy | Halo exchange on every conv with kernel > 1 | Normal bar + seams checked separately |
| **Sequence parallel / ring attention** | Long-sequence DiT scaling | KV all-gather, overlapped with compute | Normal bar |
| **Tensor parallel** | Wide hidden dims | Collective traffic every layer | Normal bar |

Check data parallel **first** — it gets skipped because it feels too easy.

If AllGather or ReduceScatter is already in the top ops, the answer is usually
**not** a different axis assignment — it is that the collective isn't
overlapping with compute. Reserve cores for the CCL (`ccl_core_grid_offset`),
give it a persistent ping-pong buffer, and prefer an op that fuses the
collective into the compute. See
`../tt-dit-benchmark-profile/existing-fast-paths.md` § "CCL overlap with
compute" — LTX and Wan attention are the reference implementations.

## 2. Kernel research

Start at `../tt-dit-benchmark-profile/existing-fast-paths.md` — it catalogues
the DiT-specific fused ops ttnn already ships (distributed RMSNorm and
LayerNorm — **not GroupNorm**, matmul+all-gather, matmul+reduce-scatter, AdaLN
`dit_minimal_matmul_addcmul_fused`, RMSNorm+activation, fused head ops,
`neighbor_pad_async` for halo) with a "profile shows X → try Y" table.

Most hot spots are a fast path that **isn't engaged** — shape guard, dtype
mismatch, or a call site predating the op. That's a config fix, not kernel work.
For anything uncovered, `../tt-dit-kernel-research/SKILL.md`.

## 3. Layout round-trips — 1.5–2×

Signature: `Untilize` and `Tilize` near the top, bracketing one op — usually
GroupNorm, which wants a different layout than its neighbours.

Measured **52.8% of warm encoder device time** on a video VAE; the conversions
cost more than the norm. Computing statistics in the neighbours' layout measured
**1.78×** on the affected block.

Generalize: any op forcing a layout change pays twice. Group data-movement ops
when ranking — 36% of one ViT layer was pure data movement.

## 4. Math fidelity — up to 2×

Sets FPU cycles per tile; does not change storage type, so it does not touch the
dtype contract. HiFi4 on elementwise (`BinaryNg`, `Unary`) is a silent 2×; HiFi2
is the sane default for most of a diffusion model.

`FLOPs %` is against the *current* fidelity's peak, so HiFi4 → HiFi2 can show
FLOPs% falling while throughput rises. Compare absolute TFLOPs.

## 5. Fusion and folding — 1.2–1.5×

Only after (1) and (2). Ask in order: does a fused op exist → can it fold into
weights at load time → into an existing op's optional inputs → only then a
kernel.

| Pattern | Fix | Measured |
|---|---|---|
| Reshape+transpose chains around attention | `ttnn.experimental.nlp_create_qkv_heads` / `nlp_concat_heads` | **1.45×** on a ViT layer |
| LayerScale, scale-shift modulation, norm affine | Fold into the adjacent projection's weights at load time | 1.015× alone — worth doing, not worth a session |
| `BinaryNg` after a matmul in an AdaLN block | `dit_minimal_matmul_addcmul_fused` | — |
| Unfused `Silu`/`Gelu` after a matmul | `fused_activation=` on the matmul | — |

Not everything folds — RoPE-adjacent elementwise typically needs real kernel
work. Check the op's share of the warm profile first.

Anomalies are cheap wins: two ops doing identical work at very different
durations, or an op on 57 cores while neighbours get 120, is usually a wrong
config.

## 6. Blocking and config sweeps — 1.5–2×, and they go last

**Sweep the blockings last.** Every lever above changes the work; a sweep run
before them is tuning that gets invalidated.

| Target | API | Tool / result |
|---|---|---|
| Conv3d blockings | `utils/conv3d.py::get_conv3d_config`, `register_conv3d_configs` | `tests/models/wan2_2/bruteforce_conv3d_sweep.py` — the canonical sweeper. **1.70×** on a video-VAE encoder |
| Matmul blocks | `utils/matmul.py::get_matmul_config`, `register_matmul_configs` | `utils/sweep_mm_block_sizes.py` — sweeps `(M,K,N,sb_h,sb_w)` in one device session |
| SDPA chunk size | ttnn SDPA config | Sweep at the attention's **real** shape — **2.95×** at `q=k=192` vs defaults |
| GroupNorm `num_out_blocks` | per-shape table | Also gates L1: too few blocks = CBs larger than L1 |

Discipline: one knob family at a time, production shape, warm, **one config per
process with a hard timeout** (`../shared/device-hangs.md` § Sweep discipline).
Record the whole curve — it tells the next agent whether the optimum is sharp or
flat — and which values were never reached.

## 7. Trace — absolutely last

Trace is a **guaranteed win where it applies**, which is exactly why it goes
last: it survives every other change, and chasing it first ships a model that
dispatches efficiently and computes slowly. Everything above cuts device time;
trace only removes dispatch gaps.

| | |
|---|---|
| **Where it pays** | Predictable shapes, many small ops — audio decode, the denoiser loop. LTX: 34.6 s → 7.6 s end-to-end on a 4×8 |
| **Where we don't** | Visual VAE — shapes vary with tiling, ops are large, little dispatch overhead to recover |
| **Verify first** | The warm-window gap distribution must show dispatch is a real share. `tt-perf-report`'s tracing headline read 97.1% on a component whose warm gap share was 16.2% |
| **Mechanics** | `utils/tracing.py::traced_function` adds a `traced=` kwarg; `Tracer` captures on first call, replays after. Needs `trace_region_size` |

**Trace is the one lever that fails silently.** Every other lever works or
raises; a mis-set-up trace returns plausible, wrong output.

| Hazard (documented on `Tracer`) | Consequence |
|---|---|
| Tensors allocated **after** capture may be overwritten during replay | Corruption lands in an unrelated tensor and reads as a model bug |
| The same output tensor objects are returned every call, overwritten in place | Holding a reference across calls gives aliased data, not two results |
| `trace_region_size` is DRAM taken from weights and activations | Oversizing pushes an allocation failure somewhere else. In-tree: 90112 to 500_000_000 |

So: gate each region behind its own env flag to bisect, and **re-check quality
end to end** against the untraced baseline. A component PCC gate cannot catch
this — it runs the untraced path.

## Not levers

| | |
|---|---|
| Weight upload | One-time construction cost. Cache it (`TT_DIT_CACHE_DIR`); never count it, never optimize it as per-step |
| Host Python between device ops, untraced | Real, but that is what trace removes |
| Fewer steps or lower resolution | A product decision. If you're proposing it, you've run out of levers — say so |
