# FLUX.2 frontier evaluation

The larger evaluation is **complete** on bh-lb-08 (reservation 221619):
48 images, raw CLIP scores, six paired PNG sheets, 28 real-QKV measurements,
and all 96 forward/reverse full-block timing measurements. See
[the complete report](replacement-suite-01/REPORT.md).

[Weight caching is working](WEIGHT_CACHE.md): full-model setup fell from
248.8 s for the initial cold conversion/write to 9.1 s warm. All 24 component
loads across the eight image runs hit cache. The fresh 50-step stock control
passed all 15 pairwise equality comparisons among three untraced and three
steady traced calls. All recorded image-run and sustained block replay checks
were exact.

The old bh-51 allocation was released after reset did not resolve overheating.
Its [partial results](suite-02/REPORT.md) and [repeatability investigation](REPEATABILITY.md)
are preserved. The historical repeatability cause is not isolated: both hardware
and host-pinning conditions changed. Current results are a small exploratory
evaluation, not a statistically powered quality ranking or full Diffusers qualification.

See [STATUS.md](STATUS.md) for reservation, environment setup, the passing
eight-device environment smoke, resolved checkpoint access, and adapter checks.

**Correctness prerequisite:** the pinned source predates the September 10
FLUX.2 fixes in main (#55225). Its Linear-topology attention output projection
drops the residual/gate, and dual-stream Q/K normalization is not per-head.
See [DIAGNOSIS.md](DIAGNOSIS.md) for the controlled failing/passing block tests.
`FLUX2_MODEL_REPAIR=main_fused` opts into isolated equivalents of those two
fixes; `both` uses an unfused diagnostic residual repair and is not a performance
baseline. Old noise images must not enter the frontier quality comparison.
`FLUX2_CONDITIONING=corrected` independently repairs guidance scaling and phase
precision. Both settings are recorded and must match across compared runs.

The evaluation keeps numerical recipes D/C/B/A/E/F/G fixed. Stock FLUX.2 is
a separate control until demonstrated equivalent to A. Encoder, VAE, model
weights, scheduler, guidance, prompts and initial noise must remain identical.
New input conditioning strategies are outside this evaluation.

## Stock bring-up

`test_pipeline.py` follows the existing FLUX.2 performance test's Blackhole LB
configuration: mesh 2x4, SP2/TP4, two links, no FSDP, no dynamic loading,
sharded prompt, and 64 KiB L1_SMALL for the VAE. It defaults to one prompt,
seed 0, 1024x1024, two denoising steps, untraced. This is a smoke test, not
a quality evaluation. Set `FLUX2_VARIANT` to stock or A through G; there is no
unsupported-input fallback. The VAE uses fixed Q64/K64 blocks to fit Blackhole
L1 with the pinned build. Frontier attention retains Q256/K512.

From the repository root in a compatible prebuilt environment:

```sh
FLUX2_OUTPUT=/absolute/fresh/output/directory \
  python -m pytest -s experiments/sdpa-l2/flux2-frontier-v1/test_pipeline.py
```

`FLUX2_CHECKPOINT` may specify an existing FLUX.2-dev checkpoint directory.
Otherwise the standard model-location fixture is used. Converted-weight caching
now has a validated full-model workaround: set `TT_DIT_CACHE_DIR` to a fresh cache
root and `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`. This bypasses the stalled
pinned transfer path, not the disk cache. See [WEIGHT_CACHE.md](WEIGHT_CACHE.md)
for the fresh-process hash checks and full-model qualification procedure.
`FLUX2_REQUIRE_WEIGHT_CACHE=1` makes an unexpected conversion/cache miss fail
explicitly; enable it only after populating and validating the cache.
Use `FLUX2_TRACED=1` for the
traced smoke, with a fresh output directory. Full evaluation uses
`FLUX2_STEPS=50 FLUX2_PROMPTS=3 FLUX2_SEEDS=2` after integration qualification.

The manifest records inputs, software versions, source hashes and stage
timings. These host timings are not per-block device performance. Detailed
device profiling and CLIP scoring are separate phases.

`FLUX2_BLOCK_BENCH=1` captures six real block inputs before model trace capture
and measures isolated blocking mesh trace replays (250 warmups, fifty samples).
These are full-block latencies including attention preprocessing and communication,
not hardware-counter measurements. `run_suite.py` runs serially and stops at the
first failed variant. `score_images.py` checks paired configurations and image
hashes and reports raw CLIP cosine, which is not a direct image-fidelity metric.

Strict qualification is the default. `FLUX2_EXPLORATORY=1` explicitly records
block/model replay inequality instead of aborting generation. It preserves
finite-value, shape, recipe and no-fallback checks; it does not change attention
numerics. Such runs are exploratory, **not qualification passes**, and use
separate output directories so original failures remain available. The mode
exists because the stock control also fails repeated-run equality.

The seven experimental variants share a gather-KV wrapper: Q stays local, KV
is preprocessed on its owning rank and gathered in its prepared format. This
is **not overlapped ring attention**. Stock ring is a separate control, and
its performance must not be represented as a numerical-scheme-only comparison.

| Variant | Denoiser attention recipe | Device input preparation |
|---|---|---|
| D | FP32 streaming, HiFi4 QK/PV, full-FP32 subtraction, refined exp | None |
| C | FP32 streaming, HiFi4 QK / HiFi2 PV, cheaper exp path | None |
| B | BF16 HiFi2 streaming, compensated recurrent state | None |
| A | Frozen main BF16 HiFi2 streaming | None |
| E | LoFi BF16 streaming, compensated state, BFP8 KV | Q RNE7; KV RNE5 then BFP8 |
| F | LoFi FP32 streaming, BFP8 KV | Q RNE7; KV RNE5 then BFP8 |
| G | LoFi BF16 streaming, compensated state, BFP4 KV | Q RNE7; KV native-group RNE+saturation |
| stock | Existing tt-dit joint ring SDPA, BF16 HiFi2, exp_approx_mode=False | None |

The table is a short label, not a replacement for the exact compile defines:
`test_recipe_defines_match_frontier_records` compares every define and fidelity
setting against the frozen measured records. All outputs are BF16. The text
encoder and VAE do not use these overrides.

## Integration gates

- Preserve joint image/text attention, actual logical lengths and head mapping.
- Preserve global softmax across SP shards; no independently normalized partial
  outputs masquerading as global attention.
- Qualify real boundary Q/K/V, preprocessing range, complete finite outputs and
  traced replay before image comparisons. No unsupported-shape fallback.
- Count on-device preprocessing and communication in model timings.
- Distinguish compressed local KV storage from compressed ring transport.
- Compare identical prompts/seeds and publish every generated image.
