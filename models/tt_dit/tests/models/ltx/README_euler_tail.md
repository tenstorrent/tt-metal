# LTX Euler tail trace experiment

`LTX_EULER_TAIL_TRACE=1` enables a separate tail trace for static, traced,
unconditioned distilled denoising. The default is off. Dynamic loading, image or
reference conditioning, eager execution and `LTX_DEBUG_STATS=1` retain the original
Euler branch. The baseline branch is unchanged, including its image pinning.

Each stage retains the DiT trace's declared velocity outputs and the existing
latent/padding-mask inputs. All six input addresses and layouts must remain
stable. The tail preserves both casts and all eight in-place operations in their
original order. Python `dt` values come from the same float32 sigma scalars;
identical values reuse a trace. Custom and shortened warmup schedules can add
entries beyond the shipped 8+2 schedule. No device scalar cache is created.

Preparation uses cloned inputs; capture records commands and executes the actual
state update once. The tail returns no temporary outputs. Cleanup releases every
tail before its DiT producer or persistent state buffers. Recipe-only execution
uses the original eager tail until a real producer trace exists. A failed replay
requires cleanup and a fresh owner; it cannot resume potentially advanced state.

CPU ownership tests (no device imports):

```sh
python -m unittest discover -s models/tt_dit/tests/unit -p test_ltx_euler_tail.py
```

These execute the actual Tracer and pipeline branch with a command-recording fake.
They cover every shipped step with FP32/BF16 velocities, A/B/A restoration, partial
and full video masks, audio padding, capture without double advancement, address
rebinding, failure/cleanup, and release ordering. They do not validate native
rounding, allocator safety or performance.

Small native contract, through the broker with a current build/source stamp:

```sh
C10_RESULTS=<fresh.pt> python -m pytest \
  'models/tt_dit/tests/models/ltx/test_euler_tail_trace.py::test_euler_tail_trace[galaxy]' \
  -s -v --timeout=500
```

`[f07]` selects its 2x4 mesh. The test preallocates independent baseline/candidate
latents, velocity inputs and masks before all captures. Tiny traced producers
supply retained declared outputs. Two sequence lengths and both velocity dtypes
exercise all 10 steps, changed inputs, per-chip exact BF16 comparisons, padding,
and a final stage revisit after all trace families coexist. Raw mismatches and
partial records are saved before failure. The five recorded tail-only timings
exclude the producer, uploads and validation readback; they are observations,
not an accepted speed claim. A recipe run with `TT_METAL_KERNEL_CAPTURE_ONLY=1`
sets the usual prewarm flag, writes no evidence and skips with no native trace.
Prewarm that manifest using the host's existing CPU tool before collection.

Native contract success is only the first gate. Follow with fresh-process
baseline/candidate full-model runs at the frozen source/build/model scope,
all 10 latent step comparisons and real prompt A/B/A, existing AV quality gates,
compile-free warm generations, and inclusive generation timings. The potential
benefit is reduced dispatch of the Euler tail; its contribution to full-pipeline
latency is not yet measured.
