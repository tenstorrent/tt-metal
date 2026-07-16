# BGE-M3 encoder-only SDPA scaffold

Status: **unverified parity scaffold; not integrated into the model**.

This directory establishes a no-`_ttnn.so`-rebuild path for the exact retained
N300 DP=2 encoder attention shape:

- Q `[6, 32, 4096, 64]` BF8
- K `[6, 16, 8192, 64]` BF4
- V `[6, 16, 8192, 64]` BF8
- non-causal, no mask, scale 1
- Q128/K2048, LoFi, FP32 destination
- fixed 8x8 worker grid

`op.py` mirrors the relevant constants, CB allocation, compile-time arguments,
and per-core global-Q scheduling from the production `SDPAProgramFactory`.
The three model-local kernel entrypoints currently include the production JIT
kernels unchanged. This separates host-descriptor parity from later kernel
specialization.

## Why forwarding semaphores are omitted

There are 6144 Q work units: `B6 * HQ32 * 32 Q chunks`. Dividing across 64
cores gives exactly 96 units/core, or three complete heads/core. No head crosses
a core boundary, so the production KV forwarding chains have no participants.
The Python descriptor therefore sends fourteen zero chain fields per core and
does not allocate the three forwarding semaphores.

This assumption must be validated on silicon before any model integration.

## First takeover steps

1. Import `build_encoder_sdpa_descriptor` only; check Python binding names and
   descriptor construction without wiring `attention.py`.
2. Run the gated parity probe in
   `models/demos/wormhole/bge_m3/tests/perf/encoder_sdpa_parity_probe.py`.
3. Resolve any binding differences, especially:
   - `KernelDescriptor(defines=...)` map conversion;
   - `TensorAccessorArgs` placeholder layout for absent optional tensors;
   - whether inactive CB id `0xFFFFFFFF` survives Python integer conversion;
   - whether explicit forwarding semaphore descriptors are required even when
     every runtime participant flag is zero.
4. Require stock-equivalent PCC and output shape.
5. Compare one warm repeat launch, program-cache reuse, and trace capture/replay.
6. Profile device duration. Parity is acceptable only within measurement noise.
7. Only then add an explicit optimization option in `attention.py`.

## Specialization order after parity

1. Copy production kernel bodies into this directory; stop including them.
2. Add internal zones around QK, row max, sub/exp/sum, PV, correction, and final
   normalization.
3. Remove inactive encoder modes and optional-accessor slots only after confirming
   they affect generated code or L1.
4. Explore CB lifetime aliasing to recover enough L1 for Q160/K2048 while
   retaining double-buffered K/V.
5. Optimize the measured dominant reduction/SFPU region one change at a time.

Do not infer a speedup from specialization alone: most production mode branches
are already compile-time eliminated.
