### Landed in-repo perf levers (model-faithful @48: ~1.08 → ~18 t/s)

The DiffusionGemma in-repo, precision-neutral perf levers already landed on `diffusion-gemma-function`, with their commits:

| # | Lever | Effect | Commit(s) |
|---|---|---|---|
| 1 | **Sparse MoE (token-gather)** — replaces dense-128 experts with true top-8 token-gather (the single biggest win) | MoE ~13× (dense 137.6 → ~10.5 ms/layer) | [`e98fabaaff4`](https://github.com/tenstorrent/tt-metal/commit/e98fabaaff4db87c9052d675a0407ebb4625d8d8) lever-A GO (12.5×, PCC 0.9997) · [`1d1ccd93a8b`](https://github.com/tenstorrent/tt-metal/commit/1d1ccd93a8bc1e8a5112bd608107e1ced59af700) land true-sparse (9.3× traced step) |
| 2 | **Matmul-geometry tuning (OPT-004, `DG_SPARSE_MOE_TUNED`)** — program-config / core-grid / L1-output tuning of the 5 sparse-MoE matmuls; verified exhausted | MoE ~3.47× vs auto (PCC 0.99967) | [`014c47177f7`](https://github.com/tenstorrent/tt-metal/commit/014c47177f70dd18e656a68b0fad5a9c7027c570) tuned geometry (opt-in) · [`9c5c999fb80`](https://github.com/tenstorrent/tt-metal/commit/9c5c999fb80fa207c3cb3888d28d85d9214c5097) default ON |
| 3 | **Terminal trim (`DG_DEDUP_ARGMAX`, ROW_MAJOR argmax)** — dedup the 2 full-vocab argmaxes; ROW_MAJOR multi-core argmax over 262144 | argmax 1240 → 14.4 ms/op (2×/step) | [`474713ec259`](https://github.com/tenstorrent/tt-metal/commit/474713ec259d96c1f94b58860574630d9669acab) DEDUP_ARGMAX · [`a39e8d63c1b`](https://github.com/tenstorrent/tt-metal/commit/a39e8d63c1b1068094dc324c9cd6153d67043fb0) ROW_MAJOR argmax + trace-safe terminal |
| 4 | **Traced denoise loop (`DG_DENOISE_TRACED`)** — trace-safe fixed-step serving loop with device canvas feedback | 2.72× traced vs eager; model-faithful @48 = 17.92 t/s | [`d25626f2636`](https://github.com/tenstorrent/tt-metal/commit/d25626f2636979180194f0016a7bc9271ddbb56f) wired into serving (bit-exact 58.29 t/s @12) · [`35e70fd1225`](https://github.com/tenstorrent/tt-metal/commit/35e70fd12259e47fd63f4573d1373ed1a9cbcc59) @48 = 17.92 (2.72×) |
| 5 | **Batched commit** — batched single-prefill commit replaces the 256-token decode-append; now default | 2.54× on the block; commit no longer dominates | [`3d71dee8a97`](https://github.com/tenstorrent/tt-metal/commit/3d71dee8a97ae7504da38ba24a7c86c611f7eb7f) land batched commit as default |

**Cumulative:** model-faithful @48 throughput went from ~1.08 t/s to ~18 t/s.

The remaining distance to 100 t/s is out-of-gate or gated on fidelity: fewer denoise steps (the single biggest multiplier, ~2–2.4×) is blocked by the #48291 sparse-MoE fidelity ceiling (early-halt never fires); an upstream fused gather-experts-combine MoE kernel would need to be upstreamed; and bf8 experts were measured and **fail** the diffusion-decision fidelity gate (see the dg-07 datatype-sweep, #47475). So the precision-neutral in-repo @48 ceiling stands at ~18 t/s.
