# AutoFix: public sampling temperature

Date: 2026-09-12. Diagnosis and isolated host-only verification by the sampling/trace audit agent; implementation changed by the coordinator.

The wrapper exposed public `temperature=T` but passed T directly into `TTSampling.reset_params`. The native sampling API documents its `temp` tensor as `1/T` (`sampling_nanobind.cpp:45`), and its compute kernel multiplies logits by that field (`device/kernels/compute/sampling.cpp:458-464`). The common parameter formatter performs the reciprocal at `models/common/sampling/generator.py:605`.

Consequently a requested T=0.8 actually produced effective temperature 1.25. Greedy T=1 and token-range-only tests cannot detect the error. The smallest repair passes `[1.0 / temperature] * 32`, retaining the persistent tensor update, seed handling and trace lifecycle.

The coordinator applied that repair. `sampling_params_host.json` records an AST-extracted test of the actual changed `set_sampling_params` method with standard-library fake Torch/vector/sampler objects. No Torch or TTNN module, model, or device was imported or run.

| Public temperature | Expected device multiplier | All 32 lanes |
| --- | --- | --- |
| 0.5 | 2.0 | Pass |
| 0.8 | 1.25 | Pass |
| 1.0 | 1.0 | Pass |

The same checks verified k=8 and p=0.9 remained unchanged, seed 12 refreshed the existing seed tensor with values 13 through 44, and unchanged sampler strategy released no trace. Generator SHA256 at test time: `718ee9c9c3a7743eca5968f061e3470571f4290f60b01137817af161c82a02f9`.

This proves public-to-native parameter translation in host control flow. It does not measure stochastic quality, native BF16 rounding, RNG replay reproducibility, or performance; hardware validation belongs to the coordinator's contract runs.
