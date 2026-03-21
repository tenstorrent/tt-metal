# Stage 1 Proposal for #38114

This PR provides the Stage-1 design proposal requested in bounty issue #38114.

## Goal
Build a modular auto-tuning infrastructure that chooses the best-performing matmul configuration for a given input shape/device topology and can be retrained when kernels/config spaces evolve.

## Proposed Architecture

1. **Feature extractor**
   - Input tensor features: M/N/K, dtype, layout, batch dims.
   - Hardware/runtime features: device count, mesh topology, memory limits.
2. **Candidate generator**
   - Enumerate valid execution configs for single/multi-device paths.
3. **Cost model**
   - Learn latency prediction from benchmark traces.
   - Fallback to rule-based heuristic when confidence low.
4. **Planner API**
   - `ttnn.matmul_auto(a, b, *, config_override=None)`
   - Torch-like API dispatching to selected config.
5. **Validation harness**
   - Correctness tests over random shape sweeps.
   - Performance tests proving picked config beats nearby alternatives.
6. **Retraining pipeline**
   - Re-benchmark selected config sets after backend changes.
   - Refit model and refresh inference artifact.

## Milestone Mapping

- **Stage 1:** this design + integration points and algorithm selection
- **Stage 2:** working prototype + preliminary planner + initial tests
- **Stage 3:** robust planner, perf evidence, model integration, retraining docs

## Initial Algorithm Choice

- Gradient-boosted tree regressor for latency prediction (fast, interpretable)
- Confidence gating with fallback heuristic for out-of-distribution inputs
- Optional small NN head after baseline if needed

## Deliverables in later stages

- planner API implementation
- functional and performance tests
- example integration in at least one model path
- update/retraining guide
