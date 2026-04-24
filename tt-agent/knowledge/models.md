# Models

Topic knowledge for model-level code on tt-metal — external references
only today; extend with patterns and traps as they surface.

## External references

- **Production LLM layers** — `tt-metal/models/tt_transformers/tt/`
  Attention, MLP, layernorm, embedding — how production models are
  structured. LightweightModule pattern, weight loading, forward pass is
  device-only.
- **Shared model utilities** — `tt-metal/models/common/`
  Weight loading helpers, PCC validation (`assert_with_pcc`), tensor
  save/load.
- **Model demos and end-to-end tests** — `tt-metal/models/demos/`
  Full inference demos including multi-chip setups.
- **Model-level tests** — `tt-metal/models/tt_transformers/tests/`
  How to test a model block in isolation: load weights, run forward,
  check PCC.
- **Model bringup tech report** — `tt-metal/tech_reports/ttnn/TTNN-model-bringup.md`
  Step-by-step protocol for porting a new model to TT hardware.
- **Advanced performance optimization** — `tt-metal/tech_reports/AdvancedPerformanceOptimizationsForModels/`
  Trace mode, 2CQ, op fusion, math fidelity tuning.
