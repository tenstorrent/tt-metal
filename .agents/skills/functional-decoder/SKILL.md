---
name: functional-decoder
description: Bring up a functionally correct TTNN implementation of HuggingFace transformer decoder layers under models/autoports/<model>. Use when producing models/autoports/<model>/tt/functional_decoder.py, reading the HF decoder architecture, validating paged prefill/decode against the HF reference, and leaving compact correctness and performance evidence.
---

# Functional Decoder Bringup

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage turns the HF decoder layer (or multiple layers if there are different kinds in the HF model) into complete, working, tested TTNN code. It's called "functional" because it should be functionally complete and ready to ship - this is not a minimal prototype, it's the heart of everything else that follows. You should bring up the decoder on a single 1x1 device mesh. No need to use multiple devices at this stage.

Useful reference when needed: `references/decoder-bringup-knowledge.md`. It points to local TTNN model code, cache behavior, trace/profiling patterns, watcher notes, and common failure modes.


## Your Part

Implement:

```text
models/autoports/<model>/tt/functional_decoder.py
```

with a model-appropriate `FunctionalDecoder` that subclasses `models.common.lightweightmodule.LightweightModule` and exposes a real weight-loading boundary:

```python
class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...
    def decode_forward(self, ...): ...
```

The exact forward signatures should fit the model. Keep them keyword-friendly and document them in the final report.

## How To Approach It

Start by understanding the HF model Load `AutoConfig`, inspect the installed `configuration_*.py` and `modeling_*.py`, and identify the decoder layer, attention, RoPE, cache API, MLP/MoE path, residual order, norm behavior, activation, bias flags, head shapes, and layer-kind differences. Trace through the model's execution line by line to make sure you understand it.

Implement correctness first. BF16, tile layout, and DRAM memory are fine while proving semantics. Move weight conversion, reshaping, dtype selection, `ttnn.as_tensor`, and cache construction into `from_state_dict` or setup helpers. Keep runtime prefill/decode free of hidden `torch`, `from_torch`, `to_torch`, or host fallback except explicit test boundaries.

For MoE models, validate the real router/gate and active experts end-to-end. Component tests for gate or experts are useful diagnostics, but the decoder result should include the gate-selected expert path a real model run would use. Target single-user bringup with a routed runtime path: prepare W0/W1/W2 using `ttnn.experimental.moe_compute_utils`, dispatch selected tokens with `ttnn.experimental.all_to_all_dispatch_metadata`, compute experts with `ttnn.experimental.moe_compute`, then use the model-appropriate score-weighted combine/reduce. If `models/common/modules/moe/tt_moe_decode.py` exists in your checkout, start there; otherwise read the DeepSeek MoE optimized path and the `all_to_all_dispatch_metadata` / `moe_compute` tests.

When PCC is low, debug it. Split the decoder into components, check HF parity, raise fidelity where useful, simplify the failing shape, and keep narrowing until the cause is understood. If the cause is a tt-metal bug, make a reproducer or an on-branch workaround and record the evidence.

## Evidence To Leave

Use `PCC >= 0.995` as the default acceptance bar for prefill and decode unless the model gives a concrete reason to choose a different bar. Final evidence should cover:

- HF-vs-TTNN prefill and decode PCC and performance for each meaningful layer kind.
- Prefill and decode on-device performance measured from warmed runs (using traced execution for decode) - these should have perf report outputs to prove it.
- Paged KV-cache behavior, including page table and current-position handling.
- Full advertised prefill/decode sequence length (test the longest you can fit if the reference model's maximum is too large for the device to fit)
- Determinism for repeated identical inputs tested.
- Evidence there are no torch or host calls required within a single prefill or decode pass - everything runs and stays on device.
- Watcher-clean run when the environment supports it, or a clear reason it was not run. Watcher should be run by setting TT_METAL_WATCHER=10, don't skip asserts or anything.
