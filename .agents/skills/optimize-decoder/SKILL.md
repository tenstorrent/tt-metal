---
name: optimize-decoder
description: Optimize a correct TTNN decoder layer under models/autoports/<model>. Use after functional-decoder bringup when producing models/autoports/<model>/tt/optimized_decoder.py, preserving correctness while improving layout, precision, sharding, program configs, data movement, and warmed latency with tt-perf-report evidence.
---

# Optimize Decoder

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage takes a decoder that already works and makes it faster without making it wrong. Here is some opportunity for exploration, insight and deep-dive performance engineering - enjoy yourself and make a real difference!

Useful reference when needed: `references/optimization-knowledge.md`. It collects TTNN optimization paths, precision defaults, DRAM-sharded matmul guidance, and `tt-perf-report` notes.

## Your Part

Implement:

```text
models/autoports/<model>/tt/optimized_decoder.py
```

with an `OptimizedDecoder(LightweightModule)` that preserves the usable interface of `FunctionalDecoder`:

```python
class OptimizedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...
    def decode_forward(self, ...): ...
```

Reuse or wrap the functional implementation when that is the clearest path, but make sure tests and perf evidence exercise the optimized path.

## How To Approach It

Start from the functional decoder report and code. Reproduce enough of the baseline to know current PCC, sequence limits, cache behavior, trace behavior, and latency.

Profile warmed prefill and decode separately. Use `tt-perf-report` as a conversation with the hardware, not as an oracle: classify bottlenecks, try applicable advice, keep changes that improve the target without unacceptable correctness or complexity cost, and record why rejected advice was rejected. We'd like to improve tt-perf-report and its advice to be more useful so please call out potential improvements in your final report.

Tune precision and fidelity one group at a time so regressions can be assigned. A common starting point is BF16 activations and norms, BFP8 attention/MLP weights, BFP8 KV cache if PCC allows it, and selective BFP4 trials for MLP/expert weights. After that follow tt-perf-report, read the kernels, explore and be methodically creative until you are satisfied we've got everything out of the hardware that we can without rewriting the ttnn ops themselves!

## Evidence To Leave

Final optimized evidence should show:

- Functional checks still pass against the optimized path.
- Prefill and decode PCC remain at the functional acceptance bar, with any material delta explained.
- Paged KV-cache and warmed trace replay still behave correctly.
- Runtime fallback audit remains clean.
- Watcher-clean run when the environment supports it, or a clear reason it was not run.
- Stress or repeated-run coverage appropriate to the risk of the changes.
- Warmed prefill and decode latency before/after optimization.
- `tt-perf-report` output and the main performance conclusions.
- Optimization checklist:
-[ ] Decoder path fully traced with no host fallbacks
-[ ] Decode activations generally width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
-[ ] Prefill activations generally DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
-[ ] Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives where the target model fits their contracts.
-[ ] Explicitly configured `memory_config`, `program_config`, and `compute_kernel_config` for important ops.
-[ ] Shard specs and core grids that divide tensor dimensions cleanly into tiles where possible, code grids as large as this and the model/hardware allows.
-[ ] DRAM-sharded decode matmuls
-[ ] For MoE models: optimized the routed active-expert path using `all_to_all_dispatch_metadata` + `moe_compute` where the model/hardware fits, including packed W0/W1/W2 weights, expert mapping, combine/reduce, and no dense all-expert runtime path.
