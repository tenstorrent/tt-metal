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

Reuse or wrap the functional implementation from models/autoports/<model>/tt/functional_decoder.py when that is the clearest path, but make sure tests and perf evidence exercise the optimized path. Also: your goal is to produce an optimized implementation. That often means doing more than just tweaking things in the functional path and require reimplementing some or all of its forward pass - in those cases feel free to copy the code into your optimized class as a starting point and change it as much as you need to!

## How To Approach It

Start from the functional decoder report and code. Reproduce enough of the baseline to know current PCC, sequence limits, cache behavior, trace behavior, and latency.

Read and follow the advice in `tech_reports/LLMs/llms.md` - optimize on-device performance and particularly for decode measure the performance of traced execution. If there are significant op/host gaps you can note them but you should still follow the steps below to optimize on-device performance. Always perform optimization using real model shapes, do not use reduced shapes!

A note on the term "sharding" - tt-metal uses this to mean two things. On-device sharding (e.g. L1-sharded activations, DRAM-sharded weights) are sharded across the cores/dram banks of a single device (which is a grid of cores). You should absolutely consider these as in-scope for this stage! Multi-chip sharding (e.g. with a mesh mapper) is about distributing tensors across multiple devices in a mesh. Any time tt-perf-report mentions sharding it is probably talking about on-device sharding and is in-scope for you.

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
- Watcher still clean. Watcher should be run by setting TT_METAL_WATCHER=10, don't skip asserts or anything.
- Optimization checklist:
-[ ] Decoder path fully traced with no host fallbacks
-[ ] Decode activations generally width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
-[ ] Prefill activations generally DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
-[ ] Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives where the target model fits their contracts.
-[ ] Explicitly configured `memory_config`, `program_config`, and `compute_kernel_config` for important ops.
-[ ] Shard specs and core grids that divide tensor dimensions cleanly into tiles where possible, code grids as large as this and the model/hardware allows.
-[ ] DRAM-sharded decode matmuls
-[ ] For MoE models: optimized the routed active-expert path using `all_to_all_dispatch_metadata` + `moe_compute` where the model/hardware fits, including packed W0/W1/W2 weights, expert mapping, combine/reduce, and no dense all-expert runtime path.

If this checklist is not completed, take this as a sign that you should go back and perform those optimization steps to improve on-device performance. For this stage that is what we are most interested in optimizing; op/host gap will be reduced by tracing.
