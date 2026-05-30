---
name: multichip-decoder
description: Parallelize a functional or optimized TTNN decoder layer under models/autoports/<model> across the available multi-chip hardware. Use when producing models/autoports/<model>/tt/multichip_decoder.py, choosing a tensor/mesh parallel strategy, comparing to the single-chip TTNN baseline, and leaving correctness plus performance evidence.
---

# Multichip Decoder

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage takes a trusted single-chip decoder and makes it run tensor-parallel over the full target mesh available on this machine.

Useful reference when needed: `references/parallelization-knowledge.md`. It points to GPT-OSS, `models/common`, CCL, RMSNorm, MoE, and Galaxy examples.

## Your Part

Implement:

```text
models/autoports/<model>/tt/multichip_decoder.py
```

with a `MultichipDecoder(LightweightModule)` that preserves the single-chip decoder interface unless the final report explains a deliberate, chainable contract change:

```python
class MultichipDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...
    def decode_forward(self, ...): ...
```

Use previous single-chip implementation as the baseline: optimized if it exists and passes, otherwise functional.

## How To Approach It

Read the baseline code and report first. Identify layer kinds, tested shapes, sequence limits, layouts, precision, cache behavior, trace behavior, and latency.

Choose a strategy for the hardware. For 1D meshes up to 8 chips, 1D tensor parallelism the starting point. For Galaxy-class meshes, make a model-specific 2D plan. For dense models this might be using 2D matmuls. For MoE models, make the routed decode pipeline part of the mesh plan: choose the dispatch axis, replicated or TP axis, expert mapping, packed-weight placement, combine/reduce path, and whether the model can afford expert replication across all layers. Prefer the `all_to_all_dispatch_metadata` + `moe_compute` path or the common MoE decode wrapper when it fits.

A good multichip design needs to keep two competing objectives in mind: apply as much flops + memory bandwidth to the workload as possible and minimize the impact of the additional data movement this requires. In practice this means considering: (1) which weight tensors to fracture across which mesh dimensions and (2) how and when to move the activations between them. The second one is somewhat analgous to the choices required for within-chip sharding performed in the decoder optimization stage. It is important to think about the residual stream and the interfaces to each module - we want to minimize time spent in collective operations here and this might require revisiting configurations and choices made in the decoder previously. All of this is on the table, our goal is the fastest possible multichip implementation, not the most convenient given our starting point.

The next step will stack decoders together, so we should ensure that the decoder's output and input formats are the same as each other, but they do NOT need to be the same as the single-chip's inputs/outputs. Clearly given our goal, it's absolutely fine and in many cases desirable for the multichip implementation to maintain chip-sharded activations on its inputs and outputs, something the single-chip version cannot do. The right thing to do in this case is to modify the test code itself to provide the correct sharding for the multi-chip decoder as in the final model as long as this will only need to be performed once at the entry/exit to the entire model and not at the boundary to every layer. If there are multiple decoder layer types it's probably important that their input+output shardings and datatypes are compatible with each other, however!

The typical advice is "get correctness before chasing speed" but do not let this mislead you: our only acceptable goal is an extremely high-performance implementation. Pursuing the correctness of a design that clearly introduces unnecessary communication patterns may not be a good use of time. To evaluate correctness we recommend comparing multi-chip TTNN output to the single-chip TTNN baseline with identical synthetic or real weights, inputs, page tables, and positions. This isolates sharding and collective bugs from HF-vs-TTNN numerical differences.

Treat RMSNorm, KV-cache layout, head ownership, bias placement, padding/slicing, and collective axes as first-class correctness questions. Most multi-chip bugs live there.

Async CCLs and semaphores are subtle. Read the relevant docs and consider reusing the CCL/semaphore helper classes that `tt_transformers`, `gpt_oss`, and `deepseek_v3` use.

Reemmber: always do a tt-perf-report pass before finishing this stage and optimize optimize optimize - our intent is to provide the highest utilization of the hardware that we reasonably can short of writing new ttnn ops!

## Evidence To Leave

Final multi-chip evidence should show:

- Which single-chip baseline was used and why.
- Target mesh shape, hardware, and chosen parallelization strategy for both weights and activtations with calculated evidence showing why this is optimal for the model and hardware vs other sensible options.
- Multi-chip prefill and decode PCC against the single-chip TTNN baseline.
- Paged KV-cache behavior on the target mesh.
- Warmed trace replay for decode on the target mesh.
- Determinism or stress coverage appropriate to the implementation risk.
- Runtime fallback audit remains clean.
- Watcher-clean run (`export TT_METAL_WATCHER=10`). If it has a false positive explain why it's a false positive.
- Warmed single-chip baseline and multi-chip latency, speedup, and efficiency.
- `tt-perf-report` output with the main communication, DRAM, compute, and data-movement findings.
