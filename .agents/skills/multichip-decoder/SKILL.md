---
name: multichip-decoder
description: Parallelize a functional or optimized TTNN decoder layer under models/autoports/<model> across the available multi-chip hardware. Use when producing models/autoports/<model>/tt/multichip_decoder.py, choosing a tensor/mesh parallel strategy, comparing to the single-chip TTNN baseline, and leaving correctness plus performance evidence.
---

# Multichip Decoder

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage takes a trusted and p single-chip decoder and makes it run tensor-parallel over the full target mesh available on this machine.

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

Choose a strategy for the hardware. For 1D meshes up to 8 chips, 1D tensor parallelism the starting point. For Galaxy-class meshes, make a model-specific 2D plan. For dense models this might be using 2D matmuls, for MoE models calculate whether we can afford to replicate the expert weights to allow multiple active experts to run in parallel ala the gpt_oss implementation - but when making this decision take into account that we will need to be able to do so for all the layers in the model, not just for one decoder layer!

Keep the residual input/output layout chainable across decoder layers. Your decoder's output and input formats should be the same.

Get correctness before chasing speed. Compare multi-chip TTNN output to the single-chip TTNN baseline with identical synthetic or real weights, inputs, page tables, and positions. This isolates sharding and collective bugs from HF-vs-TTNN numerical differences.

Treat RMSNorm, KV-cache layout, head ownership, bias placement, padding/slicing, and collective axes as first-class correctness questions. Most multi-chip bugs live there.

Async CCLs and semaphores are tricksy little hobbitses. Read the relevant docs and above all look and and consider reusing the CCL/semaphore helper classes that tt_transformers / gpt_oss / deepseek_v3 use.

## Evidence To Leave

Final multi-chip evidence should show:

- Which single-chip baseline was used and why.
- Target mesh shape, hardware, and chosen parallelization strategy.
- Multi-chip prefill and decode PCC against the single-chip TTNN baseline.
- Paged KV-cache behavior on the target mesh.
- Warmed trace replay for decode on the target mesh.
- Determinism or stress coverage appropriate to the implementation risk.
- Runtime fallback audit remains clean.
- Watcher-clean run (if it has a false positive explain why it's a false positive)
- Warmed single-chip baseline and multi-chip latency, speedup, and efficiency.
- `tt-perf-report` output with the main communication, DRAM, compute, and data-movement findings.
