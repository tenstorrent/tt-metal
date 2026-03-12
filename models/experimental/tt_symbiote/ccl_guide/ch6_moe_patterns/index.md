# Chapter 6: MoE-Specific Patterns

Mixture-of-Experts (MoE) models activate only a sparse subset of expert networks per token, creating communication patterns that no symmetric collective can handle efficiently. This chapter covers the full dispatch→compute→combine pipeline and the DeepSeek-specific broadcast primitive.

All operations in this chapter are under `ttnn.` (not `ttnn.experimental.`) except `deepseek_minimal_broadcast`, which is under `ttnn.experimental.`.

---

## Table of Contents

| Section | File | Description |
|---------|------|-------------|
| 6.1 | [MoE Overview](moe_overview.md) | Sparse activation, expert routing, why MoE needs special CCL treatment, token metadata |
| 6.2 | [Dispatch and Combine](dispatch_combine.md) | Deep dive into AllToAllDispatch + AllToAllCombine: full MoE forward pass, `local_reduce`, load imbalance, worked example |
| 6.3 | [DeepSeek Patterns](deepseek_patterns.md) | `deepseek_minimal_broadcast`, DeepSeek MoE architecture differences, MoE optimization patterns |

---

## New parameters in Ch6

| Parameter | Type | Appears in | Purpose |
|-----------|------|------------|---------|
| `expert_indices_tensor` | `ttnn.Tensor [B,S,1,K]` | `all_to_all_dispatch` | Top-K expert rankings for each token |
| `expert_mapping_tensor` | `ttnn.Tensor [1,1,E,D]` | both ops | One-hot expert→device mapping; fully replicated |
| `expert_metadata_tensor` | `ttnn.Tensor [B,S,1,K]` | `all_to_all_combine` | Gathered metadata output from dispatch; drives combine routing |
| `output_concat_dim` | `int` (default 1) | `all_to_all_dispatch` | Dimension along which dispatched tokens are concatenated (1=batch, 2=sequence) |
| `output_shard_dim` | `int` (default 1) | `all_to_all_combine` | Dimension along which combined output is sharded |
| `local_reduce` | `bool` (default False) | `all_to_all_combine` | Whether expert outputs are already locally reduced before combine |
| `sender_coord` | `MeshCoordinate` | `deepseek_minimal_broadcast` | Coordinate of the device that holds the tensor to broadcast |

---

## Relationship to Chapter 3

Chapter 3 §3.2 introduced the AllToAllDispatch and AllToAllCombine APIs. This chapter builds on that foundation:

- §3.2 covers: concept, API signatures, `AllToAllTransferType` (FullPacket vs PageByPage), kernel structure, basic example
- Ch6 covers: full MoE forward pass structure, metadata tensor lifecycle, `local_reduce` semantics, load imbalance handling, performance tuning, DeepSeek-specific patterns

Cross-references to §3.2 are used rather than re-explaining the API basics.

---

## Prerequisites

- Chapter 3 §3.2: AllToAllDispatch and AllToAllCombine API basics
- Chapter 2 §2.1: AllGather parameter conventions (topology, num_links, cluster_axis)
- Chapter 4 §4.1: Async model and SubDevice (for overlap with expert computation)

---

*Next: [6.1 MoE Overview](moe_overview.md)*
