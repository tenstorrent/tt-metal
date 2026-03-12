# 6.1 MoE Overview

## What Is Mixture-of-Experts?

A Mixture-of-Experts (MoE) layer replaces a single dense feed-forward network (FFN) with N independent expert networks, each handling a different specialization of the input distribution. At inference time, a learned router selects K of the N experts for each input token — typically K=2 from N=8 or N=64 experts. Only those K experts compute; the rest are skipped.

```
Dense FFN:
  every token  →  [FFN]  →  output

MoE layer:
  every token  →  [Router] → top-K expert indices
                           ↓
                  [Expert 0] (only for tokens routed here)
                  [Expert 1] (only for tokens routed here)
                  ...
                  [Expert N-1] (only for tokens routed here)
                           ↓
              weighted sum of K expert outputs → token output
```

This sparsity is the computational advantage: for the same total parameter count, MoE uses only K/N of the parameters per forward pass. Models like Mixtral-8x7B, DeepSeek-V3, and Qwen-MoE use this to scale parameter count without proportional compute growth.

---

## Why MoE Needs Special CCL Treatment

### The routing asymmetry problem

In a dense model distributed across D devices, every device computes on the same tensor and the all-reduce or all-gather pattern is symmetric: every device sends roughly the same amount of data to every other device.

In MoE, the router assigns each token to K specific experts. Different tokens go to different experts. If expert `i` is on device `j`, then any token selecting expert `i` must be physically moved to device `j` for computation — regardless of which device originated the token. The result is a fully general all-to-all communication:

```
Device 0: tokens [t0, t1, t2, t3]  →  some tokens go to device 0, 1, 2, 3
Device 1: tokens [t4, t5, t6, t7]  →  some tokens go to device 0, 1, 2, 3
Device 2: tokens [t8, t9, t10, t11] → some tokens go to device 0, 1, 2, 3
Device 3: tokens [t12, t13, t14, t15] → some tokens go to device 0, 1, 2, 3
```

Unlike AllGather (every device gets the full tensor) or ReduceScatter (every device gets one shard), the MoE all-to-all has *data-dependent routing*: which token goes where is determined at runtime by the router's output, not by a static topology. A standard AllGather would gather all tokens to all devices — wasteful because most of them are irrelevant to any given device's experts. A ReduceScatter has the wrong semantics entirely.

### tt-metal's solution: sparse AllToAll

`ttnn.all_to_all_dispatch` is a *sparse* all-to-all: it sends each token only to the devices that actually have a selected expert for that token. Devices that have no expert selected for a given token receive a placeholder row (garbage data). The metadata tensor produced by dispatch encodes which rows are real tokens and which are placeholders, so the expert computation and combine phases can distinguish them.

This sparse design means:
- Bandwidth scales with K×tokens×hidden_dim, not D×tokens×hidden_dim
- Expert computation on placeholder rows can be skipped or handled by masking
- The combine phase uses the metadata to correctly route results back to originating devices

---

## The Dispatch → Compute → Combine Pipeline

A complete MoE forward pass has four stages:

```
Stage 1: Router (local, no CCL)
  Each device: [B, S, 1, H] tokens → Router → [B, S, 1, K] expert_indices

Stage 2: Dispatch (all-to-all CCL)
  ttnn.all_to_all_dispatch → [sparse tokens on each device, expert_metadata]

Stage 3: Expert computation (local, no CCL)
  Each device: expert FFNs process only the rows routed to them

Stage 4: Combine (all-to-all CCL, inverse of dispatch)
  ttnn.all_to_all_combine → [combined outputs back to originating devices]

Stage 5: Weighted sum (local, no CCL)
  Each device: multiply expert outputs by router scores, sum K contributions
```

Stages 1, 3, and 5 are local operations. Only stages 2 and 4 require cross-device communication. This structure means the CCL cost of MoE is bounded by two all-to-all collective calls per MoE layer, regardless of how many experts there are.

### ASCII overview of one token's journey

```
Device 0                 Device 1                 Device 2
─────────                ─────────                ─────────
token t0 (expert 2)  →→→→→→→→→→→→→→→→→→→→→→→  placeholder
token t1 (expert 0)  → [FFN_0(t1)]              placeholder
token t2 (expert 5)  →→→→→→→→→ placeholder  →→ [FFN_5(t2)]

After expert compute:
Device 0: result(t1) by FFN_0
Device 2: result(t2) by FFN_5

Combine:
Device 0 ←←←←←←←←←←←←←←←←←←←← result(t0) from Device 1's FFN_2
Device 0 ←←←←←←←←←←←←←←←←←←←← result(t2) from Device 2's FFN_5
Device 0 accumulates: t0 = α*result0(t0) + β*result1(t0)  [weighted sum if K>1]
```

---

## Tensor Shapes and the Dimension Convention

The dimension abbreviations used throughout Ch6 (B, S, H, K, D, A, D[A], E, T) are defined in [Ch3 §3.2 — Tensor dimension conventions](../ch3_intermediate_operations/all_to_all.md#tensor-dimension-conventions-from-nanobind-docstring). See §6.2 for the concrete input and output shapes for each operation.

---

## The Expert Mapping Tensor

The `expert_mapping_tensor` is a one-hot matrix of shape `[1, 1, E, D]` that encodes which expert lives on which device. Row `e` of the matrix has a single 1 in column `d` if expert `e` is on device `d`, and 0 everywhere else:

```
             device 0  device 1  device 2  device 3
expert 0  [    1         0         0         0    ]
expert 1  [    0         1         0         0    ]
expert 2  [    0         0         1         0    ]
expert 3  [    0         0         0         1    ]
expert 4  [    1         0         0         0    ]  ← expert 4 also on device 0
expert 5  [    0         1         0         0    ]
expert 6  [    0         0         1         0    ]
expert 7  [    0         0         0         1    ]
```

This tensor is **fully replicated** — every device holds the same copy. The dispatch kernel uses it to look up the destination device for each (token, expert) pair. The combine kernel uses it to look up the source device for each (result, expert) pair.

> **Gotcha:** The expert_mapping_tensor must be fully replicated across all devices before dispatch. If different devices have different mappings, the routing will be silently incorrect — dispatch does not validate cross-device consistency of this tensor.

---

## The Expert Metadata Tensor

The `expert_metadata_tensor` output from dispatch is the key to making combine work correctly. It is essentially an all-gather of the `expert_indices_tensor`: after dispatch, every device knows the expert selections of every token across all devices.

Shape: `[1, B×D[A], S, K]` (same batch-expanded shape as the sparse token output)

Each row in the metadata tensor corresponds to the same row in the sparse token output. A metadata row contains the K expert indices that token originally selected. During combine, the metadata tells each device:
- Which rows in the sparse post-expert-compute tensor are real (token was dispatched here)
- Where to send the result (back to the token's originating device)
- How to reassemble the K results for the weighted sum

> **Gotcha:** The `expert_metadata_tensor` output from `all_to_all_dispatch` must be passed unmodified to `all_to_all_combine`. Do not reshape, pad, or reorder it. Any transformation breaks the row correspondence that combine relies on.

---

## Expert Load Balancing

In a balanced MoE system, each device receives approximately T×K/D tokens across its E local experts per forward pass. In practice, load imbalance is common:

- Some experts are more popular than others (popular experts get more tokens)
- Certain sequence positions systematically prefer the same expert
- Fine-tuned models may collapse routing to a small subset of experts

### Performance impact

Load imbalance affects CCL performance because:
1. `all_to_all_dispatch` sends tokens in batches. If device `d` receives 3× the average token count, its output tensor row allocation must be sized for the worst case. Placeholder rows fill unused slots — they are transmitted anyway at full bandwidth cost.
2. Expert computation time is proportional to tokens received. The slowest device determines end-to-end latency (pipeline stall).
3. `all_to_all_combine` sends results back. The same imbalance that made dispatch slow makes combine slow.

### Mitigation strategies

1. **Auxiliary load-balancing loss**: train the router with a term that penalizes load imbalance. Most MoE training frameworks (MegaBlocks, FairScale, Megatron-LM) include this by default.

2. **Expert capacity factor**: pre-allocate a fixed number of slots per expert per device. Tokens that overflow the capacity are dropped (with degraded quality). This bounds worst-case bandwidth.

3. **Token dropping with fallback**: dropped tokens skip the expert and use the residual stream directly. Acceptable for inference at the cost of small accuracy degradation.

4. **Static expert-to-device assignment**: assign popular experts to multiple devices (expert replication). This is only feasible if the expert weights can be replicated; it trades memory for load balance.

---

See [§6.2 — Output structure: sparse tokens](dispatch_combine.md#output-structure-sparse-tokens) for how placeholder rows work in practice.

---

*Back to [Chapter 6 Index](index.md)*

*Next: [6.2 Dispatch and Combine](dispatch_combine.md)*
