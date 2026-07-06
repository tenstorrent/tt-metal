# Sparse MoE training for DeepSeek

Adds on-device Mixture-of-Experts training for the DeepSeek model in tt-train:
sparse MoE (moe_group / moe_ungroup + variable_matmul), expert-parallel expert
sharding, and the training-script wiring to drive it. Merged latest `main`;
sequence-parallelism was moved out to `llama-tp-sp`.

## What's included
- **DeepSeek MoE FFN** with a single `moe_type` selector: `dense | sparse | sparse_ep`.
  - `dense` — on-device masked experts (reference / cross-check).
  - `sparse` — moe_group/ungroup dispatch, single chip.
  - `sparse_ep` — routed-expert list partitioned across a mesh axis (SparseMoEEP);
    each chip stores and runs `E / D_ep` experts, saving expert-weight memory
    linearly in the axis size.
- **One axis knob**: `device_config.moe_axis` selects the mesh axis (registered as
  `moe_ep`, or `"tp"` under full-model TP) that `sparse_ep` partitions experts across.
  With no such axis it degrades to single-chip `sparse`.
- **leids built once** in `MoE.__init__` (optional mesh mapper — sharded for EP,
  replicated otherwise), shared by both sparse variants; no per-forward host copy.
- **DP-shared-axis support**: when the EP axis coincides with the data-parallel
  axis, the MoE block all_gathers the batch on entry and scatters on exit, so
  DP+EP can share one axis without an all-to-all token shuffle.
- **Training driver** (`examples/train/`): DeepSeek DP/TP + MoE-EP, full-vocab
  tokenized data via `tools/dataset_to_tokens.py`, plus example configs.

## Validation (Blackhole; loss + MFU, 30-step average over steps 2–30)
Loss coincides where it must — **sparse ≡ dense** (single device) and
**sparse ≡ sparse_ep** routing (8-chip): steps 1–2 bit-identical, step 3 within
one bf16 ULP. SparseMoE-vs-dense forward+backward parity: **12/12 tests pass**.

| Mode | Model / vocab | Mesh | seq × bs | runner | MFU avg (min–max) |
|---|---|---|---|---|---|
| sparse | tiny, char vocab | 1×1 | 4096 × 8 | memory_efficient | **7.39%** (7.14–7.66) |
| sparse | tiny, full vocab | 1×1 | 2048 × 2 | default | 3.72% (3.14–5.07) |
| dense  | tiny, full vocab | 1×1 | 2048 × 2 | memory_efficient | 0.91% (0.80–1.03) |
| sparse_ep | 16B, full vocab | 1×8 | 4096 × 2 | memory_efficient | 2.37% (1.59–2.99) |

`seq × bs` are the largest that fit at each setting. Single-device **full vocab
is logits-bound** — the `[bs, seq, 102400]` logits tensor caps `seq×bs`, which is
why char vocab (negligible logits) reaches ~7.4% at 4096×8 while full vocab tops
out at 2048×2 (and is noisy at that tiny, ~3 s-per-step size). On 8 chips the LM
head is ColumnParallel, so logits shard across the axis and full vocab + longer
seq coexist.

## Notes
- SP (sequence parallelism) and the MoE **tensor-parallel** variant are intentionally
  **not** in this PR — SP lives on `llama-tp-sp`; the multi-device MoE path here is
  expert-parallel only.
- Config/CMake/nanobind cleanups keep the diff scoped to the MoE feature.
