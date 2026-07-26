# Sparse MoE training for DeepSeek

Adds on-device Mixture-of-Experts training for the DeepSeek model in tt-train:
sparse MoE (moe_group / moe_ungroup + variable_matmul), expert-parallel expert
sharding, and the training-script wiring to drive it. Merged latest `main`;
sequence-parallelism was moved out to `llama-tp-sp`.

## What's included
- **DeepSeek MoE FFN** with a single `moe_type` selector: `dense | sparse_ep`.
  - `dense` — on-device masked experts, reference / cross-check path.
  - `sparse_ep` — moe_group/ungroup sparse dispatch with the routed-expert list
    partitioned across a mesh axis (SparseMoEEP); each chip stores and runs
    `E / D_ep` experts, saving expert-weight memory linearly in the axis size.
    With no usable EP axis (single chip) it degenerates to single-device sparse
    (EP size 1) via `SparseMoE` — so there is no separate `sparse` mode.
- **One axis knob**: `device_config.moe_axis` selects the mesh axis (registered as
  `moe_ep`, or `"tp"` under full-model TP) that `sparse_ep` partitions experts across.
- **leids built once** in `MoE.__init__` (optional mesh mapper — sharded for EP,
  replicated otherwise), shared by the EP and single-device paths; no per-forward host copy.
- **DP-shared-axis support**: when the EP axis coincides with the data-parallel
  axis, the MoE block all_gathers the batch on entry and scatters on exit, so
  DP+EP can share one axis without an all-to-all token shuffle.
- **Training driver** (`examples/train/`): DeepSeek DP/TP + MoE-EP, full-vocab
  tokenized data via `tools/dataset_to_tokens.py`, plus example configs.

## Validation (Blackhole; loss + MFU, 30-step average over steps 2–30)
Loss coincides where it must — **sparse ≡ dense** (single device) and
**sparse ≡ sparse_ep** routing (8-chip): steps 1–2 bit-identical, step 3 within
one bf16 ULP. SparseMoE-vs-dense forward+backward parity: **12/12 tests pass**.

Measured on a Blackhole Galaxy host (firmware bundle 19.8.1), 30 steps, MFU
averaged over steps 2–30. Absolute MFU is sensitive to host clocks and state —
re-measure before quoting these elsewhere.

| Mode | Model / vocab | Mesh | seq × bs | runner | MFU avg (min–max) | TFLOPS | ms/step |
|---|---|---|---|---|---|---|---|
| sparse_ep (1 chip) | tiny, full vocab | 1×1 | 2048 × 8 | memory_efficient | **11.80%** (10.40–13.10) | 19 | 3229 |
| dense | tiny, full vocab | 1×1 | 2048 × 2 | memory_efficient | 1.06% (0.96–1.25) | 2 | 8975 |
| sparse_ep (1 chip) | nano, char vocab | 1×1 | 2048 × 32 | default | 8.26% (7.18–10.00) | 13 | 1161 |
| sparse_ep (8 chip) | 16B, full vocab | 1×8 | 4096 × 2 | memory_efficient | 3.64% (2.90–4.43) | 47 | 3292 |
| sparse_ep (32 chip) | 16B, full vocab | 4×8 | 4096 × 8 | memory_efficient | 1.60% (1.52–1.69) | 83 | 7401 |
| sparse_ep (32 chip) | 16B, full vocab | 4×8 | 8192 × 4 | memory_efficient | 1.82% (1.70–1.99) | 94 | 7760 |

On one chip, sparse_ep is **11.1×** the MFU of dense at the same seq (11.80% vs
1.06%). On the 4×8 mesh, doubling seq to 8192 while halving batch keeps
tokens/step at 32768 and gains ~14% MFU — attention work grows quadratically
while the MoE/FFN grows linearly, so arithmetic intensity rises.

**Sequence-length ceiling (4×8, DP=4/TP=8):** seq 8192 × bs 4 is the largest that
fits. The binding constraint is the vocab-parallel logits buffer,
`per_DP_rank_tokens × vocab × 2 B`; at 16384 tokens per rank it needs 419 MB per
DRAM bank with only ~153 MB free. Both `seq 8192 × bs 8` and `seq 16384 × bs 4`
fail on that same 3.36 GB allocation, since `batch_size` must be ≥ DP.

### Reproducing these numbers

All rows use the same driver; only the training config differs. `--max_steps 30`
matches the "average over steps 2–30" above (the checked-in configs default to
`max_steps: 20`, so pass the flag explicitly).

```bash
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
export VIRTUAL_ENV=$PWD/python_env PATH="$PWD/python_env/bin:$PATH"
# Multi-chip rows only: the MGD dims must match the config's mesh_shape.
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt-train/configs/mgd/<mgd>.textproto

python tt-train/sources/examples/train/train.py \
    -c <training-config>.yaml --max_steps 30
```

| Row | `<training-config>` | Mesh | `<mgd>` |
|---|---|---|---|
| sparse_ep (1 chip), tiny, full vocab | `tiny_deepseek_single_sparse_ep.yaml` | 1×1 | not needed |
| dense, tiny, full vocab | `tiny_deepseek_single_dense.yaml` | 1×1 | not needed |
| sparse_ep (1 chip), nano, char vocab | `training_shakespeare_nano_deepseek_char.yaml` | 1×1 | not needed |
| sparse_ep (8 chip), 16B, full vocab | `deepseek_16B_8chip_ep.yaml` | 1×8 | `bh_galaxy_1_8_line_line` |

Training configs live in `tt-train/configs/training_configs/`. MFU is printed per
step by the training loop (`MFU: x.xx%`); average steps 2–30 and ignore step 1,
which includes JIT kernel compilation. Override seq/batch without editing configs
via `--sequence-length` / `--batch-size`.

Record the exact host with `tt-smi -ls` (board type + firmware bundle) alongside
any refreshed numbers; MFU is sensitive to clocks and host state.

`seq × bs` are the largest that fit at each setting. The **memory_efficient
runner** (block-activation recompute) is what unlocks the big full-vocab batch:
it frees the activation memory the `[bs, seq, 102400]` logits tensor needs, so
single-device full vocab fits `2048 × 8` (best single-device MFU) rather than
being capped at a tiny batch. On 8 chips the LM head is ColumnParallel, so logits
shard across the axis as well.

## Notes
- SP (sequence parallelism) and the MoE **tensor-parallel** variant are intentionally
  **not** in this PR — SP lives on `llama-tp-sp`; the multi-device MoE path here is
  expert-parallel only.
- Config/CMake/nanobind cleanups keep the diff scoped to the MoE feature.
