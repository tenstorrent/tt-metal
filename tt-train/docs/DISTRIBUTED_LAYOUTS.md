# Distributed Training in the Python Stack: Meshes, Layouts, Gradients

Every distributed decision in `ttml`'s Python models, trainers and FSDP is read from two things: the **named mesh**
(which axis is which) and each tensor's **layout** on it (its `TensorTopology`). No module takes a flag describing its
neighbours, and no trainer guesses a layout from the mesh's shape. This page is the reference for that design.

The C++ models (`ttml.models.distributed.*`, the nano_gpt and pipeline-parallel examples) still configure themselves
through `ParallelismContext`; see [DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md).

## The mesh

```python
mesh = ttml.Mesh((2, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)          # validates the MGD, enables fabric, opens the devices
...
ttml.close_device_mesh()
```

`initialize_device(yaml_config)` and the GRPO completers open the same way, naming axes with
`ttml.common.utils.build_mesh(device_config)` (`dp` → `fsdp` → `tp`). Every axis has a **kind**, which decides how
gradients are combined across it:

| Axis | Kind | Gradients across it |
| --- | --- | --- |
| `dp`, `fsdp`, `cp` | data: each rank computes the loss of its own samples | averaged |
| `tp` | model: the ranks compute one loss together | partial sums added |

Other names have no kind and declare one with `ttml.Mesh(..., axis_kinds={"ep": AxisKind.Data})`.
`mesh.batch_axes` are the axes the batch is split across (`dp`, `fsdp`: `cp` splits each sequence instead), and
`mesh.batch_mapper(tdim=0)` is the mapper that splits it.

## Layouts

`ttml.Sharding.from_tensor(t).placements` is the tensor's placement on **each mesh axis**: `PlacementShard(dim)` or
`PlacementReplicate()`. ttnn's whole-mesh mappers label a tensor with one placement for the entire mesh; it is
expanded, so `placements[axis]` is always meaningful. `shard_dim(axis)` and `shape` (the full tensor's) read off it,
and `derive_mapper()` / `gather()` move a tensor on and off the mesh exactly as it is laid out.

Ops propagate labels: an output is `Shard` on every axis where an input is. After a matmul the sharded dim is not
always the output's (`ttml.ops.linear` labels its output features correctly), but whether a tensor **differs across
an axis** always is: that is what gradients rely on.

## Tensor-parallel regions

A tensor-parallel region computes locally with weights sharded across one axis (`ttml.parallel`):

- `whole(x, axis)`: the region's input, made whole on every rank: gathered if `x` is sharded across the axis, as is
  if replicated. Its backward sums the ranks' partial gradients.
- `sum_partials(y, axis, like)`: the region's partial results, added up into `like`'s layout: all-reduce for
  replicated, reduce-scatter for sharded.

`ColumnParallelLinear` makes its input whole; `RowParallelLinear` slices a replicated input itself and sums into the
layout of `like=` (the tensor the region started from); `VocabParallelEmbedding` sums into `layout=`. So a model
chooses the layout of its residual stream **once**, and every region follows it: Llama's `TPStrategy.TENSOR_SEQUENCE`
is a single decision in `Llama.forward`, where the embedding output is sharded along the sequence.

Local compute between the edges can be anything: DeepSeek's MLP runs its fused SwiGLU kernel on sharded weights
between `whole` and `sum_partials`.

Dropout reads the layout too: devices that hold the same values drop the same elements, so a per-device mask is
drawn only when the input differs on every device (`ttml.ops.dropout.dropout`'s default).

## Gradients

One convention: **a tensor replicated across an axis is one value.** Where the ranks' gradients of it agree, that is
its gradient; where they differ, each rank holds a partial sum and the gradient is their total. The collectives'
backward passes apply it (`all_reduce`, `broadcast`, `all_gather` sum a gradient that differs across the axis), and so
does `ttml.sync_gradients(params)`, once per optimizer step, per parameter and axis:

| Parameter on the axis | Axis kind | Gradient | Action |
| --- | --- | --- | --- |
| sharded (TP, FSDP) | any | any | none: the module reduced it |
| replicated | data | any | average (a mean of equal values is a no-op, so the label is not trusted here) |
| replicated | model | differs | sum (sequence parallelism's norm gains, row-parallel bias, LoRA paths) |
| replicated | model | agrees | none |
| replicated | none | differs | error |

`ttml.core.distributed.gradient_reductions(param, mesh.axis_kinds)` lists what the sync will do for a parameter.
`ttml.data_parallel_size()` is the number of ranks it averages over: a loss normalized by a global count divides by it.

`ttml.core.clip_grad_norm` after the sync takes the global norm: each gradient's sum of squares is added up across the
axes it is sharded on and counted once across those it is replicated on.

## Loss

`ttml.parallel.cross_entropy_loss(logits, targets)` reads the logits' layout: the vocabulary-parallel loss across the
axis the last dim is sharded on, plain cross-entropy for replicated logits. `SFTTrainer` uses it by default.
