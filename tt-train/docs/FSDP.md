# Fully Sharded Data Parallel (FSDP) in TTML

`ttml.fsdp` provides a torch-style FSDP implementation that shards model
parameters, gradients, and optimizer state across a mesh axis. It is the
memory-efficient sibling of DDP: same per-rank batch slicing for compute,
but each device only stores `1/N` of every wrapped parameter at rest, with
weights gathered on-the-fly during forward/backward.

If you've used PyTorch's [FSDP2](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
(`torch.distributed.fsdp.fully_shard`), the API is intentionally familiar:

```python
for block in model.blocks:
    ttml.fsdp.fully_shard(block)
ttml.fsdp.fully_shard(model)  # root: wraps only params not owned by a block
```

For a refresher on TTML's other parallelism strategies (DDP, TP, CP, PP) and
the underlying `Mesh` / MGD machinery, see
[DISTRIBUTED_TRAINING.md](./DISTRIBUTED_TRAINING.md).

---

## Quick Start

### 1. Configure the mesh in YAML

To enable fsdp in [the training example](/tt-train/sources/examples/train/train.py), add `enable_fsdp: true` to your training config's `device_config`. The
non-trivial axis of `mesh_shape` will be named `"fsdp"`:

```yaml
# configs/training_configs/training_my_model_fsdp.yaml

training_config:
# ...

device_config:
  enable_fsdp: true
  mesh_shape: [32, 1]
```

`enable_fsdp` and `enable_ddp` are mutually exclusive on a 1D / line
mesh — pick one. On a 2D mesh, they can coexist (HSDP, see
[Hybrid FSDP+DDP (HSDP)](#hybrid-fsdpddp-hsdp)). `enable_fsdp` can also
coexist with `enable_tp` on a 2D mesh (see
[Combining with TP](#combining-with-tp)).

### 2. Wrap the model before creating the optimizer

In your training script (the nano_gpt example does this for you, gated on
`device_config.enable_fsdp`):

```python
# After model creation, BEFORE create_optimizer.
if device_config.enable_fsdp:
    for block in model.blocks:
        ttml.fsdp.fully_shard(block)
    ttml.fsdp.fully_shard(model)

# Optimizer state is now allocated against the sharded parameter shapes.
optimizer = create_optimizer(model, yaml_config)
```

The order matters: `fully_shard` rewrites each parameter's `m_value` to its
local shard, so the optimizer's `zeros_like(param)` allocations are sized
for the shard rather than the full tensor.

### 3. Run training as usual

The training loop is unchanged from the DDP path:

```python
optimizer.zero_grad()
logits = model(input_tokens, mask)
loss = ttml.ops.loss.cross_entropy_loss(logits, targets, reduce=ttml.ops.ReduceType.MEAN)
loss.backward(False)

ttml.sync_gradients(model.parameters())  # no-op on a pure-FSDP mesh
optimizer.step()
```

`fully_shard` installs forward hooks that all-gather weights into full
shape before each forward pass and reduce-scatter gradients back into shard
shape after each backward pass — both transparent to the rest of the loop.

---

## API: `fully_shard`

```python
def fully_shard(
    module: AbstractModuleBase,
    shard_dim: Union[int, Literal["auto"]] = "auto",
    mesh_axis: str = "fsdp",
    reshard_after_forward: bool = True,
) -> AbstractModuleBase
```

Wraps `module` in place and returns it. After the call:
- Every parameter owned by `module` (transitively, but excluding
  parameters owned by *nested* `fully_shard`-ed submodules) is sharded
  along `shard_dim` across `mesh_axis`.
- `module.forward` is replaced with a hooked version that gathers and
  reshards on every call.
- Two convenience methods are attached:
  - `module.unshard()` — manually all-gather every managed parameter.
  - `module.reshard()` — manually swap them back to local shards.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `module` | required | An `AbstractModuleBase` instance (block, root model, etc.). |
| `shard_dim` | `"auto"` | Tensor dim to shard along, or `"auto"`. Auto picks `rank-2` (the typical "first matmul weight dim" for `[1, 1, O, I]` weights), falls back to `rank-1` if `rank-2` is already taken by another mesh axis (e.g. TP) or has size 1. Parameters whose chosen dim is not divisible by the FSDP axis size are skipped with a warning. |
| `mesh_axis` | `"fsdp"` | Name of the mesh axis to shard across. Must exist on the mesh and have size > 1. Kept distinct from `"dp"` so a 2D mesh `("fsdp", "dp")` cleanly supports hybrid sharded data parallel later. |
| `reshard_after_forward` | `True` | If `True`, weights are resharded between forward and backward to keep peak memory low; the backward-pre callback re-gathers just in time. If `False`, weights stay gathered between forward and backward — cheaper in CCL but uses more memory. |

### Recommended usage pattern (FSDP2-style root)

```python
for block in model.blocks:
    ttml.fsdp.fully_shard(block)
ttml.fsdp.fully_shard(model)
```

The outer `fully_shard(model)` is a *root* wrapper: it owns only the
parameters that *aren't* already managed by a nested wrapper (typically
`tok_emb`, `ln_fc`, `fc` for a transformer). Each block's wrapper handles
its own parameters independently. This matches PyTorch FSDP2 semantics and
is what gives the canonical "one block gathered at a time" memory profile.

You can wrap any other granularity (e.g. only every other block, or just
the root), but per-block wrapping is the granularity we test against.

---

## How it works under the hood

FSDP splits the model state across `axis_size` ranks, then "unshards"
(all-gathers) parameters just before they are needed and "reshards"
(deallocates the gathered copy) as soon as possible.

### Parameter lifecycle

```
                        ┌───────────────────────────────────┐
                        │       SHARDED  (rest state)       │
                        │  m_value = local 1/N slice        │
                        │  m_grad  = local 1/N slice        │
                        └───────────────────────────────────┘
                                       │
                            forward begins
                                       ▼
                        ┌───────────────────────────────────┐
                        │  pre_forward (Python hook)        │
                        │   - all_gather(local_shard)       │
                        │   - m_value = full tensor         │
                        └───────────────────────────────────┘
                                       │
                          original forward runs against full m_value
                                       │
                                       ▼
                        ┌───────────────────────────────────┐
                        │  post_forward (Python hook)       │
                        │   - m_value = cached local shard  │
                        │   - deallocate gathered tensor    │
                        └───────────────────────────────────┘
                                       │
                            backward begins
                                       ▼
                        ┌───────────────────────────────────┐
                        │  backward_pre (autograd callback) │
                        │   - re-gather m_value             │
                        │   - if m_grad initialized (i.e.   │
                        │     carry-over from a previous    │
                        │     micro-batch backward),        │
                        │     all-gather it too             │
                        └───────────────────────────────────┘
                                       │
                          module's internal backward closures run
                                       │
                                       ▼
                        ┌───────────────────────────────────┐
                        │  backward_post (autograd callback)│
                        │   - reduce_scatter(m_grad)        │
                        │   - mean over axis_size           │
                        │   - m_value = local shard         │
                        │   - m_grad  = local-shard grad    │
                        │   - deallocate gathered weight    │
                        │     and gathered grad             │
                        └───────────────────────────────────┘
                                       │
                            optimizer.step() runs against shards
```

### Why two pairs of hooks?

`pre_forward` and `post_forward` are pure-Python wrappers around the
module's `forward`. Easy.

`backward_pre` and `backward_post` are harder: TTML's autograd doesn't
expose tensor-level hooks the way PyTorch does. So `fully_shard` inserts
two **identity autograd callback nodes** (`ttml.autograd.callback`) into
the graph at module boundaries:

- The output of the wrapped forward is wrapped with a callback whose
  closure runs `backward_pre`. Topologically this node sits *just after*
  the module's internal forward ops, which means in reverse topo order it
  fires *just before* their backward closures.
- The first tensor argument is wrapped with a callback that runs
  `backward_post`. Topologically it sits *before* the internals, so it
  fires *after* every backward closure that contributes to its gradient.

Both callbacks are zero-copy (they share the underlying `tt::tt_metal::Tensor`
handle with their input) and only cost one graph node per FSDP module per
step.

### Initial sharding (`fully_shard` time)

Each replicated parameter is replaced with its local shard via a
host-roundtrip:

1. The full tensor is composed from the mesh as a `float32` numpy array.
2. A new `MeshMapperConfig` is built with the existing placements
   preserved on every other axis (so a TP-sharded weight stays
   TP-sharded), with `Shard{shard_dim}` installed on the FSDP axis.
3. The numpy array is redistributed back to the mesh through a fresh
   `from_numpy` call with that mapper.

---

### Gradient sync

`ttml.sync_gradients(model.parameters(), axis_names=("dp",))` continues to
work exactly like in DDP. For pure-FSDP runs (no `"dp"` axis on the mesh)
it is a no-op: gradients have already been reduce-scattered in
`backward_post`. For a hybrid mesh (`"fsdp"` and `"dp"`), each parameter
is filtered per-axis: FSDP-sharded params skip the `"fsdp"` axis (already
reduce-scattered) but still all-reduce on the `"dp"` axis (replicated
across DP groups). The same call covers both cases, no rewrites needed.

## Gradient accumulation

`fully_shard` supports `gradient_accumulation_steps > 1`. The
`backward_pre` hook auto-detects accumulation by checking
`is_grad_initialized()`; if it is, it gathers gradients.
Note: if `gradient_accumulation_steps == 1`, gradients are destroyed at the end of the step,
and created lazily in the backward pass. This removes the need for gradient all-gathering.

---

## Combining with TP

FSDP and TP are configured on different axes, so a 2D mesh with both
parallelisms enabled splits parameters along TWO dimensions:

```yaml
device_config:
  enable_fsdp: true
  enable_tp: true
  mesh_shape: [4, 8]   # FSDP=4, TP=8
```

In this layout:
- TP weights (e.g. `ColumnParallelLinear`, `RowParallelLinear`) have
  `Shard{tdim}` on the `"tp"` axis applied at construction.
- `fully_shard` then adds `Shard{shard_dim}` on the `"fsdp"` axis.
  `auto` picks a different tensor dim than the one TP already claimed.
- Forward all-gathers across FSDP (full TP-shard); backward
  reduce-scatters across FSDP (averages TP-shards across DP groups).
- TP's own all-reduce / scatter / broadcast on the `"tp"` axis is
  unchanged.

---

## Hybrid FSDP+DDP (HSDP)

FSDP and DDP can coexist on a 2D mesh: the DDP axis replicates the
sharded model state across DP groups, and the FSDP axis shards the
weights within each DP group. The same shape as PyTorch FSDP2's HSDP
(`Mesh((replicate, shard), ("replicate", "shard"))`).

```yaml
device_config:
  enable_ddp: true
  enable_fsdp: true
  mesh_shape: [4, 8]    # axis 0 = "dp" (D=4), axis 1 = "fsdp" (F=8)
```

In this layout:
- The batch is sharded across **both** the `"dp"` and `"fsdp"` axes —
  every device on the mesh sees a unique `B / (D*F)` slice.
- `fully_shard` uses the `"fsdp"` axis (default). Each block's weights
  are sharded F-way *within* a DP group and **replicated D-way** across
  DP groups.
- During backward:
  - The FSDP backward-post hook reduce-scatters grads across the
    `"fsdp"` axis (size F) — same as pure FSDP.
  - `ttml.sync_gradients(params, axis_names=("dp", "fsdp"))` then runs
    a per-param filter: FSDP-managed grads skip the `"fsdp"` axis
    (already reduce-scattered) but all-reduce across `"dp"` to average
    each shard over DP replicas. Any non-FSDP / replicated parameter
    all-reduces on both axes.

---

## Constraints and gotchas

- **Build order.** `fully_shard` must be called before
  `create_optimizer`. The optimizer's state tensors are sized from the
  parameter's shape at construction, so resharding after the optimizer
  exists would leave its state mis-sized. The training script raises if
  you try to use `--resume` or `--model_save_path` with `enable_fsdp:
  true` because the pickle checkpoint format would only capture per-rank
  slices.
- **Muon is unsupported.** `MuonComposite`'s Newton–Schulz update is not
  elementwise; running it on a shard would produce a different result
  than running it on the full weight. The constructor errors out if any
  parameter is sharded.
- **`clip_grad_norm`** raises under FSDP for the same reason it raises
  under TP: the per-rank L2 norm isn't the global norm. A
  sharding-aware clip is on the TODO list.
- **Parameters on the chosen shard dim with size 1** (e.g. RMSNorm
  `gamma` shaped `[1, 1, 1, F]` with `shard_dim` 2) are skipped
  with a warning rather than sharded. They stay replicated. For the
  small norm-style parameters this is the right behavior. If `shard_dim`
  is set to `auto`, it will try to shard on dim 2, and then dim 3 before skipping.

---

## Debugging tips

- **`module.unshard()` / `module.reshard()`** are exposed on every
  wrapped module. Useful to manually inspect a parameter's full value
  outside the training loop.
- **`ttml.fsdp.is_fsdp_managed(param.tensor)`** returns `True` for any
  parameter `fully_shard` has touched. The marker is what
  `sync_gradients` uses to decide which axes to skip per parameter.

---

## TODOs

These are known to be incomplete pieces of the FSDP prototype, in
roughly the order I'd tackle them:

- [ ] **Remove extra AG in FSDP with `RunnerType.MemoryEfficient`.**
  The memory-efficient block runner re-runs the block forward inside backward (gradient
  checkpointing). FSDP's hook placement makes it so that we all-gather and reshard weights on the second
  forward, and then immediately all-gather and reshard on the backward, which is quite foolish since
  we could just keep weights from the second forward AG.

- [ ] **Add prefetching / CCL-compute overlap.** The hook architecture is
  designed for this — each FSDPState already knows its neighbors via
  the natural module ordering. The plan is:
    - In `pre_forward` of block `i`, kick off an *async* all-gather for
      block `i+1` using `ttnn::experimental::all_gather_async` with ccl subdevice.
    - In `pre_forward` of block `i+1`, wait on the prefetched future
      instead of issuing a fresh gather.
    - Symmetric on backward: `backward_pre` of block `i` triggers async
      gather for block `i-1`.

- [ ] **Sharding-aware `clip_grad_norm`.** Square the per-rank shard
  grads, all-reduce the squared-sums on the FSDP axis, take the global
  sqrt, then scale per-rank shards by `min(1, max_norm / global_norm)`.

- [ ] **Lazy init.** The
  current host-roundtrip path is correct but spends ~`num_devices * total_params`
  bytes of host RAM transiently at `fully_shard` time. Plus it doesn't allow for a model with
  weights not fitting in a single chip memory. Need lazy init infra to enable training of a model
  like Qwen-32B or Llama-70B on a single galaxy.

- [ ] **Muon + FSDP.** Newton–Schulz needs the full weight matrix.
  Either materialize full weight inside Muon's step (one extra
  all-gather + reduce-scatter per Muon param per step), or shard
  Muon-managed weights along a different axis so they stay full on
  every FSDP rank.
