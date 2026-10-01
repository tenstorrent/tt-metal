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

Two optional knobs trade memory or a few cores for fewer exposed collectives (details in
[Fewer collectives](#fewer-collectives-keeping-blocks-gathered) and
[Overlapping collectives with compute](#overlapping-collectives-with-compute)):

```yaml
device_config:
  enable_fsdp: true
  mesh_shape: [8, 1]
  fsdp_keep_gathered_gib: 2.0          # keep the last blocks that fit in 2 GiB gathered (.inf = all)
  fsdp_overlap_collectives: true       # CCL sub-device + second command queue
  fsdp_ccl_subdevice: {columns: 1}     # default; {rows: 1} has faster collectives but a 12x9 grid
  fsdp_overlap_lookahead: 2            # blocks the collective queue may run ahead (2x that many gather slots)
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
    keep = ttml.fsdp.blocks_to_keep_gathered(model.blocks, device_config.fsdp_keep_gathered_gib)
    for i, block in enumerate(model.blocks):
        ttml.fsdp.fully_shard(block, reshard_after_forward=i not in keep)
    ttml.fsdp.fully_shard(model, reshard_after_forward=False)  # root params are live all step anyway

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
| `shard_dim` | `"auto"` | Tensor dim to shard along, or `"auto"`. Auto considers `rank-2` (the typical "first matmul weight dim" for `[1, 1, O, I]` weights) then `rank-1`, drops dims already taken by another mesh axis (e.g. TP), of size 1, or not divisible by the FSDP axis size, and prefers the dim whose per-rank shard is **tile-aligned** (a multiple of 32) — see [Tile-aligned shards](#tile-aligned-shards). Parameters with no usable dim are skipped with a warning. |
| `mesh_axis` | `"fsdp"` | Name of the mesh axis to shard across. Must exist on the mesh and have size > 1. Kept distinct from `"dp"` so a 2D mesh `("fsdp", "dp")` cleanly supports hybrid sharded data parallel later. |
| `reshard_after_forward` | `True` | If `True`, weights are resharded between forward and backward to keep peak memory low; the backward-pre callback re-gathers just in time. If `False`, weights stay gathered between forward and backward — one all-gather per parameter per step fewer, more memory. See [Fewer collectives](#fewer-collectives-keeping-blocks-gathered). |

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
                        │   - if m_grad initialized (shard  │
                        │     carried over from a previous  │
                        │     micro-batch), set it aside    │
                        │     and clear m_grad              │
                        └───────────────────────────────────┘
                                       │
                          module's internal backward closures run
                                       │
                                       ▼
                        ┌───────────────────────────────────┐
                        │  backward_post (autograd callback)│
                        │   - reduce_scatter(m_grad)        │
                        │   - mean over axis_size           │
                        │   - + the shard set aside above   │
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

`fully_shard` supports `gradient_accumulation_steps > 1` without any extra collective. The
accumulated gradient stays **sharded**: `backward_pre` detects a carried-over shard grad via
`is_grad_initialized()`, sets it aside and clears the parameter's grad, so the module's
closures build a fresh full-shape grad for this micro-batch; `backward_post` reduce-scatters
that one and adds the saved shard to the result. (Gathering the accumulated grad instead, as
an earlier version did, cost a full-size all-gather per parameter per extra micro-batch —
measured +45 ms per micro-batch for TinyLlama on 8 chips.)

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
- <a name="tile-aligned-shards"></a>**Tile-aligned shards.** `all_gather_async` and
  `reduce_scatter_minimal_async` only use their direct kernels when the gathered /
  scattered dim of every per-rank shard is a multiple of the 32-element tile;
  otherwise ttnn silently routes the call through a composite split → pad → gather →
  slice → concat chain. On a 32-chip Blackhole galaxy that path measured 4-20x slower
  per call and 10-50x more host time (a `[5632, 2048]` weight sharded 32-way on rows
  gives 176-row shards: all-gather 3.7 ms vs 0.43 ms on the aligned dim, host 3.6 ms vs
  0.16 ms), which made TinyLlama FSDP=32 host-bound: 2072 ms per step vs 1728 with the
  aligned dim, i.e. slower than DDP instead of faster. `"auto"` therefore prefers the
  tile-aligned dim (dim 3 for the cases above). If neither dim is aligned, `fully_shard`
  warns (and refuses in overlap mode, whose persistent buffers need the direct kernels);
  pad that dim to a multiple of `32 * axis_size` or pass an explicit `shard_dim`.
- **Gradient checkpointing** (`runner_type: memory_efficient`) recomputes each block's
  forward inside the backward pass. The wrapped forward recognises that situation
  (`AutoContext.is_backward_in_progress()` with gradients enabled) and keeps the weights
  gathered for the block's backward closures that follow immediately, so a checkpointed
  step costs two all-gathers per parameter (no-grad forward, recompute) instead of three.
- **Throughput numbers.** The batch is sharded over every data-parallel axis, `dp` and
  `fsdp` alike, and `ThroughputCallback` counts tokens over both. (It used to count `dp`
  only, so FSDP runs printed TPS/MFU divided by the FSDP axis size — the origin of most
  "FSDP is slow" reports.)

---

## Fewer collectives: keeping blocks gathered

FSDP issues three collectives per parameter per step: the forward all-gather, the backward
all-gather (after `reshard_after_forward` freed the weights), and the reduce-scatter. A block
wrapped with `reshard_after_forward=False` skips the second one for the price of holding its
gathered weights from its forward to its backward. The training example decides per block with
`ttml.fsdp.blocks_to_keep_gathered(blocks, budget_gib)`:

- the **last block** is always kept — its backward starts right after the forward, so
  resharding it would be undone by an immediate re-gather;
- **`fsdp_keep_gathered_gib: X`** extends that to as many preceding blocks as fit in `X` GiB of
  unsharded bf16 weights per device, counted from the end because the last blocks' gathered
  weights live the shortest;
- **`.inf`** keeps every block (the whole unsharded model must fit): measured −2 % step time for
  TinyLlama on 8 chips, −2.6 % on 32;
- the **root** is wrapped with `reshard_after_forward=False` unconditionally: its embedding is
  the first op of the forward and its LM head's gradient the last of the backward, so the
  interval a reshard would free is empty.

## Overlapping collectives with compute

By default every FSDP all-gather and reduce-scatter runs on the full core grid, serialized with
the model's compute: the device is ~96 % busy, but the collectives are pure exposed time (about
6 % of the step at 10k tokens per device, ~30 % at 2k). `fsdp_overlap_collectives: true` moves
them off the critical path:

```python
ttml.open_device_mesh(mesh, num_command_queues=2)
ttml.fsdp.enable_overlap(columns=1)     # before any fully_shard
for block in model.blocks: ttml.fsdp.fully_shard(block, ...)
ttml.fsdp.fully_shard(model, reshard_after_forward=False)
```

How it works:

- **Two sub-devices.** Each chip's Tensix grid is split into a compute sub-device (id 0) and a
  CCL sub-device (id 1: the rightmost `columns` columns or the bottom `rows` rows). A program
  must lie inside one sub-device (a hard error on a mesh command queue), so the mesh device
  reports the compute rectangle as its compute grid from then on
  (`MeshDevice::set_compute_with_storage_grid_size_override`) and every op sizes itself inside
  it. The CCL kernels take `(workers + 1 mux) × 2 directions` cores per link and use 4, 2 or 1
  workers depending on what fits: on a 12×10 Blackhole grid one column (10 cores, 8 % of the
  grid) gives 1 worker, one row (12 cores) 2 — a 2.3x faster all-gather — and two columns
  (20 cores) 4. The shape also decides the compute grid the ops see, and that matters more:
  one row leaves 12×9, which for TinyLlama at 6 samples per device was 11 % *slower* than no
  overlap at all while one column (11×10) was 3 % faster, and on 32 chips the column beat the
  row by 6 %; only at 1 sample per device on 8 chips did the row's faster collectives edge it
  (420 vs 426 ms). One column is the default.
- **Two command queues.** The dispatcher launches programs in order and a launch waits for the
  previous program on the same sub-device, so a run of collective launches on the compute queue
  would stall every compute launch behind it (measured: 8 gathers then 8 matmuls cost the sum
  of both). FSDP collectives are therefore issued on hardware queue 1 (ttnn device operations
  launch on the thread's current queue, `ttnn.command_queue(1)`) and run on the CCL sub-device;
  a collective issued on queue 0 — a tensor-parallel all-gather inside a block, the
  vocab-parallel loss — runs on the compute sub-device, because two queues launching programs
  on the same sub-device interleave on the same cores and corrupt each other.
- **Prefetch.** `pre_forward` of block *i* waits for its own gather and issues block *i+1*'s;
  `backward_pre` does the same for block *i−1*; the root prefetches the first block in forward
  and the last in backward. Compute never consumes a gather before a *CCL barrier* (record on
  queue 1, wait on queue 0).
- **Reduce-scatters queue behind the next prefetch.** `backward_post` of block *i* hands its
  gathered grads to the runtime; block *i−1*'s `backward_pre` issues its own prefetch first and
  the deferred reduce-scatters after it, behind a *compute drain* (an event recorded on queue 0
  that queue 1 waits for, so the grads are complete). The CCL queue is in order, a gather has a
  deadline (the compute waiting for it) and a reduce-scatter has none until the end of backward,
  so this order matters: issuing the reduce-scatters in `backward_post` measured 447 vs 421 ms on
  TinyLlama at 1 sample per device — every prefetch arrived late behind the previous block's
  reduce-scatters. They write fresh shard-shaped outputs, scaled by `1/N` once at the end of
  backward; the full-shape grads they read are freed at the next barrier.
- **Every buffer a collective writes across devices is persistent and rotates with slack.**
  The host frees and reallocates addresses far ahead of the device, and a collective is a
  cross-device write: when device E starts a gather it writes into every peer's copy of the
  output while a slower peer may still be reading. So all-gather outputs come from a pool of
  `2 × lookahead` persistent slots per (parameter position, shape) — the root and blocks kept
  gathered own a slot each — and a gather into a slot waits for the compute release
  *`lookahead` units after* the slot's last reader (`ttml.fsdp.SlotSchedule`; every gather is a
  rendezvous, so peers cannot lag by more than `lookahead` units, and with twice that many
  slots the one being overwritten was last read at least `lookahead` units ago everywhere).
  The same rule shapes `CCLResources`: reduce-scatter staging buffers rotate through 4 sets per
  shape and global semaphores through 8 sets per queue, and a reduce-scatter on the CCL queue
  refuses to run without persistent staging (its own temporaries would be freed by the host
  while compute could be handed their addresses). Two more consequences of the same rule: a pool
  buffer is created at first use on an address the host may have freed a moment ago, so its
  creation is followed by a full device synchronize (once per buffer, in the first step — the
  eager frees of the no-grad checkpointing forward made this visible); and the full-shape grads
  a reduce-scatter read are handed back to the host only two barriers later, because the barrier
  proves the reduce-scatter done on this device while a peer may still be reading, and the next
  fresh allocation at that address could be a tensor-parallel all-gather that writes into every
  peer (the source of a rare TP × FSDP drift). `fsdp_overlap_lookahead` defaults to 2 (four
  slots); 3 measured ~1 % faster for six slots' worth of memory.
- **Bit-exactness.** With the rules above every configuration measured — plain FSDP at 1 and 5
  samples per device on 8 and 32 chips, gradient accumulation, gradient checkpointing — trains
  with losses bit-identical to the non-overlapped run. `TTML_FSDP_SERIALIZE_COLLECTIVES=1`
  follows every collective with a barrier, the first switch to flip when a run is not.
  `tools/profiling/fsdp_bench/overlap_race_harness.py` reproduces the schedule in isolation
  and checks every block bitwise; run it before changing any of this.

Cost/benefit: the reserved column is 8 % of the grid, so overlap pays when the exposed
collectives are a larger share of the step than that. Measured with this branch (profiler off,
ring MGDs, mean of steps 3-12, 2048 tokens per sample; every overlapped run's losses are
bit-identical to its reference):

| configuration | no overlap | overlap, one column | one row |
|---|---|---|---|
| TinyLlama FSDP8, 1 sample/device | 466 ms | 426 (−9 %) | 420 (−10 %) |
| TinyLlama FSDP8, 5 samples/device | 1662 | 1605 (−3 %) | 1625 |
| TinyLlama FSDP8, 6 samples/device | 1980 | 1928 (−3 %) | 2206 (+11 %) |
| TinyLlama FSDP8, 6 samples × 2 micro-batches | 3945 | 3839 (−3 %) | 4399 |
| TinyLlama FSDP8, 1 sample, gradient checkpointing | 560 | 511 (−9 %) | 523 |
| TinyLlama FSDP32, 1 sample/device (DDP32: 572) | 502 | 451 (−10 %) | 479 |
| Llama-8B TP4 × FSDP8, 2 samples, checkpointing | 1574 | 1532 (−3 %) | — |

HSDP (a `dp` axis next to `fsdp`) has not been run in overlap mode.

## Inference / rollouts

Each forward through a `fully_shard`-ed model all-gathers all weights and frees them again.
For autoregressive generation that means one all-gather of the *whole model* per generated
token (measured 30 ms per forward for TinyLlama 1.1B, 60 % of a decode step; ~1 s per token
for a 32B model). Wrap gradient-free generation in

```python
with ttml.fsdp.unshard_for_inference(model) as kept_gathered:
    tokens = generate(model, prompts)
```

which gathers every unit once, suspends `reshard_after_forward` for the block, and reshards
on exit. It only does so when the unsharded weights fit in half of the free DRAM
(`max_fraction_of_free_dram`); otherwise it warns and leaves per-forward gathering in place.

---

## Debugging tips

- **`module.unshard()` / `module.reshard()`** are exposed on every
  wrapped module. Useful to manually inspect a parameter's full value
  outside the training loop.
- **`ttml.fsdp.is_fsdp_managed(param.tensor)`** returns `True` for any
  parameter `fully_shard` has touched. The marker is what
  `sync_gradients` uses to decide which axes to skip per parameter.
- **`tools/profiling/fsdp_bench/`** has the benchmark kit (config generator, guarded runner,
  Tracy analysis, CCL microbenchmark, overlap race harness) and the performance study behind
  the numbers quoted in this document.

---

## TODOs

These are known to be incomplete pieces of the FSDP prototype, in
roughly the order I'd tackle them:

- [ ] **Bucket / flatten per unit.** One all-gather and one reduce-scatter per block instead of
  one per parameter (FlatParam-style, with views back into the weights). FSDP issues ~700
  collective launches per step on TinyLlama; at 32 chips the per-launch cost dominates, and the
  4 KB RMSNorm gammas cost as much to launch as a matrix. Leaving tiny parameters replicated
  instead was measured and is *worse* (their all-reduces in `sync_gradients` are exposed), so
  this needs the bucketed form.

- [ ] **HSDP in overlap mode** has not been exercised (the `dp` all-reduce runs on the compute
  sub-device from queue 0, which is correct by construction but unmeasured).

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
