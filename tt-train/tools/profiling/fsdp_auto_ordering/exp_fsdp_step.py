# A real tt-train FSDP training step (Llama blocks under ttml.fsdp.fully_shard, AdamW) on a 1x2 mesh, with the
# simplest possible overlap: every block's all-gather is issued one block ahead (forward and backward) and every
# reduce-scatter as soon as its gradient exists, all on the collective stream (command queue 1, the last column),
# while compute runs on command queue 0 (the other columns). The program contains no events. Variants:
#   today     ttml's FSDP as it is: one queue, the whole grid (run with --setup full)
#   serial    the same schedule with collectives on the compute stream (cq0/sd0), ordering off: no overlap
#   serial+   serial with the ordering ledger on (its host cost)
#   overlap   collectives on cq1/sd1, ordering "order": every wait derived at launch
#   racy      overlap with ordering off
#   onequeue  collectives on cq0/sd1, compute on cq0/sd0, ordering "order" (two streams on one queue)
#   check     overlap in "check" mode: prints the first refusal
import argparse
import gc
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--setup", choices=["full", "split"], default="split")
parser.add_argument("--variants", default="serial,serial+,overlap,racy,onequeue")
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--hidden", type=int, default=2048)
parser.add_argument("--heads", type=int, default=16)
parser.add_argument("--kv-heads", type=int, default=4)
parser.add_argument("--seq", type=int, default=512)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--ccl-columns", type=int, default=1)
parser.add_argument("--log", default=None, help="ledger log file (sets TTNN_QUEUE_ORDERING_LOG)")
parser.add_argument("--out", default=None, help="JSON results file")
parser.add_argument("--init", default=None, help="initial weights (torch.save), created if missing")
args = parser.parse_args()

os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    os.path.join(os.environ["TT_METAL_HOME"], "tt-train/configs/mgd/bh_1_2_line_line.textproto"),
)
if args.setup == "split":
    os.environ["TTML_NUM_COMMAND_QUEUES"] = "2"
if args.log:
    os.environ["TTNN_QUEUE_ORDERING_LOG"] = args.log

import numpy as np
import ttnn
import ttml
import ttml.fsdp as F
from ttml.models.llama import Llama, LlamaConfig
from ttml.models import RunnerType, WeightTyingType

ttml.open_device_mesh(ttml.Mesh((1, 2), ("dp", "fsdp")))
ctx = ttml.autograd.AutoContext.get_instance()
device = ctx.get_device()
grid = device.compute_with_storage_grid_size()
print(f"grid {grid.x}x{grid.y}", flush=True)

SD0, SD1 = ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)
manager = None
if args.setup == "split":
    split = grid.x - args.ccl_columns
    compute = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(split - 1, grid.y - 1))})
    column = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(split, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([compute]), ttnn.SubDevice([column])], 0)
    device.load_sub_device_manager(manager)
    ttnn._ttnn.device.set_reserved_compute_columns(args.ccl_columns)
    narrowed = device.compute_with_storage_grid_size()
    print(
        f"compute grid narrowed to {narrowed.x}x{narrowed.y}; collectives get {args.ccl_columns} column(s)", flush=True
    )


# ---------------------------------------------------------------------------------------------------------------
# The overlap: ttml.fsdp's hooks, with collectives placed on a stream and gathers issued one block ahead.
# ---------------------------------------------------------------------------------------------------------------


class Plan:
    original = None  # ttml's FSDPState hooks, for "today"
    ccl_queue = 0
    ccl_sub_device = SD0
    order = []  # FSDP states in forward order
    prefetched = {}
    eager = False  # record an event on the collective queue after each block's collectives
    marks = False  # and one on the compute queue where each block's forward and backward end
    park = False  # keep what the collective stream read or wrote until the step ends
    parked = []


def install_hooks():
    Plan.original = {
        name: getattr(F.FSDPState, name) for name in ("pre_forward", "post_forward", "backward_pre", "backward_post")
    }

    def gather(state):
        values = [(parameter.tensor, shard_dim) for parameter, shard_dim in state.managed]
        out = []
        for tensor, shard_dim in values:
            shard = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
            state._cached_shards[id(tensor)] = shard
            with ttnn.command_queue(Plan.ccl_queue):
                out.append(
                    ttnn.all_gather(
                        shard,
                        shard_dim,
                        cluster_axis=state.axis_index,
                        subdevice_id=Plan.ccl_sub_device,
                        topology=ttnn.Topology.Linear,
                    )
                )
        mark_collectives_done()
        return out

    def mark_collectives_done():
        if Plan.eager:
            ttnn.record_event(device, Plan.ccl_queue, sub_device_ids=[Plan.ccl_sub_device])

    def unshard(state):
        if state.sharded_state == F._ShardedState.UNSHARDED:
            return
        gathered = Plan.prefetched.pop(id(state), None) or gather(state)
        for (parameter, _), value in zip(state.managed, gathered):
            parameter.tensor.set_value(value)
        state.sharded_state = F._ShardedState.UNSHARDED

    def prefetch(state):
        if state is None or state.sharded_state == F._ShardedState.UNSHARDED or id(state) in Plan.prefetched:
            return
        Plan.prefetched[id(state)] = gather(state)

    def neighbour(state, step):
        index = Plan.order.index(state) + step
        return Plan.order[index] if 0 <= index < len(Plan.order) else None

    def pre_forward(state):
        unshard(state)
        prefetch(neighbour(state, +1))

    def mark_compute_done():
        if Plan.marks:
            ttnn.record_event(device, 0, sub_device_ids=[SD0])

    def post_forward(state):
        mark_compute_done()
        if state is Plan.order[-1]:
            return  # backward starts here: keep it gathered (torch FSDP2 does the same)
        state._reshard_all_params()
        state.sharded_state = F._ShardedState.SHARDED

    def backward_pre(state):
        unshard(state)
        prefetch(neighbour(state, -1))

    def backward_post(state):
        mark_compute_done()
        scale = 1.0 / float(state.axis_size)
        for parameter, shard_dim in state.managed:
            tensor = parameter.tensor
            if not tensor.is_grad_initialized():
                continue
            gathered_grad = tensor.get_grad()
            with ttnn.command_queue(Plan.ccl_queue):
                reduced = ttnn.reduce_scatter(
                    gathered_grad,
                    shard_dim,
                    cluster_axis=state.axis_index,
                    subdevice_id=Plan.ccl_sub_device,
                    topology=ttnn.Topology.Linear,
                )
                # The mean, on the collective stream's cores (binary_ng takes a worker-cores scope, #62762).
                with ttnn.sub_device(Plan.ccl_sub_device):
                    unscaled, reduced = reduced, ttnn.multiply(reduced, scale)
                if Plan.park:
                    Plan.parked.append(unscaled)
            gathered_value = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
            tensor.set_value(state._cached_shards[id(tensor)])
            tensor.set_grad(reduced)
            ttnn.deallocate(gathered_value)
            if Plan.park:
                Plan.parked.append(gathered_grad)
            else:
                ttnn.deallocate(gathered_grad)
        mark_collectives_done()
        state.sharded_state = F._ShardedState.SHARDED

    Plan.ours = {
        "pre_forward": pre_forward,
        "post_forward": post_forward,
        "backward_pre": backward_pre,
        "backward_post": backward_post,
    }
    Plan.gather = gather


install_hooks()


def use_hooks(ours):
    for name, fn in (Plan.ours if ours else Plan.original).items():
        setattr(F.FSDPState, name, fn)


VARIANTS = {
    # name: (our hooks, collective queue, collective sub-device, ordering mode)
    "noshard": (False, 0, SD0, "off"),  # no FSDP: compute (and a full-size optimizer) only
    "ccl0": (True, 0, SD0, "off"),  # the step's collectives alone, on the compute grid
    "ccl1": (True, 1, SD1, "off"),  # the step's collectives alone, on the collective column(s)
    "park": (True, 1, SD1, "order"),
    "eager": (True, 1, SD1, "order"),  # overlap, one event recorded after each block's collectives
    "eagerpark": (True, 1, SD1, "order"),  # eager and park
    "dataonly": (True, 1, SD1, "order"),  # eagerpark with memory reuse left unordered (diagnostic, unsafe)
    "racyrec": (True, 1, SD1, "off"),  # racy with eager's records (what the records cost)
    "marks": (
        True,
        1,
        SD1,
        "order",
    ),  # eagerpark, plus an event on the compute queue at each block's end  # overlap, frees of what collectives touched parked until the step ends
    "today": (False, 0, SD0, "off"),
    "serial": (True, 0, SD0, "off"),
    "serial+": (True, 0, SD0, "order"),
    "overlap": (True, 1, SD1, "order"),
    "racy": (True, 1, SD1, "off"),
    "onequeue": (True, 0, SD1, "order"),
    "check": (True, 1, SD1, "check"),
}

config = LlamaConfig(
    hidden_size=args.hidden,
    num_hidden_layers=args.layers,
    num_attention_heads=args.heads,
    num_key_value_heads=args.kv_heads,
    vocab_size=256,
    max_position_embeddings=args.seq,
    runner_type=RunnerType.Default,
    weight_tying=WeightTyingType.Disabled,
)
rng = np.random.default_rng(0)
tokens = rng.integers(0, 256, size=(args.batch, 1, 1, args.seq)).astype(np.uint32)
targets = rng.integers(0, 256, size=(args.batch, args.seq)).astype(np.uint32)


def initial_weights(model):
    """The same starting point for every variant and process: ttml's initialization is not reproducible."""
    import torch

    if args.init and os.path.exists(args.init):
        return torch.load(args.init)
    weights = {
        name: ttnn.to_torch(ttnn.get_device_tensors(tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE))[0])
        for name, tensor in model.parameters().items()
    }
    if args.init:
        torch.save(weights, args.init)
    return weights


INITIAL = None


def load_initial_weights(model):
    global INITIAL
    if INITIAL is None:
        INITIAL = initial_weights(model)
    for name, tensor in model.parameters().items():
        value = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
        tensor.set_value(
            ttnn.from_torch(
                INITIAL[name],
                dtype=value.dtype,
                layout=value.layout,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        )


def fingerprint(model):
    """Per-device sums of every parameter, as one vector."""
    sums = []
    for name, tensor in sorted(model.parameters().items()):
        for shard in ttnn.get_device_tensors(tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)):
            sums.append(float(ttnn.to_torch(shard).float().abs().sum()))
    return np.array(sums)


def run(variant):
    ours, queue, sub_device, mode = VARIANTS[variant]
    use_hooks(ours)
    Plan.ccl_queue, Plan.ccl_sub_device = queue, sub_device
    Plan.prefetched = {}
    Plan.park, Plan.parked = variant in ("park", "eagerpark", "dataonly", "marks"), []
    Plan.eager = variant in ("eager", "eagerpark", "dataonly", "racyrec", "marks")
    Plan.marks = variant == "marks"
    if variant == "dataonly":
        os.environ["TTNN_QUEUE_ORDERING_SKIP_REUSE"] = "1"
    else:
        os.environ.pop("TTNN_QUEUE_ORDERING_SKIP_REUSE", None)
    ctx.set_seed(1234)
    model = Llama(config)
    load_initial_weights(model)
    for block in model.blocks if variant != "noshard" else []:
        F.fully_shard(block)
    Plan.order = [block._fsdp_state for block in model.blocks if getattr(block, "_fsdp_state", None) is not None]
    optimizer = ttml.optimizers.AdamW(model.parameters(), ttml.optimizers.AdamWConfig.make(1e-3, 0.9, 0.999, 1e-8, 0.0))
    x = ttml.autograd.Tensor.from_numpy(tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32)
    y = ttml.autograd.Tensor.from_numpy(targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32)
    mask = ttml.autograd.Tensor.from_numpy(
        np.tril(np.ones((1, 1, args.seq, args.seq), dtype=np.float32)), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    ttnn.synchronize_device(device)
    ttnn._ttnn.core.set_queue_ordering(mode)

    def collectives_only():
        # What the step's collectives do, without compute: each block gathered twice (forward and backward) and its
        # gradients reduce-scattered and scaled.
        for state in Plan.order:
            for value in Plan.gather(state):
                ttnn.deallocate(value)
        for state in Plan.order[::-1]:
            for value, (parameter, shard_dim) in zip(Plan.gather(state), state.managed):
                with ttnn.command_queue(Plan.ccl_queue):
                    reduced = ttnn.reduce_scatter(
                        value,
                        shard_dim,
                        cluster_axis=state.axis_index,
                        subdevice_id=Plan.ccl_sub_device,
                        topology=ttnn.Topology.Linear,
                    )
                    with ttnn.sub_device(Plan.ccl_sub_device):
                        reduced = ttnn.multiply(reduced, 0.5)
                ttnn.deallocate(value)
                ttnn.deallocate(reduced)

    def step():
        if variant.startswith("ccl"):
            collectives_only()
            return None
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = ttml.ops.loss.cross_entropy_loss(logits, y, reduce=ttml.ops.ReduceType.MEAN)
        loss.backward(False)
        ctx.reset_graph()
        optimizer.step()
        for value in Plan.parked:
            ttnn.deallocate(value)
        Plan.parked = []
        return loss

    result = {"variant": variant}
    log_marks = []
    try:
        losses = []
        for _ in range(args.warmup):
            losses.append(step())
            ttnn.synchronize_device(device)
            if args.log:
                log_marks.append(os.path.getsize(args.log))
        times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            losses.append(step())
            ttnn.synchronize_device(device)
            times.append((time.perf_counter() - start) * 1e3)
            if args.log:
                log_marks.append(os.path.getsize(args.log))
        result["step_ms"] = times
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        result["losses"] = [
            float(loss.to_numpy(ttnn.DataType.FLOAT32, composer=composer).mean()) for loss in losses if loss is not None
        ]
        result["fingerprint"] = fingerprint(model).tolist()
        result["log_marks"] = log_marks
    except Exception as error:  # check mode refuses
        import traceback

        traceback.print_exc()
        result["error"] = str(error).split("\n")[0][:1500]
        ttnn.synchronize_device(device)
    ttnn._ttnn.core.set_queue_ordering("off")
    use_hooks(False)
    del model, optimizer, x, y, mask
    gc.collect()
    ttnn.synchronize_device(device)
    return result


results = []
for variant in args.variants.split(","):
    result = run(variant)
    results.append(result)
    if "error" in result:
        print(f"{variant:9s} refused: {result['error']}", flush=True)
    else:
        t = result["step_ms"]
        print(
            f"{variant:9s} step {np.median(t):8.2f} ms (min {min(t):.2f})  losses {[round(l, 4) for l in result['losses']]}",
            flush=True,
        )

reference = next((r for r in results if r["variant"] in ("serial", "today") and "fingerprint" in r), None)
if reference is not None:
    for r in results:
        if "fingerprint" in r:
            a, b = np.array(r["fingerprint"]), np.array(reference["fingerprint"])
            r["max_rel_param_diff_vs_" + reference["variant"]] = float(
                np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-6))
            )
            print(
                f"{r['variant']:9s} params vs {reference['variant']}: max relative difference {r['max_rel_param_diff_vs_' + reference['variant']]:.3g}"
            )

if args.out:
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "grid": [grid.x, grid.y], "results": results}, f, indent=1)

if manager is not None:
    ttnn._ttnn.device.set_reserved_compute_columns(0)
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(manager)
