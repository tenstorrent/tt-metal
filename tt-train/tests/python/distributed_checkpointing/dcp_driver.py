# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run one world of a cross-world checkpoint test in its own process.

A tensor-parallel model needs the process-wide ParallelismContext, which cannot be reset, so a test that saves on one
parallelism and loads on another runs each world in its own process and passes the checkpoint through a directory.
This driver is that process. ``save`` trains a tiny Llama for a few steps, saves it, and writes a reference next to
the checkpoint: every parameter and optimizer moment in canonical layout, the logits on a fixed batch, and the
optimizer scalars. ``load`` builds the destination world, loads the checkpoint, and compares against the reference.
Both commands then take one more training step and record the reference again under ``resume:``: the run one step
after the checkpoint, which the loaded world reproduces only if the optimizer carried its moments and step count
across (``--no-resume-step`` leaves it out).

The reference is computed independently of the checkpoint module: tensors are gathered through the composer and
mapped to the canonical layout with the model's own rules written out in numpy here (per-shard segment order for
the fused projections, the logical vocab rows for the padded LM head).

    python tt-train/tests/python/distributed_checkpointing/dcp_driver.py save --dir /tmp/ckpt --mesh 2x2 --axes dp,tp --tp --steps 2
    python tt-train/tests/python/distributed_checkpointing/dcp_driver.py load --dir /tmp/ckpt --mesh 2x2 --axes dp,fsdp --fsdp
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import ttnn

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.models.llama.transformer import compute_swiglu_intermediate_size
from ttml.sharding import Sharding

NATIVE = ttml.autograd.PreferredPrecision.NATIVE
# head_dim = hidden_size / num_attention_heads must be a multiple of 32 (RoPE); the head counts must be multiples of
# the TP size. --hidden/--heads/--kv-heads override these on save; load takes them from the reference.
TINY = dict(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=130,
    max_position_embeddings=32,
)
SEQ_LEN = 32  # a whole tile: the SDPA kernel wants the sequence dimension unpadded
MESH_OPEN_FAILED = 3  # exit status when this host cannot open the requested mesh; the tests skip on it


def parse_mesh(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.lower().split("x"))


def open_world(mesh_shape: tuple[int, ...], axes: tuple[str, ...], tp: bool, mgd: str | None) -> None:
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    ttml.open_device_mesh(ttml.Mesh(mesh_shape, axes))
    if tp:
        mesh = ttml.mesh()
        ddp = mesh.has_axis("dp") and mesh.axis_size("dp") > 1
        ctx = ttml.autograd.AutoContext.get_instance()
        ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=ddp, enable_tp=True))


def build(tp: bool, fsdp: bool, tie: bool, sizes: dict):
    config = LlamaConfig(
        **{**TINY, **sizes},
        use_tp=tp,
        weight_tying=WeightTyingType.Enabled if tie else WeightTyingType.Disabled,
        # A tied LM head is vocab-parallel, so the embedding must share that layout (LlamaConfig refuses otherwise).
        embedding_placement=EmbeddingPlacement.VocabParallel if (tp and tie) else EmbeddingPlacement.Replicated,
    )
    model = Llama(config)
    if fsdp:
        ttml.fsdp.fully_shard(model)
    params = model.parameters()
    cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    optimizer = ttml.optimizers.AdamW(params, cfg)
    return config, model, params, optimizer


def inputs(config: LlamaConfig, seed: int):
    """A batch: the tokens, the causal mask and the next-token targets, the same on every device."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, config.vocab_size, size=(1, 1, 1, SEQ_LEN)).astype(np.uint32)
    targets = rng.integers(0, config.vocab_size, size=(1, SEQ_LEN)).astype(np.uint32)
    mask = np.tril(np.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=np.float32))
    return (
        ttml.autograd.Tensor.from_numpy(tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32),
        ttml.autograd.Tensor.from_numpy(mask, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16),
        ttml.autograd.Tensor.from_numpy(targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32),
    )


def padding_mask(padded_vocab: int, vocab: int, tp_axis: int):
    """``-1e4`` on the LM head's padded vocab columns and ``0`` elsewhere, sharded over the TP axis like the logits.
    Added to the logits it takes the padded entries out of the vocab-parallel loss's softmax: that loss normalizes
    over every local column (it is given no vocab size, issue #64714), so a TP world whose vocab is padded would
    otherwise train a different loss than a world whose LM head output is sliced to the logical vocab, and their
    steps could never be compared. (With a vocab of 130 and TP 2 the padding is 62 of 192 columns.) Once #64714 is
    fixed the loss leaves the padding out itself, and this mask and its use in ``loss_of`` can go."""
    mesh = ttml.mesh()
    shape = tuple(mesh.shape)
    columns = np.where(np.arange(padded_vocab) < vocab, 0.0, -1e4).astype(np.float32)
    full = np.broadcast_to(columns, (1, 1, SEQ_LEN, padded_vocab)).copy()
    dims = [None] * len(shape)
    dims[tp_axis] = 3
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=shape, dims=tuple(dims))
    return ttml.autograd.Tensor.from_numpy(
        full, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16, mapper=mapper
    )


def loss_of(model, config: LlamaConfig, tokens, mask, targets):
    """Cross entropy on the next-token targets, through the loss each world's logits layout calls for: the
    vocab-parallel one on a TP world, whose LM head keeps its output vocab-sharded and padded (the padding masked
    out, see ``padding_mask``), the plain one elsewhere, where the model slices its logits to the logical vocab.
    Both are the losses training uses, and both see the same logical vocabulary, so every world's gradient is the
    same quantity. A mean over the local logits is not: its element count and its padded columns differ per world,
    and a step taken on two worlds could never be compared."""
    logits = model(tokens, mask)
    if not config.use_tp:
        return ttml.ops.loss.cross_entropy_loss(logits, targets, ttml.ops.ReduceType.MEAN)
    tp_axis = ttml.mesh().axis_index("tp")
    if model.padded_vocab_size != config.vocab_size:
        logits = ttml.ops.binary.add(logits, padding_mask(model.padded_vocab_size, config.vocab_size, tp_axis))
    return ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
        logits, targets, cluster_axis=tp_axis, reduce=ttml.ops.ReduceType.MEAN
    )


def train(model, params, optimizer, config, steps: int, seed: int, start: int = 0) -> None:
    """``steps`` training steps numbered from ``start``: step ``n`` draws batch ``seed + n``, so a run resumed from
    a checkpoint repeats the batches the uninterrupted run would have drawn."""
    ctx = ttml.autograd.AutoContext.get_instance()
    mesh = ttml.mesh()
    needs_sync = any(mesh.axis_size(name) > 1 for name in mesh.axis_names if name != "tp")
    for step in range(start, start + steps):
        tokens, mask, targets = inputs(config, seed + step)
        optimizer.zero_grad()
        loss = loss_of(model, config, tokens, mask, targets)
        loss.backward(False)
        if needs_sync:
            ttml.sync_gradients(params)
        optimizer.step()
        ctx.reset_graph()


def gather(tensor) -> np.ndarray:
    """A tensor's global device-layout array as float32, through the composer."""
    return np.asarray(Sharding.from_tensor(tensor).gather(tensor)).astype(np.float32)


def full_logits(tensor, config: LlamaConfig) -> np.ndarray:
    """The logits on the fixed batch as ``[1, 1, S, vocab]`` in every world.

    A tensor-parallel LM head (``gather_output=False``) leaves the vocab dimension sharded over the TP axis, and its
    output label does not say so, so the shards are read directly and concatenated in TP order; a replicated head
    is read from the first device. The padded vocab columns are dropped.
    """
    value = tensor.get_value(NATIVE)
    shards = [ttnn.to_torch(s).float().numpy() for s in ttnn.get_device_tensors(value)]  # row-major mesh order
    mesh = ttml.mesh()
    if config.use_tp:
        shape = tuple(mesh.shape)
        tp_axis = mesh.axis_index("tp")
        stride = 1 if tp_axis == len(shape) - 1 else int(np.prod(shape[tp_axis + 1 :]))
        pieces = [shards[j * stride] for j in range(shape[tp_axis])]  # coordinate 0 on every other axis
        full = np.concatenate(pieces, axis=3)
    else:
        full = shards[0]
    return full[..., : config.vocab_size]


def uninterleave(array: np.ndarray, dim: int, sizes: tuple[int, ...], shards: int) -> np.ndarray:
    """Undo the per-shard segment order of a fused weight: shard ``j`` holds piece ``j`` of every segment."""
    if shards == 1:
        return array
    per_shard = array.shape[dim] // shards
    pieces = [[] for _ in sizes]
    for j in range(shards):
        cursor = j * per_shard
        for g, size in enumerate(sizes):
            n = size // shards
            pieces[g].append(np.take(array, range(cursor, cursor + n), axis=dim))
            cursor += n
    return np.concatenate([np.concatenate(p, axis=dim) for p in pieces], axis=dim)


def canonicalize(name: str, array: np.ndarray, config: LlamaConfig, tp: int) -> np.ndarray:
    head_dim = config.hidden_size // config.num_attention_heads
    qkv = (
        config.num_attention_heads * head_dim,
        config.num_key_value_heads * head_dim,
        config.num_key_value_heads * head_dim,
    )
    intermediate = config.intermediate_size or compute_swiglu_intermediate_size(config.hidden_size)
    if name.endswith("qkv_linear/weight"):
        return uninterleave(array, 2, qkv, tp)
    if name.endswith("qkv_linear/bias"):
        return uninterleave(array, 3, qkv, tp)
    if name.endswith("w_gate_up/weight"):
        return uninterleave(array, 2, (intermediate, intermediate), tp)
    if name.endswith("fc/weight") or name.endswith("tok_emb/weight"):
        return np.take(array, range(config.vocab_size), axis=2)
    return array


def canonical_name(name: str, config: LlamaConfig) -> str:
    if config.weight_tying == WeightTyingType.Enabled:
        return name.replace("/tok_emb/weight", "/fc/weight")
    return name


def reference(model, params, optimizer, config, seed: int) -> dict[str, np.ndarray]:
    tp = ttml.mesh().axis_size("tp") if config.use_tp else 1
    out: dict[str, np.ndarray] = {}
    for name, tensor in params.items():
        out["param:" + canonical_name(name, config)] = canonicalize(name, gather(tensor), config, tp)

    def walk(node, path):
        if isinstance(node, ttml.NamedParameters):
            for name, tensor in node.items():
                out["state:" + "/".join(path) + ":" + canonical_name(name, config)] = canonicalize(
                    name, gather(tensor), config, tp
                )
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (str(key),))
        else:
            out["scalar:" + "/".join(path)] = np.asarray(node)

    walk(optimizer.get_state_dict(), ())
    tokens, mask, _ = inputs(config, seed + 1000)
    logits = model(tokens, mask)
    out["logits"] = full_logits(logits, config)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return out


RESUME = "resume:"


def resume_reference(model, params, optimizer, config, seed: int, step: int) -> dict[str, np.ndarray]:
    """One more training step, numbered ``step`` so it draws that step's batch, then the reference again under the
    ``resume:`` prefix: the run one step after the checkpoint. Taken on the saving world after its save and on the
    loading world after its load, the two agree only if the optimizer carried its moments and step count across."""
    train(model, params, optimizer, config, 1, seed, start=step)
    return {RESUME + key: value for key, value in reference(model, params, optimizer, config, seed).items()}


def compare(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], logits_tolerance: float) -> list[str]:
    """Every entry of the loaded world against the saving world's: exact, except across layouts, where the forward
    and the step after the checkpoint round differently. There the logits and the post-step moments must lie within
    ``logits_tolerance`` of their scale, and the post-step parameter update within a quarter of its mean size (a
    fresh optimizer's update differs from the restored one's by about its whole size)."""
    problems = []
    exact = logits_tolerance == 0.0
    if set(actual) != set(expected):
        problems.append(
            f"keys differ: only in loaded world {sorted(set(actual) - set(expected))[:5]}, only in reference {sorted(set(expected) - set(actual))[:5]}"
        )
    for key in sorted(set(actual) & set(expected)):
        a, e = actual[key], expected[key]
        if a.shape != e.shape:
            problems.append(f"{key}: shape {a.shape} vs reference {e.shape}")
            continue
        base = key[len(RESUME) :] if key.startswith(RESUME) else None
        if key == "logits" or base == "logits":
            scale = float(np.max(np.abs(e))) + 1e-6
            rel = float(np.max(np.abs(a - e))) / scale
            print(f"{key}: max |diff| / max |ref| = {rel:.4f} (tolerance {logits_tolerance})")
            if rel > logits_tolerance:
                problems.append(f"{key} differ by {rel:.4f} of their scale (tolerance {logits_tolerance})")
        elif not exact and base is not None and base.startswith("state:"):
            scale = float(np.max(np.abs(e))) + 1e-6
            rel = float(np.max(np.abs(a - e))) / scale
            if rel > logits_tolerance:
                problems.append(f"{key}: differs by {rel:.4f} of its scale (tolerance {logits_tolerance})")
        elif not exact and base is not None and base.startswith("param:") and base in actual and base in expected:
            # The update each world applied. The loaded parameters equal the saved ones (checked under their own
            # key), so the updates differ only where the steps did, whatever the parameters' own magnitude.
            delta, delta_ref = a - actual[base], e - expected[base]
            size = float(np.mean(np.abs(delta_ref)))
            if size == 0.0:
                continue  # an update below bf16 resolution moved nothing in the reference run either
            rel = float(np.mean(np.abs(delta - delta_ref))) / size
            if rel > 0.25:
                problems.append(f"{key}: the update differs by {rel:.2f} of its mean size (tolerance 0.25)")
        elif not np.array_equal(a, e):
            n = int(np.count_nonzero(a != e))
            problems.append(f"{key}: {n} of {a.size} values differ (max |diff| {float(np.max(np.abs(a - e))):.4g})")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["save", "load"])
    parser.add_argument("--dir", required=True, help="checkpoint directory")
    parser.add_argument("--mesh", required=True, help="mesh shape, e.g. 2x2")
    parser.add_argument("--axes", required=True, help="comma-separated axis names, e.g. dp,tp")
    parser.add_argument("--tp", action="store_true", help="build the model tensor-parallel over the 'tp' axis")
    parser.add_argument("--fsdp", action="store_true", help="fully_shard the model over the 'fsdp' axis")
    parser.add_argument("--tie", action="store_true", help="tie the embedding and the LM head")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mgd", default=None, help="mesh graph descriptor to set before opening the mesh")
    parser.add_argument("--logits-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--same-layout",
        action="store_true",
        help="load: the world is the one the checkpoint was saved on; logits must match exactly",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=TINY["hidden_size"],
        help="save: hidden size; load reads the sizes from the reference",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=TINY["num_attention_heads"],
        help="save: attention heads, a multiple of the TP size",
    )
    parser.add_argument(
        "--kv-heads", type=int, default=TINY["num_key_value_heads"], help="save: KV heads, a multiple of the TP size"
    )
    parser.add_argument("--vocab", type=int, default=TINY["vocab_size"], help="save: vocab size (padded per world)")
    parser.add_argument(
        "--no-resume-step",
        dest="resume_step",
        action="store_false",
        help="skip the training step taken after the save and after the load (and its comparison)",
    )
    args = parser.parse_args(argv)

    mesh_shape = parse_mesh(args.mesh)
    axes = tuple(args.axes.split(","))
    reference_path = args.dir.rstrip("/") + ".reference.npz"
    if args.command == "load":
        expected = dict(np.load(reference_path))
        sizes = json.loads(str(expected.pop("config")))
    else:
        sizes = dict(
            hidden_size=args.hidden,
            num_attention_heads=args.heads,
            num_key_value_heads=args.kv_heads,
            vocab_size=args.vocab,
        )
    try:
        open_world(mesh_shape, axes, args.tp, args.mgd)
    except RuntimeError as err:  # the descriptor does not fit this host's fabric, too few devices, ...
        print(f"MESH_OPEN_FAILED: {str(err).splitlines()[0]}")
        return MESH_OPEN_FAILED
    try:
        config, model, params, optimizer = build(args.tp, args.fsdp, args.tie, sizes)
        if args.command == "save":
            train(model, params, optimizer, config, args.steps, args.seed)
            result = dcp.save(
                args.dir,
                model=model,
                optimizer=optimizer,
                header={"step": args.steps, "mesh": list(mesh_shape), "axes": list(axes)},
            )
            ref = reference(model, params, optimizer, config, args.seed)
            if args.resume_step:
                ref.update(resume_reference(model, params, optimizer, config, args.seed, args.steps))
            np.savez(reference_path, config=json.dumps(sizes), **ref)
            print(
                json.dumps(
                    {
                        "saved": result.path,
                        "bytes_written": result.bytes_written,
                        "files": len(result.files),
                        "keys": result.keys,
                        "stats": result.stats,
                    }
                )
            )
            return 0
        result = dcp.load(args.dir, model=model, optimizer=optimizer)
        actual = reference(model, params, optimizer, config, args.seed)
        resumed = args.resume_step and any(key.startswith(RESUME) for key in expected)
        if resumed:
            actual.update(resume_reference(model, params, optimizer, config, args.seed, args.steps))
        else:
            expected = {key: value for key, value in expected.items() if not key.startswith(RESUME)}
            print("NOTE: no training step after the load; the run one step on is not compared")
        problems = compare(actual, expected, 0.0 if args.same_layout else args.logits_tolerance)
        if result.header.get("step") != args.steps:
            problems.append(f"header step {result.header.get('step')} != {args.steps}")
        print(
            json.dumps(
                {
                    "loaded": result.path,
                    "resumed": resumed,
                    "fast_path_keys": len(result.fast_path_keys),
                    "missing": result.missing,
                    "unexpected": result.unexpected,
                    "stats": result.stats,
                }
            )
        )
        for problem in problems:
            print("MISMATCH:", problem)
        print("RESULT:", "OK" if not problems else f"{len(problems)} mismatch(es)")
        return 0 if not problems else 1
    finally:
        ttml.close_device_mesh()


if __name__ == "__main__":
    sys.exit(main())
