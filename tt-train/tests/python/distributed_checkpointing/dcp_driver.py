# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run one world of a cross-world checkpoint test in its own process.

A tensor-parallel model needs the process-wide ParallelismContext, which cannot be reset, so a test that saves on one
parallelism and loads on another runs each world in its own process and passes the checkpoint through a directory.
This driver is that process. ``save`` trains a tiny Llama for a few steps, saves it, and writes a reference next to
the checkpoint: every parameter and optimizer moment as the shards this rank holds, the logits on a fixed batch,
the optimizer scalars and what a resumed run does next. ``load`` builds the destination world, loads the
checkpoint, dumps the same, and rank 0 compares the two worlds in canonical layout. Both commands then take one
more training step and dump again under ``resume:``: the run one step after the checkpoint, which the loaded world
reproduces only if the optimizer carried its moments and step count across (``--no-resume-step`` leaves the step
out, e.g. on a mesh whose gradient sync cannot run).

Under tt-run every rank of a world runs this driver; each dumps the shards of its own coordinates, and the parts
are merged by key. Shards are placed into global arrays by their topology labels and mapped to the canonical
layout with the model's own rules written out in numpy here (per-shard segment order for the fused projections,
the logical vocab rows for the padded LM head).

    python tt-train/tests/python/distributed_checkpointing/dcp_driver.py save --dir /tmp/ckpt --mesh 2x2 --axes dp,tp --tp --steps 2
    python tt-train/tests/python/distributed_checkpointing/dcp_driver.py load --dir /tmp/ckpt --mesh 2x2 --axes dp,fsdp --fsdp
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import traceback

import numpy as np
import ttnn

import ttml
import ttml.distributed_checkpointing as dcp
from dcp_shards import local_coords, shard_arrays
from ttml.common.schedulers import CosineAnnealingScheduler
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.models.llama.transformer import compute_swiglu_intermediate_size
from ttml.common.utils import set_seed
from ttml.distributed_checkpointing._train_state import capture_process_rng, restore_process_rng

from dcp_oracle import Loader

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
T_MAX = 16  # cosine schedule length, longer than any --steps the tests use, so the LR keeps moving


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
    scheduler = CosineAnnealingScheduler(optimizer, T_max=T_MAX)
    return config, model, params, optimizer, scheduler


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


def train(
    model, params, optimizer, scheduler, loader, config, steps: int, seed: int, mark=lambda what: None, start: int = 0
) -> None:
    """``steps`` training steps numbered from ``start``: step ``n`` draws batch ``seed + n``, so a run resumed from
    a checkpoint repeats the batches the uninterrupted run would have drawn. The generators are seeded only when
    the run starts, at step 0; a resumed step continues from the generators the checkpoint restored."""
    ctx = ttml.autograd.AutoContext.get_instance()
    if start == 0:
        set_seed(seed)
    mesh = ttml.mesh()
    needs_sync = any(mesh.axis_size(name) > 1 for name in mesh.axis_names if name != "tp")
    for step in range(start, start + steps):
        tokens, mask, targets = inputs(config, seed + step)
        optimizer.zero_grad()
        loss = loss_of(model, config, tokens, mask, targets)
        loss.backward(False)
        mark(f"step {step}: forward and backward done")
        if needs_sync:
            ttml.sync_gradients(params)
            mark(f"step {step}: gradients synced over the non-TP axes")
        optimizer.step()
        scheduler.step()
        loader.advance()
        ctx.reset_graph()
        mark(f"step {step}: optimizer stepped")


def world_group() -> tuple[int, int]:
    """``(rank, size)`` of the ranks this process runs with: ``(0, 1)`` outside tt-run."""
    if ttnn.distributed_context_is_initialized() and int(ttnn.distributed_context_get_size()) > 1:
        return int(ttnn.distributed_context_get_rank()), int(ttnn.distributed_context_get_size())
    return 0, 1


def agree(ok: bool) -> bool:
    """Whether every rank succeeded so far; an allgather, so a rank that failed never leaves another blocked."""
    if ttnn.distributed_context_is_initialized() and int(ttnn.distributed_context_get_size()) > 1:
        return all(int(v) == 0 for v in ttnn.distributed_context_allgather_int(0 if ok else 1))
    return ok


def local_shards(tensor) -> dict[tuple[int, ...], np.ndarray]:
    """This process's shards of a tensor as float32, by mesh coordinate."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    return shard_arrays(tensor.get_value(NATIVE), local_coords(device))


def label_json(tensor) -> str:
    from ttml.distributed_checkpointing._state_dict import label_of

    return json.dumps(label_of(tensor.get_value(NATIVE)).to_json())


def assemble(label: str, shards: dict[tuple[int, ...], np.ndarray]) -> np.ndarray:
    """The global device-layout array of a tensor from the shards of every rank, placed by its topology label."""
    from ttml.distributed_checkpointing._topology import Label, chunk_boxes, global_shape

    lab = Label.from_json(json.loads(label))
    local_shape = next(iter(shards.values())).shape
    full = np.zeros(global_shape(lab, local_shape), dtype=np.float32)
    seen = set()
    chunks = chunk_boxes(lab, local_shape)
    for chunk in chunks:
        if chunk.mesh_coord in shards:
            full[chunk.box.slices()] = shards[chunk.mesh_coord]
            seen.add(chunk.group)
    missing = {c.group for c in chunks} - seen
    if missing:
        raise RuntimeError(f"no rank dumped a shard for replica groups {sorted(missing)}")
    return full


def replica_selections(label: str, copies: dict[tuple[int, ...], list[np.ndarray]]) -> list[dict]:
    """Ways to pick one shard per replica group, one per distinct copy a group has across the devices and ranks
    holding it: devices that should hold the same box may not (replicas drift apart when a world's collectives do
    not span its ranks), and the checkpoint holds the copy of the device that was elected to write it."""
    from ttml.distributed_checkpointing._topology import Label, chunk_boxes

    lab = Label.from_json(json.loads(label))
    local_shape = next(iter(copies.values()))[0].shape
    distinct: dict[tuple[int, ...], list[tuple[tuple[int, ...], np.ndarray]]] = {}
    for chunk in chunk_boxes(lab, local_shape):
        for array in copies.get(chunk.mesh_coord, []):
            kept = distinct.setdefault(chunk.group, [])
            if not any(np.array_equal(array, a) for _, a in kept):
                kept.append((chunk.mesh_coord, array))
    width = max(len(k) for k in distinct.values())
    selections = []
    for i in range(width):
        pick = {}
        for group, kept in distinct.items():
            coord, array = kept[min(i, len(kept) - 1)]
            pick[coord] = array
        selections.append(pick)
    return selections


def coord_key(coord: tuple[int, ...]) -> str:
    return "-".join(str(int(c)) for c in coord)


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


def local_reference(model, params, optimizer, scheduler, loader, config, seed: int, rank: int) -> dict[str, np.ndarray]:
    """This rank's part of the reference: every parameter and optimizer moment as the shards it holds, the logits
    shards on a fixed batch, the optimizer scalars and what a resumed run does next. Shard keys carry the rank, so
    the parts of ranks whose meshes share coordinates merge without colliding."""
    mesh = ttml.mesh()
    out: dict[str, np.ndarray] = {}
    out["meta"] = np.asarray(
        json.dumps(
            {
                "tp": mesh.axis_size("tp") if config.use_tp else 1,
                "tp_axis": mesh.axis_index("tp") if config.use_tp else -1,
                "mesh": list(mesh.shape),
            }
        )
    )
    tag = f"w{rank:03d}"
    for name, tensor in params.items():
        cname = canonical_name(name, config)
        out[f"label:param:{cname}"] = np.asarray(label_json(tensor))
        for coord, arr in local_shards(tensor).items():
            out[f"shard:param:{cname}:{tag}:{coord_key(coord)}"] = arr

    def walk(node, path):
        if isinstance(node, ttml.NamedParameters):
            for name, tensor in node.items():
                cname = "/".join(path) + ":" + canonical_name(name, config)
                out[f"label:state:{cname}"] = np.asarray(label_json(tensor))
                for coord, arr in local_shards(tensor).items():
                    out[f"shard:state:{cname}:{tag}:{coord_key(coord)}"] = arr
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (str(key),))
        else:
            out["scalar:" + "/".join(path)] = np.asarray(node)

    walk(optimizer.get_state_dict(), ())
    # What a resumed run does next: the scheduler's next LRs, the loader's next position and the next draw of each
    # generator (the ttml one as its state text: nothing draws from it on the host). The probes step the scheduler
    # and the loader and draw from the generators, so all three are put back: the dump leaves the run where it found
    # it, and a step taken afterwards is the run's own next step at the LR the save or the load left.
    scheduler_state, loader_state, generators = (
        scheduler.get_state_dict(),
        loader.get_state_dict(),
        capture_process_rng(),
    )
    scheduler.step()
    out["train:next_lrs"] = np.asarray(scheduler.get_last_lrs())
    loader.advance()
    out["train:loader_position"] = np.asarray(loader.position)
    out["train:ttml_generator"] = np.asarray(ttml.autograd.AutoContext.get_instance().get_generator_state())
    out["train:python_next"] = np.asarray(random.random())
    out["train:numpy_next"] = np.asarray(np.random.random())
    scheduler.set_state_dict(scheduler_state)  # writes the LRs back into the optimizer as they were
    loader.set_state_dict(loader_state)
    tokens, mask, _ = inputs(config, seed + 1000)
    logits = model(tokens, mask)
    for coord, arr in local_shards(logits).items():
        out[f"logits:{tag}:{coord_key(coord)}"] = arr
    ttml.autograd.AutoContext.get_instance().reset_graph()
    restore_process_rng(generators)  # after the forward too, which draws from the ttml generator with dropout
    return out


RESUME = "resume:"


def resume_reference(
    model, params, optimizer, scheduler, loader, config, seed: int, rank: int, step: int, mark=lambda what: None
) -> dict[str, np.ndarray]:
    """One more training step, numbered ``step`` so it draws that step's batch, then this rank's part again under the
    ``resume:`` prefix: the run one step after the checkpoint. Taken on the saving world after its save and on the
    loading world after its load, the two agree only if the optimizer carried its moments and step count across."""
    train(model, params, optimizer, scheduler, loader, config, 1, seed, mark, start=step)
    part = local_reference(model, params, optimizer, scheduler, loader, config, seed, rank)
    return {RESUME + key: value for key, value in part.items()}


def merge_parts(paths: list[str]) -> dict[str, np.ndarray]:
    """The union of the ranks' reference parts. Shards and logits are keyed by coordinate and never collide; the
    scalars, the labels and what the run does next are taken from rank 0 (another rank's generators may differ
    when the checkpoint held no entry for it)."""
    merged: dict[str, np.ndarray] = {}
    for path in sorted(paths):
        with np.load(path) as part:
            for key in part.files:
                if key in merged:
                    continue
                merged[key] = part[key]
    return merged


def canonical(parts: dict[str, np.ndarray], config: LlamaConfig) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Every tensor in canonical layout, the full logits and the scalars, from the merged parts; and, per tensor
    whose replicas differ between the devices or ranks of that world, every distinct canonical copy. The parts
    dumped after the step that follows the checkpoint come back the same way under the ``resume:`` prefix."""
    out, replicas = _canonical_part({k: v for k, v in parts.items() if not k.startswith(RESUME)}, config)
    resume = {k[len(RESUME) :]: v for k, v in parts.items() if k.startswith(RESUME)}
    if resume:
        after, after_replicas = _canonical_part(resume, config)
        out.update({RESUME + k: v for k, v in after.items()})
        replicas.update({RESUME + k: v for k, v in after_replicas.items()})
    return out, replicas


def _canonical_part(parts: dict[str, np.ndarray], config: LlamaConfig) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    meta = json.loads(str(parts["meta"]))
    out: dict[str, np.ndarray] = {}
    copies: dict[str, dict[tuple[int, ...], list[np.ndarray]]] = {}
    for key, value in parts.items():
        if key.startswith("shard:"):
            _, kind, rest = key.split(":", 2)
            name, _rank, coord = rest.rsplit(":", 2)
            copies.setdefault(f"{kind}:{name}", {}).setdefault(tuple(int(c) for c in coord.split("-")), []).append(
                value
            )
        elif key.startswith("logits:"):
            _, _rank, coord = key.split(":")
            copies.setdefault("logits", {}).setdefault(tuple(int(c) for c in coord.split("-")), []).append(value)
        elif key.startswith("scalar:") or key.startswith("train:"):
            out[key] = value
    replicas: dict[str, list] = {}
    for name, by_coord in copies.items():
        if name == "logits":
            continue
        kind, cname = name.split(":", 1)
        leaf = cname.rsplit(":", 1)[-1] if kind == "state" else cname
        distinct: list[np.ndarray] = []
        for selection in replica_selections(str(parts["label:" + name]), by_coord):
            full = canonicalize(leaf, assemble(str(parts["label:" + name]), selection), config, meta["tp"])
            if not any(np.array_equal(full, c) for c in distinct):
                distinct.append(full)
        out[name] = distinct[0]
        if len(distinct) > 1:
            replicas[name] = distinct
    by_coord = copies["logits"]
    if any(len(v) > 1 and not all(np.array_equal(v[0], c) for c in v[1:]) for v in by_coord.values()):
        replicas["logits"] = []
    if meta["tp"] > 1:
        axis = meta["tp_axis"]
        columns = []
        for j in range(meta["tp"]):
            coord = tuple(j if a == axis else 0 for a in range(len(meta["mesh"])))
            columns.append(by_coord[coord][0])
        full = np.concatenate(columns, axis=3)
    else:
        full = next(iter(by_coord.values()))[0]
    out["logits"] = full[..., : config.vocab_size]
    return out, replicas


def compare(
    actual: tuple[dict[str, np.ndarray], dict[str, list]],
    expected: tuple[dict[str, np.ndarray], dict[str, list]],
    logits_tolerance: float,
) -> list[str]:
    """Every tensor of the loaded world against the saving world's. A box with one copy must match exactly; where
    the saving world held several copies (its replicas drifted), every loaded copy must equal one of them, and the
    logits are reported but not judged, since that world's forward was not consistent.

    The entries dumped after the step that follows the checkpoint (``resume:``) are exact on the same layout. Across
    layouts the forward and the step round differently: the post-step moments must lie within ``logits_tolerance``
    of their scale and the post-step parameter update within a quarter of its mean size (a fresh optimizer's update
    differs from the restored one's by about its whole size). A saving world with drifted replicas took its step
    from inconsistent copies, so its post-step entries are not judged either."""
    actual, actual_replicas = actual
    expected, expected_replicas = expected
    problems = []
    exact = logits_tolerance == 0.0
    if expected_replicas:
        print(
            f"NOTE: {len(expected_replicas)} tensor(s) had replicas that differ between the saving world's devices or "
            f"ranks ({', '.join(sorted(expected_replicas)[:4])}{', ...' if len(expected_replicas) > 4 else ''}); a loaded "
            "value may equal any copy, and the logits are reported but not judged"
        )
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
            if rel > logits_tolerance and not expected_replicas:
                problems.append(f"{key} differ by {rel:.4f} of their scale (tolerance {logits_tolerance})")
            continue
        if base is not None and expected_replicas:
            continue
        if not exact and base is not None and base.startswith("state:"):
            scale = float(np.max(np.abs(e))) + 1e-6
            rel = float(np.max(np.abs(a - e))) / scale
            if rel > logits_tolerance:
                problems.append(f"{key}: differs by {rel:.4f} of its scale (tolerance {logits_tolerance})")
            continue
        if not exact and base is not None and base.startswith("param:") and base in actual and base in expected:
            # The update each world applied. The loaded parameters equal the saved ones (checked under their own
            # key), so the updates differ only where the steps did, whatever the parameters' own magnitude.
            delta, delta_ref = a - actual[base], e - expected[base]
            size = float(np.mean(np.abs(delta_ref)))
            if size == 0.0:
                continue  # an update below bf16 resolution moved nothing in the reference run either
            rel = float(np.mean(np.abs(delta - delta_ref))) / size
            if rel > 0.25:
                problems.append(f"{key}: the update differs by {rel:.2f} of its mean size (tolerance 0.25)")
            continue
        loaded_copies = actual_replicas.get(key, [a])
        saved_copies = expected_replicas.get(key, [e])
        if a.dtype.kind in "US" or e.dtype.kind in "US":
            if not np.array_equal(a, e):
                problems.append(f"{key}: text differs")
        elif len(saved_copies) == 1 and len(loaded_copies) == 1:
            if not np.array_equal(a, e):
                n = int(np.count_nonzero(a != e))
                problems.append(f"{key}: {n} of {a.size} values differ (max |diff| {float(np.max(np.abs(a - e))):.4g})")
        else:
            for copy in loaded_copies:
                if not any(np.array_equal(copy, saved) for saved in saved_copies):
                    problems.append(f"{key}: a loaded copy matches none of the {len(saved_copies)} saved copies")
                    break
    return problems


def part_path(base: str, rank: int) -> str:
    return f"{base}-w{rank:03d}.npz"


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
        help="skip the training step taken after the save and after the load (and its comparison); every rank of "
        "a world must agree, and a load compares the step only when the save took it too",
    )
    args = parser.parse_args(argv)

    mesh_shape = parse_mesh(args.mesh)
    axes = tuple(args.axes.split(","))
    reference_base = args.dir.rstrip("/") + ".reference"
    actual_base = args.dir.rstrip("/") + ".actual"
    saved_resume = False
    if args.command == "load":
        with np.load(reference_base + ".config.npz") as saved:
            sizes = json.loads(str(saved["config"]))
            saved_resume = "resume_step" in saved.files and bool(saved["resume_step"])
    else:
        sizes = dict(
            hidden_size=args.hidden,
            num_attention_heads=args.heads,
            num_key_value_heads=args.kv_heads,
            vocab_size=args.vocab,
        )
    resumed = args.resume_step and (args.command == "save" or saved_resume)
    opened = True
    try:
        open_world(mesh_shape, axes, args.tp, args.mgd)
    except RuntimeError as err:  # the descriptor does not fit this host's fabric, too few devices, ...
        print(f"MESH_OPEN_FAILED: {str(err).splitlines()[0]}")
        opened = False
    if not agree(opened):  # a rank that could not open its mesh must not leave the others waiting in a collective
        if opened:
            ttml.close_device_mesh()
        return MESH_OPEN_FAILED
    rank, size = world_group()

    def mark(what: str) -> None:
        """Progress markers: when a world is killed for running too long, the last one says which phase hung."""
        print(f"[driver] rank {rank}/{size}: {what}", flush=True)

    mark("mesh open")
    try:
        # Every generator a world draws from, before the model is built: ttml.init has its own generator, seeded from
        # OS entropy unless told otherwise, so without this every rank of a multi-host world would start from
        # different weights and the ranks' replicas could never agree. set_seed seeds it with the rest.
        set_seed(args.seed)
        config, model, params, optimizer, scheduler = build(args.tp, args.fsdp, args.tie, sizes)
        loader = Loader()
        mark("model built")
        if args.command == "save":
            train(model, params, optimizer, scheduler, loader, config, args.steps, args.seed, mark)
            mark("trained; saving")
            result = dcp.save(
                args.dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=loader,
                header={"step": args.steps, "mesh": list(mesh_shape), "axes": list(axes), "ranks": size},
            )
            mark("saved; dumping the reference")
            try:
                part = local_reference(model, params, optimizer, scheduler, loader, config, args.seed, rank)
                if resumed:
                    part.update(
                        resume_reference(
                            model, params, optimizer, scheduler, loader, config, args.seed, rank, args.steps, mark
                        )
                    )
                np.savez(part_path(reference_base, rank), **part)
                ok = True
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                ok = False
            if not agree(ok):
                print("RESULT: the reference could not be written on every rank")
                return 1
            if rank == 0:
                np.savez(reference_base + ".config.npz", config=json.dumps(sizes), resume_step=np.asarray(resumed))
                print(
                    json.dumps(
                        {
                            "saved": result.path,
                            "ranks": size,
                            "bytes_written": result.bytes_written,
                            "files": len(result.files),
                            "keys": result.keys,
                            "stats": result.stats,
                        }
                    )
                )
            return 0
        mark("loading")
        result = dcp.load(args.dir, model=model, optimizer=optimizer, scheduler=scheduler, dataloader=loader, rng=True)
        mark("loaded; dumping the loaded world")
        try:
            part = local_reference(model, params, optimizer, scheduler, loader, config, args.seed, rank)
            if resumed:
                part.update(
                    resume_reference(
                        model, params, optimizer, scheduler, loader, config, args.seed, rank, args.steps, mark
                    )
                )
            else:
                print("NOTE: no training step after the load; the run one step on is not compared")
            np.savez(part_path(actual_base, rank), **part)
            ok = True
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            ok = False
        if not agree(ok):
            print("RESULT: the loaded world could not be dumped on every rank")
            return 1
        problems: list[str] = []
        if rank == 0:
            try:
                expected_parts = merge_parts(glob.glob(reference_base + "-w*.npz"))
                if not resumed:
                    expected_parts = {k: v for k, v in expected_parts.items() if not k.startswith(RESUME)}
                expected = canonical(expected_parts, config)
                actual = canonical(merge_parts(glob.glob(actual_base + "-w*.npz")), config)
                problems = compare(actual, expected, 0.0 if args.same_layout else args.logits_tolerance)
            except Exception as err:  # noqa: BLE001
                traceback.print_exc()
                problems.append(f"comparison failed: {err}")
        if result.header.get("step") != args.steps:
            problems.append(f"header step {result.header.get('step')} != {args.steps}")
        expected_restored = ["scheduler", "dataloader"]
        world_rank = ttnn.distributed_context_world_rank() if size > 1 else 0
        if str(int(world_rank)) in dcp.read_train_state(args.dir).get("rng", {}).get("ranks", {}):
            expected_restored += ["rng.ttml", "rng.python", "rng.numpy", "rng.ttml_init"]
            if "torch" in sys.modules:
                expected_restored.append("rng.torch")
        if result.restored != expected_restored:
            problems.append(f"restored {result.restored} != {expected_restored}")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "loaded": result.path,
                        "ranks": size,
                        "resumed": resumed,
                        "fast_path_keys": len(result.fast_path_keys),
                        "missing": result.missing,
                        "unexpected": result.unexpected,
                        "restored": result.restored,
                        "stats": result.stats,
                    }
                )
            )
        for problem in problems:
            print("MISMATCH:", problem)
        verdict = agree(not problems)
        if rank == 0:
            print("RESULT:", "OK" if verdict else f"{len(problems)} mismatch(es)")
        return 0 if verdict else 1
    finally:
        ttml.close_device_mesh()


if __name__ == "__main__":
    sys.exit(main())
