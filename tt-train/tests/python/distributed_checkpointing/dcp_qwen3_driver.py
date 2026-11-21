# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run one world of a qwen3 cross-world checkpoint test in its own process (for ``test_dcp_qwen3.py``).

The model is the qwen3 example's ``DistributedQwen3ForCausalLM`` (``tt-train/sources/examples/qwen3``) on a
``[dp, tp]`` mesh. ``save`` trains a few steps, saves model, AdamW, a cosine scheduler, a stand-in dataloader and the
generators, and writes a reference next to the checkpoint: every parameter and AdamW moment in canonical layout, the
optimizer scalars, the logits on a fixed batch, what a resumed run does next, and the same again one training step
later. ``load`` builds a fresh model (other initial weights) on its own world, loads, and compares, under the rules of
``dcp_driver.compare``. The reference does not go through the checkpoint module: shards are gathered by their
topology and the model's TP layout is undone here (kv_proj's per-shard ``[K_j | V_j]`` rows, the padded vocab rows),
so a wrong layout declaration in the model shows up as a mismatch. A replicated tensor whose copies differ between
devices is a failure in either world: a checkpoint keeps one copy, so the model must keep them equal (#65654).

    python dcp_qwen3_driver.py save --dir /tmp/q/step_0000002 --mesh 1x2 --mgd <mgd>/bh_1_2_line_line.textproto --tie
    python dcp_qwen3_driver.py load --dir /tmp/q/step_0000002 --mesh 1x8 --mgd <mgd>/bh_1_8_line_line.textproto
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

_TT_TRAIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_TT_TRAIN, "sources", "examples", "qwen3"))  # first: the example's ``utils`` package

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

import ttml  # noqa: E402
import ttml.distributed_checkpointing as dcp  # noqa: E402
from ttml.common.schedulers import CosineAnnealingScheduler  # noqa: E402
from ttml.common.utils import set_seed  # noqa: E402
from ttml.distributed_checkpointing._train_state import capture_process_rng, restore_process_rng  # noqa: E402
from ttml.models import WeightTyingType  # noqa: E402
from ttml.models.qwen3 import Qwen3Config  # noqa: E402

from dcp_driver import (
    DESCRIPTOR_MISMATCH,
    MESH_OPEN_FAILED,
    RESUME,
    assemble,
    compare,
    label_json,
    local_shards,
    replicas_differ,
    uninterleave,
)  # noqa: E402
from dcp_oracle import Loader  # noqa: E402
from model_qwen3_distributed import DistributedQwen3ForCausalLM, tp_padded_vocab_size  # noqa: E402
from utils.lora import LORA_TARGETS_ALL, inject_adapter_in_model  # noqa: E402

SEQ_LEN = 32
T_MAX = 16
SIZES = dict(hidden=256, intermediate=512, layers=2, heads=8, kv_heads=8, head_dim=64)


def parse_mesh(text: str) -> tuple[int, int]:
    dp, tp = (int(x) for x in text.lower().split("x"))
    return dp, tp


def open_world(dp: int, tp: int, mgd: str | None) -> None:
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    ttml.open_device_mesh(ttml.Mesh((dp, tp), ("dp", "tp")))
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=dp > 1, enable_tp=True))


def make_config(sizes: dict) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=sizes["hidden"],
        intermediate_size=sizes["intermediate"],
        num_hidden_layers=sizes["layers"],
        num_attention_heads=sizes["heads"],
        num_key_value_heads=sizes["kv_heads"],
        head_dim=sizes["head_dim"],
        vocab_size=sizes["vocab"],
        max_position_embeddings=2 * SEQ_LEN,
        weight_tying=WeightTyingType.Enabled if sizes["tie"] else WeightTyingType.Disabled,
    )


def build(sizes: dict, init_seed: int):
    torch.manual_seed(init_seed)  # the example draws its initial weights from torch
    config = make_config(sizes)
    model = DistributedQwen3ForCausalLM(config, tie_word_embeddings=sizes["tie"], shard_dim=1)
    if sizes.get("lora"):
        # As the example's train.py: adapters on every target, base frozen, AdamW over the LoRA params only.
        inject_adapter_in_model(model, {"targets": list(LORA_TARGETS_ALL), "rank": 8, "alpha": 8})
        everything = model.parameters()
        for name, param in everything.items():
            if "lora" not in name:
                param.set_requires_grad(False)
        params = {name: param for name, param in everything.items() if "lora" in name}
        optimizer = ttml.optimizers.AdamW(
            params, ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
        )
        return config, model, everything, optimizer, CosineAnnealingScheduler(optimizer, T_max=T_MAX)
    params = model.parameters()
    cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    optimizer = ttml.optimizers.AdamW(params, cfg)
    return config, model, params, optimizer, CosineAnnealingScheduler(optimizer, T_max=T_MAX)


def inputs(vocab: int, seed: int):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=(1, 1, 1, SEQ_LEN)).astype(np.uint32)
    targets = rng.integers(0, vocab, size=(1, SEQ_LEN)).astype(np.uint32)
    mask = np.tril(np.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=np.float32))
    return (
        ttml.autograd.Tensor.from_numpy(tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32),
        ttml.autograd.Tensor.from_numpy(mask, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16),
        ttml.autograd.Tensor.from_numpy(targets, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32),
    )


def lm_columns(config: Qwen3Config, tie: bool) -> int:
    """The LM head's output width on this world: padded to 32 * tp when tied, the logical vocab when untied."""
    return tp_padded_vocab_size(config.vocab_size) if tie else config.vocab_size


def padding_mask(columns: int, vocab: int):
    """-1e4 on the padded vocab columns, sharded over the TP axis like the logits (the vocab-parallel loss counts
    every local column, issue #64714; masking makes every world train the same logical-vocab loss)."""
    shape = tuple(ttml.mesh().shape)
    full = np.broadcast_to(
        np.where(np.arange(columns) < vocab, 0.0, -1e4).astype(np.float32), (1, 1, SEQ_LEN, columns)
    ).copy()
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=shape, dims=(None, 3))
    return ttml.autograd.Tensor.from_numpy(
        full, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16, mapper=mapper
    )


def loss_of(model, config, tie, tokens, mask, targets):
    logits = model(tokens, mask)
    columns = lm_columns(config, tie)
    if columns != config.vocab_size:
        logits = ttml.ops.binary.add(logits, padding_mask(columns, config.vocab_size))
    if ttml.mesh().axis_size("tp") > 1:
        return ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
            logits, targets, cluster_axis=ttml.mesh().axis_index("tp"), reduce=ttml.ops.ReduceType.MEAN
        )
    return ttml.ops.loss.cross_entropy_loss(logits, targets, ttml.ops.ReduceType.MEAN)


def drifted(params, optimizer) -> list[str]:
    """Replicated parameters and moments whose copies differ between this world's devices."""
    tensors = dict(params.items())

    def walk(node, path):
        if isinstance(node, ttml.NamedParameters):
            for name, tensor in node.items():
                tensors["/".join(path) + ":" + name] = tensor
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (str(key),))

    walk(optimizer.get_state_dict(), ())
    out = []
    for name, tensor in tensors.items():
        copies = {coord: [arr] for coord, arr in local_shards(tensor).items()}
        if replicas_differ(label_json(tensor), copies):
            out.append(name)
    return out


def drift_problem(params, optimizer, when: str) -> str | None:
    """A description of the replicated tensors whose device copies differ, or None: a checkpoint keeps one copy of a
    replicated tensor, so a model that lets its copies drift apart cannot be restored as it ran (#65654)."""
    names = drifted(params, optimizer)
    if not names:
        return None
    short = sorted({n.replace("DistributedQwen3ForCausalLM/model/", "") for n in names})
    return (
        f"{when}, {len(names)} replicated tensor(s) differ between devices: {', '.join(short[:8])}"
        f"{' ...' if len(short) > 8 else ''}"
    )


def train(model, params, optimizer, scheduler, loader, config, tie, steps, seed, start=0):
    """``steps`` steps numbered from ``start``; step n draws batch seed + n. Generators are seeded only at step 0."""
    ctx = ttml.autograd.AutoContext.get_instance()
    if start == 0:
        set_seed(seed)
    needs_sync = ttml.mesh().axis_size("dp") > 1
    for step in range(start, start + steps):
        tokens, mask, targets = inputs(config.vocab_size, seed + step)
        optimizer.zero_grad()
        loss = loss_of(model, config, tie, tokens, mask, targets)
        loss.backward(False)
        if needs_sync:
            ttml.sync_gradients(params)
        optimizer.step()
        scheduler.step()
        loader.advance()
        ctx.reset_graph()


def canonicalize(name: str, array: np.ndarray, config: Qwen3Config, tp: int) -> np.ndarray:
    """Undo the model's TP layout: kv_proj rows are [K_0 | V_0 | K_1 | V_1 | ...] per shard; the embedding and LM
    head rows past the logical vocab are padding whose count depends on the TP size."""
    kv_out = config.num_key_value_heads * config.head_dim
    if "self_attn/kv_proj/" in name and (name.endswith("weight") or name.endswith("lora_B")):
        return uninterleave(array, 2, (kv_out, kv_out), tp)  # kv_proj/weight, kv_proj/base_layer/weight, lora_B
    if "self_attn/kv_proj/" in name and name.endswith("col_bias"):
        return uninterleave(array, 3, (kv_out, kv_out), tp)
    if name.endswith("lm_head/weight") or name.endswith("embed_tokens/weight"):
        return np.take(array, range(config.vocab_size), axis=2)
    return array


def full(tensor) -> np.ndarray:
    return assemble(label_json(tensor), local_shards(tensor))


def reference(model, params, optimizer, scheduler, loader, config, tie, seed) -> dict[str, np.ndarray]:
    tp = ttml.mesh().axis_size("tp")
    out: dict[str, np.ndarray] = {}
    for name, tensor in params.items():
        out["param:" + name] = canonicalize(name, full(tensor), config, tp)

    def walk(node, path):
        if isinstance(node, ttml.NamedParameters):
            for name, tensor in node.items():
                out["state:" + "/".join(path) + ":" + name] = canonicalize(name, full(tensor), config, tp)
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (str(key),))
        else:
            out["scalar:" + "/".join(path)] = np.asarray(node)

    walk(optimizer.get_state_dict(), ())
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
    out["train:torch_next"] = torch.rand(2).numpy()
    out["train:ttml_init_next"] = ttml.init._rng.random(2)
    scheduler.set_state_dict(scheduler_state)
    loader.set_state_dict(loader_state)
    tokens, mask, _ = inputs(config.vocab_size, seed + 1000)
    logits = model(tokens, mask)
    # The logits' topology label does not describe their vocab split, so join the TP shards along the vocab dim
    # (they are replicated over DP: take row 0).
    shards = local_shards(logits)
    out["logits"] = np.concatenate([shards[(0, j)] for j in range(tp)], axis=3)[..., : config.vocab_size]
    ttml.autograd.AutoContext.get_instance().reset_graph()
    restore_process_rng(generators)
    return out


def resume_reference(model, params, optimizer, scheduler, loader, config, tie, seed, step):
    train(model, params, optimizer, scheduler, loader, config, tie, 1, seed, start=step)
    part = reference(model, params, optimizer, scheduler, loader, config, tie, seed)
    return {RESUME + k: v for k, v in part.items()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["save", "load"])
    p.add_argument("--dir", required=True)
    p.add_argument("--mesh", required=True, help="dp x tp, e.g. 1x2")
    p.add_argument("--mgd", default=None)
    p.add_argument("--tie", action="store_true", help="save: tie the embedding and the LM head")
    p.add_argument("--lora", action="store_true", help="save: LoRA on every target, base frozen (as train.py)")
    p.add_argument("--vocab", type=int, default=300, help="save: vocab size (load reads it from the reference)")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--same-layout", action="store_true", help="load: the save's world; everything must be exact")
    p.add_argument("--logits-tolerance", type=float, default=0.1)
    args = p.parse_args(argv)

    dp, tp = parse_mesh(args.mesh)
    ref_path = args.dir.rstrip("/") + ".reference.npz"
    if args.command == "load":
        expected = dict(np.load(ref_path))
        sizes = json.loads(str(expected.pop("config")))
    else:
        sizes = dict(SIZES, vocab=args.vocab, tie=args.tie, lora=args.lora)
    try:
        open_world(dp, tp, args.mgd)
    except RuntimeError as err:
        if DESCRIPTOR_MISMATCH not in str(err):
            raise
        print(f"MESH_OPEN_FAILED: {str(err).splitlines()[0]}")
        return MESH_OPEN_FAILED
    try:
        init_seed = 1234 if args.command == "save" else 5678  # the load world starts from other weights
        config, model, params, optimizer, scheduler = build(sizes, init_seed)
        loader = Loader()
        tie = sizes["tie"]
        print(
            f"world {dp}x{tp}: vocab {config.vocab_size}, LM head columns {lm_columns(config, tie)}, tie {tie}, "
            f"lora {bool(sizes.get('lora'))}"
        )
        if args.command == "save":
            train(model, params, optimizer, scheduler, loader, config, tie, args.steps, args.seed)
            drift = drift_problem(params, optimizer, f"after {args.steps} training steps")
            if drift:
                print("MISMATCH:", drift)
                print("RESULT: the saving world's replicas differ")
                return 1
            result = dcp.save(
                args.dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=loader,
                header={"step": args.steps, "mesh": [dp, tp]},
            )
            ref = reference(model, params, optimizer, scheduler, loader, config, tie, args.seed)
            ref.update(
                resume_reference(model, params, optimizer, scheduler, loader, config, tie, args.seed, args.steps)
            )
            np.savez(ref_path, config=json.dumps(sizes), **ref)
            print(
                json.dumps(
                    {
                        "saved": result.path,
                        "keys": result.keys,
                        "files": len(result.files),
                        "bytes": result.bytes_written,
                    }
                )
            )
            return 0
        first = next(iter(params))
        fresh = canonicalize(first, full(params[first]), config, tp)
        result = dcp.load(args.dir, model=model, optimizer=optimizer, scheduler=scheduler, dataloader=loader, rng=True)
        actual = reference(model, params, optimizer, scheduler, loader, config, tie, args.seed)
        actual.update(resume_reference(model, params, optimizer, scheduler, loader, config, tie, args.seed, args.steps))
        problems = compare(actual, expected, 0.0 if args.same_layout else args.logits_tolerance)
        drift = drift_problem(params, optimizer, "after the load and one more step")
        if drift:
            problems.append(drift)
        if np.array_equal(fresh, expected["param:" + first]):
            problems.append("precondition: the fresh model already held the saved weights")
        if result.header.get("step") != args.steps:
            problems.append(f"header step {result.header.get('step')} != {args.steps}")
        want = ["scheduler", "dataloader", "rng.ttml", "rng.python", "rng.numpy", "rng.ttml_init", "rng.torch"]
        if result.restored != want:
            problems.append(f"restored {result.restored} != {want}")
        print(
            json.dumps(
                {
                    "loaded": result.path,
                    "fast_path_keys": len(result.fast_path_keys),
                    "missing": result.missing,
                    "unexpected": result.unexpected,
                    "restored": result.restored,
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
