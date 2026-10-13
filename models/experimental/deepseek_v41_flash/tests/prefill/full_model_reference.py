# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Streamed fp32 CPU reference of the whole DeepSeek-V4.1-Flash network, on the real checkpoint.

The checkpoint's own ``inference/model.py`` (through ``tests/reference.py``: dequantized weights, the activation
quantization simulations off) over the whole prompt in one pass at ``start_pos = 0``, as ``Transformer.forward``
runs it, but one layer resident at a time: a layer's ``model.Block`` is built, run over ``(streams, pre_mix)`` and
freed (the routed experts are built per call). An Engram layer's ``model.Engram`` runs ahead of its block, with the
checkpoint's ``NgramHashState`` hashes and its n-gram table (~101 GB, read for that layer only).

The compressed KV, index keys and selections a layer hands down the stack live in the module's ``shared_attn``;
they are saved after every kv / index source, so a run interrupted after layer ``i`` resumes from layer ``i + 1``.

Cached on disk, keyed by the prompt tokens and the layer count: the embedding streams, every layer's output streams
and ``pre_mix``, and the logits of every token (collapse with the last ``pre_mix``, final norm, ``head``). No device
is touched, so it can run ahead of the device test, alone::

    python -m models.experimental.deepseek_v41_flash.tests.prefill.full_model_reference --len 2048

The device's prefill selects with the indexer as the reference does; the reference's candidate pre-filter keeps every
block up to ``candidate_topk_blocks * candidate_block_size`` = 16384 compressed entries, so the two are the same
computation for prompts up to 16384 tokens.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from models.experimental.deepseek_v41_flash.tests.reference import (
    load_reference,
    model_args,
    reference_blocks,
    reference_engram,
    reference_engram_hash,
)
from models.experimental.deepseek_v41_flash.tt.config import dequantized
from models.experimental.deepseek_v41_flash.tt.engram_lookup import load_engram_table
from models.experimental.deepseek_v4_flash.tests.prefill.full_model_reference import (
    ReferenceStore as V4ReferenceStore,
    default_reference_dir,
)
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_SHARED_FIELDS = ("compress_kv", "index_k", "topk_idxs", "candidates")


class ReferenceStore(V4ReferenceStore):
    """V4's store plus each layer's output ``pre_mix`` (``pre_{i}``) and the shared attention state (``shared_{i}``).

    Tensors are memory-mapped on read, so a caller slicing one chunk's tokens does not read the whole file.
    """

    def get(self, name: str):
        return torch.load(self.path(name), map_location="cpu", weights_only=True, mmap=True)

    def pre_mix_input(self, layer_idx: int) -> torch.Tensor:
        """The ``pre_mix`` ``[1, T, hc]`` entering layer ``layer_idx``: one-hot ``[1, 0, ..]`` for layer 0."""
        if layer_idx > 0:
            return self.get(f"pre_{layer_idx - 1:02d}")
        streams = self.get("embedding")
        one_hot = torch.zeros(1, streams.shape[1], streams.shape[2])
        one_hot[..., 0] = 1
        return one_hot

    def pre_mix_output(self, layer_idx: int) -> torch.Tensor:
        return self.get(f"pre_{layer_idx:02d}")

    def complete(self) -> bool:
        return super().complete() and all(self.has(f"pre_{i:02d}") for i in range(self.num_layers))


def store_for(ids: torch.Tensor, num_layers: int, root: Optional[Path] = None) -> ReferenceStore:
    """The store for these token ids (keyed by the ids and the layer count)."""
    key = hashlib.sha1(ids.to(torch.int64).numpy().tobytes() + f"|v4.1|{num_layers}".encode()).hexdigest()
    directory = (root or default_reference_dir() / "v41") / f"T{ids.shape[1]}_L{num_layers}_{key[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    return ReferenceStore(directory, num_layers)


@torch.no_grad()
def run_reference(
    snapshot: Path,
    loader: DeepseekV4WeightLoader,
    tokenizer,
    ids: torch.Tensor,
    store: ReferenceStore,
    progress: Optional[Callable[[str], None]] = None,
) -> ReferenceStore:
    """Fill ``store`` for ``ids`` ``[1, T]`` (resuming after the last cached layer). Returns ``store``."""
    note = progress or (lambda message: logger.info(message))
    if store.complete():
        note(f"reference already cached in {store.dir}")
        return store
    seq_len = ids.shape[1]
    started = time.perf_counter()
    model = load_reference(snapshot)
    args = model_args(model, snapshot, seq_len)
    sources = set(args.kv_source_layers) | set(args.index_source_layers)

    if not store.has("embedding"):
        note("reference: embedding")
        table = dequantized(loader, "embed.weight")()
        store.put("embedding", table[ids].unsqueeze(2).repeat(1, 1, args.hc_mult, 1).contiguous())
        del table
    first_missing = next(
        (i for i in range(store.num_layers) if not (store.has(f"layer_{i:02d}") and store.has(f"pre_{i:02d}"))),
        store.num_layers,
    )
    streams = pre_mix = None
    if first_missing < store.num_layers:
        streams = store.layer_input(first_missing).clone()
        pre_mix = store.pre_mix_input(first_missing).clone()
        done = [i for i in sorted(sources) if i < first_missing]
        if done:
            saved = store.get(f"shared_{done[-1]:02d}")
            for name in _SHARED_FIELDS:
                setattr(model.shared_attn, name, None if saved.get(name) is None else saved[name].clone())
            note(f"reference: resuming at layer {first_missing}, shared attention state from layer {done[-1]}")

    hashes = None
    if any(first_missing <= i < store.num_layers for i in args.engram_layer_ids):
        hashes = reference_engram_hash(model, snapshot, tokenizer, seq_len)(ids, 0)  # [1, T, n_engram, n_hash_cols]

    for li in range(first_missing, store.num_layers):
        t0 = time.perf_counter()
        if li in args.engram_layer_ids:
            note(f"reference layer {li}: Engram table -> host RAM (~101 GB)")
            weight, scale = load_engram_table(loader, li)
            engram = reference_engram(model, snapshot, loader, li, weight, scale)
            streams = engram(streams, hashes[:, :, engram.layer_hash_index])
            del engram, weight, scale
        block = reference_blocks(model, snapshot, loader, [li], seq_len)[li]
        t_load = time.perf_counter() - t0
        streams, pre_mix = block(streams, 0, pre_mix, None)
        del block
        assert torch.isfinite(streams).all() and torch.isfinite(pre_mix).all(), f"layer {li}: non-finite values"
        store.put(f"pre_{li:02d}", pre_mix)
        store.put(f"layer_{li:02d}", streams)
        if li in sources:
            store.put(f"shared_{li:02d}", {name: getattr(model.shared_attn, name) for name in _SHARED_FIELDS})
        note(
            f"reference layer {li + 1}/{store.num_layers} (ratio {args.compress_ratios[li]}): build {t_load:.1f}s, "
            f"total {time.perf_counter() - t0:.1f}s, std {streams.std():.3f} "
            f"[{(time.perf_counter() - started) / 60:.1f} min so far]"
        )

    if not store.has("logits"):
        note("reference: collapse, final norm, head")
        if streams is None:
            last = store.num_layers - 1
            streams, pre_mix = store.layer_output(last), store.pre_mix_output(last)
        hidden = torch.sum(pre_mix.unsqueeze(-1) * streams.float(), dim=2)  # Block.hc_pre
        norm = model.RMSNorm(args.dim, args.norm_eps)
        norm.weight = torch.nn.Parameter(dequantized(loader, "norm.weight")(), requires_grad=False)
        store.put("logits", F.linear(norm(hidden), dequantized(loader, "head.weight")()))
    (store.dir / "meta.json").write_text(json.dumps({"tokens": seq_len, "layers": store.num_layers}))
    note(f"reference done in {(time.perf_counter() - started) / 60:.1f} min -> {store.dir}")
    return store


if __name__ == "__main__":  # pre-generate the reference for the verify test's prompt, host only
    import argparse

    from transformers import AutoTokenizer

    from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR
    from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
        _DEFAULT_PROMPT_FILE,
        _build_prompt_ids,
    )
    from models.experimental.deepseek_v4_flash.tt.weight_loader import resolve_snapshot_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--len", type=int, default=2048, help="prompt tokens (a multiple of 128)")
    parser.add_argument("--layers", type=int, default=None, help="only the first N layers")
    cli = parser.parse_args()

    snapshot_dir = resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    ckpt = DeepseekV4WeightLoader(snapshot_dir)
    tok = AutoTokenizer.from_pretrained(snapshot_dir)
    prompt_ids, _ = _build_prompt_ids(tok, json.loads(Path(_DEFAULT_PROMPT_FILE).read_text())[0], cli.len)
    with open(snapshot_dir / "inference" / "config.json") as f:
        total_layers = json.load(f)["n_layers"]
    n_layers = min(cli.layers or total_layers, total_layers)
    token_ids = torch.tensor(prompt_ids).unsqueeze(0)
    run_reference(snapshot_dir, ckpt, tok, token_ids, store_for(token_ids, n_layers))
