# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full-size reference outputs, generated once on the CPU and read back by the full-size device tests.

At the full padded length (65,536 patches) the reference takes minutes per layer, so the outputs
are saved instead of recomputed. One folder per case under `$MOONVIT_GOLDEN_DIR`:

    patch_embed.pt         `vision_tower.patch_embed(pixels, grid_thws)`
    block_00.pt ... 26     each encoder block's output, the first fed `patch_embed.pt`
    final_layernorm.pt     `encoder.final_layernorm` of the last block (the encoder's output)
    mm_projector.pt        the model's output: `tpool_patch_merger`, then `mm_projector` (h*w/4 rows)
    meta.json              what the files were made from, and the time each step took

The outputs are fp32 (folder `<grid>-seed<seed>-fp32`): the reference model cast to fp32, with
attention computed `CHUNK_SIZE` query rows at a time (`chunked_attention`). Upstream's eager attention
holds several 12 x L x L tensors at once, more than the host has. The norms keep the bf16 model's eps
(2^-7): upstream `nn.RMSNorm(eps=None)` takes eps from the input dtype, so fp32 would otherwise change
the model, not only the precision. The bf16 reference is not used: its own rounding (bf16 attention
probabilities and residual stream) is larger than the device's error in deeper layers.

Pixels are `random_patches(config, grid, seed)`. The loop is `MoonViT3dEncoder.forward` with every
block's output kept: each block gets the arguments `forward` gives it (`reference_rope`).
`test_goldens.py` checks the loop against `encoder(...)` and the model, and `chunked_attention`
against upstream `eager_attention`, on small inputs. A run resumes from the last saved block (a
finished folder only redoes the final norm and the model output). `meta.json` records hashes of the
checkpoint index, the checkpoint's `config.json`, the vision and projector shards, the reference code
and `chunked_attention`; `load` refuses a folder made from different ones.

    python -m models.experimental.kimi_k3_moonvit_v2.tests.goldens --grid 256x256 [--blocks 1]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import math
import os
import resource
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from models.experimental.kimi_k3_moonvit_v2.reference import modeling_kimi_k3 as upstream
from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import CHECKPOINT_PREFIXES, checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    REAL_MARK,
    build_reference,
    random_patches,
    reference_rope,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig

GOLDEN_DIR_ENV = "MOONVIT_GOLDEN_DIR"
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"
# The reference files the saved outputs depend on, hashed into `meta.json`. Not the image preprocessing
# (`kimi_k3_vision_processing.py`, `media_utils.py`): the outputs start from `random_patches`.
REFERENCE_FILES = (
    "__init__.py",
    "checkpoint.py",
    "configuration_kimi_k3.py",
    "kimi_k3_vision.py",
    "modeling_kimi_k3.py",
    "config.json",
)
CHUNK_SIZE = 2048
# Query rows per chunk in `use_sampled_attention`: it runs against all 65,536 keys, so smaller chunks.
SAMPLED_CHUNK_SIZE = 256
# The bf16 model's RMSNorm eps (`nn.RMSNorm(eps=None)` on bf16 input), kept when running in fp32. The
# device uses the same value (`MoonViTConfig.norm_eps`).
NORM_EPS = MoonViTConfig().norm_eps


def golden_dir() -> Path | None:
    """`$MOONVIT_GOLDEN_DIR`, or None when it is unset."""
    value = os.environ.get(GOLDEN_DIR_ENV)
    return Path(value) if value else None


def case_dir(h: int, w: int, seed: int) -> Path:
    root = golden_dir()
    if root is None:
        raise RuntimeError(f"set ${GOLDEN_DIR_ENV} to the folder that holds the saved outputs")
    return root / f"{h}x{w}-seed{seed}-fp32"


def block_file(index: int) -> str:
    return f"block_{index:02d}.pt"


def sha256(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
    return digest.hexdigest()


def vision_shards(checkpoint: Path) -> list[Path]:
    """The shards that hold the vision tower and projector weights."""
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {shard for key, shard in weight_map.items() if key.startswith(CHECKPOINT_PREFIXES)}
    return [checkpoint / shard for shard in sorted(shards)]


def chunked_attention(
    q, k, v, q_cu_seqlens=None, k_cu_seqlens=None, *, query_rows=None, chunk_size: int = CHUNK_SIZE, **kwargs
):
    """Upstream `eager_attention` for one image, `chunk_size` query rows at a time: the same scores,
    fp32 softmax and cast back to the input dtype, without the 12 x L x L tensors. One image means
    `cu_seqlens == [0, L]`, where eager's mask is all true, so there is nothing to mask.

    With `query_rows`, only those rows are computed, each against every key, and the other rows of the
    output are zero. At the full padded length this is the part of attention the CPU can afford: the
    rows after attention (`wo`, the residual, norm1, the MLP) each depend on their own row alone, so a
    block run with it is exact on `query_rows`."""
    if q_cu_seqlens.tolist() != [0, q.shape[0]] or k_cu_seqlens.tolist() != [0, k.shape[0]]:
        raise ValueError("chunked_attention handles one image (cu_seqlens [0, L]) only")
    if query_rows is None:
        query_rows = torch.arange(q.shape[0])
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))  # [heads, L, head_dim]
    out = torch.zeros_like(q)
    for start in range(0, len(query_rows), chunk_size):
        rows = query_rows[start : start + chunk_size]
        weight = q[:, rows] @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        out[:, rows] = torch.softmax(weight, dim=-1, dtype=torch.float32).to(q.dtype) @ v
    return out.transpose(0, 1).reshape(q.shape[1], -1)


@contextmanager
def _eager_attention_as(attention):
    """Run upstream's `eager` attention path as `attention` inside the block."""
    previous = upstream.VL_VISION_ATTENTION_FUNCTIONS["eager"]
    upstream.VL_VISION_ATTENTION_FUNCTIONS["eager"] = attention
    try:
        yield
    finally:
        upstream.VL_VISION_ATTENTION_FUNCTIONS["eager"] = previous


def use_chunked_attention():
    """Upstream's `eager` attention as `chunked_attention` over every row."""
    return _eager_attention_as(chunked_attention)


def use_sampled_attention(query_rows: torch.Tensor):
    """Upstream's `eager` attention as `chunked_attention` for `query_rows` (raster order) only."""
    return _eager_attention_as(
        functools.partial(chunked_attention, query_rows=query_rows, chunk_size=SAMPLED_CHUNK_SIZE)
    )


def to_fp32(reference):
    """The reference in fp32 with the bf16 model's RMSNorm eps. Only the norms built with `eps=None`
    (the encoder's) take it from the dtype; the projector's `post_norm` sets its own (1e-5) and keeps it."""
    reference = reference.float()
    for module in reference.modules():
        if isinstance(module, torch.nn.RMSNorm) and module.eps is None:
            module.eps = NORM_EPS
    return reference


@functools.cache
def checkpoint_fingerprint(checkpoint: Path) -> dict:
    """Hashes of the checkpoint files the reference reads. The shards are about 1 GB, so once per process."""
    return {
        "checkpoint_index_sha256": sha256([checkpoint / "model.safetensors.index.json"]),
        "checkpoint_config_sha256": sha256([checkpoint / "config.json"]),
        "vision_weights_sha256": sha256(vision_shards(checkpoint)),
    }


def fingerprint(checkpoint: Path) -> dict:
    """What a golden depends on besides the grid and seed."""
    return {
        "attention": "chunked " + hashlib.sha256(inspect.getsource(chunked_attention).encode()).hexdigest(),
        **checkpoint_fingerprint(checkpoint),
        "reference_sha256": sha256([REFERENCE_DIR / name for name in REFERENCE_FILES]),
        "torch": torch.__version__,
    }


# The `fingerprint` keys a golden must match; `torch` is recorded only.
FINGERPRINT_KEYS = (
    "attention",
    "checkpoint_index_sha256",
    "checkpoint_config_sha256",
    "vision_weights_sha256",
    "reference_sha256",
)
IDENTITY = ("grid", "seed", *FINGERPRINT_KEYS)


def model_output(reference, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
    """The rest of `KimiK3VisionModel.forward` after the encoder, for one image: `tpool_patch_merger`
    of the encoder output `x`, then `mm_projector`."""
    merged = upstream.tpool_patch_merger(x, grid_thws, merge_kernel_size=reference.vision_tower.merge_kernel_size)
    (out,) = reference.mm_projector(merged)
    return out


def encoder_steps(reference, x: torch.Tensor, grid_thws: torch.Tensor, start: int = 0, stop: int | None = None):
    """`MoonViT3dEncoder.forward` of `reference` (a `KimiK3VisionModel`) over blocks `start` to
    `stop`, given the input to block `start`: yields ("block_NN", output) per block, then
    ("final_layernorm", output) and ("mm_projector", the model's output) when `stop` is the last
    block. Nothing past `stop` is computed."""
    encoder = reference.vision_tower.encoder
    stop = len(encoder.blocks) if stop is None else stop
    cu_seqlens, max_seqlen, rope_freqs_cis = reference_rope(reference, grid_thws)
    for index in range(start, stop):
        x = encoder.blocks[index](x, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)
        yield block_file(index)[:-3], x
    if stop == len(encoder.blocks):
        x = encoder.final_layernorm(x)
        yield "final_layernorm", x
        yield "mm_projector", model_output(reference, x, grid_thws)


def load(h: int, w: int, seed: int, name: str) -> torch.Tensor:
    """A saved output (`patch_embed`, `block_00`, ..., `final_layernorm`, `mm_projector`), checked
    against the current checkpoint (index, config, vision weights), reference code and `chunked_attention`."""
    folder = case_dir(h, w, seed)
    meta = json.loads((folder / "meta.json").read_text())
    current = fingerprint(checkpoint_path())
    for key in FINGERPRINT_KEYS:
        if meta.get(key) != current[key]:
            raise RuntimeError(f"{folder} was made from a different {key}; regenerate it")
    return torch.load(folder / f"{name}.pt")


def generate(h: int, w: int, seed: int, blocks: int | None) -> None:
    checkpoint = checkpoint_path()
    if checkpoint is None:
        raise SystemExit("set KIMI_K3_HF_MODEL to the checkpoint folder")
    if golden_dir() is None:
        raise SystemExit(f"set {GOLDEN_DIR_ENV} to the folder to write the outputs to")
    folder = case_dir(h, w, seed)
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "meta.json"
    meta = {"grid": [1, h, w], "seed": seed, **fingerprint(checkpoint), "seconds": {}}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if any(old.get(k) != meta[k] for k in IDENTITY):
            raise SystemExit(f"{folder} holds outputs of different inputs; move it away first")
        meta["seconds"] = old["seconds"]

    def save(name: str, tensor: torch.Tensor, seconds: float) -> None:
        torch.save(tensor.clone(), folder / f"{name}.pt")
        meta["seconds"][name] = round(seconds, 1)
        meta_path.write_text(json.dumps(meta, indent=2))
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"[goldens] {name}: {seconds:.1f} s, peak RSS so far {peak_gb:.1f} GB", flush=True)

    t = time.time()
    reference = to_fp32(build_reference(REAL_MARK))
    config = MoonViTConfig.from_hf_config(checkpoint)
    print(f"[goldens] fp32 reference built in {time.time() - t:.1f} s, {config.num_hidden_layers} blocks", flush=True)
    pixels, grid_thws = random_patches(config, [(1, h, w)], seed)

    with torch.no_grad(), use_chunked_attention():
        if (folder / "patch_embed.pt").exists():
            x = torch.load(folder / "patch_embed.pt")
        else:
            t = time.time()
            x = reference.vision_tower.patch_embed(pixels.float(), grid_thws)
            save("patch_embed", x, time.time() - t)

        start = 0
        while (folder / block_file(start)).exists():
            start += 1
        if start:
            x = torch.load(folder / block_file(start - 1))
            print(f"[goldens] resuming at block {start}", flush=True)
        stop = config.num_hidden_layers if blocks is None else min(blocks, config.num_hidden_layers)

        t = time.time()
        for name, out in encoder_steps(reference, x, grid_thws, start, stop):
            save(name, out, time.time() - t)
            t = time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", default="256x256", help="HxW in patches")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=None, help="stop after this many blocks (default: all)")
    args = parser.parse_args()
    h, w = (int(v) for v in args.grid.split("x"))
    generate(h, w, args.seed, args.blocks)


if __name__ == "__main__":
    main()
