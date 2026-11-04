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

Two precisions:

- `fp32` (default, `<grid>-seed<seed>-fp32`): the reference model cast to fp32, with attention
  computed `CHUNK_SIZE` query rows at a time (`chunked_attention`). The device tests compare against
  these. Upstream's eager attention holds several 12 x L x L tensors at once: ~510 GB measured in
  bf16, roughly twice that in fp32, more than the host has; chunks need ~25 GB. The norms keep the
  bf16 model's eps (2^-7): upstream `nn.RMSNorm(eps=None)` takes eps from the input dtype, so fp32
  would otherwise change the model, not only the precision.
- `bf16` (`<grid>-seed<seed>`): the reference as it runs, bf16 with upstream eager attention. Its
  own rounding (bf16 attention probabilities and residual stream) is large where activations are
  large: at full size, layer 18 has PCC 0.983 against fp32, while the device has 0.99998. After
  the projector it is worse: on a 64x64 image the bf16 model's output has PCC 0.848 against fp32.

Pixels are `random_patches(config, grid, seed)`. The loop is `MoonViT3dEncoder.forward` with every
block's output kept: each block gets the arguments `forward` gives it (`reference_rope`).
`test_goldens.py` checks the loop against `encoder(...)` and the model, and `chunked_attention` against upstream
`eager_attention`, on small inputs. A run resumes from the last saved block (a finished folder only redoes the final norm and the
model output). `meta.json` records
hashes of the checkpoint index, the reference code and `chunked_attention`; `load` refuses a folder
made from different ones.

    python -m models.experimental.kimi_k3_moonvit_v2.tests.goldens --grid 256x256 [--dtype bf16] [--blocks 1]
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from models.experimental.kimi_k3_moonvit_v2.reference import modeling_kimi_k3 as upstream
from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.tests.common import random_patches, reference_rope

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
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CHUNK_SIZE = 2048
# The bf16 model's RMSNorm eps (`nn.RMSNorm(eps=None)` on bf16 input), kept when running in fp32.
NORM_EPS = torch.finfo(torch.bfloat16).eps


def golden_dir() -> Path | None:
    """`$MOONVIT_GOLDEN_DIR`, or None when it is unset."""
    value = os.environ.get(GOLDEN_DIR_ENV)
    return Path(value) if value else None


def case_dir(h: int, w: int, seed: int, dtype: str = "fp32") -> Path:
    root = golden_dir()
    if root is None:
        raise RuntimeError(f"set ${GOLDEN_DIR_ENV} to the folder that holds the saved outputs")
    suffix = "" if dtype == "bf16" else f"-{dtype}"
    return root / f"{h}x{w}-seed{seed}{suffix}"


def block_file(index: int) -> str:
    return f"block_{index:02d}.pt"


def sha256(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def chunked_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None, chunk_size: int = CHUNK_SIZE, **kwargs):
    """Upstream `eager_attention` for one image, `chunk_size` query rows at a time: the same scores,
    fp32 softmax and cast back to the input dtype, without the 12 x L x L tensors. One image means
    `cu_seqlens == [0, L]`, where eager's mask is all true, so there is nothing to mask."""
    if q_cu_seqlens.tolist() != [0, q.shape[0]] or k_cu_seqlens.tolist() != [0, k.shape[0]]:
        raise ValueError("chunked_attention handles one image (cu_seqlens [0, L]) only")
    q, k, v = (t.transpose(0, 1) for t in (q, k, v))  # [heads, L, head_dim]
    out = torch.empty_like(q)
    for start in range(0, q.shape[1], chunk_size):
        rows = slice(start, start + chunk_size)
        weight = q[:, rows] @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        out[:, rows] = torch.softmax(weight, dim=-1, dtype=torch.float32).to(q.dtype) @ v
    return out.transpose(0, 1).reshape(q.shape[1], -1)


@contextmanager
def use_chunked_attention():
    """Run upstream's `eager` attention path as `chunked_attention` inside the block."""
    previous = upstream.VL_VISION_ATTENTION_FUNCTIONS["eager"]
    upstream.VL_VISION_ATTENTION_FUNCTIONS["eager"] = chunked_attention
    try:
        yield
    finally:
        upstream.VL_VISION_ATTENTION_FUNCTIONS["eager"] = previous


def to_fp32(reference):
    """The reference in fp32 with the bf16 model's RMSNorm eps. Only the norms built with `eps=None`
    (the encoder's) take it from the dtype; the projector's `post_norm` sets its own (1e-5) and keeps it."""
    reference = reference.float()
    for module in reference.modules():
        if isinstance(module, torch.nn.RMSNorm) and module.eps is None:
            module.eps = NORM_EPS
    return reference


def fingerprint(checkpoint: Path, dtype: str) -> dict:
    """What a golden depends on besides the grid and seed."""
    attention = "eager"
    if dtype == "fp32":
        attention = "chunked " + hashlib.sha256(inspect.getsource(chunked_attention).encode()).hexdigest()
    return {
        "dtype": dtype,
        "attention": attention,
        "checkpoint_index_sha256": sha256([checkpoint / "model.safetensors.index.json"]),
        "reference_sha256": sha256([REFERENCE_DIR / name for name in REFERENCE_FILES]),
        "torch": torch.__version__,
    }


IDENTITY = ("grid", "seed", "dtype", "attention", "checkpoint_index_sha256", "reference_sha256")


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


def load(h: int, w: int, seed: int, name: str, dtype: str = "fp32") -> torch.Tensor:
    """A saved output (`patch_embed`, `block_00`, ..., `final_layernorm`, `mm_projector`), checked
    against the current checkpoint, reference code and (fp32) `chunked_attention`."""
    folder = case_dir(h, w, seed, dtype)
    meta = json.loads((folder / "meta.json").read_text())
    current = fingerprint(checkpoint_path(), dtype)
    for key in ("checkpoint_index_sha256", "reference_sha256", "attention"):
        # Folders written before `attention` was recorded are bf16 eager.
        if meta.get(key, "eager") != current[key]:
            raise RuntimeError(f"{folder} was made from a different {key}; regenerate it")
    return torch.load(folder / f"{name}.pt")


def generate(h: int, w: int, seed: int, dtype: str, blocks: int | None) -> None:
    from models.experimental.kimi_k3_moonvit_v2.tests.conftest import build_reference
    from models.experimental.kimi_k3_moonvit_v2.tt.model_config import MoonViTConfig

    checkpoint = checkpoint_path()
    if checkpoint is None:
        raise SystemExit("set KIMI_K3_HF_MODEL to the checkpoint folder")
    if golden_dir() is None:
        raise SystemExit(f"set {GOLDEN_DIR_ENV} to the folder to write the outputs to")
    folder = case_dir(h, w, seed, dtype)
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "meta.json"
    meta = {"grid": [1, h, w], "seed": seed, **fingerprint(checkpoint, dtype), "seconds": {}}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        old.setdefault("dtype", "bf16")
        old.setdefault("attention", "eager")
        if any(old[k] != meta[k] for k in IDENTITY):
            raise SystemExit(f"{folder} holds outputs of different inputs; move it away first")
        meta["seconds"] = old["seconds"]

    def save(name: str, tensor: torch.Tensor, seconds: float) -> None:
        torch.save(tensor.clone(), folder / f"{name}.pt")
        meta["seconds"][name] = round(seconds, 1)
        meta_path.write_text(json.dumps(meta, indent=2))
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"[goldens] {name}: {seconds:.1f} s, peak RSS so far {peak_gb:.1f} GB", flush=True)

    t = time.time()
    reference = build_reference("checkpoint", None)
    if dtype == "fp32":
        reference = to_fp32(reference)
    config = MoonViTConfig.from_hf_config(checkpoint)
    print(
        f"[goldens] {dtype} reference built in {time.time() - t:.1f} s, {config.num_hidden_layers} blocks", flush=True
    )
    pixels, grid_thws = random_patches(config, [(1, h, w)], seed)

    attention = use_chunked_attention() if dtype == "fp32" else nullcontext()
    with torch.no_grad(), attention:
        if (folder / "patch_embed.pt").exists():
            x = torch.load(folder / "patch_embed.pt")
        else:
            t = time.time()
            x = reference.vision_tower.patch_embed(pixels.to(DTYPES[dtype]), grid_thws)
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
    parser.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    parser.add_argument("--blocks", type=int, default=None, help="stop after this many blocks (default: all)")
    args = parser.parse_args()
    h, w = (int(v) for v in args.grid.split("x"))
    generate(h, w, args.seed, args.dtype, args.blocks)


if __name__ == "__main__":
    main()
