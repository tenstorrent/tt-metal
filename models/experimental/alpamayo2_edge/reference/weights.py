# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import math
import shutil
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

INDEX_FILE = "model.safetensors.index.json"
# The shard name every tensor of a random checkpoint maps to; no such file exists.
RANDOM_SHARD = "random"
_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def load_weights(model_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    """Return every checkpoint tensor under `prefix`, keyed with the prefix stripped.

    A checkpoint written by `make_random_checkpoint` yields its seeded random tensors instead of reading shards.
    """
    model_dir = Path(model_dir)
    with open(model_dir / INDEX_FILE) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    names = [name for name in weight_map if name.startswith(prefix)]
    if not names:
        raise KeyError(f"no checkpoint tensors under prefix {prefix!r}")

    random_init = index.get("metadata", {}).get("random_init")
    if random_init is not None:
        out = {}
        for name in names:
            dtype, shape = index["tensors"][name]
            out[name[len(prefix) :]] = _random_tensor(name, dtype, shape, seed=random_init["seed"])
        return out

    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    out = {}
    for shard, shard_names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt") as f:
            for name in shard_names:
                out[name[len(prefix) :]] = f.get_tensor(name)
    return out


def make_random_checkpoint(source_dir: Path, target_dir: Path, *, seed: int = 0) -> Path:
    """Write a checkpoint into `target_dir` with `source_dir`'s config and tokenizer files and random weights.

    Only an index is written for the weights:

        {"metadata": {"random_init": {"seed": seed}},
         "weight_map": {name: RANDOM_SHARD},
         "tensors": {name: [safetensors_dtype, shape]}}

    `load_weights` draws each tensor from a generator seeded by `seed` and the tensor's name, so the same seed gives
    the same weights in every process, and the index (hence the golden-cache checkpoint key) is the same in every
    session. The index differs from the source's, so goldens keyed on the checkpoint do not collide.

    Every tensor of two or more dims (linear and patch-embedding weights, embedding tables, the lm_head) is normal
    with std fan_in**-0.5, fan_in being the product of its trailing dims; 1D norm gains are 1 + N(0, 0.1²) and biases
    N(0, 0.02²): the usual fan-in init, so each matmul maps a unit-scale normed input to a unit-scale output.
    """
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    with open(source_dir / INDEX_FILE) as f:
        source_index = json.load(f)
    tensors = {}
    for shard in sorted(set(source_index["weight_map"].values())):
        with safe_open(source_dir / shard, framework="pt") as f:
            for name in f.keys():
                header = f.get_slice(name)
                tensors[name] = [header.get_dtype(), list(header.get_shape())]
    if set(tensors) != set(source_index["weight_map"]):
        raise ValueError(f"the shards of {source_dir} do not hold exactly the tensors its index names")
    unsupported = {name: dtype for name, (dtype, _) in tensors.items() if dtype not in _SAFETENSORS_DTYPES}
    if unsupported:
        raise ValueError(
            f"no random init for dtypes {sorted(set(unsupported.values()))}, e.g. {next(iter(unsupported))}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.name != INDEX_FILE and path.suffix != ".safetensors" and path.is_file():
            shutil.copyfile(path, target_dir / path.name)
    index = {
        "metadata": {"random_init": {"seed": seed}},
        "weight_map": {name: RANDOM_SHARD for name in sorted(tensors)},
        "tensors": {name: tensors[name] for name in sorted(tensors)},
    }
    with open(target_dir / INDEX_FILE, "w") as f:
        json.dump(index, f, indent=1)
    return target_dir


def _random_tensor(name: str, dtype: str, shape: list[int], *, seed: int) -> torch.Tensor:
    # crc32, not hash(), which Python salts per process. The CPU generator keeps only the low 32 bits of its seed,
    # so the seed and the name are hashed together into them.
    generator = torch.Generator().manual_seed(zlib.crc32(f"{seed}:{name}".encode()))
    values = torch.randn(shape, generator=generator)
    if len(shape) >= 2:
        values *= math.prod(shape[1:]) ** -0.5
    elif name.endswith(".bias"):
        values *= 0.02
    else:
        values = 1 + 0.1 * values
    return values.to(_SAFETENSORS_DTYPES[dtype])
