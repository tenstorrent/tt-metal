# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""h3/synth_weights.py: the synthetic H3's random weights and stand-in tokenizer (no device, no tt-metal imports).

Copied beside synth_gen.py by h3.sh. Self-check on CPU: python3 synth_weights.py --check <h3/config> <work dir>
builds the placeholder tree, generates every tensor of one shard per partition and prints min/max/std and finiteness.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

SEED = int(os.environ.get("H3_SYNTH_SEED", "4912"))


# ----------------------------------------------------------------------------------------------- random weights

_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
_POOL_N = 1 << 28  # 268M float32 normals, shared; each tensor takes a seeded window of it


class _Pool:
    def __init__(self, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.data = torch.randn(_POOL_N, generator=gen, dtype=torch.float32)

    def normal(self, key: str, numel: int) -> torch.Tensor:
        if numel > _POOL_N:
            reps = math.ceil(numel / _POOL_N)
            return self.data.repeat(reps)[:numel]
        start = zlib.crc32(f"{SEED}:{key}".encode()) % (_POOL_N - numel + 1)
        return self.data[start : start + numel]


# Host views (smoke job 194001): the transformer partition is 66 GB in bf16 and the pipeline reads it as
# one state dict (load_file per shard, all kept until the load ends); a node with 38-64 GB free killed pytest there.
# Its tensors larger than _VIEW_MIN elements are strided views over one small pool per dtype and std, so the dict costs
# about nothing; a reshape or cat in the model's own conversion copies one tensor or block at a time. Rows of a view
# start _row_step elements apart in the pool (shifted windows when the pool is too short for disjoint rows). The device
# weights path (synth_gen.py) samples a row's mean and std, the stock path copies each tensor as it converts it. An
# in-place write to an overlapping view fails loudly. H3_SYNTH_HOST_VIEWS=0: every tensor materialized as before.
# Smoke job 194267 (h3-full): with the transformer as views, the Qwen3-VL text encoder was next, 50.3 GB
# of bf16 read through safe_open.get_tensor into one state dict (pytest RSS 1.3 -> 55.9 GB, then SIGKILL). Its loader
# keeps the dict until the upload ends and converts each tensor out of place (unflatten, pad, cat, from_torch), so its
# large tensors are views too, through load_file and safe_open alike.
VIEW_PARTITIONS = ("transformer_ref", "transformer", "text_encoder")
_VIEW_MIN = 1 << 20
_VIEW_POOL_N = 1 << 24  # elements per (dtype, std) pool: 32 MB in bf16
# Shared scaled pools (h3-full jobs 194542 and 194664: py_hwm 25.6-27.7 GB, MemAvailable down to 0.8 GB):
# the video VAE owned 10.4 GB of float32 for the whole run (its load_state keeps the encoder and decoder state; each
# loads on first use), 9.7 GB of it at two stds. A std whose tensors in a SHARED_PARTITIONS partition hold more than
# _POOL_N elements gets one scaled pool, (pool * std).to(dtype), and its tensors are windows of it: the same window
# and the same float32 product per element, so the same bits as before. The VAE converts its state out of place;
# shared_writes() names any pool an in-place write reached (synth_gen.py prints it as a WARNING).
# H3_SYNTH_HOST_VIEWS=0 turns this off too.
SHARED_PARTITIONS = ("vae",)


def _row_step(rows: int, last: int, pool_n: int = _VIEW_POOL_N) -> int:
    """Pool offset between consecutive rows: `last` (disjoint rows) when they fit, else the largest step that does."""
    if last > pool_n or rows + last > pool_n + 1:
        raise ValueError(f"{rows} rows of {last} do not fit a {pool_n} pool")
    return max(1, min(last, (pool_n - last) // max(1, rows - 1))) if rows > 1 else last


def strided_view(pool: torch.Tensor, key: str, shape: list[int]) -> torch.Tensor:
    """A tensor of `shape` over `pool` without a copy; its start depends on `key` (SEED)."""
    last = shape[-1]
    rows = math.prod(shape[:-1])
    step = _row_step(rows, last, pool.numel())
    span = (rows - 1) * step + last
    start = zlib.crc32(f"{SEED}:{key}".encode()) % (pool.numel() - span + 1)
    stride = [1] * len(shape)
    if len(shape) > 1:
        stride[-2] = step
        for i in range(len(shape) - 3, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]
    return pool.as_strided(shape, stride, start)


def _scale_rule(name: str, shape: list[int]) -> tuple[str, float]:
    """('const', c) or ('normal', std) for one checkpoint tensor."""
    last = name.rsplit(".", 1)[-1]
    if last.startswith("zero_"):  # frozen zero buffers (audio VAE zero_k_bias); the converter asserts all zero
        return "const", 0.0
    if last in ("alpha", "beta", "weight_g"):  # snake activations, weight-norm gains
        return "const", 1.0
    if last in ("scale1", "scale2"):  # VAE LayerScale
        return "const", 0.1
    if last == "filter":  # anti-alias low-pass taps: a box filter
        return "const", 1.0 / max(1, shape[-1])
    if last.endswith("bias"):
        return "normal", 0.02
    if len(shape) <= 1:  # norm gains
        return "const", 1.0
    fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    return "normal", 1.0 / math.sqrt(max(1, fan_in))


class SyntheticWeights:
    """Tensor table from shapes.json.gz, keyed by the placeholder shard path."""

    def __init__(self, config_dir: Path, work: Path):
        spec = json.loads(gzip.open(config_dir / "shapes.json.gz").read())
        self.revision = spec.get("revision")
        self.root = work / "weights"
        self.shards: dict[str, dict[str, tuple[str, list[int]]]] = {}
        self._pool: _Pool | None = None
        self._views: dict[tuple[str, float], torch.Tensor] = {}
        self._shared: dict[tuple[str, float], torch.Tensor] = {}
        self._shared_n: dict[tuple[str, str, float], int] = {}  # (partition, dtype, std): normal elements in it
        self.host_views = os.environ.get("H3_SYNTH_HOST_VIEWS", "1") != "0"
        self.view_count = 0
        self.elements = 0
        self._build_tree(config_dir, spec["files"])

    def _build_tree(self, config_dir: Path, files: dict) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        shutil.copytree(config_dir, self.root, ignore=shutil.ignore_patterns("shapes.json.gz"))
        by_sub: dict[str, dict[str, dict]] = {}
        for rel, tensors in files.items():
            sub, shard = rel.split("/", 1)
            by_sub.setdefault(sub, {})[shard] = tensors
        for sub, shards in by_sub.items():
            directory = self.root / sub
            directory.mkdir(parents=True, exist_ok=True)
            for shard, tensors in shards.items():
                path = directory / shard
                path.write_bytes(b"")  # placeholder: a real open of it fails loudly
                self.shards[str(path.resolve())] = {k: (v[0], v[1]) for k, v in tensors.items()}
                self.elements += sum(int(np.prod(v[1])) for v in tensors.values())
                for k, (dt, sh) in tensors.items():
                    kind, value = _scale_rule(k, sh)
                    if sub in SHARED_PARTITIONS and kind == "normal":
                        self._shared_n[(sub, dt, value)] = self._shared_n.get((sub, dt, value), 0) + int(np.prod(sh))
            if len(shards) > 1:
                index = (
                    "model.safetensors.index.json"
                    if sub == "text_encoder"
                    else "diffusion_pytorch_model.safetensors.index.json"
                )
                weight_map = {k: shard for shard, tensors in shards.items() for k in tensors}
                (directory / index).write_text(json.dumps({"metadata": {"synthetic": True}, "weight_map": weight_map}))

    def tensor(
        self, name: str, dtype: str, shape: list[int], view: bool = False, part: str | None = None
    ) -> torch.Tensor:
        if self._pool is None:
            self._pool = _Pool(SEED)
        kind, value = _scale_rule(name, shape)
        numel = int(np.prod(shape)) if shape else 1
        if view and numel > _VIEW_MIN and len(shape) > 1 and shape[-1] + math.prod(shape[:-1]) <= _VIEW_POOL_N + 1:
            self.view_count += 1
            if kind == "const":
                return torch.full((1,) * len(shape), value, dtype=_DTYPES[dtype]).expand(shape)
            pool = self._views.get((dtype, value))
            if pool is None:
                pool = (self._pool.normal(f"view:{dtype}:{value}", _VIEW_POOL_N) * value).to(_DTYPES[dtype])
                self._views[(dtype, value)] = pool
            return strided_view(pool, name, shape)
        if kind == "const":
            out = torch.full(shape, value, dtype=torch.float32)
        elif part is not None and numel <= _POOL_N and self._shared_n.get((part, dtype, value), 0) > _POOL_N:
            pool = self._shared.get((dtype, value))
            if pool is None:
                pool = (self._pool.data * value).to(_DTYPES[dtype])
                self._shared[(dtype, value)] = pool
            start = zlib.crc32(f"{SEED}:{name}".encode()) % (_POOL_N - numel + 1)
            return pool[start : start + numel].view(shape)
        else:
            out = (self._pool.normal(name, numel) * value).reshape(shape)
        return out.to(_DTYPES[dtype])

    def table(self, path) -> dict[str, tuple[str, list[int]]]:
        key = str(Path(path).resolve())
        if key not in self.shards:
            raise FileNotFoundError(f"synthetic H3: no shard table for {path}")
        return self.shards[key]

    def views_for(self, path) -> bool:
        """Large tensors of this shard are host views (its partition is in VIEW_PARTITIONS, H3_SYNTH_HOST_VIEWS on)."""
        return self.host_views and Path(path).resolve().parent.name in VIEW_PARTITIONS

    def shared_for(self, path) -> str | None:
        """The partition whose large-std tensors are windows of a shared scaled pool, or None."""
        sub = Path(path).resolve().parent.name
        return sub if self.host_views and sub in SHARED_PARTITIONS else None

    def shared_writes(self) -> list[str]:
        """Shared pools an in-place write reached (torch bumps a base's version for a write through any view of it)."""
        return [f"{dt}:{v:g}" for (dt, v), pool in self._shared.items() if pool._version]

    def load_file(self, path, device="cpu"):
        view, part = self.views_for(path), self.shared_for(path)
        return {k: self.tensor(k, dt, sh, view, part) for k, (dt, sh) in self.table(path).items()}

    def safe_open(self, path, framework="pt", device="cpu"):
        weights = self
        view, part = self.views_for(path), self.shared_for(path)

        class _Handle:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def keys(self):
                return list(weights.table(path))

            def metadata(self):
                return {}

            def get_tensor(self, key):
                dt, sh = weights.table(path)[key]
                return weights.tensor(key, dt, sh, view, part)

        return _Handle()


class SyntheticTokenizer:
    """Words and punctuation to stable ids below the special range; the special tokens keep their ids."""

    def __init__(self, tokenizer_config: Path):
        added = json.loads(tokenizer_config.read_text()).get("added_tokens_decoder", {})
        self.special = {v["content"]: int(k) for k, v in added.items()}
        self.vocab = min(self.special.values()) if self.special else 151643
        self._re = re.compile(r"\w+|[^\w\s]")

    def __call__(self, text, add_special_tokens=False, **_):
        ids = [zlib.crc32(piece.encode()) % self.vocab for piece in self._re.findall(text)]
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        if isinstance(token, (list, tuple)):
            return [self.convert_tokens_to_ids(t) for t in token]
        return self.special.get(token, zlib.crc32(token.encode()) % self.vocab)


def _check(config_dir: str, work: str) -> int:
    weights = SyntheticWeights(Path(config_dir), Path(work))
    print(f"tree {weights.root}: {len(weights.shards)} shards, {weights.elements / 1e9:.2f} G elements")
    seen, bad = set(), 0
    for path, table in sorted(weights.shards.items()):
        sub = Path(path).parent.name
        if sub in seen:
            continue
        seen.add(sub)
        state = weights.load_file(path)
        for key, t in list(state.items())[:400]:
            f = t.float()
            if not torch.isfinite(f).all():
                bad += 1
                print(f"NONFINITE {sub} {key}")
        stds = [float(t.float().std()) for t in state.values() if t.numel() > 1]
        print(f"{sub}: {len(state)} tensors, std {min(stds):.4f}..{max(stds):.4f}")
    tok = SyntheticTokenizer(Path(config_dir) / "tokenizer" / "tokenizer_config.json")
    print(
        "tokens",
        len(tok("A short cinematic scene, with a fox.")["input_ids"]),
        tok.convert_tokens_to_ids("<|image_pad|>"),
    )
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        sys.exit(_check(sys.argv[2], sys.argv[3]))
    sys.exit("usage: synth_weights.py --check <h3/config> <work dir>")
