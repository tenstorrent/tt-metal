# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Convert the StreamRender-H3 backbone to a diffusers MiniMax-H3 snapshot the ref2va pipeline loads.

The backbone (`Efficient-Large-Model/StreamRender-H3`, `backbone/`) is a Sana `export_state` dump: `dit.*`
keys, LoRA folded in, and the fused `qkv_proj` already in runtime layout `[q_all, k_all, v_all]` (not the
per-head `[q0, k0, v0, ...]` grouping of MiniMax's released checkpoint). So every key is a rename except
two: `qkv_proj` splits into contiguous thirds, and `fc1` (`[gate, up]`, Sana's `silu(gate) * up`) swaps its
halves, because diffusers' `ff.net.0.proj` stores them the other way round.

Neither permutation mistake changes a shape, so `--verify` compares every converted tensor with the base
MiniMax-H3 ref2va transformer it was fine-tuned from: a correct mapping stays highly correlated, a row
permutation drops to ~0.

The output directory is a full snapshot: `transformer_ref/` is the converted backbone and every other
partition is a symlink into the base snapshot (the Qwen text encoder is byte-identical). The weight cache is
keyed on the subfolder name, not the path, so runs on this snapshot need their own `TT_DIT_CACHE_DIR`.

    python -m models.tt_dit.pipelines.minimax_h3.streamrender.convert_backbone \
        --backbone /data/.../StreamRender-H3/backbone --base /data/.../MiniMax-H3 --out /data/.../snapshot --verify
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_RENAMES = (
    (r"^video_patch_proj\.", "proj_in."),
    (r"^audio_patch_proj\.", "audio_proj_in."),
    (r"^condition_proj\.", "context_embedder."),
    (r"^time_embedder\.proj_in\.", "time_embedder.linear_1."),
    (r"^time_embedder\.proj_out\.", "time_embedder.linear_2."),
    (r"^final_layer\.adaln_proj\.", "norm_out."),
    (r"^final_layer\.norm\.", "norm_out.norm."),
    (r"^final_layer\.video_out\.", "proj_out."),
    (r"^final_layer\.audio_out\.", "audio_proj_out."),
    (r"^blocks\.", "transformer_blocks."),
    (r"^token_refiner\.blocks\.", "token_refiner.refiner_blocks."),
    (r"\.attn\.q_norm\.", ".attn.norm_q."),
    (r"\.attn\.k_norm\.", ".attn.norm_k."),
    (r"\.attn\.out_proj\.", ".attn.to_out.0."),
    (r"\.mlp\.fc1\.", ".ff.net.0.proj."),
    (r"\.mlp\.fc2\.", ".ff.net.2."),
)

# A fine-tuned tensor stays far above this against its base; a permuted one lands near zero.
_MIN_COSINE = 0.5


def convert_key(key: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Diffusers tensors for one Sana tensor; empty for buffers diffusers recomputes."""
    if not key.startswith("dit."):
        raise KeyError(f"unexpected backbone key {key!r}: expected a `dit.*` export")
    key = key.removeprefix("dit.")
    if key == "rope.inv_freq":
        return {}
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    if key.endswith(".attn.qkv_proj.weight"):
        q, k, v = value.chunk(3, dim=0)
        stem = key.removesuffix("qkv_proj.weight")
        return {f"{stem}to_q.weight": q, f"{stem}to_k.weight": k, f"{stem}to_v.weight": v}
    if key.endswith(".ff.net.0.proj.weight"):
        gate, up = value.chunk(2, dim=0)
        return {key: torch.cat((up, gate), dim=0)}
    return {key: value}


def _base_reader(base_transformer: Path):
    index = json.loads((base_transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, object] = {}

    def read(key: str) -> torch.Tensor:
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(base_transformer / shard, framework="pt")
        return handles[shard].get_tensor(key)

    return index, read


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def convert(backbone: Path, base: Path, out: Path, *, verify: bool) -> None:
    sana_index = json.loads((backbone / "model.safetensors.index.json").read_text())["weight_map"]
    base_transformer = base / "transformer_ref"
    base_index, read_base = _base_reader(base_transformer)

    target = out / "transformer_ref"
    target.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    worst: tuple[float, str] = (1.0, "")
    shards = sorted(set(sana_index.values()))
    for number, shard in enumerate(shards, start=1):
        name = f"diffusion_pytorch_model-{number:05d}-of-{len(shards):05d}.safetensors"
        converted: dict[str, torch.Tensor] = {}
        with safe_open(backbone / shard, framework="pt") as handle:
            for key in handle.keys():
                for new_key, value in convert_key(key, handle.get_tensor(key)).items():
                    if new_key not in base_index:
                        raise KeyError(f"{key} -> {new_key}: not a MiniMax-H3 transformer key")
                    if verify:
                        reference = read_base(new_key)
                        if reference.shape != value.shape:
                            raise ValueError(f"{new_key}: shape {tuple(value.shape)} != base {tuple(reference.shape)}")
                        similarity = _cosine(value, reference)
                        if similarity < _MIN_COSINE:
                            print(f"LOW {new_key}: {similarity:.4f}", flush=True)
                        worst = min(worst, (similarity, new_key))
                    converted[new_key] = value.contiguous()
        save_file(converted, target / name, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(converted, name))
        print(f"{name}: {len(converted)} tensors", flush=True)

    missing = sorted(set(base_index) - set(weight_map))
    if missing:
        raise KeyError(f"{len(missing)} base keys have no StreamRender tensor, e.g. {missing[:5]}")
    (target / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2)
    )
    (target / "config.json").write_text((base_transformer / "config.json").read_text())
    if verify:
        print(f"lowest cosine vs base: {worst[0]:.4f} ({worst[1]})", flush=True)
        if worst[0] < _MIN_COSINE:
            raise ValueError(f"{worst[1]} correlates {worst[0]:.4f} with its base tensor: mapping is wrong")

    for entry in base.iterdir():
        link = out / entry.name
        if entry.name not in ("transformer_ref", "transformer") and not link.exists():
            link.symlink_to(entry.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", type=Path, required=True, help="StreamRender-H3 backbone/ directory")
    parser.add_argument("--base", type=Path, required=True, help="MiniMax-H3 diffusers snapshot")
    parser.add_argument("--out", type=Path, required=True, help="output snapshot directory")
    parser.add_argument("--verify", action="store_true", help="check every tensor against the base ref2va weights")
    args = parser.parse_args()
    convert(args.backbone, args.base, args.out, verify=args.verify)


if __name__ == "__main__":
    main()
