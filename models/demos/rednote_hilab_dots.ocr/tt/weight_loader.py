# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch HF weight loader for rednote-hilab/dots.ocr.

One function per block kind; each returns a nested state_dict shaped
exactly as the corresponding TTNN module ``__init__`` expects. No TTNN,
no device touches, no I/O outside :func:`load_hf_state_dict`.

The checkpoint is fp32, sharded over two safetensors files (~3B params),
so loaders pull only the keys under a prefix rather than the whole dict.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

MODEL_ID = "rednote-hilab/dots.ocr"
VISION_PATCH_EMBED_PREFIX = "vision_tower.patch_embed.patchifier"
VISION_TOWER_PREFIX = "vision_tower"
VISION_NUM_BLOCKS = 42

_CHECKPOINT_CACHE: Dict[str, torch.Tensor] = {}


def checkpoint_dir() -> Path:
    """Local HF snapshot dir (downloads only on first call, cached after)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors"]))


def load_hf_state_dict(prefix: Optional[str] = None, keys: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """Flat {full_hf_key: fp32 tensor} dict, filtered by prefix or explicit keys.

    Results are memoized per key, so per-block loaders can call this freely.
    """
    from safetensors import safe_open

    snap = checkpoint_dir()
    weight_map = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    if keys is not None:
        wanted = list(keys)
    elif prefix is not None:
        wanted = [k for k in weight_map if k.startswith(prefix)]
    else:
        wanted = list(weight_map)

    missing = [k for k in wanted if k not in _CHECKPOINT_CACHE]
    by_shard: Dict[str, list] = {}
    for k in missing:
        by_shard.setdefault(weight_map[k], []).append(k)
    for shard, shard_keys in by_shard.items():
        with safe_open(snap / shard, framework="pt") as f:
            for k in shard_keys:
                _CHECKPOINT_CACHE[k] = f.get_tensor(k).float()
    return {k: _CHECKPOINT_CACHE[k] for k in wanted}


def _strip(hf_sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) + 1 :]: v for k, v in hf_sd.items() if k.startswith(prefix + ".")}


def vision_patch_embed_weights(hf_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """State dict for TtVisionPatchEmbed: proj.weight [E,C,P,P], proj.bias [E], norm.weight [E]."""
    if hf_sd is None:
        hf_sd = load_hf_state_dict(prefix=VISION_PATCH_EMBED_PREFIX)
    sd = _strip(hf_sd, VISION_PATCH_EMBED_PREFIX)
    return {"proj.weight": sd["proj.weight"], "proj.bias": sd["proj.bias"], "norm.weight": sd["norm.weight"]}


def vision_rmsnorm_weights(
    layer_idx: int = 0, which: str = "norm1", hf_sd: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """State dict for TtVisionRMSNorm: {"weight": [dim]}.

    ``which`` selects the norm site: ``"norm1"`` / ``"norm2"`` inside
    ``vision_tower.blocks.{layer_idx}``, or ``"post_trunk_norm"`` (the
    tower-level final norm; ``layer_idx`` is ignored).
    """
    if which == "post_trunk_norm":
        key = f"{VISION_TOWER_PREFIX}.post_trunk_norm.weight"
    elif which in ("norm1", "norm2"):
        key = f"{VISION_TOWER_PREFIX}.blocks.{layer_idx}.{which}.weight"
    else:
        raise ValueError(f"unknown vision rmsnorm site {which!r}")
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=[key])
    return {"weight": hf_sd[key]}


def vision_attention_weights(
    layer_idx: int = 0, hf_sd: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """State dict for TtVisionAttention: qkv.weight [3*dim, dim] (fused, no bias), proj.weight [dim, dim]."""
    prefix = f"{VISION_TOWER_PREFIX}.blocks.{layer_idx}.attn"
    keys = [f"{prefix}.qkv.weight", f"{prefix}.proj.weight"]
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=keys)
    return {"qkv.weight": hf_sd[keys[0]], "proj.weight": hf_sd[keys[1]]}


def vision_mlp_weights(layer_idx: int = 0, hf_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """State dict for TtVisionMLP: fc1/fc3.weight [hidden, dim], fc2.weight [dim, hidden] (SwiGLU, no biases)."""
    prefix = f"{VISION_TOWER_PREFIX}.blocks.{layer_idx}.mlp"
    keys = [f"{prefix}.fc1.weight", f"{prefix}.fc2.weight", f"{prefix}.fc3.weight"]
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=keys)
    return {"fc1.weight": hf_sd[keys[0]], "fc2.weight": hf_sd[keys[1]], "fc3.weight": hf_sd[keys[2]]}


def vision_block_weights(
    layer_idx: int = 0, hf_sd: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """State dict for TtVisionBlock: norm1/norm2.weight [dim], attn.qkv/proj.weight, mlp.fc1/fc2/fc3.weight.

    Composes the per-kind loaders so layer indexing never leaks past them;
    keys match the HF ``vision_tower.blocks.{i}.*`` layout with the block
    prefix stripped (the shape TtVisionBlock.__init__ expects).
    """
    attn = vision_attention_weights(layer_idx=layer_idx, hf_sd=hf_sd)
    mlp = vision_mlp_weights(layer_idx=layer_idx, hf_sd=hf_sd)
    return {
        "norm1.weight": vision_rmsnorm_weights(layer_idx=layer_idx, which="norm1", hf_sd=hf_sd)["weight"],
        "attn.qkv.weight": attn["qkv.weight"],
        "attn.proj.weight": attn["proj.weight"],
        "norm2.weight": vision_rmsnorm_weights(layer_idx=layer_idx, which="norm2", hf_sd=hf_sd)["weight"],
        "mlp.fc1.weight": mlp["fc1.weight"],
        "mlp.fc2.weight": mlp["fc2.weight"],
        "mlp.fc3.weight": mlp["fc3.weight"],
    }


def patch_merger_weights(hf_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """State dict for TtPatchMerger: ln_q.weight/bias [dim], mlp.0.weight [dim*m^2, dim*m^2],
    mlp.0.bias [dim*m^2], mlp.2.weight [out, dim*m^2], mlp.2.bias [out].

    HF keys live under ``vision_tower.merger`` (LayerNorm-with-bias + two
    biased Linears; the nn.Sequential indices 0/2 are HF's, kept verbatim).
    """
    prefix = f"{VISION_TOWER_PREFIX}.merger"
    if hf_sd is None:
        hf_sd = load_hf_state_dict(prefix=prefix)
    sd = _strip(hf_sd, prefix)
    names = ["ln_q.weight", "ln_q.bias", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"]
    return {k: sd[k] for k in names}


def count_params(sd) -> int:
    """Tensor-leaf element count of a (possibly nested) state dict."""
    if isinstance(sd, torch.Tensor):
        return sd.numel()
    if isinstance(sd, dict):
        return sum(count_params(v) for v in sd.values())
    if isinstance(sd, (list, tuple)):
        return sum(count_params(v) for v in sd)
    return 0


if __name__ == "__main__":
    sd = vision_patch_embed_weights()
    n = count_params(sd)
    assert sd["proj.weight"].shape == (1536, 3, 14, 14), sd["proj.weight"].shape
    assert sd["proj.bias"].shape == (1536,), sd["proj.bias"].shape
    assert sd["norm.weight"].shape == (1536,), sd["norm.weight"].shape
    print(f"vision_patch_embed: {len(sd)} tensors, {n} params OK")

    for which, idx in [("norm1", 0), ("norm2", 0), ("norm1", VISION_NUM_BLOCKS - 1), ("post_trunk_norm", 0)]:
        nsd = vision_rmsnorm_weights(layer_idx=idx, which=which)
        assert nsd["weight"].shape == (1536,), (which, idx, nsd["weight"].shape)
    print(f"vision_rmsnorm: norm1/norm2/post_trunk_norm OK, {count_params(nsd)} params each")

    for idx in (0, VISION_NUM_BLOCKS - 1):
        asd = vision_attention_weights(layer_idx=idx)
        assert asd["qkv.weight"].shape == (3 * 1536, 1536), (idx, asd["qkv.weight"].shape)
        assert asd["proj.weight"].shape == (1536, 1536), (idx, asd["proj.weight"].shape)
    print(f"vision_attention: blocks.0/blocks.{VISION_NUM_BLOCKS - 1} OK, {count_params(asd)} params each")

    for idx in (0, VISION_NUM_BLOCKS - 1):
        msd = vision_mlp_weights(layer_idx=idx)
        assert msd["fc1.weight"].shape == (4224, 1536), (idx, msd["fc1.weight"].shape)
        assert msd["fc2.weight"].shape == (1536, 4224), (idx, msd["fc2.weight"].shape)
        assert msd["fc3.weight"].shape == (4224, 1536), (idx, msd["fc3.weight"].shape)
    print(f"vision_mlp: blocks.0/blocks.{VISION_NUM_BLOCKS - 1} OK, {count_params(msd)} params each")

    for idx in (0, VISION_NUM_BLOCKS - 1):
        bsd = vision_block_weights(layer_idx=idx)
        assert bsd["norm1.weight"].shape == (1536,), (idx, bsd["norm1.weight"].shape)
        assert bsd["norm2.weight"].shape == (1536,), (idx, bsd["norm2.weight"].shape)
        assert bsd["attn.qkv.weight"].shape == (3 * 1536, 1536), (idx, bsd["attn.qkv.weight"].shape)
        assert bsd["mlp.fc2.weight"].shape == (1536, 4224), (idx, bsd["mlp.fc2.weight"].shape)
        assert len(bsd) == 7, (idx, sorted(bsd))
    print(f"vision_block: blocks.0/blocks.{VISION_NUM_BLOCKS - 1} OK, {count_params(bsd)} params each")

    psd = patch_merger_weights()
    assert psd["ln_q.weight"].shape == (1536,), psd["ln_q.weight"].shape
    assert psd["ln_q.bias"].shape == (1536,), psd["ln_q.bias"].shape
    assert psd["mlp.0.weight"].shape == (6144, 6144), psd["mlp.0.weight"].shape
    assert psd["mlp.0.bias"].shape == (6144,), psd["mlp.0.bias"].shape
    assert psd["mlp.2.weight"].shape == (1536, 6144), psd["mlp.2.weight"].shape
    assert psd["mlp.2.bias"].shape == (1536,), psd["mlp.2.bias"].shape
    print(f"patch_merger: {len(psd)} tensors, {count_params(psd)} params OK")
