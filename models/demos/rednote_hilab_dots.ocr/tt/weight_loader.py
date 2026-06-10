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
TEXT_EMBED_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"
TEXT_NUM_LAYERS = 28

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


def vision_transformer_weights(
    num_layers: int = VISION_NUM_BLOCKS, hf_sd: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """Flat state dict for TtVisionTransformer (``vision_tower.`` prefix stripped).

    Keys: patch_embed.patchifier.{proj.weight,proj.bias,norm.weight},
    blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2,mlp.fc3}.weight,
    post_trunk_norm.weight, merger.{ln_q,mlp.0,mlp.2}.{weight,bias} — exactly
    the flat shape TtVisionTransformer.__init__ expects. Composes the
    per-kind loaders so layer indexing never leaks past them; ``num_layers``
    (production default 42) powers both the reduced-config harness and the
    full-config gate without code duplication.
    """
    out = {f"patch_embed.patchifier.{k}": v for k, v in vision_patch_embed_weights(hf_sd=hf_sd).items()}
    for i in range(num_layers):
        for k, v in vision_block_weights(layer_idx=i, hf_sd=hf_sd).items():
            out[f"blocks.{i}.{k}"] = v
    out["post_trunk_norm.weight"] = vision_rmsnorm_weights(which="post_trunk_norm", hf_sd=hf_sd)["weight"]
    for k, v in patch_merger_weights(hf_sd=hf_sd).items():
        out[f"merger.{k}"] = v
    return out


def embedding_weights(hf_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """State dict for TtEmbedding: {"weight": [vocab, hidden]}.

    HF key ``model.embed_tokens.weight`` (Qwen2-style text token embedding,
    untied from ``lm_head.weight`` — ``tie_word_embeddings`` is false for
    dots.ocr, so no shared-tensor helper is involved).
    """
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=[TEXT_EMBED_KEY])
    return {"weight": hf_sd[TEXT_EMBED_KEY]}


def text_rmsnorm_weights(
    layer_idx: int = 0, which: str = "input_layernorm", hf_sd: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """State dict for TtTextRMSNorm: {"weight": [hidden]}.

    ``which`` selects the norm site: ``"input_layernorm"`` /
    ``"post_attention_layernorm"`` inside ``model.layers.{layer_idx}``, or
    ``"final_norm"`` (the decoder-stack-level ``model.norm``; ``layer_idx``
    is ignored).
    """
    if which == "final_norm":
        key = "model.norm.weight"
    elif which in ("input_layernorm", "post_attention_layernorm"):
        key = f"model.layers.{layer_idx}.{which}.weight"
    else:
        raise ValueError(f"unknown text rmsnorm site {which!r}")
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=[key])
    return {"weight": hf_sd[key]}


def text_attention_weights(
    layer_idx: int = 0, hf_sd: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """State dict for TtTextAttention: q/k/v_proj.weight+bias, o_proj.weight (no bias).

    HF keys live under ``model.layers.{layer_idx}.self_attn`` (Qwen2 GQA:
    q_proj [1536, 1536] + bias, k/v_proj [256, 1536] + bias — 2 KV heads x
    head_dim 128 — o_proj [1536, 1536] bias-free). The TTNN module does its
    own fused-QKV repack and kv_replication; the loader hands over the raw
    HF per-projection tensors exactly as ``TtTextAttention.__init__`` expects.
    """
    prefix = f"model.layers.{layer_idx}.self_attn"
    names = [
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
        "o_proj.weight",
    ]
    keys = [f"{prefix}.{n}" for n in names]
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=keys)
    return {n: hf_sd[f"{prefix}.{n}"] for n in names}


def text_mlp_weights(layer_idx: int = 0, hf_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """State dict for TtTextMLP: gate/up/down_proj.weight (Qwen2 SwiGLU, no biases).

    HF keys live under ``model.layers.{layer_idx}.mlp`` (gate_proj
    [8960, 1536], up_proj [8960, 1536], down_proj [1536, 8960]). The TTNN
    module transposes and TP-shards at construction; the loader hands over
    the raw HF tensors exactly as ``TtTextMLP.__init__`` expects.
    """
    prefix = f"model.layers.{layer_idx}.mlp"
    names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
    keys = [f"{prefix}.{n}" for n in names]
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=keys)
    return {n: hf_sd[f"{prefix}.{n}"] for n in names}


def decoder_layer_weights(
    layer_idx: int = 0, hf_sd: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """State dict for TtDecoderLayer: input_layernorm.weight, self_attn.{q,k,v}_proj.weight/.bias,
    self_attn.o_proj.weight, post_attention_layernorm.weight, mlp.{gate,up,down}_proj.weight.

    Composes the per-kind loaders (text_rmsnorm/text_attention/text_mlp) so
    layer indexing never leaks past them; keys match the HF
    ``model.layers.{i}.*`` layout with the layer prefix stripped (the shape
    ``TtDecoderLayer.__init__`` expects).
    """
    out = {
        "input_layernorm.weight": text_rmsnorm_weights(layer_idx=layer_idx, which="input_layernorm", hf_sd=hf_sd)[
            "weight"
        ],
        "post_attention_layernorm.weight": text_rmsnorm_weights(
            layer_idx=layer_idx, which="post_attention_layernorm", hf_sd=hf_sd
        )["weight"],
    }
    for k, v in text_attention_weights(layer_idx=layer_idx, hf_sd=hf_sd).items():
        out[f"self_attn.{k}"] = v
    for k, v in text_mlp_weights(layer_idx=layer_idx, hf_sd=hf_sd).items():
        out[f"mlp.{k}"] = v
    return out


def lm_head_weights(hf_sd: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """State dict for TtLMHead: {"weight": [vocab, hidden]}.

    HF key ``lm_head.weight`` [151936, 1536], no bias, untied from
    ``model.embed_tokens.weight`` (``tie_word_embeddings`` is false for
    dots.ocr). The TTNN module transposes and vocab-shards at construction;
    the loader hands over the raw HF tensor exactly as ``TtLMHead.__init__``
    expects.
    """
    if hf_sd is None:
        hf_sd = load_hf_state_dict(keys=[LM_HEAD_KEY])
    return {"weight": hf_sd[LM_HEAD_KEY]}


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

    vt = vision_transformer_weights()
    assert len(vt) == 3 + 7 * VISION_NUM_BLOCKS + 1 + 6, len(vt)
    assert vt["patch_embed.patchifier.proj.weight"].shape == (1536, 3, 14, 14)
    assert vt[f"blocks.{VISION_NUM_BLOCKS - 1}.mlp.fc2.weight"].shape == (1536, 4224)
    assert vt["post_trunk_norm.weight"].shape == (1536,)
    assert vt["merger.mlp.2.weight"].shape == (1536, 6144)
    print(f"vision_transformer: {len(vt)} tensors, {count_params(vt)} params OK")

    esd = embedding_weights()
    assert esd["weight"].shape == (151936, 1536), esd["weight"].shape
    print(f"embedding: {len(esd)} tensors, {count_params(esd)} params OK")

    for which, idx in [
        ("input_layernorm", 0),
        ("post_attention_layernorm", 0),
        ("input_layernorm", TEXT_NUM_LAYERS - 1),
        ("final_norm", 0),
    ]:
        tsd = text_rmsnorm_weights(layer_idx=idx, which=which)
        assert tsd["weight"].shape == (1536,), (which, idx, tsd["weight"].shape)
    print(f"text_rmsnorm: input/post_attention/final_norm OK, {count_params(tsd)} params each")

    for idx in (0, TEXT_NUM_LAYERS - 1):
        tasd = text_attention_weights(layer_idx=idx)
        assert tasd["q_proj.weight"].shape == (1536, 1536), (idx, tasd["q_proj.weight"].shape)
        assert tasd["q_proj.bias"].shape == (1536,), (idx, tasd["q_proj.bias"].shape)
        assert tasd["k_proj.weight"].shape == (256, 1536), (idx, tasd["k_proj.weight"].shape)
        assert tasd["k_proj.bias"].shape == (256,), (idx, tasd["k_proj.bias"].shape)
        assert tasd["v_proj.weight"].shape == (256, 1536), (idx, tasd["v_proj.weight"].shape)
        assert tasd["v_proj.bias"].shape == (256,), (idx, tasd["v_proj.bias"].shape)
        assert tasd["o_proj.weight"].shape == (1536, 1536), (idx, tasd["o_proj.weight"].shape)
        assert len(tasd) == 7, (idx, sorted(tasd))
    print(f"text_attention: layers.0/layers.{TEXT_NUM_LAYERS - 1} OK, {count_params(tasd)} params each")

    for idx in (0, TEXT_NUM_LAYERS - 1):
        tmsd = text_mlp_weights(layer_idx=idx)
        assert tmsd["gate_proj.weight"].shape == (8960, 1536), (idx, tmsd["gate_proj.weight"].shape)
        assert tmsd["up_proj.weight"].shape == (8960, 1536), (idx, tmsd["up_proj.weight"].shape)
        assert tmsd["down_proj.weight"].shape == (1536, 8960), (idx, tmsd["down_proj.weight"].shape)
        assert len(tmsd) == 3, (idx, sorted(tmsd))
    print(f"text_mlp: layers.0/layers.{TEXT_NUM_LAYERS - 1} OK, {count_params(tmsd)} params each")

    for idx in (0, TEXT_NUM_LAYERS - 1):
        dlsd = decoder_layer_weights(layer_idx=idx)
        assert dlsd["input_layernorm.weight"].shape == (1536,), (idx, dlsd["input_layernorm.weight"].shape)
        assert dlsd["post_attention_layernorm.weight"].shape == (1536,), (
            idx,
            dlsd["post_attention_layernorm.weight"].shape,
        )
        assert dlsd["self_attn.q_proj.weight"].shape == (1536, 1536), (idx, dlsd["self_attn.q_proj.weight"].shape)
        assert dlsd["self_attn.k_proj.bias"].shape == (256,), (idx, dlsd["self_attn.k_proj.bias"].shape)
        assert dlsd["mlp.down_proj.weight"].shape == (1536, 8960), (idx, dlsd["mlp.down_proj.weight"].shape)
        assert len(dlsd) == 12, (idx, sorted(dlsd))
    print(f"decoder_layer: layers.0/layers.{TEXT_NUM_LAYERS - 1} OK, {count_params(dlsd)} params each")

    lsd = lm_head_weights()
    assert lsd["weight"].shape == (151936, 1536), lsd["weight"].shape
    print(f"lm_head: {len(lsd)} tensors, {count_params(lsd)} params OK")
