# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Convert MiniMax-H3-Image (original/SGLang release layout) to the diffusers layout tt_dit's H3 loaders read.

Self-contained (torch + safetensors only). The per-tensor transforms are copied from the official diffusers H3
converter (huggingface/diffusers @ abc5e9bf, scripts/convert_minimax_h3_to_diffusers.py), which tt_dit's H3 video
port was validated against:
  * fused QKV: the raw shards are per-head interleaved `[h0: q k v, h1: q k v, ...]` -> reorder to
    `[q_all; k_all; v_all]` (`reorder_interleaved_qkv`) -> split into to_q / to_k / to_v (`split_fused_qkv`);
  * gated FFN `fc1` / `ff.w1`: the release stores `[gate; value]`, diffusers' SwiGLU reads `[value; gate]` -> swap;
  * renames (`convert_transformer_key`, `_rename_video_vae_key`); `rope.inv_freq` and `decoder.mask_token` dropped.
H3-Image differences: no audio tensors (audio-free DiT), a 2D ViT decoder (16 heads x 64, 24 layers, rope ratio 0.5,
2-axis RoPE, Conv2d post_quant), and the per-step PDD output heads, stored as `pdd_heads.{weight,bias}`.

usage: python convert_h3_image.py <H3-Image release snapshot> <output dir> [all|vae|transformer|transformer-layers=N]
(every mode also copies pdd_config.json, model_index.json, processor/, tokenizer/ and examples/)

The release ships no text encoder: H3-Image conditions on the same Qwen3-VL-32B as MiniMax-H3, so link it in:
  ln -s <MiniMax-H3 diffusers snapshot>/text_encoder <output dir>/text_encoder
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP32_SOURCE_PREFIXES = ("video_patch_proj.", "time_embedder.", "final_layer.video_out.")


def reorder_interleaved_qkv(weight: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    assert weight.shape[0] == heads * 3 * head_dim, (weight.shape, heads, head_dim)
    grouped = weight.reshape(heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat([t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim=0)


def split_qkv(weight: torch.Tensor, heads: int, head_dim: int):
    q, k, v = reorder_interleaved_qkv(weight, heads, head_dim).split(heads * head_dim, dim=0)
    return q.contiguous(), k.contiguous(), v.contiguous()


def swap_gate_value(weight: torch.Tensor) -> torch.Tensor:
    gate, value = weight.chunk(2, dim=0)
    return torch.cat([value, gate], dim=0).contiguous()


def transformer_key(key: str, tensor: torch.Tensor, heads: int, head_dim: int) -> list[tuple[str, torch.Tensor]]:
    if key == "rope.inv_freq":
        return []
    if key.startswith(("audio_patch_proj.", "final_layer.audio_out.")):
        raise ValueError(f"unexpected audio tensor in an audio-free checkpoint: {key}")
    t = key
    if t.startswith("token_refiner.blocks."):
        t = t.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif t.startswith("blocks."):
        t = t.replace("blocks.", "transformer_blocks.", 1)
    for a, b in (
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        ("video_patch_proj.", "proj_in."),
        ("condition_proj.", "context_embedder."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.video_out.", "proj_out."),
        (".attn.q_norm.", ".attn.norm_q."),
        (".attn.k_norm.", ".attn.norm_k."),
        (".attn.out_proj.", ".attn.to_out.0."),
    ):
        t = t.replace(a, b)
    if t.endswith(".attn.qkv_proj.weight"):
        q, k, v = split_qkv(tensor, heads, head_dim)
        p = t.removesuffix("qkv_proj.weight")
        return [(p + "to_q.weight", q), (p + "to_k.weight", k), (p + "to_v.weight", v)]
    if t.endswith(".mlp.fc1.weight"):
        return [(t.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight"), swap_gate_value(tensor))]
    return [(t.replace(".mlp.fc2.", ".ff.net.2."), tensor)]


def vae_key(key: str, tensor: torch.Tensor, heads: int, head_dim: int) -> list[tuple[str, torch.Tensor]]:
    if key == "decoder.mask_token":
        return []
    if ".attn.to_qkv." in key:
        q, k, v = split_qkv(tensor, heads, head_dim)
        prefix, suffix = key.split(".attn.to_qkv.")
        return [
            (f"{prefix}.attn.to_q.{suffix}", q),
            (f"{prefix}.attn.to_k.{suffix}", k),
            (f"{prefix}.attn.to_v.{suffix}", v),
        ]
    t = key
    if t.startswith("encoder.down."):
        level, rest = t.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        t = f"encoder.down_blocks.{level}.{rest}"
    t = t.replace("decoder.x_embedder.", "decoder.proj_in.").replace(".attn.to_out.", ".attn.to_out.0.")
    t = t.replace(".ff.w1.", ".ff.net.0.proj.").replace(".ff.w2.", ".ff.net.2.")
    if ".ff.w1." in key:
        return [(t, swap_gate_value(tensor))]
    return [(t, tensor)]


def convert_vae(src: str, out: str) -> None:
    raw = json.load(open(os.path.join(src, "vae", "config.json")))
    vit = raw["vit_decoder_kwargs"]
    assert raw["vit_decoder_2d"] is True and vit["qk_norm_affine"] is False and vit["ffn_use_gated"], raw
    state = {}
    with safe_open(os.path.join(src, "vae", "diffusion_pytorch_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            for target, tensor in vae_key(key, f.get_tensor(key), vit["heads"], vit["dim_head"]):
                state[target] = tensor.contiguous()
    vdir = os.path.join(out, "vae")
    os.makedirs(vdir, exist_ok=True)
    save_file(state, os.path.join(vdir, "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    dim = vit["heads"] * vit["dim_head"]
    cfg = {
        "_class_name": "MiniMaxH3ImageVAE",
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": raw["latent_channels"],
        "latents_mean": raw["latents_mean"],
        "latents_std": raw["latents_std"],
        "decoder_num_layers": vit["num_layers"],
        "decoder_num_attention_heads": vit["heads"],
        "decoder_attention_head_dim": vit["dim_head"],
        "decoder_num_register_tokens": state["decoder.register_tokens"].shape[1],
        "decoder_ffn_mult": state["decoder.transformer_blocks.0.ff.net.2.weight"].shape[1] // dim,
        "decoder_rope_theta": vit["rope_theta"],
        "decoder_rope_dim_ratio": vit["rope_dim_ratio"],
        "decoder_rope_num_axes": 2,
        "decoder_patch_size_t": 1,
        "decoder_norm_eps": 1e-5,
        "tile_size": raw["vae_tile_size"],
        "tile_overlap_min": raw["vae_tile_overlap_min"],
    }
    json.dump(cfg, open(os.path.join(vdir, "config.json"), "w"), indent=2)
    print(f"vae: {len(state)} tensors -> {vdir}", flush=True)


def convert_transformer(src: str, out: str, max_shard_bytes: int = 5 << 30, max_layers: int | None = None) -> None:
    """`max_layers`: keep only blocks [0, max_layers) (plus every non-block tensor) -- a small checkpoint for
    few-layer PCC gates; config.json then says `num_layers = max_layers`."""
    raw = json.load(open(os.path.join(src, "transformer", "config.json")))
    assert raw["audio_enabled"] is False, raw
    heads, head_dim = raw["num_attention_heads"], raw["attention_head_dim"]
    tdir = os.path.join(out, "transformer")
    os.makedirs(tdir, exist_ok=True)
    index = json.load(open(os.path.join(src, "transformer", "model.safetensors.index.json")))["weight_map"]
    weight_map, buffer, size, total, shards = {}, {}, 0, 0, []

    def flush():
        nonlocal buffer, size
        if buffer:
            name = f"diffusion_pytorch_model-{len(shards) + 1:05d}.safetensors"
            save_file(buffer, os.path.join(tdir, name), metadata={"format": "pt"})
            weight_map.update({k: name for k in buffer})
            shards.append(name)
            buffer, size = {}, 0

    rope_theta = 10000.0

    def wanted(key: str) -> bool:
        if max_layers is None or not key.startswith("blocks."):
            return True
        return int(key.split(".")[1]) < max_layers

    for shard in sorted({v for k, v in index.items() if wanted(k)}):
        with safe_open(os.path.join(src, "transformer", shard), framework="pt") as f:
            for key in f.keys():
                if not wanted(key):
                    continue
                tensor = f.get_tensor(key)
                if key == "rope.inv_freq":
                    # Dropped like the official converter does, but only after proving it is the closed form the
                    # port recomputes from `rope_theta` -- bitwise.
                    n = raw["rope_inv_freq_len"]
                    expected = 1.0 / rope_theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float32) / (2 * n))
                    if not torch.equal(expected.to(tensor.dtype), tensor):
                        raise ValueError("rope.inv_freq is not 1/10000^(2i/32); the converter's rope_theta is wrong")
                expected = torch.float32 if key.startswith(FP32_SOURCE_PREFIXES) else torch.bfloat16
                if key != "rope.inv_freq" and tensor.dtype != expected:
                    raise ValueError(f"{key}: expected {expected}, got {tensor.dtype}")
                for target, t in transformer_key(key, tensor, heads, head_dim):
                    buffer[target] = t.contiguous()
                    size += t.numel() * t.element_size()
                    total += t.numel() * t.element_size()
                if size >= max_shard_bytes:
                    flush()
        print(f"  {shard} done", flush=True)
    flush()
    n = len(shards)
    final = {}
    for i, name in enumerate(shards):
        new = f"diffusion_pytorch_model-{i + 1:05d}-of-{n:05d}.safetensors"
        os.rename(os.path.join(tdir, name), os.path.join(tdir, new))
        final[name] = new
    json.dump(
        {"metadata": {"total_size": total}, "weight_map": {k: final[v] for k, v in weight_map.items()}},
        open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json"), "w"),
        indent=2,
        sort_keys=True,
    )
    cfg = {
        "_class_name": "MiniMaxH3Transformer3DModel",
        "num_attention_heads": heads,
        "attention_head_dim": head_dim,
        "hidden_size": raw["hidden_size"],
        "num_layers": raw["num_layers"] if max_layers is None else max_layers,
        "num_refiner_layers": raw["token_refiner_num_layers"],
        "ffn_dim": raw["ffn_hidden_size"],
        "in_channels": raw["latents_dim"],
        "audio_in_channels": raw["audio_latents_dim"],
        "patch_size": raw["patch_size"],
        "text_dim": raw["text_dim"],
        "freq_dim": raw["timestep_input_dim"],
        "time_embed_hidden_dim": raw["time_embed_hidden_size"],
        "time_embed_dim": raw["time_embed_dim"],
        "rope_freq_dim": raw["rope_inv_freq_len"],
        "rope_theta": rope_theta,
        "norm_eps": raw["norm_eps"],
        "qk_norm_eps": raw["qk_norm_eps"],
        "final_norm_eps": raw["final_norm_eps"],
        "audio_enabled": False,
    }
    json.dump(cfg, open(os.path.join(tdir, "config.json"), "w"), indent=2)
    # PDD heads, in the same snapshot so the loader finds them next to the transformer.
    with safe_open(os.path.join(src, "pdd_fused_heads.safetensors"), framework="pt") as f:
        heads_state = {
            "pdd_heads.weight": f.get_tensor("video_out.weight"),
            "pdd_heads.bias": f.get_tensor("video_out.bias"),
        }
    save_file(heads_state, os.path.join(tdir, "pdd_heads.safetensors"), metadata={"format": "pt"})
    print(f"transformer: {len(weight_map)} tensors in {n} shards, {total / 2**30:.2f} GiB -> {tdir}", flush=True)


def copy_side_files(src: str, out: str) -> None:
    for name in ("pdd_config.json", "model_index.json"):
        shutil.copy2(os.path.join(src, name), os.path.join(out, name))
    for name in ("processor", "tokenizer", "examples"):
        shutil.copytree(os.path.join(src, name), os.path.join(out, name), dirs_exist_ok=True)


def is_release_snapshot(path: str) -> bool:
    """A MiniMax-H3-Image release snapshot as published (fused PDD heads beside an unconverted transformer/)."""
    return os.path.isfile(os.path.join(path, "pdd_fused_heads.safetensors")) and not is_converted(path)


def is_converted(path: str) -> bool:
    """A snapshot this script wrote (the per-step heads live under transformer/ only after conversion)."""
    return os.path.isfile(os.path.join(path, "transformer", "pdd_heads.safetensors"))


MODES = ("all", "vae", "transformer", "transformer-layers=N")


def _max_layers(what: str) -> int | None:
    """`N` for `transformer-layers=N` (N >= 1), None for every other mode; an unknown mode raises."""
    if what.startswith("transformer-layers="):
        value = what.split("=", 1)[1]
        if value.isdigit() and int(value) >= 1:
            return int(value)
    elif what in MODES:
        return None
    raise ValueError(f"unknown conversion mode {what!r}; expected one of {', '.join(MODES)}")


def convert(src: str, out: str, what: str = "all") -> None:
    max_layers = _max_layers(what)  # before any work, so a typo does not convert nothing and print ok
    os.makedirs(out, exist_ok=True)
    if what in ("vae", "all"):
        convert_vae(src, out)
    if what in ("transformer", "all"):
        convert_transformer(src, out)
    if max_layers is not None:  # e.g. transformer-layers=2
        convert_transformer(src, out, max_layers=max_layers)
    # Every mode: the pipeline reads pdd_config.json (the sigma grid), the processor and the tokenizer whichever
    # component was converted, so a transformer-only snapshot is otherwise unloadable.
    copy_side_files(src, out)
    print("ok", out)


if __name__ == "__main__":
    if not 3 <= len(sys.argv) <= 4:
        sys.exit(f"usage: python convert_h3_image.py <release snapshot> <output dir> [{'|'.join(MODES)}]")
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    try:
        _max_layers(mode)
    except ValueError as err:
        sys.exit(str(err))
    convert(sys.argv[1], sys.argv[2], mode)
