"""Convert a native Tencent `hyvideo` HunyuanVideo-1.5 DiT checkpoint state_dict
into the key layout of `diffusers.HunyuanVideo15Transformer3DModel`.

Scoped from the 720p_t2v checkpoint (54 double-blocks, 2 token-refiner blocks).
Almost everything is a 1:1 rename with identical shapes; the ONLY restructures are:
  * text token-refiner uses a FUSED qkv (6144x2048) -> split into to_q/to_k/to_v.
Note: img_in.proj is 65-channel (concat-condition), so the diffusers model must be
built with in_channels=65 (not the config's nominal 32).
"""
import re

# ---- per double-block module rename (native suffix -> diffusers suffix) ----
_BLOCK = {
    "img_attn_q": "attn.to_q",
    "img_attn_k": "attn.to_k",
    "img_attn_v": "attn.to_v",
    "img_attn_q_norm": "attn.norm_q",
    "img_attn_k_norm": "attn.norm_k",
    "img_attn_proj": "attn.to_out.0",
    "txt_attn_q": "attn.add_q_proj",
    "txt_attn_k": "attn.add_k_proj",
    "txt_attn_v": "attn.add_v_proj",
    "txt_attn_q_norm": "attn.norm_added_q",
    "txt_attn_k_norm": "attn.norm_added_k",
    "txt_attn_proj": "attn.to_add_out",
    "img_mlp.fc1": "ff.net.0.proj",
    "img_mlp.fc2": "ff.net.2",
    "txt_mlp.fc1": "ff_context.net.0.proj",
    "txt_mlp.fc2": "ff_context.net.2",
    "img_mod.linear": "norm1.linear",
    "txt_mod.linear": "norm1_context.linear",
}

# ---- top-level embedder / final-layer rename (exact native key -> diffusers key) ----
_EMB = {
    "img_in.proj.weight": "x_embedder.proj.weight",
    "img_in.proj.bias": "x_embedder.proj.bias",
    "time_in.mlp.0.weight": "time_embed.timestep_embedder.linear_1.weight",
    "time_in.mlp.0.bias": "time_embed.timestep_embedder.linear_1.bias",
    "time_in.mlp.2.weight": "time_embed.timestep_embedder.linear_2.weight",
    "time_in.mlp.2.bias": "time_embed.timestep_embedder.linear_2.bias",
    "vision_in.proj.0.weight": "image_embedder.norm_in.weight",
    "vision_in.proj.0.bias": "image_embedder.norm_in.bias",
    "vision_in.proj.1.weight": "image_embedder.linear_1.weight",
    "vision_in.proj.1.bias": "image_embedder.linear_1.bias",
    "vision_in.proj.3.weight": "image_embedder.linear_2.weight",
    "vision_in.proj.3.bias": "image_embedder.linear_2.bias",
    "vision_in.proj.4.weight": "image_embedder.norm_out.weight",
    "vision_in.proj.4.bias": "image_embedder.norm_out.bias",
    "byt5_in.fc1.weight": "context_embedder_2.linear_1.weight",
    "byt5_in.fc1.bias": "context_embedder_2.linear_1.bias",
    "byt5_in.fc2.weight": "context_embedder_2.linear_2.weight",
    "byt5_in.fc2.bias": "context_embedder_2.linear_2.bias",
    "byt5_in.fc3.weight": "context_embedder_2.linear_3.weight",
    "byt5_in.fc3.bias": "context_embedder_2.linear_3.bias",
    "byt5_in.layernorm.weight": "context_embedder_2.norm.weight",
    "byt5_in.layernorm.bias": "context_embedder_2.norm.bias",
    "cond_type_embedding.weight": "cond_type_embed.weight",
    "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
    "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
    "final_layer.linear.weight": "proj_out.weight",
    "final_layer.linear.bias": "proj_out.bias",
    "txt_in.input_embedder.weight": "context_embedder.proj_in.weight",
    "txt_in.input_embedder.bias": "context_embedder.proj_in.bias",
    "txt_in.c_embedder.linear_1.weight": "context_embedder.time_text_embed.text_embedder.linear_1.weight",
    "txt_in.c_embedder.linear_1.bias": "context_embedder.time_text_embed.text_embedder.linear_1.bias",
    "txt_in.c_embedder.linear_2.weight": "context_embedder.time_text_embed.text_embedder.linear_2.weight",
    "txt_in.c_embedder.linear_2.bias": "context_embedder.time_text_embed.text_embedder.linear_2.bias",
    "txt_in.t_embedder.mlp.0.weight": "context_embedder.time_text_embed.timestep_embedder.linear_1.weight",
    "txt_in.t_embedder.mlp.0.bias": "context_embedder.time_text_embed.timestep_embedder.linear_1.bias",
    "txt_in.t_embedder.mlp.2.weight": "context_embedder.time_text_embed.timestep_embedder.linear_2.weight",
    "txt_in.t_embedder.mlp.2.bias": "context_embedder.time_text_embed.timestep_embedder.linear_2.bias",
}

# ---- token-refiner per-block rename (native suffix -> diffusers suffix); qkv split handled separately ----
_REFINER = {
    "self_attn_proj": "attn.to_out.0",
    "mlp.fc1": "ff.net.0.proj",
    "mlp.fc2": "ff.net.2",
    "norm1": "norm1",
    "norm2": "norm2",
    "adaLN_modulation.1": "norm_out.linear",
}


def convert(native_sd):
    """native state_dict (name -> tensor) -> diffusers state_dict. Tensors may be
    real, or meta tensors for shape-only validation (chunk works on meta)."""
    out = {}
    for k, v in native_sd.items():
        m = re.match(r"double_blocks\.(\d+)\.(.+)", k)
        if m:
            i, rest = m.group(1), m.group(2)
            mod, param = rest.rsplit(".", 1)
            out[f"transformer_blocks.{i}.{_BLOCK[mod]}.{param}"] = v
            continue
        m = re.match(r"txt_in\.individual_token_refiner\.blocks\.(\d+)\.(.+)", k)
        if m:
            i, rest = m.group(1), m.group(2)
            base = f"context_embedder.token_refiner.refiner_blocks.{i}"
            if rest in ("self_attn_qkv.weight", "self_attn_qkv.bias"):
                param = rest.rsplit(".", 1)[1]
                q, kk, vv = v.chunk(3, dim=0)
                out[f"{base}.attn.to_q.{param}"] = q
                out[f"{base}.attn.to_k.{param}"] = kk
                out[f"{base}.attn.to_v.{param}"] = vv
            else:
                mod, param = rest.rsplit(".", 1)
                out[f"{base}.{_REFINER[mod]}.{param}"] = v
            continue
        if k in _EMB:
            out[_EMB[k]] = v
            continue
        raise KeyError(f"unmapped native key: {k}")
    return out


if __name__ == "__main__":
    # ---- offline validation: names + shapes, no weights needed ----
    import json
    import sys

    import torch

    SC = sys.argv[1]
    native = json.load(open(SC + "/native_keys.json"))  # name -> shape(list)

    # native "state_dict" as meta tensors (shape only, zero memory)
    nsd = {k: torch.empty(tuple(s), device="meta") for k, s in native.items()}
    conv = convert(nsd)
    conv_shapes = {k: tuple(v.shape) for k, v in conv.items()}

    # diffusers expected at the REAL config, in_channels=65 (concat-condition)
    from diffusers import HunyuanVideo15Transformer3DModel as M

    with torch.device("meta"):
        m = M(
            in_channels=65,
            out_channels=32,
            num_attention_heads=16,
            attention_head_dim=128,
            num_layers=54,
            num_refiner_layers=2,
            mlp_ratio=4,
            patch_size=1,
            patch_size_t=1,
            qk_norm="rms_norm",
            text_embed_dim=3584,
            image_embed_dim=1152,
            rope_theta=256,
            rope_axes_dim=(16, 56, 56),
            task_type="t2v",
            use_meanflow=False,
        )
    exp = {k: tuple(v.shape) for k, v in m.state_dict().items()}

    ck, ek = set(conv_shapes), set(exp)
    missing = sorted(ek - ck)  # diffusers keys the converter did NOT produce
    extra = sorted(ck - ek)  # converter produced keys diffusers doesn't want
    badshape = sorted(k for k in (ck & ek) if conv_shapes[k] != exp[k])

    print(f"native tensors in : {len(nsd)}")
    print(f"converter produced: {len(conv_shapes)}")
    print(f"diffusers expects : {len(exp)}")
    print(f"MISSING (expected, not produced): {len(missing)}")
    for k in missing[:15]:
        print("   -", k, exp[k])
    print(f"EXTRA (produced, not expected): {len(extra)}")
    for k in extra[:15]:
        print("   +", k, conv_shapes[k])
    print(f"SHAPE MISMATCH: {len(badshape)}")
    for k in badshape[:15]:
        print("   !", k, "conv", conv_shapes[k], "exp", exp[k])
    print(
        "\nRESULT:",
        "PASS - converter is a complete, shape-exact bijection"
        if not (missing or extra or badshape)
        else "FAIL - see above",
    )
