# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import (
    HfModelWrapper,
    HfAttentionWrapper,
    HfDecoderWrapper,
    convert_meta_to_hf,
)

from models.tt_transformers.tt.load_checkpoints import map_meta_to_hf_keys


def convert_vision_meta_to_hf(state_dict, head_dim):
    # state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_vision_meta_to_hf_keys(state_dict)
    return state_dict


def reference_lm_head(model_args):
    model = model_args.reference_transformer(wrap=False)
    layer = model.lm_head
    layer._load_state_dict = layer.load_state_dict
    layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, model_args.head_dim))
    return layer


def reference_transformer(model_args, wrap=True, load_checkpoint=False):
    from transformers import AutoConfig, AutoModelForCausalLM

    # HF is much faster at loading from a checkpoint than generating from config
    # so use that by preference unless we don't have a checkpoint
    if model_args.dummy_weights and not load_checkpoint:
        config = AutoConfig.from_pretrained(model_args.LOCAL_HF_PARAMS[model_args.model_name])
        config.num_layers = model_args.n_layers
        config.num_hidden_layers = model_args.n_layers
        model = AutoModelForCausalLM.from_config(config)
    else:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(model_args.CKPT_DIR, device_map="auto")

        # HACK: Assume that we want the language model layers only
        if hasattr(model, "language_model"):
            model.model = model.language_model
            # We keep language_model because transformers don't let us change or delete it
        model.model.layers = model.model.layers[: model_args.n_layers]
    if wrap:
        wrapper = HfModelWrapper(model, model_args.head_dim)
        return wrapper
    else:
        return model


def reference_rms_norm(model_args):
    model = reference_transformer(model_args, wrap=False)
    layer = model.model.norm
    layer._load_state_dict = layer.load_state_dict
    layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, model_args.head_dim))
    return layer


def reference_mlp(model_args):
    model = model_args.reference_transformer(wrap=False)
    layer = model.model.layers[0].mlp
    layer._load_state_dict = layer.load_state_dict
    layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, model_args.head_dim))
    return layer


def reference_embedding(model_args, reference_model=None):
    if reference_model is None:
        model = model_args.reference_transformer(wrap=False)
        layer = model.model.embed_tokens
    else:
        layer = reference_model.model.model.embed_tokens

    layer._load_state_dict = layer.load_state_dict
    layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, model_args.head_dim))
    return layer


def reference_decoder(model_args):
    model = model_args.reference_transformer(wrap=False)
    layer = model.model.layers[0]
    model_name_env = os.getenv("HF_MODEL")
    if hasattr(model.model, "rotary_emb_local"):
        rotary_emb_local = model.model.rotary_emb_local
    else:
        rotary_emb_local = None
    wrapper = HfDecoderWrapper(layer, model_args.head_dim, model.model.rotary_emb, rotary_emb_local)
    return wrapper


def reference_attention(model_args):
    model = model_args.reference_transformer(wrap=False)
    layer = model.model.layers[0].self_attn
    use_position_embeddings = model_args.from_hf_url or layer.__class__.__name__ in (
        "Qwen3Attention",
        "MistralAttention",
        "Gemma3Attention",
    )
    wrapper = HfAttentionWrapper(
        layer, model_args.head_dim, model.model.rotary_emb if use_position_embeddings else None
    )
    return wrapper


def reference_vision_transformer(model_args, wrap=True, load_checkpoint=False):
    from transformers import AutoConfig, AutoModelForCausalLM

    if model_args.dummy_weights and not load_checkpoint:
        config = AutoConfig.from_pretrained(model_args.LOCAL_HF_PARAMS[model_args.model_name])
        config.num_layers = model_args.n_layers
        config.num_hidden_layers = model_args.n_layers
        model = AutoModelForCausalLM.from_config(config)
    else:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(model_args.CKPT_DIR)
    if wrap:
        wrapper = HfModelWrapper(model, model_args.head_dim)
        return wrapper
    else:
        return model


def reference_vision_multi_modal(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.multi_modal_projector
    return layer


def reference_vision_rms_norm(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.multi_modal_projector.mm_soft_emb_norm
    return layer


def reference_gemma_model(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model
    layer._load_state_dict = layer.load_state_dict
    layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, model_args.head_dim))
    return layer


def reference_vision_model(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model
    return layer


def reference_vision_mlp(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model.encoder.layers[0].mlp
    return layer


def reference_siglip_patch_embed(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model.embeddings.patch_embedding
    return layer


def reference_vision_pos_embedding(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model.embeddings.position_embedding
    return layer


def reference_vision_embedding(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model.embeddings
    return layer


def reference_vision_layernorm(model_args, layer_name="layer_norm1"):
    model = reference_vision_transformer(model_args, wrap=False)
    if layer_name == "layer_norm1":
        layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm1
    elif layer_name == "layer_norm2":
        layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm2
    else:
        layer = model.vision_tower.vision_model.post_layernorm
    return layer


def reference_vision_attention(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model.encoder.layers[0].self_attn  # Common naming
    return layer


def reference_vision_encoder_block(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model.encoder.layers[0]
    return layer


def reference_vision_encoder(model_args):
    model = reference_vision_transformer(model_args, wrap=False)
    layer = model.vision_tower.vision_model.encoder
    return layer


def map_vision_meta_to_hf_keys(loaded_weights):
    language_weights = {
        key[len("language_model.") :]: tensor
        for key, tensor in loaded_weights.items()
        if key.startswith("language_model.")
    }
    mapped_language_weights = map_meta_to_hf_keys(language_weights, language_prefix="language_model.")
    other_weights = {key: tensor for key, tensor in loaded_weights.items() if not key.startswith("language_model.")}
    hf_state_dict = {**mapped_language_weights}
    loaded_weights = {**other_weights}
    meta_to_hf_mappings = {
        # vision MLP
        "c_fc.weight": "fc1.weight",
        "c_fc.bias": "fc1.bias",
        "c_proj.weight": "fc2.weight",
        "c_proj.bias": "fc2.bias",
        # vision attention
        # "wq.weight": "q_proj.weight",
        # "wk.weight": "k_proj.weight",
        # "wv.weight": "v_proj.weight",
        # "wo.weight": "out_proj.weight",
        # "wq.bias": "q_proj.bias",
        # "wk.bias": "k_proj.bias",
        # "wv.bias": "v_proj.bias",
        # "wo.bias": "out_proj.bias",
        # vision encoder block
        "attn.wq.weight": "self_attn.q_proj.weight",
        "attn.wk.weight": "self_attn.k_proj.weight",
        "attn.wv.weight": "self_attn.v_proj.weight",
        "attn.wo.weight": "self_attn.out_proj.weight",
        "attn.wq.bias": "self_attn.q_proj.bias",
        "attn.wk.bias": "self_attn.k_proj.bias",
        "attn.wv.bias": "self_attn.v_proj.bias",
        "attn.wo.bias": "self_attn.out_proj.bias",
        "ln_1.weight": "layer_norm1.weight",
        "ln_1.bias": "layer_norm1.bias",
        "ln_2.weight": "layer_norm2.weight",
        "ln_2.bias": "layer_norm2.bias",
        "mlp.c_fc.weight": "mlp.fc1.weight",
        "mlp.c_fc.bias": "mlp.fc1.bias",
        "mlp.c_proj.weight": "mlp.fc2.weight",
        "mlp.c_proj.bias": "mlp.fc2.bias",
        # vision encoder
        "layers.{layer}.attn.wq.weight": "layers.{layer}.self_attn.q_proj.weight",
        "layers.{layer}.attn.wk.weight": "layers.{layer}.self_attn.k_proj.weight",
        "layers.{layer}.attn.wv.weight": "layers.{layer}.self_attn.v_proj.weight",
        "layers.{layer}.attn.wo.weight": "layers.{layer}.self_attn.out_proj.weight",
        "layers.{layer}.attn.wq.bias": "layers.{layer}.self_attn.q_proj.bias",
        "layers.{layer}.attn.wk.bias": "layers.{layer}.self_attn.k_proj.bias",
        "layers.{layer}.attn.wv.bias": "layers.{layer}.self_attn.v_proj.bias",
        "layers.{layer}.attn.wo.bias": "layers.{layer}.self_attn.out_proj.bias",
        "layers.{layer}.ln_1.weight": "layers.{layer}.layer_norm1.weight",
        "layers.{layer}.ln_1.bias": "layers.{layer}.layer_norm1.bias",
        "layers.{layer}.ln_2.weight": "layers.{layer}.layer_norm2.weight",
        "layers.{layer}.ln_2.bias": "layers.{layer}.layer_norm2.bias",
        "layers.{layer}.mlp.c_fc.weight": "layers.{layer}.mlp.fc1.weight",
        "layers.{layer}.mlp.c_fc.bias": "layers.{layer}.mlp.fc1.bias",
        "layers.{layer}.mlp.c_proj.weight": "layers.{layer}.mlp.fc2.weight",
        "layers.{layer}.mlp.c_proj.bias": "layers.{layer}.mlp.fc2.bias",
        # vision transformer
        "encoder.layers.{layer}.attn.wq.weight": "encoder.layers.{layer}.self_attn.q_proj.weight",
        "encoder.layers.{layer}.attn.wk.weight": "encoder.layers.{layer}.self_attn.k_proj.weight",
        "encoder.layers.{layer}.attn.wv.weight": "encoder.layers.{layer}.self_attn.v_proj.weight",
        "encoder.layers.{layer}.attn.wo.weight": "encoder.layers.{layer}.self_attn.out_proj.weight",
        "encoder.layers.{layer}.attn.wq.bias": "encoder.layers.{layer}.self_attn.q_proj.bias",
        "encoder.layers.{layer}.attn.wk.bias": "encoder.layers.{layer}.self_attn.k_proj.bias",
        "encoder.layers.{layer}.attn.wv.bias": "encoder.layers.{layer}.self_attn.v_proj.bias",
        "encoder.layers.{layer}.attn.wo.bias": "encoder.layers.{layer}.self_attn.out_proj.bias",
        "ln_post.weight": "post_layernorm.weight",
        "ln_post.bias": "post_layernorm.bias",
        # Top level
        "_linear.weight": "weight",  # patch_embedding
        "_linear.bias": "bias",  # patch_embedding
        "positional_embedding": "weight",  # pos_emb
        "visual.embeddings.patch_embedding._linear.weight": "visual.embeddings.patch_embedding.weight",
        "visual.embeddings.patch_embedding._linear.bias": "visual.embeddings.patch_embedding._linear.bias",
        "visual.embeddings.position_embedding.positional_embedding": "visual.embeddings.position_embedding.weight",
        "visual.encoder.layers.{layer}.attn.wq.weight": "visual.encoder.layers.{layer}.self_attn.q_proj.weight",
        "visual.encoder.layers.{layer}.attn.wk.weight": "visual.encoder.layers.{layer}.self_attn.k_proj.weight",
        "visual.encoder.layers.{layer}.attn.wv.weight": "visual.encoder.layers.{layer}.self_attn.v_proj.weight",
        "visual.encoder.layers.{layer}.attn.wo.weight": "visual.encoder.layers.{layer}.self_attn.out_proj.weight",
        "visual.encoder.layers.{layer}.attn.wq.bias": "visual.encoder.layers.{layer}.self_attn.q_proj.bias",
        "visual.encoder.layers.{layer}.attn.wk.bias": "visual.encoder.layers.{layer}.self_attn.k_proj.bias",
        "visual.encoder.layers.{layer}.attn.wv.bias": "visual.encoder.layers.{layer}.self_attn.v_proj.bias",
        "visual.encoder.layers.{layer}.attn.wo.bias": "visual.encoder.layers.{layer}.self_attn.out_proj.bias",
        "visual.encoder.layers.{layer}.ln_1.weight": "visual.encoder.layers.{layer}.layer_norm1.weight",
        "visual.encoder.layers.{layer}.ln_1.bias": "visual.encoder.layers.{layer}.layer_norm1.bias",
        "visual.encoder.layers.{layer}.ln_2.weight": "visual.encoder.layers.{layer}.layer_norm2.weight",
        "visual.encoder.layers.{layer}.ln_2.bias": "visual.encoder.layers.{layer}.layer_norm2.bias",
        "visual.encoder.layers.{layer}.mlp.c_fc.weight": "visual.encoder.layers.{layer}.mlp.fc1.weight",
        "visual.encoder.layers.{layer}.mlp.c_fc.bias": "visual.encoder.layers.{layer}.mlp.fc1.bias",
        "visual.encoder.layers.{layer}.mlp.c_proj.weight": "visual.encoder.layers.{layer}.mlp.fc2.weight",
        "visual.encoder.layers.{layer}.mlp.c_proj.bias": "visual.encoder.layers.{layer}.mlp.fc2.bias",
        "visual.ln_post.weight": "visual.post_layernorm.weight",
        "visual.ln_post.bias": "visual.post_layernorm.bias",
    }

    for key, tensor in loaded_weights.items():
        # Handle full model paths with layer numbers
        if "model.vision_tower.vision_model.encoder.layers." in key:
            parts = key.split(".")
            layer_num = parts[5]
            remainder = ".".join(parts[6:])
            if remainder in meta_to_hf_mappings:
                new_key = f"model.vision_tower.vision_model.encoder.layers.{layer_num}.{meta_to_hf_mappings[remainder]}"
                hf_state_dict[new_key] = tensor
            continue

        # Handle full vision encoder paths with layer numbers
        if "layers." in key:
            parts = key.split(".")
            layer_num = parts[1]  # e.g. "0" in "model.layers.0.input_layernorm.weight"
            template_key = "layers.{layer}." + ".".join(parts[2:])
            if template_key in meta_to_hf_mappings:
                hf_state_dict[meta_to_hf_mappings[template_key].format(layer=layer_num)] = tensor
                continue

        # Try exact matches first
        if key in meta_to_hf_mappings:
            hf_state_dict[meta_to_hf_mappings[key]] = tensor
            continue

        # For submodule state dicts, try matching the end of the key
        matched = False
        for meta_pattern, hf_pattern in meta_to_hf_mappings.items():
            if key.endswith("." + meta_pattern):
                # Replace only the matching part at the end
                prefix = key[: -len(meta_pattern)]
                new_key = prefix + hf_pattern
                hf_state_dict[new_key] = tensor
                matched = True
                break

        # If no mapping found, keep the original key
        if not matched:
            hf_state_dict[key] = tensor

    return hf_state_dict
