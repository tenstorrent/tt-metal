# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import (
    HfModelWrapper,
    HfAttentionWrapper,
    HfDecoderWrapper,
    convert_meta_to_hf,
)


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
