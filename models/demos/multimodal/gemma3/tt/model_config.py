# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

import ttnn
from models.demos.multimodal.gemma3.tt.load_checkpoints import convert_vision_hf_to_meta, convert_vision_meta_to_hf
from models.tt_transformers.tt.common import calculate_prefill_warmup_seq_lens, cap_seq_lens_to_max_prefill_chunk_size
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, convert_meta_to_hf, standardize_hf_keys
from models.tt_transformers.tt.model_config import HfAttentionWrapper, HfDecoderWrapper, HfModelWrapper
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs

# file names for performance and accuracy mode override files
PERFORMANCE_DECODER_CONFIG_FILENAME = "performance_decoder_config.json"
ACCURACY_DECODER_CONFIG_FILENAME = "accuracy_decoder_config.json"


class ModelArgs(TTModelArgs):
    OP_KEYS = (
        # Embedding
        "EMB_WEIGHTS",
        # Feed forward
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        # Attention
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "ATTN_OUTPUT",
        "ATTN_W_LAYOUT",
        # Decoder
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
    )

    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,  # Set to False to reduce memory usage by not caching HF model
        prefetcher=None,
    ):
        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            cache_hf=cache_hf,
            prefetcher=prefetcher,
        )

        self.use_qk_fused = False  # For Gemma 3, we do not use qk fused ops (rotary embedding + paged cache update)
        self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.padded_vocab_size = 262400

    def get_warmup_prefill_supported_seq_lens(self):
        DEFAULT_VALUE = self.capped_warmup_seq_len
        # This dictionary is used to override the default ceil warmup prefill value
        model_specific_ceil_warmup_lengths = {"gemma-3-4b": 65536}

        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(self.base_model_name, DEFAULT_VALUE)
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )

        to_warmup_seq_lens = self.filter_warmup_seq_lens(to_warmup_seq_lens)

        return to_warmup_seq_lens

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        # TODO: Add more model-specific filtering here
        # This filtering is based on the current PR's (https://github.com/tenstorrent/tt-metal/pull/33143) sequence lengths that are used for warmup
        return to_warmup_seq_lens

    def get_trace_prefill_supported_seq_lens(self):
        default_supported_seq_lens = {
            # for gemma we have different default supported seq lens than in tt_transformers
            # TODO: should be empty until https://github.com/tenstorrent/tt-metal/issues/33041 is fixed
            "N150": [],
            "N300": [],
            "T3K": [],
            "TG": [],
        }

        # TODO: If no specific sequence lengths are listed for a model and device, the default one will be used (from the default_supported_seq_lens dictionary)
        # TODO: should be empty until https://github.com/tenstorrent/tt-metal/issues/33041 is fixed
        model_specific_supported_seq_lens = {
            # EXAMPLE: "gemma-3-4b": {
            #     "N150": [128, 1024, 2048],
            # }
        }

        model_name = self.base_model_name
        device_name = self.device_name

        # If there is no entry for a model in model_specific_supported_seq_lens, use the entry in default_supported_seq_lens
        result = model_specific_supported_seq_lens.get(model_name, {}).get(
            device_name, default_supported_seq_lens.get(device_name)
        )

        if result is not None:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)
        else:
            return []

    def _set_model_specific_params(self):
        self.rms_norm_add_unit_offset = True
        self.embed_scale = self.dim**0.5

    # def _set_vision_params(self, vision_config):
    #     self.vision_dim = vision_config.get("hidden_size", 1280)
    #     self.vision_mlp_ratio = vision_config.get("intermediate_size", self.vision_dim * 4) // self.vision_dim
    #     self.vision_hidden_dim = vision_config.get("intermediate_size", self.vision_dim * self.vision_mlp_ratio)
    #     self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
    #     self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
    #     self.vision_n_layers = vision_config.get("num_hidden_layers", 32)
    #     self.vision_patch_size = vision_config.get("patch_size", 14)
    #     self.vision_in_channels = vision_config.get("num_channels", 3)
    #     self.vision_act_layer = ttnn.UnaryOpType.GELU  # or read from config if variable
    #     self.vision_dropout = vision_config.get("attention_dropout", 0.0)
    #     self.vision_max_num_tiles = 4
    #     self.vision_n_global_layers = 8

    def _set_vision_params(self, vision_config):
        self.vision_chunk_size = vision_config.get("vision_chunk_size", 896)
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = vision_config.get("vision_num_cross_attention_layers", 8)
        self.vision_dim = vision_config.get("hidden_size", 1152)

        intermediate_size = vision_config.get("intermediate_size", self.vision_dim * 4)
        self.vision_mlp_ratio = intermediate_size // self.vision_dim
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads

        self.vision_n_layers = vision_config.get("num_hidden_layers", 27)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_in_channels = vision_config.get("num_channels", 3)

        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.mm_tokens_per_image = vision_config.get("mm_tokens_per_image", 256)

        # Optional vision activation layer, defaults to GELU
        act_layer = vision_config.get("act_layer", "gelu").lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)

        self.vision_n_global_layers = vision_config.get("n_global_layers", 8)

    def _set_hf_params(self, checkpoint_dir):
        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            # Merge non-nested keys into text_config
            text_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return text_config

        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            # Merge non-nested keys into vision_config
            vision_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return vision_config

        from transformers import AutoConfig

        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for gemma models for now.")
        else:
            self.hf_config = AutoConfig.from_pretrained(self.CKPT_DIR).to_dict()

        if "text_config" in self.hf_config or "vision_config" in self.hf_config:
            self._set_params_from_dict(self.hf_config)
            if "vision_config" in self.hf_config:
                merged_vision_config = merge_vision_config(self.hf_config)
                self._set_vision_params(merged_vision_config)
        else:
            self._set_params_from_dict(self.hf_config)

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if is_vision:
            text_prefix = "model.vision_tower.vision_model.encoder."
        else:
            text_prefix = ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",  # If no module is given, just get layer prefix
        }

        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "self_attn.",
            "TransformerBlock": "",
            "": "",
        }

        module_map = vision_module_map if is_vision else module_map

        return text_prefix + layer_prefix + module_map[module_name]

    # TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
    def load_state_dict(self):
        if self.dummy_weights:
            from transformers import AutoModelForCausalLM

            raise NotImplementedError("Dummy weights not supported for gemma models for now.")
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.CKPT_DIR,
                torch_dtype="auto"
                # Note that the default setting is torch.dtype.float32, but model weights are
                # may come in any dtype. If the model's weights are in torch.dtype.bfloat16, this would result in 2x memory usage from an
                # unnecessary cast.
            )
            if self.cache_hf_flag:
                self.cached_hf_model = model
            state_dict = model.state_dict()

        if self.is_multimodal:
            state_dict = convert_vision_hf_to_meta(state_dict, self.head_dim)
        else:
            state_dict = standardize_hf_keys(state_dict)
            state_dict = convert_hf_to_meta(state_dict, self.head_dim)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict

    def create_tokenizer(self):
        from transformers import AutoTokenizer

        # Mapping of base model names to their known tokenizer paths
        # These are the original models that have proper tokenizers
        base_model_tokenizer_mapping = {
            "gemma-3-4b-it": "google/gemma-3-4b-it",
        }

        logger.info(f"Tokenizer path: {self.TOKENIZER_PATH}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Base model name: {self.base_model_name}")

        try:
            # Try to load tokenizer from the original model path
            # If there is no Processor, it will return Tokenizer (useful for multimodal models)
            tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH, local_files_only=os.getenv("CI") == "true")
            logger.info(f"Successfully loaded tokenizer from {self.TOKENIZER_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {self.TOKENIZER_PATH}: {e}")

            # Try to use base model tokenizer as fallback
            fallback_tokenizer_path = base_model_tokenizer_mapping.get(self.base_model_name)

            if fallback_tokenizer_path:
                logger.info(f"Attempting to use fallback tokenizer: {fallback_tokenizer_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        fallback_tokenizer_path, local_files_only=os.getenv("CI") == "true"
                    )
                    logger.info(f"Successfully loaded fallback tokenizer from {fallback_tokenizer_path}")
                except Exception as fallback_e:
                    logger.error(f"Failed to load fallback tokenizer from {fallback_tokenizer_path}: {fallback_e}")
                    raise fallback_e
            else:
                logger.error(f"No fallback tokenizer found for base model: {self.base_model_name}")
                raise e

        # Add meta-compatible stop token list to the HF tokenizer
        if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
            tokenizer.stop_tokens = [tokenizer.eos_token_id]
        return tokenizer

    def reference_vision_multi_modal(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.multi_modal_projector
        return layer

    def reference_vision_rms_norm(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.multi_modal_projector.mm_soft_emb_norm
        return layer

    def reference_rms_norm_text(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def get_hf_model_cls(self):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

        if not self.is_multimodal:
            return AutoModelForCausalLM

        for model_cls in (AutoModelForVision2Seq, AutoModelForImageTextToText):
            if type(self.hf_config) == dict:
                return model_cls

        raise ValueError(f"Unknown model for config {type(self.hf_config)}")

    def reference_mlp(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].mlp
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_transformer(self, wrap=True, load_checkpoint=False):
        pass

        if self.dummy_weights and not load_checkpoint:
            raise NotImplementedError("Dummy weights not supported for gemma models for now.")
        else:
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(self.CKPT_DIR)
        if wrap:
            wrapper = HfModelWrapper(model, self.head_dim)
            return wrapper
        else:
            return model

    def reference_gemma_model(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_model(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model
        return layer

    def reference_vision_mlp(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder.layers[0].mlp
        return layer

    def reference_siglip_patch_embed(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings.patch_embedding
        return layer

    def reference_vision_pos_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings.position_embedding
        return layer

    def reference_vision_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings
        return layer

    def reference_vision_layernorm(self, layer_name="layer_norm1"):
        model = self.reference_vision_transformer(wrap=False)
        if layer_name == "layer_norm1":
            layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm1
        elif layer_name == "layer_norm2":
            layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm2
        else:
            layer = model.vision_tower.vision_model.post_layernorm
        return layer

    def reference_vision_attention(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder.layers[0].self_attn  # Common naming
        return layer

    def reference_vision_encoder_block(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder.layers[0]
        return layer

    def reference_vision_encoder(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder
        return layer

    # def reference_decoder(self, i=0):
    #     model = self.reference_transformer(wrap=False)
    #     layer = model.model.layers[i]
    #     rotary_emb = model.model.rotary_emb

    #     rotary_emb_local = model.model.rotary_emb_local
    #     wrapper = HfGemmaDecoderWrapper(layer, self.head_dim, rotary_emb, rotary_emb_local)

    #     return wrapper

    def reference_decoder_text(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0]
        use_position_embeddings = layer.__class__.__name__ != "Phi3DecoderLayer" or self.base_model_name in ("phi-4",)
        if hasattr(model.model, "rotary_emb_local"):
            rotary_emb_local = model.model.rotary_emb_local
        else:
            rotary_emb_local = None
        wrapper = HfDecoderWrapper(
            layer, self.head_dim, model.model.rotary_emb if use_position_embeddings else None, rotary_emb_local
        )
        return wrapper

    def reference_attention(self, rope_embeddings="global"):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].self_attn
        use_position_embeddings = layer.__class__.__name__ in ("Gemma3Attention",)
        if "gemma-3" in self.model_name:
            if rope_embeddings == "local":
                rotary_emb = model.model.rotary_emb_local
            else:
                rotary_emb = model.model.rotary_emb
        else:
            rotary_emb = model.model.rotary_emb
        wrapper = HfAttentionWrapper(layer, self.head_dim, rotary_emb if use_position_embeddings else None)
        return wrapper


class HfGemmaDecoderWrapper:
    def __init__(self, decoder, head_dim, rotary_emb, rotary_emb_local):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.rotary_emb_local = rotary_emb_local
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        # TODO: Generalize for other HF models

        position_embeddings_global = self.rotary_emb(x, position_ids)
        position_embeddings_local = self.rotary_emb_local(x, position_ids)
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
        result = self.decoder.forward(
            x,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            past_key_value=self.past_key_values,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=mask,
        )
        output = result[0]
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        return self.decoder.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim))
