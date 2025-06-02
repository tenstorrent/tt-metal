# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import math
import torch

from loguru import logger
from models.tt_transformers.tt.model_config import ModelArgs
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import nearest_multiple


class Phi3MiniModelArgs(ModelArgs):
    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
    ):
        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )

        # Set the max number of tokens for each prefill chunk based on the model and device
        max_prefill_chunk_size_div1024 = os.getenv("MAX_PREFILL_CHUNK_SIZE")
        if max_prefill_chunk_size_div1024 is None:
            # TODO Improve this to be more general to more devices and models
            MAX_PREFILL_CHUNK_SIZES_DIV1024 = {"N150": 1, "N300": 32}
            try:
                max_prefill_chunk_size_div1024 = MAX_PREFILL_CHUNK_SIZES_DIV1024[self.device_name]
            except KeyError:
                logger.warning(
                    f"Model {self.model_name} on device {self.device_name}, setting MAX_PREFILL_CHUNK_SIZE to 4 for compatibility"
                )
                logger.warning(
                    f"Try setting MAX_PREFILL_CHUNK_SIZE to larger powers of 2 up to e.g. 128 for faster performance (if you run out of L1 memory it was too high)"
                )
                max_prefill_chunk_size_div1024 = 4
            assert (
                max_prefill_chunk_size_div1024 is not None
            ), f"Unsupported model {self.model_name} on device {self.device_name}"
        else:
            max_prefill_chunk_size_div1024 = int(max_prefill_chunk_size_div1024)
        self.max_prefill_chunk_size = max_prefill_chunk_size_div1024 * 1024

        # Set the min number of tokens for each prefill chunk based on the model and device
        min_prefill_chunk_size_div1024 = os.getenv("MIN_PREFILL_CHUNK_SIZE")
        if min_prefill_chunk_size_div1024 is None:
            # TODO Improve this to be more general to more devices and models
            MIN_PREFILL_CHUNK_SIZES_DIV1024 = {"N150": 1, "N300": 2}
            try:
                min_prefill_chunk_size_div1024 = MIN_PREFILL_CHUNK_SIZES_DIV1024[self.device_name]
            except KeyError:
                logger.warning(
                    f"Model {self.model_name} on device {self.device_name}, setting MIN_PREFILL_CHUNK_SIZE to 2 for compatibility"
                )
                min_prefill_chunk_size_div1024 = 2
            assert (
                min_prefill_chunk_size_div1024 is not None
            ), f"Unsupported model {self.model_name} on device {self.device_name}"
        else:
            min_prefill_chunk_size_div1024 = int(min_prefill_chunk_size_div1024)
        self.min_prefill_chunk_size = min_prefill_chunk_size_div1024 * 1024

        assert (
            self.min_prefill_chunk_size <= self.max_prefill_chunk_size
        ), f"Min prefill chunk size {self.min_prefill_chunk_size} should not be greater than Max prefill chunk size {self.max_prefill_chunk_size}"

    def _set_params_from_dict(self, params):
        # Common params with different names between Meta and HF
        self.dim = params.get("hidden_size")
        self.n_heads = params.get("num_attention_heads")
        self.n_kv_heads = params.get("num_key_value_heads")
        self.n_layers = params.get("num_hidden_layers")
        self.full_model_n_layers = self.n_layers
        self.norm_eps = params.get("rms_norm_eps")
        self.vocab_size = params["vocab_size"]
        self.padded_vocab_size = 32 * 1024
        self.head_dim = self.dim // self.n_heads

        # Handle different MLP dimension specifications
        self.hidden_dim = params["intermediate_size"]
        self.ffn_dim_multiplier = None
        self.multiple_of = None

        if "_name_or_path" in params:
            self.model_name = os.path.basename(params["_name_or_path"])

        self.unpadded_hidden_dim = self.hidden_dim
        # Don't need to pad for CPU runs
        if self.num_devices:
            # Default padding cores for each model, 0 if not set here
            default_padded_cores = 0

            # Override MLP padding cores from env var
            mlp_padded_cores = int(os.environ.get("PAD_MLP_CORES", default_padded_cores))

            # Only pad if MLP_PADDED_CORES is non-zero
            if mlp_padded_cores > 0:
                padded_hidden_dim = nearest_multiple(
                    self.hidden_dim, mlp_padded_cores * self.tile_size * self.num_devices
                )
                if padded_hidden_dim != self.hidden_dim:
                    logger.info(
                        f"PAD_MLP_CORES={mlp_padded_cores}, padding hidden dim from {self.hidden_dim} to {padded_hidden_dim}"
                    )
                    self.hidden_dim = padded_hidden_dim

        # RoPE params
        self.rope_theta = params.get("rope_theta")
        # If use_scaled_rope is not present, assume setting rope_scaling means use scaled rope
        # If it is present and is set to false, do not use scaled rope
        # Setting self.rope_scaling_factor to None is our way of saying do not use scaled rope
        self.rope_scaling_factor = None
        if "rope_scaling" in params:
            self.max_context_len = params.get("max_position_embeddings", None)
            self.orig_context_len = params.get("original_max_position_embeddings", None)
            self.rope_scaling = params.get("rope_scaling")
            if self.rope_scaling["type"] == "longrope":
                scale = self.max_context_len / self.orig_context_len
                self.rope_scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.orig_context_len))

        # Vision params (Meta-specific)
        self.vision_chunk_size = -1
        self.vision_max_num_chunks = 4
        self.vision_num_cross_attention_layers = -1

        # Vision constants
        self.vision_dim = 1280
        self.vision_mlp_ratio = 4
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_act_layer = ttnn.UnaryOpType.GELU
        self.vision_dropout = 0.0
        self.vision_attn_n_heads = 16
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_n_layers = 32
        self.vision_n_global_layers = 8
        self.vision_max_num_tiles = 4
        self.vision_patch_size = 14
        self.vision_in_channels = 3

    def __repr__(self):
        return f"""ModelArgs(
            dim={self.dim},
            n_layers={self.n_layers},
            n_heads={self.n_heads},
            n_kv_heads={self.n_kv_heads},
            vocab_size={self.vocab_size},
            multiple_of={self.multiple_of},
            norm_eps={self.norm_eps},
            rope_theta={self.rope_theta},
            rope_scaling_factor={self.rope_scaling_factor},
            rope_scaling={self.rope_scaling},
            max_batch_size={self.max_batch_size},
            max_seq_len={self.max_seq_len},
        )"""

    def reference_decoder(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0]
        wrapper = HfDecoderWrapper(layer, self.head_dim)
        return wrapper

    def create_tokenizer(self):
        """Create and return a Tokenizer instance."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH)

        # Add stop token list to the HF tokenizer
        if not "stop_tokens" in tokenizer.__dict__:
            tokenizer.stop_tokens = [tokenizer.eos_token_id, tokenizer.encode("<|end|>")[0]]
        return tokenizer


class HfDecoderWrapper(LightweightModule):
    def __init__(self, decoder, head_dim):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
        result, self.past_key_values = self.decoder.forward(
            x,
            past_key_value=self.past_key_values,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=mask,
        )
        return result
