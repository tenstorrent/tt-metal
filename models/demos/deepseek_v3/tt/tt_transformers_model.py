# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V3 model wrapper for tt_transformers Generator compatibility.

This module provides a wrapper that makes DeepSeek-V3 compatible with
the tt_transformers Generator class by implementing the required interface.
"""

from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.mla import MLA
from models.demos.deepseek_v3.tt.model import Model
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE, get_weight_config
from models.demos.deepseek_v3.utils.hf_model_utils import load_model_weights, load_tokenizer
from models.demos.deepseek_v3.utils.run_config import create_run_config


class DeepSeekV3ModelArgs:
    """Model arguments class compatible with tt_transformers Generator."""

    def __init__(
        self, mesh_device, hf_config, tokenizer, processor=None, max_batch_size=MAX_BATCH_SIZE, max_seq_len=4096
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = hf_config.vocab_size
        self.n_layers = hf_config.num_hidden_layers
        self.model_name = "deepseek_v3"
        self.configuration = hf_config  # For compatibility
        self.max_context_len = hf_config.max_position_embeddings
        self.max_prefill_chunk_size = 1024  # TODO: tune this

    def get_model_config(self):
        """Return model config for compatibility."""
        return self.hf_config

    def encode_prompt(self, prompt, instruct=True):
        """Encode prompt using tokenizer."""
        if instruct:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
        else:
            return self.tokenizer.encode(prompt, add_special_tokens=True)


class DeepSeekV3TTTransformersModel:
    """DeepSeek-V3 model wrapper for tt_transformers Generator compatibility."""

    def __init__(
        self, model_args, mesh_device, hf_config, paged_config, ccl, model_state, model_shared_state, run_configs
    ):
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.paged_config = paged_config
        self.ccl = ccl
        self.model_state = model_state
        self.model_shared_state = model_shared_state
        self.run_configs = run_configs

        # Required attributes for tt_transformers compatibility
        self.vocab_size = hf_config.vocab_size
        self.n_layers = hf_config.num_hidden_layers
        self.args = model_args
        self.configuration = hf_config

        # Setup RoPE
        self.rope_setup = RotarySetup(device=mesh_device, batch_size=MAX_BATCH_SIZE, hf_config=hf_config)

        # Create page tables for each layer
        self.page_tables_tt = [
            MLA.create_page_table(
                paged_config=paged_config,
                mesh_device=mesh_device,
            )
            for _ in range(hf_config.num_hidden_layers)
        ]

        # Dummy attributes for compatibility
        self.device_decode_sliding_mask = None
        self.decode_sliding_mask_mat = None

    def setup_cache(self, max_batch_size):
        """Setup KV cache for the model."""
        # DeepSeek-V3 handles caching internally, return dummy cache
        return None

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        """Prepare inputs for prefill phase."""
        # Convert tokens to embeddings

        # tokens_embd = self._embed_tokens(tokens)

        # # Get RoPE matrices
        # seq_len = tokens.shape[-1]
        # rope_mats = self.rope_setup.get_rot_mats_table(seq_len)
        # rot_mats_global = rope_mats[0]
        # rot_mats_local = None  # DeepSeek-V3 doesn't use local RoPE

        # # Convert page table if provided
        # tt_page_table = None
        # if page_table is not None:
        #     tt_page_table = ttnn.from_torch(
        #         page_table,
        #         device=self.mesh_device,
        #         dtype=ttnn.int32,
        #         layout=ttnn.ROW_MAJOR_LAYOUT,
        #         mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        #     )

        # return tokens_embd, rot_mats_global, rot_mats_local, tt_page_table, None
        # TODO: implement this
        if len(tokens) > self.model_args.max_prefill_chunk_size:
            assert False, "Tokens length exceeds max prefill chunk size. Chunking is not supported for DeepSeek-V3."

        if tokens.dim() == 2:
            tokens = tokens.reshape(1, 1, -1)

        return tokens, None, None, None, None

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """Prepare inputs for decode phase."""
        # Convert tokens to embeddings
        # tokens_embd = self._embed_tokens(tokens)

        # # Get RoPE matrices for current positions
        # rope_idxs = self.rope_setup.get_rot_idxs(current_pos, on_host=True)
        # rot_mats_global = self.rope_setup.get_rot_mats(rope_idxs)
        # rot_mats_local = None  # DeepSeek-V3 doesn't use local RoPE

        # # Convert page table if provided
        # tt_page_table = None
        # if page_table is not None:
        #     tt_page_table = ttnn.from_torch(
        #         page_table,
        #         device=self.mesh_device,
        #         dtype=ttnn.int32,
        #         layout=ttnn.ROW_MAJOR_LAYOUT,
        #         mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        #     )

        # TODO: implement this

        if tokens.dim() == 2:
            tokens = tokens.reshape(1, 1, -1)

        return tokens, current_pos, None, None

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Prepare decode inputs on host."""
        return self.prepare_inputs_decode(tokens, current_pos, page_table)

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        """Forward pass for prefill phase."""
        # TODO : ingore all the inputs???

        tokens_batched = x.view(1, 1, -1)
        seq_len = tokens_batched.numel()

        # Prepare TT inputs for prefill - reshape to [1, 1, actual_seq_len]
        tt_tokens = ttnn.from_torch(
            tokens_batched,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Use DeepSeek-V3's prefill forward
        rope_setup = RotarySetup(
            device=self.mesh_device,
            batch_size=1,
            hf_config=self.hf_config,
        )
        rot_mats = rope_setup.get_rot_mats_table(seq_len)
        rope_tensors = {
            "cos_matrix": rot_mats[0],
            "sin_matrix": rot_mats[1],
            "trans_matrix": rot_mats[2],
        }
        logits_tt = Model.forward_prefill(
            tt_tokens, user_id, self.run_configs["prefill"], rope_tensors, self.page_tables_tt
        )
        # logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))

        # # Free device tensors for this step
        # ttnn.deallocate(tt_tokens)
        # ttnn.deallocate(logits_tt)

        # logger.info(f"ttnn_prefill_forward logits shape: {logits.shape}")
        return logits_tt

    def ttnn_decode_forward(
        self, x, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None, argmax_on_device=False
    ):
        """Forward pass for decode phase."""
        rope_tensors = {
            "cos_matrix": self.rope_setup.get_rot_mats(rot_mat_idxs)[0],
            "sin_matrix": self.rope_setup.get_rot_mats(rot_mat_idxs)[1],
            "trans_matrix": self.rope_setup.get_rot_mats(rot_mat_idxs)[2],
        }

        # Use DeepSeek-V3's decode forward
        logits_tt = Model.forward_decode(x, current_pos, self.run_configs["decode"], rope_tensors, self.page_tables_tt)

        return logits_tt

    def process_output_prefill(self, tt_out, last_token_idx):
        """Process prefill output to get logits."""
        # Convert TT tensor to torch tensor
        logits = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
        return logits[0, 0, last_token_idx, : self.vocab_size]

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """Process decode output to get logits."""
        if is_tokens:
            return ttnn.to_torch(tt_out)[0, 0, :B, 0]

        # Convert TT tensor to torch tensor
        logits = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
        return logits[:, :, :B, : self.vocab_size].view(B, S, -1)

    def update_attention_masks(self, current_pos):
        """Update attention masks for sliding window (not used in DeepSeek-V3)."""
        # DeepSeek-V3 doesn't use sliding window attention

    def compute_vision_tokens_masks(self, *args, **kwargs):
        """Compute vision token masks (not used in DeepSeek-V3)."""
        # DeepSeek-V3 doesn't support vision
        return None, None

    def transform_decode_inputs_device(self, tokens):
        """Transform decode inputs on device."""
        return self._embed_tokens(tokens)

    def _embed_tokens(self, tokens):
        """Embed tokens using DeepSeek-V3 embedding."""
        # Convert tokens to TT tensor
        tt_tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Use DeepSeek-V3 embedding
        from models.demos.deepseek_v3.tt.embedding import Embedding

        embedding_state = self.model_state["embedding"]
        tt_embeddings = Embedding.forward_prefill(tt_tokens, embedding_state)

        return tt_embeddings


def create_deepseek_v3_tt_transformers_model(
    mesh_device,
    max_batch_size=MAX_BATCH_SIZE,
    max_seq_len=4096,
    state_dict=None,
    model_path="models/demos/deepseek_v3/reference",
    cache_dir="generated/deepseek_v3",
):
    """
    Create DeepSeek-V3 model compatible with tt_transformers Generator.

    Returns:
        model_args: DeepSeekV3ModelArgs instance
        model: DeepSeekV3TTTransformersModel instance
        tt_kv_cache: List of KV cache tensors (empty for DeepSeek-V3)
        state_dict: Loaded state dictionary
    """
    logger.info("Creating DeepSeek-V3 model compatible with tt_transformers Generator")

    # Load HF config and tokenizer
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hf_config.max_seq_len = max_seq_len

    hf_config.num_hidden_layers = 4
    tokenizer = load_tokenizer(model_path)

    # Create model args
    model_args = DeepSeekV3ModelArgs(
        mesh_device=mesh_device,
        hf_config=hf_config,
        tokenizer=tokenizer,
        processor=None,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )

    # Setup CCL and paged attention
    ccl = CCL(mesh_device)
    mesh_shape = list(mesh_device.shape)
    dp_factor = mesh_shape[1]

    paged_config = MLA.get_valid_paged_config(max_seq_len, max_batch_size, dp_factor)

    # Load or create weights
    if not state_dict:
        logger.info(f"Loading HF weights from {model_path}")
        hf_weights = load_model_weights(model_path)
        state_dict = {
            k: v
            for k, v in hf_weights.items()
            if k.startswith("model.embed_tokens.")
            or k.startswith("model.layers.")
            or k.startswith("model.norm.")
            or k.startswith("lm_head.")
        }

    # Convert weights to TT format
    weight_cache_path = cache_dir
    logger.info("Converting weights to TTNN format")
    model_weight_config = get_weight_config(
        ModuleClass=Model,
        hf_config=hf_config,
        state_dicts=(state_dict,),
        weight_cache_path=weight_cache_path,
        mesh_device=mesh_device,
        force_recalculate=False,
    )

    # Create run configs
    logger.info("Creating run configs")
    prefill_cfg = Model.prefill_model_config(hf_config, mesh_device)
    decode_cfg = Model.decode_model_config(hf_config, mesh_device)

    # Create model states
    logger.info("Creating model states")
    model_state = Model.create_state(hf_config=hf_config, mesh_device=mesh_device, paged_config=paged_config, ccl=ccl)

    model_shared_state = Model.create_shared_state(hf_config=hf_config, mesh_device=mesh_device)

    run_config_prefill = create_run_config(prefill_cfg, model_weight_config, model_state, model_shared_state)
    run_config_decode = create_run_config(decode_cfg, model_weight_config, model_state, model_shared_state)

    run_configs = {"prefill": run_config_prefill, "decode": run_config_decode}

    # Create model wrapper
    model = DeepSeekV3TTTransformersModel(
        model_args=model_args,
        mesh_device=mesh_device,
        hf_config=hf_config,
        paged_config=paged_config,
        ccl=ccl,
        model_state=model_state,
        model_shared_state=model_shared_state,
        run_configs=run_configs,
    )

    # DeepSeek-V3 doesn't use traditional KV cache, return empty list
    tt_kv_cache = None

    logger.info("DeepSeek-V3 model created successfully")
    return model_args, model, tt_kv_cache
