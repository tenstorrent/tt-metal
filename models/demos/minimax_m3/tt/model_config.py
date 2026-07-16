# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M3 ModelArgs class that's compatible with tt_transformers interface
"""

import json
import os
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial


class ModelArgs:
    """MiniMax-M3 ModelArgs compatible with tt_transformers create_tt_model interface"""

    def __init__(
        self,
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
    ):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        if self.max_batch_size > 32:
            assert (
                self.max_batch_size % self.mesh_device.shape[0] == 0
            ), "max_batch_size must be divisible by the number of device rows"
            self.max_local_batch_size = self.max_batch_size // self.mesh_device.shape[0]
        else:
            self.max_local_batch_size = self.max_batch_size
        self.max_seq_len = max_seq_len
        if optimizations is not None:
            logger.warning("MiniMax-M3 doesn't support any performance optimizations - ignoring optimizations argument")
        self.optimizations = None
        self.cache_hf = cache_hf

        # Weights come from the HF_MODEL environment variable (tt_transformers standard); there is
        # no built-in default path. Dummy-weights mode needs no checkpoint, so a name-only
        # placeholder is enough to satisfy the model-name check below.
        hf_model = os.getenv("HF_MODEL")
        if hf_model is None:
            if not self.dummy_weights:
                raise ValueError(
                    "HF_MODEL is not set. Point it at a MiniMax-M3 checkpoint directory, "
                    "or construct ModelArgs with dummy_weights=True."
                )
            hf_model = "MiniMax-M3"  # placeholder (dummy weights — no checkpoint load)
        self.model_path = hf_model
        self.weights_path = hf_model

        logger.info(
            f"Using MiniMax-M3 model from: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

        if self.dummy_weights:
            # Skip loading HF config for testing - use default values
            logger.info("Using dummy weights mode - skipping HuggingFace config loading")

        else:
            # Load HF config to get model parameters. M3's published config is VL-wrapped
            # (model_type=minimax_m3_vl); the TEXT-backbone dims live under `.text_config`, so we
            # unwrap it and keep the wrapper (vision dims etc.) as `self.vl_config` for later.
            cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.vl_config = cfg
            self.hf_config = getattr(cfg, "text_config", cfg)
            # Set key attributes that tt_transformers expects
            self.vocab_size = self.hf_config.vocab_size
            self.n_layers = getattr(self.hf_config, "num_hidden_layers", 32)
            # head_dim (128) is set independently of hidden_size/num_attention_heads, so read it
            # directly from the config.
            self.head_dim = getattr(
                self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads
            )
            self.rope_theta = getattr(self.hf_config, "rope_theta", 10000.0)
            self.rope_scaling = getattr(self.hf_config, "rope_scaling", None)

        # Add missing attributes that Generator expects
        self.max_prefill_chunk_size = 128 * 1024
        self.model_name = Path(self.model_path).name
        assert self.model_name.startswith(
            "MiniMax-M3"
        ), f"Unrecognized model name {self.model_name} inferred from model path {self.model_path}. Expected a MiniMax-M3* checkpoint dir (e.g. MiniMax-M3 or MiniMax-M3-ref)."  # Model identifier
        self.max_context_len = max_seq_len  # Context length for tt_transformers compatibility

        if self.dummy_weights:
            # Skip tokenizer loading for testing
            self.tokenizer = None
            self.processor = None
        else:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True)
            self.processor = None  # MiniMax-M3 doesn't use vision processor

    def encode_prompt(self, prompt_text, instruct=False, system_prompt_text=None):
        """
        Encode prompts using HuggingFace tokenizer with chat template
        Compatible with tt_transformers interface
        """
        assert not instruct, "MiniMax-M3 does not support instruct mode"
        chat = []
        if isinstance(prompt_text, str):
            if system_prompt_text:
                chat.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                chat.append({"role": "user", "content": prompt_text})
            return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
        else:
            # prompt_text is already a list of chat messages
            return self.tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True):
        """Load model state dict compatible with tt_transformers

        Args:
            weights_path (str or Path): Path to the model weights directory or file.
            dummy_weights (bool): If True, returns a dummy state dict for testing purposes.
            convert_to_meta_format (bool): If True, convert HF QKV weights to Meta format for RoPE.
                Set to False when loading for HuggingFace reference models.
        """
        if dummy_weights:
            # Return dummy state dict for testing
            return {}
        else:
            # M3 ships NO modeling code (the checkpoint carries only config + multimodal
            # processors; no AutoModel in auto_map), so AutoModelForCausalLM.from_pretrained
            # CANNOT build it. Read the bf16 safetensors directly: keep only the text backbone
            # (`language_model.*`) and strip that prefix -> the plain keys the model expects
            # (`model.*` / `lm_head.*`); drop the multimodal tensors (vision_tower /
            # multi_modal_projector / patch_merge_mlp). The checkpoint has no mtp/nextn weights.
            state_dict = ModelArgs._load_text_backbone_safetensors(weights_path)

            # Convert HF QKV weights to Meta format for RoPE compatibility (if requested).
            # M3 uses PARTIAL rotary (rotary_dim < head_dim), so only the rotary slice of each
            # head is interleaved (the shared full-head permute would be wrong); the helper also
            # swizzles the per-head q/k-norm gains so per-head QK-norm stays consistent (see
            # weights.py / test_attention_vs_ref.py).
            if convert_to_meta_format:
                logger.info("Converting QKV weights from HuggingFace to Meta format for RoPE (partial rotary)")
                cfg = AutoConfig.from_pretrained(weights_path, trust_remote_code=True)
                tc = getattr(cfg, "text_config", cfg)
                rotary_dim = getattr(tc, "rotary_dim", tc.head_dim)
                state_dict = convert_hf_qkv_to_meta_format_partial(state_dict, tc.head_dim, rotary_dim)
            return state_dict

    @staticmethod
    def _load_text_backbone_safetensors(weights_path):
        """Read the M3 bf16 safetensors and return the text-backbone state dict.

        Keeps `language_model.*` (the text backbone), strips that prefix so keys match what the
        model expects (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`,
        `lm_head.weight`), and drops the multimodal tensors. Loads shards lazily from the index.
        """
        from safetensors.torch import load_file

        weights_path = str(weights_path)
        index_path = os.path.join(weights_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        # DEBUG SPEED: M3_LOAD_NLAYERS=K loads ONLY the shards holding layers 0..K-1 (+ embed/norm/lm_head)
        # instead of all 869GB — for fast single-/few-layer real-weights isolation runs.
        _ln = os.getenv("M3_LOAD_NLAYERS")
        if _ln:
            kln = int(_ln)
            keep = set()
            for k, shard in weight_map.items():
                kk = k[len("language_model.") :] if k.startswith("language_model.") else k
                if kk.startswith("model.layers."):
                    li = int(kk.split(".")[2])
                    if li < kln:
                        keep.add(shard)
                else:  # embed / norm / lm_head / non-layer
                    keep.add(shard)
            shards = sorted(keep)
            logger.info(
                f"M3_LOAD_NLAYERS={kln}: loading {len(shards)} shards (of {len(set(weight_map.values()))}) for layers 0..{kln-1}"
            )
        else:
            shards = sorted(set(weight_map.values()))

        PREFIX = "language_model."
        state_dict = {}
        for shard in tqdm(shards, desc="Loading M3 bf16 safetensors (text backbone)"):
            shard_sd = load_file(os.path.join(weights_path, shard))
            for k, v in shard_sd.items():
                if not k.startswith(PREFIX):
                    continue  # drop vision_tower / multi_modal_projector / patch_merge_mlp
                new_k = k[len(PREFIX) :]  # language_model.model.X -> model.X ; language_model.lm_head -> lm_head
                state_dict[new_k] = v.to(torch.bfloat16) if v.dtype == torch.float32 else v
        return state_dict

    def weight_cache_path(self, dtype):
        """Return weight cache path for the model"""
        cache_dir = os.getenv("TT_CACHE_PATH")
        if cache_dir:
            cache_dir = Path(cache_dir)  # If we specify a TT_CACHE_PATH, use that for the cache
        else:
            cache_dir = Path(self.model_path)  # Use same directory as model
        logger.info(f"Cache directory: {cache_dir}")
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        cache_path = cache_dir / f"tensor_cache_{dtype_str}_{self.mesh_device.shape}"

        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_model_config(self):
        """Return model configuration dict"""
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
        }

    def get_state_dict_prefix(self, prefix, layer_idx):
        """Get state dict prefix for layer weights"""
        if layer_idx is None:
            return prefix
        return f"{prefix}layers.{layer_idx}."
