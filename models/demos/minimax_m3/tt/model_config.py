# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M3 ModelArgs class that's compatible with tt_transformers interface
"""

import json
import os
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.demos.minimax_m3.utils.checkpoint import read_text_backbone, text_config_of
from models.demos.minimax_m3.utils.model_variant import VARIANTS, load_config, select_variant
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial


def default_weight_cache_root(model_path) -> Path:
    """Root under which ``tensor_cache_bfp8_{MeshShape}`` dirs live: ``TT_CACHE_PATH`` > the variant's shared cache
    root on weka (only for that variant's default checkpoint dir, when the root exists) > the checkpoint dir itself
    (the legacy NFS layout). Cache files do not record their source, so another checkpoint never defaults to a
    shared cache. The weights dirs are read-only, hence the separate cache roots."""
    cache_dir = os.getenv("TT_CACHE_PATH")
    if cache_dir:
        return Path(cache_dir)
    for variant in VARIANTS.values():
        is_default_checkpoint = os.path.realpath(model_path) == os.path.realpath(variant.weights_dir)
        if is_default_checkpoint and os.path.isdir(variant.cache_root):
            return Path(variant.cache_root)
    return Path(model_path)


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
        # no built-in default path. Dummy-weights mode needs no checkpoint: the model is built from the
        # selected variant's repo config (M3_MODEL_VARIANT, default MiniMax-M3; utils/model_variant.py).
        hf_model = os.getenv("HF_MODEL")
        if hf_model is None and not self.dummy_weights:
            raise ValueError(
                "HF_MODEL is not set. Point it at a MiniMax-M3 checkpoint directory, "
                "or construct ModelArgs with dummy_weights=True."
            )
        self.model_variant = select_variant(hf_model)
        if hf_model is None:
            hf_model = self.model_variant.name  # placeholder (dummy weights — no checkpoint load)
        self.model_path = hf_model
        self.weights_path = hf_model

        logger.info(
            f"Using {self.model_variant.name} model from: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

        if self.dummy_weights:
            from transformers import PretrainedConfig

            logger.info(f"Using dummy weights mode - model config from {self.model_variant.config_path}")
            self.vl_config = None
            self.hf_config = text_config_of(PretrainedConfig(**load_config(self.model_variant.name)))
        else:
            # M3's published config is VL-wrapped (model_type=minimax_m3_vl); the TEXT-backbone dims live
            # under `.text_config`, so we unwrap it and keep the wrapper (vision dims etc.) as `self.vl_config`.
            cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.vl_config = cfg
            self.hf_config = text_config_of(cfg)
        # Set key attributes that tt_transformers expects (text_config_of checked they are present).
        self.vocab_size = self.hf_config.vocab_size
        self.n_layers = self.hf_config.num_hidden_layers
        self.head_dim = self.hf_config.head_dim
        self.rope_theta = self.hf_config.rope_theta
        self.rope_scaling = getattr(self.hf_config, "rope_scaling", None)

        # Add missing attributes that Generator expects
        self.max_prefill_chunk_size = 128 * 1024
        self.model_name = Path(self.model_path).name
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
            # CANNOT build it. Read the safetensors directly: keep only the text backbone
            # (`language_model.*`) and strip that prefix -> the plain keys the model expects
            # (`model.*` / `lm_head.*`); drop the multimodal tensors (vision_tower /
            # multi_modal_projector / patch_merge_mlp). The checkpoint has no mtp/nextn weights.
            # Quantized (MXFP8 / NVFP4) weights are dequantized to bf16 here, before the
            # name-keyed QKV conversion below (utils/checkpoint.py).
            state_dict = ModelArgs._load_text_backbone_safetensors(weights_path)

            # Convert HF QKV weights to Meta format for RoPE compatibility (if requested).
            # M3 uses PARTIAL rotary (rotary_dim < head_dim), so only the rotary slice of each
            # head is interleaved (the shared full-head permute would be wrong); the helper also
            # swizzles the per-head q/k-norm gains so per-head QK-norm stays consistent (see
            # weights.py / test_attention_vs_ref.py).
            if convert_to_meta_format:
                logger.info("Converting QKV weights from HuggingFace to Meta format for RoPE (partial rotary)")
                tc = text_config_of(AutoConfig.from_pretrained(weights_path, trust_remote_code=True))
                state_dict = convert_hf_qkv_to_meta_format_partial(state_dict, tc.head_dim, tc.rotary_dim)
            return state_dict

    @staticmethod
    def _load_text_backbone_safetensors(weights_path):
        """Read the M3 safetensors and return the text-backbone state dict.

        Keeps `language_model.*` (the text backbone), strips that prefix so keys match what the
        model expects (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`,
        `lm_head.weight`), and drops the multimodal tensors. Quantized weights come back as bf16
        under their plain `.weight` key, without their scale tensors. Reads one tensor at a time.
        """
        weights_path = str(weights_path)
        index_path = os.path.join(weights_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        # DEBUG SPEED: M3_LOAD_NLAYERS=K loads ONLY the shards holding layers 0..K-1 (+ embed/norm/lm_head)
        # instead of all 869GB — for fast single-/few-layer real-weights isolation runs.
        _ln = os.getenv("M3_LOAD_NLAYERS")
        if _ln:
            kln = int(_ln)
            # M3_LOAD_LAYER_START=S loads the GLOBAL layer window [S, S+kln) instead of [0, kln). A
            # pipeline-parallel rank owns layers [first_layer_idx, first_layer_idx+num_layers), so its
            # adapter sets S=first_layer_idx to load exactly that slice (keys stay global-indexed).
            start = int(os.getenv("M3_LOAD_LAYER_START", "0"))
            keep = set()
            for k, shard in weight_map.items():
                kk = k[len("language_model.") :] if k.startswith("language_model.") else k
                if kk.startswith("model.layers."):
                    li = int(kk.split(".")[2])
                    if start <= li < start + kln:
                        keep.add(shard)
                else:  # embed / norm / lm_head / non-layer
                    keep.add(shard)
            shards = sorted(keep)

            # Shards also hold other layers' tensors: dequantize only this window's (and the non-layer ones).
            def _in_window(k):
                kk = k[len("language_model.") :] if k.startswith("language_model.") else k
                return not kk.startswith("model.layers.") or start <= int(kk.split(".")[2]) < start + kln

            weight_map = {k: v for k, v in weight_map.items() if _in_window(k)}
            logger.info(
                f"M3_LOAD_NLAYERS={kln} M3_LOAD_LAYER_START={start}: loading {len(shards)} shards "
                f"(of {len(set(weight_map.values()))}) for layers {start}..{start + kln - 1}"
            )
        else:
            shards = sorted(set(weight_map.values()))

        return read_text_backbone(weights_path, weight_map, tqdm(shards, desc="Loading M3 safetensors (text backbone)"))

    def weight_cache_path(self, dtype):
        """Return weight cache path for the model"""
        cache_dir = default_weight_cache_root(self.model_path)
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
