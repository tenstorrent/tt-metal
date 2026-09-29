# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS ModelArgs (config + weight loader). Mirrors ``minimax_m3/tt/model_config.py``.

Loads the HF gpt-oss config (dims live at the top level — gpt-oss is NOT a VL wrapper, but
``text_config`` is unwrapped defensively) and loads the weights via HF ``from_pretrained``
(dequantizing the MXFP4 experts to bf16 on the host), converting the q/k projections to Meta
format for the on-device (indexed) RoPE. Dim constants are cross-checked
against ``deepseek_v3_d_p/reference/gpt_oss_120b_config.py::GptOss120BConfig``.
"""

import gc
import os
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

# Bundled config.json (no network / checkpoint needed for the config path).
DEFAULT_HF_MODEL = "models/demos/gpt_oss/configs/gpt-oss-120b"


class ModelArgs:
    """GPT-OSS ModelArgs compatible with the tt_transformers create_tt_model interface."""

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
            logger.warning("GPT-OSS doesn't support perf optimizations - ignoring optimizations argument")
        self.optimizations = None
        self.cache_hf = cache_hf

        # Weights / config from HF_MODEL (tt_transformers standard); fall back to the bundled config
        # dir (config-only, e.g. dummy weights or the config path of a cache-only run).
        hf_model = os.getenv("HF_MODEL") or DEFAULT_HF_MODEL
        self.model_path = hf_model
        self.weights_path = hf_model
        logger.info(
            f"Using GPT-OSS model from: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

        # Load HF config (defensively unwrap text_config for any wrapped variant).
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.hf_config = getattr(cfg, "text_config", cfg)
        self.vocab_size = self.hf_config.vocab_size
        self.n_layers = getattr(self.hf_config, "num_hidden_layers", GptOss120BConfig.NUM_LAYERS)
        self.head_dim = getattr(
            self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads
        )
        self.rope_theta = getattr(self.hf_config, "rope_theta", GptOss120BConfig.ROPE_THETA)
        self.rope_scaling = getattr(self.hf_config, "rope_scaling", None)

        # tt_transformers-Generator-expected attributes.
        self.max_prefill_chunk_size = 128 * 1024
        self.model_name = Path(self.model_path).name
        self.max_context_len = max_seq_len

        if self.dummy_weights:
            self.tokenizer = None
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True)
            except Exception as e:  # config-only dir has no tokenizer
                logger.warning(f"No tokenizer at {self.weights_path} ({e}); tokenizer disabled")
                self.tokenizer = None
        self.processor = None

    def encode_prompt(self, prompt_text, instruct=False, system_prompt_text=None):
        """Encode prompts via the HF tokenizer chat template (tt_transformers interface)."""
        chat = []
        if isinstance(prompt_text, str):
            if system_prompt_text:
                chat.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                chat.append({"role": "user", "content": prompt_text})
            return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
        return self.tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True):
        """Load the gpt-oss state dict.

        Reads the bf16 safetensors directly (gpt-oss keys are already ``model.*`` / ``lm_head.*``, no
        wrapper prefix to strip) and, unless disabled, converts the q/k projections to Meta format for
        the on-device RoPE (the q/k weight + bias reverse_permute; see convert_hf_qkv_to_meta_format).
        The MoE expert tensors (``experts.gate_up_proj`` / ``down_proj`` [+ *_bias]) are returned
        verbatim; ``moe/weights.py::prepare_routed_expert_weights`` de-interleaves them.
        """
        if dummy_weights:
            return {}
        state_dict = ModelArgs._load_safetensors(weights_path)
        if convert_to_meta_format:
            cfg = AutoConfig.from_pretrained(weights_path, trust_remote_code=True)
            tc = getattr(cfg, "text_config", cfg)
            head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
            logger.info("Converting q/k projections from HuggingFace to Meta format for RoPE (full rotary)")
            state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
        return state_dict

    @staticmethod
    def _load_safetensors(weights_path):
        """Load the gpt-oss weights, dequantizing the MXFP4 experts to bf16.

        openai/gpt-oss ships the MoE experts MXFP4-quantized (``experts.gate_up_proj_blocks`` /
        ``_scales`` + ``down_proj_blocks`` / ``_scales``); the raw safetensors have NO dense
        ``gate_up_proj`` / ``down_proj`` tensor, so reading them directly leaves the experts packed
        and ``prepare_routed_expert_weights`` KeyErrors. Load via HF ``from_pretrained`` so
        transformers dequantizes the experts on the host, and target bf16 directly so the dequant
        intermediate is bf16 (not fp32) -- avoids ~2x host footprint / OOM on the 120B. Mirrors
        ``models/demos/gpt_oss/tt/model_config.py`` (see #48508/#48509).
        """
        model = AutoModelForCausalLM.from_pretrained(
            str(weights_path),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        state_dict = model.state_dict()
        # Drop the HF module now that we hold the weight tensors (state_dict shares storage), so the
        # peak host footprint is bounded by the bf16 weights rather than weights + a live model.
        del model
        gc.collect()
        # Safety net: cast any fp32 stragglers. No-op for the bf16-loaded tensors (rebinds refs).
        norm = state_dict.get("model.norm.weight")
        if norm is not None and norm.dtype != torch.bfloat16:
            state_dict = {
                k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v)
                for k, v in tqdm(state_dict.items(), desc="Casting fp32 stragglers -> bf16")
            }
        return state_dict

    def weight_cache_path(self, dtype):
        """Weight-cache dir for this model + mesh."""
        cache_dir = os.getenv("TT_CACHE_PATH")
        cache_dir = Path(cache_dir) if cache_dir else Path(self.model_path)
        logger.info(f"Cache directory: {cache_dir}")
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        cache_path = cache_dir / f"tensor_cache_{dtype_str}_{self.mesh_device.shape}"
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_model_config(self):
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
        }

    def get_state_dict_prefix(self, prefix, layer_idx):
        if layer_idx is None:
            return prefix
        return f"{prefix}layers.{layer_idx}."
