# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 ``ModelArgs`` — config + checkpoint loader.

Donor: ``gpt_oss_d_p/tt/model_config.py`` (``weights.loading``), which plays exactly this role —
``ModelArgs.load_state_dict`` -> ``_load_safetensors`` (dequantize on load) ->
``convert_hf_qkv_to_meta_format`` for the on-device RoPE, plus ``weight_cache_path``.

Adapted from the donor:

  * **the wrapper prefix.** gpt-oss keys are already ``model.*`` / ``lm_head.*``; Mistral ships a
    ``Mistral3ForConditionalGeneration`` checkpoint whose text backbone lives under
    ``model.language_model.*`` (verified against transformers 5.12's own state dict; the older
    ``language_model.model.*`` spelling is accepted too). Those are rewritten to the ``model.*``
    naming that both the TT model and the reference use, and a load that matches NOTHING fails loudly
    rather than building a model out of zero weights.
  * **no MoE.** Every expert key path is gone — Mistral is dense.
  * **no vision.** ``model.vision_tower.*`` and ``model.multi_modal_projector.*`` are dropped: this
    is the text backbone, and the Pixtral tower is out of scope for prefill.
  * **head_dim 128** in the Meta-format permute (the donor's is 64).
  * **fp8, not MXFP4.** The donor loads through HF ``from_pretrained`` so transformers dequantizes
    its MXFP4 experts. Here the safetensors are read directly and dequantized by
    :mod:`.fp8_dequant` (per-tensor scales), which keeps the peak host footprint at one shard rather
    than a whole live HF model, and keeps the dequantization ours to test (``test_fp8_loader.py``).
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.demos.mistral_3_5_d_p.reference.mistral_config import CONFIG_DIR
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

# Bundled config.json dir (no network / checkpoint needed for the config-only path).
DEFAULT_HF_MODEL = str(CONFIG_DIR)

# Wrapper prefixes seen on Mistral3 checkpoints, mapped onto the text-backbone naming
# (``model.layers.*`` / ``model.embed_tokens.weight`` / ``model.norm.weight``) that this package and
# the reference both use. Longest first: "model.language_model." must win over "language_model.".
_TEXT_PREFIXES = (
    ("model.language_model.", "model."),  # transformers >= 5 (verified against Mistral3ForConditionalGeneration)
    ("language_model.model.", "model."),  # older export spelling
    ("language_model.", "model."),
)
# Keys belonging to parts of the checkpoint prefill does not build.
_DROP_SUBSTRINGS = ("vision_tower", "multi_modal_projector", "patch_merger", "vision_language_adapter")


def map_checkpoint_keys(state_dict: dict) -> dict:
    """Strip the Mistral3 wrapper prefix and drop the vision-side keys.

    Returns keys in the text-backbone naming. Raises if nothing survives, because a silent empty
    result would build a model of freshly-allocated (garbage) weights.
    """
    out: dict[str, torch.Tensor] = {}
    dropped = 0
    for key, value in state_dict.items():
        if any(marker in key for marker in _DROP_SUBSTRINGS):
            dropped += 1
            continue
        for prefix, replacement in _TEXT_PREFIXES:
            if key.startswith(prefix):
                key = replacement + key[len(prefix) :]
                break
        out[key] = value
    if not out:
        raise ValueError(
            f"no text-backbone weights found in a checkpoint of {len(state_dict)} keys; expected one of "
            f"{[p for p, _ in _TEXT_PREFIXES]} or bare 'model.*' keys"
        )
    logger.info(f"[weights] mapped {len(out)} text-backbone keys ({dropped} vision/projector keys dropped)")
    return out


class ModelArgs:
    """Config + weight loading for Mistral-Medium-3.5-128B."""

    def __init__(self, mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=None, cache_hf=False):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = SPEC.max_seq_len if max_seq_len is None else max_seq_len
        self.cache_hf = cache_hf

        # Weights / config from HF_MODEL; fall back to the bundled config dir (config-only, e.g. a
        # dummy-weights or cache-populate run).
        hf_model = os.getenv("HF_MODEL") or DEFAULT_HF_MODEL
        self.model_path = hf_model
        self.weights_path = hf_model
        logger.info(
            f"Using Mistral-Medium-3.5 from: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.hf_config = getattr(cfg, "text_config", cfg)
        self.vocab_size = self.hf_config.vocab_size
        self.n_layers = getattr(self.hf_config, "num_hidden_layers", C.NUM_LAYERS)
        self.head_dim = getattr(
            self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads
        )
        self.rope_parameters = dict(getattr(self.hf_config, "rope_parameters", None) or {})

        self.max_prefill_chunk_size = SPEC.chunk_size
        self.model_name = C.MODEL_NAME
        self.max_context_len = self.max_seq_len

        if self.dummy_weights:
            self.tokenizer = None
        else:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True)
            except Exception as e:  # a config-only dir has no tokenizer
                logger.warning(f"No tokenizer at {self.weights_path} ({e}); tokenizer disabled")
                self.tokenizer = None
        self.processor = None

    def encode_prompt(self, prompt_text, instruct=False, system_prompt_text=None):
        """Encode a prompt through the HF tokenizer's chat template."""
        chat = []
        if isinstance(prompt_text, str):
            if system_prompt_text:
                chat.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                chat.append({"role": "user", "content": prompt_text})
            return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
        return self.tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True, num_layers=None):
        """Load the Mistral-Medium-3.5 state dict, dequantized and (by default) Meta-swizzled.

        Steps, in order — each is separately testable:
          1. read the safetensors shards (:meth:`_load_safetensors`);
          2. dequantize per-tensor fp8 to bf16 (:mod:`.fp8_dequant`);
          3. strip the ``Mistral3ForConditionalGeneration`` wrapper prefix, drop the vision keys;
          4. unless disabled, convert the q/k projections to Meta format for the on-device RoPE.

        ``convert_to_meta_format=False`` is what the golden generator uses: the reference is
        HF-convention, and swizzling there would rotate the golden's q/k instead of the device's.
        ``num_layers`` truncates to the first N layers for a reduced-depth run.
        """
        if dummy_weights:
            return {}
        state_dict = ModelArgs._load_safetensors(weights_path)
        from .fp8_dequant import dequantize_state_dict

        state_dict = dequantize_state_dict(state_dict)
        state_dict = map_checkpoint_keys(state_dict)
        if num_layers is not None:
            state_dict = {k: v for k, v in state_dict.items() if ModelArgs._within_layers(k, num_layers)}
        if convert_to_meta_format:
            head_dim = ModelArgs._head_dim_of(weights_path)
            logger.info(f"Converting q/k projections HF -> Meta format for the on-device RoPE (head_dim={head_dim})")
            state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
        return state_dict

    @staticmethod
    def _within_layers(key: str, num_layers: int) -> bool:
        marker = "model.layers."
        if marker not in key:
            return True
        return int(key.split(marker, 1)[1].split(".", 1)[0]) < num_layers

    @staticmethod
    def _head_dim_of(weights_path) -> int:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(weights_path), trust_remote_code=True)
        text = getattr(cfg, "text_config", cfg)
        return getattr(text, "head_dim", text.hidden_size // text.num_attention_heads)

    @staticmethod
    def _load_safetensors(weights_path):
        """Read every safetensors shard into one dict, shard by shard.

        Deliberately NOT HF ``from_pretrained``: the donor needs it because transformers owns the
        MXFP4 dequantization, whereas per-tensor fp8 is dequantized here. Reading the shards directly
        keeps the peak host footprint at one shard plus the accumulated bf16 result, instead of
        weights plus a live HF module, and — the reason that matters at 128 B — it keeps the fp8
        intermediate out of memory entirely.

        Falls back to a torch ``.pt`` (``reference_weights.pt``) when a directory has no shards,
        which is how the golden generator's own weight dump is reloaded for a device run.
        """
        path = Path(weights_path)
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            pt = path / "reference_weights.pt"
            if pt.is_file():
                logger.info(f"[weights] no safetensors in {path}; loading the torch dump {pt.name}")
                return torch.load(pt, weights_only=True)
            raise FileNotFoundError(f"no *.safetensors shards (and no reference_weights.pt) in {path}")

        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            logger.info(f"[weights] {len(shards)} shards, {len(weight_map)} tensors (index present)")

        from safetensors.torch import load_file

        state_dict: dict[str, torch.Tensor] = {}
        for shard in tqdm(shards, desc="Reading safetensors shards"):
            state_dict.update(load_file(str(shard)))
            gc.collect()
        logger.info(f"[weights] read {len(state_dict)} tensors from {len(shards)} shard(s)")
        return state_dict

    def weight_cache_path(self, dtype):
        """Weight-cache dir for this model + mesh. Mirrors the donor's layout so a cache-populate run
        and a serving run agree on the filenames."""
        cache_dir = os.getenv("TT_CACHE_PATH")
        cache_dir = Path(cache_dir) if cache_dir else Path(self.model_path)
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[dtype]
        cache_path = cache_dir / f"tensor_cache_{dtype_str}_{tuple(self.mesh_device.shape)}"
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Weight cache directory: {cache_path}")
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
