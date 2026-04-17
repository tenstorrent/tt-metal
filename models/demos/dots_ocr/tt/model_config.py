# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from loguru import logger

from models.tt_transformers.tt.model_config import ModelArgs


class DotsModelArgs(ModelArgs):
    """
    ModelArgs specialization for HF `rednote-hilab/dots.mocr`.

    Uses the same text stack as other HF causal LMs in tt_transformers (RMSNorm + RoPE + SwiGLU),
    with HF-style RoPE (`use_hf_rope=True`).

    **Wormhole LB (single chip):** set `MESH_DEVICE` to `N150` or `N300` and use a 1×1 mesh.
    Optionally cap context for DRAM: `export DOTS_MAX_SEQ_LEN_WH_LB=8192`
    """

    def __init__(self, *args, hf_config=None, **kwargs):
        if os.getenv("HF_MODEL") is None:
            if hf_config is None:
                raise ValueError("DotsModelArgs: set HF_MODEL or pass hf_config so HF_MODEL can be inferred.")
            os.environ["HF_MODEL"] = getattr(hf_config, "_name_or_path", None) or "rednote-hilab/dots.mocr"
        super().__init__(*args, **kwargs)

        # `ModelArgs.__init__` defines `self.trust_remote_code_hf` but does not accept it
        # as an init kwarg in this repo version. Set it post-init so `_set_hf_params()`
        # can load HF configs that require remote code (e.g. dots.mocr).
        self.trust_remote_code_hf = True
        self.use_hf_rope = True

        # Seed an instance-level ``LOCAL_HF_PARAMS`` entry so the parent
        # ``ModelArgs.load_state_dict()``'s ``self.LOCAL_HF_PARAMS[self.model_name]`` access
        # finds a config source for the Dots checkpoint (which is not listed in the class-level
        # ``LOCAL_HF_PARAMS``). This keeps the shared ``tt_transformers/tt/model_config.py``
        # untouched.
        self.LOCAL_HF_PARAMS = {**type(self).LOCAL_HF_PARAMS, self.model_name: self.CKPT_DIR}

        cap = os.getenv("DOTS_MAX_SEQ_LEN_WH_LB")
        if cap:
            self.max_seq_len = min(self.max_seq_len, int(cap))
            logger.info(f"DotsModelArgs: DOTS_MAX_SEQ_LEN_WH_LB={cap} -> max_seq_len={self.max_seq_len}")

        logger.info(
            f"DotsModelArgs (WH LB friendly): dim={self.dim} layers={self.n_layers} heads={self.n_heads} "
            f"kv_heads={self.n_kv_heads} max_seq_len={self.max_seq_len} device_name={self.device_name}"
        )

    def _set_hf_params(self, checkpoint_dir):
        """
        Load HF config from the real checkpoint (`HF_MODEL` / `CKPT_DIR`).

        Parent `ModelArgs._set_hf_params` uses `LOCAL_HF_PARAMS[model_name]` when `dummy_weights=True`,
        which does not include `dots.mocr`. We always load from `self.CKPT_DIR`.
        """
        from transformers import AutoConfig

        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            text_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return text_config

        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            vision_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return vision_config

        if self.dummy_weights:
            logger.info(f"DotsModelArgs: dummy_weights=True, loading HF config from CKPT_DIR={self.CKPT_DIR}")
        # Dots.mocr ships custom modelling code, so always pass trust_remote_code=True here.
        # NB: ``ModelArgs.__init__`` sets ``self.trust_remote_code_hf = False`` before calling
        # ``_set_hf_params`` and only post-init code can flip it — so we force True locally
        # instead of relying on the instance attribute.
        self.trust_remote_code_hf = True
        self.hf_config = AutoConfig.from_pretrained(
            self.CKPT_DIR,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
        config = self.hf_config.to_dict()

        if "text_config" in config or "vision_config" in config:
            merged_text_config = merge_text_config(config)
            self._set_params_from_dict(merged_text_config)

            if "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
                self._set_vision_params(config["vision_config"])
            else:
                if "vision_config" in config:
                    merged_vision_config = merge_vision_config(config)
                    self._set_vision_params({"vision_config": merged_vision_config})

            self.is_multimodal = "vision_config" in config or self.is_vision()
        else:
            self._set_params_from_dict(config)

        if "llama" in self.model_name.lower():
            if "3.2-11B" in checkpoint_dir:
                logger.warning(f"-Vision is removed from model_name {self.model_name}")
                self.model_name = "Llama-3.2-11B" + ("-Instruct" if self.instruct else "")
            elif "3.1-70B" in checkpoint_dir:
                self.is_70b = True
            elif "3.2-90B" in checkpoint_dir:
                logger.warning(f"-Vision is removed from model_name {self.model_name}")
                self.model_name = "Llama-3.2-90B" + ("-Instruct" if self.instruct else "")
                self.is_90b = True

    def get_hf_model_cls(self):
        """
        Dots OCR (``rednote-hilab/dots.mocr`` -> ``DotsOCRForCausalLM``) is a multimodal model
        that registers under ``AutoModelForCausalLM``, not ``AutoModelForVision2Seq`` /
        ``AutoModelForImageTextToText``. The parent ``get_hf_model_cls`` walks those two
        Vision2Seq registries first and raises ``ValueError`` for Dots, so override here.

        This also avoids relying on ``AutoModelForCausalLM._model_mapping`` lookups, which can
        miss ``trust_remote_code`` config classes loaded via revision-specific
        ``transformers_modules.*`` paths.
        """
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def get_state_dict_prefix(self, module_name: str, layer_num=None, is_vision: bool = False):
        """
        Dots OCR state_dict prefix helper — mirrors ``qwen25_vl.VisionModelArgs.get_state_dict_prefix``.

        Text keys use ``layers.{i}.`` under ``model.``; vision keys live under ``vision_tower.``
        (``vision_tower.blocks.{i}.``, ``vision_tower.patch_embed.``, ``vision_tower.merger.``,
        ``vision_tower.norm.``). Callers that only need a layer prefix can pass
        ``module_name=""``.
        """
        if is_vision:
            if module_name in ("VisionTransformer", ""):
                base = "vision_tower."
            elif module_name == "VisionBlock":
                base = "vision_tower.blocks."
            elif module_name == "PatchEmbed":
                return "vision_tower.patch_embed."
            elif module_name == "PatchMerger":
                return "vision_tower.merger."
            elif module_name == "Norm":
                return "vision_tower.norm."
            else:
                base = "vision_tower."
            if layer_num is not None and module_name in ("VisionBlock", "VisionTransformer", ""):
                return f"{base}{layer_num}." if module_name == "VisionBlock" else f"{base}blocks.{layer_num}."
            return base
        return f"layers.{layer_num}." if layer_num is not None else ""

    def load_real_state_dict(self) -> dict:
        """
        Load real weights (text + vision) from the Dots HF checkpoint.

        Wraps :func:`models.demos.dots_ocr.tt.load.load_dots_full_state_dict` so callers don't
        need to plumb ``head_dim`` / ``n_heads`` / ``n_kv_heads`` themselves. Raises on
        failure — callers that want to fall back to dummy weights should catch it.
        """
        from models.demos.dots_ocr.tt.load import load_dots_full_state_dict

        return load_dots_full_state_dict(
            self.CKPT_DIR,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
        )
