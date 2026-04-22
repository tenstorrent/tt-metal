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

    Supported topologies
    ---------------------
    ``dots.mocr`` is GQA with ``num_attention_heads=12`` and
    ``num_key_value_heads=2``. Tensor parallelism shards heads along
    ``cluster_shape[1]`` and the base ``ModelArgs`` asserts
    ``n_kv_heads % cluster_shape[1] == 0``, so the TP degree must divide
    ``gcd(12, 2) = 2``.

    * ``MESH_DEVICE=N150`` (mesh ``1x1``, TP=1) — fully supported.
    * ``MESH_DEVICE=N300`` (mesh ``1x2``, TP=2) — fully supported.
    * ``MESH_DEVICE=T3K`` — physical **8-device** Wormhole LLMBox (``1×8``). ``open_mesh_device()``
      can open the full mesh then ``create_submesh`` to logical ``1×1`` or ``1×2`` (TP≤2 per
      ``num_key_value_heads=2``); see ``DOTS_T3K_OPEN_FULL_MESH`` / ``DOTS_T3K_TP``.

    Use ``models.demos.dots_ocr.tt.mesh.open_mesh_device()`` / ``close_dots_mesh_device()``
    on T3K when the full 8-device path is used (see ``DOTS_T3K_OPEN_FULL_MESH``).

    Optional context cap for DRAM-constrained runs: ``export DOTS_MAX_SEQ_LEN=8192``
    (legacy ``DOTS_MAX_SEQ_LEN_WH_LB`` is still honored for back-compat).

    LM head uploads use ``ttnn.from_torch`` → tilize; the default tt_transformers column budget
    (``668 * num_lm_head_cores``) can create very wide shards for large vocabs (e.g. dots.mocr
    ~152k) and exceed Wormhole L1 circular-buffer limits. Cap with
    ``DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE`` (default 2048 when unset).
    """

    def __init__(self, *args, hf_config=None, **kwargs):
        if os.getenv("HF_MODEL") is None:
            # Keep legacy behavior (HF_MODEL drives shared tt_transformers loading),
            # but don't hard-require it for call sites that already provide HF-derived weights/configs.
            if hf_config is not None:
                os.environ["HF_MODEL"] = getattr(hf_config, "_name_or_path", None) or "rednote-hilab/dots.mocr"
        # Dots OCR TT stacks always load real checkpoint tensors (filtered loaders in ``tt/load.py``).
        kwargs["dummy_weights"] = False
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

        # Canonical env var is ``DOTS_MAX_SEQ_LEN``; honor the legacy
        # ``DOTS_MAX_SEQ_LEN_WH_LB`` for back-compat with older scripts.
        cap = os.getenv("DOTS_MAX_SEQ_LEN") or os.getenv("DOTS_MAX_SEQ_LEN_WH_LB")
        if cap:
            self.max_seq_len = min(self.max_seq_len, int(cap))
            logger.info(f"DotsModelArgs: max_seq_len capped to {self.max_seq_len} via env")

        # Shrink LM head column chunks so host tilize stays within L1 CB limits on WH (N300/N150).
        lm_cap_s = os.getenv("DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE", "2048")
        if getattr(self, "max_columns_per_device_lm_head", None) is not None and lm_cap_s.strip() != "":
            lm_cap = int(lm_cap_s)
            if lm_cap > 0:
                old_lm = self.max_columns_per_device_lm_head
                self.max_columns_per_device_lm_head = min(old_lm, lm_cap)
                if self.max_columns_per_device_lm_head < old_lm:
                    logger.info(
                        f"DotsModelArgs: max_columns_per_device_lm_head {old_lm} -> "
                        f"{self.max_columns_per_device_lm_head} (LM head tilize / L1; "
                        f"override DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE)"
                    )

        # Belt-and-braces divisibility check so we emit a dots-specific diagnostic
        # instead of a bare AssertionError from the parent. The parent also
        # checks this, so we don't repeat the assert; we just print a helpful
        # hint if the user misreads the "T3K auto-clamp" mechanism.
        tp_cols = self.cluster_shape[1] if self.cluster_shape else 1
        if self.n_kv_heads and tp_cols and self.n_kv_heads % tp_cols != 0:
            logger.error(
                f"DotsModelArgs: cluster_shape[1]={tp_cols} does not divide n_kv_heads={self.n_kv_heads}. "
                f"Open the mesh via `models.demos.dots_ocr.tt.mesh.open_mesh_device()` so T3K / large "
                f"meshes are automatically clamped to a 1x2 submesh."
            )

        logger.info(
            f"DotsModelArgs: dim={self.dim} layers={self.n_layers} heads={self.n_heads} "
            f"kv_heads={self.n_kv_heads} cluster_shape={self.cluster_shape} "
            f"max_seq_len={self.max_seq_len} device_name={self.device_name}"
        )

    def _set_hf_params(self, checkpoint_dir):
        """
        Load HF config from the real checkpoint (`HF_MODEL` / `CKPT_DIR`).

        Parent `ModelArgs._set_hf_params` may use `LOCAL_HF_PARAMS` for some models; Dots always loads
        from ``self.CKPT_DIR`` (real checkpoint).
        """
        from transformers import AutoConfig

        from models.demos.dots_ocr.reference._flash_attn_shim import install as _install_flash_attn_shim

        _install_flash_attn_shim()

        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            text_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return text_config

        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            vision_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return vision_config

        # Dots.mocr ships custom modelling code, so always pass trust_remote_code=True here.
        # NB: ``ModelArgs.__init__`` sets ``self.trust_remote_code_hf = False`` before calling
        # ``_set_hf_params`` and only post-init code can flip it — so we force True locally
        # instead of relying on the instance attribute.
        self.trust_remote_code_hf = True
        # Eager attention in config when the installed ``transformers`` supports the kwarg.
        _cfg_kw = dict(
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
        try:
            self.hf_config = AutoConfig.from_pretrained(
                self.CKPT_DIR,
                attn_implementation="eager",
                **_cfg_kw,
            )
        except TypeError:
            self.hf_config = AutoConfig.from_pretrained(self.CKPT_DIR, **_cfg_kw)
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
        Dots OCR state_dict prefix helper for ``vision_tower`` / text submodules.

        Text decoder keys follow the same Meta-style layout as ``ModelArgs`` (``layers.{i}.attention.*``,
        ``layers.{i}.feed_forward.*``) after ``convert_*_hf_to_meta*``; vision keys live under
        ``vision_tower.*``. Omitting the submodule name (e.g. ``attention``) breaks Attention/MLP
        lookups and yields invalid keys such as ``layers.0..wq.weight``.
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

        # Match ``ModelArgs.get_state_dict_prefix`` text branch (non-Llama-vision): Meta layer paths.
        text_module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",
        }
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        return layer_prefix + text_module_map[module_name]

    def load_real_state_dict(self, *, qkv_permute: bool = False) -> dict:
        """
        Load real weights (text + vision) from the Dots HF checkpoint via filtered safetensors loads.

        Does not instantiate the full ``DotsOCRForCausalLM`` on the host (avoids OOM). Raises if
        the checkpoint is missing or keys cannot be mapped.
        """
        from models.demos.dots_ocr.tt.load import load_dots_full_state_dict

        return load_dots_full_state_dict(
            self.CKPT_DIR,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            qkv_permute=qkv_permute,
        )

    def load_state_dict(self):
        """
        Same as :meth:`load_real_state_dict` with default Q/K layout (no extra HF permute).

        Prefer :meth:`load_real_state_dict` with an explicit ``qkv_permute`` to match the text stack.
        """
        return self.load_real_state_dict(qkv_permute=False)
