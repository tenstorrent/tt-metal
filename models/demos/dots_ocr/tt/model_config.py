# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from loguru import logger

from models.demos.dots_ocr.reference.hf_utils import DOTS_OCR_DEFAULT_HF_MODEL_ID
from models.tt_transformers.tt.model_config import ModelArgs, PrecisionSetting, TensorGroup


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
        # Parent ``ModelArgs`` reads ``HF_MODEL`` from the environment to set ``CKPT_DIR``.
        # Seed it when unset so Dots works without exporting HF_MODEL (prefer HF config path).
        if os.getenv("HF_MODEL") is None:
            if hf_config is not None:
                name_or_path = getattr(hf_config, "_name_or_path", None)
                os.environ["HF_MODEL"] = name_or_path if name_or_path else DOTS_OCR_DEFAULT_HF_MODEL_ID
            else:
                os.environ["HF_MODEL"] = DOTS_OCR_DEFAULT_HF_MODEL_ID

        # Dots OCR TT stacks always load real checkpoint tensors (filtered loaders in ``tt/load.py``).
        kwargs["dummy_weights"] = False
        super().__init__(*args, **kwargs)

        # `ModelArgs.__init__` defines `self.trust_remote_code_hf` but does not accept it
        # as an init kwarg in this repo version. Set it post-init so `_set_hf_params()`
        # can load HF configs that require remote code (e.g. dots.mocr).
        self.trust_remote_code_hf = True
        self.use_hf_rope = True

        # Dots OCR text quality is very sensitive to MLP weight precision.
        # Force BF16 for FFN weights (w1/w2/w3) to improve decode PCC and avoid degenerate text.
        dec_opt = getattr(self, "decoders_optimizations", None)
        if dec_opt is not None and hasattr(dec_opt, "decoder_optimizations"):
            for _dec_id, conf in dec_opt.decoder_optimizations.items():
                # Keep activations handled by callers; just force the weight tensors.
                conf.tensor_dtype_settings[TensorGroup.FF1_FF3] = PrecisionSetting.BF16
                conf.tensor_dtype_settings[TensorGroup.FF2] = PrecisionSetting.BF16
            logger.debug("DotsModelArgs: forced BF16 for FFN weights (FF1_FF3/FF2) for OCR correctness")

        # Wormhole correctness guard:
        # TTNN warns that HiFi4 + fp32_dest_acc_en can reduce accuracy on Wormhole. Dots OCR is very
        # sensitive and can collapse into repetitive garbage when this triggers. Force HiFi3 for the
        # affected compute configs on Wormhole SKUs (N150/N300/T3K).
        import ttnn

        dev = str(getattr(self, "device_name", "") or "").lower()
        is_wormhole = (
            dev in ("n150", "n300", "t3k") or dev.startswith("n150") or dev.startswith("n300") or "wormhole" in dev
        )
        if is_wormhole and hasattr(ttnn, "WormholeComputeKernelConfig") and hasattr(ttnn, "MathFidelity"):
            hifi3 = getattr(ttnn.MathFidelity, "HiFi3", None)
            if hifi3 is not None:

                def _hifi3_replacement(cfg):
                    if cfg is None or not bool(getattr(cfg, "fp32_dest_acc_en", True)):
                        return None
                    return ttnn.WormholeComputeKernelConfig(
                        math_fidelity=hifi3,
                        math_approx_mode=bool(getattr(cfg, "math_approx_mode", False)),
                        fp32_dest_acc_en=True,
                        packer_l1_acc=bool(getattr(cfg, "packer_l1_acc", True)),
                        dst_full_sync_en=bool(getattr(cfg, "dst_full_sync_en", False)),
                    )

                # Assign by fixed attribute names (no dynamic setattr) — satisfies SAST / avoids
                # "unsanitized external input in code generation" false positives on loop+setattr.
                _r = _hifi3_replacement(getattr(self, "compute_kernel_config_hifi4", None))
                if _r is not None:
                    self.compute_kernel_config_hifi4 = _r
                _r = _hifi3_replacement(getattr(self, "compute_kernel_config_hifi4_fp32", None))
                if _r is not None:
                    self.compute_kernel_config_hifi4_fp32 = _r
                _r = _hifi3_replacement(getattr(self, "compute_kernel_config_sdpa", None))
                if _r is not None:
                    self.compute_kernel_config_sdpa = _r
                logger.debug("DotsModelArgs: forced HiFi3 for fp32 accumulation on Wormhole (correctness)")

        # Dots OCR decode correctness is very sensitive to LM head output quantization.
        # `tt_transformers/tt/lm_head.py` defaults logits dtype to BF8 unless `args.lm_head_dtype`
        # is provided, which can collapse text quality (low decode PCC / repetitive garbage).
        # Force BF16 logits for Dots OCR.
        import ttnn

        self.lm_head_dtype = ttnn.bfloat16

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
            logger.debug(f"DotsModelArgs: max_seq_len capped to {self.max_seq_len} via env")

        # Shrink LM head column chunks so host tilize stays within L1 CB limits on WH (N300/N150).
        lm_cap_s = os.getenv("DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE", "2048")
        if getattr(self, "max_columns_per_device_lm_head", None) is not None and lm_cap_s.strip() != "":
            lm_cap = int(lm_cap_s)
            if lm_cap > 0:
                old_lm = self.max_columns_per_device_lm_head
                self.max_columns_per_device_lm_head = min(old_lm, lm_cap)
                if self.max_columns_per_device_lm_head < old_lm:
                    logger.debug(
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

        logger.debug(
            f"DotsModelArgs: dim={self.dim} layers={self.n_layers} heads={self.n_heads} "
            f"kv_heads={self.n_kv_heads} cluster_shape={self.cluster_shape} "
            f"max_seq_len={self.max_seq_len} device_name={self.device_name}"
        )

    def weight_cache_path(self, dtype):
        """
        Override base cache path to avoid correctness bugs from reusing tensorbins
        across incompatible Dots OCR configurations.

        TT weight tensorbins depend on how we convert HF weights (notably Q/K permute).
        If we reuse the same `tensor_cache_*` directory, toggling `text_qkv_permute`
        will silently keep using the old cached tensors and PCC won't change.
        """
        base = super().weight_cache_path(dtype)

        parts: list[str] = []
        qkvp = getattr(self, "dots_text_qkv_permute", None)
        if qkvp is not None:
            parts.append(f"qkvperm{int(bool(qkvp))}")
        host_rope = getattr(self, "dots_use_host_rope", None)
        if host_rope is not None:
            parts.append("hostrope" if bool(host_rope) else "devrope")

        env_tag = os.getenv("DOTS_CACHE_TAG", "").strip()
        if env_tag:
            parts.append(env_tag)

        if not parts:
            return base
        return base.parent / f"{base.name}__{'__'.join(parts)}"

    def _set_hf_params(self, checkpoint_dir):
        """
        Load HF config from the real checkpoint (`HF_MODEL` / `CKPT_DIR`).

        Parent `ModelArgs._set_hf_params` may use `LOCAL_HF_PARAMS` for some models; Dots always loads
        from ``self.CKPT_DIR`` (real checkpoint).
        """
        from transformers import AutoConfig

        from models.demos.dots_ocr.reference.flash_attention_shim import install as _install_flash_attn_shim

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
