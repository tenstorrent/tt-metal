# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MoonViTModelArgs — central config object for the MoonViT vision tower.

Holds:
  - HF config (loaded once from moonshotai/Kimi-VL-A3B-Instruct).
  - Vision-tower hyperparameters derived from that config.
  - Mesh device and dtype.
  - reference_*() factories that return HF submodules on CPU, used by
    per-submodule PCC tests in `tests/moonvit/`.

Per-submodule tests instantiate this class, call `reference_*()` to get
a torch reference, then build the corresponding ttnn module with the
same weights for a PCC comparison.

This class is deliberately lighter than `models.demos.qwen25_vl.tt.model_config.VisionModelArgs`
(which inherits the full LLM ModelArgs). MoonViT is small enough to
warrant a focused config object that only carries vision-relevant
fields.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

# Default HF model id; override via DEEPSEEK_MOONVIT_HF_MODEL env var or constructor kwarg.
#
# K2.6 is the production target. Its MoonViT is the 3D (video-capable)
# variant, but for image inputs (T=1) it degenerates exactly to the 2D
# MoonViT this module implements. The full K2.6 checkpoint is ~600 GB;
# we load only the vision tower + projector from shards 63/64 via
# `_k26_loader.load_k26_vision_reference` (never materializing the LLM).
#
# The legacy A3B id ("moonshotai/Kimi-VL-A3B-Instruct") still works for
# anyone with that checkpoint cached — set DEEPSEEK_MOONVIT_HF_MODEL.
DEFAULT_HF_MODEL_ID = "moonshotai/Kimi-K2.6"


@dataclass
class MoonViTModelArgs:
    """Configuration object for the MoonViT vision tower.

    Hyperparameters mirror the authoritative HF config
    (`vision_config` section of moonshotai/Kimi-VL-A3B-Instruct/config.json):

        hidden_size       = 1152
        num_hidden_layers = 27
        num_attention_heads = 16
        intermediate_size = 4304
        patch_size        = 14
        init_pos_emb_height = 64
        init_pos_emb_width  = 64
        merge_kernel_size = [2, 2]
        layer_norm_eps    = 1e-5

    Derived:
        head_dim   = hidden_size // num_attention_heads = 72
        merge_dim  = hidden_size * prod(merge_kernel_size) = 4608
    """

    # Required at construction time.
    mesh_device: Any
    dtype: Any  # ttnn.bfloat16 etc. Kept as Any to avoid importing ttnn at module import.

    # Optional: override the HF model id when constructing.
    hf_model_id: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MOONVIT_HF_MODEL", DEFAULT_HF_MODEL_ID))
    trust_remote_code: bool = True

    # Authoritative architecture values from the config.json this plan was based on.
    # Asserted against the real HF config on load — change these only if MoonViT itself changes.
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    merge_kernel_size: tuple[int, int] = (2, 2)
    layer_norm_eps: float = 1e-5

    # Lazy-loaded.
    _hf_config: Any = field(default=None, init=False, repr=False)
    _hf_model: Any = field(default=None, init=False, repr=False)
    _hf_processor: Any = field(default=None, init=False, repr=False)
    _hf_tokenizer: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Derived properties

    @property
    def is_k26(self) -> bool:
        """True when loading a Kimi-K2.x checkpoint (3D MoonViT, vision-only load).

        K2.x can't go through HF's Auto* loaders: the `.` in the repo name
        breaks the trust_remote_code dynamic-module importer, and loading
        the full model would materialize the ~595 GB LLM. We route these
        through `_k26_loader` instead.
        """
        mid = self.hf_model_id.lower()
        return "k2.6" in mid or "k2.5" in mid or "kimi-k2" in mid

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def merge_dim(self) -> int:
        h, w = self.merge_kernel_size
        return self.hidden_size * h * w

    @property
    def text_hidden_size(self) -> int:
        """Hidden size of the LLM the projector outputs into.

        Pulled from the HF config's text_config when available. For
        Kimi-VL-A3B-Instruct this is 2048; for DeepSeek-V3/V4 this is
        7168. The plan ships v1 testing against Kimi-VL's LLM end-to-end
        so we use whatever HF reports.
        """
        cfg = self.hf_config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        raise AttributeError("Could not find text hidden_size on HF config")

    # ------------------------------------------------------------------
    # HF loading (lazy)

    @property
    def hf_config(self):
        if self._hf_config is None:
            if self.is_k26:
                self._hf_config = self._build_k26_config()
            else:
                from transformers import AutoConfig

                self._hf_config = AutoConfig.from_pretrained(self.hf_model_id, trust_remote_code=self.trust_remote_code)
            self._assert_hf_matches_plan()
        return self._hf_config

    def _build_k26_config(self):
        """Build a config view from K2.6's config.json (AutoConfig is unusable).

        K2.6's vision_config uses flat `vt_*` keys; we normalize them to the
        plan field names (hidden_size, num_hidden_layers, ...) so the rest of
        this class and the assert below are source-agnostic. The raw block is
        preserved under `vision_config._raw` for anything K2.6-specific.
        """
        import json
        from types import SimpleNamespace

        from models.demos.deepseek_v3.tt.moonvit._k26_loader import _find_snapshot_dir

        snapshot_dir = _find_snapshot_dir()
        with open(os.path.join(snapshot_dir, "config.json")) as f:
            cfg = json.load(f)
        raw = cfg["vision_config"]
        vision_config = SimpleNamespace(
            hidden_size=raw["vt_hidden_size"],
            num_hidden_layers=raw["vt_num_hidden_layers"],
            num_attention_heads=raw["vt_num_attention_heads"],
            intermediate_size=raw["vt_intermediate_size"],
            patch_size=raw["patch_size"],
            init_pos_emb_height=raw["init_pos_emb_height"],
            init_pos_emb_width=raw["init_pos_emb_width"],
            merge_kernel_size=tuple(raw["merge_kernel_size"]),
            mm_hidden_size=raw["mm_hidden_size"],
            _raw=raw,
        )
        text_config = SimpleNamespace(hidden_size=cfg["text_config"]["hidden_size"])
        return SimpleNamespace(
            vision_config=vision_config,
            text_config=text_config,
            media_placeholder_token_id=cfg["media_placeholder_token_id"],
            _snapshot_dir=snapshot_dir,
        )

    def _assert_hf_matches_plan(self) -> None:
        """Cross-check the HF config against the values we hard-coded.

        If MoonViT's released config changes, this is where we'll notice.
        """
        cfg = self._hf_config
        vc = getattr(cfg, "vision_config", None)
        if vc is None:
            raise RuntimeError(
                f"HF config for {self.hf_model_id} has no `vision_config`; "
                "make sure you're loading a multimodal Kimi-VL checkpoint."
            )
        expectations = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "patch_size": self.patch_size,
            "init_pos_emb_height": self.init_pos_emb_height,
            "init_pos_emb_width": self.init_pos_emb_width,
        }
        for name, expected in expectations.items():
            actual = getattr(vc, name, None)
            if actual is not None and actual != expected:
                raise RuntimeError(
                    f"HF vision_config.{name} = {actual} but plan expects {expected}. "
                    "MoonViT config has changed — update MoonViTModelArgs."
                )

    @property
    def hf_model(self):
        """Lazily load the HF model exposing `.vision_tower` and the projector.

        For K2.6 we build a vision-only stand-in (no LLM) from shards 63/64
        via `_k26_loader`; its `.mm_projector` attribute holds the projector.
        For A3B we load the full HF model via AutoModel (`.multi_modal_projector`).
        """
        if self._hf_model is None:
            if self.is_k26:
                from models.demos.deepseek_v3.tt.moonvit._k26_loader import load_k26_vision_reference

                # Ensure the config (and thus snapshot dir) is resolved first.
                snapshot_dir = getattr(self.hf_config, "_snapshot_dir", None)
                vision_tower, projector = load_k26_vision_reference(snapshot_dir=snapshot_dir)
                self._hf_model = _K26VisionModel(vision_tower=vision_tower, mm_projector=projector)
            else:
                from transformers import AutoModel

                self._hf_model = AutoModel.from_pretrained(
                    self.hf_model_id,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.bfloat16,
                )
                self._hf_model.eval()
        return self._hf_model

    @property
    def hf_processor(self):
        if self._hf_processor is None:
            if self.is_k26:
                from models.demos.deepseek_v3.tt.moonvit._k26_loader import load_k26_image_processor

                snapshot_dir = getattr(self.hf_config, "_snapshot_dir", None)
                self._hf_processor = load_k26_image_processor(snapshot_dir=snapshot_dir)
            else:
                from transformers import AutoProcessor

                self._hf_processor = AutoProcessor.from_pretrained(
                    self.hf_model_id, trust_remote_code=self.trust_remote_code
                )
        return self._hf_processor

    @property
    def hf_tokenizer(self):
        if self._hf_tokenizer is None:
            if self.is_k26:
                raise NotImplementedError(
                    "K2.6 tokenizer loading is not wired up yet (repo-name `.` "
                    "breaks AutoTokenizer's dynamic-module import). Not needed by "
                    "the synthetic-grid PCC tests."
                )
            from transformers import AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_id, trust_remote_code=self.trust_remote_code
            )
        return self._hf_tokenizer

    @property
    def media_placeholder_token_id(self) -> int:
        """The token id used to mark image positions in the prompt.

        Plan: v1 uses Kimi-VL's tokenizer so this resolves directly. When
        DeepSeek-V4 ships, this must be reconciled with V4's tokenizer.
        """
        cfg = self.hf_config
        if hasattr(cfg, "media_placeholder_token_id"):
            return cfg.media_placeholder_token_id
        raise AttributeError("HF config has no media_placeholder_token_id")

    # ------------------------------------------------------------------
    # reference_*() factories — implemented in a sibling module to keep
    # this file declarative. See moonvit/_references.py (to be added in
    # task #4) for the actual extraction of submodules from hf_model.

    def reference_layernorm(self, layer_num: int = 0, which: str = "norm0"):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_layernorm

        return reference_layernorm(self, layer_num=layer_num, which=which)

    def reference_mlp(self, layer_num: int = 0):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_mlp

        return reference_mlp(self, layer_num=layer_num)

    def reference_attention(self, layer_num: int = 0):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_attention

        return reference_attention(self, layer_num=layer_num)

    def reference_block(self, layer_num: int = 0):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_block

        return reference_block(self, layer_num=layer_num)

    def reference_patch_embed(self):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_patch_embed

        return reference_patch_embed(self)

    def reference_pos_emb(self):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_pos_emb

        return reference_pos_emb(self)

    def reference_patch_merger(self):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_patch_merger

        return reference_patch_merger(self)

    def reference_projector(self):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_projector

        return reference_projector(self)

    def reference_vision_tower(self):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_vision_tower

        return reference_vision_tower(self)

    def reference_rope_2d(self):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_rope_2d

        return reference_rope_2d(self)

    def reference_final_layernorm(self):
        from models.demos.deepseek_v3.tt.moonvit._references import reference_final_layernorm

        return reference_final_layernorm(self)


class _K26VisionModel:
    """Vision-only stand-in for HF `KimiK25ForConditionalGeneration`.

    Exposes just the attributes the reference extractors need
    (`.vision_tower`, `.mm_projector`) so `_references.py` can pull
    submodules without the 595 GB LLM ever being constructed.
    """

    def __init__(self, vision_tower, mm_projector):
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector

    def eval(self):
        self.vision_tower.eval()
        self.mm_projector.eval()
        return self
