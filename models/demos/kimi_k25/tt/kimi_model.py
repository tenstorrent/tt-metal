# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""kimi_model.py — Kimi K2.5 model adapter for tt-metal inference.

This module wires the Kimi K2.5 HuggingFace checkpoint to the DeepSeek V3
tt-metal runtime by:

  1. Validating the HF config against :class:`KimiK25Config` (catches silent
     architecture mismatches early — e.g. n_group, rms_norm_eps, experts).
  2. Injecting :class:`KimiLazyStateDict` as the weight source so that INT4
     expert weights are dequantized transparently on first access, and the
     ``language_model.model.*`` checkpoint prefix is stripped to match what
     DSV3 modules expect.
  3. Delegating all runtime logic (mesh setup, CCL, RoPE, KV cache, decode
     loop) to :class:`~models.demos.deepseek_v3.tt.generator.DeepseekGenerator`.

Usage::

    from models.demos.kimi_k25.tt.kimi_model import KimiGenerator, load_kimi_model

    # Context manager (recommended — ensures cleanup on exit)
    with load_kimi_model(
        model_path="/workspace/extra/Kimi-K2.5",
        mesh_device=mesh_device,
        cache_dir="/workspace/extra/kimi_cache",
    ) as gen:
        results = gen.generate(prompts=["Hello, Kimi!"])

    # Manual lifecycle
    gen = load_kimi_model(
        model_path="/workspace/extra/Kimi-K2.5",
        mesh_device=mesh_device,
    )
    results = gen.generate(prompts=["Hello!"])
    gen.cleanup_all()

Key differences from :class:`DeepseekGenerator`:

* **Weight loading** — Uses :class:`KimiLazyStateDict` instead of the default
  HF safetensors loader.  The state dict strips ``language_model.model.`` from
  checkpoint keys and auto-dequantizes INT4-packed expert weights.
* **Config validation** — Calls :func:`KimiK25Config.from_hf_config` which
  raises ``ValueError`` for any critical field mismatch (e.g. wrong number of
  experts or incorrect rms_norm_eps).
* **Cache path** — Defaults to ``generated/kimi_k25`` instead of
  ``generated/deepseek_v3`` to keep the converted TTNN tensors separate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.utils.weight_config import get_weight_config
from models.demos.kimi_k25.utils.config_adapter import KimiK25Config
from models.demos.kimi_k25.utils.weight_loader import KimiLazyStateDict

if TYPE_CHECKING:
    import ttnn

__all__ = ["KimiGenerator", "load_kimi_model"]

#: Default weight cache directory (relative path resolved from CWD at runtime).
_DEFAULT_CACHE_DIR = "generated/kimi_k25"


class KimiGenerator(DeepseekGenerator):
    """DeepseekGenerator subclass tailored for Kimi K2.5.

    Overrides two hooks:

    * ``__init__`` — validates the HF config via :class:`KimiK25Config` before
      delegating to the parent.  This catches architectural mismatches
      (wrong expert count, rms_norm_eps, etc.) at startup rather than
      producing silent numerical errors at runtime.

    * ``_prepare_weight_configs`` — injects :class:`KimiLazyStateDict` as the
      state-dict source so the parent's ``get_weight_config`` call receives
      Kimi-specific weight loading (INT4 dequant + prefix stripping) without
      any further changes to the DSV3 runtime.

    All other behaviour — mesh device setup, CCL, KV cache, RoPE, decode loop,
    context manager lifecycle — is inherited from :class:`DeepseekGenerator`.

    Args:
        hf_config:   Optional pre-loaded ``transformers.AutoConfig``.  If
                     *None*, the parent will load it from *model_path*.
        mesh_device: TTNN :class:`~ttnn.MeshDevice` to run on.
        model_path:  Path to the local Kimi K2.5 checkpoint directory
                     (must contain ``model.safetensors.index.json``).
        cache_dir:   Where to store converted TTNN tensors.  Defaults to
                     ``generated/kimi_k25`` relative to the working directory.
        **kwargs:    All remaining keyword arguments are forwarded verbatim to
                     :class:`DeepseekGenerator`.

    Raises:
        ValueError: If the HF config fails :class:`KimiK25Config` validation.
    """

    def __init__(
        self,
        hf_config=None,
        mesh_device: "ttnn.MeshDevice | None" = None,
        model_path: "str | Path | None" = None,
        cache_dir: "str | Path | None" = None,
        **kwargs,
    ) -> None:
        # ------------------------------------------------------------------
        # Step 1: Load HF config early (parent would do this, but we need it
        # here to run validation before any expensive initialisation starts).
        # ------------------------------------------------------------------
        if hf_config is None and model_path is not None:
            try:
                from transformers import AutoConfig

                hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
                logger.info(f"KimiGenerator: loaded HF config from {model_path}")
            except Exception as exc:
                logger.warning(
                    f"KimiGenerator: could not load AutoConfig from {model_path!r}: {exc}. "
                    "Proceeding without pre-validation — parent will attempt to load."
                )

        # ------------------------------------------------------------------
        # Step 2: Validate config against KimiK25Config reference values.
        # This raises ValueError on critical architectural mismatches.
        # ------------------------------------------------------------------
        if hf_config is not None:
            try:
                kimi_cfg = KimiK25Config.from_hf_config(hf_config)
                logger.info(f"KimiGenerator: config validated\n{kimi_cfg.summary()}")
            except ValueError as exc:
                raise ValueError(
                    f"Kimi K2.5 config validation failed — refusing to start " f"with a mismatched model:\n{exc}"
                ) from exc

        # ------------------------------------------------------------------
        # Step 3: Delegate to parent.  Parent's __init__ will:
        #   - set self.model_path, self.hf_config, self.ccl, self.rope_setup
        #   - call self._prepare_weight_configs(cache_dir)  <- we override this
        # ------------------------------------------------------------------
        super().__init__(
            hf_config=hf_config,
            mesh_device=mesh_device,
            model_path=str(model_path) if model_path is not None else None,
            cache_dir=cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Weight config hook — inject KimiLazyStateDict
    # ------------------------------------------------------------------

    def _prepare_weight_configs(self, cache_dir: "str | Path | None") -> None:
        """Override parent to use :class:`KimiLazyStateDict` as weight source.

        The parent's default implementation calls
        ``prepare_model_state_dict(model_path=...)`` which invokes the standard
        DSV3 HF loader (including the ``_strip_model_prefix`` that strips only
        ``model.``).  We replace that with :class:`KimiLazyStateDict` which:

        * strips ``language_model.model.`` (Kimi's full text-backbone prefix),
        * transparently dequantizes ``*.weight_packed`` tensors (I32 -> BF16),
        * caches dequantized results to avoid redundant computation.

        If ``random_weights=True`` (smoke tests), we fall through to the parent
        so that random-weight generation still works without real weights.
        """
        if self.random_weights:
            logger.info(
                "KimiGenerator: random_weights=True — using parent weight prep " "(KimiLazyStateDict not injected)"
            )
            super()._prepare_weight_configs(cache_dir)
            return

        weight_cache_path = Path(cache_dir) if cache_dir is not None else Path(_DEFAULT_CACHE_DIR)
        weight_cache_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"KimiGenerator: loading weights via KimiLazyStateDict " f"from {self.model_path!r}")
        kimi_state_dict = KimiLazyStateDict(self.model_path)

        self.model_weight_config = get_weight_config(
            ModuleClass=RowBatchedModel,
            hf_config=self.hf_config,
            state_dicts=(kimi_state_dict,),
            weight_cache_path=weight_cache_path,
            mesh_device=self.mesh_device,
            force_recalculate=self.force_recalculate,
            random_weights=False,  # already handled above
            model_path=self.model_path,
            single_layer=self.single_layer,
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_kimi_model(
    model_path: "str | Path",
    mesh_device: "ttnn.MeshDevice",
    cache_dir: "str | Path | None" = None,
    **kwargs,
) -> KimiGenerator:
    """Create a :class:`KimiGenerator` ready for inference.

    This is the primary entry point for Kimi K2.5 inference on Tenstorrent
    hardware.  The returned object is a context manager; use it with ``with``
    to guarantee cleanup:

    .. code-block:: python

        with load_kimi_model(
            model_path="/workspace/extra/Kimi-K2.5",
            mesh_device=mesh_device,
            cache_dir="/workspace/extra/kimi_cache",
        ) as gen:
            output = gen.generate(prompts=["Hello!"])

    Args:
        model_path:  Local path to the Kimi K2.5 checkpoint directory.
                     Must contain ``model.safetensors.index.json`` and the 64
                     shard files (``model-00001-of-000064.safetensors`` ...).
                     Environment variable ``KIMI_HF_MODEL`` is a conventional
                     way to set this path.
        mesh_device: TTNN mesh device (single Galaxy TG = 32-chip 4x8 mesh,
                     or smaller meshes for partial testing).
        cache_dir:   Directory for converted TTNN tensors.  Created if absent.
                     Reused across runs — first run converts (slow); subsequent
                     runs reload from cache (fast).  Defaults to
                     ``generated/kimi_k25`` in the current working directory.
        **kwargs:    Forwarded to :class:`KimiGenerator` (and ultimately to
                     :class:`~models.demos.deepseek_v3.tt.generator.DeepseekGenerator`).
                     Useful overrides include:

                     * ``random_weights=True`` — smoke test without real weights
                     * ``override_num_layers=4`` — run only N layers (fast test)
                     * ``tokenizer=<AutoTokenizer>`` — pre-loaded tokenizer

    Returns:
        A :class:`KimiGenerator` instance.  Call ``.cleanup_all()`` when done,
        or use as a context manager.
    """
    return KimiGenerator(
        model_path=model_path,
        mesh_device=mesh_device,
        cache_dir=cache_dir,
        **kwargs,
    )
