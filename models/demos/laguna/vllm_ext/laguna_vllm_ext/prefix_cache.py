# SPDX-License-Identifier: Apache-2.0
"""Capability-gated TT sliding-window prefix-cache compatibility.

The public TT plugin commit used by Laguna supports ordinary prefix caching,
but its platform hook blanket-disables it when ``get_sliding_window()`` is not
``None``. Laguna's qualified p150x2 profile supports the combination under its
canonical 8192-token admission and trace-stable runtime-offset policy. This
module wraps the hook without modifying the installed plugin, restoring the
requested setting only when both prefix-cache capabilities are explicitly true.
Other profiles retain the public plugin's fail-closed behavior.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

PATCH_MARKER = "_laguna_sliding_window_prefix_cache_patch"
_BASE_CAPABILITY = "supports_prefix_caching"
_SLIDING_WINDOW_CAPABILITY = "supports_prefix_caching_with_sliding_window"


def _resolve_model_class(model_config: Any) -> type:
    """Resolve the TT-prefixed class after the public platform hook registers it."""
    from vllm.model_executor.model_loader.utils import get_model_architecture

    model_class, _ = get_model_architecture(model_config)
    return model_class


def _supports_sliding_window_prefix_cache(model_config: Any) -> bool:
    model_class = _resolve_model_class(model_config)
    capabilities = getattr(model_class, "model_capabilities", None)
    return bool(
        isinstance(capabilities, dict)
        and capabilities.get(_BASE_CAPABILITY, False)
        and capabilities.get(_SLIDING_WINDOW_CAPABILITY, False)
    )


def _patch_platform(platform_class: type) -> bool:
    """Wrap a TT platform class once; return whether this call installed it."""
    if platform_class.__dict__.get(PATCH_MARKER, False):
        return False

    original_method = platform_class.check_and_update_config
    original_function = getattr(original_method, "__func__", None)
    if original_function is None:
        raise TypeError("TTPlatform.check_and_update_config is not a classmethod")

    def check_and_update_config(cls: type, vllm_config: Any) -> None:
        cache_config = vllm_config.cache_config
        prefix_cache_was_requested = bool(cache_config.enable_prefix_caching)

        original_function(cls, vllm_config)

        # The pinned public plugin leaves the flag enabled in every case except
        # an unsupported model or a model with a sliding window. Resolve the
        # model only on the disabled candidate path, after the original hook has
        # registered and TT-prefixed its architecture.
        restored_by_capability = False
        if prefix_cache_was_requested and not cache_config.enable_prefix_caching:
            sliding_window = vllm_config.model_config.get_sliding_window()
            if sliding_window is not None and _supports_sliding_window_prefix_cache(
                vllm_config.model_config
            ):
                cache_config.enable_prefix_caching = True
                restored_by_capability = True

        # The public hook may have already logged an intermediate blanket-disable decision. Always
        # report the final engine state so the launch log remains authoritative. A rejected request
        # or a capability-gated reversal is elevated because either one deserves operator attention.
        final_enabled = bool(cache_config.enable_prefix_caching)
        log_level = (
            logging.WARNING
            if restored_by_capability or prefix_cache_was_requested != final_enabled
            else logging.INFO
        )
        logger.log(
            log_level,
            "Laguna prefix-cache final state: requested=%s enabled=%s restored_by_capability=%s",
            prefix_cache_was_requested,
            final_enabled,
            restored_by_capability,
        )

        # The public plugin owns the final vLLM configuration, so enforce the
        # canonical-cache scheduler envelope only after its hook has run. This
        # also protects direct `vllm serve` invocations that bypass our shell
        # launcher while setting TT_LAGUNA_PREFIX_CACHE=1.
        from .prefix_cache_quantum import validate_prefix_cache_vllm_config

        validate_prefix_cache_vllm_config(vllm_config)

    check_and_update_config.__name__ = original_function.__name__
    check_and_update_config.__doc__ = original_function.__doc__
    setattr(
        platform_class, "check_and_update_config", classmethod(check_and_update_config)
    )
    setattr(platform_class, PATCH_MARKER, True)
    return True


def install_sliding_window_prefix_cache_patch() -> bool:
    """Install the TT platform wrapper once in the current process."""
    from vllm_tt_plugin.platform import TTPlatform

    installed = _patch_platform(TTPlatform)
    if installed:
        logger.info("Installed capability-gated TT sliding-window prefix-cache support")
    return installed


def sliding_window_prefix_cache_patch_is_installed() -> bool:
    """Return whether the runtime TT platform carries this extension's wrapper."""
    from vllm_tt_plugin.platform import TTPlatform

    return bool(TTPlatform.__dict__.get(PATCH_MARKER, False))
