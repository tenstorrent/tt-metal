"""Handler wrapper around LoRA service scale application."""

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log
from loguru import logger


def apply_scale_to_adapter(self, adapter_name: str, scale: float) -> int:
    """Apply scale to registered targets for one adapter."""
    self._ensure_lora_registry()
    self._lora_service.set_hooks(
        warn_hook=logger.warning,
        debug_hook=lambda message: debug_log(message, mode=DEBUG_MODEL_LOADING, prefix="lora"),
    )
    modified = self._lora_service.apply_scale(adapter_name=adapter_name, scale=scale)
    sync_state = getattr(self, "_sync_lora_state_from_service", None)
    if callable(sync_state):
        sync_state()
    else:
        self._ensure_lora_registry()
    return modified
