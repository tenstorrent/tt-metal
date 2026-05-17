"""Handler wrapper around the LoRA service registry build."""

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log
from loguru import logger


def rebuild_lora_registry(self, lora_path: str | None = None) -> tuple[int, list[str]]:
    """Build explicit adapter->target mapping used for deterministic scaling."""
    self._ensure_lora_registry()
    total_targets, adapters = self._lora_service.rebuild_registry(lora_path=lora_path)
    sync_state = getattr(self, "_sync_lora_state_from_service", None)
    if callable(sync_state):
        sync_state()
    else:
        self._ensure_lora_registry()

    if not adapters:
        logger.warning("No adapter names discovered from decoder; LoRA registry will be empty.")
        debug_log(
            "No adapter names discovered; skipping adapter target registration.",
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )

    return total_targets, adapters
