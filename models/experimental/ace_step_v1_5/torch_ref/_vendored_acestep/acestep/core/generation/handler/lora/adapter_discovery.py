"""Handler wrapper for adapter discovery."""


def collect_adapter_names(self) -> list[str]:
    """Best-effort adapter name discovery across PEFT runtime variants."""
    self._ensure_lora_registry()
    return self._lora_service.discover_adapter_names()
