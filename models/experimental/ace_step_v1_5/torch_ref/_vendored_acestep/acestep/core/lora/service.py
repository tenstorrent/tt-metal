"""Stateful LoRA facade that composes core pure functions."""

from collections.abc import Callable
from typing import Any

from .introspection import collect_adapter_names
from .registry import build_lora_registry
from .scaling import apply_scale_to_adapter


class LoraService:
    """Facade over adapter discovery, registry build, and scaling application.

    When no adapter names are discoverable, a synthetic ``default`` adapter is
    used internally and filtered to adapter-agnostic target kinds only.
    """

    def __init__(
        self,
        decoder: Any | None = None,
        warn_hook: Callable[[str], None] | None = None,
        debug_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.decoder = decoder
        self.warn_hook = warn_hook
        self.debug_hook = debug_hook
        self.registry: dict[str, dict[str, Any]] = {}
        self.scale_state: dict[tuple[int, str, str], float] = {}
        self.active_adapter: str | None = None
        self.last_scale_report: dict[str, Any] = {}
        self.synthetic_default_mode = False

    def bind_decoder(self, decoder: Any | None) -> None:
        self.decoder = decoder

    def set_hooks(
        self,
        warn_hook: Callable[[str], None] | None = None,
        debug_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.warn_hook = warn_hook
        self.debug_hook = debug_hook

    def discover_adapter_names(self) -> list[str]:
        if self.decoder is None:
            return []
        return [name for name in collect_adapter_names(self.decoder) if isinstance(name, str) and name]

    @staticmethod
    def _keep_adapter_agnostic_targets(registry: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Keep only adapter-agnostic targets when no adapter names are discoverable."""
        allowed_kinds = {"scale_layer", "scaling_scalar"}
        filtered: dict[str, dict[str, Any]] = {}
        for adapter_name, meta in registry.items():
            targets = [target for target in meta.get("targets", []) if target.get("kind") in allowed_kinds]
            if targets:
                filtered[adapter_name] = {
                    "path": meta.get("path"),
                    "targets": targets,
                }
        return filtered

    def rebuild_registry(self, lora_path: str | None = None) -> tuple[int, list[str]]:
        if self.decoder is None:
            self.registry = {}
            self.scale_state = {}
            self.active_adapter = None
            self.synthetic_default_mode = False
            return 0, []

        adapter_names = self.discover_adapter_names()
        synthetic_default = not adapter_names
        self.synthetic_default_mode = synthetic_default
        effective_names = adapter_names or ["default"]

        rebuilt_registry, _ = build_lora_registry(
            decoder=self.decoder,
            adapter_names=effective_names,
            lora_path=lora_path,
        )
        if synthetic_default:
            rebuilt_registry = self._keep_adapter_agnostic_targets(rebuilt_registry)

        self.registry = rebuilt_registry
        self.scale_state = {}
        total_targets = sum(len(meta.get("targets", [])) for meta in self.registry.values())
        if self.active_adapter not in self.registry:
            self.active_adapter = next(iter(self.registry.keys()), None)
        return total_targets, list(self.registry.keys())

    def ensure_active_adapter(self) -> str | None:
        if self.active_adapter is None and self.registry:
            self.active_adapter = next(iter(self.registry.keys()))
        return self.active_adapter

    def set_active_adapter(self, adapter_name: str) -> bool:
        if adapter_name not in self.registry:
            return False
        self.active_adapter = adapter_name
        return True

    def apply_scale(self, adapter_name: str, scale: float) -> int:
        modified, report = apply_scale_to_adapter(
            registry=self.registry,
            scale_state=self.scale_state,
            adapter_name=adapter_name,
            scale=scale,
            warn_hook=self.warn_hook,
            debug_hook=self.debug_hook,
        )
        self.last_scale_report = report
        return modified

    def registry_snapshot(self, max_targets_per_adapter: int = 20) -> dict[str, Any]:
        """Return debuggable registry view and synthetic fallback mode flag."""
        adapters: dict[str, Any] = {}
        for adapter_name, meta in self.registry.items():
            targets = meta.get("targets", [])
            entries = []
            for target in targets[:max_targets_per_adapter]:
                module = target.get("module")
                entries.append(
                    {
                        "kind": target.get("kind"),
                        "module_name": target.get("module_name"),
                        "adapter": target.get("adapter"),
                        "module_class": module.__class__.__name__ if module is not None else None,
                    }
                )
            adapters[adapter_name] = {
                "path": meta.get("path"),
                "target_count": len(targets),
                "targets": entries,
                "truncated": len(targets) > max_targets_per_adapter,
            }
        return {
            "active_adapter": self.active_adapter,
            "adapter_names": list(self.registry.keys()),
            "synthetic_default_mode": self.synthetic_default_mode,
            "adapters": adapters,
        }
