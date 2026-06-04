"""Pure LoRA registry construction."""

from typing import Any

from .introspection import (
    get_peft_initial_scale,
    is_lora_like_module,
    is_peft_factor_set_scale_module,
    read_adapter_value,
)


def build_lora_registry(
    decoder: Any,
    adapter_names: list[str],
    lora_path: str | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Build explicit adapter->target mapping used for deterministic scaling."""
    registry: dict[str, dict[str, Any]] = {name: {"path": lora_path, "targets": []} for name in adapter_names}

    for module_name, module in decoder.named_modules():
        if not is_lora_like_module(module_name, module):
            continue

        if is_peft_factor_set_scale_module(module):
            for adapter in adapter_names:
                scaling = getattr(module, "scaling", None)
                current_scale = read_adapter_value(scaling, adapter)
                initial_scale = get_peft_initial_scale(module, adapter)
                base_factor = (
                    float(current_scale) / float(initial_scale)
                    if isinstance(current_scale, (int, float))
                    and isinstance(initial_scale, (int, float))
                    and initial_scale != 0
                    else None
                )
                registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "set_scale_factor",
                        "adapter": adapter,
                        "module_name": module_name,
                        "base_factor": base_factor,
                    }
                )
            continue

        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            for adapter in adapter_names:
                if adapter in module.scaling:
                    registry[adapter]["targets"].append(
                        {
                            "module": module,
                            "kind": "scaling_dict",
                            "adapter": adapter,
                            "module_name": module_name,
                            "base_scale": module.scaling[adapter],
                        }
                    )
            continue

        if hasattr(module, "set_scale"):
            for adapter in adapter_names:
                registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "set_scale_unknown",
                        "adapter": adapter,
                        "module_name": module_name,
                        "base_scale": read_adapter_value(getattr(module, "scaling", None), adapter),
                    }
                )
            continue

        if hasattr(module, "scale_layer") and len(adapter_names) == 1:
            adapter = adapter_names[0]
            base_scale = read_adapter_value(getattr(module, "scaling", None), adapter)
            registry[adapter]["targets"].append(
                {
                    "module": module,
                    "kind": "scale_layer",
                    "module_name": module_name,
                    "base_scale": float(base_scale) if isinstance(base_scale, (int, float)) else None,
                }
            )
            continue

        if hasattr(module, "scaling") and isinstance(module.scaling, (int, float)) and len(adapter_names) == 1:
            adapter = adapter_names[0]
            registry[adapter]["targets"].append(
                {
                    "module": module,
                    "kind": "scaling_scalar",
                    "module_name": module_name,
                    "base_scale": float(module.scaling),
                }
            )

    total_targets = sum(len(meta["targets"]) for meta in registry.values())
    return registry, total_targets
