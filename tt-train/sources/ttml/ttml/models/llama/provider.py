# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HF ``model_type: llama`` checkpoints (Llama 3.x) on ttml's Python ``Llama``, with LoRA adapters.

ttml's Llama fuses q, k and v into one ``qkv_linear``, and gate and up into one ``w_gate_up``. A fused layer
has one LoRA adapter, so LoRA targets all of its projections or none of them; a partly targeted layer is
rejected rather than silently widened.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np

from ..peft import check_peft_keys, lora_to_peft, peft_adapter_config, peft_to_lora, with_peft_prefix
from ..provider import AdapterState, UnsupportedTrainingConfig, model_dir

if TYPE_CHECKING:
    from ttml.unified_trainer.configs import LoraConfig, SFTTrainerConfig

SUPPORTED_DTYPES = ("bfloat16",)
# Rows and columns of the device mesh: a single chip.
SUPPORTED_MESH = (1, 1)
TILE = 32

# The HF projections each LoRA-able layer of ttml's Llama holds, and the regex ``LoraModel`` matches
# that layer's name ("blocks.3.attention.qkv_linear") with.
LORA_LAYERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("q_proj", "k_proj", "v_proj"), r"\.attention\.qkv_linear$"),
    (("o_proj",), r"\.attention\.out_linear$"),
    (("gate_proj", "up_proj"), r"\.mlp\.w_gate_up$"),
    (("down_proj",), r"\.mlp\.w2$"),
)


def _fail(field: str, reason: str) -> None:
    raise UnsupportedTrainingConfig(f"{field}: {reason}")


def lora_layers(target_modules: Sequence[str]) -> list[tuple[tuple[str, ...], str]]:
    """The ``LORA_LAYERS`` entries the HF ``target_modules`` select.

    Raises ``UnsupportedTrainingConfig`` for a name that isn't a Llama projection, and for a fused layer
    only some of whose projections are targeted.
    """
    targets = set(target_modules)
    known = {name for projections, _ in LORA_LAYERS for name in projections}
    unknown = sorted(targets - known)
    if unknown:
        _fail("model.lora.target_modules", f"{unknown} are not Llama projections; choose from {sorted(known)}")
    selected = []
    for projections, pattern in LORA_LAYERS:
        chosen = targets.intersection(projections)
        if chosen and len(chosen) != len(projections):
            _fail(
                "model.lora.target_modules",
                f"{', '.join(projections)} are one fused layer in ttml's Llama, so LoRA targets all of them or "
                f"none; got only {sorted(chosen)}",
            )
        if chosen:
            selected.append((projections, pattern))
    return selected


def hf_target_modules(lora: LoraConfig) -> list[str]:
    """The HF projections the adapters cover, as PEFT lists them in ``adapter_config.json``."""
    return sorted(name for projections, _ in lora_layers(lora.target_modules) for name in projections)


def _same_model(adapter_base: Any, model_name: str) -> bool:
    """Whether an adapter's ``base_model_name_or_path`` names ``model_name``; a hub id and a local copy of it match."""
    if not adapter_base:
        return True

    def tail(name: str) -> str:
        return os.path.basename(os.path.normpath(name))

    return adapter_base == model_name or tail(adapter_base) == tail(model_name)


def _lora_named_layers(module: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """``(name, LoraLinear)`` for each LoRA layer under ``module``, named as ``LoraModel`` matched them."""
    from ttml.modules import AbstractModuleBase, LoraLinear

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoraLinear):
            yield full_name, child
        elif isinstance(child, AbstractModuleBase):
            yield from _lora_named_layers(child, full_name)


def _read(tensor: Any) -> np.ndarray:
    """A 2-D weight stored ``[1, 1, rows, cols]`` (bf16) as a float32 ``[rows, cols]`` array; lossless."""
    import ttnn

    values = tensor.to_numpy(ttnn.DataType.FLOAT32)
    return values.reshape(values.shape[-2:])


class LlamaProvider:
    """Provider for HF Llama checkpoints."""

    def check(self, config: SFTTrainerConfig, hf_config: dict[str, Any]) -> None:
        mesh = tuple(config.device.mesh)
        if mesh != SUPPORTED_MESH:
            _fail("device.mesh", f"Llama trains on a single chip so far, {list(SUPPORTED_MESH)}; got {list(mesh)}")
        if config.dtype not in SUPPORTED_DTYPES:
            _fail("dtype", f"ttml Llama weights and adapters are bf16; use 'bfloat16', got {config.dtype!r}")
        max_length = config.max_length
        if max_length % TILE != 0:
            _fail("max_length", f"must be a multiple of {TILE} (the ttml tile size), got {max_length}")
        if max_length > hf_config.get("max_position_embeddings", max_length):
            _fail("max_length", f"exceeds the model's max_position_embeddings ({hf_config['max_position_embeddings']})")
        lora = config.model.lora
        if lora.task_type != "CAUSAL_LM":
            _fail("model.lora.task_type", f"only CAUSAL_LM is supported, got {lora.task_type!r}")
        lora_layers(lora.target_modules)
        self._llama_config(config, hf_config)

    def _llama_config(self, config: SFTTrainerConfig, hf_config: dict[str, Any]) -> Any:
        from ttml.models import RunnerType
        from ttml.models.llama import LlamaConfig

        overrides = {"runner_type": RunnerType.MemoryEfficient} if config.model.gradient_checkpointing else {}
        try:
            # Sizes the RoPE tables: every training sequence is padded to exactly max_length.
            return LlamaConfig.from_hf(hf_config, max_position_embeddings=config.max_length, **overrides)
        except ValueError as e:
            raise UnsupportedTrainingConfig(
                f"model.name: {config.model.name} can't be built as a ttml Llama: {e}"
            ) from e

    def build(self, config: SFTTrainerConfig, hf_config: dict[str, Any]) -> Any:
        from ttml.models.llama import Llama, load_from_safetensors
        from ttml.modules import LoraConfig as TtmlLoraConfig
        from ttml.modules import LoraModel

        llama_config = self._llama_config(config, hf_config)
        weights = model_dir(config.model.name)
        # Loaded before LoRA wraps it: the loader checks it covers every parameter of a plain Llama.
        model = Llama(llama_config)
        load_from_safetensors(model, weights)
        lora = config.model.lora
        lora_config = TtmlLoraConfig(
            rank=lora.r,
            # Scaling alpha / r, as in PEFT.
            alpha=float(lora.alpha),
            target_modules=[pattern for _, pattern in lora_layers(lora.target_modules)],
            lora_dropout=lora.dropout,
        )
        try:
            return LoraModel(model, lora_config)
        except (ValueError, RuntimeError) as e:
            raise UnsupportedTrainingConfig(f"model.lora.target_modules: {e}") from e

    def _adapted_layers(self, model: Any) -> Iterator[tuple[Any, tuple[Any, ...]]]:
        """``(LoraLinear, the HF tensors its weight holds)`` for each LoRA layer of a ``LoraModel``."""
        from ttml.models.llama import hf_layout

        llama = model.model
        layout = hf_layout(llama)
        root = next(iter(llama.parameters())).split("/", 1)[0]
        for name, layer in _lora_named_layers(llama):
            yield layer, layout[f"{root}/{name.replace('.', '/')}/weight"]

    def export_adapter(self, model: Any, config: SFTTrainerConfig) -> AdapterState:
        tensors: dict[str, np.ndarray] = {}
        for layer, blocks in self._adapted_layers(model):
            tensors.update(lora_to_peft(_read(layer.lora_A.tensor), _read(layer.lora_B.tensor), blocks))
        lora = config.model.lora
        return AdapterState(
            tensors=tensors,
            config=peft_adapter_config(
                config.model.name,
                r=lora.r,
                alpha=lora.alpha,
                dropout=lora.dropout,
                target_modules=hf_target_modules(lora),
                task_type=lora.task_type,
            ),
        )

    def import_adapter(self, model: Any, state: AdapterState, config: SFTTrainerConfig) -> None:
        import ttml

        lora = config.model.lora
        adapter = state.config
        expected = {
            "peft_type": "LORA",
            "r": lora.r,
            "lora_alpha": lora.alpha,
            "use_rslora": False,
            "target_modules": hf_target_modules(lora),
        }
        found = {key: adapter.get(key) for key in expected}
        if isinstance(found["target_modules"], (list, tuple, set)):
            found["target_modules"] = sorted(found["target_modules"])
        found["use_rslora"] = bool(found["use_rslora"])
        mismatched = {key: (found[key], value) for key, value in expected.items() if found[key] != value}
        if mismatched:
            details = ", ".join(f"{key}: adapter {got!r}, run {want!r}" for key, (got, want) in mismatched.items())
            _fail("checkpoint", f"the adapter doesn't match this run's LoRA ({details})")
        if not _same_model(adapter.get("base_model_name_or_path"), config.model.name):
            _fail(
                "checkpoint",
                f"the adapter is for {adapter['base_model_name_or_path']!r}, this run trains {config.model.name!r}",
            )

        tensors = with_peft_prefix(state.tensors)
        layers = list(self._adapted_layers(model))
        check_peft_keys(tensors, (blocks for _, blocks in layers))
        for layer, blocks in layers:
            for parameter, value in zip((layer.lora_A, layer.lora_B), peft_to_lora(tensors, blocks)):
                shape = tuple(parameter.tensor.shape()[-2:])
                if value.shape != shape:
                    _fail("checkpoint", f"adapter tensor of shape {value.shape} for a {shape} LoRA weight")
                # Keeps the parameter's dtype (bf16) and layout.
                ttml.autograd.assign_numpy(parameter.tensor, value.reshape(1, 1, *shape).astype(np.float32))
