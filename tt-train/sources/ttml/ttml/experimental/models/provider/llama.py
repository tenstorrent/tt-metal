# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HF ``model_type: llama`` checkpoints (Llama 3.x) on ttml's Python ``Llama``, optionally with LoRA adapters.

ttml's Llama fuses q, k and v into one ``qkv_linear``, and gate and up into one ``w_gate_up``. Each targeted
projection still gets its own adapter, as in PEFT: a fused layer holds the adapters of the projections it fuses
(``ttml.modules.LoraBlocks``), so any of them can be targeted on their own.

The weights a checkpoint holds, by ``checkpoint.format.type`` and whether ``model.lora`` is set:

- ``hf`` with LoRA: the adapters, as a PEFT adapter directory.
- ``hf`` without: the model, as ``save_pretrained`` lays one out.
- ``tt`` with LoRA: the adapters, in ``checkpoint.pkl``.
- ``tt`` without: every parameter, in ``checkpoint.pkl``.

``checkpoint.pkl`` has a header naming the base model and the LoRA settings, which a load must match.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import numpy as np

from ...errors import UnsupportedTrainingConfig
from ..utils.checkpoint import TT_CHECKPOINT, AdapterState, PathLike, read_adapter, write_adapter
from ..utils.hub import load_hf_config, model_dir
from ..utils.peft import check_peft_keys, lora_to_peft, peft_adapter_config, peft_to_lora, with_peft_prefix
from .base import ModelProvider

if TYPE_CHECKING:
    from ttml.experimental.trainer.configs import LoraConfig, SFTTrainerConfig

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


def lora_layers(target_modules: Sequence[str]) -> list[tuple[tuple[str, ...], str, tuple[bool, ...]]]:
    """The ``LORA_LAYERS`` entries the HF ``target_modules`` select, each with a flag per projection: targeted or not.

    Raises ``UnsupportedTrainingConfig`` for a name that isn't a Llama projection.
    """
    targets = set(target_modules)
    known = {name for projections, _ in LORA_LAYERS for name in projections}
    unknown = sorted(targets - known)
    if unknown:
        _fail("model.lora.target_modules", f"{unknown} are not Llama projections; choose from {sorted(known)}")
    selected = []
    for projections, pattern in LORA_LAYERS:
        adapted = tuple(name in targets for name in projections)
        if any(adapted):
            selected.append((projections, pattern, adapted))
    return selected


def hf_target_modules(lora: LoraConfig) -> list[str]:
    """The HF projections the adapters cover, as PEFT lists them in ``adapter_config.json``."""
    return sorted(set(lora.target_modules))


def _layer_name(parameter: str) -> str:
    """The name ``LoraModel`` matches a weight's layer by: "Llama/blocks/3/attention/qkv_linear/weight" ->
    "blocks.3.attention.qkv_linear"."""
    return parameter.split("/", 1)[1].removesuffix("/weight").replace("/", ".")


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


class LlamaProvider(ModelProvider):
    """Provider for HF Llama checkpoints."""

    config: SFTTrainerConfig

    def __init__(self, config: SFTTrainerConfig) -> None:
        super().__init__(config)
        # The model's HF config.json.
        self.hf_config = load_hf_config(self.model_source())

    def check(self) -> None:
        config = self.config
        mesh = tuple(config.device.mesh)
        if mesh != SUPPORTED_MESH:
            _fail("device.mesh", f"Llama trains on a single chip so far, {list(SUPPORTED_MESH)}; got {list(mesh)}")
        if config.dtype not in SUPPORTED_DTYPES:
            _fail("dtype", f"ttml Llama weights and adapters are bf16; use 'bfloat16', got {config.dtype!r}")
        max_length = config.max_length
        if max_length % TILE != 0:
            _fail("max_length", f"must be a multiple of {TILE} (the ttml tile size), got {max_length}")
        max_positions = self.hf_config.get("max_position_embeddings", max_length)
        if max_length > max_positions:
            _fail("max_length", f"exceeds the model's max_position_embeddings ({max_positions})")
        lora = config.model.lora
        if lora is not None:
            if lora.task_type != "CAUSAL_LM":
                _fail("model.lora.task_type", f"only CAUSAL_LM is supported, got {lora.task_type!r}")
            lora_layers(lora.target_modules)
        self._llama_config()

    def _llama_config(self) -> Any:
        from ttml.models import RunnerType
        from ttml.models.llama import LlamaConfig

        overrides = {"runner_type": RunnerType.MemoryEfficient} if self.config.model.gradient_checkpointing else {}
        try:
            # Sizes the RoPE tables: every training sequence is padded to exactly max_length.
            return LlamaConfig.from_hf(self.hf_config, max_position_embeddings=self.config.max_length, **overrides)
        except ValueError as e:
            raise UnsupportedTrainingConfig(
                f"model.name: {self.config.model.name} can't be built as a ttml Llama: {e}"
            ) from e

    def build(self) -> Any:
        """The ``Llama`` with the pretrained weights, wrapped in ``ttml.modules.LoraModel`` if ``model.lora`` is set."""
        from ttml.models.llama import Llama, load_from_safetensors
        from ttml.modules import LoraConfig as TtmlLoraConfig
        from ttml.modules import LoraModel

        model = Llama(self._llama_config())
        # Before wrapping: the adapters LoraModel adds have no HF tensors, and keep their initial values.
        load_from_safetensors(model, model_dir(self.model_source()))
        lora = self.config.model.lora
        if lora is None:
            return model
        layers = lora_layers(lora.target_modules)
        lora_config = TtmlLoraConfig(
            rank=lora.r,
            # Scaling alpha / r, as in PEFT.
            alpha=float(lora.alpha),
            target_modules=[pattern for _, pattern, _ in layers],
            target_blocks=self._target_blocks(model, layers),
            lora_dropout=lora.dropout,
        )
        try:
            return LoraModel(model, lora_config)
        except (ValueError, RuntimeError) as e:
            raise UnsupportedTrainingConfig(f"model.lora.target_modules: {e}") from e

    def _target_blocks(self, llama: Any, layers: Sequence[tuple[tuple[str, ...], str, tuple[bool, ...]]]) -> dict:
        """The ``LoraBlocks`` of each selected fused layer: the rows of its projections, as ``hf_layout`` places them,
        and which of them are targeted."""
        from ttml.models.llama import hf_layout
        from ttml.modules import LoraBlocks

        layout = hf_layout(llama)
        blocks = {}
        for projections, pattern, adapted in layers:
            if len(projections) == 1:
                continue
            rows = {
                tuple(tensor.shape[0] for tensor in held.tensors)
                for parameter, held in layout.items()
                if re.search(pattern, _layer_name(parameter))
            }
            # Every transformer block has the same shapes.
            (block_rows,) = rows
            blocks[pattern] = LoraBlocks(rows=block_rows, adapted=adapted)
        return blocks

    def _adapted_layers(self, model: Any) -> Iterator[tuple[Any, tuple[Any, ...], tuple[bool, ...]]]:
        """``(LoraLinear, the HF tensors its weight holds, which of them it adapts)`` for each LoRA layer of a
        ``LoraModel``."""
        from ttml.models.llama import hf_layout

        llama = model.model
        layout = hf_layout(llama)
        root = next(iter(llama.parameters())).split("/", 1)[0]
        for name, layer in _lora_named_layers(llama):
            held = layout[f"{root}/{name.replace('.', '/')}/weight"]
            yield layer, held, layer.blocks.adapted if layer.blocks is not None else (True,) * len(held.tensors)

    def load_weights(self, model: Any, directory: PathLike) -> None:
        directory = Path(directory)
        if self.config.checkpoint.format.type == "tt":
            self._load_tt_checkpoint(model, directory)
        elif self.config.model.lora is not None:
            adapter = read_adapter(directory)
            self._check_base_model(directory, adapter.config.get("base_model_name_or_path"))
            self._import_adapter(model, adapter)
        else:
            from ttml.models.llama import load_from_safetensors

            load_from_safetensors(model, directory)

    def export_weights(self, model: Any, directory: PathLike) -> None:
        directory = Path(directory)
        if self.config.checkpoint.format.type == "tt":
            from ttml import checkpointing

            checkpointing.save_checkpoint(
                str(directory / TT_CHECKPOINT), header=self._tt_header(), model_params=self._tt_parameters(model)
            )
        elif self.config.model.lora is not None:
            write_adapter(directory, self._export_adapter(model))
        else:
            from ttml.models.llama import export_hf_model

            export_hf_model(model, directory, hf_source_dir=model_dir(self.model_source(), local_only=True))

    # ------------------------------------------------------------------ tt checkpoints

    def _tt_parameters(self, model: Any) -> Any:
        """The parameters a ``tt`` checkpoint holds: the LoRA adapters of a LoRA model, else every parameter."""
        import ttml

        parameters = model.parameters()
        if self.config.model.lora is None:
            return parameters
        adapters = ttml.NamedParameters()
        for name, tensor in parameters.items():
            if name.endswith(("/lora_A", "/lora_B")):
                adapters[name] = tensor
        return adapters

    def _tt_header(self) -> dict:
        return {
            "base_model": {"name_or_path": self.config.model.name},
            "model_type": self.hf_config.get("model_type"),
            "lora": self._lora_settings(),
        }

    def _lora_settings(self) -> Optional[dict]:
        """What a ``tt`` checkpoint's LoRA adapters must share with the model that loads them."""
        lora = self.config.model.lora
        if lora is None:
            return None
        return {"r": lora.r, "alpha": lora.alpha, "target_modules": hf_target_modules(lora)}

    def _load_tt_checkpoint(self, model: Any, directory: Path) -> None:
        from ttml import checkpointing

        file = directory / TT_CHECKPOINT
        if not file.is_file():
            raise FileNotFoundError(f"{directory} is not a tt checkpoint: no {TT_CHECKPOINT}")
        header = checkpointing.read_header(str(file))
        self._check_base_model(directory, (header.get("base_model") or {}).get("name_or_path"))
        expected = {"model_type": self.hf_config.get("model_type"), "lora": self._lora_settings()}
        mismatched = {key: (header.get(key), want) for key, want in expected.items() if header.get(key) != want}
        if mismatched:
            details = ", ".join(f"{key}: checkpoint {got!r}, run {want!r}" for key, (got, want) in mismatched.items())
            _fail("checkpoint", f"{directory} doesn't match this model ({details})")
        checkpointing.load_checkpoint(str(file), model_params=self._tt_parameters(model))

    def _check_base_model(self, directory: Path, base: Optional[str]) -> None:
        """A checkpoint that records no base model is taken as this model's."""
        name = self.config.model.name
        if base and base != name:
            _fail("checkpoint", f"{directory} is for {base!r}, this model is {name!r}")

    # ------------------------------------------------------------------ PEFT adapters

    def _export_adapter(self, model: Any) -> AdapterState:
        tensors: dict[str, np.ndarray] = {}
        for layer, held, adapted in self._adapted_layers(model):
            tensors.update(lora_to_peft(_read(layer.lora_A.tensor), _read(layer.lora_B.tensor), held, adapted))
        lora = self.config.model.lora
        return AdapterState(
            tensors=tensors,
            config=peft_adapter_config(
                self.config.model.name,
                r=lora.r,
                alpha=lora.alpha,
                dropout=lora.dropout,
                target_modules=hf_target_modules(lora),
                task_type=lora.task_type,
            ),
        )

    def _import_adapter(self, model: Any, adapter: AdapterState) -> None:
        import ttml

        lora = self.config.model.lora
        expected = {
            "peft_type": "LORA",
            "r": lora.r,
            "lora_alpha": lora.alpha,
            "use_rslora": False,
            "target_modules": hf_target_modules(lora),
        }
        found = {key: adapter.config.get(key) for key in expected}
        if isinstance(found["target_modules"], (list, tuple, set)):
            found["target_modules"] = sorted(found["target_modules"])
        found["use_rslora"] = bool(found["use_rslora"])
        mismatched = {key: (found[key], value) for key, value in expected.items() if found[key] != value}
        if mismatched:
            details = ", ".join(f"{key}: adapter {got!r}, run {want!r}" for key, (got, want) in mismatched.items())
            _fail("checkpoint", f"the adapter doesn't match this run's LoRA ({details})")

        tensors = with_peft_prefix(adapter.tensors)
        layers = list(self._adapted_layers(model))
        check_peft_keys(tensors, ((held, adapted) for _, held, adapted in layers))
        for layer, held, adapted in layers:
            for parameter, value in zip((layer.lora_A, layer.lora_B), peft_to_lora(tensors, held, adapted, lora.r)):
                # Keeps the parameter's dtype (bf16) and layout.
                ttml.autograd.assign_numpy(parameter.tensor, value.reshape(1, 1, *value.shape))
