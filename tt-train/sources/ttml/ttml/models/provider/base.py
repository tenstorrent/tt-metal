# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..utils.checkpoint import (
    ADAPTER_CONFIG,
    BASE_MODEL_DIR,
    TT_CHECKPOINT,
    AdapterState,
    PathLike,
    is_adapter_dir,
    is_model_dir,
    is_tt_checkpoint,
    read_adapter,
    write_adapter,
)
from ..utils.errors import UnsupportedTrainingConfig

if TYPE_CHECKING:
    from ttml.unified_trainer.configs import TrainerConfig

# What a training run reads from a hub model: the weights, the configs and the tokenizer.
_MODEL_FILES = ["*.safetensors", "*.json", "tokenizer.model", "chat_template.jinja"]


class ModelProvider(ABC):
    """One family of HF models in ttml, for one run: whether it can train the run's config, the model, and its weights.

    The only place that decides whether a model can train a config: the trainer and its hosts never inspect
    LoRA targets, dtypes, sequence lengths or the device mesh. Every config a provider gets has ``model.lora``
    set, as the trainer rejects a run without it. A run uses its provider as::

        provider = get_provider(config)
        provider.check()  # before the device opens
        model = provider.build()
        provider.load_weights(model, checkpoint)  # the pretrained weights, then the checkpoint's if one is given
        ...
        provider.export_weights(model, directory)  # a checkpoint, in the format config.checkpoint.format picks

    ``load_weights`` and ``export_weights`` read and write the checkpoint formats (``ttml.models.utils.checkpoint``).
    A family provides what is its own, its HF model's weights and its LoRA adapters in PEFT's form, by implementing the
    abstract methods after them.
    """

    def __init__(self, config: TrainerConfig, hf_config: dict[str, Any]) -> None:
        self.config = config
        # The model's HF config.json.
        self.hf_config = hf_config

    @abstractmethod
    def check(self) -> None:
        """Raise ``UnsupportedTrainingConfig`` if ``config`` can't be trained on this model.

        Runs before any weights are downloaded and before the device is opened.
        """

    @abstractmethod
    def build(self) -> Any:
        """The model's architecture, wrapped in ``ttml.modules.LoraModel`` as ``config.model.lora`` describes. Its
        weights are initial ones until ``load_weights``.

        The model takes the standard ttml causal-LM input the trainer assumes:
        ``logits = model(input_ids, None)``, with ``input_ids`` ``[B, 1, 1, T]`` uint32, ``logits``
        ``[B, 1, T, V]``, and ``None`` for a causal mask.
        """

    def load_weights(self, model: Any, checkpoint: Optional[PathLike] = None) -> None:
        """Load the pretrained weights of ``config.model.name`` into ``model``, then those of ``checkpoint``, a
        directory ``export_weights`` wrote, if one is given.

        The checkpoint's format is told from its files: a ``tt`` checkpoint loads every weight, a PEFT adapter the LoRA
        adapters. Either must be of this run's model, with its LoRA settings. A full model directory is an error: a run
        starts from one as its ``model.name``.
        """
        self._load_hf_weights(model)
        if checkpoint is None:
            return
        directory = Path(checkpoint)
        if is_tt_checkpoint(directory):
            from ttml import checkpointing

            file = str(directory / TT_CHECKPOINT)
            self._check_tt_header(directory, checkpointing.read_header(file))
            checkpointing.load_checkpoint(file, model_params=model.parameters())
        elif is_adapter_dir(directory):
            adapter = read_adapter(directory)
            self._check_base_model(directory, adapter.config.get("base_model_name_or_path"))
            self._import_adapter(model, adapter)
        elif is_model_dir(directory):
            raise UnsupportedTrainingConfig(
                f"checkpoint: {directory} is a full model directory; to train on it, make it the model.name"
            )
        else:
            raise FileNotFoundError(
                f"{directory} is not a checkpoint: it holds neither {TT_CHECKPOINT} nor {ADAPTER_CONFIG}"
            )

    def export_weights(self, model: Any, directory: PathLike) -> None:
        """Read the weights of ``model`` from the device, a single chip so far, and write them to ``directory``, in
        the format ``config.checkpoint.format`` picks.

        ``hf`` with ``trainable``: a LoRA run's PEFT adapter, or for full fine-tuning the HF model. ``hf`` with ``all``
        also holds a LoRA run's frozen base model, in ``base_model/``. ``tt``: every parameter in ``checkpoint.pkl``,
        with a header that names the base model and the LoRA settings.
        """
        directory = Path(directory)
        checkpoint_format = self.config.checkpoint.format
        if checkpoint_format.type == "tt":
            from ttml import checkpointing

            checkpointing.save_checkpoint(
                str(directory / TT_CHECKPOINT), header=self._tt_header(), model_params=model.parameters()
            )
        elif self.config.model.lora is None:
            self._export_hf_model(model, directory)
        else:
            write_adapter(directory, self._export_adapter(model))
            if checkpoint_format.set == "all":
                self._export_hf_model(model, directory / BASE_MODEL_DIR)

    # ------------------------------------------------------------------ what a family provides

    @abstractmethod
    def _load_hf_weights(self, model: Any) -> None:
        """Load the pretrained weights of ``config.model.name`` into ``model``; its LoRA adapters keep the values
        ``build`` gave them."""

    @abstractmethod
    def _export_hf_model(self, model: Any, directory: Path) -> None:
        """Write the HF model ``model`` holds as a directory, laid out as ``save_pretrained`` lays one out, with the
        config and tokenizer files of ``config.model.name``. LoRA adapters are left out: for a LoRA model, that's
        its frozen base model."""

    @abstractmethod
    def _export_adapter(self, model: Any) -> AdapterState:
        """The LoRA adapters of ``model`` as a PEFT adapter of the HF model."""

    @abstractmethod
    def _import_adapter(self, model: Any, adapter: AdapterState) -> None:
        """Load a PEFT adapter of the run's base model into ``model``; raise ``UnsupportedTrainingConfig`` if it
        doesn't fit."""

    # ------------------------------------------------------------------ what a checkpoint records and must match

    def _base_model(self) -> dict:
        """The model the run trains, as a checkpoint records it: its name, and its hub commit."""
        return {"name_or_path": self.config.model.name, "revision": model_revision(self.config.model.name)}

    def _lora_settings(self) -> Optional[dict]:
        """What a ``tt`` checkpoint's LoRA adapters must share with the run that loads it."""
        lora = self.config.model.lora
        if lora is None:
            return None
        return {"r": lora.r, "alpha": lora.alpha, "target_modules": sorted(set(lora.target_modules))}

    def _tt_header(self) -> dict:
        return {
            "base_model": self._base_model(),
            "model_type": self.hf_config.get("model_type"),
            "lora": self._lora_settings(),
        }

    def _check_tt_header(self, directory: Path, header: dict) -> None:
        self._check_base_model(directory, (header.get("base_model") or {}).get("name_or_path"))
        expected = {"model_type": self.hf_config.get("model_type"), "lora": self._lora_settings()}
        mismatched = {key: (header.get(key), want) for key, want in expected.items() if header.get(key) != want}
        if mismatched:
            details = ", ".join(f"{key}: checkpoint {got!r}, run {want!r}" for key, (got, want) in mismatched.items())
            raise UnsupportedTrainingConfig(f"checkpoint: {directory} doesn't match this run's model ({details})")

    def _check_base_model(self, directory: Path, base: Optional[str]) -> None:
        """Raise unless a checkpoint's base model is this run's ``model.name``, or the checkpoint's own
        ``base_model/``, which a LoRA run saved with every weight can train from."""
        name = self.config.model.name
        if same_model(base, name) or Path(name).resolve() == (directory / BASE_MODEL_DIR).resolve():
            return
        raise UnsupportedTrainingConfig(f"checkpoint: {directory} is for {base!r}, this run trains {name!r}")


def load_hf_config(model_name: str) -> dict[str, Any]:
    """The ``config.json`` of a local model directory, or of a hub model (downloads only that file)."""
    path = Path(model_name) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model_name, "config.json"))
    return json.loads(path.read_text())


def model_dir(model_name: str, local_only: bool = False) -> str:
    """A local directory with the model's weights, configs and tokenizer; downloads a hub model once.

    With ``local_only``, a hub model is never looked up online: the copy downloaded before is taken.
    """
    if Path(model_name).is_dir():
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, allow_patterns=_MODEL_FILES, local_files_only=local_only)


def model_revision(model_name: str) -> Optional[str]:
    """The hub commit of the copy of a hub model that ``model_dir`` downloaded; None for a local directory."""
    if Path(model_name).is_dir():
        return None
    # The hub cache keeps each commit in snapshots/<commit>/.
    return Path(model_dir(model_name, local_only=True)).name


def same_model(base: Any, model_name: str) -> bool:
    """Whether a checkpoint's base model ``base`` names ``model_name``: a hub id and a local copy of it match, and so
    does a checkpoint that records no base."""
    if not base:
        return True

    def tail(name: str) -> str:
        return os.path.basename(os.path.normpath(name))

    return base == model_name or tail(base) == tail(model_name)
