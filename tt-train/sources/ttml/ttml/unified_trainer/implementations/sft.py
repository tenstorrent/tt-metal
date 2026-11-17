# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import ttnn

import ttml
from ttml import checkpointing
from ttml.common.utils import no_grad, validate_clip_grad_norm
from ttml.datasets.provider import DatasetProvider, clear_cache, load_tokenizer
from ttml.datasets.provider import get_provider as get_dataset_provider
from ttml.datasets.provider import providers as dataset_providers
from ttml.models.provider import UnsupportedTrainingConfig, model_revision, same_model
from ttml.models.provider import get_provider as get_model_provider

from ..configs import SFTTrainerConfig
from ..managers.device import DeviceManager
from ..utils import (
    ADAPTER_CONFIG,
    BASE_MODEL_DIR,
    TT_CHECKPOINT,
    CombinedOptimizer,
    assign_parameters,
    is_adapter_dir,
    is_model_dir,
    is_tt_checkpoint,
    read_adapter,
    read_trainer_state,
    staged_directory,
    write_adapter,
    write_trainer_state,
)
from .base import Trainer

# Device layout and dtype of each key `collate_causal_lm` produces.
BATCH_SPEC = {
    "input_ids": (ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
    "labels": (ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
    "loss_mask": (ttnn.Layout.TILE, ttnn.DataType.BFLOAT16),
}


class SFTTrainer(Trainer):
    """Supervised fine-tuning of a causal LM on ttml, on a single chip. It trains LoRA adapters with AdamW; full
    fine-tuning (every weight) is not supported yet, and a config without ``model.lora`` is rejected.

    The model comes from a model provider (``ttml.models.provider``), which also decides whether the config can be
    trained at all, and the batches from dataset providers (``ttml.datasets.provider``), one per split. Checkpoints
    are saved in the format ``checkpoint.format`` picks (see ``save_checkpoint``).

    Losses are read back to host floats once per micro-batch, as ttml's other trainers do.
    """

    config_class = SFTTrainerConfig
    config: SFTTrainerConfig

    def _check_config(self) -> None:
        if self.config.model.lora is None:
            raise UnsupportedTrainingConfig(
                "model.lora: only LoRA training is supported so far, not full fine-tuning; set model.lora"
            )
        if self.config.dataset.id.lower() not in dataset_providers():
            raise UnsupportedTrainingConfig(
                f"dataset.id: {self.config.dataset.id!r} is not one of {sorted(dataset_providers())}"
            )
        self.provider, self.hf_config = get_model_provider(self.config)
        self.provider.check(self.config, self.hf_config)

    def _seed(self) -> None:
        super()._seed()
        ttml.manual_seed(self.config.seed)

    def _load_dataloaders(self) -> tuple[DatasetProvider, Optional[DatasetProvider]]:
        """The dataset providers of the train and validation splits; the loop gets its batches from them."""
        tokenizer = load_tokenizer(self.config.model.name)
        train = get_dataset_provider(self.config, "train", tokenizer)
        if len(train) == 0:
            raise ValueError(
                f"dataset.id: the train split has {train.num_examples} examples of at most max_length="
                f"{self.config.max_length} tokens, fewer than one batch of {self.config.batch_size}"
            )
        if self.config.metrics.validation.steps_freq == 0:
            return train, None
        return train, get_dataset_provider(self.config, "validation", tokenizer)

    def _create_device_manager(self) -> DeviceManager:
        return DeviceManager(self.config.device)

    def _load_model(self) -> Any:
        return self.provider.build(self.config, self.hf_config)

    def _load_optimizer(self) -> CombinedOptimizer:
        """The config's optimizers, each over its trainable parameters (the LoRA adapters), at a constant LR."""
        if self.config.max_grad_norm > 0:
            validate_clip_grad_norm(self.model)
        names = [name for name, tensor in self.model.parameters().items() if tensor.get_requires_grad()]
        assigned = assign_parameters(names, self.config.optimizer)
        parameters = self.model.parameters()
        optimizers = []
        for optimizer, optimizer_names in zip(self.config.optimizer, assigned):
            trainable = ttml.NamedParameters()
            for name in optimizer_names:
                trainable[name] = parameters[name]
            optimizers.append(ttml.optimizers.create_optimizer(optimizer.to_ttml(), trainable))
        return CombinedOptimizer(optimizers)

    # ------------------------------------------------------------------ steps

    def _train_batches(self, epoch: int) -> DatasetProvider:
        # The shuffled order is a function of the seed and the epoch.
        self.train_dataloader.seek(epoch)
        return self.train_dataloader

    def _prepare_batch(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        # The device manager places each array on the mesh `config.device` names.
        return {key: self.device_manager.to_device(value, *BATCH_SPEC[key]) for key, value in batch.items()}

    def _forward(self, batch: dict[str, Any]) -> Any:
        # mask=None: causal attention on device. Batches are right-padded and padded positions have a zero
        # loss_mask, so causal attention is exact for every real token.
        logits = self.model(batch["input_ids"], None)
        per_token = ttml.ops.loss.cross_entropy_loss(logits, batch["labels"], ttml.ops.ReduceType.NONE)
        # loss_mask is normalized to sum to B * T, so the mean is over real targets only.
        return ttml.ops.unary.mean(per_token * batch["loss_mask"])

    def _scale_loss(self, loss: Any, divisor: int) -> Any:
        # ttml tensors multiply by a float but have no true division.
        return loss if divisor == 1 else loss * (1.0 / divisor)

    def _backward(self, loss: Any) -> None:
        loss.backward(False)

    def _accumulate_loss(self, running: Optional[float], loss: Any) -> float:
        # One host readback per micro-batch, before `_clean_graph_state` drops the graph.
        value = float(self.device_manager.to_numpy(loss).mean())
        return value if running is None else running + value

    def _clean_graph_state(self) -> None:
        # Drops this batch's autograd graph; gradients stay on the parameters until zero_grad.
        ttml.autograd.AutoContext.get_instance().reset_graph()

    def _clip_gradients(self, max_norm: float) -> None:
        # The L2 norm over every gradient; a non-finite norm is not an error.
        ttml.core.clip_grad_norm(self.model.parameters(), max_norm, 2.0, False)

    def _optimizer_step(self) -> None:
        self.optimizer.step()
        self._zero_grad()

    def _zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def _train_mode(self) -> None:
        self.model.train()

    def _eval_mode(self) -> None:
        self.model.eval()

    def _no_grad(self) -> Any:
        return no_grad()

    # ------------------------------------------------------------------ for callbacks and hosts

    def loss_to_float(self, loss: float) -> float:
        return float(loss)

    def parameter_counts(self) -> tuple[int, int]:
        tensors = list(self.model.parameters().values())
        total = sum(int(np.prod(t.shape())) for t in tensors)
        trainable = sum(int(np.prod(t.shape())) for t in tensors if t.get_requires_grad())
        return total, trainable

    def current_learning_rate(self) -> list[float]:
        return [float(lr) for lr in self.optimizer.get_group_lrs()]

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None) -> None:
        """A checkpoint directory in the format ``checkpoint.format`` picks, with ``trainer_state.json``.

        ``hf`` with ``trainable``: a LoRA run's PEFT adapter, or for full fine-tuning the HF model. ``hf`` with ``all``
        also holds a LoRA run's frozen base model, in ``base_model/``. ``tt``: every parameter in ``checkpoint.pkl``.
        """
        checkpoint_format = self.config.checkpoint.format
        with staged_directory(path) as directory:
            if checkpoint_format.type == "tt":
                checkpointing.save_checkpoint(
                    str(directory / TT_CHECKPOINT), header=self._tt_header(), model_params=self.model.parameters()
                )
            elif self.config.model.lora is None:
                self.provider.export_model(self.model, self.config, directory)
            else:
                write_adapter(directory, self.provider.export_adapter(self.model, self.config))
                if checkpoint_format.set == "all":
                    self.provider.export_model(self.model, self.config, directory / BASE_MODEL_DIR)
            write_trainer_state(
                directory,
                {**(metadata or {}), "format": checkpoint_format.model_dump(), "base_model": self._base_model()},
            )

    def load_checkpoint(self, path: str) -> dict:
        """Loads the weights of a checkpoint directory ``save_checkpoint`` wrote, in the format it finds there.

        A ``tt`` checkpoint loads every parameter, a PEFT adapter the LoRA adapters. Either must be for this run's
        model, with its LoRA settings. A full model directory is an error: a run starts from one as its
        ``model.name``.
        """
        directory = Path(path)
        if is_tt_checkpoint(directory):
            file = str(directory / TT_CHECKPOINT)
            self._check_tt_header(directory, checkpointing.read_header(file))
            checkpointing.load_checkpoint(file, model_params=self.model.parameters())
        elif is_adapter_dir(directory):
            adapter = read_adapter(directory)
            self._check_base_model(directory, adapter.config.get("base_model_name_or_path"))
            self.provider.import_adapter(self.model, adapter, self.config)
        elif is_model_dir(directory):
            raise UnsupportedTrainingConfig(
                f"checkpoint: {path} is a full model directory; to train on it, make it the model.name"
            )
        else:
            raise FileNotFoundError(
                f"{path} is not a checkpoint: it holds neither {TT_CHECKPOINT} nor {ADAPTER_CONFIG}"
            )
        return read_trainer_state(directory)

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

    def cleanup(self) -> None:
        # The tokenized-dataset cache is process-wide; the next run may use another model or length.
        clear_cache()
        super().cleanup()
