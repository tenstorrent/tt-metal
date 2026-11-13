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
from ttml.models.provider import UnsupportedTrainingConfig
from ttml.models.provider import get_provider as get_model_provider

from ..configs import SFTTrainerConfig
from ..managers.device import DeviceManager
from ..utils import (
    OPTIMIZER_STATE,
    CombinedOptimizer,
    assign_parameters,
    is_model_dir,
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
    trained at all, and the batches from dataset providers (``ttml.datasets.provider``), one per split. A checkpoint
    is a PEFT adapter directory.

    Losses are read back to host floats once per micro-batch, as ttml's other trainers do.
    """

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
        if self.config.val_steps_freq == 0:
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

    def _train_batches(self, epoch: int, start: int) -> DatasetProvider:
        # The shuffled order is a function of the seed and the epoch, so a resumed run trains on the batches the
        # run it continues would have, without collating the ones before `start`.
        self.train_dataloader.seek(epoch, start)
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

    def _optimizer_step(self) -> None:
        if self.config.max_grad_norm > 0:
            ttml.core.clip_grad_norm(self.model.parameters(), self.config.max_grad_norm, 2.0, False)
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

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None, optimizer: bool = False) -> None:
        """A PEFT adapter directory, with ``trainer_state.json``, and with ``optimizer`` ``optimizer_state.pkl``."""
        with staged_directory(path) as directory:
            write_adapter(directory, self.provider.export_adapter(self.model, self.config))
            write_trainer_state(directory, metadata or {})
            if optimizer:
                # Also the trained weights and the RNG of dropout: what a resumed run needs to continue as this
                # one would.
                checkpointing.save_checkpoint(
                    str(directory / OPTIMIZER_STATE),
                    header={"rng": ttml.autograd.AutoContext.get_instance().get_generator_state()},
                    model_params=self._trainable_parameters(),
                    optimizer=self.optimizer,
                )

    def load_checkpoint(self, path: str, optimizer: bool = False) -> dict:
        """Loads the PEFT adapter directory ``save_checkpoint`` wrote; a full model directory is an error."""
        if is_model_dir(path):
            raise UnsupportedTrainingConfig(
                f"checkpoint: {path} is a full model, not a LoRA adapter; to train adapters on it, make it "
                "the model.name"
            )
        self.provider.import_adapter(self.model, read_adapter(path), self.config)
        if optimizer:
            state_path = Path(path) / OPTIMIZER_STATE
            if not state_path.is_file():
                raise FileNotFoundError(f"{path} has no optimizer state: it was saved without checkpoint.save_optim")
            header = checkpointing.load_checkpoint(
                str(state_path), model_params=self._trainable_parameters(), optimizer=self.optimizer
            )
            ttml.autograd.AutoContext.get_instance().set_generator_state(header["rng"])
        return read_trainer_state(path)

    def _trainable_parameters(self) -> Any:
        trainable = ttml.NamedParameters()
        for name, tensor in self.model.parameters().items():
            if tensor.get_requires_grad():
                trainable[name] = tensor
        return trainable

    def cleanup(self) -> None:
        # The tokenized-dataset cache is process-wide; the next run may use another model or length.
        clear_cache()
        super().cleanup()
