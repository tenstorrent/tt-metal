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

from .checkpoint_io import (
    OPTIMIZER_STATE,
    is_adapter_dir,
    is_model_dir,
    read_adapter,
    read_trainer_state,
    staged_directory,
    write_adapter,
    write_trainer_state,
)
from .configs import SFTTrainerConfig
from .data import DATASETS, build_dataloader, clear_cache, get_dataset, load_tokenizer
from .managers.device import DeviceManager
from .optimizers import CombinedOptimizer, assign_parameters
from .providers import UnsupportedTrainingConfig, get_provider
from .trainer import Trainer

# Device layout and dtype of each key `collate_causal_lm` produces.
BATCH_SPEC = {
    "input_ids": (ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
    "labels": (ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
    "loss_mask": (ttnn.Layout.TILE, ttnn.DataType.BFLOAT16),
}


class SFTTrainer(Trainer):
    """Supervised fine-tuning of a causal LM on ttml, on a single chip: LoRA adapters, or every weight.

    The model comes from a provider (``ttml.unified_trainer.providers``), which also decides whether the
    config can be trained at all. Checkpoints follow the model: a LoRA run saves a PEFT adapter directory,
    a full fine-tune an HF model directory.

    Losses are read back to host floats once per micro-batch, as ttml's other trainers do.
    """

    config: SFTTrainerConfig

    def _check_config(self) -> None:
        if self.config.dataset_id.lower() not in DATASETS:
            raise UnsupportedTrainingConfig(f"dataset_id: {self.config.dataset_id!r} is not one of {sorted(DATASETS)}")
        self.provider, self.hf_config = get_provider(self.config)
        self.provider.check(self.config, self.hf_config)

    def _seed(self) -> None:
        super()._seed()
        ttml.manual_seed(self.config.seed)

    def _load_dataloaders(self) -> tuple[Any, Optional[Any]]:
        tokenizer = load_tokenizer(self.config.model.name)
        train_dataset = get_dataset(self.config, "train", tokenizer)
        if len(train_dataset) < self.config.batch_size:
            raise ValueError(
                f"dataset_id: the train split has {len(train_dataset)} examples of at most max_length="
                f"{self.config.max_length} tokens, fewer than one batch of {self.config.batch_size}"
            )
        train_dataloader = build_dataloader(train_dataset, self.config, shuffle=True)
        if self.config.val_steps_freq == 0:
            return train_dataloader, None
        val_dataset = get_dataset(self.config, "validation", tokenizer)
        return train_dataloader, build_dataloader(val_dataset, self.config, shuffle=False)

    def _create_device_manager(self) -> DeviceManager:
        return DeviceManager()

    def _load_model(self) -> Any:
        return self.provider.build(self.config, self.hf_config)

    def _load_optimizer(self) -> CombinedOptimizer:
        """The config's optimizers, each over its trainable parameters (the adapters, for LoRA), at a constant LR."""
        if self.config.max_grad_norm > 0:
            validate_clip_grad_norm(self.model)
        names = [name for name, tensor in self.model.parameters().items() if tensor.get_requires_grad()]
        assigned = assign_parameters(names, self.config.optimizer)
        if self.config.master_dtype == "float32":
            # Before the optimizer, whose state follows the parameters' dtype. Checkpoints load after it.
            ttml.core.store_trainable_parameters_as(self.model.parameters(), ttnn.DataType.FLOAT32)
        parameters = self.model.parameters()
        optimizers = []
        for optimizer, optimizer_names in zip(self.config.optimizer, assigned):
            trainable = ttml.NamedParameters()
            for name in optimizer_names:
                trainable[name] = parameters[name]
            optimizers.append(ttml.optimizers.create_optimizer(optimizer.to_ttml(), trainable))
        return CombinedOptimizer(optimizers)

    # ------------------------------------------------------------------ steps

    def _train_batches(self, epoch: int, start: int) -> Any:
        # The shuffled order is a function of the seed and the epoch, so a resumed run trains on the batches the
        # run it continues would have, without collating the ones before `start`.
        state = self.train_dataloader.get_state_dict()
        self.train_dataloader.set_state_dict({**state, "epoch": epoch, "position": start})
        return self.train_dataloader

    def _prepare_batch(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            key: ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(value), *BATCH_SPEC[key])
            for key, value in batch.items()
        }

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
        """A PEFT adapter directory for a LoRA run, else an HF model directory; with ``trainer_state.json``, and
        with ``optimizer`` ``optimizer_state.pkl``."""
        with staged_directory(path) as directory:
            if self.config.model.lora is not None:
                write_adapter(directory, self.provider.export_adapter(self.model, self.config))
            else:
                self.provider.export_weights(self.model, self.config, str(directory))
            write_trainer_state(directory, metadata or {})
            if optimizer:
                # Also the trained weights at the precision they train in, which the HF files may round to bf16,
                # and the RNG of dropout: what a resumed run needs to continue as this one would.
                checkpointing.save_checkpoint(
                    str(directory / OPTIMIZER_STATE),
                    header={"rng": ttml.autograd.AutoContext.get_instance().get_generator_state()},
                    model_params=self._trainable_parameters(),
                    optimizer=self.optimizer,
                )

    def load_checkpoint(self, path: str, optimizer: bool = False) -> dict:
        """Loads what ``save_checkpoint`` wrote for this kind of run; the other kind of checkpoint is an error."""
        if self.config.model.lora is not None:
            if is_model_dir(path):
                raise UnsupportedTrainingConfig(
                    f"checkpoint: {path} is a full model, not a LoRA adapter; to train adapters on it, make it "
                    "the model.name"
                )
            self.provider.import_adapter(self.model, read_adapter(path), self.config)
        else:
            if is_adapter_dir(path):
                raise UnsupportedTrainingConfig(
                    f"checkpoint: {path} is a LoRA adapter, and this run trains every weight; set model.lora to train it"
                )
            if not is_model_dir(path):
                raise FileNotFoundError(f"{path} is not an HF model directory (config.json and *.safetensors)")
            self.provider.import_weights(self.model, path, self.config)
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
