# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import itertools
import random
import sys
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional

import numpy as np
from tqdm import tqdm

from ..callbacks.callback import CallbackHandler
from ..configs import CheckpointConfig, TrainerConfig
from ..managers.checkpoint import CheckpointManager
from ..managers.logging import TrainingLogger


class Trainer(ABC):
    """The training loop: callbacks, gradient accumulation, validation scheduling and the run lifecycle.

    Holds no tensors and imports no framework; everything that does is an abstract method a subclass such
    as ``SFTTrainer`` implements. A host builds the trainer once with its callbacks and runs any number of
    jobs on it::

        trainer = SFTTrainer(callbacks=[...])
        for config in jobs:
            try:
                trainer.setup(config)
                trainer.train()
            finally:
                trainer.cleanup()
    """

    def __init__(self, callbacks: Any = None) -> None:
        self.callback_handler = CallbackHandler(self, callbacks)
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self.config: Optional[TrainerConfig] = None
        self.logger: Optional[TrainingLogger] = None
        self.global_step = 0
        self.epoch = 0
        # Sum of the scaled losses of the current accumulation window, as `_accumulate_loss` keeps it.
        self.step_loss: Any = None

    def setup(self, config: TrainerConfig) -> None:
        """Validate ``config`` and build what a run needs: data, device, model and optimizer.

        Fails on an untrainable config before the device is opened or any weights are downloaded. Any
        exception goes to every callback's ``on_setup_error`` and is re-raised; ``cleanup`` still frees what
        was built.
        """
        try:
            self._host_attributes()
            if self.config is not None:
                self.cleanup()
            self.config = config
            self.logger = TrainingLogger(config.logging, run_config=config.model_dump(mode="json"))
            self._check_config()
            self._seed()
            self.train_dataloader, self.val_dataloader = self._load_dataloaders()
            self._check_stopping_criteria()
            self.device_manager = self._create_device_manager()
            self.model = self._load_model()
            self.optimizer = self._load_optimizer()
        except Exception as exception:
            self.callback_handler("on_setup_error", exception)
            raise

    # ------------------------------------------------------------------ the loop

    @contextmanager
    def _train_lifecycle(self) -> Iterator[None]:
        """``on_train_start`` before the loop and ``on_train_end`` after it.

        Any exception, also from ``on_train_start``, goes to every callback's ``on_train_error`` and is
        re-raised, so ``on_train_end`` never sees a half-trained model.
        """
        try:
            self.callback_handler("on_train_start")
            yield
        except Exception as exception:
            self.callback_handler("on_train_error", exception)
            raise
        else:
            self.callback_handler("on_train_end")
        finally:
            # After the callbacks, which may still log.
            self.logger.finish()

    def train(self) -> None:
        if self.config is None:
            raise RuntimeError("Trainer.setup(config) must be called before train()")

        accumulation_steps = self.config.gradient_accumulation_steps
        with self._train_lifecycle():
            # After `on_train_start`, which may have resumed the run from a checkpoint.
            first_epoch, first_batch = self._start_position()
            self.epoch = first_epoch
            self.step_loss = None
            if self._validation_enabled():
                self.validate()

            self._train_mode()
            for epoch in self._epochs(first_epoch):
                if self._reached_step_limit():
                    break
                self.epoch = epoch
                self.callback_handler("on_train_epoch_start")

                # A partial window left by the previous epoch must not leak into this epoch's first step.
                accumulation_step = 0
                self.step_loss = None
                self._zero_grad()

                start = first_batch if epoch == first_epoch else 0
                for batch in tqdm(self._train_batches(epoch, start), desc=f"Training (epoch {epoch})", initial=start):
                    self.callback_handler("on_train_batch_start", batch)
                    batch = self._prepare_batch(batch)

                    self.callback_handler("on_forward_start", batch)
                    loss = self._forward(batch)
                    self.callback_handler("on_forward_end", loss)

                    # Scaled here, so `_forward` is shared with validation.
                    scaled_loss = self._scale_loss(loss, accumulation_steps)
                    self.callback_handler("on_backward_start", loss)
                    self._backward(scaled_loss)
                    self.step_loss = self._accumulate_loss(self.step_loss, scaled_loss)
                    self.callback_handler("on_backward_end", loss)
                    self._clean_graph_state()

                    accumulation_step += 1
                    if accumulation_step == accumulation_steps:
                        window_loss, self.step_loss = self.step_loss, None
                        self.callback_handler("on_optimizer_step_start")
                        self._optimizer_step()
                        self.callback_handler("on_optimizer_step_end", window_loss)

                        accumulation_step = 0
                        self.global_step += 1
                        if self._validation_enabled() and self.global_step % self._validation_steps_freq() == 0:
                            self.validate()
                        self.callback_handler("on_step_end", self.global_step)

                    self.callback_handler("on_train_batch_end")
                    if self._reached_step_limit():
                        break

                self.callback_handler("on_train_epoch_end")

    def validate(self) -> None:
        self._eval_mode()
        self.callback_handler("on_validation_start")

        total_loss, num_batches = None, 0
        with self._no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                self.callback_handler("on_validation_batch_start", batch)
                batch = self._prepare_batch(batch)
                loss = self._forward(batch)
                total_loss = self._accumulate_loss(total_loss, loss)
                num_batches += 1
                self.callback_handler("on_validation_batch_end", batch, loss)
                self._clean_graph_state()

        # 0.0 when the validation set has no full batch.
        val_loss = total_loss / num_batches if num_batches else 0.0
        self.callback_handler("on_validation_end", val_loss)
        self._train_mode()

    def _validation_enabled(self) -> bool:
        return self.val_dataloader is not None and self._validation_steps_freq() > 0

    def _validation_steps_freq(self) -> int:
        return self.config.metrics.validation.steps_freq

    def _epochs(self, first: int) -> Iterable[int]:
        """The epochs the loop trains, from epoch ``first`` on: up to the limit of an epoch criterion. A step
        criterion sets no last epoch; the loop ends once ``_reached_step_limit``."""
        stopping = self.config.stopping_criteria
        return range(first, stopping.limit) if stopping.type == "epoch" else itertools.count(first)

    def _reached_step_limit(self) -> bool:
        """Whether the run made the optimizer steps a step criterion limits it to. It then ends where it is, in
        the middle of an epoch too; that epoch still gets its ``on_train_epoch_end``."""
        stopping = self.config.stopping_criteria
        return stopping.type == "step" and self.global_step >= stopping.limit

    def _check_stopping_criteria(self) -> None:
        """Raise if the run would never end: a step limit, on epochs too short to make an optimizer step."""
        if self.config.stopping_criteria.type != "step":
            return
        batches, accumulation_steps = len(self.train_dataloader), self.config.gradient_accumulation_steps
        if batches < accumulation_steps:
            raise ValueError(
                f"stopping_criteria: an epoch of {batches} batches makes no optimizer step with "
                f"gradient_accumulation_steps={accumulation_steps}, so the run would never reach its step limit"
            )

    def _start_position(self) -> tuple[int, int]:
        """The epoch the loop starts in, and the batch of it: after the ``global_step`` steps a resumed run made.

        Every epoch makes ``len(train_dataloader) // gradient_accumulation_steps`` steps, as a partial window at
        its end is dropped.
        """
        if self.global_step == 0:
            return 0, 0
        accumulation_steps = self.config.gradient_accumulation_steps
        epoch, step = divmod(self.global_step, len(self.train_dataloader) // accumulation_steps)
        return epoch, step * accumulation_steps

    def _train_batches(self, epoch: int, start: int) -> Iterable:
        """The batches of epoch ``epoch`` from batch ``start`` on, which is 0 except where a resumed run continues.

        Takes every pass over ``train_dataloader`` to yield the same batches; a subclass whose dataloader orders
        each epoch differently overrides it.
        """
        return itertools.islice(self.train_dataloader, start, None) if start else self.train_dataloader

    # ------------------------------------------------------------------ run setup

    def _check_config(self) -> None:
        """Raise if ``self.config`` can't be trained. Runs before anything is loaded or opened."""

    def _seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    @abstractmethod
    def _load_dataloaders(self) -> tuple[Iterable, Optional[Iterable]]:
        """Return ``(train_dataloader, val_dataloader)``; ``val_dataloader`` is ``None`` without validation data.

        ``len(train_dataloader)`` is the batches of an epoch.
        """

    @abstractmethod
    def _create_device_manager(self) -> Any:
        """Open the device. The returned object's ``close()`` releases it, and must be safe to call twice."""

    @abstractmethod
    def _load_model(self) -> Any:
        """Build the model to train, with its pretrained weights."""

    @abstractmethod
    def _load_optimizer(self) -> Any:
        """Build the optimizer over the model's trainable parameters."""

    # ------------------------------------------------------------------ steps

    @abstractmethod
    def _prepare_batch(self, batch: Any) -> Any:
        """Move a host batch from the dataloader to the device."""

    @abstractmethod
    def _forward(self, batch: Any) -> Any:
        """Return the loss of one batch, unscaled. Shared by training and validation."""

    def _scale_loss(self, loss: Any, divisor: int) -> Any:
        """Scale a micro-batch loss by ``1 / divisor`` for gradient accumulation."""
        return loss / divisor

    @abstractmethod
    def _backward(self, loss: Any) -> None:
        pass

    @abstractmethod
    def _accumulate_loss(self, running: Any, loss: Any) -> Any:
        """Add ``loss`` to ``running`` (``None`` for the first loss of a window), detached from the graph.

        Its result is what callbacks get as ``window_loss`` and ``val_loss``, and pass to ``loss_to_float``.
        """

    def _clean_graph_state(self) -> None:
        """Release per-batch graph state, after a training batch's backward and after each validation batch."""

    @abstractmethod
    def _optimizer_step(self) -> None:
        """Step the optimizer and zero the gradients."""

    @abstractmethod
    def _zero_grad(self) -> None:
        pass

    @abstractmethod
    def _train_mode(self) -> None:
        pass

    @abstractmethod
    def _eval_mode(self) -> None:
        pass

    @abstractmethod
    def _no_grad(self) -> Any:
        """Return a context manager that disables gradient tracking."""

    # ------------------------------------------------------------------ for callbacks and hosts

    @abstractmethod
    def loss_to_float(self, loss: Any) -> float:
        """Read a loss from ``_accumulate_loss`` (or the ``0.0`` of an empty validation set) as a float."""

    @abstractmethod
    def parameter_counts(self) -> tuple[int, int]:
        """Return ``(total_params, trainable_params)``."""

    @abstractmethod
    def current_learning_rate(self) -> list[float]:
        """The LR of each optimizer, in ``config.optimizer`` order."""

    @abstractmethod
    def save_checkpoint(self, path: str, metadata: Optional[dict] = None, optimizer: bool = False) -> None:
        """Save what the model trains as a directory at ``path``, replacing one that is there.

        ``metadata`` (step, epoch, metrics) is stored alongside, and returned by ``load_checkpoint``. With
        ``optimizer``, so is the optimizer state, which ``resume_from_checkpoint`` needs.
        """

    @abstractmethod
    def load_checkpoint(self, path: str, optimizer: bool = False) -> dict:
        """Load a directory written by ``save_checkpoint`` into the model, and with ``optimizer`` its optimizer
        state into the optimizer; return its metadata (``{}`` if none)."""

    def resume_from_checkpoint(self, path: str) -> dict:
        """Load a checkpoint saved with its optimizer state, and continue its run: ``train`` starts with the step
        after the checkpoint's, on the batches that step would have trained on. Returns the checkpoint's metadata."""
        metadata = self.load_checkpoint(path, optimizer=True)
        if "step" not in metadata:
            raise ValueError(f"{path} records no step to resume after")
        self.global_step = metadata["step"]
        return metadata

    def create_checkpoint_manager(
        self, config: CheckpointConfig, logger: Optional[TrainingLogger] = None
    ) -> CheckpointManager:
        return CheckpointManager(config, save=self.save_checkpoint, logger=logger)

    # ------------------------------------------------------------------ teardown

    def _host_attributes(self) -> frozenset[str]:
        """What the trainer held before its first run, such as its callbacks: kept by every ``cleanup``."""
        return self.__dict__.setdefault("_host_attrs", frozenset(self.__dict__) | {"_host_attrs"})

    def cleanup(self) -> None:
        """Release everything the run created, so the next ``setup`` starts clean.

        Safe to call more than once, and after a failure anywhere in ``setup`` or ``train``. Drops all run
        state (what the trainer held before its first run is kept), frees it while the device is still
        open, and closes the device last.
        """
        device_manager = self.__dict__.get("device_manager")
        host_attributes = self._host_attributes()
        try:
            if self.logger is not None:
                self.logger.finish()
        finally:
            # From a host's `finally`, the run's exception is still in flight, and its traceback keeps the
            # failed step's frames alive, and with them device tensors.
            exception = sys.exc_info()[1]
            if exception is not None:
                traceback.clear_frames(exception.__traceback__)
            for attr in list(self.__dict__):
                if attr not in host_attributes:
                    delattr(self, attr)
            self._reset_run_state()
            gc.collect()
            if device_manager is not None:
                device_manager.close()
