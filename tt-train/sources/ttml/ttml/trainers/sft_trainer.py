# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Supervised Fine-Tuning (SFT) trainer."""

from __future__ import annotations

import logging
import os
import pickle
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ttml.modules.lora import LoraConfig

import numpy as np
import ttnn
from tqdm import tqdm

logger = logging.getLogger(__name__)

import ttml
from ttml.common.profiler_utils import profiler_marker
from ttml.common.utils import no_grad
from ttml.datasets import Batch, TTMLDataloader
from ttml.trainers.callback import TrainerCallback


@dataclass
class SFTConfig:
    """Configuration for :class:`SFTTrainer`.

    Training-loop cadence and default learning-rate settings live here.
    Optimizer-specific parameters should be passed via the *optimizer*
    argument of :class:`SFTTrainer` instead.

    Attributes:
        max_steps: Total number of optimiser steps.
        gradient_accumulation_steps: Micro-batches accumulated before each
            optimiser step.
        eval_interval: Run evaluation every *N* steps (0 to disable).
        save_interval: Save a checkpoint every *N* steps (0 to disable).
        checkpoint_dir: Directory for checkpoint files.
        seed: Optional RNG seed for reproducibility.
        max_seq_len: Maximum sequence length (currently unused; kept for
            backward compatibility).
        learning_rate: Peak learning rate, also used to initialise the default
            AdamW optimizer when no explicit optimizer is provided.
        warmup_steps: Linear warmup steps for the default schedule.
        max_grad_norm: Maximum gradient norm for clipping (0 to disable).
        log_interval: Update the progress bar / fire ``on_step_end``
            callbacks every *N* steps.
        gradient_checkpointing: Enable activation recomputation to reduce
            memory usage at the cost of ~30 % extra compute.  Sets the
            model's ``runner_type`` to ``MemoryEfficient``.
    """

    # ---- loop ----
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    eval_interval: int = 200
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    # If non-empty, checkpoint files are named "{checkpoint_prefix}_step_{N}.pkl"
    # instead of "step_{N}.pkl".
    checkpoint_prefix: str = ""
    seed: Optional[int] = None
    max_seq_len: int = 1024
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    max_grad_norm: float = 0.0
    log_interval: int = 1
    gradient_checkpointing: bool = False
    # If True, tqdm progress bar (and its loss/lr postfix) is suppressed.
    disable_progress_bar: bool = False


class SFTTrainer:
    """Supervised fine-tuning trainer for Tenstorrent devices.

    The trainer owns the training loop, optimizer, and scheduler.  It knows
    nothing about model internals or dataset structure — only the interfaces
    defined by :class:`~ttml.modules.module_base.AbstractModuleBase` and
    :class:`~ttml.datasets.TTMLDataloader`.

    Multi-device training (DDP / FSDP / HSDP): supply a **collate function**
    that shards batch tensors across the mesh (via
    ``shard_tensor_to_mesh_mapper``); gradients are all-reduced automatically
    each step across the present ``dp`` / ``fsdp`` axes.

    Loss aggregation across devices is handled automatically via a default
    ``concat_mesh_to_tensor_composer(device, 0)``.  Pass a custom
    ``loss_composer`` to override.

    Example::

        from ttml.trainers import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            config=SFTConfig(max_steps=5_000, learning_rate=2e-5, max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": 2e-5, "weight_decay": 0.01},
        )
        trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataloader: TTMLDataloader,
        eval_dataloader: Optional[TTMLDataloader],
        config: SFTConfig,
        optimizer: Any = None,
        peft_config: Optional[LoraConfig] = None,
        lr_schedule: Optional[Callable[[int], float]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        compute_loss_func: Optional[Callable] = None,
        loss_composer: Any = None,
        attention_mask: Any = None,
        checkpoint_saver: Optional[Callable[["SFTTrainer", str], None]] = None,
        checkpoint_loader: Optional[Callable[["SFTTrainer", str], int]] = None,
    ) -> None:
        """
        Args:
            model: Model to fine-tune.
            train_dataloader: Training data loader.
            eval_dataloader: Evaluation data loader (may be ``None``).
            config: Trainer configuration.
            optimizer: One of:
                - A ``ttml.optimizers.OptimizerBase`` instance (used directly).
                - A ``dict`` with at least a ``"type"`` key, forwarded to
                  ``ttml.optimizers.create_optimizer(dict, model.parameters())``.
                - ``None`` (default) -- creates a default AdamW with
                  ``learning_rate=config.learning_rate``.
            peft_config: Optional PEFT configuration (e.g. :class:`LoraConfig`).
                When provided the model is automatically wrapped with the
                appropriate adapter (currently :class:`LoraModel`).
            lr_schedule: Optional callable ``step -> lr``.
            callbacks: Optional list of :class:`TrainerCallback` instances.
            compute_loss_func: Optional callable ``(model_output, batch) -> loss``
                that replaces the default masked cross-entropy.  Must return a
                scalar :class:`ttml.autograd.Tensor` loss already reduced over
                batch and sequence dimensions.  When this hook is provided the
                caller is fully responsible for applying ``batch.loss_mask``
                (or any other prompt/pad masking strategy); the default masking
                logic is bypassed entirely.
            loss_composer: Optional mesh composer passed to
                ``loss.to_numpy(composer=...)`` when extracting scalar loss
                values for logging. By default a `concat_mesh_to_tensor_composer` with dim=0 is used.
                (Covers both single-device and multi-device DDP configurations.)
            attention_mask: Optional attention mask passed as the second
                argument to ``model(input_ids, mask)``.  ``None`` (default)
                lets the model generate a causal mask on the fly.
            checkpoint_saver: Optional ``(trainer, path) -> None`` that
                replaces the default thin save (which stores only
                ``{step, model_state}``).  Caller is free to serialise
                tokenizer, configs, RNG, etc.
            checkpoint_loader: Optional ``(trainer, path) -> int`` that
                restores ``model.parameters()`` from ``path`` and returns
                the step the checkpoint was taken at.  Called by
                :meth:`load_checkpoint` to support resume.

        Note:
            When ``config.gradient_checkpointing`` is ``True`` the model's
            ``runner_type`` is switched to ``MemoryEfficient``, which
            recomputes activations during backward instead of caching them.
        """
        if config.seed is not None:
            ttml.autograd.AutoContext.get_instance().set_seed(config.seed)

        if peft_config is not None:
            from ttml.modules.lora import LoraModel

            model = LoraModel(model, peft_config)

        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing(model)

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.step = 0  # 0-based; incremented after each optimizer step
        # Axes to all-reduce gradients across each step; empty (single-device / TP-only) = no-op.
        self._grad_sync_axes = self._resolve_grad_sync_axes()
        self._validate_clip_grad_norm()

        self._optimizer = self._build_optimizer(optimizer)
        self._lr_schedule = lr_schedule if lr_schedule is not None else self._build_lr_schedule()
        self._attention_mask = attention_mask
        self._loss_fn = self._build_loss_fn()
        self._callbacks = callbacks or []
        self._compute_loss_override = compute_loss_func
        self._loss_composer = self._build_loss_composer(loss_composer)
        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader
        self._mask_validated = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def optimizer(self):
        return self._optimizer

    def train(self) -> None:
        """Run the full training loop.

        The dataloader is cycled automatically: when one epoch ends the
        iterator is reset so that ``max_steps`` is always reached regardless
        of dataset size.
        """
        self.model.train()
        for cb in list(self._callbacks):
            cb.on_train_begin(self)
        data_iter = iter(self.train_dataloader)
        cfg = self.config

        def _next_batch() -> Batch:
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                return next(data_iter)

        # Compilation finishes after the first iteration of THIS run; track the starting step so the
        # marker fires on resume too, not only when self.step happens to equal 1.
        start_step = self.step
        bar = tqdm(range(self.step, cfg.max_steps), desc="SFTTrainer", disable=cfg.disable_progress_bar)
        for _ in bar:
            # self.step is 0-based so external lr_schedule callables (e.g.
            # SpeedrunScheduler.lr_at) receive the expected step index.
            lr = self._lr_schedule(self.step)
            self._optimizer.set_lr(lr)
            self._optimizer.zero_grad()

            micro_losses = []
            for _ in range(cfg.gradient_accumulation_steps):
                batch = _next_batch()
                profiler_marker(None, "dataloader_step_done")

                loss = self._compute_loss(batch)
                micro_loss = float(loss.to_numpy(ttnn.DataType.FLOAT32, composer=self._loss_composer).mean())
                micro_losses.append(micro_loss)
                profiler_marker(None, "forward_pass_done")
                for cb in list(self._callbacks):
                    cb.on_after_forward(self, batch, micro_loss)

                if cfg.gradient_accumulation_steps > 1:
                    loss = ttml.ops.binary.mul(loss, 1.0 / cfg.gradient_accumulation_steps)
                loss.backward(False)
                profiler_marker(None, "backward_pass_done")
                for cb in list(self._callbacks):
                    cb.on_after_backward(self, batch)
                ttml.autograd.AutoContext.get_instance().reset_graph()

            if self._grad_sync_axes:
                ttml.sync_gradients(self.model.parameters(), axis_names=self._grad_sync_axes)

            for cb in list(self._callbacks):
                cb.on_before_optimizer_step(self)

            if cfg.max_grad_norm > 0:
                ttml.core.clip_grad_norm(self.model.parameters(), cfg.max_grad_norm, 2.0, False)

            profiler_marker(None, "gradient_sync_done")

            self._optimizer.step()
            self.step += 1

            profiler_marker(None, "optimizer_step_done")

            step_loss = float(np.mean(micro_losses))
            if cfg.log_interval > 0 and self.step % cfg.log_interval == 0:
                bar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr:.2e}"}, refresh=False)
            for cb in list(self._callbacks):
                cb.on_step_end(self, self.step, step_loss, lr)

            if cfg.eval_interval > 0 and self.step % cfg.eval_interval == 0:
                if self.eval_dataloader is not None:
                    val_loss = self._eval()
                    bar.set_postfix(
                        {
                            "loss": f"{step_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        },
                        refresh=False,
                    )
                    for cb in list(self._callbacks):
                        cb.on_eval_end(self, self.step, val_loss)

            if cfg.save_interval > 0 and self.step % cfg.save_interval == 0 and self.step > 0:
                self._save_checkpoint()
                save_path = self._checkpoint_path()
                for cb in list(self._callbacks):
                    cb.on_save(self, self.step, save_path)

            profiler_marker(None, f"iteration_{self.step}", dump_results=True)
            if self.step == start_step + 1:
                profiler_marker(None, "compilation_finished")

        for cb in list(self._callbacks):
            cb.on_train_end(self)

    def _checkpoint_path(self) -> str:
        """Path used for the current-step checkpoint (honors `config.checkpoint_prefix`)."""
        name = f"step_{self.step}.pkl"
        if self.config.checkpoint_prefix:
            name = f"{self.config.checkpoint_prefix}_{name}"
        return os.path.join(self.config.checkpoint_dir, name)

    def remove_callback(self, cb: TrainerCallback) -> None:
        """Detach ``cb`` from the trainer's callback list.

        Safe to call from inside a callback method — event iteration takes a
        snapshot of the list, so the removed callback finishes the current
        event but receives no further events.
        """
        try:
            self._callbacks.remove(cb)
        except ValueError:
            # cb is not registered — removal is idempotent, so a missing entry is a no-op.
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: Batch):
        """Forward pass + loss computation.

        If *compute_loss_func* was provided it is called instead of the
        default masked cross-entropy.
        """
        logits = self.model(batch.input_ids, self._attention_mask)  # [B, 1, T, V]
        if self._compute_loss_override is not None:
            return self._compute_loss_override(logits, batch)

        # Validate loss_mask normalisation: for a standard SFT collate the mask
        # should sum to B*T (one entry per token position).  A large deviation
        # usually indicates a custom collate that forgot to normalise.
        # Checked once on the first batch — repeating it would force a
        # device→host sync every step.
        if batch.loss_mask is not None and not self._mask_validated:
            self._mask_validated = True
            mask_np = batch.loss_mask.to_numpy(ttnn.DataType.FLOAT32, composer=self._loss_composer)
            B, _, T, _ = mask_np.shape
            expected = float(B * T)
            actual = float(mask_np.sum())
            # use relative tolerance to avoid BF16 precision issues
            if abs(actual - expected) / expected > 0.01:
                logger.warning(
                    "loss_mask sum (%.2f) differs from expected B*T (%d). "
                    "If you are using a custom collate function, make sure "
                    "the mask is correctly normalised.",
                    actual,
                    int(expected),
                )

        loss = self._loss_fn(logits, batch.labels, ttml.ops.ReduceType.NONE)  # [B, 1, T, 1]
        loss = loss * batch.loss_mask  # zero out prompt + padding
        return ttml.ops.unary.mean(loss)

    def _eval(self) -> float:
        """Run one pass over the eval dataloader and return mean loss."""
        self.model.eval()
        losses = []
        with no_grad():
            for batch in self.eval_dataloader:
                loss = self._compute_loss(batch)
                losses.append(float(loss.to_numpy(ttnn.DataType.FLOAT32, composer=self._loss_composer).mean()))
                ttml.autograd.AutoContext.get_instance().reset_graph()
        self.model.train()
        return float(np.mean(losses))

    def _save_checkpoint(self) -> None:
        """Persist a checkpoint via ``checkpoint_saver`` (or the default).

        The default saver stores ``{step, model_state}`` (model parameters as
        FLOAT32 numpy arrays).  When ``checkpoint_saver`` was provided to
        ``__init__`` it is called instead — callers use this hook to embed
        tokenizers, configs, RNG state, etc.

        .. note::
            When a ``peft_config`` is used and the model is wrapped in
            :class:`LoraModel`, the default saver currently saves *all*
            parameters (base + LoRA).  Ideally only the LoRA adapter weights
            should be persisted to keep checkpoints small.

            TODO: when ``peft_config`` is set, filter ``self.model.parameters()``
            to save only LoRA adapter parameters (e.g. those whose name contains
            ``lora_``).
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = self._checkpoint_path()

        if self._checkpoint_saver is not None:
            self._checkpoint_saver(self, path)
            return

        state = {}
        for name, param in self.model.parameters().items():
            tensor = param.tensor if hasattr(param, "tensor") else param
            # In multi-device (DDP) setups parameter tensors are distributed across
            # the mesh, so the underlying host storage carries the full mesh shape
            # (e.g. [1, 64]) and ``to_numpy()`` without a composer trips the
            # single-buffer assertion in ``host_buffer::get_host_buffer``.
            # ``ttnn.get_device_tensors(device_tensor)[0]`` does not help: the
            # returned tensor still references the parent mesh storage and hits
            # the same error once moved to host.
            #
            # Aggregate via the loss composer (concat along tensor dim 0). Since
            # DDP replicates weights, every device holds an identical copy, so
            # the first ``per_replica_dim0`` rows of the concatenated array are
            # exactly one replica.
            # TODO: support TP / sharded parameters with a model-aware composer.
            param_np = tensor.to_numpy(ttnn.DataType.FLOAT32, composer=self._loss_composer)
            per_replica_dim0 = tensor.shape()[0]
            state[name] = param_np[:per_replica_dim0]

        with open(path, "wb") as f:
            pickle.dump({"step": self.step, "model_state": state}, f)

    def load_checkpoint(self, path: str) -> int:
        """Restore training state from a checkpoint and advance ``self.step``.

        Delegates to ``checkpoint_loader`` if one was supplied; otherwise reads
        the default ``{step, model_state}`` format and copies parameters via
        ``param.assign(...)``.  Returns the step the checkpoint was taken at.
        Call before :meth:`train` — the loop iterates from ``self.step`` to
        ``cfg.max_steps`` so resume picks up exactly where the run stopped.
        """
        if self._checkpoint_loader is not None:
            step = int(self._checkpoint_loader(self, path))
            self.step = step
            return step

        import ml_dtypes

        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        step = int(ckpt["step"])
        model_state = ckpt["model_state"]

        params = self.model.parameters()
        for name, arr in model_state.items():
            if name not in params:
                continue
            arr_bf16 = arr.astype(ml_dtypes.bfloat16)
            restored = ttml.autograd.Tensor.from_numpy(
                arr_bf16, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
            )
            params[name].assign(restored)

        self.step = step
        return step

    def _build_optimizer(self, optimizer: Any):
        """Resolve the *optimizer* argument into an ``OptimizerBase``.

        * ``OptimizerBase`` instance -- returned as-is.
        * ``dict`` -- forwarded to ``ttml.optimizers.create_optimizer``.
        * ``None`` -- a default AdamW is created with ``learning_rate=config.learning_rate``.
        """
        if isinstance(optimizer, ttml.optimizers.OptimizerBase):
            return optimizer

        if isinstance(optimizer, dict):
            return ttml.optimizers.create_optimizer(optimizer, self.model.parameters())

        # Default: AdamW with lr from config
        return ttml.optimizers.create_optimizer(
            {"type": "AdamW", "lr": self.config.learning_rate},
            self.model.parameters(),
        )

    def _build_lr_schedule(self):
        """Return a callable ``step -> lr`` implementing linear warmup then constant."""
        peak_lr = self.config.learning_rate
        warmup = max(0, self.config.warmup_steps)

        def schedule(step: int) -> float:
            if warmup > 0 and step < warmup:
                return peak_lr * (step + 1) / warmup
            return peak_lr

        return schedule

    def _build_loss_composer(self, loss_composer: Optional[Any] = None):
        """Create a mesh composer for aggregating loss across devices."""
        if loss_composer is not None:
            return loss_composer
        device = ttml.autograd.AutoContext.get_instance().get_device()
        return ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    def _build_loss_fn(self):
        """Pick the cross-entropy variant matching the LM head's logit sharding.

        When the active mesh has a TP axis with size > 1 the conventional
        SFTTrainer model shape (Llama with ``use_tp=True`` or
        ``models.distributed.{llama,gpt2}``) emits vocab-sharded logits via
        ``ColumnParallelLinear(gather_output=False)``.  Plain
        ``cross_entropy_loss`` would compute its softmax denominator over each
        TP shard instead of the full vocab — wrong by construction — so we
        route through ``vocab_parallel_cross_entropy_loss`` with
        ``cluster_axis`` bound to the TP axis.

        The wrapper preserves the ``(logits, labels, reduce)`` signature of
        ``cross_entropy_loss`` so the masked-CE flow in ``_compute_loss``
        (``ReduceType.NONE`` per-token loss → ``* loss_mask`` → ``mean``) is
        unchanged: the per-token shape ``[B, 1, T, 1]`` matches between the
        two ops, and the upstream-grad broadcast in vocab-parallel CE handles
        the ``loss_mask`` weighting in backward.

        Pipeline-parallel models whose last stage gathers logits to the full
        vocab fall outside this auto-detect rule (mesh has TP axis but the
        loss should stay full-vocab CE); for those, pass ``compute_loss_func``
        when constructing the trainer.
        """
        mesh = ttml.mesh()
        if mesh.has_axis("tp") and mesh.axis_size("tp") > 1:
            tp_cluster_axis = mesh.axis_index("tp")

            def vocab_parallel_loss(logits, labels, reduce):
                return ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
                    logits, labels, cluster_axis=tp_cluster_axis, reduce=reduce
                )

            return vocab_parallel_loss
        return ttml.ops.loss.cross_entropy_loss

    @staticmethod
    def _enable_gradient_checkpointing(model: Any) -> None:
        """Switch the model's runner type to ``MemoryEfficient``.

        Works with both bare models (Llama, NanoGPT) and LoraModel wrappers.
        The model config is a frozen dataclass, so we replace it wholesale.
        """
        target = model.model if hasattr(model, "model") else model
        cfg = getattr(target, "config", None)
        if cfg is None or not hasattr(cfg, "runner_type"):
            return
        target.config = replace(cfg, runner_type=ttml.models.RunnerType.MemoryEfficient)

    def _validate_clip_grad_norm(self) -> None:
        """clip_grad_norm computes a per-device norm with no cross-mesh reduction, so it is only correct
        when every parameter is replicated. Reject sharded (FSDP/TP) params up front rather than silently
        clipping by a wrong per-shard norm."""
        if self.config.max_grad_norm <= 0:
            return
        if any(not ttml.Sharding.from_tensor(p).is_fully_replicated for _, p in self.model.parameters().items()):
            raise ValueError(
                "clip_grad_norm is not supported with sharded parameters (FSDP/TP): each device holds "
                "only a shard, so the per-shard norm is wrong"
            )

    @staticmethod
    def _resolve_grad_sync_axes() -> tuple[str, ...]:
        """Mesh axes to all-reduce gradients across each step.

        The subset of ``("dp", "fsdp")`` present on the active mesh with size > 1; empty otherwise
        (single device / TP-only). FSDP-sharded params are skipped per-axis by ``ttml.sync_gradients``
        (``fully_shard``'s backward hooks already reduce-scattered them), so this covers DDP, the dp axis
        of HSDP, and non-sharded params on the fsdp axis.
        """
        mesh = ttml.maybe_mesh()
        if mesh is None:
            return ()
        return tuple(name for name in ("dp", "fsdp") if mesh.has_axis(name) and mesh.axis_size(name) > 1)
