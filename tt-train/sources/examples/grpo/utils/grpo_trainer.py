from dataclasses import dataclass
import inspect
from typing import List, Callable, Sequence

from ttml.common.config import DeviceConfig, TransformerConfig
from utils.inference import InferenceCtx, setup_inference
from utils.inference import completion_batched_multiple_prompts, deallocate_tensors
from utils.loss import compute_nlog_probs, compute_grpo_loss
import os
import numpy as np
import ttml
import ttnn
from safetensors.torch import save_file
from ttml.common.utils import create_optimizer, no_grad


class TrainerCallback:
    def on_step_end(self, trainer, step, metrics):
        pass

    def on_train_end(self, trainer):
        pass


@dataclass
class GrpoConfig:
    epsilon: float
    batch_size: int
    micro_batch_size: int
    num_iterations: int
    gradient_accumulation_steps: int
    logging_steps: int
    output_dir: str
    checkpointing: bool
    checkpoint_interval: int
    prompts_to_train: int
    temperature: float
    max_completion_length: int
    num_generations: int
    warmup_steps: int


def dispatch_reward(reward_func, completions, prompts, batch_columns):
    sig = inspect.signature(reward_func)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    data_pool = {"completions": completions, "prompts": prompts, **batch_columns}

    if has_kwargs:
        return reward_func(**data_pool)

    call_kwargs = {name: data_pool[name] for name in params if name in data_pool}
    return reward_func(**call_kwargs)


def compute_advantages(rewards_np, group_size):
    advantages_np = np.zeros_like(rewards_np)
    for start in range(0, len(rewards_np), group_size):
        end = min(start + group_size, len(rewards_np))
        rg = rewards_np[start:end]
        advantages_np[start:end] = rg - float(rg.mean())
    return advantages_np


def iter_batched_completions(
    ctx: InferenceCtx,
    prompts: Sequence[List[int]],
    batch_columns: dict,
    batch_size: int = 32,
):
    n = len(prompts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompt_batch = list(prompts[start:end])

        completions_batch = completion_batched_multiple_prompts(ctx, prompt_batch)

        prompt_batch_expanded = [item for item in prompt_batch for _ in range(ctx.group_size)]
        columns_expanded = {
            k: [v for v in col[start:end] for _ in range(ctx.group_size)] for k, col in batch_columns.items()
        }

        assert len(prompt_batch_expanded) == len(completions_batch)
        yield prompt_batch_expanded, completions_batch, columns_expanded


def iter_micro_batch(prompts, completions, micro_batch_size=16):
    for start in range(0, len(completions), micro_batch_size):
        end = min(start + micro_batch_size, len(completions))

        yield prompts[start:end], completions[start:end]


def save_checkpoint(model, step, output_dir, dp_composer=None):
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"grpo_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    tensors = {name: param.to_numpy(ttnn.DataType.FLOAT32, dp_composer) for name, param in model.parameters().items()}
    save_file(tensors, os.path.join(ckpt_dir, f"model.safetensors"))


class GrpoTrainer:
    def __init__(
        self,
        model_source: str,
        dataset,
        config: GrpoConfig,
        reward_func: Callable,
        transformer_config: dict,
        optimizer_config: dict,
        device_config: dict,
        callbacks: list = None,
    ):
        self.model_source = model_source
        self.dataset = dataset
        self.config = config
        self.reward_func = reward_func
        self.transformer_config = TransformerConfig({"transformer_config": transformer_config})
        self.device_config = DeviceConfig({"device_config": device_config})
        self.optimizer_config_dict = optimizer_config
        self.callbacks = callbacks or []
        self.model = None

    def _notify(self, method, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method)(self, *args, **kwargs)

    def train(self):
        grpo_cfg = self.config

        if self.device_config.total_devices() > 1:
            ttml.core.distributed.enable_fabric(self.device_config.total_devices())

        ttml.autograd.AutoContext.get_instance().open_device(
            self.device_config.mesh_shape, self.device_config.device_ids
        )

        inference_ctx = setup_inference(self.config, self.transformer_config, self.device_config, self.model_source)
        self.model = inference_ctx.tt_model

        optimizer = create_optimizer(inference_ctx.tt_model, self.optimizer_config_dict)
        base_lr = optimizer.get_lr()

        dataset = self.dataset.select(range(min(grpo_cfg.prompts_to_train, len(self.dataset))))
        prompts = [inference_ctx.tokenizer.encode(row["prompt"]) for row in dataset]
        extra_columns = {k: list(dataset[k]) for k in dataset.column_names if k != "prompt"}

        num_batches = 0
        num_steps = 0
        accum_count = 0
        grad_accum = grpo_cfg.gradient_accumulation_steps
        accum_rewards = []
        accum_completion_lens = []

        optimizer.zero_grad()

        for prompts_batch, completions_batch, dataset_columns_dict in iter_batched_completions(
            inference_ctx, prompts, extra_columns, grpo_cfg.batch_size
        ):
            num_batches += 1

            completions_strs = [inference_ctx.tokenizer.decode(c, skip_special_tokens=True) for c in completions_batch]
            prompts_strs = [inference_ctx.tokenizer.decode(p) for p in prompts_batch]
            rewards = dispatch_reward(self.reward_func, completions_strs, prompts_strs, dataset_columns_dict)
            rewards_np = np.array(rewards, dtype=np.float32)

            advantages_np = compute_advantages(rewards_np, inference_ctx.group_size)
            accum_rewards.append(rewards_np)
            accum_completion_lens.extend(len(c) for c in completions_batch)

            probs_old_list = []
            inference_ctx.tt_model.eval()
            with no_grad():
                for p, c in iter_micro_batch(prompts_batch, completions_batch, grpo_cfg.micro_batch_size):
                    nlog_old, mask, Tp = compute_nlog_probs(inference_ctx, p, c)
                    nlog_old.set_requires_grad(False)
                    mask.set_requires_grad(False)
                    probs_old_list.append((nlog_old, mask, Tp))

            for mini_epoch in range(grpo_cfg.num_iterations):
                inference_ctx.tt_model.train()

                for i, (p, c) in enumerate(
                    iter_micro_batch(prompts_batch, completions_batch, grpo_cfg.micro_batch_size),
                ):
                    B = len(c)
                    adv_slice = advantages_np[i * grpo_cfg.micro_batch_size : i * grpo_cfg.micro_batch_size + B]

                    adv_tt = ttml.autograd.Tensor.from_numpy(
                        adv_slice.reshape((B, 1)),
                        ttnn.Layout.ROW_MAJOR,
                        ttnn.DataType.BFLOAT16,
                        inference_ctx.dp_mapper,
                    )
                    adv_tt.set_requires_grad(False)

                    nlog_old, mask_old, Tp = probs_old_list[i]
                    nlog_probs_new, mask_new, _ = compute_nlog_probs(inference_ctx, p, c)

                    loss = compute_grpo_loss(
                        nlog_old,
                        nlog_probs_new,
                        mask_old,
                        adv_tt,
                        B,
                        Tp,
                        len(prompts_batch) * grad_accum,
                        grpo_cfg.epsilon,
                        inference_ctx,
                    )

                    loss.backward(retain_graph=False)

                    deallocate_tensors([nlog_probs_new, mask_new, adv_tt, loss])

                accum_count += 1

                if accum_count == grad_accum:
                    warmup_factor = (
                        1.0 if grpo_cfg.warmup_steps == 0 else min(1.0, (num_steps + 1) / grpo_cfg.warmup_steps)
                    )
                    optimizer.set_lr(base_lr * warmup_factor)

                    if inference_ctx.dp_mapper is not None:
                        ttml.core.distributed.synchronize_gradients(inference_ctx.tt_model.parameters())

                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0

                    num_steps += 1
                    all_rewards = np.concatenate(accum_rewards)
                    mean_reward = float(all_rewards.mean())
                    mean_completion_len = sum(accum_completion_lens) / max(len(accum_completion_lens), 1)

                    if grpo_cfg.logging_steps > 0 and num_steps % grpo_cfg.logging_steps == 0:
                        step_metrics = {
                            "reward_mean": mean_reward,
                            "reward_std": float(all_rewards.std()),
                            "mean_completion_len": mean_completion_len,
                            "lr": base_lr * warmup_factor,
                        }
                        self._notify("on_step_end", num_steps, step_metrics)

                    accum_rewards.clear()
                    accum_completion_lens.clear()

                    if grpo_cfg.checkpointing and num_steps % grpo_cfg.checkpoint_interval == 0:
                        save_checkpoint(
                            inference_ctx.tt_model, num_steps, grpo_cfg.output_dir, inference_ctx.dp_composer
                        )

            for nlog_old, mask_old, _ in probs_old_list:
                deallocate_tensors([nlog_old, mask_old])

        self._notify("on_train_end")
