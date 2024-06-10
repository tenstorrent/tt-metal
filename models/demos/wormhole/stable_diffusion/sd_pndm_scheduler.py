# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import ttnn


@dataclass
class TtSchedulerOutput:
    prev_sample: ttnn.Tensor


class TtPNDMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        skip_prk_steps: bool = False,
        set_alpha_to_one: bool = False,
        prediction_type: str = "epsilon",
        steps_offset: int = 0,
        device=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.skip_prk_steps = skip_prk_steps
        self.prediction_type = prediction_type
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.device = device
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.alphas_cumprod = self.alphas_cumprod.tolist()
        self.init_noise_sigma = 1.0
        self.pndm_order = 4

        # running values
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

        # setable values
        self.num_inference_steps = None
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
        self._timesteps += self.steps_offset

        if self.skip_prk_steps:
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]])[
                ::-1
            ].copy()
        else:
            prk_timesteps = np.array(self._timesteps[-self.pndm_order :]).repeat(2) + np.tile(
                np.array([0, self.num_train_timesteps // num_inference_steps // 2]), self.pndm_order
            )
            self.prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][::-1].copy()

        timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]).astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.prk_timesteps = self.prk_timesteps.tolist()
        self.ets = []
        self.counter = 0

    def step(
        self,
        model_output: ttnn.Tensor,
        timestep: int,
        sample: ttnn.Tensor,
        return_dict: bool = True,
    ) -> Union[TtSchedulerOutput, Tuple]:
        if self.counter < len(self.prk_timesteps) and not self.skip_prk_steps:
            return self.step_prk(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)
        else:
            return self.step_plms(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)

    def step_prk(
        self,
        model_output: ttnn.Tensor,
        timestep: int,
        sample: ttnn.Tensor,
        return_dict: bool = True,
    ) -> Union[TtSchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        diff_to_prev = 0 if self.counter % 2 else self.num_train_timesteps // self.num_inference_steps // 2
        prev_timestep = timestep - diff_to_prev
        timestep = self.prk_timesteps[self.counter // 4 * 4]

        if self.counter % 4 == 0:
            self.cur_model_output += ttnn.multiply(model_output, 1 / 6)
            self.ets.append(model_output)
            self.cur_sample = sample
        elif (self.counter - 1) % 4 == 0:
            self.cur_model_output += ttnn.multiply(model_output, 1 / 3)
        elif (self.counter - 2) % 4 == 0:
            self.cur_model_output += ttnn.multiply(model_output, 1 / 3)
        elif (self.counter - 3) % 4 == 0:
            model_output = ttnn.add(self.cur_model_output, ttnn.multiply(model_output, 1 / 6))
            self.cur_model_output = 0

        # cur_sample should not be `None`
        cur_sample = self.cur_sample if self.cur_sample is not None else sample

        prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
        self.counter += 1

        if not return_dict:
            return (prev_sample,)

        return TtSchedulerOutput(prev_sample=prev_sample)

    def step_plms(
        self,
        model_output: ttnn.Tensor,
        timestep: int,
        sample: ttnn.Tensor,
        return_dict: bool = True,
    ) -> Union[TtSchedulerOutput, Tuple]:
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        if len(self.ets) == 1 and self.counter == 0:
            model_output = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = ttnn.mul((model_output + self.ets[-1]), 1 / 2)
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            model_output = ttnn.mul((3 * self.ets[-1] - self.ets[-2]), 1 / 2)
        elif len(self.ets) == 3:
            model_output = ttnn.mul((23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]), 1 / 12)
        else:
            model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
        self.counter += 1

        if not return_dict:
            return (prev_sample,)

        return TtSchedulerOutput(prev_sample=prev_sample)

    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.prediction_type == "v_prediction":
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        elif self.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon` or `v_prediction`"
            )
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)
        prev_sample = sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output * (
            1 / model_output_denom_coeff
        )

        return prev_sample
