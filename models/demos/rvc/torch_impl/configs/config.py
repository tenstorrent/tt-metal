# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch


class Config:
    def __init__(self):
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.instead: str | None = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def params_config(self) -> tuple:
        # 5G GPU_RAM conf
        # note: keep x_query < x_center/ 2 to voice degradation
        x_pad = 1
        x_query = 3
        x_center = 12
        x_max = 16

        # x_pad = 1
        # x_query = 6
        # x_center = 24
        # x_max = 16

        # x_pad = 1
        # x_query = 6
        # x_center = 38
        # x_max = 41

        # x_pad = 1
        # x_query = 2
        # x_center = 6
        # x_max = 8

        # x_pad = 1
        # x_query = 1
        # x_center = 3
        # x_max = 4

        return x_pad, x_query, x_center, x_max

    def use_cpu(self) -> None:
        self.device = self.instead = "cpu"
        self.params_config()

    def device_config(self) -> tuple:
        return self.params_config()
