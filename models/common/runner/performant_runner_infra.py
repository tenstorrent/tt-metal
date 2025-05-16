# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class PerformantRunnerInfra(ABC):
    @abstractmethod
    def setup_dram_sharded_input(self, device):
        """
        Setup the DRAM sharded input for the model.
        """

    @abstractmethod
    def dealloc_output(self):
        """
        Deallocate the output tensor.
        """

    @abstractmethod
    def run(self):
        """
        Run the model.
        """

    @abstractmethod
    def validate(self):
        """
        Validate the model output.
        """
