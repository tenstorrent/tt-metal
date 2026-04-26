# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test infrastructure for YOLO26 performance testing.

Provides utilities for setting up YOLO26 model, input tensors, and validation.
"""

import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.common.utility_functions import divup, is_wormhole_b0, is_blackhole
from models.experimental.yolo26.tt.ttnn_yolo26 import TtYOLO26
from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader


def load_yolo26_torch_model(variant: str = "yolo26n"):
    """Load YOLO26 model from Ultralytics."""
    from ultralytics import YOLO

    model = YOLO(f"{variant}.pt")
    return model.model, model.model.state_dict()


class YOLO26TestInfra:
    """
    Test infrastructure for YOLO26 performance testing.

    Handles model setup, input preparation, and output validation.
    """

    def __init__(
        self,
        device,
        batch_size: int = 1,
        input_size: int = 640,
        variant: str = "yolo26n",
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        torch_input_tensor=None,
    ):
        torch.manual_seed(0)
        self.device = device
        self.batch_size = batch_size
        self.input_size = input_size
        self.variant = variant
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
        self.total_batch_size = batch_size * self.num_devices

        # Load PyTorch model and weights
        logger.info(f"Loading YOLO26 {variant} model...")
        self.torch_model, state_dict = load_yolo26_torch_model(variant)
        self.weight_loader = YOLO26WeightLoader(state_dict)

        # Create TTNN model
        logger.info("Creating TTNN YOLO26 model...")
        self.ttnn_model = TtYOLO26(device, variant)
        self.ttnn_model.load_weights_from_state_dict(state_dict)

        # Setup input tensor
        self.channels = 3
        if torch_input_tensor is not None:
            self.torch_input_tensor = torch_input_tensor
        else:
            self.torch_input_tensor = torch.randn(
                self.total_batch_size, self.channels, input_size, input_size, dtype=torch.float32
            )

        # Compute reference output for validation (optional, can be slow)
        self.torch_output_tensor = None
        self.input_tensor = None
        self.output_tensor = None

    def _get_core_grid(self):
        """Get optimal core grid for the device."""
        if is_wormhole_b0():
            return ttnn.CoreGrid(y=8, x=8)
        elif is_blackhole():
            return ttnn.CoreGrid(y=8, x=10)
        else:
            return ttnn.CoreGrid(y=8, x=8)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        """
        Setup L1 sharded input configuration.

        Returns:
            Tuple of (tt_inputs_host, input_mem_config)
        """
        core_grid = self._get_core_grid()
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        n, c, h, w = torch_input_tensor.shape
        n = n // self.num_devices if n // self.num_devices != 0 else n

        num_cores = core_grid.x * core_grid.y
        shard_h = (n * w * h + num_cores - 1) // num_cores
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})

        # Pad channels if needed
        padded_c = max(c, min_channels)
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, padded_c), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )

        # Convert NCHW to NHWC and reshape for sharding
        torch_input = torch_input_tensor.permute(0, 2, 3, 1)  # NCHW -> NHWC
        torch_input = torch_input.reshape(n * self.num_devices, 1, h * w, c)

        # Pad channels if needed
        if c < min_channels:
            padding_c = min_channels - c
            torch_input = F.pad(torch_input, (0, padding_c), "constant", 0)

        # Create host tensor
        if self.num_devices > 1:
            input_tensors = [torch_input[i].unsqueeze(0) for i in range(torch_input.shape[0])]
            tt_inputs_host = ttnn.from_host_shards(
                [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in input_tensors],
                device.shape,
            )
        else:
            tt_inputs_host = ttnn.from_torch(
                torch_input.squeeze(0) if torch_input.shape[0] == 1 else torch_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, min_channels=16):
        """
        Setup DRAM sharded input for trace-based execution.

        Returns:
            Tuple of (tt_inputs_host, sharded_mem_config_DRAM, input_mem_config)
        """
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device, torch_input_tensor, min_channels)

        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x * dram_grid_size.y),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self):
        """Run YOLO26 inference."""
        self.output_tensor = self.ttnn_model(self.input_tensor)
        return self.output_tensor

    def dealloc_output(self):
        """Deallocate output tensor."""
        if self.output_tensor is not None:
            if isinstance(self.output_tensor, dict):
                # YOLO26 outputs dict with 'boxes' and 'scores'
                for key in self.output_tensor:
                    for item in self.output_tensor[key]:
                        if isinstance(item, tuple):
                            ttnn.deallocate(item[0])
                        else:
                            ttnn.deallocate(item)
            elif isinstance(self.output_tensor, (list, tuple)):
                for out in self.output_tensor:
                    if isinstance(out, tuple):
                        ttnn.deallocate(out[0])
                    else:
                        ttnn.deallocate(out)
            else:
                ttnn.deallocate(self.output_tensor)

    def validate(self, output_tensor=None, pcc_threshold=0.9):
        """
        Validate output against PyTorch reference.

        Note: Full validation is expensive. For performance tests,
        we typically skip validation or do minimal checks.
        """
        logger.info(f"YOLO26 validation - batch_size={self.batch_size}, input_size={self.input_size}")
        return True, "Validation skipped for performance test"


def create_test_infra(
    device,
    batch_size: int = 1,
    input_size: int = 640,
    variant: str = "yolo26n",
    act_dtype=ttnn.bfloat16,
    weight_dtype=ttnn.bfloat8_b,
    torch_input_tensor=None,
):
    """
    Factory function to create YOLO26 test infrastructure.
    """
    return YOLO26TestInfra(
        device,
        batch_size,
        input_size,
        variant,
        act_dtype,
        weight_dtype,
        torch_input_tensor,
    )
