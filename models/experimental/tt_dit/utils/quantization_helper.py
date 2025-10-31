# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import uuid
from loguru import logger


class TensorIO:
    def __init__(self, mesh_device: ttnn.MeshDevice, output_dir: str = None, filter_key: str = None):
        self.mesh_device = mesh_device
        self.filter_key = filter_key
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "operation_tensors_" + str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=False)

        self.operation_count = 0

    def print_if_tensor(self, arg, label):
        if isinstance(arg, ttnn._ttnn.tensor.Tensor):
            logger.info(f"  {label}: dtype={arg.dtype} shape={arg.shape} layout={arg.layout}")

    def write_if_tensor(self, arg, filename):
        if isinstance(arg, ttnn._ttnn.tensor.Tensor):
            torch_tensor = ttnn.to_torch(
                arg, dtype=torch.float32, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0)
            )
            torch.save(torch_tensor.cpu(), filename)

    def pre_hook_write_io(self, operation, args, kwargs):
        if self.filter_key is not None and self.filter_key not in operation.python_fully_qualified_name:
            return
        logger.info(f"Hook called for {operation.python_fully_qualified_name}")
        for i, arg in enumerate(args):
            self.print_if_tensor(arg, f"arg[{i}]")
            self.write_if_tensor(
                arg,
                os.path.join(
                    self.output_dir,
                    f"op_{self.operation_count:05d}_{operation.python_fully_qualified_name}_input{i:02d}.pt",
                ),
            )
        self.operation_count += 1

    def post_hook_write_io(self, operation, args, kwargs, output):
        if self.filter_key is not None and self.filter_key not in operation.python_fully_qualified_name:
            return
        self.print_if_tensor(output, "output")
        self.write_if_tensor(
            output,
            os.path.join(
                self.output_dir, f"op_{self.operation_count:05d}_{operation.python_fully_qualified_name}_output.pt"
            ),
        )
