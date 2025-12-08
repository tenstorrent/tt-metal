# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import uuid
from loguru import logger
from models.common.utility_functions import comp_pcc

# Ordered from most to least precise:
DATA_TYPES = [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b]
MATH_FIDELITIES = [
    ttnn.MathFidelity.HiFi4,
    ttnn.MathFidelity.HiFi3,
    ttnn.MathFidelity.HiFi2,
    ttnn.MathFidelity.LoFi,
]


class QuantizationHookHelper:
    """
    Helper class with functionality to:

      1. Write input and output tensors for all or some operations to disk.
      2. Measure accuracy of all or some operations when the inputs are quantized.
      3. Update the output of all or some operations with the quantized result if the accuracy meets
         a minimum value.

    Example to write tensors to disk:

      from ....utils import quantization_helper

      helper = quantization_helper.QuantizationHookHelper(
                   mesh_device,
                   filter_key="all_gather_async",
                   write_tensors=True,
                   parallel_config=parallel_config,
               )

      with ttnn.register_pre_operation_hook(helper.pre_hook), ttnn.register_post_operation_hook(helper.post_hook):
          output = model(input)

    Example to compute attention operations with different quantization configurations
    and select a configuration that results in output PCC > 0.97:

      helper = quantization_helper.QuantizationHookHelper(
                   mesh_device,
                   filter_key="attention",
                   quantization_configs=[
                       (ttnn.DataType.bfloat8_b, ttnn.MathFidelity.HiFi2),
                       (ttnn.DataType.bfloat8_b, ttnn.MathFidelity.LoFi),
                       (ttnn.DataType.bfloat4_b, ttnn.MathFidelity.LoFi),
                   ],
                   update_output=True,
                   min_output_pcc=0.97,
                   parallel_config=parallel_config,
                   ccl_manager=ccl_manager,
               )

    For the hooks to be used the following environment variable must be set:

      export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        filter_key: str = None,
        write_tensors: bool = False,
        output_dir: str = "",
        quantization_configs: list[tuple[ttnn.DataType, ttnn.MathFidelity]] = [],
        update_output: bool = False,
        min_output_pcc: float = None,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.mesh_device = mesh_device
        self.filter_key = filter_key
        self.write_tensors = write_tensors

        self.output_dir = output_dir
        if write_tensors:
            if self.output_dir is None:
                self.output_dir = os.path.join(os.getcwd(), "operation_tensors_" + str(uuid.uuid4()))
            os.makedirs(self.output_dir, exist_ok=False)

        # Order quantization settings from most accurate to least accurate:
        order0_idx = {val: idx for idx, val in enumerate(DATA_TYPES)}
        order1_idx = {val: idx for idx, val in enumerate(MATH_FIDELITIES)}
        self.quantization_configs = sorted(quantization_configs, key=lambda x: (order0_idx[x[0]], order1_idx[x[1]]))

        self.update_output = update_output
        self.min_output_pcc = min_output_pcc
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.operation_count = 0

    def print_if_tensor(self, arg, label):
        """If the argument is a ttnn Tensor, print its information"""
        if isinstance(arg, tuple):
            arg = arg[0]
        if isinstance(arg, ttnn._ttnn.tensor.Tensor):
            logger.info(f"  {label}: dtype={arg.dtype} shape={arg.shape} layout={arg.layout}")

    def write_if_tensor(self, arg, filename):
        """If the argument is a ttnn Tensor, write it to disk in PyTorch format"""
        if isinstance(arg, tuple):
            arg = arg[0]
        if self.write_tensors:
            if isinstance(arg, ttnn._ttnn.tensor.Tensor):
                torch_tensor = ttnn.to_torch(
                    arg, dtype=torch.float32, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0)
                )
                logger.info(f"Writing file: {filename}")
                torch.save(torch_tensor.cpu(), filename)

    def check_input_configurations(self, operation, args, kwargs, output):
        """
        Iterates through the quantization configurations and computes the operation
        with the input tensors quantized to the specified data type and the kernel
        configuration set to the specified math fidelity.
        If a minimum PCC is set the function returns the output from the last
        quantization configuration that met this minimum PCC.
        """
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, ttnn._ttnn.tensor.Tensor):
            return None

        mesh_composer = ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0)
        output_torch = ttnn.to_torch(output, dtype=torch.float32, mesh_composer=mesh_composer)

        best_output = output
        for dtype, math_fidelity in self.quantization_configs:
            new_args = []
            for arg in args:
                if isinstance(arg, ttnn._ttnn.tensor.Tensor):
                    new_args.append(ttnn.typecast(arg, dtype))
                else:
                    new_args.append(arg)

            new_kwargs = copy.deepcopy(kwargs)
            if "compute_kernel_config" in new_kwargs:
                kernel_config = new_kwargs["compute_kernel_config"]
                kernel_config.math_fidelity = math_fidelity
                new_kwargs["compute_kernel_config"] = kernel_config

            # Hack to update data type in persistent output buffer argument.
            if "persistent_output_buffer_k" in kwargs:
                new_kwargs["persistent_output_buffer_k"] = self.ccl_manager.get_ag_ping_pong_buffer(
                    new_args[1].shape,
                    2,
                    self.parallel_config.sequence_parallel.mesh_axis,
                    dtype=dtype,
                )
            if "persistent_output_buffer_v" in kwargs:
                new_kwargs["persistent_output_buffer_v"] = self.ccl_manager.get_ag_ping_pong_buffer(
                    new_args[2].shape,
                    2,
                    self.parallel_config.sequence_parallel.mesh_axis,
                    dtype=dtype,
                )

            new_output = operation(*tuple(new_args), **new_kwargs)
            if isinstance(new_output, tuple):
                new_output = new_output[0]

            # Check the accuracy of new_output.
            if isinstance(new_output, ttnn._ttnn.tensor.Tensor):
                new_output_torch = ttnn.to_torch(new_output, dtype=torch.float32, mesh_composer=mesh_composer)
                if new_output_torch.shape == output_torch.shape:
                    max_abs_err = torch.max(torch.abs(output_torch - new_output_torch))
                    passed, pcc = comp_pcc(output_torch, new_output_torch, 1.0)
                    logger.info(
                        f"Input dtype={dtype}, math_fidelity={math_fidelity}, max(abs(error))={max_abs_err} pcc={pcc}"
                    )

                    if self.min_output_pcc is not None and pcc >= self.min_output_pcc:
                        logger.info(f"Updating best output to ({dtype}, {math_fidelity})")
                        best_output = new_output
                else:
                    logger.warning(f"Unexpected output shape {new_output_torch.shape}, expecting {output_torch.shape}")

        return best_output

    def pre_hook(self, operation, args, kwargs):
        """Hook that runs before the operation is computed"""

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

    def post_hook(self, operation, args, kwargs, output):
        """Hook that runs after the operation is computed"""

        if self.filter_key is not None and self.filter_key not in operation.python_fully_qualified_name:
            return

        self.print_if_tensor(output, "output")
        self.write_if_tensor(
            output,
            os.path.join(
                self.output_dir, f"op_{self.operation_count:05d}_{operation.python_fully_qualified_name}_output.pt"
            ),
        )

        if len(self.quantization_configs) > 0:
            best_output = self.check_input_configurations(operation, args, kwargs, output)
            if best_output is not None and self.update_output:
                best_output = ttnn.typecast(best_output, ttnn.bfloat16)
                if isinstance(output, ttnn._ttnn.tensor.Tensor):
                    ttnn.copy(best_output, output)
                elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], ttnn._ttnn.tensor.Tensor):
                    ttnn.copy(best_output, output[0])
