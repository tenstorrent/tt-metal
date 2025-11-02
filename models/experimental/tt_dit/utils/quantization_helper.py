# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import uuid
from loguru import logger
from models.common.utility_functions import comp_pcc


class TensorIO:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        output_dir: str = None,
        filter_key: str = None,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.mesh_device = mesh_device
        self.filter_key = filter_key
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), "operation_tensors_" + str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=False)

        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

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
        self.new_args = [arg for arg in args]
        self.new_kwargs = kwargs.copy()
        for i, arg in enumerate(args):
            if isinstance(arg, ttnn._ttnn.tensor.Tensor):
                torch_tensor = ttnn.to_torch(
                    arg, dtype=torch.float32, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0)
                )
                self.new_args[i] = ttnn._ttnn.tensor.Tensor(
                    torch_tensor,
                    data_type=ttnn.DataType.BFLOAT8_B,  # TODO: parameterize data type.
                    device=arg.device(),
                    layout=arg.get_layout(),
                    mem_config=arg.memory_config(),
                    tile=arg.get_tile(),
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                )
                tensor_1 = ttnn.to_torch(
                    args[i],
                    dtype=torch.float32,
                    mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0),
                )
                tensor_2 = ttnn.to_torch(
                    self.new_args[i],
                    dtype=torch.float32,
                    mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0),
                )
                print(f" tensor_1[:10]={tensor_1.flatten()[:10]}")
                print(f" tensor_2[:10]={tensor_2.flatten()[:10]}")

            self.print_if_tensor(arg, f"arg[{i}]")
            self.print_if_tensor(self.new_args[i], f"arg[{i}]")
            self.write_if_tensor(
                arg,
                os.path.join(
                    self.output_dir,
                    f"op_{self.operation_count:05d}_{operation.python_fully_qualified_name}_input{i:02d}.pt",
                ),
            )

        # TODO: Set the kernel config math fidelity.

        self.new_kwargs["persistent_output_buffer_k"] = self.ccl_manager.get_ag_ping_pong_buffer(
            args[1].shape,
            2,
            self.parallel_config.sequence_parallel.mesh_axis,
            dtype=self.new_args[1].dtype,
        )
        self.new_kwargs["persistent_output_buffer_v"] = self.ccl_manager.get_ag_ping_pong_buffer(
            args[2].shape,
            2,
            self.parallel_config.sequence_parallel.mesh_axis,
            dtype=self.new_args[2].dtype,
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

        # FIXME: Update the output with the modified version?
        test_output = operation(*tuple(self.new_args), **self.new_kwargs)
        print(f"len(output)={len(output)}")
        if len(output) > 0:
            print(f"type(output[0])={type(output[0])}")
            if isinstance(output[0], ttnn._ttnn.tensor.Tensor):
                tensor_1 = ttnn.to_torch(
                    output[0],
                    dtype=torch.float32,
                    mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0),
                )
                tensor_2 = ttnn.to_torch(
                    test_output[0],
                    dtype=torch.float32,
                    mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0),
                )
                passed, pcc = comp_pcc(tensor_1, tensor_2, 1.0)
                print(f" tensor_1[:10]={tensor_1.flatten()[:10]}")
                print(f" tensor_2[:10]={tensor_2.flatten()[:10]}")
                print(
                    f" tensor_1.shape={tensor_1.shape} mean(tensor_1)={torch.mean(tensor_1)}, std(tensor_1)={torch.std(tensor_1)}"
                )
                print(
                    f" tensor_2.shape={tensor_2.shape} mean(tensor_2)={torch.mean(tensor_2)}, std(tensor_2)={torch.std(tensor_2)}"
                )
                print(f" max(abs(diff))={torch.max(torch.abs(tensor_1-tensor_2))} pcc={pcc}")
