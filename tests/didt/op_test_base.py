# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import os

from models.utility_functions import comp_pcc, is_blackhole
import ttnn


class OpTestBase:
    def __init__(
        self,
        mesh_device,
        in0_shape,
        in1_shape,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_layout,
        in1_layout,
        program_config,
        compute_config,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
    ):
        self.mesh_device = mesh_device
        # This will be removed once we rebase onto new main with the new open_mesh_device API
        # that will allow opening mesh_device with any specific device from the available ones
        if isinstance(mesh_device, ttnn.MeshDevice):
            self.from_torch_mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            self.to_torch_mesh_mapper = ttnn.ListMeshToTensor(self.mesh_device)
            self.device_ids = self.mesh_device.get_device_ids()
        else:
            # ttnn.Device
            self.from_torch_mesh_mapper = None
            self.to_torch_mesh_mapper = None
            self.device_ids = [self.mesh_device.id()]

        self.in0_mem_config = in0_mem_config
        self.in1_mem_config = in1_mem_config
        self.out_mem_config = out_mem_config
        self.in0_shape = in0_shape
        self.in1_shape = in1_shape
        self.in0_dtype = in0_dtype
        self.in1_dtype = in1_dtype
        self.out_dtype = out_dtype
        self.in0_layout = in0_layout
        self.in1_layout = in1_layout
        self.program_config = program_config
        self.compute_config = compute_config
        self.loop_count = loop_count
        self.determinism_check_enabled = determinism_check_enabled
        self.determinism_check_iterations = determinism_check_iterations

        # Weights and activations tensors are needed for subclasses to run the operation
        self.activations = None
        self.weights = None

    def get_device(self, device_idx):
        if isinstance(self.mesh_device, ttnn.MeshDevice):
            return self.mesh_device.get_device(device_idx)
        else:
            # ttnn.Device
            return self.mesh_device

    # Override if needed
    def set_seed(self):
        torch.manual_seed(1234)

    # Override if needed
    def generate_torch_activations(self, shape):
        return torch.randn(shape)

    # Override if needed
    def generate_torch_weights(self, shape):
        return torch.randn(shape)

    def generate_tt_activations_from_torch(self, torch_tensor):
        return ttnn.from_torch(
            torch_tensor,
            dtype=self.in0_dtype,
            layout=self.in0_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )

    def generate_tt_weights_from_torch(self, torch_tensor):
        return ttnn.from_torch(
            torch_tensor,
            dtype=self.in1_dtype,
            layout=self.in1_layout,
            memory_config=self.in1_mem_config,
            device=self.mesh_device,
            mesh_mapper=self.from_torch_mesh_mapper,
        )

    def convert_activations_to_memory_config(self, activations):
        return ttnn.to_memory_config(activations, self.in0_mem_config)

    def run_device_operation(self):
        # Default Op is matmul. Override in derived class if needed.
        return ttnn.matmul(
            self.activations,
            self.weights,
            program_config=self.program_config,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_config,
        )

    def deallocate_activations(self):
        self.activations.deallocate(True)

    def run_op_test(self):
        num_devices = len(self.device_ids)
        self.set_seed()

        logger.info(f"Running on {num_devices} devices")

        a_shape = self.in0_shape
        b_shape = self.in1_shape

        num_activation_tensors = 1
        if self.determinism_check_enabled:
            # If we are running determinism checks, we want to switch activation tensors
            # every time we complete an iteration of a determinism check, to confirm that
            # device is producing new results, and not just reusing an already existing buffer
            num_activation_tensors = 10

        A = []
        for act in range(num_activation_tensors):
            A.append(self.generate_torch_activations(a_shape))
        B = self.generate_torch_weights(b_shape)

        logger.info("Pushing activations to devices...")
        a_t = []
        for act in range(num_activation_tensors):
            a_t.append(self.generate_tt_activations_from_torch(A[act]))

        logger.info("Pushing weights to devices...")
        self.weights = self.generate_tt_weights_from_torch(B)

        logger.info("Activations and weights pushed to devices!")

        if self.determinism_check_enabled:
            # Run op once to populate reference output to use for determinism checks
            num_nd_outputs = [0] * num_devices
            reference_out = [None for _ in range(num_activation_tensors)]

            for act in range(num_activation_tensors):
                # First, load activations from DRAM to required memory config
                self.activations = self.convert_activations_to_memory_config(a_t[act])
                output = self.run_device_operation()
                reference_out[act] = ttnn.to_torch(output, mesh_composer=self.to_torch_mesh_mapper)
                output.deallocate(True)
                self.deallocate_activations()

        current_act_tensor = 0
        self.activations = [None] * num_devices
        out = [None] * num_devices

        self.activations = self.convert_activations_to_memory_config(a_t[current_act_tensor])

        logger.info("Starting iterations")
        for i in range(self.loop_count):
            out = self.run_device_operation()

            # Synchronize devices in mesh in order (this ensures we sync with closer devices first -
            # - eg. if the chip is remote, we first sync its local pair);
            # The device ids don't necessarily start from 0, as it is possible to target a specific chip in the mesh
            for device_idx in self.device_ids:
                logger.info(f"Start sync device id: {device_idx}")
                ttnn.device.synchronize_device(self.get_device(device_idx))
                logger.info(f"End sync device id: {device_idx}")

            # Check if the output matches the first run output
            if self.determinism_check_enabled and i % self.determinism_check_iterations == 0:
                outputs = ttnn.to_torch(out, mesh_composer=self.to_torch_mesh_mapper)

                for output_id in range(num_devices):
                    device_idx = self.device_ids[output_id]
                    if torch.equal(reference_out[current_act_tensor][output_id], outputs[output_id]):
                        logger.info(f"Device {device_idx} PCC: 1.0")
                    else:
                        # For determinism check, we avoid calling comp_pcc func as it is heavy and with too many operations,
                        # part of the code that replaces nans/infs with zeros starts leaking memory, even if deallocation is forced,
                        # so we call it only in case we see tensors are not equal
                        _, pcc = comp_pcc(reference_out[current_act_tensor][output_id], outputs[output_id])
                        logger.info(f"Device {device_idx} PCC: {pcc}")
                        num_nd_outputs[output_id] += 1

                current_act_tensor = (current_act_tensor + 1) % num_activation_tensors
                logger.info("Switching activation tensor for new determinism iterations...")
                self.deallocate_activations()

                # Load next round of activations from DRAM to required memory config
                self.activations = self.convert_activations_to_memory_config(a_t[current_act_tensor])

            out.deallocate(True)

            logger.info(f"Iteration = {i}, done!")

        if self.determinism_check_enabled:
            for nd_output_id in range(num_devices):
                device_idx = self.device_ids[nd_output_id]
                logger.info(
                    f"Number of non-deterministic outputs on device {device_idx} is {num_nd_outputs[nd_output_id]}"
                )


def get_blackhole_grid_size(simulate_2col_harvesting):
    assert is_blackhole()

    if simulate_2col_harvesting:
        assert "TT_METAL_ETH_DISPATCH" not in os.environ
        return ttnn.CoreCoord(11, 10)
    else:
        return ttnn.CoreCoord(14, 10) if ("TT_METAL_ETH_DISPATCH" in os.environ) else ttnn.CoreCoord(13, 10)
