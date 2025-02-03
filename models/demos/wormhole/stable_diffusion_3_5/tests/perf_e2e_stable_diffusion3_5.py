# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import pytest
import ttnn
import numpy as np

from models.utility_functions import (
    profiler,
)

from models.demos.wormhole.stable_diffusion_3_5.tests.sd3_5_test_infra import create_test_infra

from models.perf.perf_utils import prep_perf_report

from diffusers import StableDiffusion3Pipeline
from models.experimental.functional_stable_diffusion3_5.reference.sd3_transformer_2d_model import SD3Transformer2DModel

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


def dump_device_profiler(device):
    if isinstance(device, ttnn.Device):
        ttnn.DumpDeviceProfiler(device)
    else:
        for dev in device.get_device_ids():
            ttnn.DumpDeviceProfiler(device.get_device(dev))


# TODO: Create ttnn apis for this
ttnn.dump_device_profiler = dump_device_profiler

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


class SD35mTrace:
    def __init__(self):
        ...

    def initialize_sd35m_trace_inference(
        self,
        device,
        device_batch_size=1,
        model_location_generator=None,
    ):
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
            model_location_generator=model_location_generator,
        )
        (
            self.tt_inputs_host_hidden_states,
            self.input_mem_config_hidden_states,
        ) = self.test_infra.setup_l1_sharded_input_hidden_state(device)
        (
            self.tt_inputs_host_encoder_hidden_states,
            self.input_mem_config_encoder_hidden_states,
        ) = self.test_infra.setup_l1_sharded_input_encoder_hidden_state(device)
        (
            self.tt_inputs_host_pooled_proj,
            self.input_mem_config_pooled_proj,
        ) = self.test_infra.setup_l1_sharded_input_pooled_proj(device)
        self.tt_inputs_host_timestep, self.input_mem_config_timestep = self.test_infra.setup_l1_sharded_input_timestep(
            device
        )

        # First run configures convs JIT
        #################################
        self.input_tensor_hidden_state = self.tt_inputs_host_hidden_states.to(
            device, self.input_mem_config_hidden_states
        )
        shape_hidden_state = self.input_tensor_hidden_state.shape
        dtype_hidden_state = self.input_tensor_hidden_state.dtype
        layout_hidden_state = self.input_tensor_hidden_state.layout
        #
        self.input_tensor_encoder_hidden_state = self.tt_inputs_host_encoder_hidden_states.to(
            device, self.input_mem_config_encoder_hidden_states
        )
        shape_encoder_hidden_state = self.input_tensor_encoder_hidden_state.shape
        dtype_encoder_hidden_state = self.input_tensor_encoder_hidden_state.dtype
        layout_encoder_hidden_state = self.input_tensor_encoder_hidden_state.layout
        #
        self.input_tensor_pooled_proj = self.tt_inputs_host_pooled_proj.to(device, self.input_mem_config_pooled_proj)
        shape_pooled_proj = self.input_tensor_pooled_proj.shape
        dtype_pooled_proj = self.input_tensor_pooled_proj.dtype
        layout_pooled_proj = self.input_tensor_pooled_proj.layout
        #
        self.input_tensor_timestep = self.tt_inputs_host_timestep.to(device, self.input_mem_config_timestep)
        shape_timestep = self.input_tensor_timestep.shape
        dtype_timestep = self.input_tensor_timestep.dtype
        layout_timestep = self.input_tensor_timestep.layout
        #
        print("0")
        #
        self.test_infra.run(
            self.input_tensor_hidden_state,
            self.input_tensor_encoder_hidden_state,
            self.input_tensor_pooled_proj,
            self.input_tensor_timestep,
        )
        self.test_infra.validate()
        self.test_infra.dealloc_output()

        print("1")

        # Optimized run
        ###############
        self.input_tensor_hidden_state = self.tt_inputs_host_hidden_states.to(
            device, self.input_mem_config_hidden_states
        )
        self.input_tensor_encoder_hidden_state = self.tt_inputs_host_encoder_hidden_states.to(
            device, self.input_mem_config_encoder_hidden_states
        )
        self.input_tensor_pooled_proj = self.tt_inputs_host_pooled_proj.to(device, self.input_mem_config_pooled_proj)
        self.input_tensor_timestep = self.tt_inputs_host_timestep.to(device, self.input_mem_config_timestep)
        #
        self.test_infra.run(
            self.input_tensor_hidden_state,
            self.input_tensor_encoder_hidden_state,
            self.input_tensor_pooled_proj,
            self.input_tensor_timestep,
        )
        self.test_infra.validate()

        print("2")

        # Capture
        ##########
        self.input_tensor_hidden_state = self.tt_inputs_host_hidden_states.to(
            device, self.input_mem_config_hidden_states
        )
        self.input_tensor_encoder_hidden_state = self.tt_inputs_host_encoder_hidden_states.to(
            device, self.input_mem_config_encoder_hidden_states
        )
        self.input_tensor_pooled_proj = self.tt_inputs_host_pooled_proj.to(device, self.input_mem_config_pooled_proj)
        self.input_tensor_timestep = self.tt_inputs_host_timestep.to(device, self.input_mem_config_timestep)
        #
        self.test_infra.dealloc_output()
        trace_input_addr_hidden_state = ttnn.buffer_address(self.input_tensor_hidden_state)
        trace_input_addr_encoder_hidden_state = ttnn.buffer_address(self.input_tensor_encoder_hidden_state)
        trace_input_addr_pooled_proj = ttnn.buffer_address(self.input_tensor_pooled_proj)
        trace_input_addr_timestep = ttnn.buffer_address(self.input_tensor_timestep)

        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.test_infra.run(
            self.input_tensor_hidden_state,
            self.input_tensor_encoder_hidden_state,
            self.input_tensor_pooled_proj,
            self.input_tensor_timestep,
        )

        self.input_tensor_hidden_state = ttnn.allocate_tensor_on_device(
            shape_hidden_state,
            dtype_hidden_state,
            layout_hidden_state,
            device,
            self.input_mem_config_hidden_states,
        )
        # self.input_tensor_encoder_hidden_state = ttnn.allocate_tensor_on_device(
        #     shape_encoder_hidden_state,
        #     dtype_encoder_hidden_state,
        #     layout_encoder_hidden_state,
        #     device,
        #     self.input_mem_config_encoder_hidden_states,
        # )
        # self.input_tensor_pooled_proj = ttnn.allocate_tensor_on_device(
        #     shape_pooled_proj,
        #     dtype_pooled_proj,
        #     layout_pooled_proj,
        #     device,
        #     self.input_mem_config_pooled_proj,
        # )
        # self.input_tensor_timestep = ttnn.allocate_tensor_on_device(
        #     shape_timestep ,
        #     dtype_timestep ,
        #     layout_timestep ,
        #     device,
        #     self.input_mem_config_timestep,
        # )
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        assert trace_input_addr_hidden_state == ttnn.buffer_address(self.input_tensor_hidden_state)
        assert trace_input_addr_encoder_hidden_state == ttnn.buffer_address(self.input_tensor_encoder_hidden_state)
        assert trace_input_addr_pooled_proj == ttnn.buffer_address(self.input_tensor_pooled_proj)
        assert trace_input_addr_timestep == ttnn.buffer_address(self.input_tensor_timestep)

        print("3")

        self.device = device

        """
        for j in range(40):

            # numpy_array = np.load(
            # "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__hidden_states_"
            # + str(j)+ ".npy")
            # torch_input_hidden_states = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)

            # numpy_array = np.load(
            #     "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__encoder_hidden_"
            #     + str(j)+ ".npy")
            # torch_input_encoder_hidden_states = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)

            # numpy_array = np.load(
            #     "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__pooled_proj_"
            #     + str(j)+ ".npy")
            # torch_input_pooled_projections = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)

            # numpy_array = np.load(
            #     "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512___timesteps_proj_"
            #     + str(j)+ ".npy")
            # torch_input_timesteps_proj = torch.from_numpy(numpy_array)  # .to(dtype=torch.bfloat16)

            # self.test_infra.torch_output_tensor = torch.from_numpy(
            #     np.load(
            #         "../../sd35_512_unopt/tt-metal/models/experimental/functional_stable_diffusion3_5/demo/demo_unoptimized_512x512__noise_pred_"
            #         + str(j) + ".npy")
            # )
            # torch_input_hidden_states = torch_input_hidden_states.permute(0, 2, 3, 1)
            # self.tt_inputs_host_hidden_states = ttnn.from_torch(torch_input_hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
            # self.tt_inputs_host_encoder_hidden_states = ttnn.from_torch(torch_input_encoder_hidden_states.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            # self.tt_inputs_host_pooled_proj = ttnn.from_torch(torch_input_pooled_projections.unsqueeze(0).unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            # self.tt_inputs_host_timestep = ttnn.from_torch(torch_input_timesteps_proj.unsqueeze(0).unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)


            torch_input_hidden_states = torch.load('pt_demo_trace/demo_in0_noTrace_'+str(j)+'.pt')
            torch_input_encoder_hidden_states = torch.load('pt_demo_trace/demo_in1_noTrace_'+str(j)+'.pt')
            torch_input_pooled_projections = torch.load('pt_demo_trace/demo_in2_noTrace_'+str(j)+'.pt')
            torch_input_timesteps_proj = torch.load('pt_demo_trace/demo_in3_noTrace_'+str(j)+'.pt')
            self.test_infra.torch_output_tensor = torch.load('pt_demo_trace/noise_pred_noTrace_'+str(j)+'.pt')

            torch_input_hidden_states = torch_input_hidden_states.permute(0, 2, 3, 1)
            self.tt_inputs_host_hidden_states = ttnn.from_torch(torch_input_hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
            self.tt_inputs_host_encoder_hidden_states = ttnn.from_torch(torch_input_encoder_hidden_states.unsqueeze(1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            self.tt_inputs_host_pooled_proj = ttnn.from_torch(torch_input_pooled_projections.unsqueeze(0).unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            self.tt_inputs_host_timestep = ttnn.from_torch(torch_input_timesteps_proj, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

            tt_inputs_host_hidden_states = self.tt_inputs_host_hidden_states
            tt_inputs_host_encoder_hidden_states = self.tt_inputs_host_encoder_hidden_states
            tt_inputs_host_pooled_proj = self.tt_inputs_host_pooled_proj
            tt_inputs_host_timestep = self.tt_inputs_host_timestep

            ttnn.copy_host_to_device_tensor(tt_inputs_host_hidden_states, self.input_tensor_hidden_state, 0)
            ttnn.copy_host_to_device_tensor(tt_inputs_host_encoder_hidden_states, self.input_tensor_encoder_hidden_state, 0)
            ttnn.copy_host_to_device_tensor(tt_inputs_host_pooled_proj, self.input_tensor_pooled_proj, 0)
            ttnn.copy_host_to_device_tensor(tt_inputs_host_timestep, self.input_tensor_timestep, 0)
            #
            #print("--", j, "--", self.input_tensor_timestep)
            ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
            output = self.test_infra.output_tensor
            self.test_infra.validate(output)
            ttnn.synchronize_devices(self.device)
        """

    def execute_sd35m_trace_inference(self, tt_inputs_host=None):
        tt_inputs_host_hidden_states = (
            self.tt_inputs_host_hidden_states if tt_inputs_host is None else tt_inputs_host[0]
        )
        tt_inputs_host_encoder_hidden_states = (
            self.tt_inputs_host_encoder_hidden_states if tt_inputs_host is None else tt_inputs_host[1]
        )
        tt_inputs_host_pooled_proj = self.tt_inputs_host_pooled_proj if tt_inputs_host is None else tt_inputs_host[2]
        tt_inputs_host_timestep = self.tt_inputs_host_timestep if tt_inputs_host is None else tt_inputs_host[3]
        #
        ttnn.copy_host_to_device_tensor(tt_inputs_host_hidden_states, self.input_tensor_hidden_state, 0)
        ttnn.copy_host_to_device_tensor(tt_inputs_host_encoder_hidden_states, self.input_tensor_encoder_hidden_state, 0)
        ttnn.copy_host_to_device_tensor(tt_inputs_host_pooled_proj, self.input_tensor_pooled_proj, 0)
        ttnn.copy_host_to_device_tensor(tt_inputs_host_timestep, self.input_tensor_timestep, 0)
        #

        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        output = self.test_infra.output_tensor
        ttnn.synchronize_devices(self.device)

        return output
        """

        self.test_infra.run(
            self.input_tensor_hidden_state,
            self.input_tensor_encoder_hidden_state,
            self.input_tensor_pooled_proj,
            self.input_tensor_timestep
            )
        return self.test_infra.output_tensor
        """

    def release_sd35m_trace_inference(self):
        ttnn.release_trace(self.device, self.tid)
        # test_infra.dealloc_output()

    ## for the server/client Demo
    """
    def run_traced_inference(self, torch_input_tensor):
        n, h, w, c = torch_input_tensor.shape
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        return self.execute_sd35m_trace_inference(tt_inputs_host)
    """
