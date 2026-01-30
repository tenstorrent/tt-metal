# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.experimental.yunet.common import load_torch_model, get_default_weights_path
from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model


class YunetPerformanceRunnerInfra:
    """
    Infrastructure class for YUNet performance testing.
    Sets up the model, manages inputs/outputs, and provides run/validate methods.
    """

    def __init__(
        self,
        device,
        batch_size=1,
        input_height=640,
        input_width=640,
        act_dtype=ttnn.bfloat16,
        torch_input_tensor=None,
    ):
        torch.manual_seed(0)
        self.device = device
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.act_dtype = act_dtype

        # Load PyTorch model
        logger.info("Loading PyTorch model...")
        self.torch_model = load_torch_model(get_default_weights_path())
        self.torch_model = self.torch_model.to(torch.bfloat16)
        self.torch_model.eval()

        # Create TTNN model
        logger.info("Creating TTNN model...")
        self.ttnn_model = create_yunet_model(device, self.torch_model)

        # Setup input tensor
        if torch_input_tensor is None:
            # NHWC format for TTNN
            self.torch_input_tensor = torch.randn((batch_size, input_height, input_width, 3), dtype=torch.bfloat16)
        else:
            self.torch_input_tensor = torch_input_tensor

        # Run PyTorch for reference output (expects NCHW)
        # Model is bfloat16, so input must also be bfloat16
        torch_input_nchw = self.torch_input_tensor.permute(0, 3, 1, 2).to(torch.bfloat16)
        self.torch_model.train()  # Get raw outputs
        with torch.no_grad():
            self.torch_outputs = self.torch_model(torch_input_nchw)

        self.input_tensor = None
        self.output_tensors = None

    def _setup_l1_sharded_input(self, device, torch_input_tensor=None):
        """Setup L1 input memory configuration (interleaved for simplicity)."""
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        # Use interleaved DRAM config - simpler and works for any input size
        input_mem_config = ttnn.DRAM_MEMORY_CONFIG

        # Convert to TTNN tensor
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return tt_inputs_host, input_mem_config

    def setup_dram_input(self, device, torch_input_tensor=None):
        """Setup DRAM input for host-to-device transfer."""
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        # Convert to TTNN tensor (host)
        tt_inputs_host = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Use interleaved DRAM memory config
        dram_mem_config = ttnn.DRAM_MEMORY_CONFIG

        return tt_inputs_host, dram_mem_config

    def run(self):
        """Run the TTNN model."""
        cls_out, box_out, obj_out, kpt_out = self.ttnn_model(self.input_tensor)
        self.output_tensors = (cls_out, box_out, obj_out, kpt_out)
        return self.output_tensors

    def validate(self, pcc_threshold=0.99):
        """Validate TTNN outputs against PyTorch reference."""
        if self.output_tensors is None:
            logger.warning("No outputs to validate. Call run() first.")
            return False

        pt_cls, pt_box, pt_obj, pt_kpt = self.torch_outputs
        tt_cls, tt_box, tt_obj, tt_kpt = self.output_tensors

        # Convert TTNN outputs to torch and compare
        from models.common.utility_functions import comp_pcc

        all_pass = True
        min_pcc = 1.0

        for i in range(3):
            # cls
            pt_tensor = pt_cls[i].permute(0, 2, 3, 1).to(torch.bfloat16).flatten()
            tt_tensor = ttnn.to_torch(tt_cls[i]).flatten()
            passed, pcc = comp_pcc(pt_tensor, tt_tensor, pcc_threshold)
            min_pcc = min(min_pcc, pcc)
            if not passed:
                all_pass = False

            # box
            pt_tensor = pt_box[i].permute(0, 2, 3, 1).to(torch.bfloat16).flatten()
            tt_tensor = ttnn.to_torch(tt_box[i]).flatten()
            passed, pcc = comp_pcc(pt_tensor, tt_tensor, pcc_threshold)
            min_pcc = min(min_pcc, pcc)
            if not passed:
                all_pass = False

            # obj
            pt_tensor = pt_obj[i].permute(0, 2, 3, 1).to(torch.bfloat16).flatten()
            tt_tensor = ttnn.to_torch(tt_obj[i]).flatten()
            passed, pcc = comp_pcc(pt_tensor, tt_tensor, pcc_threshold)
            min_pcc = min(min_pcc, pcc)
            if not passed:
                all_pass = False

            # kpt
            pt_tensor = pt_kpt[i].permute(0, 2, 3, 1).to(torch.bfloat16).flatten()
            tt_tensor = ttnn.to_torch(tt_kpt[i]).flatten()
            passed, pcc = comp_pcc(pt_tensor, tt_tensor, pcc_threshold)
            min_pcc = min(min_pcc, pcc)
            if not passed:
                all_pass = False

        logger.info(f"YUNet validation - min PCC: {min_pcc:.6f}, passed: {all_pass}")
        return all_pass

    def dealloc_output(self):
        """Deallocate output tensors."""
        if self.output_tensors is not None:
            cls_out, box_out, obj_out, kpt_out = self.output_tensors
            for tensors in [cls_out, box_out, obj_out, kpt_out]:
                for t in tensors:
                    ttnn.deallocate(t)
            self.output_tensors = None
