# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from loguru import logger

import ttnn


@dataclass(frozen=True)
class SamplingParams:
    """
    Used in Generator decode forward functions for greedy decoding / sampling on device.
    The same data class exists in vLLM at vllm/worker/tt_model_runner.py.
    """

    temperature: float
    top_k: int
    top_p: float


class Generator:
    def __init__(self, model, model_args, mesh_device):
        """
        Creating a Grok generator wrapper.

        Args:
            model: The Grok Transformer model
            model_args: Model configuration arguments
            mesh_device: The mesh device for distributed inference
        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.prev_page_table = None

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params: SamplingParams = None,
    ):
        """
        Main decode forward method.

        Args:
            tokens: Input token tensor [batch_size, 1]
            start_pos: Current position for each sequence [batch_size]
            page_table: Page table for paged attention
            enable_trace: Whether to use tracing for faster execution
            read_from_device: Whether to read output back to host
            sampling_params: Parameters for sampling/greedy decoding

        Returns:
            Logits tensor [batch_size, 1, vocab_size] if read_from_device=True,
            otherwise returns device tensor
        """
        assert (
            sampling_params is None or sampling_params.temperature == 0
        ), "Currently only supporting greedy decoding (temperature=0) on device"
        argmax_on_device = sampling_params is not None and sampling_params.temperature == 0

        decode_kwargs = {
            "tokens": tokens,
            "current_pos": start_pos,
            "page_table": page_table,
            "argmax_on_device": argmax_on_device,
        }

        if enable_trace:
            tt_decode_output = self._easy_trace(**decode_kwargs)
        else:
            tt_decode_output = self._decode_forward_no_trace(**decode_kwargs)

        if read_from_device:
            logits = self.read_decode_output(tt_decode_output)
            return self.process_decode_output_host(logits, is_tokens=(sampling_params is not None))

        return tt_decode_output

    def _decode_forward_no_trace(
        self,
        tokens,
        current_pos,
        page_table=None,
        argmax_on_device=False,
    ):
        """
        Performs decode step without tracing.
        Returns tt_logits on device.
        """
        # Prepare rotation matrices
        rot_mats = self.model.rope_setup.get_rot_mats(current_pos)

        # Convert tokens to device tensor if needed
        if torch.is_tensor(tokens):
            tt_tokens = ttnn.from_torch(
                tokens.reshape(1, 1, 1, -1),  # [batch_size, seq_len=1] -> [1, 1, 1, batch_size]
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, None), mesh_shape=self.model_args.cluster_shape
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            tt_tokens = tokens

        # Convert current_pos to device tensor if needed
        if torch.is_tensor(current_pos):
            tt_current_pos = ttnn.from_torch(
                current_pos,
                device=self.mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(None, 0) if self.model_args.num_devices == 32 else (None, None),
                    mesh_shape=self.model_args.cluster_shape,
                ),
            )
        else:
            tt_current_pos = current_pos

        # Forward pass
        tt_logits = self.model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mats=rot_mats,
            page_table=page_table,
        )

        return tt_logits

    def _capture_trace(
        self,
        tokens,
        current_pos,
        page_table=None,
        argmax_on_device=False,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        # Compile run
        self._decode_forward_no_trace(tokens, current_pos, page_table=page_table, argmax_on_device=argmax_on_device)
        logger.info("Done Compiling Model")

        # Prepare inputs for trace capture
        batch_size = tokens.shape[0]

        # Store host inputs
        host_tokens = tokens.reshape(1, 1, 1, batch_size)
        host_current_pos = current_pos

        # Create device tensors for tracing
        tt_tokens = ttnn.from_torch(
            host_tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(None, None), mesh_shape=self.model_args.cluster_shape
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tt_current_pos = ttnn.from_torch(
            host_current_pos,
            device=self.mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, 0) if self.model_args.num_devices == 32 else (None, None),
                mesh_shape=self.model_args.cluster_shape,
            ),
        )

        # Begin trace capture
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        rot_mats = self.model.rope_setup.get_rot_mats(host_current_pos)
        tt_logits = self.model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mats=rot_mats,
            page_table=page_table,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return trace_id, tt_logits, tt_tokens, tt_current_pos

    def _easy_trace(
        self,
        tokens,
        current_pos,
        page_table=None,
        argmax_on_device=False,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_id"):
            trace_id, tt_out_trace, tt_tokens, tt_current_pos = self._capture_trace(
                tokens, current_pos, page_table=page_table, argmax_on_device=argmax_on_device
            )
            self.trace_id = trace_id
            self.trace_inputs = {
                "tt_tokens": tt_tokens,
                "tt_current_pos": tt_current_pos,
            }
            self.trace_output = tt_out_trace

        # Check if page_table changed
        reset_inputs = not argmax_on_device
        if self.prev_page_table is None or (
            page_table is not None and not torch.equal(self.prev_page_table, page_table)
        ):
            reset_inputs = True
            self.prev_page_table = page_table

        if reset_inputs:
            # Update trace inputs
            batch_size = tokens.shape[0]
            host_tokens = tokens.reshape(1, 1, 1, batch_size)

            ttnn.copy_host_to_device_tensor(
                host_tokens,
                self.trace_inputs["tt_tokens"],
            )
            ttnn.copy_host_to_device_tensor(
                current_pos,
                self.trace_inputs["tt_current_pos"],
            )

        # Execute trace
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)

        return self.trace_output

    def read_decode_output(self, tt_out):
        """
        Reads output from device to host.
        Input tt_out is a ttnn device tensor.
        """
        return tt_out.cpu()

    def process_decode_output_host(self, tt_out, is_tokens=False):
        """
        Converts the input ttnn host tensor to a torch tensor.
        The input can be logits (if is_tokens=False) or tokens (if is_tokens=True).
        """
        batch_size = self.model_args.max_batch_size
        logits = self.model.process_output_decode(tt_out, batch_size, S=1)
        return logits

    def __del__(self):
        # Cleanup if needed
        if hasattr(super(Generator, self), "__del__"):
            super().__del__()
