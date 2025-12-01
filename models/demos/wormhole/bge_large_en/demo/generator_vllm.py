# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Integration for BGE-Large-EN-v1.5 Embedding Model

This module provides vLLM integration for the BGE embedding model, enabling
OpenAI Embedding API compatibility through vLLM's serving infrastructure.

Usage:
    The model can be registered with vLLM's ModelRegistry and served via
    the OpenAI Embedding API endpoint.
"""

from typing import Any, List, Optional

import torch
import transformers
from loguru import logger

import ttnn
from models.demos.bge_large_en.runner.performant_runner import BGEPerformantRunner
from models.demos.sentence_bert.reference.sentence_bert import custom_extended_mask


def get_num_devices(device):
    """Get number of devices from ttnn.Device or ttnn.MeshDevice."""
    if isinstance(device, ttnn.MeshDevice):
        return device.get_num_devices()
    elif isinstance(device, ttnn.Device):
        return 1
    else:
        raise ValueError(f"Unrecognized device type {type(device)}")


class BGEPerformantRunnerSingleCQ(BGEPerformantRunner):
    """
    Adapter for BGEPerformantRunner that works with single command queue devices.

    This is needed for vLLM integration, as vLLM devices typically only have
    command queue 0, while the original BGEPerformantRunner uses both CQ 0 and CQ 1.
    """

    def _capture_bge_trace_2cqs(self):
        """
        Override to use single command queue (CQ 0) instead of 2 command queues.
        """
        logger.info("Trace capture: Step 1 - Recording initial event")
        self.op_event = ttnn.record_event(self.device, 0)

        # Single command queue: copy first, then wait
        logger.info("Trace capture: Step 2 - Copying host to device tensors")
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 0)
        ttnn.copy_host_to_device_tensor(self.tt_tokens_host, self.tt_tokens, 0)
        ttnn.copy_host_to_device_tensor(self.tt_posids_host, self.tt_pos, 0)
        ttnn.copy_host_to_device_tensor(self.tt_ext_att_mask_host, self.tt_ext_att_mask, 0)
        ttnn.copy_host_to_device_tensor(self.tt_att_mask_host, self.tt_att_mask, 0)
        ttnn.wait_for_event(0, self.op_event)
        self.write_event = self.op_event

        # Try to reshard from DRAM to L1, fall back to interleaved L1 if it fails (e.g., small batches)
        logger.info("Trace capture: Step 3 - Resharding from DRAM to L1")
        try:
            self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
            self.runner_infra.ttnn_token_ids = ttnn.to_memory_config(self.tt_tokens, self.input_mem_config)
            self.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(self.tt_pos, self.input_mem_config)
            self.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(self.tt_ext_att_mask, self.input_mem_config)
            self.runner_infra.ttnn_att_mask = ttnn.to_memory_config(self.tt_att_mask, self.input_mem_config)
            logger.info("Trace capture: Step 3 - Resharding successful")
        except RuntimeError as e:
            # If reshard fails (e.g., due to alignment issues with small batches),
            # use interleaved L1 input directly instead of sharded
            error_msg = str(e).lower()
            if (
                "circular buffer" in error_msg
                or "divisible" in error_msg
                or "page size" in error_msg
                or "aligned" in error_msg
            ):
                # For very small batches, use interleaved layout to avoid alignment issues
                # Create tensors directly on device with interleaved L1 memory config
                self.runner_infra.ttnn_input_ids = ttnn.from_torch(
                    self.runner_infra.input_ids,
                    dtype=ttnn.uint32,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                self.runner_infra.ttnn_token_ids = ttnn.from_torch(
                    self.runner_infra.token_type_ids,
                    dtype=ttnn.uint32,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                self.runner_infra.ttnn_pos_ids = ttnn.from_torch(
                    self.runner_infra.position_ids,
                    dtype=ttnn.uint32,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                self.runner_infra.ttnn_ext_att_mask = ttnn.from_torch(
                    self.runner_infra.extended_mask,
                    dtype=ttnn.bfloat16,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                self.runner_infra.ttnn_att_mask = ttnn.from_torch(
                    self.runner_infra.attention_mask,
                    dtype=ttnn.bfloat16,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                # Update input_mem_config to interleaved L1 for subsequent operations
                self.input_mem_config = ttnn.L1_MEMORY_CONFIG
            else:
                raise

        spec_input = self.runner_infra.ttnn_input_ids.spec
        spec_token = self.runner_infra.ttnn_token_ids.spec
        spec_pos = self.runner_infra.ttnn_pos_ids.spec
        spec_att = self.runner_infra.ttnn_ext_att_mask.spec
        spec_att_2 = self.runner_infra.ttnn_att_mask.spec

        logger.info("Trace capture: Step 4 - First run (JIT configuration)")
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        logger.info("Trace capture: Step 4 - First run completed, validating...")
        self.runner_infra.validate()
        logger.info("Trace capture: Step 4 - Validation passed, deallocating output...")
        self.runner_infra.dealloc_output()

        # Optimized run
        logger.info("Trace capture: Step 5 - Optimized run")
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 0)
        ttnn.copy_host_to_device_tensor(self.tt_tokens_host, self.tt_tokens, 0)
        ttnn.copy_host_to_device_tensor(self.tt_posids_host, self.tt_pos, 0)
        ttnn.copy_host_to_device_tensor(self.tt_ext_att_mask_host, self.tt_ext_att_mask, 0)
        ttnn.copy_host_to_device_tensor(self.tt_att_mask_host, self.tt_att_mask, 0)
        ttnn.wait_for_event(0, self.op_event)
        self.write_event = ttnn.record_event(self.device, 0)
        # Use same memory config approach as first run
        if self.input_mem_config == ttnn.L1_MEMORY_CONFIG:
            # Already using interleaved, reuse the same approach
            self.runner_infra.ttnn_input_ids = ttnn.from_torch(
                self.runner_infra.input_ids, dtype=ttnn.uint32, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.runner_infra.ttnn_token_ids = ttnn.from_torch(
                self.runner_infra.token_type_ids,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.runner_infra.ttnn_pos_ids = ttnn.from_torch(
                self.runner_infra.position_ids,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.runner_infra.ttnn_ext_att_mask = ttnn.from_torch(
                self.runner_infra.extended_mask,
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.runner_infra.ttnn_att_mask = ttnn.from_torch(
                self.runner_infra.attention_mask,
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
            self.runner_infra.ttnn_token_ids = ttnn.to_memory_config(self.tt_tokens, self.input_mem_config)
            self.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(self.tt_pos, self.input_mem_config)
            self.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(self.tt_ext_att_mask, self.input_mem_config)
            self.runner_infra.ttnn_att_mask = ttnn.to_memory_config(self.tt_att_mask, self.input_mem_config)
        logger.info("Trace capture: Step 5 - Optimized run executing...")
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        logger.info("Trace capture: Step 5 - Optimized run completed, validating...")
        self.runner_infra.validate()
        logger.info("Trace capture: Step 5 - Validation passed")

        # Capture
        logger.info("Trace capture: Step 6 - Starting trace capture")
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 0)
        ttnn.copy_host_to_device_tensor(self.tt_tokens_host, self.tt_tokens, 0)
        ttnn.copy_host_to_device_tensor(self.tt_posids_host, self.tt_pos, 0)
        ttnn.copy_host_to_device_tensor(self.tt_ext_att_mask_host, self.tt_ext_att_mask, 0)
        ttnn.copy_host_to_device_tensor(self.tt_att_mask_host, self.tt_att_mask, 0)
        ttnn.wait_for_event(0, self.op_event)
        self.write_event = ttnn.record_event(self.device, 0)
        # Use same memory config approach as previous runs
        if self.input_mem_config == ttnn.L1_MEMORY_CONFIG:
            # Already using interleaved, reuse the same approach
            self.runner_infra.ttnn_input_ids = ttnn.from_torch(
                self.runner_infra.input_ids, dtype=ttnn.uint32, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.runner_infra.ttnn_token_ids = ttnn.from_torch(
                self.runner_infra.token_type_ids,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.runner_infra.ttnn_pos_ids = ttnn.from_torch(
                self.runner_infra.position_ids,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.runner_infra.ttnn_ext_att_mask = ttnn.from_torch(
                self.runner_infra.extended_mask,
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.runner_infra.ttnn_att_mask = ttnn.from_torch(
                self.runner_infra.attention_mask,
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
            self.runner_infra.ttnn_token_ids = ttnn.to_memory_config(self.tt_tokens, self.input_mem_config)
            self.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(self.tt_pos, self.input_mem_config)
            self.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(self.tt_ext_att_mask, self.input_mem_config)
            self.runner_infra.ttnn_att_mask = ttnn.to_memory_config(self.tt_att_mask, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.ttnn_input_ids.buffer_address()
        trace_input_addr2 = self.runner_infra.ttnn_token_ids.buffer_address()
        trace_input_addr3 = self.runner_infra.ttnn_pos_ids.buffer_address()
        trace_input_addr4 = self.runner_infra.ttnn_ext_att_mask.buffer_address()
        trace_input_addr5 = self.runner_infra.ttnn_att_mask.buffer_address()
        logger.info("Trace capture: Step 6 - Beginning trace capture...")
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        logger.info("Trace capture: Step 6 - Running model within trace...")
        self.runner_infra.run()
        logger.info("Trace capture: Step 6 - Allocating output tensors...")
        self.ttnn_input_ids = ttnn.allocate_tensor_on_device(spec_input, self.device)
        self.ttnn_token_ids = ttnn.allocate_tensor_on_device(spec_token, self.device)
        self.ttnn_pos_ids = ttnn.allocate_tensor_on_device(spec_pos, self.device)
        self.ttnn_ext_att_mask = ttnn.allocate_tensor_on_device(spec_att, self.device)
        self.ttnn_att_mask = ttnn.allocate_tensor_on_device(spec_att_2, self.device)
        logger.info("Trace capture: Step 6 - Ending trace capture...")
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        logger.info("Trace capture: Step 6 - Synchronizing device...")
        ttnn.synchronize_device(self.device)
        logger.info("Trace capture: Step 6 - Trace capture completed")
        assert trace_input_addr == self.ttnn_input_ids.buffer_address()
        assert trace_input_addr2 == self.ttnn_token_ids.buffer_address()
        assert trace_input_addr3 == self.ttnn_pos_ids.buffer_address()
        assert trace_input_addr4 == self.ttnn_ext_att_mask.buffer_address()
        assert trace_input_addr5 == self.ttnn_att_mask.buffer_address()
        self._has_2_cqs = False

    def _execute_bge_trace_2cqs_inference(
        self, tt_inputs_host=None, tt_tokens=None, tt_posids=None, tt_ext_att_mask=None, tt_att_mask=None
    ):
        """
        Override to use single command queue (CQ 0) instead of 2 command queues.
        """
        if tt_inputs_host is None:
            tt_inputs_host = self.tt_inputs_host
            tt_tokens = self.tt_tokens_host
            tt_posids = self.tt_posids_host
            tt_ext_att_mask = self.tt_ext_att_mask_host
            tt_att_mask = self.tt_att_mask_host

        # Single command queue: wait for previous event, then copy and execute
        ttnn.wait_for_event(0, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_inputs, 0)
        ttnn.copy_host_to_device_tensor(tt_tokens, self.tt_tokens, 0)
        ttnn.copy_host_to_device_tensor(tt_posids, self.tt_pos, 0)
        ttnn.copy_host_to_device_tensor(tt_ext_att_mask, self.tt_ext_att_mask, 0)
        ttnn.copy_host_to_device_tensor(tt_att_mask, self.tt_att_mask, 0)
        self.write_event = ttnn.record_event(self.device, 0)
        ttnn.wait_for_event(0, self.write_event)
        # Handle both sharded and interleaved memory configs
        if self.input_mem_config == ttnn.L1_MEMORY_CONFIG:
            # Using interleaved, create tensors directly
            self.ttnn_input_ids = ttnn.from_torch(
                self.runner_infra.input_ids, dtype=ttnn.uint32, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.ttnn_token_ids = ttnn.from_torch(
                self.runner_infra.token_type_ids,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.ttnn_pos_ids = ttnn.from_torch(
                self.runner_infra.position_ids,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.ttnn_ext_att_mask = ttnn.from_torch(
                self.runner_infra.extended_mask,
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.ttnn_att_mask = ttnn.from_torch(
                self.runner_infra.attention_mask,
                dtype=ttnn.bfloat16,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            # Using sharded, reshard as normal
            self.ttnn_input_ids = ttnn.reshard(self.tt_inputs, self.input_mem_config, self.ttnn_input_ids)
            self.ttnn_token_ids = ttnn.reshard(self.tt_tokens, self.input_mem_config, self.ttnn_token_ids)
            self.ttnn_pos_ids = ttnn.reshard(self.tt_pos, self.input_mem_config, self.ttnn_pos_ids)
            self.ttnn_ext_att_mask = ttnn.reshard(self.tt_ext_att_mask, self.input_mem_config, self.ttnn_ext_att_mask)
            self.ttnn_att_mask = ttnn.reshard(self.tt_att_mask, self.input_mem_config, self.ttnn_att_mask)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        return self.runner_infra.ttnn_output_tensor[0]


class BGEForEmbedding:
    """
    vLLM-compatible wrapper for BGE-Large-EN-v1.5 embedding model.

    This class implements the interface required by vLLM for embedding models,
    enabling OpenAI Embedding API compatibility.
    """

    # Class-level storage for device from initialize_vllm_model
    _device_cache = {}

    def __init__(
        self,
        vllm_config=None,
        prefix: str = "",
        device: Optional[ttnn.Device] = None,
        model_location_generator=None,
        max_batch_size: int = 8,
        max_seq_len: int = 384,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        model_name: str = "BAAI/bge-large-en-v1.5",
    ):
        """
        Initialize the BGE embedding model for vLLM.

        Args:
            vllm_config: vLLM configuration (required by VllmModel protocol)
            prefix: Model prefix (required by VllmModel protocol)
            device: TTNN device instance (required)
            model_location_generator: Optional function to generate model weight paths
            max_batch_size: Maximum batch size for inference
            max_seq_len: Maximum sequence length (default 384 for BGE)
            act_dtype: Activation data type
            weight_dtype: Weight data type
            model_name: HuggingFace model name
        """
        # Store vLLM config if provided
        self.vllm_config = vllm_config

        # Fallback: try to get device from vllm_config if still None
        if device is None and vllm_config is not None:
            # For TT backend, device might be in vllm_config
            device = getattr(vllm_config, "device", None)

        if device is None:
            raise ValueError(
                "device must be provided. " "This usually means initialize_vllm_model() was not called first."
            )

        self.device = device
        self.model_location_generator = model_location_generator
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.model_name = model_name

        # Load config
        self.config = transformers.BertConfig.from_pretrained(model_name)
        # Set attention implementation for reference model
        if not hasattr(self.config, "_attn_implementation") or self.config._attn_implementation is None:
            self.config._attn_implementation = "eager"

        # Determine hardcoded batch size based on device configuration
        # batch_size_per_device = 8 (for N150, N300, etc.)
        # For N300 (2 devices): total batch_size = 8 * 2 = 16
        num_devices = get_num_devices(device)
        self.batch_size_per_device = 8
        self.fixed_batch_size = self.batch_size_per_device * num_devices
        self.fixed_seq_len = 384  # Fixed sequence length like demo.py

        # BERT/BGE models use pad_token_id = 0
        self.pad_token_id = getattr(self.config, "pad_token_id", 0)

        logger.info(
            f"BGE model configured: num_devices={num_devices}, "
            f"batch_size_per_device={self.batch_size_per_device}, "
            f"fixed_batch_size={self.fixed_batch_size}, fixed_seq_len={self.fixed_seq_len}"
        )

        # Initialize runner (will be set up during first forward pass with fixed dimensions)
        self.runner: Optional[BGEPerformantRunner] = None
        self._is_initialized = False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: transformers.PretrainedConfig,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = None,
        tt_data_parallel: int = 1,
        model_location_generator=None,
        optimizations=None,
        vllm_config=None,
    ) -> "BGEForEmbedding":
        """
        Initialize the model for vLLM.

        This is the entry point called by vLLM's TTModelLoader.

        Args:
            hf_config: HuggingFace model configuration
            mesh_device: TTNN mesh device
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length (defaults to 384 for BGE)
            tt_data_parallel: Number of data parallel devices (default: 1)
            model_location_generator: Optional function to generate model weight paths
            optimizations: Optional optimizations (not used for BGE)

        Returns:
            Initialized BGEForEmbedding instance
        """
        if max_seq_len is None:
            max_seq_len = 384  # Default for BGE-large-en-v1.5

        if optimizations is not None:
            logger.warning("Custom optimizations are not supported for BGE model, ignoring optimizations argument")

        logger.info(
            f"Initializing BGE-Large-EN-v1.5 for vLLM: "
            f"max_batch_size={max_batch_size}, max_seq_len={max_seq_len}, tt_data_parallel={tt_data_parallel}"
        )

        # For now, BGE only supports single device (tt_data_parallel=1)
        # Multi-device support can be added later if needed
        if tt_data_parallel != 1:
            logger.warning(
                f"BGE model currently only supports tt_data_parallel=1, "
                f"but {tt_data_parallel} was requested. Using single device."
            )

        # Create and return an instance directly
        # Note: vLLM wraps the class before calling initialize_vllm_model,
        # so cls is already wrapped in ModelForPooling, which requires vllm_config
        instance = cls(
            vllm_config=vllm_config,  # Required by ModelForPooling wrapper
            prefix="",
            device=mesh_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        return instance

    def _initialize_runner(self):
        """Initialize the BGE runner with hardcoded batch size and sequence length based on device configuration."""
        if self._is_initialized and self.runner is not None:
            return

        # Use hardcoded dimensions based on device configuration
        batch_size = self.fixed_batch_size
        sequence_length = self.fixed_seq_len

        logger.info(
            f"Initializing BGE runner with fixed dimensions: batch_size={batch_size}, seq_len={sequence_length}"
        )

        # Create dummy inputs for initialization with fixed dimensions
        input_ids = torch.randint(
            low=0,
            high=self.config.vocab_size - 1,
            size=[batch_size, sequence_length],
            dtype=torch.int64,
        )
        attention_mask = torch.ones(batch_size, sequence_length)
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        token_type_ids = torch.zeros([batch_size, sequence_length], dtype=torch.int64)
        position_ids = torch.arange(0, sequence_length, dtype=torch.int64).unsqueeze(dim=0)

        # Initialize runner with single-CQ adapter for vLLM
        # vLLM devices typically only have 1 command queue (CQ 0)
        self.runner = BGEPerformantRunnerSingleCQ(
            device=self.device,
            model_location_generator=self.model_location_generator,
            device_batch_size=batch_size,
            sequence_length=sequence_length,
            input_ids=input_ids,
            extended_mask=extended_mask,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            act_dtype=self.act_dtype,
            weight_dtype=self.weight_dtype,
            model_name=self.model_name,
        )

        # Capture trace for optimized inference (uses single CQ)
        logger.info("Starting trace capture...")
        try:
            self.runner._capture_bge_trace_2cqs()  # Actually uses single CQ via override
            logger.info("Trace capture completed successfully")
        except Exception as e:
            logger.error(f"Trace capture failed: {e}")
            raise
        self._is_initialized = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for embedding generation.

        This is the main interface method called by vLLM for embedding inference.
        Inputs are padded to match the hardcoded batch_size and seq_len based on device configuration.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask tensor
            token_type_ids: Optional token type IDs tensor
            position_ids: Optional position IDs tensor

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        input_batch_size, input_seq_len = input_ids.shape
        original_batch_size = input_batch_size
        original_seq_len = input_seq_len

        # Initialize runner with fixed dimensions if not already done
        if not self._is_initialized:
            self._initialize_runner()

        # Pad inputs to match fixed dimensions
        # Pad batch dimension if needed
        if input_batch_size < self.fixed_batch_size:
            batch_pad = self.fixed_batch_size - input_batch_size
            # Pad with pad_token_id for input_ids, 0 for others
            input_ids = torch.nn.functional.pad(input_ids, (0, 0, 0, batch_pad), value=self.pad_token_id)
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, batch_pad), value=0)
            else:
                attention_mask = torch.ones(self.fixed_batch_size, input_seq_len, dtype=torch.float32)
                attention_mask[input_batch_size:] = 0  # Set padding to 0
            if token_type_ids is not None:
                token_type_ids = torch.nn.functional.pad(token_type_ids, (0, 0, 0, batch_pad), value=0)
            else:
                token_type_ids = torch.zeros([self.fixed_batch_size, input_seq_len], dtype=torch.int64)
            if position_ids is not None:
                # Repeat position_ids for padded batches
                position_ids = torch.cat(
                    [
                        position_ids,
                        position_ids[-1:].repeat(batch_pad, 1)
                        if position_ids.shape[0] > 0
                        else torch.arange(0, input_seq_len, dtype=torch.int64).unsqueeze(0).repeat(batch_pad, 1),
                    ],
                    dim=0,
                )
            else:
                position_ids = (
                    torch.arange(0, input_seq_len, dtype=torch.int64).unsqueeze(0).repeat(self.fixed_batch_size, 1)
                )
            input_batch_size = self.fixed_batch_size
        elif input_batch_size > self.fixed_batch_size:
            raise ValueError(
                f"Input batch size {input_batch_size} exceeds fixed batch size {self.fixed_batch_size}. "
                f"Please ensure vLLM max_batch_size <= {self.fixed_batch_size}"
            )
        else:
            # Batch size matches, but ensure other tensors are created if None
            if attention_mask is None:
                attention_mask = torch.ones(input_batch_size, input_seq_len, dtype=torch.float32)
            if token_type_ids is None:
                token_type_ids = torch.zeros([input_batch_size, input_seq_len], dtype=torch.int64)
            if position_ids is None:
                position_ids = (
                    torch.arange(0, input_seq_len, dtype=torch.int64).unsqueeze(0).repeat(input_batch_size, 1)
                )

        # Pad sequence length dimension if needed
        if input_seq_len < self.fixed_seq_len:
            seq_pad = self.fixed_seq_len - input_seq_len
            input_ids = torch.nn.functional.pad(input_ids, (0, seq_pad), value=self.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, seq_pad), value=0)
            token_type_ids = torch.nn.functional.pad(token_type_ids, (0, seq_pad), value=0)
            # For position_ids, extend the sequence
            if position_ids.shape[1] == input_seq_len:
                # Extend position_ids with sequential values
                extended_positions = (
                    torch.arange(input_seq_len, self.fixed_seq_len, dtype=torch.int64)
                    .unsqueeze(0)
                    .repeat(input_batch_size, 1)
                )
                position_ids = torch.cat([position_ids, extended_positions], dim=1)
            else:
                # Recreate position_ids for the full sequence
                position_ids = (
                    torch.arange(0, self.fixed_seq_len, dtype=torch.int64).unsqueeze(0).repeat(input_batch_size, 1)
                )
            input_seq_len = self.fixed_seq_len
        elif input_seq_len > self.fixed_seq_len:
            # Truncate if longer (shouldn't happen if max_seq_len is set correctly)
            logger.warning(
                f"Input sequence length {input_seq_len} exceeds fixed length {self.fixed_seq_len}, truncating"
            )
            input_ids = input_ids[:, : self.fixed_seq_len]
            attention_mask = attention_mask[:, : self.fixed_seq_len]
            token_type_ids = token_type_ids[:, : self.fixed_seq_len]
            position_ids = position_ids[:, : self.fixed_seq_len]
            input_seq_len = self.fixed_seq_len

        # Create extended attention mask
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)

        # Run inference with padded inputs
        output = self.runner.run(
            input_ids=input_ids,
            tokens=token_type_ids,
            posids=position_ids,
            ext_att_mask=extended_mask,
            att_mask=attention_mask,
        )

        # Convert TTNN tensor to PyTorch tensor
        embeddings = ttnn.to_torch(
            output,
            dtype=torch.float32,
            mesh_composer=self.runner.runner_infra.output_mesh_composer,
        )

        # Squeeze batch dimension if needed (BGE returns shape [batch_size, 1, embedding_dim])
        if embeddings.dim() == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)

        # Return only the embeddings for the original (non-padded) batch
        if original_batch_size < self.fixed_batch_size:
            embeddings = embeddings[:original_batch_size]

        return embeddings

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[torch.Tensor]] = None,
        prompt_lens: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        sampling_params: Optional[Any] = None,
        cross_page_table: Optional[torch.Tensor] = None,
        empty_slots: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Prefill forward pass for embedding generation.

        This method is called by vLLM's TT model runner for prefill phase.
        For embedding models, this is the same as forward() since we don't
        have separate prefill/decode phases and don't use KV cache.

        Args:
            tokens: Token IDs tensor of shape (batch_size, seq_len)
            page_table: Block table (not used for embedding models)
            kv_cache: KV cache (not used for embedding models)
            prompt_lens: Prompt lengths (not used for embedding models)
            start_pos: Start positions (not used for embedding models)
            sampling_params: Sampling parameters (not used for embedding models)
            cross_page_table: Cross attention page table (not used)
            empty_slots: Empty slots (not used)
            **kwargs: Additional arguments

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        # For embedding models, we only need the tokens
        # Ignore KV cache, page tables, and other generation-specific parameters
        input_ids = tokens

        # Extract optional parameters from kwargs
        attention_mask = kwargs.get("attention_mask", None)
        token_type_ids = kwargs.get("token_type_ids", None)

        # Generate position_ids from tokens shape if not provided
        batch_size, seq_len = input_ids.shape
        if "positions" in kwargs:
            position_ids = kwargs["positions"]
        elif start_pos is not None:
            # Use start_pos if provided
            position_ids = start_pos
        else:
            # Generate default position IDs
            position_ids = torch.arange(0, seq_len, dtype=torch.int64).unsqueeze(dim=0).repeat(batch_size, 1)

        # Call the main forward method
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.config.hidden_size

    def get_max_seq_len(self) -> int:
        """Return the maximum sequence length."""
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        """Return the maximum batch size."""
        return self.max_batch_size


def register_model():
    """
    Register the BGE model with vLLM's ModelRegistry.

    This function should be called to make the model available to vLLM.
    Typically called from vLLM's model registration code.

    Note: vLLM constructs model class names as "TT" + architecture name.
    For BertModel architecture, it expects "TTBertModel".
    """
    try:
        from vllm.model_executor.model_loader import ModelRegistry

        # Register with name "TTBertModel" (vLLM constructs this from architecture "BertModel")
        ModelRegistry.register_model(
            "TTBertModel",
            "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
        )
        logger.info("Successfully registered TTBertModel (BGE-Large-EN-v1.5) with vLLM ModelRegistry")
    except ImportError:
        logger.warning(
            "vLLM ModelRegistry not available. "
            "Make sure vLLM is installed and the model is registered in vLLM's model loader."
        )
