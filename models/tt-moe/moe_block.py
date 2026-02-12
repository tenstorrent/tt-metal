# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unified MoE Block implementation that can be configured via JSON.

This module provides a single MoEBlock class that can be configured to support
different MoE architectures including DeepSeek-V3 and GPT-OSS through JSON configuration.
"""

import json
from pathlib import Path

import torch
from loguru import logger

import ttnn

try:
    # Try relative imports (when used as a module)
    from .components.experts.distributed_expert import DistributedExpert
    from .components.experts.shared_expert import SharedExpert
    from .components.routers.moe_gate import MoEGateRouter
    from .utils.lazy_state_dict import LazyStateDict
except ImportError:
    # Fall back to absolute imports (when testing)
    from components.experts.distributed_expert import DistributedExpert
    from components.experts.shared_expert import SharedExpert
    from components.routers.moe_gate import MoEGateRouter
    from utils.lazy_state_dict import LazyStateDict


class MoEBlock:
    """
    Configurable MoE block that supports multiple architectures.

    This class can be configured via JSON to support:
    - DeepSeek-V3 style MoE with shared experts
    - GPT-OSS style MoE with throughput experts
    - Other MoE architectures through configuration
    """

    def __init__(self, config_path: str, mesh_device: ttnn.MeshDevice, ccl=None):
        """
        Initialize MoE block from configuration file.

        Args:
            config_path: Path to JSON configuration file
            mesh_device: TTNN mesh device for tensor placement
            ccl: CCL instance for collective operations
        """
        self.mesh_device = mesh_device
        self.ccl = ccl

        # Load configuration
        self.config = self._load_config(config_path)

        # Extract tensor parallel configuration
        self.tp_config = self.config.get("tensor_parallel", {"enabled": False})
        self.tp_enabled = self.tp_config.get("enabled", False)

        if self.tp_enabled:
            self.tp_axis = self.tp_config["cluster_axis"]
            self.tp_size = mesh_device.shape[self.tp_axis]
        else:
            self.tp_axis = None
            self.tp_size = 1

        # Initialize components
        self._init_router()
        self._init_experts()
        self._init_expert_mapping_tensors()

        # Load weights if path is specified
        self._load_weights_if_specified()

    def _load_config(self, config_path: str) -> dict:
        """Load JSON configuration file."""
        with open(config_path, "r") as f:
            full_config = json.load(f)
        return full_config["moe_block"]

    def _init_router(self):
        """Initialize router based on configuration."""
        router_config = self.config["router"]
        router_type = router_config["type"]

        if router_type == "moe_gate":
            self.router = MoEGateRouter(router_config["config"], self.mesh_device)
        elif router_type == "topk":
            # Import and create TopKRouter when implemented
            # self.router = TopKRouter(router_config["config"], self.mesh_device)
            raise NotImplementedError("TopKRouter not yet implemented")
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    def _init_experts(self):
        """Initialize expert configurations based on JSON configuration."""
        experts_config = self.config["experts"]

        # Setup distributed expert configuration if enabled
        self.distributed_expert_enabled = False
        self.distributed_expert_config = None
        self.distributed_expert_weights = None

        if experts_config.get("distributed", {}).get("enabled", False):
            dist_config = experts_config["distributed"]
            # Add n_routed_experts from router config if not present
            if "n_routed_experts" not in dist_config:
                dist_config["n_routed_experts"] = self.config["router"]["config"].get("n_routed_experts", 256)

            self.distributed_expert_enabled = True
            self.distributed_expert_config = dist_config
            self.ep_axis = dist_config["dispatch_cluster_axis"]

            # Calculate num_experts_per_device from config
            n_routed_experts = dist_config["n_routed_experts"]
            num_devices = self.mesh_device.get_num_devices()
            self.num_experts_per_device = n_routed_experts // num_devices

        # Setup shared expert configuration if enabled
        self.shared_expert_enabled = False
        self.shared_expert_config = None
        self.shared_expert_weights = None

        if experts_config.get("shared", {}).get("enabled", False):
            shared_config = experts_config["shared"]
            self.shared_expert_enabled = True
            self.shared_expert_config = shared_config
            self.shared_parallel = shared_config.get("parallel_with_moe", False)

    def _init_expert_mapping_tensors(self):
        """Initialize expert mapping tensors for all-to-all operations."""
        if not self.distributed_expert_enabled:
            return

        num_devices = self.mesh_device.get_num_devices()
        num_experts_per_device = self.num_experts_per_device
        num_dispatch_device_rows = self.mesh_device.shape[0] if hasattr(self.mesh_device, "shape") else 1

        # Create expert mapping tensor for dispatch/combine
        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(num_devices, dtype=torch.int32)
            .repeat_interleave(num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Create remap topk mask
        n_routed_experts = self.config["router"]["config"].get("n_routed_experts", 256)
        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, num_dispatch_device_rows, 1, n_routed_experts), dtype=torch.bfloat16),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _load_weights_if_specified(self):
        """Load weights from model path if specified in JSON configuration."""
        weight_path = self.config.get("weight_path")
        if not weight_path:
            return

        module_prefix = self.config.get("module_prefix", "")
        layer_index = self.config.get("layer_index", 0)

        # Load state dict lazily from model files
        from loguru import logger

        logger.info(f"Loading weights from {weight_path} (as specified in JSON config)")
        logger.info(f"Using module_prefix: {module_prefix}, layer_index: {layer_index}")

        with LazyStateDict(Path(weight_path)) as lazy_dict:
            if module_prefix:
                # Create a view for the specific module
                state_dict = lazy_dict.view_with_prefix(module_prefix + ".")
            else:
                state_dict = lazy_dict

            # Load weights for each component
            self.load_weights(state_dict)

    def load_weights(self, state_dict: dict):
        """
        Load weights for all components.

        Args:
            state_dict: Dictionary containing weights for router and experts
        """
        # Load router weights (MoEGate in DeepSeek)
        if hasattr(self, "router"):
            router_state = {}
            # MoEGate weights in DeepSeek - pass through directly
            for key in ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]:
                if key in state_dict:
                    # Remove the mlp.gate. prefix
                    simple_key = key.replace("mlp.gate.", "")
                    router_state[simple_key] = state_dict[key]

            if router_state:
                self.router.load_weights(router_state)

        # Convert distributed expert weights using class method
        if self.distributed_expert_enabled:
            # Pass through all mlp.experts.* keys for per-expert loading
            expert_state = {k.replace("mlp.", ""): v for k, v in state_dict.items() if k.startswith("mlp.experts.")}

            if expert_state:
                # Create a config object from JSON config for weight conversion
                class ExpertConfig:
                    def __init__(self, json_config):
                        self.n_routed_experts = json_config["n_routed_experts"]
                        self.hidden_size = json_config["hidden_size"]
                        self.moe_intermediate_size = json_config["intermediate_size"]
                        self.quantization_config = {
                            "weight_block_size": json_config.get("weight_block_size", [128, 128])
                        }

                config = ExpertConfig(self.distributed_expert_config)

                # Convert weights using class method
                weight_configs = DistributedExpert.convert_weights(
                    config, (expert_state,), Path("/tmp/moe_block_expert_weights"), self.mesh_device
                )

                # Store the weight configs for forward pass
                self.distributed_expert_weights = weight_configs

                # Also create decode model config for forward pass
                self.distributed_expert_decode_config = DistributedExpert.decode_model_config(config, self.mesh_device)

                # Merge weight configs into decode config
                for key, value in self.distributed_expert_weights.items():
                    if key in self.distributed_expert_decode_config:
                        self.distributed_expert_decode_config[key].update(value)
                    else:
                        self.distributed_expert_decode_config[key] = value

        # Convert shared expert weights using class method
        if self.shared_expert_enabled:
            shared_state = {}
            # Shared expert weights in DeepSeek
            if "mlp.shared_experts.gate_proj.weight" in state_dict:
                shared_state["gate_proj.weight"] = state_dict["mlp.shared_experts.gate_proj.weight"]
            if "mlp.shared_experts.up_proj.weight" in state_dict:
                shared_state["up_proj.weight"] = state_dict["mlp.shared_experts.up_proj.weight"]
            if "mlp.shared_experts.down_proj.weight" in state_dict:
                shared_state["down_proj.weight"] = state_dict["mlp.shared_experts.down_proj.weight"]

            if shared_state:
                # Create a config object from JSON config for weight conversion
                class SharedConfig:
                    def __init__(self, json_config):
                        self.hidden_size = json_config["hidden_size"]
                        self.moe_intermediate_size = json_config["intermediate_size"]
                        self.quantization_config = {
                            "weight_block_size": json_config.get("weight_block_size", [128, 128])
                        }

                config = SharedConfig(self.shared_expert_config)

                # Convert weights using class method
                weight_configs = SharedExpert.convert_weights(
                    config, (shared_state,), Path("/tmp/moe_block_shared_weights"), self.mesh_device
                )

                # Store the weight configs for forward pass
                self.shared_expert_weights = weight_configs

                # Also create decode model config for forward pass
                self.shared_expert_decode_config = SharedExpert.decode_model_config(config, self.mesh_device)

                # Merge weight configs into decode config
                for key in ["w1", "w2", "w3"]:
                    if key in weight_configs:
                        self.shared_expert_decode_config[key].update(weight_configs[key])

    def _prepare_expert_weights(self, weights: ttnn.Tensor, mode: str = "decode"):
        """
        Prepare expert weights by repeating and permuting.

        This follows the _fwd_repeat_permute_expert_weights pattern from DeepSeek.

        Args:
            weights: Expert weights from router [batch, seq_len, num_experts_per_tok]
            mode: "decode" or "prefill" mode

        Returns:
            Prepared weights tensor
        """
        # Get configuration for weight repeat
        weight_repeat_config = self.config.get("topk_weights_repeat", {})
        if not weight_repeat_config:
            # Default repeat dimensions based on hidden size
            hidden_size = self.config["experts"]["distributed"]["hidden_size"]
            repeat_dims = (hidden_size, 1, 1, 1)
        else:
            repeat_dims = weight_repeat_config.get("repeat_dims", (1, 1, 1, 1))

        # Convert to ROW_MAJOR for repeat
        weights_rm = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT)

        # Repeat weights (use DRAM for large repeat operations)
        weights_rm = ttnn.repeat(weights_rm, ttnn.Shape(repeat_dims), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Permute dimensions (3, 1, 2, 0) following DeepSeek pattern
        # Use DRAM for permute of large tensors
        weights_rm = ttnn.permute(weights_rm, (3, 1, 2, 0), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Convert back to TILE_LAYOUT (can move to L1 if needed for subsequent operations)
        weights = ttnn.to_layout(weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(weights_rm)

        return weights

    def _forward_moe(self, x: ttnn.Tensor, indices: ttnn.Tensor, weights: ttnn.Tensor, mode: str = "decode"):
        """
        Forward pass through MoE experts following DeepSeek's _fwd_moe pattern.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            indices: Expert indices from router
            weights: Expert weights from router (already prepared)
            mode: "decode" or "prefill" mode

        Returns:
            MoE output tensor
        """
        # Get dimensions
        batch_size_per_device = x.shape[-2]  # In prefill, this is seq_len_per_device
        seq_len = 1  # For a2a dispatch/combine compatibility
        hidden_size = x.shape[-1]

        # Get configuration
        num_dispatch_devices = self.mesh_device.shape[0] if hasattr(self.mesh_device, "shape") else 1
        batch_size = batch_size_per_device * num_dispatch_devices
        num_experts_per_tok = self.config["router"]["config"]["num_experts_per_tok"]

        # Convert to ROW_MAJOR and reshape
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, hidden_size))

        indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        indices_rm = ttnn.reshape(indices_rm, shape=(batch_size_per_device, 1, seq_len, num_experts_per_tok))

        # Chunking for prefill to avoid OOM
        chunk_config = self.config.get("moe_chunking", {})
        chunk_size = min(batch_size_per_device, max(1, chunk_config.get("chunk_size", batch_size_per_device)))

        output_chunks = []
        tokens = batch_size * seq_len

        for batch_start in range(0, batch_size_per_device, chunk_size):
            batch_end = min(batch_start + chunk_size, batch_size_per_device)
            batch_chunk = batch_end - batch_start
            batch_size_chunk = batch_chunk * num_dispatch_devices

            # Slice inputs for this chunk
            x_chunk = ttnn.slice(x_rm, [batch_start, 0, 0, 0], [batch_end, 1, seq_len, hidden_size])
            indices_chunk = ttnn.slice(indices_rm, [batch_start, 0, 0, 0], [batch_end, 1, seq_len, num_experts_per_tok])

            # All-to-all dispatch
            logger.info(
                f"DEBUG: Before all_to_all_dispatch - x_chunk shape: {x_chunk.shape}, indices_chunk shape: {indices_chunk.shape}"
            )
            logger.info(
                f"DEBUG: EP axis: {self.ep_axis if self.distributed_expert_enabled else 0}, num_experts_per_device: {self.num_experts_per_device}"
            )
            logger.info(
                f"DEBUG: expert_mapping_tensors type: {type(self.expert_mapping_tensors)}, length: {len(self.expert_mapping_tensors) if hasattr(self.expert_mapping_tensors, '__len__') else 'N/A'}"
            )
            logger.info(f"DEBUG: memory_config: {self._get_memory_config(mode, 'dispatch')}")
            logger.info("DEBUG: *** CALLING all_to_all_dispatch NOW ***")

            dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
                x_chunk,
                indices_chunk,
                self.expert_mapping_tensors,
                cluster_axis=self.ep_axis if self.distributed_expert_enabled else 0,
                memory_config=self._get_memory_config(mode, "dispatch"),
                topology=ttnn.Topology.Linear,
            )

            logger.info("DEBUG: *** all_to_all_dispatch COMPLETED ***")
            ttnn.deallocate(x_chunk)
            ttnn.deallocate(indices_chunk)

            logger.info(f"DEBUG: After all_to_all_dispatch - dispatch_output shape: {dispatch_output.shape}")

            # Reshape and repeat activations
            dispatch_chunk = ttnn.reshape(dispatch_output, shape=(1, 1, batch_size_chunk * seq_len, hidden_size))
            logger.info(f"DEBUG: After reshape - dispatch_chunk shape: {dispatch_chunk.shape}")

            # Apply activations_repeat configuration
            activations_repeat_config = self.config.get("activations_repeat", {})
            if activations_repeat_config:
                # Use the repeat dims from config or default to num_experts_per_device
                repeat_dims = activations_repeat_config.get("repeat_dims", [1, self.num_experts_per_device, 1, 1])
                logger.info(
                    f"DEBUG: Repeating with dims: {repeat_dims} (num_experts_per_device={self.num_experts_per_device})"
                )
                # Use appropriate memory config based on mode
                repeat_memory_config = self._get_memory_config(mode, "dispatch")
                dispatch_chunk = ttnn.repeat(
                    dispatch_chunk, ttnn.Shape(repeat_dims), memory_config=repeat_memory_config
                )
                logger.info(f"DEBUG: After repeat - dispatch_chunk shape: {dispatch_chunk.shape}")

            dispatch_chunk = ttnn.to_layout(dispatch_chunk, ttnn.TILE_LAYOUT)
            ttnn.deallocate(dispatch_output)

            # Run experts
            if self.distributed_expert_enabled:
                # Use class method forward_decode with the stored config
                experts_output = DistributedExpert.forward_decode(dispatch_chunk, self.distributed_expert_decode_config)
            else:
                # Fallback if no distributed expert configured
                experts_output = dispatch_chunk
            ttnn.deallocate(dispatch_chunk)

            # Reshape expert output
            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
            experts_output = ttnn.reshape(
                experts_output, shape=(self.num_experts_per_device, batch_size_chunk, seq_len, hidden_size)
            )

            # Reshape metadata for combine
            dispatch_metadata = ttnn.reshape(
                dispatch_metadata, shape=(1, batch_size_chunk, seq_len, num_experts_per_tok)
            )

            # All-to-all combine
            combine_output = ttnn.all_to_all_combine(
                experts_output,
                dispatch_metadata,
                self.expert_mapping_tensors,
                cluster_axis=self.ep_axis if self.distributed_expert_enabled else 0,
                memory_config=self._get_memory_config(mode, "combine"),
                topology=ttnn.Topology.Linear,
            )
            ttnn.deallocate(experts_output)
            ttnn.deallocate(dispatch_metadata)

            # Reshape and apply weights
            post_combine = ttnn.reshape(
                combine_output, shape=(num_experts_per_tok, 1, batch_chunk * seq_len, hidden_size)
            )
            post_combine = ttnn.to_layout(post_combine, ttnn.TILE_LAYOUT)

            # Slice weights for this chunk
            token_start = batch_start * seq_len
            token_end = batch_end * seq_len
            weights_chunk = ttnn.slice(
                weights, [0, 0, token_start, 0], [num_experts_per_tok, 1, token_end, hidden_size]
            )

            # Multiply by weights
            post_combine = ttnn.mul(post_combine, weights_chunk, memory_config=self._get_memory_config(mode, "combine"))
            ttnn.deallocate(weights_chunk)

            # Sum across experts dimension
            post_combine = ttnn.sum(post_combine, dim=0, keepdim=True)
            output_chunks.append(post_combine)

        # Combine chunks if multiple
        if len(output_chunks) == 1:
            output = output_chunks[0]
        else:
            output = ttnn.concat(output_chunks, dim=2)
            for chunk in output_chunks:
                ttnn.deallocate(chunk)

        # Cleanup
        ttnn.deallocate(x_rm)
        ttnn.deallocate(indices_rm)

        return output

    def forward(self, x: ttnn.Tensor, mode: str = "decode"):
        """
        Forward pass through MoE block.

        This implements the pattern:
        1. Optional all-gather if tensor parallel is enabled
        2. Router forward to get expert assignments
        3. Expert forward (distributed and/or shared)
        4. Combine outputs if multiple experts
        5. Optional reduce-scatter if tensor parallel is enabled

        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or TP-sharded
            mode: "decode" or "prefill" mode

        Returns:
            MoE output tensor with same shape as input
        """
        logger.info(f"DEBUG MoEBlock.forward: Starting with input shape {x.shape}, mode={mode}")
        logger.info(
            f"DEBUG MoEBlock.forward: TP enabled={self.tp_enabled}, TP axis={self.tp_axis}, TP size={self.tp_size}"
        )

        # Check for chunking in prefill mode
        if mode == "prefill":
            prefill_chunk_config = self.config.get("prefill_chunking", {})
            chunk_tokens = prefill_chunk_config.get("chunk_size", 16384)
            num_dispatch_devices = self.mesh_device.shape[0] if hasattr(self.mesh_device, "shape") else 1
            global_tokens = x.shape[2] * num_dispatch_devices

            if global_tokens > chunk_tokens:
                chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
                return self._forward_chunked_prefill(x, chunk_size)

        # Step 1: All-gather if TP is enabled and input is sharded
        x_was_gathered = False
        is_sharded = self._is_tp_sharded(x)
        logger.info(
            f"DEBUG MoEBlock.forward: _is_tp_sharded returned {is_sharded} (actual_dim={x.shape[-1]}, expected_sharded={self.config['experts']['distributed']['hidden_size'] // self.tp_size if self.tp_enabled else 'N/A'})"
        )

        if self.tp_enabled and self._is_tp_sharded(x):
            # Get CCL parameters for all_gather
            all_gather_config = {
                "cluster_axis": self.tp_axis,
                "dim": -1,  # Last dimension for hidden states
                "memory_config": self._get_memory_config(mode, "all_gather"),
                "topology": ttnn.Topology.Linear,
            }

            # Use CCL to populate runtime arguments (semaphores)
            if self.ccl is not None:
                all_gather_args = self.ccl.populate_all_gather_runtime_args(all_gather_config)
            else:
                raise ValueError("CCL instance is required for tensor parallel operations")

            x = ttnn.experimental.all_gather_async(x, **all_gather_args)
            x_was_gathered = True

        # Step 2: Router forward
        logger.info(f"DEBUG MoEBlock.forward: Calling router.forward with input shape {x.shape}")
        weights, indices = self.router.forward(x, mode)
        logger.info(
            f"DEBUG MoEBlock.forward: Router returned weights shape={weights.shape}, indices shape={indices.shape}"
        )

        # Prepare expert weights
        logger.info(f"DEBUG MoEBlock.forward: Preparing expert weights")
        weights_prepared = self._prepare_expert_weights(weights, mode)
        logger.info(f"DEBUG MoEBlock.forward: Weights prepared shape={weights_prepared.shape}")

        # Step 3: Expert computation
        outputs = []

        # Distributed experts with MoE forward
        if self.distributed_expert_enabled:
            logger.info(f"DEBUG MoEBlock.forward: Calling _forward_moe with distributed experts")
            moe_output = self._forward_moe(x, indices, weights_prepared, mode)
            logger.info(
                f"DEBUG MoEBlock.forward: _forward_moe returned shape={moe_output.shape if moe_output else 'None'}"
            )
            outputs.append(moe_output)

        # Shared expert (if enabled and parallel with MoE)
        if self.shared_expert_enabled and self.shared_parallel:
            logger.info(f"DEBUG MoEBlock.forward: Running SharedExpert.forward_decode")
            # Use class method forward_decode with the stored config
            shared_out = SharedExpert.forward_decode(x, self.shared_expert_decode_config)
            logger.info(
                f"DEBUG MoEBlock.forward: SharedExpert returned shape={shared_out.shape if shared_out else 'None'}"
            )
            outputs.append(shared_out)

        # Step 4: Combine outputs
        if len(outputs) > 1:
            # Add MoE and shared expert outputs
            output = ttnn.add(outputs[0], outputs[1])
            for out in outputs:
                if out is not outputs[-1]:  # Don't deallocate the last one (it's output)
                    ttnn.deallocate(out)
        else:
            output = outputs[0]

        # Step 5: Reduce-scatter if we did all-gather
        if x_was_gathered:
            # Get CCL parameters for reduce_scatter
            reduce_scatter_config = {
                "cluster_axis": self.tp_axis,
                "dim": 3,  # Last dimension after batching
                "memory_config": self._get_memory_config(mode, "reduce_scatter"),
                "topology": ttnn.Topology.Linear,
            }

            # Use CCL to populate runtime arguments (semaphores)
            if self.ccl is not None:
                reduce_scatter_args = self.ccl.populate_reduce_scatter_runtime_args(reduce_scatter_config)
            else:
                raise ValueError("CCL instance is required for tensor parallel operations")

            output = ttnn.experimental.reduce_scatter_minimal_async(output, **reduce_scatter_args)
            # Cleanup gathered tensor
            ttnn.deallocate(x)

        # Cleanup
        ttnn.deallocate(weights_prepared)

        return output

    def _forward_chunked_prefill(self, x: ttnn.Tensor, chunk_size: int):
        """
        Forward pass with chunking for large prefill sequences.

        Args:
            x: Input tensor [batch, 1, seq_len, hidden_dim]
            chunk_size: Size of each chunk

        Returns:
            MoE output tensor
        """
        chunk_size = max(1, chunk_size)
        _, _, seq_len, _ = x.shape
        output_chunks = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x_chunk = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, x.shape[3]])
            output_chunks.append(self.forward(x_chunk, mode="prefill"))
            ttnn.deallocate(x_chunk)

        if len(output_chunks) == 1:
            return output_chunks[0]

        output = ttnn.concat(output_chunks, dim=2)
        for chunk in output_chunks:
            ttnn.deallocate(chunk)
        return output

    def _is_tp_sharded(self, x: ttnn.Tensor) -> bool:
        """
        Check if input is tensor-parallel sharded.

        Args:
            x: Input tensor

        Returns:
            True if tensor is TP-sharded, False otherwise
        """
        if not self.tp_enabled:
            return False

        hidden_size = self.config["experts"]["distributed"]["hidden_size"]
        actual_dim = x.shape[-1]
        expected_sharded_dim = hidden_size // self.tp_size

        return actual_dim == expected_sharded_dim

    def _get_memory_config(self, mode: str, op_type: str):
        """
        Get memory configuration for specific operation and mode.

        Args:
            mode: "decode" or "prefill"
            op_type: Type of operation (all_gather, reduce_scatter, dispatch, combine, etc.)

        Returns:
            TTNN memory configuration
        """
        # Check for specific memory config in configuration
        memory_config_overrides = self.config.get("memory_configs", {})

        # Look for mode-specific override first
        mode_configs = memory_config_overrides.get(mode, {})
        if op_type in mode_configs:
            return getattr(ttnn, mode_configs[op_type])

        # Default configurations
        if mode == "decode":
            if op_type in ["all_gather", "reduce_scatter"]:
                return ttnn.create_sharded_memory_config(
                    shape=(32, self.config["experts"]["distributed"]["hidden_size"] // self.tp_size),
                    core_grid=ttnn.CoreGrid(y=7, x=4),
                    strategy=ttnn.ShardStrategy.WIDTH,
                )
            else:
                return ttnn.L1_MEMORY_CONFIG
        else:  # prefill
            return ttnn.DRAM_MEMORY_CONFIG

    @classmethod
    def from_deepseek_state(cls, state_dict: dict, mesh_device: ttnn.MeshDevice, ccl=None, config_path: str = None):
        """
        Create MoEBlock from DeepSeek-V3 state dict.

        This is a convenience method to create a MoEBlock configured for
        DeepSeek-V3 and load its weights.

        Args:
            state_dict: DeepSeek-V3 state dict with MoE weights
            mesh_device: TTNN mesh device
            ccl: CCL instance
            config_path: Path to config, defaults to deepseek_v3.json

        Returns:
            Initialized MoEBlock ready for inference
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "deepseek_v3.json")

        moe_block = cls(config_path, mesh_device, ccl)
        moe_block.load_weights(state_dict)
        return moe_block
