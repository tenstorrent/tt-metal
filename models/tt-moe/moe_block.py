# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Simplified MoE Block implementation that can be configured via JSON.

This module provides a single MoEBlock class that can be configured to support
different MoE architectures including DeepSeek-V3 and GPT-OSS through simplified JSON configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch
from loguru import logger

import ttnn

# Component imports
try:
    from .components.collective.all_to_all_ops import AllToAllConfig
    from .components.experts.distributed_expert import DistributedExpert
    from .components.experts.shared_expert import SharedExpert
    from .components.routers.moe_gate import MoEGateRouter
    from .components.routers.topk_router import TopKRouter
    from .utils.lazy_state_dict import LazyStateDict
except ImportError:
    from components.collective.all_to_all_ops import AllToAllConfig
    from components.experts.distributed_expert import DistributedExpert
    from components.experts.shared_expert import SharedExpert
    from components.routers.moe_gate import MoEGateRouter
    from components.routers.topk_router import TopKRouter
    from utils.lazy_state_dict import LazyStateDict


class MoEBlock:
    """
    Simplified configurable MoE block that supports multiple architectures.

    Key simplifications:
    - Unified model_params section for core parameters
    - Automatic derivation of redundant parameters
    - Simplified memory configuration (auto/L1/DRAM)
    - Cleaner weight loading logic
    - Removed unused experimental features
    """

    def __init__(self, config_path: str, mesh_device: ttnn.MeshDevice, ccl=None):
        """
        Initialize MoE block from simplified configuration file.

        Args:
            config_path: Path to JSON configuration file
            mesh_device: TTNN mesh device for tensor placement
            ccl: CCL instance for collective operations
        """
        self.mesh_device = mesh_device
        self.ccl = ccl

        # Load and validate configuration
        self.config = self._load_and_validate_config(config_path)

        # Extract core parameters
        self.model_params = self.config["model_params"]
        self.num_experts = self.model_params["num_experts"]
        self.num_experts_per_tok = self.model_params["num_experts_per_tok"]
        self.hidden_size = self.model_params["hidden_size"]
        self.intermediate_size = self.model_params["intermediate_size"]

        # Derive expert distribution
        self.num_devices = mesh_device.get_num_devices()
        self.num_experts_per_device = self.num_experts // self.num_devices

        # Setup parallelism
        self._setup_parallelism()

        # Initialize all-to-all configuration
        self.all_to_all_config = AllToAllConfig.from_config(self.config, self.ep_axis)

        # Initialize components
        self._init_router()
        self._init_experts()
        self._init_expert_mapping_tensors()

        # Load weights if specified (unless mock weights requested)
        if self.config.get("weight_path") and not self.config.get("use_mock_weights", False):
            self._load_weights_from_path()

    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate JSON configuration."""
        with open(config_path, "r") as f:
            full_config = json.load(f)

        config = full_config["moe_block"]

        # Validate required fields
        required_fields = ["model_params", "router", "experts"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Validate model params
        required_params = ["num_experts", "num_experts_per_tok", "hidden_size", "intermediate_size"]
        for param in required_params:
            if param not in config["model_params"]:
                raise ValueError(f"Missing required model parameter: {param}")

        return config

    def _setup_parallelism(self):
        """Setup tensor and expert parallelism configurations."""
        # Tensor parallel setup
        tp_config = self.config.get("tensor_parallel", {"enabled": False})
        self.tp_enabled = tp_config.get("enabled", False)
        self.tp_axis = tp_config.get("axis", 1) if self.tp_enabled else None
        self.tp_size = self.mesh_device.shape[self.tp_axis] if self.tp_enabled else 1

        # Expert parallel setup
        ep_config = self.config.get("expert_parallel", {"enabled": True})
        self.ep_enabled = ep_config.get("enabled", True)
        self.ep_axis = ep_config.get("axis", 0) if self.ep_enabled else None

    def _init_router(self):
        """Initialize router based on simplified configuration."""
        router_config = self.config["router"]
        router_type = router_config["type"]

        # Build router configuration with derived parameters
        router_params = {
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "hidden_size": self.hidden_size,
            "n_routed_experts": self.num_experts,  # Always same as num_experts
        }

        if router_type == "moe_gate":
            # DeepSeek-style router
            router_params.update(
                {
                    "score_correction_bias": router_config.get("score_correction_bias", True),
                    "n_group": 8,  # Default for DeepSeek
                    "topk_group": 3,  # Default for DeepSeek
                    "routed_scaling_factor": 1.0,
                    "memory_config": "L1_MEMORY_CONFIG",
                    "compute_kernel_config": "HIFI2",
                }
            )
            self.router = MoEGateRouter(router_params, self.mesh_device)

        elif router_type == "topk":
            # GPT-OSS style router
            self.router = TopKRouter(self.mesh_device, router_params)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    def _init_experts(self):
        """Initialize expert configurations based on simplified JSON."""
        experts_config = self.config["experts"]

        # All experts now use distributed implementation
        # GPT-OSS uses DistributedExpert with clamped SwiGLU activation
        expert_type = "distributed"

        self.expert_type = expert_type
        logger.info(f"Using expert type: {expert_type}")

        # Setup experts based on type
        self.distributed_expert_enabled = experts_config.get("distributed", True)
        if self.distributed_expert_enabled:
            base_config = {
                "n_routed_experts": self.num_experts,
                "num_experts": self.num_experts,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_experts_per_device": self.num_experts_per_device,
                "dispatch_cluster_axis": self.ep_axis,
                "activation": experts_config.get("activation", "swiglu"),
                "use_quantized_weights": experts_config.get("quantized", False),
                "memory_config": "L1_MEMORY_CONFIG",  # Default for decode
                "output_memory_config": "L1_MEMORY_CONFIG",
            }

            # Add activation parameters if present (for GPT-OSS clamped SwiGLU)
            if "swiglu_alpha" in experts_config:
                base_config["swiglu_alpha"] = experts_config["swiglu_alpha"]
            if "swiglu_limit" in experts_config:
                base_config["swiglu_limit"] = experts_config["swiglu_limit"]

            # Add quantization parameters if needed
            if experts_config.get("quantized", False):
                base_config["weight_block_size"] = experts_config.get("weight_block_size", [128, 128])

            self.distributed_expert_config = base_config

        # Setup shared experts (DeepSeek only)
        self.shared_expert_enabled = experts_config.get("shared", False)
        if self.shared_expert_enabled:
            self.shared_expert_config = {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "use_quantized_weights": experts_config.get("quantized", False),
                "memory_config": "L1_MEMORY_CONFIG",
            }

            if experts_config.get("quantized", False):
                self.shared_expert_config["weight_block_size"] = experts_config.get("weight_block_size", [128, 128])

            self.shared_parallel = experts_config.get("shared_parallel", True)

    def _init_expert_mapping_tensors(self):
        """Initialize expert mapping tensors for all-to-all operations."""
        if not self.distributed_expert_enabled:
            return

        # Log configuration for debugging
        logger.info(
            f"Creating expert mapping for {self.num_experts} experts with {self.num_experts_per_device} per device"
        )

        # Create expert mapping tensor
        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(self.num_devices, dtype=torch.int32)
            .repeat_interleave(self.num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Verify expert mapping tensor shape
        expected_shape = (1, 1, self.num_experts, self.num_devices)
        logger.info(f"Expert mapping tensor shape: {self.expert_mapping_tensors.shape}, expected: {expected_shape}")

        # Create remap topk mask
        num_dispatch_device_rows = self.mesh_device.shape[0] if hasattr(self.mesh_device, "shape") else 1
        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, num_dispatch_device_rows, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _load_weights_from_path(self):
        """Load weights from model path specified in configuration."""
        weight_path = Path(self.config["weight_path"])
        module_prefix = self.config.get("module_prefix", "")

        logger.info(f"Loading weights from {weight_path}")

        with LazyStateDict(weight_path) as lazy_dict:
            if module_prefix:
                state_dict = lazy_dict.view_with_prefix(module_prefix + ".")
            else:
                state_dict = lazy_dict

            self.load_weights(state_dict)

    def load_weights(self, state_dict: dict):
        """
        Load weights for all components with simplified logic.

        Args:
            state_dict: Dictionary containing weights for router and experts
        """
        # Load router weights
        router_state = self._extract_router_weights(state_dict)
        if router_state:
            self.router.load_weights(router_state)

        # Load distributed expert weights
        if self.distributed_expert_enabled:
            expert_state = self._extract_expert_weights(state_dict)
            if expert_state:
                self._load_distributed_expert_weights(expert_state)

        # Load shared expert weights (DeepSeek only)
        if self.shared_expert_enabled:
            shared_state = self._extract_shared_expert_weights(state_dict)
            if shared_state:
                self._load_shared_expert_weights(shared_state)

    def _extract_router_weights(self, state_dict: dict) -> dict:
        """Extract router weights from state dict."""
        router_state = {}

        if isinstance(self.router, MoEGateRouter):
            # DeepSeek style
            for key in ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]:
                if key in state_dict:
                    simple_key = key.replace("mlp.gate.", "")
                    router_state[simple_key] = state_dict[key]
        else:
            # GPT-OSS style (TopKRouter)
            # Try multiple possible key formats
            for key in state_dict.keys():
                if "router.weight" in key or key == "mlp.router.weight":
                    router_state["weight"] = state_dict[key]
                elif "router.bias" in key or key == "mlp.router.bias":
                    router_state["bias"] = state_dict[key]
                elif key == "mlp.gate.weight" and "weight" not in router_state:
                    # Fallback for gate-style naming
                    router_state["weight"] = state_dict[key]
                elif key == "mlp.gate.bias" and "bias" not in router_state:
                    router_state["bias"] = state_dict[key]

        return router_state

    def _extract_expert_weights(self, state_dict: dict) -> dict:
        """Extract distributed expert weights from state dict."""
        # All models now use DistributedExperts, weights are under mlp.experts.
        return {k.replace("mlp.", ""): v for k, v in state_dict.items() if k.startswith("mlp.experts.")}

    def _extract_shared_expert_weights(self, state_dict: dict) -> dict:
        """Extract shared expert weights from state dict."""
        shared_state = {}
        prefix = "mlp.shared_experts."

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            weight_key = f"{prefix}{proj}.weight"
            scale_key = f"{prefix}{proj}.weight_scale_inv"

            if weight_key in state_dict:
                shared_state[f"{proj}.weight"] = state_dict[weight_key]
            if scale_key in state_dict:
                shared_state[f"{proj}.weight_scale_inv"] = state_dict[scale_key]

        return shared_state

    def _load_distributed_expert_weights(self, expert_state: dict):
        """Load distributed expert weights with simplified conversion."""

        # Create minimal config for weight conversion
        class ExpertConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)

                # Add quantization config if needed
                if hasattr(self, "use_quantized_weights") and self.use_quantized_weights:
                    self.quantization_config = {"weight_block_size": getattr(self, "weight_block_size", [128, 128])}

        config = ExpertConfig(self.distributed_expert_config)

        # Convert weights using DistributedExpert
        weight_configs = DistributedExpert.convert_weights(
            config, (expert_state,), Path("/tmp/moe_expert_weights"), self.mesh_device
        )

        # Create decode config
        self.distributed_expert_decode_config = DistributedExpert.decode_model_config(config, self.mesh_device)

        # Add all-to-all and other required configurations
        self.distributed_expert_decode_config["cluster_axis"] = self.all_to_all_config.cluster_axis
        self.distributed_expert_decode_config["dispatch_topology"] = self.all_to_all_config.dispatch_topology
        self.distributed_expert_decode_config["combine_topology"] = self.all_to_all_config.combine_topology
        self.distributed_expert_decode_config["num_experts"] = self.num_experts
        self.distributed_expert_decode_config["num_experts_per_tok"] = self.num_experts_per_tok
        self.distributed_expert_decode_config["hidden_size"] = self.hidden_size

        # DistributedExpert returns a dict, merge the weights
        for key, value in weight_configs.items():
            if key in self.distributed_expert_decode_config:
                self.distributed_expert_decode_config[key].update(value)
            else:
                self.distributed_expert_decode_config[key] = value

    def _load_shared_expert_weights(self, shared_state: dict):
        """Load shared expert weights with simplified conversion."""

        # Create minimal config for weight conversion
        class SharedConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)

                # Check if we have quantized weights
                if "gate_proj.weight_scale_inv" in shared_state:
                    self.quantization_config = {"weight_block_size": getattr(self, "weight_block_size", [128, 128])}

        config = SharedConfig(self.shared_expert_config)

        # Convert weights
        weight_configs = SharedExpert.convert_weights(
            config, (shared_state,), Path("/tmp/moe_shared_weights"), self.mesh_device
        )

        # Create decode config
        self.shared_expert_decode_config = SharedExpert.decode_model_config(config, self.mesh_device)

        # Merge weight configs
        for key in ["w1", "w2", "w3"]:
            if key in weight_configs:
                self.shared_expert_decode_config[key].update(weight_configs[key])

    def forward(self, x: ttnn.Tensor, mode: str = "decode") -> ttnn.Tensor:
        """
        Simplified forward pass through MoE block.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or TP-sharded
            mode: "decode" or "prefill" mode

        Returns:
            MoE output tensor with same shape as input
        """
        # Handle chunking for large prefill sequences
        if mode == "prefill" and self._should_chunk_prefill(x):
            return self._forward_chunked_prefill(x)

        # All-gather if tensor parallel
        x_was_gathered = False
        if self.tp_enabled and self._is_tp_sharded(x):
            x = self._all_gather(x, mode)
            x_was_gathered = True

        # Router forward
        weights, indices = self.router.forward(x, mode)

        # Prepare weights for experts
        weights_prepared = self._prepare_expert_weights(weights)

        # Run experts
        outputs = []

        # Distributed experts
        if self.distributed_expert_enabled:
            moe_output = self._forward_moe(x, indices, weights_prepared, mode)
            outputs.append(moe_output)

        # Shared expert (if enabled and parallel)
        if self.shared_expert_enabled and self.shared_parallel:
            shared_output = SharedExpert.forward_decode(x, self.shared_expert_decode_config)
            outputs.append(shared_output)

        # Combine outputs
        if len(outputs) > 1:
            output = ttnn.add(outputs[0], outputs[1])
            for out in outputs[:-1]:
                ttnn.deallocate(out)
        else:
            output = outputs[0]

        # Reduce-scatter if we gathered
        if x_was_gathered:
            output = self._reduce_scatter(output, mode)
            ttnn.deallocate(x)

        # Cleanup
        ttnn.deallocate(weights_prepared)

        return output

    def _prepare_expert_weights(self, weights: ttnn.Tensor) -> ttnn.Tensor:
        """Simplified weight preparation."""
        # Convert to ROW_MAJOR for operations
        weights_rm = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT)

        # Repeat weights to match hidden dimension
        repeat_dims = (self.hidden_size, 1, 1, 1)
        weights_rm = ttnn.repeat(weights_rm, ttnn.Shape(repeat_dims), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Permute dimensions (3, 1, 2, 0) - standard pattern
        weights_rm = ttnn.permute(weights_rm, (3, 1, 2, 0), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Convert back to TILE_LAYOUT
        weights = ttnn.to_layout(weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(weights_rm)

        return weights

    def _forward_moe(
        self, x: ttnn.Tensor, indices: ttnn.Tensor, weights: ttnn.Tensor, mode: str = "decode"
    ) -> ttnn.Tensor:
        """Simplified MoE forward pass."""
        # Get dimensions
        batch_size_per_device = x.shape[-2]
        seq_len = 1  # For compatibility
        hidden_size = x.shape[-1]

        # Check batch mode
        use_replicated_batch = self.config.get("replicated_batch_mode", False)
        num_dispatch_devices = self.mesh_device.shape[0] if hasattr(self.mesh_device, "shape") else 1

        if use_replicated_batch:
            batch_size = batch_size_per_device
        else:
            batch_size = batch_size_per_device * num_dispatch_devices

        # Reshape inputs
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, hidden_size))

        indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        indices_rm = ttnn.reshape(indices_rm, shape=(batch_size_per_device, 1, seq_len, self.num_experts_per_tok))

        # Get chunk size
        chunk_config = self.config.get("chunking", {})
        chunk_size = chunk_config.get("moe_chunk_size", batch_size_per_device)
        chunk_size = min(batch_size_per_device, max(1, chunk_size))

        output_chunks = []

        for batch_start in range(0, batch_size_per_device, chunk_size):
            batch_end = min(batch_start + chunk_size, batch_size_per_device)

            # Slice inputs
            x_chunk = ttnn.slice(x_rm, [batch_start, 0, 0, 0], [batch_end, 1, seq_len, hidden_size])
            indices_chunk = ttnn.slice(
                indices_rm, [batch_start, 0, 0, 0], [batch_end, 1, seq_len, self.num_experts_per_tok]
            )

            # Prepare weights for experts
            batch_chunk = batch_end - batch_start
            token_start = batch_start * seq_len
            token_end = batch_end * seq_len
            weights_chunk = ttnn.slice(
                weights, [0, 0, token_start, 0], [self.num_experts_per_tok, 1, token_end, hidden_size]
            )

            # Convert x_chunk to TILE layout before passing to experts
            x_chunk = ttnn.to_layout(x_chunk, ttnn.TILE_LAYOUT)

            # Run experts based on expert type
            # DistributedExpert with integrated all-to-all (now used for all models)
            experts_output = DistributedExpert.forward_decode(
                x_chunk,  # Original input, not dispatch_output
                indices_chunk,
                weights_chunk,
                self.distributed_expert_decode_config,
                self.expert_mapping_tensors,
                self.mesh_device,
            )

            # Cleanup
            ttnn.deallocate(x_chunk)
            ttnn.deallocate(indices_chunk)
            ttnn.deallocate(weights_chunk)

            # Experts now return the final weighted and summed output
            # Shape: [1, 1, batch_chunk * seq_len, hidden_size]
            output_chunks.append(experts_output)

        # Combine chunks
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

    def _should_chunk_prefill(self, x: ttnn.Tensor) -> bool:
        """Check if prefill should be chunked."""
        if "chunking" not in self.config:
            return False

        prefill_chunk_size = self.config["chunking"].get("prefill_chunk_size", 16384)
        num_dispatch_devices = self.mesh_device.shape[0] if hasattr(self.mesh_device, "shape") else 1
        global_tokens = x.shape[2] * num_dispatch_devices

        return global_tokens > prefill_chunk_size

    def _forward_chunked_prefill(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with chunking for large prefill sequences."""
        chunk_size = self.config["chunking"].get("prefill_chunk_size", 16384)
        num_dispatch_devices = self.mesh_device.shape[0] if hasattr(self.mesh_device, "shape") else 1
        chunk_size = max(1, chunk_size // max(1, num_dispatch_devices))

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
        """Check if tensor is TP-sharded."""
        if not self.tp_enabled:
            return False

        actual_dim = x.shape[-1]
        expected_sharded_dim = self.hidden_size // self.tp_size
        return actual_dim == expected_sharded_dim

    def _get_memory_config(self, mode: str):
        """Simplified memory configuration."""
        # Simple default: L1 for decode, DRAM for prefill
        if mode == "decode":
            return ttnn.L1_MEMORY_CONFIG
        else:
            return ttnn.DRAM_MEMORY_CONFIG

    def _all_gather(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """Perform all-gather operation."""
        if self.ccl is None:
            raise ValueError("CCL instance is required for tensor parallel operations")

        all_gather_config = {
            "cluster_axis": self.tp_axis,
            "dim": -1,
            "memory_config": self._get_memory_config(mode),
            "topology": ttnn.Topology.Linear,
        }

        all_gather_args = self.ccl.populate_all_gather_runtime_args(all_gather_config)
        return ttnn.experimental.all_gather_async(x, **all_gather_args)

    def _reduce_scatter(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """Perform reduce-scatter operation."""
        if self.ccl is None:
            raise ValueError("CCL instance is required for tensor parallel operations")

        reduce_scatter_config = {
            "cluster_axis": self.tp_axis,
            "dim": 3,
            "memory_config": self._get_memory_config(mode),
            "topology": ttnn.Topology.Linear,
        }

        reduce_scatter_args = self.ccl.populate_reduce_scatter_runtime_args(reduce_scatter_config)
        return ttnn.experimental.reduce_scatter_minimal_async(x, **reduce_scatter_args)

    @classmethod
    def from_state_dict(cls, state_dict: dict, mesh_device: ttnn.MeshDevice, ccl=None, config_path: str = None):
        """
        Create MoEBlock from state dict with auto-detected configuration.

        Args:
            state_dict: State dict with MoE weights
            mesh_device: TTNN mesh device
            ccl: CCL instance
            config_path: Path to config file

        Returns:
            Initialized MoEBlock ready for inference
        """
        if config_path is None:
            # Auto-detect based on state dict structure
            if "mlp.shared_experts.gate_proj.weight" in state_dict:
                config_path = str(Path(__file__).parent / "configs" / "deepseek_v3_simplified.json")
            else:
                config_path = str(Path(__file__).parent / "configs" / "gpt_oss_simplified.json")

        moe_block = cls(config_path, mesh_device, ccl)
        moe_block.load_weights(state_dict)
        return moe_block
