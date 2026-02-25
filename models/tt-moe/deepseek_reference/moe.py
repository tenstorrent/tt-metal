# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    AllToAllCombineConfig,
    AllToAllDispatchConfig,
    MeshDeviceStub,
    MulConfig,
    ReduceScatterAsyncMinimalConfig,
    RepeatConfig,
)
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.demos.deepseek_v3.utils.shared_state_addon import SharedStateAddOn

from .ccl import CCL
from .experts import Experts as MoEExperts
from .moe_gate import MoEGate


class MoE(SharedStateAddOn, AbstractModule):
    """MoE module from DeepSeek-R1.
    See the `AbstractModule` docstring for usage info.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"MoE expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts
        assert state_dict is not None

        return {
            "moe_gate": MoEGate.convert_weights(
                hf_config, (state_dict,), output_path / "moe_gate", mesh_device, "gate."
            ),
            "moe_experts": MoEExperts.convert_weights(
                hf_config, (state_dict,), output_path / "moe_experts", mesh_device
            ),
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelState:
        """Create shared model state containing tensors that are constant across all instances.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelState containing shared tensors
        """
        num_devices = mesh_device.get_num_devices()
        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
        num_dispatch_device_rows = mesh_device.shape[0]

        expert_mapping_tensors = ttnn.from_torch(
            torch.eye(num_devices, dtype=torch.int32)
            .repeat_interleave(num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        remap_topk_mask = ttnn.from_torch(
            torch.ones((1, num_dispatch_device_rows, 1, hf_config.n_routed_experts), dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return {
            "expert_mapping_tensors": expert_mapping_tensors,
            "remap_topk_mask": remap_topk_mask,
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        ccl: CCL,
    ) -> ModelState:
        """Create model state containing CCL-related communication configurations.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            ccl: CCL instance for communication configuration
        Returns:
            ModelState containing CCL configurations
        """
        # Store CCL object for runtime semaphore initialization
        return {
            # CCL-specific parameters (semaphores and num_links)
            "all_to_all_dispatch": {
                "num_links": 4,
            },
            "all_to_all_combine": {
                "num_links": 4,
            },
            "ccl": ccl,
        }

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        mode: str,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG

            USERS_PER_ROW = 32
            HIDDEN_SIZE = hf_config.hidden_size
            TP_SIZE = mesh_device.shape[1]

            input_output_memory_config = ttnn.create_sharded_memory_config(
                shape=(USERS_PER_ROW, HIDDEN_SIZE // TP_SIZE),
                core_grid=ttnn.CoreGrid(y=7, x=4),
                strategy=ttnn.ShardStrategy.WIDTH,
            )

            # Construct the config
            return {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
                "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode, topk_fallback=topk_fallback),
                "all_to_all_dispatch_output_memory_config": memory_config,
                "all_to_all_dispatch_metadata_memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "activations_repeat": RepeatConfig(repeat_dims=ttnn.Shape((1, num_experts_per_device, 1, 1))),
                "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
                "all_to_all_combine_output_memory_config": memory_config,
                "topk_weights_repeat": RepeatConfig(repeat_dims=ttnn.Shape((hf_config.hidden_size, 1, 1, 1))),
                "mul_experts_output_with_weights": MulConfig(memory_config=memory_config),
                "input_memory_config": input_output_memory_config,
                "output_memory_config": input_output_memory_config,
                "all_to_all_dispatch": AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
                "all_to_all_combine": AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
                "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                    cluster_axis=1,
                    dim=3,
                    memory_config=input_output_memory_config,
                ),
                "revert_tp": AllGatherAsyncConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dim=-1,  # Last dimension
                    # memory_config=ttnn.create_sharded_memory_config(  # Bad PCC
                    #     shape=(USERS_PER_ROW, HIDDEN_SIZE),
                    #     core_grid=ttnn.CoreGrid(y=7, x=8),
                    #     strategy=ttnn.ShardStrategy.WIDTH,
                    # ),
                    memory_config=memory_config,
                    cluster_axis=1,
                ),
            }
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            # Construct the config
            return {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
                "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode, topk_fallback=topk_fallback),
                "all_to_all_dispatch_output_memory_config": memory_config,
                "all_to_all_dispatch_metadata_memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "activations_repeat": RepeatConfig(repeat_dims=ttnn.Shape((1, num_experts_per_device, 1, 1))),
                "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
                "all_to_all_combine_output_memory_config": memory_config,
                "topk_weights_repeat": RepeatConfig(repeat_dims=ttnn.Shape((hf_config.hidden_size, 1, 1, 1))),
                "mul_experts_output_with_weights": MulConfig(memory_config=memory_config),
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
                "all_to_all_dispatch": AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
                "all_to_all_combine": AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
                "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                    cluster_axis=1,
                    dim=3,
                    memory_config=memory_config,
                ),
                "revert_tp": AllGatherAsyncConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dim=-1,  # Last dimension
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cluster_axis=1,
                ),
            }

    @classmethod
    def decode_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, topk_fallback: bool = False
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, "decode", topk_fallback=topk_fallback)

    @classmethod
    def prefill_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, topk_fallback: bool = False
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, "prefill", topk_fallback=topk_fallback)

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # Chunk the full MoE prefill path at 16K tokens to avoid OOM.
        # Use global token count (local seq_len * num_dispatch_devices) to decide.
        chunk_tokens = int(cfg.get("prefill_chunk_size", 16384))
        num_dispatch_devices = int(cfg.get("num_dispatch_devices", 1))
        global_tokens = x.shape[2] * num_dispatch_devices
        if global_tokens > chunk_tokens:
            chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
            return cls._forward_chunked_prefill(x, cfg, chunk_size)
        return cls._forward_impl(x, cfg)

    @classmethod
    def _forward_chunked_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig, chunk_size: int) -> ttnn.Tensor:
        chunk_size = max(1, chunk_size)
        _, _, seq_len, _ = x.shape
        output_chunks: list[ttnn.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x_chunk = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, x.shape[3]])
            output_chunks.append(cls._forward_impl(x_chunk, cfg))
            ttnn.deallocate(x_chunk)

        if len(output_chunks) == 1:
            return output_chunks[0]
        output = ttnn.concat(output_chunks, dim=2)
        for chunk in output_chunks:
            ttnn.deallocate(chunk)
        return output

    @classmethod
    def _forward_impl(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # breakpoint()
        ccl = cfg["ccl"]  # CCL runtime initialization in execution order
        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

        # All Gather (only if input is TP-sharded)
        hidden_size = cfg["hidden_size"]
        tp_size = cfg["mesh_device"].shape[1]
        x_dim = x.shape[-1]

        if x_dim == hidden_size:
            # Already full hidden size; skip all_gather
            pass
        elif x_dim == hidden_size // tp_size:
            x = cls._fwd_all_gather(x, cfg)
        else:
            logger.warning(
                f"MoE forward: unexpected input hidden dim {x_dim} (hidden_size={hidden_size}, tp_size={tp_size}); "
                "running all_gather as fallback."
            )
            x = cls._fwd_all_gather(x, cfg)

        # MoE Gate

        # TRACE: Router input
        if os.environ.get("TRACE_FLOW", "0") == "1":
            import sys

            sys.path.append("/home/ntarafdar/tt-moe/tt-metal")
            from trace_helpers import trace_point

            trace_point(
                stage="1_router_input",
                impl="reference",
                tensors={"x": x},
                configs={"router_config": cfg.get("moe_gate", {})},
                metadata={"seq_len": seq_len, "batch_size": batch_size},
                mesh_device=cfg["mesh_device"],
            )

        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)

        # TRACE: Router output
        if os.environ.get("TRACE_FLOW", "0") == "1":
            trace_point(
                stage="2_router_output",
                impl="reference",
                tensors={"weights": topk_experts_weights, "indices": topk_experts_indices},
                mesh_device=cfg["mesh_device"],
            )

        # Save router outputs for comparison
        if os.environ.get("SAVE_ROUTER_OUTPUTS", "0") == "1":
            from pathlib import Path

            import torch

            save_dir = Path("/tmp/moe_activations/reference")
            save_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Convert to torch for saving
                mesh_device = cfg["mesh_device"].get() if hasattr(cfg["mesh_device"], "get") else cfg["mesh_device"]
                weights_torch = ttnn.to_torch(
                    topk_experts_weights,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )
                indices_torch = ttnn.to_torch(
                    topk_experts_indices,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "weights": weights_torch,
                        "indices": indices_torch,
                        "weights_shape": weights_torch.shape,
                        "indices_shape": indices_torch.shape,
                        "weights_dtype": weights_torch.dtype,
                        "indices_dtype": indices_torch.dtype,
                        "implementation": "MoE_Reference",
                    },
                    save_dir / "router_outputs.pt",
                )

                print(f"[Reference] Saved router outputs: weights {weights_torch.shape}, indices {indices_torch.shape}")
            except Exception as e:
                print(f"[Reference] Failed to save router outputs: {e}")

        # Repeat + Permute Expert weights

        topk_experts_weights = cls._fwd_repeat_permute_expert_weights(topk_experts_weights, cfg)

        # MOE

        post_combine_output_tensor = cls._fwd_moe(
            x,
            topk_experts_indices,
            topk_experts_weights,
            cfg,
            batch_size_per_device,
            batch_size,
            seq_len,
        )

        # Reduce Scatter

        post_combine_output_tensor = cls._fwd_reduce_scatter(post_combine_output_tensor, cfg, ccl)

        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gate(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return MoEGate.forward(x, cfg["moe_gate"])

    @classmethod
    def _fwd_repeat_permute_expert_weights(
        cls, topk_experts_weights: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig
    ) -> ttnn.Tensor:
        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            import sys

            sys.path.append("/home/ntarafdar/tt-moe/tt-metal")
            from tensor_flow_debug import save_tensor_checkpoint

            # Before any transformations
            save_tensor_checkpoint(
                "weights_before_repeat_permute",
                topk_experts_weights,
                "0_weights_processing",
                "reference",
                cfg.get("mesh_device"),
                {"original_shape": list(topk_experts_weights.shape)},
            )

        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            save_tensor_checkpoint(
                "weights_after_row_major",
                topk_experts_weights_rm,
                "0_weights_processing",
                "reference",
                cfg.get("mesh_device"),
            )

        topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, **cfg["topk_weights_repeat"])

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            save_tensor_checkpoint(
                "weights_after_repeat",
                topk_experts_weights_rm,
                "0_weights_processing",
                "reference",
                cfg.get("mesh_device"),
                {"repeat_dims": str(cfg["topk_weights_repeat"])},
            )

        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            save_tensor_checkpoint(
                "weights_after_permute",
                topk_experts_weights_rm,
                "0_weights_processing",
                "reference",
                cfg.get("mesh_device"),
                {"permute_dims": "(3, 1, 2, 0)"},
            )

        topk_experts_weights = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            save_tensor_checkpoint(
                "weights_after_tile_layout",
                topk_experts_weights,
                "0_weights_processing",
                "reference",
                cfg.get("mesh_device"),
            )

        return topk_experts_weights

    @classmethod
    def _fwd_moe(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig | RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        tokens = batch_size * seq_len

        # Save tensor flow for debugging
        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            import sys

            sys.path.append("/home/ntarafdar/tt-moe/tt-metal")
            from tensor_flow_debug import save_tensor_checkpoint

            # Stage 1: Initial inputs to _fwd_moe
            save_tensor_checkpoint(
                "hidden_states",
                x,
                "1_fwd_moe_input",
                "reference",
                cfg["mesh_device"],
                {"batch_size": batch_size, "seq_len": seq_len, "tokens": tokens},
            )
            save_tensor_checkpoint("weights", topk_experts_weights, "1_fwd_moe_input", "reference", cfg["mesh_device"])
            save_tensor_checkpoint("indices", topk_experts_indices, "1_fwd_moe_input", "reference", cfg["mesh_device"])

        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            # Stage 2: After ROW_MAJOR conversion
            save_tensor_checkpoint("hidden_states_rm", x_rm, "2_after_row_major", "reference", cfg["mesh_device"])

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            # CRITICAL: Save before reshape to understand the transformation
            from pathlib import Path

            import numpy as np

            save_tensor_checkpoint(
                "hidden_states_before_reshape",
                x_rm,
                "2b_before_reshape",
                "reference",
                cfg["mesh_device"],
                {
                    "current_shape": list(x_rm.shape),
                    "target_shape": (batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
                    "batch_size_per_device": batch_size_per_device,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                },
            )

            # Save binary for exact comparison
            x_rm_torch = ttnn.to_torch(
                x_rm,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    cfg["mesh_device"], dims=(-2, -1), mesh_shape=cfg["mesh_device"].shape
                ),
            )
            binary_dir = Path("/tmp/tensor_flow_binary/reference/2b_before_reshape")
            binary_dir.mkdir(parents=True, exist_ok=True)
            np.save(binary_dir / "hidden_states.npy", x_rm_torch.cpu().float().numpy())

        x_rm = ttnn.reshape(
            x_rm,
            shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
        )

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            # Stage 3: After reshape
            save_tensor_checkpoint(
                "hidden_states_reshaped",
                x_rm,
                "3_after_reshape",
                "reference",
                cfg["mesh_device"],
                {"shape": (batch_size_per_device, 1, seq_len, cfg["hidden_size"])},
            )

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            save_tensor_checkpoint(
                "indices_rm", topk_experts_indices_rm, "2_after_row_major", "reference", cfg["mesh_device"]
            )

        # TRACE: After ROW_MAJOR conversion
        if os.environ.get("TRACE_FLOW", "0") == "1":
            import sys

            sys.path.append("/home/ntarafdar/tt-moe/tt-metal")
            from trace_helpers import trace_point

            trace_point(
                stage="3_after_row_major",
                impl="reference",
                tensors={"x_rm": x_rm, "indices_rm": topk_experts_indices_rm},
                metadata={"batch_size_per_device": batch_size_per_device, "seq_len": seq_len},
                mesh_device=cfg["mesh_device"],
            )

        # DEBUG: Save tensor before reshape
        if os.environ.get("DEBUG_SAVE_CHUNKS", "0") == "1":
            from pathlib import Path

            import torch

            debug_dir = Path("/tmp/moe_debug/ref_impl")
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Save indices shape info before reshape
            mesh_device = cfg["mesh_device"]
            indices_torch = ttnn.to_torch(
                topk_experts_indices_rm,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            )

            torch.save(
                {
                    "data": indices_torch.cpu() if hasattr(indices_torch, "cpu") else indices_torch,
                    "shape": tuple(topk_experts_indices_rm.shape)
                    if hasattr(topk_experts_indices_rm, "shape")
                    else indices_torch.shape,
                    "target_shape": (batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"]),
                    "batch_size_per_device": batch_size_per_device,
                    "seq_len": seq_len,
                    "num_experts_per_tok": cfg["num_experts_per_tok"],
                },
                debug_dir / "indices_before_reshape.pt",
            )

            print(f"[DEBUG] Ref impl - indices before reshape: shape={topk_experts_indices_rm.shape}")
            print(
                f"[DEBUG] Ref impl - target shape: ({batch_size_per_device}, 1, {seq_len}, {cfg['num_experts_per_tok']})"
            )

        if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
            # Critical: Log batch_size_per_device to understand reshape
            save_tensor_checkpoint(
                "config_info",
                x_rm,
                "2a_batch_size_calculation",
                "reference",
                cfg["mesh_device"],
                {
                    "batch_size_per_device": batch_size_per_device,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "x_rm_shape": list(x_rm.shape),
                    "target_reshape": (batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"]),
                },
            )

        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"])
        )

        # DEBUG: Save weights after preparation, before chunking
        if os.environ.get("DEBUG_SAVE_CHUNKS", "0") == "1":
            # Save weights shape info
            weights_torch = ttnn.to_torch(
                topk_experts_weights,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            )

            torch.save(
                {
                    "data": weights_torch.cpu() if hasattr(weights_torch, "cpu") else weights_torch,
                    "shape": tuple(topk_experts_weights.shape)
                    if hasattr(topk_experts_weights, "shape")
                    else weights_torch.shape,
                    "note": "weights after _fwd_repeat_permute_expert_weights, before chunking",
                },
                debug_dir / "weights_before_chunking.pt",
            )

            print(f"[DEBUG] Ref impl - weights before chunking: shape={topk_experts_weights.shape}")

        # Chunk along local batch dimension to keep all_to_all_dispatch output small in prefill.
        chunk_size = min(batch_size_per_device, max(1, cfg.get("moe_chunk_size", batch_size_per_device)))
        output_chunks: list[ttnn.Tensor] = []

        def _slice_topk_weights(batch_start: int, batch_end: int) -> ttnn.Tensor:
            token_start = batch_start * seq_len
            token_end = batch_end * seq_len

            if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
                save_tensor_checkpoint(
                    f"weights_slice_input_{batch_start}",
                    topk_experts_weights,
                    f"5_weight_slice_{batch_start}_{batch_end}",
                    "reference",
                    cfg["mesh_device"],
                    {
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "token_start": token_start,
                        "token_end": token_end,
                        "input_shape": list(topk_experts_weights.shape),
                    },
                )

            sliced = ttnn.slice(
                topk_experts_weights,
                [0, 0, token_start, 0],
                [cfg["num_experts_per_tok"], 1, token_end, cfg["hidden_size"]],
            )

            if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
                save_tensor_checkpoint(
                    f"weights_sliced_{batch_start}",
                    sliced,
                    f"5_weight_slice_{batch_start}_{batch_end}",
                    "reference",
                    cfg["mesh_device"],
                    {"output_shape": list(sliced.shape)},
                )

            return sliced

        for batch_start in range(0, batch_size_per_device, chunk_size):
            batch_end = min(batch_start + chunk_size, batch_size_per_device)
            batch_chunk = batch_end - batch_start
            batch_size_chunk = batch_chunk * cfg["num_dispatch_devices"]

            x_chunk = ttnn.slice(
                x_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["hidden_size"]],
            )
            topk_indices_chunk = ttnn.slice(
                topk_experts_indices_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["num_experts_per_tok"]],
            )

            if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
                # Stage 4: After slicing for chunk
                chunk_info = {
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "chunk_size": chunk_size,
                    "batch_chunk": batch_chunk,
                }
                save_tensor_checkpoint(
                    f"x_chunk_{batch_start}",
                    x_chunk,
                    f"4_after_slice_chunk_{batch_start}_{batch_end}",
                    "reference",
                    cfg["mesh_device"],
                    chunk_info,
                )
                save_tensor_checkpoint(
                    f"indices_chunk_{batch_start}",
                    topk_indices_chunk,
                    f"4_after_slice_chunk_{batch_start}_{batch_end}",
                    "reference",
                    cfg["mesh_device"],
                    chunk_info,
                )

            # TRACE: After slicing
            if os.environ.get("TRACE_FLOW", "0") == "1":
                import sys

                sys.path.append("/home/ntarafdar/tt-moe/tt-metal")
                from trace_helpers import trace_point

                trace_point(
                    stage=f"4_after_slice_chunk_{batch_start}_{batch_end}",
                    impl="reference",
                    tensors={"x_chunk": x_chunk, "indices_chunk": topk_indices_chunk},
                    metadata={"batch_start": batch_start, "batch_end": batch_end, "chunk_size": chunk_size},
                    mesh_device=cfg["mesh_device"],
                )

            # Save chunks for comparison - only activations, not weights
            if os.environ.get("DEBUG_SAVE_CHUNKS", "0") == "1":
                from pathlib import Path

                import torch

                chunk_dir = Path(f"/tmp/moe_chunks/ref_impl/chunk_{batch_start}_{batch_end}")
                chunk_dir.mkdir(parents=True, exist_ok=True)

                # Save x_chunk - handle distributed tensors
                # x_chunk is sliced from x_rm which is distributed on the mesh
                mesh_device = cfg["mesh_device"]
                x_chunk_torch = ttnn.to_torch(
                    x_chunk,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
                )
                torch.save(
                    {
                        "data": x_chunk_torch,
                        "shape": x_chunk_torch.shape,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "chunk_size": chunk_size,
                    },
                    chunk_dir / "x_chunk.pt",
                )

                # Save indices_chunk - also distributed
                indices_chunk_torch = ttnn.to_torch(
                    topk_indices_chunk,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
                )
                torch.save(
                    {
                        "data": indices_chunk_torch,
                        "shape": indices_chunk_torch.shape,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                    },
                    chunk_dir / "indices_chunk.pt",
                )

                print(f"[DEBUG] Saved ref x_chunk and indices_chunk for batch {batch_start}-{batch_end} to {chunk_dir}")

            # Save exact binary inputs to all_to_all_dispatch for comparison
            if os.environ.get("SAVE_DISPATCH_INPUTS", "0") == "1":
                import hashlib
                from pathlib import Path

                import numpy as np
                import torch

                dispatch_input_dir = Path("/tmp/dispatch_inputs/reference")
                dispatch_input_dir.mkdir(parents=True, exist_ok=True)

                # Save hidden states (x_chunk)
                x_chunk_torch = ttnn.to_torch(
                    x_chunk,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        cfg["mesh_device"], dims=(-2, -1), mesh_shape=cfg["mesh_device"].shape
                    ),
                )
                torch.save(
                    {
                        "tensor": x_chunk_torch.cpu(),
                        "shape": list(x_chunk_torch.shape),
                        "dtype": str(x_chunk_torch.dtype),
                    },
                    dispatch_input_dir / "hidden_states.pt",
                )
                # Convert to float32 for numpy (bfloat16 not supported in numpy)
                np.save(dispatch_input_dir / "hidden_states.npy", x_chunk_torch.cpu().float().numpy())

                # Calculate and save hash (use float32 for consistent hashing)
                x_bytes = x_chunk_torch.cpu().float().numpy().tobytes()
                x_hash = hashlib.md5(x_bytes).hexdigest()
                with open(dispatch_input_dir / "hidden_states_hash.txt", "w") as f:
                    f.write(f"MD5: {x_hash}\n")
                    f.write(f"SHA256: {hashlib.sha256(x_bytes).hexdigest()}\n")
                    f.write(f"Shape: {x_chunk_torch.shape}\n")
                    f.write(f"Dtype: {x_chunk_torch.dtype}\n")

                # Save indices (topk_indices_chunk)
                indices_torch = ttnn.to_torch(
                    topk_indices_chunk,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        cfg["mesh_device"], dims=(-2, -1), mesh_shape=cfg["mesh_device"].shape
                    ),
                )
                torch.save(
                    {
                        "tensor": indices_torch.cpu(),
                        "shape": list(indices_torch.shape),
                        "dtype": str(indices_torch.dtype),
                    },
                    dispatch_input_dir / "indices.pt",
                )
                # Indices are int16, can save directly to numpy
                np.save(dispatch_input_dir / "indices.npy", indices_torch.cpu().numpy())

                # Calculate and save hash
                idx_bytes = indices_torch.cpu().numpy().tobytes()
                idx_hash = hashlib.md5(idx_bytes).hexdigest()
                with open(dispatch_input_dir / "indices_hash.txt", "w") as f:
                    f.write(f"MD5: {idx_hash}\n")
                    f.write(f"SHA256: {hashlib.sha256(idx_bytes).hexdigest()}\n")
                    f.write(f"Shape: {indices_torch.shape}\n")
                    f.write(f"Dtype: {indices_torch.dtype}\n")

                # Save expert mapping (cfg["expert_mapping_tensors"])
                expert_mapping_torch = ttnn.to_torch(
                    cfg["expert_mapping_tensors"],
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        cfg["mesh_device"], dims=(-2, -1), mesh_shape=cfg["mesh_device"].shape
                    ),
                )
                torch.save(
                    {
                        "tensor": expert_mapping_torch.cpu(),
                        "shape": list(expert_mapping_torch.shape),
                        "dtype": str(expert_mapping_torch.dtype),
                    },
                    dispatch_input_dir / "expert_mapping.pt",
                )

                print(f"[REF] Saved dispatch inputs to {dispatch_input_dir}")
                print(f"  hidden_states MD5: {x_hash}")
                print(f"  indices MD5: {idx_hash}")

            if os.environ.get("SAVE_TENSOR_FLOW", "0") == "1":
                # Save RIGHT BEFORE dispatch - the exact inputs
                save_tensor_checkpoint(
                    f"dispatch_input_x_{batch_start}",
                    x_chunk,
                    f"6_dispatch_input_{batch_start}_{batch_end}",
                    "reference",
                    cfg["mesh_device"],
                    {"batch_start": batch_start, "batch_end": batch_end},
                )
                save_tensor_checkpoint(
                    f"dispatch_input_indices_{batch_start}",
                    topk_indices_chunk,
                    f"6_dispatch_input_{batch_start}_{batch_end}",
                    "reference",
                    cfg["mesh_device"],
                    {"batch_start": batch_start, "batch_end": batch_end},
                )

            all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch(
                x_chunk,
                topk_indices_chunk,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_dispatch"],
            )

            # Save dispatch output for comparison
            if os.environ.get("SAVE_EXPERT_CHECKPOINTS", "0") == "1":
                from pathlib import Path

                import torch

                expert_debug_dir = Path("/tmp/expert_debug/reference")
                expert_debug_dir.mkdir(parents=True, exist_ok=True)

                # Save dispatch output
                dispatch_output_torch = ttnn.to_torch(
                    all_to_all_dispatch_output_tensors, mesh_composer=ttnn.ConcatMeshToTensor(cfg["mesh_device"], dim=0)
                )
                torch.save(
                    {
                        "tensor": dispatch_output_torch.cpu(),
                        "shape": list(dispatch_output_torch.shape),
                        "dtype": str(dispatch_output_torch.dtype),
                    },
                    expert_debug_dir / "dispatch_output.pt",
                )

                print(f"[REF] Saved dispatch output shape: {dispatch_output_torch.shape}")
                print(f"[REF] Dispatch output infinities: {torch.isinf(dispatch_output_torch).sum().item()}")
                print(f"[REF] Dispatch output NaNs: {torch.isnan(dispatch_output_torch).sum().item()}")

            ttnn.deallocate(x_chunk)
            ttnn.deallocate(topk_indices_chunk)

            dispatch_chunk = ttnn.reshape(
                all_to_all_dispatch_output_tensors,
                shape=(1, 1, batch_size_chunk * seq_len, cfg["hidden_size"]),
            )
            dispatch_chunk = ttnn.repeat(dispatch_chunk, **cfg["activations_repeat"])
            dispatch_chunk = ttnn.to_layout(dispatch_chunk, ttnn.TILE_LAYOUT)
            ttnn.deallocate(all_to_all_dispatch_output_tensors)

            experts_output = MoEExperts._forward(dispatch_chunk, cfg["moe_experts"])
            ttnn.deallocate(dispatch_chunk)

            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
            experts_output = ttnn.reshape(
                experts_output, shape=(cfg["num_experts_per_device"], batch_size_chunk, seq_len, cfg["hidden_size"])
            )

            all_to_all_dispatch_metadata_tensors = ttnn.reshape(
                all_to_all_dispatch_metadata_tensors,
                shape=(1, batch_size_chunk, seq_len, cfg["num_experts_per_tok"]),
            )

            all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
                experts_output,
                all_to_all_dispatch_metadata_tensors,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_combine"],
            )
            ttnn.deallocate(experts_output)
            ttnn.deallocate(all_to_all_dispatch_metadata_tensors)

            post_combine_output_tensor = ttnn.reshape(
                all_to_all_combine_output_tensors,
                shape=(cfg["num_experts_per_tok"], 1, batch_chunk * seq_len, cfg["hidden_size"]),
            )
            post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)

            topk_weights_chunk = _slice_topk_weights(batch_start, batch_end)

            # Skip saving weights_chunk - focus on activations only
            # The routing weights can be derived from the model if needed

            post_combine_output_tensor = ttnn.mul(
                post_combine_output_tensor, topk_weights_chunk, **cfg["mul_experts_output_with_weights"]
            )
            ttnn.deallocate(topk_weights_chunk)

            post_combine_output_tensor = ttnn.sum(post_combine_output_tensor, dim=0, keepdim=True)
            output_chunks.append(post_combine_output_tensor)

        if len(output_chunks) == 1:
            post_combine_output_tensor = output_chunks[0]
        else:
            post_combine_output_tensor = ttnn.concat(output_chunks, dim=2)
            for chunk in output_chunks:
                ttnn.deallocate(chunk)

        # Save final output checkpoint
        if os.environ.get("SAVE_EXPERT_CHECKPOINTS", "0") == "1":
            from pathlib import Path

            import torch

            expert_debug_dir = Path("/tmp/expert_debug/reference")
            expert_debug_dir.mkdir(parents=True, exist_ok=True)

            # Save final output
            final_output_torch = ttnn.to_torch(
                post_combine_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(cfg["mesh_device"], dim=0)
            )
            torch.save(
                {
                    "tensor": final_output_torch.cpu(),
                    "shape": list(final_output_torch.shape),
                    "dtype": str(final_output_torch.dtype),
                },
                expert_debug_dir / "final_output.pt",
            )

            print(f"[REF] Saved final output shape: {final_output_torch.shape}")
            print(f"[REF] Final output infinities: {torch.isinf(final_output_torch).sum().item()}")
            print(f"[REF] Final output NaNs: {torch.isnan(final_output_torch).sum().item()}")

        ttnn.deallocate(x_rm)
        ttnn.deallocate(topk_experts_indices_rm)
        return post_combine_output_tensor

    @classmethod
    def _fwd_reduce_scatter(
        cls, post_combine_output_tensor: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig, ccl: CCL
    ) -> ttnn.Tensor:
        return ttnn.experimental.reduce_scatter_minimal_async(
            post_combine_output_tensor, **ccl.populate_reduce_scatter_runtime_args(cfg["final_output_reduce_scatter"])
        )

    @classmethod
    def _fwd_all_gather(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        return ttnn.experimental.all_gather_async(x, **cfg["ccl"].populate_all_gather_runtime_args(cfg["revert_tp"]))

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls.forward(x, cfg)
