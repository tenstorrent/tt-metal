# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Normalization layer implementations for TTNN."""

from torch import nn
import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, tree_map, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig, trace_enabled


class TTNNLayerNorm(TTNNModule):
    """TTNN-accelerated LayerNorm."""

    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm):
        """Create TTNNLayerNorm from PyTorch LayerNorm."""
        if layer_norm.weight is None:
            print(f"Warning: LayerNorm layer {layer_norm} has no weight. Using standard LayerNorm.")
            return layer_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = layer_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        """Preprocess LayerNorm weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = nn.LayerNorm(normalized_shape=1)
        self.tt_weight = ttnn.from_torch(self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias = ttnn.from_torch(self.torch_layer.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through LayerNorm."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.layer_norm(
            input_tensor,
            weight=self.tt_weight,
            bias=self.tt_bias,
        )
        return tt_output


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TTNNRMSNorm(TTNNModule):
    @classmethod
    def from_torch(cls, rms_norm: DeepseekV2RMSNorm):
        """Create from PyTorch RMSNorm."""
        if rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def preprocess_weights_impl(self):
        """Preprocess RMSNorm weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = DeepseekV2RMSNorm(hidden_size=1)
        self.tt_weight = ttnn.from_torch(
            self.torch_layer.weight.unsqueeze(0).expand(32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.torch_layer.variance_epsilon)
        return x


@trace_enabled
class TTNNDistributedRMSNorm(TTNNModule):
    """
    Distributed RMSNorm implementation that performs the reduction across devices in the forward pass.

    """

    @property
    def _is_distributed(self):
        """True when running on a multi-device mesh (col-sharded activations; CCL not required for output metadata)."""
        return self.device is not None and self.device.get_num_devices() > 1

    def set_output_tensors_config_impl(self, output_tensors):
        """Col-sharded activations: same mesh composer / logical shape as Qwen decoder attention (dim=-1)."""

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    e.set_distributed_tensor_config(
                        DistributedTensorConfig(
                            mesh_mapper=mesh_mapper,
                            mesh_composer=mesh_composer,
                            logical_shape_fn=logical_shape_for_col_sharded,
                        )
                    )
            return e

        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)
        return tree_map(set_col_sharded_config, output_tensors)

    @classmethod
    def from_torch(cls, rms_norm: "RMSNorm"):
        """Create from PyTorch RMSNorm."""
        if rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Using standard RMSNorm.")
            return rms_norm
        new_layer_norm = cls()
        new_layer_norm._fallback_torch_layer = rms_norm
        return new_layer_norm

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        dim = int(self.torch_layer.weight.shape[0])
        assert dim % 32 == 0, f"TTNNDistributedRMSNorm gamma length {dim} must be divisible by 32"
        w_bf16 = self.torch_layer.weight.to(torch.bfloat16)

        if self.device is None or self.device.get_num_devices() <= 1:
            relayout = w_bf16.view(1, 1, dim // 32, 32)
            self.weight_distributed = ttnn.as_tensor(relayout, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
            return

        mesh_shape = list(self.device.shape)
        ncol = int(mesh_shape[-1])
        n_dev = int(self.device.get_num_devices())
        ntiles = dim // 32

        # ShardTensor2dMesh(..., dims=(None, 2), mesh_shape=(1, ncol)) requires the sharded axis
        # (tile rows) to align with the mesh — e.g. code_predictor q_norm (dim=128 → 4 tiles) on T3K
        # (ncol=8) does not shard as 8 chunks. Width-sharding [1,1,1,dim] breaks
        # rms_norm_post_all_gather (gamma last dim must pad to TILE_WIDTH=32). Use PyTorch RMSNorm
        # for those subgraphs instead (see test_qwen_omni ``_restore_torch_rmsnorm_in_code_predictor``).
        if ntiles % ncol == 0:
            relayout = w_bf16.view(1, 1, ntiles, 32)
            mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=mesh_shape)
        else:
            raise RuntimeError(
                f"TTNNDistributedRMSNorm: gamma (dim={dim}, ntiles={ntiles}) is incompatible with mesh {mesh_shape}: "
                f"need (dim//32) % mesh_width == 0. For small norms (e.g. talker code_predictor), keep HF RMSNorm on CPU."
            )

        self.weight_distributed = ttnn.as_tensor(
            relayout,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, inp):
        original_shape = inp.shape
        if len(original_shape) == 3:
            inp = ttnn.unsqueeze(inp, 1)  # Add batch dimension for RMSNorm
        if inp.layout != ttnn.TILE_LAYOUT:
            inp = ttnn.to_layout(inp, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)
        # AllGather stats
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=self.torch_layer.variance_epsilon,
            weight=self.weight_distributed,
        )
        tt_stats.deallocate(True)

        # Squeeze back to original shape if we added a batch dimension
        if len(original_shape) == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])

        return tt_out


@trace_enabled
class TTNNQwenLayerNorm(TTNNModule):
    """HF ``nn.LayerNorm`` → ``ttnn.layer_norm`` (single device) or distributed pre/all_gather/post (mesh)."""

    @staticmethod
    def _normalized_numel(normalized_shape) -> int:
        if isinstance(normalized_shape, int):
            return int(normalized_shape)
        n = 1
        for d in normalized_shape:
            n *= int(d)
        return n

    @property
    def _is_distributed(self) -> bool:
        return self.device is not None and self.device.get_num_devices() > 1

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        use_row_major_workaround: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.embedding_dim = self._normalized_numel(self.normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = elementwise_affine and bias
        self.use_row_major_workaround = use_row_major_workaround
        self.compute_kernel_config = None
        self.tt_weight = None
        self.tt_bias = None
        self.weight_distributed = None
        self.bias_distributed = None
        # Mesh: True when (emb//32) % mesh_width != 0 — use all_gather + full-width replicated LN (vision 1152 on 1×8).
        self._distributed_gather_layernorm = False
        if self.embedding_dim % 32 != 0:
            raise ValueError(
                f"TTNNQwenLayerNorm: embedding_dim ({self.embedding_dim}) must be divisible by 32 for TTNN tile ops"
            )
        if self.elementwise_affine:
            self.torch_weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.torch_bias = nn.Parameter(torch.zeros(self.normalized_shape)) if self.use_bias else None
        else:
            self.torch_weight = None
            self.torch_bias = None

    def set_output_tensors_config_impl(self, output_tensors):
        """Col-sharded last dim on mesh, or replicated full hidden when using gather + full ``layer_norm``."""

        def set_gather_output_config(e):
            """Replicated activations on mesh; ConcatMeshToTensor(dim=0) for host unwrap (no MeshComposerConfig([0, rank]))."""
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None and self.device is not None:
                e.set_distributed_tensor_config(
                    DistributedTensorConfig(
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                )
            return e

        def materialize_merger_ln_q_one_replica(e):
            """Vision merger ``ln_q`` then HF ``.view``: avoid mesh compose; match ``TTNNQwen3OmniVisionMLP`` slice-after-concat."""
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    e.set_distributed_tensor_config(
                        DistributedTensorConfig(
                            mesh_mapper=mesh_mapper,
                            mesh_composer=mesh_composer,
                            logical_shape_fn=logical_shape_for_col_sharded,
                        )
                    )
            return e

        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)
        if getattr(self, "_distributed_gather_layernorm", False):
            name = self.module_name or ""
            # Do not dim-0-shard LN output: vision rotary cos/sin stay full seq length; sharding breaks q*cos broadcast.
            if "merger" in name and "ln_q" in name:
                return tree_map(materialize_merger_ln_q_one_replica, output_tensors)
            return tree_map(set_gather_output_config, output_tensors)
        return tree_map(set_col_sharded_config, output_tensors)

    @classmethod
    def from_torch(cls, layer_norm: nn.LayerNorm, use_row_major_workaround: bool = False):
        """Symbiote calls ``from_torch(hf_module)`` only — use ``set_device`` for the mesh."""
        if not layer_norm.elementwise_affine:
            return layer_norm
        emb = cls._normalized_numel(layer_norm.normalized_shape)
        if emb % 32 != 0:
            return layer_norm
        new_layer = cls(
            normalized_shape=layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine,
            bias=layer_norm.bias is not None,
            use_row_major_workaround=use_row_major_workaround,
        )
        if layer_norm.weight is not None:
            new_layer.torch_weight = nn.Parameter(layer_norm.weight.data.clone())
        if layer_norm.bias is not None:
            new_layer.torch_bias = nn.Parameter(layer_norm.bias.data.clone())
        new_layer._fallback_torch_layer = layer_norm
        return new_layer

    def preprocess_weights_impl(self):
        if not self.elementwise_affine:
            self.tt_weight = None
            self.tt_bias = None
            return
        # Mesh: sharded ROW_MAJOR gamma in ``move_weights_to_device_impl``, or TILE + replicate when ntiles % width != 0.
        if self.device is not None and self.device.get_num_devices() > 1:
            ncol = int(list(self.device.shape)[-1])
            ntiles = self.embedding_dim // 32
            if ntiles % ncol != 0:
                self._distributed_gather_layernorm = True
                # Host TT tensors without mesh placement; ``move_weights_to_device_impl`` uses
                # ``from_torch(..., device=..., mesh_mapper=...)`` (``to_device`` has no mesh_mapper).
                self.tt_weight = None
                self.tt_bias = None
                self.weight_distributed = None
                self.bias_distributed = None
                return
            self._distributed_gather_layernorm = False
            self.tt_weight = None
            self.tt_bias = None
            return
        weight = self.torch_weight
        bias = self.torch_bias
        if self.use_row_major_workaround:
            layout = ttnn.ROW_MAJOR_LAYOUT
            weight_reshaped = weight.reshape(-1, 32)
            bias_reshaped = bias.reshape(-1, 32) if bias is not None else None
        else:
            layout = ttnn.TILE_LAYOUT
            weight_reshaped = weight.reshape(1, -1)
            bias_reshaped = bias.reshape(1, -1) if bias is not None else None
        self.tt_weight = ttnn.from_torch(weight_reshaped, dtype=ttnn.bfloat16, layout=layout)
        if bias_reshaped is not None:
            self.tt_bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16, layout=layout)
        else:
            self.tt_bias = None

    def _build_sharded_gamma_beta_row_major(self):
        """``[1,1,ntiles,32]`` + ``ShardTensor2dMesh`` on tile dim — matches ``TTNNDistributedRMSNorm``."""
        emb = self.embedding_dim
        mesh_shape = list(self.device.shape)
        ncol = int(mesh_shape[-1])
        ntiles = emb // 32
        assert ntiles % ncol == 0, "gather path should not call _build_sharded_gamma_beta_row_major"
        w_bf16 = self.torch_weight.reshape(-1).to(torch.bfloat16)
        relayout_w = w_bf16.view(1, 1, ntiles, 32)
        mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=mesh_shape)
        self.weight_distributed = ttnn.as_tensor(
            relayout_w,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
        if self.use_bias and self.torch_bias is not None:
            b_bf16 = self.torch_bias.reshape(-1).to(torch.bfloat16)
            relayout_b = b_bf16.view(1, 1, ntiles, 32)
            self.bias_distributed = ttnn.as_tensor(
                relayout_b,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
            self.bias_distributed = ttnn.to_device(self.bias_distributed, self.device)
        else:
            z = torch.zeros(emb, dtype=torch.bfloat16)
            relayout_z = z.view(1, 1, ntiles, 32)
            self.bias_distributed = ttnn.as_tensor(
                relayout_z,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
            self.bias_distributed = ttnn.to_device(self.bias_distributed, self.device)

    def move_weights_to_device_impl(self):
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if not self.elementwise_affine:
            return
        if self.device.get_num_devices() > 1:
            if getattr(self, "_distributed_gather_layernorm", False):
                rep = ttnn.ReplicateTensorToMesh(self.device)
                w = self.torch_weight.reshape(1, -1).to(torch.bfloat16)
                self.tt_weight = ttnn.from_torch(
                    w,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=rep,
                )
                if self.use_bias and self.torch_bias is not None:
                    b = self.torch_bias.reshape(1, -1).to(torch.bfloat16)
                    self.tt_bias = ttnn.from_torch(
                        b,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        mesh_mapper=rep,
                    )
                else:
                    self.tt_bias = None
                return
            self._build_sharded_gamma_beta_row_major()
            return
        if self.tt_weight is not None:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
        if self.tt_bias is not None:
            self.tt_bias = ttnn.to_device(self.tt_bias, self.device)

    def _forward_distributed(self, x: ttnn.Tensor, original_shape: tuple) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rank = len(original_shape)
        if rank == 2:
            x = ttnn.unsqueeze(x, 0)
            x = ttnn.unsqueeze(x, 0)
        elif rank == 3:
            x = ttnn.unsqueeze(x, 1)
        elif rank != 4:
            raise RuntimeError(f"TTNNQwenLayerNorm: expected rank 2–4 activations, got rank {rank}")

        tt_stats = ttnn.layer_norm_pre_all_gather(
            x,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        tt_out = ttnn.layer_norm_post_all_gather(
            x,
            tt_stats,
            epsilon=self.eps,
            weight=self.weight_distributed,
            bias=self.bias_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_stats.deallocate(True)

        if rank == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        elif rank == 2 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [int(tt_out.shape[2]), int(tt_out.shape[3])])
        return tt_out

    def _forward_distributed_gather_ln(self, x: ttnn.Tensor, original_shape: tuple) -> ttnn.Tensor:
        """Col-shard width does not tile-shard evenly on mesh (e.g. vision 1152 on 8 devices): gather, full LN, replicate out."""
        emb = self.embedding_dim
        n_dev = int(self.device.get_num_devices())
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wloc = int(x.shape[-1])
        if wloc * n_dev == emb:
            x = ttnn.all_gather(
                x,
                dim=-1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        elif wloc != emb:
            raise RuntimeError(
                f"TTNNQwenLayerNorm gather path: need col-shard {emb}/{n_dev} or full width {emb}, got last dim {wloc}"
            )

        rank = len(original_shape)
        if rank == 2:
            x = ttnn.unsqueeze(ttnn.unsqueeze(x, 0), 0)
        elif rank == 3:
            x = ttnn.unsqueeze(x, 1)
        elif rank != 4:
            raise RuntimeError(f"TTNNQwenLayerNorm: gather path expected rank 2–4 activations, got rank {rank}")

        tt_out = ttnn.layer_norm(
            x,
            weight=self.tt_weight,
            bias=self.tt_bias,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )

        if rank == 3 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]])
        elif rank == 2 and len(tt_out.shape) == 4:
            tt_out = ttnn.reshape(tt_out, [int(tt_out.shape[2]), int(tt_out.shape[3])])
        return tt_out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        original_shape = tuple(int(d) for d in x.shape)
        if self._is_distributed and self.elementwise_affine:
            if getattr(self, "_distributed_gather_layernorm", False):
                return self._forward_distributed_gather_ln(x, original_shape)
            return self._forward_distributed(x, original_shape)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.layer_norm(
            x,
            weight=self.tt_weight,
            bias=self.tt_bias,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )
