# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...parallel.config import EncoderParallelManager
from ...utils.substate import substate

# Ensure that the fabric is initialized in 1-D mode so that collective ops such as
# all_gather_async can correctly establish routes on a (N, 4) line mesh.
# When running inside the SD-3.5 test-suite the fixtures take care of this, but
# standalone tests (like the CLIP encoder unit-test) need this safeguard.
# ttnn_fabric.set_fabric_config(ttnn_fabric.FabricConfig.FABRIC_1D)

# TODO: merge param and isntance classes


class CLIPAttentionParameters:
    """Parameters for CLIP attention layer"""

    def __init__(self, q_proj, k_proj, v_proj, out_proj):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.out_proj = out_proj

    @classmethod
    def from_torch(
        cls,
        state_dict: dict,
        config,
        mesh_device=None,
        parallel_manager: EncoderParallelManager = None,
    ):
        """Create attention parameters"""
        q_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_manager.tensor_parallel.mesh_axis,
        )
        k_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_manager.tensor_parallel.mesh_axis,
        )
        v_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_manager.tensor_parallel.mesh_axis,
        )

        out_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_manager.tensor_parallel.mesh_axis,
        )

        q_proj.load_state_dict(substate(state_dict, "q_proj"))
        k_proj.load_state_dict(substate(state_dict, "k_proj"))
        v_proj.load_state_dict(substate(state_dict, "v_proj"))
        out_proj.load_state_dict(substate(state_dict, "out_proj"))

        return cls(q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, out_proj=out_proj)


class CLIPMLPParameters:
    """Parameters for CLIP MLP layer"""

    def __init__(self, fc1, fc2):
        self.fc1 = fc1
        self.fc2 = fc2

    @classmethod
    def from_torch(
        cls,
        state_dict: dict,
        config,
        mesh_device=None,
        parallel_manager: EncoderParallelManager = None,
    ):
        """Create MLP parameters from torch state dict"""
        fc1 = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_manager.tensor_parallel.mesh_axis if parallel_manager else 0,
        )

        fc2 = RowParallelLinear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_manager.tensor_parallel.mesh_axis if parallel_manager else 0,
            ccl_manager=parallel_manager,
        )

        fc1.load_state_dict(substate(state_dict, "fc1"))
        fc2.load_state_dict(substate(state_dict, "fc2"))

        return cls(fc1=fc1, fc2=fc2)


class CLIPMLP:
    """CLIP MLP implementation with parallelism"""

    def __init__(self, parameters: CLIPMLPParameters, config, parallel_manager: EncoderParallelManager = None):
        self.config = config
        self.parallel_manager = parallel_manager

        self.fc1 = parameters.fc1
        self.fc2 = parameters.fc2

        self.activation = config.hidden_act

    def __call__(self, hidden_states: ttnn.Tensor, parallel_manager: EncoderParallelManager = None) -> ttnn.Tensor:
        """
        Forward pass of CLIP MLP

        Args:
            hidden_states: Input tensor
            parallel_manager: Parallel manager for distributed operations

        Returns:
            MLP output tensor
        """
        hidden_states = self.fc1(hidden_states)

        if self.activation == "quick_gelu":
            # Quick GELU: x * sigmoid(1.702 * x)
            hidden_states = hidden_states * ttnn.sigmoid(1.702 * hidden_states)
        elif self.activation == "gelu":
            hidden_states = ttnn.gelu(hidden_states)
        elif self.activation == "relu":
            hidden_states = ttnn.relu(hidden_states)
        else:
            # default to quick_gelu if unknown activation
            hidden_states = hidden_states * ttnn.sigmoid(1.702 * hidden_states)

        hidden_states = self.fc2(hidden_states)

        return hidden_states


class CLIPAttention:
    """CLIP attention implementation with head parallelism"""

    def __init__(self, parameters: CLIPAttentionParameters, config, parallel_manager: EncoderParallelManager = None):
        self.config = config
        self.parallel_manager = parallel_manager

        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.tensor_parallel_factor = parallel_manager.tensor_parallel.factor if parallel_manager else 1
        self.num_local_heads = self.num_heads // self.tensor_parallel_factor
        print(
            f"INIT DEBUG: num_heads={self.num_heads}, tp_factor={self.tensor_parallel_factor}, num_local_heads={self.num_local_heads}"
        )

        self.q_proj = parameters.q_proj
        self.k_proj = parameters.k_proj
        self.v_proj = parameters.v_proj
        self.out_proj = parameters.out_proj

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor = None,
        parallel_manager: EncoderParallelManager = None,
    ) -> ttnn.Tensor:
        """
        Forward pass of CLIP attention

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
            causal_attention_mask: Optional attention mask
            parallel_manager: Parallel manager for distributed operations

        Returns:
            Attention output tensor
        """
        batch_size, seq_length, _ = hidden_states.shape

        # project to Q, K, V (column parallel - each device gets a subset of heads)
        q = self.q_proj(hidden_states)  # (batch, seq_len, local_hidden_size)
        k = self.k_proj(hidden_states)  # (batch, seq_len, local_hidden_size)
        v = self.v_proj(hidden_states)  # (batch, seq_len, local_hidden_size)

        q = q * self.scale

        num_devices = parallel_manager.tensor_parallel.factor if parallel_manager else 1
        num_local_heads = self.num_heads // num_devices
        local_hidden_size = self.embed_dim // num_devices

        # debug
        print(f"DEBUG attention shapes:")
        print(f"  hidden_states.shape: {hidden_states.shape}")
        print(f"  q.shape after projection: {q.shape}")
        print(f"  self.num_heads: {self.num_heads}")
        print(f"  self.num_local_heads (stored): {self.num_local_heads}")
        print(f"  runtime_num_local_heads: {num_local_heads}")
        print(f"  self.head_dim: {self.head_dim}")
        print(f"  self.tensor_parallel_factor (stored): {self.tensor_parallel_factor}")
        print(f"  runtime_tp_factor: {num_devices}")
        print(f"  runtime_local_hidden_size: {local_hidden_size}")
        print(
            f"  target reshape volume: {batch_size} * {seq_length} * {num_local_heads} * {self.head_dim} = {batch_size * seq_length * num_local_heads * self.head_dim}"
        )

        q = ttnn.reshape(q, (batch_size, seq_length, num_local_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, seq_length, num_local_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, seq_length, num_local_heads, self.head_dim))

        # transpose to [batch_size, num_heads, seq_length, head_dim]
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))

        if causal_attention_mask is not None:
            scores = scores + causal_attention_mask

        attn_weights = ttnn.softmax(scores, dim=-1)

        # TODO: Add dropout when ttnn.dropout is available
        # attn_weights = ttnn.dropout(attn_weights, self.config.attention_dropout)

        attn_output = ttnn.matmul(attn_weights, v)  # (batch, num_local_heads, seq_len, head_dim)

        attn_output = ttnn.transpose(attn_output, 1, 2)

        attn_output = ttnn.reshape(attn_output, (1, batch_size, seq_length, self.embed_dim // num_devices))

        tp_mesh = parallel_manager.tp_mesh
        attn_output = ttnn.experimental.all_gather_async(
            input_tensor=attn_output,
            dim=len(attn_output.shape) - 1,
            cluster_axis=parallel_manager.tensor_parallel.mesh_axis,  # 1d
            mesh_device=parallel_manager.mesh_device,
            topology=parallel_manager.topology,
            multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(),
            num_links=parallel_manager.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        dense_out = self.out_proj(attn_output)

        dense_out = ttnn.experimental.all_gather_async(
            input_tensor=dense_out,
            dim=len(dense_out.shape) - 1,
            cluster_axis=parallel_manager.tensor_parallel.mesh_axis,  # 1d
            mesh_device=parallel_manager.mesh_device,
            topology=parallel_manager.topology,
            multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(),
            num_links=parallel_manager.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dense_out_shape[2] = orig_shape[2]
        dense_out = ttnn.reshape(dense_out, tuple(dense_out_shape), dense_out.shape)

        return ttnn.reshape(dense_out, tuple(dense_out.shape)[1:])
