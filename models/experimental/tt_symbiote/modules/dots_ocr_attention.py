# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.attention import (
    TTNNPagedAttentionKVCache,
    TTNNSDPAAttention,
)
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearLLamaIColShardedWAllReduced,
    TTNNLinearLLamaIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.rope import BailingRotarySetup

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object


@trace_enabled
class TTNNDotsOCRAttention(TTNNModule):
    _shared_rotary_setups = {}

    def __init__(self):
        super().__init__()
        self.num_attention_heads = None
        self.num_key_value_heads = None
        self.num_key_value_groups = None
        self.head_dim = None
        self.hidden_size = None
        self.scaling = None
        self.is_causal = True
        self.layer_idx = None
        self.qkv_proj = None
        self.o_proj = None
        self.sdpa = None
        self.core_grid = None
        self._q_bias_torch = None
        self._k_bias_torch = None
        self._v_bias_torch = None
        self._q_bias = None
        self._k_bias = None
        self._v_bias = None
        self._qkv_bias_torch = None
        self._qkv_bias = None
        self._q_size = None
        self._kv_size = None

    @classmethod
    def from_torch(cls, hf_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = hf_attn

        config = hf_attn.config
        new_attn.num_attention_heads = config.num_attention_heads
        new_attn.num_key_value_heads = config.num_key_value_heads
        new_attn.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        new_attn.head_dim = config.hidden_size // config.num_attention_heads
        new_attn.hidden_size = config.hidden_size
        new_attn.scaling = getattr(hf_attn, "scaling", new_attn.head_dim**-0.5)
        new_attn.layer_idx = hf_attn.layer_idx

        q_size = new_attn.num_attention_heads * new_attn.head_dim
        kv_size = new_attn.num_key_value_heads * new_attn.head_dim
        new_attn._q_size = q_size
        new_attn._kv_size = kv_size

        fused_weight = torch.cat(
            [
                hf_attn.q_proj.weight.data,
                hf_attn.k_proj.weight.data,
                hf_attn.v_proj.weight.data,
            ],
            dim=0,
        )

        fused_linear = torch.nn.Linear(new_attn.hidden_size, q_size + 2 * kv_size, bias=False)
        fused_linear.weight.data = fused_weight
        new_attn.qkv_proj = TTNNLinearLLamaIColShardedWAllReduced.from_torch(fused_linear)

        # Store bias tensors for manual application
        if hf_attn.q_proj.bias is not None:
            new_attn._q_bias_torch = hf_attn.q_proj.bias.data.clone()
        if hf_attn.k_proj.bias is not None:
            new_attn._k_bias_torch = hf_attn.k_proj.bias.data.clone()
        if hf_attn.v_proj.bias is not None:
            new_attn._v_bias_torch = hf_attn.v_proj.bias.data.clone()

        # Fused QKV bias for nlp_create_qkv_heads path
        if (
            new_attn._q_bias_torch is not None
            and new_attn._k_bias_torch is not None
            and new_attn._v_bias_torch is not None
        ):
            new_attn._qkv_bias_torch = torch.cat(
                [
                    new_attn._q_bias_torch,
                    new_attn._k_bias_torch,
                    new_attn._v_bias_torch,
                ],
                dim=0,
            )

        # O projection
        new_attn.o_proj = TTNNLinearLLamaIReplicatedWColSharded.from_torch(hf_attn.o_proj)

        new_attn.sdpa = TTNNSDPAAttention()
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()

        grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            )
            self.sdpa.decode_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.sdpa.decode_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

        # Override QKV compute config: HiFi2 for decode
        self.qkv_proj.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if self.device.get_num_devices() > 1:
            self._decode_cur_pos = ttnn.from_torch(
                torch.zeros(1, dtype=torch.int32),
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self._decode_cur_pos = None

        if self._q_bias_torch is not None:
            self._q_bias = ttnn.from_torch(
                self._q_bias_torch.unsqueeze(0).unsqueeze(0),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._k_bias_torch is not None:
            self._k_bias = ttnn.from_torch(
                self._k_bias_torch.unsqueeze(0).unsqueeze(0),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._v_bias_torch is not None:
            self._v_bias = ttnn.from_torch(
                self._v_bias_torch.unsqueeze(0).unsqueeze(0),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if self._qkv_bias_torch is not None:
            self._qkv_bias = ttnn.from_torch(
                self._qkv_bias_torch.unsqueeze(0).unsqueeze(0),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        config = self._fallback_torch_layer.config
        rope_params = getattr(config, "rope_parameters", {}) or {}
        rope_theta = getattr(config, "rope_theta", rope_params.get("rope_theta", 1000000.0))
        setup_key = (id(self.device), self.head_dim, rope_theta, 1.0)
        if setup_key not in TTNNDotsOCRAttention._shared_rotary_setups:
            TTNNDotsOCRAttention._shared_rotary_setups[setup_key] = BailingRotarySetup(
                device=self.device,
                head_dim=self.head_dim,
                max_seq_len=131072,
                rope_theta=rope_theta,
                partial_rotary_factor=1.0,
                rope_convention="half_half",
            )
        self._rotary_setup = TTNNDotsOCRAttention._shared_rotary_setups[setup_key]

    def _get_cur_pos_device_tensor(self, cache_position, batch_size):
        cp = cache_position
        if isinstance(cp, TorchTTNNTensor):
            cp = cp.ttnn_tensor
        if len(cp.shape) > 1:
            total_elems = 1
            for d in cp.shape:
                total_elems *= d
            cp = ttnn.reshape(cp, (total_elems,))
        if cp.shape[0] > batch_size:
            cp = ttnn.slice(cp, [0], [batch_size])
        if self._decode_cur_pos is not None:
            ttnn.copy(cp, self._decode_cur_pos)
            return self._decode_cur_pos
        return cp

    def _project_qkv_fused(self, hidden_states, batch_size, seq_length):
        """Project hidden states to fused QKV tensor, ready for nlp_create_qkv_heads."""
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qkv_states = self.qkv_proj(hidden_states)

        if self._qkv_bias is not None:
            qkv_states = ttnn.add(qkv_states, self._qkv_bias)

        # Reshape to 4D for nlp_create_qkv_heads: [B, 1, S, D]
        qkv_states = ttnn.reshape(qkv_states, (batch_size, 1, seq_length, -1))

        return qkv_states

    def _forward_prefill(self, hidden_states, attention_mask, past_key_values, cache_position):
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        qkv_states = self._project_qkv_fused(hidden_states, batch_size, seq_length)
        query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
            qkv_states,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_states)

        seq_len = query_states.shape[2]
        cos, sin = self._rotary_setup.get_cos_sin_for_prefill(seq_len)

        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        query_states = ttnn.experimental.rotary_embedding(query_states, cos, sin)
        key_states = ttnn.experimental.rotary_embedding(key_states, cos, sin)

        if query_states.shape[2] != seq_len:
            query_states = query_states[:, :, :seq_len, :]
        if key_states.shape[2] != seq_len:
            key_states = key_states[:, :, :seq_len, :]

        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)
        if past_key_values is not None and use_paged:
            past_key_values.paged_fill_on_device(key_states, value_states, layer_idx=self.layer_idx, batch_idx=0)

        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=True,
        )

        attn_shape = list(attn_output.shape)
        attn_output = ttnn.reshape(
            attn_output, (attn_shape[0], attn_shape[1], self.num_attention_heads * self.head_dim)
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def _forward_decode_paged(
        self,
        hidden_states,
        attention_mask,
        past_key_values,
        cache_position,
        decode_cur_pos_tt=None,
        decode_cos_sin=None,
    ):
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        if decode_cur_pos_tt is not None:
            cur_pos_tt = decode_cur_pos_tt
        else:
            cur_pos_tt = self._get_cur_pos_device_tensor(cache_position, batch_size)

        qkv_states = self._project_qkv_fused(hidden_states, batch_size, seq_length)

        # Decode (seq_length=1): qkv_states is tiny (B*1*(q_dim+2*kv_dim) bf16, ~9 KB).
        # Routing it to L1 lets nlp_create_qkv_heads run on `(in0:l1_interleaved)` instead
        # of the slower `dram_interleaved` bucket — same pattern as the MLP slice fix.
        is_decode = int(seq_length) == 1
        nlp_heads_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        if is_decode and qkv_states.memory_config().buffer_type != ttnn.BufferType.L1:
            qkv_states = ttnn.to_memory_config(qkv_states, ttnn.L1_MEMORY_CONFIG)

        query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
            qkv_states,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=nlp_heads_mc,
        )
        ttnn.deallocate(qkv_states)

        if isinstance(query_states, ttnn.Tensor) and query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if isinstance(key_states, ttnn.Tensor) and key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        if decode_cos_sin is not None:
            cos, sin = decode_cos_sin
        else:
            cos, sin = self._rotary_setup.get_cos_sin_for_decode(cur_pos_tt)
        # K stays in L1 (reshaped/sharded before paged_update_on_device which accepts any
        # memory layout). Q must end up DRAM for paged_sdpa_decode (`Q_memcfg.buffer_type()
        # == DRAM` asserted in sdpa_decode_device_operation.cpp:89). Since rotary_embedding
        # defaults its output memory_config to the input's, we explicitly request DRAM
        # for Q because nlp_create_qkv_heads now produces L1 in decode.
        query_states = ttnn.experimental.rotary_embedding(query_states, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key_states = ttnn.experimental.rotary_embedding(key_states, cos, sin, memory_config=ttnn.L1_MEMORY_CONFIG)

        if query_states.shape[2] != seq_length:
            query_states = query_states[:, :, :seq_length, :]
        if key_states.shape[2] != seq_length:
            key_states = ttnn.slice(
                key_states,
                [0, 0, 0, 0],
                [int(key_states.shape[0]), int(key_states.shape[1]), int(seq_length), int(key_states.shape[3])],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # Permute [B, H, S, D] -> [S, B, H, D] for paged attention kernels
        query_states = ttnn.permute(query_states, (2, 0, 1, 3))
        kv_key = ttnn.permute(key_states, (2, 0, 1, 3))
        kv_value = ttnn.permute(value_states, (2, 0, 1, 3))

        if isinstance(kv_value, ttnn.Tensor) and kv_value.dtype != ttnn.bfloat16:
            kv_value = ttnn.typecast(kv_value, ttnn.bfloat16)

        tile_size = 32
        shard_h = ((self.num_key_value_heads + tile_size - 1) // tile_size) * tile_size

        core_grid = ttnn.CoreGrid(y=1, x=batch_size)
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(shard_h, self.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        kv_key = ttnn.to_memory_config(kv_key, shard_cfg)
        kv_value = ttnn.to_memory_config(kv_value, shard_cfg)

        past_key_values.paged_update_on_device(kv_key, kv_value, layer_idx=self.layer_idx, current_pos=cur_pos_tt)
        ttnn.deallocate(kv_key)
        ttnn.deallocate(kv_value)

        attn_output = past_key_values.paged_sdpa_decode(
            query_states,
            self.layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=getattr(self.sdpa, "decode_program_config", self.sdpa.program_config),
            compute_kernel_config=getattr(self.sdpa, "decode_compute_kernel_config", self.sdpa.compute_kernel_config),
        )

        attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_attention_heads * self.head_dim))

        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        seq_length = hidden_states.shape[1]
        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)

        if use_paged and seq_length == 1:
            return self._forward_decode_paged(
                hidden_states,
                attention_mask,
                past_key_values,
                cache_position,
                decode_cur_pos_tt=kwargs.get("decode_cur_pos_tt"),
                decode_cos_sin=kwargs.get("decode_cos_sin"),
            )
        else:
            return self._forward_prefill(hidden_states, attention_mask, past_key_values, cache_position)
