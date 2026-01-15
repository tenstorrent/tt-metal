# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.common import RopeScaling, gather_cos_sin, get_rot_transformation_mat
from ttnn import ShardTensor2dMesh, replicate_tensor_to_mesh_mapper


# Copied from DeepseekV3RotaryEmbedding: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L114
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float, device: Optional[Any] = None) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    @staticmethod
    def permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Undo the HF permute
        cos = cos[:, : cos.shape[1] // 2]
        cos = torch.stack((cos, cos), dim=-1).flatten(-2)

        sin = sin[:, : sin.shape[1] // 2]
        sin = torch.stack((sin, sin), dim=-1).flatten(-2)

        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, dim]

        return cos, sin

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        self.register_buffer("freqs_cis", torch.complex(cos.float(), sin.float()), persistent=False)

        cos, sin = self.permute_to_meta_format(cos, sin)
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = x.shape[-2]  # Get sequence length from input tensor
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class ScaledRotaryEmbedding(RotaryEmbedding, ABC):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        device: Optional[Any] = None,
    ) -> None:
        self.scaling_factor = factor
        super().__init__(dim, max_position_embeddings, base, device)

    @abstractmethod
    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        pass

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        t = torch.arange(seq_len * 2.0)
        freqs = self.apply_scaling(freqs)
        freqs = torch.outer(t, freqs).float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("freqs_cis", torch.complex(cos.float(), sin.float()), persistent=False)

        cos, sin = gather_cos_sin(torch.arange(seq_len), cos, sin)
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)


# Copied from DeepseekV3YarnRotaryEmbedding: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L262
class YarnRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        original_max_position_embeddings: int,
        beta_fast: float,
        beta_slow: float,
        mscale: float,
        mscale_all_dim: float,
        device: Optional[Any] = None,
    ) -> None:
        self.scaling_factor = factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    # Inverse dim formula to find dim based on number of rotations
    @staticmethod
    def yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # Find dim range bounds based on rotations
    @staticmethod
    def yarn_find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
    ) -> Tuple[int, int]:
        low = math.floor(YarnRotaryEmbedding.yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(YarnRotaryEmbedding.yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    @staticmethod
    def yarn_get_mscale(scale: float, mscale: float) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def yarn_linear_ramp_mask(min: float, max: float, dim: int) -> torch.Tensor:
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = YarnRotaryEmbedding.yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - YarnRotaryEmbedding.yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            YarnRotaryEmbedding.yarn_get_mscale(self.scaling_factor, self.mscale)
            / YarnRotaryEmbedding.yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * _mscale
        sin = emb.sin() * _mscale
        cos, sin = self.permute_to_meta_format(cos, sin)

        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)


class LinearScaledRotaryEmbedding(ScaledRotaryEmbedding):
    def __init__(
        self, dim: int, max_position_embeddings: int, base: float, factor: float, device: Optional[Any] = None
    ) -> None:
        super().__init__(dim, max_position_embeddings, base, factor, device)

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        return freqs / self.scaling_factor


class LlamaRotaryEmbedding(ScaledRotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        original_max_position_embeddings: int,
        low_freq_factor: float,
        high_freq_factor: float,
        device: Optional[Any] = None,
    ) -> None:
        self.orig_context_len = original_max_position_embeddings
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        super().__init__(dim, max_position_embeddings, base, factor, device)

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        # Llama-3.x specific scaling
        # Values obtained from grid search
        low_freq_wavelen = self.orig_context_len / self.low_freq_factor
        high_freq_wavelen = self.orig_context_len / self.high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / self.scaling_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (self.orig_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / self.scaling_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


class Phi3RotaryEmbedding(ScaledRotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        original_max_position_embeddings: int,
        long_factor: List[int],
        short_factor: List[int],
        device: Optional[Any] = None,
    ) -> None:
        self.orig_context_len = original_max_position_embeddings
        self.long_factor = long_factor
        self.short_factor = short_factor
        scale = 1024 * 128 / self.orig_context_len  # Specific for Phi-3-mini-128k
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.orig_context_len))
        super().__init__(dim, max_position_embeddings, base, scaling_factor, device)

    def apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        if self.max_seq_len_cached > self.orig_context_len:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32)
        assert freqs.shape[-1] == ext_factors.shape[-1]
        return freqs / ext_factors

    def _set_cos_sin_cache(self, seq_len: int, device: Any, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        inv_freq_shape = torch.arange(0, self.dim, 2).float().to(device) / self.dim
        self.inv_freq = 1.0 / (self.base**inv_freq_shape)
        self.inv_freq = self.apply_scaling(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq.to(t.device))

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.scaling_factor
        sin = emb.sin() * self.scaling_factor
        cos, sin = self.permute_to_meta_format(cos, sin)
        self.register_buffer("cos_cached", cos.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.to(dtype), persistent=False)


def rotary_embedding_factory(
    dim: int,
    max_position_embeddings: int,
    base: float,
    rope_scaling: Optional[RopeScaling] = None,
    device: Optional[Any] = None,
) -> Union[RotaryEmbedding, ScaledRotaryEmbedding]:
    if rope_scaling is None:
        return RotaryEmbedding(dim, max_position_embeddings, base, device)
    else:
        if rope_scaling.rope_type.value == "linear":
            rotary_embedding = LinearScaledRotaryEmbedding
        elif rope_scaling.rope_type.value == "llama3":
            rotary_embedding = LlamaRotaryEmbedding
        elif rope_scaling.rope_type.value == "yarn":
            rotary_embedding = YarnRotaryEmbedding
        elif rope_scaling.rope_type.value == "longrope":
            rotary_embedding = Phi3RotaryEmbedding
        else:
            raise ValueError(f"Invalid rope_scaling: {rope_scaling}")
        return rotary_embedding(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            **rope_scaling.model_dump(exclude_none=True),
        )


def compute_freqs_cis(
    dhead: int, end: int, theta: float, rope_scaling: Optional[RopeScaling]
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary_embedding = rotary_embedding_factory(
        dim=dhead, max_position_embeddings=end // 2, base=theta, rope_scaling=rope_scaling
    )
    return rotary_embedding.freqs_cis


def compute_gather_cos_sin(
    dhead: int, end: int, theta: float, rope_scaling: Optional[RopeScaling]
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary_embedding = rotary_embedding_factory(
        dim=dhead, max_position_embeddings=end // 2, base=theta, rope_scaling=rope_scaling
    )
    return rotary_embedding.cos_cached, rotary_embedding.sin_cached


def get_rot_mats(
    head_dim: int,
    device: Any,
    seq_len: int,
    theta: float,
    rope_scaling: Optional[RopeScaling],
    datatype: Any = ttnn.bfloat16,
) -> List[ttnn.Tensor]:
    cos_matrix, sin_matrix = compute_gather_cos_sin(
        dhead=head_dim,
        end=2 * seq_len,
        theta=theta,
        rope_scaling=rope_scaling,
    )

    cos_matrix = ttnn.from_torch(
        cos_matrix,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=datatype,
        mesh_mapper=replicate_tensor_to_mesh_mapper(device),
    )
    sin_matrix = ttnn.from_torch(
        sin_matrix,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=datatype,
        mesh_mapper=replicate_tensor_to_mesh_mapper(device),
    )

    return [cos_matrix, sin_matrix]


class RotarySetup(LightweightModule):
    def __init__(
        self,
        device: Any,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        rope_scaling: Optional[RopeScaling] = None,
        use_qk_fused: bool = False,
        datatype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.use_qk_fused = use_qk_fused
        self.original_batch_size = batch_size

        # NOTE: If qk fused ops (rotary embedding + paged cache update) are used
        # we need to double the batch size in order to replicate the transformation matrix on double the batch size number of cores
        self.doubled_batch_size = self.original_batch_size * 2 if use_qk_fused else self.original_batch_size
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1
        if self.num_devices == 32:
            self.batch_size_per_device_group = max(self.doubled_batch_size // list(device.shape)[1], 1)
        else:
            self.batch_size_per_device_group = self.doubled_batch_size
        self.core_grid = (
            ttnn.CoreCoord(8, 8) if ttnn.get_arch_name() == "blackhole" else device.compute_with_storage_grid_size()
        )

        # Generate the cos/sin matrices needed for ttnn.embedding op
        self.cos_matrix, self.sin_matrix = get_rot_mats(
            head_dim=head_dim,
            device=device,
            seq_len=max_seq_len,
            theta=rope_theta,
            rope_scaling=rope_scaling,
            datatype=datatype,
        )

        self.batch_grid = ttnn.num_cores_to_corerangeset(self.doubled_batch_size, self.core_grid, row_wise=True)

        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1,
            1,
            self.doubled_batch_size,
            1,
            # 1, 1, num_cores, 1
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=(
                ShardTensor2dMesh(
                    device,
                    dims=(None, 2) if (self.num_devices == 32 and batch_size > 1) else (None, None),
                    mesh_shape=list(device.shape),
                )
                if self.is_mesh_device
                else None
            ),
        )

        # TODO: Colman, should this be TILE_SIZE or head_dim? Why should it be different for prefill and decode?
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_to_mesh_mapper(device),
        )

    def get_both_trans_mats(self) -> Dict[str, ttnn.Tensor]:
        assert self.transformation_mat is not None, "Transformation matrix not initialized"
        assert self.transformation_mat_prefill is not None, "Prefill Transformation matrix not initialized"
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs: torch.Tensor, on_host: bool = False) -> ttnn.Tensor:
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"

        if self.use_qk_fused:
            # NOTE: For fused QK ops (rotary embedding + paged cache update), we intentionally double the batch dimension so that
            # the rotary indices can be used for Q and K tensors each.
            position_idxs = position_idxs.repeat(2)
            assert (
                position_idxs.shape[0] == self.batch_size_per_device_group
            ), "Position idxs must be the same as the batch size per device group"

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)  # [1, 1, 1, batch]
        assert position_idxs.shape == (1, batch), "Position idxs must be a [1, batch] tensor"
        assert torch.min(position_idxs) >= 0, "Position idxs must be non-negative"

        # Add padding if needed
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

        if on_host:  # If tensor is on host, don't pass a mesh mapper if single-device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=replicate_tensor_to_mesh_mapper(self.device),
            )
        else:  # On device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_to_mesh_mapper(self.device),
            )

        return rot_idxs

    def get_rot_mats(
        self, position_idxs: Union[torch.Tensor, ttnn.Tensor], return_rot_idxs: bool = False
    ) -> List[ttnn.Tensor]:
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"
        # Send the idxs to device

        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=embedding_layout)  # [1, batch, head_dim]
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=embedding_layout)  # [1, batch, head_dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, head_dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1[32], head_dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch, 1[32], head_dim]

        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
