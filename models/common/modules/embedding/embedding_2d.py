# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTTv2 embedding for the canonical Wormhole Galaxy (8, 4) mesh."""

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight

_GALAXY_MESH_SHAPE = (8, 4)


@dataclass(frozen=True)
class Embedding2DConfig:
    """Resolved placement and execution policy for :class:`Embedding2D`.

    ``weights`` has host shape ``[vocab_size, dim]``. It is replicated over
    mesh rows and sharded over the hidden dimension on mesh columns.
    """

    weights: LazyWeight
    mesh_device: ttnn.MeshDevice | None = None
    vocab_size: int | None = None
    dim: int | None = None
    max_batch_size: int = 32
    embed_scale: float = 1.0
    weights_dtype: ttnn.DataType | None = None
    weights_memcfg: ttnn.MemoryConfig | None = None
    decode_output_dtype: ttnn.DataType | None = None
    decode_output_memcfg: ttnn.MemoryConfig | None = None
    prefill_output_dtype: ttnn.DataType | None = None
    prefill_output_memcfg: ttnn.MemoryConfig | None = None

    def is_resolved(self) -> bool:
        return all(getattr(self, field) is not None for field in self.__dataclass_fields__)


class Embedding2D(LightweightModule):
    """Embedding lookup whose output is hidden-sharded over Galaxy columns."""

    def __init__(self, weights: LazyWeight, embed_scale: float = 1.0):
        super().__init__()
        self.config = _resolve_embedding2d_config(Embedding2DConfig(weights=weights, embed_scale=embed_scale))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: Embedding2DConfig):
        instance = object.__new__(cls)
        super(Embedding2D, instance).__init__()
        instance.config = _resolve_embedding2d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.weights = self.config.weights.get_device_weight()
        self._device_weights_loaded = True

    def decode_forward(self, token_ids: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        return self._forward(token_ids, self.config.decode_output_dtype, self.config.decode_output_memcfg)

    def prefill_forward(self, token_ids: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        return self._forward(token_ids, self.config.prefill_output_dtype, self.config.prefill_output_memcfg)

    def forward(self, token_ids: ttnn.Tensor | LazyWeight, mode: str) -> ttnn.Tensor:
        if mode == "decode":
            return self.decode_forward(token_ids)
        if mode == "prefill":
            return self.prefill_forward(token_ids)
        raise ValueError(f"Unknown mode: {mode!r}; expected 'decode' or 'prefill'")

    def _forward(self, token_ids, output_dtype, output_memcfg):
        self.load_device_weights()
        token_ids, owns_token_ids = _load_token_ids(token_ids, self.config)
        try:
            output = ttnn.embedding(
                token_ids,
                self.weights,
                layout=ttnn.TILE_LAYOUT,
                dtype=output_dtype,
                memory_config=output_memcfg,
            )
        finally:
            if owns_token_ids:
                ttnn.deallocate(token_ids)
        output = ttnn.reshape(output, ttnn.Shape((1, 1, output.shape[-2], output.shape[-1])))
        if self.config.embed_scale == 1.0:
            return output
        scaled_output = ttnn.multiply(output, self.config.embed_scale, memory_config=output_memcfg)
        if scaled_output is not output:
            ttnn.deallocate(output)
        return scaled_output

    def release(self) -> None:
        value = self.config.weights._value
        if value is not None:
            ttnn.deallocate(value)
            self.config.weights._value = None
        if hasattr(self, "weights"):
            del self.weights
        self._device_weights_loaded = False


def _resolve_embedding2d_config(config: Embedding2DConfig) -> Embedding2DConfig:
    mesh_device = _derive_mesh_device("Embedding2D", config.mesh_device, (config.weights,))
    _validate_galaxy_mesh("Embedding2D", mesh_device)

    shape = tuple(config.weights.source.shape)
    if len(shape) != 2:
        raise ValueError(f"Embedding2D weight must have host shape [vocab_size, dim], got {shape}")
    vocab_size, dim = shape
    if config.vocab_size is not None and config.vocab_size != vocab_size:
        raise ValueError(f"Embedding2D vocab_size {config.vocab_size} does not match weight shape {shape}")
    if config.dim is not None and config.dim != dim:
        raise ValueError(f"Embedding2D dim {config.dim} does not match weight shape {shape}")
    if dim % _GALAXY_MESH_SHAPE[1]:
        raise ValueError(f"Embedding2D dim {dim} must be divisible by 4 Galaxy columns")
    if config.max_batch_size != 32:
        raise ValueError("Embedding2D supports Galaxy physical batch 32 only")
    if config.embed_scale <= 0:
        raise ValueError("Embedding2D embed_scale must be positive")

    weights_dtype = config.weights_dtype or ttnn.bfloat16
    weights_memcfg = config.weights_memcfg or ttnn.DRAM_MEMORY_CONFIG
    mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape(*_GALAXY_MESH_SHAPE),
    )
    physical_source = config.weights.source
    reshape = getattr(physical_source, "reshape", None)
    if callable(reshape):
        physical_source = reshape(1, 1, vocab_size, dim)
    weights = resolve_lazy_weight(
        replace(config.weights, source=physical_source),
        device=mesh_device,
        dtype=weights_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=weights_memcfg,
        mesh_mapper_config=mapper,
    )
    resolved = replace(
        config,
        weights=weights,
        mesh_device=mesh_device,
        vocab_size=vocab_size,
        dim=dim,
        weights_dtype=weights_dtype,
        weights_memcfg=weights_memcfg,
        decode_output_dtype=config.decode_output_dtype or ttnn.bfloat16,
        decode_output_memcfg=config.decode_output_memcfg or ttnn.L1_MEMORY_CONFIG,
        prefill_output_dtype=config.prefill_output_dtype or ttnn.bfloat8_b,
        prefill_output_memcfg=config.prefill_output_memcfg or ttnn.DRAM_MEMORY_CONFIG,
    )
    if not resolved.is_resolved():
        raise ValueError("Embedding2D config did not resolve completely")
    return resolved


def _load_token_ids(token_ids: ttnn.Tensor | LazyWeight, config: Embedding2DConfig) -> tuple[ttnn.Tensor, bool]:
    if isinstance(token_ids, LazyWeight):
        if token_ids.device is not None and token_ids.device is not config.mesh_device:
            raise ValueError("Embedding2D token_ids belong to a different mesh")
        token_ids = replace(
            token_ids,
            device=config.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=None,
        ).get_device_weight()
        return token_ids, True
    if not isinstance(token_ids, ttnn.Tensor):
        raise TypeError("Embedding2D token_ids must be a TTNN tensor or LazyWeight")
    return token_ids, False


def _derive_mesh_device(name: str, configured, weights: tuple[LazyWeight, ...]):
    mesh_device = configured or weights[0].device or ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError(f"{name} mesh_device must be provided")
    for index, weight in enumerate(weights):
        if weight.device is not None and weight.device is not mesh_device:
            raise ValueError(f"{name} weight {index} belongs to a different mesh")
    return mesh_device


def _validate_galaxy_mesh(name: str, mesh_device) -> None:
    shape = tuple(mesh_device.shape)
    if shape != _GALAXY_MESH_SHAPE:
        raise ValueError(f"{name} requires logical mesh shape (8, 4), got {shape}")
    if mesh_device.get_num_devices() != 32:
        raise ValueError(f"{name} requires exactly 32 devices")
    if mesh_device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        raise ValueError(f"{name} supports Wormhole only")
