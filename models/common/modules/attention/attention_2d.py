# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-resolved 2D attention recipes for the Wormhole Galaxy mesh."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.rmsnorm.rmsnorm_2d import RMSNorm2D, RMSNorm2DConfig, RMSNorm2DGeometry

GALAXY_MESH_SHAPE = (8, 4)
GALAXY_DEVICE_COUNT = 32


class PrefillRowMode(str, Enum):
    SINGLE_ROW = "single-row"
    CONCAT_32 = "concat-32"


class PrefillCollectiveMode(str, Enum):
    REGULAR = "regular"
    RING = "ring"


class PrefillAttentionMode(str, Enum):
    REGULAR = "regular"
    PREFIX_CHUNKED = "prefix/chunked"


@dataclass(frozen=True, order=True)
class PrefillRecipeIdentity:
    """The complete identity of one statically tuned prefill recipe."""

    sequence_length: int
    row_mode: PrefillRowMode
    collective_mode: PrefillCollectiveMode
    attention_mode: PrefillAttentionMode

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("prefill recipe sequence_length must be positive")
        object.__setattr__(self, "row_mode", PrefillRowMode(self.row_mode))
        object.__setattr__(self, "collective_mode", PrefillCollectiveMode(self.collective_mode))
        object.__setattr__(self, "attention_mode", PrefillAttentionMode(self.attention_mode))


@dataclass(frozen=True)
class Attention2DSequenceConfig:
    """Frozen program, memory, and kernel values for one prefill recipe."""

    identity: PrefillRecipeIdentity
    qkv_program_config: Any
    sdpa_program_config: Any
    wo_program_config: Any
    qkv_output_memory_config: Any
    heads_memory_config: Any
    kv_memory_config: Any
    sdpa_output_memory_config: Any
    concat_memory_config: Any
    wo_output_memory_config: Any
    qkv_kernel_config: Any
    sdpa_kernel_config: Any
    wo_kernel_config: Any
    activation_dtype: Any
    chunk_alignment: int = 128

    def __post_init__(self) -> None:
        if self.chunk_alignment <= 0:
            raise ValueError("chunk_alignment must be positive")
        if self.identity.sequence_length % self.chunk_alignment:
            raise ValueError("sequence_length must be divisible by chunk_alignment")
        missing = tuple(
            name
            for name in self.__dataclass_fields__
            if name not in {"identity", "chunk_alignment"} and getattr(self, name) is None
        )
        if missing:
            raise ValueError(f"prefill recipe policy is incomplete: {missing}")

    @property
    def sequence_length(self) -> int:
        return self.identity.sequence_length


@dataclass(frozen=True)
class PagedKVMetadata:
    block_size: int
    max_num_blocks: int
    cache_dtype: Any
    page_table_dtype: Any

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.max_num_blocks <= 0:
            raise ValueError("paged KV block_size and max_num_blocks must be positive")
        if self.cache_dtype is None or self.page_table_dtype is None:
            raise ValueError("paged KV cache and page-table dtypes are required")


@dataclass(frozen=True)
class KVCacheBinding:
    keys: Any
    values: Any
    owner: object
    metadata: PagedKVMetadata | None = None
    mesh_device: Any = None

    def __post_init__(self) -> None:
        if self.keys is None or self.values is None:
            raise ValueError("both key and value cache tensors are required")
        if self.owner is None:
            raise ValueError("KV cache binding requires a non-None owner token")


@dataclass(frozen=True)
class DecodeMetadata:
    current_positions: Any
    page_table: Any = None


@dataclass(frozen=True)
class PrefillMetadata:
    sequence_length: int
    user_ids: tuple[int, ...] = (0,)
    collective_mode: PrefillCollectiveMode = PrefillCollectiveMode.REGULAR
    page_table: Any = None
    chunk_page_table: Any = None
    chunk_start: int | None = None
    chunk_start_tensor: Any = None
    prefix_user_id: int | None = None

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.user_ids:
            raise ValueError("user_ids must not be empty")
        object.__setattr__(self, "collective_mode", PrefillCollectiveMode(self.collective_mode))


RotaryCallable = Callable[..., tuple[Any, Any]]
CollectiveCallable = Callable[..., Any]
RuntimeTensorFactory = Callable[[tuple[int, ...], tuple[int, ...], tuple[int, ...], Any], tuple[Any, Any, Any]]
TensorReleaser = Callable[[Any], None]


def _collective_output_is_owned(_tensor: Any) -> bool:
    return False


@dataclass(frozen=True)
class Attention2DLowLevelCallables:
    """Narrow adapters for 2D APIs whose resource signatures are not stable."""

    rotary: RotaryCallable
    reduce_qkv: CollectiveCallable
    gather_heads: CollectiveCallable
    reduce_output: CollectiveCallable
    is_borrowed_output: Callable[[Any], bool] = _collective_output_is_owned
    reduce_create_qkv_heads: CollectiveCallable | None = None
    gather_users: CollectiveCallable | None = None

    def __post_init__(self) -> None:
        required = ("rotary", "reduce_qkv", "gather_heads", "reduce_output", "is_borrowed_output")
        missing = tuple(name for name in required if not callable(getattr(self, name)))
        if missing:
            raise TypeError(f"low-level attention callables must be callable: {missing}")
        if self.reduce_create_qkv_heads is not None and not callable(self.reduce_create_qkv_heads):
            raise TypeError("low-level reduce_create_qkv_heads must be callable when provided")
        if self.gather_users is not None and not callable(self.gather_users):
            raise TypeError("low-level gather_users must be callable when provided")


@dataclass(frozen=True)
class Attention2DConfig:
    """Complete immutable model tuning and placement policy."""

    wqkv: LazyWeight
    wo: LazyWeight
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    low_level: Attention2DLowLevelCallables
    runtime_tensor_factory: RuntimeTensorFactory
    runtime_tensor_releaser: TensorReleaser

    wqkv_bias: LazyWeight | None = None
    prefill_wqkv: LazyWeight | None = None
    prefill_wo: LazyWeight | None = None
    q_norm_config: RMSNorm2DConfig | None = None
    k_norm_config: RMSNorm2DConfig | None = None
    mesh_device: Any = None
    mesh_shape: tuple[int, int] = GALAXY_MESH_SHAPE
    architecture: Any = None
    dim: int | None = None
    qkv_size: int | None = None
    scale: float | None = None
    users_per_column: int = 8

    wqkv_mesh_mapper_config: Any = None
    wo_mesh_mapper_config: Any = None
    bias_mesh_mapper_config: Any = None
    weight_memory_config: Any = None
    wo_weight_memory_config: Any = None
    weight_layout: Any = None
    wqkv_dtype: Any = None
    wo_dtype: Any = None
    bias_dtype: Any = None

    decode_input_placement: Any = None
    decode_output_placement: Any = None
    prefill_input_placement: Any = None
    prefill_output_placement: Any = None
    decode_qkv_output_memory_config: Any = None
    decode_heads_memory_config: Any = None
    decode_kv_memory_config: Any = None
    decode_sdpa_output_memory_config: Any = None
    decode_concat_memory_config: Any = None
    decode_concat_sub_core_grids: Any = None
    decode_wo_output_memory_config: Any = None
    decode_program_config: Any = None
    decode_sdpa_program_config: Any = None
    decode_wo_program_config: Any = None
    decode_qkv_kernel_config: Any = None
    decode_sdpa_kernel_config: Any = None
    decode_wo_kernel_config: Any = None
    decode_activation_dtype: Any = None

    prefill_sequence_configs: Mapping[PrefillRecipeIdentity, Attention2DSequenceConfig] = field(default_factory=dict)
    decode_prefetch_context: Any = None
    prefill_prefetch_context: Any = None
    intermediate_releaser: TensorReleaser = ttnn.deallocate
    kv_cache: KVCacheBinding | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefill_sequence_configs", MappingProxyType(dict(self.prefill_sequence_configs)))

    @property
    def batch_offsets(self) -> tuple[int, ...]:
        return tuple(range(0, self.max_batch_size, self.users_per_column))

    @property
    def prefix_lower_bounds(self) -> tuple[int, ...]:
        return self.batch_offsets

    @property
    def prefix_upper_bounds(self) -> tuple[int, ...]:
        return tuple(offset + self.users_per_column for offset in self.batch_offsets)

    def sequence_config(self, identity: PrefillRecipeIdentity) -> Attention2DSequenceConfig:
        try:
            return self.prefill_sequence_configs[identity]
        except KeyError as error:
            supported = tuple(
                sorted(
                    self.prefill_sequence_configs,
                    key=lambda item: (
                        item.sequence_length,
                        item.row_mode.value,
                        item.collective_mode.value,
                        item.attention_mode.value,
                    ),
                )
            )
            raise ValueError(f"no prefill config for recipe={identity}; supported={supported}") from error


def _source_shape(weight: LazyWeight) -> tuple[int, ...]:
    shape = getattr(getattr(weight, "source", None), "shape", None)
    if shape is None:
        raise ValueError("attention weights must expose source.shape")
    return tuple(int(value) for value in shape)


def _matrix_shape(name: str, weight: LazyWeight) -> tuple[int, int]:
    shape = _source_shape(weight)
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 2:
        raise ValueError(f"{name} must have a matrix source shape, got {_source_shape(weight)}")
    return shape


def _architecture_name(value: Any) -> str:
    if value is None:
        return ""
    candidate = getattr(value, "name", None)
    if candidate is None:
        candidate = getattr(value, "value", value)
    if not isinstance(candidate, (str, int)):
        candidate = str(candidate)
    return str(candidate).rsplit(".", 1)[-1].replace("-", "_").strip().lower()


def _mesh_architecture(mesh_device: Any) -> str:
    arch = mesh_device.arch() if callable(getattr(mesh_device, "arch", None)) else getattr(mesh_device, "arch", None)
    return _architecture_name(arch)


def _tensor_shape(name: str, tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ValueError(f"{name} must expose shape")
    return tuple(int(value) for value in shape)


def _tensor_dtype(name: str, tensor: Any) -> Any:
    if not hasattr(tensor, "dtype"):
        raise ValueError(f"{name} must expose dtype")
    return tensor.dtype


def _tensor_memory_config(tensor: Any) -> Any:
    value = getattr(tensor, "memory_config", None)
    return value() if callable(value) else value


def _require_placement(name: str, tensor: Any, expected: Any) -> None:
    actual = _tensor_memory_config(tensor)
    if actual != expected:
        raise ValueError(f"{name} placement must be exactly {expected!r}, got {actual!r}")


def _require_activation(name: str, tensor: Any, placement: Any, dtype: Any) -> None:
    _require_placement(name, tensor, placement)
    if _tensor_dtype(name, tensor) != dtype:
        raise ValueError(f"{name} dtype must be exactly {dtype!r}, got {_tensor_dtype(name, tensor)!r}")


def _prefetch_kwargs(context: Any) -> dict[str, Any]:
    if context is None:
        return {}
    return {
        "global_cb": getattr(context, "global_cb", None),
        "sub_device_id": getattr(context, "worker_sub_device_id", getattr(context, "sub_device_id", None)),
    }


def _validate_same_mesh(name: str, value: Any, mesh_device: Any) -> None:
    collaborator_mesh = getattr(value, "mesh_device", getattr(value, "device", None))
    if collaborator_mesh is not None and collaborator_mesh is not mesh_device:
        raise ValueError(f"{name} belongs to a different mesh")


def _require_exact_weight_policy(name: str, weight: LazyWeight, config: Attention2DConfig) -> None:
    memory_config = config.wo_weight_memory_config if name == "wo" else config.weight_memory_config
    expected = {
        "device": config.mesh_device,
        "mesh_mapper_config": getattr(config, f"{name}_mesh_mapper_config"),
        "memory_config": memory_config,
        "layout": config.weight_layout,
        "dtype": getattr(config, f"{name}_dtype"),
    }
    mismatched = tuple(field for field, value in expected.items() if value is None or getattr(weight, field) != value)
    if mismatched:
        raise ValueError(f"{name} weight placement must exactly match config fields: {mismatched}")


def _required_policy(config: Attention2DConfig) -> None:
    optional = {
        "wqkv_bias",
        "prefill_wqkv",
        "prefill_wo",
        "q_norm_config",
        "k_norm_config",
        "mesh_device",
        "architecture",
        "dim",
        "qkv_size",
        "scale",
        "bias_mesh_mapper_config",
        "bias_dtype",
        "wo_weight_memory_config",
        "kv_cache",
    }
    required_names = {
        "wqkv_mesh_mapper_config",
        "wo_mesh_mapper_config",
        "weight_memory_config",
        "weight_layout",
        "wqkv_dtype",
        "wo_dtype",
        "decode_input_placement",
        "decode_output_placement",
        "prefill_input_placement",
        "prefill_output_placement",
        "decode_qkv_output_memory_config",
        "decode_heads_memory_config",
        "decode_kv_memory_config",
        "decode_sdpa_output_memory_config",
        "decode_concat_memory_config",
        "decode_wo_output_memory_config",
        "decode_program_config",
        "decode_sdpa_program_config",
        "decode_wo_program_config",
        "decode_qkv_kernel_config",
        "decode_sdpa_kernel_config",
        "decode_wo_kernel_config",
        "decode_activation_dtype",
    }
    missing = tuple(sorted(name for name in required_names if name not in optional and getattr(config, name) is None))
    if missing:
        raise ValueError(f"Attention2D placement/program policy is incomplete: {missing}")
    for name in ("runtime_tensor_factory", "runtime_tensor_releaser", "intermediate_releaser"):
        if not callable(getattr(config, name)):
            raise TypeError(f"Attention2D requires callable {name}")


def resolve_attention2d_config(config: Attention2DConfig) -> Attention2DConfig:
    """Validate all host-visible policy without materializing a TTNN tensor."""

    if not isinstance(config, Attention2DConfig):
        raise TypeError("resolve_attention2d_config expects Attention2DConfig")
    mesh_device = config.mesh_device or config.wqkv.device or config.wo.device
    if mesh_device is None:
        raise ValueError("mesh_device is required")
    mesh_shape = tuple(getattr(mesh_device, "shape", config.mesh_shape))
    if mesh_shape != GALAXY_MESH_SHAPE or tuple(config.mesh_shape) != GALAXY_MESH_SHAPE:
        raise ValueError(f"Attention2D requires logical mesh shape {GALAXY_MESH_SHAPE}, got {mesh_shape}")
    get_num_devices = getattr(mesh_device, "get_num_devices", None)
    count = get_num_devices() if callable(get_num_devices) else mesh_shape[0] * mesh_shape[1]
    if count != GALAXY_DEVICE_COUNT:
        raise ValueError(f"Attention2D requires {GALAXY_DEVICE_COUNT} devices, got {count}")
    detected = _mesh_architecture(mesh_device)
    requested = _architecture_name(config.architecture) or detected
    if "wormhole" not in requested or (detected and "wormhole" not in detected):
        raise ValueError(f"Attention2D supports Wormhole only, got requested={requested!r}, mesh={detected!r}")
    if config.users_per_column != 8 or config.max_batch_size != GALAXY_DEVICE_COUNT:
        raise ValueError("Attention2D requires max_batch_size=32 and users_per_column=8")
    if min(config.n_heads, config.n_kv_heads, config.head_dim, config.max_seq_len) <= 0:
        raise ValueError("head counts, head_dim, and max_seq_len must be positive")
    if config.n_heads % GALAXY_MESH_SHAPE[0] or config.n_kv_heads % GALAXY_MESH_SHAPE[0]:
        raise ValueError("n_heads and n_kv_heads must be divisible by the mesh row partition")

    qkv_size = config.head_dim * (config.n_heads + 2 * config.n_kv_heads)
    # Architectures that decouple head_dim from the hidden size (Qwen3) project
    # attention into n_heads * head_dim before WO reduces back to dim.
    attention_dim = config.head_dim * config.n_heads
    wqkv_shape, wo_shape = _matrix_shape("wqkv", config.wqkv), _matrix_shape("wo", config.wo)
    dim = config.dim or wqkv_shape[0]
    if config.qkv_size is not None and config.qkv_size != qkv_size:
        raise ValueError(f"qkv_size must equal {qkv_size}")
    if wqkv_shape != (dim, qkv_size):
        raise ValueError(f"wqkv source shape must be {(dim, qkv_size)}, got {wqkv_shape}")
    if attention_dim % GALAXY_MESH_SHAPE[0]:
        raise ValueError("n_heads * head_dim must be divisible by the mesh row partition")
    if wo_shape != (attention_dim, dim):
        raise ValueError(f"wo source shape must be {(attention_dim, dim)}, got {wo_shape}")
    if (config.prefill_wqkv is None) != (config.prefill_wo is None):
        raise ValueError("prefill_wqkv and prefill_wo must be supplied together")
    if config.prefill_wqkv is not None:
        if _matrix_shape("prefill_wqkv", config.prefill_wqkv) != wqkv_shape:
            raise ValueError("prefill_wqkv source shape must match wqkv")
        if _matrix_shape("prefill_wo", config.prefill_wo) != wo_shape:
            raise ValueError("prefill_wo source shape must match wo")
    if dim % GALAXY_MESH_SHAPE[1]:
        raise ValueError("dim must be divisible by the mesh column partition")
    if config.wqkv_bias is not None and _source_shape(config.wqkv_bias) not in {
        (qkv_size,),
        (1, qkv_size),
        (1, 1, 1, qkv_size),
    }:
        raise ValueError("wqkv_bias source shape must end in qkv_size")
    if (config.q_norm_config is None) != (config.k_norm_config is None):
        raise ValueError("q_norm_config and k_norm_config must be supplied together")
    for name, norm in (("q_norm_config", config.q_norm_config), ("k_norm_config", config.k_norm_config)):
        if norm is not None:
            if _source_shape(norm.weight)[-1] != config.head_dim:
                raise ValueError(f"{name} weight must normalize head_dim={config.head_dim}")
            if norm.geometry is not RMSNorm2DGeometry.HEAD_LOCAL:
                raise ValueError(f"{name} must use head-local RMSNorm2D geometry")

    resolved = replace(
        config,
        mesh_device=mesh_device,
        wo_weight_memory_config=config.wo_weight_memory_config or config.weight_memory_config,
    )
    if not isinstance(resolved.low_level, Attention2DLowLevelCallables):
        raise TypeError("Attention2D low_level must be Attention2DLowLevelCallables")
    _required_policy(resolved)
    _require_exact_weight_policy("wqkv", resolved.wqkv, resolved)
    _require_exact_weight_policy("wo", resolved.wo, resolved)
    if resolved.wqkv_bias is not None:
        _require_exact_weight_policy("bias", resolved.wqkv_bias, resolved)
    for name, value in (
        ("wqkv", resolved.wqkv),
        ("wo", resolved.wo),
        ("prefill_wqkv", resolved.prefill_wqkv),
        ("prefill_wo", resolved.prefill_wo),
        ("wqkv_bias", resolved.wqkv_bias),
    ):
        if value is not None:
            _validate_same_mesh(name, value, mesh_device)
    for mode, context in (
        ("decode", resolved.decode_prefetch_context),
        ("prefill", resolved.prefill_prefetch_context),
    ):
        if context is None:
            continue
        _validate_same_mesh(f"{mode} prefetch context", context, mesh_device)
        if getattr(context, "mode", mode) != mode:
            raise ValueError(f"{mode} prefetch context has incompatible mode")

    if not resolved.prefill_sequence_configs:
        raise ValueError("at least one identity-keyed prefill config is required")
    for key, recipe in resolved.prefill_sequence_configs.items():
        if key != recipe.identity:
            raise ValueError("prefill recipe keys must exactly match recipe.identity")
        if key.sequence_length > resolved.max_seq_len:
            raise ValueError("prefill recipe exceeds max_seq_len")
    wqkv = resolve_lazy_weight(
        resolved.wqkv,
        device=mesh_device,
        mesh_mapper_config=resolved.wqkv_mesh_mapper_config,
        memory_config=resolved.weight_memory_config,
        layout=resolved.weight_layout,
        dtype=resolved.wqkv_dtype,
    )
    wo = resolve_lazy_weight(
        resolved.wo,
        device=mesh_device,
        mesh_mapper_config=resolved.wo_mesh_mapper_config,
        memory_config=resolved.wo_weight_memory_config,
        layout=resolved.weight_layout,
        dtype=resolved.wo_dtype,
    )
    prefill_wqkv = resolved.prefill_wqkv or wqkv
    prefill_wo = resolved.prefill_wo or wo
    bias = resolved.wqkv_bias
    if bias is not None:
        bias = resolve_lazy_weight(
            bias,
            device=mesh_device,
            mesh_mapper_config=resolved.bias_mesh_mapper_config,
            memory_config=resolved.weight_memory_config,
            layout=resolved.weight_layout,
            dtype=resolved.bias_dtype,
        )
    return replace(
        resolved,
        wqkv=wqkv,
        wo=wo,
        prefill_wqkv=prefill_wqkv,
        prefill_wo=prefill_wo,
        wqkv_bias=bias,
        architecture="wormhole",
        dim=dim,
        qkv_size=qkv_size,
        scale=resolved.scale if resolved.scale is not None else resolved.head_dim**-0.5,
    )


class Attention2D(LightweightModule):
    """Straight-line direct-TTNN decode and prefill attention."""

    def __init__(
        self,
        wqkv: LazyWeight,
        wo: LazyWeight,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        **overrides: Any,
    ) -> None:
        super().__init__()
        self._initialize(
            Attention2DConfig(wqkv, wo, n_heads, n_kv_heads, head_dim, max_batch_size, max_seq_len, **overrides)
        )

    @classmethod
    def from_config(cls, config: Attention2DConfig) -> "Attention2D":
        if not isinstance(config, Attention2DConfig):
            raise TypeError("Attention2D.from_config expects Attention2DConfig")
        instance = object.__new__(cls)
        super(Attention2D, instance).__init__()
        instance._initialize(config)
        return instance

    def _initialize(self, config: Attention2DConfig) -> None:
        self.config = resolve_attention2d_config(config)
        self._loaded_weight_modes: set[str] = set()
        self._runtime_tensors: tuple[Any, Any, Any] | None = None
        self._intermediates: dict[int, Any] = {}
        self._closed = False
        self._kv_cache = self.config.kv_cache
        self._q_norm = RMSNorm2D.from_config(self.config.q_norm_config) if self.config.q_norm_config else None
        self._k_norm = RMSNorm2D.from_config(self.config.k_norm_config) if self.config.k_norm_config else None
        if self._kv_cache is not None:
            _validate_same_mesh("kv_cache", self._kv_cache, self.config.mesh_device)
            self._validate_cache(self._kv_cache)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Attention2D is closed")

    def output_is_borrowed(self, tensor: Any) -> bool:
        """Report whether a returned collective tensor remains owned by its resource provider."""

        return self.config.low_level.is_borrowed_output(tensor)

    @property
    def kv_cache_binding(self) -> KVCacheBinding | None:
        return self._kv_cache

    def bind_kv_cache(self, binding: KVCacheBinding) -> None:
        self._require_open()
        if not isinstance(binding, KVCacheBinding):
            raise TypeError("bind_kv_cache expects KVCacheBinding")
        _validate_same_mesh("kv_cache", binding, self.config.mesh_device)
        self._validate_cache(binding)
        if self._kv_cache is not None and self._kv_cache is not binding:
            raise RuntimeError("KV cache is already bound; unbind it through its owner first")
        self._kv_cache = binding

    def unbind_kv_cache(self, owner: object) -> KVCacheBinding:
        self._require_open()
        if self._kv_cache is None:
            raise RuntimeError("KV cache is not bound")
        if self._kv_cache.owner is not owner:
            raise PermissionError("only the binding owner may unbind the KV cache")
        binding, self._kv_cache = self._kv_cache, None
        return binding

    def _validate_cache(self, binding: KVCacheBinding) -> None:
        keys_shape = _tensor_shape("key cache", binding.keys)
        values_shape = _tensor_shape("value cache", binding.values)
        if keys_shape != values_shape or len(keys_shape) != 4:
            raise ValueError("key and value cache shapes must be identical rank-4 shapes")
        if _tensor_dtype("key cache", binding.keys) != _tensor_dtype("value cache", binding.values):
            raise ValueError("key and value cache dtypes must match")
        local_kv_heads = self.config.n_kv_heads // GALAXY_MESH_SHAPE[0]
        if binding.metadata is None:
            expected = (self.config.users_per_column, local_kv_heads, self.config.max_seq_len, self.config.head_dim)
            if keys_shape != expected:
                raise ValueError(f"contiguous KV cache shape must be {expected}, got {keys_shape}")
            return
        meta = binding.metadata
        expected = (meta.max_num_blocks, local_kv_heads, meta.block_size, self.config.head_dim)
        if keys_shape != expected:
            raise ValueError(f"paged KV cache shape must be {expected}, got {keys_shape}")
        if _tensor_dtype("key cache", binding.keys) != meta.cache_dtype:
            raise ValueError("paged KV cache dtype does not match metadata")
        if meta.max_num_blocks * meta.block_size < self.config.max_seq_len:
            raise ValueError("paged KV cache capacity is smaller than max_seq_len")

    def _validate_page_table(
        self, name: str, table: Any, binding: KVCacheBinding, users: tuple[int, ...], needed: int
    ) -> None:
        """Validate a prefill page table, which carries one row per filled user.

        ``paged_fill_cache`` indexes the table by ``batch_idx``, so the
        device-local table must reach the highest user this request fills.
        """

        shape = self._page_table_prologue(name, table, binding)
        if shape is None:
            return
        if shape[0] <= max(users):
            raise ValueError(f"{name} must have one row for every addressed user")
        self._validate_page_table_capacity(name, table, shape, binding, needed)

    def _validate_decode_page_table(self, table: Any, binding: KVCacheBinding) -> None:
        """Validate the decode page table against the device-local SDPA batch.

        Decode attends to one mesh column's users on each device, so
        ``paged_update_cache`` and ``paged_scaled_dot_product_attention_decode``
        both require the device-local table to carry exactly
        ``users_per_column`` rows — or, when the table is L1-sharded, that batch
        repeated once per core. A table sized to the full physical batch is the
        prefill layout and is rejected here rather than at the first op.
        """

        name = "page_table"
        shape = self._page_table_prologue(name, table, binding)
        if shape is None:
            return
        per_column = self.config.users_per_column
        if shape[0] < per_column or shape[0] % per_column:
            raise ValueError(
                f"decode {name} must carry {per_column} device-local rows (or that batch repeated "
                f"once per core), got {shape}"
            )
        self._validate_page_table_capacity(name, table, shape, binding, self.config.max_seq_len)

    def _page_table_prologue(self, name: str, table: Any, binding: KVCacheBinding) -> tuple[int, ...] | None:
        """Return the table's shape, or ``None`` when the cache is contiguous."""

        if binding.metadata is None:
            if table is not None:
                raise ValueError(f"{name} is only valid with a paged KV cache")
            return None
        if table is None:
            raise ValueError(f"{name} is required with a paged KV cache")
        shape = _tensor_shape(name, table)
        if len(shape) != 2:
            raise ValueError(f"{name} must be a rank-2 [users, blocks] tensor, got {shape}")
        return shape

    def _validate_page_table_capacity(
        self, name: str, table: Any, shape: tuple[int, ...], binding: KVCacheBinding, needed: int
    ) -> None:
        meta = binding.metadata
        required_blocks = (needed + meta.block_size - 1) // meta.block_size
        if shape[1] < required_blocks or shape[1] > meta.max_num_blocks:
            raise ValueError(f"{name} width cannot address the required KV capacity")
        if _tensor_dtype(name, table) != meta.page_table_dtype:
            raise ValueError(f"{name} dtype does not match paged KV metadata")

    def load_device_weights(self, mode: str = "decode") -> None:
        self._require_open()
        if mode not in {"decode", "prefill"}:
            raise ValueError(f"unsupported attention weight mode: {mode}")
        if mode in self._loaded_weight_modes:
            return
        if mode == "decode":
            self.wqkv = self.config.wqkv.get_device_weight()
            self.wo = self.config.wo.get_device_weight()
        else:
            self.prefill_wqkv = self.config.prefill_wqkv.get_device_weight()
            self.prefill_wo = self.config.prefill_wo.get_device_weight()
        self.wqkv_bias = self.config.wqkv_bias.get_device_weight() if self.config.wqkv_bias else None
        self._loaded_weight_modes.add(mode)

    def _require_cache(self) -> KVCacheBinding:
        self._require_open()
        if self._kv_cache is None:
            raise RuntimeError("KV cache must be bound before attention execution")
        return self._kv_cache

    def _ensure_runtime_tensors(self) -> tuple[Any, Any, Any]:
        self._require_open()
        if self._runtime_tensors is None:
            tensors = self.config.runtime_tensor_factory(
                self.config.batch_offsets,
                self.config.prefix_lower_bounds,
                self.config.prefix_upper_bounds,
                self.config.mesh_device,
            )
            if not isinstance(tensors, tuple) or len(tensors) != 3 or any(value is None for value in tensors):
                raise RuntimeError("runtime_tensor_factory must return exactly three tensors")
            self._runtime_tensors = tensors
        return self._runtime_tensors

    def _own(self, value: Any) -> Any:
        if isinstance(value, tuple):
            for item in value:
                self._own(item)
        elif value is not None:
            self._intermediates[id(value)] = value
        return value

    def _release(self, value: Any) -> None:
        if isinstance(value, tuple):
            for item in value:
                self._release(item)
            return
        owned = self._intermediates.pop(id(value), None)
        if owned is not None:
            self.config.intermediate_releaser(owned)

    def _transition(self, old: Any, new: Any, *, borrowed: bool = False) -> Any:
        """Own a stage result and release inputs that are not aliases of it."""

        new_values = new if isinstance(new, tuple) else (new,)
        new_ids = {id(value) for value in new_values if value is not None}
        if not borrowed:
            self._own(new)
        old_values = old if isinstance(old, tuple) else (old,)
        for value in old_values:
            if value is not None and id(value) not in new_ids:
                self._release(value)
        return new

    def _apply_qk_norm(self, q: Any, k: Any, *, mode: str) -> tuple[Any, Any]:
        q_norm, k_norm = self._q_norm, self._k_norm
        if q_norm is None or k_norm is None:
            return q, k

        input_field = f"{mode}_input_memcfg"
        q_target = getattr(getattr(q_norm, "config", None), input_field, None)
        k_target = getattr(getattr(k_norm, "config", None), input_field, None)
        source_q, source_k = q, k
        placed_q = (
            ttnn.to_memory_config(q, q_target) if q_target is not None and _tensor_memory_config(q) != q_target else q
        )
        placed_k = (
            ttnn.to_memory_config(k, k_target) if k_target is not None and _tensor_memory_config(k) != k_target else k
        )
        if placed_q is not source_q:
            self._own(placed_q)
        if placed_k is not source_k:
            self._own(placed_k)

        normalize = "decode_forward" if mode == "decode" else "prefill_forward"
        normalized_q = self._own(getattr(q_norm, normalize)(placed_q))
        normalized_k = self._own(getattr(k_norm, normalize)(placed_k))
        if placed_q is not source_q:
            self._release(placed_q)
        if placed_k is not source_k:
            self._release(placed_k)
        # Create-head outputs can share one allocation. Keep that allocation
        # alive through the sibling V transition instead of force-deallocating Q/K.
        self._disown(source_q)
        self._disown(source_k)
        return normalized_q, normalized_k

    def _disown(self, value: Any) -> None:
        self._intermediates.pop(id(value), None)

    def _release_all_intermediates(self) -> None:
        pending, self._intermediates = tuple(self._intermediates.values()), {}
        for tensor in pending:
            self.config.intermediate_releaser(tensor)

    def _recipe_identity(self, metadata: PrefillMetadata) -> PrefillRecipeIdentity:
        if len(metadata.user_ids) == 1:
            row_mode = PrefillRowMode.SINGLE_ROW
        elif len(metadata.user_ids) == GALAXY_DEVICE_COUNT:
            row_mode = PrefillRowMode.CONCAT_32
        else:
            raise ValueError("prefill recipes support exactly one row or concat-32 users")
        prefix_chunked = any(
            value is not None
            for value in (
                metadata.prefix_user_id,
                metadata.chunk_page_table,
                metadata.chunk_start,
                metadata.chunk_start_tensor,
            )
        )
        return PrefillRecipeIdentity(
            metadata.sequence_length,
            row_mode,
            metadata.collective_mode,
            PrefillAttentionMode.PREFIX_CHUNKED if prefix_chunked else PrefillAttentionMode.REGULAR,
        )

    def _validate_prefill(self, metadata: PrefillMetadata, binding: KVCacheBinding) -> Attention2DSequenceConfig:
        if len(set(metadata.user_ids)) != len(metadata.user_ids):
            raise ValueError("prefill user_ids must be unique")
        if any(user < 0 or user >= self.config.max_batch_size for user in metadata.user_ids):
            raise ValueError("prefill user_ids must be within the physical batch")
        if metadata.prefix_user_id is not None and metadata.prefix_user_id not in metadata.user_ids:
            raise ValueError("prefix_user_id must identify an active prefill user")
        if metadata.chunk_start is not None and metadata.chunk_start_tensor is not None:
            raise ValueError("provide either chunk_start or chunk_start_tensor, not both")
        recipe = self.config.sequence_config(self._recipe_identity(metadata))
        if metadata.chunk_start is not None and (
            metadata.chunk_start < 0 or metadata.chunk_start % recipe.chunk_alignment
        ):
            raise ValueError("chunk_start must be non-negative and aligned to chunk_alignment")
        if recipe.identity.attention_mode is PrefillAttentionMode.PREFIX_CHUNKED:
            self._validate_page_table(
                "page_table", metadata.page_table, binding, metadata.user_ids, self.config.max_seq_len
            )
            fill_table = metadata.chunk_page_table if metadata.chunk_page_table is not None else metadata.page_table
            self._validate_page_table(
                "chunk_page_table", fill_table, binding, metadata.user_ids, metadata.sequence_length
            )
        else:
            self._validate_page_table(
                "page_table", metadata.page_table, binding, metadata.user_ids, metadata.sequence_length
            )
            if metadata.chunk_page_table is not None:
                raise ValueError("chunk_page_table requires a prefix/chunked recipe")
        return recipe

    def decode_forward(self, x: Any, rot_mats: Any, metadata: DecodeMetadata) -> Any:
        self._require_open()
        if not isinstance(metadata, DecodeMetadata):
            raise TypeError("decode_forward requires DecodeMetadata")
        binding = self._require_cache()
        self._validate_decode_page_table(metadata.page_table, binding)
        position_shape = _tensor_shape("current_positions", metadata.current_positions)
        valid_position_shapes = {
            (self.config.users_per_column,),
            (self.config.max_batch_size,),
            (1, 1, 1, self.config.users_per_column),
            (1, 1, 1, self.config.max_batch_size),
        }
        if position_shape not in valid_position_shapes:
            raise ValueError("current_positions must address the complete physical batch")
        _require_activation("decode input", x, self.config.decode_input_placement, self.config.decode_activation_dtype)
        batch_offsets, lower, upper = self._ensure_runtime_tensors()
        self.load_device_weights("decode")
        cfg, low = self.config, self.config.low_level
        prefetch_kwargs = _prefetch_kwargs(cfg.decode_prefetch_context)
        try:
            qkv = self._own(
                ttnn.linear(
                    x,
                    self.wqkv,
                    bias=self.wqkv_bias,
                    program_config=cfg.decode_program_config,
                    compute_kernel_config=cfg.decode_qkv_kernel_config,
                    dtype=cfg.decode_activation_dtype,
                    memory_config=cfg.decode_qkv_output_memory_config,
                    **prefetch_kwargs,
                )
            )
            # WH Galaxy decode is hardware-qualified with the production fused QKV
            # collective. The fallback below is an injectable compatibility path;
            # failed fallback recipes are not evidence of a generic CCL defect.
            if low.reduce_create_qkv_heads is not None:
                heads = self._own(
                    low.reduce_create_qkv_heads(
                        qkv,
                        mode="decode",
                        config=cfg,
                        batch_offsets=batch_offsets,
                    )
                )
                self._release(qkv)
            else:
                reduced = low.reduce_qkv(qkv, mode="decode", config=cfg, batch_offsets=batch_offsets)
                reduced = self._transition(qkv, reduced, borrowed=low.is_borrowed_output(reduced))
                heads = self._own(
                    ttnn.experimental.nlp_create_qkv_heads_decode(
                        reduced,
                        num_heads=cfg.n_heads // GALAXY_MESH_SHAPE[0],
                        num_kv_heads=cfg.n_kv_heads // GALAXY_MESH_SHAPE[0],
                        memory_config=cfg.decode_heads_memory_config,
                    )
                )
                self._release(reduced)
            q, k, v = heads
            q, k = self._apply_qk_norm(q, k, mode="decode")
            q, k = self._transition((q, k), low.rotary(q, k, rot_mats, mode="decode", config=cfg))
            if _tensor_memory_config(k) != cfg.decode_kv_memory_config:
                k = self._transition(k, ttnn.to_memory_config(k, cfg.decode_kv_memory_config))
            if _tensor_memory_config(v) != cfg.decode_kv_memory_config:
                v = self._transition(v, ttnn.to_memory_config(v, cfg.decode_kv_memory_config))
            if binding.metadata:
                ttnn.experimental.paged_update_cache(
                    binding.keys, k, update_idxs_tensor=metadata.current_positions, page_table=metadata.page_table
                )
                ttnn.experimental.paged_update_cache(
                    binding.values, v, update_idxs_tensor=metadata.current_positions, page_table=metadata.page_table
                )
                attention = self._own(
                    ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q,
                        binding.keys,
                        binding.values,
                        page_table_tensor=metadata.page_table,
                        cur_pos_tensor=metadata.current_positions,
                        scale=cfg.scale,
                        program_config=cfg.decode_sdpa_program_config,
                        compute_kernel_config=cfg.decode_sdpa_kernel_config,
                        memory_config=cfg.decode_sdpa_output_memory_config,
                    )
                )
            else:
                ttnn.experimental.paged_update_cache(binding.keys, k, update_idxs_tensor=metadata.current_positions)
                ttnn.experimental.paged_update_cache(binding.values, v, update_idxs_tensor=metadata.current_positions)
                attention = self._own(
                    ttnn.transformer.scaled_dot_product_attention_decode(
                        q,
                        binding.keys,
                        binding.values,
                        cur_pos_tensor=metadata.current_positions,
                        scale=cfg.scale,
                        program_config=cfg.decode_sdpa_program_config,
                        compute_kernel_config=cfg.decode_sdpa_kernel_config,
                        memory_config=cfg.decode_sdpa_output_memory_config,
                    )
                )
            self._release((q, k, v))
            if low.gather_users is not None:
                attention = self._transition(
                    attention,
                    low.gather_users(attention, mode="decode", recipe=None, config=cfg, prefix_bounds=(lower, upper)),
                )
            concat_kwargs = {
                "num_heads": cfg.n_heads // GALAXY_MESH_SHAPE[0],
                "memory_config": cfg.decode_concat_memory_config,
            }
            if cfg.decode_concat_sub_core_grids is not None:
                concat_kwargs["sub_core_grids"] = cfg.decode_concat_sub_core_grids
            concat = self._transition(
                attention,
                ttnn.experimental.nlp_concat_heads_decode(attention, **concat_kwargs),
            )
            if _tensor_memory_config(concat) != cfg.decode_concat_memory_config:
                concat = self._transition(concat, ttnn.to_memory_config(concat, cfg.decode_concat_memory_config))
            gathered = self._transition(
                concat, low.gather_heads(concat, mode="decode", recipe=None, config=cfg, prefix_bounds=(lower, upper))
            )
            projected = self._transition(
                gathered,
                ttnn.linear(
                    gathered,
                    self.wo,
                    program_config=cfg.decode_wo_program_config,
                    compute_kernel_config=cfg.decode_wo_kernel_config,
                    dtype=cfg.decode_activation_dtype,
                    memory_config=cfg.decode_wo_output_memory_config,
                    **prefetch_kwargs,
                ),
            )
            output = low.reduce_output(projected, mode="decode", recipe=None, config=cfg)
            if output is projected:
                self._disown(output)
            else:
                self._release(projected)
            _require_activation("decode output", output, cfg.decode_output_placement, cfg.decode_activation_dtype)
            return output
        finally:
            self._release_all_intermediates()

    def _fill_prefill_cache(self, binding: KVCacheBinding, k: Any, v: Any, metadata: PrefillMetadata) -> None:
        for name, tensor in (
            ("key cache", binding.keys),
            ("value cache", binding.values),
            ("key heads", k),
            ("value heads", v),
        ):
            is_allocated = getattr(tensor, "is_allocated", None)
            if callable(is_allocated) and not is_allocated():
                raise RuntimeError(f"{name} must remain allocated through prefill cache fill")
        if binding.metadata:
            table = metadata.chunk_page_table if metadata.chunk_page_table is not None else metadata.page_table
            if len(metadata.user_ids) > 1:
                for row, user in enumerate(metadata.user_ids):
                    k_user = self._own(k[row : row + 1, :, :, :])
                    v_user = self._own(v[row : row + 1, :, :, :])
                    table_user = self._own(table[user : user + 1, :])
                    ttnn.experimental.paged_fill_cache(binding.keys, k_user, table_user, batch_idx=0)
                    ttnn.experimental.paged_fill_cache(binding.values, v_user, table_user, batch_idx=0)
                    self._release((k_user, v_user, table_user))
                return
            for user in metadata.user_ids:
                ttnn.experimental.paged_fill_cache(binding.keys, k, table, batch_idx=user)
                ttnn.experimental.paged_fill_cache(binding.values, v, table, batch_idx=user)
        else:
            if len(metadata.user_ids) > 1:
                for row, user in enumerate(metadata.user_ids):
                    k_user = self._own(k[row : row + 1, :, :, :])
                    v_user = self._own(v[row : row + 1, :, :, :])
                    ttnn.fill_cache(binding.keys, k_user, user)
                    ttnn.fill_cache(binding.values, v_user, user)
                    self._release((k_user, v_user))
                return
            for user in metadata.user_ids:
                ttnn.fill_cache(binding.keys, k, user)
                ttnn.fill_cache(binding.values, v, user)

    def _sdpa_page_table(self, metadata: PrefillMetadata) -> Any:
        """Return the page table view chunked SDPA can read.

        The prefill page table carries one row per user because
        ``paged_fill_cache`` indexes it by ``batch_idx``. Chunked SDPA instead
        requires the table's leading dimension to equal Q's batch, which for a
        single-row prefill is one, so the addressed user's row is sliced out.
        A concatenated prefill already matches and is passed through.
        """

        table = metadata.page_table
        if len(metadata.user_ids) != 1:
            return table
        rows = _tensor_shape("page_table", table)[0]
        if rows == 1:
            return table
        user = metadata.prefix_user_id if metadata.prefix_user_id is not None else metadata.user_ids[0]
        return self._own(table[user : user + 1, :])

    def prefill_forward(self, x: Any, rot_mats: Any, metadata: PrefillMetadata) -> Any:
        self._require_open()
        if not isinstance(metadata, PrefillMetadata):
            raise TypeError("prefill_forward requires PrefillMetadata")
        binding = self._require_cache()
        recipe = self._validate_prefill(metadata, binding)
        _require_activation("prefill input", x, self.config.prefill_input_placement, recipe.activation_dtype)
        batch_offsets, lower, upper = self._ensure_runtime_tensors()
        self.load_device_weights("prefill")
        cfg, low = self.config, self.config.low_level
        prefetch_kwargs = _prefetch_kwargs(cfg.prefill_prefetch_context)
        try:
            qkv = self._own(
                ttnn.linear(
                    x,
                    self.prefill_wqkv,
                    bias=self.wqkv_bias,
                    program_config=recipe.qkv_program_config,
                    compute_kernel_config=recipe.qkv_kernel_config,
                    dtype=recipe.activation_dtype,
                    memory_config=recipe.qkv_output_memory_config,
                    **prefetch_kwargs,
                )
            )
            reduced = low.reduce_qkv(
                qkv,
                mode="prefill",
                recipe=recipe.identity,
                config=cfg,
                batch_offsets=batch_offsets,
            )
            reduced = self._transition(qkv, reduced, borrowed=low.is_borrowed_output(reduced))
            heads = self._own(
                ttnn.experimental.nlp_create_qkv_heads(
                    reduced,
                    num_heads=cfg.n_heads // GALAXY_MESH_SHAPE[0],
                    num_kv_heads=cfg.n_kv_heads // GALAXY_MESH_SHAPE[0],
                    transpose_k_heads=False,
                    memory_config=recipe.heads_memory_config,
                )
            )
            self._release(reduced)
            q, k, v = heads
            q, k = self._apply_qk_norm(q, k, mode="prefill")
            q, k = self._transition(
                (q, k), low.rotary(q, k, rot_mats, mode="prefill", recipe=recipe.identity, config=cfg)
            )
            cache_dtype = _tensor_dtype("key cache", binding.keys)
            k_needs_transition = (
                _tensor_memory_config(k) != recipe.kv_memory_config or _tensor_dtype("key heads", k) != cache_dtype
            )
            v_needs_transition = (
                _tensor_memory_config(v) != recipe.kv_memory_config or _tensor_dtype("value heads", v) != cache_dtype
            )
            if k_needs_transition or v_needs_transition:
                temporary_stages: list[Any] = []

                def convert_cache_heads(tensor: Any, name: str) -> Any:
                    converted = tensor
                    if _tensor_memory_config(converted) != recipe.kv_memory_config:
                        converted = self._own(
                            ttnn.to_memory_config(
                                converted,
                                recipe.kv_memory_config,
                                dtype=_tensor_dtype(name, converted),
                            )
                        )
                    if _tensor_dtype(name, converted) != cache_dtype:
                        relocated = converted
                        converted = self._own(ttnn.typecast(relocated, dtype=cache_dtype))
                        if relocated is not tensor:
                            temporary_stages.append(relocated)
                    return converted

                converted_k = convert_cache_heads(k, "key heads") if k_needs_transition else k
                converted_v = convert_cache_heads(v, "value heads") if v_needs_transition else v
                k, v = self._transition((k, v), (converted_k, converted_v))
                self._release(tuple(temporary_stages))
            self._fill_prefill_cache(binding, k, v, metadata)
            if recipe.identity.attention_mode is PrefillAttentionMode.PREFIX_CHUNKED:
                kwargs = dict(
                    input_tensor_q=q,
                    input_tensor_k=binding.keys,
                    input_tensor_v=binding.values,
                    page_table_tensor=self._sdpa_page_table(metadata),
                    program_config=recipe.sdpa_program_config,
                    compute_kernel_config=recipe.sdpa_kernel_config,
                    memory_config=recipe.sdpa_output_memory_config,
                )
                if metadata.chunk_start_tensor is not None:
                    kwargs["chunk_start_idx_tensor"] = metadata.chunk_start_tensor
                else:
                    kwargs["chunk_start_idx"] = metadata.chunk_start or 0
                attention = self._own(ttnn.transformer.chunked_scaled_dot_product_attention(**kwargs))
            else:
                attention = self._own(
                    ttnn.transformer.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=True,
                        scale=cfg.scale,
                        program_config=recipe.sdpa_program_config,
                        compute_kernel_config=recipe.sdpa_kernel_config,
                        memory_config=recipe.sdpa_output_memory_config,
                    )
                )
            self._release((q, k, v))
            concat = self._transition(
                attention, ttnn.experimental.nlp_concat_heads(attention, memory_config=recipe.concat_memory_config)
            )
            gathered = self._transition(
                concat,
                low.gather_heads(
                    concat, mode="prefill", recipe=recipe.identity, config=cfg, prefix_bounds=(lower, upper)
                ),
            )
            projected = self._transition(
                gathered,
                ttnn.linear(
                    gathered,
                    self.prefill_wo,
                    program_config=recipe.wo_program_config,
                    compute_kernel_config=recipe.wo_kernel_config,
                    dtype=recipe.activation_dtype,
                    memory_config=recipe.wo_output_memory_config,
                    **prefetch_kwargs,
                ),
            )
            output = low.reduce_output(projected, mode="prefill", recipe=recipe.identity, config=cfg)
            if output is projected:
                self._disown(output)
            else:
                self._release(projected)
            _require_activation("prefill output", output, cfg.prefill_output_placement, recipe.activation_dtype)
            return output
        finally:
            self._release_all_intermediates()

    def forward(self, x: Any, rot_mats: Any, *, mode: str, metadata: DecodeMetadata | PrefillMetadata) -> Any:
        self._require_open()
        if mode == "decode":
            return self.decode_forward(x, rot_mats, metadata)
        if mode == "prefill":
            return self.prefill_forward(x, rot_mats, metadata)
        raise ValueError(f"unsupported attention mode: {mode}")

    def close(self) -> None:
        if self._closed:
            return
        self._release_all_intermediates()
        if self._runtime_tensors is not None:
            for tensor in self._runtime_tensors:
                self.config.runtime_tensor_releaser(tensor)
        self._runtime_tensors = None
        self._closed = True
