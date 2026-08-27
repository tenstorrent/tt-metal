# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTTv2 LM head for the canonical Wormhole Galaxy (8, 4) mesh."""

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.tensor_utils import TILE_SIZE

_GALAXY_MESH_SHAPE = (8, 4)
Collective = Callable[[ttnn.Tensor], ttnn.Tensor]


def _no_sub_device() -> None:
    """Name no sub-device, which is what ``ttnn.linear`` defaults to anyway."""

    return None


@dataclass(frozen=True)
class LMHead2DConfig:
    """Static LM-head projection, placement, and collective policy.

    Each weight source has shape ``[dim, split_vocab]``. Vocabulary is sharded
    over eight mesh rows and hidden reduction over four mesh columns. For more
    than one projection split, each source is packed row-major: all eight row
    shards for that split are contiguous in the source. Local split outputs can
    then be concatenated without changing global vocabulary order. The injected
    collective must complete the column reduction.
    """

    output_weights: Sequence[LazyWeight]
    vocab_size: int
    decode_collective: Collective
    prefill_output_weights: Sequence[LazyWeight] | None = None
    prefill_collective: Collective | None = None
    mesh_device: ttnn.MeshDevice | None = None
    dim: int | None = None
    padded_vocab_size: int | None = None
    max_batch_size: int = 32
    decode_program_configs: Sequence | None = None
    prefill_program_configs: Sequence | None = None
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    decode_weights_memcfgs: Sequence[ttnn.MemoryConfig] | None = None
    prefill_weights_memcfgs: Sequence[ttnn.MemoryConfig] | None = None
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_input_memcfg: ttnn.MemoryConfig | None = None
    decode_output_memcfg: ttnn.MemoryConfig | None = None
    prefill_output_memcfg: ttnn.MemoryConfig | None = None
    decode_output_dtype: ttnn.DataType = ttnn.bfloat8_b
    prefill_output_dtype: ttnn.DataType = ttnn.bfloat8_b
    #: Resolve the mode's worker sub-device at call time. A ``gather_in0``
    #: matmul intersects its ring with ``device->worker_cores(TENSIX,
    #: sub_device_id)`` and, given none, with sub-device **0** -- which under
    #: the Galaxy decode manager is the *prefetch sender* set, disjoint from
    #: the ring. The intersection is then empty and the op dies building its
    #: semaphores:
    #:     TT_FATAL ... Expecting a non-empty CoreRangeSet!
    #: Callables, not values: the id belongs to the live operation-boundary
    #: context, which is created after this config.
    decode_sub_device_id: Callable[[], object] = _no_sub_device
    prefill_sub_device_id: Callable[[], object] = _no_sub_device
    #: Place the invalid-logits mask into the mode's output placement before
    #: adding it. Resolved, never supplied: it is true exactly when that
    #: placement is sharded, because the mask itself is interleaved DRAM.
    decode_stage_mask: bool = False
    prefill_stage_mask: bool = False
    _invalid_logits_mask: LazyWeight | None = None

    def is_resolved(self) -> bool:
        return all(getattr(self, field) is not None for field in self.__dataclass_fields__)


class LMHead2D(LightweightModule):
    """Large output projection with explicit mode-specific lazy weights."""

    def __init__(self, output_weights: Sequence[LazyWeight], vocab_size: int, collective: Collective):
        super().__init__()
        self.config = _resolve_lm_head2d_config(
            LMHead2DConfig(output_weights=output_weights, vocab_size=vocab_size, decode_collective=collective)
        )
        self._initialize_load_state()

    @classmethod
    def from_config(cls, config: LMHead2DConfig):
        instance = object.__new__(cls)
        super(LMHead2D, instance).__init__()
        instance.config = _resolve_lm_head2d_config(config)
        instance._initialize_load_state()
        return instance

    def _initialize_load_state(self) -> None:
        self._decode_weights_loaded = False
        self._prefill_weights_loaded = False
        self._mask_loaded = False

    def load_device_weights(self, mode: str) -> None:
        if mode not in ("decode", "prefill"):
            raise ValueError(f"Unknown mode: {mode!r}; expected 'decode' or 'prefill'")
        loaded_name = f"_{mode}_weights_loaded"
        if not getattr(self, loaded_name):
            weights = getattr(self.config, "output_weights" if mode == "decode" else "prefill_output_weights")
            setattr(
                self,
                "output_weights" if mode == "decode" else "prefill_output_weights",
                [weight.get_device_weight() for weight in weights],
            )
            setattr(self, loaded_name, True)
        if not self._mask_loaded:
            self.invalid_logits_mask = self.config._invalid_logits_mask.get_device_weight()
            self._mask_loaded = True

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        self.load_device_weights("decode")
        x, owns_input = _load_input(x, self.config, self.config.decode_input_memcfg, mode="decode")
        try:
            return self._project(
                x,
                self.output_weights,
                self.config.decode_program_configs,
                self.config.decode_output_dtype,
                self.config.decode_output_memcfg,
                self.config.decode_collective,
                self.config.decode_stage_mask,
                self.config.decode_sub_device_id(),
            )
        finally:
            if owns_input:
                ttnn.deallocate(x)

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        self.load_device_weights("prefill")
        x, owns_input = _load_input(x, self.config, self.config.prefill_input_memcfg, mode="prefill")
        try:
            return self._project(
                x,
                self.prefill_output_weights,
                self.config.prefill_program_configs,
                self.config.prefill_output_dtype,
                self.config.prefill_output_memcfg,
                self.config.prefill_collective,
                self.config.prefill_stage_mask,
                self.config.prefill_sub_device_id(),
            )
        finally:
            if owns_input:
                ttnn.deallocate(x)

    def forward(self, x: ttnn.Tensor | LazyWeight, mode: str) -> ttnn.Tensor:
        if mode == "decode":
            return self.decode_forward(x)
        if mode == "prefill":
            return self.prefill_forward(x)
        raise ValueError(f"Unknown mode: {mode!r}; expected 'decode' or 'prefill'")

    def release(self) -> None:
        failures = []
        seen_weights = set()
        seen_values = set()
        weights = (*self.config.output_weights, *self.config.prefill_output_weights, self.config._invalid_logits_mask)
        for weight in weights:
            if id(weight) in seen_weights:
                continue
            seen_weights.add(id(weight))
            if weight._value is not None:
                value_id = id(weight._value)
                if value_id not in seen_values:
                    seen_values.add(value_id)
                    try:
                        ttnn.deallocate(weight._value)
                    except BaseException as error:
                        failures.append(error)
                weight._value = None
        for name in ("output_weights", "prefill_output_weights", "invalid_logits_mask"):
            if hasattr(self, name):
                delattr(self, name)
        self._initialize_load_state()
        if failures:
            raise failures[0]

    def _project(
        self, x, weights, program_configs, dtype, output_memcfg, collective, stage_mask=False, sub_device_id=None
    ):
        outputs = []
        try:
            for weight, program_config in zip(weights, program_configs):
                partial = ttnn.linear(
                    x,
                    weight,
                    program_config=program_config,
                    compute_kernel_config=self.config.compute_kernel_config,
                    dtype=dtype,
                    memory_config=output_memcfg,
                    sub_device_id=sub_device_id,
                )
                try:
                    reduced = collective(partial)
                    if reduced is partial:
                        raise RuntimeError("LMHead2D collective must return a distinct owned output tensor")
                finally:
                    ttnn.deallocate(partial)
                outputs.append(reduced)
            logits = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=-1, memory_config=output_memcfg)
            mask = self.invalid_logits_mask
            staged_mask = mask
            try:
                # The mask is one module-owned *interleaved* DRAM tensor shared by
                # both modes, but a mode's `output_memcfg` may be sharded - the
                # Galaxy decode LM head lands its logits width-sharded on a
                # 24-core ring, because that is the only placement whose circular
                # buffers fit beside the resident decode activations. Placing the
                # mask into the output's placement first keeps the add on the
                # logits' own cores: `interleaved_to_sharded` runs on its
                # *output* shard's cores, so on a partitioned mesh it stays
                # inside the sub-device the logits already live in, where a mixed
                # sharded/interleaved binary add would not.
                #
                # Whether to do this is decided once, in
                # `_resolve_lm_head2d_config`, and arrives here as a bool. This
                # method must not interrogate `output_memcfg` itself: it is the
                # module's one seam for injecting placements, and its unit tests
                # drive it with opaque sentinels precisely so that the plumbing
                # is testable without a mesh.
                if stage_mask:
                    staged_mask = ttnn.interleaved_to_sharded(mask, output_memcfg)
                return ttnn.add(logits, staged_mask, memory_config=output_memcfg)
            finally:
                if staged_mask is not mask:
                    ttnn.deallocate(staged_mask)
                if len(outputs) > 1:
                    ttnn.deallocate(logits)
        finally:
            for output in outputs:
                ttnn.deallocate(output)


def _resolve_lm_head2d_config(config: LMHead2DConfig) -> LMHead2DConfig:
    if config.vocab_size <= 0:
        raise ValueError(f"LMHead2D vocab_size must be positive, got {config.vocab_size}")
    if not config.output_weights:
        raise ValueError("LMHead2D requires at least one decode output weight")
    if not callable(config.decode_collective):
        raise ValueError("LMHead2D requires a decode collective callable")
    prefill_uses_decode_weights = config.prefill_output_weights is None
    prefill_weights = config.output_weights if prefill_uses_decode_weights else config.prefill_output_weights
    prefill_collective = config.decode_collective if config.prefill_collective is None else config.prefill_collective
    if not callable(prefill_collective):
        raise ValueError("LMHead2D requires a prefill collective callable")

    all_weights = tuple(config.output_weights) + tuple(prefill_weights)
    mesh_device = config.mesh_device or all_weights[0].device or ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("LMHead2D mesh_device must be provided")
    for index, weight in enumerate(all_weights):
        if weight.device is not None and weight.device is not mesh_device:
            raise ValueError(f"LMHead2D weight {index} belongs to a different mesh")
    _validate_galaxy_mesh(mesh_device)
    _validate_collective("decode", config.decode_collective, mesh_device)
    _validate_collective("prefill", prefill_collective, mesh_device)

    decode_shapes = _validate_weight_shapes("decode", config.output_weights, config.dim)
    dim = config.dim or decode_shapes[0][0]
    prefill_shapes = _validate_weight_shapes("prefill", prefill_weights, dim)
    decode_padded_shapes = [_padded_weight_shape(shape) for shape in decode_shapes]
    prefill_padded_shapes = [_padded_weight_shape(shape) for shape in prefill_shapes]
    padded_vocab_size = sum(shape[1] for shape in decode_padded_shapes)
    if sum(shape[1] for shape in prefill_padded_shapes) != padded_vocab_size:
        raise ValueError("LMHead2D decode and prefill weights must cover the same padded vocabulary")
    required_multiple = _GALAXY_MESH_SHAPE[0] * TILE_SIZE
    minimum_padded = ((config.vocab_size + required_multiple - 1) // required_multiple) * required_multiple
    # A *multiple* of the vocabulary-shard tile, not the *minimal* one. This used
    # to demand exactly `minimum_padded`, and that forbade the only padding the
    # decode chain can actually run: `all_reduce_async`'s reduction kernel waits
    # for a full shard on every output core, so the width must be an exact
    # multiple of `cores * shard_width`, and Llama's minimal 128256 gives 501
    # tiles per device - a width no usable core count divides. See D-B19 and
    # `galaxy_padded_vocab_size`, which now pads to a ring-exact width.
    #
    # The bound below still fails closed on a nonsense width: padding may not add
    # a whole extra vocabulary shard per mesh row.
    if padded_vocab_size % required_multiple or padded_vocab_size < minimum_padded:
        raise ValueError(
            f"LMHead2D weights cover padded vocab {padded_vocab_size}; expected a multiple of "
            f"{required_multiple} at least {minimum_padded} for logical vocab {config.vocab_size}"
        )
    if padded_vocab_size >= minimum_padded + required_multiple * _GALAXY_MESH_SHAPE[0]:
        raise ValueError(
            f"LMHead2D padded vocab {padded_vocab_size} pads logical vocab {config.vocab_size} by more than "
            f"one vocabulary shard per mesh row; that is a geometry mistake, not a padding choice"
        )
    _validate_vocab_coverage("decode", decode_shapes, config.vocab_size, padded_vocab_size)
    _validate_vocab_coverage("prefill", prefill_shapes, config.vocab_size, padded_vocab_size)
    if config.padded_vocab_size is not None and config.padded_vocab_size != padded_vocab_size:
        raise ValueError("LMHead2D padded_vocab_size does not match output weights")
    if config.max_batch_size != 32:
        raise ValueError("LMHead2D supports Galaxy physical batch 32 only")
    if config._invalid_logits_mask is not None:
        raise ValueError("LMHead2D invalid-logits mask is module-owned")

    kernel = config.compute_kernel_config or ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1), ttnn.PlacementShard(-2)],
        mesh_shape_override=ttnn.MeshShape(*_GALAXY_MESH_SHAPE),
    )
    decode_memcfgs = tuple(config.decode_weights_memcfgs or [ttnn.DRAM_MEMORY_CONFIG] * len(config.output_weights))
    prefill_memcfgs = tuple(config.prefill_weights_memcfgs or [ttnn.DRAM_MEMORY_CONFIG] * len(prefill_weights))
    if len(decode_memcfgs) != len(config.output_weights) or len(prefill_memcfgs) != len(prefill_weights):
        raise ValueError("LMHead2D requires one weight memory config per split")
    decode_weights = _resolve_weights(config.output_weights, mesh_device, mapper, decode_memcfgs)
    prefill_weights = (
        decode_weights
        if prefill_uses_decode_weights and prefill_memcfgs == decode_memcfgs
        else _resolve_weights(prefill_weights, mesh_device, mapper, prefill_memcfgs)
    )

    # The mask is interleaved DRAM, so a sharded output placement needs it
    # placed first; that decision belongs here, where the real memory configs
    # are, not in `_project`, whose unit tests drive it with sentinels.
    decode_output_memcfg = config.decode_output_memcfg or ttnn.L1_MEMORY_CONFIG
    prefill_output_memcfg = config.prefill_output_memcfg or ttnn.DRAM_MEMORY_CONFIG
    # `padded_vocab_size`, and *not* the width of the output placement. A ring
    # matmul's output shard spec over-covers its tensor -- 24 cores x 672 = 16128
    # columns of spec for a 16032-column tensor -- and the logits keep the logical
    # width, so a mask sized to the placement would be wider than what it is added
    # to. Measured on device: an attempt to widen it here failed the resource
    # lookup at `(1, 1, 32, 16032)`, which is the width TTNN reports.
    mask_source = torch.zeros((1, 1, 1, padded_vocab_size), dtype=torch.float32)
    mask_source[..., config.vocab_size :] = float("-inf")
    mask = _LMHead2DMaskLazyWeight(
        source=mask_source,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1), ttnn.PlacementReplicate()],
            mesh_shape_override=ttnn.MeshShape(*_GALAXY_MESH_SHAPE),
        ),
    )
    resolved = replace(
        config,
        output_weights=decode_weights,
        prefill_output_weights=prefill_weights,
        prefill_collective=prefill_collective,
        mesh_device=mesh_device,
        dim=dim,
        padded_vocab_size=padded_vocab_size,
        decode_program_configs=tuple(config.decode_program_configs or [None] * len(decode_weights)),
        prefill_program_configs=tuple(config.prefill_program_configs or [None] * len(prefill_weights)),
        compute_kernel_config=kernel,
        decode_weights_memcfgs=decode_memcfgs,
        prefill_weights_memcfgs=prefill_memcfgs,
        decode_input_memcfg=config.decode_input_memcfg or ttnn.L1_MEMORY_CONFIG,
        prefill_input_memcfg=config.prefill_input_memcfg or ttnn.DRAM_MEMORY_CONFIG,
        decode_output_memcfg=decode_output_memcfg,
        prefill_output_memcfg=prefill_output_memcfg,
        decode_stage_mask=decode_output_memcfg.is_sharded(),
        prefill_stage_mask=prefill_output_memcfg.is_sharded(),
        _invalid_logits_mask=mask,
    )
    if len(resolved.decode_program_configs) != len(decode_weights):
        raise ValueError("LMHead2D requires one decode program config per split")
    if len(resolved.prefill_program_configs) != len(prefill_weights):
        raise ValueError("LMHead2D requires one prefill program config per split")
    if not resolved.is_resolved():
        raise ValueError("LMHead2D config did not resolve completely")
    return resolved


def _validate_weight_shapes(mode: str, weights: Sequence[LazyWeight], configured_dim: int | None):
    shapes = [tuple(weight.source.shape) for weight in weights]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError(f"LMHead2D {mode} weights must have host shape [dim, split_vocab], got {shapes}")
    dim = configured_dim or shapes[0][0]
    if dim % _GALAXY_MESH_SHAPE[1]:
        raise ValueError(f"LMHead2D dim {dim} must be divisible by 4 Galaxy columns")
    for shape in shapes:
        if shape[0] != dim:
            raise ValueError(f"LMHead2D {mode} weight hidden dimensions do not match {dim}: {shapes}")
        if shape[1] <= 0:
            raise ValueError("LMHead2D vocabulary splits must be non-empty")
    return shapes


def _padded_weight_shape(shape: tuple[int, int]) -> tuple[int, int]:
    hidden_multiple = _GALAXY_MESH_SHAPE[1] * TILE_SIZE
    vocab_multiple = _GALAXY_MESH_SHAPE[0] * TILE_SIZE
    return (
        ((shape[0] + hidden_multiple - 1) // hidden_multiple) * hidden_multiple,
        ((shape[1] + vocab_multiple - 1) // vocab_multiple) * vocab_multiple,
    )


@dataclass
class _LMHead2DLazyWeight(LazyWeight):
    @property
    def padded_shape(self) -> tuple[int, ...]:
        return _padded_weight_shape(tuple(self.source.shape))


@dataclass
class _LMHead2DInputLazyWeight(LazyWeight):
    @property
    def padded_shape(self) -> tuple[int, ...]:
        shape = list(self.source.shape)
        hidden_multiple = _GALAXY_MESH_SHAPE[1] * TILE_SIZE
        shape[-1] = ((shape[-1] + hidden_multiple - 1) // hidden_multiple) * hidden_multiple
        return tuple(shape)


@dataclass
class _LMHead2DMaskLazyWeight(LazyWeight):
    @property
    def padded_shape(self) -> tuple[int, ...]:
        shape = list(self.source.shape)
        vocab_multiple = _GALAXY_MESH_SHAPE[0] * TILE_SIZE
        shape[-1] = ((shape[-1] + vocab_multiple - 1) // vocab_multiple) * vocab_multiple
        return tuple(shape)


def _validate_vocab_coverage(
    mode: str,
    shapes: Sequence[tuple[int, int]],
    vocab_size: int,
    padded_vocab_size: int,
) -> None:
    source_vocab_size = sum(shape[1] for shape in shapes)
    if source_vocab_size not in (vocab_size, padded_vocab_size):
        raise ValueError(
            f"LMHead2D {mode} weights must cover exactly logical vocab {vocab_size} or padded vocab "
            f"{padded_vocab_size}, got {source_vocab_size}"
        )
    vocab_multiple = _GALAXY_MESH_SHAPE[0] * TILE_SIZE
    if any(shape[1] % vocab_multiple for shape in shapes[:-1]):
        raise ValueError(f"LMHead2D {mode} may pad only the final vocabulary split")


def _resolve_weights(weights, mesh_device, mapper, memcfgs):
    return tuple(
        resolve_lazy_weight(
            _LMHead2DLazyWeight(**{field: getattr(weight, field) for field in LazyWeight.__dataclass_fields__}),
            device=mesh_device,
            dtype=weight.dtype or ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memcfg,
            mesh_mapper_config=mapper,
        )
        for weight, memcfg in zip(weights, memcfgs)
    )


def _validate_collective(mode: str, collective: Collective, mesh_device) -> None:
    if getattr(collective, "mesh_device", None) is not mesh_device:
        raise ValueError(f"LMHead2D {mode} collective must declare the resolved mesh_device")
    if getattr(collective, "cluster_axis", None) != 1:
        raise ValueError(f"LMHead2D {mode} collective must reduce Galaxy column axis 1")
    if getattr(collective, "consumes_input", None) is not False:
        raise ValueError(f"LMHead2D {mode} collective must declare consumes_input=False")
    if getattr(collective, "returns_owned_output", None) is not True:
        raise ValueError(f"LMHead2D {mode} collective must declare returns_owned_output=True")


def _load_input(
    x: ttnn.Tensor | LazyWeight, config: LMHead2DConfig, memory_config, *, mode: str
) -> tuple[ttnn.Tensor, bool]:
    if isinstance(x, LazyWeight):
        if x.device is not None and x.device is not config.mesh_device:
            raise ValueError("LMHead2D input belongs to a different mesh")
        _validate_input_shape(x.source.shape, config, mode)
        if x._value is not None:
            raise ValueError("LMHead2D does not accept a materialized LazyWeight input; pass its TTNN tensor instead")
        mapper = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape(*_GALAXY_MESH_SHAPE),
        )
        return (
            resolve_lazy_weight(
                _LMHead2DInputLazyWeight(**{field: getattr(x, field) for field in LazyWeight.__dataclass_fields__}),
                device=config.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                mesh_mapper_config=mapper,
            ).get_device_weight(),
            True,
        )
    if not isinstance(x, ttnn.Tensor):
        raise TypeError("LMHead2D input must be a TTNN tensor or LazyWeight")
    _validate_input_shape(x.shape, config, mode)
    return x, False


def _validate_input_shape(shape, config: LMHead2DConfig, mode: str) -> None:
    shape = tuple(int(value) for value in shape)
    # A host source carries the complete hidden dimension; a device activation
    # produced by the column-sharded residual stream carries its column shard.
    column_local_dim = config.dim // _GALAXY_MESH_SHAPE[1]
    if len(shape) != 4 or shape[-1] not in (config.dim, column_local_dim):
        raise ValueError(
            f"LMHead2D {mode} input must have shape [N, C, S, {config.dim}] or "
            f"[N, C, S, {column_local_dim}], got {shape}"
        )
    if mode == "decode" and shape[-2] != config.max_batch_size:
        raise ValueError(f"LMHead2D decode input requires physical batch {config.max_batch_size}, got {shape[-2]}")


def _validate_galaxy_mesh(mesh_device) -> None:
    if tuple(mesh_device.shape) != _GALAXY_MESH_SHAPE:
        raise ValueError(f"LMHead2D requires logical mesh shape (8, 4), got {tuple(mesh_device.shape)}")
    if mesh_device.get_num_devices() != 32:
        raise ValueError("LMHead2D requires exactly 32 devices")
    if mesh_device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        raise ValueError("LMHead2D supports Wormhole only")
