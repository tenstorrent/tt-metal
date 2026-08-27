# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTTv2 Qwen3-32B reconstruction for the WH Galaxy `(8, 4)` mesh.

Architecture: dense Qwen3 transformer with per-head Q/K RMSNorm, bias-free QKV
projections, and a ``head_dim`` decoupled from the hidden size.

    dim = 5120, layers = 64, n_heads = 64, n_kv_heads = 8, head_dim = 128,
    hidden = 25600, vocab = 151936 (padded 152064), rope_theta = 1000000,
    rms_eps = 1e-6

Because ``n_heads * head_dim`` is 8192 while ``dim`` is 5120, the output
projection reduces 8192 to 5120: the row partition shards 1024 attention
columns per mesh row. Q/K normalization is head-local — one 128-wide norm per
head, no collective — never a distributed hidden-dimension norm.

This package owns its own graph: the checkpoint contract, the precision recipe,
provider conversion, every 2D module config, the decoder layer, and the tensor
model. It borrows only topology-owned, model-neutral machinery from
``models/common/models/galaxy`` — the `(8, 4)` geometry and placement recipes,
the collective-resource plans, the Attention2D/LMHead2D collective adapters, the
prefetch construction policy, and the paged-KV metadata view — plus the reusable
2D modules themselves. It never imports another model-named package.

Ownership order is explicit because the prefetcher is the resource root:

1. resolve geometry and placements;
2. resolve the Galaxy collective-resource policy;
3. build every ``LazyWeight``;
4. materialize and register the prefetched decode weights, then seal;
5. create the Galaxy CCL/subdevice owner over the sealed prefetcher; and
6. assemble the module configs and the tensor model.

Residual convention (identical to the qualified WH Galaxy dataflow): a layer
consumes ``(x, h)`` where ``x`` is the previous stage's contribution and ``h``
is the running residual sum, and returns the new pair. Decode fuses ``h += x``
into the distributed norm; prefill accumulates explicitly in DRAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.galaxy.collectives import (
    GalaxyAttentionCollectives,
    GalaxyColumnAllReduce,
    GalaxyColumnUserSelector,
    deallocate_if_allocated,
    galaxy_runtime_tensor_factory,
)
from models.common.models.galaxy.kv_contract import (
    GalaxyAttentionKVSpec,
    GalaxyPagedAttentionConfig,
    GalaxyPagedKVContract,
)
from models.common.models.galaxy.plans import build_galaxy_resources_config, select_galaxy_resource
from models.common.models.galaxy.prefetch import (
    GALAXY_GLOBAL_CB_SIZE,
    build_galaxy_prefetcher,
    galaxy_dram_prefetch_start,
)
from models.common.models.galaxy.recipes import (
    GALAXY_DEVICE_COUNT,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GalaxyDecodePlacements,
    GalaxyDenseGeometry,
    GalaxyPrefillPlacements,
    compute_kernel_config,
    dram_sharded_weight_memory_config,
    galaxy_padded_vocab_size,
    resolve_galaxy_decode_placements,
    resolve_galaxy_prefill_placements,
    rope_core_grids,
    sampling_core_grids,
    validate_galaxy_mesh,
    worker_cores,
)
from models.common.models.galaxy.resources import create_galaxy_resources
from models.common.modules.attention.attention_2d import (
    Attention2D,
    Attention2DConfig,
    Attention2DSequenceConfig,
    DecodeMetadata,
    KVCacheBinding,
    PrefillAttentionMode,
    PrefillCollectiveMode,
    PrefillMetadata,
    PrefillRecipeIdentity,
    PrefillRowMode,
)
from models.common.modules.embedding.embedding_2d import Embedding2D, Embedding2DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_2d import LMHead2D, LMHead2DConfig
from models.common.modules.mlp.mlp_2d import MLP2D, MLP2DConfig
from models.common.modules.rmsnorm.rmsnorm_2d import (
    RMSNorm2D,
    RMSNorm2DConfig,
    RMSNorm2DGeometry,
    RMSNorm2DResidualPolicy,
)
from models.common.modules.rope.rope_2d import RotarySetup2D, RotarySetup2DConfig
from models.common.modules.sampling.sampling_2d import Sampling2D, Sampling2DConfig

QWEN3_32B_GALAXY_HF_MODEL = "Qwen/Qwen3-32B"
DEFAULT_HF_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"

#: Checkpoint geometry this package accepts, and nothing else.
QWEN3_32B_CHECKPOINT_CONTRACT = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 25600,
    "vocab_size": 151936,
    "head_dim": 128,
}

#: Decode weights streamed by the prefetcher, in the order a layer issues them.
#:
#: **The attention projections are not in this list, and that is a correctness
#: requirement, not a performance choice.** The prefetcher's global circular
#: buffer is received by the 24 ring cores (``galaxy_sender_receiver_mapping``),
#: and a prefetched matmul reads its weight from that buffer in registration
#: order. Only the MLP's three projections run on the ring: `recipes.py` builds
#: `mlp_w1_w3_program_config` / `mlp_w2_program_config` with
#: ``ring_matmul_program_config`` but `attention_qkv_program_config` /
#: `attention_wo_program_config` with ``dense_matmul_program_config``, i.e. a
#: confined worker rectangle (Milestone A limitation L3), and a matmul there
#: cannot take its weight from a buffer whose receivers are the ring.
#:
#: Registering them anyway put two unconsumed entries per layer into the global
#: CB, so every later consumer was shifted by one and the MLP's `w1` read the
#: entry meant for `wqkv`. Measured for Llama on `(8, 4)` as D-B25a: decode
#: attention PCC 0.737 and decode MLP 0.096 against Hugging Face, with the MLP
#: wrong even as a *function* of its own input, and every configuration field
#: correct. This package carried the same defect unchanged.
#:
#: The per-head Q/K norms were never prefetched either: they are 128-element
#: vectors consumed by a head-local norm, not ring matmul operands.
QWEN3_32B_PREFETCHED_WEIGHT_NAMES = ("w1", "w3", "w2")

_MLP_PREFILL_RESHAPE_CUTOFF = 1024


# ============================================================================
# Precision recipe
# ============================================================================


@dataclass(frozen=True)
class Qwen3_32BGalaxyPrecision:
    """Per-model precision and math-fidelity recipe.

    Defaults are the accuracy recipe. They reproduce the dtypes the Milestone A
    WH Galaxy module qualifications used for this geometry, with one Qwen
    departure: the narrow 5120-wide hidden dimension makes the MLP projections
    the accuracy bottleneck, so ``w1``/``w2``/``w3`` stay bfloat16 while the
    attention path stays BFP8. That matches the qualified MLP2D Qwen recipe.
    """

    wqkv_dtype: Any = ttnn.bfloat8_b
    wo_dtype: Any = ttnn.bfloat8_b
    kv_cache_dtype: Any = ttnn.bfloat8_b
    mlp_w1_w3_dtype: Any = ttnn.bfloat16
    mlp_w2_dtype: Any = ttnn.bfloat16
    embedding_dtype: Any = ttnn.bfloat16
    norm_dtype: Any = ttnn.bfloat16
    lm_head_dtype: Any = ttnn.bfloat8_b

    # Activations. The residual stream stays bfloat16 so a 64-layer running sum
    # is never re-quantized; MLP internals and collectives run bfloat8_b.
    decode_activation_dtype: Any = ttnn.bfloat16
    decode_mlp_activation_dtype: Any = ttnn.bfloat8_b
    decode_residual_dtype: Any = ttnn.bfloat16
    prefill_activation_dtype: Any = ttnn.bfloat16
    prefill_mlp_activation_dtype: Any = ttnn.bfloat8_b
    prefill_residual_dtype: Any = ttnn.bfloat16
    attention_collective_dtype: Any = ttnn.bfloat8_b
    # The decode logits, and with them the LM head's column all-reduce buffer.
    # bfloat8_b, not `decode_activation_dtype`, for two reasons that agree:
    #
    #  * it is the qualified precision. The production Galaxy LM head calls
    #    `ttnn.linear(..., dtype=ttnn.bfloat8_b)` and allocates its buffer at
    #    bfloat8_b, and the accuracy gates this milestone reuses were set
    #    against that;
    #  * the reduction buffer is `GALAXY_COLUMNS` times the width of the logits,
    #    so at bfloat16 it is ~96 kB per core and clashes with the ring matmul's
    #    circular buffers on the cores they share:
    #        TT_THROW ... Statically allocated circular buffers in program N
    #        clash with L1 buffers on core range [5-6 - 6-7]
    #
    # The *accumulation* is unaffected: the all-reduce runs `fp32_dest_acc=True`,
    # because a bfloat16 cross-device sum of the logits is order-dependent on ETH
    # ring arrival. Only the stored result is bfloat8_b, exactly as upstream.
    lm_head_output_dtype: Any = ttnn.bfloat8_b

    attention_kernel_config: Any = field(default_factory=compute_kernel_config)
    mlp_ff1_ff3_kernel_config: Any = field(default_factory=lambda: compute_kernel_config(packer_l1_acc=True))
    mlp_ff2_kernel_config: Any = field(default_factory=lambda: compute_kernel_config(packer_l1_acc=True))
    norm_kernel_config: Any = field(default_factory=compute_kernel_config)
    lm_head_kernel_config: Any = field(default_factory=lambda: compute_kernel_config(packer_l1_acc=True))


QWEN3_32B_GALAXY_ACCURACY = Qwen3_32BGalaxyPrecision()

# The performance recipe quantizes the feed-forward projections to BFP8 and
# drops the FF1/FF3 matmuls to LoFi fidelity.
QWEN3_32B_GALAXY_PERFORMANCE = Qwen3_32BGalaxyPrecision(
    mlp_w1_w3_dtype=ttnn.bfloat8_b,
    mlp_w2_dtype=ttnn.bfloat8_b,
    mlp_ff1_ff3_kernel_config=compute_kernel_config(math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True),
)


# ============================================================================
# Provider-neutral host weights
# ============================================================================


@dataclass(frozen=True)
class Qwen3_32BGalaxyLayerWeights:
    """Host tensors for one decoder layer, already in TT layout.

    ``wqkv`` has shape ``[dim, qkv_size]`` with the eight mesh-row shards packed
    contiguously (``[Q_row, K_row, V_row]`` per row). ``wo`` has shape
    ``[n_heads * head_dim, dim]``, which for this model is ``[8192, 5120]``.
    MLP weights are the transposed projections ``[dim, hidden]``,
    ``[hidden, dim]``, ``[dim, hidden]``. ``q_norm`` and ``k_norm`` are
    ``head_dim``-wide per-head vectors.

    ``wqkv_bias`` exists because the provider conversion detects and packs a
    biased checkpoint; Qwen3-32B has none, and lazy-weight resolution rejects
    one rather than silently dropping checkpoint data.
    """

    wqkv: Any
    wo: Any
    w1: Any
    w2: Any
    w3: Any
    attention_norm: Any
    ff_norm: Any
    q_norm: Any
    k_norm: Any
    wqkv_bias: Any = None


@dataclass(frozen=True)
class Qwen3_32BGalaxyWeights:
    """Host tensors for one complete model."""

    embedding: Any
    rope_cos: Any
    rope_sin: Any
    layers: tuple[Qwen3_32BGalaxyLayerWeights, ...]
    final_norm: Any
    lm_head: Any


# ============================================================================
# Model parameters
# ============================================================================


@dataclass(frozen=True)
class Qwen3_32BGalaxyModelParameters:
    """Provider-neutral dimensions for the Galaxy Qwen reconstruction."""

    dim: int = 5120
    n_heads: int = 64
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 25600
    vocab_size: int = 151936
    n_layers: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_batch_size: int = GALAXY_PHYSICAL_BATCH
    max_seq_len: int = 2048
    prefill_sequence_lengths: tuple[int, ...] = (128,)
    #: Per-row lengths served by concatenated physical-batch-32 prefill. Each
    #: entry adds one recipe and one set of collective resources.
    batched_prefill_sequence_lengths: tuple[int, ...] = ()
    #: Sequence lengths that additionally resolve a prefix-cached/chunked recipe.
    chunked_prefill_sequence_lengths: tuple[int, ...] = ()
    qk_norm: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefill_sequence_lengths", tuple(sorted(set(self.prefill_sequence_lengths))))
        object.__setattr__(
            self, "batched_prefill_sequence_lengths", tuple(sorted(set(self.batched_prefill_sequence_lengths)))
        )
        object.__setattr__(
            self, "chunked_prefill_sequence_lengths", tuple(sorted(set(self.chunked_prefill_sequence_lengths)))
        )
        unknown = set(self.chunked_prefill_sequence_lengths) - set(self.prefill_sequence_lengths)
        if unknown:
            raise ValueError(f"chunked prefill lengths must also be plain prefill lengths, got {sorted(unknown)}")
        if not 1 <= self.n_layers <= QWEN3_32B_CHECKPOINT_CONTRACT["num_hidden_layers"]:
            raise ValueError(
                f"n_layers must be in [1, {QWEN3_32B_CHECKPOINT_CONTRACT['num_hidden_layers']}], got {self.n_layers}"
            )

    @property
    def padded_vocab_size(self) -> int:
        return galaxy_padded_vocab_size(self.vocab_size)

    @property
    def attention_dim(self) -> int:
        """Attention projection width, which Qwen3 decouples from ``dim``."""

        return self.n_heads * self.head_dim

    def geometry(self) -> GalaxyDenseGeometry:
        return GalaxyDenseGeometry(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            prefill_sequence_lengths=self.prefill_sequence_lengths,
            batched_prefill_sequence_lengths=self.batched_prefill_sequence_lengths,
        )

    def with_layers(self, n_layers: int) -> "Qwen3_32BGalaxyModelParameters":
        """Return the same parameters for a layer subset, e.g. a one-layer model."""

        return replace(self, n_layers=n_layers)

    def rope_table_len(self) -> int:
        """Return a tile-aligned RoPE table long enough for the served context."""

        return ((max(self.max_seq_len * 2, 8192) + 127) // 128) * 128


def validate_qwen3_32b_checkpoint(hf_config: Any, *, n_layers: int | None = None) -> None:
    """Fail closed unless the checkpoint is exactly Qwen/Qwen3-32B."""

    actual = {name: getattr(hf_config, name, None) for name in QWEN3_32B_CHECKPOINT_CONTRACT}
    mismatches = {
        name: (actual[name], expected)
        for name, expected in QWEN3_32B_CHECKPOINT_CONTRACT.items()
        if actual[name] != expected
    }
    if mismatches:
        raise ValueError(f"Unexpected Qwen3-32B geometry (actual, expected): {mismatches}")
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("Qwen3-32B requires bias-free QKV projections")
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("Qwen3-32B requires an untied LM head")
    if n_layers is not None and not 1 <= n_layers <= int(hf_config.num_hidden_layers):
        raise ValueError(f"n_layers must be in [1, {hf_config.num_hidden_layers}], got {n_layers}")


def parameters_from_hf_config(
    hf_config: Any,
    *,
    n_layers: int | None = None,
    max_batch_size: int = GALAXY_PHYSICAL_BATCH,
    max_seq_len: int = 2048,
    prefill_sequence_lengths: tuple[int, ...] = (128,),
    batched_prefill_sequence_lengths: tuple[int, ...] = (),
    chunked_prefill_sequence_lengths: tuple[int, ...] = (),
) -> Qwen3_32BGalaxyModelParameters:
    """Derive the model parameters from a validated HF config."""

    validate_qwen3_32b_checkpoint(hf_config, n_layers=n_layers)
    return Qwen3_32BGalaxyModelParameters(
        dim=int(hf_config.hidden_size),
        n_heads=int(hf_config.num_attention_heads),
        n_kv_heads=int(hf_config.num_key_value_heads),
        head_dim=int(hf_config.head_dim),
        hidden_dim=int(hf_config.intermediate_size),
        vocab_size=int(hf_config.vocab_size),
        n_layers=int(hf_config.num_hidden_layers if n_layers is None else n_layers),
        rms_norm_eps=float(hf_config.rms_norm_eps),
        rope_theta=float(getattr(hf_config, "rope_theta", 1000000.0)),
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        prefill_sequence_lengths=prefill_sequence_lengths,
        batched_prefill_sequence_lengths=batched_prefill_sequence_lengths,
        chunked_prefill_sequence_lengths=chunked_prefill_sequence_lengths,
    )


def default_paged_attention_config(
    params: Qwen3_32BGalaxyModelParameters, *, block_size: int = 32
) -> GalaxyPagedAttentionConfig:
    """Return a paged geometry covering the physical batch at max sequence."""

    blocks_per_user = (params.max_seq_len + block_size - 1) // block_size
    return GalaxyPagedAttentionConfig(
        block_size=block_size,
        max_num_blocks=blocks_per_user * params.max_batch_size,
    )


# ============================================================================
# Lazy weights
# ============================================================================


@dataclass(frozen=True)
class Qwen3_32BGalaxyLazyLayerWeights:
    """Resolved but unmaterialized device weights for one layer.

    Decode and prefill keep distinct materializations: decode uses the DRAM
    width-sharded ring layout the gather-in0 matmuls require, prefill uses
    interleaved DRAM. The per-head Q/K norms are mode-neutral.
    """

    wqkv: LazyWeight
    wo: LazyWeight
    w1: LazyWeight
    w2: LazyWeight
    w3: LazyWeight
    prefill_wqkv: LazyWeight
    prefill_wo: LazyWeight
    prefill_w1: LazyWeight
    prefill_w2: LazyWeight
    prefill_w3: LazyWeight
    attention_norm: LazyWeight
    ff_norm: LazyWeight
    q_norm: LazyWeight
    k_norm: LazyWeight

    def prefetched(self) -> tuple[LazyWeight, ...]:
        return tuple(getattr(self, name) for name in QWEN3_32B_PREFETCHED_WEIGHT_NAMES)


@dataclass(frozen=True)
class Qwen3_32BGalaxyLazyWeights:
    """Resolved but unmaterialized device weights for one complete model."""

    embedding: LazyWeight
    rope_cos: LazyWeight
    rope_sin: LazyWeight
    layers: tuple[Qwen3_32BGalaxyLazyLayerWeights, ...]
    final_norm: LazyWeight
    lm_head: LazyWeight

    def prefetch_registration(self) -> tuple[tuple[str, LazyWeight], ...]:
        """Return the decode weights to register, in per-layer issue order."""

        return tuple(
            (f"layer[{index}].{name}", weight)
            for index, layer in enumerate(self.layers)
            for name, weight in zip(QWEN3_32B_PREFETCHED_WEIGHT_NAMES, layer.prefetched())
        )


def _lazy(
    source: Any,
    *,
    mesh_device: Any,
    dtype: Any,
    mesh_mapper_config: Any = None,
    memory_config: Any = ttnn.DRAM_MEMORY_CONFIG,
    layout: Any = ttnn.TILE_LAYOUT,
    cache: tuple[Path, str] | None = None,
) -> LazyWeight:
    return LazyWeight(
        source=source,
        device=mesh_device,
        dtype=dtype,
        mesh_mapper_config=mesh_mapper_config,
        memory_config=memory_config,
        layout=layout,
        cache_dir_weight_name=cache,
    )


def _mesh_mapper(*placements: Any) -> ttnn.MeshMapperConfig:
    return ttnn.MeshMapperConfig(placements=list(placements), mesh_shape_override=ttnn.MeshShape(*GALAXY_MESH_SHAPE))


class _UnprefetchedContext:
    """A prefetch context that names the worker sub-device but no global CB.

    `Attention2D` reads `global_cb` and `worker_sub_device_id` off its
    `decode_prefetch_context` at every call. The confined attention decode
    matmuls must still be told their sub-device - without it a ttnn matmul
    defaults to sub-device *zero*, the prefetch senders (D-B13) - but they must
    **not** be handed a global circular buffer they cannot receive from. See
    `QWEN3_32B_PREFETCHED_WEIGHT_NAMES`.
    """

    def __init__(self, context: Any):
        self._context = context

    @property
    def global_cb(self) -> None:
        return None

    @property
    def worker_sub_device_id(self) -> Any:
        return self._context.worker_sub_device_id

    @property
    def sub_device_id(self) -> Any:
        return self._context.worker_sub_device_id

    @property
    def mesh_device(self) -> Any:
        return self._context.mesh_device

    @property
    def mode(self) -> Any:
        return getattr(self._context, "mode", None)


def _row_output_mapper() -> ttnn.MeshMapperConfig:
    """Rows shard the output dimension, columns shard the reduced dimension."""

    return _mesh_mapper(ttnn.PlacementShard(-1), ttnn.PlacementShard(-2))


def _row_reduction_mapper() -> ttnn.MeshMapperConfig:
    """Rows shard the reduced dimension, columns shard the output dimension."""

    return _mesh_mapper(ttnn.PlacementShard(-2), ttnn.PlacementShard(-1))


def _reject_qkv_bias(index: int, bias: Any) -> None:
    """Reject a fused QKV bias until Attention2D can place one explicitly."""

    if bias is not None:
        raise ValueError(
            f"layer {index} carries a fused QKV bias, which the Galaxy Qwen path does not support yet: "
            "Attention2D validates the bias against the projection's DRAM-sharded weight placement"
        )


def build_qwen3_32b_galaxy_lazy_weights(
    *,
    mesh_device: Any,
    geometry: GalaxyDenseGeometry,
    precision: Qwen3_32BGalaxyPrecision,
    weights: Qwen3_32BGalaxyWeights,
    cache_path: Path | None = None,
) -> Qwen3_32BGalaxyLazyWeights:
    """Resolve every device weight placement without materializing anything."""

    validate_galaxy_mesh("Galaxy Qwen weights", mesh_device)
    wqkv_memcfg = dram_sharded_weight_memory_config(mesh_device, geometry.local_dim, geometry.local_qkv_size)
    wo_memcfg = dram_sharded_weight_memory_config(mesh_device, geometry.local_attention_dim, geometry.local_dim)
    w1_w3_memcfg = dram_sharded_weight_memory_config(mesh_device, geometry.local_dim, geometry.local_hidden_dim)
    w2_memcfg = dram_sharded_weight_memory_config(mesh_device, geometry.local_hidden_dim, geometry.local_dim)
    row_output, row_reduction = _row_output_mapper(), _row_reduction_mapper()

    def cache(index: int, kind: str, name: str) -> tuple[Path, str] | None:
        return (cache_path / kind, f"layer{index}_{name}") if cache_path else None

    def layer_weights(index: int, layer: Qwen3_32BGalaxyLayerWeights) -> Qwen3_32BGalaxyLazyLayerWeights:
        # Neither the checkpoint nor Attention2D supports a fused QKV bias, and
        # the module validates a bias against the projection's own DRAM-sharded
        # placement, which a bias vector cannot satisfy. Fail loudly rather than
        # silently dropping checkpoint data.
        _reject_qkv_bias(index, layer.wqkv_bias)
        # One row per projection: (name, host tensor, cache kind, dtype, mesh
        # mapper, decode ring placement). Decode and prefill share the dtype and
        # the mapper by construction and differ only in placement.
        projections = (
            ("wqkv", layer.wqkv, "attn", precision.wqkv_dtype, row_output, wqkv_memcfg),
            ("wo", layer.wo, "attn", precision.wo_dtype, row_reduction, wo_memcfg),
            ("w1", layer.w1, "mlp", precision.mlp_w1_w3_dtype, row_output, w1_w3_memcfg),
            ("w2", layer.w2, "mlp", precision.mlp_w2_dtype, row_reduction, w2_memcfg),
            ("w3", layer.w3, "mlp", precision.mlp_w1_w3_dtype, row_output, w1_w3_memcfg),
        )
        matrices: dict[str, LazyWeight] = {}
        for name, source, kind, dtype, mapper, ring_memory_config in projections:
            matrices[name] = _lazy(
                source,
                mesh_device=mesh_device,
                dtype=dtype,
                mesh_mapper_config=mapper,
                memory_config=ring_memory_config,
                cache=cache(index, kind, f"{name}_ring"),
            )
            matrices[f"prefill_{name}"] = _lazy(
                source,
                mesh_device=mesh_device,
                dtype=dtype,
                mesh_mapper_config=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache=cache(index, kind, name),
            )

        def norm(source: Any, name: str) -> LazyWeight:
            return _lazy(
                source,
                mesh_device=mesh_device,
                dtype=precision.norm_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache=cache(index, "norm", name),
            )

        def head_local_norm(source: Any, name: str) -> LazyWeight:
            if source is None:
                raise ValueError(f"layer {index} is missing the per-head {name} weight Qwen3-32B requires")
            return norm(source, name)

        return Qwen3_32BGalaxyLazyLayerWeights(
            **matrices,
            attention_norm=norm(layer.attention_norm, "attention_norm"),
            ff_norm=norm(layer.ff_norm, "ff_norm"),
            q_norm=head_local_norm(layer.q_norm, "q_norm"),
            k_norm=head_local_norm(layer.k_norm, "k_norm"),
        )

    return Qwen3_32BGalaxyLazyWeights(
        embedding=_lazy(
            weights.embedding,
            mesh_device=mesh_device,
            dtype=precision.embedding_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
        ),
        rope_cos=_lazy(
            weights.rope_cos,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache=(cache_path / "rope", "cos") if cache_path else None,
        ),
        rope_sin=_lazy(
            weights.rope_sin,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache=(cache_path / "rope", "sin") if cache_path else None,
        ),
        layers=tuple(layer_weights(index, layer) for index, layer in enumerate(weights.layers)),
        final_norm=_lazy(
            weights.final_norm,
            mesh_device=mesh_device,
            dtype=precision.norm_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache=(cache_path / "norm", "final_norm") if cache_path else None,
        ),
        lm_head=_lazy(
            weights.lm_head,
            mesh_device=mesh_device,
            dtype=precision.lm_head_dtype,
            mesh_mapper_config=_row_output_mapper(),
            cache=(cache_path / "lm_head", "output") if cache_path else None,
        ),
    )


# ============================================================================
# Module configs
# ============================================================================


@dataclass(frozen=True)
class Qwen3_32BGalaxyBlockConfig:
    """Immutable configuration for one Galaxy Qwen decoder layer."""

    attention_norm_config: RMSNorm2DConfig
    attention_config: Attention2DConfig
    ff_norm_config: RMSNorm2DConfig
    mlp_config: MLP2DConfig
    kv_spec: GalaxyAttentionKVSpec
    decode_attention_input_memcfg: Any
    decode_mlp_input_memcfg: Any
    decode_mlp_input_dtype: Any
    decode_residual_memcfg: Any
    prefill_attention_input_memcfg: Any
    prefill_mlp_input_memcfg: Any
    prefill_mlp_input_dtype: Any
    prefill_residual_memcfg: Any
    prefill_residual_dtype: Any


@dataclass(frozen=True)
class Qwen3_32BGalaxyTransformer2DConfig:
    """Complete immutable configuration for the Galaxy Qwen tensor model."""

    geometry: GalaxyDenseGeometry
    mesh_device: Any
    resources: Any
    prefetcher: Any
    embedding_config: Embedding2DConfig
    rope_config: RotarySetup2DConfig
    block_configs: tuple[Qwen3_32BGalaxyBlockConfig, ...]
    norm_config: RMSNorm2DConfig
    lm_head_config: LMHead2DConfig
    attention_collectives: GalaxyAttentionCollectives
    decode_placements: GalaxyDecodePlacements
    prefill_placements: GalaxyPrefillPlacements
    sampling_config: Sampling2DConfig | None = None
    cache_path: str | None = None
    owns_shared_resources: bool = False

    @property
    def n_layers(self) -> int:
        return len(self.block_configs)

    @property
    def num_devices(self) -> int:
        return GALAXY_DEVICE_COUNT

    @property
    def vocab_size(self) -> int:
        return self.geometry.vocab_size

    @property
    def max_batch_size(self) -> int:
        return self.geometry.max_batch_size

    @property
    def max_seq_len(self) -> int:
        return self.geometry.max_seq_len

    @property
    def dim(self) -> int:
        return self.geometry.dim


def _norm_config(
    weight: LazyWeight,
    *,
    mesh_device: Any,
    geometry: GalaxyDenseGeometry,
    precision: Qwen3_32BGalaxyPrecision,
    resources: Any,
    prefetch_contexts: tuple[Any, Any],
    decode_placements: GalaxyDecodePlacements,
    eps: float,
    residual_policy: RMSNorm2DResidualPolicy,
) -> RMSNorm2DConfig:
    prefill_context, decode_context = prefetch_contexts
    return RMSNorm2DConfig(
        weight=weight,
        cluster_shape=GALAXY_MESH_SHAPE,
        eps=eps,
        residual_policy=residual_policy,
        geometry=RMSNorm2DGeometry.DISTRIBUTED,
        mesh_device=mesh_device,
        tt_ccl=resources.ccl,
        collective_resource_selector=select_galaxy_resource,
        decode_prefetch_context=decode_context,
        prefill_prefetch_context=prefill_context,
        max_batch_size=geometry.max_batch_size,
        decode_input_memcfg=decode_placements.residual_memcfg,
        decode_residual_memcfg=decode_placements.residual_memcfg,
        decode_output_memcfg=decode_placements.residual_memcfg,
        # decode_stats_memcfg is deliberately not passed: RMSNorm2D resolves the
        # fused-statistics placement from decode_input_memcfg, and only that
        # placement satisfies its own _require_fused_stats_placement check (D1).
        prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config_prefill=precision.norm_kernel_config,
    )


#: How many worker cores the head-local Q/K decode norm runs its kernel on. The
#: tensor it normalizes is ``users_per_column`` users of one *tile* of padded
#: heads - 8 x 32 = 256 rows on this mesh, for the 8 local Q heads and the 1
#: local K head alike - so eight cores is one tile row each. `RMSNorm2D` derives
#: the shard shape from the tensor and will refuse a count that does not divide
#: it into whole tiles.
_HEAD_LOCAL_DECODE_NORM_CORES = 8


def _head_local_decode_norm_cores() -> Any:
    """Return the worker cores a head-local decode norm runs its kernel on.

    One core wide, ``_HEAD_LOCAL_DECODE_NORM_CORES`` tall, taken from the decode
    worker envelope's first column. Three properties are load-bearing, and each
    is something `ttnn.rms_norm` refuses without:

    * **worker cores**, because an interleaved head-local norm spreads over the
      whole compute grid and aborts under the decode sub-device manager
      (D-B26);
    * **a rectangle**, which the sharded layernorm requires outright
      (``Sharded layernorm does not support non-rectangular core grids``);
    * **one core wide**, so the full ``head_dim``-wide row lands on a single
      core and the reduction needs no multicast.

    `worker_cores()` is ``{[1-0 - 3-9], [5-0 - 6-9]}`` on this mesh, so
    ``x = 1, y = 0..7`` is worker-owned and disjoint from the prefetch senders.
    """

    workers = worker_cores()
    origin = workers.bounding_box().start
    core_range = ttnn.CoreRange(
        ttnn.CoreCoord(origin.x, origin.y),
        ttnn.CoreCoord(origin.x, origin.y + _HEAD_LOCAL_DECODE_NORM_CORES - 1),
    )
    # The envelope is not contiguous - the `x = 4` prefetch sender column splits
    # it - and its bounding box therefore includes cores no sub-device owns, so
    # membership is checked core by core rather than taken from the box.
    for y in range(core_range.start.y, core_range.end.y + 1):
        if not workers.contains(ttnn.CoreCoord(core_range.start.x, y)):
            raise ValueError(f"head-local decode norm core ({core_range.start.x}, {y}) is not a worker core")
    return ttnn.CoreRangeSet({core_range})


def _head_local_norm_config(
    weight: LazyWeight,
    *,
    mesh_device: Any,
    precision: Qwen3_32BGalaxyPrecision,
    decode_placements: GalaxyDecodePlacements,
    eps: float,
) -> RMSNorm2DConfig:
    """Return the per-head Q/K norm config `Attention2D` requires.

    Qwen3 normalizes each ``head_dim``-wide head independently, so there is no
    column reduction and no collective. ``Attention2D`` rejects any other
    geometry, and it also rejects a weight whose width is not ``head_dim``.

    **Decode must name the created heads' own placement, not interleaved DRAM.**
    Measured on `(8, 4)` (defect D-B26, this job's run `a2_03_qknorm`): with
    ``ttnn.DRAM_MEMORY_CONFIG`` the prefill norm is correct at PCC 0.99998 on all
    32 devices and the *decode* norm aborts before producing any number at all -

        TT_FATAL: Kernel group cores do not match sub device cores for
                  programmable core type TENSIX
        program.cpp:2205: num_intersections == num_cores

    - because an interleaved ``ttnn.rms_norm`` resolves
    ``LayerNormDefaultProgramConfig``, which splits its rows over
    ``device->compute_with_storage_grid_size()``: the whole compute grid,
    including the prefetch sender columns the loaded decode manager does not
    own. This is the unresolved half of Milestone A's D2, whose own defect was
    that head-local decode aborted in op validation before producing a
    numerical result; the module's D2 fix made interleaved DRAM the *default*
    for this geometry, which is right for prefill and unplaceable for decode.

    The answer is that decode names **no** placement at all and instead names
    the cores its kernel may run on. ``Attention2D`` relocates the created heads
    to ``decode_input_memcfg`` before calling, and Q and K arrive on *disjoint*
    core sets - ``nlp_create_qkv_heads_decode`` gives Q the first ``batch`` cores
    of the head grid and K the next ``batch`` - which the fused QK rotary
    downstream requires (``Q and K must not overlap``). Naming any single
    placement, ``attention_heads_memcfg`` included, relocates both onto it and
    destroys that. Leaving it unset makes the relocation a no-op, and
    ``RMSNorm2D`` puts each tensor back exactly where it found it. The kernel
    itself runs in a third, block-sharded placement over the named worker cores,
    because the created heads are HEIGHT_SHARDED and ``ttnn.rms_norm`` rejects
    that layout outright (``layernorm_device_operation.cpp:166``, a standing
    TODO). See ``RMSNorm2D._decode_head_local``.

    Prefill keeps interleaved DRAM: its mode plan is a single sub-device over the
    full grid, its heads are ``[1, local_heads, sequence, head_dim]`` in DRAM
    already, and it is qualified there at PCC >= 0.9999.
    """

    return RMSNorm2DConfig(
        weight=weight,
        cluster_shape=GALAXY_MESH_SHAPE,
        eps=eps,
        geometry=RMSNorm2DGeometry.HEAD_LOCAL,
        mesh_device=mesh_device,
        # Decode names no input or output placement, and that is the point: see
        # the docstring above and `RMSNorm2D._decode_head_local`. Q and K arrive
        # on disjoint core sets and must leave on the same ones.
        prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        decode_compute_cores=_head_local_decode_norm_cores(),
        compute_kernel_config_prefill=precision.norm_kernel_config,
    )


def _attention_sequence_configs(
    geometry: GalaxyDenseGeometry,
    precision: Qwen3_32BGalaxyPrecision,
    prefill: GalaxyPrefillPlacements,
    chunked_lengths: tuple[int, ...] = (),
) -> dict[PrefillRecipeIdentity, Attention2DSequenceConfig]:
    """Resolve one frozen recipe per prefill shape this model may be asked for.

    Three families share the interleaved DRAM placements and differ only in the
    program configs their geometry requires:

    - single-row prefill, one user per request;
    - prefix-cached/chunked single-row prefill, whose SDPA reads the paged cache
      and is therefore chunk-aligned rather than sequence-length tuned;
    - concatenated physical-batch-32 prefill, whose projections see 32 rows of
      tokens at once while SDPA still runs one causal sequence per row.
    """

    recipes: dict[PrefillRecipeIdentity, Attention2DSequenceConfig] = {}

    def add(identity: PrefillRecipeIdentity, qkv: Any, sdpa: Any, wo: Any) -> None:
        recipes[identity] = Attention2DSequenceConfig(
            identity=identity,
            qkv_program_config=qkv,
            sdpa_program_config=sdpa,
            wo_program_config=wo,
            qkv_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            heads_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            kv_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sdpa_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            concat_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            wo_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            qkv_kernel_config=precision.attention_kernel_config,
            sdpa_kernel_config=precision.attention_kernel_config,
            wo_kernel_config=precision.attention_kernel_config,
            activation_dtype=precision.prefill_activation_dtype,
            chunk_alignment=geometry.chunk_alignment,
        )

    for length in geometry.prefill_sequence_lengths:
        add(
            PrefillRecipeIdentity(
                length, PrefillRowMode.SINGLE_ROW, PrefillCollectiveMode.REGULAR, PrefillAttentionMode.REGULAR
            ),
            prefill.attention_program_configs[length],
            prefill.attention_sdpa_program_configs[length],
            prefill.attention_wo_program_configs[length],
        )
    for length in chunked_lengths:
        add(
            PrefillRecipeIdentity(
                length, PrefillRowMode.SINGLE_ROW, PrefillCollectiveMode.REGULAR, PrefillAttentionMode.PREFIX_CHUNKED
            ),
            prefill.attention_program_configs[length],
            prefill.chunked_sdpa_program_config,
            prefill.attention_wo_program_configs[length],
        )
    for length in geometry.batched_prefill_sequence_lengths:
        add(
            PrefillRecipeIdentity(
                length, PrefillRowMode.CONCAT_32, PrefillCollectiveMode.REGULAR, PrefillAttentionMode.REGULAR
            ),
            prefill.batched_attention_program_configs[length],
            prefill.batched_attention_sdpa_program_configs[length],
            prefill.batched_attention_wo_program_configs[length],
        )
    return recipes


def _build_block_config(
    index: int,
    lazy: Qwen3_32BGalaxyLazyLayerWeights,
    *,
    geometry: GalaxyDenseGeometry,
    precision: Qwen3_32BGalaxyPrecision,
    mesh_device: Any,
    resources: Any,
    prefetch_contexts: tuple[Any, Any],
    decode: GalaxyDecodePlacements,
    prefill: GalaxyPrefillPlacements,
    collectives: GalaxyAttentionCollectives,
    norm_eps: float,
    paged_attention_config: GalaxyPagedAttentionConfig | None,
    chunked_lengths: tuple[int, ...] = (),
) -> Qwen3_32BGalaxyBlockConfig:
    prefill_context, decode_context = prefetch_contexts
    attention_config = Attention2DConfig(
        wqkv=lazy.wqkv,
        wo=lazy.wo,
        n_heads=geometry.n_heads,
        n_kv_heads=geometry.n_kv_heads,
        head_dim=geometry.head_dim,
        max_batch_size=geometry.max_batch_size,
        max_seq_len=geometry.max_seq_len,
        low_level=collectives.callables(),
        runtime_tensor_factory=galaxy_runtime_tensor_factory,
        runtime_tensor_releaser=deallocate_if_allocated,
        # Qwen3 has no QKV bias, and normalizes Q and K per head.
        prefill_wqkv=lazy.prefill_wqkv,
        prefill_wo=lazy.prefill_wo,
        q_norm_config=_head_local_norm_config(
            lazy.q_norm,
            mesh_device=mesh_device,
            precision=precision,
            decode_placements=decode,
            eps=norm_eps,
        ),
        k_norm_config=_head_local_norm_config(
            lazy.k_norm,
            mesh_device=mesh_device,
            precision=precision,
            decode_placements=decode,
            eps=norm_eps,
        ),
        mesh_device=mesh_device,
        architecture=mesh_device.arch(),
        dim=geometry.dim,
        users_per_column=geometry.users_per_column,
        wqkv_mesh_mapper_config=lazy.wqkv.mesh_mapper_config,
        wo_mesh_mapper_config=lazy.wo.mesh_mapper_config,
        weight_memory_config=lazy.wqkv.memory_config,
        wo_weight_memory_config=lazy.wo.memory_config,
        weight_layout=ttnn.TILE_LAYOUT,
        wqkv_dtype=precision.wqkv_dtype,
        wo_dtype=precision.wo_dtype,
        decode_input_placement=decode.attention_input_memcfg,
        decode_output_placement=decode.residual_memcfg,
        prefill_input_placement=ttnn.DRAM_MEMORY_CONFIG,
        prefill_output_placement=ttnn.DRAM_MEMORY_CONFIG,
        decode_qkv_output_memory_config=decode.attention_qkv_output_memcfg,
        decode_heads_memory_config=decode.attention_heads_memcfg,
        decode_kv_memory_config=decode.attention_kv_memcfg,
        decode_sdpa_output_memory_config=decode.attention_sdpa_output_memcfg,
        decode_concat_memory_config=decode.attention_concat_memcfg,
        decode_concat_sub_core_grids=decode.attention_gather_users_memcfg.shard_spec.grid,
        decode_wo_output_memory_config=decode.attention_wo_output_memcfg,
        decode_program_config=decode.attention_qkv_program_config,
        decode_sdpa_program_config=decode.attention_sdpa_program_config,
        decode_wo_program_config=decode.attention_wo_program_config,
        decode_qkv_kernel_config=precision.attention_kernel_config,
        decode_sdpa_kernel_config=precision.attention_kernel_config,
        decode_wo_kernel_config=precision.attention_kernel_config,
        decode_activation_dtype=precision.decode_activation_dtype,
        prefill_sequence_configs=_attention_sequence_configs(geometry, precision, prefill, chunked_lengths),
        # No global circular buffer for the confined attention decode matmuls -
        # they do not run on the ring that receives it - but still the worker
        # sub-device id, without which ttnn defaults to the prefetch senders.
        decode_prefetch_context=_UnprefetchedContext(decode_context),
        prefill_prefetch_context=prefill_context,
        intermediate_releaser=deallocate_if_allocated,
    )
    mlp_config = MLP2DConfig(
        w1=lazy.w1,
        w2=lazy.w2,
        w3=lazy.w3,
        prefill_w1=lazy.prefill_w1,
        prefill_w2=lazy.prefill_w2,
        prefill_w3=lazy.prefill_w3,
        mesh_device=mesh_device,
        tt_ccl=resources.ccl,
        collective_resource_selector=select_galaxy_resource,
        decode_prefetch_context=decode_context,
        prefill_prefetch_context=prefill_context,
        dim=geometry.dim,
        hidden_dim=geometry.hidden_dim,
        max_batch_size=geometry.max_batch_size,
        w1_w3_memcfg=lazy.w1.memory_config,
        w2_memcfg=lazy.w2.memory_config,
        decode_input_memcfg=decode.mlp_input_memcfg,
        decode_w2_input_memcfg=decode.mlp_w2_input_memcfg,
        decode_w1_w3_prg_config=decode.mlp_w1_w3_program_config,
        decode_w2_prg_config=decode.mlp_w2_program_config,
        decode_w1_w3_output_memcfg=decode.mlp_w1_w3_output_memcfg,
        decode_w2_output_memcfg=decode.mlp_w2_output_memcfg,
        ff1_out_reduce_scatter_memcfg=decode.mlp_reduce_scatter_memcfg,
        ff2_out_reduce_scatter_memcfg=decode.residual_memcfg,
        sharded_attn_input_memcfg=decode.residual_memcfg,
        prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_w1_w3_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_w2_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        w1_w3_dtype=precision.mlp_w1_w3_dtype,
        w2_dtype=precision.mlp_w2_dtype,
        decode_activation_dtype=precision.decode_mlp_activation_dtype,
        # The final decode collective produces the residual contribution, so it
        # keeps the residual stream's dtype.
        decode_ccl_dtype=precision.decode_residual_dtype,
        decode_mul_dtype=precision.decode_mlp_activation_dtype,
        prefill_activation_dtype=precision.prefill_mlp_activation_dtype,
        prefill_ccl_dtype=precision.prefill_residual_dtype,
        prefill_mul_dtype=precision.prefill_mlp_activation_dtype,
        ff1_3_compute_kernel_cfg=precision.mlp_ff1_ff3_kernel_config,
        ff2_compute_kernel_cfg=precision.mlp_ff2_kernel_config,
        prefill_len_cutoff=_MLP_PREFILL_RESHAPE_CUTOFF,
    )
    return Qwen3_32BGalaxyBlockConfig(
        attention_norm_config=_norm_config(
            lazy.attention_norm,
            mesh_device=mesh_device,
            geometry=geometry,
            precision=precision,
            resources=resources,
            prefetch_contexts=prefetch_contexts,
            decode_placements=decode,
            eps=norm_eps,
            # The first layer creates the residual stream from its input, so it
            # has nothing to fuse. Every later layer fuses `h += x`.
            residual_policy=(RMSNorm2DResidualPolicy.NONE if index == 0 else RMSNorm2DResidualPolicy.FUSED_DECODE),
        ),
        attention_config=attention_config,
        ff_norm_config=_norm_config(
            lazy.ff_norm,
            mesh_device=mesh_device,
            geometry=geometry,
            precision=precision,
            resources=resources,
            prefetch_contexts=prefetch_contexts,
            decode_placements=decode,
            eps=norm_eps,
            residual_policy=RMSNorm2DResidualPolicy.FUSED_DECODE,
        ),
        mlp_config=mlp_config,
        kv_spec=GalaxyAttentionKVSpec.from_geometry(
            n_kv_heads=geometry.n_kv_heads,
            head_dim=geometry.head_dim,
            kv_cache_dtype=precision.kv_cache_dtype,
            paged_attention_config=paged_attention_config,
        ),
        decode_attention_input_memcfg=decode.attention_input_memcfg,
        decode_mlp_input_memcfg=decode.mlp_input_memcfg,
        decode_mlp_input_dtype=precision.decode_mlp_activation_dtype,
        decode_residual_memcfg=decode.residual_memcfg,
        prefill_attention_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_mlp_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_mlp_input_dtype=precision.prefill_mlp_activation_dtype,
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_residual_dtype=precision.prefill_residual_dtype,
    )


def build_qwen3_32b_galaxy_transformer_2d_config(
    *,
    mesh_device: Any,
    geometry: GalaxyDenseGeometry,
    precision: Qwen3_32BGalaxyPrecision,
    lazy_weights: Qwen3_32BGalaxyLazyWeights,
    resources: Any,
    prefetcher: Any,
    norm_eps: float,
    rope_theta: float,
    # True, matching the production Galaxy model. The Galaxy attention decode
    # path selects the fused op on exactly the prefetcher condition -
    #     if self.use_prefetcher:
    #         q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(...)
    #     else:
    #         ... rotary_embedding_llama(q, ...); rotary_embedding_llama(k, ...)
    # - so on a prefetcher mesh the non-fused pair is the *fallback* path, kept
    # for Blackhole, and it expects a different cos/sin layout: `get_rot_mats`
    # returns [1, 1, local_batch, head_dim] for the non-fused decode op, while
    # `get_rm_rot_mats` expands to [1, expanded_batch, heads, head_dim] for the
    # fused one.
    #
    # Measured for Llama on `(8, 4)` with the non-fused pair (D-B25b): the decode
    # step wrote a K of |max| = inf into the cache at the current position while
    # V, which does not pass through RoPE, was exact at PCC 0.99973 - and the
    # prefix prefill wrote was still 0.99993. Qwen3-32B has 64 heads against 8 KV
    # heads, so the head-row asymmetry that exposes this is larger here, not
    # smaller.
    use_qk_fused_rotary: bool = True,
    paged_attention_config: GalaxyPagedAttentionConfig | None = None,
    enable_device_sampling: bool = True,
    chunked_prefill_sequence_lengths: tuple[int, ...] = (),
    decode_placements: GalaxyDecodePlacements | None = None,
    cache_path: Path | str | None = None,
    owns_shared_resources: bool = False,
) -> Qwen3_32BGalaxyTransformer2DConfig:
    """Resolve every 2D module config for the Galaxy Qwen model."""

    validate_galaxy_mesh("Galaxy Qwen3-32B", mesh_device)
    if not lazy_weights.layers:
        raise ValueError("the Galaxy Qwen model requires at least one decoder layer")
    decode = decode_placements or resolve_galaxy_decode_placements(geometry, mesh_device)
    prefill = resolve_galaxy_prefill_placements(geometry, mesh_device)
    prefetch_contexts = (prefetcher.context("prefill"), prefetcher.context("decode"))
    rope_core_grid, rope_batch_grid = rope_core_grids(mesh_device, use_qk_fused=use_qk_fused_rotary)
    # The RoPE transformation matrices are owned by RotarySetup2D, which the
    # model constructs from this config. The collectives borrow them through a
    # provider the model binds, so nothing materializes at config time.
    collectives = GalaxyAttentionCollectives(
        resources=resources,
        mesh_device=mesh_device,
        geometry=geometry,
        decode_placements=decode,
        use_fused_qk_rotary=use_qk_fused_rotary,
        collective_dtype=precision.attention_collective_dtype,
        head_dtype=precision.decode_activation_dtype,
    )
    embedding_config = Embedding2DConfig(
        weights=lazy_weights.embedding,
        mesh_device=mesh_device,
        vocab_size=geometry.vocab_size,
        dim=geometry.dim,
        max_batch_size=geometry.max_batch_size,
        embed_scale=1.0,
        decode_output_dtype=precision.decode_activation_dtype,
        # The residual placement itself, not L1_MEMORY_CONFIG and not DRAM.
        # `ttnn.embedding` takes its program grid from a *sharded* output's shard
        # grid, and only from there: with an interleaved output - L1 or DRAM - it
        # spreads over the whole compute grid, including the two prefetch sender
        # columns, and cannot place its own static circular buffers around the
        # prefetcher's L1 there:
        #     TT_THROW ... Statically allocated circular buffers in program N
        #     clash with L1 buffers on core range [0-0 - 0-0]
        # Naming the residual placement confines the program to the worker cores
        # that placement already occupies, and makes the relocation in
        # `embed_decode` a no-op instead of a second copy. Milestone B job 1
        # found and fixed this for Llama; this copy carried it unchanged.
        decode_output_memcfg=decode.residual_memcfg,
        prefill_output_dtype=precision.prefill_activation_dtype,
        prefill_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
    )
    rope_config = RotarySetup2DConfig(
        cos_matrix=lazy_weights.rope_cos,
        sin_matrix=lazy_weights.rope_sin,
        max_batch_size=geometry.max_batch_size,
        head_dim=geometry.head_dim,
        mesh_device=mesh_device,
        users_per_column=geometry.users_per_column,
        use_qk_fused=use_qk_fused_rotary,
        rope_theta=rope_theta,
        # Qwen3-32B applies unscaled GPT-NeoX RoPE.
        rope_scaling_factor=None,
        original_context_len=None,
        core_grid=rope_core_grid,
        batch_grid=rope_batch_grid,
    )
    block_configs = tuple(
        _build_block_config(
            index,
            layer,
            geometry=geometry,
            precision=precision,
            mesh_device=mesh_device,
            resources=resources,
            prefetch_contexts=prefetch_contexts,
            decode=decode,
            prefill=prefill,
            collectives=collectives,
            norm_eps=norm_eps,
            paged_attention_config=paged_attention_config,
            chunked_lengths=tuple(chunked_prefill_sequence_lengths),
        )
        for index, layer in enumerate(lazy_weights.layers)
    )
    norm_config = _norm_config(
        lazy_weights.final_norm,
        mesh_device=mesh_device,
        geometry=geometry,
        precision=precision,
        resources=resources,
        prefetch_contexts=prefetch_contexts,
        decode_placements=decode,
        eps=norm_eps,
        residual_policy=RMSNorm2DResidualPolicy.FUSED_DECODE,
    )
    lm_head_config = LMHead2DConfig(
        output_weights=(lazy_weights.lm_head,),
        vocab_size=geometry.vocab_size,
        # One collective per mode, each naming its own worker sub-device. Without
        # the sub-device id the reduction places workers on the whole compute
        # grid, which the loaded decode manager does not own.
        decode_collective=GalaxyColumnAllReduce(
            mesh_device,
            subdevice_id=lambda: resources.context("decode").worker_sub_device_id,
            # Decode goes through the keyed persistent buffer, because
            # `ttnn.all_reduce`'s buffer-less path falls back to a composite
            # all-gather whose internal `concat` is not sub-device aware.
            resources=resources,
            placements=decode,
            dtype=precision.lm_head_output_dtype,
        ),
        prefill_collective=GalaxyColumnAllReduce(
            mesh_device,
            subdevice_id=lambda: resources.context("prefill").worker_sub_device_id,
        ),
        mesh_device=mesh_device,
        dim=geometry.dim,
        padded_vocab_size=geometry.padded_vocab_size,
        max_batch_size=geometry.max_batch_size,
        compute_kernel_config=precision.lm_head_kernel_config,
        # Both weights stay interleaved DRAM, and saying so here is the honest
        # description rather than a choice: `resolve_lazy_weight` fills only the
        # *None* fields of a LazyWeight, and `_lazy` already gave this one
        # `ttnn.DRAM_MEMORY_CONFIG`, so any other value here would be silently
        # discarded rather than applied.
        decode_weights_memcfgs=(ttnn.DRAM_MEMORY_CONFIG,),
        prefill_weights_memcfgs=(ttnn.DRAM_MEMORY_CONFIG,),
        # Decode runs the 24-core gather-in0 ring - the same ring the MLP uses,
        # and the one the production LM head has always used. Prefill keeps
        # interleaved DRAM: it has many row tiles, so a 2D multicast matmul can
        # spread over them, and its mode plan is not partitioned the way
        # decode's is.
        decode_input_memcfg=decode.lm_head_input_memcfg,
        prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        decode_output_memcfg=decode.lm_head_output_memcfg,
        prefill_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        decode_program_configs=(decode.lm_head_program_config,),
        # The ring matmul intersects its cores with this sub-device, and defaults
        # to sub-device 0 - the prefetch senders - if it is not told.
        decode_sub_device_id=lambda: resources.context("decode").worker_sub_device_id,
        prefill_sub_device_id=lambda: resources.context("prefill").worker_sub_device_id,
        decode_output_dtype=precision.lm_head_output_dtype,
        prefill_output_dtype=precision.prefill_activation_dtype,
    )
    # Sampling2D consumes logits whose users are sharded over the four mesh
    # columns, while the decode graph keeps the physical batch replicated across
    # columns. The model therefore only owns the resolved sampler; selecting each
    # column's user slice is an executor responsibility (Milestone C).
    sampling_sub_core_grids, sampling_topk_grid, sampling_start_core = sampling_core_grids()
    sampling_config = (
        Sampling2DConfig(
            vocab_size=geometry.vocab_size,
            padded_vocab_size=geometry.padded_vocab_size,
            mesh_device=mesh_device,
            max_batch_size=geometry.max_batch_size,
            sub_core_grids=sampling_sub_core_grids,
            sub_core_grid_topk=sampling_topk_grid,
            start_core=sampling_start_core,
        )
        if enable_device_sampling
        else None
    )
    return Qwen3_32BGalaxyTransformer2DConfig(
        geometry=geometry,
        mesh_device=mesh_device,
        resources=resources,
        prefetcher=prefetcher,
        embedding_config=embedding_config,
        rope_config=rope_config,
        block_configs=block_configs,
        norm_config=norm_config,
        lm_head_config=lm_head_config,
        attention_collectives=collectives,
        decode_placements=decode,
        prefill_placements=prefill,
        sampling_config=sampling_config,
        cache_path=str(cache_path) if cache_path is not None else None,
        owns_shared_resources=owns_shared_resources,
    )


# ============================================================================
# Tensor model
# ============================================================================


def _relocate(tensor: Any, memory_config: Any, dtype: Any = None) -> Any:
    """Place and recast a tensor without leaving the decode sub-device partition.

    Ported from ``llama33_70b_galaxy.model._relocate``, which Milestone B job 1
    rewrote after the three-argument form aborted Llama's first decode on
    silicon. The defect is not model-specific and this copy carried it unchanged.

    The decode sub-device manager owns only part of the compute grid, and
    tt-metal rejects a program that touches a core the manager does not own:

        TT_FATAL ... Kernel group cores do not match sub device cores
                     for programmable core type TENSIX

    Three obvious spellings are unsafe:

    * ``to_memory_config(t, memcfg, dtype)`` - the three-argument form - goes to
      ``ttnn::prim::copy``, whose factory splits work over
      ``device->compute_with_storage_grid_size()`` (standing TODO to use the
      sub-device's worker cores instead). **This is what this function used to
      call**, at every placement hop in the Qwen decode graph.
    * ``ttnn.typecast(t, dtype)`` on sharded activations reaches the same
      full-grid split.
    * ``to_memory_config(t, memcfg)`` between two shard specs differing in both
      grid and width resolves to ``reshard_program_factory_generic``, also
      full-grid.

    What *is* safe is the explicit pair: ``sharded_to_interleaved`` runs on its
    input's ``shard_spec.grid`` and ``interleaved_to_sharded`` on its output
    shard's cores, both worker-confined. Both accept ``output_dtype``, so the
    recast rides along instead of needing an op of its own.

    The cost is one DRAM round trip per placement hop - a real decode-latency
    cost that belongs on the performance follow-up list, not a correctness
    argument.
    """

    source_memcfg = tensor.memory_config()
    needs_placement = memory_config is not None and source_memcfg != memory_config
    needs_dtype = dtype is not None and tensor.dtype != dtype
    if not needs_placement and not needs_dtype:
        return tensor

    target_memcfg = memory_config if needs_placement else source_memcfg
    cast_to = dtype if needs_dtype else None

    if source_memcfg.is_sharded():
        if not target_memcfg.is_sharded():
            # Sharded to interleaved in ONE hop, straight into the requested
            # interleaved config. `sharded_to_interleaved` runs on its *input's*
            # `shard_spec.grid`, so it stays inside the partition whatever the
            # destination buffer type is.
            #
            # The two-hop version this replaced staged into DRAM first and then
            # asked `to_memory_config` to move DRAM -> L1, which is an
            # interleaved-to-interleaved move and therefore `ttnn::prim::copy` on
            # the full compute grid. For Llama that aborted the very first decode
            # step to reach the LM head, which at the time asked for
            # `ttnn.L1_MEMORY_CONFIG`:
            #     TT_FATAL ... Kernel group cores do not match sub device cores
            #                  for programmable core type TENSIX
            # (defect D-B10). The bug is latent for every interleaved target that
            # is not DRAM, in prefill as well as decode; this package carried the
            # two-hop form unchanged.
            placed = ttnn.sharded_to_interleaved(tensor, target_memcfg, output_dtype=cast_to)
            if placed is not tensor:
                deallocate_if_allocated(tensor)
            return placed
        staged = ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=cast_to)
        deallocate_if_allocated(tensor)
        placed = ttnn.interleaved_to_sharded(staged, target_memcfg)
        deallocate_if_allocated(staged)
        return placed

    if target_memcfg.is_sharded():
        placed = ttnn.interleaved_to_sharded(tensor, target_memcfg, output_dtype=cast_to)
        if placed is not tensor:
            deallocate_if_allocated(tensor)
        return placed

    # Interleaved to interleaved. No sharded staging is available to carry the
    # recast, so this is the one path still exposed to the full-grid typecast
    # factory. It is not reached by the decode graph, whose activations are
    # sharded wherever a recast is asked for; if a future recipe does reach it
    # under a loaded partition, it will abort loudly rather than silently.
    if needs_dtype:
        recast = ttnn.typecast(tensor, dtype=dtype)
        if recast is not tensor:
            deallocate_if_allocated(tensor)
        tensor = recast
    if needs_placement and tensor.memory_config() != target_memcfg:
        placed = ttnn.to_memory_config(tensor, target_memcfg)
        if placed is not tensor:
            deallocate_if_allocated(tensor)
        tensor = placed
    return tensor


def _release_unless(tensor: Any, *keep: Any) -> None:
    if tensor is None:
        return
    if any(tensor is other for other in keep):
        return
    deallocate_if_allocated(tensor)


class Qwen3_32BTransformerBlock2D(LightweightModule):
    """One Galaxy Qwen decoder layer: norm, attention, norm, MLP, residual."""

    def __init__(self, config: Qwen3_32BGalaxyBlockConfig):
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm2D.from_config(config.attention_norm_config)
        self.attention = Attention2D.from_config(config.attention_config)
        self.ff_norm = RMSNorm2D.from_config(config.ff_norm_config)
        self.feed_forward = MLP2D.from_config(config.mlp_config)
        self.kv_spec = config.kv_spec
        # Strategy is bound once: layer 0 creates the residual stream, later
        # layers fuse the accumulation into their distributed norm.
        self._decode_attention_norm = (
            self._decode_attention_norm_fused
            if config.attention_norm_config.residual_policy is RMSNorm2DResidualPolicy.FUSED_DECODE
            else self._decode_attention_norm_initial
        )

    @classmethod
    def from_config(cls, config: Qwen3_32BGalaxyBlockConfig) -> "Qwen3_32BTransformerBlock2D":
        return cls(config)

    # Decode

    def _decode_attention_norm_initial(self, x: Any, h: Any) -> tuple[Any, Any]:
        if h is not None:
            raise ValueError("the first Galaxy layer creates the residual stream and takes h=None")
        return self.attention_norm.decode_forward(x), x

    def _decode_attention_norm_fused(self, x: Any, h: Any) -> tuple[Any, Any]:
        if h is None:
            raise ValueError("fused residual decode requires the running residual sum")
        normed, residual = self.attention_norm.decode_forward(x, residual=h)
        _release_unless(x, normed, residual)
        return normed, residual

    def decode_forward(self, x: Any, h: Any, rot_mats: Any, metadata: DecodeMetadata) -> tuple[Any, Any]:
        attention_input, h = self._decode_attention_norm(x, h)
        attention_input = _relocate(attention_input, self.config.decode_attention_input_memcfg)
        attention_output = self.attention.decode_forward(attention_input, rot_mats, metadata)
        mlp_input, h = self.ff_norm.decode_forward(attention_output, residual=h)
        _release_unless(attention_output, mlp_input, h)
        mlp_input = _relocate(mlp_input, self.config.decode_mlp_input_memcfg, self.config.decode_mlp_input_dtype)
        return self.feed_forward.decode_forward(mlp_input), h

    # Prefill

    def prefill_forward(self, x: Any, h: Any, rot_mats: Any, metadata: PrefillMetadata) -> tuple[Any, Any]:
        residual_memcfg = self.config.prefill_residual_memcfg
        residual_dtype = self.config.prefill_residual_dtype
        if h is None:
            h = x
        else:
            accumulated = ttnn.add(h, x, memory_config=residual_memcfg, dtype=residual_dtype)
            _release_unless(h, accumulated)
            _release_unless(x, accumulated)
            h = accumulated
        attention_input = self.attention_norm.prefill_forward(h)
        attention_input = _relocate(attention_input, self.config.prefill_attention_input_memcfg)
        attention_output = self.attention.prefill_forward(attention_input, rot_mats, metadata)
        residual = ttnn.add(h, attention_output, memory_config=residual_memcfg, dtype=residual_dtype)
        _release_unless(attention_output, residual)
        _release_unless(h, residual)
        mlp_input = self.ff_norm.prefill_forward(residual)
        mlp_input = _relocate(mlp_input, self.config.prefill_mlp_input_memcfg, self.config.prefill_mlp_input_dtype)
        return self.feed_forward.prefill_forward(mlp_input), residual

    def forward(self, x: Any, h: Any, rot_mats: Any, *, mode: str, metadata: Any) -> tuple[Any, Any]:
        if mode == "decode":
            return self.decode_forward(x, h, rot_mats, metadata)
        if mode == "prefill":
            return self.prefill_forward(x, h, rot_mats, metadata)
        raise ValueError(f"unsupported Galaxy layer mode: {mode}")

    def close(self) -> None:
        self.attention.close()


class Qwen3_32BGalaxyTransformer2D(LightweightModule):
    """Qwen3-32B as a full-mesh Galaxy `(8, 4)` tensor model."""

    hf_model = QWEN3_32B_GALAXY_HF_MODEL

    def __init__(self, config: Qwen3_32BGalaxyTransformer2DConfig):
        super().__init__()
        self.config = config
        self.geometry = config.geometry
        self.mesh_device = config.mesh_device
        self.resources = config.resources
        self.prefetcher = config.prefetcher
        self.embedding = Embedding2D.from_config(config.embedding_config)
        self.rope_setup = RotarySetup2D.from_config(config.rope_config)
        self.attention_collectives = config.attention_collectives
        # Binding a provider keeps the transformation matrices lazy: nothing
        # materializes until the first forward pass asks for them.
        self.attention_collectives.bind_transformation_matrices(self.rope_setup.get_both_trans_mats)
        self.layers = [Qwen3_32BTransformerBlock2D.from_config(block) for block in config.block_configs]
        self.norm = RMSNorm2D.from_config(config.norm_config)
        self.lm_head = LMHead2D.from_config(config.lm_head_config)
        self.sampling = Sampling2D.from_config(config.sampling_config) if config.sampling_config else None
        self.supports_on_device_sampling = self.sampling is not None
        # Decode logits carry the whole physical batch on every column while the
        # sampler consumes one column's user slice, so device sampling needs the
        # selector between them. It allocates nothing until first use.
        self.column_user_selector = (
            GalaxyColumnUserSelector(
                config.mesh_device,
                max_batch_size=config.geometry.max_batch_size,
                users_per_column=config.geometry.users_per_column,
            )
            if self.sampling is not None
            else None
        )
        self.n_layers = config.n_layers
        self.num_devices = GALAXY_DEVICE_COUNT
        self.vocab_size = config.geometry.vocab_size
        self.padded_vocab_size = config.geometry.padded_vocab_size
        self.decode_residual_memcfg = config.decode_placements.residual_memcfg
        self.prefill_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG
        self.model_args = None
        self._kv_owner = object()
        self._kv_specs = tuple(block.kv_spec for block in config.block_configs)
        self._kv_bound = False
        self._closed = False

    # Executor and runtime contracts

    def iter_executor_named_modules(self) -> Iterator[tuple[str, Any]]:
        for index, layer in enumerate(self.layers):
            for suffix, submodule in (
                ("attn_norm", layer.attention_norm),
                ("attention", layer.attention),
                ("ff_norm", layer.ff_norm),
                ("mlp", layer.feed_forward),
            ):
                yield f"layer[{index}].{suffix}", submodule
        yield "final_norm", self.norm
        yield "lm_head", self.lm_head

    @property
    def kv_specs(self) -> tuple[GalaxyAttentionKVSpec, ...]:
        return self._kv_specs

    def paged_kv_contract(self) -> GalaxyPagedKVContract:
        """Return the per-layer KV metadata view for the common KV manager."""

        return GalaxyPagedKVContract(self, self._kv_specs)

    def configure_paged_attention(self, *, block_size: int, max_num_blocks: int) -> None:
        """Install the final paged block geometry before any cache is bound."""

        if self._kv_bound:
            raise RuntimeError("paged attention cannot be reconfigured while a KV cache is bound")
        paged = GalaxyPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
        self._kv_specs = tuple(spec.with_paged_config(paged) for spec in self._kv_specs)
        for layer, spec in zip(self.layers, self._kv_specs):
            layer.kv_spec = spec

    def set_kv_cache(self, kv_cache: Sequence[Sequence[Any]] | None) -> None:
        """Bind or unbind every layer's KV cache transactionally."""

        if self._closed:
            raise RuntimeError("the model is closed")
        if kv_cache is None:
            self._unbind_kv_cache()
            return
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} entries but the model has {len(self.layers)} layers")
        self._unbind_kv_cache()
        bound: list[Any] = []
        try:
            for index, (layer, spec, pair) in enumerate(zip(self.layers, self._kv_specs, kv_cache)):
                tensors = tuple(pair)
                if len(tensors) != 2:
                    raise ValueError(f"kv_cache layer {index} must contain exactly two K/V tensors")
                layer.attention.bind_kv_cache(
                    KVCacheBinding(
                        keys=tensors[0],
                        values=tensors[1],
                        owner=self._kv_owner,
                        metadata=spec.paged_kv_metadata(),
                        mesh_device=self.mesh_device,
                    )
                )
                bound.append(layer.attention)
        except BaseException:
            for attention in reversed(bound):
                attention.unbind_kv_cache(self._kv_owner)
            raise
        self._kv_bound = True

    def _unbind_kv_cache(self) -> None:
        for layer in self.layers:
            if layer.attention.kv_cache_binding is not None:
                layer.attention.unbind_kv_cache(self._kv_owner)
        self._kv_bound = False

    # Operation-boundary lifecycle

    def activate(self, mode: str) -> Any:
        """Activate the prefetch/CCL context for one operation boundary."""

        return self.resources.activate(mode)

    def synchronize(self, mode: str) -> None:
        self.resources.synchronize(mode)

    # Input staging helpers

    def embed_decode(self, tokens: Any) -> Any:
        """Embed one replicated `[1, 32]` decode token row into the residual stream."""

        return _relocate(self.embedding.decode_forward(tokens), self.decode_residual_memcfg)

    def embed_prefill(self, tokens: Any) -> Any:
        """Embed one replicated `[1, sequence]` prefill token row."""

        return self.embedding.prefill_forward(tokens)

    def prepare_decode_rot_mats(self, position_idxs: Any) -> list[Any]:
        """Return `(cos, sin)` for the physical decode batch."""

        return self.rope_setup.decode_forward(position_idxs)

    def prepare_prefill_rot_mats(self, start_pos: int, seq_len: int) -> list[Any]:
        return self.rope_setup.prefill_forward(start_pos=start_pos, seq_len=seq_len)

    def get_rot_idxs(self, position_idxs: Any, on_host: bool = False) -> Any:
        return self.rope_setup.get_rot_idxs(position_idxs, on_host=on_host)

    # Graph methods

    def decode_forward(
        self,
        x_embed: Any,
        current_pos: Any,
        rot_mats: Any,
        page_table: Any = None,
    ) -> Any:
        """Run one physical-batch-32 decode step and return padded logits."""

        metadata = DecodeMetadata(current_positions=current_pos, page_table=page_table)
        x, h = x_embed, None
        for layer in self.layers:
            x, h = layer.decode_forward(x, h, rot_mats, metadata)
        normed, residual = self.norm.decode_forward(x, residual=h)
        _release_unless(x, normed, residual)
        _release_unless(residual, normed)
        return self.lm_head.decode_forward(_relocate(normed, self.config.lm_head_config.decode_input_memcfg))

    def prefill_forward(
        self,
        x_embed: Any,
        rot_mats: Any,
        *,
        sequence_length: int,
        user_ids: tuple[int, ...] = (0,),
        page_table: Any = None,
        chunk_page_table: Any = None,
        chunk_start: int | None = None,
        chunk_start_tensor: Any = None,
        prefix_user_id: int | None = None,
        collective_mode: PrefillCollectiveMode = PrefillCollectiveMode.REGULAR,
        return_hidden_state: bool = False,
    ) -> Any:
        """Run one prefill request and return logits or the final hidden state."""

        metadata = PrefillMetadata(
            sequence_length=sequence_length,
            user_ids=tuple(user_ids),
            collective_mode=collective_mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start=chunk_start,
            chunk_start_tensor=chunk_start_tensor,
            prefix_user_id=prefix_user_id,
        )
        x, h = x_embed, None
        for layer in self.layers:
            x, h = layer.prefill_forward(x, h, rot_mats, metadata)
        hidden = ttnn.add(h, x, memory_config=self.prefill_residual_memcfg)
        _release_unless(x, hidden)
        _release_unless(h, hidden)
        if return_hidden_state:
            return hidden
        return self.project_hidden_state(hidden)

    def project_hidden_state(self, hidden: Any, *, mode: str = "prefill") -> Any:
        """Apply the final norm and LM head to an already-extracted hidden state.

        Decode never reaches this path: its residual sum is fused into the final
        norm inside :meth:`decode_forward`.
        """

        if mode != "prefill":
            raise ValueError("project_hidden_state is the prefill last-token projection")
        normed = self.norm.prefill_forward(hidden)
        _release_unless(hidden, normed)
        return self.lm_head.prefill_forward(_relocate(normed, self.config.lm_head_config.prefill_input_memcfg))

    def project_prefill_logits(
        self,
        hidden: Any,
        *,
        rows: int = 1,
        sequence_length: int | None = None,
        token_indices: Sequence[int] | None = None,
    ) -> tuple[Any, ...]:
        """Normalize a prefill hidden state and project one token per row.

        The final norm runs over the whole token stream because its distributed
        statistics gather is keyed by that geometry; only then is one token
        selected per prefill row. Each row is projected separately, which is the
        only shape ``LMHead2D`` can consume for a row count below the physical
        batch, so the result is one logits tensor per row.

        ``token_indices`` addresses each row's real last token, which is not
        ``sequence_length - 1`` whenever a prompt was padded up to a supported
        recipe length.
        """

        normed = self.norm.prefill_forward(hidden)
        _release_unless(hidden, normed)
        tokens = int(normed.shape[-2])
        if rows < 1:
            raise ValueError("prefill projection needs at least one row")
        if sequence_length is None:
            sequence_length = tokens // rows
        if rows * sequence_length != tokens:
            raise ValueError(f"{rows} rows of {sequence_length} tokens do not cover {tokens} prefill tokens")
        indices = tuple(token_indices) if token_indices is not None else (sequence_length - 1,) * rows
        if len(indices) != rows or any(not 0 <= index < sequence_length for index in indices):
            raise ValueError(f"token_indices must hold one in-range index per row, got {indices}")
        parts = (normed,) if rows == 1 else tuple(ttnn.split(normed, sequence_length, dim=2))
        outputs: list[Any] = []
        try:
            for part, index in zip(parts, indices):
                placed = _relocate(part[:, :, index : index + 1, :], self.config.lm_head_config.prefill_input_memcfg)
                try:
                    outputs.append(self.lm_head.prefill_forward(placed))
                finally:
                    deallocate_if_allocated(placed)
            return tuple(outputs)
        finally:
            if rows > 1:
                for part in parts:
                    deallocate_if_allocated(part)
            deallocate_if_allocated(normed)

    def select_decode_column_users(self, logits: Any) -> Any:
        """Return each mesh column's user slice of a column-replicated tensor."""

        if self.column_user_selector is None:
            raise RuntimeError("column user selection requires the device sampler to be enabled")
        return self.column_user_selector(logits)

    def sample_decode(
        self,
        logits: Any,
        *,
        top_k: Any = 1,
        top_p: Any = 1.0,
        temperature: Any = 0.0,
        seed: Any = None,
        forced_argmax: Any = False,
        slot_ids: Sequence[int] | None = None,
    ) -> Any:
        """Sample one token per user from decode logits on device."""

        if self.sampling is None:
            raise RuntimeError("device sampling is disabled for this model")
        column_logits = self.select_decode_column_users(logits)
        try:
            return self.sampling.decode_forward(
                column_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                forced_argmax=forced_argmax,
                slot_ids=slot_ids,
            )
        finally:
            deallocate_if_allocated(column_logits)

    def forward(self, x: Any, *args: Any, mode: str = "decode", **kwargs: Any) -> Any:
        if mode == "decode":
            return self.decode_forward(x, *args, **kwargs)
        if mode == "prefill":
            return self.prefill_forward(x, *args, **kwargs)
        raise ValueError(f"unsupported Galaxy model mode: {mode}")

    # Teardown

    def close(self) -> None:
        """Release model-owned device state; terminal and idempotent."""

        if self._closed:
            return
        failures: list[BaseException] = []

        def attempt(action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - collect, then raise the first
                failures.append(error)

        attempt(self._unbind_kv_cache)
        for layer in self.layers:
            attempt(layer.close)
        attempt(self.attention_collectives.cleanup)
        attempt(self.rope_setup.release)
        attempt(self.lm_head.release)
        if self.sampling is not None:
            attempt(self.sampling.release)
        if self.column_user_selector is not None:
            attempt(self.column_user_selector.release)
        attempt(self.embedding.release)
        if self.config.owns_shared_resources:
            attempt(self.resources.cleanup)
            attempt(self.prefetcher.cleanup)
        self._closed = True
        if failures:
            raise failures[0]


# ============================================================================
# Assembly
# ============================================================================


def build_qwen3_32b_galaxy_model(
    mesh_device: Any,
    *,
    params: Qwen3_32BGalaxyModelParameters,
    weights: Qwen3_32BGalaxyWeights,
    precision: Qwen3_32BGalaxyPrecision = QWEN3_32B_GALAXY_ACCURACY,
    paged_attention_config: GalaxyPagedAttentionConfig | None = None,
    enable_device_sampling: bool = True,
    # True, matching the production Galaxy model. The Galaxy attention decode
    # path selects the fused op on exactly the prefetcher condition -
    #     if self.use_prefetcher:
    #         q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(...)
    #     else:
    #         ... rotary_embedding_llama(q, ...); rotary_embedding_llama(k, ...)
    # - so on a prefetcher mesh the non-fused pair is the *fallback* path, kept
    # for Blackhole, and it expects a different cos/sin layout: `get_rot_mats`
    # returns [1, 1, local_batch, head_dim] for the non-fused decode op, while
    # `get_rm_rot_mats` expands to [1, expanded_batch, heads, head_dim] for the
    # fused one.
    #
    # Measured for Llama on `(8, 4)` with the non-fused pair (D-B25b): the decode
    # step wrote a K of |max| = inf into the cache at the current position while
    # V, which does not pass through RoPE, was exact at PCC 0.99973 - and the
    # prefix prefill wrote was still 0.99993. Qwen3-32B has 64 heads against 8 KV
    # heads, so the head-row asymmetry that exposes this is larger here, not
    # smaller.
    use_qk_fused_rotary: bool = True,
    cache_path: Path | str | None = None,
    global_cb_size: int | None = GALAXY_GLOBAL_CB_SIZE,
    prefetcher_injections: dict[str, Any] | None = None,
) -> Qwen3_32BGalaxyTransformer2D:
    """Own the complete Galaxy construction order and return the tensor model.

    The returned model owns the Galaxy resources and the prefetcher until a
    model-owned executor takes over that role, so ``close()`` is the single
    teardown entry point.
    """

    if len(weights.layers) != params.n_layers:
        raise ValueError(f"expected {params.n_layers} layer weight sets, got {len(weights.layers)}")
    if params.qk_norm and any(layer.q_norm is None or layer.k_norm is None for layer in weights.layers):
        raise ValueError("Qwen3-32B requires per-head Q and K normalization weights for every layer")
    validate_galaxy_mesh("Galaxy Qwen3-32B", mesh_device)
    geometry = params.geometry()
    decode = resolve_galaxy_decode_placements(geometry, mesh_device)
    resources_config = build_galaxy_resources_config(mesh_device, geometry, decode)
    lazy_weights = build_qwen3_32b_galaxy_lazy_weights(
        mesh_device=mesh_device,
        geometry=geometry,
        precision=precision,
        weights=weights,
        cache_path=Path(cache_path) if cache_path is not None else None,
    )
    registration = lazy_weights.prefetch_registration()
    prefetcher = build_galaxy_prefetcher(
        mesh_device,
        resources_config,
        expected_weight_count=len(registration),
        global_cb_size=global_cb_size,
        prefetch_num_layers=len(lazy_weights.layers),
        dram_prefetch_start=galaxy_dram_prefetch_start(
            tensors_per_layer=len(QWEN3_32B_PREFETCHED_WEIGHT_NAMES),
            num_layers=len(lazy_weights.layers),
        ),
        **(prefetcher_injections or {}),
    )
    resources = None
    try:
        for name, weight in registration:
            prefetcher.register_weight(name, weight.get_device_weight())
        prefetcher.seal()
        resources = create_galaxy_resources(mesh_device, config=resources_config, prefetcher=prefetcher)
        config = build_qwen3_32b_galaxy_transformer_2d_config(
            mesh_device=mesh_device,
            geometry=geometry,
            precision=precision,
            lazy_weights=lazy_weights,
            resources=resources,
            prefetcher=prefetcher,
            norm_eps=params.rms_norm_eps,
            rope_theta=params.rope_theta,
            use_qk_fused_rotary=use_qk_fused_rotary,
            paged_attention_config=paged_attention_config,
            enable_device_sampling=enable_device_sampling,
            chunked_prefill_sequence_lengths=params.chunked_prefill_sequence_lengths,
            decode_placements=decode,
            cache_path=cache_path,
            owns_shared_resources=True,
        )
        return Qwen3_32BGalaxyTransformer2D(config)
    except BaseException:
        if resources is not None:
            try:
                resources.cleanup()
            except BaseException:
                pass
        prefetcher.cleanup()
        raise


__all__ = [
    "DEFAULT_HF_REVISION",
    "QWEN3_32B_CHECKPOINT_CONTRACT",
    "QWEN3_32B_GALAXY_ACCURACY",
    "QWEN3_32B_GALAXY_HF_MODEL",
    "QWEN3_32B_GALAXY_PERFORMANCE",
    "QWEN3_32B_PREFETCHED_WEIGHT_NAMES",
    "Qwen3_32BGalaxyBlockConfig",
    "Qwen3_32BGalaxyLayerWeights",
    "Qwen3_32BGalaxyLazyLayerWeights",
    "Qwen3_32BGalaxyLazyWeights",
    "Qwen3_32BGalaxyModelParameters",
    "Qwen3_32BGalaxyPrecision",
    "Qwen3_32BGalaxyTransformer2D",
    "Qwen3_32BGalaxyTransformer2DConfig",
    "Qwen3_32BGalaxyWeights",
    "Qwen3_32BTransformerBlock2D",
    "build_qwen3_32b_galaxy_lazy_weights",
    "build_qwen3_32b_galaxy_model",
    "build_qwen3_32b_galaxy_transformer_2d_config",
    "default_paged_attention_config",
    "parameters_from_hf_config",
    "validate_qwen3_32b_checkpoint",
]
