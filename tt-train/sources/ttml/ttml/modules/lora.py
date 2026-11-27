# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import ttnn
import ttml

from .linear import LinearLayer, ColumnParallelLinear, RowParallelLinear
from .module_base import AbstractModuleBase, ModuleDict, ModuleList
from .parameter import Buffer, Parameter
from _ttml.modules import RunMode


@dataclass(frozen=True)
class LoraBlocks:
    """The row blocks of a fused linear layer, each a projection with its own LoRA adapter.

    A fused layer stacks several projections in its output rows, as Llama's ``qkv_linear`` stacks q, k and v.
    ``rows`` are the row counts of its blocks, in order, and ``adapted`` flags the blocks that get an adapter.
    Each of the ``n`` adapted blocks gets its own rank-``r`` pair ``A_i``, ``B_i``, as a separate projection would:
    ``lora_A`` stacks the ``A_i`` (``[n*r, in]``), and ``lora_B`` holds each ``B_i`` in its block's rows
    (``[out, r]``). The rows of the blocks that aren't adapted stay zero.
    """

    rows: tuple[int, ...]
    adapted: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.adapted):
            raise ValueError(f"LoraBlocks: {len(self.rows)} blocks of rows, {len(self.adapted)} adapted flags")
        if not any(self.adapted):
            raise ValueError("LoraBlocks: no block is adapted")


@dataclass
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adaptation) injection.

    Attributes:
        rank: Rank of the low-rank decomposition (A and B matrices).
        alpha: Scaling numerator.  The effective scaling factor is
            ``alpha / sqrt(rank)`` when *use_rslora* is ``True``, else
            ``alpha / rank``.
        target_modules: List of regex patterns matched against fully-qualified
            module names.  Matching linear layers are replaced with their LoRA
            counterparts.
        target_blocks: Regex patterns, matched like *target_modules*, mapped
            to the blocks of the fused layers they match (see ``LoraBlocks``):
            each adapted block of such a layer gets its own adapter.  A
            matching layer that no pattern here matches gets one adapter over
            all of its rows.
        use_rslora: Use rank-stabilized LoRA scaling (``alpha / sqrt(rank)``)
            instead of the original ``alpha / rank``.
        is_bias_trainable: If ``True``, bias parameters in matched layers remain
            trainable after injection.
        trainable_modules: List of name substrings; any parameter whose path
            contains one of these substrings is unfrozen after injection.  This
            is useful for keeping specific non-LoRA parameters trainable
            (e.g. layer norms).
        lora_dropout: Dropout probability applied to the LoRA input during
            training.
    """

    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = field(default_factory=list)
    target_blocks: dict[str, LoraBlocks] = field(default_factory=dict)
    use_rslora: bool = False
    is_bias_trainable: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    lora_dropout: float = 0.0
    verbose: bool = False


def _create_lora_A(in_features: int, rank: int, mapper=None):
    """Initialize LoRA A (down-projection) with kaiming uniform, shape (1, 1, rank, in_features)."""
    bound = 1.0 / np.sqrt(in_features)
    weight_np = np.random.uniform(
        low=-bound,
        high=bound,
        size=(1, 1, rank, in_features),
    ).astype(np.float32)
    return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _create_lora_B(rank: int, out_features: int, mapper=None):
    """Initialize LoRA B (up-projection) with zeros, shape (1, 1, out_features, rank)."""
    weight_np = np.zeros((1, 1, out_features, rank), dtype=np.float32)
    return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _constant(values: np.ndarray):
    """A ``(1, 1, rows, cols)`` bf16 tensor that takes no gradient."""
    tensor = ttml.autograd.Tensor.from_numpy(values[None, None], ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
    tensor.set_requires_grad(False)
    return tensor


def _create_block_constants(blocks: LoraBlocks, rank: int):
    """The constants that lay out a fused layer's ``lora_B`` block-diagonally (see ``LoraLinear``).

    ``expand`` ``(n*r, r)`` is ``n`` stacked identities: ``linear(lora_B, expand)`` is ``n`` copies of ``lora_B``
    side by side. ``mask`` ``(out, n*r)`` is 1 where the rows of the ``j``-th adapted block meet columns
    ``j*r:(j+1)*r``, and 0 elsewhere.
    """
    slots = sum(blocks.adapted)
    expand = np.tile(np.eye(rank, dtype=np.float32), (slots, 1))
    mask = np.zeros((sum(blocks.rows), slots * rank), dtype=np.float32)
    start, slot = 0, 0
    for rows, adapted in zip(blocks.rows, blocks.adapted):
        if adapted:
            mask[start : start + rows, slot * rank : (slot + 1) * rank] = 1.0
            slot += 1
        start += rows
    return _constant(expand), _constant(mask)


def lora_b_layout(base_weight: Parameter) -> dict:
    """The checkpoint layout declarations of ``base_weight`` that describe its output rows (dim 2): a fused segment
    order and a padded vocab's logical extent. lora_B's rows add onto those rows shard by shard, so lora_B is laid out
    the same way and needs the same declarations to reshard across TP sizes."""
    declared = getattr(base_weight, "checkpoint_layout", None) or {}
    return {kind: value for kind, value in declared.items() if int(value[0]) in (2, -2)}


class LoraLinear(AbstractModuleBase):
    """LinearLayer with frozen base weights and a trainable low-rank path.

    Computes: linear(x, weight, bias) + linear(linear(x, lora_A), lora_B) * scaling

    With ``blocks``, the layer is fused and each adapted block has its own adapter (``LoraBlocks``). The update
    then takes ``lora_B`` laid out block-diagonally (``(out, n*r)``), so that block ``i``'s rows only see the
    columns of ``linear(x, lora_A)`` that ``A_i`` produces. It's built each forward from two constant buffers,
    ``lora_expand`` and ``lora_mask``, which ``shared`` (a dict) lets the layers of one model share. The masked
    entries get no gradient, so the rows of blocks that aren't adapted stay zero.
    """

    def __init__(
        self,
        base_layer: LinearLayer,
        config: LoraConfig,
        blocks: LoraBlocks | None = None,
        shared: dict | None = None,
    ):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.blocks = blocks

        self.dropout_prob = config.lora_dropout
        self.scaling = config.alpha / math.sqrt(config.rank) if config.use_rslora else config.alpha / config.rank

        self.weight = base_layer.weight
        self.bias = base_layer.bias

        # Freeze weights
        self.weight.tensor.set_requires_grad(False)
        if self.bias is not None:
            self.bias.tensor.set_requires_grad(config.is_bias_trainable)

        slots = 1
        if blocks is not None:
            if sum(blocks.rows) != self.out_features:
                raise ValueError(f"LoraBlocks of {sum(blocks.rows)} rows for a layer of {self.out_features}")
            slots = sum(blocks.adapted)
            constants = shared if shared is not None else {}
            key = (blocks, config.rank)
            if key not in constants:
                constants[key] = _create_block_constants(blocks, config.rank)
            expand, mask = constants[key]
            self.lora_expand = Buffer(expand)
            self.lora_mask = Buffer(mask)

        self.lora_A = Parameter(_create_lora_A(self.in_features, slots * config.rank))
        self.lora_B = Parameter(_create_lora_B(config.rank, self.out_features))
        self.lora_B.checkpoint_layout = lora_b_layout(self.weight)

    def forward(self, x):
        bias = self.bias.tensor if self.bias is not None else None
        base = ttml.ops.linear.linear(x, self.weight.tensor, bias)
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
        lora_update = ttml.ops.linear.linear(h, self._lora_B(), None)
        return base + lora_update * self.scaling

    def _lora_B(self):
        if self.blocks is None:
            return self.lora_B.tensor
        expanded = ttml.ops.linear.linear(self.lora_B.tensor, self.lora_expand.tensor, None)
        return ttml.ops.binary.mul(expanded, self.lora_mask.tensor)


class LoraColumnParallelLinear(AbstractModuleBase):
    """ColumnParallelLinear with frozen base weights and a trainable low-rank path.

    lora_A is replicated (full in_features).
    lora_B is column-sharded on dim 2 (out_features split across TP devices).
    """

    def __init__(self, base_layer: ColumnParallelLinear, config: LoraConfig):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.gather_output = base_layer.gather_output
        self.axis_name = base_layer.axis_name
        self.cluster_axis = base_layer.cluster_axis

        self.dropout_prob = config.lora_dropout
        self.scaling = config.alpha / math.sqrt(config.rank) if config.use_rslora else config.alpha / config.rank

        self.weight = base_layer.weight
        self.bias = base_layer.bias

        # Freeze weights
        self.weight.tensor.set_requires_grad(False)
        if self.bias is not None:
            self.bias.tensor.set_requires_grad(config.is_bias_trainable)

        self.lora_A = Parameter(_create_lora_A(self.in_features, config.rank))
        lora_B_mapper = ttml.mesh().axis_mapper(self.axis_name, tdim=2)
        self.lora_B = Parameter(_create_lora_B(config.rank, self.out_features, mapper=lora_B_mapper))
        # On a fused base layer lora_B's rows are in the same per-shard [Q_d | K_d | V_d] order as the base output's.
        self.lora_B.checkpoint_layout = lora_b_layout(self.weight)

    def forward(self, x):
        bias_t = self.bias.tensor if self.bias is not None else None
        x = ttml.ops.distributed.broadcast(x, self.cluster_axis)
        base = ttml.ops.linear.linear(x, self.weight.tensor, bias_t)
        if self.gather_output:
            base = ttml.ops.distributed.all_gather(
                base, 3, self.cluster_axis, ttml.ops.distributed.GradOutputType.REPLICATED
            )
        # lora_A is replicated so it operates on the full input; lora_B is
        # column-sharded, producing a sharded update that must be gathered
        # whenever the base path is gathered.
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        # Each device's grad of the replicated lora_A only carries its own
        # lora_B shard's output columns; broadcast (identity forward, all_reduce
        # backward) sums it across TP so every copy gets the full grad.
        lora_A = ttml.ops.distributed.broadcast(self.lora_A.tensor, self.cluster_axis)
        h = ttml.ops.linear.linear(lora_input, lora_A, None)
        lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        if self.gather_output:
            lora_update = ttml.ops.distributed.all_gather(
                lora_update, 3, self.cluster_axis, ttml.ops.distributed.GradOutputType.REPLICATED
            )
        return base + lora_update * self.scaling


class LoraRowParallelLinear(AbstractModuleBase):
    """RowParallelLinear with frozen base weights and a trainable low-rank path.

    lora_A is row-sharded on dim 3 (in_features split across TP devices).
    lora_B is replicated (full out_features).
    """

    def __init__(self, base_layer: RowParallelLinear, config: LoraConfig):
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.input_is_parallel = base_layer.input_is_parallel
        self.axis_name = base_layer.axis_name
        self.cluster_axis = base_layer.cluster_axis

        self.dropout_prob = config.lora_dropout
        self.scaling = config.alpha / math.sqrt(config.rank) if config.use_rslora else config.alpha / config.rank

        self.weight = base_layer.weight
        self.bias = base_layer.bias

        # Freeze weights
        self.weight.tensor.set_requires_grad(False)
        if self.bias is not None:
            self.bias.tensor.set_requires_grad(config.is_bias_trainable)

        lora_A_mapper = ttml.mesh().axis_mapper(self.axis_name, tdim=3)
        self.lora_A = Parameter(_create_lora_A(self.in_features, config.rank, mapper=lora_A_mapper))
        self.lora_B = Parameter(_create_lora_B(config.rank, self.out_features))
        self.lora_B.checkpoint_layout = lora_b_layout(self.weight)

    def forward(self, x):
        if not self.input_is_parallel:
            x = ttml.ops.distributed.scatter(x, 3, self.cluster_axis)
        base = ttml.ops.linear.linear(x, self.weight.tensor, None)
        base = ttml.ops.distributed.all_reduce(base, self.input_is_parallel, self.cluster_axis)
        if self.bias is not None:
            base = ttml.ops.binary.add(base, self.bias.tensor)
        # lora_A is row-sharded (each device sees a slice of in_features), so
        # an all_reduce is needed after lora_A to sum partial projections before
        # the replicated lora_B can be applied.
        lora_input = x
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            lora_input = ttml.ops.dropout.dropout(x, self.dropout_prob)
        h = ttml.ops.linear.linear(lora_input, self.lora_A.tensor, None)
        h = ttml.ops.distributed.all_reduce(h, self.input_is_parallel, self.cluster_axis)
        lora_update = ttml.ops.linear.linear(h, self.lora_B.tensor, None)
        return base + lora_update * self.scaling


_LORA_MODULE_MAP: dict[type, type] = {
    ColumnParallelLinear: LoraColumnParallelLinear,
    RowParallelLinear: LoraRowParallelLinear,
    LinearLayer: LoraLinear,
}


class LoraModel(AbstractModuleBase):
    """Wraps a model, freezes its parameters, and injects LoRA adapters.

    Linear layers whose fully-qualified name matches any pattern in
    ``config.target_modules`` are replaced with their LoRA counterpart via
    ``_LORA_MODULE_MAP`` (``LinearLayer`` -> ``LoraLinear``,
    ``ColumnParallelLinear`` -> ``LoraColumnParallelLinear``, etc.).
    """

    def __init__(self, model: AbstractModuleBase, config: LoraConfig) -> None:
        super().__init__()
        self.model = model
        verbose = config.verbose

        if verbose and not config.target_modules:
            print("[LoRA] target_modules is empty. No injection will occur.")

        if verbose:
            print(
                f"[LoRA] Config: rank={config.rank}, alpha={config.alpha}, "
                f"targets={config.target_modules}, dropout={config.lora_dropout}, "
                f"use_rslora={config.use_rslora}, is_bias_trainable={config.is_bias_trainable}"
            )

        if verbose:
            print("[LoRA] Freezing weights")
        for _, tensor in model.named_parameters():
            tensor.set_requires_grad(False)

        patterns = [re.compile(p) for p in config.target_modules]
        blocks = [(re.compile(p), layer_blocks) for p, layer_blocks in config.target_blocks.items()]
        injected = self._inject(model, "", patterns, blocks, config, shared={})
        if verbose:
            print("[LoRA] Injecting LoRA")
            for name in injected:
                print(f"  - {name}")

        unmatched = [p.pattern for p in patterns if not any(p.search(name) for name in injected)]
        if unmatched:
            raise ValueError(f"[LoRA] target_modules pattern(s) matched no LinearLayer in the model: {unmatched}")
        unmatched = [p.pattern for p, _ in blocks if not any(p.search(name) for name in injected)]
        if unmatched:
            raise ValueError(f"[LoRA] target_blocks pattern(s) matched no layer LoRA was injected into: {unmatched}")

        if config.trainable_modules:
            self._unfreeze_trainable(model, config.trainable_modules)

        if verbose:
            unfrozen = [name for name, t in model.parameters().items() if t.get_requires_grad()]
            print("[LoRA] Unfreezing weights:")
            for name in unfrozen:
                print(f"  - {name}")

    def _inject(
        self,
        module: AbstractModuleBase,
        prefix: str,
        patterns: list[re.Pattern],
        blocks: list[tuple[re.Pattern, LoraBlocks]],
        config: LoraConfig,
        shared: dict,
    ) -> list[str]:
        """Recursively replace matching linear layers with their LoRA wrappers.

        ModuleList and ModuleDict require index/key-based assignment instead of
        ``setattr`` because their ``__setitem__`` maintains internal bookkeeping.
        """
        injected: list[str] = []
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if any(p.search(full_name) for p in patterns):
                lora_cls = _LORA_MODULE_MAP.get(type(child))
                if lora_cls is None:
                    raise RuntimeError(
                        f"Cannot apply LoRA to '{full_name}': "
                        f"'{type(child).__name__}' is not a supported layer type. "
                        f"Supported: {', '.join(cls.__name__ for cls in _LORA_MODULE_MAP)}"
                    )
                layer_blocks = next((b for p, b in blocks if p.search(full_name)), None)
                if layer_blocks is None:
                    lora_layer = lora_cls(child, config)
                elif lora_cls is LoraLinear:
                    lora_layer = LoraLinear(child, config, layer_blocks, shared)
                else:
                    raise RuntimeError(
                        f"Cannot apply per-block LoRA to '{full_name}': only {LinearLayer.__name__} supports "
                        f"target_blocks, not '{type(child).__name__}'"
                    )

                if isinstance(module, ModuleList):
                    module[int(name)] = lora_layer
                elif isinstance(module, ModuleDict):
                    module[name] = lora_layer
                else:
                    setattr(module, name, lora_layer)
                injected.append(full_name)
            elif isinstance(child, AbstractModuleBase):
                injected.extend(self._inject(child, full_name, patterns, blocks, config, shared))
        return injected

    @staticmethod
    def _unfreeze_trainable(model: AbstractModuleBase, trainable_modules: list[str]) -> None:
        """Unfreeze parameters whose full path starts with any of the given prefixes."""
        for param_path, tensor in model.parameters().items():
            if any(prefix in param_path for prefix in trainable_modules):
                tensor.set_requires_grad(True)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
