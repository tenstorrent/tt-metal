# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the Shared Expert module.

- One shared_expert_ffn op = typecast + unified routed expert + mesh_partition: gate/up replicated,
  down split on emb across mesh columns, so each column computes only its own output slice; no CCL
- Composed GLU helpers (``situ_glu``, ``clamped_silu_glu``) for TtFfn's TP-sharded forward
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.debug_logging import DEBUG_LOGGING_ENABLED

# GLU activations this module can run over its gate/up pair. Spelled as the HF ``hidden_act``
# and as the torch reference (reference/tt/moe/expert.py), so a model config's string maps across
# without translation -- unlike the routed expert, whose fused kernel needs an enum.
ACTIVATION_SILU = "silu"
ACTIVATION_SITU = "situ"
ACTIVATION_CLAMPED_SILU_GLU = "clamped_silu_glu"
SUPPORTED_ACTIVATIONS = (ACTIVATION_SILU, ACTIVATION_SITU, ACTIVATION_CLAMPED_SILU_GLU)

# The fused routed-expert op's variant per activation. Its kernel bakes the constants in
# (SituGluConfigKimi, ClampedSiluGluConfigDsV4), so a config that disagrees would silently run a
# different activation.
_FUSED_ACTIVATIONS = {
    ACTIVATION_SILU: ttnn.RoutedExpertActivation.Silu,
    ACTIVATION_SITU: ttnn.RoutedExpertActivation.SituGlu,
    ACTIVATION_CLAMPED_SILU_GLU: ttnn.RoutedExpertActivation.ClampedSiluGlu,
}
_FUSED_SITU_BETAS = (4.0, 25.0)
_FUSED_CLAMPED_SILU_GLU_LIMIT = 10.0


def situ_glu(
    gate_out: ttnn.Tensor,
    up_out: ttnn.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
    sub_core_grids: Optional[ttnn.CoreRangeSet] = None,
) -> ttnn.Tensor:
    """Kimi-K3's SiTU-GLU over a raw gate/up matmul pair, consuming (deallocating) both.

        softcap(gate, situ_beta) * sigmoid(gate) * softcap(up, situ_linear_beta)

    ``ttnn.situ_glu`` is the math; this wrapper exists only to free the two matmul accumulators,
    which an op may not do to its own inputs.

    Blackhole only, since ``ttnn.softcap`` is.

    ``sub_core_grids`` confines every composed step to the shared expert's sub-device, so this can
    run overlapped with the MoE dispatch. It also keeps the intermediates in DRAM: the op's L1 fast
    path allocates interleaved, i.e. on the dispatch sub-device's cores as well.
    """
    activated = ttnn.situ_glu(gate_out, up_out, situ_beta, situ_linear_beta, sub_core_grids=sub_core_grids)
    ttnn.deallocate(gate_out)
    ttnn.deallocate(up_out)
    return activated


def clamped_silu_glu(
    gate_out: ttnn.Tensor,
    up_out: ttnn.Tensor,
    limit: float,
    sub_core_grids: Optional[ttnn.CoreRangeSet] = None,
) -> ttnn.Tensor:
    """DeepSeek-V4's clamped SiLU-GLU over a raw gate/up matmul pair, consuming (deallocating) both.

        silu(min(gate, limit)) * clamp(up, -limit, limit)

    The wrapper exists only to free the two matmul accumulators, which an op may not do to its own
    inputs. ``sub_core_grids`` confines it to the shared expert's sub-device so it can run
    overlapped with the MoE dispatch.
    """
    activated = ttnn.clamped_silu_glu(gate_out, up_out, limit, sub_core_grids=sub_core_grids)
    ttnn.deallocate(gate_out)
    ttnn.deallocate(up_out)
    return activated


COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


# Larger K blocks keep getting faster, but the gate matmul's PCC slides from 0.9998 at 16 tiles to
# 0.9994 at 56: bf16 dest accumulates more terms before the packer flushes. 16 keeps ~97% of the
# speed at the best accuracy.
MAX_IN0_BLOCK_W = 16


def _in0_block_w(k_tiles: int) -> int:
    """Largest K block that tiles k_tiles without a remainder.

    The in0/in1 CBs are double-buffered only when K splits into more than one block, so 24 tiles
    over one pass is the same L1 as 12 over two. Full K is therefore free exactly when the capped
    block would have needed two passes. Of the shapes in this family only DeepSeek-V4-Pro's 24-tile
    down projection is changed by it -- 12 becomes 24, worth 9.1%; Kimi-K3's 48 tiles would overflow
    L1 by 3.8% and stay capped.

    A ratio rather than an L1 budget, so it holds without tracking the factory's buffering depth,
    tile alignment and unreserved base.
    """
    assert k_tiles > 0, f"k_tiles must be positive, got {k_tiles}"
    capped = 1
    for w in range(min(MAX_IN0_BLOCK_W, k_tiles), 0, -1):
        if k_tiles % w == 0:
            capped = w
            break
    return k_tiles if k_tiles <= 2 * capped else capped


def _out_subblock(per_core_M: int, per_core_N: int, deep_k: bool) -> tuple[int, int]:
    """Subblock tiling the per-core output block within the 8-tile DST budget.

    Which shape wins splits on K, measured across every model's gate and down projection: the
    gate's deep K (128-224 tiles) prefers one tall column, the down's shallow K (16-48) prefers the
    widest block that fits.

    The search below is the down projections' rule -- gate and up return above it. Only a per_core_M
    past the DST budget sends deep K into it: unreachable below a ~2300-token chunk, unreported above.
    """
    if deep_k and per_core_M <= 8:
        return (per_core_M, 1)
    best = (1, 1)
    for h in range(1, per_core_M + 1):
        for w in range(1, per_core_N + 1):
            if per_core_M % h or per_core_N % w or h * w > 8:
                continue
            if (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


class TtSharedExpert(LightweightModule):
    """
    TTNN implementation of Shared Expert MLP, replicated on every chip.

    Forward (one op):
        Input: x [1, seq_len, emb_dim] TILE bf16 (replicated across mesh columns)
        output = shared_expert_ffn(x) → [1, seq_len, emb_dim / num_cols] bf8
        (typecast + gate / up / GLU activation / down + mesh_partition over mesh columns, fused:
        the op reads bf16 x directly and computes only this column's emb slice of the output)

    Weights: gate_proj [emb_dim, hidden_dim] and up_proj [emb_dim, hidden_dim] whole on every chip;
    down_proj [hidden_dim, emb_dim / num_cols], split on emb across mesh columns.
    """

    # TtFfn reuses the weight code below and keeps its TP-sharded layout.
    REPLICATED_WEIGHTS = True

    @classmethod
    def _weight_layout(cls, proj: str) -> tuple[tuple, str]:
        """Mesh-mapper dims and cache basename for one projection.

        The fused layout gets its own basenames: as_tensor's cache key ignores the mapper, so a file
        of another layout under the same name would load with the wrong shape. Its down_proj is
        split on emb (its N) across mesh columns, so each column computes only its own output
        slice -- what mesh_partition of the full output kept.
        """
        if cls.REPLICATED_WEIGHTS:
            if proj == "down_proj":
                return (None, -1), "down_proj_emb_sharded"
            return (None, None), f"{proj}_replicated"
        return ((None, -2) if proj == "down_proj" else (None, -1)), proj

    @classmethod
    def check_cache_complete(cls, cache_path: Path, cache_name_prefix: str) -> bool:
        """Check if all shared expert weight cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            _, name = cls._weight_layout(proj)
            if not pattern_exists(f"{cache_name_prefix}.{name}*.tensorbin", "SharedExpert"):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{name}")
                return False
        return True

    @classmethod
    def _convert_and_cache_weights(
        cls,
        torch_weights: dict,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,
    ):
        """
        Shared logic for converting gate/up/down projections to ttnn with caching.

        Args:
            torch_weights: Dict with 'gate_proj', 'up_proj', 'down_proj' [out_features, in_features]
            emb_dim: Embedding dimension
            hidden_dim: Hidden dimension
            mesh_device: Mesh device reference (for mesh_mapper)
            weights_dtype: Weight data type
            cache_path: Cache directory
            cache_name_prefix: Prefix for cache files
            device: None for cache-only, mesh_device for cache+load

        Returns:
            Dict of ttnn.Tensor if device is not None, else None
        """

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        # Prepare post-transpose tensors
        if torch_weights is not None:
            gate_w = torch_weights["gate_proj"].T.contiguous()
            up_w = torch_weights["up_proj"].T.contiguous()
            down_w = torch_weights["down_proj"].T.contiguous()
        else:
            gate_w = torch.empty(emb_dim, hidden_dim)
            up_w = torch.empty(emb_dim, hidden_dim)
            down_w = torch.empty(hidden_dim, emb_dim)

        def _to_ttnn(tensor, dims, name):
            mesh_mapper = ttnn.ShardTensor2dMesh(
                mesh_device,
                mesh_shape=mesh_device.shape,
                dims=dims,
            )
            return ttnn.as_tensor(
                tensor,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
                cache_file_name=_cache_name(name),
            )

        gate_tt = _to_ttnn(gate_w, *cls._weight_layout("gate_proj"))
        up_tt = _to_ttnn(up_w, *cls._weight_layout("up_proj"))
        down_tt = _to_ttnn(down_w, *cls._weight_layout("down_proj"))

        if device is None:
            # Cache built, free host tensors
            del gate_tt, up_tt, down_tt
            return None
        else:
            # Return device tensors for __init__
            return {"gate": gate_tt, "up": up_tt, "down": down_tt}

    @classmethod
    def build_ttnn_cache(
        cls,
        torch_weights: dict,
        emb_dim: int,
        hidden_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Build TTNN cache for shared expert without device copy."""
        cls._convert_and_cache_weights(
            torch_weights, emb_dim, hidden_dim, mesh_device, weights_dtype, cache_path, cache_name_prefix, device=None
        )

    def __init__(
        self,
        mesh_device,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        torch_weights: dict = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        subdevice_id: Optional[ttnn.SubDeviceId] = None,
        subdevice_cores: Optional[ttnn.CoreRangeSet] = None,
        activation: str = ACTIVATION_SILU,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        clamped_silu_glu_limit: Optional[float] = None,
        seq_len_per_chip: Optional[int] = None,
    ):
        """
        Initialize TtSharedExpert module.

        Args:
            mesh_device: TTNN mesh device
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden dimension (default: 2048)
            torch_weights: Optional dict with keys 'gate_proj', 'up_proj', 'down_proj' containing torch tensors
            num_links: Number of ethernet links to use for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Linear)
            activations_dtype: Data type for activations (default: bfloat16)
            weights_dtype: Data type for weights (default: bfloat8_b)
            compute_kernel_config: Compute kernel configuration
            weight_cache_path: Optional path for caching TTNN weight tensors
            cache_name_prefix: Optional prefix for cache file names
            subdevice_cores: the shared expert's sub-device when it overlaps the MoE dispatch. The
                fused op's fixed grid is placed at its top-left core.
            activation: GLU activation over the gate/up pair -- "silu" (default), "situ" for
                Kimi-K3's SiTU-GLU, or "clamped_silu_glu" for DeepSeek-V4's. A plain string, shared
                with TtFfn, which composes it from ttnn ops.
            situ_beta / situ_linear_beta: SiTU softcap betas (K3: 4.0 / 25.0). Required, and
                non-zero, when activation == "situ"; ignored otherwise. The fused op accepts only
                the K3 pair.
            clamped_silu_glu_limit: the clamp limit (V4: 10.0). Required, and positive, when
                activation == "clamped_silu_glu"; ignored otherwise. The fused op accepts only 10.0.
            seq_len_per_chip: tokens per chip in every forward. The fused op runs the shared expert
                as one expert owning all of them, described once here as count [seq_len_per_chip],
                expert index [0] and region start [0].
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_devices = mesh_device.get_num_devices()
        self.num_links = num_links
        self.topology = topology
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        self.subdevice_id = subdevice_id
        self.subdevice_cores = subdevice_cores
        self.weight_cache_path = weight_cache_path
        self.seq_len_per_chip = seq_len_per_chip

        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; expected one of {SUPPORTED_ACTIVATIONS}")
        if activation == ACTIVATION_SITU:
            # ttnn.softcap, which both SiTU halves go through, is Blackhole-only. Raise rather than
            # fall back to SiLU: a silently different activation is a wrong model, not a slow one.
            if not is_blackhole():
                raise ValueError(f"activation {activation!r} needs ttnn.softcap, which is Blackhole-only")
            # softcap precomputes 1/beta, so a zero (or missing) beta would emit inf.
            if not situ_beta or not situ_linear_beta:
                raise ValueError(
                    f"activation {activation!r} requires non-zero situ_beta / situ_linear_beta, "
                    f"got {situ_beta} / {situ_linear_beta}"
                )
        if activation == ACTIVATION_CLAMPED_SILU_GLU:
            if clamped_silu_glu_limit is None or clamped_silu_glu_limit <= 0:
                raise ValueError(
                    f"activation {activation!r} requires a positive clamped_silu_glu_limit, "
                    f"got {clamped_silu_glu_limit}"
                )
        self.activation = activation
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.clamped_silu_glu_limit = clamped_silu_glu_limit
        if self.REPLICATED_WEIGHTS:
            self._init_fused_expert()
        self.cache_name_prefix = cache_name_prefix

        logger.debug(f"Initializing TtSharedExpert with emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")
        logger.debug(f"CCL config: num_links={num_links}, topology={topology}")

        # Create sharded weights
        if torch_weights is not None:
            logger.debug("Creating weights from provided torch tensors")
            weights = self._convert_and_cache_weights(
                torch_weights,
                emb_dim,
                hidden_dim,
                mesh_device,
                self.weights_dtype,
                weight_cache_path,
                cache_name_prefix,
                device=mesh_device,
            )
            self.gate_proj = weights["gate"]
            self.up_proj = weights["up"]
            self.down_proj = weights["down"]
        elif weight_cache_path is not None:
            logger.debug("Loading weights from cache")
            weights = self._convert_and_cache_weights(
                None,
                emb_dim,
                hidden_dim,
                mesh_device,
                self.weights_dtype,
                weight_cache_path,
                cache_name_prefix,
                device=mesh_device,
            )
            self.gate_proj = weights["gate"]
            self.up_proj = weights["up"]
            self.down_proj = weights["down"]
        else:
            logger.debug("Creating random sharded weights")
            self.gate_proj = self._create_random_sharded_weight(
                (emb_dim, hidden_dim), *self._weight_layout("gate_proj"), dtype=self.weights_dtype
            )
            self.up_proj = self._create_random_sharded_weight(
                (emb_dim, hidden_dim), *self._weight_layout("up_proj"), dtype=self.weights_dtype
            )
            self.down_proj = self._create_random_sharded_weight(
                (hidden_dim, emb_dim), *self._weight_layout("down_proj"), dtype=self.weights_dtype
            )

    def _init_fused_expert(self):
        """Validate the config against unified_routed_expert_moe and build its index tensors."""
        if not is_blackhole():
            raise ValueError("TtSharedExpert runs shared_expert_ffn, which is Blackhole-only")
        if self.activation == ACTIVATION_SITU and (self.situ_beta, self.situ_linear_beta) != _FUSED_SITU_BETAS:
            raise ValueError(
                f"the fused op bakes SiTU betas {_FUSED_SITU_BETAS}, got {(self.situ_beta, self.situ_linear_beta)}"
            )
        if (
            self.activation == ACTIVATION_CLAMPED_SILU_GLU
            and self.clamped_silu_glu_limit != _FUSED_CLAMPED_SILU_GLU_LIMIT
        ):
            raise ValueError(
                f"the fused op bakes clamp limit {_FUSED_CLAMPED_SILU_GLU_LIMIT}, got {self.clamped_silu_glu_limit}"
            )
        self.fused_activation = _FUSED_ACTIVATIONS[self.activation]
        # The op's grid starts at (0, 0) unless moved; the dispatch sub-device owns those cores.
        self.core_grid_origin = (
            self.subdevice_cores.bounding_box().start if self.subdevice_cores is not None else ttnn.CoreCoord(0, 0)
        )
        # The one expert's layout for the unified routed-expert program: all tokens, expert 0, row 0.
        if self.seq_len_per_chip is None:
            raise ValueError("TtSharedExpert needs seq_len_per_chip to describe its single expert")
        self.expert_token_counts = self._uint32_vector(self.seq_len_per_chip)
        self.global_expert_idx_table = self._uint32_vector(0)
        self.expert_region_offsets = self._uint32_vector(0)

    def _uint32_vector(self, value: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _cache_name(self, name: str) -> Optional[str]:
        if self.weight_cache_path is None or self.cache_name_prefix is None:
            return None
        return str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")

    def _to_sharded_ttnn(self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType) -> ttnn.Tensor:
        """
        Convert torch weight to sharded ttnn tensor.

        Args:
            torch_weight: PyTorch weight tensor in TTNN format [in_features, out_features]
            dims: Sharding dimensions for mesh_mapper (e.g., (None, -1) or (-2, None))
            name: Weight name for logging
            dtype: Data type for the weight tensor

        Returns:
            Sharded ttnn tensor
        """
        logger.debug(f"Creating sharded weight {name} with dims={dims}, shape={torch_weight.shape}")

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=self.mesh_device.shape,
            dims=dims,
        )

        tt_weight = ttnn.as_tensor(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=self._cache_name(name),
        )

        logger.debug(f"Created {name}: {tt_weight.shape}")
        return tt_weight

    def _create_sharded_weight_from_torch(
        self, torch_weight: torch.Tensor, dims: tuple, name: str, dtype: ttnn.DataType
    ) -> ttnn.Tensor:
        """
        Convert HuggingFace torch weight to sharded ttnn tensor.

        HF/PyTorch nn.Linear weights are [out_features, in_features], but TTNN matmul(x, W)
        expects [in_features, out_features], so we transpose before sharding.
        """
        torch_weight = torch_weight.T.contiguous()
        return self._to_sharded_ttnn(torch_weight, dims, name, dtype)

    def _create_random_sharded_weight(self, shape: tuple, dims: tuple, name: str, dtype: ttnn.DataType) -> ttnn.Tensor:
        """
        Create random sharded weight in TTNN format [in_features, out_features].
        """
        torch_weight = torch.randn(*shape, dtype=torch.float32)
        return self._to_sharded_ttnn(torch_weight, dims, name, dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: one fused op computing, on every chip, its mesh column's emb slice of the expert.

        Args:
            x: Input tensor [1, seq_len, emb_dim] (replicated across mesh columns)

        Returns:
            Output tensor [1, seq_len, emb_dim / num_cols], bfloat8_b
        """
        if DEBUG_LOGGING_ENABLED:
            logger.debug(f"Forward pass: input shape={x.shape}")

        assert (
            x.shape[-1] == self.emb_dim
        ), f"Input must be replicated (full emb_dim={self.emb_dim}), got shape[-1]={x.shape[-1]}"
        assert (
            x.shape[-2] == self.seq_len_per_chip
        ), f"Expected seq_len_per_chip={self.seq_len_per_chip} tokens, got {x.shape[-2]}"

        # Reads x as given (no typecast) and writes a fresh bf8 output, never over x, which the MoE
        # dispatch reads concurrently.
        output = ttnn.experimental.deepseek_prefill.shared_expert_ffn(
            x,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            self.expert_token_counts,
            self.global_expert_idx_table,
            self.expert_region_offsets,
            compute_kernel_config=self.compute_kernel_config,
            activation=self.fused_activation,
            core_grid_origin=self.core_grid_origin,
        )
        if DEBUG_LOGGING_ENABLED:
            logger.debug(f"After shared_expert_ffn: {output.shape}")

        return output
