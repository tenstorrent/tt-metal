# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Routed Expert module for processing dispatched tokens.

This module processes tokens that have been dispatched to local experts.
Unlike TtSharedExpert, this module:
- Does NOT use CCL (no all-gather, no reduce-scatter)
- Processes tokens that are already dispatched to each device
- Each device holds weights for `experts_per_chip` local experts
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

# Model configs are torch-only and so name their activation as a string; this is the one place
# that maps those names onto the kernel enum. Keys match the HF ``hidden_act`` spelling.
ROUTED_EXPERT_ACTIVATION_BY_NAME = {
    "silu": ttnn.RoutedExpertActivation.Silu,
    "swiglu_oai": ttnn.RoutedExpertActivation.SwiGluOai,
    "situ": ttnn.RoutedExpertActivation.SituGlu,
}

# Activations whose fused kernel path carries the bias branch (gate/up bias before the
# activation, down bias after the down matmul). SiLU has no bias branch.
_BIAS_CAPABLE_ACTIVATIONS = (
    ttnn.RoutedExpertActivation.SwiGluOai,
    ttnn.RoutedExpertActivation.SituGlu,
)

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


class TtRoutedExpert(LightweightModule):
    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str, experts_per_chip: int) -> bool:
        """Check if all routed expert weight cache files exist."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for local_expert_idx in range(experts_per_chip):
            for proj in ["gate", "up", "down"]:
                pattern = f"{cache_name_prefix}.local_{local_expert_idx}_{proj}*.tensorbin"
                if not pattern_exists(pattern, "RoutedExpert"):
                    logger.debug(f"TTNN cache missing: {cache_name_prefix}.local_{local_expert_idx}_{proj}")
                    return False
        return True

    @staticmethod
    def dram_nd_shard_spec(mesh_device, n_dim: int) -> "ttnn.NdShardSpec":
        """DRAM ND shard spec that lets UnifiedRoutedExpertFfn fetch a whole K-row
        weight slice in ONE NoC request instead of one per tile.

        The shard is a single tile-row tall and per_core_N tiles wide, i.e. exactly the
        slice one FFN core consumes for one K-row. Two properties matter:

        * WIDTH: matching per_core_N is what makes the slice a single shard, hence one
          contiguous request. per_core_N = ceil(n_tiles / FFN_GRID_X) mirrors the op's
          own N split; the op validates the width and fails loudly if they diverge.
        * HEIGHT of exactly one tile-row: shards are distributed ROUND_ROBIN_1D, so
          shard id = k * shard_grid_n + gx and consecutive K-rows land in DIFFERENT DRAM
          banks. That rotation is the whole point — measured on a P150, a core pinned to
          one bank saturates near 30 GB/s no matter how big the request (13/27/55 KB all
          gave ~245 GB/s aggregate), while the same bytes with the bank rotating reach
          ~370 GB/s. A taller shard would be one request per K-BLOCK but would pin the
          core to a bank, which measured no faster than plain interleaved. It would also
          couple this spec to the op's in0_block_w, which its L1 guard can lower.

        n_tiles need not be a multiple of per_core_N: the last shard is partially valid
        and those columns are dropped by the op's existing N-bounds guards.
        """
        # UnifiedRoutedExpertFfn's N-parallel grid width. Must match the op's GRID_X or the
        # op's check_shard_width TT_FATAL fires (per_core_N is derived from it on both sides),
        # which is exactly what caught a GRID_X=12 experiment where only this side was 11.
        FFN_GRID_X = 11
        n_tiles = n_dim // ttnn.TILE_SIZE
        per_core_n = (n_tiles + FFN_GRID_X - 1) // FFN_GRID_X
        dram_grid = mesh_device.dram_grid_size()
        return ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([ttnn.TILE_SIZE, per_core_n * ttnn.TILE_SIZE]),
            grid=ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))]
            ),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    @staticmethod
    def _convert_and_cache_expert_weights(
        torch_weights: list[dict] | None,
        experts_per_chip: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,
        *,
        emb_dim: int | None = None,
        hidden_dim: int | None = None,
        dram_sharded: bool = False,
    ):
        """
        Shared logic for converting expert weights to ttnn with caching.

        Args:
            torch_weights: List of expert weight dicts, or None for cache-only loading.
                When None, emb_dim and hidden_dim must be provided.
            experts_per_chip: Number of experts per chip (8 for 8x4 mesh)
            mesh_device: Mesh device reference
            weights_dtype: Weight data type
            cache_path: Cache directory
            cache_name_prefix: Prefix for cache files
            device: None for cache-only, mesh_device for cache+load
            emb_dim: Required when torch_weights is None
            hidden_dim: Required when torch_weights is None

        Returns:
            (gate_projs, up_projs, down_projs) if device is not None, else None
        """
        from tqdm import tqdm

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        mesh_rows, mesh_cols = mesh_device.shape
        gate_tensors, up_tensors, down_tensors = [], [], []

        mode = "build-cache" if device is None else ("load-cache" if torch_weights is None else "convert")
        for local_expert_idx in tqdm(range(experts_per_chip), desc=f"Expert weights ({mode})"):
            if torch_weights is not None:
                gate_weights, up_weights, down_weights = ExpertMapping.gather_weights_for_mesh_distribution(
                    torch_weights, local_expert_idx, mesh_rows, mesh_cols, experts_per_chip
                )

                stacked_gate = torch.stack([w.T.contiguous() for w in gate_weights], dim=0)
                in_f, out_f = stacked_gate.shape[1], stacked_gate.shape[2]
                stacked_gate = stacked_gate.reshape(mesh_rows, mesh_cols, in_f, out_f)

                stacked_up = torch.stack([w.T.contiguous() for w in up_weights], dim=0).reshape(
                    mesh_rows, mesh_cols, in_f, out_f
                )

                stacked_down = torch.stack([w.T.contiguous() for w in down_weights], dim=0)
                in_f_down, out_f_down = stacked_down.shape[1], stacked_down.shape[2]
                stacked_down = stacked_down.reshape(mesh_rows, mesh_cols, in_f_down, out_f_down)
            else:
                assert emb_dim is not None and hidden_dim is not None
                stacked_gate = torch.empty(mesh_rows, mesh_cols, emb_dim, hidden_dim)
                stacked_up = torch.empty(mesh_rows, mesh_cols, emb_dim, hidden_dim)
                stacked_down = torch.empty(mesh_rows, mesh_cols, hidden_dim, emb_dim)

            mem = ttnn.DRAM_MEMORY_CONFIG if device else None
            mapper = ExpertMapping.get_weights_mesh_mapper(mesh_device)

            gate_tt = ttnn.as_tensor(
                stacked_gate,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_gate"),
            )
            up_tt = ttnn.as_tensor(
                stacked_up,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_up"),
            )
            down_tt = ttnn.as_tensor(
                stacked_down,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=mem,
                cache_file_name=_cache_name(f"local_{local_expert_idx}_down"),
            )

            if device is None:
                del gate_tt, up_tt, down_tt
            else:
                gate_tt = ttnn.squeeze(ttnn.squeeze(gate_tt, dim=0), dim=0)
                up_tt = ttnn.squeeze(ttnn.squeeze(up_tt, dim=0), dim=0)
                down_tt = ttnn.squeeze(ttnn.squeeze(down_tt, dim=0), dim=0)

                # DRAM ND-sharded weights let the FFN reader fetch a whole K-row weight
                # slice in ONE NoC request instead of one per tile (see
                # dram_nd_shard_spec). Done as a device-side reshard here, after the
                # normal interleaved build and the squeeze to 2D, rather than by handing
                # as_tensor an ND memory config: the mesh-mapper path rank-squeezes the
                # 4D weight and ND-sharded tensors reject that view. It also keeps the
                # on-disk cache layout-independent — the cached tensor stays interleaved,
                # so switching layouts needs no cache rebuild.
                if dram_sharded:
                    gate_tt = ttnn.to_memory_config(
                        gate_tt,
                        ttnn.MemoryConfig(
                            buffer_type=ttnn.BufferType.DRAM,
                            nd_shard_spec=TtRoutedExpert.dram_nd_shard_spec(mesh_device, gate_tt.shape[-1]),
                        ),
                    )
                    up_tt = ttnn.to_memory_config(
                        up_tt,
                        ttnn.MemoryConfig(
                            buffer_type=ttnn.BufferType.DRAM,
                            nd_shard_spec=TtRoutedExpert.dram_nd_shard_spec(mesh_device, up_tt.shape[-1]),
                        ),
                    )
                    down_tt = ttnn.to_memory_config(
                        down_tt,
                        ttnn.MemoryConfig(
                            buffer_type=ttnn.BufferType.DRAM,
                            nd_shard_spec=TtRoutedExpert.dram_nd_shard_spec(mesh_device, down_tt.shape[-1]),
                        ),
                    )

                gate_tensors.append(gate_tt)
                up_tensors.append(up_tt)
                down_tensors.append(down_tt)

        return (gate_tensors, up_tensors, down_tensors) if device else None

    @staticmethod
    def _convert_expert_biases(
        torch_biases: list[dict],
        experts_per_chip: int,
        mesh_device: ttnn.MeshDevice,
        bias_dtype=ttnn.bfloat16,
    ):
        """Convert per-expert gate/up/down biases to mesh-distributed ttnn tensors.

        Each torch bias dict has keys gate_proj_bias/up_proj_bias/down_proj_bias (1D,
        length = projection N). Returns (gate, up, down) lists of one (1, N) TILE tensor
        per local expert, distributed across the mesh exactly like the routed weights
        (reusing the weight gather + mesh mapper by remapping the bias keys).
        """
        mesh_rows, mesh_cols = mesh_device.shape
        mapper = ExpertMapping.get_weights_mesh_mapper(mesh_device)
        # Remap bias keys so the weight-distribution gather can be reused verbatim.
        as_weights = [
            {
                "gate_proj": d["gate_proj_bias"],
                "up_proj": d["up_proj_bias"],
                "down_proj": d["down_proj_bias"],
            }
            for d in torch_biases
        ]

        def _to_tt(per_pos_biases):
            # each entry is 1D (N,) -> (1, N); stack per mesh position -> (rows, cols, 1, N)
            stacked = torch.stack([b.reshape(1, -1) for b in per_pos_biases], dim=0)
            n = stacked.shape[-1]
            stacked = stacked.reshape(mesh_rows, mesh_cols, 1, n)
            tt = ttnn.as_tensor(
                stacked,
                mesh_mapper=mapper,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                dtype=bias_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.squeeze(ttnn.squeeze(tt, dim=0), dim=0)  # per device: (1, N)

        gate_biases, up_biases, down_biases = [], [], []
        for local_expert_idx in range(experts_per_chip):
            gb, ub, db = ExpertMapping.gather_weights_for_mesh_distribution(
                as_weights, local_expert_idx, mesh_rows, mesh_cols, experts_per_chip
            )
            gate_biases.append(_to_tt(gb))
            up_biases.append(_to_tt(ub))
            down_biases.append(_to_tt(db))
        return gate_biases, up_biases, down_biases

    @staticmethod
    def build_ttnn_cache(
        torch_weights: list[dict],
        experts_per_chip: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Build TTNN cache for routed experts without device copy."""
        TtRoutedExpert._convert_and_cache_expert_weights(
            torch_weights, experts_per_chip, mesh_device, weights_dtype, cache_path, cache_name_prefix, device=None
        )

    """
    TTNN implementation of Routed Expert module.

    Processes dispatched tokens through local experts. Each device holds
    `experts_per_chip` experts and processes the tokens dispatched to them.

    Architecture (per expert):
        gate_out = x @ gate_proj
        up_out = x @ up_proj
        activated = silu(gate_out) * up_out
        output = activated @ down_proj

    Weight Layout:
        - Each expert has gate_proj, up_proj, down_proj
        - Weights are NOT sharded across devices (each device has full local expert weights)
        - gate_proj, up_proj: (emb_dim, hidden_dim)
        - down_proj: (hidden_dim, emb_dim)
    """

    def __init__(
        self,
        mesh_device,
        experts_per_chip: int,
        global_expert_idx_table: ttnn.Tensor,
        emb_dim: int = 7 * 1024,
        hidden_dim: int = 2 * 1024,
        max_tokens: int = 1600,
        torch_weights: list[dict] = None,
        torch_biases: list[dict] = None,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        compute_kernel_config: ttnn.WormholeComputeKernelConfig = COMPUTE_KERNEL_CONFIG_LOFI,
        weights_dram_sharded: bool = False,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        *,
        activation: "ttnn.RoutedExpertActivation",
    ):
        """
        Initialize TtRoutedExpert module.

        Args:
            mesh_device: TTNN mesh device
            experts_per_chip: Number of local experts per chip
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden/intermediate dimension (default: 2048)
            max_tokens: Maximum tokens per expert (default: 1600, used for program config).
                          The FFN kernel sizes chunk_M_tiles/per_core_M to each expert's
                          ACTUAL token count at runtime (read device-side), so no expected-
                          token hint is needed.
            torch_weights: Optional list of dicts with keys 'gate_proj', 'up_proj', 'down_proj'
                          containing torch tensors. Length must be num_devices * experts_per_chip
                          (total routed experts), with weights ordered by global expert index.
                          Note: torch weights are in HuggingFace format (out_features, in_features)
                          so they need to be transposed for TTNN matmul.
            activations_dtype: Data type for activations (default: bfloat8_b)
            weights_dtype: Data type for weights (default: bfloat4_b)
            compute_kernel_config: Compute kernel configuration
            global_expert_idx_table: TTNN tensor mapping local expert slots to global expert ids.
                          Produced by sharding ExpertMapping.create_global_expert_idx_table via
                          get_ep_mesh_mapper, so each device holds (1, 1, experts_per_chip) of
                          global ids. Required.
            activation: Required ttnn.RoutedExpertActivation selecting the fused kernel's
                          activation. Pass RoutedExpertActivation.Silu for the DeepSeek path
                          (byte-identical), RoutedExpertActivation.SwiGluOai for the
                          MiniMax-M3 / gpt-oss clamped swigluoai activation, or
                          RoutedExpertActivation.SituGlu for Kimi K3's SiTU-GLU. Keyword-only
                          and without a default so the caller must choose explicitly.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.experts_per_chip = experts_per_chip
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.num_devices = mesh_device.get_num_devices()
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        # DRAM ND-sharded weights: one NoC request per K-row slice instead of one per
        # tile, with the shard->bank round-robin rotating banks across K-rows. See
        # dram_nd_shard_spec.
        self.weights_dram_sharded = weights_dram_sharded
        self.compute_kernel_config = compute_kernel_config
        self.weight_cache_path = weight_cache_path
        self.cache_name_prefix = cache_name_prefix
        self.global_expert_idx_table = global_expert_idx_table
        # Activation variant for the fused unified_routed_expert_moe kernel.
        # Required RoutedExpertActivation, chosen explicitly by the caller (no
        # silent default): pass ttnn.RoutedExpertActivation.Silu for the DeepSeek
        # path (byte-identical), .SwiGluOai for the MiniMax-M3 / gpt-oss clamped
        # swigluoai activation, or .SituGlu for Kimi K3's SiTU-GLU. Enforcing presence
        # avoids silently running the wrong activation when a caller forgets to set it.
        if activation is None:
            raise ValueError(
                "TtRoutedExpert requires an explicit `activation` "
                "(ttnn.RoutedExpertActivation.Silu, .SwiGluOai or .SituGlu)"
            )
        self.activation = activation

        # Every non-SiLU activation lives in the fused Blackhole kernel only; the Wormhole
        # fallback in forward() calls routed_expert_ffn, which has no activation parameter and
        # always computes SiLU. Reject here rather than silently returning SiLU output.
        if activation != ttnn.RoutedExpertActivation.Silu and not is_blackhole():
            raise NotImplementedError(
                f"TtRoutedExpert {activation} is only supported on the Blackhole fused path; "
                "the fallback path computes SiLU"
            )

        # Optional per-expert projection biases (gpt-oss). Supported by any fused binary
        # activation (the kernel adds gate/up bias before the activation and down bias
        # after the down matmul). Converted + distributed like the weights below.
        if torch_biases is not None and activation not in _BIAS_CAPABLE_ACTIVATIONS:
            raise ValueError(
                "TtRoutedExpert expert biases require a fused binary activation "
                "(RoutedExpertActivation.SwiGluOai or .SituGlu); the SiLU path has no bias branch."
            )

        total_experts = self.num_devices * experts_per_chip
        logger.debug(f"Initializing TtRoutedExpert with experts_per_chip={experts_per_chip}")
        logger.debug(f"emb_dim={emb_dim}, hidden_dim={hidden_dim}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}, total_experts={total_experts}")

        # Store weights for each local expert
        # Each expert has (gate_proj, up_proj, down_proj)
        self.gate_projs = []
        self.up_projs = []
        self.down_projs = []

        self.gate_projs_pc = None
        self.up_projs_pc = None
        self.down_projs_pc = None

        if torch_weights is not None:
            assert len(torch_weights) == total_experts, (
                f"Expected {total_experts} expert weights (num_devices={self.num_devices} * "
                f"experts_per_chip={experts_per_chip}), got {len(torch_weights)}"
            )
            logger.debug(f"Creating weights from provided torch tensors ({total_experts} experts)")
            result = self._convert_and_cache_expert_weights(
                torch_weights,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                self.weight_cache_path,
                self.cache_name_prefix,
                device=self.mesh_device,
                dram_sharded=weights_dram_sharded,
            )
        elif weight_cache_path is not None:
            logger.debug(f"Loading weights from cache ({experts_per_chip} local experts)")
            result = self._convert_and_cache_expert_weights(
                None,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                self.weight_cache_path,
                self.cache_name_prefix,
                device=self.mesh_device,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                dram_sharded=weights_dram_sharded,
            )
        else:
            logger.debug(f"Creating dummy tensors for testing ({total_experts} experts)")
            torch_weights = []
            for _ in range(total_experts):
                torch_weights.append(
                    {
                        "gate_proj": torch.empty(hidden_dim, emb_dim),
                        "up_proj": torch.empty(hidden_dim, emb_dim),
                        "down_proj": torch.empty(emb_dim, hidden_dim),
                    }
                )
            result = self._convert_and_cache_expert_weights(
                torch_weights,
                experts_per_chip,
                self.mesh_device,
                self.weights_dtype,
                None,
                None,
                device=self.mesh_device,
                dram_sharded=weights_dram_sharded,
            )

        assert result is not None, "Expected weight tensors to be returned when device is provided"
        self.gate_projs, self.up_projs, self.down_projs = result

        # Convert + distribute optional per-expert biases (gpt-oss), one (1, N)
        # tensor per local expert, mesh-distributed like the weights.
        self.gate_biases = None
        self.up_biases = None
        self.down_biases = None
        if torch_biases is not None:
            assert (
                len(torch_biases) == total_experts
            ), f"Expected {total_experts} expert biases, got {len(torch_biases)}"
            self.gate_biases, self.up_biases, self.down_biases = self._convert_expert_biases(
                torch_biases, experts_per_chip, self.mesh_device
            )

    @staticmethod
    def shard_expert_token_counts(
        mesh_device: ttnn.MeshDevice,
        expert_token_counts: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Convert and shard the expert token counts tensor across mesh devices.

        Args:
            mesh_device: The mesh device to place the tensor on
            expert_token_counts: Total tokens per expert (sparse per group, replicated across dispatch_group_size)
                Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts) - from get_gate_outputs()

        Returns:
            TTNN tensor sharded across mesh devices.
            Per-device shape: (1, num_routed_experts)
        """
        logger.debug(f"[shard_expert_token_counts] INPUT: expert_token_counts.shape={expert_token_counts.shape}")
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(1, 0),
        )
        result = ttnn.from_torch(
            expert_token_counts,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        result = ttnn.squeeze(result, 0)
        logger.debug(f"[shard_expert_token_counts] OUTPUT: result.shape={result.shape}")
        return result

    def _cache_name(self, name: str) -> Optional[str]:
        if self.weight_cache_path is None or self.cache_name_prefix is None:
            return None
        return str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")

    def _create_random_weight(self, shape: tuple, name: str) -> ttnn.Tensor:
        """
        Allocate uninitialized weight tensor on device DRAM (fast, no host transfer).

        Args:
            shape: Weight shape (in_features, out_features) for TTNN matmul
            name: Weight name for logging

        Returns:
            Uninitialized TTNN tensor on device DRAM
        """
        logger.debug(f"Allocating uninitialized weight {name} with shape {shape} on device DRAM")

        tt_weight = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            self.weights_dtype,
            ttnn.TILE_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        return tt_weight

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        On Blackhole, delegates the per-local-expert work to the
        `unified_routed_expert_moe` C++ composite (no Python per-expert loop,
        no host-device count sync). On non-Blackhole archs (Wormhole), falls
        back to the per-expert Python loop with extract → routed_expert_ffn
        → insert.

        Args:
            dispatched_buffer: Dispatched tokens
                shape: (max_dispatch_buffer_token_size, emb_dim)
            expert_token_counts: Token counts per expert per chip
                Shape per device: (1, num_routed_experts).
            expert_region_offsets: Expert region start offsets per expert
                (shared across source devices in a dispatch group). Produced by
                offset_cumsum. Shape per device: (1, num_routed_experts).

        Returns:
            expert_outputs: Expert output tensor, same shape as dispatched_buffer
        """
        logger.debug(f"Forward pass: dispatched_buffer shape={dispatched_buffer.shape}")

        if is_blackhole():
            # Fused path. The composite op selects its strategy from the input
            # layout: a ROW_MAJOR bf16 buffer is consumed directly (x tilized and
            # bf8-packed internally, fresh output); a TILE buffer takes the
            # non-fused read path and is written in place. TILE mode requires x to
            # be bf8, so cast a mismatched TILE input; the ROW_MAJOR fast path is
            # left untouched.
            if dispatched_buffer.layout == ttnn.TILE_LAYOUT and dispatched_buffer.dtype != self.activations_dtype:
                logger.warning(f"{dispatched_buffer.dtype=} typecasting to {self.activations_dtype}")
                dispatched_buffer = ttnn.typecast(dispatched_buffer, self.activations_dtype)
            signpost(header="UnifiedRoutedExpertMoe")
            expert_outputs = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
                dispatched_buffer,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                self.gate_projs,
                self.up_projs,
                self.down_projs,
                max_dispatched_tokens_per_expert=self.max_tokens,
                compute_kernel_config=self.compute_kernel_config,
                activation=self.activation,
                gate_biases=self.gate_biases,
                up_biases=self.up_biases,
                down_biases=self.down_biases,
            )
            logger.debug(f"Final expert_outputs shape: {expert_outputs.shape}")
            return expert_outputs

        if self.gate_biases is not None:
            raise NotImplementedError("Expert bias is only supported on the Blackhole fused path")

        # Wormhole fallback: the per-expert extract → FFN → insert loop needs a
        # TILE activations_dtype buffer. A ROW_MAJOR input is tilized and cast in
        # one to_layout; an already-TILE input only needs the dtype cast (to_layout
        # would not cast a TILE→TILE tensor). expert_outputs aliases this buffer;
        # the insert writes each expert's result back in place.
        if dispatched_buffer.layout == ttnn.TILE_LAYOUT:
            if dispatched_buffer.dtype != self.activations_dtype:
                logger.warning(f"{dispatched_buffer.dtype=} typecasting to {self.activations_dtype}")
                dispatched_buffer = ttnn.typecast(dispatched_buffer, self.activations_dtype)
        else:
            dispatched_buffer = ttnn.to_layout(dispatched_buffer, ttnn.TILE_LAYOUT, dtype=self.activations_dtype)
        expert_outputs = dispatched_buffer
        for local_expert in range(self.experts_per_chip):
            signpost(f"Expert {local_expert+1}/{self.experts_per_chip}")

            tokens = ttnn.experimental.deepseek_prefill.extract(
                dispatched_buffer,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=local_expert,
                max_dispatched_tokens_per_expert=self.max_tokens,
            )
            logger.debug(f"Expert {local_expert}: input shape {tokens.shape}")

            output = ttnn.experimental.deepseek_prefill.routed_expert_ffn(
                tokens,
                self.gate_projs[local_expert],
                self.up_projs[local_expert],
                self.down_projs[local_expert],
                compute_kernel_config=self.compute_kernel_config,
                output=None,
            )
            logger.debug(f"Expert {local_expert}: output shape {output.shape}")

            expert_outputs = ttnn.experimental.deepseek_prefill.insert(
                expert_outputs,
                output,
                expert_region_offsets,
                expert_token_counts,
                self.global_expert_idx_table,
                local_expert_id=local_expert,
            )

        logger.debug(f"Final expert_outputs shape: {expert_outputs.shape}")
        return expert_outputs
