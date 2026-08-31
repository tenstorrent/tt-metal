# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi-K3 "LatentMoE" projections: the shared low-rank pair that wraps the routed experts.

K3's MoE is DeepSeek-V3's MoE run in a reduced latent space. Two shared projections and one RMSNorm
form the boundary:

    x [emb_dim]  --down_proj-->  [routed_emb_dim]  --dispatch/experts/combine/reduce-->
                 [routed_emb_dim]  --norm--> --up_proj-->  [emb_dim]

Shapes for real K3: ``down_proj`` 7168 -> 3584, ``up_proj`` 3584 -> 7168, ``norm`` over 3584. Both
projections are plain bf16 in the checkpoint -- only the routed experts' ``w1/w2/w3`` are MXFP4.

Why this is worth ~10% extra FLOPs: it halves the row width that dispatch moves over fabric AND
halves the per-chip routed-expert weight footprint. Folding the projections into the expert weights
instead (``(Wg_i . W_down) x``) is mathematically valid and strictly worse -- it would restore the
7168 dispatch payload and double expert weight memory, because the factor is 896 experts. Contrast
MLA, which *does* absorb ``wkv_b1`` into Q, a win there because it is per-head over 128 heads.

Tensor-parallel placement, mirroring the surrounding modules:
  * ``down_proj`` is COLUMN-parallel (``dims=(None, -1)``, as ``TtSharedExpert``'s gate/up). Its input
    is the already-all-gathered full-width ``x``, so each device produces ``routed_emb_dim/tp`` and one
    all-gather makes the latent whole again -- dispatch needs complete rows.
  * ``up_proj`` is ROW-parallel (``dims=(None, -2)``, as ``TtSharedExpert``'s down / ``TtFfn``'s down).
    Its input is already ``routed_emb_dim/tp`` because ``TtReduceModule`` fuses the top-k weighted sum
    with a TP reduce-scatter, so it produces full-width partial sums that reduce-scatter back to
    ``emb_dim/tp`` -- exactly the block's output contract.
  * the norm is a ``TtDistributedRmsNorm``, which normalises a TP-sharded tensor using all-gathered
    statistics, so it needs no re-gather of the activations themselves.

Structurally this is the same shape as ``ttMLA._q_a_latent`` (``q_a_proj`` 7168->1536 followed by
``q_a_layernorm`` on the latent); the difference is only which collective each side needs.
"""

import math
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

# Both projections are the shared expert's shapes with a narrower latent -- down_proj is its gate
# (K 224 tiles, N 28) and up_proj its down (K 28, N 224) -- so the block rules measured there apply
# unchanged rather than being re-derived here.
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import _in0_block_w, _out_subblock
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm

# Cache-file basenames. Kept as a tuple so check_cache_complete and the writer cannot drift.
_PROJ_NAMES = ("down_proj", "up_proj")


def _program_config(grid: ttnn.CoreCoord, m: int, k: int, n: int):
    """2D program config for one latent projection, derived for the 640-token prefill chunk.

    Takes the matmul's M, K and N in elements and tiles them here, so callers pass dimensions they
    already hold rather than each dividing by the tile side.

    Both projections otherwise take a default that tiles them badly: the down projection measured
    178.0us on 100 cores against 56.4us for this one on the same 100 cores, and the up projection
    108.4us on 75 cores against 62.8us on 110. Occupancy is only half of it -- the down projection's
    3.2x comes entirely from the per-core block.
    """
    m_tiles, k_tiles, n_tiles = (d // ttnn.TILE_SIZE for d in (m, k, n))
    per_core_M = math.ceil(m_tiles / grid.y)
    per_core_N = math.ceil(n_tiles / grid.x)
    subblock_h, subblock_w = _out_subblock(per_core_M, per_core_N, deep_k=k_tiles >= 2 * n_tiles)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=_in0_block_w(k_tiles),
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


class TtLatentMoeProjections(LightweightModule):
    """The down/up projection pair plus latent RMSNorm that bracket K3's routed experts."""

    def forward(self, *args, **kwargs):
        """Not a single-pass module: the two halves sit on opposite sides of dispatch/combine.

        LightweightModule.__call__ delegates here, so without this override ``module(x)`` would raise
        a bare AttributeError. Use ``to_latent()`` before dispatch and ``from_latent()`` after the
        reduce.
        """
        raise NotImplementedError(
            "TtLatentMoeProjections has no single forward pass; call to_latent() before dispatch and "
            "from_latent() after the reduce."
        )

    # ------------------------------------------------------------------ cache

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str, use_norm: bool = True) -> bool:
        """Check that both projections (and the latent norm, if used) are cached."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for proj in _PROJ_NAMES:
            if not pattern_exists(f"{cache_name_prefix}.{proj}*.tensorbin", "LatentMoeProj"):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{proj}")
                return False
        if use_norm and not TtDistributedRmsNorm.check_cache_complete(cache_path, f"{cache_name_prefix}.norm"):
            logger.debug(f"TTNN cache missing: {cache_name_prefix}.norm")
            return False
        return True

    @staticmethod
    def _require_weight_source(
        torch_weights: dict | None,
        weight_cache_path: Path | None,
        cache_name_prefix: str | None,
        use_norm: bool,
    ) -> None:
        """Refuse to build from nothing. Raises ValueError unless weights or a complete cache exist.

        Load-bearing, not defensive: ``_convert_and_cache_weights``' placeholder branch builds
        ``torch.empty`` -- uninitialised memory -- and ``ttnn.as_tensor`` will both push that to device
        AND persist it as a legitimate-looking cache file on a miss. Silently-random projections would
        then produce plausible-but-wrong outputs with nothing anywhere reporting a problem. The
        placeholder path exists only for the cache-only pass in ``build_ttnn_cache`` (device=None).
        """
        if torch_weights is not None:
            # A missing "norm" reaches TtDistributedRmsNorm's no-weight branch, which fabricates a
            # random gamma. `is None` rather than truthiness: bool() on a tensor raises, and an
            # all-zero weight is legitimate.
            expected_keys = (*_PROJ_NAMES, "norm") if use_norm else tuple(_PROJ_NAMES)
            missing = [k for k in expected_keys if torch_weights.get(k) is None]
            if missing:
                raise ValueError(
                    f"TtLatentMoeProjections: torch_weights entries {missing} are missing or None; "
                    f"expected {list(expected_keys)} (use_norm={use_norm})."
                )
            return
        # A direct glob, not check_cache_complete: that needs the fast_cache_checker global to have
        # been initialised, and would otherwise fail for a reason unrelated to this precondition.
        cache_is_usable = weight_cache_path is not None and cache_name_prefix is not None
        if cache_is_usable:
            expected = [f"{cache_name_prefix}.{proj}*.tensorbin" for proj in _PROJ_NAMES]
            if use_norm:
                expected.append(f"{cache_name_prefix}.norm_weight*.tensorbin")
            cache_is_usable = all(any(Path(weight_cache_path).glob(pat)) for pat in expected)
        if not cache_is_usable:
            raise ValueError(
                "TtLatentMoeProjections requires either torch_weights or a complete on-disk cache, but "
                f"got neither (weight_cache_path={weight_cache_path}, "
                f"cache_name_prefix={cache_name_prefix!r}). Refusing to fall back to torch.empty "
                "placeholders, which would silently become the model's weights. If you got here from "
                "TtPrefillBlock: no state-dict producer emits a 'latent_weights' key yet, so K3 weight "
                "extraction still needs implementing (see extract_layer_state_dict)."
            )

    @staticmethod
    def _convert_and_cache_weights(
        torch_weights: dict | None,
        emb_dim: int,
        routed_emb_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path | None,
        cache_name_prefix: str | None,
        device: ttnn.MeshDevice | None = None,
    ):
        """Convert the two projections to TTNN with caching.

        Args:
            torch_weights: dict with 'down_proj' [routed_emb_dim, emb_dim] and
                'up_proj' [emb_dim, routed_emb_dim], HF ``(out_features, in_features)`` convention.
                None builds correctly-shaped placeholders, for a cache-only pass.
            device: None for cache-only, mesh_device for cache+load.

        Returns:
            dict of ttnn.Tensor when device is not None, else None.
        """

        def _cache_name(name):
            if cache_path is None or cache_name_prefix is None:
                return None
            return str(cache_path / f"{cache_name_prefix}.{name}")

        # TTNN holds (in_features, out_features), so the matmul is x @ W with no runtime transpose.
        if torch_weights is not None:
            down_w = torch_weights["down_proj"].T.contiguous()  # (emb_dim, routed_emb_dim)
            up_w = torch_weights["up_proj"].T.contiguous()  # (routed_emb_dim, emb_dim)
        else:
            down_w = torch.empty(emb_dim, routed_emb_dim)
            up_w = torch.empty(routed_emb_dim, emb_dim)

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

        # (None, -1): shard OUT features across the TP axis -> column-parallel.
        # (None, -2): shard IN  features across the TP axis -> row-parallel.
        # Names come from _PROJ_NAMES so the writer and check_cache_complete cannot drift apart.
        down_name, up_name = _PROJ_NAMES
        down_tt = _to_ttnn(down_w, (None, -1), down_name)
        up_tt = _to_ttnn(up_w, (None, -2), up_name)

        if device is None:
            del down_tt, up_tt
            return None
        return {"down": down_tt, "up": up_tt}

    @staticmethod
    def build_ttnn_cache(
        torch_weights: dict | None,
        emb_dim: int,
        routed_emb_dim: int,
        mesh_device: ttnn.MeshDevice,
        weights_dtype: ttnn.DataType,
        cache_path: Path,
        cache_name_prefix: str,
        use_norm: bool = True,
    ):
        """Write the projection (and latent-norm) caches without copying to device.

        ``torch_weights`` must be a complete dict (down_proj / up_proj [/ norm]); None is accepted only
        when the cache is already complete, in which case ``ttnn.as_tensor`` loads each file and writes
        nothing. Anything else raises rather than persisting placeholders -- see _require_weight_source.
        """
        # Enforced here rather than at the call site because this is public: without it the
        # placeholder branch persists torch.empty as a cache check_cache_complete calls good.
        TtLatentMoeProjections._require_weight_source(torch_weights, cache_path, cache_name_prefix, use_norm)
        TtLatentMoeProjections._convert_and_cache_weights(
            torch_weights,
            emb_dim,
            routed_emb_dim,
            mesh_device,
            weights_dtype,
            cache_path,
            cache_name_prefix,
            device=None,
        )
        if use_norm:
            TtDistributedRmsNorm.build_ttnn_cache(
                torch_weight=torch_weights.get("norm") if torch_weights else None,
                emb_dim=routed_emb_dim,
                mesh_device=mesh_device,
                cache_path=cache_path,
                cache_name_prefix=f"{cache_name_prefix}.norm",
            )

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        emb_dim: int,
        routed_emb_dim: int,
        torch_weights: dict | None = None,
        use_norm: bool = True,
        rms_norm_eps: float = 1e-5,
        weights_dtype: ttnn.DataType = ttnn.bfloat8_b,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        compute_kernel_config=None,
    ):
        """
        Args:
            emb_dim: full model hidden (K3: 7168)
            routed_emb_dim: latent hidden the routed side runs at (K3: 3584)
            torch_weights: dict with down_proj / up_proj / norm; None loads from cache
            use_norm / rms_norm_eps: K3 sets latent_moe_use_norm=True and rms_norm_eps=1e-5.
                Note the eps MUST be passed through: TtDistributedRmsNorm defaults to 1e-6, and
                silently inheriting that would be a quiet accuracy loss.
            num_links / topology: CCL config for the TP-axis collectives (column axis).
        """
        super().__init__()
        self._require_weight_source(torch_weights, weight_cache_path, cache_name_prefix, use_norm)

        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.routed_emb_dim = routed_emb_dim
        self.use_norm = use_norm
        self.num_links = num_links
        self.topology = topology
        self.tp_factor = mesh_device.shape[1]
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.compute_kernel_config = compute_kernel_config

        weights = self._convert_and_cache_weights(
            torch_weights,
            emb_dim,
            routed_emb_dim,
            mesh_device,
            weights_dtype,
            weight_cache_path,
            cache_name_prefix,
            device=mesh_device,
        )
        self.down_proj = weights["down"]
        self.up_proj = weights["up"]

        self.norm = (
            TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=routed_emb_dim,
                epsilon=rms_norm_eps,
                torch_weight=torch_weights.get("norm") if torch_weights else None,
                cluster_axis=1,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"{cache_name_prefix}.norm" if cache_name_prefix else None,
            )
            if use_norm
            else None
        )

        logger.debug(
            f"TtLatentMoeProjections: {emb_dim} -> {routed_emb_dim} -> {emb_dim}, "
            f"tp={self.tp_factor}, use_norm={use_norm}, eps={rms_norm_eps}"
        )

    # --------------------------------------------------------------- forward

    # The tp_factor == 1 branches here and in from_latent are exercised by local runs only; every CI
    # row pins fabric2d-mesh-8x4.
    def to_latent(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """emb_dim (replicated) -> routed_emb_dim (replicated). Runs before dispatch.

        ``x`` is the tensor TtMoe has already all-gathered to full ``emb_dim`` for the shared expert,
        so no collective is needed on the input. The column-parallel matmul leaves the latent sharded
        ``routed_emb_dim/tp``, and dispatch needs whole rows, hence the all-gather on the output.
        Layout and dtype are preserved so dispatch sees exactly the kind of tensor it does today,
        only narrower.
        """
        assert x.shape[-1] == self.emb_dim, (
            f"latent down_proj expects replicated full emb_dim={self.emb_dim}, got shape[-1]={x.shape[-1]}. "
            "It must run AFTER TtMoe's all-gather of x."
        )

        latent = ttnn.matmul(
            x,
            self.down_proj,
            program_config=_program_config(
                self.mesh_device.compute_with_storage_grid_size(),
                x.padded_shape[-2],
                self.emb_dim,
                self.routed_emb_dim // self.tp_factor,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        logger.debug(f"[LatentMoe.to_latent] after down_proj: {latent.shape}")

        if self.tp_factor > 1:
            latent = ttnn.experimental.all_gather_async(
                latent,
                dim=-1,
                cluster_axis=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
                num_links=self.num_links,
                topology=self.topology,
            )
        assert latent.shape[-1] == self.routed_emb_dim, f"{latent.shape=} != ..{self.routed_emb_dim}"
        logger.debug(f"[LatentMoe.to_latent] after all_gather: {latent.shape}")
        return latent

    def from_latent(self, y: ttnn.Tensor) -> ttnn.Tensor:
        """routed_emb_dim/tp -> emb_dim/tp. Runs after the top-k weighted sum.

        Order matters and matches upstream ``KimiSparseMoeBlock.forward``: the RMSNorm sits AFTER the
        weighted sum, so it sees the summed latent rather than per-expert outputs. The norm is
        TP-distributed, so it consumes and returns the sharded tensor unchanged in shape.
        """
        expected_in = self.routed_emb_dim // self.tp_factor
        assert y.shape[-1] == expected_in, (
            f"latent up_proj expects TP-sharded latent {expected_in} "
            f"(= {self.routed_emb_dim}/{self.tp_factor}), got shape[-1]={y.shape[-1]}"
        )
        # The distributed norm is rank-4 only, and a rank-3 input dies inside the op with an index
        # error that names nothing.
        assert len(y.shape) == 4, f"latent from_latent() requires a rank-4 tensor (norm is rank-4 only), got {y.shape}"

        if self.use_norm:
            y = self.norm(y)
            logger.debug(f"[LatentMoe.from_latent] after latent norm: {y.shape}")

        out_full = ttnn.matmul(
            y,
            self.up_proj,
            program_config=_program_config(
                self.mesh_device.compute_with_storage_grid_size(),
                y.padded_shape[-2],
                expected_in,
                self.emb_dim,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        logger.debug(f"[LatentMoe.from_latent] after up_proj: {out_full.shape}")

        if self.tp_factor > 1:
            # Plain reduce_scatter rather than TtSharedExpert's persistent intermediate: that buffer
            # is shaped by its first caller, so a second op with a different dtype would collide.
            out = ttnn.reduce_scatter(
                out_full,
                dim=-1,
                cluster_axis=1,
                num_links=self.num_links,
                topology=self.topology,
            )
        else:
            out = out_full
        logger.debug(f"[LatentMoe.from_latent] after reduce_scatter: {out.shape}")
        return out
