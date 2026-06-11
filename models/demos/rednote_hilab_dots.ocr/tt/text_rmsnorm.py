# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text RMSNorm for dots.ocr (Qwen2RMSNorm, eps=1e-6).

Qwen2RMSNorm: fp32 variance — x * rsqrt(mean(x^2, -1) + eps) — cast back, then
scale by the learned weight. The decomposed pow -> mean -> rsqrt -> mul chain
maps onto the single fused ``ttnn.rms_norm`` op (cf. KB entry ttnn_pow: the
chain "is a fusion candidate into ttnn.rms_norm"); reference_impl
models/common/rmsnorm.py uses the identical fused op with a
[1, 1, dim//32, 32] ROW_MAJOR gamma.

Parallelism plan (ARCHITECTURE.md / inventory): decoder_stack is sharded
4-way; text_rmsnorm placement=shard via the distributed norm recipe of
models/common/rmsnorm.py / tt_transformers distributed_norm.py, with the
gamma replicated logically (each device holds its hidden slice). Mirroring
the reference_impl, this block carries BOTH paths:

- ``forward``: replicated activation -> fused ``ttnn.rms_norm`` with a
  replicated gamma (the tt_transformers non-TG path; correct whenever the
  decoder keeps a replicated residual stream between CCLs). Optimization
  phase: when the shape qualifies (hidden tiles divisible by 12, per-core
  shard within L1 budget — gated on the PADDED seq so the decode token row
  [1,1,1,H] -> 32 phys rows qualifies too), the norm runs WIDTH-SHARDED on
  a 4x3 grid via ``LayerNormShardedMultiCoreProgramConfig`` with an
  i2s/s2i bounce that keeps the DRAM-interleaved output contract (consumers
  are matmuls; the tick-23 L1-interleaved-into-matmul stall forbids an L1
  pin). Default interleaved LN caps at padded_seq//32 cores (1 @ decode,
  4 @ seq 128); the sharded kernel uses 12. Measured at the production
  operating points: fp32 [1,1,128,1536] 24.4 -> 21.1 us/iter (-13.4%);
  DECODE bf16 [1,1,1,1536] traced 18.57 -> 6.92 us/device (-63%, see grid
  sweep at ``_SHARD_GRID_X``). Larger and smaller grids measured worse
  (block_w=4 with subblock_w=4 is the sweet spot at both shapes).
- ``forward_distributed``: hidden-sharded activation [.., dim/N per device]
  -> ``ttnn.rms_norm_pre_all_gather`` (per-device sum(x^2) stats) ->
  ``ttnn.all_gather(dim=3, Topology.Linear)`` (sync variant, acceptable for
  first-pass correctness per tp-guidance; async in optimization phase) ->
  ``ttnn.rms_norm_post_all_gather`` with the dim-2-sharded gamma (KB entry
  ttnn_rms_norm_post_all_gather). Output stays hidden-sharded.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class TtTextRMSNorm(LightweightModule):
    """dots.ocr text RMSNorm (Qwen2, eps=1e-6) with replicated + distributed paths.

    Args:
        mesh_device: ttnn mesh device handle.
        state_dict: {"weight": [dim]} torch tensor (HF key e.g.
            model.layers.N.input_layernorm.weight).
        dtype: on-device weight dtype.
        eps: RMSNorm epsilon (Qwen2 text decoder uses 1e-6).
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, eps=1e-6):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps

        weight = state_dict["weight"]
        dim = weight.shape[-1]
        # ttnn.rms_norm gamma format: [1, 1, dim//32, 32] in ROW_MAJOR for
        # bf16. The ROW_MAJOR gamma path is bf16-only (an fp32 ROW_MAJOR
        # gamma is misread on device, PCC ~0) — fp32 gammas use TILE
        # [1, 1, 1, dim] instead (cf. tt/vision_rmsnorm.py).
        if dtype == ttnn.float32:
            gamma, gamma_layout = weight.reshape(1, 1, 1, dim), ttnn.TILE_LAYOUT
        else:
            gamma, gamma_layout = weight.reshape(1, 1, dim // TILE, TILE), ttnn.ROW_MAJOR_LAYOUT
        # Replicated gamma for the fused single-op path.
        self.weight = ttnn.from_torch(
            gamma,
            dtype=dtype,
            layout=gamma_layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Hidden-sharded gamma for the distributed path: shard the
        # [1, 1, dim//32, 32] ROW_MAJOR gamma on dim 2 — per-device
        # [1, 1, (dim/N)//32, 32], matching the per-device hidden slice
        # (models/common/rmsnorm.py weight_distributed, ShardTensor2dMesh
        # dims=(None, 2); 1xN line -> plain dim-2 shard).
        self.weight_distributed = ttnn.from_torch(
            weight.reshape(1, 1, dim // TILE, TILE),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        )
        # Reference_impl models/common/rmsnorm.py runs HiFi2 + fp32 dest acc
        # for the norm; init_device_compute_kernel_config picks the
        # arch-correct config struct (Blackhole here).
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Sharded-path configs cached per (seq, dim, dtype-size) — shape-only
        # keys, so control flow stays deterministic under metal trace.
        self._sharded_cfg_cache = {}

    # Width-sharded grid: 12 cores (4x3), block_w = dim_tiles/12. Measured
    # best at BOTH production points:
    # - prefill gate shape (fp32 [1,1,128,1536]): 24.4 -> 21.1 us/iter
    #   (-13.4%) vs default interleaved (which caps at seq//32 cores);
    #   48/24/6-core grids measured worse.
    # - DECODE token row (bf16 [1,1,1,1536] padded to 32 rows, fp32 TILE
    #   gamma, traced replay, occupancy redo tick-62): interleaved fallback
    #   was 1/110 cores at 18.57 us/device; sharded total 6.92 us/device
    #   (LN 5.04 + i2s/s2i bounce 1.88), -63%. Max-core lever A/B at the
    #   queried 11x10 grid: 48c 7.66 / 24c 7.21 / 16c 7.18 / 12c(6x2) 7.14 /
    #   8c 6.90 / 6c 7.32 us/device — the single-row op is latency-bound,
    #   MORE cores measured WORSE; 8-12 cores is the measured ceiling
    #   (<70%-occupancy wave-off backed by the losing max-core A/B).
    #   Composition note (decoder_layer/perf tick): a residual stream kept
    #   WIDTH_SHARDED across the whole decode step would drop the 1.88 us
    #   bounce per norm call (57 calls/step); per-norm the bounce pays for
    #   itself vs the 1-core fallback.
    _SHARD_GRID_X, _SHARD_GRID_Y = 4, 3
    # Per-core shard byte cap: keeps the kernel's static CBs well inside the
    # 1.46 MB per-core L1 for long-seq prefill (gate falls back to the fused
    # interleaved path beyond it).
    _SHARD_PER_CORE_BYTE_CAP = 256 * 1024

    def _sharded_cfgs(self, x: ttnn.Tensor):
        """Return (program_config, sharded_memory_config) or None if the
        shape/layout doesn't qualify for the width-sharded fast path."""
        if x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return None
        # PADDED row count: the decode token row is logical [1,1,1,H] but
        # physical [1,1,32,H] — the sharded kernel works on the padded tile
        # rows (LN has no cross-row interaction, pad rows are discarded by
        # the logical shape). Gating on the logical seq left decode on the
        # 1-core interleaved fallback (occupancy redo: 18.57 -> 6.92 us/dev).
        seq, dim = x.padded_shape[-2], x.shape[-1]
        dtype_bytes = 4 if x.dtype == ttnn.float32 else 2
        key = (seq, dim, dtype_bytes)
        if key in self._sharded_cfg_cache:
            return self._sharded_cfg_cache[key]
        cfg = None
        num_cores = self._SHARD_GRID_X * self._SHARD_GRID_Y
        if seq % TILE == 0 and dim % TILE == 0 and (dim // TILE) % num_cores == 0:
            block_w = dim // TILE // num_cores
            if seq * block_w * TILE * dtype_bytes <= self._SHARD_PER_CORE_BYTE_CAP:
                subblock_w = min(4, block_w)
                while block_w % subblock_w:
                    subblock_w -= 1
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(self._SHARD_GRID_X, self._SHARD_GRID_Y),
                    subblock_w=subblock_w,
                    block_h=seq // TILE,
                    block_w=block_w,
                    inplace=False,
                )
                smc = ttnn.create_sharded_memory_config(
                    (1, 1, seq, dim),
                    ttnn.CoreGrid(y=self._SHARD_GRID_Y, x=self._SHARD_GRID_X),
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                cfg = (pc, smc)
        self._sharded_cfg_cache[key] = cfg
        return cfg

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Replicated path. x: [..., dim] TILE_LAYOUT, replicated across the mesh.

        Returns: same shape, replicated, DRAM interleaved (block-output
        contract — every decoder_layer consumer is a matmul).
        """
        cfgs = self._sharded_cfgs(x)
        if cfgs is None:
            return ttnn.rms_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                compute_kernel_config=self.compute_kernel_config,
            )
        pc, smc = cfgs
        x_sh = ttnn.interleaved_to_sharded(x, smc)
        out_sh = ttnn.rms_norm(
            x_sh,
            epsilon=self.eps,
            weight=self.weight,
            program_config=pc,
            memory_config=smc,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x_sh)
        out = ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_sh)
        return out

    def forward_distributed(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Distributed path. x: [1, 1, seq, dim/N] per device (hidden-sharded).

        Per-device partial sum(x^2) stats are all-gathered so every device
        normalizes by the FULL-hidden variance, then scales by its local
        gamma slice. Returns: same shape, still hidden-sharded.
        """
        stats = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.bfloat16)
        stats = ttnn.all_gather(stats, dim=3, topology=ttnn.Topology.Linear)
        out = ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self.eps,
            weight=self.weight_distributed,
            compute_kernel_config=self.compute_kernel_config,
        )
        stats.deallocate(True)
        return out
