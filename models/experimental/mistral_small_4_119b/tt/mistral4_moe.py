# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral4 Mixture-of-Experts (MoE) layer — fully on device, no host fallback.

Architecture:
  • 128 routed experts  (Mistral4NaiveMoe)
  • 4 active experts per token  (Mistral4TopkRouter)
  • 1 shared expert  (Mistral4MLP, always active)
  • Each expert / shared expert: gate_proj + up_proj → SiLU gate → down_proj
    dimensions: HIDDEN_SIZE(4096) → EXPERT_INTERMEDIATE_SIZE(2048) → HIDDEN_SIZE(4096)

Sparse execution (the only path): the token-compacting
``ttnn.experimental.moe_compute`` pipeline. Expert weights are sharded along the
expert dimension (dim=0) — device k holds experts [k*EPD : (k+1)*EPD] — and only
the top-k experts each token routes to are computed (real sparsity, not a
post-matmul mask). Shared-expert and gate weights are replicated on every device.
Requires a Blackhole 1x8 mesh (P150x8).

Routing + forward (fully on device, no host round-trip):
  gate → softmax → top-k → sum-normalize → mesh_partition (DP-shard tokens+idx)
  → all_to_all_dispatch → moe_compute (grouped expert FFN + combine)
  → ×scores + Σ over k → all_gather (restore replicated) → + shared expert + residual.
  See ``_compute_routing_sparse`` / ``_forward_moe_compute`` / ``_init_moe_compute``.

Weight loading (moe_compute prepared format):
  Hugging Face ``Mistral4NaiveMoe`` (Mistral-Small-4) uses fused parameters:
      ``mlp.experts.gate_up_proj``  [num_experts, 2 * intermediate, hidden]
      ``mlp.experts.down_proj``     [num_experts, hidden, intermediate]
  ``_init_moe_compute`` splits gate vs up, prepares + bf4-quantizes them via
  ``prepare_w0_w1_tensor_for_moe_compute`` / ``prepare_w2_tensor_for_moe_compute``
  (disk-cached). The replicated gate (router) weight is loaded separately.
"""

from __future__ import annotations


import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.constants import (
    EXPERT_INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
    NUM_ACTIVE_EXPERTS,
    NUM_EXPERTS,
    SHARED_EXPERT_INTERMEDIATE_SIZE,
)

# ── Weight loading helpers ─────────────────────────────────────────────────


def _bf16(t: torch.Tensor, scale_inv: torch.Tensor | None = None) -> torch.Tensor:
    """Cast to bfloat16, dequantizing FP8 weights if scale_inv is provided.

    scale_inv may be scalar () or per-expert [N]; it is reshaped to broadcast
    against the weight tensor correctly.
    """
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = t.to(torch.float32)
        if scale_inv is not None:
            s = scale_inv.to(torch.float32)
            # Reshape for broadcasting: expand trailing dims to match weight ndim
            while s.dim() < t.dim():
                s = s.unsqueeze(-1)
            t = t * s
    return t.to(torch.bfloat16).contiguous()


def _load_replicated(
    state_dict: dict,
    key: str,
    transpose: bool,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
    cache_file_name=None,
) -> ttnn.Tensor:
    scale_inv = state_dict.get(key.replace(".weight", ".weight_scale_inv"))
    w = _bf16(state_dict[key], scale_inv)
    if transpose:
        w = w.T.contiguous()
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=cache_file_name,
    )


def _load_norm_weight_1d(
    state_dict: dict,
    key: str,
    dim: int,
    mesh_device: ttnn.MeshDevice,
    cache_file_name=None,
) -> ttnn.Tensor:
    w = _bf16(state_dict[key]).reshape(1, 1, dim)
    return ttnn.as_tensor(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=cache_file_name,
    )


# ── Shared (always-active) Expert MLP ─────────────────────────────────────


class TtMistral4SharedMLP(LightweightModule):
    """Always-active shared expert: gate_proj / up_proj / down_proj."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        prefix: str,
        intermediate_size: int,
        dtype: ttnn.DataType,
        compute_kernel_config,
        cache_dir=None,
    ):
        super().__init__()
        self.compute_kernel_config = compute_kernel_config
        self.intermediate_size = intermediate_size
        self._decode_bf8_output = os.environ.get("MISTRAL4_DECODE_BF8_SHARED_EXPERT", "0") == "1"
        _cf = (lambda key: str(cache_dir / key)) if cache_dir is not None else (lambda _: None)

        # Fused gate+up weight: swiglu(z) = z[:I] * SiLU(z[I:])
        # Place up in the first half and gate in the second so that
        # swiglu produces SiLU(gate_proj(x)) * up_proj(x).
        gate_w = _bf16(
            state_dict[prefix + "gate_proj.weight"],
            state_dict.get(prefix + "gate_proj.weight_scale_inv"),
        )  # [I, H]
        up_w = _bf16(
            state_dict[prefix + "up_proj.weight"],
            state_dict.get(prefix + "up_proj.weight_scale_inv"),
        )  # [I, H]
        gate_up_w = torch.cat([up_w, gate_w], dim=0).T.contiguous()  # [H, 2I]
        self.gate_up_proj = ttnn.as_tensor(
            gate_up_w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(prefix + "gate_up_proj"),
        )  # [HIDDEN_SIZE, 2 * intermediate_size]

        self.down_proj = _load_replicated(
            state_dict,
            prefix + "down_proj.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
            cache_file_name=_cf(prefix + "down_proj.weight"),
        )  # [intermediate_size, HIDDEN_SIZE]

    def forward(self, x: ttnn.Tensor, gu_pc=None, d_pc=None) -> ttnn.Tensor:
        """
        Args:  x: [1, 1, seq, HIDDEN_SIZE]
               gu_pc / d_pc: optional 1D-mcast program configs for the gate_up and
                             down matmuls. Without them the default ~12-core path
                             runs; with them the shared expert hits the same
                             64-core parallelism as the routed experts.
        Returns:   [1, 1, seq, HIDDEN_SIZE]
        """
        # Matmul outputs stay in DRAM (gate_up is [1,1,seq,2I] — largest tensor).
        # Slice/silu/multiply intermediates in L1 so the elementwise chain doesn't
        # round-trip through DRAM and the final down_proj matmul reads an L1 in0.
        _mem = ttnn.L1_MEMORY_CONFIG
        seq_len = x.shape[2]
        I = self.intermediate_size
        gate_up = ttnn.linear(
            x,
            self.gate_up_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=gu_pc,
        )  # [1, 1, seq, 2I] — up in [0:I], gate in [I:2I]

        # Split along the last dim (tile-aligned at I=2048); avoids ttnn.swiglu which
        # pads the seq dim to tile size and breaks non-tile-aligned seq lengths.
        up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, seq_len, I], memory_config=_mem)
        gate = ttnn.slice(gate_up, [0, 0, 0, I], [1, 1, seq_len, 2 * I], memory_config=_mem)
        ttnn.deallocate(gate_up)

        hidden = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=_mem,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        out = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=d_pc,
        )
        ttnn.deallocate(hidden)
        return out

    def forward_decode(self, x: ttnn.Tensor, gu_pc=None, d_pc=None) -> ttnn.Tensor:
        """Decode variant: all activations in L1 (seq=1 tensors are tiny)."""
        _mem = ttnn.L1_MEMORY_CONFIG
        I = self.intermediate_size
        gate_up = ttnn.matmul(
            x,
            self.gate_up_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b if self._decode_bf8_output else ttnn.bfloat16,
            memory_config=_mem,
            program_config=gu_pc,
        )
        up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, 1, I], memory_config=_mem)
        gate = ttnn.slice(gate_up, [0, 0, 0, I], [1, 1, 1, 2 * I], memory_config=_mem)
        ttnn.deallocate(gate_up)
        hidden = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=_mem,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.matmul(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b if self._decode_bf8_output else ttnn.bfloat16,
            memory_config=_mem,
            program_config=d_pc,
        )
        ttnn.deallocate(hidden)
        return out


# ── Main MoE Layer ─────────────────────────────────────────────────────────


class TtMistral4MoELayer(LightweightModule):
    """
    Mistral4 MoE: 128 routed experts (device-sharded) + 1 shared expert.

    Sparse-only via ``ttnn.experimental.moe_compute`` (P150x8, 1x8 mesh):
      - experts_per_device = 128 / num_devices (16 on P150x8)
      - device k holds experts [k*EPD : (k+1)*EPD]
      - Routing (gate → softmax → top-k → normalize) is fully on device.
      - Tokens are DP-sharded, all_to_all_dispatched to their experts, run through
        the grouped expert FFN + combine, weighted by their routing scores, summed
        over k, then all_gathered back to replicated.
      - Shared expert (replicated) + residual are added after combine.
    Both prefill ``forward`` and ``forward_decode`` delegate to ``_forward_moe_compute``.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_prefix: str,
        expert_dtype: ttnn.DataType = ttnn.bfloat16,
        cache_dir=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.num_experts = NUM_EXPERTS
        self.num_active = NUM_ACTIVE_EXPERTS

        assert self.num_experts % self.num_devices == 0, (
            f"num_experts ({self.num_experts}) must be divisible by " f"num_devices ({self.num_devices})"
        )
        self.experts_per_device = self.num_experts // self.num_devices

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # LoFi is safe for bf4 expert weights: quantization error (~0.0625) dominates
        # HiFi2's extra FPU precision, so halving FPU cycles has no meaningful PCC cost.
        self.expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        mlp_prefix = layer_prefix + "mlp."
        _cf = (lambda key: str(cache_dir / key)) if cache_dir is not None else (lambda _: None)
        self._cf = _cf  # reused by _init_moe_compute for the moe_compute weight cache

        # ── Gate (router) weight ───────────────────────────────────────────
        # shape: [num_experts, HIDDEN_SIZE] → [HIDDEN_SIZE, num_experts] for matmul
        # bfloat8_b: saves 18 MB / 12 banks ≈ 1.5 MB/bank across 36 layers.
        self.gate_weight = _load_replicated(
            state_dict,
            mlp_prefix + "gate.weight",
            transpose=True,
            dtype=ttnn.bfloat8_b,
            mesh_device=mesh_device,
            cache_file_name=_cf(mlp_prefix + "gate.weight"),
        )  # [HIDDEN_SIZE, NUM_EXPERTS]

        # Gate correction bias (additive, applied before softmax; uploaded to device or None)
        gate_bias_key = mlp_prefix + "gate.e_score_correction_bias"
        if gate_bias_key in state_dict:
            gate_bias_t = state_dict[gate_bias_key].to(torch.bfloat16).reshape(1, 1, 1, -1)
            self.gate_bias_tt = ttnn.as_tensor(
                gate_bias_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )  # [1, 1, 1, NUM_EXPERTS] replicated on all devices (tiny; skip caching)
        else:
            self.gate_bias_tt = None

        # ── Shared expert ──────────────────────────────────────────────────
        # bfloat8_b: 36 layers × 3 shared-expert weights × 8 MB = 0.86 GB vs 1.73 GB at bf16.
        self.shared_expert = TtMistral4SharedMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            prefix=mlp_prefix + "shared_experts.",
            intermediate_size=SHARED_EXPERT_INTERMEDIATE_SIZE,
            dtype=ttnn.bfloat4_b,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            cache_dir=cache_dir,
        )

        self._expert_pc_cache: dict = {}

        # ── Level-2 sparse path (ttnn.experimental.moe_compute) ─────────────
        # Token-compacting pipeline (dispatch → grouped expert matmul → combine),
        # fully on device, no host fallback. Requires BH 1x8 (P150x8).
        self._init_moe_compute(state_dict, mlp_prefix)

    def _expert_1d_mcast_pc(self, m_tiles: int, k_tiles: int, n_tiles: int):
        """1D-mcast program config for a single non-batched expert matmul on Blackhole.

        Caches by (m, k, n). Finds the largest rectangle within the device grid
        whose core-count exactly divides n_tiles, eliminating idle cores and
        reducing mcast fan-out overhead.
        """
        key = (m_tiles, k_tiles, n_tiles)
        cached = self._expert_pc_cache.get(key)
        if cached is not None:
            return cached
        grid_full = self.mesh_device.compute_with_storage_grid_size()
        max_x, max_y = grid_full.x, grid_full.y

        # Find the largest num_cores (as a rectangle within max_x×max_y) that
        # divides n_tiles exactly, so every selected core has a full tile share.
        best_nc, best_x, best_y = 1, 1, 1
        for py in range(1, max_y + 1):
            for px in range(1, max_x + 1):
                nc = px * py
                if n_tiles % nc == 0 and nc > best_nc:
                    best_nc, best_x, best_y = nc, px, py

        grid_x, grid_y = best_x, best_y
        per_core_M = m_tiles
        per_core_N = n_tiles // best_nc
        # in0_block_w: largest divisor of K that fits, capped at 8.
        in0_block_w = 1
        for cand in (8, 4, 2):
            if k_tiles % cand == 0:
                in0_block_w = cand
                break
        # Subblocks: out_subblock_h * out_subblock_w <= 8 (DST capacity, fp32_dest_acc=False).
        out_subblock_w = 1
        for cand in (4, 2, 1):
            if per_core_N % cand == 0:
                out_subblock_w = cand
                break
        out_subblock_h = 1
        for cand in (4, 2, 1):
            if per_core_M % cand == 0 and cand * out_subblock_w <= 8:
                out_subblock_h = cand
                break
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self._expert_pc_cache[key] = pc
        return pc

    # ── Level-2 sparse MoE (ttnn.experimental.moe_compute) ──────────────────

    def _sharded_l1(self, shape, dtype):
        """L1 HEIGHT_SHARDED config placing a [rows, cols] tensor on the drain tilize core."""
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(self._mc_drain_core, self._mc_drain_core)})
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [shape[0], shape[1]], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _init_moe_compute(self, state_dict: dict, mlp_prefix: str) -> None:
        """Build moe_compute weights + scaffolding once. On-device; no host fallback in forward."""
        from ttnn.operations.ccl import MoEActivationFunction
        from ttnn.experimental.moe_compute_utils import (
            auto_output_width_shard_dim,
            effective_matmul_ring_size,
            get_tilize_drain_core,
        )

        md = self.mesh_device
        D = self.num_devices
        EPD = self.experts_per_device
        H = HIDDEN_SIZE
        I = EXPERT_INTERMEDIATE_SIZE

        self._mc_bh_ring_size = 8  # P150 has 8 DRAM banks
        ring_n = effective_matmul_ring_size(md, self._mc_bh_ring_size)
        self._mc_out_h = 4
        self._mc_out_w = auto_output_width_shard_dim(H, matmul_ring_size=ring_n)
        # mistral4 expert = up * SiLU(gate) == moe_compute SILU(silu(x@w0)*(x@w1)) with w0=gate, w1=up.
        self._mc_activation = MoEActivationFunction.SILU
        self._mc_drain_core = get_tilize_drain_core()

        # Per-bank/per-expert DRAM-sharded mem configs (cheap — no weights touched).
        wmc = ttnn.experimental.get_weight_mem_configs(
            md,
            num_layers=1,
            experts_per_device=EPD,
            hidden_size=H,
            intermediate_size=I,
            has_bias=False,
            bh_ring_size=self._mc_bh_ring_size,
        )

        # ── Prepared + bf4-quantized routed-expert weights (disk-cached) ────
        # Building these is expensive: host FP8→bf16 dequant of all 128 experts +
        # prepare_* + quantize_weights_via_host, per layer. The dense path caches its
        # stacked experts via cache_file_name; mirror that here with dump/load_tensor so
        # only the first run pays the cost. load_tensor restores the sharded layout to
        # the mesh; reapply the target memcfg if the on-disk layout differs (DS idiom).
        cf = getattr(self, "_cf", lambda _: None)
        w0w1_cache = cf(mlp_prefix + "mc_w0w1_bf4.tensorbin")  # dump_tensor requires .tensorbin
        w2_cache = cf(mlp_prefix + "mc_w2_bf4.tensorbin")

        def _load_cached(path, target_memcfg):
            t = ttnn.load_tensor(path, device=md)
            if t.memory_config().memory_layout != target_memcfg.memory_layout:
                t = ttnn.to_memory_config(t, target_memcfg)
            return t

        if w0w1_cache and w2_cache and os.path.exists(w0w1_cache) and os.path.exists(w2_cache):
            self._mc_w0w1 = _load_cached(w0w1_cache, wmc.w0_w1)
            self._mc_w2 = _load_cached(w2_cache, wmc.w2)
        else:
            # HF Mistral4NaiveMoe: experts.gate_up_proj [E, 2I, H] = [gate; up];
            # experts.down_proj [E, H, I]. moe_compute wants w0/w1 = (L,E,K=H,N=I),
            # w2 = (L,E,N=I,K=H).
            gu = _bf16(
                state_dict[mlp_prefix + "experts.gate_up_proj"],
                state_dict.get(mlp_prefix + "experts.gate_up_proj_scale_inv"),
            )
            w0 = gu[:, :I, :].permute(0, 2, 1).contiguous().unsqueeze(0)  # gate → [1,E,H,I]
            w1 = gu[:, I:, :].permute(0, 2, 1).contiguous().unsqueeze(0)  # up   → [1,E,H,I]
            dn_key = mlp_prefix + "experts.down_proj"
            if dn_key not in state_dict:
                dn_key = mlp_prefix + "experts.down_proj.weight"
            dn = _bf16(state_dict[dn_key], state_dict.get(dn_key.replace(".weight", "") + "_scale_inv"))
            w2 = dn.permute(0, 2, 1).contiguous().unsqueeze(0)  # down → [1,E,I,H]

            def _shard_experts(t):
                # [1, NUM_EXPERTS, *, *] sharded on expert dim → [1, EPD, *, *] per device.
                return ttnn.from_torch(
                    t, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(md, dim=1)
                )

            tt_w0, tt_w1, tt_w2 = _shard_experts(w0), _shard_experts(w1), _shard_experts(w2)

            w0w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(
                tt_w0, tt_w1, L=1, E=EPD, K=H, N=I, bh_ring_size=self._mc_bh_ring_size
            )
            ttnn.deallocate(tt_w0)
            ttnn.deallocate(tt_w1)
            self._mc_w0w1 = ttnn.experimental.quantize_weights_via_host(
                w0w1_prepped, dtype=ttnn.bfloat4_b, memory_config=wmc.w0_w1
            )
            ttnn.deallocate(w0w1_prepped)
            w2_prepped = ttnn.experimental.prepare_w2_tensor_for_moe_compute(
                tt_w2, L=1, E=EPD, N=I, K=H, bh_ring_size=self._mc_bh_ring_size
            )
            ttnn.deallocate(tt_w2)
            self._mc_w2 = ttnn.experimental.quantize_weights_via_host(
                w2_prepped, dtype=ttnn.bfloat4_b, memory_config=wmc.w2
            )
            ttnn.deallocate(w2_prepped)

            if w0w1_cache:
                ttnn.dump_tensor(w0w1_cache, self._mc_w0w1)
            if w2_cache:
                ttnn.dump_tensor(w2_cache, self._mc_w2)

        # ── Expert→device mappings (expert e on device e // EPD) ────────────
        disp = torch.zeros(1, 1, self.num_experts, D, dtype=torch.int16)  # one-hot, for all_to_all_dispatch
        mc = torch.zeros(D, self.num_experts, dtype=torch.int16)  # device-index, for moe_compute
        for e in range(self.num_experts):
            disp[0, 0, e, e // EPD] = 1
            mc[:, e] = e // EPD
        self._mc_dispatch_mapping = ttnn.from_torch(
            disp,
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        self._mc_mapping = ttnn.from_torch(
            mc,
            device=md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )

        # ── Combine scaffolding (cores, barrier semaphore, mux cores) ───────
        out_cores = ttnn.experimental.get_moe_combine_cores(md, self._mc_out_h, self._mc_out_w)
        combine_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in out_cores])
        self._mc_semaphore = ttnn.create_global_semaphore(md, combine_crs, 0)
        self._mc_mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])

        # Cache of round-robin expert indices for the padding rows, keyed by
        # (S_orig, S, k). Built once per shape (see _balanced_padding_idx).
        self._mc_pad_idx_cache = {}
        # Cache of the moe_compute combine-output buffer, keyed by S. Allocated once
        # and reused (moe_compute overwrites it), so the decode step holds no per-call
        # allocation/host write — required for trace capture (see _combine_out_buffer).
        self._mc_combine_cache = {}

    def _combine_out_buffer(self, k: int, S: int, H: int):
        """Persistent combine-output buffer for moe_compute: global [k,S,H] sharded on
        dim1 (per device [k, tpd, H]), zero-initialised, built fully on device.

        Allocated once per S and reused — moe_compute fully overwrites it each call, so
        reuse is safe (matches DeepSeek's preallocated combine output). Reusing it instead
        of a per-call ``ttnn.zeros`` keeps the decode step free of the buffer write that a
        captured trace forbids ("Writes are not supported during trace capture")."""
        cached = self._mc_combine_cache.get(S)
        if cached is not None:
            return cached
        _drm = ttnn.DRAM_MEMORY_CONFIG
        _z = ttnn.zeros(
            [k, S, H], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device, memory_config=_drm
        )
        buf = ttnn.mesh_partition(_z, dim=1, cluster_axis=1, memory_config=_drm)
        ttnn.deallocate(_z)
        self._mc_combine_cache[S] = buf
        return buf

    def _padding_rr_idx(self, S_orig: int, S: int, k: int):
        """Round-robin expert indices [1,1,S-S_orig,k] (uint16) for the padding rows.

        Padding row r is assigned experts ((r*k)..(r*k+k-1)) mod NUM_EXPERTS, so the
        padding tokens collectively touch every expert — and therefore every device —
        keeping the all_to_all_dispatch / moe_compute combine balanced. Built once per
        shape and cached. Constructed fully on device (ttnn.arange % NUM_EXPERTS) — no
        torch/host tensor — so it is safe under trace capture as well."""
        key = (S_orig, S, k)
        cached = self._mc_pad_idx_cache.get(key)
        if cached is not None:
            return cached
        npad = S - S_orig
        _drm = ttnn.DRAM_MEMORY_CONFIG
        # Reshape while float32 (reshape rejects uint16), then do the elementwise mod +
        # typecast in TILE, and untilize last → uint16 ROW_MAJOR [1,1,npad,k].
        rr = ttnn.arange(0, npad * k, 1, dtype=ttnn.float32, device=self.mesh_device, memory_config=_drm)
        rr = ttnn.reshape(rr, [1, 1, npad, k])
        rr = ttnn.to_layout(rr, ttnn.TILE_LAYOUT, memory_config=_drm)
        rr = ttnn.remainder(rr, float(self.num_experts), memory_config=_drm)
        rr = ttnn.typecast(rr, ttnn.uint16, memory_config=_drm)
        rr = ttnn.to_layout(rr, ttnn.ROW_MAJOR_LAYOUT, memory_config=_drm)
        self._mc_pad_idx_cache[key] = rr
        return rr

    def _balanced_padding_idx(self, topk_idx: ttnn.Tensor, S_orig: int, S: int, k: int) -> ttnn.Tensor:
        """Replace the padding rows [S_orig:S] of topk_idx with round-robin expert
        indices so every device receives dispatched tokens.

        A single real token (decode) — or a small ragged prefill chunk — otherwise
        routes to only a few experts on one device, starving the rest; the moe_compute
        combine's per-device line/ring reduction then deadlocks on mismatched wait
        counts. Padding outputs are sliced off downstream, so balancing them is free.
        Consumes ``topk_idx`` and returns a new [1,1,S,k] uint16 TILE tensor — same
        layout as the input, so the downstream ``to_layout(ROW_MAJOR)`` calls still
        copy rather than alias-and-free the shared buffer. (The reference impls keep
        indices ROW_MAJOR throughout; this model's working prefill path tilizes, so we
        match that here rather than refactor it.)"""
        _drm = ttnn.DRAM_MEMORY_CONFIG
        idx_rm = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=_drm)
        ttnn.deallocate(topk_idx)
        real = ttnn.slice(idx_rm, [0, 0, 0, 0], [1, 1, S_orig, k], memory_config=_drm)
        ttnn.deallocate(idx_rm)
        pad = self._padding_rr_idx(S_orig, S, k)
        merged = ttnn.concat([real, pad], dim=2, memory_config=_drm)
        ttnn.deallocate(real)
        merged_tile = ttnn.to_layout(merged, ttnn.TILE_LAYOUT, memory_config=_drm)
        ttnn.deallocate(merged)
        return merged_tile

    def _compute_routing_sparse(self, x: ttnn.Tensor, seq_len: int):
        """Gate → softmax → top-k → sum-normalize, fully on device. No host round-trip.

        Returns topk_vals [1,1,seq,k] (bf16) and topk_idx [1,1,seq,k] (topk index dtype).
        """
        m_tiles = (seq_len + 31) // 32
        gate_pc = self._expert_1d_mcast_pc(m_tiles, HIDDEN_SIZE // 32, NUM_EXPERTS // 32)
        logits = ttnn.linear(
            x,
            self.gate_weight,
            bias=self.gate_bias_tt,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=gate_pc,
        )
        probs = ttnn.softmax(logits, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(logits)
        vals, idx = ttnn.topk(probs, k=self.num_active, dim=-1)
        ttnn.deallocate(probs)
        s = ttnn.sum(vals, dim=-1, keepdim=True)
        vals = ttnn.div(vals, s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        return vals, idx

    def _forward_moe_compute(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Level-2 sparse prefill: dispatch → moe_compute (grouped expert FFN + combine) → weight+sum.

        Args:  x: [1, 1, seq, HIDDEN_SIZE] replicated.   Returns: [1, 1, seq, HIDDEN_SIZE] replicated.
        """
        md = self.mesh_device
        D = self.num_devices
        H = HIDDEN_SIZE
        I = EXPERT_INTERMEDIATE_SIZE
        k = self.num_active
        _drm = ttnn.DRAM_MEMORY_CONFIG

        # Pad the sequence so each device gets a tile-aligned token count: total_tokens must
        # be a multiple of (num_devices × TILE) so mesh_partition splits evenly and the
        # per-device token count is tile-aligned for moe_compute. Padding tokens (zeros) flow
        # through routing+experts and are sliced off at the end.
        S_orig = x.shape[2]
        align = D * ttnn.TILE_SIZE
        S = ((S_orig + align - 1) // align) * align
        if S != S_orig:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, S - S_orig), (0, 0)], value=0.0)
        tpd = S // D

        # Keep x in DRAM: moe_compute statically reserves most of L1 for circular buffers
        # across the worker cores, so any replicated L1 activation held live across the
        # moe_compute call (x is reused for the gate and the shared expert) clashes with
        # those CBs in the full model. The gate / mesh_partition / shared-expert reads all
        # work from DRAM.
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        topk_vals, topk_idx = self._compute_routing_sparse(x, S)  # [1,1,S,k]
        topk_idx = ttnn.typecast(topk_idx, ttnn.uint16)
        if S != S_orig:
            # Balance the padding rows' routing so every device gets tokens (else the
            # combine's cross-device reduction deadlocks; see _balanced_padding_idx).
            topk_idx = self._balanced_padding_idx(topk_idx, S_orig, S, k)

        # ── DP-shard tokens + indices, then all_to_all_dispatch ─────────────
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=_drm)
        x_dp = ttnn.mesh_partition(x_rm, dim=2, cluster_axis=1, memory_config=_drm)  # [1,1,tpd,H]
        ttnn.deallocate(x_rm)
        x_disp = ttnn.reshape(x_dp, [tpd, 1, 1, H])
        ttnn.deallocate(x_dp)

        idx_rm = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=_drm)
        idx_dp = ttnn.mesh_partition(idx_rm, dim=2, cluster_axis=1, memory_config=_drm)  # [1,1,tpd,k]
        ttnn.deallocate(idx_rm)
        idx_disp = ttnn.reshape(idx_dp, [tpd, 1, 1, k])
        ttnn.deallocate(idx_dp)

        sparse_buf, _meta = ttnn.all_to_all_dispatch(
            x_disp,
            idx_disp,
            self._mc_dispatch_mapping,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_concat_dim=1,
        )
        ttnn.deallocate(x_disp)
        ttnn.deallocate(idx_disp)
        ttnn.deallocate(_meta)  # dispatch metadata unused (combine routing comes from idx/mapping)
        # Dispatch output is [1, S, 1, H] per device, sharded on dim0 (→ global
        # [num_devices, S, 1, H]). moe_compute expects the sparse buffer as the dim0-shard
        # of [num_devices, S, H], i.e. per device [1, S, H]. Squeeze ONLY the singleton seq
        # dim (dim2): collapsing to 2D [S, H] makes the mesh treat dim0 (=S) as the shard →
        # wrong inferred global token count → oversized single-core internal buffer (OOM).
        sparse_buf = ttnn.reshape(sparse_buf, [1, S, H])

        # ── Full replicated idx/scores on the drain tilize core ─────────────
        idx_full = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=_drm)
        ttnn.deallocate(topk_idx)
        idx_full = ttnn.reshape(idx_full, [S, k])
        idx_full = ttnn.to_memory_config(idx_full, self._sharded_l1([S, k], ttnn.uint16))

        scores_rm = ttnn.to_layout(topk_vals, ttnn.ROW_MAJOR_LAYOUT, memory_config=_drm)
        scores_full = ttnn.reshape(scores_rm, [S, k])
        scores_full = ttnn.to_memory_config(scores_full, self._sharded_l1([S, k], ttnn.bfloat16))

        # ── moe_compute: grouped expert FFN + combine ───────────────────────
        # Combine output: persistent [k,S,H] dim1-sharded buffer, allocated once per S
        # and reused (moe_compute overwrites it). Reuse — rather than a per-call alloc —
        # is what makes the decode step trace-capturable.
        combine_out = self._combine_out_buffer(k, S, H)
        if os.environ.get("MISTRAL4_MOE_DEBUG") == "1":
            try:
                ttnn.dump_device_memory_state(self.mesh_device, prefix="moedbg_")
            except Exception as _e:
                print(f"[MOEDBG] dump failed: {_e}", flush=True)
        outputs = ttnn.experimental.moe_compute(
            sparse_buf,
            idx_full,
            scores_full,
            self._mc_mapping,
            self._mc_w0w1,
            self._mc_w2,
            layer_id=0,
            output_height_shard_dim=self._mc_out_h,
            intermediate_size=I,
            has_bias=False,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            num_links=2,
            mux_core_range_set=self._mc_mux,
            optional_output_tensor=combine_out,
            optional_cross_device_semaphore=self._mc_semaphore,
            activation_type=self._mc_activation,
            bh_ring_size=self._mc_bh_ring_size,
        )
        ttnn.deallocate(sparse_buf)
        ttnn.deallocate(idx_full)
        ttnn.deallocate(scores_full)
        combine = outputs[5]  # [k, S, H] sharded dim1 → per device [k, tpd, H] (UNWEIGHTED)
        # moe_compute returns 6 tensors; only slot 5 (combine) is consumed. Slots 0–3 are
        # large per-call L1 scratch (expert-activation + tilize output); free them now so
        # they don't stay resident in L1 and clash with downstream rms_norm/lm_head CBs.
        # Slot 4 is an RM alias of slot 3's buffer — freed with it, so skip it here.
        for _scratch in outputs[:4]:
            ttnn.deallocate(_scratch)

        # ── Apply routing weights (per device's token slice) + sum over k ───
        sc_dp = ttnn.mesh_partition(scores_rm, dim=2, cluster_axis=1, memory_config=_drm)  # [1,1,tpd,k]
        ttnn.deallocate(scores_rm)
        sc_dp = ttnn.reshape(sc_dp, [tpd, k])
        sc_dp = ttnn.permute(sc_dp, [1, 0])  # [k, tpd]
        sc_dp = ttnn.reshape(sc_dp, [k, tpd, 1])  # broadcast over H
        weighted = ttnn.multiply(combine, sc_dp, memory_config=_drm)
        ttnn.deallocate(sc_dp)
        routed_dp = ttnn.sum(weighted, dim=0)  # [tpd, H]
        ttnn.deallocate(weighted)
        routed_dp = ttnn.reshape(routed_dp, [1, 1, tpd, H])

        # ── Restore replicated layout, add shared expert + residual ─────────
        routed = ttnn.all_gather(
            routed_dp, dim=2, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear, memory_config=_drm
        )  # [1,1,S,H]
        ttnn.deallocate(routed_dp)
        routed = ttnn.to_layout(routed, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        shared_out = self.shared_expert.forward(x)
        out = ttnn.add(routed, shared_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared_out)
        if S != S_orig:
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, S_orig, H], memory_config=ttnn.L1_MEMORY_CONFIG)
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:  x: [1, 1, seq, HIDDEN_SIZE]  (replicated on all devices)
        Returns:   [1, 1, seq, HIDDEN_SIZE]  (replicated on all devices)
        """
        return self._forward_moe_compute(x)

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Single-token decode step.

        Routes through the same token-compacting moe_compute pipeline as prefill.
        _forward_moe_compute is seq-agnostic — a single decode token (seq=1) is
        padded up to num_devices×TILE, processed, and sliced back to [1,1,1,H].
        """
        return self._forward_moe_compute(x)
