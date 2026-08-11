# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mo_e` (HunyuanMoE) of tencent/HunyuanImage-3.0.

HunyuanMoE is a mixed shared/routed MoE:

    shared = shared_mlp(x)                       # SwiGLU
    logits = gate.wg(x)                          # [tokens, num_experts]
    gates  = softmax(logits, dim=experts)
    top-k (=8) routing, weights normalized by the sum of the top-k gates
    routed = sum_e router_weight[:, e] * expert_e(x)   # each expert a SwiGLU
    out    = shared + routed

The canonical Mixtral `TtMoeLayer` models a different routing (8 experts,
top-2, no shared expert, no gate-and-up fusion), so this component is ported
directly with TTNN ops. With `moe_drop_tokens=False` the reference uses an
expert capacity equal to the max tokens-per-expert, so NO token is ever
dropped -- making this dense per-expert formulation numerically identical to
the reference dispatch/combine einsum.

Tensor-parallel (TP) shard path
-------------------------------
When `build()` is handed a `ttnn.MeshDevice` the MoE graduates DIRECTLY
tensor-parallel (EXPERT-parallel). This 6U Blackhole Galaxy only brings the
fabric up on the FULL physical mesh, so the harness opens MeshShape(rows, cols)
(e.g. (8, 4)); TP runs across the mesh axis whose length divides num_experts
(the length-8 axis) and DP-REPLICATES across the other axis:

  * the routed experts are split disjointly `num_experts/TP` per TP device
    (`ShardTensor2dMesh` over a stacked expert-weight tensor);
  * the router is computed REPLICATED (softmax / top-k need all experts) and
    each TP device selects ITS expert columns via a sharded one-hot selection
    matmul; an all-reduce over the TP axis (cluster_axis) sums the per-device
    routed partials;
  * the shared expert, router `wg` are REPLICATED (added once).

The gathered device-0 output equals the single-device golden. The single-device
path (no mesh) is unchanged and still composes the graduated `top_k_gate` stub.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0._stubs import top_k_gate as _top_k_gate

HF_MODEL_ID = "tencent/HunyuanImage-3.0"


# Host dtype matching the target ttnn dtype: convert the weight to bf16 ON THE
# HOST before from_torch so the fp32->bf16 cast happens once at build time (host
# side) rather than as a per-forward on-device TypecastDeviceOperation on the
# lazily-materialized fp32 upload. bfloat8_b has no host torch equivalent, so it
# stays fp32 (its pack happens on device at build regardless).
def _host_of(dtype):
    return torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32


def _to_ttnn(t, device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.to(_host_of(dtype)),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _linear_weight(w, device, dtype=ttnn.bfloat16):
    # nn.Linear stores [out, in]; ttnn.linear(x, W) = x @ W needs [in, out].
    return _to_ttnn(w.t().contiguous(), device, dtype=dtype)


def _is_mesh_device(device) -> bool:
    # A 1-device mesh (or a plain single device) is treated as single-chip: the
    # TP/EP collectives below then no-op, so the model runs fabric-free on one
    # chip (matches the hunyuan-image3-bringup path; unblocks single-chip PCC when
    # the inter-chip fabric is unavailable). Real multi-chip (>1 device) unchanged.
    try:
        if isinstance(device, ttnn.MeshDevice):
            return device.get_num_devices() > 1
    except AttributeError:
        pass
    return hasattr(device, "get_num_devices") and hasattr(device, "get_device_ids") and device.get_num_devices() > 1


# --- env-gated CHEAP perf knobs (ladder rungs 1/2/5), OFF by default ----------
# Shared by the mo_e MoE matmuls AND the image3_decoder_layer attention (which
# imports this module). Each is an isolated experiment for the perf sweep.
def _mm_cfg():
    """compute_kernel_config from HUNYUAN_MM_FIDELITY (lofi|hifi2|hifi3|hifi4);
    None => ttnn default. LoFi = 1-pass math -> biggest throughput on a
    compute-bound matmul (decode already uses LoFi). Token/PCC-gate."""
    f = os.environ.get("HUNYUAN_MM_FIDELITY", "").lower()
    fid = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }.get(f)
    if fid is None:
        return None
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )


def _mm_grid(device):
    """Full CoreGrid by DEFAULT (HUNYUAN_MM_FULLGRID=0 disables -> op default).
    Forces matmuls that don't already pin a grid (e.g. the gate_up + attn linears)
    onto the whole grid for max DRAM-read/compute throughput. Measured -5.9% steady
    ms/step (6093->5736) on the EP=32 mesh; grid-only, no math change (PCC identical)."""
    if os.environ.get("HUNYUAN_MM_FULLGRID", "1") == "0":
        return None
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=g.y, x=g.x)


def _ccl_links():
    """num_links for collectives from HUNYUAN_CCL_LINKS (default 2).

    2 links measured -4.3% steady ms/step (6368->6093) vs 1 on the EP=32 full
    (8,4) mesh -- the 2-axis reduce likes the extra bandwidth. HUNYUAN_CCL_LINKS=1
    restores single-link.
    """
    try:
        return max(1, int(os.environ.get("HUNYUAN_CCL_LINKS", "2")))
    except ValueError:
        return 2


def _ep_links():
    """num_links for the EP-axis all_reduce from HUNYUAN_EP_LINKS (default 1).

    LEVER 4: the EP-axis (DP-axis) reduce on the EP=32 full mesh was hardcoded to
    1 link. HUNYUAN_EP_LINKS=2 bumps it to match the TP-axis reduce; unset/1 keeps
    behavior byte-identical to the prior baseline.
    """
    try:
        return max(1, int(os.environ.get("HUNYUAN_EP_LINKS", "1")))
    except ValueError:
        return 1


def _sp_fused_on():
    """HUNYUAN_SP_FUSED=1 -> SP Step 2. On TOP of SP Step 1 (auto-enabled via _sp_on),
    ALSO H-shards the residual/activation HIDDEN dim across the TP axis (axis 0) so the
    residual stream is [1, S/sp, H/tp], and swaps the plain matmuls for collective
    matmuls: col-parallel -> AG+MM (all_gather_minimal_matmul_async, gathers H then
    matmuls) and row-parallel -> MM + reduce_scatter (sums the TP/expert partials AND
    re-scatters H). The two RMSNorms become DISTRIBUTED (variance reduced over the full
    H via an all_reduce over the TP axis). OFF by default -> byte-identical SP-only path.

    NOTE: the FUSED matmul+reduce_scatter op (minimal_matmul_strided_reduce_scatter_async)
    requires Ring topology; this Galaxy runs FABRIC_1D (Linear), so the row-parallel side
    uses the non-fused MM + ccl_manager.reduce_scatter (Linear) instead -- same math, same
    H/tp-sharded output, no fabric change. AG+MM is Linear-native and used as-is."""
    return os.environ.get("HUNYUAN_SP_FUSED", "0") == "1" or _sp_ring_on()


def _sp_ring_on():
    """HUNYUAN_SP_RING=1 -> open the mesh under FABRIC_1D_RING and run every SP
    collective with Ring topology, AND (unless HUNYUAN_SP_RING_FUSEDMM=0) swap the
    row-parallel MM+reduce_scatter for the FUSED minimal_matmul_strided_reduce_scatter_async
    (Ring-only op). Implies SP_FUSED (Step 2) + SP. OFF -> the FABRIC_1D (Linear)
    SP_FUSED path, byte-identical. EXPLORATORY (ring-fabric overlap experiment)."""
    return os.environ.get("HUNYUAN_SP_RING", "0") == "1"


def _sp_topology():
    """Ring under HUNYUAN_SP_RING (ring fabric), else Linear (FABRIC_1D). Every SP
    collective reads this so the OFF path stays byte-identical."""
    return ttnn.Topology.Ring if _sp_ring_on() else ttnn.Topology.Linear


def _sp_ring_fusedmm_on():
    """Under SP_RING, use the FUSED MM+RS for row-parallel matmuls (default ON;
    HUNYUAN_SP_RING_FUSEDMM=0 falls back to Ring-topology non-fused MM + reduce_scatter
    for ablation)."""
    return _sp_ring_on() and os.environ.get("HUNYUAN_SP_RING_FUSEDMM", "1") == "1"


def _sp_on():
    """HUNYUAN_SP=1 -> Sequence-Parallel (SP Step 1). Shards the residual/activation
    SEQUENCE dim across the DP mesh axis (axis 1) and shrinks expert-parallel to the
    TP axis only (EP=8, see _TtMoE.__init__). A PURE RESHARD: no math change, so it
    must PCC-match the EP=32 path. OFF by default -> byte-identical EP=32 behavior.
    HUNYUAN_SP_FUSED implies SP (Step 2 builds on Step 1's seq-shard + EP=8 layout)."""
    return os.environ.get("HUNYUAN_SP", "0") == "1" or _sp_fused_on()


# --- SP Step 2 (HUNYUAN_SP_FUSED) collective-matmul / distributed-norm helpers ----
# Module-level so BOTH stubs (mo_e AND image3_decoder_layer) share ONE implementation.
def _ag_last(x, tp_axis, num_links=None):
    """all_gather x's LAST (hidden) dim over the TP axis -> full H. x 3D [1,S,H/tp] ->
    [1,S,H]. Explicit Linear-topology gather (the fused AG+MM gathers internally; this
    is for the one-shot gather feeding the router + plain expert matmuls)."""
    nlinks = num_links if num_links is not None else _ccl_links()
    x4 = ttnn.unsqueeze_to_4D(x)
    g = ttnn.all_gather(x4, dim=3, cluster_axis=tp_axis, num_links=nlinks, topology=_sp_topology())
    return ttnn.reshape(g, [int(x.shape[0]), int(g.shape[-2]), int(g.shape[-1])])


def _reduce_scatter_last(x, ccl_manager, tp_axis):
    """reduce_scatter a row-parallel partial [1,S,H] over the TP axis on the hidden dim
    -> [1,S,H/tp]. SIMULTANEOUSLY sums the per-device (head / expert) partials AND
    re-scatters H so the output matches the H-sharded residual. Linear topology."""
    x4 = ttnn.unsqueeze_to_4D(x)
    rs = ccl_manager.reduce_scatter(x4, dim=3, mesh_axis=tp_axis)
    return ttnn.reshape(rs, [int(x.shape[0]), int(rs.shape[-2]), int(rs.shape[-1])])


def _mmrs_last(x, w, device, ccl_manager, tp_axis, *, compute_kernel_config=None):
    """SP_RING: FUSED row-parallel matmul + strided reduce_scatter over the TP axis
    (minimal_matmul_strided_reduce_scatter_async, Ring-only). x [1,S,K] @ w [K,N] ->
    matmul partial [1,S,N] whose per-device (head/expert) partials are summed AND
    re-scattered on the hidden dim in ONE op -> [1,S,N/tp]. Overlaps the MM with the RS
    on the ring fabric. Mirrors RowParallelLinear.forward_fused_addcmul (no addcmul)."""
    from models.tt_dit.utils.matmul import get_fused_mmrs_config

    x4 = ttnn.unsqueeze_to_4D(x)  # [1,1,S,K]
    M, K, N = int(x4.padded_shape[-2]), int(x4.padded_shape[-1]), int(w.padded_shape[-1])
    core_grid = device.compute_with_storage_grid_size()
    ck = compute_kernel_config
    if ck is None:
        # The fused op REQUIRES a compute_kernel_config (no default in the binding);
        # our matmuls normally run with None, so build a sane default here.
        ck = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    _, rs = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        input_tensor=x4,
        weight_tensor=w,
        dim=3,
        multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(tp_axis),
        **get_fused_mmrs_config(M, K, N, core_grid, ccl_manager.num_links),
        bias=None,
        memory_config_mm=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        rs_output_mem_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        topology=_sp_topology(),
        cluster_axis=tp_axis,
        compute_kernel_config=ck,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(tp_axis),
        dtype=None,
    )
    return ttnn.reshape(rs, [int(x.shape[0]), int(rs.shape[-2]), int(rs.shape[-1])])


def _agmm(x, w, device, ccl_manager, tp_axis, *, compute_kernel_config=None, fused_activation=None):
    """Column-parallel AG+MM: all_gather_minimal_matmul_async gathers x's H-shard over
    the TP axis then matmuls with the local weight shard. x 3D [1,S,H/tp]; w 2D [H, N/tp]
    (K = full H, output col-fractured). Returns 3D [1,S,N/tp]. The compute grid frees a
    row (full.y-1) for the CCL worker zone (the NOC lesson: a full grid hits Illegal-NOC)."""
    from models.tt_dit.utils.matmul import get_matmul_config

    x4 = ttnn.unsqueeze_to_4D(x)  # [1,1,S,H/tp]
    M, K, N = int(x4.shape[-2]), int(w.shape[-2]), int(w.shape[-1])
    full = device.compute_with_storage_grid_size()
    cg = ttnn.CoreCoord(full.x, full.y - 1)
    cfg = get_matmul_config(M, K, N, cg, None)
    nlinks = _ccl_links()
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=w,
        bias_tensor=None,
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=compute_kernel_config,
        persistent_output_buffer=ccl_manager.get_ag_ping_pong_buffer(x4.shape, 3, tp_axis, dtype=x4.get_dtype()),
        multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(tp_axis),
        num_links=nlinks,
        topology=_sp_topology(),
        cluster_axis=tp_axis,
        barrier_semaphore=None,
        force_transpose=True,
        num_workers_per_link=full.x // nlinks,
        num_buffers_per_channel=24,
        chunks=1,
        dtype=None,
    )[0]
    return ttnn.reshape(out, [int(x.shape[0]), M, N])


def _dist_rmsnorm(x, weight, tp_axis, eps, hidden_size, num_links=None):
    """Distributed RMSNorm on an H-sharded activation x [1,S,H/tp] with H-sharded weight
    [1,1,H/tp]. The variance reduction spans the FULL H, so the local sum-of-squares is
    all_reduced over the TP axis. Validated to PCC 1.000001 vs torch RMSNorm on-mesh."""
    nlinks = num_links if num_links is not None else _ccl_links()
    sq = ttnn.multiply(x, x)
    ssl = ttnn.sum(sq, dim=-1, keepdim=True)  # [1,S,1] local sum-of-squares
    ttnn.deallocate(sq)
    ss = ttnn.all_reduce(ssl, cluster_axis=tp_axis, num_links=nlinks, topology=_sp_topology())
    ttnn.deallocate(ssl)
    inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / float(hidden_size)), eps))  # [1,S,1]
    ttnn.deallocate(ss)
    xn = ttnn.multiply(x, inv)  # broadcast over H/tp
    ttnn.deallocate(inv)
    out = ttnn.multiply(xn, weight)  # [1,1,H/tp] broadcast over S
    ttnn.deallocate(xn)
    return out


def _minmm_on():
    """HUNYUAN_MINMM=1 -> use flux2 experimental.minimal_matmul (block/core-grid
    optimized matmul, NO collective) as a drop-in for our ttnn.matmul/linear.
    OFF by default (byte-identical fallback). Mixed act/weight dtype is allowed
    by minimal_matmul (validate() only requires each dtype in {bf16,bf8_b,bf4_b,fp32}),
    so it drops into the bf16-act x bf4_b/bf8_b-weight expert matmuls too."""
    return os.environ.get("HUNYUAN_MINMM", "0") == "1"


# --- LEVER 1: swept minimal_matmul block-size winners (HUNYUAN_MMCFG) ----------
# From a single-device `minimal_matmul` block sweep (grid 12x10, M=4096 fixed),
# keyed on per-device (K, N) = (weight.shape[-2], weight.shape[-1]). Values are in
# TILE units (minimal_matmul config semantics). Applied ONLY when HUNYUAN_MMCFG=1.
# (M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
_MMCFG_WINNERS = {
    (4096, 12288): (6, 8, 16, 2, 2),  # expert gate_up
    (6144, 4096): (4, 8, 13, 4, 1),  # expert down
    (4096, 768): (11, 8, 3, 1, 3),  # qkv_proj
    (512, 4096): (10, 2, 12, 2, 2),  # o_proj
    (384, 4096): (8, 4, 13, 4, 1),  # shared_down
}
_MMCFG_SWEEP_GRID = (12, 10)  # (x, y) the sweep ran on; skip (fall back) if runtime grid differs
_MMCFG_CFG_CACHE = {}  # (gx, gy) -> {(K, N): ttnn.MinimalMatmulConfig}, built once per runtime grid
_MMCFG_GRID_WARNED = set()


def _mmcfg_on():
    """HUNYUAN_MMCFG=1 -> apply the swept-winner MinimalMatmulConfig block sizes
    (per (K, N)) to matmuls routed through _minmm. OFF by default: _minmm behaves
    byte-identically to today. The ON path is a strict superset -- a shape with no
    swept winner (or a runtime grid != the swept 12x10) falls through unchanged."""
    return os.environ.get("HUNYUAN_MMCFG", "0") == "1"


def _mmcfg_winners_for_grid(gx, gy):
    """Build (once, cached) the {(K,N): MinimalMatmulConfig} dict for a runtime grid.
    Grid comes from the device at call time (matches _mm_grid), NOT hardcoded."""
    key = (gx, gy)
    d = _MMCFG_CFG_CACHE.get(key)
    if d is None:
        cc = ttnn.CoreCoord(gx, gy)
        d = {
            kn: ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=cc,
            )
            for kn, (mb, kb, nb, sh, sw) in _MMCFG_WINNERS.items()
        }
        _MMCFG_CFG_CACHE[key] = d
    return d


def _mmcfg_lookup(w):
    """Return the swept MinimalMatmulConfig for weight w's (K, N) grid-matched to the
    runtime device, or None to fall back (no winner, grid mismatch, or device error)."""
    kn = (int(w.shape[-2]), int(w.shape[-1]))
    if kn not in _MMCFG_WINNERS:
        return None
    try:
        g = w.device().compute_with_storage_grid_size()
        gx, gy = int(g.x), int(g.y)
    except Exception:
        return None
    if (gx, gy) != _MMCFG_SWEEP_GRID:
        if (gx, gy) not in _MMCFG_GRID_WARNED:
            _MMCFG_GRID_WARNED.add((gx, gy))
            print(
                f"[mo_e] HUNYUAN_MMCFG: runtime grid ({gx},{gy}) != swept {_MMCFG_SWEEP_GRID}; "
                f"skipping swept minimal_matmul configs (fallback to current behavior)."
            )
        return None
    return _mmcfg_winners_for_grid(gx, gy)[kn]


def _minmm(x, w, *, compute_kernel_config=None, core_grid=None, fallback=None):
    """Drop-in matmul. HUNYUAN_MMCFG=1: dispatch minimal_matmul with the swept block
    config for w's (K, N) when one exists (LEVER 1). Otherwise the CURRENT behavior
    EXACTLY: minimal_matmul (no config) when HUNYUAN_MINMM=1, else `fallback`
    (default ttnn.matmul). The OFF path is byte-identical to today. minimal_matmul
    picks its own core grid via its config, so core_grid is only forwarded to the
    fallback."""
    if _mmcfg_on():
        cfg = _mmcfg_lookup(w)
        if cfg is not None:
            return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=compute_kernel_config, config=cfg)
        # no swept winner / grid mismatch -> fall through to the current behavior below
    if _minmm_on():
        return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=compute_kernel_config)
    fb = fallback if fallback is not None else ttnn.matmul
    return fb(x, w, compute_kernel_config=compute_kernel_config, core_grid=core_grid)


class _TtMoE:
    def __init__(self, device, torch_module, ccl_manager=None):
        self.device = device
        self.ccl_manager = ccl_manager  # SP Step 1: shared mesh CCLManager (used from Step 2)
        self.is_mesh = _is_mesh_device(device)
        cfg = torch_module.config
        layer_idx = getattr(torch_module, "layer_idx", 0) or 0

        self.use_shared = bool(getattr(cfg, "use_mixed_mlp_moe", False))
        topk = torch_module.gate.moe_topk
        self.moe_topk = int(topk if isinstance(topk, int) else topk[layer_idx])
        self.num_experts = int(torch_module.num_experts)
        # Gate 2 real-invocation counter.
        self.num_calls = 0

        # Full compute grid for the DRAM-bw-bound down matmul: aggregate DRAM
        # read bandwidth scales with the number of cores issuing NoC reads, so a
        # partial grid caps the weight-read throughput. Force the whole grid.
        _g = device.compute_with_storage_grid_size()
        self.mm_core_grid = ttnn.CoreGrid(y=_g.y, x=_g.x)

        if self.is_mesh:
            self.mesh_shape = tuple(int(x) for x in device.shape)
            self.tp_axis = self._pick_tp_axis()
            self.tp = int(self.mesh_shape[self.tp_axis])
            # EP=32 (63cfd0eb26): shard routed experts across ALL mesh chips (2/chip)
            # instead of TP-axis + DP-replicate. Re-expressed onto the merged-2D-matmul MoE.
            # DEFAULT ON (2026-08-01): PCC-verified (test_mo_e_sharded 0.9940 == EP-off) AND
            # measured -18% steady ms/step (7770->6368, E2E 157.8->143.1s) on the full (8,4)
            # mesh. HUNYUAN_EP_FULLMESH=0 forces OFF; requires num_experts divisible by the
            # device count (else the TP-axis shard fallback below).
            _ep_on = os.environ.get("HUNYUAN_EP_FULLMESH", "1") != "0"
            self._ep_fullmesh = _ep_on and (self.num_experts % int(self.device.get_num_devices()) == 0)
            # SP Step 1 (HUNYUAN_SP): mesh axis 1 (the DP axis) becomes
            # sequence-parallel, so the routed experts can no longer shard across
            # it -> expert-parallel shrinks to the TP axis (EP=8). Forcing
            # _ep_fullmesh OFF makes _build_sharded shard experts on the TP axis
            # only (n_shard = tp = mesh_shape[tp_axis] = 8, epd = num_experts/8)
            # AND makes _mesh_reduce sum the per-device expert partials over the
            # TP axis ONLY (dropping the axis-1 EP reduce) -- exactly the EP=8 /
            # single-axis reduce SP requires. This reuses the known-good pre-EP=32
            # path; sp_axis is recorded for Step 2 (fused collectives). OFF by
            # default -> _ep_fullmesh unchanged -> byte-identical EP=32 behavior.
            self._sp = _sp_on()
            self._sp_fused = _sp_fused_on() and self._sp
            if self._sp:
                self._ep_fullmesh = False
                self.sp_axis = 1 - self.tp_axis
            self._build_sharded(torch_module)
        else:
            self._sp = False
            self._sp_fused = False
            self._build_single(torch_module)

    # ------------------------------------------------------------------
    def _pick_tp_axis(self) -> int:
        """Longest mesh axis (>1) that divides num_experts, so expert-parallel
        MoE splits cleanly across it."""
        best = None
        for ax, sz in enumerate(self.mesh_shape):
            if sz > 1 and self.num_experts % sz == 0:
                if best is None or sz > self.mesh_shape[best]:
                    best = ax
        return 0 if best is None else best

    def _repl(self, t, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.to(_host_of(dtype)),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def _shard(self, t, dim, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        dims = [None, None]
        dims[self.tp_axis] = dim
        return ttnn.from_torch(
            t.to(_host_of(dtype)),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, dims=tuple(dims), mesh_shape=self.mesh_shape),
        )

    def _shard_fullmesh(self, t, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        """EP=32 (63cfd0eb26): shard dim 0 across ALL mesh devices (both axes) so routed
        experts are disjoint (num_experts/num_devices per chip), vs _shard TP-axis-only + DP
        replicate. UNVERIFIED on this MoE form pending the mesh."""
        return ttnn.from_torch(
            t.to(_host_of(dtype)),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )

    # ------------------------------------------------------------------
    def _build_single(self, torch_module):
        device = self.device
        # Composed graduated gate: HunyuanMoE.gate == HunyuanTopKGate. The gate
        # stub owns the router (softmax + top-k) AND the load-balance l_aux; its
        # router weights feed the expert-combine below (main forward path).
        self.gate = _top_k_gate.build(device, torch_module.gate)
        # NOTE: the router weight `wg` now lives in the composed gate stub;
        # mo_e no longer holds a duplicate copy.

        def _mlp_weights(mlp):
            return (
                _linear_weight(mlp.gate_and_up_proj.weight, device),
                _linear_weight(mlp.down_proj.weight, device),
                int(mlp.gate_and_up_proj.weight.shape[0] // 2),  # intermediate (post-split)
            )

        if self.use_shared:
            self.shared_gu, self.shared_down, self.shared_inter = _mlp_weights(torch_module.shared_mlp)
        self.experts = [_mlp_weights(e) for e in torch_module.experts]

    # ------------------------------------------------------------------
    def _build_sharded(self, torch_module):
        tp = self.tp
        # EP shard degree: TP axis by default; ALL devices if HUNYUAN_EP_FULLMESH (63cfd0eb26).
        n_shard = int(self.device.get_num_devices()) if getattr(self, "_ep_fullmesh", False) else tp
        assert self.num_experts % n_shard == 0, f"n_shard={n_shard} must divide num_experts={self.num_experts}"
        epd = self.num_experts // n_shard

        def _shard0(t, dtype=ttnn.bfloat16):
            return (
                self._shard_fullmesh(t, dtype=dtype)
                if getattr(self, "_ep_fullmesh", False)
                else self._shard(t, dim=0, dtype=dtype)
            )

        self.experts_per_dev = epd

        # Composed graduated gate: HunyuanMoE.gate == HunyuanTopKGate. On a mesh
        # the gate stub does the column-parallel router matmul + all_gather and
        # returns the FULL replicated [1, S, E] top-k router weights AND the
        # load-balance l_aux, so this expert-parallel MoE composes it exactly as
        # the single-device path does (keeps top_k_gate on the real forward path
        # instead of an inline duplicate router).
        self.gate = _top_k_gate.build(self.device, torch_module.gate)

        # routed experts as TWO merged 2D matmuls (block-matmul identity).
        # Every routed expert consumes the SAME token input x, and the routed
        # output is sum_e w_e * (silu(x@Wg_e)*(x@Wu_e)) @ Wd_e. Because the down
        # matmul's expert sum is a contraction, the whole epd-expert compute folds
        # into two single 2D matmuls per device — no x-broadcast, no batched
        # matmul, no per-expert reshape/permute, no expert-axis reduce:
        #   gate_up:  gu = x @ Wgu_cat,  Wgu_cat = [Wg_0..Wg_{epd-1} | Wu_0..Wu_{epd-1}]  [H, 2*epd*I]
        #   down:     y  = act @ Wd_stack, Wd_stack = vstack(Wd_0..Wd_{epd-1})            [epd*I, H]
        # so act[:, :, e*I:(e+1)*I] @ Wd_stack[e*I:(e+1)*I, :] summed over e == the
        # per-expert down+sum. Single big-N/big-K matmuls tile the full grid (near
        # roofline) vs epd small batched matmuls. Weights bf8_b (DRAM-bw bound);
        # activations bf16 (mixed-dtype matmul).
        I = int(torch_module.experts[0].gate_and_up_proj.weight.shape[0] // 2)
        self.expert_inter = I
        gu_t = [e.gate_and_up_proj.weight.t().contiguous() for e in torch_module.experts]  # [H, 2I] each
        dn_t = [e.down_proj.weight.t().contiguous() for e in torch_module.experts]  # [I, H] each
        gu_cat = torch.stack(
            [
                torch.cat(
                    [gu_t[d * epd + e][:, :I] for e in range(epd)]  # all gates
                    + [gu_t[d * epd + e][:, I:] for e in range(epd)],  # all ups
                    dim=-1,
                )
                for d in range(n_shard)
            ],
            dim=0,
        )  # [tp, H, 2*epd*I]
        dn_stack = torch.stack(
            [torch.cat([dn_t[d * epd + e] for e in range(epd)], dim=0) for d in range(n_shard)], dim=0
        )  # [tp, epd*I, H]
        self.exp_gu_cat = _shard0(gu_cat, dtype=ttnn.bfloat4_b)  # per TP device [1, H, 2*epd*I]
        self.exp_down_stack = _shard0(dn_stack, dtype=ttnn.bfloat4_b)  # per TP device [1, epd*I, H]

        # per-TP-device selection+expand matrix: picks this device's expert
        # columns out of the replicated router AND repeats each expert's weight
        # across its I-wide down-matmul block, so `router @ sel` directly yields
        # the [1, S, epd*I] per-column router weights the merged down matmul needs
        # (no reshape/broadcast to expand epd -> epd*I).
        sel = torch.zeros(n_shard, self.num_experts, epd * I)
        for d in range(n_shard):
            for e in range(epd):
                sel[d, d * epd + e, e * I : (e + 1) * I] = 1.0
        self.sel = _shard0(sel)

        # shared expert: REPLICATED by default. With full-mesh EP it is SHARDED across all chips
        # on its intermediate dim (025dbff313) so each chip computes a PARTIAL that folds into the
        # SAME 2-axis all_reduce (no new CCL). Requires si % n_shard == 0. UNVERIFIED pending mesh.
        if self.use_shared:
            si = int(torch_module.shared_mlp.gate_and_up_proj.weight.shape[0] // 2)
            self.shared_inter = si
            gu_t = torch_module.shared_mlp.gate_and_up_proj.weight.t().contiguous()  # [H, 2*si]
            down_t = torch_module.shared_mlp.down_proj.weight.t().contiguous()  # [si, H]
            self.shared_sharded = getattr(self, "_ep_fullmesh", False) and (si % n_shard == 0)
            if self.shared_sharded:
                self.shared_per = si // n_shard
                _Hs = int(gu_t.shape[0])
                self.shared_gu = ttnn.from_torch(
                    gu_t.reshape(_Hs, 2, si).to(torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=2),
                )
                self.shared_gu = ttnn.reshape(self.shared_gu, [_Hs, 2 * self.shared_per])
                self.shared_down = ttnn.from_torch(
                    down_t.to(torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
                )
            else:
                self.shared_gu = self._repl(gu_t)
                self.shared_down = self._repl(down_t)

    # ------------------------------------------------------------------
    def _mesh_reduce(self, x):
        """All-reduce (sum) a per-device partial across the TP mesh axis.

        Fused ring `ttnn.all_reduce` (cluster_axis=TP axis) instead of the naive
        `all_gather(dim=0)+sum`: the old path materialised [tp, S, hidden] (tp× the
        bytes) on every chip before a local reduce; the ring all_reduce moves ~2×
        the shard bytes/chip and drops the separate sum — the exact prefill-MoE
        reduce gpt_oss/gemma4/deepseek use. Same math, same shape."""
        if not _is_mesh_device(self.device):
            return x  # single chip: the per-device partial already IS the full sum
        x = ttnn.all_reduce(x, cluster_axis=self.tp_axis, num_links=_ccl_links(), topology=_sp_topology())
        if getattr(self, "_ep_fullmesh", False):
            # EP=32: experts sharded over BOTH mesh axes -> also sum over the DP axis (63cfd0eb26).
            x = ttnn.all_reduce(x, cluster_axis=(1 - self.tp_axis), num_links=_ep_links(), topology=_sp_topology())
        return x

    def _swiglu(self, x, gu_w, down_w, inter):
        gu = ttnn.linear(x, gu_w, compute_kernel_config=_mm_cfg(), core_grid=_mm_grid(self.device))
        x1 = ttnn.slice(gu, [0, 0, 0], [gu.shape[0], gu.shape[1], inter])
        x2 = ttnn.slice(gu, [0, 0, inter], [gu.shape[0], gu.shape[1], 2 * inter])
        ttnn.deallocate(gu)
        # SwiGLU: fuse silu into the multiply (silu(x2) * x1 in one op).
        act = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        out = _minmm(
            act, down_w, compute_kernel_config=_mm_cfg(), core_grid=_mm_grid(self.device), fallback=ttnn.linear
        )
        ttnn.deallocate(act)
        return out

    # ------------------------------------------------------------------
    def _forward_sharded_sparse(self, hidden_states, return_l_aux=False):
        """EXPERIMENT (HUNYUAN_SPARSE_MOE=1): local per-device token gather so the
        two expert matmuls run on only the ~C tokens routing to THIS device's epd
        experts, not all S. Targets the ~30% device-MoE slice (see stage profile).

        DRAFT — the gather/scatter/topk op layouts are marked `# VERIFY` and need
        on-device checking; `_forward_sharded` wraps this in try/except and falls
        back to the (exact) dense path on any error, so a run stays alive. Capacity
        C < S DROPS routed tokens (the reference is no-drop) -> PCC-gate every run;
        size C safely (default S/2)."""
        x = hidden_states
        S = int(x.shape[1])
        H = int(x.shape[-1])
        I = self.expert_inter
        eI = self.experts_per_dev * I
        C = min(int(os.environ.get("HUNYUAN_SPARSE_CAP", str(S // 2))), S)

        l_aux, router = self.gate(x, return_router=True, need_l_aux=return_l_aux)
        router_local = ttnn.matmul(router, self.sel)  # [1, S, epd*I], 0 for non-selected
        ttnn.deallocate(router)

        # per-token "routed to THIS device" weight (max over this device's cols) -> [1, S]
        tok_w = ttnn.reshape(ttnn.max(router_local, dim=-1), [1, S])  # VERIFY: max keepdim/shape
        # top-C routed tokens. <C routed -> padding tokens (weight 0 -> contribute 0);
        # >C routed -> DROP (PCC risk). topi [1, C] (uint).
        _tv, topi = ttnn.topk(tok_w, C, dim=-1, largest=True, sorted=False)  # VERIFY topk sig
        topi = ttnn.typecast(topi, ttnn.uint32)  # topk idx is uint16; reshape/gather need uint32/int32
        ttnn.deallocate(tok_w)
        ttnn.deallocate(_tv)

        # expand topi [1,C] -> gather indices [1,C,H] and [1,C,epd*I]
        topi3 = ttnn.reshape(topi, [1, C, 1])
        idx_h = ttnn.repeat(topi3, ttnn.Shape([1, 1, H]))  # VERIFY repeat/expand of uint idx
        idx_e = ttnn.repeat(topi3, ttnn.Shape([1, 1, eI]))
        ttnn.deallocate(topi)
        ttnn.deallocate(topi3)

        x_g = ttnn.gather(x, dim=1, index=idx_h)  # [1, C, H]  VERIFY gather API/layout
        rl_g = ttnn.gather(router_local, dim=1, index=idx_e)  # [1, C, epd*I]
        ttnn.deallocate(router_local)
        ttnn.deallocate(idx_e)

        gu = ttnn.matmul(
            x_g, self.exp_gu_cat, compute_kernel_config=_mm_cfg(), core_grid=_mm_grid(self.device)
        )  # [1, C, 2*epd*I]
        ttnn.deallocate(x_g)
        x1 = ttnn.slice(gu, [0, 0, 0], [1, C, eI])
        x2 = ttnn.slice(gu, [0, 0, eI], [1, C, 2 * eI])
        ttnn.deallocate(gu)
        act = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        act = ttnn.multiply(act, rl_g)
        ttnn.deallocate(rl_g)
        comb_g = ttnn.matmul(
            act, self.exp_down_stack, core_grid=self.mm_core_grid, compute_kernel_config=_mm_cfg()
        )  # [1, C, H]
        ttnn.deallocate(act)

        # scatter [1,C,H] back into a per-device zeroed [1,S,H] at the gathered rows
        combined = ttnn.zeros(
            [1, S, H],
            dtype=comb_g.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # VERIFY: per-device zero on mesh (matches dense per-device partial)
        # uint16 index clears the scatter 256-cap TILE guard (scatter.cpp is_i32=uint32+int32;
        # uint16 exempt+supported). idx are token positions < S < 65535 -> lossless. micro-repro PCC 1.0
        idx_h16 = ttnn.typecast(idx_h, ttnn.uint16)
        combined = ttnn.scatter(combined, dim=1, index=idx_h16, src=comb_g)  # VERIFY scatter API
        ttnn.deallocate(idx_h16)
        ttnn.deallocate(comb_g)
        ttnn.deallocate(idx_h)

        routed = self._mesh_reduce(combined)  # all-reduce over TP -> full expert sum
        ttnn.deallocate(combined)
        if self.use_shared:
            shared = self._swiglu(x, self.shared_gu, self.shared_down, self.shared_inter)
            out = ttnn.add(shared, routed)
            ttnn.deallocate(shared)
            ttnn.deallocate(routed)
        else:
            out = routed
        if return_l_aux:
            return out, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return out

    def _forward_sharded(self, hidden_states, return_l_aux=False):
        if os.environ.get("HUNYUAN_SPARSE_MOE") == "1":
            try:
                return self._forward_sharded_sparse(hidden_states, return_l_aux=return_l_aux)
            except Exception as e:  # draft path: keep the run alive with the exact dense fallback
                print(f"[mo_e] SPARSE path failed ({type(e).__name__}: {e}) -> dense fallback")
        if getattr(self, "_sp_fused", False):
            return self._forward_sp_fused(hidden_states, return_l_aux=return_l_aux)
        x = hidden_states

        # --- routing via the composed graduated gate (top_k_gate) ---
        # The gate returns the load-balance l_aux AND the FULL normalized top-k
        # router weights [1, S, E] (replicated across the mesh); this device then
        # selects ITS expert columns out of that replicated router.
        l_aux, router = self.gate(x, return_router=True, need_l_aux=return_l_aux)

        router_local = ttnn.matmul(router, self.sel)  # [1, S, epd*I] (per-column weights, expanded)
        ttnn.deallocate(router)

        # --- routed experts: TWO merged 2D matmuls (block-matmul identity) ---
        # gu = x @ Wgu_cat gives all experts' gate|up in one big-N matmul; SwiGLU
        # in the flat [1,S,epd*I] layout (gates in the first half, ups in the
        # second) needs only two slices at the midpoint — no reshape/permute.
        # Router weights (already expanded to epd*I by `sel`) scale act per column,
        # then act @ Wd_stack contracts over epd*I == the per-expert down + sum in
        # ONE matmul. Numerically identical to the batched grouped-matmul, but the
        # two matmuls tile the full grid (near roofline) and the concat/reshape/
        # permute/expert-sum overhead is gone.
        I = self.expert_inter
        eI = self.experts_per_dev * I
        gu = _minmm(
            x, self.exp_gu_cat, compute_kernel_config=_mm_cfg(), core_grid=_mm_grid(self.device)
        )  # [1, S, 2*epd*I] = [all gates | all ups]
        x1 = ttnn.slice(gu, [0, 0, 0], [gu.shape[0], gu.shape[1], eI])  # gates [1,S,epd*I]
        x2 = ttnn.slice(gu, [0, 0, eI], [gu.shape[0], gu.shape[1], 2 * eI])  # ups   [1,S,epd*I]
        ttnn.deallocate(gu)
        # SwiGLU: fuse silu into the multiply (silu(x2) * x1 in one op).
        act = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])  # [1, S, epd*I]
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        act = ttnn.multiply(act, router_local)  # per-expert-column router weight
        ttnn.deallocate(router_local)
        combined = _minmm(
            act, self.exp_down_stack, core_grid=self.mm_core_grid, compute_kernel_config=_mm_cfg()
        )  # [1, S, H] (down + expert-sum fused)
        ttnn.deallocate(act)

        # shared expert: when SHARDED (full-mesh EP) add its PARTIAL to the routed partial BEFORE
        # the reduce so ONE 2-axis all_reduce sums both (025dbff313); else the original post-reduce add.
        if self.use_shared and getattr(self, "shared_sharded", False):
            shared_p = self._swiglu(x, self.shared_gu, self.shared_down, self.shared_per)
            combined = ttnn.add(combined, shared_p)
            ttnn.deallocate(shared_p)
            out = self._mesh_reduce(combined)
            ttnn.deallocate(combined)
        elif self.use_shared:
            routed = self._mesh_reduce(combined)  # all-reduce over TP -> full expert sum
            ttnn.deallocate(combined)
            shared = self._swiglu(x, self.shared_gu, self.shared_down, self.shared_inter)
            out = ttnn.add(shared, routed)
            ttnn.deallocate(shared)
            ttnn.deallocate(routed)
        else:
            out = self._mesh_reduce(combined)
            ttnn.deallocate(combined)
        if return_l_aux:
            return out, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return out

    # ------------------------------------------------------------------
    def _forward_sp_fused(self, hidden_states, return_l_aux=False):
        """SP Step 2 MoE: the residual is H-sharded [1, S/sp, H/tp]. Gather H ONCE
        (feeds the replicated router + the plain merged-2D expert/shared matmuls, which
        want full H), run the exact dense MoE, then reduce_scatter the [1,S/sp,H] partial
        back to [1,S/sp,H/tp] -- the reduce_scatter simultaneously sums the EP=8 expert
        partials and re-scatters H. The shared expert stays REPLICATED (full-H output);
        folding shared/tp into the routed partial makes the single reduce_scatter sum
        tp*(shared/tp) = shared exactly (1/tp is an exact bf16 exponent shift), so no
        si-divisibility constraint and no extra collective."""
        x_sh = hidden_states  # [1, S/sp, H/tp]
        x = _ag_last(x_sh, self.tp_axis)  # [1, S/sp, H] full hidden (one gather, reused below)

        l_aux, router = self.gate(x, return_router=True, need_l_aux=return_l_aux)
        router_local = ttnn.matmul(router, self.sel)  # [1, S/sp, epd*I]
        ttnn.deallocate(router)

        I = self.expert_inter
        eI = self.experts_per_dev * I
        gu = _minmm(x, self.exp_gu_cat, compute_kernel_config=_mm_cfg(), core_grid=_mm_grid(self.device))
        x1 = ttnn.slice(gu, [0, 0, 0], [gu.shape[0], gu.shape[1], eI])
        x2 = ttnn.slice(gu, [0, 0, eI], [gu.shape[0], gu.shape[1], 2 * eI])
        ttnn.deallocate(gu)
        act = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        act = ttnn.multiply(act, router_local)
        ttnn.deallocate(router_local)
        if _sp_ring_fusedmm_on():
            # SP_RING: FUSED routed-down matmul + reduce_scatter (overlaps the dominant
            # expert-down MM with its RS on the ring fabric). The replicated shared expert
            # is scattered to matching H/tp blocks via a separate Ring reduce_scatter of
            # shared/tp (sum over tp of replicated shared/tp = shared; block d -> device d).
            out = _mmrs_last(
                act,
                self.exp_down_stack,
                self.device,
                self.ccl_manager,
                self.tp_axis,
                compute_kernel_config=_mm_cfg(),
            )  # [1, S/sp, H/tp] routed sum
            ttnn.deallocate(act)
            if self.use_shared:
                shared = self._swiglu(x, self.shared_gu, self.shared_down, self.shared_inter)  # [1,S/sp,H] full
                shared = ttnn.multiply(shared, 1.0 / float(self.tp))
                shared_sh = _reduce_scatter_last(shared, self.ccl_manager, self.tp_axis)  # [1,S/sp,H/tp] block d
                ttnn.deallocate(shared)
                out = ttnn.add(out, shared_sh)
                ttnn.deallocate(shared_sh)
            ttnn.deallocate(x)
        else:
            combined = _minmm(
                act, self.exp_down_stack, core_grid=self.mm_core_grid, compute_kernel_config=_mm_cfg()
            )  # [1, S/sp, H] partial (EP=8 per-device expert sum, pre-reduce)
            ttnn.deallocate(act)

            if self.use_shared:
                shared = self._swiglu(x, self.shared_gu, self.shared_down, self.shared_inter)  # [1,S/sp,H] full
                shared = ttnn.multiply(shared, 1.0 / float(self.tp))  # fold into the tp-way reduce
                combined = ttnn.add(combined, shared)
                ttnn.deallocate(shared)
            ttnn.deallocate(x)

            out = _reduce_scatter_last(combined, self.ccl_manager, self.tp_axis)  # [1, S/sp, H/tp]
            ttnn.deallocate(combined)
        if return_l_aux:
            return out, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return out

    # ------------------------------------------------------------------
    def __call__(self, hidden_states, return_l_aux=False, **kwargs):
        self.num_calls += 1
        if self.is_mesh:
            return self._forward_sharded(hidden_states, return_l_aux=return_l_aux)

        x = hidden_states

        # --- routing via the composed graduated gate (top_k_gate) ---
        # The gate returns the load-balance l_aux AND the normalized top-k
        # router weights [1, S, E] the experts are combined with below.
        l_aux, router = self.gate(x, return_router=True, need_l_aux=return_l_aux)

        # --- shared expert ---
        combined = None
        if self.use_shared:
            combined = self._swiglu(x, self.shared_gu, self.shared_down, self.shared_inter)

        # --- routed experts (dense; no token dropping in this config) ---
        for e in range(self.num_experts):
            gu_w, down_w, inter = self.experts[e]
            y = self._swiglu(x, gu_w, down_w, inter)
            w = ttnn.slice(router, [0, 0, e], [router.shape[0], router.shape[1], e + 1])
            y = ttnn.multiply(y, w)
            ttnn.deallocate(w)
            if combined is None:
                combined = y
            else:
                combined = ttnn.add(combined, y)
                ttnn.deallocate(y)
        ttnn.deallocate(router)
        if return_l_aux:
            return combined, l_aux
        if l_aux is not None:
            ttnn.deallocate(l_aux)
        return combined


def build(device, torch_module=None, ccl_manager=None):
    if torch_module is None:
        raise RuntimeError("mo_e native port requires the HF torch_module to extract weights.")
    return _TtMoE(device, torch_module, ccl_manager=ccl_manager)


def mo_e(device, torch_module=None):
    return build(device, torch_module)
