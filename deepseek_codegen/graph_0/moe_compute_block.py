# SPDX-License-Identifier: Apache-2.0
"""moe_compute-based routed-expert block for moe_test.py.

Replaces the graph's all_to_all_dispatch + 2x sparse_matmul + all_to_all_combine
expert path with the fused ttnn.experimental.moe_compute pipeline:

    all_to_all_dispatch_metadata -> moe_compute -> (manual weighted-k-sum
    -> all_reduce_async(axis1) -> mesh_partition)

Weights come from the SAME ce_cache the sparse path used (main_const_eval_gate_up =
concat(W0,W1) and main_const_eval_39 = W2), re-prepared into moe_compute's bf4
DRAM-sharded layout (each device keeps its own 8 experts). The expert_mapping is
derived from the graph's one-hot var_76 (main_const_eval_37).

cluster_axis=0 (4 dispatch devices), 8 replicated (axis 1), 8 experts/device,
256 routed experts, hidden=7168, N=2048, k=8, tokens_per_device=32.
"""
import os
from pathlib import Path
import torch
import ttnn
import utils
import main as M
from ttnn.operations.ccl import MoEActivationFunction
from ttnn.experimental.moe_compute_utils import get_tilize_drain_core

THIS_DIR = Path(__file__).resolve().parent
WCACHE = THIS_DIR / "moe_io" / "wcache"  # prepared bf4 weights cache

MESH = (4, 8)
CLUSTER_AXIS = 0
HIDDEN = 7168
N = 2048
K_SEL = 8
EPD = 8  # experts per device
TOKENS_PER_DEV = 32
NUM_DISPATCH = MESH[CLUSTER_AXIS]  # 4
NUM_REPL = (MESH[0] * MESH[1]) // NUM_DISPATCH  # 8
TOTAL_TOKENS = TOKENS_PER_DEV * NUM_DISPATCH  # 128
SHARD_DIMS = (0, None)  # tokens on axis0, replicated axis1
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _dram():
    return DRAM


DEBUG_SYNC = os.environ.get("MOE_DEBUG_SYNC") == "1"


def _dbg(dev, name):
    """When MOE_DEBUG_SYNC=1, synchronize after an op and print, so a device hang is
    pinpointed to the op AFTER the last printed step (synchronize blocks on the hang)."""
    if DEBUG_SYNC:
        ttnn.synchronize_device(dev)
        print(f"[moe_dbg] {name} OK", flush=True)


def derive_expert_mapping(var_76, dev):
    """Build moe_compute's canonical [num_devices, experts] uint16 linear-coord
    expert_mapping from the graph's one-hot var_76 [.,.,256,32] (replicated)."""
    onehot = ttnn.to_torch(ttnn.get_device_tensors(var_76)[0]).to(torch.int64).reshape(256, 32)
    dev_of_expert = onehot.argmax(dim=-1).to(torch.int32)  # [256]
    canon = dev_of_expert.unsqueeze(0).repeat(32, 1)  # [32, 256]
    return ttnn.from_torch(
        canon,
        device=dev,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=MESH),
    )


def _wcache_ready():
    return (WCACHE / "tt_w0w1.tensorbin").exists() and (WCACHE / "tt_w2.tensorbin").exists()


def prepare_moe_weights(ce_cache, dev, rebuild=False):
    """Slice ce_cache gate_up -> w0/w1, use const_eval_39 -> w2, run the C++ device
    prepare_* + quantize to bf4 DRAM-sharded (per-device, 8 experts each).

    Cached to disk PER TENSOR (so a w2 failure doesn't force a w0/w1 rebuild). Inputs to
    the prepare ops are deallocated AFTER their quantize: prepare_w2 on a sharded input
    returns a tensor that shares storage with its input, so an early deallocate frees it
    (-> "Tensor is not allocated" in quantize_weights_via_host's from_device)."""
    WCACHE.mkdir(parents=True, exist_ok=True)
    mc = ttnn.experimental.get_weight_mem_configs(
        dev, num_layers=1, experts_per_device=EPD, hidden_size=HIDDEN, intermediate_size=N, has_bias=False
    )
    _sfx = os.environ.get("MOE_WEIGHT_DTYPE", "bf4")
    _qdt = {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b}.get(_sfx, ttnn.bfloat4_b)
    _tag = "" if _sfx == "bf4" else f"_{_sfx}"
    w0w1_path = WCACHE / f"tt_w0w1{_tag}.tensorbin"
    w2_path = WCACHE / f"tt_w2{_tag}.tensorbin"

    # The C++ prepare_* helpers expect ROW_MAJOR bf16 input (the layout the reference
    # uploads via from_torch). The ce_cache weights are bf8 TILE; feeding TILE both makes
    # the internal torch-style reshapes operate on tile-ordered elements (wrong layout) AND
    # triggers a latent bug in prepare_w2 (its final to_layout(TILE) is a no-op on an
    # already-TILE tensor, then the input is force-deallocated -> returned tensor unallocated).
    def _rm_bf16(t):
        return ttnn.to_layout(
            ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=DRAM),
            ttnn.Layout.ROW_MAJOR,
            None,
            memory_config=DRAM,
        )

    if w0w1_path.exists() and not rebuild:
        tt_w0w1 = ttnn.load_tensor(str(w0w1_path), device=dev)
    else:
        gate_up = _rm_bf16(ce_cache["main_const_eval_gate_up"])  # [1,8,7168,4096] RM bf16
        w0 = ttnn.slice(gate_up, [0, 0, 0, 0], [1, EPD, HIDDEN, N], memory_config=DRAM)
        w1 = ttnn.slice(gate_up, [0, 0, 0, N], [1, EPD, HIDDEN, 2 * N], memory_config=DRAM)
        ttnn.deallocate(gate_up)
        w0w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(w0, w1, L=1, E=EPD, K=HIDDEN, N=N)
        tt_w0w1 = ttnn.experimental.quantize_weights_via_host(w0w1_prepped, dtype=_qdt, memory_config=mc.w0_w1)
        ttnn.deallocate(w0w1_prepped)
        ttnn.deallocate(w0)
        ttnn.deallocate(w1)
        ttnn.dump_tensor(str(w0w1_path), tt_w0w1)

    if w2_path.exists() and not rebuild:
        tt_w2 = ttnn.load_tensor(str(w2_path), device=dev)
    else:
        w2 = _rm_bf16(ce_cache["main_const_eval_39"])  # [1,8,2048,7168] RM bf16
        w2_prepped = ttnn.experimental.prepare_w2_tensor_for_moe_compute(w2, L=1, E=EPD, N=N, K=HIDDEN)
        tt_w2 = ttnn.experimental.quantize_weights_via_host(w2_prepped, dtype=_qdt, memory_config=mc.w2)
        ttnn.deallocate(w2_prepped)
        ttnn.deallocate(w2)
        ttnn.dump_tensor(str(w2_path), tt_w2)

    return tt_w0w1, tt_w2


class MoEComputeState:
    """Persistent weights, expert_mapping, semaphores, buffers, mem configs for the
    moe_compute routed-expert pipeline. Built once outside the per-decode timing."""

    def __init__(self, ce_cache, var_76, dev, rebuild_weights=False):
        self.dev = dev
        self.tt_w0w1, self.tt_w2 = prepare_moe_weights(ce_cache, dev, rebuild=rebuild_weights)
        self.tt_em = derive_expert_mapping(var_76, dev)
        self.drain = get_tilize_drain_core()
        self.pools = M.main_const_eval_all_reduce_semaphores(dev)

        grid = dev.compute_with_storage_grid_size()
        worker = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.disp_sem = ttnn.create_global_semaphore(dev, worker, 0)
        self.comb_sem = ttnn.create_global_semaphore(dev, worker, 0)
        self.mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])

        # dispatch output buffers (preallocated, persistent)
        self.sparse_buf = ttnn.from_torch(
            torch.zeros([NUM_DISPATCH, TOTAL_TOKENS, HIDDEN], dtype=torch.bfloat16),
            device=dev,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=SHARD_DIMS, mesh_shape=MESH),
        )
        out_shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(self.drain, self.drain)}),
            [TOTAL_TOKENS, K_SEL],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.out_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
        self.out_idx = ttnn.from_torch(
            torch.zeros([NUM_DISPATCH, TOTAL_TOKENS, K_SEL], dtype=torch.uint16),
            device=dev,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=self.out_l1,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=SHARD_DIMS, mesh_shape=MESH),
        )
        self.out_scr = ttnn.from_torch(
            torch.zeros([NUM_DISPATCH, TOTAL_TOKENS, K_SEL], dtype=torch.bfloat16),
            device=dev,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=self.out_l1,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=SHARD_DIMS, mesh_shape=MESH),
        )
        self.combine_out = ttnn.from_torch(
            torch.zeros([K_SEL, TOKENS_PER_DEV, HIDDEN], dtype=torch.bfloat16),
            device=dev,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )

        # dispatch input mem config (L1 height-sharded on a token grid)
        ncy = min(8, TOKENS_PER_DEV)
        ncx = (TOKENS_PER_DEV + ncy - 1) // ncy
        in_shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncx - 1, ncy - 1))}),
            [1, K_SEL],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.in_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_shard)


def run_routed_experts(x_normed, indices_i32, scores_bf16, st):
    """Routed MoE via moe_compute. Returns the routed output [1,32,896] fp32 per device
    (= the graph's ttnn_typecast_101).

    Inputs (per device, tokens on axis0, hidden TP-sharded on axis1):
      x_normed   [32,1,896] bf16 (graph ttnn_typecast_78, reshaped)
      indices_i32 [32,8]    int32 (graph ttnn_typecast_86)
      scores_bf16 [32,1,8]  bf16  (graph ttnn_multiply_58)
    """
    dev = st.dev
    # All 3 dispatch inputs must be ROW_MAJOR (router produces TILE): the L1 height-sharded
    # indices/scores use non-tile [1,8] shards (TILE requires 32x32 shards), and the canonical
    # dispatch input x is ROW_MAJOR too.
    # x: gather hidden along axis1 -> full 7168, shape [32,1,1,7168], L1
    x = ttnn.to_layout(x_normed, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM)
    x = ttnn.reshape(x, [TOKENS_PER_DEV, 1, 1, HIDDEN // NUM_REPL], memory_config=DRAM)
    x = ttnn.all_gather(
        input_tensor=x,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=DRAM,
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    _dbg(dev, "all_gather_x")
    # indices: int32 -> ROW_MAJOR -> reshape -> uint16 (reshape rejects uint16) -> L1 height-shard
    idx = ttnn.to_layout(indices_i32, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM)
    idx = ttnn.reshape(idx, [TOKENS_PER_DEV, 1, 1, K_SEL], memory_config=DRAM)
    idx = ttnn.typecast(idx, ttnn.DataType.UINT16, memory_config=DRAM)
    idx = ttnn.to_memory_config(idx, st.in_l1)
    # scores: bf16 -> ROW_MAJOR -> reshape -> L1 height-shard
    scr = ttnn.to_layout(scores_bf16, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM)
    scr = ttnn.reshape(scr, [TOKENS_PER_DEV, 1, 1, K_SEL], memory_config=DRAM)
    scr_l1 = ttnn.to_memory_config(scr, st.in_l1)

    d_sparse, d_idx, d_scr = ttnn.experimental.all_to_all_dispatch_metadata(
        x,
        idx,
        scr_l1,
        st.tt_em,
        cluster_axis=CLUSTER_AXIS,
        num_links=None,
        worker_mode=ttnn.WorkerMode.DIRECT,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
        output_tensors=(st.sparse_buf, st.out_idx, st.out_scr),
        cross_device_semaphore=st.disp_sem,
    )
    ttnn.deallocate(x)
    ttnn.deallocate(scr_l1)
    _dbg(dev, "all_to_all_dispatch_metadata")

    # DIAGNOSTIC: compute_only=True runs only the expert matmuls (NO combine/mux), isolating
    # whether the matmul half works on BH 4x8 (the combine hangs on both Ring and Linear topology).
    if os.environ.get("MOE_COMPUTE_ONLY") == "1":
        co = ttnn.experimental.moe_compute(
            d_sparse,
            d_idx,
            d_scr,
            st.tt_em,
            st.tt_w0w1,
            st.tt_w2,
            layer_id=0,
            output_height_shard_dim=4,
            intermediate_size=N,
            has_bias=False,
            activation_type=MoEActivationFunction.SILU,
            compute_only=True,
        )
        _dbg(dev, "moe_compute(compute_only)")
        print(f"[moe_dbg] compute_only matmul output (slot4) shape={tuple(co[4].shape)}", flush=True)
        # diagnostic only: return zeros so moe_test completes (PCC will fail; we just want no-hang)
        return ttnn.from_torch(
            torch.zeros([1, TOKENS_PER_DEV, HIDDEN // NUM_REPL], dtype=torch.float32),
            device=dev,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )

    # Combine config is env-driven so the BH-4x8 hang can be swept without recompiling:
    #   MOE_TOPOLOGY = Ring | Linear | (unset->auto)
    #   MOE_NUM_LINKS = 1 | 2 | (unset->auto)
    #   MOE_BH_RING   = 8 | 12 | 16 | (unset->default 12)
    #   MOE_OHSD      = output_height_shard_dim (default 4)
    #   MOE_SEM_COMBINE_CORES = 1  -> put the combine semaphore on get_moe_combine_cores (vs full grid)
    _topo = {"Ring": ttnn.Topology.Ring, "Linear": ttnn.Topology.Linear}.get(os.environ.get("MOE_TOPOLOGY"))
    _nl = os.environ.get("MOE_NUM_LINKS")
    _nl = int(_nl) if _nl else None
    _bhr = os.environ.get("MOE_BH_RING")
    _bhr = int(_bhr) if _bhr else None
    _ohsd = int(os.environ.get("MOE_OHSD", "4"))
    _kw = {}
    if _bhr is not None:
        _kw["bh_ring_size"] = _bhr
    print(
        f"[moe_dbg] moe_compute config: topology={os.environ.get('MOE_TOPOLOGY','auto')} "
        f"num_links={_nl} bh_ring={_bhr} ohsd={_ohsd}",
        flush=True,
    )
    outs = ttnn.experimental.moe_compute(
        d_sparse,
        d_idx,
        d_scr,
        st.tt_em,
        st.tt_w0w1,
        st.tt_w2,
        layer_id=0,
        output_height_shard_dim=_ohsd,
        intermediate_size=N,
        has_bias=False,
        cluster_axis=CLUSTER_AXIS,
        topology=_topo,
        num_links=_nl,
        mux_core_range_set=st.mux,
        optional_output_tensor=st.combine_out,
        optional_cross_device_semaphore=st.comb_sem,
        activation_type=MoEActivationFunction.SILU,
        **_kw,
    )
    tt_combine = outs[5]  # [k, tokens_per_device, hidden] per device (partial across axis1)
    _dbg(dev, "moe_compute")
    if os.environ.get("MOE_DUMP_ROUTED"):
        _s = ttnn.to_torch(ttnn.get_device_tensors(scr)[0]).float().reshape(TOKENS_PER_DEV, K_SEL)
        _ssum = _s.sum(-1)
        print(
            f"[moe_dbg] scores per-token sum: mean={float(_ssum.mean()):.4f} "
            f"first5={[round(float(v),4) for v in _ssum[:5]]}",
            flush=True,
        )
        _c = ttnn.to_torch(ttnn.get_device_tensors(tt_combine)[0]).float()
        print(
            f"[moe_dbg] combine_out: shape={tuple(_c.shape)} norm={float(_c.norm()):.4f} "
            f"absmax={float(_c.abs().max()):.4f} nonzero_frac={float((_c.abs()>1e-6).float().mean()):.4f}",
            flush=True,
        )

    # ---- manual tail: weighted-k-sum -> all_reduce(axis1) -> mesh_partition ----
    c = ttnn.to_layout(tt_combine, ttnn.Layout.TILE, None, memory_config=DRAM)
    c = ttnn.typecast(c, ttnn.DataType.FLOAT32, memory_config=DRAM)
    if os.environ.get("MOE_NO_TAIL_SCORE"):
        # DIAGNOSTIC: skip the score-multiply to test whether moe_compute's combine already
        # applied scores internally (if e2e PCC jumps to ~0.99, the tail must NOT re-apply).
        weighted = c
    else:
        sc = ttnn.reshape(scr, [TOKENS_PER_DEV, K_SEL], memory_config=DRAM)
        sc = ttnn.to_layout(sc, ttnn.Layout.TILE, None, memory_config=DRAM)
        sc = ttnn.typecast(sc, ttnn.DataType.FLOAT32, memory_config=DRAM)
        sc = ttnn.permute(sc, [1, 0], memory_config=DRAM)  # [k, tokens]
        sc = ttnn.reshape(sc, [K_SEL, TOKENS_PER_DEV, 1], memory_config=DRAM)
        if os.environ.get("MOE_DUMP_ROUTED"):
            _scd = ttnn.to_torch(ttnn.get_device_tensors(sc)[0]).float().reshape(K_SEL, TOKENS_PER_DEV)
            _scr0 = ttnn.to_torch(ttnn.get_device_tensors(scr)[0]).float().reshape(TOKENS_PER_DEV, K_SEL)
            print(f"[moe_dbg] sc[:,tok0] after permute (k=0..7): {[round(float(v),4) for v in _scd[:, 0]]}", flush=True)
            print(
                f"[moe_dbg] scr[tok0,:] raw           (k=0..7): {[round(float(v),4) for v in _scr0[0, :]]}", flush=True
            )
        weighted = ttnn.multiply(c, sc, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(c)
        ttnn.deallocate(sc)
    summed = ttnn.sum(weighted, [0], False, memory_config=DRAM)  # [tokens, hidden] partial
    ttnn.deallocate(weighted)
    summed = ttnn.reshape(summed, [1, 1, TOKENS_PER_DEV, HIDDEN], memory_config=DRAM)
    _dbg(dev, "weighted_k_sum")
    ar = ttnn.experimental.all_reduce_async(
        summed,
        cluster_axis=1,
        mesh_device=dev,
        barrier_semaphores=st.pools[0][0],
        rs_global_semaphores=st.pools[1][0],
        ag_global_semaphores=st.pools[2][0],
        math_op=ttnn.ReduceType.Sum,
        memory_config=DRAM,
    )
    ttnn.deallocate(summed)
    _dbg(dev, "all_reduce_async")
    part = ttnn.mesh_partition(input_tensor=ar, dim=3, cluster_axis=1, memory_config=DRAM)  # [1,1,32,896]
    _dbg(dev, "mesh_partition")
    ttnn.deallocate(ar)
    routed = ttnn.reshape(part, [1, TOKENS_PER_DEV, HIDDEN // NUM_REPL], memory_config=DRAM)  # [1,32,896]
    ttnn.deallocate(part)
    # Match the sparse path's ttnn_typecast_101 dtype (BFLOAT16, moe_test.py:760). The downstream
    # add (moe_test.py:913) does add(matmul_33[bf16], typecast_101); feeding FLOAT32 here mismatches
    # the bf16 tile layout the add expects -> scrambled values. Return BFLOAT16 like the sparse path.
    routed = ttnn.typecast(routed, ttnn.DataType.BFLOAT16, memory_config=DRAM)
    if os.environ.get("MOE_ZERO_ROUTED"):
        # DIAGNOSTIC: zero the routed AFTER running the full combine+tail, to separate the routed
        # VALUE (→ e2e returns to 0.955 if this fixes it) from tail CCL side-effects on the shared
        # path (→ e2e stays ~0.778 even with routed zeroed).
        routed = ttnn.multiply(routed, 0.0, memory_config=DRAM)
    return routed
