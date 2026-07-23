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
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_num_devices") and hasattr(device, "get_device_ids")


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
    """Full CoreGrid from HUNYUAN_MM_FULLGRID=1 (else None => op default). Forces
    matmuls that don't already pin a grid (e.g. the gate_up + attn linears) onto
    the whole grid for max DRAM-read/compute throughput."""
    if os.environ.get("HUNYUAN_MM_FULLGRID") != "1":
        return None
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=g.y, x=g.x)


def _ccl_links():
    """num_links for collectives from HUNYUAN_CCL_LINKS (default 1)."""
    try:
        return max(1, int(os.environ.get("HUNYUAN_CCL_LINKS", "1")))
    except ValueError:
        return 1


class _TtMoE:
    def __init__(self, device, torch_module):
        self.device = device
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
            self._build_sharded(torch_module)
        else:
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
        assert self.num_experts % tp == 0, f"TP={tp} must divide num_experts={self.num_experts} for expert-parallel MoE"
        epd = self.num_experts // tp
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
                for d in range(tp)
            ],
            dim=0,
        )  # [tp, H, 2*epd*I]
        dn_stack = torch.stack(
            [torch.cat([dn_t[d * epd + e] for e in range(epd)], dim=0) for d in range(tp)], dim=0
        )  # [tp, epd*I, H]
        self.exp_gu_cat = self._shard(gu_cat, dim=0, dtype=ttnn.bfloat4_b)  # per TP device [1, H, 2*epd*I]
        self.exp_down_stack = self._shard(dn_stack, dim=0, dtype=ttnn.bfloat4_b)  # per TP device [1, epd*I, H]

        # per-TP-device selection+expand matrix: picks this device's expert
        # columns out of the replicated router AND repeats each expert's weight
        # across its I-wide down-matmul block, so `router @ sel` directly yields
        # the [1, S, epd*I] per-column router weights the merged down matmul needs
        # (no reshape/broadcast to expand epd -> epd*I).
        sel = torch.zeros(tp, self.num_experts, epd * I)
        for d in range(tp):
            for e in range(epd):
                sel[d, d * epd + e, e * I : (e + 1) * I] = 1.0
        self.sel = self._shard(sel, dim=0)

        # shared expert REPLICATED (added once, after the routed all-reduce)
        if self.use_shared:
            self.shared_gu = self._repl(torch_module.shared_mlp.gate_and_up_proj.weight.t().contiguous())
            self.shared_down = self._repl(torch_module.shared_mlp.down_proj.weight.t().contiguous())
            self.shared_inter = int(torch_module.shared_mlp.gate_and_up_proj.weight.shape[0] // 2)

    # ------------------------------------------------------------------
    def _mesh_reduce(self, x):
        """All-reduce (sum) a per-device partial across the TP mesh axis.

        Fused ring `ttnn.all_reduce` (cluster_axis=TP axis) instead of the naive
        `all_gather(dim=0)+sum`: the old path materialised [tp, S, hidden] (tp× the
        bytes) on every chip before a local reduce; the ring all_reduce moves ~2×
        the shard bytes/chip and drops the separate sum — the exact prefill-MoE
        reduce gpt_oss/gemma4/deepseek use. Same math, same shape."""
        return ttnn.all_reduce(x, cluster_axis=self.tp_axis, num_links=_ccl_links(), topology=ttnn.Topology.Linear)

    def _swiglu(self, x, gu_w, down_w, inter):
        gu = ttnn.linear(x, gu_w, compute_kernel_config=_mm_cfg(), core_grid=_mm_grid(self.device))
        x1 = ttnn.slice(gu, [0, 0, 0], [gu.shape[0], gu.shape[1], inter])
        x2 = ttnn.slice(gu, [0, 0, inter], [gu.shape[0], gu.shape[1], 2 * inter])
        ttnn.deallocate(gu)
        # SwiGLU: fuse silu into the multiply (silu(x2) * x1 in one op).
        act = ttnn.multiply(x2, x1, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        out = ttnn.linear(act, down_w, compute_kernel_config=_mm_cfg(), core_grid=_mm_grid(self.device))
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
        gu = ttnn.matmul(
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
        combined = ttnn.matmul(
            act, self.exp_down_stack, core_grid=self.mm_core_grid, compute_kernel_config=_mm_cfg()
        )  # [1, S, H] (down + expert-sum fused)
        ttnn.deallocate(act)

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


def build(device, torch_module=None):
    if torch_module is None:
        raise RuntimeError("mo_e native port requires the HF torch_module to extract weights.")
    return _TtMoE(device, torch_module)


def mo_e(device, torch_module=None):
    return build(device, torch_module)
