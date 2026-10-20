# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 routed experts on the Kimi K3 token-split MoE block (GLM_EXPERTS_MODE=ts; a hack until the block has an API
for several models): the block's routed_only mode, split layout with the full-mesh MoE input only.

    x [S/n, H] and the router's top-k of this chip's own tokens
      -> routing rows all-gathered, the combine planner (every per-call table on the device)
      -> sparse dispatch (sparse_all_gather_2d: a token's row only to the chips holding one of its experts)
      -> flat_routed_expert (this layer's weights) with the router-forwarded combine fused in: each y row from the down
         cores' L1 to its home chip's fp32 accumulators -> this chip's own tokens' sum [S/n, H]

One block per (mesh, chunk) serves every layer (built at the first call, after the model's other ops ran once: the
arena's address); each layer has its own FlatRoutedExpert (the weights) of the block's plan: 64 gate/up cores with
DRISC weight readers only (the Tensix reader cells go to the combine), 32 down cores, two x helpers. The shared expert
keeps its TP path on the gathered x tiles (one full-mesh gather here: the dispatch is sparse).
Needs fabric_payload_bytes >= 8224 (a 8192 B token row + the combine's 32 B trailer in one packet).
"""

from __future__ import annotations

import json
import os

from ttnn.bringup.flat_routed_expert_ttnn.flat_layouts import GLM_FLASH

import ttnn
from models.demos.glm53_flash_d_p.tt.experts import LazyExpertWeights
from models.demos.glm53_flash_d_p.tt.experts_ag import (
    _dram,
    all_gather_rows,
    default_gids,
    flat_cache_dir,
    hook_ctx,
    moe_hooks,
)

# the routed experts' layout: DRISC-only weight readers, 32 down cores (128 output tile columns, 4 a core), one x
# helper (a second one costs ~104 KB of every core's arena (1208 vs 1104 KB): in the whole model the dispatch's 103 KB
# manifest then no longer fits beside it at S 640; no gain on GLM in the overlap harness)
FLAT_LAYOUT = GLM_FLASH
_BLOCKS = {}
_HOST = {}
_DONE = {}  # mesh -> the done words every layer's flat expert shares
_GX = {}  # (mesh, T) -> the gathered x tiles and the full-mesh topology


class TtExpertsTs:
    """GLM routed experts on the K3 token-split block; the same call as TtExpertsAg (split + full_mesh only)."""

    def __init__(
        self,
        mesh,
        layer: int,
        torch_weights,
        num_experts: int,
        emb_dim: int,
        hidden_dim: int,
        top_k: int = 8,
        limit: float = 10.0,
        max_seq_len: int = 8192,
        weights_dtype=ttnn.bfloat4_b,
        cache: bool = True,
        num_links: int = 2,
        shared_weights=None,
        shared_axis: int = 1,
        shared_impl: str = os.environ.get("GLM_SE_IMPL", "ttnn"),  # (probe: ttnn / unified / flat)
    ):
        """shared_weights (gate, up [I_s, H], down [H, I_s] torch, HF layout): the shared expert folded into the block
        (split over each mesh row's chips, its partials summed by the combine): the call then returns routed + shared
        and gathers no x for a separate shared expert. shared_axis: its group is each mesh row (1) or column (0).
        shared_impl: ttnn (matmuls + glu_act; default), unified (the bring-up unified op, high precision, ClampedSiluGlu
        fused, x tilized inside: L4 HiFi3 chunk 5120 366 + 51 untilize vs 363 us; 2048 174 vs 181) or flat (the flat expert
        op, no tilize / untilize: 424 vs 338 us at HiFi2 - too few gate/up cores for one I 1024 expert).
        """
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

        assert float(limit) == 10.0, f"clamped_silu is fixed at limit 10, got {limit}"
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        assert num_experts % n == 0, tuple(mesh.shape)
        self.mesh, self.layer, self.links = mesh, layer, num_links
        self.E, self.K, self.H, self.I = num_experts, top_k, emb_dim, hidden_dim
        self.epc = num_experts // n
        self.max_seq_len = max_seq_len
        self.gids = default_gids(n, self.epc)
        self.gathered_x = None
        self.shared = None
        if shared_weights is not None:
            from models.demos.glm53_flash_d_p.tt.shared_tp_row import TtSharedFlat, TtSharedTpRow

            if shared_impl == "unified":  # (the bring-up unified op, high precision)
                from models.demos.glm53_flash_d_p.tt.shared_tp_row import TtSharedUnified

                self.shared = TtSharedUnified(mesh, *shared_weights, limit=limit, axis=shared_axis)
            elif shared_impl == "flat":  # (one set of done words for every layer's shared expert)
                d0 = _DONE.get(("se", id(mesh)))
                self.shared = TtSharedFlat(
                    mesh,
                    *shared_weights,
                    limit=limit,
                    axis=shared_axis,
                    max_rows=max_seq_len,
                    done=d0,
                    **json.loads(os.environ.get("GLM_SE_FLAT_OPTS", "{}")),  # (layout probes: x_bf16, layout)
                )
                if d0 is None:
                    _DONE[("se", id(mesh))] = self.shared.flat.done
            else:
                prec = {}  # (probe GLM_SE_PREC=bf16_hifi4: the model MLP's precision, for accuracy A/B)
                if os.environ.get("GLM_SE_PREC") == "bf16_hifi4":
                    prec = dict(w_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi4)
                elif os.environ.get("GLM_SE_PREC") == "bf8_hifi2":
                    prec = dict(fidelity=ttnn.MathFidelity.HiFi2)
                self.shared = TtSharedTpRow(mesh, *shared_weights, limit=limit, axis=shared_axis, **prec)
        wdtype = {ttnn.bfloat4_b: "bf4", ttnn.bfloat8_b: "bf8"}[weights_dtype]
        fake = isinstance(torch_weights, str)
        prefix = None
        if cache and not fake:
            d = flat_cache_dir(mesh)
            d.mkdir(parents=True, exist_ok=True)
            prefix = str(d / f"layer_{layer}")

        def weights_per_dev():
            t = lambda w: w.T.contiguous()  # noqa: E731
            out = []
            for gl in self.gids:
                dev = []
                for g in gl:
                    w = torch_weights[g]
                    dev.append((t(w["gate_proj"]), t(w["up_proj"]), t(w["down_proj"])))
                out.append(dev)
            return out

        # (the per-expert token cap: the whole chunk, as the all-gather path)
        self.flat = FlatRoutedExpert(
            mesh,
            torch_weights if fake else weights_per_dev,
            m=max_seq_len,
            H=emb_dim,
            I=hidden_dim,
            gids=self.gids,
            n_global=num_experts,
            wdtype=wdtype,
            act="clamped_silu",
            pin=1,
            cache_prefix=prefix,
            layout=FLAT_LAYOUT,
        )
        # (one done-words buffer for every layer: 2 KB of L1 on each down coordinator per layer otherwise - 86 KB over
        # 42 layers, under the token-split block's arena; launches leave it zeroed and the layers run one at a time)
        shared = _DONE.get(id(mesh))
        if shared is None or shared[0] is not mesh:
            _DONE[id(mesh)] = (mesh, self.flat.done)
        else:
            ttnn.deallocate(self.flat.done)
            self.flat.done = shared[1]

    def _block(self, s, joint=False):
        from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import K3Dims, TtK3TokenSplitMoE

        key = (id(self.mesh), s, self.H, self.K, self.E, self.shared.axis if self.shared is not None else None, joint)
        hit = _BLOCKS.get(key)
        # (built at the first call; rebuilt if L1 the model allocated since - e.g. a later op's first-call semaphores -
        # moved the arena: its address is baked into the kernels)
        if hit is not None and hit.mesh is self.mesh and not hit.arena_fits():
            del _BLOCKS[key]
            hit = None
        if hit is None or hit.mesh is not self.mesh:
            from loguru import logger

            logger.info(f"GLM experts ts: building the token-split block (S {s} per chip, layer {self.layer})")
            n = self.mesh.get_num_devices()
            d = K3Dims(H=self.H, latent=self.H, I=self.I, E=self.E, K=self.K, act="clamped_silu")
            hit = _BLOCKS[key] = TtK3TokenSplitMoE(
                self.mesh,
                None,
                d,
                s,
                cap=self.max_seq_len,
                rows_cap=-(-n * s * min(self.K, self.epc) // 32) * 32 + 32 * self.epc,
                routed_only=True,
                flat_layout=FLAT_LAYOUT,
                shared_tp=self.shared is not None,
                shared_axis=self.shared.axis if self.shared is not None else 1,
                joint_routing=joint,
                # (hang hunting: block variants, e.g. {"acc_on_ports": false} or {"deterministic": true})
                **json.loads(os.environ.get("GLM_TS_BLOCK_OPTS", "{}")),
            )
            sh_ = hit._arena_spec[4].shard_spec
            ab = int(sh_.shape[0]) * int(sh_.shape[1]) * 4
            logger.info(
                f"GLM experts ts: arena at {hit.arena_addr}, {ab >> 10} KB a core (L1 below it: the dispatch manifest, the "
                f"CBs; without it - the shared expert's time - L1 up to {(hit.arena_addr + ab) >> 10} KB is free)"
            )
            if os.environ.get("GLM_TS_L1_REPORT") == "1":  # (debug: the L1 buffers the model holds besides the arena)
                bufs = [
                    b
                    for b in ttnn._ttnn.reports.get_buffers(self.mesh)
                    if "L1" in str(b.buffer_type) and b.device_id == 0
                ]
                for b in sorted(bufs, key=lambda b: -b.address):
                    logger.info(
                        f"GLM experts ts L1: addr {b.address} size/bank {b.max_size_per_bank} {b.buffer_type} {b.buffer_layout}"
                    )
        return hit

    def _gather_x(self, x, T):
        """The shared expert's input: every chip's own rows as tiles [1, 1, T, H] (one full-mesh gather; the inputs'
        topology relabelled dim-2-over-all-chips as the all-gather path's gather_full)."""
        key = (id(self.mesh), T)
        if key not in _GX:
            rows, cols = tuple(self.mesh.shape)
            dist = ttnn.MeshShape(rows, cols)
            coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
            topo = ttnn.TensorTopology(dist, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
            _GX[key] = (_dram(self.mesh, [1, 1, T, self.H], ttnn.bfloat16, ttnn.TILE_LAYOUT), topo)
        gx, topo = _GX[key]
        x.update_tensor_topology(topo)
        all_gather_rows(x, gx, None, self.links)
        return gx

    def __call__(self, x, dense=None, idx=None, wts=None, split=False, full_mesh=False):
        """x [1, 1, S/n, H] (this chip's own rows), idx / wts [1, 1, S/n, K] -> [1, 1, S/n, H] bf16 TILE, the routed
        experts' sum for this chip's rows (the split residual layout); gathered_x: all S rows as tiles (the shared
        expert's input, valid until the next MoE layer's call). wts None: idx is the router's joint rows (TtRouter.joint:
        32-token pages of ids and weights' bf16 bits)."""
        assert split and full_mesh and idx is not None, "GLM_EXPERTS_MODE=ts: the split layout's full-mesh MoE input"
        joint = wts is None
        s = x.shape[-2]
        n = self.mesh.get_num_devices()
        tmp = []
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
            tmp.append(x)
        if not joint and idx.dtype != ttnn.uint16:
            idx = ttnn.typecast(idx, ttnn.uint16)
            tmp.append(idx)
        if not joint and wts.dtype != ttnn.bfloat16:
            wts = ttnn.typecast(wts, ttnn.bfloat16)
            tmp.append(wts)
        import time as _t

        t0 = _t.perf_counter()
        self.gathered_x = self._gather_x(x, n * s) if self.shared is None else None
        t1 = _t.perf_counter()
        blk = self._block(s, joint)
        x2 = ttnn.reshape(x, [s, self.H])
        t2 = _t.perf_counter()
        hk = moe_hooks()
        ctx = hook_ctx(self.mesh, self.layer, "ts", x, idx, wts) if hk else None
        if hk and hasattr(hk, "before_moe"):  # (hang hunting: GLM_HANG_HOOKS)
            hk.before_moe(ctx)
        op0 = ttnn._ttnn.get_device_operation_id()
        ids2 = idx if joint else ttnn.reshape(idx, [s, self.K])
        wts2 = None if joint else ttnn.reshape(wts, [s, self.K])
        y = blk.routed(x2, ids2, wts2, self.flat, shared=self.shared)
        if hk and hasattr(hk, "after_moe"):
            hk.after_moe(ctx, y)
        if (
            os.environ.get("GLM_TS_OPLOG") == "1"
        ):  # (device op ids per layer's call: maps a triage's hung op id to a layer)
            print(f"[ts-op] layer {self.layer}: ops {op0}..{ttnn._ttnn.get_device_operation_id() - 1}", flush=True)
        if os.environ.get("K3_HOST_LOG") == "1":
            h = _HOST.setdefault("t", [0.0, 0.0, 0.0, 0])
            h[0] += t1 - t0
            h[1] += t2 - t1
            h[2] += _t.perf_counter() - t2
            h[3] += 1
            if h[3] % 42 == 0:
                from loguru import logger

                logger.info(
                    f"GLM ts host ms per call: x gather {1e3 * h[0] / h[3]:.2f}, block lookup + reshape "
                    f"{1e3 * h[1] / h[3]:.2f}, routed {1e3 * h[2] / h[3]:.2f}"
                )
        for t in tmp:
            ttnn.deallocate(t)
        return ttnn.reshape(y, [1, 1, s, self.H])


def build_experts_ts(mesh, loader, cfg, layer: int, max_chunk: int, weights_dtype=ttnn.bfloat4_b) -> TtExpertsTs:
    """GLM_SHARED_EXPERT (spec device.shared_expert): separate, or folded into the block - split over each mesh row
    (folded) or each mesh column (folded_sp)."""
    shared = None
    mode = os.environ.get("GLM_SHARED_EXPERT", "separate")
    assert mode in (
        "separate",
        "folded",
        "folded_sp",
    ), f"device.shared_expert: separate / folded / folded_sp, got {mode}"
    if mode != "separate":
        from models.demos.glm53_flash_d_p.reference.weights import PREFIX

        p = f"{PREFIX}layers.{layer}.mlp.shared_experts."
        shared = tuple(loader.weight(p + f"{n}_proj.weight") for n in ("gate", "up", "down"))
    if not _HOST.get("logged_shared"):  # (the resolved option, once)
        from loguru import logger

        logger.info(f"GLM experts ts: shared expert {mode} (device.shared_expert)")
        _HOST["logged_shared"] = True
    return TtExpertsTs(
        mesh,
        layer,
        "fake" if getattr(loader, "fake", False) else LazyExpertWeights(loader, layer, cfg.n_routed_experts),
        num_experts=cfg.n_routed_experts,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=cfg.num_experts_per_tok,
        limit=cfg.swiglu_limit,
        max_seq_len=max_chunk,
        weights_dtype=weights_dtype,
        num_links=int(os.environ.get("GLM_MOE_LINKS", "1")),
        cache=os.environ.get("GLM_EXPERTS_CACHE", "1") != "0",
        shared_weights=shared,
        shared_axis=0 if mode == "folded_sp" else 1,
    )
