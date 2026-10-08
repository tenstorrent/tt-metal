# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RING decode MoE for MiMo-V2.6-Flash (bfp8 ring-layout expert weights, a SECOND weight copy next to the unified per-expert prefill copy).

Pipeline per layer (the DSV4 decode MoE block pattern on the generic TTMoEDecode module): router_select -> all_to_all_dispatch_metadata -> moe_compute (ring matmul with fused
combine, HiFi3 + fp32 accumulate) -> tilize + deepseek_moe_fast_reduce_nc_fused (score-weighted sum over k) -> reduce_scatter over the mesh columns (hidden-sharded) -> all_gather over the
columns (full hidden, replicated like the rest of the decode path).

Interface (same as tt/moe_decode.MiMoMoEDecode, so tt/layer.py can pick either):
    ex  = RingExpertWeights(mesh, layer)                     # ring weights of one layer (NFS cache MIMO_WCACHE, default /mnt/tt-data/ssinghal/mimo-wcache)
    dec = MiMoRingMoEDecode(mesh, gate, ex)                  # gate: MiMoGate
    out = dec.forward(x, mask)                               # x [1,1,T,4096] bf16 tile per device (T = tokens of this mesh row, 1..32), mask: persistent live mask [1,1,64,1]
    mask helpers make_mask_host / alloc_mask / update_mask are the MiMoMoEDecode ones.
Numerics: the process environment of the moe_compute program must be MOE_COMPUTE_BFP8_WEIGHTS=1, MOE_COMPUTE_FP32_ACC=1, MOE_COMPUTE_FIDELITY=HiFi3 (set by default at import; LoFi/HiFi2 are
NOT accurate: alpha/rel L2 gates fail) and the patched _ttnncpp.so (moe_compute env knobs). Dead slots (mask 0) get balanced routing with weight 0 (no zero-token chips).
"""
import os

import torch

os.environ.setdefault("MOE_COMPUTE_BFP8_WEIGHTS", "1")
os.environ.setdefault("MOE_COMPUTE_FP32_ACC", "1")
os.environ.setdefault("MOE_COMPUTE_FIDELITY", "HiFi3")
os.environ.setdefault(
    "MOE_COMPUTE_FID_W2", "0"
)  # W2 (down proj) matmul at LoFi: numerics identical to HiFi2, within noise of HiFi3, -11% ring MoE time (needs a lib with the per-matmul fidelity knob, else ignored)

import ttnn  # noqa: E402
from models.common.modules.moe.tt_moe_decode import (  # noqa: E402
    TTMoEDecode,
    _TTMoEDecodeBuffers,
    _TTMoEDecodeExpertState,
)
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.moe_decode import MiMoMoEDecode, dead_routing  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.moe_prefill import HIDDEN, TOPK, free  # noqa: E402

CFG_TEXT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "mimo_v26.yaml")).read()
WCACHE = os.environ.get("MIMO_WCACHE", "/mnt/tt-data/ssinghal/mimo-wcache")
_CFG, _BUF = {}, {}


def ring_config(T, mesh):
    key = (id(mesh), T)
    if key not in _CFG:
        cfg = TTMoEDecodeConfig.from_yaml(
            CFG_TEXT.replace("batch_per_device: 4\n", f"batch_per_device: {T}\n", 1), topology=ttnn.Topology.Linear
        )
        if tuple(mesh.shape) != tuple(cfg.mesh_shape):
            cfg = cfg.with_mesh_shape(tuple(mesh.shape))
        if cfg.num_fast_reduce_outputs == 1:  # generic reduce_scatter path wants interleaved DRAM
            cfg = cfg.model_copy(
                update={"reduce": cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})}
            )
        _CFG[key] = cfg
    return _CFG[key]


_WEIGHTS = __import__("weakref").WeakSet()


def release_ring_buffers(mesh):
    """Free every scratch/semaphore set of this mesh (their L1 clashes with the static CBs of prefill programs, e.g. rms_norm, when a prefill follows decode in
    the same process: Generator.reconfigure calls this after closing the decode trace). Rebuilt lazily on the next decode forward (not traced: needs a warm-up).
    """
    import gc

    for k in [k for k in _BUF if k[0] == id(mesh)]:
        _BUF.pop(k)
        for w in list(_WEIGHTS):
            w._decs.pop(k[1], None)
    gc.collect()


def ring_buffers(T, mesh):
    """ONE persistent scratch/semaphore set per (mesh, T), shared by all layers (each set costs ~13 KB of L1 per bank for the rest of the run)."""
    key = (id(mesh), T)
    if key not in _BUF:
        if os.environ.get("MIMO_RING_KEEP_ALL_T", "0") != "1":
            # a new T (Generator.reconfigure to another U): release the scratch sets of the other T of this mesh (their L1 would otherwise stay allocated for the rest of the
            # run and clash with the statically allocated CBs of later programs, e.g. prefill rms_norm). Their traces are closed by reconfigure.
            for k in [k for k in _BUF if k[0] == id(mesh)]:
                old = _BUF.pop(k)
                for w in list(_WEIGHTS):
                    w._decs.pop(k[1], None)
                del old
            import gc

            gc.collect()
        from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size

        cfg = ring_config(T, mesh)
        bd = cfg.buffers.model_dump()
        bd["compute_tilize_drain_core"] = ttnn.experimental.get_moe_tilize_drain_core(
            mesh,
            cfg.compute.output_height_shard_dim,
            auto_output_width_shard_dim(cfg.hidden_size, matmul_ring_size=effective_matmul_ring_size(mesh)),
            cfg.hidden_size,
            mux_core_range_set=cfg.compute.mux_core_range_set,
        )
        _BUF[key] = _TTMoEDecodeBuffers(mesh, **bd)
    return _BUF[key]


class RingExpertWeights:
    """Ring-layout bfp8 weights of ONE layer (batch independent: shared by every T). Built from the same bf16-dequantised experts as the unified copy;
    the packed+quantized host tensors are cached on NFS (MIMO_WCACHE/ring_L<layer>) so the conversion is not repeated.
    """

    def __init__(self, mesh, layer, w_of_expert=None):
        self.mesh, self.layer = mesh, layer
        cfg = ring_config(32, mesh)
        cdir = os.path.join(WCACHE, f"ring_L{layer}")
        files = _TTMoEDecodeExpertState._cache_files(cdir, None, False)
        warm = all(os.path.exists(f) for f in files.values())
        w0 = w1 = w2 = None
        if not warm:
            if w_of_expert is None:
                from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts

                W = load_layer_experts(layer)
                w_of_expert = lambda e: W[e]
            w0 = torch.stack([w_of_expert(e)[0] for e in range(256)])[None]
            w1 = torch.stack([w_of_expert(e)[1] for e in range(256)])[None]
            w2 = torch.stack([w_of_expert(e)[2] for e in range(256)])[None]
        os.makedirs(cdir, exist_ok=True)
        self.state = _TTMoEDecodeExpertState(mesh, w0, w1, w2, **cfg.experts.model_dump(), weight_cache_dir=cdir)
        self.warm = warm
        self._decs = {}
        _WEIGHTS.add(self)

    def decode(self, T):
        if T not in self._decs:
            d = object.__new__(TTMoEDecode)
            d.config = ring_config(T, self.mesh)
            d.expert_state = self.state
            d.buffers = ring_buffers(T, self.mesh)
            self._decs[T] = d
        return self._decs[T]


class MiMoRingMoEDecode(MiMoMoEDecode):
    """Ring decode MoE; inherits the mask helpers (make_mask_host / alloc_mask / update_mask) of MiMoMoEDecode."""

    def __init__(self, mesh, gate, experts, router="fast", dead="spread32"):
        self.mesh, self.gate, self.experts, self.router, self.dead = mesh, gate, experts, router, dead
        self._sp = {}

    def _dead_consts(self, T):
        if T not in self._sp:
            rep = ttnn.ReplicateTensorToMesh(self.mesh)
            sidx, sw = dead_routing(self.dead)
            mk = lambda t: ttnn.from_torch(
                t.reshape(1, 1, -1, TOPK)[:, :, :T].bfloat16(),
                device=self.mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
            )
            ones = ttnn.from_torch(
                torch.ones(1, 1, T, 1).bfloat16(),
                device=self.mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
            )
            self._sp[T] = (mk(sidx), mk(sw), ones)
        return self._sp[T]

    def _route(self, x, mask):
        T = x.shape[2]
        if self.router == "fast":
            wts, idx = self.gate.route_v(x, "sel_mm1x8")
        else:
            wts, idx = self.gate.route(x, "exact")
        if mask is None:
            return wts, idx
        sp_i, sp_w, ones = self._dead_consts(T)
        m = ttnn.slice(mask, [0, 0, 0, 0], [1, 1, T, 1])
        i = ttnn.typecast(ttnn.to_layout(ttnn.reshape(idx, [1, 1, T, TOPK]), ttnn.TILE_LAYOUT), ttnn.bfloat16)
        w = ttnn.to_layout(ttnn.reshape(wts, [1, 1, T, TOPK]), ttnn.TILE_LAYOUT)
        dead = ttnn.subtract(ones, m)
        i2 = ttnn.add(ttnn.multiply(i, m), ttnn.multiply(sp_i, dead))
        w2 = ttnn.add(ttnn.multiply(w, m), ttnn.multiply(sp_w, dead))
        i2 = ttnn.reshape(ttnn.to_layout(ttnn.typecast(i2, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, TOPK])
        w2 = ttnn.reshape(ttnn.to_layout(w2, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, TOPK])
        return w2, i2

    def forward(self, x, mask=None, upto=99):
        """upto (stage timing): 0 router (+dead handling), 1 dispatch, 2 moe_compute, 3 tilize+fast_reduce, 99 full (reduce_scatter + all_gather)."""
        T = x.shape[2]
        wts, idx = self._route(x, mask)
        if upto == 0:
            return wts, idx
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, HIDDEN])
        dec = self.experts.decode(T)
        out = _staged(dec, x_rm, wts, idx, upto)
        if upto != 99:
            return out
        full = ttnn.all_gather(out, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        free(out)
        return full


def _staged(dec, tt_x, tt_scores, tt_indices, upto=99):
    """TTMoEDecode.forward, optionally cut after a stage (1 dispatch, 2 moe_compute, 3 tilize+fast_reduce)."""
    if upto == 99:
        return dec.forward(tt_x, tt_scores, tt_indices, layer_id=0)
    (xi, dx), (ii, di), (si, ds) = dec._format_dispatch_inputs(tt_x, tt_indices, tt_scores)
    sp, oi, osc = ttnn.experimental.all_to_all_dispatch_metadata(
        xi,
        ii,
        si,
        dec.expert_state.tt_expert_mapping,
        **dec.config.dispatch.model_dump(),
        output_tensors=dec.buffers.tt_dispatch_output_tensors,
        cross_device_semaphore=dec.buffers.dispatch_global_semaphore,
    )
    if upto == 1:
        return sp
    _, _, _, l1, _, comb = ttnn.experimental.moe_compute(
        sp,
        oi,
        osc,
        dec.expert_state.tt_expert_mapping,
        dec.expert_state.tt_w0_w1,
        dec.expert_state.tt_w2,
        layer_id=0,
        **dec.config.compute.model_dump(),
        optional_output_tensor=dec.buffers.tt_combine_output,
        optional_cross_device_semaphore=dec.buffers.combine_global_semaphore,
    )
    if upto == 2:
        return comb
    un = dec._pad_for_fast_reduce(ttnn.unsqueeze(comb, dim=1))
    if dec.config.use_post_combine_tilize:
        til = ttnn.experimental.deepseek_moe_post_combine_tilize(un, **dec.config.post_combine_tilize.model_dump())
    else:
        shp = list(un.shape)
        shp[2] = ((shp[2] + 31) // 32) * 32
        til = ttnn.tilize_with_val_padding(
            un, output_tensor_shape=shp, pad_value=0.0, **dec.config.tilize_with_val_padding.model_dump()
        )
    red = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        til, ii, dec.expert_state.tt_expert_mapping, **dec.config.reduce.model_dump(), scores_tensor=tt_scores
    )
    return red[0]


# ----------------------------------------------------------------------------- DRAM headroom / automatic fallback
RING_MB_PER_BANK_PER_LAYER = (
    27.0  # measured: 1268 MB per bank for 47 layers (ring copy ~214 MB per chip = 27 MB per bank per layer)
)
_DECISION = {}


def free_dram_gib_per_chip(mesh):
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return mv.total_bytes_free_per_bank * mv.num_banks / 2**30


def ring_enabled(mesh, layers_to_build=None, min_headroom_gib=1.0, log=print):
    """Decide ONCE per process whether the ring copy may be built: ring only if the free DRAM per chip now, minus the ring copy of all layers still to build, leaves >= min_headroom_gib
    (MIMO_RING_MIN_HEADROOM_GIB overrides, MIMO_RING_FORCE=1 skips the check); otherwise False (caller falls back to the unified decode MoE) and a line is logged. The check must run
    AFTER the KV pool / prefill buffers are allocated (first decode call), layers_to_build defaults to MIMO_RING_LAYERS (47).
    """
    key = id(mesh)
    if key in _DECISION:
        return _DECISION[key]
    n = int(layers_to_build or os.environ.get("MIMO_RING_LAYERS", "47"))
    need = n * RING_MB_PER_BANK_PER_LAYER * 8 / 1024
    free = free_dram_gib_per_chip(mesh)
    margin = float(os.environ.get("MIMO_RING_MIN_HEADROOM_GIB", min_headroom_gib))
    ok = os.environ.get("MIMO_RING_FORCE", "0") == "1" or free - need >= margin
    log(
        f"[moe_ring] free DRAM {free:.2f} GiB per chip, ring copy of {n} layers needs {need:.2f} GiB, headroom after {free - need:.2f} GiB "
        f"(min {margin:.2f}): {'RING decode MoE enabled' if ok else 'NOT enough room -> falling back to the unified decode MoE'}"
    )
    _DECISION[key] = ok
    return ok
