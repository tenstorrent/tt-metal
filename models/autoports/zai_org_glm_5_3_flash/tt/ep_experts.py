# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b (PRODUCTION_PATH.md M5, the 45-layer image): EP-SHARDED routed-expert weights for the TP mesh.

Stage 14's TP sparse_kda layer (tt/tp8_sparse.py) routes EP8 — chip r runs experts [E_loc r, E_loc (r + 1)), E_loc =
288 / 8 = 36 — but builds its MoE with the decoder's REPLICATED expert tensors (all 288 on every chip: ~4.1 GB per
layer, so a handful of sparse layers fit, not 42). Here every chip holds only its own 36:

  slot i (0 <= i < E_loc) = ONE mesh tensor whose chip-r shard is expert E_loc r + i (`ttnn.from_host_shards`, the
  fused_experts bfloat4_b ND-sharded DRAM layout, the same bytes as the replicated load), allocated lock-step -> the
  same address on every chip.

The returned gate_up / down lists are indexed by the GLOBAL expert id (entry e = slot e mod E_loc), so the code that
builds chip r's expert address table from entries [E_loc r, E_loc (r + 1)) — TPSparseKDALayer — gets the slot addresses
for every chip unchanged, and ReaderExperts' e_lo (compile time, per chip) maps a routed id to its slot. 42 sparse layers
x 4.1 GB / 8 = ~21.4 GB per chip instead of 172 GB."""

from __future__ import annotations

import os

import ttnn


class EpList:
    """global expert id -> the slot tensor holding it on its owning chip (slot = id mod E_loc)"""

    def __init__(self, slots: list, num_experts: int):
        self.slots, self.n = list(slots), int(num_experts)

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self.slots[e % len(self.slots)] for e in range(*k.indices(self.n))]
        if not -self.n <= k < self.n:
            raise IndexError(k)
        return self.slots[k % self.n % len(self.slots)]

    def __iter__(self):
        return (self[e] for e in range(self.n))


def load_ep_experts(source, layer: int, num_experts: int, mesh, cache_dir: str, keep: bool | None = None):
    """(gate_up, down) EpLists of `layer`'s routed experts, EP-sharded over `mesh` (see the module docstring).
    The fused_experts gate_up permutation is read from the shared derived cache (`cache_dir`) when it is there; a
    missing one is permuted into a temporary file and removed after the load unless keep (GLM53_EP_KEEP_PERMUTED=1):
    the 42-layer image is ~114 GB of derived files, more than the box's root disk holds (2026-10-20 06:20Z: a
    full-model load filled it)."""
    if keep is None:
        keep = os.environ.get("GLM53_EP_KEEP_PERMUTED", "0") == "1"
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FE_CORES, _nd_dram, _permuted_gate_up_file

    ndev = int(mesh.get_num_devices())
    assert num_experts % ndev == 0, (num_experts, ndev)
    e_loc = num_experts // ndev
    gu_cfg = dn_cfg = None
    gu_slots, dn_slots = [], []
    for i in range(e_loc):
        gus, dns = [], []
        for r in range(ndev):
            e = e_loc * r + i
            ge = source._entry(layer, f"mlp.experts.{e}.gate_up_proj")
            de = source._entry(layer, f"mlp.experts.{e}.down_proj")
            assert ge["dtype"] == "bfloat4_b" and de["dtype"] == "bfloat4_b", (ge["dtype"], de["dtype"])
            src = os.path.join(source.cache_dir, ge["path"])
            dst = os.path.join(cache_dir, f"L{layer:02d}", f"{e:03d}.gate_up_proj.{ge['sha256'][:16]}.tensorbin")
            transient = not os.path.exists(dst) and not keep
            if transient:  # permuted into a temporary file, loaded, removed (one expert's ~9 MB at a time; under the
                # coordinator's scratch area for generated caches, 06:31Z rule)
                tmp = os.environ.get("GLM53_EP_SCRATCH", "/home/alex/glm53/scratch-big/14b")
                os.makedirs(tmp, exist_ok=True)
                dst = os.path.join(tmp, f"ep_gate_up.{os.getpid()}.L{layer:02d}.{e:03d}.tensorbin")
            if not os.path.exists(dst):
                _permuted_gate_up_file(src, dst, ge["shape"], 576)
            if gu_cfg is None:
                gu_cfg = _nd_dram(ge["shape"][0], ge["shape"][1] // FE_CORES)
                dn_cfg = _nd_dram(de["shape"][0], de["shape"][1] // FE_CORES)
            gus.append(ttnn.load_tensor(dst))
            if transient:
                os.remove(dst)
            dns.append(ttnn.load_tensor(os.path.join(source.cache_dir, de["path"])))
        gu_slots.append(ttnn.to_device(ttnn.from_host_shards(gus, mesh.shape), mesh, memory_config=gu_cfg))
        dn_slots.append(ttnn.to_device(ttnn.from_host_shards(dns, mesh.shape), mesh, memory_config=dn_cfg))
    return EpList(gu_slots, num_experts), EpList(dn_slots, num_experts)


def build_moe_ep(mesh, cfg, source, layer: int, ocfg, routed: bool = True):
    """tt/tp8_sparse.build_moe with EP-sharded routed experts (router, router tail and shared expert replicated as
    there). routed=False: no routed-expert weights (the layer brings its own: Stage 7a's sliced XSL layout,
    GLM53_TP_XSL_SLICED — the EP copy would double the expert bytes per chip)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedMoE, FusedRouter, FusedSwiGLU
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
        OptimizedMoE,
        OptimizedRouter,
        OptimizedSwiGLU,
        _check_expert_weights,
        _upgrade,
    )

    dev = lambda name: source.device(layer, name, mesh)  # noqa: E731
    host = lambda name: source.host(layer, name)  # noqa: E731
    shared = FusedSwiGLU(mesh, dev("mlp.shared_experts.gate_proj.weight"), dev("mlp.shared_experts.up_proj.weight"),
                         dev("mlp.shared_experts.down_proj.weight"), limit=cfg.swiglu_limit)  # fmt: skip
    shared.fcfg = ocfg
    router = FusedRouter(mesh, dev("mlp.gate.weight"), host("mlp.gate.e_score_correction_bias"),
                         num_experts=cfg.n_routed_experts, top_k=cfg.num_experts_per_tok, fcfg=ocfg)  # fmt: skip
    if routed:
        gate_up, down = load_ep_experts(source, layer, cfg.n_routed_experts, mesh, ocfg.fused_experts_cache_dir)
    else:
        gate_up, down = [None] * cfg.n_routed_experts, [None] * cfg.n_routed_experts
    moe = FusedMoE(mesh, router, gate_up, down, shared, intermediate=cfg.moe_intermediate_size, hidden=cfg.hidden_size,
                   limit=cfg.swiglu_limit, scaling=cfg.routed_scaling_factor, cfg=ocfg)  # fmt: skip
    if routed:
        _check_expert_weights(source, layer, moe.gate_up, moe.down)  # shape / dtype per expert id (its slot's spec)
    _upgrade(moe.shared, OptimizedSwiGLU, ocfg, "shared")
    _upgrade(moe.router, OptimizedRouter, ocfg)
    moe.__class__ = OptimizedMoE
    moe.ep_sharded = True
    return moe
