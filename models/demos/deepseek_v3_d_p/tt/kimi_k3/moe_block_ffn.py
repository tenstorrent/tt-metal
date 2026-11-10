# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A prefill model's MoE layers on the MoE block (models/demos/moe_block/tt/moe_block.py, the
ttnn.experimental.moe_block C++ ops): router, shared expert, latent projections and routed experts in one call, on the
post-ffn_norm residual as it is.

    x [1, 1, S/SP, H/TP] bf16 TILE (ffn_norm's output: rows over the SP axis, columns over TP)
    -> MoEBlock -> routed + shared [1, 1, S/SP, H/TP] bf16 TILE (what TtMoe returns, unsqueezed)

Kimi-K3 (kimi_k3/block.py, K3_MOE_IMPL=block) and GLM-5.3 (tt_prefill_block.py, MOE_IMPL=block). The model's own
weights, per layer: the gate (rows permuted to the block's expert order), the shared expert and (K3) the latent pair
from the checkpoint ($KIMI_K3_CKPT / $GLM53_HF_MODEL, fp8 dequantized); the routed experts laid out on the device from
TtMoe's bfp4 ttnn cache (cache_path, K3: $K3_BLOCK_WEKA_CACHE or CI's on Weka) by flat_gather.

Expert order: TtMoe holds checkpoint expert gtab[c, r, le] in slot le of chip (r, c) (ExpertMapping, column-major);
the block holds expert b = (r C + c) EPC + le there. The experts stay where TtMoe put them and the gate's rows (and
bias) are permuted to match.

One MoEBlock (and its MoEBlockResources) per (mesh, chunk) serves every layer, built at the first call (after the
model's own ops placed their L1); each layer holds its MoEBlockWeights. The block's persistent words live in L1_SMALL;
MoEBlockFfnOptions holds the block's few non-default choices (the combine's snap, ...), production's by default.
release_blocks(mesh) frees the blocks' resources (persistent L1_SMALL / L1 / DRAM) at once: call it before closing the
mesh (else they stay allocated until garbage collection, and the resources fail on a closed mesh).
"""

from __future__ import annotations

import json
import os
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.moe_block.tt.flat_gather import flat_face_map, flat_weights_module, gather, gather_batches
from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_block_words_slab
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights, sp_tp, to_mesh_order

LINKS = 2
GATHER_EXPERTS = 4  # (local experts a flat_gather batch)
_WEKA_CACHE = "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3-Cache/Kimi-K3-Cache-prefill"
_BLOCKS: dict = {}  # (id(mesh), chunk, options) -> (mesh, MoEBlockResources, MoEBlock)
_FACE_MAPS: dict = {}  # ((H, I, E, layout), experts a batch) -> gather_batches of flat_face_map: the same every layer


def release_blocks(mesh_device=None):
    """Drop the cached blocks (of mesh_device; None: of every mesh) and release their resources (arena, words) now."""
    for key in [k for k in list(_BLOCKS) if mesh_device is None or k[0] == id(mesh_device)]:
        _, res, _ = _BLOCKS.pop(key)
        res.release()


@dataclass(frozen=True)
class MoEBlockFfnOptions:
    """The MoE block's choices a caller may change; None / the defaults are production's.

    combine_snap: the combine's snap q (a power of two: every routed term rounded to a multiple of q, so the sum is
        bit-identical in any landing order); None: the MoEBlockConfig default (2^-14); 0: off.
    combine_landing: the combine's landing pool (CombineLanding): "rings" (one ring a source chip, in the accumulators'
        L1), "shared" (slot grants from one free list), "flat_cores" (rings in the flat expert cores' free L1),
        "shared_flat_cores" (shared's grants on flat_cores' slots); None: the planner's (rings or flat_cores, ranked).
    sp_axis: the mesh axis SP runs on (1: the 4x4 tray-half stage, SP on the axis-1 ring, TP on the axis-0 line); the
        block's ops run on the transposed mesh (MoEBlockConfig.sp_axis)."""

    combine_snap: Optional[float] = None
    combine_landing: Optional[str] = None
    sp_axis: int = 0


def _combine_config(mb, opts: "MoEBlockFfnOptions"):
    """The combine's CombineConfig for these options (None: MoEBlock's default)."""
    if opts.combine_landing is None:
        return None
    landing = {
        "rings": mb.CombineLanding.Rings,
        "shared": mb.CombineLanding.Shared,
        "flat_cores": mb.CombineLanding.FlatCores,
        "shared_flat_cores": mb.CombineLanding.SharedFlatCores,
    }[opts.combine_landing]
    cc = mb.CombineConfig()
    cc.landing = landing
    return cc


def block_dims(model_cfg):
    """The block's MoEModelDims (GLM-5.3 / Kimi-K3), checked against the model's config - or a config's own
    moe_block_dims() -> (MoEModelDims, flat activation name), for a model the block runs without a branch here (the block
    tests' DeepSeek V4.1-Flash)."""
    own = getattr(model_cfg, "moe_block_dims", None)
    if own is not None:
        return own()
    mb = ttnn.experimental.moe_block
    E, K, H, I = (
        model_cfg.NUM_ROUTED_EXPERTS,
        model_cfg.NUM_EXPERTS_PER_TOKEN,
        model_cfg.EMB_SIZE,
        model_cfg.MOE_INTERMEDIATE_SIZE,
    )
    if (E, K, H, I) == (256, 8, 6144, 2048):  # GLM-5.3
        silu = mb.ExpertActivation(kind=mb.ExpertActivationKind.Silu)
        return (
            mb.MoEModelDims(
                num_experts=E,
                top_k=K,
                emb_dim=H,
                routed_expert_hidden_dim=I,
                routed_expert_activation=silu,
                router=mb.RouterSpec(score_func=mb.RouterScoreFunc.Sigmoid, route_scale=2.5),
                shared_expert_hidden_dim=2048,
                shared_expert_activation=silu,
            ),
            "silu",
        )
    got = (E, K, H, I, model_cfg.ROUTED_EXPERT_HIDDEN_SIZE)
    assert got == (896, 16, 7168, 3072, 3584), f"MoE block: config (E, top-k, H, I, latent) {got}: not GLM-5.3 / K3"
    situ = mb.ExpertActivation(kind=mb.ExpertActivationKind.Situ, beta=4.0, linear_beta=25.0)
    return (
        mb.MoEModelDims(
            num_experts=E,
            top_k=K,
            emb_dim=H,
            routed_expert_hidden_dim=I,
            routed_expert_activation=situ,
            router=mb.RouterSpec(score_func=mb.RouterScoreFunc.Sigmoid, route_scale=1.0),
            shared_expert_hidden_dim=6144,
            shared_expert_activation=situ,
            latent_dim=3584,
            latent_norm=True,
        ),
        "situ",
    )


def _gtab(mesh, epc):
    """gtab[c, r, le]: the checkpoint expert TtMoe holds in slot le of chip (r, c)."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

    R, C = tuple(mesh.shape)
    return ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=R, num_dispatch_groups=C
    )


def _k3_checkpoint() -> Path:
    """Kimi-K3's checkpoint: $KIMI_K3_CKPT, else the prefill runner's $PREFILL_HF_MODEL (its K3 runs set only that)."""
    return Path(os.environ.get("KIMI_K3_CKPT") or os.environ.get("PREFILL_HF_MODEL") or "")


def _cache_dir(mesh):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import _checkpoint_identity

    R, C = tuple(mesh.shape)
    ckpt = _k3_checkpoint()
    root = Path(os.environ.get("K3_BLOCK_WEKA_CACHE") or os.environ.get("TT_KIMI_K3_PREFILL_TTNN_CACHE") or _WEKA_CACHE)
    return root / f"kimi_k3_bh_{R * C}dev" / _checkpoint_identity(ckpt) / f"mesh{R}x{C}.tpaxis1"


def _glm_nonrouted(layer_idx: int, hf_config):
    """GLM-5.3 layer's gate and shared expert from $GLM53_HF_MODEL, fp8 dequantized."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_tensors
    from models.demos.deepseek_v3_d_p.utils.test_utils import dequantize_state_dict

    ckpt = Path(
        os.environ.get("GLM53_HF_MODEL")
        or os.environ.get("PREFILL_HF_MODEL")  # (the prefill runner's)
        or "/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4"
    )
    with (ckpt / "model.safetensors.index.json").open(encoding="utf-8") as f:
        wm = json.load(f)["weight_map"]
    p = f"model.layers.{layer_idx}.mlp."
    names = [k for k in wm if k.startswith(p + "gate.") or k.startswith(p + "shared_experts.")]
    assert names, f"MOE_IMPL=block: no {p}gate / shared_experts keys in {ckpt}"
    t = load_tensors(ckpt, {k: k for k in names})
    if hf_config is None or getattr(hf_config, "quantization_config", None) is None:
        with (ckpt / "config.json").open(encoding="utf-8") as f:  # (only its quantization_config is read)
            hf_config = types.SimpleNamespace(quantization_config=json.load(f).get("quantization_config"))
    return dequantize_state_dict(t, hf_config, dtype=torch.float32), p


def host_weights(mesh, dims, layer_idx: int, hf_config=None):
    """The layer's non-routed weights in the block's expert order: dict of float32 torch tensors (router weight / bias,
    shared gate / up / down, K3: latent down / up / norm)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import (
        GATE_KEYS,
        LATENT_KEYS,
        SHARED_EXPERT_KEYS,
        layer_prefix,
        load_tensors,
    )

    R, C = tuple(mesh.shape)
    epc = dims.num_experts // (R * C)
    gtab = _gtab(mesh, epc)
    perm = torch.tensor(
        [int(gtab[c, r, le]) for r in range(R) for c in range(C) for le in range(epc)], dtype=torch.long
    )
    if dims.latent_dim is None:  # (GLM-5.3)
        t, p = _glm_nonrouted(layer_idx, hf_config)
        sp = p + "shared_experts."
        return {
            "router_weight": t[p + "gate.weight"].float()[perm].contiguous(),
            "router_bias": t[p + "gate.e_score_correction_bias"].float()[perm].contiguous(),
            "gate": t[sp + "gate_proj.weight"].float(),
            "up": t[sp + "up_proj.weight"].float(),
            "down": t[sp + "down_proj.weight"].float(),
        }
    ckpt = _k3_checkpoint()
    assert (ckpt / "model.safetensors.index.json").is_file(), f"K3 MoE block: no checkpoint at {ckpt}"
    p = layer_prefix(ckpt, layer_idx)
    keys = {f"g.{k}": p + v for k, v in GATE_KEYS.items()}
    keys.update({f"s.{k}": p + v for k, v in SHARED_EXPERT_KEYS.items()})
    keys.update({f"l.{k}": p + v for k, v in LATENT_KEYS.items()})
    t = load_tensors(ckpt, keys)
    return {
        "router_weight": t["g.weight"].float()[perm].contiguous(),
        "router_bias": t["g.e_score_correction_bias"].float()[perm].contiguous(),
        "gate": t["s.gate_proj"].float(),
        "up": t["s.up_proj"].float(),
        "down": t["s.down_proj"].float(),
        "latent_down": t["l.down_proj"].float(),
        "latent_up": t["l.up_proj"].float(),
        "latent_norm": t["l.norm"].float(),
    }


def device_sources(mesh, layer_idx: int, epc: int, cache, experts=None):
    """TtMoe's cached routed experts of a layer straight to DRAM (bfp4 as stored, interleaved): flat_gather's sources,
    s = le * 3 + (0 gate, 1 up, 2 down) - each [1, 1, in, out] a chip (W^T: the flat op's orientation). experts: the
    local experts to load (default all epc), s counted from the first."""
    srcs = []
    for le in range(epc) if experts is None else experts:
        for proj in ("gate", "up", "down"):
            hits = sorted(Path(cache).glob(f"layer_{layer_idx}.routed_expert.local_{le}_{proj}_dtype_*.tensorbin"))
            assert hits, f"MoE block: no layer_{layer_idx}.routed_expert.local_{le}_{proj} in {cache}"
            assert "BFLOAT4_B" in hits[0].name, f"flat_gather copies bfp4 faces: {hits[0].name}"
            # (straight to the device, as TtMoe loads its cache - no host tensor of all 32 chips in Python)
            t = ttnn.load_tensor(str(hits[0]), device=mesh)
            if t.memory_config().buffer_type != ttnn.BufferType.DRAM or t.memory_config().is_sharded():
                t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
            srcs.append(t)
    return srcs


class TtK3MoEBlockFfn:
    """One MoE layer on the shared MoE block: __call__(ffn_norm's x [1, 1, S/SP, H/TP]) -> [1, 1, S/SP, H/TP]."""

    def __init__(
        self,
        mesh_device,
        model_cfg,
        layer_idx: int,
        hf_config=None,
        cache_path=None,
        state_dict=None,
        options: Optional[MoEBlockFfnOptions] = None,
    ):
        self.mesh, self.layer_idx, self.hf_config = mesh_device, int(layer_idx), hf_config
        self.options = options or MoEBlockFfnOptions()
        self.cache = Path(cache_path) if cache_path is not None else None
        # (a TtPrefillBlock state dict holding the MoE's host weights - gate_weights, shared_expert_weights,
        # routed_expert_weights, K3: latent_weights - as the synthetic-weight tests give TtMoe: the block's weights are
        # made from them, expert e on chip e // (E / N), instead of from the checkpoint and the TtMoe cache)
        self.host_sd = state_dict if state_dict and state_dict.get("routed_expert_weights") is not None else None
        self.dims, self.act = block_dims(model_cfg)
        self.model_name = model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", "")
        # (this layer's weights in the ops' layouts, made at the first call that needs them: each part kept by what its
        # layout depends on - _packing - so a block with another plan gets weights laid out for it)
        self._heads: dict = {}  # head packing key -> MoEHeadWeights
        self._routed_w: dict = {}  # routed packing key -> (gidx, w_gu, w_d)
        self._w_up = None  # (a latent model's tail W': its layout does not depend on the chunk)
        self._prepared: dict = {}  # (head key, routed key) -> MoEBlockWeights
        self._by_block: dict = {}  # id(block) -> (block, its MoEBlockWeights): a call's lookup, no planning

    def _block(self, chunk, forward_chunks: int = 1):
        opts = self.options
        key = (id(self.mesh), chunk, int(forward_chunks), opts)
        hit = _BLOCKS.get(key)
        if hit is not None and hit[0] is not self.mesh:  # (another mesh at this id: its block goes now)
            _BLOCKS.pop(key)[1].release()
            hit = None
        if hit is None:
            mb = ttnn.experimental.moe_block
            cfg = mb.MoEBlockConfig(
                chunk_tokens=chunk,
                num_links=LINKS,
                persist_l1_small=True,  # (the ops' words in L1_SMALL: MoEBlock)
            )
            cfg.sp_axis = opts.sp_axis
            cfg.forward_chunks = int(forward_chunks)
            if opts.combine_snap is not None:  # (else the config's default snap)
                cfg.combine_snap = opts.combine_snap
            # (the model's slab, the one its mesh opened L1_SMALL for: moe_impl.moe_block_words_slab)
            cfg.words_slab_bytes = moe_block_words_slab(self.model_name)
            dtypes = mb.MoEDtypes()
            t0 = time.time()
            res = mb.MoEBlockResources(self.mesh, self.dims, cfg, dtypes)
            block = MoEBlock(
                self.mesh,
                res,
                dtypes=dtypes,
                max_tokens_per_expert=max(chunk, 256),
                # (the flat plan's own choices: the gate/up read pieces from its weight block's bytes, the sub-blocks
                # fitted into the combine's arena)
                layout=mb.FlatLayoutOptions(),
                combine=_combine_config(mb, opts),
            )
            logger.info(f"MoE block: {opts}, flat layout {block.expert_kw['layout']}")
            logger.info(
                f"MoE block: built for chunk {chunk} at layer {self.layer_idx} in {time.time() - t0:.1f} s - "
                f"accumulators {block.accumulators}, capacity {block.expert_kw['max_tokens_per_expert']}, arena "
                f"{block.arena_spec.memory_config.shard_spec.shape[0] * 64 >> 10} KB a core"
            )
            fit = mb.plan_moe_head(res, dtypes, block.accumulators)
            if fit.ok and len(fit.planned):
                p, lay = fit.planned[0], fit.planned[0].layout
                logger.info(
                    f"MoE block: head engine {lay.engine_columns}x{lay.engine_rows}, MT {lay.row_tiles_per_core}, "
                    f"{lay.row_chunks} chunk(s), KB {lay.k_step_tiles}, "
                    f"predicted {p.predicted_us:.0f} us {fit.note}"
                )
            hit = _BLOCKS[key] = (self.mesh, res, block)
        return hit[2]

    # The flat plan's fields the routed experts' layout is made from (flat_weight_regions / flat_weight_shapes: the
    # gate/up weight slots and their column groups, the down cores' columns and bank rotation, the banks).
    _ROUTED_PACKING = ("banks", "n_rd", "n_rd_sg", "rg", "np", "nu", "nk_gu", "nd", "pcds", "col0s", "wd_rot_tiles")

    def _routed_layout(self, block) -> dict:
        """The flat plan (as_dict) the block's routed experts are laid out for."""
        mb = ttnn.experimental.moe_block
        R, C = tuple(self.mesh.shape)
        N, E = R * C, self.dims.num_experts
        kw = block.expert_kw
        return mb.flat_routed_expert_plan(
            self.mesh,
            self.dims.routed_width,
            self.dims.routed_expert_hidden_dim,
            E // N,
            E,
            kw["max_tokens_per_expert"],
            weights_bf8=False,
            pin=1,
            layout=kw["layout"],
        ).as_dict()

    def _packing(self, block) -> tuple:
        """(head key, routed key): what this layer's prepared weights' layouts depend on, for the block. The chunk picks
        the block's plans, so two blocks may need differently packed weights:
        * the head's (prepare_moe_head_weights): its planned layout's GX x GY matmul cores (the regions, GY K blocks
          dealt round-robin over a column's cores), MT (the gate / up and down pass widths) and K step KB (the blocks'
          K rows), and the shared expert's weight dtype - not NCH or NIT (row chunks / router iterations);
        * the routed experts' (flat_gather): the flat plan's _ROUTED_PACKING fields.
        The tail's W' (prepare_moe_tail_weights) is laid out by the engine's columns and K ranges, from the grid and
        the model's dims - the same at every chunk."""
        mb = ttnn.experimental.moe_block
        fit = mb.plan_moe_head(block.resources, block.dtypes, block.accumulators)
        assert fit.ok and len(fit.planned), f"MoE block layer {self.layer_idx}: the head does not plan: {fit.reason}"
        lay = fit.planned[0].layout
        head = (
            lay.engine_columns,
            lay.engine_rows,
            lay.row_tiles_per_core,
            lay.k_step_tiles,
            str(block.dtypes.shared_weights),
        )
        flat = self._routed_layout(block)
        routed = tuple(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v) for k in self._ROUTED_PACKING for v in [flat.get(k)]
        )
        return head, routed

    def _routed(self, block, sources=None):
        """This layer's flat expert tensors (gidx, w_gu, w_d), laid out on the device from TtMoe's bfp4 cache - or from
        sources(experts): the same plain sources (s = le * 3 + gate / up / down, [1, 1, in, out] bfp4 a chip, DRAM) made
        some other way (_host_sources)."""
        fw = flat_weights_module()
        R, C = tuple(self.mesh.shape)
        N, E = R * C, self.dims.num_experts
        epc, Hr, I = E // N, self.dims.routed_width, self.dims.routed_expert_hidden_dim
        kw = block.expert_kw
        lay = self._routed_layout(block)
        lo = kw["layout"]
        key = (Hr, I, epc, tuple(lo.key()), kw["max_tokens_per_expert"])
        # (the layer's sources go to DRAM GATHER_EXPERTS local experts at a time and are freed after their gather:
        # all 3 epc of them next to the laid-out dest held one layer's routed weights twice - 65 MB a bank on K3, which
        # the last layer of an SC4 stage at 100 slots did not have)
        per = GATHER_EXPERTS
        t0 = time.time()
        batches = _FACE_MAPS.get((key, per))
        if batches is None:
            maps = flat_face_map(Hr, I, epc, lay)
            NT = (Hr // 32) * (I // 32)
            batches = _FACE_MAPS[(key, per)] = gather_batches(maps, NT, epc, per)
            del maps
            logger.info(
                f"MoE block gather: face maps in {time.time() - t0:.0f} s, {len(batches)} batches of {per} experts"
            )
        shapes = fw.flat_weight_shapes(Hr, I, epc, lay)
        dst = {
            n: fw.uninitialized_bank_sharded(*shapes[n], lay["banks"], ttnn.bfloat4_b, self.mesh)
            for n in ("w_gu", "w_d")
        }
        if sources is not None:
            cache = "host weights"  # (the log's label: no TtMoe cache, no checkpoint looked up)
        else:
            cache = self.cache if self.cache is not None else _cache_dir(self.mesh)
        for n in dst:
            pages = (int(dst[n].shape[-2]) // 32) * (int(dst[n].shape[-1]) // 32)
            got = sum(int(sub[n][0].numel()) for _, _, sub in batches)
            assert pages == got, f"{n}: {pages} pages, the face map {got}"
        t_load = t_gather = 0.0
        n_srcs = 0
        for e0, e1, sub in batches:
            t1 = time.time()
            srcs = (
                sources(range(e0, e1))
                if sources is not None
                else device_sources(self.mesh, self.layer_idx, epc, cache, experts=range(e0, e1))
            )
            t2 = time.time()
            for n, (rows, fm) in sub.items():
                gather(self.mesh, srcs, fm, dst[n], pages=rows)
            for t in srcs:
                ttnn.deallocate(t)
            n_srcs += len(srcs)
            t_load, t_gather = t_load + t2 - t1, t_gather + time.time() - t2
        gids = torch.arange(E, dtype=torch.int32).reshape(N, 1, epc)  # (the block's chip n holds n epc ..)
        gids = gids[to_mesh_order(list(range(N)), self.mesh.shape, self.options.sp_axis)]  # (in device order)
        gidx = ttnn.from_torch(
            gids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        logger.info(
            f"MoE block layer {self.layer_idx}: {n_srcs} tensorbins to DRAM {t_load:.0f} s, gather {t_gather:.1f} s "
            f"in {len(batches)} batches ({cache})"
        )
        return gidx, dst["w_gu"], dst["w_d"]

    def _host_sources(self, experts):
        """The state dict's routed experts as flat_gather's plain sources (device_sources' layout): per local expert le
        and projection, chip n's expert n * epc + le transposed to [in, out], bfp4 in DRAM - a batch of experts at a
        time, so the host holds one projection of every chip at once (not the layer packed)."""
        R, C = tuple(self.mesh.shape)
        N = R * C
        epc = self.dims.num_experts // N
        rw = self.host_sd["routed_expert_weights"]
        E = self.dims.num_experts
        assert len(rw) == E, f"MoE block: {len(rw)} routed experts in the state dict, the model has {E}"
        same = N > 1 and all(rw[n * epc + i] is rw[i] for n in range(1, N) for i in range(epc))
        srcs = []
        for le in experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if same:
                    x, mapper = rw[le][proj].T.contiguous().reshape(1, 1, *rw[le][proj].T.shape), None
                else:
                    chips = to_mesh_order(list(range(N)), self.mesh.shape, self.options.sp_axis)  # (device order)
                    x = torch.stack([rw[n * epc + le][proj].T for n in chips]).unsqueeze(1).contiguous()
                    mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)
                srcs.append(
                    ttnn.from_torch(
                        x,
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mapper if mapper is not None else ttnn.ReplicateTensorToMesh(self.mesh),
                    )
                )
                del x
        return srcs

    def _host_sd_weights(self):
        """The non-routed weights from the state dict (the model's expert order: no permutation), as host_weights."""
        sd = self.host_sd
        g, s = sd["gate_weights"], sd["shared_expert_weights"]
        w = {
            "router_weight": g["weight"].float().contiguous(),
            "router_bias": g["e_score_correction_bias"].float().contiguous(),
            "gate": s["gate_proj"].float(),
            "up": s["up_proj"].float(),
            "down": s["down_proj"].float(),
        }
        lat = sd.get("latent_weights")
        if self.dims.latent_dim is not None:
            assert lat is not None, "MoE block: a latent model's state dict without latent_weights"
            w.update(
                latent_down=lat["down_proj"].float(), latent_up=lat["up_proj"].float(), latent_norm=lat["norm"].float()
            )
        return w

    def _weights(self, block):
        """This layer's MoEBlockWeights for the block: each part laid out for the block's plans (_packing) - made for
        the first block that needs that layout, reused by every block with the same one."""
        seen = self._by_block.get(id(block))
        if seen is not None and seen[0] is block:  # (held: its id is not reused while the entry lives)
            return seen[1]
        head_key, routed_key = self._packing(block)
        hit = self._prepared.get((head_key, routed_key))
        if hit is not None:
            self._by_block[id(block)] = (block, hit)
            return hit
        mb = ttnn.experimental.moe_block
        res = block.resources
        t0 = time.time()
        if self.host_sd is None and self.options.sp_axis != 0:
            # (the checkpoint path reads TtMoe's routed-expert cache and its expert placement, which with SP on mesh
            # axis 1 is #61481's - not on this branch)
            raise NotImplementedError("MoE block: SP on mesh axis 1 takes state-dict weights only on this branch")
        head, w_up = self._heads.get(head_key), self._w_up
        if head is None and self.host_sd is None:
            cached = self._load_head_cache(block)
            if cached is not None:
                head = cached[0]
                w_up = w_up if w_up is not None else cached[1]
        need_host = head is None or (self.dims.latent_dim is not None and w_up is None)
        fut, pool = None, None
        if need_host and self.host_sd is None:
            # (the checkpoint read + dequant on a thread while the routed experts' tensorbins load: ~9 s of a K3
            # layer's ~50 s hidden behind the NFS-bound reads)
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=1)
            fut = pool.submit(host_weights, self.mesh, self.dims, self.layer_idx, self.hf_config)
        routed = self._routed_w.get(routed_key)
        if routed is None:
            routed = (
                self._routed(block, sources=self._host_sources) if self.host_sd is not None else self._routed(block)
            )
            self._routed_w[routed_key] = routed
        t1 = time.time()
        if need_host:
            w = self._host_sd_weights() if self.host_sd is not None else fut.result()
            if pool is not None:
                pool.shutdown()
            host = lambda k: ttnn.from_torch(w[k], dtype=ttnn.float32)  # noqa: E731
            if head is None:
                head = mb.prepare_moe_head_weights(
                    host("gate"),
                    host("up"),
                    host("down"),
                    host("latent_down") if "latent_down" in w else None,
                    host("router_weight"),
                    host("router_bias"),
                    resources=res,
                    dtypes=block.dtypes,
                    accumulators=block.accumulators,
                )
            if w_up is None and "latent_up" in w:
                w_up = mb.prepare_moe_tail_weights(
                    host("latent_up"), host("latent_norm"), resources=res, row_blocks=block.row_blocks
                )
            del w
            if self.host_sd is None:
                self._save_head_cache(block, head, w_up)
        self._heads[head_key], self._w_up = head, w_up
        gidx, w_gu, w_d = routed
        weights = MoEBlockWeights(head=head, expert_ids=gidx, gate_up=w_gu, down=w_d, w_up=w_up)
        self._prepared[(head_key, routed_key)] = weights
        self._by_block[id(block)] = (block, weights)
        logger.info(
            f"MoE block layer {self.layer_idx}: weights for head {head_key}, {len(self._heads)} head / "
            f"{len(self._routed_w)} routed layouts held; routed {t1 - t0:.0f} s (the checkpoint read alongside), then "
            f"the head / tail weights {time.time() - t1:.0f} s; {time.time() - t0:.0f} s in all"
        )
        return weights

    def _head_cache_dir(self, block):
        """Where this layer's head / tail weights in the ops' layouts are kept (device-ready tensorbins), so the
        checkpoint is dequantized once: $MOE_BLOCK_WEIGHT_CACHE (unset: none). Keyed by the checkpoint, the mesh, the layer and everything the layouts depend on (the
        head's plan, the dtypes, the accumulators)."""
        import hashlib

        root = os.environ.get("MOE_BLOCK_WEIGHT_CACHE")
        if not root:  # (opt-in until the hit path has run in CI: TODO default next to the TtMoe cache)
            return None
        mb = ttnn.experimental.moe_block
        fit = mb.plan_moe_head(block.resources, block.dtypes, block.accumulators)
        lay = fit.planned[0].layout if fit.ok and len(fit.planned) else None
        ckpt = (
            os.environ.get("KIMI_K3_CKPT") or os.environ.get("PREFILL_HF_MODEL") or os.environ.get("GLM53_HF_MODEL", "")
        )
        key = repr(
            (
                "v1",
                Path(ckpt).name,
                tuple(self.mesh.shape),
                self.options.sp_axis,
                self.dims.num_experts,
                self.dims.emb_dim,
                self.dims.latent_dim,
                block.accumulators,
                str(block.dtypes.shared_weights),
                None
                if lay is None
                else (lay.engine_columns, lay.engine_rows, lay.row_tiles_per_core, lay.row_chunks, lay.k_step_tiles),
            )
        )
        return Path(root) / hashlib.sha1(key.encode()).hexdigest()[:16] / f"layer_{self.layer_idx}"

    _HEAD_FIELDS = ("w", "w_router", "router_bias", "router_index")

    def _load_head_cache(self, block):
        d = self._head_cache_dir(block)
        if d is None or not (d / "done").exists():
            return None
        mb = ttnn.experimental.moe_block
        load = lambda n: ttnn.load_tensor(str(d / f"{n}.tensorbin"), device=self.mesh)  # noqa: E731
        head = mb.MoEHeadWeights(*(load(n) for n in self._HEAD_FIELDS))
        w_up = load("w_up") if (d / "w_up.tensorbin").exists() else None
        return head, w_up

    def _save_head_cache(self, block, head, w_up):
        d = self._head_cache_dir(block)
        if d is None:
            return
        if d.exists():
            return
        tmp = d.with_name(f"{d.name}.tmp{os.getpid()}")
        try:
            # (written aside, then renamed into place: a concurrent run never reads a half-written layer)
            tmp.mkdir(parents=True, exist_ok=True)
            for n in self._HEAD_FIELDS:
                ttnn.dump_tensor(str(tmp / f"{n}.tensorbin"), getattr(head, n))
            if w_up is not None:
                ttnn.dump_tensor(str(tmp / "w_up.tensorbin"), w_up)
            (tmp / "done").touch()
            os.rename(tmp, d)
        except OSError as e:  # (read-only, full, or another run renamed first: the cache is optional)
            logger.warning(f"MoE block layer {self.layer_idx}: head weights not cached in {d}: {e}")

    def __call__(
        self,
        x,
        actual_isl: Optional[int] = None,
        padding_side: str = "right",
        actual_start: Optional[int] = None,
        metadata=None,
        forward_metadata=None,
        chunks=None,
        forward_chunks: Optional[int] = None,
    ):
        """x: ffn_norm's [1, 1, S/SP, H/TP]. The padding as TtMoe takes it (main's one-chunk contract): actual_isl None
        - every row real; else the chunk's real tokens, from actual_start (None: 0) - eager, host-written words - or,
        on the traced path, metadata = main's (slot_id, actual_start, actual_end) device tensors (read on device every
        call; actual_isl then only turns padding awareness on, as in TtMoe). A packed forward (common/prefill/
        prefill_chunk.py: forward_chunks chunks back to back on each SP chip): forward_metadata = the forward's metadata
        words as one uint32 device tensor (traceable), or chunks = its PrefillChunks (eager); forward_chunks defaults to
        len(chunks), else 1. Padding tokens are not routed, dispatched or computed by the routed experts; the shared
        expert and the tail skip the row tiles holding none (MoEBlock.__call__)."""
        R = sp_tp(self.mesh.shape, self.options.sp_axis)[0]  # (SP: the token shards)
        F = int(forward_chunks) if forward_chunks is not None else (len(chunks) if chunks is not None else 1)
        block = self._block(int(x.shape[-2]) * R, F)
        weights = self._weights(block)
        tmp = None
        if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            x = tmp = ttnn.to_layout(ttnn.typecast(x, ttnn.bfloat16), ttnn.TILE_LAYOUT)
        span = {}
        if forward_metadata is not None or chunks is not None:
            assert actual_isl is None and metadata is None, "MoE block: a packed forward's padding or one chunk's"
            span = dict(metadata=forward_metadata) if forward_metadata is not None else dict(chunks=chunks)
        elif actual_isl is not None:
            assert padding_side == "right", f"MoE block: padding_side {padding_side!r} (main's chunks pad on the right)"
            assert F == 1, f"MoE block: a forward of {F} chunks takes forward_metadata or chunks"
            if metadata is not None:  # (traced: main's device scalars, read every call)
                span = dict(metadata=metadata)
            else:
                start = int(actual_start or 0)
                span = dict(actual_start=start, actual_end=start + int(actual_isl))
        y = block(x, weights, **span)
        if not getattr(block, "_logged_l1_small", False):  # (the block's persistent words, all made)
            from models.demos.moe_block.tt.moe_block import l1_small_bytes

            mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1_SMALL)
            logger.info(
                f"MoE block: L1_SMALL {l1_small_bytes(self.mesh)} B a bank allocated after its first call (the mesh "
                f"has {int(mv.total_bytes_allocated_per_bank) + int(mv.total_bytes_free_per_bank)} B)"
            )
            block._logged_l1_small = True
        if getattr(block, "_logged_arena", None) != block.arena_address:  # (where each call's arena lands)
            logger.info(
                f"MoE block layer {self.layer_idx}: arena at {block.arena_address} (was "
                f"{getattr(block, '_logged_arena', None)}); x {x.memory_config().buffer_type} {tuple(x.shape)}"
            )
            block._logged_arena = block.arena_address
        if tmp is not None:
            ttnn.deallocate(tmp)
        return y
