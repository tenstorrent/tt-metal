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
model's own ops placed their L1); each layer holds its MoEBlockWeights. Env: MOE_COMBINE_SNAP ("2^-14" or a float:
the combine's snap), PERSIST_L1=small (the block's persistent words in L1_SMALL).
"""

from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.moe_block.tt.flat_gather import flat_face_map, flat_weights_module, gather, gather_batches
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

LINKS = 2
_WEKA_CACHE = "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3-Cache/Kimi-K3-Cache-prefill"
_BLOCKS: dict = {}  # (id(mesh), chunk) -> (mesh, MoEBlockResources, MoEBlock)
_REPLAY: dict = {}  # (MOE_BLOCK_REPLAY=1: the first MoE call's x, weights and output - replayed before the next)
_FACE_MAPS: dict = {}  # ((H, I, E, layout), experts a batch) -> gather_batches of flat_face_map: the same every layer


def _bytes_of(d):
    """A device shard's values as bytes (bf16 viewed as int16: numpy has no bfloat16)."""
    t = ttnn.to_torch(d).contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def _host_all(t):
    """Every chip's shard of t, flattened (a diagnostic)."""
    return torch.cat([ttnn.to_torch(d).float().flatten() for d in ttnn.get_device_tensors(t)])


def fabric_payload(H: int) -> int:
    """Fabric payload bytes the combine's L1 hand-off needs: one piece a row (H bf16) plus its trailer, 64-B aligned."""
    return max(7232, (2 * H + 32 + 63) // 64 * 64)


def block_dims(model_cfg):
    """The block's MoEModelDims (GLM-5.3 / Kimi-K3), checked against the model's config."""
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


def log_words_layout(res, l1s_before=None, l1s_after=None):
    """The block's persistent words in L1_SMALL: each set's cores (the union over the chips), bytes a core and address,
    the per-tensor footprint (each set reserves its bytes on every core) vs the packed slab's."""
    rows = res.words_layout()
    slab = res.words_slab()
    per_core = {}
    for w in rows:
        for c in w["cores"]:
            per_core[c] = per_core.get(c, 0) + w["bytes"]
        logger.info(
            f"MOE_WORDS {w['name']}: {len(w['cores'])} cores, {w['bytes']} B a core, at {w['address']}"
            f"{' (slab +' + str(w['address'] - slab['address']) + ')' if w['packed'] else ' (own tensor)'}"
        )
    own = sum(w["bytes"] for w in rows if not w["packed"])
    logger.info(
        f"MOE_WORDS total: {len(rows)} sets, sum {sum(w['bytes'] for w in rows)} B (a tensor a set: every core), "
        f"max a core {max(per_core.values(), default=0)} B; slab {slab}; own tensors {own} B; footprint "
        f"{(slab['capacity'] if slab else 0) + own} B; mesh L1_SMALL before the block {l1s_before} B, after {l1s_after} B"
    )


class TtK3MoEBlockFfn:
    """One MoE layer on the shared MoE block: __call__(ffn_norm's x [1, 1, S/SP, H/TP]) -> [1, 1, S/SP, H/TP]."""

    def _default_gu_read_pieces(self) -> str:
        """The gate/up reads' pieces (FlatLayoutOptions.gu_read_pieces) with the default Tensix reads: the fastest
        DRISC-free choice a model - Kimi-K3 2 (L1-L4 golden, traced, full block: +3.0% over the DRISC senders, 1 piece
        +6.0%), GLM-5.3 1 (L78 chunked prefill, traced: 302.4 ms a chunk, +1.7%; 2 / 4 pieces 305.1 / 304.9, DRISC
        297.3). -1 (the op's default) for the DRISC senders."""
        if os.environ.get("MOE_GU_SOURCE", "tensix") == "drisc":
            return "-1"
        return "2" if self.model_name == "KimiK3Config" else "1"

    def __init__(self, mesh_device, model_cfg, layer_idx: int, hf_config=None, cache_path=None, state_dict=None):
        self.mesh, self.layer_idx, self.hf_config = mesh_device, int(layer_idx), hf_config
        self.cache = Path(cache_path) if cache_path is not None else None
        # (a TtPrefillBlock state dict holding the MoE's host weights - gate_weights, shared_expert_weights,
        # routed_expert_weights, K3: latent_weights - as the synthetic-weight tests give TtMoe: the block's weights are
        # made from them, expert e on chip e // (E / N), instead of from the checkpoint and the TtMoe cache)
        self.host_sd = state_dict if state_dict and state_dict.get("routed_expert_weights") is not None else None
        self.dims, self.act = block_dims(model_cfg)
        self.model_name = model_cfg if isinstance(model_cfg, str) else getattr(model_cfg, "__name__", "")
        self.weights = None  # (this layer's MoEBlockWeights, made at its first call)
        self.calls = 0

    def _block(self, chunk):
        key = (id(self.mesh), chunk)
        hit = _BLOCKS.get(key)
        if hit is None or hit[0] is not self.mesh:
            mb = ttnn.experimental.moe_block
            sv = os.environ.get("MOE_COMBINE_SNAP", "")
            snap = None if not sv or sv == "0" else (2.0 ** int(sv[2:]) if sv.startswith("2^") else float(sv))
            cfg = mb.MoEBlockConfig(
                chunk_tokens=chunk,
                num_links=LINKS,
                combine_snap=snap,
                persist_l1_small=True,  # (the ops' words and semaphores in L1_SMALL: MoEBlock)
            )
            if os.environ.get("MOE_WORDS_SLAB"):  # (the persistent words' packed slab, B a core; 0: a tensor a set)
                cfg.words_slab_bytes = int(os.environ["MOE_WORDS_SLAB"])
            dtypes = mb.MoEDtypes()
            t0 = time.time()
            from models.demos.moe_block.tt.moe_block import l1_small_bytes

            l1s_before = l1_small_bytes(self.mesh)  # (the model's own L1_SMALL users so far)
            res = mb.MoEBlockResources(self.mesh, self.dims, cfg, dtypes)
            # (the reference's sizing - models/demos/moe_block RoutedExperts.cap_of / layout_of and plan_combine's
            # accumulators at 8x4, chunk 5120: GLM-5.3 11, Kimi-K3 15; MOE_ACCUMULATORS overrides)
            acc = int(os.environ.get("MOE_ACCUMULATORS", "0")) or (11 if self.dims.latent_dim is None else 15)
            block = MoEBlock(
                self.mesh,
                res,
                dtypes=dtypes,
                accumulators=acc,
                max_tokens_per_expert=max(chunk, 256),
                activation=flat_weights_module().ACTS[self.act],
                layout=mb.FlatLayoutOptions(
                    rows_hint=round(chunk * self.dims.top_k / self.dims.num_experts),
                    x_helpers=1,
                    # (which DRISC tiles of each DRAM bank send the gate/up weights: both / free_only /
                    # noc1_endpoint_only - FlatLayoutOptions.drisc_tiles; MOE_DRISC_TILES overrides "both")
                    drisc_tiles=getattr(mb.FlatDriscTiles, os.environ.get("MOE_DRISC_TILES", "both")),
                    # (who reads the gate/up weights - FlatLayoutOptions.gu_source drisc | tensix | tensix_split |
                    # tensix_noc1 | placed | auto_tensix | auto_hybrid; MOE_GU_PLACE: a JSON file of gu_place words for
                    # placed). Default: the gate/up cores read their own weights on NOC0 (K3: each block as 2 reads) - no
                    # DRISC sender is launched on the DRAM cores (K3 L1-4 golden, traced: +3.0% over the DRISC senders,
                    # the fastest DRISC-free choice; MOE_GU_SOURCE=drisc restores the senders)
                    gu_source=getattr(mb.FlatGuSource, os.environ.get("MOE_GU_SOURCE", "tensix")),
                    gu_read_depth=int(os.environ.get("MOE_GU_READ_DEPTH", "-1")),
                    gu_read_pieces=int(os.environ.get("MOE_GU_READ_PIECES", self._default_gu_read_pieces())),
                    gu_place=json.load(open(os.environ["MOE_GU_PLACE"])) if os.environ.get("MOE_GU_PLACE") else [],
                ),
            )
            _lo = block.expert_kw["layout"]
            logger.info(
                f"MoE block: flat layout drisc_tiles {_lo.drisc_tiles} gu_source {_lo.gu_source} "
                f"gu_read_depth {_lo.gu_read_depth} gu_read_pieces {_lo.gu_read_pieces} gu_place {len(_lo.gu_place)}"
            )
            logger.info(
                f"MoE block: built for chunk {chunk} at layer {self.layer_idx} in {time.time() - t0:.1f} s - "
                f"accumulators {block.accumulators}, capacity {block.expert_kw['max_tokens_per_expert']}, arena "
                f"{block.arena_spec.memory_config.shard_spec.shape[0] * 64 >> 10} KB a core"
            )
            fit = mb.plan_fused_head(res, dtypes, block.accumulators, block.l1_free)
            if fit.ok and len(fit.planned):
                p, lay = fit.planned[0], fit.planned[0].lay
                logger.info(
                    f"MoE block: head engine {lay.GX}x{lay.GY}, MT {lay.MT}, {lay.NCH} chunk(s), KB {lay.KB}, router on "
                    f"the engine {lay.on_engine}, x in DRAM {lay.xdram}, predicted {p.t:.0f} us {fit.note}"
                )
            block._l1s_before = l1s_before
            hit = _BLOCKS[key] = (self.mesh, res, block)
        return hit[2]

    def _routed(self, block):
        """This layer's flat expert tensors (gidx, w_gu, w_d), laid out on the device from TtMoe's bfp4 cache."""
        mb, fw = ttnn.experimental.moe_block, flat_weights_module()
        R, C = tuple(self.mesh.shape)
        N, E = R * C, self.dims.num_experts
        epc, Hr, I = E // N, self.dims.routed_width, self.dims.routed_expert_hidden_dim
        kw = block.expert_kw
        lay = mb.flat_routed_expert_plan(
            self.mesh, Hr, I, epc, E, kw["max_tokens_per_expert"], weights_bf8=False, pin=1, layout=kw["layout"]
        ).as_dict()
        lo = kw["layout"]
        key = (Hr, I, epc, lo.x_helpers, lo.pin_ratio, lo.rows_hint, lo.x_ready, kw["max_tokens_per_expert"])
        # (the layer's sources go to DRAM MOE_GATHER_EXPERTS local experts at a time and are freed after their gather:
        # all 3 epc of them next to the laid-out dest held one layer's routed weights twice - 65 MB a bank on K3, which
        # the last layer of an SC4 stage at 100 slots did not have)
        per = int(os.environ.get("MOE_GATHER_EXPERTS", "4"))
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
        cache = self.cache if self.cache is not None else _cache_dir(self.mesh)
        for n in dst:
            pages = (int(dst[n].shape[-2]) // 32) * (int(dst[n].shape[-1]) // 32)
            got = sum(int(sub[n][0].numel()) for _, _, sub in batches)
            assert pages == got, f"{n}: {pages} pages, the face map {got}"
        t_load = t_gather = 0.0
        n_srcs = 0
        for e0, e1, sub in batches:
            t1 = time.time()
            srcs = device_sources(self.mesh, self.layer_idx, epc, cache, experts=range(e0, e1))
            t2 = time.time()
            for n, (rows, fm) in sub.items():
                gather(self.mesh, srcs, fm, dst[n], pages=rows)
            for t in srcs:
                ttnn.deallocate(t)
            n_srcs += len(srcs)
            t_load, t_gather = t_load + t2 - t1, t_gather + time.time() - t2
        gids = torch.arange(E, dtype=torch.int32).reshape(N, 1, epc)
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

    def _routed_host(self, block):
        """This layer's flat expert tensors from the state dict's host routed experts (expert e on chip e // epc, in
        the chips' row-major order: the block's global expert e is the model's expert e)."""
        mb, fw = ttnn.experimental.moe_block, flat_weights_module()
        R, C = tuple(self.mesh.shape)
        N, E = R * C, self.dims.num_experts
        epc, Hr, I = E // N, self.dims.routed_width, self.dims.routed_expert_hidden_dim
        kw = block.expert_kw
        lay = mb.flat_routed_expert_plan(
            self.mesh, Hr, I, epc, E, kw["max_tokens_per_expert"], weights_bf8=False, pin=1, layout=kw["layout"]
        ).as_dict()
        rw = self.host_sd["routed_expert_weights"]
        assert len(rw) == E, f"MoE block: {len(rw)} routed experts in the state dict, the model has {E}"
        f = lambda t: t.float().T.contiguous()  # noqa: E731  ((out, in) Linear weights -> the flat op's [in, out])
        same = N > 1 and all(rw[n * epc + i] is rw[i] for n in range(1, N) for i in range(epc))
        per_dev = [
            [(f(rw[e]["gate_proj"]), f(rw[e]["up_proj"]), f(rw[e]["down_proj"])) for e in range(n * epc, (n + 1) * epc)]
            for n in range(1 if same else N)
        ]
        # (every chip the same expert objects - a test's pool repeated: pack once, replicate; K3's 896 distinct
        # experts packed a chip each would hold ~120 GB of fp32)
        packed = [fw.flat_weight_regions(W_l, lay) for W_l in per_dev]
        w_gu = fw.bank_sharded([p[0] for p in packed], lay["banks"], ttnn.bfloat4_b, self.mesh, replicate=same)
        w_d = fw.bank_sharded([p[1] for p in packed], lay["banks"], ttnn.bfloat4_b, self.mesh, replicate=same)
        gidx = ttnn.from_torch(
            torch.arange(E, dtype=torch.int32).reshape(N, 1, epc),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        return gidx, w_gu, w_d

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
        if self.weights is None:
            mb = ttnn.experimental.moe_block
            res = block.resources
            t0 = time.time()
            cached = None if self.host_sd is not None else self._load_head_cache(block)
            fut, pool = None, None
            if cached is None and self.host_sd is None:
                # (the checkpoint read + dequant on a thread while the routed experts' tensorbins load: ~9 s of a K3
                # layer's ~50 s hidden behind the NFS-bound reads)
                from concurrent.futures import ThreadPoolExecutor

                pool = ThreadPoolExecutor(max_workers=1)
                fut = pool.submit(host_weights, self.mesh, self.dims, self.layer_idx, self.hf_config)
            gidx, w_gu, w_d = self._routed_host(block) if self.host_sd is not None else self._routed(block)
            t1 = time.time()
            if cached is not None:
                head, w_up = cached
            else:
                w = self._host_sd_weights() if self.host_sd is not None else fut.result()
                if pool is not None:
                    pool.shutdown()
                host = lambda k: ttnn.from_torch(w[k], dtype=ttnn.float32)  # noqa: E731
                head = mb.prepare_fused_head_weights(
                    host("gate"),
                    host("up"),
                    host("down"),
                    host("latent_down") if "latent_down" in w else None,
                    host("router_weight"),
                    host("router_bias"),
                    resources=res,
                    dtypes=block.dtypes,
                    accumulators=block.accumulators,
                    l1_free=block.l1_free,
                )
                w_up = None
                if "latent_up" in w:
                    w_up = mb.prepare_fused_tail_weights(host("latent_up"), host("latent_norm"), resources=res)
                del w
                if self.host_sd is None:
                    self._save_head_cache(block, head, w_up)
            self.weights = MoEBlockWeights(head=head, expert_ids=gidx, gate_up=w_gu, down=w_d, w_up=w_up)
            logger.info(
                f"MoE block layer {self.layer_idx}: routed {t1 - t0:.0f} s (the checkpoint read alongside), then the "
                f"head / tail weights {time.time() - t1:.0f} s; {time.time() - t0:.0f} s in all"
            )
        return self.weights

    def _head_cache_dir(self, block):
        """Where this layer's head / tail weights in the ops' layouts are kept (device-ready tensorbins), so the
        checkpoint is dequantized once: $MOE_BLOCK_WEIGHT_CACHE (unset: none). Keyed by the checkpoint, the mesh, the layer and everything the layouts depend on (the
        head's plan, the dtypes, the accumulators)."""
        import hashlib

        root = os.environ.get("MOE_BLOCK_WEIGHT_CACHE")
        if not root:  # (opt-in until the hit path has run in CI: TODO default next to the TtMoe cache)
            return None
        mb = ttnn.experimental.moe_block
        fit = mb.plan_fused_head(block.resources, block.dtypes, block.accumulators, block.l1_free)
        lay = fit.planned[0].lay if fit.ok and len(fit.planned) else None
        ckpt = (
            os.environ.get("KIMI_K3_CKPT") or os.environ.get("PREFILL_HF_MODEL") or os.environ.get("GLM53_HF_MODEL", "")
        )
        key = repr(
            (
                "v1",
                Path(ckpt).name,
                tuple(self.mesh.shape),
                self.dims.num_experts,
                self.dims.emb_dim,
                self.dims.latent_dim,
                block.accumulators,
                str(block.dtypes.shared_weights),
                str(block.dtypes.router_weights),
                str(block.dtypes.projection_weights),
                None if lay is None else (lay.GX, lay.GY, lay.MT, lay.NCH, lay.KB, lay.on_engine, lay.xdram),
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
        head = mb.FusedHeadWeights(*(load(n) for n in self._HEAD_FIELDS))
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

    def _hashes(self, block):
        """sha1 a chip of every device buffer the block reads (MOE_BLOCK_REPLAY: what changed between two calls)."""
        import hashlib

        w, h = _REPLAY["w"], block.last_head
        named = {"x": _REPLAY["x"], "expert_ids": w.expert_ids}
        for n in ("w", "w_router", "router_bias", "router_index"):
            named["head." + n] = getattr(w.head, n)
        for n in ("se", "rows", "xbuf", "route", "counts", "regions", "token_index", "table", "plan", "latent"):
            t = getattr(h, n, None)
            if t is not None:
                named["out." + n] = t
        if w.w_up is not None:
            named["w_up"] = w.w_up
        out = {}
        for name, t in named.items():
            devs = ttnn.get_device_tensors(t)
            out[name] = [hashlib.sha1(_bytes_of(d)).hexdigest()[:10] for d in devs]
        for name, t in (("gate_up", w.gate_up), ("down", w.down)):  # (large: chips 0 and the last)
            devs = ttnn.get_device_tensors(t)
            out[name] = [hashlib.sha1(_bytes_of(d)).hexdigest()[:10] for d in (devs[0], devs[-1])]
        return out

    def _replay(self, block, when):
        """MOE_BLOCK_REPLAY=1: the first MoE call again (its x and weights) - is the block's state intact?"""
        if "y" not in _REPLAY or _REPLAY.get("mesh") is not self.mesh:
            return
        if when == "before its weights" and not getattr(self, "_replayed_twice", False):
            self._replayed_twice = True  # (then once more: is only the first call after the model's ops wrong?)
            self._replay(block, "before its weights (1st)")
            when = "before its weights (2nd)"
        if "hashes" in _REPLAY and when == "before its weights":
            now = self._hashes(block)
            diff = {
                n: sum(a != b for a, b in zip(now[n], _REPLAY["hashes"][n]))
                for n in now
                if now[n] != _REPLAY["hashes"][n]
            }
            logger.info(
                f"MOE_BLOCK_HASH layer {self.layer_idx}: {len(now)} buffers, changed since layer "
                f"{_REPLAY['layer']}'s replay (chips): {diff if diff else 'none'}"
            )
        comp = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 3))
        y_again = block(_REPLAY["x"], _REPLAY["w"])
        again = ttnn.to_torch(y_again, mesh_composer=comp).float()
        ttnn.deallocate(y_again)
        heads = {n: _host_all(getattr(block.last_head, n)) for n in ("rows", "se", "xbuf")}
        logger.info(
            f"MOE_BLOCK_REPLAY layer {self.layer_idx} {when}: head outputs differ - "
            + ", ".join(f"{n} {int((heads[n] != _REPLAY[n]).sum())}" for n in heads)
        )
        ref = _REPLAY["y"]
        fin = torch.isfinite(again) & torch.isfinite(ref)
        logger.info(
            f"MOE_BLOCK_REPLAY layer {self.layer_idx} {when}: layer {_REPLAY['layer']}'s call again - "
            f"{int((again[fin] != ref[fin]).sum())} finite values differ, PCC "
            f"{torch.corrcoef(torch.stack([again[fin], ref[fin]]))[0, 1].item():.5f}; non-finite {int((~torch.isfinite(again)).sum())}"
            f" (first call {int((~torch.isfinite(ref)).sum())})"
        )

    def __call__(self, x):
        R = tuple(self.mesh.shape)[0]
        block = self._block(int(x.shape[-2]) * R)
        replay = os.environ.get("MOE_BLOCK_REPLAY") == "1" and self.layer_idx <= 5
        if replay:
            self._replay(block, "before its weights")
        weights = self._weights(block)
        if replay:
            self._replay(block, "after its weights")
        tmp = None
        if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            x = tmp = ttnn.to_layout(ttnn.typecast(x, ttnn.bfloat16), ttnn.TILE_LAYOUT)
        y = block(x, weights)
        if replay and ("y" not in _REPLAY or _REPLAY.get("mesh") is not self.mesh):
            comp = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, 3))
            _REPLAY.update(
                mesh=self.mesh,
                layer=self.layer_idx,
                x=ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                w=weights,
                y=ttnn.to_torch(y, mesh_composer=comp).float(),
                **{n: _host_all(getattr(block.last_head, n)) for n in ("rows", "se", "xbuf")},
            )
            self._replay(block, "right after")
            _REPLAY["hashes"] = self._hashes(block)
        if not getattr(block, "_logged_l1_small", False):  # (the block's persistent words + semaphores, all made)
            from models.demos.moe_block.tt.moe_block import l1_small_bytes

            mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1_SMALL)
            ex = block.exchange_spec
            logger.info(
                f"MoE block: L1_SMALL {l1_small_bytes(self.mesh)} B a bank allocated after its first call (the mesh "
                f"has {int(mv.total_bytes_allocated_per_bank) + int(mv.total_bytes_free_per_bank)} B); "
                + (
                    f"two arenas: the exchange's {ex.memory_config.shard_spec.shape[0] * 64 >> 10} KB"
                    if ex
                    else "one arena"
                )
            )
            block._logged_l1_small = True
            log_words_layout(block.resources, getattr(block, "_l1s_before", None), l1_small_bytes(self.mesh))
        if getattr(block, "_logged_arena", None) != block.arena_address:  # (where each call's arena lands)
            logger.info(
                f"MoE block layer {self.layer_idx}: arena at {block.arena_address} (was "
                f"{getattr(block, '_logged_arena', None)}); x {x.memory_config().buffer_type} {tuple(x.shape)}"
            )
            block._logged_arena = block.arena_address
        self.calls += 1
        if self.calls <= int(os.environ.get("MOE_BLOCK_CHECK", "0")) and self.layer_idx <= int(
            os.environ.get("MOE_BLOCK_CHECK_LAYERS", "1000")
        ):
            self._check(block, x, y)
        if tmp is not None:
            ttnn.deallocate(tmp)
        return y

    def _check(self, block, x, y):
        """MOE_BLOCK_CHECK=N: the first N calls of each layer (up to MOE_BLOCK_CHECK_LAYERS) against the block's math on
        the host in fp32, on the device's own routing (the head's routing rows) and this layer's weights."""
        R, C = tuple(self.mesh.shape)
        N, E, K = R * C, self.dims.num_experts, self.dims.top_k
        epc = E // N
        comp = ttnn.ConcatMesh2dToTensor(self.mesh, mesh_shape=(R, C), dims=(2, 3))
        xs = ttnn.to_torch(x, mesh_composer=comp).float()
        T = xs.shape[-2]
        xs = xs.reshape(T, -1)
        ys = ttnn.to_torch(y, mesh_composer=comp).float().reshape(T, -1)
        S = T // N
        ids, wts = [], []
        for d in ttnn.get_device_tensors(block.last_head.rows):  # (row-major chips: chip k owns tokens k S ..)
            u = ttnn.to_torch(d).to(torch.int64).reshape(-1)[: S * 2 * K].reshape(S, 2 * K) & 0xFFFF
            ids.append(u[:, :K])
            wb = u[:, K:]
            wts.append(torch.where(wb >= 32768, wb - 65536, wb).to(torch.int16).view(torch.bfloat16).float())
        ids, wts = torch.cat(ids), torch.cat(wts)
        w = host_weights(self.mesh, self.dims, self.layer_idx, self.hf_config)
        cache = self.cache if self.cache is not None else _cache_dir(self.mesh)
        mats = {}
        for le in range(epc):
            for proj in ("gate", "up", "down"):
                hit = sorted(
                    Path(cache).glob(f"layer_{self.layer_idx}.routed_expert.local_{le}_{proj}_dtype_*.tensorbin")
                )
                host = ttnn.load_tensor(str(hit[0]))
                mats[le, proj] = [
                    ttnn.to_torch(d).float().reshape(d.shape[-2], d.shape[-1]) for d in ttnn.get_device_tensors(host)
                ]
        h = xs if self.dims.latent_dim is None else xs @ w["latent_down"].T
        acc = torch.zeros(T, self.dims.routed_width)
        F = torch.nn.functional
        act = (
            (lambda g, u: F.silu(g) * u)
            if self.act == "silu"
            else (lambda g, u: 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25))
        )
        for e in torch.unique(ids).tolist():
            tok, kk = (ids == e).nonzero(as_tuple=True)
            k, le = e // epc, e % epc
            wg, wu, wd = (mats[le, p][k] for p in ("gate", "up", "down"))  # (W^T: [in, out])
            a = act(h[tok] @ wg, h[tok] @ wu)
            acc.index_add_(0, tok, (a @ wd) * wts[tok, kk].unsqueeze(1))
        out = acc
        if self.dims.latent_dim is not None:
            out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.dims.norm_eps) * w["latent_norm"]
            out = out @ w["latent_up"].T
        out = out + act(xs @ w["gate"].T, xs @ w["up"].T) @ w["down"].T
        p = torch.corrcoef(torch.stack([ys.flatten(), out.flatten()]))[0, 1].item()
        # (the routing too: the host router's top-K on the same x)
        logits = xs @ w["router_weight"].T
        sc = torch.sigmoid(logits) + w["router_bias"]
        hk = sc.topk(K, dim=-1).indices.sort(-1).values
        same = float((hk == ids.sort(-1).values).all(-1).float().mean())
        logger.info(
            f"MOE_BLOCK_CHECK layer {self.layer_idx} call {self.calls}: PCC {p:.5f} vs the host model (device routing); "
            f"routing: {same:.4f} of tokens pick the host's top-{K}; |y| max {ys.abs().max():.3g}, host {out.abs().max():.3g}"
        )


def prebuild(mesh, model_cfg, chunk: int):
    """The block for (mesh, chunk), built and run once on zeros with placeholder weights, so every persistent L1 word
    of its ops exists before any program of the model is compiled: call it before the model runs anything on the device.
    (A program is checked against the L1 buffers only when it is compiled; on its cache hits its circular buffers would
    overwrite words the ops placed later - the MoE layers' outputs turned to garbage from the second one on.)"""
    key = (id(mesh), chunk)
    hit = _BLOCKS.get(key)
    if hit is not None and hit[0] is mesh:
        return
    ffn = TtK3MoEBlockFfn(mesh, model_cfg, layer_idx=-1)
    block = ffn._block(chunk)
    mb, fw = ttnn.experimental.moe_block, flat_weights_module()
    d = ffn.dims
    R, C = tuple(mesh.shape)
    N, E, H, Is = R * C, d.num_experts, d.emb_dim, d.shared_expert_hidden_dim
    epc, Hr, I = E // N, d.routed_width, d.routed_expert_hidden_dim
    host = lambda *sh: ttnn.from_torch(torch.zeros(*sh), dtype=ttnn.float32)
    head = mb.prepare_fused_head_weights(
        host(Is, H), host(Is, H), host(H, Is), host(Hr, H) if d.latent_dim else None, host(E, H), host(E),
        resources=block.resources, dtypes=block.dtypes, accumulators=block.accumulators, l1_free=block.l1_free,
    )  # fmt: skip
    w_up = mb.prepare_fused_tail_weights(host(H, Hr), host(Hr), resources=block.resources) if d.latent_dim else None
    kw = block.expert_kw
    lay = mb.flat_routed_expert_plan(
        mesh, Hr, I, epc, E, kw["max_tokens_per_expert"], weights_bf8=False, pin=1, layout=kw["layout"]
    ).as_dict()
    shapes = fw.flat_weight_shapes(Hr, I, epc, lay)
    w_gu, w_d = (fw.uninitialized_bank_sharded(*shapes[n], lay["banks"], ttnn.bfloat4_b, mesh) for n in ("w_gu", "w_d"))
    gidx = ttnn.from_torch(
        torch.arange(E, dtype=torch.int32).reshape(N, 1, epc),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    x = ttnn.from_torch(
        torch.zeros(1, 1, chunk, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
    )
    ttnn.deallocate(block(x, MoEBlockWeights(head=head, expert_ids=gidx, gate_up=w_gu, down=w_d, w_up=w_up)))
    ttnn.synchronize_device(mesh)
    for t in (x, gidx, w_gu, w_d):
        ttnn.deallocate(t)
    logger.info(f"MoE block: prebuilt for chunk {chunk} (its persistent L1 in place before the model's programs)")
