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
from models.demos.moe_block.tt.flat_gather import flat_face_map, flat_weights_module, gather
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

LINKS = 2
_WEKA_CACHE = "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3-Cache/Kimi-K3-Cache-prefill"
_BLOCKS: dict = {}  # (id(mesh), chunk) -> (mesh, MoEBlockResources, MoEBlock)
_FACE_MAPS: dict = {}  # (H, I, E, layout) -> flat_face_map: the same every layer


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


def _cache_dir(mesh):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import _checkpoint_identity

    R, C = tuple(mesh.shape)
    ckpt = Path(os.environ.get("KIMI_K3_CKPT", ""))
    root = Path(os.environ.get("K3_BLOCK_WEKA_CACHE", _WEKA_CACHE))
    return root / f"kimi_k3_bh_{R * C}dev" / _checkpoint_identity(ckpt) / f"mesh{R}x{C}.tpaxis1"


def _glm_nonrouted(layer_idx: int, hf_config):
    """GLM-5.3 layer's gate and shared expert from $GLM53_HF_MODEL, fp8 dequantized."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_tensors
    from models.demos.deepseek_v3_d_p.utils.test_utils import dequantize_state_dict

    ckpt = Path(os.environ.get("GLM53_HF_MODEL", "/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4"))
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
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import GATE_KEYS, LATENT_KEYS, SHARED_EXPERT_KEYS
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import layer_prefix, load_tensors

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
    ckpt = Path(os.environ.get("KIMI_K3_CKPT", ""))
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


def device_sources(mesh, layer_idx: int, epc: int, cache):
    """TtMoe's cached routed experts of a layer straight to DRAM (bfp4 as stored, interleaved): flat_gather's sources,
    s = le * 3 + (0 gate, 1 up, 2 down) - each [1, 1, in, out] a chip (W^T: the flat op's orientation)."""
    srcs = []
    for le in range(epc):
        for proj in ("gate", "up", "down"):
            hits = sorted(Path(cache).glob(f"layer_{layer_idx}.routed_expert.local_{le}_{proj}_dtype_*.tensorbin"))
            assert hits, f"MoE block: no layer_{layer_idx}.routed_expert.local_{le}_{proj} in {cache}"
            assert "BFLOAT4_B" in hits[0].name, f"flat_gather copies bfp4 faces: {hits[0].name}"
            host = ttnn.load_tensor(str(hits[0]))
            srcs.append(ttnn.to_device(host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            del host
    return srcs


class TtK3MoEBlockFfn:
    """One MoE layer on the shared MoE block: __call__(ffn_norm's x [1, 1, S/SP, H/TP]) -> [1, 1, S/SP, H/TP]."""

    def __init__(self, mesh_device, model_cfg, layer_idx: int, hf_config=None, cache_path=None):
        self.mesh, self.layer_idx, self.hf_config = mesh_device, int(layer_idx), hf_config
        self.cache = Path(cache_path) if cache_path is not None else None
        self.dims, self.act = block_dims(model_cfg)
        self.weights = None  # (this layer's MoEBlockWeights, made at its first call)

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
                persist_l1_small=os.environ.get("PERSIST_L1") == "small",
            )
            dtypes = mb.MoEDtypes()
            t0 = time.time()
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
                    rows_hint=round(chunk * self.dims.top_k / self.dims.num_experts), x_helpers=1
                ),
            )
            logger.info(
                f"MoE block: built for chunk {chunk} at layer {self.layer_idx} in {time.time() - t0:.1f} s - "
                f"accumulators {block.accumulators}, capacity {block.expert_kw['max_tokens_per_expert']}"
            )
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
        t0 = time.time()
        maps = _FACE_MAPS.get(key)
        if maps is None:
            maps = _FACE_MAPS[key] = flat_face_map(Hr, I, epc, lay)
            logger.info(f"MoE block gather: face maps in {time.time() - t0:.0f} s")
        shapes = fw.flat_weight_shapes(Hr, I, epc, lay)
        dst = {
            n: fw.uninitialized_bank_sharded(*shapes[n], lay["banks"], ttnn.bfloat4_b, self.mesh)
            for n in ("w_gu", "w_d")
        }
        cache = self.cache if self.cache is not None else _cache_dir(self.mesh)
        t1 = time.time()
        srcs = device_sources(self.mesh, self.layer_idx, epc, cache)
        t2 = time.time()
        for n, fm in maps.items():
            pages = (int(dst[n].shape[-2]) // 32) * (int(dst[n].shape[-1]) // 32)
            assert pages == fm.shape[0], f"{n}: {pages} pages, the face map {fm.shape[0]}"
            gather(self.mesh, srcs, fm, dst[n])
        for t in srcs:
            ttnn.deallocate(t)
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
            f"MoE block layer {self.layer_idx}: {len(srcs)} tensorbins to DRAM {t2 - t1:.0f} s, gather "
            f"{time.time() - t2:.1f} s ({cache})"
        )
        return gidx, dst["w_gu"], dst["w_d"]

    def _weights(self, block):
        if self.weights is None:
            mb = ttnn.experimental.moe_block
            res = block.resources
            t0 = time.time()
            w = host_weights(self.mesh, self.dims, self.layer_idx, self.hf_config)
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
            t1 = time.time()
            gidx, w_gu, w_d = self._routed(block)
            self.weights = MoEBlockWeights(head=head, expert_ids=gidx, gate_up=w_gu, down=w_d, w_up=w_up)
            logger.info(
                f"MoE block layer {self.layer_idx}: checkpoint weights {t1 - t0:.0f} s, routed {time.time() - t1:.0f} s"
            )
        return self.weights

    def __call__(self, x):
        R = tuple(self.mesh.shape)[0]
        block = self._block(int(x.shape[-2]) * R)
        weights = self._weights(block)
        tmp = None
        if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            x = tmp = ttnn.to_layout(ttnn.typecast(x, ttnn.bfloat16), ttnn.TILE_LAYOUT)
        y = block(x, weights)
        if tmp is not None:
            ttnn.deallocate(tmp)
        return y
