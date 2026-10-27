# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The DSpark drafter on the mesh (speculative decode): the checkpoint's three ``mtp.*`` stages run as one
draft call on the decode tile.

The reference (``reference/deepseek_v41/model.py`` ``Transformer.forward_spec`` at ``start_pos = p > 0``): stage 0
projects the target's tap ``main_hidden(p)`` (the hc-mean block inputs of layers 37-39, concatenated) to ``main_x``
and every stage writes ``main_kv = QDQ(RoPE_p(kv_norm(wkv(main_x))))`` into its window ring at slot ``p % 128``; the
draft block ``[x[p], noise x 4]`` at positions ``p + 1 .. p + 5`` then runs the three stages (each the backbone block's
shape: mHC, a ratio-0 attention whose keys are the ring's ``min(128, p + 1)`` slots plus the block's own rows,
non-causal, with the sink; a 128-expert top-3 MoE with one shared expert), the last stage's ``norm``, the tied
head on the five rows, and the Markov loop (``forward_head``: row ``i``'s logits plus ``markov_head(out[i])``,
``out[i + 1] = argmax``).

Device mapping: the stages are built from the backbone's modules on the
32-row decode tile replicated on every mesh row, as ``tt/v41/decode.py`` runs one token: ``TtV41Attention`` places
the projections (bfp8, TP), ``TtV41HyperConnections`` the mHC, ``TtDistributedRmsNorm`` the norms, ``TtV41Moe`` (at
the drafter's expert count, EP-32: four experts a chip a stage, in the drafter's own cache directory) the MoE, built
sequentially as the decode MoEs are; ``TtV41DSpark`` (the prefill seeder) owns ``main_proj`` / ``main_norm`` and the
stages' bf16 ``wkv`` / ``kv_norm`` for the ring write. Each stage keeps one ``[1, 1, 128 + 32, 512]`` bf16 row-major
KV buffer: rows ``[0, 128)`` are the slot's ring (the prefill seeding's ``new_rings`` values joined in), rows
``[128, 160)`` the block's keys, written each call; the attention's index rows list the valid ring slots, then rows
``128 .. 128 + block``, then the sentinel. Rows 5 to 31 of the tile are noise-token rows at later positions whose
outputs are never read (every op is row-local or masked by the per-row index rows).

The tied embedding and head are the backbone's device modules (``TtV41Embedding``, ``TtV41Head`` with the last
stage's ``norm``); the Markov loop runs on the host from the five rows' logits (``draft`` returns them), which is
one readback of ``5 x vocab`` fp32 per call; a device loop is.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest import mock

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.decode import DECODE_ROUTING, DECODE_TAIL, MOE_ROWS, ROWS, SHARED_IN_FFN
from models.demos.deepseek_v3_d_p.tt.v41.decode import SHARED_IN_FFN_DTYPE
from models.demos.deepseek_v3_d_p.tt.v41.dspark import TtV41DSpark
from models.demos.deepseek_v3_d_p.tt.v41.dspark_weights import (
    DSPARK_BLOCK_SIZE,
    DSPARK_EXPERTS_PER_TOKEN,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_ROUTED_EXPERTS,
    dspark_cache_dir,
    dspark_layer_id,
    load_stage,
    load_stage_dense,
    seeding_weights,
)
from models.demos.deepseek_v3_d_p.tt.v41.head import TtV41Embedding, TtV41Head
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.mhc import TtV41HyperConnections
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from models.demos.deepseek_v3_d_p.tt.v41.moe_shared import install_shared_in_ffn
from models.demos.deepseek_v3_d_p.tt.v41.moe_tail import install_decode_tail
from models.demos.deepseek_v3_d_p.tt.v41.pool import RING, SENTINEL, scatter_rows
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin
from models.demos.deepseek_v3_d_p.tt.v41.routing import install_decode_routing
from models.demos.deepseek_v3_d_p.tt.v41.weights import begin_layer, complete_layer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

KV_ROWS = RING + ROWS  # a stage's KV buffer: the ring, then the block tile
INDEX_WIDTH = -(-(RING + DSPARK_BLOCK_SIZE) // TOPK_ALIGN) * TOPK_ALIGN  # 160: the index rows' width
# V41_DSPARK_MOE=prefill: the plain TtMoe routing on the 64-row repeated tile whatever the decode flags say (the
# decode routing kernels were written for the backbone's 384 experts); the default follows tt/v41/decode.py
DSPARK_MOE = os.environ.get("V41_DSPARK_MOE", "decode")
assert DSPARK_MOE in ("decode", "prefill"), DSPARK_MOE


class DSparkMoeConfig:
    """The config the drafter's ``TtV41Moe`` reads: the backbone's at the drafter's expert count."""

    @staticmethod
    def of(config):
        return type(
            "DSparkMoeConfig",
            (config,),
            {"NUM_ROUTED_EXPERTS": DSPARK_ROUTED_EXPERTS, "NUM_EXPERTS_PER_TOKEN": DSPARK_EXPERTS_PER_TOKEN},
        )


def _norm(mesh_device, config, weight):
    return TtDistributedRmsNorm(
        mesh_device=mesh_device,
        emb_dim=config.EMB_SIZE,
        torch_weight=weight,
        epsilon=config.RMS_NORM_EPS,
        cluster_axis=TP_AXIS,
        num_links=fabric_num_links(),
        topology=per_axis_topology()[TP_AXIS],
    )


class TtDSparkStage:
    """One ``mtp.<stage>`` block on the decode tile."""

    def __init__(self, mesh_device, config, stage: int, weights: dict, cache_dir: Path, routed_expert_weights_dtype):
        self.mesh_device, self.config, self.stage = mesh_device, config, stage
        self.layer = dspark_layer_id(stage, config)
        self.sp, self.tp = tuple(mesh_device.shape)
        self.attn = TtV41Attention(mesh_device, config, self.layer, weights["attn"])
        assert self.attn.ratio == 0 and not self.attn.is_kv_source and not self.attn.is_index_source
        self.attn_norm = _norm(mesh_device, config, weights["attn_norm"])
        self.ffn_norm = _norm(mesh_device, config, weights["ffn_norm"])
        self.residual = TtV41HyperConnections(mesh_device, config, weights["hc_attn"], weights["hc_ffn"])
        self.moe_rows = MOE_ROWS if DSPARK_MOE == "decode" else 64
        self.ffn = self._moe(weights, cache_dir, routed_expert_weights_dtype)
        self.kv = ttnn.from_torch(
            torch.zeros(1, 1, KV_ROWS, config.HEAD_DIM),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def _moe(self, weights: dict, cache_dir: Path, dtype) -> TtV41Moe:
        """The stage's ``TtV41Moe`` on ``moe_rows`` rows a chip, shared expert and dispatch in sequence (as
        ``TtV41Decoder._decode_moe``), the routed experts converted into (or loaded from) ``cache_dir`` under the
        ``layer_<40 + stage>`` prefix; the decode routing / tail / shared-expert installs follow the backbone's flags.
        """
        build_moe = TtPrefillBlock._build_moe

        def build_sequential(*args, **kwargs):
            return build_moe(*args, **{**kwargs, "overlap_shared_expert_with_dispatch": False})

        moe_weights = {
            k: weights[k]
            for k in ("gate_weights", "gate_bias_vl", "shared_expert_weights", "routed_expert_weights")
            if k in weights
        }
        with mock.patch.object(TtPrefillBlock, "_build_moe", staticmethod(build_sequential)):
            moe = TtV41Moe(
                self.mesh_device,
                DSparkMoeConfig.of(self.config),
                self.layer,
                moe_weights,
                self.moe_rows * self.sp,
                routed_expert_weights_dtype=dtype,
                weight_cache_path=cache_dir,
            )
        moe.moe.shared_expert.topology = ttnn.Topology.Linear
        if DSPARK_MOE == "decode":
            if DECODE_ROUTING:
                install_decode_routing(moe, None)
            if DECODE_TAIL:
                install_decode_tail(moe, ccl=self.attn.ccl)
            if SHARED_IN_FFN != "0":
                install_shared_in_ffn(moe, SHARED_IN_FFN, SHARED_IN_FFN_DTYPE)
        return moe

    def forward_ffn(self, h):
        """The MoE on the tile repeated to ``moe_rows`` rows -> ROWS rows."""
        if self.moe_rows == ROWS:
            return self.ffn(h)
        out = self.ffn(ttnn.concat([h] * (self.moe_rows // ROWS), dim=2))
        return ttnn.slice(out, [0, 0, 0, 0], [1, 1, ROWS, out.shape[3]])

    def forward_attention(self, x, cos, sin, index_rows):
        """``x`` [1, 1, ROWS, hidden/tp] bf16 (after attn_norm), the block rows' RoPE ``cos`` / ``sin`` and the index
        rows [1, 1, ROWS, INDEX_WIDTH] -> [1, 1, ROWS, hidden/tp] bf16. Writes the block's keys at rows
        [RING, RING + ROWS) of the stage's KV buffer."""
        attn = self.attn
        xq = fp8_qdq(x)
        qr = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wq_a")), weight=attn.q_norm, epsilon=attn.eps)
        q = attn._dense(fp8_qdq(qr), "wq_b")  # [1, 1, ROWS, H/tp * head_dim]
        kv = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wkv")), weight=attn.kv_norm, epsilon=attn.eps)
        kv = attn._rope(kv, cos, sin)
        rows = ttnn.to_layout(fp8_qdq(kv), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(rows, self.kv, [0, 0, RING, 0], [1, 1, KV_ROWS, rows.shape[3]], [1, 1, 1, 1])
        ttnn.deallocate(rows)
        q = attn.ccl.tp_all_gather(q, dim=3)  # all heads, in head order
        out = ttnn.transformer.sparse_sdpa(
            q_heads(q, cos, sin, attn.trans_mat, attn.heads, attn.rope_dim),
            self.kv,
            index_rows,
            attn.head_dim,
            kv_format=MlaKvCacheFormat.BF16_RM.sparse_sdpa_format,
            scale=attn.scale,
            k_chunk_size=next(c for c in (128, 64, 32) if index_rows.shape[-1] % c == 0),
            attention_sink=attn.sink,
        )
        out = o_heads(out, cos, ttnn.neg(sin), attn.trans_mat, attn.o_groups, attn.rope_dim)
        if self.tp > 1:
            out = ttnn.mesh_partition(out, dim=1, cluster_axis=TP_AXIS)
        return attn._o_proj(out)


class TtDSparkDrafter:
    """The three stages, the seeder's projections and the tied embedding / head as one draft call."""

    def __init__(
        self,
        mesh_device,
        config,
        ckpt,
        embedding: TtV41Embedding,
        head_weight: torch.Tensor,
        cache_root: Path,
        routed_expert_weights_dtype=ttnn.bfloat8_b,
        stages: int | None = None,
        head: TtV41Head | None = None,
    ):
        """``ckpt``: the ``V41Checkpoint`` the ``mtp.*`` tensors are read from; ``embedding``: the backbone's device
        embedding (tied); ``head_weight``: the tied ``head.weight`` (the head is built here with the last stage's
        ``norm``, unless ``head`` is given); ``cache_root``: ``TT_V41_WEIGHT_CACHE`` (the drafter's directory is
        ``dspark_cache_dir`` under it); ``stages``: how many stages to build (tests; default all)."""
        self.mesh_device, self.config = mesh_device, config
        self.sp, self.tp = tuple(mesh_device.shape)
        self.embedding = embedding
        self.cache_dir = dspark_cache_dir(
            cache_root,
            mesh_device.shape,
            routed_expert_weights_dtype.name,
            (TtRoutedExpert._convert_and_cache_expert_weights, TtV41Moe.__init__),
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.build_s = {}
        n = config.NUM_DSPARK_LAYERS if stages is None else stages
        self.stages: list[TtDSparkStage] = []
        self.markov = None
        self.head = head
        for k in range(n):
            t0 = time.time()
            layer = dspark_layer_id(k, config)
            cached = begin_layer(self.cache_dir, layer)
            weights = (load_stage_dense if cached else load_stage)(ckpt, k, config)
            self.build_s[f"load{k}"] = round(time.time() - t0, 1)
            t0 = time.time()
            self.stages.append(
                TtDSparkStage(mesh_device, config, k, weights, self.cache_dir, routed_expert_weights_dtype)
            )
            complete_layer(self.cache_dir, layer)
            if k == config.NUM_DSPARK_LAYERS - 1:
                # the Markov tables for the host loop: the embedding rows bf16, the head as fp32 once (the reference's
                # ``F.linear(embed.float(), head)`` with a bf16-valued head; a per-call conversion was 75 ms)
                self.markov = {
                    "embed": weights["markov"]["embed"].detach().to(torch.bfloat16),
                    "head": weights["markov"]["head"].detach().to(torch.bfloat16).float(),
                }
                if self.head is None:
                    self.head = TtV41Head(mesh_device, config, weights["norm"], head_weight)
            self.build_s[f"build{k}"] = round(time.time() - t0, 1)
        t0 = time.time()
        self.seeder = TtV41DSpark(mesh_device, config, seeding_weights(ckpt, config))
        self.build_s["seeder"] = round(time.time() - t0, 1)
        pre = torch.zeros(1, 1, ROWS, config.HC_MULT)
        pre[..., 0] = 1.0
        self.pre_mix = self._replicated(pre, ttnn.float32)

    def _replicated(self, host: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT, device=True):
        return ttnn.from_torch(
            host,
            device=self.mesh_device if device else None,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    # --- host inputs ---------------------------------------------------------------------------------------------
    def tap_rows(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """The host tap ``[n_taps * hidden]`` (the reference's ``main_hidden`` row) -> ``[1, 1, ROWS, n_taps * hidden]``
        in (chip, tap, j) column order, the row on every tile row: sharded over TP on dim 3 it gives chip ``t`` the
        local concatenation ``[tap_0 slice t | tap_1 slice t | ...]`` that ``TtV41DSpark.main_proj`` multiplies."""
        n_taps, hidden = len(self.config.DSPARK_TARGET_LAYER_IDS), self.config.EMB_SIZE
        t = main_hidden.detach().to(torch.bfloat16).reshape(n_taps, self.tp, hidden // self.tp).transpose(0, 1)
        return t.reshape(1, 1, 1, n_taps * hidden).expand(1, 1, ROWS, -1).contiguous()

    def host_inputs(self, token: int, main_hidden: torch.Tensor, position: int) -> dict:
        """The draft call's host tensors at ``position`` = the reference's ``start_pos`` ``p``: the tap is position
        ``p``'s (its ``main_kv`` goes to ring slot ``p % 128``), ``token`` is the target's output at ``p`` (``x[p + 1]``,
        the block's first row at position ``p + 1``), the other rows noise at ``p + 2 ..``."""
        p, cfg = position, self.config
        ids = torch.full((1, 1, ROWS), DSPARK_NOISE_TOKEN_ID, dtype=torch.int32)
        ids[0, 0, 0] = token
        inputs = {"token": self._replicated(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False)}
        inputs["tap"] = ttnn.from_torch(
            self.tap_rows(main_hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(None, 3)),
        )
        main = cos_sin(cfg, False, torch.full((ROWS,), p, dtype=torch.int64))
        inputs["rope_main"] = tuple(self._replicated(t, ttnn.float32, device=False) for t in main)
        block = cos_sin(cfg, False, torch.arange(p + 1, p + 1 + ROWS))
        inputs["rope_block"] = tuple(self._replicated(t, ttnn.bfloat16, device=False) for t in block)
        ring_idx = torch.full((1, 1, 1, ROWS), SENTINEL, dtype=torch.int64)
        ring_idx[..., 0] = p % RING
        inputs["ring_idx"] = self._replicated(
            ring_idx.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False
        )
        valid = min(RING, p + 1)  # the reference's get_dspark_topk_idxs: the written ring slots, then the block
        rows = [SENTINEL] * INDEX_WIDTH
        rows[:valid] = range(valid)
        rows[valid : valid + DSPARK_BLOCK_SIZE] = range(RING, RING + DSPARK_BLOCK_SIZE)
        index = torch.tensor(rows, dtype=torch.int64).view(1, 1, 1, -1).expand(1, 1, ROWS, -1).contiguous()
        inputs["index_rows"] = self._replicated(index.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False)
        return inputs

    def to_device(self, inputs: dict) -> dict:
        out = {}
        for k, v in inputs.items():
            out[k] = (
                tuple(ttnn.to_device(t, self.mesh_device) for t in v)
                if isinstance(v, tuple)
                else ttnn.to_device(v, self.mesh_device)
            )
        return out

    # --- the ring write (the decode half of the seeding) ---------------------------------------------------------
    def write_main_kv(self, inputs: dict):
        """stage 0's ``main_x`` from the tap rows, every stage's ``main_kv`` into its ring at ``ring_idx`` (row 0 of the
        tile; the other rows' index is the sentinel). Returns ``main_x`` [1, 1, ROWS, hidden] bf16 replicated."""
        s = self.seeder
        y = ttnn.linear(
            fp8_qdq(inputs["tap"]), s.main_proj, dtype=ttnn.float32, compute_kernel_config=s.compute_kernel_config
        )
        y = ttnn.typecast(s.ccl.tp_all_reduce(y), ttnn.bfloat16)
        main_x = ttnn.rms_norm(y, weight=s.main_norm, epsilon=s.eps)
        xq = fp8_qdq(main_x)
        cos, sin = inputs["rope_main"]
        for layer, stage in zip(s.layers, self.stages):
            kv = ttnn.linear(xq, layer["wkv"], compute_kernel_config=s.compute_kernel_config)
            kv = s._rope(ttnn.rms_norm(kv, weight=layer["kv_norm"], epsilon=s.eps), cos, sin)
            kv = ttnn.to_layout(fp8_qdq(kv), ttnn.ROW_MAJOR_LAYOUT)
            scatter_rows(kv, stage.kv, inputs["ring_idx"], rows=1)
            ttnn.deallocate(kv)
        return main_x

    # --- the draft block ---------------------------------------------------------------------------------------
    def forward_block(self, inputs: dict, on_stage=None):
        """The block through the stages -> the final collapsed stream [1, 1, ROWS, hidden/tp] bf16 (before the last
        stage's norm). ``on_stage(k, streams)`` after each stage (acceptance)."""
        cos, sin = inputs["rope_block"]
        h = self.embedding(inputs["token"])
        x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
        pre = self.pre_mix
        for k, stage in enumerate(self.stages):
            x, pre = stage.residual(
                x,
                pre,
                attention=lambda h, s=stage: s.forward_attention(s.attn_norm(h), cos, sin, inputs["index_rows"]),
                ffn=lambda h, s=stage: s.forward_ffn(s.ffn_norm(h)),
            )
            if on_stage is not None:
                on_stage(k, x)
        return ttnn.typecast(self.stages[-1].residual.final_collapse(x, pre), ttnn.bfloat16)

    def draft(self, inputs: dict, on_stage=None):
        """One draft call on device inputs: the ring write, the block, the head -> (logits tile [1, 1, ROWS,
        vocab/tp] fp32 per chip (row j the block's row j), main_x)."""
        main_x = self.write_main_kv(inputs)
        final = self.forward_block(inputs, on_stage)
        logits, _ = self.head(final, 0)
        return logits, main_x

    def logits_rows(self, logits, rows: int = DSPARK_BLOCK_SIZE) -> torch.Tensor:
        """The block's first ``rows`` rows of the logits tile on the host, ``[rows, vocab]`` fp32 (``scored_rows``:
        the device untilizes and slices the rows, the host reads ``rows`` instead of 32; consumes ``logits``)."""
        return self.head._tile_to_host(self.head.scored_rows(logits, 0, rows), 0)

    def markov_loop(self, token: int, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The reference's ``forward_head`` loop on the host: ``logits`` [block, vocab] fp32 (modified in place:
        row ``i`` plus ``markov_head(out[i])``), ``out[0] = token``, ``out[i + 1] = argmax(row i)``; greedy.
        Returns (output ids [block + 1], the biased logits)."""
        embed, head = self.markov["embed"], self.markov["head"]
        out = torch.empty(logits.shape[0] + 1, dtype=torch.int64)
        out[0] = token
        for i in range(logits.shape[0]):
            logits[i] += torch.mv(head, embed[out[i]].float())
            out[i + 1] = logits[i].argmax()
        return out, logits

    def reset_rings(self) -> None:
        for stage in self.stages:
            ttnn.experimental.slice_write(
                self._replicated(torch.zeros(1, 1, RING, self.config.HEAD_DIM), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                stage.kv,
                [0, 0, 0, 0],
                [1, 1, RING, self.config.HEAD_DIM],
                [1, 1, 1, 1],
            )

    def load_rings(self, rings: list[torch.Tensor]) -> None:
        """Rows [0, 128) of every stage's KV buffer from host rings ``[128, head_dim]`` (slot order)."""
        for stage, ring in zip(self.stages, rings):
            t = self._replicated(ring.to(torch.bfloat16).view(1, 1, RING, -1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.slice_write(t, stage.kv, [0, 0, 0, 0], [1, 1, RING, ring.shape[-1]], [1, 1, 1, 1])
            ttnn.deallocate(t)

    def ring_rows(self, stage: int) -> torch.Tensor:
        """The stage's ring on the host (chip 0), ``[128, head_dim]``."""
        return ttnn.to_torch(ttnn.get_device_tensors(self.stages[stage].kv)[0])[0, 0, :RING]
