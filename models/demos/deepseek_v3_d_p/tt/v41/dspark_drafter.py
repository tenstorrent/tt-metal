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
stage's ``norm``); the Markov loop runs in the graph (``V41_DSPARK_MARKOV=device``) or on the host from the five
rows' logits (``draft`` returns them; one readback of ``5 x vocab`` fp32 per call).

With ``ring_slots`` above 128 the rings are position-based as the verify decoder's window rings (a position's slot
is ``position % ring_slots``; ``tt/v41/spec_verify.py``): the verify step then writes every verify row's tap into
the rings and runs the drafter inside its own capture on every accept outcome's block (``fused_draft``: block
``a`` is ``[argmax(row a), noise x 4]`` at positions ``p + a + 1 ..`` with its index rows bounded to ``p + a``;
``(K + 1) x 5`` rows of one tile), so a round has one replay and one readback.
"""

from __future__ import annotations

import copy
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
from models.demos.deepseek_v3_d_p.tt.v41 import dspark_blocks
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.decode import (
    DECODE_ROUTING,
    DECODE_TAIL,
    MOE_ROWS,
    ROWS,
    SHARED_IN_FFN,
    SHARED_IN_FFN_DTYPE,
)
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
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

KV_ROWS = RING + ROWS  # a stage's KV buffer at the default ring: the ring, then the block tile
INDEX_WIDTH = -(-(RING + DSPARK_BLOCK_SIZE) // TOPK_ALIGN) * TOPK_ALIGN  # 160: the index rows' width
# V41_DSPARK_MOE=prefill: the plain TtMoe routing on the 64-row repeated tile whatever the decode flags say (the
# decode routing kernels were written for the backbone's 384 experts); the default follows tt/v41/decode.py
DSPARK_MOE = os.environ.get("V41_DSPARK_MOE", "decode")
assert DSPARK_MOE in ("decode", "prefill"), DSPARK_MOE
# V41_DSPARK_MARKOV=host: the Markov loop on the host from the five logits rows (the reference's fp32 arithmetic);
# device (default): in the graph, each row's bias from the Markov tables on device and its argmax through the
# sampler's top-32-per-chip path (bf16 candidates, as the target's own argmax)
DSPARK_MARKOV = os.environ.get("V41_DSPARK_MARKOV", "device")
assert DSPARK_MARKOV in ("device", "host"), DSPARK_MARKOV
# V41_DSPARK_HOST_INPUTS=one (default): the standalone call's per-rank inputs (token, ring index, tap selector,
# index rows, the two RoPE pairs) travel as one uint32 row-major page per rank, uploaded with one copy and unpacked
# inside the call's graph (after the step's ``StepPacker`` "one" mode: every mesh ``from_torch``
# and every copy costs 0.2 to 0.4 ms whatever its size, and the call had nine and ten); split: one host tensor and
# one copy per input, as before. A host tap (the single-block tests) keeps the split path.
DSPARK_HOST_INPUTS = os.environ.get("V41_DSPARK_HOST_INPUTS", "one")
assert DSPARK_HOST_INPUTS in ("one", "split"), DSPARK_HOST_INPUTS
ONE = "one"  # the packed page's key in ``host_inputs_blocks``'s dict and the buffers


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

    def __init__(
        self,
        mesh_device,
        config,
        stage: int,
        weights: dict,
        cache_dir: Path,
        routed_expert_weights_dtype,
        ring_slots: int = RING,
        users: int = 1,
    ):
        """``users``: rings per chip (one a slot of the rank: user ``u``'s at rows
        ``u ring_slots ..``); the block tile follows the rings (``block_base``)."""
        self.mesh_device, self.config, self.stage = mesh_device, config, stage
        self.ring_slots, self.users = ring_slots, users
        self.block_base = users * ring_slots
        self.kv_rows = self.block_base + ROWS
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
            torch.zeros(1, 1, self.kv_rows, config.HEAD_DIM),
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
        [ring_slots, ring_slots + ROWS) of the stage's KV buffer."""
        attn = self.attn
        xq = fp8_qdq(x)
        qr = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wq_a")), weight=attn.q_norm, epsilon=attn.eps)
        q = attn._dense(fp8_qdq(qr), "wq_b")  # [1, 1, ROWS, H/tp * head_dim]
        kv = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wkv")), weight=attn.kv_norm, epsilon=attn.eps)
        kv = attn._rope(kv, cos, sin)
        rows = ttnn.to_layout(fp8_qdq(kv), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(
            rows, self.kv, [0, 0, self.block_base, 0], [1, 1, self.kv_rows, rows.shape[3]], [1, 1, 1, 1]
        )
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
        shared_head: TtV41Head | None = None,
        trace: bool = False,
        warmup: int = 2,
        ring_slots: int = RING,
        users: int = 1,
    ):
        """``ckpt``: the ``V41Checkpoint`` the ``mtp.*`` tensors are read from; ``embedding``: the backbone's device
        embedding (tied); ``head_weight``: the tied ``head.weight`` (the head is built here with the last stage's
        ``norm``), or ``shared_head``: the backbone's ``TtV41Head``, whose projection is reused under the last
        stage's norm (no second copy of the 330 MB a chip); ``cache_root``: ``TT_V41_WEIGHT_CACHE`` (the drafter's
        directory is ``dspark_cache_dir`` under it); ``stages``: how many stages to build (tests; default all).
        ``trace``: ``run`` captures the draft call after ``warmup`` eager calls and replays it. ``ring_slots``: the
        stages' ring slots (128: the reference's ``p % 128``; more: position-based, the module docstring).
        ``users``: rings a chip, one per slot of the rank (the standalone call then proposes
        for up to ``users`` slots at once, ``run_blocks``; ``tt/v41/dspark_blocks.py`` has the row layout)."""
        self.mesh_device, self.config = mesh_device, config
        self.sp, self.tp = tuple(mesh_device.shape)
        assert ring_slots >= RING and ring_slots % 32 == 0, ring_slots
        assert 1 <= users <= ROWS // DSPARK_BLOCK_SIZE, users
        self.ring_slots, self.users = ring_slots, users
        self.block_base = users * ring_slots
        self._consts: dict = {}  # the fused call's constants by key
        self.embedding = embedding
        self.trace, self.warmup = trace, warmup
        self.buffers: dict = {}
        self._zero_tap = None  # the host zero tap of the standalone call (built once)
        self.packed = DSPARK_HOST_INPUTS == "one" and DSPARK_MARKOV == "device"
        # the standalone call's drafts gathered over SP inside the graph (``gather_sp``: the pool's, set by
        # ``TtV41VerifyDecoder.attach_drafter``) so the eight ranks' rows are one readback of chip 0
        # (eight per-rank reads were 2.5 to 4.5 ms a round); None: the per-rank reads
        self.gather_sp = None
        self.gathered = False  # ``out`` is the gathered tile (``read_ranks`` reads chip 0's)
        # V41_DECODE_REPLAY_SYNC=0 (the decoders' flag): no device sync between the staging
        # and the replay; the copies and the replay are on one command queue, in order (the sync waited 2.8 ms
        # for the one-page copy to land)
        self.replay_sync = os.environ.get("V41_DECODE_REPLAY_SYNC", "1") == "1"
        self._staged_use_tap32 = None  # the ``use_tap32`` value the buffer holds (copied when it changes)
        self._tap_is_zero = False  # the ``tap`` buffer holds the zero tap (no copy needed for a zero host tap)
        self.captures: dict[int, tuple] = {}  # the standalone call's capture per block count
        self.eager_counts: dict[int, int] = {}
        self.blocks = 1  # the staged standalone call's block count
        self.out = None  # the last call's output tensor (``read_ranks``)
        self._eager_out = None
        self.last: dict = {}
        # V41_DECODE_INPUT_TIMING=1 (as the decoders): ``last["items"]`` holds the standalone call's host time per
        # item (ms): the inputs (``in.*``), the staging (``stage.*``), the call (``run.*``) and the readback
        self.input_timing = os.environ.get("V41_DECODE_INPUT_TIMING", "0") == "1"
        self.items: dict = {}
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
        self.head = None
        for k in range(n):
            t0 = time.time()
            layer = dspark_layer_id(k, config)
            cached = begin_layer(self.cache_dir, layer)
            weights = (load_stage_dense if cached else load_stage)(ckpt, k, config)
            self.build_s[f"load{k}"] = round(time.time() - t0, 1)
            t0 = time.time()
            self.stages.append(
                TtDSparkStage(
                    mesh_device, config, k, weights, self.cache_dir, routed_expert_weights_dtype, ring_slots, users
                )
            )
            complete_layer(self.cache_dir, layer)
            if k == config.NUM_DSPARK_LAYERS - 1:
                # the Markov tables for the host loop: the embedding rows bf16, the head as fp32 once (the reference's
                # ``F.linear(embed.float(), head)`` with a bf16-valued head; a per-call conversion was 75 ms)
                self.markov = {
                    "embed": weights["markov"]["embed"].detach().to(torch.bfloat16),
                    "head": weights["markov"]["head"].detach().to(torch.bfloat16).float(),
                }
                if shared_head is not None:
                    self.head = copy.copy(shared_head)
                    self.head.norm = _norm(mesh_device, config, weights["norm"])
                else:
                    self.head = TtV41Head(mesh_device, config, weights["norm"], head_weight)
                self._markov_device_tables(weights["markov"])
            self.build_s[f"build{k}"] = round(time.time() - t0, 1)
        t0 = time.time()
        self.seeder = TtV41DSpark(mesh_device, config, seeding_weights(ckpt, config))
        self.build_s["seeder"] = round(time.time() - t0, 1)
        pre = torch.zeros(1, 1, ROWS, config.HC_MULT)
        pre[..., 0] = 1.0
        self.pre_mix = self._replicated(pre, ttnn.float32)
        if self.markov is not None:  # the standalone call's Markov masks for every block count, before any capture
            for blocks in range(1, users + 1):
                for i in range(DSPARK_BLOCK_SIZE):
                    self._block_mask(i, blocks)

    def _markov_device_tables(self, markov: dict) -> None:
        """The Markov tables on device (``V41_DSPARK_MARKOV=device``): ``embed`` ``[vocab, 256]`` bf16 row-major
        replicated (one ``ttnn.embedding`` of a token), ``head`` transposed ``[1, 1, 256, vocab / tp]`` bf16 tiled,
        column-parallel over TP as the tied head; the row masks, the vocabulary offsets and the sampler's constants
        (top-1: the argmax) of the in-graph loop (``_markov_device``)."""
        mesh, shape = self.mesh_device, tuple(self.mesh_device.shape)
        vocab = self.config.VOCAB_SIZE
        self.markov_embed = ttnn.from_torch(
            markov["embed"].detach().to(torch.bfloat16),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.markov_head_t = ttnn.from_torch(
            markov["head"].detach().to(torch.bfloat16).T.contiguous()[None, None],
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, shape, dims=(None, 3)),
        )
        per_chip = vocab // self.tp
        offsets = torch.arange(self.tp, dtype=torch.int32).view(self.tp, 1, 1, 1).expand(self.tp, 1, ROWS, 32)
        self.vocab_offsets = ttnn.from_torch(
            (offsets * per_chip).contiguous(),
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, shape, dims=(None, 0)),
        )
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.sample_k = self._replicated(torch.ones(ROWS, dtype=torch.int32), ttnn.uint32, rm)
        self.sample_p = self._replicated(torch.zeros(ROWS), ttnn.bfloat16, rm)
        self.sample_temp = self._replicated(torch.ones(ROWS), ttnn.bfloat16, rm)
        self.zero_id = self._replicated(torch.zeros(1, 1, 1, 1, dtype=torch.int32), ttnn.uint32, rm)
        self.seeded = False
        self.markov_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    def _seed_sampler(self) -> None:
        if self.seeded:
            return
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        ttnn.manual_seed(
            seeds=ttnn.from_torch(
                torch.randint(1, 2**31 - 1, (ROWS,), dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                mesh_mapper=mapper,
            ),
            user_ids=ttnn.from_torch(
                torch.arange(ROWS, dtype=torch.int32), device=self.mesh_device, dtype=ttnn.uint32, mesh_mapper=mapper
            ),
        )
        self.seeded = True

    def _argmax_rows(self, logits):
        """Every row's argmax over the full vocabulary from ``logits`` [1, 1, ROWS, vocab / tp] fp32 per chip (not
        consumed): the sampler's path (top-32 bf16 per chip, gathered over TP, top-1) -> [1, 1, 1, ROWS] uint32."""
        ccl = self.stages[0].attn.ccl
        values, indices = ttnn.topk(ttnn.typecast(logits, ttnn.bfloat16), 32, dim=-1, largest=True, sorted=True)
        indices = ttnn.add(ttnn.typecast(indices, ttnn.int32), self.vocab_offsets)
        values = ccl.tp_all_gather(values, dim=3)
        indices = ttnn.to_layout(ccl.tp_all_gather(indices, dim=3), ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.sampling(values, indices, self.sample_k, self.sample_p, self.sample_temp)

    def _block_mask(self, i: int, blocks: int):
        """``[1, 1, ROWS, 1]`` fp32: 1 at rows ``5 a + i`` for ``a < blocks`` (the rows whose Markov bias is row
        ``i``'s of their block)."""
        key = ("mask", i, blocks)
        if key not in self._consts:
            m = torch.zeros(1, 1, ROWS, 1)
            for a in range(blocks):
                m[0, 0, DSPARK_BLOCK_SIZE * a + i, 0] = 1.0
            self._consts[key] = self._replicated(m, ttnn.float32)
        return self._consts[key]

    def _shift_ids(self, ids):
        """``[1, 1, 1, ROWS]`` uint32 row-major -> ``[1, 1, ROWS]``: row ``r + 1`` = ``ids[r]`` (row 0 = 0)."""
        body = ttnn.slice(ids, [0, 0, 0, 0], [1, 1, 1, ROWS - 1])
        return ttnn.reshape(ttnn.concat([self.zero_id, body], dim=3), (1, 1, ROWS))

    def _markov_device(self, logits, tokens, blocks: int = 1):
        """The reference's ``forward_head`` loop in the graph on ``blocks`` candidate blocks of the tile (block ``a``
        at rows ``5 a .. 5 a + 4``): iteration ``i`` adds ``markov_head(out[i])`` to row ``5 a + i`` of every block
        and takes its argmax (``out[i + 1]``); ``tokens`` ``[1, 1, ROWS]`` uint32 (row ``5 a`` the block's first
        token ``out[0]``) -> the drafts ``[1, 1, 1, ROWS]`` uint32 row-major, row ``5 a + i`` block ``a``'s draft
        ``i + 1`` (the later iterations touch other rows only, so every row's argmax is final in the last)."""
        self._seed_sampler()
        tok = tokens
        ids = None
        for i in range(DSPARK_BLOCK_SIZE):
            e = ttnn.reshape(ttnn.embedding(tok, self.markov_embed, layout=ttnn.TILE_LAYOUT), (1, 1, ROWS, -1))
            bias = ttnn.matmul(e, self.markov_head_t, dtype=ttnn.float32, compute_kernel_config=self.markov_compute)
            logits = ttnn.add(logits, ttnn.multiply(bias, self._block_mask(i, blocks)))
            ids = self._argmax_rows(logits)  # rows 5 a + i are final; the later rows lack their bias yet
            if i + 1 < DSPARK_BLOCK_SIZE:
                tok = self._shift_ids(ids)  # row 5 a + i + 1 = out[i + 1] of block a
        return ids

    def _tick(self, prefix: str):
        """A timer as ``TtV41Decoder._tick``: ``tick(name)`` adds the time since the last tick to
        ``items[prefix.name]`` (ms) under ``V41_DECODE_INPUT_TIMING``; a no-op otherwise."""
        if not self.input_timing:
            return lambda name: None
        last = [time.perf_counter()]

        def tick(name):
            now = time.perf_counter()
            self.items[f"{prefix}.{name}"] = round(self.items.get(f"{prefix}.{name}", 0.0) + 1e3 * (now - last[0]), 3)
            last[0] = now

        return tick

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

    def host_inputs(
        self, token: int, main_hidden: torch.Tensor | None, position: int, write: bool | None = None, tap_row: int = 0
    ) -> dict:
        """The draft call's host tensors at ``position`` = the reference's ``start_pos`` ``p``: the tap is position
        ``p``'s (its ``main_kv`` goes to ring slot ``p % 128``), ``token`` is the target's output at ``p`` (``x[p + 1]``,
        the block's first row at position ``p + 1``), the other rows noise at ``p + 2 ..``."""
        # no tap: the first call after a prefill (whose seeding wrote the slot; no ring write), or (``write``) a
        # device tap copied over the host one by ``stage``
        if write is None:
            write = main_hidden is not None
        entry = {"user": 0, "token": token, "position": position, "taps": [(tap_row, position)] if write else []}
        return self.host_inputs_blocks([[entry]] * self.sp, main_hidden)

    def host_inputs_blocks(self, entries_by_rank: list[list[dict]], main_hidden: torch.Tensor | None = None) -> dict:
        """The draft call's host tensors for up to ``users`` blocks a rank (``tt/v41/dspark_blocks.py``: an entry
        is a slot's ``user``, ``token``, ``position`` ``p`` and the ``taps`` to write, ``(tap row, position)``),
        one list per rank (``sp`` of them; a rank's slice sharded to its chips): row ``5 a`` of the tile is block
        ``a``'s first token at ``p + 1``, rows ``5 a + i`` carry its slot's tap writes. ``main_hidden``: a host tap
        on every row (the single-block tests; the served and harness calls select the verify step's tap tile
        through ``sel``, ``stage(tap32=)``). Sets ``blocks`` (the count every rank must share)."""
        cfg, sp = self.config, self.sp
        assert len(entries_by_rank) == sp, (len(entries_by_rank), sp)
        blocks = {len(entries) for entries in entries_by_rank}
        assert len(blocks) == 1, f"one block count on every rank: {blocks}"
        (self.blocks,) = blocks
        assert 1 <= self.blocks <= self.users, (self.blocks, self.users)
        self.items = {}
        tick = self._tick("in")
        t = dspark_blocks.mesh_inputs(entries_by_rank, self.ring_slots, self.users)
        tick("blocks")
        rm = ttnn.ROW_MAJOR_LAYOUT
        packed = self.packed and main_hidden is None
        inputs = {} if packed else {"token": self._per_rank(t["ids"].view(sp, 1, ROWS), ttnn.uint32, rm)}
        tick("token")
        if main_hidden is None:
            # no host tap (the served and harness calls select the step's tap tile on device): the zero tap, one
            # host tensor for the drafter's life; ``stage`` copies it once and skips it while the buffer holds it
            if self._zero_tap is None:
                self._zero_tap = self._host_tap(torch.zeros(len(cfg.DSPARK_TARGET_LAYER_IDS) * cfg.EMB_SIZE))
            inputs["tap"] = self._zero_tap
        else:
            inputs["tap"] = self._host_tap(main_hidden)
        tick("tap")
        # the row selector of a device tap tile (``stage(..., tap32=)``: the target step's taps of every tile row):
        # ``sel @ tap32`` puts the selected tap row on each writing row (one nonzero term a row: exact); without
        # a device tile the host tap is used (use_tap32 0)
        inputs["use_tap32"] = self._use_tap32(False)
        main = cos_sin(cfg, False, t["main_pos"].flatten())
        block = cos_sin(cfg, False, t["block_pos"].flatten())
        tick("rope")
        if packed:
            inputs[ONE] = self._pack_one(t, main, block)
            tick("pack")
            return inputs
        inputs["sel"] = self._per_rank(t["sel"].view(sp, 1, ROWS, ROWS), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        tick("sel")
        inputs["rope_main"] = tuple(
            self._per_rank(r.view(sp, 1, ROWS, -1), ttnn.float32, ttnn.TILE_LAYOUT) for r in main
        )
        inputs["rope_block"] = tuple(
            self._per_rank(r.view(sp, 1, ROWS, -1), ttnn.bfloat16, ttnn.TILE_LAYOUT) for r in block
        )
        tick("rope_tensors")
        inputs["ring_idx"] = self._per_rank(t["ring_idx"].view(sp, 1, 1, ROWS), ttnn.uint32, rm)
        inputs["index_rows"] = self._per_rank(t["index"].view(sp, 1, ROWS, INDEX_WIDTH), ttnn.uint32, rm)
        tick("index")
        return inputs

    # the packed page's columns (uint32 each): token [ROWS], ring_idx [ROWS], sel [ROWS * ROWS] (bf16 bits << 16),
    # index [ROWS * INDEX_WIDTH], rope_main cos, sin [ROWS * rope_dim] each (fp32 bits), rope_block cos, sin
    # [ROWS * rope_dim] each (bf16 bits << 16)
    def _one_layout(self) -> dict[str, tuple[int, int]]:
        if not hasattr(self, "_one_cols"):
            rope = self.config.QK_ROPE_HEAD_DIM
            widths = [
                ("token", ROWS),
                ("ring_idx", ROWS),
                ("sel", ROWS * ROWS),
                ("index_rows", ROWS * INDEX_WIDTH),
                ("main_cos", ROWS * rope),
                ("main_sin", ROWS * rope),
                ("block_cos", ROWS * rope),
                ("block_sin", ROWS * rope),
            ]
            cols, start = {}, 0
            for name, w in widths:
                cols[name] = (start, start + w)
                start += w
            self._one_cols, self._one_width = cols, start
        return self._one_cols

    @staticmethod
    def _bf16_bits(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) << 16

    def _pack_one(self, t: dict, main: tuple, block: tuple):
        """The ranks' inputs (``dspark_blocks.mesh_inputs``'s tensors, the RoPE pairs ``[1, 1, sp * ROWS, rope_dim]``
        fp32) -> the one host page per rank, ``[1, 1, 1, width]`` uint32 on every chip of the rank
        (``from_host_shards``; the host rows reused)."""
        sp, tp = self.sp, self.tp
        cols = self._one_layout()
        if getattr(self, "_one_host", None) is None:
            self._one_host = torch.zeros(sp, 1, 1, self._one_width, dtype=torch.int32)
        host = self._one_host
        lo, hi = cols["token"]
        host[:, 0, 0, lo:hi] = t["ids"].view(sp, ROWS)
        lo, hi = cols["ring_idx"]
        host[:, 0, 0, lo:hi] = t["ring_idx"].view(sp, ROWS)
        lo, hi = cols["sel"]
        host[:, 0, 0, lo:hi] = self._bf16_bits(t["sel"]).view(sp, ROWS * ROWS)
        lo, hi = cols["index_rows"]
        host[:, 0, 0, lo:hi] = t["index"].view(sp, ROWS * INDEX_WIDTH)
        for name, r in zip(("main_cos", "main_sin"), main):
            lo, hi = cols[name]
            host[:, 0, 0, lo:hi] = r.contiguous().view(torch.int32).view(sp, -1)
        for name, r in zip(("block_cos", "block_sin"), block):
            lo, hi = cols[name]
            host[:, 0, 0, lo:hi] = self._bf16_bits(r).view(sp, -1)
        parts = []
        for r in range(sp):
            shard = ttnn.Tensor(host[r : r + 1], ttnn.uint32)  # [1, 1, 1, width]: this rank's, on every TP chip
            parts.extend([shard] * tp)
        return ttnn.from_host_shards(parts, ttnn.MeshShape(sp, tp))

    def _unpack_one(self, one) -> dict:
        """The page on device -> the graph's tensors as the split path keys them (a dozen tiny ops inside the
        capture): the integer rows sliced, ``sel`` and the block RoPE bitcast to fp32, typecast to bf16 (exact: the
        low half is zero) and tilized, the main RoPE bitcast to fp32 and tilized, the index rows reshaped."""
        cols = self._one_layout()
        rope = self.config.QK_ROPE_HEAD_DIM

        def part(name, shape):
            lo, hi = cols[name]
            return ttnn.reshape(ttnn.slice(one, [0, 0, 0, lo], [1, 1, 1, hi]), shape)

        out = {
            "token": part("token", (1, 1, ROWS)),
            "ring_idx": part("ring_idx", (1, 1, 1, ROWS)),
            "index_rows": part("index_rows", (1, 1, ROWS, INDEX_WIDTH)),
        }
        sel = ttnn.typecast(ttnn.bitcast(part("sel", (1, 1, ROWS, ROWS)), ttnn.float32), ttnn.bfloat16)
        out["sel"] = ttnn.tilize(sel)
        out["rope_main"] = tuple(
            ttnn.to_layout(ttnn.bitcast(part(name, (1, 1, ROWS, rope)), ttnn.float32), ttnn.TILE_LAYOUT)
            for name in ("main_cos", "main_sin")
        )
        out["rope_block"] = tuple(
            ttnn.tilize(ttnn.typecast(ttnn.bitcast(part(name, (1, 1, ROWS, rope)), ttnn.float32), ttnn.bfloat16))
            for name in ("block_cos", "block_sin")
        )
        return out

    def _host_tap(self, main_hidden: torch.Tensor):
        """``tap_rows(main_hidden)`` as the host tensor of the ``tap`` buffer (sharded over TP on its columns)."""
        return ttnn.from_torch(
            self.tap_rows(main_hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(None, 3)),
        )

    def _use_tap32(self, use: bool):
        """The ``use_tap32`` host constant (0 or 1), one host tensor each for the drafter's life."""
        key = ("use_tap32", use)
        if key not in self._consts:
            self._consts[key] = self._replicated(torch.full((1, 1, 1, 1), float(use)), ttnn.bfloat16, device=False)
        return self._consts[key]

    def _per_rank(self, host: torch.Tensor, dtype, layout):
        """``host`` ``[sp, ...]`` -> each mesh row its slice, replicated over its TP chips (a host tensor)."""
        return ttnn.from_torch(
            host.contiguous(),
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, None)),
        )

    def tap_rows_from_streams(self, streams: list, row: int):
        """The device tap of tile row ``row`` from the target's block input streams of the tap layers (``streams``:
        per tap layer ``[1, 1, ROWS, hc * hidden/tp]`` fp32 per chip, the outputs of the blocks before them) ->
        ``[1, 1, ROWS, n_taps * hidden/tp]`` bf16 tiled per chip, the row on every tile row (``write_main_kv``'s
        ``tap``): the chip-major order of ``tap_rows`` is what the per-chip concatenation already is."""
        assert len(streams) == len(self.config.DSPARK_TARGET_LAYER_IDS)
        taps = ttnn.concat([self.seeder.tap(x) for x in streams], dim=3)  # [1, 1, ROWS, n_taps * hidden/tp] bf16
        rm = ttnn.to_layout(taps, ttnn.ROW_MAJOR_LAYOUT)
        one = ttnn.slice(rm, [0, 0, row, 0], [1, 1, row + 1, rm.shape[3]])
        out = ttnn.to_layout(ttnn.concat([one] * ROWS, dim=2), ttnn.TILE_LAYOUT)
        for t in (taps, rm, one):
            ttnn.deallocate(t)
        return out

    def stage(self, inputs: dict, tap=None, tap32=None) -> None:
        """Copy the host inputs of ``host_inputs`` into the fixed device buffers (allocated by the first call);
        ``tap``: a device tap (``tap_rows_from_streams``) copied over the host one; ``tap32``: the target step's
        tap tile of every row (``TtV41VerifyDecoder.taps``), copied into its buffer and selected by ``sel`` in the
        graph (``use_tap32`` set to 1)."""
        if tap32 is not None:
            inputs = dict(inputs, use_tap32=self._use_tap32(True))
        tick = self._tick("stage")
        zero = inputs["tap"] is self._zero_tap and self._zero_tap is not None
        if not self.buffers:
            self.buffers = self.to_device(inputs)
            self.buffers["tap32"] = ttnn.clone(self.buffers["tap"]) if tap32 is None else ttnn.clone(tap32)
            self._tap_is_zero = zero
            self._staged_use_tap32 = inputs["use_tap32"]
        else:
            for k, v in inputs.items():
                if k == "tap" and zero and self._tap_is_zero:
                    continue  # the buffer already holds the zero tap (0.5 MB a chip, 32 chips, every round)
                if k == "use_tap32":
                    if self._staged_use_tap32 is v:
                        continue  # the buffer holds this constant
                    self._staged_use_tap32 = v
                if isinstance(v, tuple):
                    for host, dev in zip(v, self.buffers[k]):
                        ttnn.copy_host_to_device_tensor(host, dev)
                else:
                    ttnn.copy_host_to_device_tensor(v, self.buffers[k])
                    if k == "tap":
                        self._tap_is_zero = zero
                tick(k)
        if tap is not None:
            ttnn.copy(tap, self.buffers["tap"])
            self._tap_is_zero = False
        if tap32 is not None:
            ttnn.copy(tap32, self.buffers["tap32"])
        tick("tap32")

    def _graph(self):
        """The draft call's graph on the buffers -> the device tensor read after it: the blocks' drafts
        (``DSPARK_MARKOV=device``) or the five logits rows (host loop, one block)."""
        if DSPARK_MARKOV == "device":
            inputs = self.buffers
            if ONE in inputs:
                inputs = dict(inputs)
                inputs.update(self._unpack_one(inputs[ONE]))
            ids, logits, _ = self.draft_ids(inputs, blocks=self.blocks)
            ttnn.deallocate(logits)
            self.gathered = self.gather_sp is not None
            if self.gathered:
                # as ``DeviceSampler.gather``: the row on every tile row, tilized, gathered over SP (a gathered
                # 128-byte row-major row arrives on the other ranks' chips as stale memory)
                tile = ttnn.to_layout(ttnn.concat([ids] * ROWS, dim=2), ttnn.TILE_LAYOUT)
                return self.gather_sp(tile, dim=2)  # [1, 1, sp * ROWS, ROWS] uint32 tiled
            return ids
        assert self.blocks == 1, "the host Markov loop takes one block"
        logits, _ = self.draft(self.buffers)
        return self.head.scored_rows(logits, 0, DSPARK_BLOCK_SIZE)

    def _read(self, out, tokens: list[int]) -> list[list[int]]:
        """The graph's output tensor -> every block's five drafts (the host loop on the logits rows in host
        mode); ``tokens``: the blocks' first tokens (the host loop's ``out[0]``). The rank-0 chip's row: the
        harness's lockstep slots; the served round reads every rank (``read_ranks``)."""
        if DSPARK_MARKOV == "device":
            if self.gathered:
                return self.read_ranks(out)[0]
            return dspark_blocks.read_drafts(ttnn.to_torch(ttnn.get_device_tensors(out)[0]), self.blocks)
        rows = self.head._tile_to_host(out, 0)
        ids, _ = self.markov_loop(tokens[0], rows)
        return [ids[1:].tolist()]

    def read_ranks(self, out) -> list[list[list[int]]]:
        """Every rank's blocks' drafts from the call's output (``run_blocks``'s ``out``): rank ``r`` -> block ``a``
        -> its five drafts (the TP chips of a rank hold the same row; the first is read)."""
        assert DSPARK_MARKOV == "device"
        tick = self._tick("read")
        if self.gathered:  # chip 0's tile: rows [r * 32, (r + 1) * 32) rank r's row on every row
            t = ttnn.to_torch(ttnn.get_device_tensors(out)[0])[0, 0]  # [sp * ROWS, ROWS]
            rows = t.view(self.sp, ROWS, ROWS)[:, 0, :]
            ranks = [dspark_blocks.read_drafts(rows[r], self.blocks) for r in range(self.sp)]
            tick("gathered")
            return ranks
        chips = ttnn.get_device_tensors(out)
        assert len(chips) == self.sp * self.tp, len(chips)
        ranks = [dspark_blocks.read_drafts(ttnn.to_torch(chips[r * self.tp]), self.blocks) for r in range(self.sp)]
        tick("ranks")
        return ranks

    def _capture(self):
        controller = SubDeviceTraceController(self.mesh_device)
        for stage in self.stages:
            stage.ffn.set_trace_controller(controller)
        ttnn.synchronize_device(self.mesh_device)
        controller.begin_capture()
        try:
            out = self._graph()
        finally:
            controller.end_capture()
            for stage in self.stages:
                stage.ffn.set_trace_controller(None)
        return controller, out

    def run(self, token: int) -> list[int]:
        """One draft call on the staged buffers (one block) -> the five drafts (``token`` the block's first
        token, for the host loop): eager, or (``trace``) captured after ``warmup`` eager calls and replayed.
        ``last`` keeps the mode and times (``device_s``: the call to a synchronize; ``read_s``: the readback and
        any host loop)."""
        return self.run_blocks([token])[0]

    def run_blocks(self, tokens: list[int], read: bool = True) -> list[list[int]]:
        """One draft call on the staged buffers (``host_inputs_blocks``: ``blocks`` blocks a rank) -> rank 0's
        blocks' drafts, five each (``tokens``: the blocks' first tokens, for the host loop). The capture is per
        block count. ``read=False``: no readback; ``out`` is the call's output tensor (``read_ranks``)."""
        assert len(tokens) == self.blocks, (len(tokens), self.blocks)
        blocks = self.blocks
        if self._eager_out is not None:  # the previous eager call's output, read by now
            ttnn.deallocate(self._eager_out)
            self._eager_out = None
        t0 = time.time()
        if not self.trace:
            mode = "eager"
        elif blocks in self.captures:
            mode = "replay"
        elif self.eager_counts.get(blocks, 0) >= self.warmup:
            mode = "capture"
        else:
            mode = "eager"
        if mode == "eager":
            self.eager_counts[blocks] = self.eager_counts.get(blocks, 0) + 1
            out = self._graph()
            ttnn.synchronize_device(self.mesh_device)
            t1 = time.time()
            drafts = self._read(out, tokens) if read else None
            self.out = self._eager_out = out
        else:
            if mode == "capture":
                self.captures[blocks] = self._capture()
            controller, out = self.captures[blocks]
            tick = self._tick("run")
            if self.replay_sync:
                ttnn.synchronize_device(self.mesh_device)
            tick("sync")
            t0 = time.time()
            controller.replay(blocking=False)
            tick("issue")
            ttnn.synchronize_device(self.mesh_device)
            tick("device")
            t1 = time.time()
            drafts = self._read(out, tokens) if read else None
            self.out = out
        self.last = {
            "mode": mode,
            "blocks": blocks,
            "device_s": round(t1 - t0, 4),
            "read_s": round(time.time() - t1, 4),
        }
        if self.input_timing:
            self.last["items"] = self.items
        return drafts

    @property
    def capture(self):
        """The one-block call's capture (the tests' and the harness's ``capture is not None``)."""
        return self.captures.get(1)

    def release(self) -> None:
        for controller, _ in self.captures.values():
            controller.release()
        self.captures = {}

    def load_rings_from_device(
        self, rings: list, start: int | None = None, user: int = 0, rank: int | None = None
    ) -> None:
        """User ``user``'s ring rows (``[u ring_slots, u ring_slots + 128)`` of every stage's KV buffer) from
        device rings ``[1, 1, 128, head_dim]`` bf16 row-major (a prefill state's ``dspark_rings``: slot
        ``q % 128`` holds position ``q`` of the last 128 before ``start``); with position-based rings
        (``ring_slots > 128``) each row goes to the user's slot ``q % ring_slots`` (``start`` required). On every
        rank (the rings are replicated), or (``rank``) on that mesh row alone: the served join of one slot, whose
        user index the other rows' slots share."""
        assert 0 <= user < self.users, (user, self.users)
        base = user * self.ring_slots
        if self.ring_slots == RING and rank is None:
            for stage, ring in zip(self.stages, rings):
                ttnn.experimental.slice_write(
                    ring, stage.kv, [0, 0, base, 0], [1, 1, base + RING, ring.shape[3]], [1, 1, 1, 1]
                )
            return
        assert start is not None and start >= RING, start
        idx = torch.full((self.sp, 1, 1, RING), SENTINEL, dtype=torch.int64)
        for r in range(self.sp) if rank is None else [rank]:
            for q in range(start - RING, start):
                idx[r, 0, 0, q % RING] = base + q % self.ring_slots
        idx = ttnn.from_torch(
            idx.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, None)),
        )
        for stage, ring in zip(self.stages, rings):
            scatter_rows(ring, stage.kv, idx, rows=RING)
        ttnn.deallocate(idx)

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
        tap = inputs["tap"]
        if "tap32" in inputs:  # the target step's tap tile: row ``sel`` on every row, or the host tap (use_tap32 0)
            picked = ttnn.matmul(inputs["sel"], inputs["tap32"], compute_kernel_config=s.compute_kernel_config)
            use = inputs["use_tap32"]
            tap = ttnn.add(ttnn.multiply(picked, use), ttnn.multiply(tap, ttnn.rsub(use, 1.0)))
        return self._write_main_kv(tap, inputs["rope_main"], inputs["ring_idx"], ROWS)

    def _write_main_kv(self, tap, rope_main: tuple, ring_idx, rows: int):
        """``tap`` [1, 1, ROWS, n_taps * hidden/tp] bf16 (every row its own), ``rope_main`` the fp32 pair of every
        row's position, ``ring_idx`` [1, 1, 1, ROWS] uint32 (the slot of each of the first ``rows`` rows, the
        sentinel for no write) -> ``main_x``; every stage's ring written at the index."""
        s = self.seeder
        y = ttnn.linear(fp8_qdq(tap), s.main_proj, dtype=ttnn.float32, compute_kernel_config=s.compute_kernel_config)
        y = ttnn.typecast(s.ccl.tp_all_reduce(y), ttnn.bfloat16)
        main_x = ttnn.rms_norm(y, weight=s.main_norm, epsilon=s.eps)
        xq = fp8_qdq(main_x)
        cos, sin = rope_main
        for layer, stage in zip(s.layers, self.stages):
            kv = ttnn.linear(xq, layer["wkv"], compute_kernel_config=s.compute_kernel_config)
            kv = s._rope(ttnn.rms_norm(kv, weight=layer["kv_norm"], epsilon=s.eps), cos, sin)
            kv = ttnn.to_layout(fp8_qdq(kv), ttnn.ROW_MAJOR_LAYOUT)
            scatter_rows(kv, stage.kv, ring_idx, rows=rows)
            ttnn.deallocate(kv)
        return main_x

    # --- the fused call (inside the verify step's capture) -------------------------------------------------------
    def _noise_run(self, n: int):
        key = ("noise", n)
        if key not in self._consts:
            self._consts[key] = self._replicated(
                torch.full((1, 1, 1, n), DSPARK_NOISE_TOKEN_ID, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
        return self._consts[key]

    def block_tokens(self, ids, rows: tuple):
        """``ids`` [1, 1, 1, ROWS] uint32 row-major (the verify step's argmax per tile row; ``rows[a]`` verify row
        ``a``'s) -> the block tile [1, 1, ROWS] uint32: row ``5 a`` = ``ids[rows[a]]`` for ``a < len(rows)``, the
        noise token elsewhere."""
        blocks = len(rows)
        parts = []
        for a in range(blocks):
            parts.append(ttnn.slice(ids, [0, 0, 0, rows[a]], [1, 1, 1, rows[a] + 1]))
            parts.append(self._noise_run(DSPARK_BLOCK_SIZE - 1))
        rest = ROWS - DSPARK_BLOCK_SIZE * blocks
        if rest:
            parts.append(self._noise_run(rest))
        return ttnn.reshape(ttnn.concat(parts, dim=3), (1, 1, ROWS))

    def block_index_rows(self, wslots, rows: tuple, ring_base: int = 0):
        """``wslots`` [1, 1, ROWS, 128] fp32 tiled (the verify step's window slots per tile row: the slots of
        positions ``p + j - 127 .. p + j`` of verify row ``j`` at tile row ``rows[j]``) -> the fused call's index
        rows [1, 1, ROWS, INDEX_WIDTH] uint32 row-major: row ``5 a + i`` lists verify row ``a``'s window (the ring
        slots of positions ``<= p + a``, offset by ``ring_base``: the user's ring with rings per slot), then its
        block's five rows ``block_base + 5 a ..``, then the sentinel; the rows past the blocks block 0's window
        alone."""
        blocks = len(rows)
        key = ("index", tuple(rows))
        if key not in self._consts:
            sel = torch.zeros(1, 1, ROWS, ROWS)
            cols = torch.full((1, 1, ROWS, INDEX_WIDTH - RING), -1, dtype=torch.int32)  # -1: the sentinel
            for r in range(ROWS):
                a = r // DSPARK_BLOCK_SIZE
                sel[0, 0, r, rows[min(a, blocks - 1)]] = 1.0
                if a < blocks:
                    cols[0, 0, r, :DSPARK_BLOCK_SIZE] = torch.arange(
                        self.block_base + DSPARK_BLOCK_SIZE * a, self.block_base + DSPARK_BLOCK_SIZE * (a + 1)
                    )
            self._consts[key] = (self._replicated(sel, ttnn.float32), self._replicated(cols, ttnn.int32))
        if wslots is None:
            return None  # ``prepare_fused``
        sel, cols = self._consts[key]
        w = ttnn.matmul(sel, wslots, compute_kernel_config=self.markov_compute)  # one 1.0 a row: exact (< 2^8)
        if ring_base:
            w = ttnn.add(w, float(ring_base))
        rows_t = ttnn.concat([ttnn.typecast(w, ttnn.int32), cols], dim=3)
        return ttnn.to_layout(ttnn.typecast(rows_t, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)

    def prepare_fused(self, layouts: list) -> None:
        """Allocate the fused call's constants for every row layout in ``layouts`` (tuples of tile rows: the
        verify rows a block takes) before any capture (a persistent tensor allocated after a capture may sit in
        that capture's freed intermediates)."""
        for rows in layouts:
            blocks = len(rows)
            for i in range(DSPARK_BLOCK_SIZE):
                self._block_mask(i, blocks)
            self._noise_run(DSPARK_BLOCK_SIZE - 1)
            if ROWS - DSPARK_BLOCK_SIZE * blocks:
                self._noise_run(ROWS - DSPARK_BLOCK_SIZE * blocks)
            self.block_index_rows(None, tuple(rows))

    def fused_draft(self, taps, ids, inputs: dict, rows: tuple, user: int = 0):
        """The draft call inside the verify step's graph (``TtV41VerifyDecoder._forward``), after the sampler:
        ``taps`` [1, 1, ROWS, n_taps * hidden/tp] bf16 (tile row ``rows[j]`` the tap of verify row ``j`` at position
        ``p + j``), ``ids`` the rows' argmaxes, ``inputs`` the step's unpacked inputs (``dring``: the ring row of
        every verify row (the user's ring with rings per slot), the sentinel elsewhere; ``dmain``: the fp32 RoPE
        pair of every row's position; ``dblock``: the bf16 pair of the block rows' positions, row ``5 a + i`` at
        ``p + a + 1 + i``; ``wslots``), ``user`` the live slot's user index (its ring with rings per slot)
        -> the drafts of every accept outcome ``a < len(rows)`` [1, 1, 1, ROWS] uint32 row-major (row ``5 a + i``
        the ``i + 1``-th draft after accepting ``a``). Every verify row's tap reaches the rings (a rejected row's
        write sits past the frontier and is rewritten when its position is really processed)."""
        assert DSPARK_MARKOV == "device", "the fused draft call keeps the Markov loop in the graph"
        assert self.ring_slots > RING, "the fused draft call needs position-based rings (ring_slots > 128)"
        blocks = len(rows)
        assert blocks * DSPARK_BLOCK_SIZE <= ROWS, blocks
        self._write_main_kv(taps, inputs["dmain"], inputs["dring"], ROWS)
        tokens = self.block_tokens(ids, rows)
        index_rows = self.block_index_rows(inputs["wslots"], rows, user * self.ring_slots if self.users > 1 else 0)
        final = self.forward_block({"token": tokens, "rope_block": inputs["dblock"], "index_rows": index_rows})
        logits, _ = self.head(final, 0)
        drafts = self._markov_device(logits, tokens, blocks)
        ttnn.deallocate(logits)
        return drafts

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
        """One draft call on device inputs: the ring writes (every row with a ring index), the block, the head ->
        (logits tile [1, 1, ROWS, vocab/tp] fp32 per chip (row j the block's row j), main_x)."""
        main_x = self.write_main_kv(inputs)
        final = self.forward_block(inputs, on_stage)
        logits, _ = self.head(final, 0)
        return logits, main_x

    def draft_ids(self, inputs: dict, on_stage=None, blocks: int = 1):
        """``draft`` followed by the device Markov loop on ``blocks`` blocks -> (the drafts [1, 1, 1, ROWS] uint32
        on device, row ``5 a + i`` block ``a``'s draft ``i + 1``; the logits tile before the biases; main_x)."""
        logits, main_x = self.draft(inputs, on_stage)
        return self._markov_device(logits, inputs["token"], blocks), logits, main_x

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
        n = self.block_base
        for stage in self.stages:
            ttnn.experimental.slice_write(
                self._replicated(torch.zeros(1, 1, n, self.config.HEAD_DIM), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                stage.kv,
                [0, 0, 0, 0],
                [1, 1, n, self.config.HEAD_DIM],
                [1, 1, 1, 1],
            )

    def load_rings(self, rings: list[torch.Tensor]) -> None:
        """Rows [0, 128) of every stage's KV buffer from host rings ``[128, head_dim]`` (slot order)."""
        for stage, ring in zip(self.stages, rings):
            t = self._replicated(ring.to(torch.bfloat16).view(1, 1, RING, -1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.slice_write(t, stage.kv, [0, 0, 0, 0], [1, 1, RING, ring.shape[-1]], [1, 1, 1, 1])
            ttnn.deallocate(t)

    def ring_rows(self, stage: int, user: int = 0) -> torch.Tensor:
        """The stage's ring of ``user`` on the host (chip 0), ``[ring_slots, head_dim]``."""
        base = user * self.ring_slots
        return ttnn.to_torch(ttnn.get_device_tensors(self.stages[stage].kv)[0])[0, 0, base : base + self.ring_slots]
