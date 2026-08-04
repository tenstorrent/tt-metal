# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 audio+video block, modelled from the source (phase 5 validation).

Mirrors ``models/tt_dit/models/transformers/ltx/attention_ltx.py`` (LTXAttention)
and ``transformer_ltx.py`` (LTXTransformerBlock.forward) for the two shipped
Blackhole configurations, which differ only in CCL topology:

===============  ============  ==========  ===================================
config           mesh          topology    ``use_nonfused_agmm``
===============  ============  ==========  ===================================
``ltx_block_bh_4x8``  (4, 8)   Ring        False -> every ColParallelLinear
                                           gathers inside its fused AGMM
``ltx_block_bh_2x4``  (2, 4)   Linear      True -> one explicit gather up front,
                                           then plain matmuls
===============  ============  ==========  ===================================

`sp_axis=1, tp_axis=0` in both (pipeline_ltx.py:402-445).

The structure that matters, from ``LTXAttention.forward``:

    use_nonfused_agmm = topology is Linear and tp > 1
    if use_nonfused_agmm:                       # Linear only
        spatial = all_gather(spatial, dim=3, tp_axis)
    qkv_parallel_config = None if use_nonfused_agmm else self.parallel_config
    gate = self._compute_gate(spatial, qkv_parallel_config)   # to_gate_logits(...)
    q, k, v = self.to_qkv(spatial, parallel_config=qkv_parallel_config)

On Ring, ``qkv_parallel_config`` is not None, so **both** ``to_gate_logits`` and
``to_qkv`` take the fused ``all_gather_minimal_matmul_async`` path and each
gathers the *same* activation over the TP axis. The gate projection's output is
tiny (``num_heads`` columns) but its gather moves the whole activation.
"""

from __future__ import annotations

from typing import List, Optional

from ..builder import GraphBuilder, Value
from ..ir import Graph, Mesh

ATTN = "models/tt_dit/models/transformers/ltx/attention_ltx.py"
TB = "models/tt_dit/models/transformers/ltx/transformer_ltx.py"

# LTX-2.3 shapes (pipeline_ltx.py defaults; sequence lengths are the SP-padded
# stage-2 values that appear in LTXAttention's chunk-size tables).
VIDEO_DIM = 32 * 128  # 4096
VIDEO_HEADS = 32
VIDEO_HEAD_DIM = 128
VIDEO_N = 38912  # ring_sdpa_chunk_by_n key (True, 8, 4, 38912)
VIDEO_FFN = 4 * VIDEO_DIM
AUDIO_DIM = 32 * 64  # 2048
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_N = 256  # cross_ring_sdpa_q_chunk_map comment: "assumes audio_N=256"
AUDIO_FFN = 4 * AUDIO_DIM
TEXT_L = 32  # sdpa_chunk_by_shape keys (True, 1216, 32) / (True, 4864, 32)
LAYERS = 48
STEPS = 8  # LTX-2.3 Fast distilled, two stages


class _Attn:
    """One LTXAttention instance, modelled branch-for-branch."""

    def __init__(self, b: GraphBuilder, name: str, dim: int, heads: int, head_dim: int, tp: int, sp: int, ring: bool):
        self.b = b
        self.name = name
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.tp, self.sp = tp, sp
        self.ring = ring
        self.use_nonfused_agmm = not ring  # Linear topology and tp > 1

    def _w(self, tag: str, k: int, n: int, shard_n: bool = True) -> Value:
        # ColParallelLinear fractures output columns over the TP axis.
        return self.b.param("%s.%s.weight" % (self.name, tag), [k, n], shard={self.tp: 1 if shard_n else 0})

    def forward(
        self,
        spatial: Value,
        *,
        query_input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        is_self: bool,
        prompt: Optional[Value] = None,
        prompt_tp_sharded: bool = False,
        kv_replicated: bool = False,
        use_ring_cross: bool = False,
        kv_needs_sp_gather: bool = False,
        masked: bool = False,
        gated: bool = True,
        rope: Optional[List[Value]] = None,
        residual: Optional[Value] = None,
    ) -> Value:
        b, name = self.b, self.name
        dim, heads, head_dim = self.dim, self.heads, self.head_dim
        q_in = query_input_dim or dim
        out_dim = output_dim or dim

        # attention_ltx.py:414-420
        if self.use_nonfused_agmm:
            spatial = b.all_gather(spatial, dim=-1, mesh_axis=self.tp, label=name + ".ag_spatial", loc=ATTN + ":418")

        def col_linear(x: Value, w: Value, label: str, loc: str) -> Value:
            """ColParallelLinear.forward: AGMM when parallel_config is passed."""
            if self.use_nonfused_agmm:
                return b.matmul(x, w, label=label, loc=loc)
            return b.agmm(x, w, mesh_axis=self.tp, dim=-1, label=label, loc=loc)

        def col_linear_chunks(x: Value, ws: List[Value], labels: List[str], loc: str) -> List[Value]:
            if self.use_nonfused_agmm:
                return [b.matmul(x, w, label=labels[i], loc=loc) for i, w in enumerate(ws)]
            return b.agmm_chunks(x, ws, mesh_axis=self.tp, dim=-1, labels=labels, label=labels[0], loc=loc)

        # attention_ltx.py:425 -- gate is computed from spatial before QKV consumes it
        gate = None
        if gated:
            gate_logits = col_linear(
                spatial, self._w("to_gate_logits", q_in, heads), name + ".to_gate_logits", ATTN + ":382"
            )
            gate = b.pointwise("sigmoid", [gate_logits], label=name + ".gate_sigmoid", loc=ATTN + ":383")
            gate = b.permute(gate, [1, 3, 2, 0] if gate.symbol.ndim == 4 else [0, 2, 1], label=name + ".gate_permute")

        if is_self:  # attention_ltx.py:428, to_qkv(chunks=3)
            q, k, v = col_linear_chunks(
                spatial,
                [self._w("to_q", q_in, dim), self._w("to_k", q_in, dim), self._w("to_v", q_in, dim)],
                [name + ".to_qkv_q", name + ".to_qkv_k", name + ".to_qkv_v"],
                ATTN + ":428",
            )
        else:  # attention_ltx.py:433-456
            kv_input = prompt if prompt is not None else spatial
            kv_fused = False
            if prompt is not None and prompt_tp_sharded:
                if self.use_nonfused_agmm:
                    kv_input = b.all_gather(
                        kv_input, dim=-1, mesh_axis=self.tp, label=name + ".ag_kv", loc=ATTN + ":442"
                    )
                else:
                    kv_fused = True  # kv_parallel_config = self.parallel_config
            q = col_linear(spatial, self._w("to_q", q_in, dim), name + ".to_q", ATTN + ":447")
            kv_dim = kv_input.shape[-1]
            k_w, v_w = self._w("to_k", kv_dim, dim), self._w("to_v", kv_dim, dim)
            if kv_fused:
                k, v = b.agmm_chunks(
                    kv_input,
                    [k_w, v_w],
                    mesh_axis=self.tp,
                    dim=-1,
                    labels=[name + ".to_kv_k", name + ".to_kv_v"],
                    label=name + ".to_kv",
                    loc=ATTN + ":452",
                )
            else:
                k = b.matmul(kv_input, k_w, label=name + ".to_kv_k", loc=ATTN + ":452")
                v = b.matmul(kv_input, v_w, label=name + ".to_kv_v", loc=ATTN + ":452")

        # attention_ltx.py:459-469 -- RMSNorm fused with the head split; V split explicitly
        q = b.split_heads(
            b.dist_norm(q, label=name + ".norm_q", loc=ATTN + ":459"), heads, head_dim, label=name + ".q_BHNE"
        )
        k = b.split_heads(
            b.dist_norm(k, label=name + ".norm_k", loc=ATTN + ":460"), heads, head_dim, label=name + ".k_BHNE"
        )
        v = b.split_heads(v, heads, head_dim, label=name + ".v_BHNE")

        # attention_ltx.py:471-489 -- cross-attn K/V gathered over SP only when genuinely sharded
        if kv_needs_sp_gather:
            k = b.all_gather(k, dim=2, mesh_axis=self.sp, label=name + ".ag_k_sp", loc=ATTN + ":488")
            v = b.all_gather(v, dim=2, mesh_axis=self.sp, label=name + ".ag_v_sp", loc=ATTN + ":489")

        if rope:  # attention_ltx.py:491-499
            q = b.pointwise("rope", [q] + rope, label=name + ".rope_q", loc=ATTN + ":494")
            k = b.pointwise("rope", [k] + rope, label=name + ".rope_k", loc=ATTN + ":497")

        if is_self and masked:  # attention_ltx.py:535 -- gather K/V, keep Q sharded, local SDPA
            k = b.all_gather(k, dim=2, mesh_axis=self.sp, label=name + ".masked_k_ag", loc=ATTN + ":538")
            v = b.all_gather(v, dim=2, mesh_axis=self.sp, label=name + ".masked_v_ag", loc=ATTN + ":539")
            attn = b.sdpa(q, k, v, label=name + ".masked_sdpa", loc=ATTN + ":540")
        elif is_self:  # ring_joint_sdpa gathers K/V over SP internally
            tag = "ring_sdpa:" + name
            kg = b.all_gather(k, dim=2, mesh_axis=self.sp, label=name + ".ring_k_ag", loc=ATTN + ":506", fused_in=tag)
            vg = b.all_gather(v, dim=2, mesh_axis=self.sp, label=name + ".ring_v_ag", loc=ATTN + ":506", fused_in=tag)
            attn = b.sdpa(q, kg, vg, label=name + ".ring_sdpa", loc=ATTN + ":506")
        elif use_ring_cross:  # attention_ltx.py:559 -- is_cross ring SDPA fuses the K/V gather
            tag = "ring_cross_sdpa:" + name
            kg = b.all_gather(k, dim=2, mesh_axis=self.sp, label=name + ".cross_k_ag", loc=ATTN + ":563", fused_in=tag)
            vg = b.all_gather(v, dim=2, mesh_axis=self.sp, label=name + ".cross_v_ag", loc=ATTN + ":563", fused_in=tag)
            attn = b.sdpa(q, kg, vg, label=name + ".cross_ring_sdpa", loc=ATTN + ":563")
        else:  # local SDPA: K/V already full-sequence
            attn = b.sdpa(q, k, v, label=name + ".sdpa", loc=ATTN + ":589")
        _ = kv_replicated

        if gate is not None:  # attention_ltx.py:600
            attn = b.pointwise("mul", [attn, gate], label=name + ".apply_gate", loc=ATTN + ":601")

        merged = b.merge_heads(attn, label=name + ".concat_heads")

        # attention_ltx.py:606-627 -- Ring fuses to_out's TP gather; Linear gathers explicitly
        w_out = self._w("to_out", dim, out_dim)
        if self.use_nonfused_agmm:
            merged = b.all_gather(merged, dim=-1, mesh_axis=self.tp, label=name + ".ag_to_out", loc=ATTN + ":610")
            out = b.matmul(merged, w_out, label=name + ".to_out", loc=ATTN + ":623")
        else:
            out = b.agmm(merged, w_out, mesh_axis=self.tp, dim=-1, label=name + ".to_out", loc=ATTN + ":337")
        if residual is not None:  # fused addcmul epilogue
            out = b.add(residual, out, label=name + ".to_out_addcmul")
        return out


def _ffn(b: GraphBuilder, name: str, x: Value, dim: int, inner: int, tp: int, ring: bool) -> Value:
    """_modulated_ffn: Ring fuses ff1(AG) + ff2 + RS + addcmul; Linear gathers explicitly."""
    normed = b.dist_norm(x, label=name + ".norm", loc=TB + ":284")
    w1 = b.param("%s.ff1.weight" % name, [dim, inner], shard={tp: 1})
    w2 = b.param("%s.ff2.weight" % name, [inner, dim], shard={tp: 0})
    if ring:
        h = b.agmm(normed, w1, mesh_axis=tp, dim=-1, label=name + ".ff1", loc=TB + ":287")
        h = b.pointwise("gelu_tanh", [h], label=name + ".gelu")
        out = b.matmul_rs(h, w2, mesh_axis=tp, dim=-1, label=name + ".ff2", loc=TB + ":287")
    else:
        normed = b.all_gather(normed, dim=-1, mesh_axis=tp, label=name + ".ag_ff", loc=TB + ":296")
        h = b.matmul(normed, w1, label=name + ".ff1", loc=TB + ":298")
        h = b.pointwise("gelu_tanh", [h], label=name + ".gelu")
        out = b.matmul_rs(h, w2, mesh_axis=tp, dim=-1, label=name + ".ff2", loc=TB + ":298")
    return b.add(x, out, label=name + ".ff_addcmul")


def ltx_block(mesh_shape=(4, 8), ring: bool = True, gated: bool = True) -> Graph:
    tp_axis, sp_axis = 0, 1  # pipeline_ltx.py: sp_axis=1, tp_axis=0 on both BH configs
    mesh = Mesh(shape=mesh_shape, axis_names=("tp", "sp"), arch="blackhole", topology="Ring" if ring else "Linear")
    b = GraphBuilder(
        "ltx_block_bh_%dx%d" % mesh_shape,
        mesh,
        steps=STEPS,
        model="LTX-2.3 (audio + video)",
        note="one LTXTransformerBlock, x%d layers, x%d steps; %s topology"
        % (LAYERS, STEPS, "Ring" if ring else "Linear"),
        parallel="sp=%d (axis1), tp=%d (axis0), gated attention %s"
        % (mesh_shape[sp_axis], mesh_shape[tp_axis], "on" if gated else "off"),
    )

    # Activations arrive SP-fractured on the sequence axis and TP-fractured on the
    # feature axis (what the FFN's reduce-scatter leaves behind).
    video = b.input("video_1BND", [1, VIDEO_N, VIDEO_DIM], shard={sp_axis: 1, tp_axis: 2})
    audio = b.input("audio_1BND", [1, AUDIO_N, AUDIO_DIM], shard={sp_axis: 1, tp_axis: 2})
    video_prompt = b.input("video_prompt", [1, TEXT_L, VIDEO_DIM])  # replicated text embeddings
    audio_prompt = b.input("audio_prompt", [1, TEXT_L, AUDIO_DIM])

    # adaLN modulation chunks. Every use site has its *own* shift/scale pair
    # (scale_shift_table / *_a2v_ca_* chunks), which is what makes the four A<->V
    # cross-attention operands four different values rather than one.
    def mod(name: str, dim: int):
        return (
            b.input("mod_%s_shift" % name, [1, 1, dim], shard={tp_axis: 2}),
            b.input("mod_%s_scale" % name, [1, 1, dim], shard={tp_axis: 2}),
        )

    mods = {
        n: mod(n, d)
        for n, d in (
            ("v_sa", VIDEO_DIM),
            ("v_ca", VIDEO_DIM),
            ("a_sa", AUDIO_DIM),
            ("a_ca", AUDIO_DIM),
            ("v_q_a2v", VIDEO_DIM),
            ("a_kv_a2v", AUDIO_DIM),
            ("a_q_v2a", AUDIO_DIM),
            ("v_kv_v2a", VIDEO_DIM),
        )
    }
    pad_mask_a2v = b.input("audio_padding_mask_full", [1, AUDIO_N, AUDIO_DIM], shard={tp_axis: 2})

    def modulate(x: Value, which: str, label: str) -> Value:
        shift, scale = mods[which]
        return b.pointwise("addcmul", [x, shift, scale], label=label, loc=TB + ":444")

    v_rope = [b.input("video_rope_cos", [1, 1, VIDEO_N, VIDEO_HEAD_DIM], shard={sp_axis: 2})]
    a_rope = [b.input("audio_rope_cos", [1, 1, AUDIO_N, AUDIO_HEAD_DIM], shard={sp_axis: 2})]

    def attn(name, dim, heads, head_dim):
        return _Attn(b, name, dim, heads, head_dim, tp_axis, sp_axis, ring)

    with b.block(calls=LAYERS, loc=TB):
        # --- video self-attention (transformer_ltx.py:344) ---
        v_normed = modulate(b.dist_norm(video, label="norm1", loc=TB + ":344"), "v_sa", "mod_v_sa")
        video = attn("attn1", VIDEO_DIM, VIDEO_HEADS, VIDEO_HEAD_DIM).forward(
            v_normed, is_self=True, gated=gated, rope=v_rope, residual=video
        )

        # --- video text cross-attention (transformer_ltx.py:369, kv_replicated=True) ---
        v_ca_in = modulate(b.dist_norm(video, label="norm2", loc=TB + ":358"), "v_ca", "mod_v_ca")
        video = attn("attn2", VIDEO_DIM, VIDEO_HEADS, VIDEO_HEAD_DIM).forward(
            v_ca_in, is_self=False, prompt=video_prompt, kv_replicated=True, gated=gated, residual=video
        )

        # --- audio self-attention (masked: gather K/V over SP, local SDPA) ---
        a_normed = modulate(b.dist_norm(audio, label="audio_norm1", loc=TB + ":390"), "a_sa", "mod_a_sa")
        audio = attn("audio_attn1", AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM).forward(
            a_normed, is_self=True, gated=gated, rope=a_rope, residual=audio, masked=True
        )

        # --- audio text cross-attention ---
        a_ca_in = modulate(b.dist_norm(audio, label="audio_norm2", loc=TB + ":404"), "a_ca", "mod_a_ca")
        audio = attn("audio_attn2", AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM).forward(
            a_ca_in, is_self=False, prompt=audio_prompt, kv_replicated=True, gated=gated, residual=audio
        )

        # --- bidirectional A<->V cross-attention (transformer_ltx.py:431-485) ---
        # norm3 / audio_norm3 are computed once and feed both directions, but each
        # direction applies its own shift/scale, so Q and K/V are distinct values.
        v_x = b.dist_norm(video, label="norm3_xattn", loc=TB + ":441")
        a_x = b.dist_norm(audio, label="audio_norm3_xattn", loc=TB + ":442")

        video_q_a2v = modulate(v_x, "v_q_a2v", "mod_v_q_a2v")
        audio_kv_a2v = modulate(a_x, "a_kv_a2v", "mod_a_kv_a2v")
        # A->V: the block gathers audio K/V over SP first (transformer_ltx.py:447)
        audio_kv_a2v = b.all_gather(audio_kv_a2v, dim=1, mesh_axis=sp_axis, label="ag_audio_kv_a2v", loc=TB + ":448")
        audio_kv_a2v = b.mul(audio_kv_a2v, pad_mask_a2v, label="zero_audio_padding")
        video = attn("a2v_attn", AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM).forward(
            video_q_a2v,
            is_self=False,
            query_input_dim=VIDEO_DIM,
            output_dim=VIDEO_DIM,
            prompt=audio_kv_a2v,
            prompt_tp_sharded=True,
            gated=gated,
            rope=v_rope,
            residual=video,
        )

        # V->A: video K/V stay SP-sharded; the ring cross SDPA gathers them internally
        audio_q_v2a = modulate(a_x, "a_q_v2a", "mod_a_q_v2a")
        video_kv_v2a = modulate(v_x, "v_kv_v2a", "mod_v_kv_v2a")
        audio = attn("v2a_attn", AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM).forward(
            audio_q_v2a,
            is_self=False,
            prompt=video_kv_v2a,
            prompt_tp_sharded=True,
            use_ring_cross=True,
            gated=gated,
            rope=a_rope,
            residual=audio,
        )

        # --- feed forwards ---
        video = _ffn(b, "ffn", video, VIDEO_DIM, VIDEO_FFN, tp_axis, ring)
        audio = _ffn(b, "audio_ff", audio, AUDIO_DIM, AUDIO_FFN, tp_axis, ring)

    return b.finish([video, audio])


def ltx_block_bh_4x8() -> Graph:
    """Blackhole 4x8, Ring topology: the shipped large config."""
    return ltx_block(mesh_shape=(4, 8), ring=True)


def ltx_block_bh_2x4() -> Graph:
    """Blackhole 2x4, Linear topology: same source, explicit-gather path."""
    return ltx_block(mesh_shape=(2, 4), ring=False)
