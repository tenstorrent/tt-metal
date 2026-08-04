# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gold graphs modelled on the real SD3.5-large joint transformer block.

``sd35_block`` mirrors ``models/tt_dit/blocks/transformer_block.py`` +
``blocks/attention.py`` + ``layers/{linear,feedforward}.py`` on a 2x4 mesh
(SP=2 on axis 0, TP=4 on axis 1). Every collective in it is load-bearing, so it
is the analyzer's precision test: it must report **no** findings.

``sd35_block_double_gather`` is the same block with the ColParallelLinear calls
switched to the fused ``all_gather_minimal_matmul_async`` path while the
explicit pre-gathers in ``transformer_block``/``attention`` are left in place --
the mistake class behind "12 AGMM collectives could be 6". Each fused kernel
then re-gathers a tensor that is already replicated on the TP axis.
"""

from __future__ import annotations

from ..builder import GraphBuilder
from ..ir import Graph, Mesh

SP, TP = 0, 1

# SD3.5-large shapes at 1024x1024, heads padded 38 -> 40 for TP=4.
S = 4096  # spatial sequence (patch tokens)
P = 352  # prompt sequence (padded)
D = 2432  # hidden dim
HEADS = 40
HEAD_DIM = 64
INNER = HEADS * HEAD_DIM  # 2560
FF_INNER = 4 * D  # 9728
BLOCKS = 38
STEPS = 28

TB = "models/tt_dit/blocks/transformer_block.py"
ATTN = "models/tt_dit/blocks/attention.py"
LIN = "models/tt_dit/layers/linear.py"


def sd35_block(fused_agmm: bool = False) -> Graph:
    name = "sd35_large_block" + ("_double_gather" if fused_agmm else "")
    b = GraphBuilder(
        name,
        Mesh(shape=(2, 4), axis_names=("sp", "tp")),
        steps=STEPS,
        model="Stable Diffusion 3.5 Large",
        note="one joint transformer block, x%d blocks, x%d denoise steps" % (BLOCKS, STEPS),
        parallel="cfg=1, sp=2 (axis0), tp=4 (axis1)",
    )

    # Block inputs. Both activations arrive TP-fractured on the feature axis
    # (that is what the RowParallel reduce-scatter at the end of a block leaves
    # behind); spatial is additionally SP-fractured on the sequence axis.
    spatial = b.input("spatial", [1, S, D], shard={SP: 1, TP: 2})
    prompt = b.input("prompt", [1, P, D], shard={TP: 2})

    # adaLN modulation chunks (produced by norm1_linear, a ColParallelLinear, so
    # they are TP-fractured on the feature axis like the activations).
    mod = {
        n: b.input("mod_" + n, [1, 1, D], shard={TP: 2})
        for n in ("shift_attn", "scale_attn", "gate_attn", "shift_ff", "scale_ff", "gate_ff")
    }
    pmod = {
        n: b.input("pmod_" + n, [1, 1, D], shard={TP: 2})
        for n in ("shift_attn", "scale_attn", "gate_attn", "shift_ff", "scale_ff", "gate_ff")
    }
    rope_cos = b.input("rope_cos", [1, 1, S, HEAD_DIM], shard={SP: 2})
    rope_sin = b.input("rope_sin", [1, 1, S, HEAD_DIM], shard={SP: 2})

    # Weights. ColParallelLinear fractures N (output columns) over TP;
    # RowParallelLinear fractures K (input rows) over TP.
    w_qkv = b.param("attn.to_qkv.weight", [D, 3 * INNER], shard={TP: 1})
    w_out = b.param("attn.to_out.weight", [INNER, D], shard={TP: 1})
    w_add_out = b.param("attn.to_add_out.weight", [INNER, D], shard={TP: 1})
    w_ff1 = b.param("ff.ff1.weight", [D, FF_INNER], shard={TP: 1})
    w_ff2 = b.param("ff.ff2.weight", [FF_INNER, D], shard={TP: 0})
    w_cff1 = b.param("ff_context.ff1.weight", [D, FF_INNER], shard={TP: 1})
    w_cff2 = b.param("ff_context.ff2.weight", [FF_INNER, D], shard={TP: 0})

    def col_linear(x, w, label, loc):
        """ColParallelLinear: x must be K-complete on the TP axis."""
        if fused_agmm:
            return b.agmm(x, w, mesh_axis=TP, dim=-1, label=label, loc=loc)
        return b.matmul(x, w, label=label, loc=loc)

    with b.block(calls=BLOCKS, loc=TB):
        # ---- norm1 + gather before attention --------------------------------
        spatial_normed = b.dist_norm(spatial, [mod["scale_attn"], mod["shift_attn"]], label="norm1", loc=TB + ":232")
        prompt_normed = b.dist_norm(
            prompt, [pmod["scale_attn"], pmod["shift_attn"]], label="norm1_context", loc=TB + ":257"
        )
        spatial_g = b.all_gather(spatial_normed, dim=2, mesh_axis=TP, label="ag_spatial_pre_attn", loc=TB + ":267")
        prompt_g = b.all_gather(prompt_normed, dim=2, mesh_axis=TP, label="ag_prompt_pre_attn", loc=TB + ":270")

        # ---- attention ------------------------------------------------------
        qkv = col_linear(spatial_g, w_qkv, "attn.to_qkv", ATTN + ":238")
        q, k, v = b.split_qkv_heads(qkv, heads=HEADS, head_dim=HEAD_DIM, label="attn.qkv")
        q = b.norm(q, label="attn.norm_q", loc=ATTN + ":250")
        k = b.norm(k, label="attn.norm_k", loc=ATTN + ":251")
        q = b.pointwise("rope", [q, rope_cos, rope_sin], label="attn.rope_q", loc=ATTN + ":354")
        k = b.pointwise("rope", [k, rope_cos, rope_sin], label="attn.rope_k", loc=ATTN + ":354")

        add_qkv = col_linear(prompt_g, w_qkv, "attn.add_qkv_proj", ATTN + ":259")
        aq, ak, av = b.split_qkv_heads(add_qkv, heads=HEADS, head_dim=HEAD_DIM, label="attn.add_qkv")
        aq = b.norm(aq, label="attn.norm_added_q", loc=ATTN + ":263")
        ak = b.norm(ak, label="attn.norm_added_k", loc=ATTN + ":264")

        # ring_joint_scaled_dot_product_attention gathers K/V over the SP axis
        # inside the kernel; the joint (prompt) K/V are already replicated there.
        ring = "ring_sdpa:attn"
        k_g = b.all_gather(k, dim=2, mesh_axis=SP, label="attn.ring_k_ag", loc=ATTN + ":275", fused_in=ring)
        v_g = b.all_gather(v, dim=2, mesh_axis=SP, label="attn.ring_v_ag", loc=ATTN + ":275", fused_in=ring)
        spatial_attn = b.sdpa(q, k_g, v_g, extra=[ak, av], label="attn.sdpa_spatial", loc=ATTN + ":275")
        prompt_attn = b.sdpa(aq, k_g, v_g, extra=[ak, av], label="attn.sdpa_prompt", loc=ATTN + ":275")

        spatial_attn = b.merge_heads(spatial_attn, label="attn.concat_heads")
        prompt_attn = b.merge_heads(prompt_attn, label="attn.concat_heads_prompt")

        spatial_attn_g = b.all_gather(spatial_attn, dim=2, mesh_axis=TP, label="ag_attn_out", loc=ATTN + ":325")
        spatial_attn_o = col_linear(spatial_attn_g, w_out, "attn.to_out", ATTN + ":328")
        prompt_attn_g = b.all_gather(prompt_attn, dim=2, mesh_axis=TP, label="ag_attn_add_out", loc=ATTN + ":331")
        prompt_attn_o = col_linear(prompt_attn_g, w_add_out, "attn.to_add_out", ATTN + ":334")

        # ---- residual + feed forward ---------------------------------------
        spatial_plus = b.add(spatial, b.mul(spatial_attn_o, mod["gate_attn"]), label="spatial+attn")
        prompt_plus = b.add(prompt, b.mul(prompt_attn_o, pmod["gate_attn"]), label="prompt+attn")

        spatial_normed2 = b.dist_norm(spatial_plus, [mod["scale_ff"], mod["shift_ff"]], label="norm2", loc=TB + ":288")
        spatial_ff_in = b.all_gather(spatial_normed2, dim=2, mesh_axis=TP, label="ag_spatial_pre_ff", loc=TB + ":297")
        ff1 = col_linear(spatial_ff_in, w_ff1, "ff.ff1", LIN + ":296")
        ff1 = b.pointwise("gelu", [ff1], label="ff.gelu")
        # RowParallelLinear: K is fractured, so the matmul is partial and a
        # reduce-scatter both reduces and re-fractures the output.
        spatial_ff = b.matmul_rs(ff1, w_ff2, mesh_axis=TP, dim=-1, label="ff.ff2", loc=LIN + ":416")
        spatial_out = b.add(spatial_plus, b.mul(spatial_ff, mod["gate_ff"]), label="spatial+ff")

        prompt_normed2 = b.dist_norm(
            prompt_plus, [pmod["scale_ff"], pmod["shift_ff"]], label="norm2_context", loc=TB + ":313"
        )
        prompt_ff_in = b.all_gather(prompt_normed2, dim=2, mesh_axis=TP, label="ag_prompt_pre_ff", loc=TB + ":322")
        cff1 = col_linear(prompt_ff_in, w_cff1, "ff_context.ff1", LIN + ":296")
        cff1 = b.pointwise("gelu", [cff1], label="ff_context.gelu")
        prompt_ff = b.matmul_rs(cff1, w_cff2, mesh_axis=TP, dim=-1, label="ff_context.ff2", loc=LIN + ":416")
        prompt_out = b.add(prompt_plus, b.mul(prompt_ff, pmod["gate_ff"]), label="prompt+ff")

    return b.finish([spatial_out, prompt_out])


def sd35_block_double_gather() -> Graph:
    return sd35_block(fused_agmm=True)
