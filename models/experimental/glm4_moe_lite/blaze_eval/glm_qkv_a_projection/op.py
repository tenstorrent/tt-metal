# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GLMQKVAProjection: q_a_proj and kv_a_proj from ONE shared activation.

Built for GLM-4.7-Flash on a 1x-harvested BH Galaxy, out of micro-ops verified working at GLM's
dims on a 12x10 grid. It exists because blaze's own MLA fused ops cannot express this model:
`layout_plan` requires n_heads_per_device % 8 == 0 and GLM has 20 heads, which no TP divisor
fixes. DRAMStreamingMatmul has no such constraint -- it is correct at all six of GLM's decode
matmul shapes -- so the projections can be fused directly.

WHY THESE TWO. In GLM's decode both read the same normed hidden state (2048) and are the largest
per-call gap measured against ttnn:

    q_a_proj   2048 -> 768   ttnn 44.7 us   blaze 4.9 us   9.17x
    kv_a_proj  2048 -> 576   ttnn 44.7 us   blaze 5.0 us   9.03x

ttnn runs them as two independent matmuls, each resharding the activation across up to 64 cores.
Fusing them prepares the activation once and lets both matmuls consume it from the same CB, on
the 8 DRAM-bank workers, with no DRAM round-trip in between.

    act ──► DRAMStreamingMatmul(w_q_a)  ──► q_a_out      pop_act=False, act stays live
        └─► DRAMStreamingMatmul(w_kv_a) ──► kv_a_out     pop_act=True, last consumer

Modelled on `swiglu`'s gate/up pair, which is the same shape of composition. Unlike swiglu this
stops before any Gather, deliberately: swiglu's post-gather mcast is where the routed expert
deadlocks at GLM's dims (its gather stalls with 16 sender frames stuck), and neither projection
here needs a cross-core reduction -- each core owns a disjoint N slice of the output.

cb_in1 is NOT shared between the two. swiglu shares it when the weight page geometry matches, but
GLM's per-core shards differ (768/8 = 96 against 576/8 = 72), so they get independent weight CBs.
"""


from ...blaze_op import BlazeOp, FusedOp, Input, Output
from ...fused_program import MultiOutput
from ..dram_streaming_matmul import DRAMStreamingMatmul


class GLMQKVAProjection(FusedOp):
    """Fused q_a / kv_a projection sharing a single activation."""

    name: str = "glm_qkv_a_projection"
    math_fidelity: str = "LoFi"
    math_approx_mode: bool = True

    act: Input = Input()
    q_a_weights: Input = Input()
    kv_a_weights: Input = Input()
    q_a_out: Output = Output()
    kv_a_out: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(
            f,
            tensors["act"],
            tensors["q_a_weights"],
            tensors["kv_a_weights"],
            q_a_out=tensors.get("q_a_out"),
            kv_a_out=tensors.get("kv_a_out"),
            prefix=user_args.get("prefix", "glm_qkv_a"),
            fp32_dest_acc_en=bool(user_args.get("fp32_dest_acc_en", True)),
            q_a_subblock_k=user_args.get("q_a_subblock_k"),
            kv_a_subblock_k=user_args.get("kv_a_subblock_k"),
        )

    @staticmethod
    def emit(
        f,
        act,
        q_a_weights,
        kv_a_weights,
        *,
        q_a_out=None,
        kv_a_out=None,
        prefix: str = "glm_qkv_a",
        fp32_dest_acc_en: bool = True,
        q_a_subblock_k: int | None = None,
        kv_a_subblock_k: int | None = None,
    ) -> MultiOutput:
        """Emit both projections against one activation.

        Declaration order is q_a then kv_a, which is also the order the model runs them in. That
        ordering is the ONLY sequencing guarantee -- the two matmuls occupy the same core set, so
        they serialise on it rather than overlapping; the win here is the shared activation and
        the absence of a DRAM round-trip, not parallelism.
        """
        q = DRAMStreamingMatmul.emit(
            f,
            act,
            q_a_weights,
            index=None,
            bias=None,
            out=q_a_out,
            prefix=BlazeOp.child_prefix(prefix, "q_a_proj"),
            fp32_dest_acc_en=fp32_dest_acc_en,
            subblock_k=q_a_subblock_k,
            fused_activation=None,
            index_offset=0,
            # kv_a still needs the activation, so do not pop it here.
            wait_for_out=False,
            pop_index=False,
            pop_act=False,
        )
        kv = DRAMStreamingMatmul.emit(
            f,
            act,
            kv_a_weights,
            index=None,
            bias=None,
            out=kv_a_out,
            prefix=BlazeOp.child_prefix(prefix, "kv_a_proj"),
            fp32_dest_acc_en=fp32_dest_acc_en,
            subblock_k=kv_a_subblock_k,
            fused_activation=None,
            index_offset=0,
            wait_for_out=False,
            pop_index=False,
            # Last consumer of the activation.
            pop_act=True,
        )
        return MultiOutput({"q_a_out": q, "kv_a_out": kv})

    @staticmethod
    def golden(act, q_a_weights, kv_a_weights):
        """Reference: the same activation through both projections."""
        return (
            DRAMStreamingMatmul.golden(act, q_a_weights),
            DRAMStreamingMatmul.golden(act, kv_a_weights),
        )
