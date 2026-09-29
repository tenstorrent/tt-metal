# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 MoE: the in-tree ``TtMoe`` at V4.1 dims (bead F1).

Reference: ``inference/model.py`` ``MoE`` / ``Gate`` / ``Expert``. 384 routed experts, top-6 of
``sqrt(softplus(x @ W_gate))`` scores (fp32 gate GEMM; the correction bias steers selection only,
weights normalized with ``+1e-20`` and scaled by ``route_scale``), clamped SwiGLU experts
(``SWIGLU_LIMIT``) and one shared expert every token goes through.

Device mapping: gate ``GateComputeMode.DEVICE_FP32`` (bf16 operands, fp32 accumulation, sqrtsoftplus in
``moe_grouped_topk``); routed experts in ``routed_expert_weights_dtype`` (bfp8 = bring-up default, D-B),
bfp8 activations; shared expert bfp8 weights, bf16 activations.

Activation quantization (dev-spec D-I): the reference FP8-quantize-dequantizes (e4m3, block 32, ue8m0) the
input of every expert GEMM (``linear()`` on FP4/FP8 weights) but not the gate's (fp32 GEMM on bf16 input).
The device reproduces the expert-input QDQ bit for bit (``qdq.fp8_qdq``, row-local, so applying it before
dispatch equals the reference's per-expert application) and routes on the unquantized input. Measured on
V4.1 layer 2 (2x4, synthetic, bfp8): shared expert PCC 0.99863 -> 0.99936 (G1 bar 0.999), routed
0.99390 -> 0.99467, final 0.99609 -> 0.99682. The QDQ of the ``w2`` input inside the experts is not
emulated (it sits inside the fused expert ops).

Correction bias: the device gate holds ``e_score_correction_bias`` in bf16. The checkpoint's biases sit near
10.76 with a spread of ~0.04 (layer 2), below the bf16 step there (0.0625), so storing them as is collapses
the selection (measured top-6 recall 0.90 on real layer 2). Selection is ``topk(scores + bias)`` and the
weights come from the unbiased scores, so subtracting the mean bias changes nothing in the reference
semantics while keeping the offsets within bf16 precision (CPU: recall 0.998 with bf16 logits).
"""

from types import SimpleNamespace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq


class TtV41Moe(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        seq_len: int,
        num_links: int = 1,
        topology=ttnn.Topology.Linear,
        routed_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path=None,
    ):
        """``weights``: the ``TtMoe`` state-dict entries ``gate_weights`` (weight, e_score_correction_bias),
        ``routed_expert_weights`` (one {gate_proj, up_proj, down_proj} per expert) and
        ``shared_expert_weights``, in checkpoint ``[out, in]`` orientation (``weights.load_layer``)."""
        gate = weights.get("gate_weights")  # absent when the device tensors come from weight_cache_path
        if gate is not None:
            bias = gate["e_score_correction_bias"].float()
            weights = {**weights, "gate_weights": {**gate, "e_score_correction_bias": bias - bias.mean()}}
        self.moe = TtPrefillBlock._build_moe(
            mesh_device=mesh_device,
            model_cfg=config,
            config=SimpleNamespace(rms_norm_eps=config.RMS_NORM_EPS),
            state_dict=weights,
            seq_len=seq_len,
            sp_axis=0,
            emb_dim=config.EMB_SIZE,
            num_links=num_links,
            topology=topology,
            gate_fallback_mode=GateComputeMode.DEVICE_FP32,
            routed_expert_activations_dtype=ttnn.bfloat8_b,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
            shared_expert_activations_dtype=ttnn.bfloat16,
            shared_expert_weights_dtype=ttnn.bfloat8_b,
            dispatch_buffer_capacity_factor=2,
            layer_idx=layer,
            weight_cache_path=weight_cache_path,
        )

    def forward(self, x: ttnn.Tensor, return_intermediates: bool = False):
        """x ``[1, 1, S/sp, hidden/tp]`` bf16, the sequence a contiguous SP shard -> same shape.

        With ``return_intermediates`` also returns ``TtMoe``'s intermediates (gate logits/scores/indices,
        shared and routed outputs) for op acceptance."""
        out, intermediates = self.moe(
            ttnn.squeeze(x, dim=0),
            return_intermediates=return_intermediates,
            actual_isl=None,
            padding_side="right",
            actual_start=0,
            expert_x=ttnn.squeeze(fp8_qdq(x), dim=0),
        )
        out = ttnn.unsqueeze(out, dim=0)
        return (out, intermediates) if return_intermediates else out
