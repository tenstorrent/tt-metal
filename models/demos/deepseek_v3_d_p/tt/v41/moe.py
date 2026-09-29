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

VL routing bias (bead 10.2): image-span tokens select with ``bias_vl`` instead (reference ``Gate``; checkpoint layer
0: mean 21.2, spread 0.10). A per-row constant does not change a row's top-k either, so ``bias_vl`` is recentred by
its own mean. With an image mask the gate reads a per-token bias ``where(image, bias_vl, bias)`` for that call
(``moe_grouped_topk`` takes a bias of the scores' shape, one row per token); without one it is the text path.
"""

from types import SimpleNamespace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq


class TtV41Moe(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        seq_len: int,
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
        bias_vl = weights.get("gate_bias_vl")  # not cached with the MoE tensors: always in the dense weights
        self.moe = TtPrefillBlock._build_moe(
            mesh_device=mesh_device,
            model_cfg=config,
            config=SimpleNamespace(rms_norm_eps=config.RMS_NORM_EPS),
            state_dict=weights,
            seq_len=seq_len,
            sp_axis=0,
            emb_dim=config.EMB_SIZE,
            num_links=fabric_num_links(),
            topology=per_axis_topology(),  # (SP, TP), one per mesh axis
            gate_fallback_mode=GateComputeMode.DEVICE_FP32,
            routed_expert_activations_dtype=ttnn.bfloat8_b,
            routed_expert_weights_dtype=routed_expert_weights_dtype,
            shared_expert_activations_dtype=ttnn.bfloat16,
            shared_expert_weights_dtype=ttnn.bfloat8_b,
            # every token can land all top-k experts on one chip: top-k is the only factor that never drops a
            # token (factor 2 overflowed on repetitive real text, silently and non-deterministically); in-tree
            # DeepSeek-V3 prefill uses its top-k (8) the same way
            dispatch_buffer_capacity_factor=config.NUM_EXPERTS_PER_TOKEN,
            layer_idx=layer,
            weight_cache_path=weight_cache_path,
        )
        self.bias_vl = None
        if bias_vl is not None:
            text_bias = self.moe.gate.bias  # [tokens per chip, experts], one row per token
            assert bias_vl.shape == (text_bias.shape[-1],), (tuple(bias_vl.shape), tuple(text_bias.shape))
            bias_vl = bias_vl.float() - bias_vl.float().mean()
            self.bias_vl = ttnn.from_torch(
                bias_vl.repeat(text_bias.shape[0], 1),
                device=mesh_device,
                dtype=text_bias.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

    def set_trace_controller(self, controller):
        """Route ``TtMoe``'s sub-device load / clear (shared-expert and dispatch overlap) through a
        ``SubDeviceTraceController`` while a trace is captured (None: direct calls). A trace cannot contain a
        sub-device manager switch, so the controller splits the capture there."""
        self.moe.set_trace_controller(controller)

    def forward(self, x: ttnn.Tensor, return_intermediates: bool = False, image_mask: ttnn.Tensor | None = None):
        """x ``[1, 1, S/sp, hidden/tp]`` bf16, the sequence a contiguous SP shard -> same shape. ``image_mask``
        ``[1, 1, S/sp, 1]`` (1 = image token, same sharding; None = text only) selects ``bias_vl`` per token.

        With ``return_intermediates`` also returns ``TtMoe``'s intermediates (gate logits/scores/indices,
        shared and routed outputs) for op acceptance."""
        gate, text_bias = self.moe.gate, self.moe.gate.bias
        if image_mask is not None:
            assert self.bias_vl is not None, "image tokens need the layer's gate_bias_vl"
            rows = image_mask.shape[2]
            assert rows == text_bias.shape[0], f"mask of {rows} rows for a {text_bias.shape[0]}-row gate"
            gate.bias = ttnn.where(ttnn.reshape(image_mask, (rows, 1)), self.bias_vl, text_bias)
        try:
            out, intermediates = self.moe(
                ttnn.squeeze(x, dim=0),
                return_intermediates=return_intermediates,
                actual_isl=None,
                padding_side="right",
                actual_start=0,
                expert_x=ttnn.squeeze(fp8_qdq(x), dim=0),
            )
        finally:
            if image_mask is not None:
                ttnn.deallocate(gate.bias)
                gate.bias = text_bias
        out = ttnn.unsqueeze(out, dim=0)
        return (out, intermediates) if return_intermediates else out
