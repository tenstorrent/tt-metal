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

Expert placement (bead 8y7.9.8): the in-tree MoE stores expert slot ``s`` on linear chip ``s // experts_per_chip``.
``expert_placement.expert_order`` relabels the experts so that the chips carry a balanced MoE cost on real text:
device slot ``s`` holds checkpoint expert ``expert_order[s]`` (gate rows, both biases and the routed experts are
permuted together, so routing and output are unchanged up to top-k ties). The relabelling stays internal: the
intermediates return gate logits and indices in checkpoint expert ids. ``expert_order`` is None (checkpoint order)
for layers without a load profile.
"""

from types import SimpleNamespace

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.v41.ccl import fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.expert_placement import expert_order
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
        self.expert_order = expert_order(layer, config.NUM_ROUTED_EXPERTS, *tuple(mesh_device.shape))  # (SP, TP)
        if self.expert_order is not None:
            weights = _to_slot_order(weights, list(self.expert_order))
        self._checkpoint_id_tables = {}
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
        if return_intermediates and self.expert_order is not None:
            self._to_checkpoint_ids(intermediates)
        return (out, intermediates) if return_intermediates else out

    def _id_table(self, name: str, shape, dtype) -> ttnn.Tensor:
        """Replicated lookup of ``shape`` whose last dim is ``slot_of`` (checkpoint expert -> slot) or ``order``
        (slot -> checkpoint expert)."""
        shape = tuple(shape)
        key = (name, shape, dtype)
        if key not in self._checkpoint_id_tables:
            order = torch.tensor(self.expert_order, dtype=torch.int64)
            row = torch.argsort(order) if name == "slot_of" else order
            self._checkpoint_id_tables[key] = ttnn.from_torch(
                row.expand(*shape).contiguous().to(torch.int32),
                device=self.moe.mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.moe.mesh_device),
            )
        return self._checkpoint_id_tables[key]

    def _to_checkpoint_ids(self, intermediates) -> None:
        """Gate logits (expert columns) and indices of ``intermediates`` from device slots to checkpoint experts."""
        logits = intermediates.gate_logits
        intermediates.gate_logits = ttnn.gather(logits, -1, self._id_table("slot_of", logits.shape, ttnn.uint32))
        indices = intermediates.gate_indices
        tiled = ttnn.to_layout(indices, ttnn.TILE_LAYOUT) if indices.layout != ttnn.TILE_LAYOUT else indices
        table = self._id_table("order", (*tuple(tiled.shape)[:-1], len(self.expert_order)), tiled.dtype)
        ids = ttnn.gather(table, -1, tiled)
        intermediates.gate_indices = ttnn.to_layout(ids, indices.layout) if indices.layout != ttnn.TILE_LAYOUT else ids


def _to_slot_order(weights: dict, order: list[int]) -> dict:
    """``weights`` with every per-expert entry present (gate rows, correction bias, VL bias, routed experts) in
    device slot order: slot ``s`` gets checkpoint expert ``order[s]``."""
    weights = dict(weights)
    if (gate := weights.get("gate_weights")) is not None:
        weights["gate_weights"] = {k: v[order] for k, v in gate.items()}  # weight rows, e_score_correction_bias
    if (bias_vl := weights.get("gate_bias_vl")) is not None:
        weights["gate_bias_vl"] = bias_vl[order]
    if (experts := weights.get("routed_expert_weights")) is not None:
        weights["routed_expert_weights"] = [experts[e] for e in order]
    return weights
