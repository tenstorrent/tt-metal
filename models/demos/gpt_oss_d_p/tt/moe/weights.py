# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS routed-expert weight preparation for the EP MoE (TtGptOssMoE / TtRoutedExpert).

HF GPT-OSS packs each expert's gate and up projections into a single interleaved tensor
``experts.gate_up_proj`` of shape ``[E, hidden, 2*inter]`` (gate = even columns, up = odd
columns), with a matching interleaved ``experts.gate_up_proj_bias`` of shape ``[E, 2*inter]``.
The down projection lives in ``experts.down_proj`` ``[E, inter, hidden]`` with bias
``experts.down_proj_bias`` ``[E, hidden]``.

This module:
  * de-interleaves ``gate_up_proj[..., ::2]`` -> gate, ``[..., 1::2]`` -> up (and the matching
    bias split from ``gate_up_proj_bias``),
  * transposes every projection to HuggingFace ``(out_features, in_features)`` per expert — the
    layout ``TtRoutedExpert`` expects (it transposes back to ``(in, out)`` internally for the
    matmul). De-interleaved gate/up are ``(hidden=in, inter=out)`` and ``down`` is
    ``(inter=in, hidden=out)``, so all three are transposed,
  * emits the per-expert ``routed_expert_weights`` list in GLOBAL expert-id order 0..E-1
    (``TtRoutedExpert`` / ``ExpertMapping.gather_weights_for_mesh_distribution`` performs the mesh
    gather itself — no explicit permutation here).

Biases are emitted into a SEPARATE structure (``routed_expert_biases``) and are NOT attached to
the weight dicts: the current ``unified_routed_expert_moe`` kernel on this branch does NOT accept
bias args (see #49619). The biased hookup is stubbed in ``TtGptOssMoE``.

Target dtypes (applied downstream by TtRoutedExpert, not here): weights bfloat4_b, activations
bfloat8_b. This module returns plain torch tensors.
"""


def prepare_routed_expert_weights(state_dict, num_experts, hidden_size, intermediate_size):
    """Build the per-expert routed-expert weights + biases from an ``experts.*`` sub state dict.

    Args:
        state_dict: experts sub-dict with keys ``gate_up_proj`` ``[E, hidden, 2*inter]``,
            ``gate_up_proj_bias`` ``[E, 2*inter]``, ``down_proj`` ``[E, inter, hidden]``,
            ``down_proj_bias`` ``[E, hidden]``.
        num_experts: number of routed experts E (global).
        hidden_size: model embedding dim (hidden).
        intermediate_size: MoE FFN hidden dim (inter).

    Returns:
        (routed_expert_weights, routed_expert_biases):
          * routed_expert_weights: list[dict] len E, global order 0..E-1, keys
            ``gate_proj``/``up_proj``/``down_proj`` in HF ``(out, in)`` layout.
          * routed_expert_biases: list[dict] len E, global order 0..E-1, keys
            ``gate_bias``/``up_bias``/``down_bias`` (1D). SEPARATE from the weights (#49619).
    """
    gate_up_proj = state_dict["gate_up_proj"]  # [E, hidden, 2*inter]
    gate_up_proj_bias = state_dict["gate_up_proj_bias"]  # [E, 2*inter]
    down_proj = state_dict["down_proj"]  # [E, inter, hidden]
    down_proj_bias = state_dict["down_proj_bias"]  # [E, hidden]

    # De-interleave: gate = even columns, up = odd columns.
    gate = gate_up_proj[..., ::2].reshape(num_experts, hidden_size, intermediate_size)  # (in=hidden, out=inter)
    up = gate_up_proj[..., 1::2].reshape(num_experts, hidden_size, intermediate_size)  # (in=hidden, out=inter)
    gate_bias = gate_up_proj_bias[..., ::2].reshape(num_experts, intermediate_size)  # [E, inter]
    up_bias = gate_up_proj_bias[..., 1::2].reshape(num_experts, intermediate_size)  # [E, inter]
    down = down_proj.reshape(num_experts, intermediate_size, hidden_size)  # (in=inter, out=hidden)
    down_bias = down_proj_bias.reshape(num_experts, hidden_size)  # [E, hidden]

    routed_expert_weights = []
    routed_expert_biases = []
    for e in range(num_experts):
        # Transpose to HF (out, in): gate/up -> (inter, hidden); down -> (hidden, inter).
        routed_expert_weights.append(
            {
                "gate_proj": gate[e].transpose(0, 1).contiguous(),
                "up_proj": up[e].transpose(0, 1).contiguous(),
                "down_proj": down[e].transpose(0, 1).contiguous(),
            }
        )
        routed_expert_biases.append(
            {
                "gate_bias": gate_bias[e].contiguous(),
                "up_bias": up_bias[e].contiguous(),
                "down_bias": down_bias[e].contiguous(),
            }
        )

    return routed_expert_weights, routed_expert_biases
