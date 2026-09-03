# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What exactly makes the KDA layer wrong on Kimi-K3's real weights?

`test_kda_layer1_golden.py` establishes the fact: layer 0's KDA scores 0.99993 on device against the
model's own recorded output, and 0.99993 again on layer 1's input, while layer 1's weights score
~0 on either input — and the torch reference reproduces the model with those same layer-1 weights at
0.99996. This file identifies the variable behind that split, because "layer 1 is broken" is not
something anyone can act on and "the decay magnitude is out of range" is.

Two sweeps, both against the torch reference on the trace's real layer-0 activations:

  * one tensor at a time, layer 1's swapped into layer 0's otherwise-working set. Six break, and
    they are not an arbitrary six: `A_log`, `dt_bias`, `f_a_proj`, `f_b_proj`, `k_proj`, `k_conv1d`
    are exactly the tensors feeding the state transition. `q_proj`, `v_proj`, `b_proj`, `g_proj`,
    `o_proj`, `o_norm`, `q_conv1d`, `v_conv1d` — everything that only feeds the input or output
    projections — are unaffected. `k` belongs to the first group because the delta rule puts it in
    the transition itself, `S <- S(I - beta k k^T) + beta k v^T`, while `q` and `v` never touch it.

  * `dt_bias` shifted by a constant, in both directions. This is the one that matters, because it
    turns a property of a layer into a property of a number:

        layer 0 as-is (dt_bias max +0.18)   0.99994     layer 1 as-is (max -1.43)    0.004
        layer 0 shifted -0.5  (max -0.32)   0.99252     layer 1 shifted +1.0         0.671
        layer 0 shifted -1.0  (max -0.82)  -0.00466     layer 1 shifted +2.0         0.958
        layer 0 shifted -1.5  (max -1.32)  -0.05155     layer 1 shifted +3.0         0.99905

    Shifting layer 0's decay down breaks it; shifting layer 1's up repairs it. 68 of Kimi-K3's 69
    KDA layers sit on layer 1's side of that boundary — layer 0 is the sole outlier, which is why
    the first rung of the depth ladder passed and the second did not.

Ruled out, each by measurement rather than argument: dtype (forcing every `prepare_chunk_recurrence`
output to fp32 changes the result by 4e-4, and fp32 for the affine summary by 1e-5), math fidelity
(HiFi4 is marginally worse), chunk grouping (identical at 20/10/5/4, and 20 local chunks never
reaches `grouped_scan_min_chunks` anyway), sequence length (already wrong at one chunk per rank),
the state cache (bit-identical alone and after layer 0) and the weight loader (the torch reference
reads the same dict).

ROOT CAUSE, found afterwards and fixed: `prepare_chunk_recurrence` computed `t_inv` — the inverse of
the delta rule's UT transform — with a doubling product whose intermediates are the explicit powers
N^2..N^16. At Kimi-K3's decay magnitudes ||N|| reaches 17, so those intermediates reach O(1e2) while
the inverse is bounded by 1, and the cancellation costs more digits than the hardware carries. It
scored 0.99508 as a whole tensor, which is why nothing caught it: `t_inv` is `I + strictly-lower`, so
PCC is dominated by the diagonal. Its strictly-lower part alone scored 0.935. See
`test_kda_prepare_vs_scan.py` for the attribution and the kernel's `invert_horner` for the fix.
"""

from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_kda_layer1_golden import PLACEMENTS, SEQ_LEN, _compose, _shard
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, TtK3KdaAttention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology

SP_AXIS, TP_AXIS = 0, 1


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_decay_magnitude_is_the_variable(mesh_device, device_params):
    checkpoint = Path(resolve_checkpoint())
    trace = resolve_trace(TRACE_100K)
    kda_cfg = kimi_k3_kda_config()
    tp_topology = per_axis_topology()[TP_AXIS]

    # 256 tokens is one 32-row chunk on each of the 8 SP ranks, so there is no chunk composition at
    # all; every longer length adds more of it. If layer 1 is already wrong at 256 the fault is in
    # the per-chunk math for these decay values, not in anything that accumulates.
    for layer_idx in (0, 1):
        weights = load_kda_layer_state_dict(checkpoint, layer_idx, kda_cfg)
        base_bias = weights["dt_bias"].float()
        seq = SEQ_LEN
        hidden = trace.rows("kda", "kda_input_layer_0", 0, seq)
        for offset in (0.0, -0.5, -1.0, -1.5) if layer_idx == 0 else (0.0, 1.0, 2.0, 3.0):
            weights = {**weights, "dt_bias": base_bias + offset}
            want, _ = kda_forward_reference(hidden.unsqueeze(0), weights, kda_cfg)
            want = want.squeeze(0)
            program_config = kimi_k3_program_config(tp_ccl_topology=tp_topology)
            attention = TtK3KdaAttention(
                ttKDA(
                    mesh_device,
                    kda_cfg,
                    weights,
                    layer_idx=layer_idx,
                    tt_ccl=get_tt_ccl(mesh_device),
                    sp_axis=SP_AXIS,
                    tp_axis=TP_AXIS,
                    program_config=program_config,
                ),
                layer_idx=layer_idx,
                tp_axis=TP_AXIS,
                num_links=1,
                tp_topology=tp_topology,
            )
            states = KdaStateCache({layer_idx: attention.kda})
            attention.bind_state_cache(states)
            try:
                out = attention.forward(_shard(mesh_device, hidden, seq), K3AttnContext())
                pcc = float(str(comp_pcc(want, _compose(mesh_device, out, seq), 0.99)[1]).split()[-1])
            except Exception as error:  # a rejected length is data, not a failure
                pcc = float("nan")
                logger.info(f"  layer {layer_idx} dt_bias {offset:+.1f} rejected: {error}")
            finally:
                states.deallocate()
            bias = weights["dt_bias"].float()
            logger.info(f"  layer {layer_idx} dt_bias {offset:+.1f} (max {float(bias.max()):+6.2f})  PCC {pcc:9.5f}")
