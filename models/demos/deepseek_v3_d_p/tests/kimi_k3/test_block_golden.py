# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 layer 0 on device against the captured vLLM trace.

Layer 0 is where the trace and the block can be compared without any of the model that is not
built yet. It is dense (``first_k_dense_replace=1``) and KDA, so no MXFP4 dequantizer is needed,
and its attention input is a plain RMSNorm of the residual stream: the attn-res read that
replaces the residual add elsewhere in K3 sees an empty snapshot history at layer 0.

The comparison stops at the KDA output. Layer 0's *pre-MLP* attn-res read is not empty, so the
residual stream diverges from the trace from the FFN input onward until the walk lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    compare_cpu_device,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.harness import (
    SP_AXIS,
    TP_AXIS,
    build_layer_0,
    kda_sequence_length,
    shard_activation,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.trace import KimiK3Trace
from models.demos.deepseek_v3_d_p.tt.kda.state_store import KdaStateStore

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
    pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True),
]

# The RMSNorm is elementwise on a BF16 residual, so it is held to the tighter bound; the
# recurrence and its carry get the threshold the rest of the KDA device tests use.
NORM_PCC = 0.999
KDA_PCC = 0.98


def test_kimi_k3_layer_0_kda_matches_vllm_trace(
    mesh_device: ttnn.MeshDevice,
    kimi_k3_checkpoint_dir: Path,
    kimi_k3_trace: KimiK3Trace,
    kimi_k3_tt_cache_root: Path,
) -> None:
    """PCC the block's KDA path against the trace, from residual stream to recurrent carry.

    Drives ``attn_norm`` and ``_kda_path`` rather than ``forward`` so the comparison ends where
    the trace stops being comparable, but that is still the whole integration under test: the
    tensor-parallel gather into KDA's full-width input projection, the carry the store hands over
    and takes back, and the reduce-scattered residual-shaped result.
    """
    trace = kimi_k3_trace
    assert trace.metadata["kda_layer"] == 0, "the trace captures KDA on a layer this test does not build"

    seq_len = kda_sequence_length(mesh_device)
    state_every = int(trace.metadata["kda_state_every"])
    assert (
        seq_len % state_every == 0
    ), f"the trace snapshots the carry every {state_every} tokens, which does not line up with T={seq_len}"

    hidden = trace.rows("decoder_input_layer_0", seq_len).reshape(1, 1, seq_len, KimiK3Config.EMB_SIZE)
    block, _ = build_layer_0(mesh_device, kimi_k3_checkpoint_dir, kimi_k3_tt_cache_root, seq_len)
    assert block.is_kda, "layer 0 must be a KDA layer"

    kda_states = KdaStateStore({0: block.kda})
    attn_norm_out = block.attn_norm(shard_activation(mesh_device, hidden))
    actual_norm = reconstruct_sp_tp_tensor(attn_norm_out, mesh_device, SP_AXIS, TP_AXIS, tp_dim=2, sp_dim=1)
    attn_out = block._kda_path(attn_norm_out, kda_states)
    actual_attn = reconstruct_sp_tp_tensor(attn_out, mesh_device, SP_AXIS, TP_AXIS, tp_dim=2, sp_dim=1)
    actual_recurrent = {
        sp_rank: reconstruct_state_at_sp_rank(kda_states.get(0).recurrent, mesh_device, SP_AXIS, TP_AXIS, sp_rank)
        for sp_rank in range(tuple(mesh_device.shape)[SP_AXIS])
    }

    failures: list[str] = []
    for name, golden, actual, threshold in (
        ("attn_norm output", trace.rows("kda_input_layer_0", seq_len), actual_norm, NORM_PCC),
        ("KDA output", trace.rows("kda_output_layer_0", seq_len), actual_attn, KDA_PCC),
    ):
        _, stream_failures = compare_cpu_device(
            f"layer 0 {name}",
            golden.reshape(1, seq_len, KimiK3Config.EMB_SIZE),
            actual.to(golden.dtype),
            pcc_threshold=threshold,
        )
        failures.extend(stream_failures)

    # The recurrence chains across the sequence-parallel ranks, so every rank ends up holding the
    # same carry: the one the trace snapshots after the last token of the window.
    # The trace stores the carry as [heads, value, key]; ttKDA carries [heads, key, value]. The
    # trace metadata names neither axis, so the orientation is established empirically: an
    # unweighted k^T v proxy correlates with the transposed snapshot and not with the snapshot.
    golden_recurrent = trace.rows("kda_recurrent_state_layer_0", 1, start=seq_len // state_every - 1)
    golden_recurrent = golden_recurrent.transpose(-2, -1)
    for sp_rank, actual in actual_recurrent.items():
        _, state_failures = compare_cpu_device(
            f"layer 0 recurrent carry sp_rank={sp_rank}",
            golden_recurrent,
            actual.to(golden_recurrent.dtype),
            pcc_threshold=KDA_PCC,
        )
        failures.extend(state_failures)

    assert not failures, "\n".join(failures)
