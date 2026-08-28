# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TtRoutedExpert with the hybrid routed-expert dispatch, graded against torch.

The op-level test proves the two active-token bands tile the experts exactly once.
This one proves the MODULE wires them up: TtRoutedExpert(hybrid_token_threshold=T)
must produce the same numbers as TtRoutedExpert(hybrid_token_threshold=None), and
both must match TorchExpert per expert.

The expert loads straddle T in both directions and include zero-count experts, so a
threshold that leaked would leave an expert holding another op's numbers or the raw
allocation. `None` is checked in the same run because it is the production path and
the hybrid must not perturb it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc

EMB = 7168
HIDDEN = 2048
MAX_TOKENS = 512
WEIGHT_SCALE = 0.02
PCC = 0.97

# Straddles the threshold both ways, with zero-count experts between live ones and a
# ragged tail on each side so neither band gets a uniformly full schedule.
EXPERT_TOKENS = (96, 512, 0, 300, 256, 0, 480, 128)
THRESHOLD = 256


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=True,
)
@pytest.mark.skipif(not is_blackhole(), reason="the hybrid dispatch is Blackhole-only")
def test_tt_routed_expert_hybrid(mesh_device):
    device = mesh_device
    torch.manual_seed(20260827)
    experts = len(EXPERT_TOKENS)

    weights = [
        {
            "gate_proj": torch.randn(HIDDEN, EMB, dtype=torch.float32) * WEIGHT_SCALE,
            "up_proj": torch.randn(HIDDEN, EMB, dtype=torch.float32) * WEIGHT_SCALE,
            "down_proj": torch.randn(EMB, HIDDEN, dtype=torch.float32) * WEIGHT_SCALE,
        }
        for _ in range(experts)
    ]

    # One shared dispatch buffer: expert e owns rows [e*MAX_TOKENS, e*MAX_TOKENS + count).
    total_rows = MAX_TOKENS * experts
    torch_input = torch.zeros(total_rows, EMB, dtype=torch.float32)
    for e, count in enumerate(EXPERT_TOKENS):
        if count:
            base = e * MAX_TOKENS
            torch_input[base : base + count] = torch.randn(count, EMB, dtype=torch.float32)

    references = {}
    with torch.no_grad():
        for e, count in enumerate(EXPERT_TOKENS):
            if not count:
                continue
            base = e * MAX_TOKENS
            torch_expert = TorchExpert(EMB, HIDDEN, weights[e], activation=ACTIVATION_SILU)
            references[e] = torch_expert(torch_input[base : base + count])

    def idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
        )

    global_expert_idx_tt = idx_tensor(list(range(experts)))
    counts_tt = idx_tensor(list(EXPERT_TOKENS))
    offsets_tt = idx_tensor([e * MAX_TOKENS for e in range(experts)])

    def run(threshold):
        tt_expert = TtRoutedExpert(
            mesh_device=device,
            experts_per_chip=experts,
            global_expert_idx_table=global_expert_idx_tt,
            emb_dim=EMB,
            hidden_dim=HIDDEN,
            max_tokens=MAX_TOKENS,
            torch_weights=weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat4_b,
            activation=ttnn.RoutedExpertActivation.Silu,
            hybrid_token_threshold=threshold,
        )
        tt_input = ttnn.from_torch(
            torch_input,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )
        out = tt_expert(tt_input, counts_tt, offsets_tt)
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)).float()

    single = run(None)  # composite only -- the production path
    hybrid = run(THRESHOLD)  # both ops, bands split at THRESHOLD
    # A threshold at or above the region capacity leaves the composite nothing to do, so
    # TtRoutedExpert drops that dispatch and the fused op owns every expert.
    fused_only = run(MAX_TOKENS)

    failures = []
    for e, count in enumerate(EXPERT_TOKENS):
        if not count:
            continue
        base = e * MAX_TOKENS
        owner = "fused" if count <= THRESHOLD else "unified"
        rows = hybrid[base : base + count]
        if not torch.isfinite(rows).all():
            failures.append(f"expert {e} (count {count}, {owner}) non-finite")
            continue
        ok_torch, msg_torch = comp_pcc(references[e], rows, PCC)
        if not ok_torch:
            failures.append(f"expert {e} (count {count}, {owner}) vs torch: {msg_torch}")
        # The single-op path is production; the hybrid must not move the numbers.
        ok_single, msg_single = comp_pcc(single[base : base + count], rows, 0.999)
        if not ok_single:
            failures.append(f"expert {e} (count {count}, {owner}) vs single-op: {msg_single}")
        fused_rows = fused_only[base : base + count]
        if not torch.isfinite(fused_rows).all():
            failures.append(f"expert {e} (count {count}) non-finite in fused-only mode")
        else:
            ok_fused, msg_fused = comp_pcc(references[e], fused_rows, PCC)
            if not ok_fused:
                failures.append(f"expert {e} (count {count}) fused-only vs torch: {msg_fused}")
        logger.info(f"expert {e}: count={count} owner={owner} torch={msg_torch}")
    assert not failures, "hybrid TtRoutedExpert disagreed: " + "; ".join(failures)
