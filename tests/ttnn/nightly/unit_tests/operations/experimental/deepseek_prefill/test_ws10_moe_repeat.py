# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DS41F-0169 repro (tt-blaze postmortem): unified_routed_expert_moe re-run on byte-identical inputs.

One chip, V4.1-Flash-1017's routed-expert geometry (dim 5120, moe_inter_dim 2304, 12 experts a chip, bf8 weights and
activations, LoFi + packer_l1_acc, ROW_MAJOR bf16 dispatch buffer, tile-padded expert regions), per-expert counts of
the chip that flipped in the prefill (layer 17, chunk 26, chip 8). Runs the op WS10_REPS times on the same device
inputs and compares every run's valid rows with the first run's.

    WS10_REPS=200 WS10_SEED=0 WS10_COUNTS=610,20,59,... pytest -s test_ws10_moe_repeat.py
"""

import hashlib
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

EMB, HID, E = 5120, 2304, 12
MAX_TOKENS = 8 * 512  # dispatch_group_size x seq_len_per_chip (compute_constants)
BUF_ROWS = MAX_TOKENS * 2 + ttnn.TILE_SIZE * (E - 1)
COUNTS_DEFAULT = "610,20,59,83,49,0,40,67,13,4,13,3"


def _u32(dev, values):
    return ttnn.from_torch(
        torch.tensor([values], dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=["mesh_device", "device_params"],
)
def test_ws10_moe_repeat(mesh_device):
    reps = int(os.environ.get("WS10_REPS", "100"))
    seed = int(os.environ.get("WS10_SEED", "0"))
    counts = [int(c) for c in os.environ.get("WS10_COUNTS", COUNTS_DEFAULT).split(",")]
    assert len(counts) == E
    starts, s = [], 0
    for c in counts:
        starts.append(s)
        s += -(-c // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    valid = torch.zeros(BUF_ROWS, dtype=torch.bool)
    for st, c in zip(starts, counts):
        valid[st : st + c] = True
    rows_used = s

    torch.manual_seed(seed)
    scale = float(os.environ.get("WS10_WSCALE", "0.02"))
    weights = [
        {
            "gate_proj": torch.randn(HID, EMB) * scale,
            "up_proj": torch.randn(HID, EMB) * scale,
            "down_proj": torch.randn(EMB, HID) * scale,
        }
        for _ in range(E)
    ]
    x = torch.zeros(BUF_ROWS, EMB)
    x[valid] = torch.randn(int(valid.sum()), EMB)

    n_routed = 384
    cnt = [0] * n_routed
    off = [0] * n_routed
    for e in range(E):
        cnt[e], off[e] = counts[e], starts[e]
    dev = mesh_device
    act = getattr(ttnn.RoutedExpertActivation, "SiluClamped", ttnn.RoutedExpertActivation.Silu)
    expert = TtRoutedExpert(
        mesh_device=dev,
        experts_per_chip=E,
        global_expert_idx_table=_u32(dev, list(range(E))),
        emb_dim=EMB,
        hidden_dim=HID,
        max_tokens=MAX_TOKENS,
        torch_weights=weights,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        activation=act,
        weights_dram_nd_sharded=False,
    )
    tt_x = ttnn.from_torch(
        x, mesh_mapper=ttnn.ReplicateTensorToMesh(dev), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16
    )
    tt_cnt, tt_off = _u32(dev, cnt), _u32(dev, off)

    def run():
        out = expert(tt_x, tt_cnt, tt_off)
        sl = ttnn.slice(out, [0, 0], [rows_used, EMB])
        t = ttnn.to_torch(sl, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0)).float()
        ttnn.deallocate(sl)
        ttnn.deallocate(out)
        return t[:rows_used]

    vmask = valid[:rows_used]
    ref = run()
    alts, ndiff = {}, 0
    for r in range(reps):
        t = run()
        d = (t != ref) & vmask[:, None]
        if d.any():
            ndiff += 1
            idx = d.nonzero()
            key = hashlib.md5(t[d].numpy().tobytes() + idx.numpy().tobytes()).hexdigest()[:8]
            if key not in alts:
                rr = sorted(set(idx[:, 0].tolist()))
                cc = idx[:, 1]
                exp = [next(e for e in range(E) if starts[e] <= x_ < starts[e] + max(counts[e], 1)) for x_ in rr[:8]]
                logger.warning(
                    f"WS10 rep {r}: NEW alternate {key}: {int(d.sum())} values, rows {rr[:16]} (experts "
                    f"{exp}), cols {int(cc.min())}..{int(cc.max())}, max |d| "
                    f"{float((t - ref)[d].abs().max()):.6f}"
                )
                alts[key] = 0
            alts[key] += 1
    logger.warning(
        f"WS10 RESULT seed {seed} counts {counts}: {ndiff} of {reps} re-runs differ from the first; "
        f"alternates {alts}"
    )
