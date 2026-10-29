# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Our ttnn target's DFlash2 taps for the token ids of a resident taps file (dfl_verify real --taps-out), and the
per-layer comparison with the resident verify model's taps.

capture (pytest, opens the mesh): prefills the ids (one chunk) with models/demos/blackhole/qwen36 (Qwen36Model) with
    the five tap layers wrapped, as tests/dflash2_capture_real.py does, and saves the taps [5, S, 5120].
    DFL_TAPS_IN=<resident taps .pt> DFL_TAPS_OUT=<out .pt> pytest qwen36_resident/tests/dfl_ttnn_taps.py -s
compare (host): python -m qwen36_resident.tests.dfl_ttnn_taps <resident .pt> <ttnn .pt>
"""
import os
import sys

import pytest
import torch

import ttnn

TAPS = [5, 19, 33, 47, 61]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "num_command_queues": 2,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 256 * 1024 * 1024,
        }
    ],
    indirect=True,
)
def test_capture_ttnn_taps(mesh_device):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    src = torch.load(os.environ["DFL_TAPS_IN"])
    S = min(128, len(src["ids"]))
    ids = torch.tensor(src["ids"][:S], dtype=torch.int32).reshape(1, S)
    md = mesh_device
    model = Qwen36Model.from_pretrained(md, max_batch_size=1, max_seq_len=4096)
    cap, origs = {}, {}
    for i in TAPS:
        origs[i] = model.layers[i].forward

        def mk(idx, orig):
            def wrapped(*a, **k):
                out = orig(*a, **k)
                cap[idx] = out
                return out

            return wrapped

        model.layers[i].forward = mk(i, origs[i])
    model.prefill_tp(ids)
    for i in TAPS:
        model.layers[i].forward = origs[i]
    taps = [
        ttnn.to_torch(cap[i], mesh_composer=ttnn.ConcatMeshToTensor(md, dim=-1)).float().reshape(-1, 5120)[:S]
        for i in TAPS
    ]
    torch.save(
        dict(ids=ids[0].tolist(), taps=torch.stack(taps).bfloat16(), tap_layers=TAPS), os.environ["DFL_TAPS_OUT"]
    )


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def compare(res_path, ttnn_path):
    r, t = torch.load(res_path), torch.load(ttnn_path)
    S = t["taps"].shape[1]
    assert r["ids"][:S] == t["ids"], "different token ids"
    for k, layer in enumerate(TAPS):
        a, b = r["taps"][k, :S].float(), t["taps"][k].float()
        rows = [pcc(a[p], b[p]) for p in range(S)]
        print(
            f"tap L{layer}: PCC all {pcc(a, b):.5f}, per position min {min(rows):.5f} median "
            f"{sorted(rows)[S // 2]:.5f}, rel {float((a - b).norm() / b.norm()):.4f}"
        )


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
