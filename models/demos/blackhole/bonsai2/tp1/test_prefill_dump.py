# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dump per-layer prefill outputs for an A/B comparison (Agent C, SCALE16 prefill debug).

Runs ONE eager model.prefill on a fixed-seed prompt (PF_PROMPT_LEN tokens, default 2642: one 2048 chunk plus a
tail, the traced_4k shapes) and saves, for every decoder layer, the last 32 rows of its output, plus the final
logits, to PF_DUMP (a .pt file). Run it twice with different env (e.g. BONSAI_SCALE16=0 / 1), then compare with
    python -m models.demos.blackhole.bonsai2.tp1.test_prefill_dump <a.pt> <b.pt>
"""
import os
import sys

import pytest
import torch

PROMPT_LEN = int(os.environ.get("PF_PROMPT_LEN", "2642"))


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
def test_prefill_dump(mesh_device):
    import ttnn
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    out = os.environ["PF_DUMP"]
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=72 * 64, enable_mtp=False)
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(1000, 100000, (1, PROMPT_LEN), dtype=torch.long, generator=g)
    dumps = {}

    for i, layer in enumerate(model.layers):
        orig = layer.forward

        def wrapped(x, *a, _orig=orig, _i=i, **kw):
            y = _orig(x, *a, **kw)
            t = ttnn.to_torch(y).float().reshape(-1, y.shape[-1])
            dumps.setdefault(_i, []).append(t[-32:].clone())  # last 32 rows of each chunk this layer processes
            return y

        layer.forward = wrapped
    logits = ttnn.to_torch(model.prefill(tokens)).float().reshape(-1)
    torch.save({"layers": dumps, "logits": logits, "argmax": int(logits.argmax()), "env": dict(os.environ)}, out)
    print(f"PFDUMP saved {out}: {len(dumps)} layers, argmax {int(logits.argmax())}", flush=True)


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


if __name__ == "__main__":
    A, B = torch.load(sys.argv[1]), torch.load(sys.argv[2])
    print(f"argmax {A['argmax']} vs {B['argmax']}; logits pcc {_pcc(A['logits'], B['logits']):.6f}")
    for i in sorted(A["layers"]):
        pc = [_pcc(a, b) for a, b in zip(A["layers"][i], B["layers"][i])]
        print(f"layer {i:2d}: " + " ".join(f"chunk{j} {p:.6f}" for j, p in enumerate(pc)))
