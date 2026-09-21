"""Phase B: capture REAL target hidden taps [5,19,33,47,61] from a real-prompt prefill.

Wraps the 5 tap layers to grab their residual output, prefills a real prompt, gathers each tap to
full hidden (5120), dumps target_hidden_cat (1,S,25600) + prompt + anchor. Feeds the real-input
drafter validation. Run as a pytest test so the mesh_device fixture sets up fabric.

Run:  MESH_DEVICE=P150x4 pytest tests/dflash2_capture_real.py -v -s
"""
import os

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

TAPS = [5, 19, 33, 47, 61]
OUT = os.environ.get("DFLASH_REAL", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_real.npz")
PROMPT = (
    "Explain, step by step, how a transformer neural network processes a sequence of tokens to predict the next one."
)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "num_command_queues": 2,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 1024 * 1024 * 1024,
        }
    ],
    indirect=True,
)
def test_capture_real_taps(mesh_device):
    md = mesh_device
    md.enable_program_cache()
    model = Qwen36Model.from_pretrained(md, max_batch_size=1, max_seq_len=4096)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    ids = tok(PROMPT * 8, return_tensors="pt").input_ids.to(torch.int32)[:, :128]  # one full 128-chunk
    S = ids.shape[1]
    print(f"[capture] prompt {S} tokens")

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

    taps = []
    for i in TAPS:
        t = ttnn.to_torch(cap[i], mesh_composer=ttnn.ConcatMeshToTensor(md, dim=-1)).float()
        print(f"[capture] tap L{i} raw shape {tuple(t.shape)}")
        taps.append(t.reshape(1, -1, 5120)[:, :S, :])
    target_hidden_cat = torch.cat(taps, dim=-1)  # (1,S,25600)
    anchor = int(ids[0, -1])
    np.savez(OUT, target_hidden_cat=target_hidden_cat.numpy(), ids=ids.numpy(), S=np.int64(S), anchor=np.int64(anchor))
    print(f"[capture] wrote {OUT}: target_hidden_cat {tuple(target_hidden_cat.shape)}, anchor={anchor}")
    assert target_hidden_cat.shape == (1, S, 25600)
