# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for the TTNN ResNet speaker encoder (Block 2) vs the coqui golden.

Block boundary: log-mel `logmel` [1,64,T] (mel front-end on CPU) -> d-vector [1,512,1].

Reference intermediates are computed from the golden logmel via the CPU core (no torchaudio),
which itself matches coqui at PCC 1.0, so this test needs only the golden logmel +
speaker_embedding tensors.
"""
import os

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.reference.xtts_speaker_ref import SpeakerReference
from models.experimental.xtts_v2.tt.ttnn_xtts_speaker import (
    TTNNSpeakerEncoder,
    preprocess_speaker_parameters,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "golden", "speaker")
TARGET_PCC = 0.99

# how to bring each ttnn intermediate into the reference's layout for PCC
#   nhwc: ttnn [1,H,W,C] -> compare against ref NCHW [1,C,H,W]
#   tf:   ttnn [1,T,F]   -> compare against ref [1,F,T]
#   same: identical layout
_ALIGN = {
    "instancenorm": "nhwc",
    "conv1": "nhwc",
    "layer1": "nhwc",
    "layer2": "nhwc",
    "layer3": "nhwc",
    "layer4": "nhwc",
    "reshape": "tf",
    "attn_w": "same",
    "pool": "same",
    "fc": "same",
    "emb": "same",
}


def _to_ref_layout(t, mode):
    if mode == "nhwc":
        return t.permute(0, 3, 1, 2).contiguous()
    if mode == "tf":
        return t.permute(0, 2, 1).contiguous()
    return t


def run_speaker_pcc(device, verbose=True):
    logmel = torch.load(os.path.join(GOLDEN, "logmel.pt"))  # [1,64,T]
    emb_gold = torch.load(os.path.join(GOLDEN, "speaker_embedding.pt"))  # [1,512,1]

    ref = SpeakerReference()
    _, ref_inter = ref.core(logmel, l2_norm=True, return_intermediates=True)

    params = preprocess_speaker_parameters(device)
    model = TTNNSpeakerEncoder(device, params)

    logmel_tt = ttnn.from_torch(logmel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    emb_tt, inter = model(logmel_tt, return_intermediates=True)

    if verbose:
        for key, mode in _ALIGN.items():
            if key not in inter or key not in ref_inter:
                continue
            got = _to_ref_layout(ttnn.to_torch(inter[key]).to(torch.float32), mode)
            want = ref_inter[key].to(torch.float32)
            _, msg = comp_pcc(want, got, pcc=0.0)
            print(f"  [{key:12s}] ttnn {tuple(got.shape)} vs ref {tuple(want.shape)}  {msg}")

    emb = ttnn.to_torch(emb_tt).to(torch.float32).reshape(1, 512, 1)
    passed, msg = comp_pcc(emb_gold, emb, pcc=TARGET_PCC)
    print(f"speaker_embedding {tuple(emb.shape)} vs coqui golden {tuple(emb_gold.shape)}  pcc: {msg}")
    return passed, msg


# ttnn.conv2d's halo/sliding-window config tensor lives in L1_SMALL — the device fixture
# must be opened with a non-zero l1_small_size (see CLAUDE_XTTS_BUGS.md BUG-2).
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_speaker_pcc(device):
    passed, msg = run_speaker_pcc(device)
    assert passed, f"speaker encoder PCC below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys
    import time

    dev = None
    for attempt in range(20):
        try:
            dev = ttnn.open_device(device_id=0, l1_small_size=32768)
            break
        except Exception as e:  # device momentarily busy (shared with main session)
            print(f"open_device attempt {attempt} failed ({e}); retrying in 45s")
            time.sleep(45)
    if dev is None:
        print("FAILED could not open device")
        sys.exit(1)
    try:
        dev.enable_program_cache()
        ok, msg = run_speaker_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
