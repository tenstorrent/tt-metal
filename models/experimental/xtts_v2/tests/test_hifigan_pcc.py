# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for the TTNN HiFi-GAN generator (Block 4) vs the coqui golden wav.

Block boundary: generator input z [1,1024,L] + d-vector g [1,512,1] -> waveform [1,1,L*256].
The generator carries the 1D signal as NHWC [1,1,L,C]; goldens are fed as-is.

Per-stage oracles (golden/hifigan/dbg_*.pt), captured from the CPU reference which matches
coqui at PCC 1.0 at every stage, are printed to localize any divergence.
"""
import os

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import (
    TTNNHifiganGenerator,
    preprocess_hifigan_parameters,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "golden", "hifigan")
TARGET_PCC = 0.99

# ttnn intermediate is NHWC [1,1,L,C]; reference dbg is [C,L] or [1,C,L].
_DBG = {
    "conv_pre": "dbg_conv_pre.pt",
    "ups0": "dbg_ups0.pt",
    "ups1": "dbg_ups1.pt",
    "ups2": "dbg_ups2.pt",
    "ups3": "dbg_ups3.pt",
}


def _nhwc_to_ncl(t):  # host torch [1,1,L,C] -> [1,C,L]
    return t.reshape(t.shape[2], t.shape[3]).permute(1, 0).unsqueeze(0)


def run_hifigan_pcc(device, verbose=True):
    z = torch.load(os.path.join(GOLDEN, "z.pt"))  # [1024, L]
    if z.dim() == 2:
        z = z.unsqueeze(0)  # [1,1024,L]
    g = torch.load(os.path.join(GOLDEN, "g.pt"))  # [1,512,1]
    wav_gold = torch.load(os.path.join(GOLDEN, "wav.pt"))  # [1,1,L*256]

    params = preprocess_hifigan_parameters(device)
    model = TTNNHifiganGenerator(device, params)

    L = z.shape[-1]
    # z [1,1024,L] -> NHWC [1,1,L,1024]. fp32 (the model runs fp32 activations — bf16 tops out
    # at waveform PCC ~0.96 on this oscillatory output; see the module docstring / BUG-3).
    z_nhwc = z.permute(0, 2, 1).reshape(1, 1, L, 1024)
    z_tt = ttnn.from_torch(z_nhwc, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    g_tt = ttnn.from_torch(g.reshape(1, 512), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    wav_tt, inter = model(z_tt, g_tt, return_intermediates=True)

    if verbose:
        for key, fname in _DBG.items():
            ref = torch.load(os.path.join(GOLDEN, fname)).to(torch.float32)
            if ref.dim() == 2:
                ref = ref.unsqueeze(0)  # [1,C,L]
            got = _nhwc_to_ncl(inter[key])  # already host torch [1,1,L,C]
            _, msg = comp_pcc(ref, got, pcc=0.0)
            print(f"  [{key:9s}] ttnn {tuple(got.shape)} vs ref {tuple(ref.shape)}  {msg}")

    wav = ttnn.to_torch(wav_tt).to(torch.float32).reshape(1, 1, -1)
    passed, msg = comp_pcc(wav_gold, wav, pcc=TARGET_PCC)
    print(f"waveform {tuple(wav.shape)} vs coqui golden {tuple(wav_gold.shape)}  pcc: {msg}")
    return passed, msg


# ttnn.conv sliding-window/halo config lives in L1_SMALL — device must be opened with a
# non-zero l1_small_size (BUG-2). The fp32 conv config tensor needs more than bf16's 32768,
# so use 65536 (BUG-3).
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_hifigan_pcc(device):
    passed, msg = run_hifigan_pcc(device)
    assert passed, f"HiFi-GAN generator PCC below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys
    import time

    dev = None
    for attempt in range(20):
        try:
            dev = ttnn.open_device(device_id=0, l1_small_size=65536)
            break
        except Exception as e:  # device momentarily busy (shared with main session)
            print(f"open_device attempt {attempt} failed ({e}); retrying in 45s")
            time.sleep(45)
    if dev is None:
        print("FAILED could not open device")
        sys.exit(1)
    try:
        dev.enable_program_cache()
        ok, msg = run_hifigan_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
