"""Direct old-code-vs-new-code numerical comparison for the streaming=False (default,
non-streaming, already-shipped) path -- answers "which test asserts streaming=False is
bit-exact to the pre-change behavior" precisely: NONE of the 152 do a literal old-vs-new
diff (they check TT-vs-torch-reference PCC>=0.99, same threshold before and after, which
is strong but not bit-exact evidence). This script IS that direct comparison.

Usage: run once BEFORE `git stash push -- .../tt/flow/encoder.py` (saves output to
/tmp/streaming_false_check_new.pt), then again AFTER the stash (loads that file, compares
against the OLD code's output for the identical input/seed).

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 120 /opt/venv/bin/python streaming_false_old_vs_new_check.py
"""
import os
import sys

import torch
import ttnn

sys.path.insert(0, "/home/user/tt-metal")

from models.demos.audio.cosyvoice2.tt.flow.encoder import TtUpsampleConformerEncoder, UpsampleConformerEncoderRef

SAVE_PATH = "/tmp/streaming_false_check_new.pt"

torch.manual_seed(0)
enc = UpsampleConformerEncoderRef()
enc.eval()
b, t_len, d = 1, 40, 512
x = torch.randn(b, t_len, d) * 0.1

device = ttnn.CreateDevice(0, l1_small_size=32768)
try:
    tt_enc = TtUpsampleConformerEncoder(device, enc)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt_enc(x_dev, t_len, 1)).float()  # default call, exactly what every existing caller uses
finally:
    ttnn.CloseDevice(device)

if os.path.exists(SAVE_PATH):
    prev = torch.load(SAVE_PATH)
    max_diff = float((out - prev).abs().max())
    rel = max_diff / float(prev.abs().max().clamp_min(1e-8))
    identical = torch.equal(out, prev)
    print(f"NEW output loaded from {SAVE_PATH}, comparing against THIS run's output.")
    print(f"torch.equal (bit-exact): {identical}")
    print(f"max|diff|: {max_diff:.8f}   relative to max|prev|: {rel:.2e}")
    torch.save(out, SAVE_PATH.replace(".pt", "_latest.pt"))
else:
    torch.save(out, SAVE_PATH)
    print(f"Saved this run's output to {SAVE_PATH}. Now: git stash push -- .../tt/flow/encoder.py, then rerun this script.")
