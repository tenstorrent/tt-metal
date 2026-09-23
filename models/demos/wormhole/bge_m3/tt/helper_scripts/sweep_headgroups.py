"""In-model sweep of the head-split and concat head_groups at B8/S512.

The GenericOp that splits and concatenates heads costs 1871 us at B8, 10.2% of
the wall. B8 inherited head_groups=4 for the split and 16 for the concat from
the B16/B32 line; neither was swept at B8.

head_groups must divide num_heads=16, so the legal values are 1, 2, 4, 8, 16.
Work units are batch * seq_tiles * head_groups, and the p150 offers 130 cores.

Each candidate is patched in, then the traced wall time is measured by the same
path the perf test uses. An isolated microbench is not used here: the earlier
attempt built its tensors with the wrong memory config and the result did not
hold, and three later items showed isolated gains vanishing in the model.
"""

import argparse
import time

import torch

import ttnn

REPEATS = 8


def wall_ms(model, args, tokens, mask):
    trace = model.capture_trace(tokens, attention_mask=mask)
    best = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        model.execute_trace(trace, blocking=True)
        elapsed = (time.perf_counter() - start) * 1000
        best = elapsed if best is None else min(best, elapsed)
    model.release_trace(trace)
    return best


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=8)
args_cli = parser.parse_args()

from models.demos.wormhole.bge_m3.tt import attention as attn_mod
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

device = ttnn.open_device(device_id=0)
args, model, _ = create_tt_model(
    mesh_device=device, max_batch_size=args_cli.batch, max_seq_len=512, dtype=ttnn.bfloat8_b
)
torch.manual_seed(0)
tokens = torch.randint(0, 1000, (args_cli.batch, 512))
mask = torch.ones((args_cli.batch, 512), dtype=torch.int32)

print("  split concat     ms     vs base")
base = None
for split_groups in (4, 8, 16):
    for concat_groups in (4, 8, 16):
        attn_mod._SWEEP_SPLIT_GROUPS = split_groups
        attn_mod._SWEEP_CONCAT_GROUPS = concat_groups
        try:
            ms = wall_ms(model, args, tokens, mask)
        except Exception as exc:
            print("  %5d %6d   FAILED  %s" % (split_groups, concat_groups, str(exc)[:40]))
            continue
        if base is None:
            base = ms
        print("  %5d %6d %6.3f   %+6.3f" % (split_groups, concat_groups, ms, ms - base))

ttnn.close_device(device)
