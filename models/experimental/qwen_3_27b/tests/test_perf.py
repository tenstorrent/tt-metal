# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device perf workload. Not a pass/fail test -- it exists to be run under Tracy,
which records every device op into a CSV. summarize_perf.py turns that into a
per-op table.

Each case wraps its measured forwards in signpost("<name>_START"/"_END") so the
summarizer can drop weight loading and compilation, which are one-time and would
otherwise dominate. The first forward is a warmup: it compiles kernels.

    python3 -m tracy -p -r -o /tmp/prof -a device_kernel_duration \\
        -m 'pytest models/experimental/qwen_3_27b/tests/test_perf.py -k deltanet'
    python models/experimental/qwen_3_27b/tests/summarize_perf.py /tmp/prof
"""

import pytest
import torch
from tracy import signpost

import ttnn
from models.experimental.qwen_3_27b.tests.test_gated_deltanet import build_reference, to_device
from models.experimental.qwen_3_27b.tt.tt_embedding import TtQwen36Embedding
from models.experimental.qwen_3_27b.tt.tt_gated_deltanet import D
from models.experimental.qwen_3_27b.tt.tt_rms_norm import TtQwen36RmsNorm

VOCAB_SIZE = 248320
ITERS = 4  # measured forwards, after one warmup


# Stages to signpost individually inside a deltanet forward. _causal_conv1d nests
# inside _project, so its region is reported as a child of it.
STAGES = ["_project", "_causal_conv1d", "_gates", "_l2norm_qk", "_delta_rule_recurrent", "_gated_norm"]


def _measure(name, forward, reset=None, warmup=True):
    """Run ITERS forwards, each in its own signposted region."""
    if warmup:
        ttnn.deallocate(forward())  # compiles kernels, not measured
        if reset:
            reset()

    for _ in range(ITERS):
        signpost(f"{name}_START")
        out = forward()
        signpost(f"{name}_END")  # reset/dealloc stay outside the measured region
        ttnn.deallocate(out)
        if reset:
            reset()


def _instrument(model, tag):
    """Wrap each stage method so every call emits its own nested signpost region."""

    def wrap(fn, region):
        def inner(*args, **kwargs):
            signpost(f"{region}_START")
            result = fn(*args, **kwargs)
            signpost(f"{region}_END")
            return result

        return inner

    for stage in STAGES:
        setattr(model, stage, wrap(getattr(model, stage), f"{tag}{stage}"))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 32, 128], ids=["t1", "t32", "t128"])
def test_perf_deltanet(mesh_device, seq_len, reset_seeds):
    model = to_device(mesh_device, build_reference())
    x = ttnn.from_torch(
        torch.randn(1, seq_len, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    tag = f"deltanet_t{seq_len}"
    # Warm up BEFORE instrumenting: a warmup call under the stage wrappers would
    # emit their regions outside the root, with compile-time durations in them.
    ttnn.deallocate(model(x))
    model.reset_state()

    _instrument(model, tag)
    _measure(tag, lambda: model(x), reset=model.reset_state, warmup=False)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 32, 128], ids=["t1", "t32", "t128"])
def test_perf_embedding(mesh_device, seq_len, reset_seeds):
    model = TtQwen36Embedding(mesh_device, torch.randn(VOCAB_SIZE, D, dtype=torch.bfloat16))
    token_ids = ttnn.from_torch(
        torch.randint(0, VOCAB_SIZE, (1, seq_len), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    _measure(f"embedding_t{seq_len}", lambda: model(token_ids))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 32, 128], ids=["t1", "t32", "t128"])
def test_perf_rms_norm(mesh_device, seq_len, reset_seeds):
    model = TtQwen36RmsNorm(mesh_device, (0.1 * torch.randn(D)).to(torch.bfloat16))
    x = ttnn.from_torch(
        torch.randn(1, seq_len, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    _measure(f"rms_norm_t{seq_len}", lambda: model(x))
