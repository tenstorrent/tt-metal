# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-ablation matrix for --profile runs: which of read / compute / write binds the device time.

Each variant compiles the kernels with LAYER_NORM_ABLATE_{READS,COMPUTE,WRITES} defines, which drop that
stage's payload but keep every CB handshake. Outputs are garbage for ablated variants; only timing is read.

Interpretation (per shape):
  full ~= max(reads_only, compute_only, writes_only)  -> stages overlap; the max is the binding stage
  full ~= reads_only + compute_only                    -> read and compute serialize

Dispatch order is appended to $LAYER_NORM_ABLATION_LOG (one case id per dispatch) so the profiler CSV,
which has one row per dispatch in the same order, maps back to cases.
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm

WIDTH = 768
RUNS_PER_CASE = 2
SHAPES = [
    (32, 1, 32, WIDTH),  # 32 tile-rows: 1 row per core
    (8, 1, 512, WIDTH),  # 128 tile-rows: 1-2 rows per core
    (256, 1, 16, WIDTH),  # 256 tile-rows: 2-3 rows per core
]
STAGES = ("READS", "COMPUTE", "WRITES")
VARIANTS = {
    "full": (),
    "no_reads": ("READS",),
    "no_compute": ("COMPUTE",),
    "no_writes": ("WRITES",),
    "reads_only": ("COMPUTE", "WRITES"),
    "compute_only": ("READS", "WRITES"),
    "writes_only": ("READS", "COMPUTE"),
    "floor": STAGES,
}


@pytest.fixture
def ablate(monkeypatch, device):
    # The generic-op program cache key ignores kernel defines, so a cached program from another variant
    # would silently run instead. Clear it per case.
    device.clear_program_cache()

    def apply(stages):
        # Patch the namespace the called function resolves names in: the op module is importable under two
        # package names (two module objects), so patching an imported module object can miss.
        op_globals = layer_norm.__globals__
        original = op_globals["create_program_descriptor"]

        def patched(*args, **kwargs):
            descriptor = original(*args, **kwargs)
            for kernel in descriptor.kernels:
                kernel.defines = list(kernel.defines) + [(f"LAYER_NORM_ABLATE_{stage}", "1") for stage in stages]
            return descriptor

        monkeypatch.setitem(op_globals, "create_program_descriptor", patched)

    yield apply
    device.clear_program_cache()


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("with_residual", [False, True], ids=["no_residual", "residual"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_layer_norm_tile_ablation(device, ablate, request, shape, with_residual, variant):
    ablate(VARIANTS[variant])
    torch.manual_seed(0)

    def to_device(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    x = to_device(torch.randn(shape).to(torch.bfloat16))
    residual = to_device(torch.randn(shape).to(torch.bfloat16)) if with_residual else None
    affine = ttnn.from_torch(
        torch.randn(WIDTH).to(torch.bfloat16).reshape(1, 1, WIDTH // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    log_path = os.environ.get("LAYER_NORM_ABLATION_LOG")
    for _ in range(RUNS_PER_CASE):
        out = layer_norm(x, affine, affine, residual)
        assert tuple(out.shape) == shape
        ttnn.deallocate(out)
        if log_path:
            with open(log_path, "a") as log:
                log.write(request.node.callspec.id + "\n")
