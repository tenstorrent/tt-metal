# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Targeted experiments. Each non-baseline changes one mechanism.

No environment-driven hash mutation: configure the emulator/host mapping before
loading buffers. The kernel snapshots the actual mapping into each result.
"""

from helpers.format_config import DataFormat
from quasar.test_dest_write_path_quasar import DestWriteKernel as Kernel
from quasar.test_dest_write_path_quasar import SrcsIssueOpt as Opt
from quasar.test_dest_write_path_quasar import (
    run_dest_write_path_eltwise,
)

# Shared controls are executed once, not copied into each sweep.
CASES = [
    ("baseline", {}),
    ("startup", {"num_blocks": 2, "loop_factor": 1}),
    ("producer_gaps", {"producer_gap": 512}),
    ("consumer_backpressure", {"consumer_gap": 512}),
    ("sparse_input", {"input_stride": 3}),
    ("sfpu_light", {"kernel": Kernel.Chain, "math_depth": 1}),
    ("sfpu_heavy", {"kernel": Kernel.Chain, "math_depth": 16}),
    ("exp_same_math", {"kernel": Kernel.Exp}),
    ("stamp_overhead", {"stamp_every": 0}),
    ("stamp_sparse", {"stamp_every": 4}),
    ("writer_proxy", {"writer_only": True}),
    ("partial_bank", {"tiles_per_bank": 1}),
]


def run_case(name, options, path, *, is_perf=False, perf_report=None):
    options = dict(options)
    kernel = options.pop("kernel", Kernel.Copy)
    stride = options.get("input_stride", 1)
    # Address-mode optimization is bit-exact copy only. Arithmetic gets bank
    # unpacking, which is legal with the existing slice arithmetic helper.
    opt = (
        (Opt.BOTH if kernel == Kernel.Copy else Opt.BANK_UNPACK)
        if path.srcs_writer
        else Opt.BASE
    )
    if stride != 1:
        opt = Opt.ADDR_MOD if path.srcs_writer else Opt.BASE
    if options.get("pack1"):
        opt = Opt.BASE
    run_dest_write_path_eltwise(
        path=path,
        tiles_per_bank=options.pop("tiles_per_bank", 4),
        data_format=DataFormat.Float32,
        kernel=kernel,
        srcs_opt=opt,
        loop_factor=options.pop("loop_factor", 4),
        experimental=True,
        is_perf=is_perf,
        perf_report=perf_report,
        **options,
    )
