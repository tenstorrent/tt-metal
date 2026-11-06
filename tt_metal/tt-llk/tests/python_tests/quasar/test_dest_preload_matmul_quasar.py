# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
FPU kernel that needs Dest, on both 32-bit Dest write paths (see test_dest_write_path_quasar.py):
Out = Partial + A x B, with the fp32 partial preloaded into Dest by the path under test (the
"FPU must add on top" case that SrcS cannot otherwise serve). L1-acc is the baseline without a
preload: the packer accumulates A x B onto the partial already in the output buffer.
"""

from enum import Enum

import torch
from helpers.data_format_inference import data_formats
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from quasar.test_dest_write_path_quasar import (
    DEST_WRITE_PATH,
    NUM_BLOCKS,
    TILE_ELEMENTS,
    DestWriteKernel,
    DestWritePath,
    SrcsIssueOpt,
    _stamps_buffer,
    report_bank_stamps,
)

TILE_DIM = 32
MATMUL_CT_SWEEP = [1, 4]
MATMUL_KT_SWEEP = [1, 4]
MATMUL_FIDELITIES = [MathFidelity.LoFi, MathFidelity.HiFi4]


class PartialPath(Enum):
    """How the partial is added: preloaded into Dest along a DestWritePath, or accumulated in L1 by PACK0.
    Duck-types DestWritePath for DEST_WRITE_PATH and the stamp records."""

    A_DRAIN = DestWritePath.A_DRAIN.value
    A_HW = DestWritePath.A_HW.value
    B_DRAIN = DestWritePath.B_DRAIN.value
    B_DONE = DestWritePath.B_DONE.value
    L1_ACC = "L1-acc"

    @property
    def pack_l1_acc(self) -> bool:
        return self == PartialPath.L1_ACC

    @property
    def srcs_writer(self) -> bool:
        return not self.pack_l1_acc and DestWritePath(self.value).srcs_writer

    @property
    def hw_bank_switch(self) -> bool:
        return not self.pack_l1_acc and DestWritePath(self.value).hw_bank_switch


def run_dest_preload_matmul(
    path: PartialPath,
    ct_dim: int,
    kt_dim: int,
    math_fidelity: MathFidelity,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor: int = 1,
    is_perf: bool = False,
    perf_report=None,
    srcs_opt: SrcsIssueOpt = SrcsIssueOpt.BASE,
):
    assert (
        srcs_opt == SrcsIssueOpt.BASE or path.srcs_writer
    ), "SrcsIssueOpt only applies to path B"
    torch.manual_seed(7)
    in_format = DataFormat.Float16_b
    out_format = DataFormat.Float32
    dest_acc = DestAccumulation.Yes

    dims_A = [TILE_DIM, TILE_DIM * kt_dim]
    dims_B = [TILE_DIM * kt_dim, TILE_DIM * ct_dim]
    src_A = torch.rand(dims_A[0] * dims_A[1], dtype=torch.bfloat16)
    src_B = torch.rand(dims_B[0] * dims_B[1], dtype=torch.bfloat16)
    partial_tiles = NUM_BLOCKS * ct_dim
    partials = torch.empty(partial_tiles * TILE_ELEMENTS, dtype=torch.float32).uniform_(
        -16.0, 16.0
    )

    stimuli = StimuliConfig(
        tilize_block(src_A, dimensions=dims_A, stimuli_format=in_format).flatten(),
        in_format,
        tilize_block(src_B, dimensions=dims_B, stimuli_format=in_format).flatten(),
        in_format,
        out_format,
        tile_count_A=kt_dim,
        tile_count_B=kt_dim * ct_dim,
        tile_count_res=partial_tiles,
        buffer_S=partials,
        stimuli_S_format=DataFormat.Float32,
        tile_count_S=partial_tiles,
        buffer_C=_stamps_buffer(),
        stimuli_C_format=DataFormat.Int32,
        tile_count_C=1,
        srcs_layout_operands=frozenset(),
        num_faces=4,
    )

    test_config_kwargs = {
        "test_name": "sources/quasar/dest_preload_matmul_quasar_test.cpp",
        "formats": InputOutputFormat(in_format, out_format),
        "templates": [
            DEST_SYNC(DestSync.Half),
            MATH_FIDELITY(math_fidelity),
            DEST_WRITE_PATH(
                dwp_path=path,
                dwp_tiles_per_bank=ct_dim,
                dwp_kernel=DestWriteKernel.Matmul,
                dwp_loop_factor=loop_factor,
                dwp_kt_dim=kt_dim,
                dwp_pack_l1_acc=path.pack_l1_acc,
                dwp_srcs_addr_mod_step=srcs_opt.addr_mod_step,
                dwp_srcs_bank_unpack=srcs_opt.bank_unpack,
                dwp_srcs_multi_bank_unpack=srcs_opt.multi_bank_unpack,
            ),
        ],
        "runtimes": [LOOP_FACTOR(loop_factor), TILE_COUNT(partial_tiles)],
        "variant_stimuli": stimuli,
        "dest_acc": dest_acc,
    }
    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )

    if is_perf:
        configuration.run(perf_report)
    else:
        outcome = configuration.run()
        pack_src_format = data_formats(
            input_format=in_format,
            output_format=out_format,
            is_fp32_dest_acc_en=dest_acc,
            num_iterations=1,
        )[0].pack_src
        golden_mm = get_golden_generator(MatmulGolden)(
            src_A,
            src_B,
            out_format,
            math_fidelity,
            input_A_dimensions=dims_A,
            input_B_dimensions=dims_B,
            tilize=True,
            input_A_format=in_format,
            input_B_format=in_format,
            math_format=pack_src_format,
            dest_acc=dest_acc,
        ).to(torch.float32)
        golden = (partials.view(NUM_BLOCKS, -1) + golden_mm.view(1, -1)).flatten()
        res = torch.tensor(outcome.result, dtype=format_dict[out_format])
        assert res.numel() == golden.numel(), "result size"
        assert passed_test(
            golden, res, out_format
        ), f"matmul {path.value} ct={ct_dim} kt={kt_dim} {math_fidelity.name}"

    report_bank_stamps(
        stimuli,
        DestWriteKernel.Matmul,
        path,
        ct_dim,
        out_format,
        is_perf,
        loop_factor=loop_factor,
        kt_dim=kt_dim,
        fidelity=math_fidelity.name,
        srcs_opt=srcs_opt.value,
        stamp_sync=False,
    )


@parametrize(
    path=list(PartialPath),
    ct_dim=MATMUL_CT_SWEEP,
    kt_dim=MATMUL_KT_SWEEP,
    math_fidelity=MATMUL_FIDELITIES,
)
def test_dest_preload_matmul_quasar(path, ct_dim, kt_dim, math_fidelity):
    run_dest_preload_matmul(path, ct_dim, kt_dim, math_fidelity)


# Fidelity does not move the matmul cycles (see the results doc), so the option sweep runs LoFi only.
MATMUL_SRCS_OPTS = [SrcsIssueOpt.BOTH, SrcsIssueOpt.MULTI_BANK]


@parametrize(
    path=[PartialPath.B_DRAIN, PartialPath.B_DONE],
    ct_dim=MATMUL_CT_SWEEP,
    kt_dim=MATMUL_KT_SWEEP,
    srcs_opt=MATMUL_SRCS_OPTS,
)
def test_dest_preload_matmul_srcs_opt_quasar(path, ct_dim, kt_dim, srcs_opt):
    run_dest_preload_matmul(path, ct_dim, kt_dim, MathFidelity.LoFi, srcs_opt=srcs_opt)
