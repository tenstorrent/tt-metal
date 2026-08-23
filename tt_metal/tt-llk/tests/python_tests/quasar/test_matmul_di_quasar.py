# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Direct-indexing (DI) matmul correctness test.

The DI MOP is a separate instruction stream from the auto-increment one: each MVMULDI
carries explicit srcb/srca/dest indices (4-row granular, so an 8-row FPU steps indices
by 2) and the replayed instructions apply no addr_mod increments. Only the DI+X2 variant
(MxFp4_2x_A/B register format hint) is covered by test_matmul_quasar.py today, so plain
DI gets its own test here while it is brought up.

Sweep helpers are imported from test_matmul_quasar so the coverage shape matches; the
kernel (sources/quasar/matmul_di_quasar_test.cpp) forces ENABLE_DIRECT_INDEXING on and
2x off. Once this passes, both files fold back into the matmul test/kernel pair.
"""

import pytest
import torch
from helpers.data_format_inference import data_formats
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MatmulGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import PerfRunType, Transpose, format_dict
from helpers.matmul_sweep import generate_tile_dims
from helpers.param_config import parametrize, runtime
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test
from quasar.test_matmul_quasar import (
    MATMUL_FORMAT,
    matmul_dest_acc_modes,
    matmul_dest_sync_modes,
    matmul_dimensions,
    matmul_implied_math_formats,
    matmul_math_fidelities,
)


@pytest.mark.quasar
@parametrize(
    format=MATMUL_FORMAT,
    math_fidelity=lambda format: matmul_math_fidelities(format),
    dest_sync_mode=lambda: matmul_dest_sync_modes(),
    dest_acc=matmul_dest_acc_modes,
    dimensions=runtime(
        lambda dest_acc, dest_sync_mode: matmul_dimensions(dest_acc, dest_sync_mode)
    ),
    implied_math_format=lambda format: matmul_implied_math_formats(format),
)
def test_matmul_di(
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    dimensions,
    format,
    implied_math_format,
):
    input_A_dimensions, input_B_dimensions = dimensions

    if format.input_format == DataFormat.Int8:
        stimuli_spec = StimuliSpec.uniform(low=-127.0, high=127.0)
    else:
        stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=format.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=format.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
        output_format=format.output_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=format.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=format.input_format
    )

    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    torch_format = format_dict[format.output_format]
    src_A_golden = src_A
    src_B_golden = src_B
    # MX inputs reach the FPU already quantized by the unpacker, so the golden has to
    # start from the quantized values, not the raw stimuli.
    if format.input_format.is_mx_format():
        tilized_A_golden = quantize_mx_tensor_chunked(
            tilized_A.flatten().to(torch.bfloat16), format.input_format
        ).reshape(tilized_A.shape)
        tilized_B_golden = quantize_mx_tensor_chunked(
            tilized_B.flatten().to(torch.bfloat16), format.input_format
        ).reshape(tilized_B.shape)
        src_A_golden = untilize_block(
            tilized_A_golden,
            stimuli_format=format.input_format,
            dimensions=input_A_dimensions,
        )
        src_B_golden = untilize_block(
            tilized_B_golden,
            stimuli_format=format.input_format,
            dimensions=input_B_dimensions,
        )

    formats_config = data_formats(
        input_format=format.input_format,
        input_format_B=format.input_format_B,
        output_format=format.output_format,
        is_fp32_dest_acc_en=dest_acc,
        num_iterations=1,
        unpacking_to_dest=False,
        disable_format_inference=format.input_format.is_mx_format(),
    )[0]
    pack_src_format = formats_config.pack_src

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A_golden,
        src_B_golden,
        format.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        input_A_format=format.input_format,
        input_B_format=format.input_format,
        math_format=pack_src_format,  # For accumulation of results in matmul we require to calculate in pack_src_format.
        dest_acc=dest_acc,
    )

    num_faces = 4

    templates = [
        MATH_FIDELITY(math_fidelity),
        IMPLIED_MATH_FORMAT(implied_math_format),
        DEST_SYNC(dest_sync_mode),
        UNPACK_TRANS_FACES(Transpose.No),
    ]
    runtimes = [
        CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
        TILE_COUNT(matmul_dims.output_tile_cnt * matmul_dims.kt_dim),
        NUM_FACES(num_faces, num_faces, num_faces),
        LOOP_FACTOR(1),
    ]
    variant_stimuli = StimuliConfig(
        tilized_A.flatten(),
        format.input_format,
        tilized_B.flatten(),
        format.input_format,
        format.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=matmul_dims.output_tile_cnt,
        num_faces=num_faces,
    )

    test_config_kwargs = {
        "test_name": "sources/quasar/matmul_di_quasar_test.cpp",
        "formats": format,
        "templates": templates,
        "runtimes": runtimes,
        "variant_stimuli": variant_stimuli,
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
        "disable_format_inference": format.input_format.is_mx_format(),
    }

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=[PerfRunType.L1_TO_L1],
        test_config_kwargs=test_config_kwargs,
        boot_mode=BootMode.TRISC,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # For MX outputs, model the packer: quantize the golden onto the MX lattice (from the
    # math/pack_src format the result was produced in) so the comparison validates the
    # device's MX output quantization, not just matmul-math-to-MX-precision.
    if format.output_format.is_mx_format():
        golden_tensor = quantize_mx_tensor_chunked(
            golden_tensor.to(format_dict[pack_src_format]), format.output_format
        ).to(torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        format.output_format,
    ), "Assert against golden failed"
