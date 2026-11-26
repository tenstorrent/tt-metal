# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
softmax_k SFPU test (Blackhole only).

Covers hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_softmax_k.h
(``_calculate_softmax_k_<is_fp32_dest_acc_en, NUM_ROWS>(k)``).

Per-row softmax over the first ``k`` columns (``1 <= k <= 16``) of ``NUM_ROWS``
(``1, 2, 4, or 8``) independent rows in face 0:

  out[r][c] = exp(x[r][c] - max_{c' < k} x[r][c']) / sum_{c' < k} exp(x[r][c'] - max)   c < k
  out[r][c] = 0                                                                          c >= k

Unlike the retired single-band implementation, this kernel:
  - computes the per-row maximum internally in SFPU (no caller-staged maxima needed),
  - handles zero logits inside ``[0, k)`` without skipping lanes,
  - masks dirty tail elements in ``[k, 16)`` before max and zeroes them after exp,
  - supports arbitrary odd and even ``k in [1, 16]`` across both BF16 and FP32 DEST,
  - handles 1, 2, 4 (1 SFPU row-group: rows 0..3) and 8 (2 row-groups: rows 0..7) rows.
"""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.constraints import skip_if_32bit_input_without_dest_acc
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    SoftmaxKGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import execute_llk
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    SOFTMAX_K,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test, result_tensor

pytestmark = [skip_for_wormhole, skip_for_quasar]

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)
DEST_SYNCS = [DestSync.Half]

FACE_DIM = 16
NUM_ROWS = [1, 2, 4, 8]
# Covers boundary k=1, odd & even k, mid-range, k=15, and full-face k=16.
ALL_K = [1, 2, 3, 4, 7, 8, 12, 15, 16]


def _build_input_tile(
    num_rows: int, k: int, torch_format
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a [32, 32] input tile and return (tile, logits[num_rows, k]).

    Active entries [0:num_rows, 0:k] include both random logits and an explicit 0.0 logit
    (when k >= 2) to verify zero logits are not treated as padding. Tail entries
    [0:num_rows, k:16] are filled with a large dirty sentinel (100.0) that would dominate
    softmax if not masked. Untouched regions (face 1 and rows outside the active SFPU band)
    hold non-zero sentinels to verify the kernel does not clobber them.
    """
    tile = torch.zeros((TILE_DIM, TILE_DIM), dtype=torch.float32)

    # Non-zero sentinels in untouched regions (face 1 and rows >= active SFPU row band).
    band_rows = 8 if num_rows == 8 else 4
    tile[:band_rows, FACE_DIM:] = 3.5
    tile[band_rows:, :] = -2.25

    logits = torch.randn((num_rows, k), dtype=torch.float32) * 2.0
    if k >= 2:
        logits[0, 0] = 0.0
    logits = logits.to(torch_format).to(torch.float32)

    tile[:num_rows, :k] = logits
    if k < FACE_DIM:
        # Dirty tail values that would dominate softmax if not masked before max.
        tile[:num_rows, k:FACE_DIM] = 100.0

    return tile.to(torch_format), logits


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=DEST_SYNCS,
    num_rows=NUM_ROWS,
    k=ALL_K,
)
def test_sfpu_softmax_k(
    formats,
    dest_acc,
    dest_sync,
    num_rows,
    k,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    torch.manual_seed(100 + num_rows * 31 + k)
    torch_format = format_dict[formats.input_format]

    input_tile, logits = _build_input_tile(num_rows, k, torch_format)

    golden_generator = get_golden_generator(SoftmaxKGolden)
    golden_tile = golden_generator(input_tile, logits, k, dest_acc, num_rows, FACE_DIM)

    src_A = tilize_block(
        input_tile.flatten(), [TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name="sources/sfpu_softmax_k_test.cpp",
            formats=formats,
            templates=[
                SOFTMAX_K(softmax_k=k, softmax_num_rows=num_rows),
                DEST_SYNC(dest_sync),
            ],
            runtimes=[
                LOOP_FACTOR(1),
                TILE_COUNT(1),
            ],
            variant_stimuli=StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=1,
                tile_count_B=1,
                tile_count_res=1,
            ),
            dest_acc=dest_acc,
            unpack_to_dest=formats.input_format.is_32_bit(),
        ),
    )
    if is_perf:
        return

    res_from_L1 = outcome.result
    res_tensor = result_tensor(res_from_L1, formats.output_format)
    res_tile = untilize_block(
        res_tensor, formats.output_format, [TILE_DIM, TILE_DIM]
    ).reshape(TILE_DIM, TILE_DIM)

    atol = 1e-2 if formats.output_format == DataFormat.Float16_b else 1e-4
    rtol = 1e-2 if formats.output_format == DataFormat.Float16_b else 1e-3
    torch.testing.assert_close(
        res_tile[:num_rows, :k].to(torch.float32),
        golden_tile[:num_rows, :k].to(torch.float32),
        atol=atol,
        rtol=rtol,
    )

    if k < FACE_DIM:
        padding = res_tile[:num_rows, k:FACE_DIM]
        assert bool((padding.to(torch.float32) == 0.0).all()), (
            f"columns {k}-{FACE_DIM - 1} are tail padding and must come back exactly 0.0 "
            f"(num_rows={num_rows}, k={k}):\n{padding}"
        )

    for row in range(num_rows):
        row_sum = res_tile[row, :k].to(torch.float32).sum().item()
        assert (
            abs(row_sum - 1.0) < 2e-2
        ), f"row {row} softmax sums to {row_sum}, expected 1.0 (num_rows={num_rows}, k={k})"

    # Face 1 of the processed band must be untouched (verifies VectorMode::RC_custom).
    band_rows = 8 if num_rows == 8 else 4
    assert passed_test(
        golden_tile[:band_rows, FACE_DIM:].flatten(),
        res_tile[:band_rows, FACE_DIM:].flatten(),
        formats.output_format,
    ), "softmax_k modified face 1 (columns 16-31) of its own row band"

    # Rows outside the processed SFPU band must be untouched.
    assert passed_test(
        golden_tile[band_rows:, :].flatten(),
        res_tile[band_rows:, :].flatten(),
        formats.output_format,
    ), "softmax_k modified DEST rows outside its active row band"
