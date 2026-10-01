# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import chain, product

import pytest
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    StochasticRounding,
)
from helpers.matmul_sweep import (
    DEST_HALF_BFP_PACK_HANG_REASON,
    DEST_RT_CT_BLOCKS,
    MatmulConfig,
    generate_face_layout_config_sweep,
    generate_tile_dims,
    is_dest_half_bfp_pack_hang,
    mid_fill_rt_ct_pairs,
    skip_matmul_combination,
    sweep_tiny_tiles_matmul,
)
from helpers.param_config import DEST_SYNC_TILE_LIMITS, input_output_formats
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Bfp8_b,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]
STOCHASTIC_ROUNDING_MODES = [StochasticRounding.No]
MATH_FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]

# Dest-fill RT x CT from dest capacity, dest occupancy blocks, and half-dest
# mid-fill when they fit. KT 1 and 4 (long-K lives on perf_matmul).
PERF_KT_DIMS = (1, 4)
THROTTLE_LEVELS = (0, 5)
DEST_HANDOFF_NUM_BLOCKS = 4


def _dest_capacity(dest_sync, dest_acc) -> int:
    return DEST_SYNC_TILE_LIMITS[dest_sync] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )


def _dest_fill_rt_ct_pairs(max_tiles):
    """Power-of-two dest-fill (rt, ct) pairs, dest occupancy blocks, and mid-fill when they fit dest."""
    pairs = []
    rt_dim = 1
    while rt_dim <= max_tiles:
        if max_tiles % rt_dim == 0:
            pairs.append((rt_dim, max_tiles // rt_dim))
        rt_dim *= 2
    pairs.extend((rt, ct) for rt, ct in DEST_RT_CT_BLOCKS if rt * ct <= max_tiles)
    pairs.extend(mid_fill_rt_ct_pairs(max_tiles))
    return list(dict.fromkeys(pairs))


def _fits_tiny_perf_tile_shape(cfg) -> bool:
    rt_dim = cfg.tile_dimensions.rt_dim
    ct_dim = cfg.tile_dimensions.ct_dim
    kt_dim = cfg.tile_dimensions.kt_dim
    max_tiles = _dest_capacity(cfg.dest_sync, cfg.dest_acc)
    half = max_tiles // 2
    return (
        cfg.dst_index == 0
        and rt_dim == 1
        and kt_dim == 1
        and ct_dim in (1, half, max_tiles)
    )


def generate_perf_matmul_combinations():
    """Regular matmul: dest-filling RT x CT grids, dest occupancy blocks, and mid-fill when they fit dest, with KT in {1, 4}."""
    combinations = []
    bfloat16_formats = {DataFormat.Float16_b, DataFormat.Float32}

    for fmt in MATMUL_FORMATS:
        is_fpu_bfloat16 = (
            fmt.input_format in bfloat16_formats
            and fmt.output_format in bfloat16_formats
        )
        for dest_acc in DEST_ACC_MODES:
            if is_dest_acc_needed(fmt) and dest_acc == DestAccumulation.No:
                continue
            # Don't add invalid variants. If these variants are added LLK_ASSERTs are hit in math_matmul and unpack_matmul tests.
            # In test_config.py, when compiling the test itself, dest_acc is changed to DestAccumulation.Yes, which causes the assert to be hit.
            # Furthermore, this combo is not valid because Float16_b has 8-bit exponent and Float16 has 5-bit exponent which, when doing calculations with these formats it needs to be expanded to Float32, which requires dest_acc to be true
            if (
                dest_acc == DestAccumulation.No
                and fmt.input_format == DataFormat.Float16_b
                and fmt.output_format == DataFormat.Float16
            ):
                continue

            for dest_sync in DEST_SYNC_MODES:
                max_tiles = _dest_capacity(dest_sync, dest_acc)
                for stochastic_mode in STOCHASTIC_ROUNDING_MODES:
                    for rt_dim, ct_dim in _dest_fill_rt_ct_pairs(max_tiles):
                        for kt_dim in PERF_KT_DIMS:
                            if skip_matmul_combination(
                                stochastic_mode,
                                dest_acc,
                                is_fpu_bfloat16,
                                kt_dim,
                            ):
                                continue
                            tile_dims = generate_tile_dims(
                                (
                                    [rt_dim * TILE_DIM, kt_dim * TILE_DIM],
                                    [kt_dim * TILE_DIM, ct_dim * TILE_DIM],
                                )
                            )
                            for face_layout_config in generate_face_layout_config_sweep(
                                math_matmul=True
                            ):
                                combinations.append(
                                    MatmulConfig(
                                        tile_dimensions=tile_dims,
                                        face_layout_config=face_layout_config,
                                        formats=fmt,
                                        stochastic_rnd=stochastic_mode,
                                        dst_index=0,
                                        dest_sync=dest_sync,
                                        dest_acc=dest_acc,
                                    )
                                )
    return combinations


MATMUL_COMBINATIONS = generate_perf_matmul_combinations()

TINY_TILES_MATMUL_COMBINATIONS = [
    cfg
    for cfg in sweep_tiny_tiles_matmul(
        MATMUL_FORMATS,
        DEST_ACC_MODES,
        STOCHASTIC_ROUNDING_MODES,
        DEST_SYNC_MODES,
        math_matmul=True,
    )
    if _fits_tiny_perf_tile_shape(cfg)
]

ALL_TEST_PARAMS = list(
    chain(
        (
            (fidelity, cfg, throttle, 1)
            for fidelity, cfg, throttle in product(
                MATH_FIDELITIES, MATMUL_COMBINATIONS, THROTTLE_LEVELS
            )
        ),
        (
            (fidelity, cfg, 0, 1)
            for fidelity, cfg in product(
                MATH_FIDELITIES, TINY_TILES_MATMUL_COMBINATIONS
            )
        ),
        (
            (fidelity, cfg, 0, DEST_HANDOFF_NUM_BLOCKS)
            for fidelity, cfg in product(MATH_FIDELITIES, MATMUL_COMBINATIONS)
            if cfg.tile_dimensions.kt_dim == 1
        ),
    )
)


# Experiment (#58068 nop sweep, not for merge): only the configs whose Wormhole
# PACK_ISOLATE TILE_LOOP was bistable in the base arm (969), plus 150 that were
# stable there as controls. PACK_ISOLATE only.
import os as _os

_os.environ.setdefault("LLK_PERF_RUN_TYPES", "PACK_ISOLATE")
_SELECT = [726, 737, 773, 777, 883, 887, 1004, 1010, 1022, 1046, 1070, 1671, 1672, 1678, 1681, 1686, 1688, 1692, 1695, 1697, 1704, 1705, 1706, 1710, 1711, 1715, 1720, 1737, 1740, 1741, 1745, 1747, 1753, 1762, 1763, 1768, 1769, 1777, 1788, 1791, 1802, 1804, 1807, 1809, 1812, 1817, 1819, 2015, 2019, 2026, 2030, 2039, 2076, 2090, 2107, 2111, 2114, 2118, 2123, 2129, 2131, 2144, 2152, 2158, 2174, 2179, 2183, 2187, 2195, 2199, 2215, 2219, 2223, 2226, 2231, 2262, 2263, 2265, 2278, 2291, 2353, 2355, 2359, 2364, 2367, 2376, 2390, 2396, 2398, 2408, 2416, 2420, 2425, 2428, 2429, 2431, 2438, 2448, 2451, 2452, 2453, 2455, 2576, 3546, 3549, 3555, 3565, 3566, 3597, 3625, 3651, 3652, 3658, 3685, 3688, 3712, 3716, 3717, 3728, 3734, 3735, 3736, 3740, 3747, 3781, 3801, 3866, 3886, 3899, 3907, 3908, 3909, 3911, 3930, 3932, 3934, 3938, 3940, 3942, 3946, 3962, 3971, 3973, 3993, 3996, 4007, 4011, 4022, 4029, 4055, 4059, 4094, 4097, 4101, 4107, 4111, 4114, 4123, 4127, 4141, 4142, 4146, 4149, 4154, 4179, 4183, 4187, 4199, 4203, 4207, 4215, 4219, 4232, 4237, 4266, 4269, 4270, 4278, 4430, 5040, 5047, 5049, 5050, 5061, 5065, 5069, 5071, 5077, 5080, 5081, 5083, 5093, 5109, 5111, 5113, 5115, 5121, 5125, 5127, 5129, 5134, 5136, 5137, 5141, 5146, 5149, 5150, 5151, 5153, 5161, 5162, 5168, 5169, 5172, 5177, 5180, 5181, 5185, 5188, 5189, 5371, 5379, 5383, 5394, 5402, 5418, 5441, 5447, 5475, 5482, 5513, 5514, 5518, 5524, 5526, 5539, 5543, 5551, 5555, 5559, 5567, 5583, 5587, 5591, 5633, 5637, 5638, 5642, 5697, 5729, 5752, 5754, 5764, 5770, 5784, 5793, 5820, 5821, 5835, 5839, 5850, 5854, 5856, 5861, 5867, 5893, 5918, 5921, 5925, 5931, 5935, 5938, 5942, 5947, 5950, 5951, 5953, 5974, 5980, 6007, 6015, 6019, 6026, 6027, 6030, 6035, 6039, 6047, 6052, 6081, 6089, 6090, 6093, 6096, 6102, 6220, 6228, 7364, 7395, 7454, 7541, 7542, 7543, 7631, 7634, 7635, 8400, 8408, 8409, 8417, 8422, 8439, 8465, 8477, 8485, 8488, 8496, 8504, 8505, 8512, 8513, 8516, 8517, 8518, 8524, 8532, 8533, 8539, 8545, 8546, 8749, 8751, 8772, 8791, 8815, 8823, 8838, 8863, 8865, 8881, 8919, 8931, 8935, 8985, 8987, 9057, 9077, 9090, 9133, 9165, 10316, 10325, 10333, 10343, 10345, 10348, 10349, 10353, 10357, 10361, 10365, 10369, 10371, 10373, 10380, 10385, 10388, 10393, 10399, 10413, 10431, 10435, 10478, 10505, 10508, 10510, 10511, 10537, 10541, 10555, 10566, 10593, 10605, 10631, 10671, 10687, 10693, 10695, 10721, 10723, 10739, 10783, 10789, 10795, 10807, 10838, 10841, 10847, 10881, 10985, 10993, 11005, 11010, 11769, 11772, 11776, 11781, 11788, 11790, 11792, 11794, 11797, 11798, 11803, 11804, 11811, 11817, 11821, 11833, 11837, 11842, 11844, 11845, 11847, 11849, 11867, 11868, 11870, 11872, 11877, 11882, 11884, 11892, 11896, 11897, 11901, 11909, 11912, 11913, 11925, 12107, 12136, 12167, 12171, 12197, 12202, 12206, 12295, 12357, 12391, 12395, 12400, 12404, 12437, 12476, 12479, 12503, 12545, 12547, 12551, 12571, 12573, 12575, 12619, 12643, 12662, 12669, 12697, 12699, 12822, 12917, 12952, 14098, 14211, 14245, 14265, 14371, 14374, 15137, 15138, 15154, 15156, 15158, 15161, 15164, 15166, 15184, 15185, 15204, 15209, 15216, 15217, 15220, 15232, 15267, 15271, 15285, 15292, 15508, 15527, 15533, 15535, 15555, 15577, 15643, 15665, 15698, 15709, 15730, 15746, 15761, 15765, 15794, 15795, 15830, 15883, 15891, 17017, 17024, 17026, 17039, 17052, 17065, 17093, 17101, 17109, 17112, 17129, 17131, 17133, 17163, 17175, 17190, 17211, 17218, 17223, 17231, 17238, 17245, 17263, 17271, 17273, 17277, 17283, 17359, 17370, 17374, 17445, 17475, 17519, 17523, 17531, 17555, 17579, 17581, 17630, 17659, 17662, 17667, 17691, 17719, 17727, 17737, 17741, 18512, 18519, 18522, 18524, 18528, 18536, 18551, 18556, 18564, 18568, 18572, 18577, 18585, 18597, 18598, 18600, 18604, 18609, 18626, 18628, 18632, 18640, 18647, 18648, 18649, 18653, 18660, 18663, 18855, 18875, 18877, 18915, 18917, 18923, 18939, 18942, 18945, 18947, 18949, 18951, 18967, 18971, 19039, 19063, 19069, 19087, 19089, 19116, 19131, 19136, 19169, 19172, 19208, 19235, 19299, 19307, 19328, 19355, 19379, 19383, 19401, 19405, 19407, 19427, 19515, 19564, 19573, 19672, 19688, 19692, 19696, 20947, 20983, 20993, 21010, 21037, 21095, 21101, 21872, 21880, 21889, 21897, 21898, 21901, 21904, 21905, 21909, 21915, 21918, 21920, 21948, 21960, 21965, 21973, 21976, 21979, 21981, 21983, 21984, 21988, 21989, 21992, 21993, 21997, 22001, 22007, 22008, 22012, 22014, 22016, 22017, 22018, 22020, 22021, 22024, 22028, 22030, 22219, 22247, 22251, 22267, 22271, 22279, 22315, 22319, 22335, 22383, 22387, 22391, 22399, 22427, 22449, 22486, 22494, 22497, 22499, 22501, 22502, 22521, 22522, 22566, 22587, 22599, 23753, 23755, 23781, 23783, 23795, 23797, 23805, 23807, 23813, 23825, 23841, 23853, 23855, 23865, 23878, 23882, 23885, 23903, 23909, 23935, 23942, 23977, 23993, 24006, 24009, 24019, 24041, 24050, 24065, 24067, 24069, 24089, 24095, 24099, 24106, 24110, 24135, 24151, 24191, 24199, 24217, 24223, 24275, 24306, 24310, 24317, 24319, 24335, 24341, 24351, 24423, 24427, 24486, 24571, 25241, 25260, 25262, 25264, 25272, 25275, 25279, 25283, 25285, 25288, 25290, 25304, 25309, 25312, 25316, 25328, 25332, 25345, 25352, 25353, 25360, 25361, 25364, 25376, 25380, 25381, 25385, 25386, 25389, 25390, 25392, 25395, 25396, 25589, 25591, 25611, 25623, 25631, 25633, 25635, 25643, 25655, 25659, 25663, 25678, 25681, 25683, 25705, 25709, 25747, 25767, 25787, 25795, 25806, 25834, 25859, 25887, 25939, 25979, 25981, 25983, 26003, 26029, 26045, 26068, 26075, 26111, 26113, 26134, 26163, 26239, 26247, 26285, 26389, 27498, 27600, 30765, 30771, 30799, 30813, 30968, 30970, 30971, 30974, 30977, 30978, 30991, 30993, 31067, 31101, 31129, 31142, 31181, 31439, 31447, 31455, 31466, 31490, 31520, 31536, 31542, 31567, 31579, 31809, 31816, 31819, 31821, 31832, 31833, 31836, 31839, 31893, 31915, 31944, 31950, 31951, 31956, 31974, 31982, 31984, 32029, 32051, 32066, 32100, 32424, 32453, 32652, 32664, 32666, 32672, 32678, 32680, 32681, 32784, 32810, 33119, 33138, 33159, 33169, 33203, 33236, 33285, 33486, 33488, 33490, 33502, 33512, 33518, 33661, 34332, 34333, 34334, 34355, 34366, 34416, 34429, 34437, 34800, 34802, 34851, 34852, 34855, 34874, 34875, 34887, 34937, 35170, 35174, 35176, 35182, 35197, 35198, 35202, 35206, 35207, 35323, 35329, 35367, 35393, 35466, 35792, 35797, 36017, 36020, 36024, 36028, 36033, 36048, 36049, 36173, 36175, 36185, 36523, 36529, 36571, 36600, 36621, 36661, 36861, 36870, 36872, 36882, 36890, 36891, 37007, 37020, 37045, 37051, 260, 1031, 1700, 1760, 1979, 2157, 2456, 2592, 2684, 3469, 3578, 4770, 5105, 5156, 5727, 5758, 6164, 6660, 7536, 7597, 7718, 7857, 8543, 8671, 8707, 8737, 8871, 8878, 8894, 9180, 9481, 10066, 10132, 10764, 10786, 10992, 11132, 11207, 11496, 11777, 11860, 11924, 11964, 11975, 12018, 12077, 12088, 12157, 12248, 12277, 12533, 13753, 14252, 14262, 14481, 14935, 15143, 15417, 15590, 15662, 15686, 15691, 15731, 15855, 15939, 16342, 16723, 17234, 17322, 17460, 17733, 17739, 17823, 18547, 19050, 19102, 19191, 19281, 19325, 19329, 20056, 20657, 21025, 21163, 22022, 22136, 22496, 22547, 22633, 22787, 22826, 22904, 23672, 24140, 24236, 24238, 24304, 24566, 24772, 25597, 25606, 25640, 25712, 25825, 26005, 26200, 26252, 26415, 26446, 27713, 27874, 28489, 28627, 28660, 28663, 28905, 29059, 29265, 29322, 29455, 29469, 29846, 30127, 30441, 31153, 31190, 31929, 32095, 32713, 32848, 33195, 33317, 33360, 34139, 34527, 34565, 34822, 35252, 35298, 35362, 35455, 35469, 35819, 36074, 36681, 36901, 36947, 36951, 37015, 37086]
ALL_TEST_PARAMS = [ALL_TEST_PARAMS[_i] for _i in _SELECT]

@pytest.mark.perf
@pytest.mark.parametrize(
    "math_fidelity,matmul_config,throttle,num_blocks", ALL_TEST_PARAMS
)
def test_perf_math_matmul(
    math_fidelity,
    matmul_config,
    throttle,
    num_blocks,
    perf_report,
):
    """
    Performance test for matmul operations.

    Regular matmul uses dest-filling RT x CT grids sized to dest capacity, dest
    occupancy blocks (1×N through dest Half, 2×N, 4×1 / 4×2), and half-dest
    mid-fill (2×2, 1×(cap/2), (cap/2)×1) when they fit, with KT in {1, 4} and
    throttle 0 or 5. Tiny tiles cover ct=1, cap/2, and dest-fill ct. A
    dest-handoff slice repeats dest_index=0 grids (NUM_BLOCKS=4, KT=1,
    throttle=0).
    """
    formats = matmul_config.formats
    in0_dimensions = matmul_config.tile_dimensions.in0_dimensions
    in1_dimensions = matmul_config.tile_dimensions.in1_dimensions
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    num_faces_in0 = matmul_config.face_layout_config.num_faces_in0
    num_faces_in1 = matmul_config.face_layout_config.num_faces_in1
    num_faces = matmul_config.face_layout_config.num_faces

    if is_dest_acc_needed(formats) and matmul_config.dest_acc == DestAccumulation.No:
        pytest.skip("Dest accumulation must be enabled for this format")

    if is_dest_half_bfp_pack_hang(
        matmul_config.dest_sync, matmul_config.dest_acc, formats
    ):
        pytest.skip(DEST_HALF_BFP_PACK_HANG_REASON)

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    variant_tile_count = (
        matmul_config.tile_dimensions.rt_dim
        * matmul_config.tile_dimensions.ct_dim
        * matmul_config.tile_dimensions.kt_dim
    )

    configuration = PerfConfig(
        "sources/math_matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(matmul_config.dest_sync),
            THROTTLE_LEVEL(throttle),
        ],
        runtimes=[
            DEST_INDEX(matmul_config.dst_index),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            TILE_COUNT(variant_tile_count * num_blocks),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(
                matmul_config.tile_dimensions.rt_dim
                * matmul_config.tile_dimensions.ct_dim
            ),
            NUM_FACES(
                num_faces, num_faces_in0, num_faces_in1
            ),  # In0 -> Input A, In1 -> Input B
            PARTIAL_FACE(  # In0 -> Input A, In1 -> Input B
                partial_a=matmul_config.face_layout_config.partial_face_in0,
                partial_face_pack=matmul_config.face_layout_config.partial_face_pack,
                partial_b=matmul_config.face_layout_config.partial_face_in1,
                partial_face_math=matmul_config.face_layout_config.partial_face_math,
            ),
            CRK_TILE_DIMM(
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.kt_dim,
            ),
            IN_TILE_DIMS(
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            LOOP_FACTOR(1024),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_in0,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_in1,
            tile_count_res=matmul_config.tile_dimensions.output_tile_cnt * num_blocks,
        ),
        dest_acc=matmul_config.dest_acc,
    )

    configuration.run(perf_report)
