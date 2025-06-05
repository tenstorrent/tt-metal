# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .format_arg_mapping import (
    ApproximationMode,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    math_dict,
    pack_dst_dict,
    pack_src_dict,
    unpack_A_dst_dict,
    unpack_A_src_dict,
    unpack_B_dst_dict,
    unpack_B_src_dict,
)
from .format_config import InputOutputFormat


class ProfilerBuild(Enum):
    Yes = "true"
    No = "false"


def generate_make_command(
    test_config, profiler_build: ProfilerBuild = ProfilerBuild.No
):
    make_cmd = f"make -j 6 --silent "
    formats = test_config.get("formats")
    testname = test_config.get("testname")

    make_cmd += f"llk_profiler={profiler_build.value} "

    dest_acc = test_config.get("dest_acc", DestAccumulation.No)
    unpack_to_dest = str(test_config.get("unpack_to_dest", False)).lower()

    if isinstance(formats, InputOutputFormat):
        make_cmd += f"unpack_A_src={unpack_A_src_dict[formats.input_format]} pack_dst={pack_dst_dict[formats.output_format]} "
    else:
        make_cmd += f"unpack_A_src={unpack_A_src_dict[formats.unpack_A_src]} unpack_A_dst={unpack_A_dst_dict[formats.unpack_A_dst]} unpack_B_src={unpack_B_src_dict[formats.unpack_B_src]} unpack_B_dst={unpack_B_dst_dict[formats.unpack_B_dst]} "
        make_cmd += f"fpu={math_dict[formats.math]} pack_src={pack_src_dict[formats.pack_src]} pack_dst={pack_dst_dict[formats.pack_dst]} "

    make_cmd += f"testname={testname} dest_acc={dest_acc.value} unpack_to_dest={unpack_to_dest} "
    mathop = test_config.get("mathop", "no_mathop")
    approx_mode = test_config.get("approx_mode", ApproximationMode.No)
    math_fidelity = test_config.get("math_fidelity", MathFidelity.LoFi)

    make_cmd += f" math_fidelity={math_fidelity.value} approx_mode={approx_mode.value} "

    reduce_dim = test_config.get("reduce_dim", ReduceDimension.No)
    pool_type = test_config.get("pool_type", ReduceDimension.No)

    if mathop != "no_mathop":
        if testname != "multiple_tiles_eltwise_test":  # single tile option
            if mathop in [
                MathOperation.ReduceColumn,
                MathOperation.ReduceRow,
                MathOperation.ReduceScalar,
            ]:
                make_cmd += f"mathop={mathop.value} "
                make_cmd += f"reduce_dim={reduce_dim.value} "
                make_cmd += f"pool_type={pool_type.value} "
            else:
                make_cmd += f"mathop={mathop.value} "
        else:  # multiple tiles handles mathop as int. we don't access value but return ENUM directly which is position in the class + 1

            make_cmd += f"mathop={mathop.value} "
            kern_cnt = str(test_config.get("kern_cnt", 1))
            pack_addr_cnt = str(test_config.get("pack_addr_cnt"))
            pack_addrs = test_config.get("pack_addrs")

            make_cmd += f"kern_cnt={kern_cnt} pack_addr_cnt={pack_addr_cnt} pack_addrs={pack_addrs} "

    return make_cmd
