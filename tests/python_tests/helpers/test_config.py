# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .format_arg_mapping import (
    unpack_src_dict,
    unpack_dst_dict,
    pack_src_dict,
    pack_dst_dict,
    math_dict,
    mathop_args_dict,
    reduce_dim_args,
    reduce_pool_args,
)


def generate_make_command(test_config):
    make_cmd = f"make --silent --always-make "
    formats = test_config.get("formats")
    testname = test_config.get("testname")
    dest_acc = test_config.get("dest_acc", " ")  # default is not 32 bit dest_acc

    make_cmd += f"unpack_src={unpack_src_dict[formats.unpack_src]} unpack_dst={unpack_dst_dict[formats.unpack_dst]} fpu={math_dict[formats.math]} pack_src={pack_src_dict[formats.pack_src]} pack_dst={pack_dst_dict[formats.pack_dst]} "

    make_cmd += f"testname={testname} dest_acc={dest_acc} "
    mathop = test_config.get("mathop", "no_mathop")
    approx_mode = test_config.get("approx_mode", "false")
    math_fidelity = test_config.get("math_fidelity", 0)

    make_cmd += f" math_fidelity={math_fidelity} approx_mode={approx_mode} "

    reduce_dim = test_config.get("reduce_dim", "no_reduce_dim")
    pool_type = test_config.get("pool_type", "no_reduce_dim")

    if mathop != "no_mathop":
        if isinstance(mathop, str):  # single tile option
            if mathop in ["reduce_col", "reduce_row", "reduce_scalar"]:
                make_cmd += f"mathop={mathop} "
                make_cmd += f"reduce_dim={reduce_dim_args[reduce_dim]} "
                make_cmd += f"pool_type={reduce_pool_args[pool_type]} "
            else:
                make_cmd += f"mathop={mathop_args_dict[mathop]} "
        else:  # multiple tiles handles mathop as int
            mathop_map = {
                1: "ELTWISE_BINARY_ADD",
                2: "ELTWISE_BINARY_SUB",
                3: "ELTWISE_BINARY_MUL",
            }
            make_cmd += f"mathop={mathop_map.get(mathop, 'ELTWISE_BINARY_MUL')} "

            kern_cnt = str(test_config.get("kern_cnt"))
            pack_addr_cnt = str(test_config.get("pack_addr_cnt"))
            pack_addrs = test_config.get("pack_addrs")

            make_cmd += f" kern_cnt={kern_cnt} pack_addr_cnt={pack_addr_cnt} pack_addrs={pack_addrs}"

    print(make_cmd)
    return make_cmd
