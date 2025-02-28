# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .format_arg_mapping import format_args_dict,mathop_args_dict, reduce_dim_args, reduce_pool_args

def generate_make_command(test_config):
    make_cmd = f"make --silent --always-make "

    input_format = test_config.get("input_format", "Float16_b") # Flolat16_b is default
    output_format = test_config.get("output_format", "Float16_b")
    testname = test_config.get("testname")
    dest_acc = test_config.get("dest_acc", " ") # default is not 32 bit dest_acc 

    make_cmd += f"format={format_args_dict[output_format]} testname={testname} dest_acc={dest_acc} " # jsut for now take output_format
    
    mathop = test_config.get("mathop", "no_mathop")
    approx_mode = test_config.get("approx_mode","false")
    math_fidelity = test_config.get("math_fidelity",0)

    make_cmd += f" math_fidelity={math_fidelity} "
    make_cmd += f" approx_mode={approx_mode} "

    reduce_dim =  test_config.get("reduce_dim","no_reduce_dim")
    pool_type =  test_config.get("pool_type","no_reduce_dim")

    if(mathop != "no_mathop"):
        if isinstance(mathop,str): # single tile option
            if(mathop == "reduce"):
                make_cmd += f"mathop={  mathop_args_dict[mathop]} "
                make_cmd += f"reduce_dim={reduce_dim_args[reduce_dim]} "
                make_cmd += f"pool_type={reduce_pool_args[pool_type]} "
            else:
                make_cmd += f"mathop={  mathop_args_dict[mathop]} "
        else: # multiple tiles handles mathop as int

            if(mathop == 1):
                make_cmd += " mathop=ELTWISE_BINARY_ADD "
            elif(mathop == 2):
                make_cmd += " mathop=ELTWISE_BINARY_SUB "
            else:
                make_cmd += " mathop=ELTWISE_BINARY_MUL "

            kern_cnt = str(test_config.get("kern_cnt"))
            pack_addr_cnt = str(test_config.get("pack_addr_cnt"))
            pack_addrs = test_config.get("pack_addrs")

            make_cmd += f" kern_cnt={kern_cnt} "
            make_cmd += f" pack_addr_cnt={pack_addr_cnt} pack_addrs={pack_addrs}" 

    print(make_cmd)

    return make_cmd
