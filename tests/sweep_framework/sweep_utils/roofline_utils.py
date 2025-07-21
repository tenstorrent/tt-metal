# SPDX-FileCopyrightText: © 2054 TenstorrentAI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc


def get_product(lst):
    count = 1
    for val in lst:
        count *= val
    return count


def get_byte_count(tensor):
    shape = tensor.padded_shape
    dtype = tensor.dtype
    if dtype == ttnn.float32:
        count = 4
    elif dtype == ttnn.bfloat16:
        count = 2
    elif dtype == ttnn.bfloat8_b:
        count = 1.0625
    else:
        logger.warning("NEED TO PROVIDE DATA TYPE SIZE")
        count = 1
    return count * get_product(tensor.shape)


WH_DRAM_THROUGHPUT = 256  # bytes/ns


def get_rooofline_message(tensors, throughput=WH_DRAM_THROUGHPUT):
    byte_count = 0
    for tensor in tensors:
        byte_count += get_byte_count(tensor)

    # bytes / bytes/ns = ns
    roofline = byte_count / throughput
    return f"ROOFLINE {roofline} BYTECOUNT {byte_count}"


def update_check_result(check_result, messages):
    status, message = check_result
    if not status:
        return check_result
    new_message = message
    for msg in messages:
        new_message += " " + msg
    return (status, "PCC " + new_message)


def get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf, flop_counts=None):
    check_result = check_with_pcc(torch_output_tensor, output_tensor, expected_pcc)
    messages = [get_rooofline_message(tensors)]
    if flop_counts:
        messages.append(f"FLOP_COUNT {get_product(flop_counts)}")
    updated_check_result = update_check_result(check_result, messages)
    return [updated_check_result, e2e_perf]


def get_updated_message(message, perf_result):
    if perf_result is None:
        return message
    key = "DEVICE FW DURATION [ns]"
    if key in perf_result:
        value = float(perf_result[key])
        tokens = message.split()
        for i in range(len(tokens)):
            if tokens[i] == "ROOFLINE":
                message += f" ROOFLINE% {float(tokens[i+1])/value}"
            if tokens[i] == "FLOP_COUNT":
                message += f" TFLOPS {float(tokens[i+1])/1e3/value}"  # flop_count / tera(10^12)*nano(10^-9) / ns
    return message
