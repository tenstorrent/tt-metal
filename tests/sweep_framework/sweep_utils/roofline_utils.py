# SPDX-FileCopyrightText: © 2054 TenstorrentAI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import is_wormhole_b0


def get_product(lst):
    count = 1
    for val in lst:
        count *= val
    return count


def get_byte_count(tensor):
    shape = tensor.padded_shape
    dtype = tensor.dtype
    if dtype in (ttnn.float32, ttnn.uint32, ttnn.int32):
        count = 4
    elif dtype == ttnn.bfloat16:
        count = 2
    elif dtype == ttnn.uint8:
        count = 1
    elif dtype == ttnn.bfloat8_b:
        count = 1.0625
    else:
        logger.warning("NEED TO PROVIDE DATA TYPE SIZE")
        count = 1
    return count * get_product(tensor.shape)


ARCH2DRAM_THROUGHPUT = {
    "wormhole_b0": 256,  # bytes/ns
    "blackhole": 512,  # bytes/ns
}


def get_roofline_metrics(tensors, throughput=None):
    if throughput is None:
        throughput = ARCH2DRAM_THROUGHPUT[ttnn.get_arch_name()]

    byte_count = 0
    for tensor in tensors:
        byte_count += get_byte_count(tensor)

    # bytes / bytes/ns = ns
    roofline = byte_count / throughput
    return {"ROOFLINE": roofline, "BYTECOUNT": byte_count}


def update_check_result(check_result, metrics):
    status, message = check_result
    if not status:
        return check_result
    metrics.update({"PCC": message})
    message = " ".join([f"{key} {value}" for key, value in metrics.items()])
    return status, message


def get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf, flop_counts=None):
    check_result = check_with_pcc(torch_output_tensor, output_tensor, expected_pcc)
    metrics = get_roofline_metrics(tensors)
    if isinstance(e2e_perf, dict):
        metrics.update(e2e_perf)
        e2e_perf = metrics["E2E_PERF"]
    else:
        metrics.update({"E2E_PERF": e2e_perf})
    if flop_counts:
        metrics.update({"FLOP_COUNT": get_product(flop_counts)})
    updated_check_result = update_check_result(check_result, metrics)
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
