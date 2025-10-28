# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict

import pandas as pd
import torch
from loguru import logger

from models.tt_transformers.tt.model_config import HfAttentionWrapper, HfDecoderWrapper, HfModelWrapper


def _extract_dtype_from_state_dict(model):
    """Helper to extract dtype from model's state_dict."""
    try:
        state_dict = model.state_dict()
        for key, param in state_dict.items():
            if "weight" in key:
                print(f"get_ref_model_dype: key={key}, dtype={param.dtype}")
                return param.dtype
    except Exception as e:
        pass
    return None


def get_ref_model_dype(ref_model, model_name):
    default_dype = torch.float32

    if ref_model is None and model_name is None:
        return default_dype

    try:
        models_to_check = []
        if isinstance(ref_model, HfAttentionWrapper):
            models_to_check.append(ref_model.attention)
        elif isinstance(ref_model, HfDecoderWrapper):
            models_to_check.append(ref_model.decoder)
        elif isinstance(ref_model, HfModelWrapper):
            models_to_check.append(ref_model.model)
        else:
            models_to_check = [ref_model]

        # Try all models until one works
        for model in models_to_check:
            if model is not None:
                dtype = _extract_dtype_from_state_dict(model)
                if dtype is not None:
                    return dtype

    except Exception as e:
        pass

    # try hardcoded dtypes
    if model_name and isinstance(model_name, str):
        model_name_lower = model_name.lower()
        if "mistral-7b" in model_name_lower:
            return torch.bfloat16
        if "llama" in model_name_lower:
            return torch.bfloat16
        if "phi-3-mini" in model_name_lower or "phi-4" in model_name_lower:
            return torch.bfloat16

    return default_dype


### UTIL FUNCTIONS FOR DEVICE PERF


def build_duration_dict(raw_dict, column_name):
    """Build a dictionary of op codes to list of durations."""
    op_code_dict = {}
    for entry in raw_dict:
        if column_name not in entry:
            logger.warning(f"Warning: {entry} does not have column {column_name}")
        op_code = entry["OP CODE"]
        duration = entry[column_name]
        if op_code not in op_code_dict:
            op_code_dict[op_code] = []
        op_code_dict[op_code].append(duration)
    return op_code_dict


def build_duration_per_instance_dict(input_dict, num_layers):
    """Build a dictionary of op codes to list of durations per instance."""
    per_instance_dict = {}
    for op_code in input_dict:
        num_ops_with_op_code = len(input_dict[op_code])
        num_instances = num_ops_with_op_code // num_layers
        if num_ops_with_op_code % num_layers != 0:
            logger.warning(f"Warning: {op_code} has {num_ops_with_op_code} ops, not a multiple of {num_layers} layers")
            print_dict(input_dict, "input_dict")
            assert num_ops_with_op_code % num_layers == 0
        for iteration_id in range(num_layers):
            for instance_id in range(num_instances):
                op_code_with_id = f"{op_code}_{instance_id}"
                if op_code_with_id not in per_instance_dict:
                    per_instance_dict[op_code_with_id] = []
                per_instance_dict[op_code_with_id].append(
                    input_dict[op_code][iteration_id * num_instances + instance_id]
                )
    return per_instance_dict


def merge_device_rows(df):
    block_by_device = defaultdict(list)

    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]

        if op_type == "tt_dnn_device":
            device_id = int(row["DEVICE ID"])
            block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []

    global_index = 0
    while max(len(block_by_device[device_id]) for device_id in device_ids) > 0:
        blocks = []
        op_name = None
        missing_devices = []
        for device_id in device_ids:
            if not len(block_by_device[device_id]):
                logger.warning(f"Warning: Device {device_id} is missing operation {op_name} at index {global_index}")
                continue
            if op_name is None:
                op_name = block_by_device[device_id][0][0]
            elif op_name != block_by_device[device_id][0][0]:
                missing_devices.append(device_id)
                continue

            blocks.append(block_by_device[device_id].pop(0))

        if missing_devices:
            logger.warning(
                f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices {missing_devices} - do not trust data for this op or directly subsequent ops with the same name"
            )

        if not blocks:
            break

        if "AllGather" in op_name or "ReduceScatter" in op_name or "AllReduce" or "Matmul_RS" in op_name:
            # For collective ops, take the average duration over all rows within a block
            device_kernel_durations = [
                d["DEVICE KERNEL DURATION [ns]"]
                for _, d in blocks
                if "DEVICE KERNEL DURATION [ns]" in d and not math.isnan(d["DEVICE KERNEL DURATION [ns]"])
            ]

            average_duration = (
                sum(device_kernel_durations) / len(device_kernel_durations) if device_kernel_durations else float("nan")
            )
            # Use the first block's data but update its duration with the average
            base_block = blocks[0][1].copy()
            base_block["DEVICE KERNEL DURATION [ns]"] = average_duration
            merged_blocks.append(base_block)
        else:
            # For non-collective ops, take the row with maximum duration
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

        global_index += 1

    return pd.DataFrame(merged_blocks)


def process_measurements(df, num_layers):
    raw_dict = df[
        ["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]", "DEVICE KERNEL FIRST TO LAST START [ns]"]
    ].to_dict(orient="records")

    # Kernel duration
    kernel_duration_dict = build_duration_dict(raw_dict, "DEVICE KERNEL DURATION [ns]")
    kernel_duration_per_instance_dict = build_duration_per_instance_dict(kernel_duration_dict, num_layers)
    kernel_duration_per_instance_aggregate_dict = {
        "avg": aggregate_per_instance_dict(kernel_duration_per_instance_dict, lambda v: sum(v) / len(v)),
        "min": aggregate_per_instance_dict(kernel_duration_per_instance_dict, min),
        "max": aggregate_per_instance_dict(kernel_duration_per_instance_dict, max),
    }

    # Dispatch duration
    dispatch_duration_dict = build_duration_dict(raw_dict, "OP TO OP LATENCY [ns]")
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers)
    dispatch_duration_per_instance_aggregate_dict = {
        "avg": aggregate_per_instance_dict(dispatch_duration_per_instance_dict, lambda v: sum(v) / len(v)),
        "min": aggregate_per_instance_dict(dispatch_duration_per_instance_dict, min),
        "max": aggregate_per_instance_dict(dispatch_duration_per_instance_dict, max),
    }
    # First to last start
    first_to_last_start_dict = build_duration_dict(raw_dict, "DEVICE KERNEL FIRST TO LAST START [ns]")
    first_to_last_start_per_instance_dict = build_duration_per_instance_dict(first_to_last_start_dict, num_layers)
    first_to_last_start_per_instance_aggregate_dict = {
        "avg": aggregate_per_instance_dict(first_to_last_start_per_instance_dict, lambda v: sum(v) / len(v)),
        "min": aggregate_per_instance_dict(first_to_last_start_per_instance_dict, min),
        "max": aggregate_per_instance_dict(first_to_last_start_per_instance_dict, max),
    }

    return (
        kernel_duration_per_instance_aggregate_dict,
        dispatch_duration_per_instance_aggregate_dict,
        first_to_last_start_per_instance_aggregate_dict,
    )


def print_dict(input_dict, dict_name):
    # print dict as a readable python dict
    logger.info(f"\n{dict_name} = {{")
    for op_code_with_id in input_dict:
        logger.info(f'"{op_code_with_id}": {input_dict[op_code_with_id]},')
    logger.info("}")


def aggregate_per_instance_dict(input_dict, agg_fn, default=0):
    result = {}
    for key, values in input_dict.items():
        clean_values = [v if v is not None else 0 for v in values]
        result[key] = agg_fn(clean_values) if clean_values else default
    return result


def split_compile_and_trace(
    df: pd.DataFrame,
    num_head_ops: int = 0,
    num_tail_ops: int = 0,
    mode: str = "prefill",
    num_runs: int = 1,
    num_layers: int = None,
    tail_start_index: int = None,
):
    """
    Split a concatenated ops DataFrame into compile and runtime-trace segments,
    and further partition those into first layer, mid layers, and model tail DataFrames.

    The ops CSV typically contains three consecutive phases: compile, capture/trace,
    and runtime trace. When an extra sampling compile pass is present (to enable
    random sampling), it contributes a fixed number of rows that should not be used
    to determine the thirds split.

    Parameters:
        df:                the input DataFrame (all ops)
        num_tail_ops:      fixed rows from a sampling-only compile run (excluded from thirds logic)
        num_runs:          number of runs in the CSV (typically 3: compile, capture, trace)
        num_layers:        number of core layers to partition (required for further splits)
        op_start_index:    slice index for start of core layers region (inclusive)
        op_end_index:      slice index for end of core layers region (exclusive)
        tail_start_index:  slice index for start of model tail ops (e.g. lmhead+sampling)

    Returns:
        (
            df_model_compilation, df_model_trace,
            df_first_layer_compilation, df_first_layer_trace,
            df_mid_layers_compilation, df_mid_layers_trace,
            df_model_tail_compilation, df_model_tail_trace
        )
        Any of the additional outputs may be None if slicing arguments are not provided.
    """
    adjusted_len = len(df)
    first_run_end = int(adjusted_len / num_runs)
    last_run_start = int((num_runs - 1) * (adjusted_len / num_runs))

    df_model_compilation = df[:first_run_end]
    df_model_trace = df[last_run_start:]

    # Excluding model embeddings and tail ops
    # [op_start_index:op_end_index] = all core layers region
    op_start_index = num_head_ops
    op_end_index = len(df_model_compilation) - num_tail_ops
    df_layers_compilation = df_model_compilation[op_start_index:op_end_index]
    df_layers_trace = df_model_trace[op_start_index:op_end_index]

    # First layer: always first 'len/num_layers'
    split_point = int(len(df_layers_compilation) / num_layers)
    df_first_layer_compilation = df_layers_compilation[:split_point]
    df_first_layer_trace = df_layers_trace[:split_point]

    # Mid layers: remainder of layers region
    if num_layers > 1:
        df_mid_layers_compilation = df_layers_compilation[split_point:]
        df_mid_layers_trace = df_layers_trace[split_point:]
    else:
        df_mid_layers_compilation = None
        df_mid_layers_trace = None

    # Model tail ops (e.g. lmhead/sampling): [tail_start_index:]
    if tail_start_index is not None:
        df_model_tail_compilation = df_model_compilation[tail_start_index:]
        df_model_tail_trace = df_model_trace[tail_start_index:]
    else:
        df_model_tail_compilation = None
        df_model_tail_trace = None

    return (
        df_model_compilation,
        df_model_trace,
        df_first_layer_compilation,
        df_first_layer_trace,
        df_mid_layers_compilation,
        df_mid_layers_trace,
        df_model_tail_compilation,
        df_model_tail_trace,
    )


def verify_value_within_margin(value, target, margin, op_code_with_id, perf_type):
    upper_limit = target + margin * target
    lower_limit = target - margin * target

    passing = True

    if value > upper_limit:
        passing = False
        logger.warning(
            f"{op_code_with_id} {perf_type}: {value} ns is larger than target "
            f"({target}) ns, difference: "
            f"{abs(value - upper_limit)} ns, margin: "
            f"{margin}, "
            f"relative margin to pass would be: "
            f"{(abs(target - value) / target) if target != 0 else -1}"
        )
    elif value < lower_limit:
        passing = False
        logger.warning(
            f"{op_code_with_id} {perf_type}: {value} ns is smaller than target "
            f"({target}) ns, difference: "
            f"{abs(value - lower_limit)} ns, margin: "
            f"{margin}, "
            f"relative margin to pass would be: "
            f"{(abs(target - value) / target) if target != 0 else -1}"
        )
    return passing
