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

    # Ensure string message
    message = str(message)

    key = "DEVICE FW DURATION [ns]"

    def _append_metrics(base_message, duration_ns, label_suffix=None):
        """Append ROOFLINE%/TFLOPS using provided duration. Optionally label suffix (e.g., "_cached")."""
        try:
            value = float(duration_ns)
        except Exception:
            return base_message

        tokens = base_message.split()
        # Build once from the original tokens to avoid cascading on appended tokens
        added_parts = []
        for i in range(len(tokens)):
            if tokens[i] == "ROOFLINE":
                if (i + 1) < len(tokens):
                    suffix = label_suffix or ""
                    added_parts.append(f" ROOFLINE%{suffix} {float(tokens[i+1])/value}")
            if tokens[i] == "FLOP_COUNT":
                if (i + 1) < len(tokens):
                    suffix = label_suffix or ""
                    # flop_count / tera(10^12)*nano(10^-9) / ns
                    added_parts.append(f" TFLOPS{suffix} {float(tokens[i+1])/1e3/value}")
        if added_parts:
            logger.info("Message processing complete")
        return base_message + "".join(added_parts)

    # Handle nested cached/uncached structure
    if isinstance(perf_result, dict) and ("uncached" in perf_result or "cached" in perf_result):
        updated = message
        uncached_dict = perf_result.get("uncached") if isinstance(perf_result.get("uncached"), dict) else None
        cached_dict = perf_result.get("cached") if isinstance(perf_result.get("cached"), dict) else None

        # Prefer uncached as canonical (unlabeled). If missing, fall back to cached as canonical.
        canonical = uncached_dict if uncached_dict is not None else cached_dict
        if isinstance(canonical, dict) and key in canonical:
            updated = _append_metrics(updated, canonical[key], label_suffix=None)

        # Add the non-canonical variant with labels for clarity
        if uncached_dict is not None and uncached_dict is not canonical and key in uncached_dict:
            updated = _append_metrics(updated, uncached_dict[key], label_suffix="_uncached")
        if cached_dict is not None and cached_dict is not canonical and key in cached_dict:
            updated = _append_metrics(updated, cached_dict[key], label_suffix="_cached")

        return updated

    # Flat dict behavior (backward compatible)
    if isinstance(perf_result, dict) and key in perf_result:
        try:
            tokens = message.split()
            logger.info(f"Processing message with {len(tokens)} tokens")
        except Exception:
            tokens = message.split()

        return _append_metrics(message, perf_result[key], label_suffix=None)

    return message
