# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger


def clamp(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    return value


def is_default_value(values, default):
    """Check if values match a default, handling None, scalar, and iterable inputs."""
    if values is None:
        return True
    if isinstance(values, (int, float)):
        return values == default
    return all(value == default for value in values)


def filter_none(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def split_list(lst, n):
    """Split list into n equal parts."""
    chunk_size = len(lst) // n
    return [list(lst[i * chunk_size : (i + 1) * chunk_size]) for i in range(n)]


def compact_debug_list(values, max_items=12):
    if values is None:
        return None
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        values = values.reshape(-1).tolist()
    elif isinstance(values, tuple):
        values = list(values)
    elif not isinstance(values, list):
        values = list(values) if isinstance(values, range) else [values]
    if len(values) <= max_items:
        return values
    half = max(1, max_items // 2)
    return {"len": len(values), "head": values[:half], "tail": values[-half:]}


def is_llama33_70b_model(args) -> bool:
    if isinstance(args, list):
        args = args[0] if args else None
    if args is None:
        return False

    fields_to_check = (
        "model_name",
        "base_model_name",
        "model_base_path",
        "model_cache_path",
        "tokenizer_path",
        "CKPT_DIR",
        "LLAMA_DIR",
        "hf_model",
        "HF_MODEL",
    )
    for field in fields_to_check:
        value = getattr(args, field, None)
        if value is None:
            continue
        normalized = str(value).lower().replace("_", "-")
        if "llama" in normalized and "3.3-70b" in normalized:
            return True
    return False


def log_sampling_debug(enabled, message, **kwargs):
    if not enabled:
        return
    compact = {key: value for key, value in kwargs.items() if value is not None}
    logger.info(f"SamplingDBG {message}: {compact}")


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def upper_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
