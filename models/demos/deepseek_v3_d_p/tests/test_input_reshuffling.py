# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger


def _format_device_tokens(values: list[int]) -> str:
    parts = []
    i = 0
    n = len(values)
    while i < n:
        if values[i] == -1:
            j = i
            while j < n and values[j] == -1:
                j += 1
            parts.append(f"-1({j - i})")
            i = j
        else:
            j = i + 1
            while j < n and values[j] != -1 and values[j] == values[j - 1] + 1:
                j += 1
            if j - i == 1:
                parts.append(str(values[i]))
            else:
                parts.append(f"{values[i]}..{values[j - 1]}")
            i = j
    return "[" + ", ".join(parts) + "]"


def pad_and_reshuffle(
    tokens: torch.Tensor,
    number_of_padded_tokens: int,
    num_sp_devices: int,
    num_tokens_stored_in_kv_cache: int,
) -> torch.Tensor:
    pad_token_value = -1

    if number_of_padded_tokens > tokens.shape[0]:
        tokens = torch.cat(
            [
                tokens,
                torch.full(
                    (number_of_padded_tokens - tokens.shape[0],),
                    pad_token_value,
                    dtype=tokens.dtype,
                ),
            ]
        )

    isl_per_chip = number_of_padded_tokens // num_sp_devices

    # Global position of input token i (after padding) is num_tokens_stored_in_kv_cache + i.
    # In a block-sequential layout that wraps round-robin, that position lives on:
    #   device = ((kv + i) // isl_per_chip) % num_sp_devices
    positions = torch.arange(number_of_padded_tokens, dtype=torch.int64)
    device_of_token = ((num_tokens_stored_in_kv_cache + positions) // isl_per_chip) % num_sp_devices

    # Stable sort groups tokens by device while preserving natural receive order
    # within each device (so dev 0's wrap-around chunk comes after its first chunk).
    sort_order = torch.argsort(device_of_token, stable=True)
    out = tokens[sort_order]

    sorted_devs = device_of_token[sort_order]
    cursor = 0
    for d in range(num_sp_devices):
        count = int((sorted_devs == d).sum().item())
        device_tokens = out[cursor : cursor + count].tolist()
        print(f"device {d}: {count} tokens -> {_format_device_tokens(device_tokens)}")
        cursor += count

    return out


SEQ_LEN_5K = 5 * 1024


@pytest.mark.parametrize("number_of_padded_tokens", [SEQ_LEN_5K])
@pytest.mark.parametrize("actual_sequence_length", [SEQ_LEN_5K - 32, SEQ_LEN_5K, 100, 3200])
@pytest.mark.parametrize("num_sp_devices", [8])
@pytest.mark.parametrize("num_tokens_stored_in_kv_cache", [0, 64, 128, 640, 5120, 65216])
def test_input_reshuffling(
    number_of_padded_tokens, actual_sequence_length, num_sp_devices, num_tokens_stored_in_kv_cache
):
    assert (
        actual_sequence_length <= number_of_padded_tokens
    ), "actual_sequence_length must be less than or equal to number_of_padded_tokens"
    assert num_tokens_stored_in_kv_cache % 32 == 0, "num_tokens_stored_in_kv_cache must be divisible by 32"
    assert number_of_padded_tokens % num_sp_devices == 0, "number_of_padded_tokens must be divisible by num_sp_devices"

    # for wraparound purposes, we just need 5k portion of the last kv cache chunk
    num_tokens_stored_in_kv_cache = num_tokens_stored_in_kv_cache % number_of_padded_tokens

    tokens = torch.arange(0, actual_sequence_length, dtype=torch.int32)
    logger.debug(f"Input tokens shape: {tokens.shape}")

    out = pad_and_reshuffle(tokens, number_of_padded_tokens, num_sp_devices, num_tokens_stored_in_kv_cache)

    isl_per_chip = number_of_padded_tokens // num_sp_devices
    print(f"isl_per_chip: {isl_per_chip}")
    print(out)

    assert out.shape == (number_of_padded_tokens,)
