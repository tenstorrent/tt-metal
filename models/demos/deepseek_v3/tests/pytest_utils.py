# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

DEFAULT_PREFILL_SEQ_LEN = 128


def expand_test_cases_with_position_ids_ranges(base_cases):
    """
    Expand test cases where position_ids ranges are expanded into individual test cases.

    Args:
        base_cases: List of tuples (mode, seq_len, batch_size_per_row, decode_position_ids)
            where decode_position_ids can be:
            - None: random position_ids
            - int: single position_id
            - tuple(start, end): range from start to end (inclusive), step=1
            - tuple(start, end, step): range from start to end (inclusive) with given step

    Returns:
        List of expanded test cases with individual position_ids

    Examples:
        >>> base_cases = [
        ...     ("decode", 1, 32, None),  # Random position_ids
        ...     ("decode", 1, 32, 1024),  # Single position_id
        ...     ("decode", 1, 32, (4096, 4100, 2)),  # Range with step=2: 4096, 4098, 4100
        ... ]
        >>> expand_test_cases_with_position_ids_ranges(base_cases)
        [
            ("decode", 1, 32, None),
            ("decode", 1, 32, 1024),
            ("decode", 1, 32, 4096),
            ("decode", 1, 32, 4098),
            ("decode", 1, 32, 4100),
        ]
    """

    def _process_tuple(mode, seq_len, batch_size_per_row, decode_position_ids):
        expanded_cases = []
        if isinstance(decode_position_ids, tuple):
            if len(decode_position_ids) == 2:
                # Expand range into individual position_ids with step=1
                start, end = decode_position_ids
                step = 1
            elif len(decode_position_ids) == 3:
                # Expand range into individual position_ids with given step
                start, end, step = decode_position_ids
                if step <= 0:
                    raise ValueError(f"step must be > 0, got {step}")
            else:
                raise ValueError(
                    f"Invalid range format: {decode_position_ids}. Expected (start, end) or (start, end, step)"
                )

            for pos_id in range(start, end + 1, step):
                expanded_cases.append((mode, seq_len, batch_size_per_row, pos_id))
        else:
            expanded_cases.append((mode, seq_len, batch_size_per_row, decode_position_ids))

        return expanded_cases

    expanded_cases = []
    for sample in base_cases:
        values = getattr(sample, "values", None)
        is_pytest_param = values is not None and hasattr(sample, "marks") and hasattr(sample, "id")
        if is_pytest_param:
            mode, seq_len, batch_size_per_row, decode_position_ids = values
        else:
            mode, seq_len, batch_size_per_row, decode_position_ids = sample

        cur_cases = _process_tuple(mode, seq_len, batch_size_per_row, decode_position_ids)
        if is_pytest_param:
            cur_cases = [pytest.param(*case, marks=sample.marks, id=sample.id) for case in cur_cases]
        expanded_cases.extend(cur_cases)

    return expanded_cases


def build_expanded_test_ids(expanded_cases):
    """
    Build pytest ids for expanded test cases.

    This is intended to be used with the output of
    `expand_test_cases_with_position_ids_ranges`, so the IDs align with the
    expanded (mode, seq_len, batch_size_per_row, decode_position_ids) tuples.
    For prefill cases, `decode_position_ids` is not applicable and is omitted
    from the generated ID.

    Examples:
        >>> expanded_cases = [
        ...     ("decode", 1, 32, None),
        ...     ("decode", 1, 32, 1024),
        ...     ("prefill", 128, 1, None),
        ... ]
        >>> build_expanded_test_ids(expanded_cases)
        [
            "mode_decode_seq_1_batch_32_pos_random",
            "mode_decode_seq_1_batch_32_pos_1024",
            "mode_prefill_seq_128_batch_1",
        ]
    """
    expanded_ids = []
    for val in expanded_cases:
        if not isinstance(val, tuple) and getattr(val, "values", None) is None:
            expanded_ids.append(str(val))
            continue

        values = getattr(val, "values", None)
        if values is not None:
            mode, seq_len, batch_size_per_row, decode_pos = values
        else:
            mode, seq_len, batch_size_per_row, decode_pos = val
        if mode == "decode":
            pos_str = decode_pos if decode_pos is not None else "random"
            expanded_ids.append(f"mode_{mode}_seq_{seq_len}_batch_{batch_size_per_row}_pos_{pos_str}")
        else:
            expanded_ids.append(f"mode_{mode}_seq_{seq_len}_batch_{batch_size_per_row}")
    return expanded_ids


def get_base_test_cases(users_per_row, prefill_seq_len, include_decode_random_pos_ids=True):
    """
    Build base test cases for decode and prefill paths.

    This helper is only exercised by these tests.:
        - models/demos/deepseek_v3/tests/test_mla.py
        - models/demos/deepseek_v3/tests/test_decoder_block.py
        - models/demos/deepseek_v3/tests/test_model.py

    Args:
        users_per_row: Number of users per row (USERS_PER_ROW).
        prefill_seq_len: Prefill sequence length to use when DEEPSEEK_MAX_SEQ_LEN_OVERRIDE is not set.
        include_decode_random_pos_ids: If True, include ("decode", 1, users_per_row, None).

    The environment variable DEEPSEEK_MAX_SEQ_LEN_OVERRIDE is primarily a CI override to expand
    prefill and decode coverage.

    Behavior:
        - Decode cases:
          - when DEEPSEEK_MAX_SEQ_LEN_OVERRIDE is set, additionally includes:
            - position_id 0
            - position_id max_seq_len - 1
        - Prefill cases:
          - when DEEPSEEK_MAX_SEQ_LEN_OVERRIDE is not set, includes one prefill case using
            prefill_seq_len: ("prefill", prefill_seq_len, 1, None)
          - when DEEPSEEK_MAX_SEQ_LEN_OVERRIDE is set, replaces the prefill list with a single case:
            ("prefill", max_seq_len, 1, None)

    """
    base_cases = []
    if include_decode_random_pos_ids:
        base_cases += [("decode", 1, users_per_row, None)]

    max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
    if max_seq_len_env is None:
        # If DEEPSEEK_MAX_SEQ_LEN_OVERRIDE is not set, use the default prefill sequence length.
        base_cases += [("prefill", prefill_seq_len, 1, None)]
    else:
        # If DEEPSEEK_MAX_SEQ_LEN_OVERRIDE is set, use it to expand prefill and decode coverage.
        max_seq_len = int(max_seq_len_env)
        base_cases += [
            ("decode", 1, users_per_row, 0),  # decode position_id 0
            ("decode", 1, users_per_row, max_seq_len - 1),  # decode position_id max_seq_len - 1
            ("prefill", max_seq_len, 1, None),  # prefill at max_seq_len
        ]
    return base_cases


def build_test_cases_and_ids(users_per_row, prefill_seq_len, include_decode_random_pos_ids=True):
    """
    Build base test cases and return expanded cases with matching pytest IDs.

    This combines:
      - get_base_test_cases
      - expand_test_cases_with_position_ids_ranges
      - build_expanded_test_ids
    """
    base_cases = get_base_test_cases(users_per_row, prefill_seq_len, include_decode_random_pos_ids)
    expanded_cases = expand_test_cases_with_position_ids_ranges(base_cases)
    expanded_ids = build_expanded_test_ids(expanded_cases)
    return expanded_cases, expanded_ids
