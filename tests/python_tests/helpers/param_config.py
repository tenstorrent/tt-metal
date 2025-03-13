# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from dataclasses import dataclass
from typing import List, Optional
from .format_config import FormatConfig, DataFormat


def manage_included_params(func):
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "included_params"):
            wrapper.included_params = []
        return func(wrapper.included_params, *args, **kwargs)

    return wrapper


@manage_included_params
def generate_format_combinations(
    included_params, formats: List[DataFormat], all_same: bool
) -> List[FormatConfig]:
    """
    Generates a list of FormatConfig instances based on the given formats and the 'all_same' flag.
    This is used for pytesting in order to utilize pytest.mark.parametrize to test different format combinations.

    If the 'all_same' flag is set to True, the function returns combinations where all format attributes are the same.
    If the 'all_same' flag is set to False, the function returns all possible combinations of formats for each attribute.
        This is good to use when looking to test on full format flush.

    Parameters:
    formats (List[DataFormat]): A list of formats that are supported for this test. Combinations are generated based on these formats.
    all_same (bool): A flag indicating whether to return combinations with all formats being the same
                     (True) or all possible combinations (False).

    Returns:
    List[FormatConfig]: A list of FormatConfig instances representing the generated format combinations.

    Example:
    >>> generate_format_combinations([DataFormat.Float16, DataFormat.Float32], True)
    [FormatConfig(unpack_src=DataFormat.Float16, unpack_dst=DataFormat.Float16, math=DataFormat.Float16, pack_src=DataFormat.Float16, pack_dst=DataFormat.Float16),
     FormatConfig(unpack_src=DataFormat.Float32, unpack_dst=DataFormat.Float32, math=DataFormat.Float32, pack_src=DataFormat.Float32, pack_dst=DataFormat.Float32)]

    >>> generate_format_combinations([DataFormat.Float16, "Float32"], False)
    [FormatConfig(unpack_src=DataFormat.Float16, unpack_dst=DataFormat.Float16, math=DataFormat.Float16, pack_src=DataFormat.Float16, pack_dst=DataFormat.Float16),
     FormatConfig(unpack_src=DataFormat.Float16, unpack_dst=DataFormat.Float16, math=DataFormat.Float16, pack_src=DataFormat.Float16, pack_dst=DataFormat.Float32),
     ...
     FormatConfig(unpack_src=DataFormat.Float32, unpack_dst=DataFormat.Float32, math=DataFormat.Float32, pack_src=DataFormat.Float32, pack_dst=DataFormat.Float32)]
    """
    if all_same:
        return [FormatConfig(fmt, fmt, fmt, fmt, fmt) for fmt in formats]
    return [
        FormatConfig(unpack_src, unpack_dst, math, pack_src, pack_dst)
        for unpack_src in formats
        for unpack_dst in formats
        for math in formats
        for pack_src in formats
        for pack_dst in formats
    ]


@manage_included_params
def generate_params(
    included_params,
    testnames: List[str],
    format_combos: List[FormatConfig],
    dest_acc: Optional[List[str]] = None,
    approx_mode: Optional[List[str]] = None,
    mathop: Optional[List[str]] = None,
    math_fidelity: Optional[List[int]] = None,
    tile_cnt: Optional[List[int]] = None,
    reduce_dim: Optional[List[str]] = None,
    pool_type: Optional[List[str]] = None,
) -> List[tuple]:
    """
    Generates a list of parameter combinations for test configurations.

    This function creates all possible combinations of the provided test parameters, including optional ones,
    while filtering out any None values. The function returns these combinations as tuples, which can be used
    for setting up tests or experiments.

    Returns:
    List[tuple]: A list of tuples, where each tuple represents a combination of parameters with any `None` values filtered out.

    Example:
    >>> testnames = ["Test1", "Test2"]
    >>> format_combos = [FormatConfig(DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16)]
    >>> generate_params(testnames, format_combos, dest_acc=["DEST_ACC"], approx_mode=["Approx1"])
    [
        ("Test1", FormatConfig(DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16), "DEST_ACC", "Approx1", None, None, None, None, None),
        ("Test2", FormatConfig(DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16), "DEST_ACC", "Approx1", None, None, None, None, None)
    ]
    """

    # Build a list of parameter names (`included_params`) that are non-None.
    # This allows later code in generate_param_ids(...) to conditionally include
    # only the parameters that were actually provided (not None) when generating the ID.
    included_params.extend(
        [
            param
            for param, value in [
                ("dest_acc", dest_acc),
                ("approx_mode", approx_mode),
                ("mathop", mathop),
                ("math_fidelity", math_fidelity),
                ("tile_cnt", tile_cnt),
                ("reduce_dim", reduce_dim),
                ("pool_type", pool_type),
            ]
            if value is not None
        ]
    )

    return [
        (
            testname,
            format_config,
            acc_mode,
            approx,
            math,
            fidelity,
            num_tiles,
            dim,
            pool,
        )
        for testname in testnames
        for format_config in format_combos
        for acc_mode in (dest_acc if dest_acc is not None else [None])
        for approx in (approx_mode if approx_mode is not None else [None])
        for math in (mathop if mathop is not None else [None])
        for fidelity in (math_fidelity if math_fidelity is not None else [None])
        for num_tiles in (tile_cnt if tile_cnt is not None else [None])
        for dim in (reduce_dim if reduce_dim is not None else [None])
        for pool in (pool_type if pool_type is not None else [None])
    ]


@manage_included_params
def clean_params(included_params, all_params: List[tuple]) -> List[tuple]:
    """
    Cleans up the list of parameter combinations by removing any `None` values.

    This function filters out any `None` values from the provided list of parameter combinations.
    It is used to clean up the list of parameters before generating parameter IDs for test cases.

    Returns:
    List[tuple]: A list of tuples, where each tuple represents a combination of parameters with any `None` values filtered out.

    Example:
    >>> all_params = [
    ...     ("Test1", FormatConfig(DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16), "Acc1", "Approx1", None, None, None, None, None),
    ...     ("Test2", FormatConfig(DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16), "Acc1", "Approx1", None, None, None, None, None)
    ... ]
    >>> clean_params(all_params)
    [
        ("Test1", FormatConfig(DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16), "Acc1", "Approx1"),
        ("Test2", FormatConfig(DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16, DataFormat.Float16), "Acc1", "Approx1")
    ]
    """

    return [tuple(param for param in comb if param is not None) for comb in all_params]


@manage_included_params
def generate_param_ids(included_params, all_params: List[tuple]) -> List[str]:
    """
    Generates a list of parameter IDs based on the provided parameter combinations.

    Creates a string ID for each combination, including only those parameters that are present in the `included_params` list.
    If a parameter is not included (i.e., it is `None`), it will be excluded from the final ID.

    Used to format output of our test cases in a more readable way. Function return is passed into `ids` parameter of `@pytest.mark.parametrize`.

    Parameters:
    all_params (List[tuple]): A list of tuples, where each tuple contains a combination of parameters.
                               The second element in the tuple is expected to be a `FormatConfig` object,
                               and the rest are the parameter values (like `dest_acc`, `approx_mode`, etc.).

    Returns:
    List[str]: A list of formatted strings representing each combination of parameters.

    Example:
    >>> all_params = [
    ...     ("Test1", FormatConfig("Float16", "Float16", "Float16", "Float16", "Float16"), "Acc1", "Approx1", "Add", 1, 4, "Dim1", "Pool1"),
    ...     ("Test2", FormatConfig("Float32", "Float32", "Float32", "Float32", "Float32"), "Acc2", None, "Mul", 2, 6, "Dim2", None)
    ... ]
    >>> generate_param_ids(all_params)
    [
        'unpack_src=Float16 | unpack_dst=Float16 | math=Add | pack_src=Float16 | pack_dst=Float16 | dest_acc=Acc1 | approx_mode=Approx1 | mathop=Add | math_fidelity=1 | tile_cnt=4 | reduce_dim=Dim1 | pool_type=Pool1',
        'unpack_src=Float32 | unpack_dst=Float32 | math=Mul | pack_src=Float32 | pack_dst=Float32 | dest_acc=Acc2 | mathop=Mul | math_fidelity=2 | tile_cnt=6 | reduce_dim=Dim2'
    ]
    """

    def format_combination(comb: tuple) -> str:
        """
        Helper function to format a single combination of parameters into a readable string.

        Args:
        comb (tuple): A tuple containing a combination of parameters, including the FormatConfig object.

        Returns:
        str: A formatted string of parameter names and values.
        """
        # Extract the FormatConfig and other parameters
        testname, format_config, *params = comb

        # Start with the FormatConfig information
        result = [
            f"unpack_src={format_config.unpack_src.value}",
            f"unpack_dst={format_config.unpack_dst.value}",
            f"math={format_config.math.value}",
            f"pack_src={format_config.pack_src.value}",
            f"pack_dst={format_config.pack_dst.value}",
        ]

        # Include optional parameters based on `included_params`
        param_names = [
            ("dest_acc", params[0] if params[0] else None),
            ("approx_mode", params[1] if params[1] else None),
            ("mathop", params[2] if params[2] else None),
            ("math_fidelity", params[3] if params[3] else None),
            ("tile_cnt", params[4] if params[4] else None),
            ("reduce_dim", params[5] if params[5] else None),
            ("pool_type", params[6] if params[6] else None),
        ]

        # Loop through the parameters and add them to the result if they are not None
        for param_name, value in param_names:
            if value is not None and param_name in included_params:
                result.append(f"| {param_name}={value}")

        # Join the result list into a single string with appropriate spacing
        return " | ".join(result)

    # Generate and return formatted strings for all parameter combinations
    return [format_combination(comb) for comb in all_params if comb[0] is not None]
