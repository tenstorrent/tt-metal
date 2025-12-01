# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import inspect
from itertools import product
from typing import Iterator, List, Tuple

import pytest
from typing_extensions import deprecated

from .format_config import (
    DataFormat,
    FormatConfig,
    InputOutputFormat,
)
from .llk_params import DestAccumulation, DestSync

checked_formats_and_dest_acc = {}


def format_combination_sweep(
    formats: List[DataFormat],
    all_same: bool,
    same_src_reg_format: bool = True,
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
    >>> format_combination_sweep([DataFormat.Float16, DataFormat.Float32], True)
    [FormatConfig(unpack_src=DataFormat.Float16, unpack_dst=DataFormat.Float16, math=DataFormat.Float16, pack_src=DataFormat.Float16, pack_dst=DataFormat.Float16),
     FormatConfig(unpack_src=DataFormat.Float32, unpack_dst=DataFormat.Float32, math=DataFormat.Float32, pack_src=DataFormat.Float32, pack_dst=DataFormat.Float32)]

    >>> format_combination_sweep([DataFormat.Float16, "Float32"], False)
    [FormatConfig(unpack_src=DataFormat.Float16, unpack_dst=DataFormat.Float16, math=DataFormat.Float16, pack_src=DataFormat.Float16, pack_dst=DataFormat.Float16),
     FormatConfig(unpack_src=DataFormat.Float16, unpack_dst=DataFormat.Float16, math=DataFormat.Float16, pack_src=DataFormat.Float16, pack_dst=DataFormat.Float32),
     ...
     FormatConfig(unpack_src=DataFormat.Float32, unpack_dst=DataFormat.Float32, math=DataFormat.Float32, pack_src=DataFormat.Float32, pack_dst=DataFormat.Float32)]
    """
    if all_same:
        return [
            FormatConfig(
                unpack_A_src=fmt,
                unpack_A_dst=fmt,
                math=fmt,
                pack_src=fmt,
                pack_dst=fmt,
                same_src_format=same_src_reg_format,
            )
            for fmt in formats
        ]
    return [
        FormatConfig(
            unpack_A_src=unpack_src,
            unpack_A_dst=unpack_dst,
            math=math,
            pack_src=pack_src,
            pack_dst=pack_dst,
            same_src_format=same_src_reg_format,
        )
        for unpack_src in formats
        for unpack_dst in formats
        for math in formats
        for pack_src in formats
        for pack_dst in formats
    ]


# When reading this file, keep in mind that parameter means FORMAL PARAMETER and argument means ACTUAL PARAMETER


class UnknownDependenciesError(Exception):
    """Raised when a dependency is not a known parameter."""

    @staticmethod
    def _format_dependencies(dependencies: set[str]) -> str:
        return "\n".join([f"    - {dependency}" for dependency in dependencies])

    @staticmethod
    def _format_parameter(parameter: str, dependencies: set[str]) -> str:
        dependencies_list = UnknownDependenciesError._format_dependencies(dependencies)
        return f"- {parameter} has missing dependencies:\n{dependencies_list}"

    @staticmethod
    def _format_parameters(parameters: dict[str, set[str]]) -> str:
        return "\n".join(
            [
                UnknownDependenciesError._format_parameter(parameter, dependencies)
                for parameter, dependencies in parameters.items()
            ]
        )

    def __init__(self, missing: dict[str, set[str]]):
        self.missing = missing
        parameters_list = UnknownDependenciesError._format_parameters(missing)
        super().__init__(
            f"Following parameters have unknown dependencies:\n{parameters_list}"
        )


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected among: \n[{', '.join(cycle)}]")


class ResolutionError(Exception):
    """Raised when a resolution error is detected."""

    def __init__(self, error: Exception):
        self.error = error
        super().__init__(f"Constraint function raised an exception: {error}")


def _param_dependencies(parameter: str, argument: any) -> List[str]:
    """Extract parameter names from a callable using introspection."""
    if callable(argument):
        dependencies = inspect.signature(argument).parameters.keys()
        return list(dependencies)
    return []


def _verify_dependency_map(dependency_map: dict[str, list[str]]) -> None:
    """
    Verifies that all dependencies are known parameters.
    """
    parameters = set(dependency_map.keys())

    missing = {
        parameter: difference
        for parameter, dependencies in dependency_map.items()
        if (difference := set(dependencies) - parameters)
    }

    if missing:
        raise UnknownDependenciesError(missing)


def _compute_dependency_map(**params: any) -> dict[str, list[str]]:
    dependency_map = {
        param: _param_dependencies(param, value) for param, value in params.items()
    }

    _verify_dependency_map(dependency_map)

    return dependency_map


def _compute_dependency_matrix(**params: any) -> list[list[int]]:
    param_idx = {param: idx for idx, param in enumerate(params.keys())}
    dependency_map = _compute_dependency_map(**params)

    return [
        [param_idx[dependency] for dependency in dependency_map[param]]
        for param in params.keys()
    ]


def _find_next_resolvable(matrix: list[list[int]], resolved: list[bool]) -> list[int]:
    """
    Returns a list of indices that became resolvable after one iteration of propagation.
    """

    def _is_resolvable_now(idx: int) -> bool:
        if resolved[idx]:
            return False

        for dep in matrix[idx]:
            if not resolved[dep]:
                return False

        return True

    return [idx for idx in range(len(matrix)) if _is_resolvable_now(idx)]


def _compute_resolution_order(
    parameter_names: list[str], dependency_matrix: list[list[int]]
) -> List[int]:
    """
    Builds a map of parameters used to resolve the constrained cartesian product

    The ordering of the keys in the map is the order in which the parameters are resolved.

    The key (int) is the index of the parameter in the result tuple.
    The values (set[int]) are the indices of the parameters on which the current parameter depends.

    """

    topological = []
    resolved = [False] * len(dependency_matrix)

    while len(topological) < len(dependency_matrix):

        resolvable = _find_next_resolvable(dependency_matrix, resolved)

        if not resolvable:
            unresolved = [
                parameter_names[i]
                for i, is_resolved in enumerate(resolved)
                if not is_resolved
            ]
            raise CircularDependencyError(unresolved)

        for idx in resolvable:
            resolved[idx] = True
            topological.append(idx)

    return topological


def _params_solve_dependencies(**kwargs: any) -> List[Tuple]:
    """
    Compute constrained cartesian product by resolving parameter dependencies.

    Uses recursive backtracking to generate all valid combinations where
    callable parameters can depend on previously resolved parameters.

    Returns:
        List of tuples representing all valid parameter combinations
    """
    parameters = tuple(range(len(kwargs)))
    arguments = tuple(kwargs.values())

    dependency_matrix = _compute_dependency_matrix(**kwargs)
    resolution_order = _compute_resolution_order(parameters, dependency_matrix)

    def _resolve_param_values(resolved: list[any], parameter: int) -> list:
        """Get possible values for a parameter given resolved dependencies."""
        argument = arguments[parameter]

        if callable(argument):
            dependencies = dependency_matrix[parameter]
            dependency_values = [resolved[dependency] for dependency in dependencies]

            try:
                result = argument(*dependency_values)
            except Exception as ex:
                raise ResolutionError(ex)

            # if constraint function returns a single value, wrap it in a list
            if not isinstance(result, list):
                return [result]

            return result

        if isinstance(argument, list):
            return argument

        return [argument]

    def _solve_recursive(resolved: list[any], resolution_index: int) -> Iterator[Tuple]:
        if resolution_index >= len(resolution_order):
            yield tuple(resolved)
            return

        parameter = resolution_order[resolution_index]
        arguments = _resolve_param_values(resolved, parameter)

        for argument in arguments:
            resolved[parameter] = argument
            yield from _solve_recursive(resolved, resolution_index + 1)

    # Initialize resolved list with None values
    resolved = [None] * len(parameters)
    return list(_solve_recursive(resolved, 0))


def parametrize(**kwargs: any):
    parameters = tuple(kwargs.keys())
    parameters_string = ",".join(parameters)
    parameter_values = _params_solve_dependencies(**kwargs)

    def decorator(test_function):
        return pytest.mark.parametrize(parameters_string, parameter_values)(
            test_function
        )

    return decorator


@deprecated("Try using parametrize or python inbuilt product function")
def generate_params(**kwargs: any) -> List[tuple]:
    wrap_list = lambda x: [x] if not isinstance(x, list) else x
    arguments = [wrap_list(value) for value in kwargs.values() if value is not None]

    return product(*arguments)


def input_output_formats(
    formats: List[DataFormat], same: bool = False
) -> List[InputOutputFormat]:
    """
    Generates a list of InputOutputFormat instances based on the given formats.
    This function is used to create input-output format combinations for testing.
    Parameters:
    formats (List[DataFormat]): A list of formats that are supported for this test.
    Returns:
    List[InputOutputFormat]: A list of InputOutputFormat instances representing the generated format combinations.
    """
    if same:
        return [InputOutputFormat(input, input) for input in formats]
    return [InputOutputFormat(input, output) for input in formats for output in formats]


def generate_combination(formats: List[Tuple[DataFormat]]) -> List[FormatConfig]:
    """
    A function that creates a list of FormatConfig objects from a list of DataFormat objects that client wants to test.
    This function is useful for creating a list of FormatConfig objects for testing multiple formats combinations
    and cases which the user has specifically defined and wants to particularly test instead of a full format flush.
    Args:
    formats (List[Tuple[DataFormat]]): A list of tuples of DataFormat objects for which FormatConfig objects need to be created.
    Returns:
    List[FormatConfig]: A list of FormatConfig objects created from the list of DataFormat objects passed as input.
    Example:
    >>> formats = [(DataFormat.Float16, DataFormat.Float32, DataFormat.Float16, DataFormat.Float32, DataFormat.Float32)]
    >>> format_configs = generate_combination(formats)
    >>> print(format_configs[0].unpack_A_src)
    DataFormat.Float16
    >>> print(format_configs[0].unpack_B_src)
    DataFormat.Float16
    """
    return [
        (
            FormatConfig(
                unpack_A_src=tuple[0],
                unpack_A_dst=tuple[1],
                pack_src=tuple[2],
                pack_dst=tuple[3],
                math=tuple[4],
            )
            if len(tuple) == 5
            else FormatConfig(
                unpack_A_src=tuple[0],
                unpack_A_dst=tuple[1],
                unpack_B_src=tuple[2],
                unpack_B_dst=tuple[3],
                pack_src=tuple[4],
                pack_dst=tuple[5],
                math=tuple[6],
                same_src_format=False,
            )
        )
        for tuple in formats
    ]


def calculate_edgecase_dest_indices(
    dest_acc: bool, result_tiles: int, dest_sync_modes: List[DestSync] = [DestSync.Half]
):
    """
    Generate the lowest and highest possible dest index depending on the DestSync mode and whether dest is 32bit or not.

    Key rules:
    1. The lowest possible dest index is always 0.
    2. When DestSync.Half:  max_dst_tiles=8 (if dest is 16bit) or max_dst_tiles=4 (if dest is 32bit)
    3. When DestSync.Full:  max_dst_tiles=16 (if dest is 16bit) or max_dst_tiles=8 (if dest is 32bit)

    Args:
        dest_acc: Dest 16/32 bit mode, has to match is_fp32_dest_acc_en from C++
        result_tiles: Number of tiles in the result matrix
        dest_sync_modes: List of DestSync modes to generate indices for. If None, uses [DestSync.Half]

    Returns:
        List of tuples: (dest_sync, dst_index)
    """

    combinations = []

    DEST_SYNC_TILE_LIMITS = {
        DestSync.Half: 8,
        DestSync.Full: 16,
    }

    capacity_divisor = 2 if dest_acc else 1

    for dest_sync in dest_sync_modes:
        base_tile_limit = DEST_SYNC_TILE_LIMITS[dest_sync]
        max_tiles = base_tile_limit // capacity_divisor
        max_index = max_tiles - result_tiles

        if max_index < 0:
            raise ValueError(
                f"Too many result tiles ({result_tiles}) for destination capacity ({max_tiles}) with {dest_sync.name}"
            )

        # Add both combinations: lowest possible index = 0 and at max possible index
        # If max_index = 0 add only (dest_sync, 0) to avoid duplicates
        combinations.extend([(dest_sync, 0)])
        if max_index != 0:
            combinations.extend([(dest_sync, max_index)])

    return combinations


def get_max_dst_index(dest_sync: DestSync, dest_acc: bool, result_tiles: int) -> int:
    DEST_SYNC_TILE_LIMITS = {
        DestSync.Half: 8 if not dest_acc else 4,
        DestSync.Full: 16 if not dest_acc else 8,
    }
    return max(DEST_SYNC_TILE_LIMITS[dest_sync] - result_tiles, 0)


def generate_unary_input_dimensions(dest_acc, dest_sync=DestSync.Half):
    """Generate all possible input dimensions for unary operations.
    These dimensions are determined by the number of tiles that can fit into dest, which is determined by dest_sync and dest_acc.
    The generated input dimensions should ensure that all of the data fits into dest without any overflow when running unary operations.

    Key rules:
    1. When DestSync.Half:  max_tiles_in_dest=8 (if dest is 16bit) or max_tiles_in_dest=4 (if dest is 32bit)
    2. When DestSync.Full:  max_tiles_in_dest=16 (if dest is 16bit) or max_tiles_in_dest=8 (if dest is 32bit)

    Args:
        dest_acc: Dest 16/32 bit mode
        dest_sync: DestSync mode. Defaults to DestSync.Half

    Returns:
        List of input dimensions
    """

    DEST_SYNC_TILE_LIMITS = {
        DestSync.Half: 8,
        DestSync.Full: 16,
    }
    capacity_divisor = 2 if dest_acc == DestAccumulation.Yes else 1
    max_tiles_in_dest = DEST_SYNC_TILE_LIMITS[dest_sync] // capacity_divisor

    num_tile_rows = 32
    num_tile_cols = 32

    return [
        [row * num_tile_rows, column * num_tile_cols]
        for row in range(1, max_tiles_in_dest + 1)
        for column in range(1, (max_tiles_in_dest // row) + 1)
    ]
