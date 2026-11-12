# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Iterator, Mapping

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import (
    BLACKHOLE_DATA_FORMAT_ENUM_VALUES,
    QUASAR_DATA_FORMAT_ENUM_VALUES,
    WORMHOLE_DATA_FORMAT_ENUM_VALUES,
)
from helpers.logger import logger

from .arch_common import _get_parser
from .config_parser import FUSER_CONFIG_DIR, FuserConfigSchema
from .fuser_config import FuserConfig
from .operand import L1_PACKERS
from .validator import SUPPORTED_TILE_SIZES

SweepValue = str | bool | int | float | tuple[int, int]
ConfigPath = tuple[str | int, ...]
AllValues = tuple[SweepValue, ...] | Callable[[dict], Iterable[SweepValue]]


@dataclass(frozen=True)
class SweepParameter:
    value_type: type[str] | type[bool] | type[int] | type[float] | type[tuple] = str
    all_values: AllValues | None = None
    node_types: tuple[str, ...] = ()
    id_field: str | None = None


@dataclass(frozen=True)
class _SweepAxis:
    path: ConfigPath
    label: str
    choices: tuple[SweepValue, ...]


SweepRegistry = Mapping[str, Mapping[str, SweepParameter]]


def _sfpu_operations(node: dict) -> list[str]:
    parser = _get_parser()
    supported_ops = {
        "UnarySfpu": parser.UNARY_SFPU_OPS,
        "BinarySfpu": parser.BINARY_SFPU_OPS,
        "TernarySfpu": parser.TERNARY_SFPU_OPS,
    }
    return sorted(op.name for op in supported_ops[node["type"]])


def _operand_formats(_node: dict) -> list[str]:
    formats = {
        ChipArchitecture.WORMHOLE: WORMHOLE_DATA_FORMAT_ENUM_VALUES,
        ChipArchitecture.BLACKHOLE: BLACKHOLE_DATA_FORMAT_ENUM_VALUES,
        ChipArchitecture.QUASAR: QUASAR_DATA_FORMAT_ENUM_VALUES,
    }[get_chip_architecture()]
    return sorted(fmt.name for fmt in formats if fmt in L1_PACKERS)


_SFPU_OPERATION = SweepParameter(
    all_values=_sfpu_operations,
    node_types=("UnarySfpu", "BinarySfpu", "TernarySfpu"),
    id_field="",
)

_SFPU_APPROXIMATION_MODE = SweepParameter(
    value_type=bool,
    all_values=(False, True),
    node_types=("UnarySfpu", "BinarySfpu", "TernarySfpu"),
)

# Register new fields here; discovery and expansion are shared by every parameter.
SWEEP_PARAMETERS: dict[str, dict[str, SweepParameter]] = {
    "config": {},
    "operation": {},
    "math": {
        "operation": _SFPU_OPERATION,
        "approximation_mode": _SFPU_APPROXIMATION_MODE,
        "unpack_to_dest": SweepParameter(
            value_type=bool, all_values=(False, True), node_types=("Fpu",)
        ),
    },
    "pack": {
        "operation": _SFPU_OPERATION,
        "approximation_mode": _SFPU_APPROXIMATION_MODE,
    },
    "operand": {
        "format": SweepParameter(all_values=_operand_formats),
        "tile_dims": SweepParameter(
            value_type=tuple, all_values=tuple(sorted(SUPPORTED_TILE_SIZES))
        ),
    },
}


def _iter_nodes(definition: dict) -> Iterator[tuple[str, ConfigPath, str, dict]]:
    yield "config", (), "", definition
    for stage_index, stage in enumerate(definition.get("operations", [])):
        path = ("operations", stage_index)
        label = f"op{stage_index + 1}"
        yield "operation", path, label, stage
        for section in ("math", "pack"):
            for node_index, node in enumerate(stage.get(section, [])):
                yield (
                    section,
                    (*path, section, node_index),
                    f"{label}_{section}{node_index + 1}",
                    node,
                )
    for index, operand in enumerate(definition.get("operands", [])):
        yield "operand", ("operands", index), f"operand{index + 1}", operand


def _sweep_choices(
    value, parameter: SweepParameter, node: dict, path: ConfigPath
) -> tuple[SweepValue, ...] | None:
    location = ".".join(str(key) for key in path)
    if value == "all":
        provider = parameter.all_values
        if provider is None:
            raise ValueError(f"{location}: 'all' is not defined; use an explicit list")
        values = list(provider(node) if callable(provider) else provider)
    elif isinstance(value, list):
        if (
            value
            and parameter.value_type is tuple
            and all(type(item) is int for item in value)
        ):
            return None
        values = value
    else:
        return None

    if not values:
        raise ValueError(f"{location}: sweep choices cannot be empty")
    if parameter.value_type is tuple:
        if any(
            not isinstance(choice, (list, tuple))
            or len(choice) != 2
            or any(type(dim) is not int for dim in choice)
            for choice in values
        ):
            raise ValueError(f"{location}: sweep choices must be [rows, cols] pairs")
        values = [tuple(choice) for choice in values]
    if any(type(choice) is not parameter.value_type for choice in values):
        raise ValueError(
            f"{location}: sweep choices must be {parameter.value_type.__name__} values"
        )
    return tuple(dict.fromkeys(values))


def _iter_axes(definition: dict, parameters: SweepRegistry) -> Iterator[_SweepAxis]:
    for scope, path, label, node in _iter_nodes(definition):
        for field, parameter in parameters.get(scope, {}).items():
            if parameter.node_types and node.get("type") not in parameter.node_types:
                continue
            field_path = (*path, field)
            choices = _sweep_choices(node.get(field), parameter, node, field_path)
            if choices is None:
                continue
            id_field = field if parameter.id_field is None else parameter.id_field
            yield _SweepAxis(
                path=field_path,
                label="_".join(part for part in (label, id_field) if part),
                choices=choices,
            )


def _compatible_tile_shapes(definition: dict, shapes: dict) -> bool:
    """Prune geometry mismatches before copying or constructing a fuser config."""
    for operation in definition.get("operations", []):
        outputs = [
            shapes.get(p["output"]) for p in operation.get("pack", []) if "output" in p
        ]
        for node in operation.get("math", []):
            if node.get("in0") is None:
                continue
            src_a = shapes.get(node["in0"])
            src_b = shapes.get(node.get("in1") or node["in0"])
            if src_a is None or src_b is None:
                return True  # Let schema validation report undeclared operands.
            if node.get("operation") in ("Matmul", "MatmulNoMop"):
                if src_a[1] != src_b[0]:
                    return False
                outputs.append((src_a[0], src_b[1]))
            else:
                if src_a != src_b:
                    return False
                outputs.append(src_a)
        if None in outputs:
            return True
        if outputs and any(shape != outputs[0] for shape in outputs[1:]):
            return False
    return True


def _tile_shape_suffix(shapes: dict) -> list[str]:
    if not shapes or all(shape == (32, 32) for shape in shapes.values()):
        return []
    if len(set(shapes.values())) == 1:
        rows, cols = next(iter(shapes.values()))
        return [f"tiles_{rows}x{cols}"]
    return [f"{name}_{rows}x{cols}" for name, (rows, cols) in shapes.items()]


def expand_fuser_configs(
    test_name: str,
    definition: dict,
    *,
    parameters: SweepRegistry | None = None,
) -> Iterator[tuple[str, dict]]:
    if parameters is None:
        parameters = SWEEP_PARAMETERS
    axes = list(_iter_axes(definition, parameters))
    shape_axes = [
        axis
        for axis in axes
        if axis.path[0] == "operands" and axis.path[-1] == "tile_dims"
    ]
    other_axes = [axis for axis in axes if axis not in shape_axes]
    for shape_choices in product(*(axis.choices for axis in shape_axes)):
        selected = {
            definition["operands"][axis.path[1]]["name"]: shape
            for axis, shape in zip(shape_axes, shape_choices)
        }
        shapes = {
            operand["name"]: tuple(operand.get("tile_dims") or (32, 32))
            for operand in definition.get("operands", [])
            if operand["name"] not in selected
        }
        shapes.update(selected)
        if shape_axes and not _compatible_tile_shapes(definition, shapes):
            continue
        yield from _expand_axes(
            test_name,
            definition,
            shape_axes,
            shape_choices,
            other_axes,
            _tile_shape_suffix(selected),
        )


def _expand_axes(
    test_name, definition, shape_axes, shape_choices, other_axes, shape_suffix
):
    for choices in product(*(axis.choices for axis in other_axes)):
        config_dict = deepcopy(definition)
        suffix = list(shape_suffix)
        for axis, value in zip(shape_axes + other_axes, shape_choices + choices):
            node = config_dict
            for key in axis.path[:-1]:
                node = node[key]
            node[axis.path[-1]] = value
            if axis in shape_axes:
                continue
            value_id = "x".join(map(str, value)) if isinstance(value, tuple) else value
            # The original datacopy/SFPU cases used unpack_to_dest: true.
            if axis.path[-1] != "unpack_to_dest" or value is not True:
                suffix.append(f"{axis.label}_{value_id}")
        case_name = "__".join([test_name, *suffix])
        yield case_name, config_dict


def validated_fuser_configs(
    test_name: str, definition: dict
) -> Iterator[tuple[str, FuserConfig]]:
    """Build valid cases before parametrization, filtering rejected sweep combinations."""
    supported_archs = definition.get("supported_archs")
    if (
        supported_archs is not None
        and get_chip_architecture().value not in supported_archs
    ):
        return

    valid_count = 0
    rejected_count = 0
    has_sweep = any(_iter_axes(definition, SWEEP_PARAMETERS))
    for case_name, config_dict in expand_fuser_configs(test_name, definition):
        try:
            config = FuserConfigSchema.load(case_name, config_dict)
        except ValueError as error:
            if not has_sweep:
                raise
            rejected_count += 1
            logger.debug(f"Excluding invalid fuser sweep case {case_name}: {error}")
            continue
        valid_count += 1
        yield case_name, config

    if not valid_count:
        raise ValueError(
            f"Fuser sweep '{test_name}' has no valid cases ({rejected_count} rejected)"
        )


def collect_fuser_cases(yaml_files):
    cases = {}
    for yaml_path in yaml_files:
        test_name = str(yaml_path.relative_to(FUSER_CONFIG_DIR).with_suffix(""))
        definition = FuserConfigSchema.load_definition(test_name)
        cases.update(validated_fuser_configs(test_name, definition))
    return cases
