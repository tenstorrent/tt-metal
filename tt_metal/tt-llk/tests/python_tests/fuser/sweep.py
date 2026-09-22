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

from .arch_common import _get_parser
from .config_parser import FUSER_CONFIG_DIR, FuserConfigSchema
from .operand import L1_PACKERS

SweepValue = str | bool | int | float
ConfigPath = tuple[str | int, ...]
AllValues = tuple[SweepValue, ...] | Callable[[dict], Iterable[SweepValue]]


@dataclass(frozen=True)
class SweepParameter:
    value_type: type[str] | type[bool] | type[int] | type[float] = str
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
    node_types=("UnarySfpu", "BinarySfpu"),
    id_field="",
)

_SFPU_APPROXIMATION_MODE = SweepParameter(
    value_type=bool,
    all_values=(False, True),
    node_types=("UnarySfpu", "BinarySfpu"),
)

# Register new fields here; discovery and expansion are shared by every parameter.
SWEEP_PARAMETERS: dict[str, dict[str, SweepParameter]] = {
    "config": {},
    "operation": {},
    "math": {
        "operation": _SFPU_OPERATION,
        "approximation_mode": _SFPU_APPROXIMATION_MODE,
    },
    "pack": {
        "operation": _SFPU_OPERATION,
        "approximation_mode": _SFPU_APPROXIMATION_MODE,
    },
    "operand": {
        "format": SweepParameter(all_values=_operand_formats),
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
        values = value
    else:
        return None

    if not values:
        raise ValueError(f"{location}: sweep choices cannot be empty")
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


def expand_fuser_configs(
    test_name: str,
    definition: dict,
    *,
    parameters: SweepRegistry | None = None,
) -> Iterator[tuple[str, dict]]:
    if parameters is None:
        parameters = SWEEP_PARAMETERS
    axes = list(_iter_axes(definition, parameters))
    for choices in product(*(axis.choices for axis in axes)):
        config_dict = deepcopy(definition)
        suffix = []
        for axis, value in zip(axes, choices):
            node = config_dict
            for key in axis.path[:-1]:
                node = node[key]
            node[axis.path[-1]] = value
            suffix.append(f"{axis.label}_{value}")
        case_name = "__".join([test_name, *suffix])
        yield case_name, config_dict


def collect_fuser_cases(yaml_files):
    cases = {}
    for yaml_path in yaml_files:
        test_name = str(yaml_path.relative_to(FUSER_CONFIG_DIR).with_suffix(""))
        definition = FuserConfigSchema.load_definition(test_name)
        for case_name, config_dict in expand_fuser_configs(test_name, definition):
            cases[case_name] = config_dict
    return cases
