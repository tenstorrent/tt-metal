# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from pathlib import Path
from typing import Annotated, List, Optional, Tuple

import pytest
import yaml
from helpers.data_format_inference import is_format_combination_outlier
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.logger import logger
from helpers.tile_constants import validate_tile_dimensions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .fuser_config import FuserConfig, GlobalConfig
from .operand import OperandRegistry

FUSER_CONFIG_DIR = (
    Path(os.environ.get("LLK_HOME", ".")) / "tests" / "python_tests" / "fuser" / "tests"
)

from helpers.chip_architecture import get_chip_architecture

from .arch_common import _get_parser

arch = get_chip_architecture()
OperationSchema = _get_parser().OperationSchema


def _format_loc(loc):
    parts = []
    i = 0
    while i < len(loc):
        part = loc[i]
        if isinstance(part, int):
            parent = parts[-1] if parts else ""
            ordinal = part + 1
            if parent == "operations":
                parts[-1] = f"Operation {ordinal}"
            elif parent == "math":
                parts[-1] = f"Math node {ordinal}"
            elif parent == "pack":
                parts[-1] = f"Pack entry {ordinal}"
            elif parent == "operands":
                parts[-1] = f"Operand {ordinal}"
            else:
                parts[-1] = f"{parent}[{part}]"
        elif isinstance(part, str):
            is_discriminator = (
                parts
                and parts[-1].startswith(("Math node", "Pack entry"))
                and i + 1 < len(loc)
                and isinstance(loc[i + 1], str)
            )
            if is_discriminator:
                parts[-1] += f" ({part})"
            else:
                parts.append(part)
        i += 1
    return parts


def format_validation_error(error: ValidationError) -> str:
    messages = []
    for err in error.errors():
        loc_parts = _format_loc(err["loc"])
        msg = err["msg"]

        if "Input should be" in msg:
            inp = err.get("input")
            valid_values = re.findall(r"'([^']+)'", msg)
            expected = ", ".join(valid_values) if valid_values else msg
            error_msg = f"got '{inp}', expected: {expected}"
        elif "Extra inputs are not permitted" in msg:
            error_msg = "unknown field"
        elif "Field required" in msg:
            error_msg = "required field"
        else:
            error_msg = msg.removeprefix("Value error, ")

        for i, part in enumerate(loc_parts):
            indent = "  " * i
            if i == len(loc_parts) - 1:
                messages.append(f"{indent}{part}: {error_msg}")
            else:
                messages.append(f"{indent}{part}")

    return "\n".join(messages)


class OperandDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    dims: Annotated[Tuple[int, int], Field(min_length=2, max_length=2)]
    format: DataFormat
    const_value: Optional[float] = None
    # Optional per-operand tile geometry (rows, cols). Defaults to a full 32x32 tile
    # (4 faces). Use (16, 32) for a 16x32 tiny tile (num_faces=2, one face-row).
    tile_dims: Optional[
        Annotated[Tuple[int, int], Field(min_length=2, max_length=2)]
    ] = None

    @field_validator("dims")
    @classmethod
    def validate_dims(cls, v: List[int]) -> Tuple[int, int]:
        for dim in v:
            if dim <= 0:
                raise ValueError(f"must be positive, got {dim}")
        return tuple(v)

    @field_validator("tile_dims", mode="before")
    @classmethod
    def validate_tile_dims(cls, v):
        if v is None:
            return v
        v = tuple(v)
        validate_tile_dimensions(v)
        return v

    @model_validator(mode="after")
    def validate_dims_align_to_tiles(self) -> "OperandDefinition":
        tile_r, tile_c = self.tile_dims if self.tile_dims is not None else (32, 32)
        if self.dims[0] % tile_r != 0:
            raise ValueError(
                f"dims[0]={self.dims[0]} must be a multiple of tile row dimension {tile_r}"
            )
        if self.dims[1] % tile_c != 0:
            raise ValueError(
                f"dims[1]={self.dims[1]} must be a multiple of tile column dimension {tile_c}"
            )
        return self

    @field_validator("format", mode="before")
    @classmethod
    def parse_data_format(cls, v):
        if isinstance(v, DataFormat):
            return v
        if isinstance(v, str):
            try:
                return DataFormat[v]
            except KeyError:
                pass
        return v


class FuserConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dest_acc: DestAccumulation = DestAccumulation.No
    loop_factor: Annotated[int, Field(ge=1)] = 16
    quasar_use_dvalid: bool = False
    operands: List[OperandDefinition] = Field(..., min_length=1)
    operations: List[OperationSchema] = Field(..., min_length=1)

    @staticmethod
    def _declared_format(formats: dict, name: str) -> DataFormat:
        if name not in formats:
            raise ValueError(
                f"Operand '{name}' is not declared in the 'operands' section"
            )
        return formats[name]

    @model_validator(mode="after")
    def validate_config(self) -> "FuserConfigSchema":
        formats = {op_def.name: op_def.format for op_def in self.operands}
        seen_operands: set[str] = set()

        for op in self.operations:
            src_a_name = None
            for node in op.math:
                for operand_name in (
                    getattr(node, "in0", None),
                    getattr(node, "in1", None),
                ):
                    if operand_name is None:
                        continue
                    self._declared_format(formats, operand_name)
                    seen_operands.add(operand_name)
                if src_a_name is None:
                    src_a_name = getattr(node, "in0", None)

            pack_schemas = op.pack_schemas

            for pack_entry in pack_schemas:
                if pack_entry.output in seen_operands:
                    raise ValueError(
                        f"cannot use '{pack_entry.output}' as output twice"
                    )
                seen_operands.add(pack_entry.output)

                if src_a_name is not None:
                    input_fmt = self._declared_format(formats, src_a_name)
                    output_fmt = self._declared_format(formats, pack_entry.output)
                    if is_format_combination_outlier(
                        input_fmt, output_fmt, self.dest_acc
                    ):
                        raise ValueError(
                            f"Dest Accumulation must be enabled for {input_fmt.name} input and {output_fmt.name} output"
                        )

            if len(pack_schemas) > 1:
                pack_formats = [
                    self._declared_format(formats, e.output) for e in pack_schemas
                ]
                first_exp_b = pack_formats[0].is_exponent_B()
                if any(f.is_exponent_B() != first_exp_b for f in pack_formats[1:]):
                    names = [e.output for e in pack_schemas]
                    logger.warning(
                        f"Pack outputs {names} have mixed exponent families, "
                        f"unpack/math format inference will use {pack_schemas[0].output} as reference",
                    )

        return self

    def to_fuser_config(self, test_name: str):
        operands = OperandRegistry()

        for op_def in self.operands:
            operands.create(
                name=op_def.name,
                dimensions=op_def.dims,
                data_format=op_def.format,
                const_value=op_def.const_value,
                tile_dims=op_def.tile_dims,
            )

        pipeline = []
        for i, op in enumerate(self.operations):
            try:
                pipeline.append(
                    op.to_l1_operation(operands, dest_acc=self.dest_acc.value)
                )
            except ValueError as e:
                raise ValueError(f"Operation {i + 1}\n  {e}") from None

        num_stages = len(pipeline)
        for i, operation in enumerate(pipeline):
            operation.stage_id = i + 1
            operation.needs_pack_sync = any(
                (node.src_a is not None and node.src_a.is_output)
                or (node.src_b is not None and node.src_b.is_output)
                for node in operation.math.math_nodes
                if hasattr(node, "unpacker") and node.unpacker is not None
            )

        for i, operation in enumerate(pipeline):
            operation.has_pack_consumer = (
                i + 1 < num_stages and pipeline[i + 1].needs_pack_sync
            )

        return FuserConfig(
            pipeline=pipeline,
            global_config=GlobalConfig(
                dest_acc=self.dest_acc,
                test_name=test_name,
                loop_factor=self.loop_factor,
                quasar_use_dvalid=self.quasar_use_dvalid,
            ),
            operand_registry=operands,
        )

    @classmethod
    def validate_string(cls, yaml_content: str) -> "FuserConfigSchema":
        config_dict = yaml.safe_load(yaml_content)
        try:
            return cls.model_validate(config_dict)
        except ValidationError as e:
            raise ValueError(
                f"Validation failed:\n{format_validation_error(e)}"
            ) from None

    @classmethod
    def load(cls, test_name: str):
        yaml_path = (FUSER_CONFIG_DIR / f"{test_name}.yaml").resolve()
        if not yaml_path.exists():
            yaml_path = (FUSER_CONFIG_DIR / arch.value / f"{test_name}.yaml").resolve()
        if not yaml_path.is_relative_to(FUSER_CONFIG_DIR.resolve()):
            raise ValueError(f"Invalid test name: {test_name}")
        if not yaml_path.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError(f"Invalid config in {yaml_path.name}")

        supported_archs = config_dict.pop("supported_archs", None)
        if supported_archs is not None:
            if arch.value not in supported_archs:
                pytest.skip(f"Test '{test_name}' not supported on {arch.value}")

        try:
            schema = cls.model_validate(config_dict)
        except ValidationError as e:
            raise ValueError(
                f"Validation failed for {yaml_path.name}:\n{format_validation_error(e)}"
            ) from None

        try:
            return schema.to_fuser_config(test_name)
        except ValueError as e:
            raise ValueError(f"Validation failed for {yaml_path.name}:\n{e}") from None
