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

from .fused_operand import OperandRegistry
from .fuser_config import FuserConfig, GlobalConfig
from .validator import PackSchema

FUSER_CONFIG_DIR = (
    Path(os.environ.get("LLK_HOME", ".")) / "tests" / "python_tests" / "fuser" / "tests"
)

from helpers.chip_architecture import get_chip_architecture

from .arch_common import _get_parser

arch = get_chip_architecture()
OperationSchema = _get_parser().OperationSchema


def format_validation_error(error: ValidationError) -> str:
    messages = []
    for err in error.errors():
        loc = ".".join(str(x) for x in err["loc"])
        msg = err["msg"]

        if "Input should be" in msg:
            inp = err.get("input")
            valid_values = re.findall(r"'([^']+)'", msg)
            expected = ", ".join(valid_values) if valid_values else msg
            messages.append(f"'{loc}': got '{inp}', expected: {expected}")
        elif "Extra inputs are not permitted" in msg:
            messages.append(f"'{loc}': unknown field")
        elif "Field required" in msg:
            messages.append(f"'{loc}': required field missing")
        else:
            clean_msg = msg.removeprefix("Value error, ")
            messages.append(f"'{loc}': {clean_msg}")

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
    operands: List[OperandDefinition] = Field(..., min_length=1)
    operations: List[OperationSchema] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_config(self) -> "FuserConfigSchema":
        formats = {op_def.name: op_def.format for op_def in self.operands}
        seen_operands: set[str] = set()

        for op in self.operations:
            src_a_name = None
            for node in op.math:
                if hasattr(node, "src_a"):
                    if src_a_name is None:
                        src_a_name = node.src_a
                    seen_operands.add(node.src_a)
                if hasattr(node, "src_b"):
                    seen_operands.add(node.src_b)

            pack_schemas = [e for e in op.pack if isinstance(e, PackSchema)]

            for pack_entry in pack_schemas:
                if pack_entry.output in seen_operands:
                    raise ValueError(
                        f"cannot use '{pack_entry.output}' as output twice"
                    )
                seen_operands.add(pack_entry.output)

                if src_a_name is not None:
                    input_fmt = formats[src_a_name]
                    output_fmt = formats[pack_entry.output]
                    if is_format_combination_outlier(
                        input_fmt, output_fmt, self.dest_acc
                    ):
                        raise ValueError(
                            f"Dest Accumulation must be enabled for {input_fmt.name} input and {output_fmt.name} output"
                        )

            if len(pack_schemas) > 1:
                pack_formats = [formats[e.output] for e in pack_schemas]
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

        pipeline = [
            op.to_fused_operation(operands, dest_acc=self.dest_acc.value)
            for op in self.operations
        ]

        return FuserConfig(
            pipeline=pipeline,
            global_config=GlobalConfig(
                dest_acc=self.dest_acc,
                test_name=test_name,
                loop_factor=self.loop_factor,
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

        return schema.to_fuser_config(test_name)
