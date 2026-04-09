# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from pathlib import Path
from typing import Annotated, List, Union

import yaml
from helpers.llk_params import DestAccumulation
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from .fused_operand import OperandRegistry
from .fuser_config import FuserConfig, GlobalConfig

FUSER_CONFIG_DIR = (
    Path(os.environ.get("LLK_HOME", ".")) / "tests" / "python_tests" / "fuser_config"
)

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture

arch = get_chip_architecture()

if arch == ChipArchitecture.WORMHOLE:
    from .wormhole.parser import OperationSchema
elif arch == ChipArchitecture.BLACKHOLE:
    from .blackhole.parser import OperationSchema
else:
    assert False, "Architecture is not supported"


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


class FuserConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dest_acc: DestAccumulation = DestAccumulation.No
    loop_factor: Annotated[int, Field(ge=1)] = 16
    operations: List[OperationSchema] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_config(self) -> "FuserConfigSchema":
        outputs = set()
        for i, op in enumerate(self.operations):
            if op.output in outputs:
                raise ValueError(f"op[{i}].output='{op.output}' already defined")
            outputs.add(op.output)
        return self

    def to_fuser_config(self, test_name: str):
        operands = OperandRegistry()
        pipeline = [op.to_fused_operation(operands) for op in self.operations]

        return FuserConfig(
            pipeline=pipeline,
            global_config=GlobalConfig(
                dest_acc=self.dest_acc,
                test_name=test_name,
                loop_factor=self.loop_factor,
            ),
        )

    @classmethod
    def validate_file(cls, yaml_path: Union[str, Path]) -> "FuserConfigSchema":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        try:
            return cls.model_validate(config_dict)
        except ValidationError as e:
            raise ValueError(
                f"Validation failed for {yaml_path.name}:\n{format_validation_error(e)}"
            ) from None

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
        yaml_path = FUSER_CONFIG_DIR / f"{test_name}.yaml"
        schema = cls.validate_file(yaml_path)
        return schema.to_fuser_config(test_name)
