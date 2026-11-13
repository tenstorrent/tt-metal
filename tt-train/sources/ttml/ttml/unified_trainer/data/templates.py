# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates, character for character those of tt-blacksmith's datasets.

The same prompts tokenize to the same examples, so a run here and the same run on tt-blacksmith train on
identical data.
"""

from __future__ import annotations

from string import Template
from typing import Mapping, Optional

# ------------------------------------------------------------------ SST-2

SST2_PROMPT_TEMPLATE = Template("""Review: $input\nOutput:""")
SST2_RESPONSE_TEMPLATE = Template('{"label": "$label"}')
SST2_LABELS = {0: "negative", 1: "positive"}

# ------------------------------------------------------------------ Alpaca (tatsu-lab/alpaca)

ALPACA_PROMPT_INTRO = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

ALPACA_PROMPT_TEMPLATE = Template(
    f"""
{ALPACA_PROMPT_INTRO}

### Instruction:
$instruction

### Input:
$input

### Response:
"""
)

ALPACA_PROMPT_TEMPLATE_NO_INPUT = Template(
    f"""
{ALPACA_PROMPT_INTRO}

### Instruction:
$instruction

### Response:
"""
)

# ------------------------------------------------------------------ custom datasets

CUSTOM_ALPACA_PROMPT_INTRO = (
    "Below is an instruction that describes a task, paired with an optional input that provides further context. "
    "Write a response that appropriately completes the request."
)

CUSTOM_ALPACA_INPUT_BLOCK_TEMPLATE = Template(
    """### Input:
$input
"""
)

CUSTOM_ALPACA_PROMPT_TEMPLATE = Template(
    f"""
{CUSTOM_ALPACA_PROMPT_INTRO}

### Instruction:
$instruction

$input_section

### Response:
"""
)

# The example fields each custom template reads; a column mapping maps them to dataset columns.
CUSTOM_TEMPLATE_KEYS = {"alpaca": {"required": {"instruction", "output"}, "optional": {"input"}}}


def _check_template(template: str) -> str:
    name = template.strip().lower()
    if name not in CUSTOM_TEMPLATE_KEYS:
        raise ValueError(f"dataset.custom_options.template: {template!r} is not one of {sorted(CUSTOM_TEMPLATE_KEYS)}")
    return name


def custom_prompt_and_response(example: Mapping, template: str, column_mapping: Mapping[str, str]) -> tuple[str, str]:
    """The prompt and response of one custom-dataset example, read through ``column_mapping``."""
    _check_template(template)
    instruction = example[column_mapping["instruction"]]
    input_column = column_mapping.get("input", "")
    input_text = example[input_column] if input_column else ""
    input_section = CUSTOM_ALPACA_INPUT_BLOCK_TEMPLATE.substitute(input=input_text) if input_text.strip() else ""
    prompt = CUSTOM_ALPACA_PROMPT_TEMPLATE.substitute(instruction=instruction, input_section=input_section)
    return prompt, example[column_mapping["output"]]


def resolve_column_mapping(
    template: str, column_mapping: Optional[Mapping[str, str]], dataset_columns: set[str]
) -> dict[str, str]:
    """``column_mapping`` completed with the template's fields that are dataset columns of the same name.

    Raises ``ValueError`` for a mapping onto a column the dataset lacks, a field the template doesn't
    read, or a required field left unmapped.
    """
    keys = CUSTOM_TEMPLATE_KEYS[_check_template(template)]
    required, optional = keys["required"], keys["optional"]
    resolved = dict(column_mapping or {})

    missing_columns = set(resolved.values()) - dataset_columns
    if missing_columns:
        raise ValueError(
            f"dataset.custom_options.column_mapping: no dataset columns {sorted(missing_columns)}; "
            f"the dataset has {sorted(dataset_columns)}"
        )
    unsupported = set(resolved) - required - optional
    if unsupported:
        raise ValueError(
            f"dataset.custom_options.column_mapping: template {template!r} reads no {sorted(unsupported)}; "
            f"it reads {sorted(required | optional)}"
        )

    for key in sorted(required | optional):
        if key in dataset_columns and key not in resolved:
            resolved[key] = key
    unmapped = required - set(resolved)
    if unmapped:
        raise ValueError(
            f"dataset.custom_options.column_mapping: template {template!r} needs {sorted(unmapped)}, which are "
            "neither mapped nor dataset columns"
        )
    return resolved
