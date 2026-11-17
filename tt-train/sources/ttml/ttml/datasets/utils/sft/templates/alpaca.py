# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Alpaca (tatsu-lab/alpaca): the prompt of an instruction, with its input or without one."""

from string import Template

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
