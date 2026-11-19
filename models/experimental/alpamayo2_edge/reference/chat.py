# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Driving conversation with relative image times (port of NVlabs `chat_template.py`) and its jinja rendering."""

import json
from datetime import datetime

import jinja2
import jinja2.ext
import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}
DEFAULT_SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
DRIVING_TASK = (
    "Your task: output the chain-of-thought reasoning of the driving process, then output the future trajectory."
)
# Driving answers follow an empty think block, which the prompt supplies.
CLOSED_THINKING = "<think></think>"
HISTORY_START = "<|traj_history_start|>"
HISTORY_PAD = "<|traj_history|>"
HISTORY_END = "<|traj_history_end|>"


def relative_time_labels(timestamps: torch.Tensor) -> list[str]:
    """Label capture times relative to the final camera-major input timestamp."""
    timestamps = timestamps.reshape(-1)
    labels = []
    for delta in (timestamps - timestamps[-1]).tolist():
        # Upstream formats with `z.1f` (Python >= 3.11): a delta that rounds to -0.0 prints as 0.0.
        text = f"{delta / 1_000_000:.1f}"
        labels.append(f"<t={'0.0' if text == '-0.0' else text}s>")
    return labels


def build_conversation(
    images: list,
    camera_indices: torch.Tensor,
    timestamps: torch.Tensor,
    pixel_bounds: dict[int, tuple[int, int]],
    nav_text: str,
    num_tokens_per_history_traj: int = 57,
    include_camera_ids: bool = True,
    system_prompt: str | None = None,
) -> list[dict]:
    """Compose camera-major sorted frames, history placeholders and navigation into the driving prompt."""
    content = []
    previous_camera = None
    for image, camera, label in zip(images, camera_indices.tolist(), relative_time_labels(timestamps), strict=True):
        if include_camera_ids and camera != previous_camera:
            content.append({"type": "text", "text": CAMERA_DISPLAY_NAMES[camera] + ": "})
        content.append({"type": "text", "text": label})
        lower, upper = pixel_bounds[camera]
        content.append({"type": "image", "image": image, "min_pixels": lower, "max_pixels": upper})
        previous_camera = camera
    content.extend(
        [
            {"type": "text", "text": "\n"},
            {"type": "text", "text": HISTORY_START + HISTORY_PAD * num_tokens_per_history_traj + HISTORY_END},
            {"type": "text", "text": "\n"},
            {"type": "text", "text": "Navigation input is: " + nav_text},
            {"type": "text", "text": "\n"},
            {"type": "text", "text": DRIVING_TASK},
        ]
    )
    system = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def render_chat_template(template: str, messages: list[dict], **template_kwargs) -> str:
    """Render a checkpoint chat template in the same jinja environment `transformers` uses."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(template).render(messages=messages, tools=None, documents=None, **template_kwargs)
