# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The image sizes Ideogram 4 serves on each mesh, its prompt limit, and the request checks over them.

Sizes are (width, height). This module imports no device code at load, so a server process can
import it to validate requests before any worker opens a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# The four 2K serving sizes.
IDEOGRAM4_SERVING_SIZES = ((2048, 2048), (2048, 1152), (1152, 2048), (1360, 2048))

# Longest prompt, chat template included; the encoder runs at this fixed length. Like the reference,
# a longer prompt is rejected, not truncated.
IDEOGRAM4_MAX_TEXT_TOKENS = 2048

_SERVED_RESOLUTIONS: dict[tuple[int, ...], tuple[tuple[int, int], ...]] = {
    (2, 4): IDEOGRAM4_SERVING_SIZES,
    (4, 8): IDEOGRAM4_SERVING_SIZES,
}


def served_resolutions(mesh_shape: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return _SERVED_RESOLUTIONS.get(tuple(mesh_shape), ())


def validate_size(*, mesh_shape: Sequence[int], width: int, height: int) -> None:
    served = served_resolutions(mesh_shape)
    if (width, height) not in served:
        msg = f"unsupported size {width}x{height}; this deployment serves {list(served)}"
        raise ValueError(msg)


def validate_prompt(prompt: str, count_text_tokens: Callable[[str], int]) -> None:
    prompt_tokens = count_text_tokens(prompt)
    if prompt_tokens > IDEOGRAM4_MAX_TEXT_TOKENS:
        msg = f"prompt is {prompt_tokens} tokens, over the {IDEOGRAM4_MAX_TEXT_TOKENS}-token limit"
        raise ValueError(msg)


def validate_request(
    *, mesh_shape: Sequence[int], width: int, height: int, prompt: str, count_text_tokens: Callable[[str], int]
) -> None:
    validate_size(mesh_shape=mesh_shape, width=width, height=height)
    validate_prompt(prompt, count_text_tokens)
