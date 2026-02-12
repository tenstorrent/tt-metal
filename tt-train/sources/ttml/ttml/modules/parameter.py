# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter and Buffer wrappers for tensor registration."""

from typing import Any


class Parameter:
    """Wrapper marking a tensor as a trainable parameter."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"Parameter({self.tensor})"


class Buffer:
    """Wrapper marking a tensor as a non-trainable buffer."""

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"Buffer({self.tensor})"
