# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


class LightweightModule:
    """Torch modules add a surprising amount of host overhead for attribute
    access and method calls. This class is a lightweight alternative that
    just wraps a forward function for now."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
