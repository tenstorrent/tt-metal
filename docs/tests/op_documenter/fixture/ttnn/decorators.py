# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Only the operation wrapper interface needed by Sphinx; no device runtime."""


class Operation:
    def __init__(self, name, function, doc=None):
        self.__name__ = name
        self.function = function
        self.__doc__ = doc

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class FastOperation(Operation):
    pass
