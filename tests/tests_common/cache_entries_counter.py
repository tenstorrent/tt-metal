# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager


class CacheEntriesCounter:
    def __init__(self, device):
        self.device = device
        self.total = 0

    def reset(self):
        """Reset the accumulated cache diff to zero."""
        self.total = 0

    @contextmanager
    def measure(self):
        before = self.device.num_program_cache_entries()
        print(f"[before measure]device.num_program_cache_entries(): {before}")
        yield
        self.total += self.device.num_program_cache_entries() - before
        print(f"[after measure]device.num_program_cache_entries(): {self.device.num_program_cache_entries()}")

    def decorator(self, fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with self.measure():
                return fn(*args, **kwargs)

        return wrapper
