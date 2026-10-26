# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


class SeededRng:
    """Reproducible test inputs from a seeded torch.Generator, with the random.Random calls the spec tests use."""

    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def randrange(self, start, stop=None):
        if stop is None:
            start, stop = 0, start
        return int(torch.randint(start, stop, (), generator=self.generator))

    def choice(self, items):
        return items[self.randrange(len(items))]

    def random(self):
        return float(torch.rand((), dtype=torch.float64, generator=self.generator))
