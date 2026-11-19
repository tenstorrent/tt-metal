# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct numerical and watcher gate for the actual production recurrence helper."""

from importlib import import_module

from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_probe as probe

if __name__ == "__main__":
    probe.fused = import_module("models.autoports.ornith_ai_ornith_1_5_9b.tt.flat_ordered_post_loop")
    probe.main()
