# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def default_device():
    device = ttnn.open_device(device_id=0)

    yield (device, "default")

    ttnn.close_device(device)
    del device
