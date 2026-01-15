# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def default_device():
    """Context manager for default single-device setup.

    Lazily imports ttnn to avoid import overhead when device is not needed.
    """
    import ttnn

    device = ttnn.open_device(device_id=0)
    device_name = ttnn.get_arch_name()

    yield (device, device_name)

    ttnn.close_device(device)
    del device
