#!/usr/bin/env python3
"""
Test script to verify 1x32 mesh device setup and all-gather operation.
This test uses DeviceGetter to open a mesh device with logical shape 1x32.
"""

import ttnn
import utils


def _main():
    device = utils.DeviceGetter.get_device((1, 32))

    # Tensor of ones on device
    ones_tensor = ttnn.ones(
        shape=[1, 1, 256, 256],
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED,
            ttnn.BufferType.DRAM,
            None,
        ),
    )

    # All-gather across the 32-way dimension of the mesh
    gathered = ttnn.all_gather(
        input_tensor=ones_tensor,
        dim=2,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=None,
        num_links=1,
        topology=None,
        # topology = ttnn.Topology.Ring,
    )

    return [gathered]


def main():
    _ = _main()
    return 0


if __name__ == "__main__":
    main()
