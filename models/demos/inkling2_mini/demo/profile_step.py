# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Device profile of one traced decode step (max_ctx 4K by default).

    flock /tmp/tt-card.lock python -m tracy -r -p --no-web-server --op-support-count 12000 \
        -m models.demos.inkling2_mini.demo.profile_step [max_ctx]
then: python -m models.demos.inkling2_mini.demo.profile_report <ops_perf_results csv>
"""

import sys

import ttnn
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.model import Inkling2Model


def main():
    from tracy import signpost

    max_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20)
    try:
        model = Inkling2Model(dev, max_ctx=max_ctx, log=lambda s: None)
        gen = Generator(model)
        gen.capture()
        ttnn.ReadDeviceProfiler(dev)  # drop compile/capture data
        for i in range(2):
            gen.decode(1, 2048 + i)
            ttnn.synchronize_device(dev)
            signpost(header=f"replay{i}")
            ttnn.ReadDeviceProfiler(dev)
        gen.release()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
