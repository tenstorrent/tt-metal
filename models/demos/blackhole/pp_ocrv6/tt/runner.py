# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TracedRunner:
    """Captures `fn(device_input)` once as a trace and replays it per call.

    The input lives in one fixed DRAM buffer. With two command queues the host-to-device
    write goes on CQ 1 while CQ 0 only replays traces; events keep them ordered.

    Every persistent device buffer (prepared conv weights) must exist before the first trace of
    any runner is captured: a later allocation can land in another trace's scratch memory and get
    overwritten on replay. Run each model once per shape untraced before building runners.
    The input buffer may still share another trace's scratch, so the write waits for all queued
    CQ 0 work, not just this runner's last replay.
    """

    def __init__(self, device, fn, host_input, two_cq=True):
        self.device = device
        self.two_cq = two_cq
        self.input = ttnn.allocate_tensor_on_device(host_input.spec, device)
        ttnn.copy_host_to_device_tensor(host_input, self.input)
        fn(self.input)
        ttnn.synchronize_device(device)
        self.trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self.output = fn(self.input)
        ttnn.end_trace_capture(device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(device)

    def __call__(self, host_input):
        if self.two_cq:
            ttnn.wait_for_event(1, ttnn.record_event(self.device, 0))
            ttnn.copy_host_to_device_tensor(host_input, self.input, cq_id=1)
            ttnn.wait_for_event(0, ttnn.record_event(self.device, 1))
        else:
            ttnn.copy_host_to_device_tensor(host_input, self.input, cq_id=0)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        return self.output

    def release(self):
        ttnn.release_trace(self.device, self.trace_id)
