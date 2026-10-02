# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Metal Trace + two command queues for ModernBERT.

Warm the kernels, capture the forward pass once, then replay it while cq1 stages
the next input.

Two device buffers, not one: `op_event` is recorded before `execute_trace`, so
with a single buffer cq1 could overwrite the input while the trace was still
reading it. Staging absorbs that write.

`ttnn.copy` rather than `ttnn.reshard` because `input_ids` is consumed exactly as
written, so the two buffers are identically specified.

Measured against an untraced loop that uploads an input every iteration, trace is
worth -29.9% at b1s256 (10.46 -> 7.33 ms) and -1.8% at b8s256 (22.31 -> 21.90).
The gap is host dispatch: small shapes sit on a ~10.5 ms floor that is independent
of how much work they do, and the traced path replays device work without paying
it. b8s256 is device-bound, so there is little there to recover.
"""

import ttnn
from models.experimental.modernbert.runner.performant_runner_infra import ModernBertPerformanceRunnerInfra


class ModernBertPerformantRunner:
    """Modes: "trace_2cq" (default), "trace" and "2cq".

    Kept separate so a failure localises to one mechanism. They differ only in how
    an input reaches `tt_input`, but need different `device_params`: "trace" wants
    a trace region and one queue, "2cq" two queues and no trace region.
    `ttnn.record_event(device, 1)` is a hard error on a one-queue device, which is
    why the single-queue path skips `_stage`.
    """

    def __init__(self, device, device_batch_size=1, sequence_length=256, input_ids=None, mode="trace_2cq"):
        if mode not in ("trace_2cq", "trace", "2cq"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.device = device
        self.runner_infra = ModernBertPerformanceRunnerInfra(
            device=device,
            batch_size=device_batch_size,
            seq_len=sequence_length,
            input_ids=input_ids,
        )

        self.tt_inputs_host = self.runner_infra.setup_inputs()
        # Staging is written by cq1; the trace input is written by cq0 and read by
        # the trace. Allocated up front and never reallocated, so the address the
        # trace bakes in stays valid for the life of the runner. Trace-only has one
        # queue and therefore nothing to stage against.
        self.tt_staging = ttnn.allocate_tensor_on_device(self.tt_inputs_host.spec, device) if mode != "trace" else None
        self.tt_input = ttnn.allocate_tensor_on_device(self.tt_inputs_host.spec, device)
        self.runner_infra.input_tensor = self.tt_input
        self.tid = None

    def _stage(self, tt_inputs_host):
        """Move one host input into the trace's input buffer across both queues."""
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_staging, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        ttnn.copy(self.tt_staging, self.tt_input)
        self.op_event = ttnn.record_event(self.device, 0)

    def _capture_modernbert_trace_2cqs(self):
        self.op_event = ttnn.record_event(self.device, 0)

        # Two full passes before capture. Every kernel has to be compiled and every
        # program config resolved outside the trace, or capture records the JIT.
        for _ in range(2):
            self._stage(self.tt_inputs_host)
            self.runner_infra.run()
            self.runner_infra.validate()
            self.runner_infra.dealloc_output()

        self._stage(self.tt_inputs_host)
        trace_input_addr = self.tt_input.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)

        # The trace holds this address; if anything reallocated the input during
        # capture, every later replay would silently read the wrong memory.
        assert self.tt_input.buffer_address() == trace_input_addr, "trace input buffer moved during capture"

    def _execute_modernbert_trace_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        self._stage(tt_inputs_host)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.output_tensor

    def _capture_modernbert_trace(self):
        """Trace alone, one command queue. The input is written straight to the
        trace buffer on cq0, so there is no staging buffer and no event."""
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_input, 0)
        for _ in range(2):
            self.runner_infra.run()
            self.runner_infra.validate()
            self.runner_infra.dealloc_output()

        trace_input_addr = self.tt_input.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        assert self.tt_input.buffer_address() == trace_input_addr, "trace input buffer moved during capture"

    def _execute_modernbert_trace(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_input, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.output_tensor

    def _warmup_2cqs(self):
        """Two queues, no trace. Only the first event needs seeding; after that
        `_stage` maintains the handshake itself."""
        self.op_event = ttnn.record_event(self.device, 0)
        for _ in range(2):
            self._stage(self.tt_inputs_host)
            self.runner_infra.run()
            self.runner_infra.validate()
            self.runner_infra.dealloc_output()

    def _execute_2cqs_inference(self, tt_inputs_host=None):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        self._stage(tt_inputs_host)
        self.runner_infra.run()
        return self.runner_infra.output_tensor

    def setup(self):
        """Capture or warm up according to mode, so callers do not branch."""
        if self.mode == "trace_2cq":
            self._capture_modernbert_trace_2cqs()
        elif self.mode == "trace":
            self._capture_modernbert_trace()
        else:
            self._warmup_2cqs()

    def run(self, input_ids=None):
        tt_inputs_host = self.tt_inputs_host if input_ids is None else self.runner_infra.setup_inputs(input_ids)
        if self.mode == "trace_2cq":
            return self._execute_modernbert_trace_2cqs_inference(tt_inputs_host)
        if self.mode == "trace":
            return self._execute_modernbert_trace(tt_inputs_host)
        return self._execute_2cqs_inference(tt_inputs_host)

    def validate(self):
        return self.runner_infra.validate()

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.device, self.tid)
            self.tid = None
        self.runner_infra.release()
