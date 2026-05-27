# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Two-command-queue decode trace pipelining for Seamless M4T v2.

CQ1 stages the next decode step's token/position H2D copies while CQ0 executes
the current decode trace.  Synchronization via ``ttnn.record_event`` /
``ttnn.wait_for_event`` prevents buffer overwrites.

Usage pattern (caller owns the trace; capture via ``TTSeamlessM4Tv2Model.capture_text_decoder_decode_trace``)::

    rt = model._kv_decode_rt          # pre-allocated decode buffers
    helper = DecodeTrace2CQ(device)

    for step in range(max_new_tokens):
        # 1. Stage next-step inputs on CQ1 while CQ0 is done / executing.
        helper.stage_inputs_cq1(model, cur_tok, cur_pos, batch_size)

        # 2. Execute trace on CQ0 (waits for CQ1 uploads internally).
        helper.execute_trace_cq0(rt.trace_id)

        # 3. Read result (blocking sync already done inside execute_trace_cq0).
        logits = rt.logits_tt
        ...

Pattern from: models/experimental/devstral2_large/demo/decode_trace_2cq.py
(branch ``remotes/origin/ign/devstral2_123B_instruct``)
"""

from __future__ import annotations

from typing import Optional

import ttnn


class DecodeTrace2CQ:
    """Helper that pipeline-overlaps CQ1 H2D uploads with CQ0 trace execution.

    Attributes:
        mesh_device: The ttnn mesh/device.
        _op_event:   CQ0 event — signals that CQ0 finished the previous trace
                     (i.e. decode buffers are available for CQ1 to overwrite).
        _write_event: CQ1 event — signals that CQ1 finished the H2D upload
                      (i.e. CQ0 can start the next trace).
    """

    def __init__(self, mesh_device: ttnn.Device) -> None:
        self.mesh_device = mesh_device
        # Initial CQ0 event: CQ1 waits for this before writing its first inputs.
        self._op_event: Optional[object] = ttnn.record_event(mesh_device, 0)
        self._write_event: Optional[object] = None

    # ------------------------------------------------------------------

    def stage_inputs_cq1(
        self,
        model: "TTSeamlessM4Tv2Model",  # type: ignore[name-defined]
        token_id: int,
        position: int,
        batch_size: int = 1,
    ) -> None:
        """Upload token + position for the current decode step on CQ1.

        Waits for CQ0 to signal that the previous trace is done (so the decode
        buffers are free) before issuing the H2D copies on CQ1.
        """
        # CQ1 must wait until CQ0 has finished using the decode input buffers.
        if self._op_event is not None:
            ttnn.wait_for_event(1, self._op_event)
        model._upload_kv_decode_step_inputs_cq1(token_id, position, batch_size)
        # Signal CQ0 that the uploads are done.
        self._write_event = ttnn.record_event(self.mesh_device, 1)

    def execute_trace_cq0(self, trace_id: int) -> None:
        """Execute the captured decode trace on CQ0.

        Waits for CQ1's write event before starting so the trace reads the
        freshly uploaded inputs, then records a CQ0 op event to signal CQ1.
        """
        if self._write_event is not None:
            ttnn.wait_for_event(0, self._write_event)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        # Signal CQ1 that CQ0 is done with the decode buffers.
        self._op_event = ttnn.record_event(self.mesh_device, 0)
