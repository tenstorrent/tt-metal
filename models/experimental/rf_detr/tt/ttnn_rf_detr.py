# SPDX-License-Identifier: Apache-2.0
"""End-to-end RF-DETR on Tenstorrent.

On-device chain: windowed DINOv2 backbone -> C2f projector -> two-stage deformable
transformer + heads. The only remaining host glue is the backbone embeddings
(patch conv + window partition) and feature-map shaping (LN + window unpartition),
which are reshape-heavy.

The projector + transformer region is dispatch-bound (many small ops on 300-query
tensors), so it is captured into a single metal-trace and replayed per inference:
the 4 backbone feature maps are written into persistent device input buffers each
call (host->device on cq_id=0), then ``ttnn.execute_trace`` replays the whole
projector+transformer device graph and the logits/boxes are read back with
``ttnn.to_torch`` (which syncs). Tracing this tail lifted FPS from ~19.4 to ~21.7.

2-CQ overlap (upload on cq_id=1 + event so compute waits) was implemented and
measured: it *regressed* FPS (~19.0) because the per-call input upload is tiny
relative to the host-bound backbone, and the benchmark runs each inference fully
synchronized (no cross-inference pipelining to overlap), so the event sync only
adds overhead. It was therefore reverted and the device stays on a single CQ.
"""

import ttnn
from models.experimental.rf_detr.reference.modeling_rf_detr import RfDetrOutput
from models.experimental.rf_detr.tt.ttnn_backbone import TtDinoBackbone
from models.experimental.rf_detr.tt.ttnn_projector import TtProjector
from models.experimental.rf_detr.tt.ttnn_transformer import TtTransformer

N_QUERIES = 300
NUM_CLASSES = 91


class TtRfDetr:
    def __init__(self, ref_model, device):
        self.ref = ref_model.eval()
        self.device = device
        self.backbone = TtDinoBackbone(ref_model, device)
        self.projector = TtProjector(ref_model, device)
        self.transformer = TtTransformer(ref_model, device)

        device.enable_program_cache()

        self._trace_id = None
        self._persistent_in = None   # list of 4 device tensors [1,1600,384] TILE bf16
        self._logits_out = None      # device tensor (trace output)
        self._boxes_out = None       # device tensor (trace output)

    def _feats_to_host(self, pixel_values):
        """Backbone -> 4 host torch tensors channels-last [1,1600,384]."""
        feats = self.backbone.feature_maps(pixel_values)
        return [f.flatten(2).transpose(1, 2).contiguous() for f in feats]

    def _capture_trace(self, feats_cl_host):
        device = self.device
        # Persistent device input buffers the trace reads from.
        self._persistent_in = [
            ttnn.from_torch(f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for f in feats_cl_host
        ]

        # Warm run (eager) so conv2d prepared-weights are cached and the program cache
        # is populated; mutating ops (p["w"] = prepared) must run OUTSIDE the capture.
        source = self.projector(self._persistent_in)
        logits, boxes = self.transformer.forward_device(source)
        ttnn.synchronize_device(device)

        # Capture the projector + transformer device graph.
        self._trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        source = self.projector(self._persistent_in)
        self._logits_out, self._boxes_out = self.transformer.forward_device(source)
        ttnn.end_trace_capture(device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(device)

    def _run_trace(self, feats_cl_host):
        device = self.device
        # Upload the 4 feature maps into the persistent input buffers, then replay.
        for host_t, dev_t in zip(feats_cl_host, self._persistent_in):
            ht = ttnn.from_torch(host_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(ht, dev_t, cq_id=0)
        ttnn.execute_trace(device, self._trace_id, cq_id=0, blocking=False)
        logits_t = ttnn.to_torch(self._logits_out).float().reshape(1, N_QUERIES, NUM_CLASSES)
        boxes_t = ttnn.to_torch(self._boxes_out).float().reshape(1, N_QUERIES, 4)
        return logits_t, boxes_t

    def __call__(self, pixel_values):
        feats_cl_host = self._feats_to_host(pixel_values)
        if self._trace_id is None:
            self._capture_trace(feats_cl_host)
        logits, pred_boxes = self._run_trace(feats_cl_host)
        return RfDetrOutput(logits=logits, pred_boxes=pred_boxes)
