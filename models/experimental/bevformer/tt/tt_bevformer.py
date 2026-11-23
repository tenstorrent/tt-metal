# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the BEVFormer detector (``reference/bevformer.py``): the ResNet101-DCN backbone,
the FPN, the perception transformer and the detection head, on device end to end.

A frame's host-derived inputs (CAN bus, ego shift, previous-BEV rotation, the cameras' metas) are
prepared by :meth:`TTBEVFormer.prepare_frame`. The caller carries the previous BEV between frames,
as upstream's ``forward_test`` does. Preparing a frame also projects its cameras and queues the
readback of the counts that size the spatial cross-attention's plan, which follow what the cameras
see.

Through :meth:`TTBEVFormer.prepare_frame` and :meth:`TTBEVFormer.__call__` the detector runs live.
``tt_bevformer_trace.TTBEVFormerTrace`` replays it from three traces around that readback, each
recording steps of this class that are device work only:

================  =====================================================================
Trace             Steps
================  =====================================================================
``cameras``       :meth:`TTBEVFormer.project_cameras`
``features``      :meth:`TTBEVFormer.features`
``bev``           :meth:`TTBEVFormer.build_plan`, :meth:`TTBEVFormer.bev`, :meth:`TTBEVFormer.detect`
outside, on host  :meth:`TTBEVFormer.host_frame`, :meth:`TTBEVFormer.refill_frame`, the planner's
                  ``read_counts`` and ``size``, :meth:`TTBEVFormer.get_bboxes`
================  =====================================================================

:meth:`TTBEVFormer.detect` runs upstream's ``get_bboxes`` on device: the coder's top-k, the box
decoding, the range test as a validity mask and z moved to the box's bottom face, all fixed
``(bs, max_num)`` tensors. :meth:`TTBEVFormer.get_bboxes` only drops the invalid
boxes on the host, as no device tensor can hold a data-dependent count (see ``TTNMSFreeCoder``).
"""

import ttnn

from models.experimental.bevformer.tt.tt_common import signpost
from models.experimental.bevformer.tt.tt_fpn import TTFPN
from models.experimental.bevformer.tt.tt_head import TTBEVFormerHead
from models.experimental.bevformer.tt.tt_nms_free_coder import TTNMSFreeCoder
from models.experimental.bevformer.tt.tt_perception_transformer import TTPerceptionTransformer
from models.experimental.bevformer.tt.tt_resnet import TTResNet


def _to_dram(*tensors):
    """The detector's returned tensors, in DRAM: a caller may hold them into the next frame, and a
    buffer left in L1 there can block the backbone's circular buffers."""
    return tuple(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in tensors)


class TTBEVFormer:
    """The detector, pinned to the image shape ``create_bevformer_parameters`` recorded the
    backbone's and FPN's convs for."""

    def __init__(self, params, device, all_layers=False):
        """``params`` from ``model_preprocessing.create_bevformer_parameters``; ``all_layers`` as
        ``TTBEVFormerHead`` takes it, off by default."""
        config = params.config
        assert (
            config.num_cams == params.transformer.config.num_cams
        ), f"{config.num_cams} cameras in the images, {params.transformer.config.num_cams} in the transformer"
        assert (config.bev_h, config.bev_w) == tuple(
            params.head.bev_shape
        ), f"the detector's BEV is {config.bev_h}x{config.bev_w}, the head's {tuple(params.head.bev_shape)}"
        self.params = params
        self.batch_size = config.batch_size
        self.backbone = TTResNet(params.backbone, device, out_indices=config.out_indices)
        self.neck = TTFPN(params.neck, device, input_dtypes=self.backbone.output_dtypes)
        self.transformer = TTPerceptionTransformer(
            params.transformer, device, bev_h=config.bev_h, bev_w=config.bev_w, spatial_shapes=config.spatial_shapes
        )
        self.head = TTBEVFormerHead(params.head, device, all_layers=all_layers)
        bev_pos = params.bev_pos
        if self.batch_size > 1:
            bev_pos = ttnn.repeat(bev_pos, ttnn.Shape((self.batch_size, 1, 1)))
        self.bev_pos = bev_pos
        self.coder = TTNMSFreeCoder(device, **vars(params.coder))

    @property
    def planner(self):
        """The encoder's ``SCARebatchPlanner``, which sizes every frame's spatial cross-attention plan."""
        return self.transformer.encoder.rebatch_planner

    # The live frame: the steps below, chained, with the host waiting on the counts mid-frame.

    def prepare_frame(self, img_metas, frame=None):
        """See ``TTPerceptionTransformer.prepare_frame``."""
        self._check_batch(img_metas)
        return self.transformer.prepare_frame(img_metas, frame)

    def __call__(self, img, frame, prev_bev=None):
        """``img`` the ``bs * num_cams`` normalized images, sample-major, one DRAM tensor per camera
        as TTResNet reads them; ``frame`` from :meth:`prepare_frame`; ``prev_bev`` the
        previous frame's ``(bs, bev_h * bev_w, C)`` BEV or None. Returns the class logits
        ``(L, bs, num_query, num_classes)``, boxes ``(L, bs, num_query, code_size)`` in
        ``TTBEVFormerHead``'s channel order, and this frame's BEV ``(bs, bev_h * bev_w, C)``. L is
        the number of decoder layers with ``all_layers``, else 1, the last layer. The FPN frees
        the backbone's features, not ``img``."""
        return self.bev(self.features(img), frame, prev_bev=prev_bev)

    # Host steps: they compute on the host, write device buffers or read results back, so they stay
    # outside traces.

    def host_frame(self, img_metas):
        """See ``TTPerceptionTransformer.host_frame``."""
        self._check_batch(img_metas)
        return self.transformer.host_frame(img_metas)

    def refill_frame(self, img_metas, frame=None, host_frame=None):
        """See ``TTPerceptionTransformer.refill_frame``."""
        self._check_batch(img_metas)
        return self.transformer.refill_frame(img_metas, frame, host_frame)

    @staticmethod
    def get_bboxes(scores, labels, boxes, valid):
        """:meth:`detect`'s outputs to one ``(boxes, scores, labels)`` per sample on the host, the
        valid ones, as ``reference.bevformer.BEVFormer.get_bboxes`` returns them."""
        return [
            (sample["bboxes"], sample["scores"], sample["labels"])
            for sample in TTNMSFreeCoder.compact(scores, labels, boxes, valid)
        ]

    # Device steps: they only queue device work, with no readback, upload or wait, which is what
    # lets a trace record them.

    def project_cameras(self, frame):
        """``frame.cameras``: the cameras projected through ``frame``'s buffers; ``planner.read_counts``
        queues the counts' readback. The ``cameras`` trace records it."""
        frame.cameras = self.transformer.encoder.project_cameras_on_device(frame.projection)
        return frame.cameras

    def features(self, img):
        """The FPN levels of :meth:`__call__`'s ``img``. The ``features`` trace records it."""
        return self.neck(list(self.backbone(img)))

    def build_plan(self, frame):
        """``frame.cameras`` with the spatial cross-attention plan built on it; the planner must have
        sized this frame (``planner.size``). The ``bev`` trace records it."""
        cameras = frame.cameras
        frame.cameras = cameras._replace(
            rebatch_plan=self.planner.build(cameras.reference_points_cam, cameras.visibility)
        )
        return frame.cameras

    def bev(self, levels, frame, prev_bev=None):
        """:meth:`__call__` from the FPN ``levels``. Device work only when :meth:`build_plan` has put
        the plan on ``frame.cameras``, as in the ``bev`` trace; without one it builds the plan here,
        waiting for the counts."""
        bev_embed = self.transformer(levels, self.params.bev_queries, self.bev_pos, frame, prev_bev=prev_bev)
        cls_scores, bbox_preds = self.head(bev_embed)
        return _to_dram(cls_scores, bbox_preds) + (bev_embed,)

    def detect(self, cls_scores, bbox_preds, score_threshold=None):
        """The last decoder layer's detections, on device: ``TTNMSFreeCoder.topk``'s scores, labels
        and query indexes, and ``TTNMSFreeCoder.finalize``'s boxes and validity mask, which also
        drops scores below ``score_threshold`` when given. The ``bev`` trace records it."""
        signpost(header="TTNN Detect Start")
        scores, labels, query_index, boxes = self.coder.topk(cls_scores[-1], bbox_preds[-1])
        boxes, valid = self.coder.finalize(scores, boxes)
        if score_threshold is not None:
            # topk's scores are ROW_MAJOR; valid is tiled.
            above = ttnn.ge(ttnn.to_layout(scores, ttnn.TILE_LAYOUT), score_threshold)
            valid = ttnn.logical_and(valid, above)
        outputs = _to_dram(scores, labels, query_index, boxes, valid)
        signpost(header="TTNN Detect End")
        return outputs

    def _check_batch(self, img_metas):
        assert (
            len(img_metas) == self.batch_size
        ), f"{len(img_metas)} samples, the detector is built for {self.batch_size}"
