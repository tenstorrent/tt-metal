# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
"""Folded image input, Hiera/FPN execution, bounded host transport, and image prediction."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.vision.segmentation.sam2.tt.tt_hiera import TtHiera
from models.demos.vision.segmentation.sam2.tt.tt_mask_decoder import TtMaskDecoder
from models.demos.vision.segmentation.sam2.tt.tt_prompt_encoder import TtPromptEncoder
from models.tt_cnn.tt.builder import Conv2dConfiguration, HeightShardedStrategyConfiguration, TtConv2d


@dataclass
class EncodedFrame:
    top_nhwc: ttnn.Tensor
    high_res_0: ttnn.Tensor
    high_res_1: ttnn.Tensor
    owns_tensors: bool = True
    _released: bool = False
    _bridge_slot: int | None = None
    _bridge_ready: bool = False
    _encoder_slot: int | None = None
    _encoder_ready_event: object | None = None

    def _ensure_live(self):
        if self._released:
            raise RuntimeError("EncodedFrame has been released")

    @property
    def high_res(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self._ensure_live()
        return self.high_res_0, self.high_res_1

    def tensors(self) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        self._ensure_live()
        return self.top_nhwc, self.high_res_0, self.high_res_1

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.owns_tensors:
            for tensor in (self.top_nhwc, self.high_res_0, self.high_res_1):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)


@dataclass
class PendingBridgeTransfer:
    source: EncodedFrame
    slot: int
    host_tensors: tuple[ttnn.Tensor, ...]
    sender_done_event: object
    finished: bool = False


class HostTensorBridge:
    """Two-slot host bridge; CQ1 upload can overlap CQ0 tracking."""

    _SLOT_COUNT = 2

    def __init__(self, sender_device, receiver_device, upload_cq_id):
        if tuple(sender_device.shape) != (1, 1) or tuple(receiver_device.shape) != (1, 1):
            raise ValueError("SAM2 host bridge endpoints must be unit meshes")
        if set(sender_device.get_device_ids()) & set(receiver_device.get_device_ids()):
            raise ValueError("SAM2 host bridge endpoints must use different ASICs")
        if upload_cq_id not in (0, 1):
            raise ValueError(f"SAM2 host bridge upload queue must be 0 or 1, got {upload_cq_id}")
        self.sender_device = sender_device
        self.receiver_device = receiver_device
        self.upload_cq_id = upload_cq_id
        self._receiver_encodings = []
        self._host_slots = [None] * self._SLOT_COUNT
        self._pending_uploads = [None] * self._SLOT_COUNT
        self._ready_events = [None] * self._SLOT_COUNT
        self._read_events = [None] * self._SLOT_COUNT
        self._next_slot = 0
        self._closed = False

    def _allocate_receiver_encodings(self, host_tensors):
        slots = []
        current_tensors = []
        try:
            for _ in range(self._SLOT_COUNT):
                current_tensors = []
                for host_tensor in host_tensors:
                    current_tensors.append(
                        ttnn.empty(
                            list(host_tensor.shape),
                            dtype=host_tensor.dtype,
                            layout=host_tensor.layout,
                            device=self.receiver_device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    )
                slots.append(EncodedFrame(*current_tensors))
                current_tensors = []
        except BaseException:
            for slot in slots:
                slot.release()
            for tensor in current_tensors:
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            raise
        self._receiver_encodings = slots

    def _host_slot(self, slot, source_tensors):
        host_tensors = self._host_slots[slot]
        if host_tensors is None:
            host_tensors = tuple(
                ttnn.allocate_tensor_on_host(
                    tensor.shape,
                    tensor.dtype,
                    tensor.layout,
                    self.sender_device,
                    tensor.memory_config(),
                )
                for tensor in source_tensors
            )
            self._host_slots[slot] = host_tensors
        return host_tensors

    def _retire_pending_upload(self, slot):
        pending = self._pending_uploads[slot]
        if pending is None:
            return
        event, _ = pending
        ttnn.event_synchronize(event)
        self._pending_uploads[slot] = None

    def begin_transfer(self, encoding: EncodedFrame) -> PendingBridgeTransfer:
        if self._closed:
            raise RuntimeError("SAM2 host bridge is closed")

        source_tensors = encoding.tensors()
        slot = self._next_slot
        self._next_slot = (slot + 1) % self._SLOT_COUNT
        self._retire_pending_upload(slot)
        host_tensors = self._host_slot(slot, source_tensors)
        try:
            if encoding._encoder_ready_event is not None:
                ttnn.wait_for_event(0, encoding._encoder_ready_event)
            for source_tensor, host_tensor in zip(source_tensors, host_tensors):
                ttnn.copy_device_to_host_tensor(
                    source_tensor,
                    host_tensor,
                    blocking=False,
                    cq_id=0,
                )
            sender_done_event = ttnn.record_event(self.sender_device, 0)
        except BaseException:
            ttnn.synchronize_device(self.sender_device)
            encoding.release()
            raise
        return PendingBridgeTransfer(encoding, slot, host_tensors, sender_done_event)

    def finish_transfer(self, pending: PendingBridgeTransfer) -> EncodedFrame:
        if self._closed:
            raise RuntimeError("SAM2 host bridge is closed")
        if pending.finished:
            raise RuntimeError("SAM2 host bridge transfer has already finished")

        receiver_work_started = False
        upload_event = None
        try:
            ttnn.event_synchronize(pending.sender_done_event)
            if not self._receiver_encodings:
                self._allocate_receiver_encodings(pending.host_tensors)
            destination = self._receiver_encodings[pending.slot]
            previous_read = self._read_events[pending.slot]
            if previous_read is not None:
                ttnn.wait_for_event(self.upload_cq_id, previous_read)
            receiver_work_started = True
            for host_tensor, receiver_tensor in zip(pending.host_tensors, destination.tensors()):
                ttnn.copy_host_to_device_tensor(host_tensor, receiver_tensor, self.upload_cq_id)
            upload_event = ttnn.record_event(self.receiver_device, self.upload_cq_id)
            self._pending_uploads[pending.slot] = (upload_event, pending.host_tensors)
            self._ready_events[pending.slot] = upload_event
            return EncodedFrame(*destination.tensors(), owns_tensors=False, _bridge_slot=pending.slot)
        finally:
            if receiver_work_started and upload_event is None:
                ttnn.synchronize_device(self.receiver_device)
            pending.finished = True
            pending.host_tensors = ()
            pending.source.release()

    def abort_transfer(self, pending: PendingBridgeTransfer) -> None:
        if pending.finished:
            return
        try:
            ttnn.event_synchronize(pending.sender_done_event)
        finally:
            pending.finished = True
            pending.host_tensors = ()
            pending.source.release()

    def wait_until_ready(self, encoding: EncodedFrame) -> None:
        slot = encoding._bridge_slot
        if slot is None or encoding._bridge_ready:
            return
        encoding._ensure_live()
        event = self._ready_events[slot]
        if event is None:
            raise RuntimeError(f"SAM2 host bridge slot {slot} has no upload event")
        ttnn.wait_for_event(0, event)
        encoding._bridge_ready = True

    def mark_consumed(self, encoding: EncodedFrame) -> None:
        slot = encoding._bridge_slot
        if slot is None:
            return
        if self._closed:
            raise RuntimeError("SAM2 host bridge is closed")
        self.wait_until_ready(encoding)
        self._read_events[slot] = ttnn.record_event(self.receiver_device, 0)
        encoding._bridge_slot = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for slot in range(self._SLOT_COUNT):
                self._retire_pending_upload(slot)
            for event in self._read_events:
                if event is not None:
                    ttnn.event_synchronize(event)
        finally:
            for encoding in self._receiver_encodings:
                encoding.release()
            self.__dict__.clear()
            self._closed = True


FPN_OUTPUT_DTYPE = ttnn.bfloat16


class TtFpnNeck:
    def __init__(self, parameters, device, backbone_channel_list, fpn_top_down_levels):
        self.backbone_channels = tuple(backbone_channel_list)
        self.top_down_levels = frozenset(fpn_top_down_levels)
        self.convs = [
            TtConv2d(
                Conv2dConfiguration(
                    input_height=input_size,
                    input_width=input_size,
                    in_channels=in_channels,
                    out_channels=layer.conv.out_channels,
                    batch_size=1,
                    kernel_size=(1, 1),
                    weight=layer.conv.weight,
                    bias=layer.conv.bias,
                    sharding_strategy=HeightShardedStrategyConfiguration(),
                    enable_weights_double_buffer=False,
                    deallocate_activation=True,
                ),
                device,
            )
            for layer, input_size, in_channels in zip(parameters.convs, (32, 64, 128, 256), self.backbone_channels)
        ]

    def __call__(self, xs: list[ttnn.Tensor]):
        outputs = [None] * len(xs)
        previous = None
        final_level = len(xs) - 1
        for level in range(final_level, -1, -1):
            parameter_index = final_level - level
            conv = self.convs[parameter_index]
            lateral, (height, width) = conv(xs[level], return_output_dim=True)
            lateral = ttnn.to_memory_config(lateral, ttnn.DRAM_MEMORY_CONFIG)
            lateral = ttnn.reshape(lateral, (1, height, width, conv.configuration.out_channels))
            if level in self.top_down_levels and previous is not None:
                upsampled = ttnn.upsample(previous, scale_factor=2, mode="nearest")
                previous = ttnn.add(
                    lateral,
                    upsampled,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=FPN_OUTPUT_DTYPE,
                )
                ttnn.deallocate(upsampled)
            else:
                previous = lateral
            outputs[level] = previous
        return outputs


class FoldedImageInputs:
    def __init__(self, image_size, pipeline_depth):
        self.image_size = image_size
        folded_size = image_size // 4 + 1
        self._storage = [
            torch.zeros((1, folded_size, folded_size, 3, 4, 4), dtype=torch.bfloat16) for _ in range(pipeline_depth)
        ]

    def prepare(self, pixel_values, slot):
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("pixel_values must be a torch.Tensor")
        expected_shape = (1, 3, self.image_size, self.image_size)
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(f"pixel_values must have shape {expected_shape}, got {tuple(pixel_values.shape)}")
        if pixel_values.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("pixel_values must use torch.float32 or torch.bfloat16")

        folded = self._storage[slot]
        for local_y in range(4):
            source_y = (local_y - 3) % 4
            block_y = (source_y + 3) // 4
            for local_x in range(4):
                source_x = (local_x - 3) % 4
                block_x = (source_x + 3) // 4
                source = pixel_values[:, :, source_y::4, source_x::4]
                source_height, source_width = source.shape[-2:]
                folded[
                    :,
                    block_y : block_y + source_height,
                    block_x : block_x + source_width,
                    :,
                    local_y,
                    local_x,
                ].copy_(source.permute(0, 2, 3, 1))
        return folded.view(1, folded.shape[1], folded.shape[2], 48)

    def close(self):
        self._storage = []


class TtSam2ImageEncoder:
    def __init__(self, params, device, cfg, io_cq_id=0):
        vision_config = cfg.vision_config
        self.device = device
        self.num_feature_levels = vision_config.num_feature_levels
        self.trunk = TtHiera(
            params.trunk,
            device,
        )
        self.neck = TtFpnNeck(
            params.neck,
            device,
            backbone_channel_list=vision_config.backbone_channel_list,
            fpn_top_down_levels=vision_config.fpn_top_down_levels,
        )
        self.io_cq_id = io_cq_id
        self.pipeline_depth = 2 if io_cq_id == 1 else 1
        self._trace_ids = []
        self._trace_inputs = []
        self._trace_outputs = []
        self._pending_host_inputs = [None] * self.pipeline_depth
        self._upload_events = [None] * self.pipeline_depth
        self._output_read_events = [None] * self.pipeline_depth
        self._next_slot = 0
        self._closed = False

    def forward_image(self, frame_nhwc):
        features = self.neck(self.trunk.forward(frame_nhwc))
        fpn = []
        for feature in features[-self.num_feature_levels - 1 : -1]:
            if feature.layout == ttnn.TILE_LAYOUT:
                fpn.append(feature)
            else:
                fpn.append(ttnn.to_layout(feature, ttnn.TILE_LAYOUT))
                ttnn.deallocate(feature)
        if features[-1].is_allocated():
            ttnn.deallocate(features[-1])
        for index in (0, 1):
            compressed = ttnn.typecast(
                fpn[index],
                ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if fpn[index].is_allocated():
                ttnn.deallocate(fpn[index])
            fpn[index] = compressed
        return EncodedFrame(fpn[-1], fpn[0], fpn[1])

    def _run_graph(self, frame_nhwc, output=None):
        fresh = self.forward_image(frame_nhwc)
        if output is None:
            return fresh
        try:
            for source, destination in zip(fresh.tensors(), output.tensors()):
                ttnn.copy(source, destination)
        finally:
            fresh.release()
        return output

    @property
    def next_input_slot(self):
        return self._next_slot

    def _capture_traces(self, host_input):
        warmup_output = None
        current_trace_id = None
        trace_ids = []
        trace_inputs = []
        trace_outputs = []
        try:
            trace_inputs = [
                ttnn.to_device(
                    host_input,
                    self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(self.pipeline_depth)
            ]
            warmup_output = self._run_graph(trace_inputs[0])
            ttnn.synchronize_device(self.device)
            for _ in range(self.pipeline_depth):
                output_tensors = [
                    ttnn.empty(
                        list(tensor.shape),
                        dtype=tensor.dtype,
                        layout=tensor.layout,
                        device=self.device,
                        memory_config=tensor.memory_config(),
                    )
                    for tensor in warmup_output.tensors()
                ]
                trace_outputs.append(EncodedFrame(*output_tensors))
            for output in trace_outputs:
                for source, destination in zip(warmup_output.tensors(), output.tensors()):
                    ttnn.copy(source, destination)
            ttnn.synchronize_device(self.device)
            warmup_output.release()
            warmup_output = None

            # All buffers that survive replay are allocated before the first trace.
            # The second capture may reuse the first trace's temporary addresses;
            # those temporaries are released before either trace is replayed, and
            # CQ0 serializes the two traces.
            for slot in range(self.pipeline_depth):
                current_trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
                try:
                    self._run_graph(trace_inputs[slot], trace_outputs[slot])
                finally:
                    ttnn.end_trace_capture(self.device, current_trace_id, cq_id=0)
                trace_ids.append(current_trace_id)
                current_trace_id = None

            ttnn.execute_trace(self.device, trace_ids[0], cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
        except BaseException:
            if current_trace_id is not None:
                ttnn.release_trace(self.device, current_trace_id)
            if warmup_output is not None:
                warmup_output.release()
            for trace_id in trace_ids:
                ttnn.release_trace(self.device, trace_id)
            for trace_output in trace_outputs:
                trace_output.release()
            for tensor in trace_inputs:
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            raise

        self._trace_ids = trace_ids
        self._trace_inputs = trace_inputs
        self._trace_outputs = trace_outputs

    def forward_eager_folded(self, folded_input):
        if self._closed:
            raise RuntimeError("SAM2 image encoder is closed")
        if self._trace_ids:
            raise RuntimeError("release SAM2 encoder traces before eager execution")
        host_input = ttnn.from_torch(
            folded_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        device_input = ttnn.to_device(
            host_input,
            self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            return self._run_graph(device_input)
        finally:
            if device_input.is_allocated():
                ttnn.deallocate(device_input)

    def forward_folded(self, folded_input, slot=None):
        if self._closed:
            raise RuntimeError("SAM2 image encoder is closed")
        if slot is None:
            slot = self._next_slot
        if slot != self._next_slot:
            raise RuntimeError(f"expected SAM2 image encoder slot {self._next_slot}, got {slot}")
        host_input = ttnn.from_torch(
            folded_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ready_event = None
        if not self._trace_ids:
            self._capture_traces(host_input)
        else:
            read_event = self._output_read_events[slot]
            if read_event is not None:
                ttnn.event_synchronize(read_event)
                self._output_read_events[slot] = None
            if self._pending_host_inputs[slot] is not None:
                raise RuntimeError(f"finish SAM2 image encode slot {slot} before reusing it")
            self._pending_host_inputs[slot] = host_input
            try:
                ttnn.copy_host_to_device_tensor(
                    host_input,
                    self._trace_inputs[slot],
                    cq_id=self.io_cq_id,
                )
                upload_event = ttnn.record_event(self.device, self.io_cq_id)
                self._upload_events[slot] = upload_event
                if self.io_cq_id != 0:
                    ttnn.wait_for_event(0, upload_event)
                ttnn.execute_trace(self.device, self._trace_ids[slot], cq_id=0, blocking=False)
                ready_event = ttnn.record_event(self.device, 0)
            except BaseException:
                self.abort_pending_host_input(slot)
                raise
        output = EncodedFrame(
            *self._trace_outputs[slot].tensors(),
            owns_tensors=False,
            _encoder_slot=slot,
            _encoder_ready_event=ready_event,
        )
        self._next_slot = (slot + 1) % self.pipeline_depth
        return output

    def mark_output_read(self, encoding, event):
        slot = encoding._encoder_slot
        if slot is None:
            return
        if self._output_read_events[slot] is not None:
            raise RuntimeError(f"SAM2 image encoder slot {slot} already has a pending read")
        self._output_read_events[slot] = event

    def release_pending_host_input(self, slot=None):
        slots = range(self.pipeline_depth) if slot is None else (slot,)
        for current_slot in slots:
            if self._pending_host_inputs[current_slot] is None:
                continue
            try:
                upload_event = self._upload_events[current_slot]
                if upload_event is not None:
                    ttnn.event_synchronize(upload_event)
            finally:
                self._pending_host_inputs[current_slot] = None
                self._upload_events[current_slot] = None

    def abort_pending_host_input(self, slot=None):
        slots = range(self.pipeline_depth) if slot is None else (slot,)
        if not any(self._pending_host_inputs[current_slot] is not None for current_slot in slots):
            return
        try:
            ttnn.synchronize_device(self.device)
        finally:
            for current_slot in slots:
                self._pending_host_inputs[current_slot] = None
                self._upload_events[current_slot] = None

    def release_traces(self):
        self.abort_pending_host_input()
        if self._trace_ids:
            ttnn.synchronize_device(self.device)
        for read_event in self._output_read_events:
            if read_event is not None:
                ttnn.event_synchronize(read_event)
        for trace_id in self._trace_ids:
            ttnn.release_trace(self.device, trace_id)
        for trace_output in self._trace_outputs:
            trace_output.release()
        for tensor in self._trace_inputs:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        self._trace_ids = []
        self._trace_outputs = []
        self._trace_inputs = []
        self._pending_host_inputs = [None] * self.pipeline_depth
        self._upload_events = [None] * self.pipeline_depth
        self._output_read_events = [None] * self.pipeline_depth
        self._next_slot = 0

    def close(self):
        if self._closed:
            return
        self.release_traces()
        self._closed = True


class TtSam2ImageHead:
    """Image-only prompt and mask head colocated with the Hiera encoder."""

    def __init__(self, params, device):
        self.hidden_dim = 256
        self.feat_hw = (64, 64)
        self.prompt_encoder = TtPromptEncoder(params.sam_prompt_encoder, device)
        self.sam_mask_decoder = TtMaskDecoder(params.sam_mask_decoder, device)
        self.dense_pe_seq = params.sam_prompt_encoder.dense_pe_seq
        self.no_mask_dense_seq = params.sam_prompt_encoder.no_mask_dense_seq
        self.no_mem_embed = params.no_mem_embed_dev
        self._closed = False

    def predict(
        self,
        encoding,
        *,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        input_masks=None,
        multimask_output=True,
    ):
        if self._closed:
            raise RuntimeError("SAM2 image head is closed")
        sparse_is_cached = input_points is None and input_boxes is None
        sparse = None if sparse_is_cached else self.prompt_encoder.embed_sparse(input_points, input_labels, input_boxes)
        dense_bchw = None
        dense_is_cached = input_masks is None
        if dense_is_cached:
            dense = self.no_mask_dense_seq
        else:
            dense_bchw = self.prompt_encoder.embed_dense(input_masks)
            dense_nhwc = ttnn.permute(dense_bchw, (0, 2, 3, 1))
            dense_sequence = ttnn.reshape(dense_nhwc, (1, self.feat_hw[0] * self.feat_hw[1], self.hidden_dim))
            if dense_sequence.layout == ttnn.TILE_LAYOUT:
                dense = ttnn.clone(dense_sequence, dtype=dense_sequence.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                dense = ttnn.to_layout(dense_sequence, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if dense_nhwc.is_allocated():
                ttnn.deallocate(dense_nhwc)
        batch, height, width, _ = encoding.top_nhwc.shape
        image_features = ttnn.add(
            ttnn.reshape(encoding.top_nhwc, (batch, height * width, self.hidden_dim)),
            self.no_mem_embed,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            masks, iou_scores, mask_tokens, object_score_logits = self.sam_mask_decoder(
                image_embeddings=image_features,
                image_pe=self.dense_pe_seq,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=multimask_output,
                high_res_features=encoding.high_res,
            )
        finally:
            if image_features.is_allocated():
                ttnn.deallocate(image_features)
            if not sparse_is_cached and sparse is not None and sparse.is_allocated():
                ttnn.deallocate(sparse)
            if not dense_is_cached:
                if dense.is_allocated():
                    ttnn.deallocate(dense)
                if dense_bchw.is_allocated():
                    ttnn.deallocate(dense_bchw)
        return {
            "low_res_masks": masks,
            "iou_scores": iou_scores,
            "mask_tokens": mask_tokens,
            "object_score_logits": object_score_logits,
        }

    def close(self):
        if self._closed:
            return
        self.__dict__.clear()
        self._closed = True
