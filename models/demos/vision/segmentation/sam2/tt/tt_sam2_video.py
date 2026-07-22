# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
"""Public SAM2 facade, bounded video sessions, lifecycle, and N300 construction."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Lock

import ttnn
from models.demos.vision.segmentation.sam2.tt.tt_sam2_image import (
    FoldedImageInputs,
    HostTensorBridge,
    PendingBridgeTransfer,
    TtSam2ImageEncoder,
    TtSam2ImageHead,
)
from models.demos.vision.segmentation.sam2.tt.tt_sam2_tracker import TtSam2Tracker

SAM2_L1_SMALL_SIZE = 12 * 8192


class TtSam2VideoModel:
    def __init__(
        self,
        image_parameters,
        encoder_parameters,
        tracker_parameters,
        encoder_device,
        tracker_device,
        cfg,
        *,
        num_maskmem=7,
        max_obj_ptrs_in_encoder=16,
        owned_submeshes,
        bridge_upload_cq_id=0,
    ):
        self.encoder_device = encoder_device
        self.tracker_device = tracker_device
        self._owned_submeshes = tuple(owned_submeshes)
        self._feature_bridge = HostTensorBridge(encoder_device, tracker_device, bridge_upload_cq_id)
        self._encoder_lock = Lock()
        self._closed = False
        self.num_maskmem = num_maskmem
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        self.hidden_dim = cfg.vision_config.fpn_hidden_size
        self.image_size = cfg.image_size
        self.image_encoder = TtSam2ImageEncoder(
            encoder_parameters,
            encoder_device,
            cfg,
            io_cq_id=bridge_upload_cq_id,
        )
        self._folded_inputs = FoldedImageInputs(self.image_size, self.image_encoder.pipeline_depth)
        self.image_head = TtSam2ImageHead(image_parameters, encoder_device)
        self.tracker = TtSam2Tracker(
            tracker_parameters,
            tracker_device,
            num_maskmem=num_maskmem,
            max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
            output_cq_id=bridge_upload_cq_id,
        )
        self._image_cache = None
        self._active_video_session = None

    def _release_image_encoding(self, encoding):
        if encoding is not None and not encoding._released:
            self._feature_bridge.mark_consumed(encoding)
            encoding.release()

    def _encode_image_local(self, pixel_values):
        slot = self.image_encoder.next_input_slot
        folded_input = self._folded_inputs.prepare(pixel_values, slot)
        return self.image_encoder.forward_folded(folded_input, slot)

    def _begin_image_transfer(self, encoding):
        pending = self._feature_bridge.begin_transfer(encoding)
        try:
            self.image_encoder.mark_output_read(encoding, pending.sender_done_event)
        except BaseException:
            self._feature_bridge.abort_transfer(pending)
            raise
        return pending

    def _finish_image_encode(self, encoding):
        source = encoding.source if isinstance(encoding, PendingBridgeTransfer) else encoding
        slot = source._encoder_slot
        try:
            if isinstance(encoding, PendingBridgeTransfer):
                return self._feature_bridge.finish_transfer(encoding)
            pending = self._begin_image_transfer(encoding)
            return self._feature_bridge.finish_transfer(pending)
        finally:
            self.image_encoder.release_pending_host_input(slot)

    def _abort_image_encode(self, encoding):
        if encoding is None:
            return
        source = encoding.source if isinstance(encoding, PendingBridgeTransfer) else encoding
        try:
            self.image_encoder.abort_pending_host_input(source._encoder_slot)
        finally:
            if isinstance(encoding, PendingBridgeTransfer):
                self._feature_bridge.abort_transfer(encoding)
            else:
                self._release_image_encoding(encoding)

    def _encode_image(self, pixel_values):
        with self._encoder_lock:
            encoding = self._encode_image_local(pixel_values)
            try:
                return self._finish_image_encode(encoding)
            except BaseException:
                self._abort_image_encode(encoding)
                raise

    def start_video_session(self):
        if self._active_video_session is not None:
            self._active_video_session.close()
        self.reset_image()
        session = TtSam2VideoSession(self)
        self._active_video_session = session
        return session

    def reset_image(self):
        encoding = self._image_cache
        self._image_cache = None
        if encoding is not None and not encoding._released:
            try:
                ttnn.synchronize_device(self.encoder_device)
            finally:
                encoding.release()

    def set_image(self, pixel_values):
        if self._active_video_session is not None:
            self._active_video_session.close()
        self.reset_image()
        with self._encoder_lock:
            # Prompt/mask-head allocations are unsafe while an encoder trace is active.
            self.image_encoder.release_traces()
            folded_input = self._folded_inputs.prepare(pixel_values, 0)
            encoding = self.image_encoder.forward_eager_folded(folded_input)
        self._image_cache = encoding

    def predict(
        self,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        input_masks=None,
        multimask_output=True,
    ):
        if self._image_cache is None:
            raise RuntimeError("set_image() must be called before predict()")
        if self._image_cache._encoder_ready_event is not None:
            ttnn.wait_for_event(0, self._image_cache._encoder_ready_event)
        return self.image_head.predict(
            self._image_cache,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
            multimask_output=multimask_output,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        stack = ExitStack()
        for submesh in self._owned_submeshes:
            stack.callback(ttnn.close_mesh_device, submesh)
        stack.callback(self.tracker.close)
        stack.callback(self.image_head.close)
        stack.callback(self.image_encoder.close)
        stack.callback(self._feature_bridge.close)
        stack.callback(self._folded_inputs.close)
        stack.callback(self.reset_image)
        if self._active_video_session is not None:
            stack.callback(self._active_video_session.close)
        try:
            stack.close()
        finally:
            self.__dict__.clear()
            self._closed = True


def release_track_output(output, *, force=False):
    if output is None or (output.get("_trace_owned", False) and not force):
        return
    for tensor in (
        output.get("pred_masks"),
        output.get("pred_masks_high_res"),
        output.get("_pred_masks_high_res_device"),
        output.get("obj_ptr"),
        output.get("object_score_logits"),
        output.get("v_maskmem"),
        *output.get("k_maskmem_rope", ()),
        *output.get("obj_ptr_k", ()),
        output.get("obj_ptr_v"),
    ):
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


class TtSam2VideoSession:
    """Bounded one-object streaming state; returned tensors remain session-owned."""

    def __init__(self, model):
        self.model = model
        self.frame_idx = 0
        self.bank = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self._non_cond_limit = max(model.num_maskmem - 1, model.max_obj_ptrs_in_encoder - 1)
        self._active_iterator = None
        self._closed = False

    def _prompt_inputs(self, prompts):
        if prompts is None:
            return None
        if not isinstance(prompts, dict):
            raise TypeError("prompts must be a dictionary")
        if prompts.get("input_masks") is not None:
            raise ValueError("video sessions accept point and box prompts, not mask prompts")
        points = prompts.get("input_points")
        labels = prompts.get("input_labels")
        boxes = prompts.get("input_boxes")
        if (points is None) != (labels is None):
            raise ValueError("video point prompts require both input_points and input_labels")
        if points is not None:
            points, labels = self.model.tracker.prompt_encoder._single_object_points(points, labels)
        if points is None and boxes is None:
            raise ValueError("the first video frame requires a point or box prompt")
        return {"point_coords": points, "point_labels": labels, "boxes": boxes}

    def _validate_first_frame_prompts(self, prompts):
        prompt_inputs = self._prompt_inputs(prompts)
        if self.frame_idx == 0 and prompt_inputs is None:
            raise ValueError("the first video frame requires a point or box prompt")
        if self.frame_idx > 0 and prompt_inputs is not None:
            raise ValueError("video sessions accept prompts only on the first conditioning frame")
        return prompt_inputs

    def _release_bank(self):
        for outputs in self.bank.values():
            for output in outputs.values():
                release_track_output(output)
            outputs.clear()

    def _close_after_failure(self):
        try:
            ttnn.synchronize_device(self.model.tracker_device)
        finally:
            self._release_bank()
            self._closed = True
            if self.model._active_video_session is self:
                self.model._active_video_session = None

    def _track_encoding(self, encoding, prompt_inputs):
        self.model._feature_bridge.wait_until_ready(encoding)
        try:
            output = self.model.tracker.track_step(
                self.frame_idx,
                encoding,
                prompt_inputs,
                self.bank,
            )
        finally:
            self.model._release_image_encoding(encoding)
        if self.frame_idx == 0:
            self.bank["cond_frame_outputs"][self.frame_idx] = output
        else:
            non_cond = self.bank["non_cond_frame_outputs"]
            non_cond[self.frame_idx] = output
            while len(non_cond) > self._non_cond_limit:
                oldest_frame = min(non_cond)
                release_track_output(non_cond.pop(oldest_frame))
        self.frame_idx += 1
        return output

    def _run_pipelined(self, frames, first_frame_prompts):
        pending_encoding = None
        try:
            try:
                first_frame = next(frames)
            except StopIteration:
                raise ValueError("video input iterable must contain at least one frame") from None

            prompt_inputs = self._validate_first_frame_prompts(first_frame_prompts)
            pending_encoding = self.model._encode_image(first_frame)
            while True:
                encoding = pending_encoding
                try:
                    next_frame = next(frames)
                except StopIteration:
                    pending_encoding = None
                    output = self._track_encoding(encoding, prompt_inputs)
                    ttnn.synchronize_device(self.model.tracker_device)
                    yield output
                    break

                local_next = None
                transfer_next = None
                with self.model._encoder_lock:
                    try:
                        local_next = self.model._encode_image_local(next_frame)
                        transfer_next = self.model._begin_image_transfer(local_next)
                        local_next = None
                        pending_encoding = None
                        output = self._track_encoding(encoding, prompt_inputs)
                        prompt_inputs = None
                        pending_encoding = self.model._finish_image_encode(transfer_next)
                        transfer_next = None
                    finally:
                        self.model._abort_image_encode(transfer_next if transfer_next is not None else local_next)
                yield output
        except GeneratorExit:
            raise
        except BaseException:
            self._close_after_failure()
            raise
        finally:
            self.model._release_image_encoding(pending_encoding)
            self._active_iterator = None

    def _run_deep_pipelined(self, frames, first_frame_prompts):
        remote_encoding = None
        local_encoding = None
        staged_encoding = None
        transfer = None
        encode_future = None
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam2-encoder")
        try:
            try:
                first_frame = next(frames)
            except StopIteration:
                raise ValueError("video input iterable must contain at least one frame") from None

            prompt_inputs = self._validate_first_frame_prompts(first_frame_prompts)
            with self.model._encoder_lock:
                local_encoding = self.model._encode_image_local(first_frame)
                try:
                    second_frame = next(frames)
                except StopIteration:
                    transfer = self.model._begin_image_transfer(local_encoding)
                    local_encoding = None
                    remote_encoding = self.model._finish_image_encode(transfer)
                    transfer = None
                else:
                    transfer = self.model._begin_image_transfer(local_encoding)
                    local_encoding = None
                    staged_encoding = self.model._encode_image_local(second_frame)
                    remote_encoding = self.model._finish_image_encode(transfer)
                    transfer = None
                    local_encoding = staged_encoding
                    staged_encoding = None

            if local_encoding is None:
                encoding = remote_encoding
                remote_encoding = None
                output = self._track_encoding(encoding, prompt_inputs)
                ttnn.synchronize_device(self.model.tracker_device)
                yield output
                return

            while True:
                try:
                    following_frame = next(frames)
                except StopIteration:
                    with self.model._encoder_lock:
                        transfer = self.model._begin_image_transfer(local_encoding)
                        local_encoding = None
                        encoding = remote_encoding
                        remote_encoding = None
                        output = self._track_encoding(encoding, prompt_inputs)
                        prompt_inputs = None
                        remote_encoding = self.model._finish_image_encode(transfer)
                        transfer = None
                    yield output

                    encoding = remote_encoding
                    remote_encoding = None
                    output = self._track_encoding(encoding, prompt_inputs)
                    ttnn.synchronize_device(self.model.tracker_device)
                    yield output
                    break

                with self.model._encoder_lock:
                    try:
                        transfer = self.model._begin_image_transfer(local_encoding)
                        local_encoding = None

                        encode_future = executor.submit(self.model._encode_image_local, following_frame)
                        encoding = remote_encoding
                        remote_encoding = None
                        output = self._track_encoding(encoding, prompt_inputs)
                        prompt_inputs = None
                        staged_encoding = encode_future.result()
                        encode_future = None
                        remote_encoding = self.model._finish_image_encode(transfer)
                        transfer = None
                        local_encoding = staged_encoding
                        staged_encoding = None
                    finally:
                        self.model._abort_image_encode(transfer)
                        self.model._abort_image_encode(staged_encoding)
                        transfer = staged_encoding = None
                yield output
        except GeneratorExit:
            raise
        except BaseException:
            self._close_after_failure()
            raise
        finally:
            if encode_future is not None:
                try:
                    abandoned_encoding = encode_future.result()
                except BaseException:
                    pass
                else:
                    self.model._abort_image_encode(abandoned_encoding)
                encode_future = None
            for encoding in (transfer, staged_encoding, local_encoding):
                self.model._abort_image_encode(encoding)
            self.model._release_image_encoding(remote_encoding)
            executor.shutdown(wait=True, cancel_futures=True)
            self._active_iterator = None

    def step(self, frame, prompts=None):
        if self._closed:
            raise RuntimeError("video session is closed")
        prompt_inputs = self._validate_first_frame_prompts(prompts)
        encoding = self.model._encode_image(frame)
        return self._track_encoding(encoding, prompt_inputs)

    def run(self, frames, first_frame_prompts):
        if self._closed:
            raise RuntimeError("video session is closed")
        if self._active_iterator is not None:
            raise RuntimeError("video session already has an active iterator")
        pipeline = self._run_deep_pipelined if self.model.image_encoder.pipeline_depth > 1 else self._run_pipelined
        iterator = pipeline(iter(frames), first_frame_prompts)
        self._active_iterator = iterator
        return iterator

    def close(self):
        if self._closed:
            return
        self._closed = True
        active_iterator = self._active_iterator
        self._active_iterator = None
        try:
            if active_iterator is not None:
                active_iterator.close()
        finally:
            try:
                if self.model.tracker_device is not None:
                    ttnn.synchronize_device(self.model.tracker_device)
            finally:
                self._release_bank()
                if self.model._active_video_session is self:
                    self.model._active_video_session = None


def _validate_n300_device(device) -> None:
    arch_name = ttnn.get_arch_name()
    if "wormhole_b0" not in arch_name:
        raise ValueError(f"SAM2 N300 requires Wormhole B0, got {arch_name}")

    mesh_shape = tuple(device.shape)
    device_ids = tuple(device.get_device_ids())
    pcie_device_ids = tuple(ttnn.get_pcie_device_ids())
    if mesh_shape not in ((1, 2), (2, 1)) or len(device_ids) != 2:
        raise ValueError(f"SAM2 requires an exact two-ASIC N300 mesh, got shape {mesh_shape} and IDs {device_ids}")
    if ttnn.get_num_devices() != 2:
        raise ValueError(f"SAM2 requires exactly two discovered N300 ASICs, got {ttnn.get_num_devices()}")
    if len(pcie_device_ids) != 1 or pcie_device_ids[0] not in device_ids:
        raise ValueError(
            "SAM2 requires an N300 with one PCIe-attached ASIC and one remote ASIC; "
            f"got mesh IDs {device_ids} and PCIe IDs {pcie_device_ids}"
        )


def build_tt_sam2_model(
    hf_model,
    device,
    num_maskmem=7,
    max_obj_ptrs_in_encoder=16,
    bridge_upload_cq_id=0,
):
    from models.demos.vision.segmentation.sam2.tt.model_preprocessing import (
        preprocess_sam2_image_head_parameters,
        preprocess_sam2_video_encoder_parameters,
        preprocess_sam2_video_tracker_parameters,
    )

    _validate_n300_device(device)

    owned_submeshes = tuple(device.create_submeshes(ttnn.MeshShape(1, 1)))
    pcie_device_id = tuple(ttnn.get_pcie_device_ids())[0]
    if tuple(owned_submeshes[0].get_device_ids()) == (pcie_device_id,):
        encoder_device, tracker_device = owned_submeshes
    else:
        tracker_device, encoder_device = owned_submeshes

    hf_model = hf_model.eval()
    try:
        encoder_parameters = preprocess_sam2_video_encoder_parameters(hf_model, encoder_device)
        image_parameters = preprocess_sam2_image_head_parameters(hf_model, encoder_device)
        tracker_parameters = preprocess_sam2_video_tracker_parameters(hf_model, tracker_device)
        return TtSam2VideoModel(
            image_parameters,
            encoder_parameters,
            tracker_parameters,
            encoder_device,
            tracker_device,
            hf_model.config,
            owned_submeshes=owned_submeshes,
            num_maskmem=num_maskmem,
            max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
            bridge_upload_cq_id=bridge_upload_cq_id,
        )
    except BaseException:
        for submesh in owned_submeshes:
            ttnn.close_mesh_device(submesh)
        raise
