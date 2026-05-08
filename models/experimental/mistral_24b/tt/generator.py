# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.generator import Generator

try:
    from tracy import signpost
except ImportError:

    def signpost(*args, **kwargs):
        pass


class MistralGenerator(Generator):
    """
    Mistral-specific trace prefill override that keeps multimodal kwargs
    (`processed_inputs`, `vision_model`) on capture and replay.
    """

    def _capture_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        kv_cache=None,
        model_id=-1,
        global_user_id=None,
        batch_size=1,
        user_id=0,
        **kwargs,
    ):
        signpost("Mistral24B::TraceCapturePrefill::Start")
        if batch_size > 1:
            prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": user_id}
            if global_user_id is not None:
                prefill_kwargs["global_user_id"] = global_user_id
            prefill_kwargs.update(kwargs)
            signpost("Mistral24B::PrefillPreprocess::Start", f"batch_size={batch_size}")
            host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
            signpost("Mistral24B::PrefillPreprocess::End", f"batch_size={batch_size}")
            tt_rot_mats_prefill_global = host_inputs[1]
            tt_rot_mats_prefill_local = host_inputs[2]
            host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

            signpost("Mistral24B::DeviceTransfer::HostToDevice::Start", "prefill trace compile inputs")
            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            signpost("Mistral24B::DeviceTransfer::HostToDevice::End", "prefill trace compile inputs")
            signpost("Mistral24B::PrefillCompileWarmup::Start", f"batch_size={batch_size}")
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
            signpost("Mistral24B::PrefillCompileWarmup::End", f"batch_size={batch_size}")
            signpost("Mistral24B::Synchronize::Start", "prefill trace compile")
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            signpost("Mistral24B::Synchronize::End", "prefill trace compile")

            signpost("Mistral24B::DeviceTransfer::HostToDevice::Start", "prefill trace capture inputs")
            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            signpost("Mistral24B::DeviceTransfer::HostToDevice::End", "prefill trace capture inputs")
            signpost("Mistral24B::TraceCapture::Start", f"prefill batch_size={batch_size}")
            trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
            ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
            signpost("Mistral24B::TraceCapture::End", f"prefill trace_id={trace_id}")
            signpost("Mistral24B::Synchronize::Start", "prefill trace capture")
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            signpost("Mistral24B::Synchronize::End", "prefill trace capture")
            signpost("Mistral24B::TraceCapturePrefill::End")
            return trace_id, tt_out_trace, *device_inputs

        prefill_kwargs = {"page_table": page_table}
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id
        prefill_kwargs.update(kwargs)
        signpost("Mistral24B::PrefillPreprocess::Start", "batch_size=1")
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        signpost("Mistral24B::PrefillPreprocess::End", "batch_size=1")
        tt_rot_mats_prefill_global = host_inputs[1]
        tt_rot_mats_prefill_local = host_inputs[2]
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        signpost("Mistral24B::DeviceTransfer::HostToDevice::Start", "prefill trace compile inputs")
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        signpost("Mistral24B::DeviceTransfer::HostToDevice::End", "prefill trace compile inputs")
        signpost("Mistral24B::PrefillCompileWarmup::Start", "batch_size=1")
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        signpost("Mistral24B::PrefillCompileWarmup::End", "batch_size=1")
        signpost("Mistral24B::Synchronize::Start", "prefill trace compile")
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        signpost("Mistral24B::Synchronize::End", "prefill trace compile")

        signpost("Mistral24B::DeviceTransfer::HostToDevice::Start", "prefill trace capture inputs")
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        signpost("Mistral24B::DeviceTransfer::HostToDevice::End", "prefill trace capture inputs")
        signpost("Mistral24B::TraceCapture::Start", "prefill batch_size=1")
        trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
        signpost("Mistral24B::TraceCapture::End", f"prefill trace_id={trace_id}")
        signpost("Mistral24B::Synchronize::Start", "prefill trace capture")
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        signpost("Mistral24B::Synchronize::End", "prefill trace capture")
        signpost("Mistral24B::TraceCapturePrefill::End")
        return trace_id, tt_out_trace, *device_inputs

    def prefill_forward_text(self, *args, **kwargs):
        signpost(
            "Mistral24B::Prefill::Start",
            f"trace_enabled={kwargs.get('enable_trace', False)} multimodal={kwargs.get('processed_inputs') is not None}",
        )
        try:
            return super().prefill_forward_text(*args, **kwargs)
        finally:
            signpost("Mistral24B::Prefill::End")

    def decode_forward(self, *args, **kwargs):
        signpost("Mistral24B::DecodeStep::Start", f"trace_enabled={kwargs.get('enable_trace', False)}")
        try:
            return super().decode_forward(*args, **kwargs)
        finally:
            signpost("Mistral24B::DecodeStep::End")

    def _capture_decode_trace_text(self, *args, **kwargs):
        signpost("Mistral24B::TraceCapture::Start", "decode text")
        try:
            return super()._capture_decode_trace_text(*args, **kwargs)
        finally:
            signpost("Mistral24B::TraceCapture::End", "decode text")

    def _decode_forward_trace_text(self, *args, **kwargs):
        signpost("Mistral24B::TraceReplay::Start", "decode text")
        try:
            return super()._decode_forward_trace_text(*args, **kwargs)
        finally:
            signpost("Mistral24B::TraceReplay::End", "decode text")

    def _decode_forward_no_trace_text(self, *args, **kwargs):
        signpost("Mistral24B::DecodeNoTrace::Start", "text")
        try:
            return super()._decode_forward_no_trace_text(*args, **kwargs)
        finally:
            signpost("Mistral24B::DecodeNoTrace::End", "text")

    def _easy_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        user_id=0,
        last_token_idx=None,
        kv_cache=None,
        model_id=-1,
        prefill_seq_len=None,
        batch_size=1,
        **kwargs,
    ):
        global_user_id = kwargs.get("global_user_id", None)
        processed_inputs = kwargs.get("processed_inputs", None)
        use_multimodal_trace = (
            processed_inputs is not None
            and processed_inputs.get("pixel_values", None) is not None
            and kwargs.get("vision_model", None) is not None
        )
        trace_key = f"{prefill_seq_len}_{model_id}_{batch_size}_{'multimodal' if use_multimodal_trace else 'text'}"

        if self.trace_id_prefill[trace_key] is None:
            signpost("Mistral24B::TracePrefillCacheMiss", trace_key)
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                prefill_ids,
                page_table=page_table,
                kv_cache=kv_cache,
                model_id=model_id,
                global_user_id=global_user_id,
                batch_size=batch_size,
                user_id=user_id,
                **kwargs,
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace

        signpost("Mistral24B::TraceReplay::Start", f"prefill {trace_key}")
        tt_out_trace = self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            prefill_ids,
            page_table=page_table,
            model_id=model_id,
            global_user_id=global_user_id,
            batch_size=batch_size,
            user_id=user_id,
            **kwargs,
        )
        signpost("Mistral24B::TraceReplay::End", f"prefill {trace_key}")
        return tt_out_trace

    def _prefill_forward_trace(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        prefill_ids,
        user_id=0,
        page_table=None,
        model_id=-1,
        global_user_id=None,
        batch_size=1,
        **kwargs,
    ):
        prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": user_id}
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id
        prefill_kwargs.update(kwargs)
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        signpost("Mistral24B::DeviceTransfer::HostToDevice::Start", "prefill trace replay inputs")
        copy_host_to_device(
            host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
        )
        signpost("Mistral24B::DeviceTransfer::HostToDevice::End", "prefill trace replay inputs")
        signpost("Mistral24B::TraceExecute::Start", f"prefill trace_id={trace_id}")
        ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)
        signpost("Mistral24B::TraceExecute::End", f"prefill trace_id={trace_id}")
        return tt_out_trace
