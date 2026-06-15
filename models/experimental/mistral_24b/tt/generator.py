# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.generator import Generator


class MistralGenerator(Generator):
    """Mistral prefill trace override; caches fused vision+text host_inputs to skip redundant vision forward on replay."""

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
        _pi = kwargs.get("processed_inputs", None)
        _is_multimodal = (
            _pi is not None and _pi.get("pixel_values") is not None
        )  # Detect multimodal input for cache path.

        if batch_size > 1:
            prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": user_id}
            if global_user_id is not None:
                prefill_kwargs["global_user_id"] = global_user_id
            prefill_kwargs.update(kwargs)
            host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
            tt_rot_mats_prefill_global = host_inputs[1]
            tt_rot_mats_prefill_local = host_inputs[2]
            host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])
            if _is_multimodal:
                # Cache fused embeddings so _prefill_forward_trace can skip re-running the vision model on replay.
                self._prefill_replay_cache = host_inputs

            # Warmup pass: run once un-traced so KV cache is populated before trace capture begins.
            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
            transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
            self.model[model_id].ttnn_prefill_forward(
                x=transformed_inputs[0],
                rot_mats_global=tt_rot_mats_prefill_global,
                rot_mats_local=tt_rot_mats_prefill_local,
                page_table=transformed_inputs[1],
                chunk_page_table=transformed_inputs[2],
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)

            # Capture pass: record the steady-state forward graph into a replayable trace.
            device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
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
            ttnn.synchronize_device(self.model_args[model_id].mesh_device)
            return trace_id, tt_out_trace, *device_inputs

        prefill_kwargs = {"page_table": page_table}
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id
        prefill_kwargs.update(kwargs)
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        tt_rot_mats_prefill_global = host_inputs[1]
        tt_rot_mats_prefill_local = host_inputs[2]
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])
        if _is_multimodal:
            # Cache fused embeddings so _prefill_forward_trace can skip re-running the vision model on replay.
            self._prefill_replay_cache = host_inputs

        # Warmup pass: run once un-traced so KV cache is populated before trace capture begins.
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)

        # Capture pass: record the steady-state forward graph into a replayable trace.
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
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
        ttnn.synchronize_device(self.model_args[model_id].mesh_device)
        return trace_id, tt_out_trace, *device_inputs

    def prefill_forward_text(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        # Per-step full device barrier: the post-#45166 self-feeding on-device decode lacks the
        # implicit per-step drain the host-argmax path has, exposing a logits read-before-CCL-complete
        # race that makes greedy decode non-deterministic. Draining each step restores determinism.
        out = super().decode_forward(*args, **kwargs)
        ttnn.synchronize_device(self.model_args[0].mesh_device)
        return out

    def _capture_decode_trace_text(self, *args, **kwargs):
        return super()._capture_decode_trace_text(*args, **kwargs)

    def _decode_forward_trace_text(self, *args, **kwargs):
        return super()._decode_forward_trace_text(*args, **kwargs)

    def _decode_forward_no_trace_text(self, *args, **kwargs):
        return super()._decode_forward_no_trace_text(*args, **kwargs)

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
        # Separate trace keys for multimodal vs text-only: they differ in input dtype (embeddings vs token ids).
        use_multimodal_trace = (
            processed_inputs is not None
            and processed_inputs.get("pixel_values", None) is not None
            and kwargs.get("vision_model", None) is not None
        )
        # Per-configuration key prevents trace collisions across seq lengths, model IDs, batch sizes, and modality.
        trace_key = f"{prefill_seq_len}_{model_id}_{batch_size}_{'multimodal' if use_multimodal_trace else 'text'}"

        if self.trace_id_prefill[trace_key] is None:
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
        cached = getattr(self, "_prefill_replay_cache", None)
        if cached is not None:
            # Reuse host_inputs from capture to skip re-running the vision model and host round-trips.
            host_inputs = cached
            self._prefill_replay_cache = None
        else:
            prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": user_id}
            if global_user_id is not None:
                prefill_kwargs["global_user_id"] = global_user_id
            prefill_kwargs.update(kwargs)
            host_inputs_full = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
            host_inputs = (host_inputs_full[0], host_inputs_full[3], host_inputs_full[4])

        copy_host_to_device(
            host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
        )
        ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)
        return tt_out_trace
