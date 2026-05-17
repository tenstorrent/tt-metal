# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal TT ``Ministral3Model`` (text stack): embeddings -> decoder layers -> final RMSNorm.

Optionally owns :class:`TtMinistral3RotaryEmbedding` (device cos/sin from ``Ministral3Config`` via
``ministral_text_config``). If configured, ``forward_prefill`` / ``forward_prefill_from_embeddings`` may
omit ``rot_mats`` (``None``) and slice tables in-model; otherwise pass ``rot_mats`` as before.

Composes existing TT submodules from this experimental folder; no Torch fallback in forward.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional


# Load ``fp8_dequantize_compat`` by path so we do not import ``devstral_utils`` package ``__init__``
# (which pulls ``multimodal_demo_helpers`` and would circular-import this module).
def _ensure_fp8_scalar_compat() -> None:
    _mod_name = "_devstarl2_fp8_dequantize_compat_exec"
    if _mod_name in sys.modules:
        sys.modules[_mod_name].apply_fp8_dequantize_compat()
        return
    _path = Path(__file__).resolve().parent.parent / "devstral_utils" / "fp8_dequantize_compat.py"
    _spec = importlib.util.spec_from_file_location(_mod_name, _path)
    if _spec is None or _spec.loader is None:
        return
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules[_mod_name] = _mod
    _mod.apply_fp8_dequantize_compat()


_ensure_fp8_scalar_compat()

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.experimental.devstarl2_small.tt.tt_ministral3_decoder_layer import TtMinistral3DecoderLayer
from models.experimental.devstarl2_small.tt.tt_ministral_rotary_emb import TtMinistral3RotaryEmbedding
from models.tt_dit.utils.tracing import traced_function
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.embedding import Embedding

if TYPE_CHECKING:
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config


class TtMinistral3Model(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        model_args,
        meta_state_dict,
        weight_cache_path,
        dtype,
        transformation_mats,
        configuration,
        llama_4_scaling_beta=None,
        original_max_position_embeddings=None,
        ministral_text_config: Optional["Ministral3Config"] = None,
        tt_rotary_embedding: Optional[TtMinistral3RotaryEmbedding] = None,
    ):
        super().__init__()
        self.args = model_args
        self.mesh_device = mesh_device
        self.n_layers = int(model_args.n_layers)

        if tt_rotary_embedding is not None and ministral_text_config is not None:
            raise ValueError("Pass at most one of tt_rotary_embedding and ministral_text_config.")

        if tt_rotary_embedding is not None:
            self.tt_rotary_embedding = tt_rotary_embedding
        elif ministral_text_config is not None:
            self.tt_rotary_embedding = TtMinistral3RotaryEmbedding(
                device=mesh_device,
                batch_size=model_args.max_batch_size,
                head_dim=model_args.head_dim,
                max_seq_len=model_args.max_seq_len,
                config=ministral_text_config,
                datatype=ttnn.bfloat16,
            )
        else:
            self.tt_rotary_embedding = None

        self.embed_tokens = Embedding(
            mesh_device=mesh_device,
            args=model_args,
            weight_cache_path=weight_cache_path,
            state_dict=meta_state_dict,
            dtype=ttnn.bfloat16,
        )

        self.layers = [
            TtMinistral3DecoderLayer(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                model_args=model_args,
                meta_state_dict=meta_state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=transformation_mats,
                configuration=configuration,
                llama_4_scaling_beta=llama_4_scaling_beta,
                original_max_position_embeddings=original_max_position_embeddings,
            )
            for i in range(self.n_layers)
        ]

        self.norm = RMSNorm(
            device=mesh_device,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            state_dict=meta_state_dict,
            weight_key="norm",
            state_dict_prefix=model_args.get_state_dict_prefix("", None),
            weight_cache_path=None if model_args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            is_distributed=model_args.is_distributed_norm,
            add_unit_offset=model_args.rms_norm_add_unit_offset,
            ccl_topology=model_args.ccl_topology(),
            tt_ccl=tt_ccl,
        )

        # Persistent device tensors owned by the trace; allocated on first forward_decode(enable_trace=True).
        self._decode_trace_token_ids: Optional[ttnn.Tensor] = None
        self._decode_trace_pos_uint32: Optional[ttnn.Tensor] = None
        self._decode_trace_pos_int32: Optional[ttnn.Tensor] = None

        # Optional LM head included inside the decode trace for maximum throughput.
        # Set via set_decode_lm_head() before the first traced decode call.
        self._decode_lm_head = None
        self._decode_lm_head_args = None

    def forward_prefill_from_embeddings(
        self,
        hidden_states_11SH: ttnn.Tensor,
        rot_mats,
        position_ids,
        rope_start_pos: int = 0,
    ) -> ttnn.Tensor:
        if rot_mats is None:
            if self.tt_rotary_embedding is None:
                raise ValueError(
                    "rot_mats is required when tt_rotary_embedding is not set (pass ministral_text_config "
                    "or tt_rotary_embedding to TtMinistral3Model.__init__, or supply rot_mats explicitly)."
                )
            seq_len = int(hidden_states_11SH.shape[2])
            rot_mats = self.tt_rotary_embedding.slice_rot_mats_prefill(rope_start_pos, seq_len)
        h = hidden_states_11SH
        for layer in self.layers:
            h = layer.forward_prefill(h, rot_mats, position_ids=position_ids)
        return self.norm(h, Mode.PREFILL)

    def forward_prefill(
        self,
        input_ids_tt: ttnn.Tensor,
        position_ids,
        rot_mats=None,
        rope_start_pos: int = 0,
    ) -> ttnn.Tensor:
        if rot_mats is None:
            if self.tt_rotary_embedding is None:
                raise ValueError(
                    "rot_mats is required when tt_rotary_embedding is not set (pass ministral_text_config "
                    "or tt_rotary_embedding to TtMinistral3Model.__init__, or supply rot_mats explicitly)."
                )
            seq_len = int(input_ids_tt.shape[-1])
            rot_mats = self.tt_rotary_embedding.slice_rot_mats_prefill(rope_start_pos, seq_len)
        h = self.embed_tokens(input_ids_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.unsqueeze_to_4D(h)
        return self.forward_prefill_from_embeddings(h, rot_mats, position_ids, rope_start_pos)

    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False)
    def _forward_decode_inner(
        self,
        token_ids_tt: ttnn.Tensor,
        pos_uint32: ttnn.Tensor,
        pos_int32: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Traceable body of a single decode step.

        All inputs are pre-allocated device tensors; no host-device copies or dynamic allocations
        occur here so the region is safe for ``ttnn.begin_trace_capture`` / ``ttnn.execute_trace``.
        Decorated with ``@traced_function``: call with ``traced=True`` to capture on the first
        invocation and replay on subsequent ones; ``traced=False`` (default) runs the function
        directly with no overhead.
        """
        # ttnn.embedding on pre-allocated cos/sin tables: safe inside a trace.
        rot_mats = self.tt_rotary_embedding.get_rot_mats(pos_uint32)
        h = self.embed_tokens(token_ids_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.unsqueeze_to_4D(h)
        residual_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        h = ttnn.to_memory_config(h, residual_mem_cfg)
        for layer in self.layers:
            h = layer.forward_decode(h, pos_int32, rot_mats)
        h_dram = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
        return self.norm(h_dram, Mode.DECODE)

    def forward_decode(
        self,
        token_ids_tt: ttnn.Tensor,
        decode_pos: int,
        enable_trace: bool = False,
    ) -> ttnn.Tensor:
        """Single-token decode using the KV cache filled by the preceding ``forward_prefill`` call.

        ``token_ids_tt``: device uint32 tensor ``[1, 1]`` holding the last generated token id.
        ``decode_pos``: absolute 0-based position of this token in the sequence (= prompt length + step).
        ``enable_trace``: when True the first call captures a device-command trace; subsequent calls
            replay it (zero Python host-dispatch overhead per token).  The model owns the persistent
            input tensors for the trace; callers may freely deallocate the ``token_ids_tt`` they pass.

        Returns hidden states after the final RMSNorm (DRAM, same shape as decode residual).
        """
        if self.tt_rotary_embedding is None:
            raise ValueError("forward_decode requires tt_rotary_embedding (pass ministral_text_config to __init__).")

        _torch = __import__("torch")
        _pos_torch = _torch.tensor([[decode_pos]], dtype=_torch.int32)
        _dev_kw = dict(
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        if not enable_trace:
            # Original untraced path: create temporaries, run, clean up.
            pos_uint32 = ttnn.from_torch(_pos_torch, dtype=ttnn.uint32, **_dev_kw)
            pos_int32 = ttnn.from_torch(_pos_torch, dtype=ttnn.int32, **_dev_kw)
            result = self._forward_decode_inner(token_ids_tt, pos_uint32, pos_int32, traced=False)
            ttnn.deallocate(pos_uint32)
            ttnn.deallocate(pos_int32)
            return result

        # --- Traced path ---
        # Persistent tensors are allocated once and updated in-place on every subsequent call so
        # the Tracer always sees the same buffer addresses (no copy-into required by _update_input).
        if self._decode_trace_token_ids is None:
            # First call: allocate the persistent trace inputs. ttnn.clone creates a new device
            # buffer that the Tracer will store as self._args; the caller's tensor is not stored.
            self._decode_trace_token_ids = ttnn.clone(token_ids_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._decode_trace_pos_uint32 = ttnn.from_torch(_pos_torch, dtype=ttnn.uint32, **_dev_kw)
            self._decode_trace_pos_int32 = ttnn.from_torch(_pos_torch, dtype=ttnn.int32, **_dev_kw)
        else:
            # Subsequent calls: update the persistent buffers with new values before replay.
            _host_kw = dict(layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(_pos_torch, dtype=ttnn.uint32, **_host_kw), self._decode_trace_pos_uint32
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(_pos_torch, dtype=ttnn.int32, **_host_kw), self._decode_trace_pos_int32
            )
            if token_ids_tt.buffer_address() != self._decode_trace_token_ids.buffer_address():
                ttnn.copy(token_ids_tt, self._decode_trace_token_ids)

        return self._forward_decode_inner(
            self._decode_trace_token_ids,
            self._decode_trace_pos_uint32,
            self._decode_trace_pos_int32,
            traced=True,
        )


__all__ = ["TtMinistral3Model"]
