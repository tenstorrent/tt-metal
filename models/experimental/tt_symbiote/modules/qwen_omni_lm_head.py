# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN thinker LM head for Qwen3-Omni-MoE (HF ``nn.Linear(hidden, vocab)`` → logits)."""

from __future__ import annotations

import os

import torch
from torch import nn
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import tree_map

# Prefill can build a full ``[batch, seq, vocab]`` logits tensor (~GB bf16). One ``ttnn.linear`` output buffer
# (and internal layout / transpose temps) can exceed Wormhole DRAM — especially when DRAM is fragmented.
# Chunk along flattened batch×seq and stitch on host (see :meth:`TTNNQwenOmniThinkerLmHead.forward`).
# Large prefills: optional chunking when estimated logits/hidden bytes exceed thresholds (see ``forward``).
_LM_HEAD_MAX_OUTPUT_BYTES = int(os.environ.get("TT_SYMBIOTE_LM_HEAD_MAX_OUTPUT_BYTES", str(32 * 1024 * 1024)))
_LM_HEAD_MAX_INPUT_BYTES = int(os.environ.get("TT_SYMBIOTE_LM_HEAD_MAX_INPUT_BYTES", str(48 * 1024 * 1024)))
_LM_HEAD_CHUNK_TOKENS = max(1, int(os.environ.get("TT_SYMBIOTE_LM_HEAD_CHUNK_TOKENS", "16")))
_LM_HEAD_MAX_CHUNK_OUTPUT_BYTES = int(
    os.environ.get("TT_SYMBIOTE_LM_HEAD_MAX_CHUNK_OUTPUT_BYTES", str(64 * 1024 * 1024))
)
_LM_HEAD_MAX_CHUNK_INPUT_BYTES = int(os.environ.get("TT_SYMBIOTE_LM_HEAD_MAX_CHUNK_INPUT_BYTES", str(32 * 1024 * 1024)))
# Per-chunk caps (fixed defaults; row count is also bounded in :meth:`TTNNQwenOmniThinkerLmHead._effective_lm_head_chunk_rows`).


def _lm_head_logits_dtensor_config(mesh_device):
    """Replicated logits: compose then slice dim 0 so HF sampling sees ``[batch, …]`` not ``[batch*n_dev, …]``."""
    if mesh_device is None or mesh_device.get_num_devices() <= 1:
        return None
    return DistributedTensorConfig(
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        replicate_compose_slice_dim0_to_leading=True,
    )


class TTNNQwenOmniThinkerLmHead(TTNNModule):
    """
    Final logits projection for the thinker.

    Final RMSNorm output may be **width-sharded** on a multi-device mesh; this layer
    all-gathers the hidden width when needed (same CCL pattern as ``TTNNQwen3OmniAttention``),
    then runs ``ttnn.linear`` with replicated weights. When estimated bf16 activations exceed
    ``TT_SYMBIOTE_LM_HEAD_MAX_{INPUT,OUTPUT}_BYTES``, logits are computed in row chunks (bounded by
    ``TT_SYMBIOTE_LM_HEAD_CHUNK_TOKENS`` and ``TT_SYMBIOTE_LM_HEAD_MAX_CHUNK_*``), read to host, and
    concatenated on CPU.

    Full prefill logits do not fit device DRAM (~1.7 GiB+); re-uploading them with ``from_torch`` OOMs,
    so the merged result stays **host ``torch``** (wrapped as :class:`TorchTTNNTensor` by symbiote).
    Sampling / ``argmax`` run on CPU tensors as in typical HF flows.
    """

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        m = cls()
        m._fallback_torch_layer = linear
        m.in_features = int(linear.in_features)
        m.out_features = int(linear.out_features)
        m.weight = linear.weight
        m.bias = linear.bias
        return m

    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None

    def deallocate_weights_impl(self):
        if getattr(self, "tt_weight", None) is not None:
            ttnn.deallocate(self.tt_weight)
            self.tt_weight = None
        if getattr(self, "tt_bias", None) is not None:
            ttnn.deallocate(self.tt_bias)
            self.tt_bias = None
        super().deallocate_weights_impl()

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _to_ttnn(self, tensor):
        return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    def _is_symbiote_replicated(self, tensor) -> bool:
        if isinstance(tensor, TorchTTNNTensor):
            cfg = tensor.ttnn_distributed_tensor_config
            if cfg is not None and cfg.mesh_mapper is not None:
                return "Replicate" in type(cfg.mesh_mapper).__name__
        return False

    def _maybe_all_gather_hidden(self, tensor):
        t = self._to_ttnn(tensor)
        if not self._is_distributed:
            return t
        # Full-width activations (replicated or already gathered): skip CCL.
        if int(t.shape[-1]) == self.in_features:
            return t
        if self._is_symbiote_replicated(tensor):
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Ring,
        )

    def _forward_linear_4d(self, x, input_tensor_shape):
        """``ttnn.linear`` with the same 4D padding/reshape as the original LM head."""
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        x_ = ttnn.reshape(x, input_shape)
        tt_output = ttnn.linear(x_, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(tt_output, list(input_tensor_shape[:-1]) + [int(self.out_features)])

    def _linear_chunk_to_torch(self, tt_logits_2d: ttnn.Tensor) -> torch.Tensor:
        """(N, V) logits on mesh → one bf16 CPU tensor (replica trim, vocab trim)."""
        n = int(tt_logits_2d.shape[0])
        v = int(self.out_features)
        if self.device is None or self.device.get_num_devices() <= 1:
            pt = ttnn.to_torch(tt_logits_2d)
        else:
            pt = ttnn.to_torch(tt_logits_2d, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
        if pt.shape[-1] > v:
            pt = pt[..., :v]
        return pt.to(torch.bfloat16).contiguous()

    def _effective_lm_head_chunk_rows(self, h_dim: int, n_flat: int) -> int:
        """Bound rows per chunk so input/output activations stay under per-chunk DRAM caps."""
        chunk = min(int(_LM_HEAD_CHUNK_TOKENS), max(1, n_flat))
        max_out = max(1, _LM_HEAD_MAX_CHUNK_OUTPUT_BYTES // max(1, int(self.out_features) * 2))
        max_in = max(1, _LM_HEAD_MAX_CHUNK_INPUT_BYTES // max(1, int(h_dim) * 2))
        return max(1, min(chunk, max_out, max_in))

    def _forward_chunked_host_logits(self, x, input_tensor_shape, n_flat: int) -> TorchTTNNTensor:
        """Chunked ``ttnn.linear`` + host stitch to avoid a single multi-GB device logits allocation."""
        assert n_flat >= 1
        h_dim = int(input_tensor_shape[-1])
        x2d = ttnn.reshape(x, (n_flat, h_dim))
        chunk = self._effective_lm_head_chunk_rows(h_dim, n_flat)
        parts = []
        for s in range(0, n_flat, chunk):
            e = min(s + chunk, n_flat)
            x_chunk = ttnn.slice(x2d, (s, 0), (e, h_dim))
            tt_chunk_out = self._forward_linear_4d(x_chunk, [e - s, h_dim])
            tt_2d = ttnn.reshape(tt_chunk_out, (e - s, int(self.out_features)))
            parts.append(self._linear_chunk_to_torch(tt_2d))
            try:
                ttnn.deallocate(tt_2d)
            except Exception:
                pass
            try:
                ttnn.deallocate(x_chunk)
            except Exception:
                pass
        try:
            ttnn.deallocate(x)
        except Exception:
            pass
        merged = torch.cat(parts, dim=0)
        rest = list(input_tensor_shape[:-1])
        merged = merged.view(*rest, int(self.out_features))
        return TorchTTNNTensor(merged)

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _lm_head_logits_dtensor_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def forward(self, hidden_states):
        x = self._maybe_all_gather_hidden(hidden_states)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(x.shape)
        n_flat = 1
        for d in input_tensor_shape[:-1]:
            n_flat *= int(d)
        h_dim = int(input_tensor_shape[-1])
        est_out_bytes = n_flat * int(self.out_features) * 2
        est_in_bytes = n_flat * h_dim * 2
        if (est_out_bytes > _LM_HEAD_MAX_OUTPUT_BYTES or est_in_bytes > _LM_HEAD_MAX_INPUT_BYTES) and n_flat > 1:
            return self._forward_chunked_host_logits(x, input_tensor_shape, n_flat)
        return self._forward_linear_4d(x, input_tensor_shape)


def replace_thinker_lm_head_with_ttnn(thinker: nn.Module) -> None:
    """Replace ``thinker.lm_head`` with :class:`TTNNQwenOmniThinkerLmHead` when it is ``nn.Linear``."""
    if "lm_head" not in getattr(thinker, "_modules", {}):
        return
    old = thinker._modules.get("lm_head")
    if old is None or not isinstance(old, nn.Linear):
        return
    # ``TTNNModule`` is not an ``nn.Module``; assign via ``_modules`` like ``register_module_replacement_dict``.
    thinker._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)


def replace_code_predictor_lm_head_with_ttnn(talker: nn.Module) -> None:
    """Replace ``talker.code_predictor.lm_head`` (``nn.ModuleList[nn.Linear]``) with TTNN heads.

    Uses a plain ``list`` for multiple heads: ``nn.ModuleList`` only accepts ``nn.Module`` children,
    while :class:`TTNNQwenOmniThinkerLmHead` is a ``TTNNModule`` (not ``nn.Module``).
    HF only does ``self.lm_head[i](hidden_states)``, so a list is sufficient.
    """
    cp = getattr(talker, "code_predictor", None)
    if cp is None:
        return
    old = getattr(cp, "lm_head", None)
    if old is None:
        return
    if isinstance(old, nn.ModuleList):
        new_heads = [TTNNQwenOmniThinkerLmHead.from_torch(m) if isinstance(m, nn.Linear) else m for m in old]
        if "lm_head" in cp._modules:
            del cp._modules["lm_head"]
        cp.lm_head = new_heads
        return
    if isinstance(old, nn.Linear):
        cp._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)


def replace_talker_codec_head_with_ttnn(talker: nn.Module) -> None:
    """Replace ``talker.codec_head`` (``nn.Linear``) with :class:`TTNNQwenOmniThinkerLmHead`."""
    if "codec_head" not in getattr(talker, "_modules", {}):
        return
    old = talker._modules.get("codec_head")
    if old is None or not isinstance(old, nn.Linear):
        return
    talker._modules["codec_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
