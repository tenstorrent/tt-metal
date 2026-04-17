# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN thinker LM head for Qwen3-Omni-MoE (HF ``nn.Linear(hidden, vocab)`` → logits)."""

from __future__ import annotations

import math
import os

import torch
from torch import nn
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import tree_map


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
    """Thinker logits: all-gather hidden if sharded (like TTNNQwen3OmniAttention), linear, replicated readback with dim0 slice for generate. Chunked matmul when env byte caps hit."""

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

    def _maybe_all_gather_hidden(self, tensor):
        """All-gather last dim when activations are column-sharded (``last * num_devices == in_features``).

        Do **not** skip based on ``TorchTTNNTensor`` "replicated" metadata alone: symbiote can tag
        tensors replicated while the physical ttnn width is still a shard (e.g. 256 of 2048), which
        broke chunked LM-head readback (ragged ``vocab`` dims across chunks).
        """
        t = self._to_ttnn(tensor)
        last = int(t.shape[-1])
        if last == self.in_features:
            return t
        mesh = self.device
        if mesh is None or not hasattr(mesh, "get_num_devices") or mesh.get_num_devices() <= 1:
            return t
        n = int(mesh.get_num_devices())
        if last * n != self.in_features:
            return t
        if self._is_distributed:
            return ttnn.experimental.all_gather_async(
                t,
                dim=-1,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Ring,
            )
        return ttnn.all_gather(
            t,
            dim=-1,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _lm_head_logits_dtensor_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def _readback_logits_bf16(self, tt_out: ttnn.Tensor, expected_token_rows: int) -> torch.Tensor:
        """Device logits → host ``(expected_token_rows, out_features)`` bf16 for chunk concat."""
        n = 1 if self.device is None else int(self.device.get_num_devices())
        wid = int(tt_out.shape[-1])
        if n > 1 and wid * n == self.out_features:
            pt = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1))
        elif n > 1:
            pt = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        else:
            pt = ttnn.to_torch(tt_out)
        pt = pt.to(torch.bfloat16)
        while pt.dim() > 2:
            pt = pt.reshape(-1, pt.shape[-1])
        if pt.shape[0] > expected_token_rows:
            pt = pt[:expected_token_rows]
        if pt.shape[-1] > self.out_features:
            pt = pt[..., : self.out_features]
        return pt.contiguous()

    def _forward_linear_4d(self, x: ttnn.Tensor) -> ttnn.Tensor:
        input_tensor_shape = list(x.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        x4 = ttnn.reshape(x, input_shape)
        out = ttnn.linear(x4, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, input_tensor_shape[:-1] + [self.out_features])

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    def _effective_chunk_rows(self, n_inner: int, chunk_cap: int, hidden_last: int) -> int:
        max_chunk_out = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_CHUNK_OUTPUT_BYTES", 64 * 1024 * 1024)
        max_chunk_in = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_CHUNK_INPUT_BYTES", 32 * 1024 * 1024)
        r = min(max(1, chunk_cap), max(1, n_inner))
        while r > 1:
            if r * self.out_features * 2 <= max_chunk_out and r * hidden_last * 2 <= max_chunk_in:
                return r
            r = max(1, r // 2)
        return 1

    def forward(self, hidden_states):
        x = self._maybe_all_gather_hidden(hidden_states)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(x.shape)
        n_inner = int(math.prod(int(d) for d in input_tensor_shape[:-1]))
        hidden = int(input_tensor_shape[-1])
        if n_inner == 0:
            return ttnn.reshape(x, input_tensor_shape[:-1] + [self.out_features])

        max_out = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_OUTPUT_BYTES", 32 * 1024 * 1024)
        max_in = self._env_int("TT_SYMBIOTE_LM_HEAD_MAX_INPUT_BYTES", 48 * 1024 * 1024)
        chunk_cap = self._env_int("TT_SYMBIOTE_LM_HEAD_CHUNK_TOKENS", 16)
        est_out = n_inner * self.out_features * 2
        est_in = n_inner * hidden * 2
        # ``chunk_cap <= 0`` disables chunking (single device-sized matmul; can OOM on long prefill).
        if chunk_cap <= 0 or (est_out <= max_out and est_in <= max_in):
            return self._forward_linear_4d(x)

        row_step = self._effective_chunk_rows(n_inner, chunk_cap, hidden)
        x2 = ttnn.reshape(x, (n_inner, hidden))
        parts: list[torch.Tensor] = []
        for r0 in range(0, n_inner, row_step):
            r1 = min(r0 + row_step, n_inner)
            sub = ttnn.slice(x2, (r0, 0), (r1, hidden))
            sub4 = ttnn.reshape(sub, (1, 1, r1 - r0, hidden))
            tt_chunk = self._forward_linear_4d(sub4)
            parts.append(self._readback_logits_bf16(tt_chunk, r1 - r0))
        logits_cpu = torch.cat(parts, dim=0).reshape(*input_tensor_shape[:-1], self.out_features)
        return TorchTTNNTensor(logits_cpu)


def replace_thinker_lm_head_with_ttnn(thinker: nn.Module) -> None:
    """Replace ``thinker.lm_head`` with :class:`TTNNQwenOmniThinkerLmHead` when it is ``nn.Linear``.

    If :data:`torch.nn.Linear` was already swapped to :class:`~models.experimental.tt_symbiote.modules.linear.TTNNLinear`
    by a thinker-wide op map, recover weights from ``_fallback_torch_layer`` and install this head instead.
    """
    if "lm_head" not in getattr(thinker, "_modules", {}):
        return
    old = thinker._modules.get("lm_head")
    if old is None or isinstance(old, TTNNQwenOmniThinkerLmHead):
        return
    if isinstance(old, nn.Linear):
        thinker._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
        return
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(old, TTNNLinear):
        tl = getattr(old, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            new_head = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(old, "_unique_name", None) is not None:
                new_head._unique_name = old._unique_name
            thinker._modules["lm_head"] = new_head


def _lm_head_from_linear_or_ttnn_linear(m):
    if isinstance(m, TTNNQwenOmniThinkerLmHead):
        return m
    if isinstance(m, nn.Linear):
        return TTNNQwenOmniThinkerLmHead.from_torch(m)
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(m, TTNNLinear):
        tl = getattr(m, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            nh = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(m, "_unique_name", None) is not None:
                nh._unique_name = m._unique_name
            return nh
    return m


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
        new_heads = [_lm_head_from_linear_or_ttnn_linear(m) for m in old]
        if "lm_head" in cp._modules:
            del cp._modules["lm_head"]
        cp.lm_head = new_heads
        return
    if isinstance(old, nn.Linear):
        cp._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
        return
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(old, TTNNLinear):
        tl = getattr(old, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            nh = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(old, "_unique_name", None) is not None:
                nh._unique_name = old._unique_name
            cp._modules["lm_head"] = nh


def replace_talker_codec_head_with_ttnn(talker: nn.Module) -> None:
    """Replace ``talker.codec_head`` (``nn.Linear``) with :class:`TTNNQwenOmniThinkerLmHead`."""
    if "codec_head" not in getattr(talker, "_modules", {}):
        return
    old = talker._modules.get("codec_head")
    if old is None or isinstance(old, TTNNQwenOmniThinkerLmHead):
        return
    if isinstance(old, nn.Linear):
        talker._modules["codec_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
        return
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear

    if isinstance(old, TTNNLinear):
        tl = getattr(old, "_fallback_torch_layer", None)
        if isinstance(tl, nn.Linear):
            new_head = TTNNQwenOmniThinkerLmHead.from_torch(tl)
            if getattr(old, "_unique_name", None) is not None:
                new_head._unique_name = old._unique_name
            talker._modules["codec_head"] = new_head
