# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN thinker LM head for Qwen3-Omni-MoE (HF ``nn.Linear(hidden, vocab)`` → logits)."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import tree_map
from models.experimental.tt_symbiote.modules.activation import _ttnn_mesh_to_torch_one_replica


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
    then runs ``ttnn.linear`` with replicated weights. Readback uses a replicated mesh config that
    slices dim 0 after compose so ``torch.argmax`` / generation see ``[batch, seq, vocab]``.

    Long prefills allocate logits ``[batch, seq, vocab]`` in BF16; a single ``ttnn.linear`` can
    exceed device DRAM (e.g. multimodal prompts). When ``seq`` exceeds
    ``TT_SYMBIOTE_LM_HEAD_CHUNK_TOKENS`` (default ``128``), the forward runs **chunked** along
    sequence. Chunk **hidden** states are read back once per chunk; logits use **PyTorch**
    ``F.linear`` on host with the same ``weight``/``bias`` as HuggingFace (avoids per-chunk
    ``ttnn.linear`` + readback drift vs a single full matmul, which was skewing sampling and
    corrupting late talker / code2wav audio). Device DRAM is not used for the full logits tensor.
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

    def set_output_tensors_config_impl(self, output_tensors):
        cfg = _lm_head_logits_dtensor_config(self.device)
        if cfg is None:
            return super().set_output_tensors_config_impl(output_tensors)

        def apply(e):
            if isinstance(e, TorchTTNNTensor):
                e.set_distributed_tensor_config(cfg)
            return e

        return tree_map(apply, output_tensors)

    def _hidden_chunk_to_torch(self, x_c):
        """Hidden slice (TTNN) → one logical host tensor (same mesh rules as activations)."""
        return _ttnn_mesh_to_torch_one_replica(x_c, getattr(self, "device", None))

    def _linear_chunked_along_seq(self, x, *, b: int, seq_len: int, h: int, input_tensor_shape: list):
        """Chunk along sequence; compute logits with host ``F.linear`` (HF weights) to match reference math."""
        chunk = int(os.environ.get("TT_SYMBIOTE_LM_HEAD_CHUNK_TOKENS", "128"))
        if chunk <= 0:
            chunk = seq_len
        if seq_len <= chunk:
            tt_out = ttnn.linear(x, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.reshape(tt_out, input_tensor_shape[:-1] + [self.out_features])

        W, wb = self.weight, self.bias
        chunks_pt = []
        for s0 in range(0, seq_len, chunk):
            s1 = min(s0 + chunk, seq_len)
            x_c = ttnn.slice(x, (0, 0, s0, 0), (b, 1, s1, h))
            h_t = self._hidden_chunk_to_torch(x_c)
            # Do not deallocate slice output: it may share storage with ``x`` and would corrupt later chunks.
            h_eff = h_t[..., : self.in_features] if int(h_t.shape[-1]) > self.in_features else h_t
            lead = h_eff.shape[:-1]
            logits_flat = F.linear(
                h_eff.reshape(-1, self.in_features).to(device=W.device, dtype=W.dtype),
                W,
                wb,
            )
            pt = logits_flat.reshape(*lead, self.out_features)
            chunks_pt.append(pt)

        if not chunks_pt:
            raise RuntimeError("TTNNQwenOmniThinkerLmHead: chunked path produced no logits chunks")

        # Sequence is the dimension immediately before vocab (same for 3D or 4D activations).
        cat_dim = chunks_pt[0].dim() - 2
        full_pt = torch.cat(chunks_pt, dim=cat_dim)
        return full_pt.reshape(tuple(input_tensor_shape[:-1]) + (self.out_features,))

    def forward(self, hidden_states):
        x = self._maybe_all_gather_hidden(hidden_states)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(x.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        x = ttnn.reshape(x, input_shape)
        b = int(x.shape[0])
        seq_len = int(x.shape[2])
        h = int(x.shape[3])
        return self._linear_chunked_along_seq(x, b=b, seq_len=seq_len, h=h, input_tensor_shape=input_tensor_shape)


def replace_thinker_lm_head_with_ttnn(thinker: nn.Module) -> None:
    """Replace ``thinker.lm_head`` with :class:`TTNNQwenOmniThinkerLmHead` when it is ``nn.Linear``."""
    if "lm_head" not in getattr(thinker, "_modules", {}):
        return
    old = thinker._modules.get("lm_head")
    if old is None or not isinstance(old, nn.Linear):
        return
    # ``TTNNModule`` is not an ``nn.Module``; assign via ``_modules`` like ``register_module_replacement_dict``.
    thinker._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
