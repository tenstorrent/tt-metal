# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN thinker LM head for Qwen3-Omni-MoE (HF ``nn.Linear(hidden, vocab)`` → logits)."""

from __future__ import annotations

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
    """
    Final logits projection for the thinker.

    Final RMSNorm output may be **width-sharded** on a multi-device mesh; this layer
    all-gathers the hidden width when needed (same CCL pattern as ``TTNNQwen3OmniAttention``),
    then runs ``ttnn.linear`` with replicated weights. Readback uses a replicated mesh config that
    slices dim 0 after compose so ``torch.argmax`` / generation see ``[batch, seq, vocab]``.
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

    def forward(self, hidden_states):
        x = self._maybe_all_gather_hidden(hidden_states)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(x.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        x = ttnn.reshape(x, input_shape)
        tt_output = ttnn.linear(x, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


def replace_thinker_lm_head_with_ttnn(thinker: nn.Module) -> None:
    """Replace ``thinker.lm_head`` with :class:`TTNNQwenOmniThinkerLmHead` when it is ``nn.Linear``."""
    if "lm_head" not in getattr(thinker, "_modules", {}):
        return
    old = thinker._modules.get("lm_head")
    if old is None or not isinstance(old, nn.Linear):
        return
    # ``TTNNModule`` is not an ``nn.Module``; assign via ``_modules`` like ``register_module_replacement_dict``.
    thinker._modules["lm_head"] = TTNNQwenOmniThinkerLmHead.from_torch(old)
