# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from torch import nn
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import TTNNQwenOmniIColShardedWAllReduced
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIReplicatedWColSharded
import ttnn
from models.experimental.tt_symbiote.core.utils import tree_map


def _normalize_qwen_omni_vision_act(torch_mlp) -> str:
    """Map HF ACT2FN to gelu vs silu (use class name for nn.Module; __name__ alone was wrong for GELUTanh)."""
    act_fn = getattr(torch_mlp, "act_fn", None)
    if act_fn is None:
        return "silu"
    if isinstance(act_fn, nn.Module):
        cn = act_fn.__class__.__name__.lower()
        if "gelu" in cn:
            return "gelu"
        if "silu" in cn or "swish" in cn:
            return "silu"
    name = (getattr(act_fn, "__name__", None) or type(act_fn).__name__ or "").lower()
    if "gelu" in name:
        return "gelu"
    if "silu" in name or "swish" in name:
        return "silu"
    return "silu"


class TTNNQwen3OmniVisionMLP(TTNNModule):
    """TTNN implementation of Qwen3OmniMoeVisionMLP (fc1 -> act -> fc2)."""

    def __init__(self):
        super().__init__()
        self.hidden_size = None
        self.intermediate_size = None

        self.linear_fc1 = None
        self.linear_fc2 = None

        # Normalized: "gelu" | "silu" (see _normalize_qwen_omni_vision_act).
        self.act_fn = None

    @classmethod
    def from_torch(cls, torch_mlp):
        module = cls()
        module._fallback_torch_layer = torch_mlp

        module.hidden_size = torch_mlp.hidden_size
        module.intermediate_size = torch_mlp.intermediate_size

        # TP MLP: fc1 col-shard intermediate; fc2 all-reduce to replicated output.
        module.linear_fc1 = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.linear_fc1)
        module.linear_fc2 = TTNNQwenOmniIColShardedWAllReduced.from_torch(torch_mlp.linear_fc2)

        module.act_fn = _normalize_qwen_omni_vision_act(torch_mlp)

        return module

    def preprocess_weights_impl(self):
        self.linear_fc1.preprocess_weights()
        self.linear_fc2.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.linear_fc1.move_weights_to_device()
        self.linear_fc2.move_weights_to_device()

    def deallocate_weights_impl(self):
        self.linear_fc1.deallocate_weights()
        self.linear_fc2.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        """After FC2: materialize one [N,hidden] replica on elem (avoid dim=-1 concat vs residual); see post_process_ttnn_module_output."""
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.hidden_size)
            # Replicated per device: concat on batch dim, then take first replica (MoE-style).
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        # Most TTNN paths already provide a ttnn.Tensor; keep this conversion as a safety net.
        if not isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.from_torch(
                hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        # Full hidden width: AG if sharded; slice if concat width > hidden (mesh artifacts).
        in_width = int(hidden_states.shape[-1])
        if in_width > int(self.hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)
        elif in_width < int(self.hidden_size):
            hidden_states = ttnn.all_gather(
                hidden_states,
                dim=-1,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.linear_fc1(hidden_states)

        # HF vision ACT2FN (often GELUTanh).
        if self.act_fn == "gelu":
            hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.silu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.linear_fc2(hidden_states)

        # Slice FC2 output if width > hidden (concat semantics).
        out_width = int(hidden_states.shape[-1])
        if out_width > int(self.hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)

        return hidden_states


class TTNNQwen3OmniTalkerResizeMLP(TTNNModule):
    """TalkerResizeMLP: same TP as TTNNQwen3OmniVisionMLP (fc1 col-shard, fc2 all-reduce); thinker_hidden → text hidden."""

    def __init__(self):
        super().__init__()
        self.input_hidden_size = None
        self.intermediate_size = None
        self.output_hidden_size = None
        self.linear_fc1 = None
        self.linear_fc2 = None
        self.act_fn = None

    @classmethod
    def from_torch(cls, torch_mlp):
        module = cls()
        module._fallback_torch_layer = torch_mlp
        module.input_hidden_size = int(torch_mlp.linear_fc1.in_features)
        module.intermediate_size = int(torch_mlp.linear_fc1.out_features)
        module.output_hidden_size = int(torch_mlp.linear_fc2.out_features)
        module.linear_fc1 = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.linear_fc1)
        module.linear_fc2 = TTNNQwenOmniIColShardedWAllReduced.from_torch(torch_mlp.linear_fc2)
        module.act_fn = _normalize_qwen_omni_vision_act(torch_mlp)
        return module

    def preprocess_weights_impl(self):
        self.linear_fc1.preprocess_weights()
        self.linear_fc2.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.linear_fc1.move_weights_to_device()
        self.linear_fc2.move_weights_to_device()

    def deallocate_weights_impl(self):
        self.linear_fc1.deallocate_weights()
        self.linear_fc2.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        if self.device_state is None or self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.output_hidden_size)
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        if not isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.from_torch(
                hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        in_w = int(hidden_states.shape[-1])
        if in_w > int(self.input_hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.input_hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)
        elif in_w < int(self.input_hidden_size):
            hidden_states = ttnn.all_gather(
                hidden_states,
                dim=-1,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.linear_fc1(hidden_states)
        if self.act_fn == "gelu":
            hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.silu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self.linear_fc2(hidden_states)

        out_w = int(hidden_states.shape[-1])
        if out_w > int(self.output_hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.output_hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)

        return hidden_states
