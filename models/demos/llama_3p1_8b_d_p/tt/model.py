# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Eager Llama-3.1-8B prefill stack, with a device-only numerical forward."""

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.attention import FullCausalAttention, _validate_device_tensor
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.decoder import DecoderLayer
from models.demos.llama_3p1_8b_d_p.tt.input import validate_chunk_range
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import DEFAULT_MAX_SEQ_LEN, DEFAULT_NUM_USERS
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PrefillGeometry, validate_mesh
from models.demos.llama_3p1_8b_d_p.tt.rms_norm import RMSNorm
from models.demos.llama_3p1_8b_d_p.tt.rope import build_indexed_rope, build_transformation_mat
from models.demos.llama_3p1_8b_d_p.tt.weights import CheckpointWeights


def _validate_weight(weight, shape, name):
    if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU torch.Tensor")
    if tuple(weight.shape) != shape or not weight.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor with shape {shape}")


def _validate_tokens(tokens, mesh_device):
    if not isinstance(tokens, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(tokens):
        raise ValueError("prefill tokens must be a device ttnn.Tensor")
    if tokens.device() != mesh_device or len(ttnn.get_device_tensors(tokens)) != layout.num_devices:
        raise ValueError("prefill tokens must cover the constructor Galaxy")
    if tuple(tokens.shape) not in ((1, 1, layout.local_sequence), (1, 1, 1, layout.local_sequence)):
        raise ValueError("prefill tokens must have local shape [1,1,256] or [1,1,1,256]")
    if tokens.dtype != ttnn.uint32 or tokens.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("prefill tokens must use UINT32 ROW_MAJOR_LAYOUT")
    if tokens.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("prefill tokens must use interleaved DRAM")


class TokenEmbedding:
    """Replicate BF16 embedding weights; consume caller-owned encounter-ordered SP IDs."""

    def __init__(self, mesh_device, weight):
        _validate_weight(weight, (Model.VOCAB_SIZE, Model.EMB_SIZE), "embedding weight")
        self.mesh_device = mesh_device
        # 1,050,673,152 bytes per chip. This deliberately simple first implementation fits in
        # Galaxy DRAM; vocabulary sharding would add a collective to each input embedding.
        self.weight = ttnn.from_torch(
            weight.bfloat16().reshape(1, 1, Model.VOCAB_SIZE, Model.EMB_SIZE),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def __call__(self, token_ids):
        _validate_tokens(token_ids, self.mesh_device)
        # ID bounds are established by the host packer or the external H2D producer. Reading IDs
        # back here would turn the numerical forward into a host-dependent path.
        embedded = ttnn.embedding(
            token_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # The public embedding operation returns [batch,sequence,hidden], even for 4-D IDs.
        # Unsqueeze is a view; do not forcibly deallocate the shared embedding storage here.
        return ttnn.unsqueeze_to_4D(embedded)

    def close(self):
        if self.weight is not None:
            self.weight.deallocate(True)
            self.weight = None


class FinalNormHead:
    """Final RMSNorm and BF16 vocabulary shards; outputs stay on device for diagnostics.

    Each TP column owns exactly 16032 vocabulary entries (501 tiles). There are no invented
    vocabulary entries or power-of-two sampling pads. This class performs no sampling.
    """

    VOCAB_PER_TP = Model.VOCAB_SIZE // layout.tp

    def __init__(self, mesh_device, mesh_config, norm_weight, head_weight):
        validate_mesh(mesh_device, mesh_config, "FinalNormHead")
        _validate_weight(norm_weight, (Model.EMB_SIZE,), "final norm weight")
        _validate_weight(head_weight, (Model.VOCAB_SIZE, Model.EMB_SIZE), "LM head weight")
        self.mesh_device = mesh_device
        self.norm = RMSNorm(mesh_device, norm_weight)
        self.weight = ttnn.from_torch(
            head_weight.bfloat16().transpose(0, 1).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, hidden):
        _validate_device_tensor(
            hidden,
            self.mesh_device,
            name="final head input",
            shape=(1, 1, layout.local_sequence, Model.EMB_SIZE),
            dtype=ttnn.bfloat16,
        )
        normalized = self.norm(hidden)
        try:
            return ttnn.matmul(
                normalized,
                self.weight,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )
        finally:
            normalized.deallocate(True)

    def close(self):
        if self.weight is not None:
            self.weight.deallocate(True)
            self.norm.weight.deallocate(True)
            self.weight = None


class PrefillModel:
    """Own model weights/resources; borrow per-call token and packed KV tensors.

    The default builds all 32 layers. ``num_layers`` permits a prefix of real checkpoint layers
    for the reduced correctness probe. It never changes the 32-layer packed-cache geometry.

    Inputs are SP-row ordered, TP-replicated UINT32 IDs, either local [1,1,256] or
    the host upload helper's [1,1,1,256]. Outputs are caller-owned hidden states
    [1,1,256,4096], or TP-vocabulary-sharded logits [1,1,256,16032]. Only rows below actual_end
    have meaning. Continuations require a contiguous prefix written through this cache object.
    Calls must remain sequential because layers share attention buffers and KV storage.
    """

    def __init__(
        self,
        mesh_device,
        checkpoint_path,
        *,
        num_layers=Model.NUM_LAYERS,
        cache_dtype=ttnn.bfloat8_b,
        enable_lm_head=True,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        num_users=DEFAULT_NUM_USERS,
    ):
        if type(num_layers) is not int or not 1 <= num_layers <= Model.NUM_LAYERS:
            raise ValueError("num_layers must be an integer in [1,32]")
        if type(enable_lm_head) is not bool:
            raise TypeError("enable_lm_head must be a bool")
        self.mesh_device = mesh_device
        self.geometry = PrefillGeometry(max_seq_len, num_users)
        self.max_seq_len = self.geometry.max_seq_len
        self.num_users = self.geometry.num_users
        self.mesh_config = MeshConfig(layout.mesh_shape, layout.tp, tp_axis=layout.tp_axis)
        validate_mesh(mesh_device, self.mesh_config, "PrefillModel")
        weights = CheckpointWeights(checkpoint_path, max_seq_len=self.max_seq_len)
        self.num_layers = num_layers
        self.embedding = None
        self.head = None
        self.layers = []
        self.attention = None
        self.rope_tables = ()
        self.transformation_mat = None
        self.closed = False
        try:
            self.attention = FullCausalAttention(
                mesh_device,
                self.mesh_config,
                cache_dtype=cache_dtype,
                max_seq_len=self.max_seq_len,
                num_users=self.num_users,
            )
            self.rope_tables = tuple(
                build_indexed_rope(mesh_device, max_seq_len=self.max_seq_len, chunk_size=layout.chunk_size)
            )
            self.transformation_mat = build_transformation_mat(mesh_device)
            self.embedding = TokenEmbedding(mesh_device, weights.embedding())
            for layer_idx in range(num_layers):
                # Only this layer's raw CPU weights remain live during its constructor.
                layer_weights = weights.layer(layer_idx)
                self.layers.append(
                    DecoderLayer(
                        mesh_device,
                        self.mesh_config,
                        layer_weights,
                        layer_idx=layer_idx,
                        attention=self.attention,
                        rope_tables=self.rope_tables,
                        transformation_mat=self.transformation_mat,
                    )
                )
                del layer_weights
            if enable_lm_head:
                self.head = FinalNormHead(mesh_device, self.mesh_config, weights.final_norm(), weights.lm_head())
        except Exception:
            self.close()
            raise

    def prefill_chunk(
        self,
        token_ids,
        kv_cache,
        *,
        slot_idx,
        actual_start,
        actual_end,
        skip_lm_head=True,
        layer_observer=None,
    ):
        """Enqueue one prefill chunk and return an owned device output.

        ``layer_observer(index, hidden)`` is a diagnostic hook. It borrows the current output until
        it returns and MUST NOT free or change it. Invocation means enqueue finished, not device
        KV completion. Synchronize the device before consuming host-visible completion.
        """
        if self.closed:
            raise RuntimeError("PrefillModel is closed")
        validate_chunk_range(actual_start, actual_end, max_seq_len=self.max_seq_len)
        _validate_tokens(token_ids, self.mesh_device)
        if type(skip_lm_head) is not bool:
            raise TypeError("skip_lm_head must be a bool")
        if not skip_lm_head and self.head is None:
            raise ValueError("the diagnostic LM head was not constructed")
        if layer_observer is not None and not callable(layer_observer):
            raise TypeError("layer_observer must be callable or None")
        # All model-independent request checks happen before the first layer can write cache.
        self.attention.validate_request(
            kv_cache, slot_idx=slot_idx, layer_idx=0, actual_start=actual_start, actual_end=actual_end
        )
        populated_end = kv_cache.populated_end(slot_idx, self.num_layers)
        if actual_start > populated_end:
            raise ValueError(
                f"prefill continuation starts at {actual_start}, beyond populated cache prefix {populated_end}; "
                "prefill the missing tokens first"
            )
        topology = ttnn.get_usable_topology(kv_cache.k, topology=ttnn.Topology.Ring, cluster_axis=layout.tp_axis)
        if topology != ttnn.Topology.Ring:
            raise RuntimeError(f"prefill requires a live TP ring; TTNN selected {topology}")
        # Invalidate downstream layers too: a restart that fails after an early layer must
        # not leave the previous prompt's suffix advertised as a valid continuation.
        kv_cache.truncate_prefix(slot_idx, actual_start)
        hidden = self.embedding(token_ids)
        try:
            for layer in self.layers:
                output = layer(hidden, kv_cache, slot_idx=slot_idx, actual_start=actual_start, actual_end=actual_end)
                hidden.deallocate(True)
                hidden = output
                if layer_observer is not None:
                    layer_observer(layer.layer_idx, hidden)
            if skip_lm_head:
                result, hidden = hidden, None
                return result
            return self.head(hidden)
        finally:
            if hidden is not None:
                hidden.deallocate(True)

    def close(self):
        """Release model-owned allocations only, after callers finish using returned outputs."""
        if self.closed:
            return
        for layer in self.layers:
            for tensor in (
                layer.input_norm.weight,
                layer.post_attention_norm.weight,
                layer.qkv.qkv_weight,
                layer.output_projection.o_weight,
                layer.mlp.gate_weight,
                layer.mlp.up_weight,
                layer.mlp.down_weight,
            ):
                tensor.deallocate(True)
        self.layers.clear()
        if self.embedding is not None:
            self.embedding.close()
        if self.head is not None:
            self.head.close()
        if self.attention is not None:
            self.attention._release_chunk_mask()
            for tensor in (
                self.attention.gathered_k,
                self.attention.gathered_v,
                self.attention.query_position_table,
                self.attention.key_positions,
            ):
                tensor.deallocate(True)
        for tensor in self.rope_tables:
            tensor.deallocate(True)
        if self.transformation_mat is not None:
            self.transformation_mat.deallocate(True)
        self.closed = True
