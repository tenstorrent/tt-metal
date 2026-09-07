# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 TTNN prefill model. Adapted from ``gpt_oss_d_p/tt/model.py``.

    parallel embedding -> [DecoderLayer] x num_hidden_layers -> final RMSNorm -> lm_head

Differences from the gpt-oss donor:

  * **Every layer is identical** — dense GQA + dense SwiGLU, full-causal. No MoE, no per-layer type
    dispatch, no sliding/full schedule, so the 88 layers come out of one loop with one config.
  * **The embedding is SHARDED** (``tt/parallel_embedding.py``, the recipe's fixed reference), not
    replicated. The donor leaves a TODO to do this; at Mistral's ``[131072, 12288]`` table it is not
    optional — 3.2 GiB per chip replicated against 0.10 GiB 2D-sharded.
  * **No on-device sampling.** The donor keeps ``SamplingGenerator`` hooks for a later decode
    bring-up; decode is out of scope here and prefill runs ``skip_lm_head=True``, so the hooks are
    left out rather than carried as dead code.

RoPE is the on-device INDEXED rope: the whole-cache block-cyclic SP cos/sin is built ONCE by the
runtime (``tt/rope.build_indexed_rope``) and passed into ``prefill_forward`` as ``rot_mats_global``.
This model therefore owns only the replicated RoPE transformation matrix.
"""

import torch
from loguru import logger

import ttnn
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.config import MeshConfig
from models.demos.mistral_3_5_d_p.tt.parallel_embedding import TtParallelEmbedding, cache_name_for, embed_shard_2d
from models.demos.mistral_3_5_d_p.tt.rope import build_transformation_mat
from models.demos.mistral_3_5_d_p.utils.general_utils import get_cache_file_name, get_matmul_compute_config
from models.demos.mistral_3_5_d_p.utils.substate import substate

from .layer import DecoderLayer
from .rms_norm import RMSNorm


def compute_per_device_vocab(vocab_size, num_tp):
    """Per-device LM-head vocab width: tile-aligned, then rounded up to a power of two.

    The power-of-two rounding is what lets ``ttnn.topk``'s multi-core (bitonic) path run on the
    logits, and it must match both the lm_head weight padding and any sampling device-offset stride.
    Mistral needs no padding at all — 131072 / 8 = 16384 is already tile-aligned AND a power of two —
    but the computation is kept so a different TP or a padded vocab still lands consistently.
    """
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()


class Model:
    """Mistral-Medium-3.5-128B TTNN prefill model (dense GQA + dense SwiGLU)."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_local_batch_size=1,
        max_seq_len=None,
        sequence_parallel=False,
        num_layers=None,
        first_layer_idx=0,
        weight_dtype=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            hf_config: HF text config
            state_dict: the whole model's HF state dict (q/k already Meta-swizzled), or {} for a
                cache-only load
            ccl_manager: Communication manager
            dtype: residual-stream activation dtype
            tensor_cache_path: root for the tilized weight cache
            mesh_config: Mesh parallelization config (defaults to TP = mesh cols)
            max_local_batch_size: users packed per device
            max_seq_len: KV cache CAPACITY in tokens, threaded into every layer
            sequence_parallel: take the SP cache-backed ring attention path
            num_layers: build only the first N layers (a reduced-depth run); default: all
            first_layer_idx: GLOBAL index of this instance's first layer, so the weight-cache keys
                and the state-dict lookups stay global under a pipeline split
            weight_dtype: override the spec's weight dataformats (bring-up A/B only; see DecoderLayer)
        """
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.vocab_size = hf_config.vocab_size
        self.head_dim = hf_config.head_dim
        self.max_local_batch_size = max_local_batch_size
        self.sequence_parallel = sequence_parallel
        self.ccl_manager = ccl_manager
        self.first_layer_idx = first_layer_idx
        self.num_layers = hf_config.num_hidden_layers if num_layers is None else num_layers
        max_seq_len = SPEC.cache_capacity if max_seq_len is None else max_seq_len

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])

        # Replicated RoPE transformation matrix. The cos/sin themselves are the whole-cache indexed
        # rope, built once by the runtime and passed into prefill_forward.
        self.transformation_mats = {"prefill": build_transformation_mat(mesh_device)}

        # --- token embedding (sharded; see tt/parallel_embedding.py) ---
        embedding_weight = substate(state_dict, "model.embed_tokens")["weight"] if state_dict else None
        shard_vocab_on_sp = embed_shard_2d()
        self.embedding = TtParallelEmbedding(
            mesh_device,
            vocab_size=self.vocab_size,
            emb_dim=hf_config.hidden_size,
            mesh_config=self.mesh_config,
            ccl_manager=ccl_manager,
            torch_weight=embedding_weight,
            cache_file_name=get_cache_file_name(tensor_cache_path, cache_name_for(shard_vocab_on_sp)),
            shard_vocab_on_sp=shard_vocab_on_sp,
        )

        # --- decoder stack ---
        logger.info(
            f"Building {self.num_layers} decoder layers (global {first_layer_idx}.."
            f"{first_layer_idx + self.num_layers - 1}), max_seq_len={max_seq_len}"
        )
        self.layers = [
            DecoderLayer(
                mesh_device,
                hf_config,
                # Cache keys and state-dict keys are GLOBAL, so a pipeline rank's local layer i is
                # global first_layer_idx + i.
                substate(state_dict, f"model.layers.{first_layer_idx + i}"),
                i,  # LOCAL index: this is the layer's offset inside each user's KV cache slot
                ccl_manager,
                dtype=dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"model.layers.{first_layer_idx + i}"),
                mesh_config=self.mesh_config,
                transformation_mats=self.transformation_mats,
                max_seq_len=max_seq_len,
                max_local_batch_size=max_local_batch_size,
                sequence_parallel=sequence_parallel,
                weight_dtype=weight_dtype,
            )
            for i in range(self.num_layers)
        ]

        # --- final norm ---
        self.norm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "model.norm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "norm"),
            mesh_config=self.mesh_config,
        )

        # --- lm head (column-parallel over the vocab) ---
        sampling_splits = mesh_device.shape[self.mesh_config.tp_axis]
        self.per_device_vocab = compute_per_device_vocab(self.vocab_size, sampling_splits)
        self.padded_vocab_size = self.per_device_vocab * sampling_splits
        if state_dict:
            lm_head_weight = substate(state_dict, "lm_head")["weight"].transpose(0, 1)  # [hidden, vocab]
            if lm_head_weight.shape[1] < self.padded_vocab_size:
                lm_head_weight = torch.nn.functional.pad(
                    lm_head_weight, (0, self.padded_vocab_size - lm_head_weight.shape[1]), "constant", 0
                )
        else:
            lm_head_weight = None
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head_padded_pow2.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.column_parallel(mesh_device),
        )
        self.matmul_config = get_matmul_compute_config(mesh_device)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def embed(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """SP-sharded uint32 token ids -> the bf16 residual stream the layers consume.

        bf16 (not bf8) on purpose: the residual stream keeps its full dynamic range, which bf8's
        per-tile shared exponent would crush once large activations appear deep in the stack.
        """
        return self.embedding.forward(tokens)

    def _forward_layers_and_head(
        self,
        hidden_states,
        rope_mats,
        current_pos,
        get_last_token=-1,
        user_id=0,
        batch_size=1,
        skip_lm_head=False,
        on_layer_complete=None,
        kv_cache=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """Prefill forward through the decoder layers plus the final projection.

        ``on_layer_complete``: optional ``fn(layer_idx)`` invoked after each layer — the SEAM for
        per-layer KV migration / validation in the disaggregated pipeline. Default None = no-op.
        ``cached_len``: valid prefix already in the cache before this chunk (0 = first/only chunk).
        """
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=rope_mats,
                position_idx=current_pos,
                kv_cache=kv_cache,
                user_id=user_id,
                batch_size=batch_size,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            if on_layer_complete is not None:
                on_layer_complete(i)

        if get_last_token != -1:
            hidden_states = self._slice_last_token(hidden_states, get_last_token, batch_size)

        if skip_lm_head:
            return hidden_states

        hidden_states_normed = self.norm(hidden_states)
        hidden_states.deallocate(True)
        # bf16 logits, not the donor's bf8. The spec fixes activations at bfloat16, and the logits are
        # the one activation a caller reads directly and argmaxes: bf8's per-tile shared exponent
        # costs measurable top-1 agreement on a 131072-wide row for no memory that matters (the
        # logits are a single tile of tokens by the time anyone keeps them).
        logits = ttnn.matmul(
            hidden_states_normed,
            self.lm_head_weight,
            dtype=SPEC.activation_dtype,
            compute_kernel_config=self.matmul_config,
        )
        hidden_states_normed.deallocate(True)
        return logits

    def _slice_last_token(self, logits, get_last_token, batch_size):
        """Keep only the tile containing each user's last real token (decode hand-off)."""
        if len(logits.shape) == 3:
            logits = ttnn.unsqueeze(logits, dim=1)
        if batch_size > 1:
            per_user_seq = logits.shape[2] // batch_size
            tiles = []
            for b in range(batch_size):
                start = b * per_user_seq + get_last_token
                tiles.append(ttnn.slice(logits, (0, 0, start, 0), (1, 1, start + 32, logits.shape[-1])))
            logits.deallocate(True)
            out = ttnn.concat(tiles, dim=2)
            for tile in tiles:
                tile.deallocate(True)
            return out
        sliced = ttnn.slice(logits, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, logits.shape[-1]))
        logits.deallocate(True)
        return sliced

    def prefill_forward(
        self,
        x,
        rot_mats_global=None,
        user_id=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        skip_lm_head=False,
        on_layer_complete=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """Prefill forward pass. ``rot_mats_global`` is the whole-cache indexed rope from the runtime."""
        assert rot_mats_global is not None, (
            "Mistral prefill uses the on-device indexed rope; pass rot_mats_global " "(tt/rope.build_indexed_rope)"
        )
        return self._forward_layers_and_head(
            hidden_states=x,
            rope_mats=rot_mats_global,
            current_pos=None,
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            user_id=user_id,
            batch_size=batch_size,
            skip_lm_head=skip_lm_head,
            on_layer_complete=on_layer_complete,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )

    def prepare_inputs_prefill(self, tokens, start_pos=0, trace_enabled=False, batch_size=1, user_id=0, **kwargs):
        """Embed + SP-shard one chunk's token ids into the model input.

        RoPE is NOT built here — prefill uses the on-device indexed rope, which the runtime builds
        once and passes into ``prefill_forward``. Returns ``(tokens_embd, None, None)`` to keep the
        3-tuple interface the tt_transformers generator expects.
        """
        device = None if trace_enabled else self.mesh_device
        if tokens.dim() == 1:
            tokens = tokens.reshape(1, -1)
        seq_total = tokens.shape[-1]

        if self.sequence_parallel:
            sp = self.mesh_device.shape[self.mesh_config.sp_axis]
            assert seq_total % sp == 0, f"SP prefill needs seq_len ({seq_total}) divisible by sp ({sp})"
            tdims = [None, None]
            tdims[self.mesh_config.sp_axis] = 3  # seq across the SP rows
            mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=tuple(tdims), mesh_shape=tuple(self.mesh_device.shape)
            )
        else:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        tt_tokens = ttnn.from_torch(
            tokens.reshape(1, 1, 1, seq_total),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        tokens_embd = self.embed(ttnn.reshape(tt_tokens, [1, 1, tt_tokens.shape[-1]]))
        tt_tokens.deallocate(True)
        return tokens_embd, None, None

    def process_output_prefill(self, tt_out, last_token_idx):
        """Host-side TP gather of the last-token logits (the lm_head is column-parallel on vocab)."""
        tp = self.mesh_config.tp
        device_tensors = ttnn.get_device_tensors(tt_out)
        if tp > 1:
            torch_output = torch.cat([ttnn.to_torch(device_tensors[i]) for i in range(tp)], dim=-1)
        else:
            torch_output = ttnn.to_torch(device_tensors[0])
        return torch_output[..., last_token_idx, : self.vocab_size]
