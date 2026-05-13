# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full Qwen3.6-27B text transformer on BH GLX 8×4 mesh — Task 8.

TtQwen36Transformer stacks:
  - Token embedding  (vocab → hidden)
  - N TtQwen36DecoderLayer  (from T7)
  - Final DistributedNorm   (zero_centered=True)
  - LM head linear          (hidden → vocab)

Design principles
-----------------
- Weights arrive as pre-loaded Python dicts (no weight-cache path needed for
  bring-up). The caller loads them via safetensors and passes them in.
- All intermediate tensors live in DRAM. L1-sharded optimisation is a later task.
- Embedding and LM head use simple replicated matmul (no TP split for bring-up).
- KV caches and DeltaNet states are explicitly managed so the caller can do
  decode steps after prefill.

Forward API
-----------
  forward_prefill(input_ids) -> logits [B, T, vocab_size] (CPU torch tensor)
  forward_prefill(input_ids, return_caches=True)
        -> (logits, kv_caches, dn_states, conv_states)
  forward_decode(next_id, current_pos, kv_caches, dn_states, conv_states)
        -> (logits [B, 1, vocab_size], kv_caches, dn_states, conv_states)

State layout
-----------
  kv_caches : list[None]  (len=num_layers, None for linear_attention layers)
              For now, full_attention layers also use None (no KV cache in bring-up).
  dn_states  : list[ttnn.Tensor | None]  (len=num_layers, only set for linear_attention)
  conv_states: list[ttnn.Tensor | None]  (len=num_layers, only set for linear_attention)
"""
from __future__ import annotations

from typing import List, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm
from models.demos.qwen3_6_galaxy.tt.llama_decoder import TtQwen36DecoderLayer, _gather_from_cols, _shard_across_cols

TILE_HEIGHT = 32


class TtQwen36Transformer(LightweightModule):
    """Full Qwen3.6-27B text-only transformer on BH GLX 8×4 mesh.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        The full 8×4 BH GLX mesh.
    args : TtQwen36ModelArgs
        Model configuration.
    global_weights : dict
        Dict with keys:
          tok_embeddings.weight  [vocab_size, hidden_size]
          norm.weight            [hidden_size]
          output.weight          [vocab_size, hidden_size]
    layers_weights : list[dict]
        Per-layer weight dicts (same format as T7's TtQwen36DecoderLayer).
        Length must equal num_layers.
    num_layers : int
        Number of decoder layers to load (default = args.num_hidden_layers = 64).
    dtype : ttnn.DataType
        Activation dtype (default bfloat16).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        args,
        global_weights: dict,
        layers_weights: list,
        num_layers: Optional[int] = None,
        dtype=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype or ttnn.bfloat16
        self.num_layers = num_layers if num_layers is not None else args.num_hidden_layers
        self.cluster_shape = list(args.cluster_shape)  # [8, 4]

        # T14b.2: persistent input_ids device buffers, keyed by (B, T).  Each
        # entry is a uint32 ROW_MAJOR replicated [1,1,1,B*T] tensor reused
        # across forward calls via ``ttnn.copy_host_to_device_tensor`` so the
        # captured trace can replay the embedding op without allocating fresh.
        self._input_ids_buffers: dict[tuple[int, int], ttnn.Tensor] = {}

        assert (
            len(layers_weights) == self.num_layers
        ), f"layers_weights has {len(layers_weights)} entries, but num_layers={self.num_layers}"

        # Build RoPE setup (needed for full_attention layers inside each decoder)
        from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup

        self.rope_setup = Qwen36RopeSetup(
            mesh_device=mesh_device,
            args=args,
            batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
        )

        # ------------------------------------------------------------------
        # Embedding
        # ------------------------------------------------------------------
        # Weight: [vocab_size, hidden] → uploaded replicated for simplicity
        emb_w = global_weights["tok_embeddings.weight"]  # [vocab_size, hidden]
        self.emb_weight = ttnn.from_torch(
            emb_w.unsqueeze(0).unsqueeze(0),  # [1, 1, vocab_size, hidden]
            device=mesh_device,
            dtype=ttnn.bfloat16,  # embedding requires bfloat16
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.vocab_size = emb_w.shape[0]
        self.hidden_size = emb_w.shape[1]

        # ------------------------------------------------------------------
        # Decoder layers
        # ------------------------------------------------------------------
        from tqdm import tqdm

        self.layers = [
            TtQwen36DecoderLayer(
                mesh_device=mesh_device,
                args=args,
                state_dict=layers_weights[i],
                layer_idx=i,
                dtype=self.dtype,
                rope_setup=self.rope_setup,
            )
            for i in tqdm(range(self.num_layers), desc="Loading decoder layers")
        ]

        # ------------------------------------------------------------------
        # Final norm (zero_centered=True, same as each layer's input_layernorm)
        # Uses DistributedNorm with HiFi4+fp32_dest_acc on H/4=1280 per col.
        # ------------------------------------------------------------------
        norm_w = global_weights["norm.weight"].float()
        self.final_norm = DistributedNorm(
            mesh_device=mesh_device,
            weight_torch=norm_w,
            eps=args.norm_eps,
            zero_centered=True,
        )

        # ------------------------------------------------------------------
        # LM head: [hidden, padded_vocab] replicated for bring-up
        # output.weight in HF: [vocab_size, hidden] → we need [hidden, vocab]
        # ------------------------------------------------------------------
        lm_head_w = global_weights["output.weight"]  # [vocab_size, hidden]
        padded_vocab = args.padded_vocab_size  # 248832
        if lm_head_w.shape[0] < padded_vocab:
            pad = torch.zeros(padded_vocab - lm_head_w.shape[0], lm_head_w.shape[1], dtype=lm_head_w.dtype)
            lm_head_w_padded = torch.cat([lm_head_w, pad], dim=0)  # [padded_vocab, hidden]
        else:
            lm_head_w_padded = lm_head_w[:padded_vocab]

        # On-device LM head weight, vocab-sharded across 4 mesh cols (cluster_axis=1).
        # Stored as [hidden, padded_vocab] so ttnn.linear(x [B,T,H], w [H, V/4]) → [B,T, V/4]
        # partial per col, then ConcatMesh2dToTensor (dims=(None, -1)) gathers
        # the per-col slices into full [B, T, padded_vocab].
        # Per-chip: 5120 × 62208 × 2 B = 637 MB (vs 2548 MB replicated).
        lm_head_w_t = lm_head_w_padded.T.contiguous()  # [hidden, padded_vocab]
        vocab_col_shard = ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 1),
            mesh_shape=list(mesh_device.shape),
        )
        self.lm_head_weight = ttnn.from_torch(
            lm_head_w_t,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=vocab_col_shard,
        )  # per-chip: [hidden, padded_vocab/4 = 62208]
        self.padded_vocab_size = padded_vocab
        self.vocab_per_col = padded_vocab // self.cluster_shape[1]  # 62208

        # LM head compute kernel: HiFi4 + fp32 dest accumulation.  The 5120-dim
        # accumulation into a 62208-wide per-col output benefits from fp32 accumulation
        # the same way attention/MLP wide-output linears do.
        self._compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------
    # Private: embedding helper
    # ------------------------------------------------------------------

    def _get_input_ids_buffer(self, B: int, T: int) -> ttnn.Tensor:
        """Return a preallocated, replicated device buffer for input_ids of
        shape ``[1, 1, 1, B*T]`` uint32.

        Lazily allocates on first call for that ``(B, T)`` shape; subsequent
        calls reuse the same on-device handle.  The buffer is persistent —
        callers MUST NOT deallocate it.
        """
        key = (B, T)
        if key not in self._input_ids_buffers:
            placeholder = torch.zeros(1, 1, 1, B * T, dtype=torch.int64)
            self._input_ids_buffers[key] = ttnn.from_torch(
                placeholder,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self._input_ids_buffers[key]

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """Embed input_ids and return replicated [B, T, H] TTNN tensor.

        T14b.2: refreshes a preallocated persistent device buffer via
        ``ttnn.copy_host_to_device_tensor`` instead of allocating a fresh
        device tensor with ``ttnn.from_torch`` each call.  This keeps the
        input_ids tensor at a stable device address so a captured trace can
        replay the embedding op without per-call allocation.

        Args:
            input_ids: CPU torch tensor [B, T] int64

        Returns:
            TTNN tensor [B, T, H] bfloat16, replicated across all mesh devices.
        """
        B, T = input_ids.shape
        full_BT = B * T

        # ttnn.embedding expects input shape [1, 1, 1, B*T]
        ids_flat = input_ids.reshape(1, 1, 1, full_BT)

        # Build a HOST tensor with the SAME dtype/layout/mesh_mapper as the
        # persistent device buffer so copy_host_to_device_tensor can do an
        # in-place per-shard write (no device= argument keeps it host-only;
        # the Python reference goes out of scope after the copy).
        ids_host_tt = ttnn.from_torch(
            ids_flat,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ids_buf = self._get_input_ids_buffer(B, T)
        ttnn.copy_host_to_device_tensor(ids_host_tt, ids_buf)

        # Embedding lookup.  TILE_LAYOUT tile-pads the row dim to a multiple of 32,
        # which is fine for prefill (T is already a multiple of 32) but corrupts
        # the logical shape for decode (T=1 stored in a 32-row tile).
        # For decode (B*T < 32), do the lookup in ROW_MAJOR_LAYOUT — exact rows,
        # no padding — then convert to TILE_LAYOUT after reshaping to [B, T, H].
        if full_BT < TILE_HEIGHT:
            x_tt = ttnn.embedding(
                ids_buf,
                self.emb_weight,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            # [1, 1, B*T, H] (no padding) → [B, T, H]
            x_tt = ttnn.reshape(x_tt, ttnn.Shape([B, T, self.hidden_size]))
            x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)
        else:
            x_tt = ttnn.embedding(
                ids_buf,
                self.emb_weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            x_tt = ttnn.reshape(x_tt, ttnn.Shape([B, T, self.hidden_size]))
        # DO NOT deallocate ids_buf — it's the persistent buffer reused across calls.
        return x_tt

    # ------------------------------------------------------------------
    # Private: final_norm + lm_head helper
    # ------------------------------------------------------------------

    def _norm_and_lm_head(self, x: ttnn.Tensor, output_as_ttnn: bool = False):
        """Apply final norm and LM head ENTIRELY on device, then optionally gather logits.

        Plan
        ----
        1. DistributedNorm on the replicated [B,T,H] → normed [B,T,H] replicated.
        2. LM head matmul against the vocab-col-sharded weight
           ``self.lm_head_weight [H, padded_vocab/4]``:
             - x [B,T,H] replicated × W [H, V/4] per col → [B,T, V/4] per col
             - Across rows the result is identical (rows replicate the same weight shard).
             - Across cols each col holds a different V/4 slice.
        3. If ``output_as_ttnn=False`` (eager default), gather to host via
           ``ConcatMesh2dToTensor(dims=(0, -1))`` which concatenates dim=0 over the
           8 rows (giving us redundant row copies to slice out) and dim=-1 over the
           4 cols (giving us the full vocab).  Take the first row's copy →
           [B, T, padded_vocab] CPU float32.

           If ``output_as_ttnn=True`` (trace path), return the on-device TTNN
           logits tensor.  The caller is responsible for the same
           ``ttnn.to_torch(...)`` gather AFTER ``ttnn.execute_trace`` returns so
           the gather is not part of the captured trace.

        Args:
            x: TTNN tensor [B, T, H] replicated.
            output_as_ttnn: If True, return the on-device TTNN logits tensor
                instead of gathering to a CPU torch tensor.  Default False
                preserves the eager API.

        Returns:
            If ``output_as_ttnn=False``: CPU torch float32 tensor
            [B, T, padded_vocab_size].
            If ``output_as_ttnn=True``: ttnn.Tensor with per-chip shape
            [B, T, padded_vocab/cluster_cols].  Caller does the
            ConcatMesh2dToTensor gather + row-slice to produce the final
            [B, T, padded_vocab] CPU tensor.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG

        # 1. DistributedNorm — shard across cols → norm → gather back to replicated.
        x_sharded = _shard_across_cols(x, self.mesh_device)
        x_normed_sharded = self.final_norm(x_sharded)
        x_sharded.deallocate(True)
        x_normed = _gather_from_cols(x_normed_sharded, self.mesh_device)
        x_normed_sharded.deallocate(True)

        # 2. On-device LM head matmul.  Per-chip output: [B, T, V_per_col=62208].
        logits_tt = ttnn.linear(
            x_normed,
            self.lm_head_weight,
            dtype=self.dtype,
            memory_config=mem,
            compute_kernel_config=self._compute_kernel,
        )
        x_normed.deallocate(True)

        if output_as_ttnn:
            # Trace path: caller will gather to host AFTER ttnn.execute_trace.
            return logits_tt

        # 3. Eager path: gather to host — concat row-replicates on dim=0 (giving
        #    8×B rows) and cols' vocab shards on dim=-1 (giving full padded_vocab).
        logits_cpu_concat = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device,
                dims=(0, -1),
                mesh_shape=self.cluster_shape,
            ),
        )  # [8*B, T, padded_vocab]
        logits_tt.deallocate(True)

        # Take the first row's copy (rows are replicated)
        n_rows = self.cluster_shape[0]
        B = logits_cpu_concat.shape[0] // n_rows
        logits_cpu = logits_cpu_concat[:B].float()  # [B, T, padded_vocab]
        return logits_cpu

    # ------------------------------------------------------------------
    # Private: allocate empty caches
    # ------------------------------------------------------------------

    def _empty_caches(self):
        """Return (kv_caches, dn_states, conv_states) all None-filled lists."""
        kv_caches = [None] * self.num_layers
        dn_states = [None] * self.num_layers
        conv_states = [None] * self.num_layers
        return kv_caches, dn_states, conv_states

    # ------------------------------------------------------------------
    # Public: forward_prefill
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        return_caches: bool = False,
        page_table=None,
        output_as_ttnn: bool = False,
    ):
        """Run prefill forward pass.

        Args:
            input_ids: CPU torch tensor [B, T] int64.  T must be a multiple of 32.
            return_caches: If True, return caches alongside logits.
            page_table: Optional TTNN int32 tensor [B, max_blocks_per_seq] for
                paged KV cache. Required when args.use_paged_kv_cache=True.
            output_as_ttnn: If True, return the on-device TTNN logits tensor
                from the LM head (per-chip shape [B, T, padded_vocab/cluster_cols])
                instead of gathering to a CPU torch tensor.  Used by the trace
                capture path so the final ``ttnn.to_torch`` gather happens AFTER
                ``ttnn.execute_trace``.  Default False preserves the eager API.

        Returns:
            If return_caches=False:
                logits: CPU torch tensor [B, T, padded_vocab_size] float32,
                or ttnn.Tensor when output_as_ttnn=True.
            If return_caches=True:
                (logits, kv_caches, dn_states, conv_states)
        """
        B, T = input_ids.shape
        assert T % 32 == 0, f"Prefill T={T} must be a multiple of 32 for tile alignment"

        # Build MRoPE cos/sin for text positions [0, T)
        cos_tt, sin_tt = self.rope_setup.get_cos_sin_for_prefill(T)
        rot_mats = (cos_tt, sin_tt)

        # Embedding
        x = self._embed(input_ids)  # [B, T, H] replicated on device

        # Initialise caches
        kv_caches, dn_states, conv_states = self._empty_caches()

        # Decoder layers
        for i, layer in enumerate(self.layers):
            x, dn_states[i], conv_states[i] = layer.forward(
                x,
                current_pos=0,
                rot_mats=rot_mats,
                kv_cache=kv_caches[i],
                page_table=page_table,
                deltanet_state=dn_states[i],
                deltanet_conv_state=conv_states[i],
                mode="prefill",
            )

        # Slice x back to T in case tile-padding happened inside decoder
        x_shape = list(x.shape)
        T_out = x_shape[1] if len(x_shape) == 3 else x_shape[-2]
        if T_out != T:
            x = ttnn.slice(x, [0, 0, 0], [B, T, self.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Final norm + LM head
        logits = self._norm_and_lm_head(x, output_as_ttnn=output_as_ttnn)  # [B, T, padded_vocab]

        if return_caches:
            return logits, kv_caches, dn_states, conv_states
        return logits

    # ------------------------------------------------------------------
    # Public: forward_prefill_hidden
    # ------------------------------------------------------------------

    def forward_prefill_hidden(
        self,
        input_ids: torch.Tensor,
    ) -> tuple:
        """Run prefill and return the hidden state AFTER all decoder layers, BEFORE final norm.

        This is the true block-level correctness metric for the transformer stack.
        PCC > 0.99 on this pre-norm hidden state proves that the full stack of
        (embedding → decoder layers) is numerically correct — identical to what
        T7 (4-layer decoder test) proved at PCC = 0.999949.

        Note: we deliberately stop before the final distributed RMSNorm because:
        1. The norm is a BF16 distributed operation with extra host roundtrip overhead.
        2. T7 already validated the decoder stack (which includes input/post layernorms).
        3. Top-1 generation correctness (argmax after LM head) verifies the end-to-end
           output separately.

        Args:
            input_ids: CPU torch tensor [B, T] int64.  T must be a multiple of 32.

        Returns:
            (emb_cpu, hidden_cpu)
            emb_cpu:    CPU torch float32 tensor [B, T, hidden_size] — BF16 embedding output
                        (same values as what the decoder loop starts from on device).
            hidden_cpu: CPU torch float32 tensor [B, T, hidden_size] — pre-final-norm decoder
                        output.  Compare this against ref_model forward with emb_cpu as input.
        """
        B, T = input_ids.shape
        assert T % 32 == 0, f"Prefill T={T} must be a multiple of 32 for tile alignment"

        # Build MRoPE cos/sin for text positions [0, T)
        cos_tt, sin_tt = self.rope_setup.get_cos_sin_for_prefill(T)
        rot_mats = (cos_tt, sin_tt)

        # Embedding
        x = self._embed(input_ids)  # [B, T, H] replicated on device

        # Capture embedding output for reference comparison (before decoder layers)
        emb_cpu_raw = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        emb_cpu = emb_cpu_raw[:B].float()  # [B, T, H] — BF16 values as float32

        # Initialise caches
        kv_caches, dn_states, conv_states = self._empty_caches()

        # Decoder layers
        for i, layer in enumerate(self.layers):
            x, dn_states[i], conv_states[i] = layer.forward(
                x,
                current_pos=0,
                rot_mats=rot_mats,
                kv_cache=kv_caches[i],
                page_table=None,  # forward_prefill_hidden is non-paged (diagnostic only)
                deltanet_state=dn_states[i],
                deltanet_conv_state=conv_states[i],
                mode="prefill",
            )

        # Slice x back to T in case tile-padding happened inside decoder
        x_shape = list(x.shape)
        T_out = x_shape[1] if len(x_shape) == 3 else x_shape[-2]
        if T_out != T:
            x = ttnn.slice(x, [0, 0, 0], [B, T, self.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Gather pre-norm hidden state to CPU — use first device from replicated tensor
        # x is [B, T, H] replicated across 32 devices; ConcatMeshToTensor(dim=0) → [32*B, T, H]
        x_cpu = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        x.deallocate(True)

        # Take first device's copy (all are identical — replicated)
        hidden_cpu = x_cpu[:B].float()  # [B, T, hidden_size]
        return emb_cpu, hidden_cpu

    # ------------------------------------------------------------------
    # Public: forward_decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        current_pos: int,
        kv_caches: Optional[List] = None,
        dn_states: Optional[List] = None,
        conv_states: Optional[List] = None,
        output_as_ttnn: bool = False,
    ):
        """Run a single decode step.

        Args:
            input_ids: CPU torch tensor [B, 1] int64.
            current_pos: Current sequence position (for RoPE).
            kv_caches: List[None|...] of length num_layers.
            dn_states: List[None|ttnn.Tensor] of length num_layers.
            conv_states: List[None|ttnn.Tensor] of length num_layers.
            output_as_ttnn: If True, return the on-device TTNN logits tensor
                from the LM head instead of a CPU torch tensor.  Used by the
                trace capture path so the final ``ttnn.to_torch`` gather happens
                AFTER ``ttnn.execute_trace``.  Default False preserves the
                eager API.

        Returns:
            (logits [B, 1, padded_vocab_size] CPU torch — or ttnn.Tensor when
            output_as_ttnn=True, kv_caches, dn_states, conv_states)
        """
        B, T = input_ids.shape
        assert T == 1, f"forward_decode expects T=1, got T={T}"

        if kv_caches is None:
            kv_caches = [None] * self.num_layers
        if dn_states is None:
            dn_states = [None] * self.num_layers
        if conv_states is None:
            conv_states = [None] * self.num_layers

        # MRoPE for single decode position
        cos_tt, sin_tt = self.rope_setup.get_cos_sin_for_decode(current_pos)
        rot_mats = (cos_tt, sin_tt)

        # Embedding
        x = self._embed(input_ids)  # [B, 1, H] replicated

        # Decoder layers
        new_dn_states = [None] * self.num_layers
        new_conv_states = [None] * self.num_layers
        for i, layer in enumerate(self.layers):
            x, new_dn_states[i], new_conv_states[i] = layer.forward(
                x,
                current_pos=current_pos,
                rot_mats=rot_mats,
                kv_cache=kv_caches[i],
                page_table=None,
                deltanet_state=dn_states[i],
                deltanet_conv_state=conv_states[i],
                mode="decode",
            )

        # Slice back to T=1 if tile-padded
        x_shape = list(x.shape)
        T_out = x_shape[1] if len(x_shape) == 3 else x_shape[-2]
        if T_out > 1:
            x = ttnn.slice(x, [0, 0, 0], [B, 1, self.hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Final norm + LM head
        logits = self._norm_and_lm_head(x, output_as_ttnn=output_as_ttnn)  # [B, 1, padded_vocab]

        return logits, kv_caches, new_dn_states, new_conv_states
