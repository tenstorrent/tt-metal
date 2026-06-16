# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B text model — prefill and decode modes.

``TtMistral4TextModel`` wraps the decoder-layer stack with per-layer KV caches.
RoPE ``(cos, sin)`` tables are uploaded once via ``cache_rope_tables`` and live
in device DRAM; per-step lookups are on-device ``ttnn.slice`` calls keyed on
``current_pos``, so no host→device cos/sin upload happens during the decode loop.

Usage::

    model = TtMistral4TextModel(...)
    cos_full, sin_full = rotary(...)              # once, on host
    model.cache_rope_tables(cos_full, sin_full)   # one-shot upload

    next_tok = model.prefill_next_token(input_ids)        # on-device argmax
    for pos in range(prefill_len, prefill_len + n_tokens):
        tok = torch.tensor([[next_tok]], dtype=torch.long)
        next_tok = model.decode_next_token(tok, pos)      # on-device argmax
"""

from __future__ import annotations

import os
import pathlib

import torch
import math

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_NUM_LAYERS,
    HIDDEN_SIZE,
    QK_ROPE_HEAD_DIM,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _load_norm_weight
from models.experimental.mistral_small_4_119b.tt.mistral4_text_prefill import (
    TtMistral4DecoderLayer,
    _rms_norm,
    _rms_norm_sharded_decode,
)


class TtMistral4TextModel:
    """
    Mistral-Small-4 text model with prefill and decode support.

    Args:
        mesh_device:        TTNN MeshDevice (e.g. 4×P150 → [1, 4] mesh)
        state_dict:         HF checkpoint dict (filtered to required prefixes)
        text_config:        HF ``text_config`` object
        num_decoder_layers: layers to instantiate (1..36; default 36)
        max_seq_len:        maximum total tokens (prefill + decode); sets cache size
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        text_config,
        num_decoder_layers: int = EXPECTED_NUM_LAYERS,
        max_seq_len: int = 4096,
        cache_dir=None,
    ):
        self.mesh_device = mesh_device
        self.num_decoder_layers = num_decoder_layers
        # Prefill runs in fixed-size chunks. A ragged (non-chunk-multiple) final chunk makes
        # the projection matmuls pick a per_core_M/out_subblock layout whose circular buffers
        # clash with L1 (assert.hpp:104), so _run_prefill_chunked pads the final chunk up to
        # this size. Round the KV-cache capacity up to a chunk multiple so the padded rows fit.
        self._prefill_chunk = int(os.environ.get("MISTRAL4_PREFILL_CHUNK", "512"))
        self.max_seq_len = ((max_seq_len + self._prefill_chunk - 1) // self._prefill_chunk) * self._prefill_chunk

        # Resolve cache dir: explicit arg > env var > None (no caching).
        # Encode num_devices in the path so cached sharded tensors don't
        # bleed across meshes with different device counts.
        _cache_dir_base = cache_dir or os.environ.get("MISTRAL4_WEIGHT_CACHE_DIR")
        if _cache_dir_base is not None:
            num_devices = mesh_device.get_num_devices()
            _effective_cache_dir = pathlib.Path(_cache_dir_base) / f"n{num_devices}"
            _effective_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            _effective_cache_dir = None
        _cf = (lambda key: str(_effective_cache_dir / key)) if _effective_cache_dir is not None else (lambda _: None)

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # ── Embedding ──────────────────────────────────────────────────
        embed_w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY].to(torch.bfloat16)
        self.embed_weight = ttnn.as_tensor(
            embed_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY),
        )

        # ── Decoder layers + per-layer KV caches ───────────────────────
        self.decoder_layers: list[TtMistral4DecoderLayer] = []
        self.kv_caches: list[tuple] = []

        for i in range(num_decoder_layers):
            layer = TtMistral4DecoderLayer(
                mesh_device=mesh_device,
                state_dict=state_dict,
                layer_idx=i,
                compute_kernel_config=self.compute_kernel_config,
                cache_dir=_effective_cache_dir,
            )
            self.decoder_layers.append(layer)
            self.kv_caches.append(layer.attn.allocate_kv_cache(self.max_seq_len))

        # ── Final norm + LM head ────────────────────────────────────────
        self.final_norm_w = _load_norm_weight(
            state_dict,
            "language_model.model.norm.weight",
            HIDDEN_SIZE,
            mesh_device,
            cache_file_name=_cf("language_model.model.norm.weight"),
        )

        lm_head_w = state_dict["language_model.lm_head.weight"].to(torch.bfloat16).T.contiguous()
        # bfloat4_b: ~32 MB per device after sharding across 4 devices (vs 64 MB at bfloat8_b).
        # Paired with LoFi math (self.lm_head_compute_kernel_config) for ~2x speedup on the
        # LM head matmul. EXPECTED to regress generation quality — gate behind a PCC test.
        TILE = 32
        N_PER_DEVICE = 131072 // mesh_device.get_num_devices()  # 16384 for P150x8
        dram = mesh_device.dram_grid_size()  # P150 → x=8, y=1
        self.num_banks = dram.x
        shard_width = math.ceil(math.ceil(N_PER_DEVICE / self.num_banks) / TILE) * TILE
        lm_head_in1_memcfg = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.DRAM,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, 0))}),
                [4096, shard_width],  # [K, N_per_bank]
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_w,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=lm_head_in1_memcfg,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            cache_file_name=_cf("language_model.lm_head.weight"),  # bump or remove the cache, layout changed!
        )

        NT_per_device = N_PER_DEVICE // TILE  # 512
        # in0_block_w=2 (not 8): the DRAM-sharded weight-prefetch CB scales with
        # in0_block_w, and at 8 the static CB region needs ~1.23 MB contiguous L1.
        # On the sparse moe_compute path L1 is heavily fragmented (small persistent
        # buffers cap the largest contiguous free block at ~0.84 MB even though L1 is
        # 97% free), so the 8-wide CBs clash with those buffers in the full-depth
        # decode/prefill logits paths. in0_block_w=2 shrinks the weight CB ~4× so the
        # region fits the available contiguous block; correctness is unchanged (just
        # more K-blocks), and smaller CBs only add headroom for the production decode.
        self.lm_head_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=2,
            per_core_M=1,
            per_core_N=NT_per_device // self.num_banks,  # 512/8 = 64
            fused_activation=None,
        )

        # Second weight + program config for prefill paths (M = seq_len > 1 tile).
        # DS requires M == 1 tile (matmul_device_operation.cpp:757), so prefill cannot
        # share the DS tensor — it needs DRAM-interleaved weights + the 1D mcast path.
        self.lm_head_weight_prefill = ttnn.as_tensor(
            lm_head_w,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            cache_file_name=_cf("language_model.lm_head.weight.prefill_interleaved"),
        )
        # 1D-mcast prefill PC: N (512 tiles/device) is split across the 8×8=64-core
        # grid (per_core_N=8 → 64 cores), and each core loops the FULL M on-core, so
        # per_core_M MUST equal the M-tile count. Hardcoding per_core_M=1 makes the op
        # also fan M across cores (num_blocks_total = m_tiles × 64), which trips the
        # `num_blocks_total <= num_cores` check for any seq_len > 32. Build the PC per
        # m_tiles instead (see _lm_head_prefill_pc); m_tiles=1 reproduces the old config.
        self._lm_per_core_N = max(1, NT_per_device // 64)  # P150x8 → 8
        self._lm_head_prefill_pc_cache: dict = {}

        self.lm_head_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # RoPE tables — uploaded once via ``cache_rope_tables``. Per-step lookup
        # is a ``ttnn.slice`` keyed on ``current_pos``; no host crossing.
        self.cos_table_tt: ttnn.Tensor | None = None
        self.sin_table_tt: ttnn.Tensor | None = None
        self._rope_table_positions: int = 0
        # CPU copies retained so the decode-step rope buffers can be updated
        # cheaply via CPU indexing + copy_host_to_device_tensor (avoids device
        # slice → new allocation each step, and enables trace replay).
        self._cos_cpu: torch.Tensor | None = None
        self._sin_cpu: torch.Tensor | None = None

        # ── Pre-allocated per-step device tensors for decode ──────────────
        # The decode loop calls ttnn.as_tensor(...) for input_id and
        # current_pos every step. as_tensor on a 4-device mesh allocates
        # fresh device buffers and runs ReplicateTensorToMesh. Pre-allocating
        # once and updating in-place via ttnn.copy_host_to_device_tensor
        # skips the per-step device allocation and the mesh-mapper dispatch.
        self._decode_input_id_device = ttnn.as_tensor(
            torch.zeros((1, 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self._decode_cur_pos_device = ttnn.as_tensor(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Pre-allocated single-position RoPE buffers for decode — updated each
        # step from _cos_cpu/_sin_cpu before the decode kernel (or trace replay).
        self._cos_decode = ttnn.as_tensor(
            torch.zeros(1, 1, 1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self._sin_decode = ttnn.as_tensor(
            torch.zeros(1, 1, 1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Pre-allocated single-element buffer for the on-device argmax result.
        # _argmax_to_device writes here; _readback_argmax reads from here.
        # Separating the two lets _argmax_to_device run inside the trace while
        # _readback_argmax (ttnn.to_torch) runs outside it.
        self._decode_token_out = ttnn.as_tensor(
            torch.zeros(1, 1, 1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Captured decode-step trace (set by capture_decode_trace); None = eager.
        self._decode_trace_id = None
        self._last_decode_pos = 0
        # CQ0 "op" event for the 2-CQ decode pipeline (armed by begin_decode_2cq).
        self._decode_2cq_op_event = None

    # ── RoPE table caching ─────────────────────────────────────────────────

    def cache_rope_tables(self, cos_full: torch.Tensor, sin_full: torch.Tensor) -> None:
        """
        Upload the full RoPE ``(cos, sin)`` table to device DRAM once.

        Accepts HF ``Mistral4RotaryEmbedding`` output (shape ``[1, S, D]`` or
        ``[1, 1, S, D]``); the last dim is trimmed to ``QK_ROPE_HEAD_DIM`` if
        wider. Replicated on every mesh device; TILE layout so subsequent
        ``ttnn.slice`` calls produce TILE tensors ready for the RoPE math.
        """

        def _upload(t: torch.Tensor) -> ttnn.Tensor:
            t = t.to(torch.bfloat16)
            while t.dim() < 4:
                t = t.unsqueeze(0)
            if t.shape[-1] > QK_ROPE_HEAD_DIM:
                t = t[..., :QK_ROPE_HEAD_DIM]
            t = t.contiguous()
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        if self.cos_table_tt is not None:
            ttnn.deallocate(self.cos_table_tt)
        if self.sin_table_tt is not None:
            ttnn.deallocate(self.sin_table_tt)

        # Normalise to [1, 1, S, D] bfloat16 on CPU before uploading, so
        # _cos_cpu/_sin_cpu are always consistently shaped for index lookups.
        def _normalise_cpu(t: torch.Tensor) -> torch.Tensor:
            t = t.to(torch.bfloat16)
            while t.dim() < 4:
                t = t.unsqueeze(0)
            if t.shape[-1] > QK_ROPE_HEAD_DIM:
                t = t[..., :QK_ROPE_HEAD_DIM]
            return t.contiguous()

        self._cos_cpu = _normalise_cpu(cos_full)
        self._sin_cpu = _normalise_cpu(sin_full)

        self.cos_table_tt = _upload(cos_full)
        self.sin_table_tt = _upload(sin_full)
        self._rope_table_positions = self.cos_table_tt.shape[-2]

    # ── Internals ──────────────────────────────────────────────────────────

    def _rope_slice(self, start: int, end: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """On-device slice of the cached cos/sin tables for positions ``[start, end)``."""
        assert (
            self.cos_table_tt is not None and self.sin_table_tt is not None
        ), "cache_rope_tables() must be called before prefill/decode"
        assert (
            0 <= start < end <= self._rope_table_positions
        ), f"position range [{start}, {end}) outside cached RoPE table of size {self._rope_table_positions}"
        cos = ttnn.slice(self.cos_table_tt, [0, 0, start, 0], [1, 1, end, QK_ROPE_HEAD_DIM])
        sin = ttnn.slice(self.sin_table_tt, [0, 0, start, 0], [1, 1, end, QK_ROPE_HEAD_DIM])
        return cos, sin

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        ids_tt = ttnn.as_tensor(
            input_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x = ttnn.embedding(ids_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ids_tt)
        return x

    def embed_tokens(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """
        Public token-embedding lookup for multimodal scatter.

        Returns a ttnn tensor of shape ``[1, 1, seq_len, HIDDEN_SIZE]`` in
        TILE layout, replicated on the mesh. Callers may slice/concat this
        with vision embeddings before calling ``prefill_from_embeds``.
        """
        seq_len = input_ids.shape[1]
        x = self._embed(input_ids)
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def _lm_head_prefill_pc(self, m_tiles: int):
        """1D-mcast LM-head program config for a full-logits prefill of ``m_tiles`` M-tiles.

        N is fixed across the 64-core grid (per_core_N); per_core_M spans the full M so
        the op loops M on-core rather than fanning it across cores (which would exceed
        the grid for seq_len > 32). Cached by m_tiles.
        """
        cached = self._lm_head_prefill_pc_cache.get(m_tiles)
        if cached is not None:
            return cached
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=m_tiles,
            per_core_N=self._lm_per_core_N,
            fuse_batch=True,
            mcast_in0=True,
        )
        self._lm_head_prefill_pc_cache[m_tiles] = pc
        return pc

    def _to_logits(self, x: ttnn.Tensor) -> torch.Tensor:
        """Final norm → lm_head → gather to host, chunked over positions.

        The prefill lm_head writes its output to L1 with ``per_core_M = M-tiles``, so running
        it over the whole sequence at once overflows L1 for long prompts. Process the sequence
        in fixed chunks (padding a ragged final chunk so the matmul always uses the validated
        ``m_tiles = chunk//32`` program config) and gather each chunk's logits to host, where
        the full ``[seq_len, vocab]`` tensor is reassembled. DS would assert here (M == 1 tile).
        """
        seq_len = x.shape[2]
        # Cap the logits chunk at 64 (per_core_M=2): the 1D-mcast lm_head's output +
        # partials CBs scale with per_core_M (= chunk//32), and at the default 512
        # (per_core_M=16) the static CB region needs ~1.2 MB contiguous L1 — which the
        # sparse moe_compute path's L1 fragmentation (largest contiguous free ~0.84 MB)
        # can't satisfy, clashing in the full-logits prefill path. A smaller chunk shrinks
        # the CB to fit; rows are independent so the logits are identical (just more
        # iterations). This is the test/PCC path only — production prefill uses
        # prefill_next_token (single-position lm_head, m_tiles=1).
        chunk = min(self._prefill_chunk, 64)
        pc = self._lm_head_prefill_pc(chunk // 32)
        # lm_head_weight is column-sharded across devices (dim=1 of [hidden, vocab]); each
        # device produces partial logits [1, 1, chunk, vocab/n_devices], concatenated on host.
        # Norm is applied per chunk (rows are independent) so no full [seq, HIDDEN] activation
        # stays resident in L1 at long context.
        blocks = []
        for s in range(0, seq_len, chunk):
            e = min(s + chunk, seq_len)
            real = e - s
            xb = ttnn.slice(x, [0, 0, s, 0], [1, 1, e, HIDDEN_SIZE])
            if real < chunk:
                xb = ttnn.pad(xb, [(0, 0), (0, 0), (0, chunk - real), (0, 0)], value=0.0)
            # Width-sharded final norm (32 cores) rather than the interleaved variant:
            # the interleaved layernorm needs ~1 MB of *contiguous* L1, which the
            # moe_compute path's persistent global semaphore fragments (it lands mid-L1,
            # capping the largest contiguous free block below what the op needs). The
            # sharded norm spreads across cores so it never needs that contiguous block.
            # Matches the decode path's final norm; returns L1-interleaved for the lm_head.
            xb = _rms_norm_sharded_decode(xb, self.final_norm_w, self.compute_kernel_config)
            lb = ttnn.linear(
                xb,
                self.lm_head_weight_prefill,
                program_config=pc,
                compute_kernel_config=self.lm_head_compute_kernel_config,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(xb)
            lb_host = ttnn.to_torch(lb, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
            ttnn.deallocate(lb)
            blocks.append(lb_host[0, 0, :real].to(torch.bfloat16))  # [real, vocab]
        ttnn.deallocate(x)
        return torch.cat(blocks, dim=0).unsqueeze(0)  # [1, seq_len, vocab]

    def _argmax_to_device(self, x_last: ttnn.Tensor) -> ttnn.Tensor:
        """
        On-device greedy argmax from one hidden state — trace-safe half.

        Runs rms_norm → lm_head → all_gather → argmax entirely on device and
        writes the result into the pre-allocated ``_decode_token_out`` buffer.
        Returns ``_decode_token_out`` so it can be deallocated by the caller
        when not running under a trace (trace replay re-uses the same buffer).

        Does NOT call ttnn.to_torch — host readback must happen outside the
        trace via ``_readback_argmax()``.
        """
        x_normed = _rms_norm_sharded_decode(x_last, self.final_norm_w, self.compute_kernel_config)

        in0_memcfg = ttnn.create_sharded_memory_config(
            (1, 1, 32, 4096),  # M=32 (padded), K=4096
            core_grid=ttnn.CoreGrid(y=1, x=self.num_banks),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        x_normed = ttnn.to_memory_config(x_normed, in0_memcfg)
        partial = ttnn.linear(
            x_normed,
            self.lm_head_weight,
            program_config=self.lm_head_program_config,
            compute_kernel_config=self.lm_head_compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.MemoryConfig(  # out: L1 width-sharded (matches DS)
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
            ),
        )

        ttnn.deallocate(x_normed)

        num_devices = self.mesh_device.get_num_devices()
        if num_devices > 1:
            full = ttnn.all_gather(
                partial,
                dim=3,
                num_links=2,  # P150x8 has 2 ethernet channels per device pair
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )  # per device: [1, 1, 1, vocab]
            ttnn.deallocate(partial)
        else:
            full = partial
        full = ttnn.typecast(full, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Multicore argmax requires ROW_MAJOR input (argmax_device_operation.cpp:153).
        full = ttnn.to_layout(full, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Write directly into the pre-allocated buffer so trace replay can overwrite
        # the same slot every step — saves one Copy/assign op and the L1 temp.
        ttnn.argmax(full, dim=-1, use_multicore=True, output_tensor=self._decode_token_out)
        ttnn.deallocate(full)
        return self._decode_token_out

    def _readback_argmax(self) -> int:
        """Read the argmax token id from ``_decode_token_out`` to host (not traced)."""
        idx_host = ttnn.to_torch(
            self._decode_token_out,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )
        return int(idx_host.flatten()[0].item())

    def _next_token_on_device(self, x_last: ttnn.Tensor) -> int:
        """Convenience wrapper used by prefill paths (not traced)."""
        self._argmax_to_device(x_last)
        return self._readback_argmax()

    # ── Public API ─────────────────────────────────────────────────────────

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run prefill and populate all KV caches for positions ``[0, seq_len)``.

        Returns host-side logits; used by PCC tests to compare against a torch
        reference. Inference callers should use ``prefill_next_token`` (on-device
        argmax) instead.

        Args:
            input_ids: [1, seq_len] long tensor on CPU
        Returns:
            logits: [1, seq_len, vocab_size] bfloat16 CPU tensor
        """
        seq_len = input_ids.shape[1]
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

        x = self._embed(input_ids)
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        return self._to_logits(x)

    def prefill_device(self, input_ids_tt: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """
        Device-only prefill for trace capture/execution.

        Identical to ``prefill`` but: takes a pre-uploaded device tensor for
        input_ids, returns a device tensor for logits (no host download), and
        does NOT deallocate the RoPE slices (so the trace can be re-executed
        without re-caching the RoPE tables).

        Args:
            input_ids_tt: uint32 device tensor of shape ``[1, seq_len]``,
                          replicated across the mesh.
            seq_len:      sequence length (passed explicitly so it's a Python
                          int known at trace-capture time).

        Returns:
            logits_tt: device tensor ``[1, 1, seq_len, vocab/n_devices]``
                       column-sharded across the mesh.
        """
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

        x = ttnn.embedding(
            input_ids_tt,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        # NOTE: do not deallocate cos_tt/sin_tt — they may alias the cached
        # RoPE buffer; deallocation would invalidate the buffer for trace replay.

        x = _rms_norm(x, self.final_norm_w, self.compute_kernel_config)
        # Prefill path: M = seq_len (> 1 tile) — interleaved weight + 1D mcast.
        logits_tt = ttnn.linear(
            x,
            self.lm_head_weight_prefill,
            program_config=self._lm_head_prefill_pc((seq_len + 31) // 32),
            compute_kernel_config=self.lm_head_compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        return logits_tt

    def prefill_device_last_token_logits(self, input_ids_tt: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """
        Device-only prefill that returns LM-head logits for the final position only.

        This still runs the full decoder stack and fills KV caches for all
        prefill positions, but slices ``x[:, :, seq_len - 1:seq_len, :]`` before
        the final norm + LM head. Generation only needs this last-position
        distribution, so this avoids the full ``seq_len × vocab`` LM-head matmul.
        """
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

        x = ttnn.embedding(
            input_ids_tt,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        if seq_len == 1:
            # Full-extent slice of a 1-row tensor aliases x; don't deallocate it (use directly).
            x_last = x
        else:
            x_last = ttnn.slice(x, [0, 0, seq_len - 1, 0], [1, 1, seq_len, HIDDEN_SIZE])
            ttnn.deallocate(x)

        x_last = _rms_norm(x_last, self.final_norm_w, self.compute_kernel_config)
        # Last-token-only path: M == 1 tile after the slice, so the DS weight + DS
        # program config are legal here and reuse the decode-tuned pattern.
        in0_memcfg = ttnn.create_sharded_memory_config(
            (1, 1, 32, HIDDEN_SIZE),
            core_grid=ttnn.CoreGrid(y=1, x=self.num_banks),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        x_last = ttnn.to_memory_config(x_last, in0_memcfg)
        logits_tt = ttnn.linear(
            x_last,
            self.lm_head_weight,
            program_config=self.lm_head_program_config,
            compute_kernel_config=self.lm_head_compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.MemoryConfig(  # DS output: L1 width-sharded
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
            ),
        )
        ttnn.deallocate(x_last)
        return logits_tt

    def _decode_upload_step_state(self, input_id: torch.Tensor, current_pos: int, cq_id: int = 0) -> None:
        """Update all pre-allocated decode input tensors in-place for this step.

        Updates: token id, current position scalar, and the single-position
        cos/sin RoPE buffers.  All three must be refreshed before either a
        direct kernel dispatch or a trace replay so the correct position's
        embeddings and KV-cache slot are used.

        ``cq_id`` selects the command queue for the host→device copies (0 for the
        single-queue path; 1 for the 2-CQ decode pipeline, where the writes overlap
        CQ0's trace replay — see decode_next_token_2cq).
        """
        input_id_host = ttnn.from_torch(
            input_id.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(input_id_host, self._decode_input_id_device, cq_id=cq_id)

        cur_pos_host = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(cur_pos_host, self._decode_cur_pos_device, cq_id=cq_id)
        self._last_decode_pos = current_pos

        # Update single-position RoPE buffers from the retained CPU table.
        # CPU indexing is free; copy_host_to_device_tensor reuses the pre-
        # allocated device buffer so no new device allocation occurs.
        assert self._cos_cpu is not None, "cache_rope_tables() must be called before decode"
        cos_pos = self._cos_cpu[:, :, current_pos : current_pos + 1, :].contiguous()
        sin_pos = self._sin_cpu[:, :, current_pos : current_pos + 1, :].contiguous()
        cos_host = ttnn.from_torch(
            cos_pos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        sin_host = ttnn.from_torch(
            sin_pos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(cos_host, self._cos_decode, cq_id=cq_id)
        ttnn.copy_host_to_device_tensor(sin_host, self._sin_decode, cq_id=cq_id)

    # ── Greedy generation entry points (on-device argmax) ──────────────────

    def prefill_next_token(self, input_ids: torch.Tensor) -> int:
        """
        Run prefill (filling all KV caches) and return the greedy next token id.

        Same as ``prefill`` but the argmax over the last position's logits runs
        on device — only a single uint32 crosses the PCIe boundary.
        """
        seq_len = input_ids.shape[1]
        x = self._embed(input_ids)
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Chunk-major prefill over the paged cache → scales past the single-pass limit.
        x = self._run_prefill_chunked(x, last_only=True)

        # Extract the last position's hidden state: [1, 1, 1, HIDDEN_SIZE].
        last_len = x.shape[-2]
        if last_len == 1:
            # Slicing [0:1] of a 1-row tensor is a full-extent slice that ALIASES x, so
            # deallocating x would free x_last's buffer. Use x directly. Hit when the final
            # prefill chunk is a single token (prompt_len ≡ 1 mod prefill_chunk).
            x_last = x
        else:
            x_last = ttnn.slice(x, [0, 0, last_len - 1, 0], [1, 1, last_len, HIDDEN_SIZE])
            ttnn.deallocate(x)
        token_id = self._next_token_on_device(x_last)
        ttnn.deallocate(x_last)
        return token_id

    def _decode_kernel(self, current_pos: int) -> None:
        """
        Single decode step using pre-allocated buffers (_decode_input_id_device,
        _decode_cur_pos_device, _cos_decode, _sin_decode).

        Token id, position and RoPE all come from those persistent buffers, and the
        KV write is tensor-indexed (paged_update_cache reads _decode_cur_pos_device),
        so this graph is replayable as a trace — see capture_decode_trace. The
        ``current_pos`` arg is now vestigial (only forward_decode's no-tensor fallback
        would use it); the buffers drive everything. The argmax result lands in
        _decode_token_out; host readback happens outside via _readback_argmax().
        """
        x = ttnn.embedding(
            self._decode_input_id_device,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [1, 1, 1, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_decode(
                x, self._cos_decode, self._sin_decode, kv_cache, current_pos, self._decode_cur_pos_device
            )

        self._argmax_to_device(x)
        ttnn.deallocate(x)

    def capture_decode_trace(self) -> None:
        """Capture the single-token decode step as a replayable trace.

        Call once after at least one eager decode step has run, so every op program
        is already compiled and in the program cache (capture must not compile). The
        captured graph reads token id / position / RoPE from the persistent decode
        buffers, so it replays correctly at any position once those buffers are
        refreshed by _decode_upload_step_state. Idempotent.
        """
        if self._decode_trace_id is not None:
            return
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_kernel(self._last_decode_pos)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self._decode_trace_id = trace_id

    def decode_next_token(self, input_id: torch.Tensor, current_pos: int) -> int:
        """Decode one token and return the greedy next token id (on-device argmax).

        Replays the captured trace if capture_decode_trace() has been called;
        otherwise runs the step eagerly (which also compiles the kernels so a
        subsequent capture is a cache hit).
        """
        self._decode_upload_step_state(input_id, current_pos)
        if self._decode_trace_id is not None:
            ttnn.execute_trace(self.mesh_device, self._decode_trace_id, cq_id=0, blocking=False)
        else:
            self._decode_kernel(current_pos)
        return self._readback_argmax()

    def decode_logits(self, input_id: torch.Tensor, current_pos: int) -> torch.Tensor:
        """Run one decode step and return full-vocabulary logits to host.

        Like decode_next_token but returns a [vocab] float32 CPU tensor instead of the
        greedy token id. Used by PCC tests to compare against prefill logits.
        """
        self._decode_upload_step_state(input_id, current_pos)
        x = ttnn.embedding(
            self._decode_input_id_device,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [1, 1, 1, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_decode(
                x, self._cos_decode, self._sin_decode, kv_cache, current_pos, self._decode_cur_pos_device
            )
        x_normed = _rms_norm_sharded_decode(x, self.final_norm_w, self.compute_kernel_config)
        ttnn.deallocate(x)
        in0_memcfg = ttnn.create_sharded_memory_config(
            (1, 1, 32, HIDDEN_SIZE),
            core_grid=ttnn.CoreGrid(y=1, x=self.num_banks),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        x_normed = ttnn.to_memory_config(x_normed, in0_memcfg)
        # Column-sharded across devices: each produces [1, 1, 32, vocab/n_devices]
        # (DRAM-sharded matmul requires a width-sharded L1 output).
        partial = ttnn.linear(
            x_normed,
            self.lm_head_weight,
            program_config=self.lm_head_program_config,
            compute_kernel_config=self.lm_head_compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
            ),
        )
        ttnn.deallocate(x_normed)
        # Convert sharded → interleaved, then ConcatMeshToTensor(dim=3) reassembles the
        # full vocab from the per-device column shards on host — same as the prefill path
        # (_to_logits). Do NOT all_gather first: that replicates the full vocab on every
        # device, so the host concat would build n_devices × vocab.
        partial = ttnn.to_memory_config(partial, ttnn.L1_MEMORY_CONFIG)
        logits = ttnn.to_torch(partial, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
        ttnn.deallocate(partial)
        return logits[0, 0, 0, :].to(torch.float32)  # [vocab]

    # ── 2-CQ decode pipeline ────────────────────────────────────────────────

    def begin_decode_2cq(self) -> None:
        """Arm the 2-CQ decode pipeline by recording the initial CQ0 event.

        Call once after capture_decode_trace() and before the first
        decode_next_token_2cq() step. The event seeds the cross-queue handshake so
        the first CQ1 input write has a prior CQ0 completion to wait on.
        """
        assert self._decode_trace_id is not None, "capture_decode_trace() must run before begin_decode_2cq()"
        self._decode_2cq_op_event = ttnn.record_event(self.mesh_device, 0)

    def decode_next_token_2cq(self, input_id: torch.Tensor, current_pos: int) -> int:
        """One decode step with the input upload on CQ1 overlapping the trace on CQ0.

        Same result as decode_next_token, but dispatch is split across two command
        queues: CQ1 writes the per-step token/position/RoPE buffers while CQ0 owns
        the captured compute trace, fenced by events so neither queue races the
        other on the shared input buffers. The argmax token is read back on CQ0
        (token feedback is inherently serial for greedy decode, so only the input
        H2D — not compute — overlaps). Requires capture_decode_trace() +
        begin_decode_2cq().
        """
        assert self._decode_trace_id is not None, "decode_next_token_2cq requires a captured trace"
        assert self._decode_2cq_op_event is not None, "call begin_decode_2cq() before decode_next_token_2cq()"
        # CQ1 must not overwrite the input buffers until CQ0's previous trace finished reading them.
        ttnn.wait_for_event(1, self._decode_2cq_op_event)
        self._decode_upload_step_state(input_id, current_pos, cq_id=1)
        write_event = ttnn.record_event(self.mesh_device, 1)
        # CQ0 must not start the trace until CQ1's writes have landed.
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(self.mesh_device, self._decode_trace_id, cq_id=0, blocking=False)
        # Mark trace completion on CQ0 so the next step's CQ1 write can proceed.
        self._decode_2cq_op_event = ttnn.record_event(self.mesh_device, 0)
        return self._readback_argmax()

    # ── Embedding-input entry points (multimodal) ──────────────────────────

    def _run_prefill_chunked(self, x: ttnn.Tensor, last_only: bool) -> ttnn.Tensor:
        """Run all decoder layers over the prompt in bounded chunks (chunk-major),
        filling the paged KV caches. Each chunk passes through all 36 layers; the
        only cross-chunk dependency is the KV cache (attention), which earlier chunks
        fill first — so this is exact, with activations bounded to one chunk.

        Returns the full ``[1,1,seq,HIDDEN]`` hidden when ``last_only=False``, or just
        the final chunk (whose last row is the overall last token) when ``last_only=True``.
        A single chunk (``seq ≤ chunk``) uses ``chunk_start_idx=0`` — identical to the
        pre-chunking single-pass path.
        """
        seq_len = x.shape[-2]
        chunk = self._prefill_chunk
        if seq_len <= chunk:
            cos_tt, sin_tt = self._rope_slice(0, seq_len)
            for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
                x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache, chunk_start_idx=0)
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
            return x

        outs: list = []
        last = None
        for s in range(0, seq_len, chunk):
            e = min(s + chunk, seq_len)
            real = e - s
            xc = ttnn.slice(x, [0, 0, s, 0], [1, 1, e, HIDDEN_SIZE])
            cos_c, sin_c = self._rope_slice(s, e)
            if real < chunk:
                # Ragged final chunk: pad up to a full chunk so every projection matmul runs at
                # the fixed m_tiles = chunk//32 program config (a ragged m_tiles picks a per_core_M/
                # out_subblock layout whose circular buffers clash with L1). Padded rows are sliced
                # off below; their KV writes land in [seq_len, padded) — covered by the chunk-rounded
                # cache and overwritten by causal decode before they are ever read.
                pad = chunk - real
                xc = ttnn.pad(xc, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
                cos_c = ttnn.pad(cos_c, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
                sin_c = ttnn.pad(sin_c, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
            for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
                xc = layer.forward_with_cache(xc, cos_c, sin_c, kv_cache, chunk_start_idx=s, valid_tokens=real)
            ttnn.deallocate(cos_c)
            ttnn.deallocate(sin_c)
            if real < chunk:
                xc = ttnn.slice(xc, [0, 0, 0, 0], [1, 1, real, HIDDEN_SIZE])
            if last_only:
                if last is not None:
                    ttnn.deallocate(last)
                last = xc
            else:
                # Retain each chunk output in DRAM, not L1. Accumulating all chunks'
                # [chunk, HIDDEN] outputs in L1 (last_only=False) starves later chunks'
                # projection matmuls of L1 headroom and clashes their circular buffers at
                # long context (assert.hpp:104). last_only=True keeps only one chunk, so the
                # e2e/decode path is unaffected.
                if xc.memory_config().buffer_type != ttnn.BufferType.DRAM:
                    xc_dram = ttnn.to_memory_config(xc, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(xc)
                    xc = xc_dram
                outs.append(xc)
        ttnn.deallocate(x)
        if last_only:
            return last
        full = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for o in outs:
            ttnn.deallocate(o)
        return full

    def prefill_from_embeds(self, inputs_embeds: ttnn.Tensor) -> torch.Tensor:
        """
        Prefill from a caller-built embedding sequence (skip ``embed_tokens``).

        Args:
            inputs_embeds: ttnn [1, 1, seq_len, HIDDEN_SIZE], replicated on mesh.
                           Typically built by ``embed_tokens`` for text positions
                           and the multi-modal projector for image positions,
                           spliced together via slice/concat on device.
        Returns:
            logits: [1, seq_len, vocab_size] bf16 CPU tensor.
        """
        x = self._run_prefill_chunked(inputs_embeds, last_only=False)
        return self._to_logits(x)

    def prefill_from_embeds_next_token(self, inputs_embeds: ttnn.Tensor) -> int:
        """
        Same as ``prefill_from_embeds`` but returns the greedy next-token id with
        on-device argmax — only a single uint32 crosses the PCIe boundary.
        """
        x = self._run_prefill_chunked(inputs_embeds, last_only=True)
        last_len = x.shape[-2]
        if last_len == 1:
            # x is already the single last token. Slicing [0:1] of a 1-row tensor is a
            # full-extent slice that ALIASES x, so deallocating x would free x_last's buffer
            # (crash in the argmax path). Use x directly. Hit when the final prefill chunk is
            # a single token (prompt_len ≡ 1 mod prefill_chunk).
            x_last = x
        else:
            x_last = ttnn.slice(x, [0, 0, last_len - 1, 0], [1, 1, last_len, HIDDEN_SIZE])
            ttnn.deallocate(x)
        token_id = self._next_token_on_device(x_last)
        ttnn.deallocate(x_last)
        return token_id
