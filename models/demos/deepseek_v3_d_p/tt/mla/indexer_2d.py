# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DSA lightning indexer on the 2D tensor-parallel layout of the Hy4-preview SC1 unit (TP32 / DP1, #65887).

``TtIndexer`` (indexer.py) replicates ``wq_b`` on every chip and shards ``wk`` / ``weights_proj`` over TP only, so on
the 8x4 mesh each indexer weight sits on all 8 SP chips of a TP column. ``TtIndexer2D`` shards every indexer weight
ONCE over all 32 chips, as ``ttMLA2DTP`` (mla_2d.py, #65885) does the attention projections. It is the indexer
``ttMLA2DTP`` builds on a ``full`` indexer layer; ``shared`` layers still take the top-k of the last full layer.

Input / output, as ``ttMLA2DTP`` holds them: ``hidden_states`` is the block residual ``[S / sp, hidden / tp]``
all-gathered over SP, ``[1, 1, S, hidden / tp]`` (ttMLA2DTP gathers it once for every hidden-side matmul); ``qr`` is
the q_a latent after q_a_layernorm, the whole ``[1, 1, S, q_lora]`` row on every chip. The output is ttMLA's top-k,
``[1, 1, S / (sp * tp), index_topk]`` uint32 per chip, the query rows SP-major (chip (s, t) holds block ``s * tp + t``),
which is what ttMLA's sparse core (heads -> sequence all-to-alls, all 64 heads per chip) takes. The index-key cache is
the context-parallel one of the dev layout: 128-token blocks block-cyclic over all 32 chips (#65886), written by each
chip's 1 / tp window of its SP row and scored by the fused full-mesh ring.

Why not shard the index head dim (128) over TP: the score is ``sum_h w[t, h] * relu(q[t, h] . k[s])``. The ReLU needs
each head's complete 128-wide dot product, so a split head dim would need an all-reduce of the per-head partial
logits ``[S, 32 heads, T]`` before the ReLU, far more than moving q. ``wk`` and ``weights_proj`` contract over the
hidden, which the residual splits over TP (axis 1), so TP must carry their K; and ``weights_proj``'s N (32 heads) is
one tile, so it cannot split 8 ways either. Whole heads per chip instead:

  =============  ===================  ==============================================  ==============
  weight         [K, N]               split                                           per chip
  =============  ===================  ==============================================  ==============
  wq_b           [2048, 32 * 128]     N: one whole head per chip (head t * sp + s)    [2048, 128]
  wk             [6144, 128]          K (hidden) over TP x SP, as ttMLA2DTP's kv_a    [192, 128]
  weights_proj   [6144, 32]           K (hidden) over TP x SP                         [192, 32]
  k_norm (+b)    [128]                replicated (the only replicated indexer tensors)
  =============  ===================  ==============================================  ==============

  Chip (s, t) owns index head ``t * sp + s`` (in general heads ``(t * sp + s) * hpc .. + hpc``, ``hpc`` heads per
  chip): after the two all-to-alls below the core sees chip (s, t)'s heads at position ``t * sp + s`` (TP column t
  holds chips (0, t) .. (sp - 1, t), ``ttMLA2DTP.core_head_order``), so the 32 heads arrive in checkpoint order and
  ``weights_proj`` keeps its column order. Hidden index of ``wk`` / ``weights_proj`` row k on chip (s, t):
  ``t * hidden / tp + s * hidden / (sp * tp) + k``, the SP slice of the residual's TP slice (``mesh_partition``).

Per forward (chunk of S tokens; Hy4 at S = 4096: q 1 MiB, k / weights partials 1 MiB / 256 KiB per chip):

  q:       qr [S, 2048] @ wq_b (one head)       -> [1, 1, S, 128]
           -> all-to-all(SP): heads in, sequence out  -> [1, 8, S / 8, 128]
           -> all-to-all(TP): heads in, sequence out  -> [1, 32, S / 32, 128] -> RoPE -> Hadamard -> bf8
  k:       x[:, K_ts] @ wk -> reduce-scatter(SP, sequence) -> [S / 8, 128], partial over TP
           -> reduce-scatter(TP, cols) + all-gather(TP) -> [S / 8, 128] -> LayerNorm -> RoPE -> Hadamard -> the
              block-cyclic index-key cache (each TP chip writes its 1 / tp window)
  weights: x[:, K_ts] @ weights_proj -> reduce-scatter(SP, sequence) -> reduce-scatter(TP, sequence) -> [S / 32, 32]
  score:   the fused full-mesh ring indexer + local top-k, unchanged (``TtIndexerBase.score``); no top-k movement.

Every reduction accumulates in fp32 (``matmul_reduce_scatter_prefill``; the TP reduce-scatters use the fp32 compute
kernel config).

Weight cache: ``layer_{i}.idx2d_{layout}.{wq_b,wk,weights_proj,k_norm,k_norm_bias}`` (``layout_tag``, e.g.
``idx2d_8x4_sp8ax0_tp4ax1``) with a layout stamp ``layer_{i}.idx2d_{layout}.layout.stamp`` (utils/sharded_cache.py:
module, ``CACHE_FORMAT``, mesh, axes, dtypes, RoPE pairing and index-key frame). TtIndexer's TP-only files
(``layer_{i}.mla.indexer_*``) never match these names, and shards without a matching stamp refuse to load. Only a
full build from real weights stamps; a cache-only load needs every shard under a matching stamp (it never converts
placeholders); a ``kv_only`` build reads / fills the cache only under a stamp a full build wrote, otherwise it
converts uncached (as ``ttMLA2DTP``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
from models.demos.deepseek_v3_d_p.utils.sharded_cache import (
    LayoutCacheGuard,
    as_tensor_file,
    layout_descriptor,
    layout_tag,
)
from models.demos.deepseek_v3_d_p.utils.weight_cache_dump import weight_cache_dump_mode


class TtIndexer2D(TtIndexer):
    accepts_tp_local_qr = False  # takes ttMLA2DTP's full q_a row

    CACHE_PREFIX = "idx2d"  # + the layout tag (cache_name); never TtIndexer's TP-only layer_{i}.mla.indexer_* files
    CACHE_FORMAT = 1  # bump when the shards change without an axis moving
    # The frame the cached wq_b / wk / k_norm rows are in: the device ropes the FIRST 64 index dims with interleaved
    # (2i, 2i + 1) pairs (rotary_embedding_indexed); Hy4's reader moves the checkpoint's rope block there
    # (hy4_checkpoint.indexer_to_device_frame). Recorded in the stamp so shards cut for another frame refuse to load.
    ROPE_PAIRING = "interleaved"
    INDEX_K_FRAME = "rope_first"
    SHORT_NAMES = {
        "indexer.wq_b": "wq_b",
        "indexer.wk": "wk",
        "indexer.weights_proj": "weights_proj",
        "indexer.k_norm": "k_norm",
        "indexer.k_norm_bias": "k_norm_bias",
    }
    VECTOR_WEIGHTS = ("k_norm", "k_norm_bias")

    # ------------------------------------------------------------------------------------------- host math
    @staticmethod
    def heads_per_chip(config, sp: int, tp: int) -> int:
        return config.index_n_heads // (sp * tp)

    @staticmethod
    def heads_of_chip(num_heads: int, sp: int, tp: int, s: int, t: int) -> list[int]:
        """Index heads chip (s on the SP axis, t on the TP axis) projects: position ``t * sp + s`` of the order the two
        all-to-alls (SP, then TP) gather the heads in, so the gathered heads are in checkpoint order."""
        hpc = num_heads // (sp * tp)
        p = t * sp + s
        return list(range(p * hpc, (p + 1) * hpc))

    @staticmethod
    def check_dims(config, sp: int, tp: int) -> None:
        H, D, hidden = config.index_n_heads, config.index_head_dim, config.hidden_size
        T = ttnn.TILE_SIZE
        checks = {
            "index heads over SP x TP": H % (sp * tp) == 0,
            "heads per chip, whole tiles": (H // (sp * tp)) * D % T == 0 if H % (sp * tp) == 0 else False,
            "hidden over TP x SP, whole tiles": hidden % (sp * tp * T) == 0,
            "index head dim over TP, whole tiles (the k reduce-scatter)": D % (tp * T) == 0,
            "index heads, whole tiles (weights_proj's N)": H % T == 0,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"2D TP indexer (SP {sp} x TP {tp}) does not tile: {bad}")

    @classmethod
    def has_weights(cls, idx_host) -> bool:
        return bool(idx_host) and all(name in idx_host for name in cls.SHORT_NAMES)

    @classmethod
    def host_weights_2d(cls, idx_host, config, sp: int, tp: int) -> dict:
        """Host tensors for the 2D mesh mappers, ``short name -> (tensor, sp_dim, tp_dim)`` (None = replicated over that
        axis). ``idx_host``: ``TtIndexer.extract_host_weights`` (HF ``[out, in]`` weights in the device frame); falsy ->
        placeholders of the right shapes (cache-only load). Pure torch: the CPU tests check the layout the device
        loads (``chip_shard``)."""
        H, D, Q, hidden = config.index_n_heads, config.index_head_dim, config.q_lora_rank, config.hidden_size
        hpc = H // (sp * tp)
        k_chip = hidden // (sp * tp)
        if cls.has_weights(idx_host):
            wq_b = idx_host["indexer.wq_b"].transpose(-2, -1)  # [Q, H * D]
            wk = idx_host["indexer.wk"].transpose(-2, -1)  # [hidden, D]
            wproj = idx_host["indexer.weights_proj"].transpose(-2, -1)  # [hidden, H]
            knorm, knorm_b = idx_host["indexer.k_norm"], idx_host["indexer.k_norm_bias"]
        else:
            e = torch.empty
            wq_b, wk, wproj, knorm, knorm_b = e(Q, H * D), e(hidden, D), e(hidden, H), e(D), e(D)
        # N of wq_b in head blocks: head (t * sp + s) * hpc + j -> [t, s, Q, hpc * D].
        wq_b = wq_b.reshape(Q, tp, sp, hpc * D).permute(1, 2, 0, 3)
        return {
            "wq_b": (wq_b, 1, 0),
            # K over both axes: hidden index t * (hidden / tp) + s * (hidden / (sp * tp)) + k.
            "wk": (wk.reshape(tp, sp, k_chip, D), 1, 0),
            "weights_proj": (wproj.reshape(tp, sp, k_chip, H), 1, 0),
            "k_norm": (knorm, None, None),
            "k_norm_bias": (knorm_b, None, None),
        }

    @staticmethod
    def chip_shard(t, sp_dim, tp_dim, sp: int, tp: int, s: int, c: int) -> torch.Tensor:
        """What ``ShardTensor2dMesh`` puts on chip (s, c) for a ``host_weights_2d`` entry (CPU tests)."""
        if sp_dim is not None:
            t = t.chunk(sp, dim=sp_dim)[s]
        if tp_dim is not None:
            t = t.chunk(tp, dim=tp_dim)[c]
        return t

    # ------------------------------------------------------------------------------------------- caches
    @classmethod
    def _dtype(cls, short: str):
        return cls.WEIGHT_DTYPES[next(k for k, v in cls.SHORT_NAMES.items() if v == short)]

    @classmethod
    def cache_name(cls, mesh_shape, sp_axis: int, tp_axis: int) -> str:
        """``idx2d_{rows}x{cols}_sp{n}ax{a}_tp{n}ax{a}``: the shards' file prefix after ``layer_{i}.``."""
        return f"{cls.CACHE_PREFIX}_{layout_tag(mesh_shape, {'sp': sp_axis, 'tp': tp_axis})}"

    @classmethod
    def cache_guard(cls, cache_path, layer_idx: int, mesh_shape, sp_axis: int, tp_axis: int) -> LayoutCacheGuard:
        name = f"layer_{layer_idx}.{cls.cache_name(mesh_shape, sp_axis, tp_axis)}"
        descriptor = layout_descriptor(
            "TtIndexer2D",
            cls.CACHE_FORMAT,
            mesh_shape,
            {"sp": sp_axis, "tp": tp_axis},
            {short: cls._dtype(short).name for short in cls.SHORT_NAMES.values()},
            extra={"rope_pairing": cls.ROPE_PAIRING, "index_k_frame": cls.INDEX_K_FRAME},
        )
        return LayoutCacheGuard(cache_path, name, f"{name}.*.tensorbin", descriptor)

    @classmethod
    def cache_files(cls, cache_path, layer_idx: int, mesh_shape, sp_axis: int, tp_axis: int) -> list[Path]:
        prefix = Path(cache_path) / f"layer_{layer_idx}.{cls.cache_name(mesh_shape, sp_axis, tp_axis)}"
        return [
            as_tensor_file(f"{prefix}.{short}", cls._dtype(short), ttnn.TILE_LAYOUT)
            for short in cls.SHORT_NAMES.values()
        ]

    @classmethod
    def check_cache_complete(
        cls, cache_path, cache_name_prefix: str, *, mesh_shape=None, sp_axis=None, tp_axis=None
    ) -> bool:
        """``cache_name_prefix``: ttMLA's ``.../layer_{i}.mla`` (as TtIndexer takes it). Complete = every shard at its
        dtype plus a matching layout stamp. The shards are named by the mesh layout, so it is required."""
        if mesh_shape is None or sp_axis is None or tp_axis is None:
            raise ValueError(
                "TtIndexer2D.check_cache_complete needs mesh_shape, sp_axis and tp_axis: they name the shards"
            )
        if cache_path is None:
            return False
        name = Path(str(cache_name_prefix)).name
        if not (name.startswith("layer_") and name.endswith(".mla")):
            raise ValueError(f"expected ttMLA's '.../layer_{{i}}.mla' prefix, got {cache_name_prefix!r}")
        layer_idx = int(name[len("layer_") : -len(".mla")])
        if not all(f.is_file() for f in cls.cache_files(cache_path, layer_idx, mesh_shape, sp_axis, tp_axis)):
            logger.debug(f"TTNN 2D indexer cache missing shards: layer {layer_idx}")
            return False
        return cls.cache_guard(cache_path, layer_idx, mesh_shape, sp_axis, tp_axis).stamped()

    @classmethod
    def build_ttnn_cache(
        cls, idx_host, cache_path, mesh_device, config, layer_idx, sp_axis: int, tp_axis: int, kv_only: bool = False
    ) -> None:
        cls._convert_and_cache_weights(
            idx_host,
            mesh_device,
            config,
            layer_idx,
            sp_axis,
            tp_axis,
            cache_path=cache_path,
            device=None,
            kv_only=kv_only,
        )

    @classmethod
    def _convert_and_cache_weights(
        cls,
        idx_host,
        mesh_device,
        config,
        layer_idx,
        sp_axis: int,
        tp_axis: int,
        cache_path=None,
        device=None,
        kv_only: bool = False,
    ):
        """Indexer weights -> device (or cache only, ``device=None``), sharded once over the mesh. Returns
        ``short name -> device tensor`` (the 2D weights reshaped to rank 2), or None for a cache-only build."""
        mesh_shape = tuple(mesh_device.shape)
        sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
        cls.check_dims(config, sp, tp)
        host = cls.host_weights_2d(idx_host, config, sp, tp)
        real = cls.has_weights(idx_host)
        guard = cls.cache_guard(cache_path, layer_idx, mesh_shape, sp_axis, tp_axis)
        guard.check()  # shards from another layout (or unstamped) refuse to load
        # The stamp vouches for the whole indexer (check_cache_complete trusts it), so only a full build from real
        # weights writes it; a kv_only build uses the cache only under a stamp a full build wrote (ttMLA2DTP's rule).
        use_cache = cache_path is not None and (not kv_only or guard.stamped())
        if device is None and not use_cache:
            logger.info(f"layer {layer_idx}: kv_only 2D indexer cache build on an unstamped cache writes nothing")
            return None
        prefix = f"layer_{layer_idx}.{cls.cache_name(mesh_shape, sp_axis, tp_axis)}"
        todo = {}
        for short, (t, sp_dim, tp_dim) in host.items():
            cache_file = str(Path(cache_path) / f"{prefix}.{short}") if use_cache else None
            todo[short] = (t, sp_dim, tp_dim, cls._dtype(short), cache_file)
        if not real:
            # Placeholders: as_tensor would dump torch.empty on a miss, so every shard must already be cached.
            guard.require_cached([as_tensor_file(f, dt, ttnn.TILE_LAYOUT) for (_, _, _, dt, f) in todo.values() if f])
        mem = ttnn.DRAM_MEMORY_CONFIG if device is not None else None
        result = {}
        for short, (t, sp_dim, tp_dim, dtype, cache_file) in todo.items():
            if sp_dim is None and tp_dim is None:
                mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            else:
                dims = [None, None]
                dims[sp_axis], dims[tp_axis] = sp_dim, tp_dim
                mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims)
            result[short] = ttnn.as_tensor(
                t.contiguous().to(torch.bfloat16),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
                cache_file_name=cache_file,
                cache_dump_mode=weight_cache_dump_mode(),
            )
        if real and not kv_only and cache_path is not None:
            guard.write()
        if device is None:
            return None
        for short in ("wq_b", "wk", "weights_proj"):  # [1, 1, K, N] shards -> the rank-2 weights the matmuls take
            w = result[short]
            result[short] = ttnn.reshape(w, (w.shape[-2], w.shape[-1]))
        return result

    # ------------------------------------------------------------------------------------------- device
    def __init__(self, idx_host, *, kv_only: bool = False, **kwargs):
        self.kv_only = kv_only
        mesh_device, sp_axis, tp_axis = kwargs["mesh_device"], kwargs["sp_axis"], kwargs["tp_axis"]
        sp, tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
        self.check_dims(kwargs["config"], sp, tp)
        self.heads_per_chip_2d = self.heads_per_chip(kwargs["config"], sp, tp)
        super().__init__(idx_host, **kwargs)
        logger.info(
            f"Indexer layer {self.layer_idx}: 2D TP, {self.heads_per_chip_2d} index head(s) per chip, wk / "
            f"weights_proj K over SP x TP (SP axis {self.sp_axis}, TP axis {self.tp_axis})"
        )

    def _upload_weights(self, idx_host):
        w = self._convert_and_cache_weights(
            idx_host,
            self.mesh_device,
            self.config,
            self.layer_idx,
            self.sp_axis,
            self.tp_axis,
            cache_path=self.weight_cache_path,
            device=self.mesh_device,
            kv_only=self.kv_only,
        )
        self._idx_wq_b = w["wq_b"]
        self._idx_wk = w["wk"]
        wproj = w["weights_proj"]
        self._idx_wproj = ttnn.multiply(
            wproj, self.index_args.index_n_heads**-0.5 * self.index_args.index_head_dim**-0.5
        )
        ttnn.deallocate(wproj)
        self._idx_knorm_w = w["k_norm"]
        self._idx_knorm_b = w["k_norm_bias"]

    def _mm_rs_sp(self, x, w):
        """x @ w with the partials summed (fp32) over SP and the sequence scattered over SP."""
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            x,
            w,
            cluster_axis=self.sp_axis,
            scatter_dim=-2,
            topology=self.sp_ccl_topology,
            num_links=self.ccl_num_links,
        )

    def _hidden_slice(self, hidden_states):
        """The gathered ``[1, 1, S, hidden / tp]`` -> this chip's SP slice of it, ``[1, 1, S, hidden / (sp * tp)]``."""
        assert (
            hidden_states.shape[2] == self.active_seq_len
        ), f"2D indexer takes the SP-gathered chunk ({self.active_seq_len} rows), got {tuple(hidden_states.shape)}"
        return ttnn.mesh_partition(hidden_states, dim=3, cluster_axis=self.sp_axis)

    def _index_k_proj(self, hidden_states, seq_len):
        x = self._hidden_slice(hidden_states)
        k = self._mm_rs_sp(x, self._idx_wk)  # [S / sp, D], partial over TP
        ttnn.deallocate(x)
        cols = ttnn.experimental.reduce_scatter_minimal_async(
            k,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.tp_axis,
            compute_kernel_config=self.hifi4_fp32_compute_kernel_config,
        )  # [S / sp, D / tp], reduced
        ttnn.deallocate(k)
        out = self._tp_all_gather_k(cols)  # [S / sp, D] on every chip of the TP row
        ttnn.deallocate(cols)
        return out

    def _tp_all_gather_k(self, t):
        """TtIndexer's k all-gather, on ttMLA's shared high-bandwidth semaphores: without them each gather program parks
        two of its own in L1_SMALL, and the 2D layer's all-to-alls already take most of the prefill's 1536 B."""
        assert tuple(t.shape) == (1, 1, self.active_seq_len_local, self.index_args.index_head_dim // self.tp_factor)
        self._k_all_gather_output = self._get_high_bw_all_gather_buffer(
            name="indexer_k_all_reduce",
            shape=[1, 1, self.active_seq_len_local, self.index_args.index_head_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=t.device(),
        )
        ready, data_valid = self.tt_ccl.get_mla_high_bw_gather_semaphores()
        return ttnn.experimental.high_bw_all_gather(
            t,
            dim=3,
            output_tensor=self._k_all_gather_output,
            num_links=self.ccl_num_links,
            cluster_axis=self.tp_axis,
            ready_semaphore=ready,
            data_valid_semaphore=data_valid,
        )

    def _all_to_all(self, x, axis: int):
        """Gather the heads (dim 1) and split the sequence (dim 2) over ``axis``."""
        return ttnn.experimental.all_to_all_async_generic(
            x,
            in_dim=1,
            out_dim=2,
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=axis,
        )

    def _index_q_heads(self, qr, sq_local, qr_tp_local):
        assert not qr_tp_local, "the 2D indexer projects the whole q_a row (its heads are sharded, not its rows)"
        D, hpc = self.index_args.index_head_dim, self.heads_per_chip_2d
        q = ttnn.linear(
            qr,
            self._idx_wq_b,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b if getattr(self, "_q_hadamard_folded", False) else ttnn.bfloat16,
        )  # [1, 1, S, hpc * D]: this chip's heads, every token
        # Moved as `parts` pieces of `width` per head: at Hy4's dims [1, 2, S, 64] per chip, the geometry of ttMLA2DTP's
        # q_rope all-to-alls, so these hit the same cached programs. Every all_to_all_async_generic program parks two
        # semaphores of its own in L1_SMALL, and a 2D full layer's programs already fill the prefill's 1536 B.
        width = self._q_move_width()
        parts = D // width
        if hpc * parts > 1:
            heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                q, num_heads=hpc * parts, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(q)
            q = heads
        assert tuple(q.shape) == (1, hpc * parts, self.active_seq_len, width), tuple(q.shape)
        rows = self._all_to_all(q, self.sp_axis)  # [1, hpc * parts * sp, S / sp, width]
        ttnn.deallocate(q)
        out = self._all_to_all(rows, self.tp_axis)  # [1, H * parts, S / (sp * tp), width], checkpoint head order
        ttnn.deallocate(rows)
        if parts == 1:
            return out
        # Head h's pieces sit at h * parts .. h * parts + parts - 1: put them back side by side.
        H, R = self.index_args.index_n_heads, out.shape[2]
        pieces = ttnn.reshape(out, (H, parts, R, width))
        joined = ttnn.permute(pieces, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [H, R, parts, width]
        ttnn.deallocate(out)
        result = ttnn.reshape(joined, (1, H, R, D))
        return result

    def _q_move_width(self) -> int:
        """Width of the pieces q moves in: the MLA rope head width when it splits the index head (Hy4: 64 of 128)."""
        width = int(getattr(self.config, "qk_rope_head_dim", 0) or 0)
        D = self.index_args.index_head_dim
        return width if width and D % width == 0 and width % ttnn.TILE_SIZE == 0 else D

    def _index_head_weights(self, hidden_states, seq_len):
        x = self._hidden_slice(hidden_states)
        w = self._mm_rs_sp(x, self._idx_wproj)  # [S / sp, H], partial over TP
        ttnn.deallocate(x)
        out = self._tp_reduce_scatter_sequence(w)  # [S / (sp * tp), H], SP-major rows
        ttnn.deallocate(w)
        return out
