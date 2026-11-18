# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""2D tensor-parallel MLA projections for the Hy4-preview SC1 layout (TP32 / DP1, #65885).

ttMLA shards its projections over ONE mesh axis (TP) and replicates them over the other (SP, which carries the
sequence). ``ttMLA2DTP`` shards every projection over BOTH axes, once, with nothing replicated.

Residual contract: the prefill block's, unchanged: ``[1, 1, S / sp, hidden / tp]`` per chip, the sequence over the SP
axis and the hidden over the TP axis. On SC1 (#65884: hidden over its 4 rows, heads over its 8 columns) that is the
physical 8x4 torus with SP 8 on axis 0 and TP 4 on axis 1, ``[S / 8, 6144 / 4]`` per chip (no logical 4x8 view). The
2D shared expert / dense FFN (``TtMlp2D``, #65890) and iHC (#65891, ``MeshPlan(hidden_axis=1)``) use the same, so the
sublayers hand the residual to each other with no reshuffle.

Roles of the two axes:

  * TP (4): the hidden of every hidden-side contraction (q_a, kv_a, gate: K over TP; o_proj: N over TP, as the
    residual), and q_b's K (q_lora);
  * SP (8): the residual's sequence; the heads, 8 per chip, of q_b's N, the gate's N and o_proj's K (``8 heads per
    chip``, #65884); q_a's N; the sequence inside the attention core (ttMLA's SP axis).

Per-weight sharding (``sp`` = 8, ``tp`` = 4; per-chip shapes for Hy4):

  ============  ==========================  =============================================  =====================
  weight        [K, N]                      split                                          per chip
  ============  ==========================  =============================================  =====================
  q_a_proj      [6144, 2048]                K (hidden) over TP, N over SP                  [1536, 256]
  q_b_proj      [2048, 64 * 256]            K (q_lora) over TP, N (8-head blocks) over SP  [512, 8 heads * 256]
  kv_a_proj     [6144, 576]                 K over TP x SP (576 is not 8 tiles)            [192, 576]
  wkv_b1        [64, 192, 512] per head     2 heads per chip                               [2, 192, 512]
  wkv_b2        [64, 512, 256] per head     2 heads per chip                               [2, 512, 256]
  g_proj        [6144, 64 * 256]            K (hidden) over TP, N (8-head blocks) over SP  [1536, 8 heads * 256]
  o_proj        [64 * 256, 6144]            K (8-head blocks) over SP, N (hidden) over TP  [8 heads * 256, 1536]
  ============  ==========================  =============================================  =====================

  Chip (s, t) owns heads ``8 s + 2 t`` and ``8 s + 2 t + 1`` in the per-head ops (wkv_b1, wkv_b2, the gate multiply):
  q_b's and the gate's 8-head block s, reduce-scattered over TP, chunk t. q_b, wkv_b, the gate and o_proj keep the
  checkpoint's head order; only the sink is stored in the order the attention core sees the 64 heads after its two
  all-to-alls (``core_head_order``: TP column t holds the heads of chips (0, t) .. (sp - 1, t)). Only the two norms
  (2048 + 512) and the 64 sinks are replicated.

Collectives (every reduce accumulates in fp32 inside ``matmul_reduce_scatter_prefill``):

  in:    all-gather(SP, sequence): x [S, hidden / tp]
  q_a:   x @ Wqa -> reduce-scatter(TP, N) -> all-gather(TP) -> all-gather(SP) -> q_a_layernorm on the full row
  q_b:   q_lat[:, K_t] @ Wqb -> reduce-scatter(TP, N): this chip's 2 heads, every token -> wkv_b1 (2 heads)
         -> all-to-all(SP) + all-to-all(TP), heads -> sequence: [64 heads, S / 32] -> RoPE -> the sparse core
  kv_a:  x[:, K_ts] @ Wkva -> reduce-scatter(SP, sequence) -> ttMLA's KV stem (TP gather + fp32 reduce)
         -> kv_a_layernorm, RoPE, KV cache write (ttMLA's SP x TP block-cyclic cache)
  core:  ttMLA's sparse path, unchanged: sparse_sdpa over all 64 heads per chip with the learnable sink
         (``attention_sink``), each query's whole top-k on one chip -> all-to-all back to [16 heads, S / sp]
  wkv_b2: all-to-all(SP), sequence -> heads: [2 heads, S] -> wkv_b2 (2 heads)
  gate:  x @ Wg -> reduce-scatter(TP, N) -> sigmoid. The reduce over the hidden (TP) completes BEFORE the sigmoid
         (sigmoid of a partial sum would be wrong); the scatter only picks this chip's 2 heads.
  o:     (v * g) -> all-gather(TP) [8 heads, S] -> @ Wo -> reduce-scatter(SP, sequence): [S / sp, hidden / tp], the
         residual layout.

With SP on mesh axis 0 the shared layers' sparse KV push addresses the chips in the cache's stripe order
(``ttMLA.push_matches_stripe_order``), so the 2D layout uses the push, as GLM-5.3 does.

The attention core is ttMLA's: per chip, all 64 heads over S / 32 queries, with the selected KV rows gathered to
that chip, so each query's softmax is complete on one chip and the sink goes into ``sparse_sdpa`` exactly once. The
context-parallel stripe-and-LSE-merge core of #65884 (each chip attends over its own KV stripe, partials merged by
LSE) is NOT used here; if it lands (#65886 / #65887) the sink must move to the merge,
``out = merged * sigmoid(lse - sink)`` (``tests/test_hy4_attention_math.py::test_lse_fold_is_the_same_function``).

Not here: the DSA indexer on this layout (#65887; TtIndexer tiles at TP 4 but shards over TP only, replicated over
SP); this module takes the top-k as ``indexer_indices``.

Cache files: ``layer_{i}.mla2d_{layout}.*`` with the layout in the name (``cache_name``: mesh, SP / TP sizes and axes,
e.g. ``mla2d_8x4_sp8ax0_tp4ax1`` on SC1) and a stamp ``layer_{i}.mla2d_{layout}.layout.stamp`` next to them holding the
layout descriptor (``utils/sharded_cache.py``: module, ``CACHE_FORMAT``, mesh, axes, dtypes). Shards without a stamp,
or with a stamp for another layout / format, refuse to load. ``CACHE_FORMAT`` is bumped whenever the shards change
without an axis moving (format 2: q_b's K over TP; the first split cached as ``mla2d`` / ``mla2d_v2`` and never loads).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCache
from models.demos.deepseek_v3_d_p.utils.sharded_cache import LayoutCacheGuard, layout_descriptor, layout_tag
from models.demos.deepseek_v3_d_p.utils.weight_cache_dump import weight_cache_dump_mode

# The config switch: models that set it (Hy4-preview on the tp32 layout) build ttMLA2DTP; GLM-5.x, DeepSeek and
# Kimi never set it and keep ttMLA.
MLA_2D_TP_FLAG = "mla_2d_tp"


def uses_2d_tp(config) -> bool:
    return bool(getattr(config, MLA_2D_TP_FLAG, False))


def mla_class(config):
    """ttMLA2DTP when the config opts in to 2D TP attention, else ttMLA."""
    return ttMLA2DTP if uses_2d_tp(config) else ttMLA


class ttMLA2DTP(ttMLA):
    CACHE_PREFIX = "mla2d"  # + the layout tag (cache_name); never ttMLA's TP-only shards
    CACHE_FORMAT = 2  # bump when the shards change without an axis moving (module docstring)

    # ------------------------------------------------------------------------------------------- host math
    @staticmethod
    def heads_of_chip(num_heads: int, sp: int, tp: int, s: int, t: int) -> list[int]:
        """The heads chip (s on the SP axis, t on the TP axis) owns in the per-head ops: q_b's and the gate's SP block
        s (``num_heads / sp`` heads), reduce-scattered over TP, chunk t."""
        block = num_heads // sp
        per_chip = block // tp
        return [s * block + t * per_chip + j for j in range(per_chip)]

    @staticmethod
    def core_head_order(num_heads: int, sp: int, tp: int) -> list[int]:
        """The order the attention core sees the 64 heads in after the all-to-alls over SP then TP (and the sink is
        stored in): TP column t holds the heads of chips (0, t) .. (sp - 1, t)."""
        return [h for t in range(tp) for s in range(sp) for h in ttMLA2DTP.heads_of_chip(num_heads, sp, tp, s, t)]

    @staticmethod
    def check_dims(config, sp: int, tp: int) -> None:
        H, hidden, Q = config.num_attention_heads, config.hidden_size, config.q_lora_rank
        v, qk = config.v_head_dim, config.qk_nope_head_dim + config.qk_rope_head_dim
        T = ttnn.TILE_SIZE
        checks = {
            "heads over SP x TP": H % (tp * sp) == 0,
            "hidden over TP x SP, whole tiles": hidden % (sp * tp * T) == 0,
            "q_lora over SP x TP, whole tiles": Q % (sp * tp * T) == 0,
            "heads per chip, whole tiles": (H // (tp * sp)) * qk % T == 0 and (H // (tp * sp)) * v % T == 0,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"2D TP attention (SP {sp} x TP {tp}) does not tile: {bad}")

    @staticmethod
    def host_weights_2d(state_dict: dict | None, config, sp: int, tp: int) -> dict:
        """Host tensors laid out for the 2D mesh mappers, as ``name -> (tensor, sp_dim, tp_dim)``: the tensor dim
        sharded over the SP axis and the one sharded over the TP axis (None = replicated over that axis).

        ``state_dict`` holds ttMLA's canonical weights (``q_a_proj.weight`` ... ``g_proj.weight``, ``attn_sink.weight``,
        with Hy4's RoPE frames already reordered by ``hy4_checkpoint.hy4_mla_weights``). None/empty -> placeholders
        of the right shapes (cache-only load). Pure torch, so the CPU tests check the same layout the device loads."""
        H, hidden = config.num_attention_heads, config.hidden_size
        Q, L = config.q_lora_rank, config.kv_lora_rank
        N, R, V = config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim
        use_gate = bool(getattr(config, "mla_use_output_gate", False))
        use_sink = bool(getattr(config, "mla_learnable_sink", False))
        real = bool(state_dict) and "q_a_proj.weight" in state_dict
        order = torch.tensor(ttMLA2DTP.core_head_order(H, sp, tp))
        if real:
            sd = state_dict
            q_a = sd["q_a_proj.weight"].transpose(-2, -1)
            q_b = sd["q_b_proj.weight"].transpose(-2, -1)
            kv_a = sd["kv_a_proj_with_mqa.weight"].transpose(-2, -1)
            kv_b = sd["kv_b_proj.weight"].reshape(H, N + V, L)
            w_uk = kv_b[:, :N, :]  # [H, N, L]: q_nope @ w_uk -> latent
            w_uv = kv_b[:, N:, :].transpose(-2, -1)  # [H, L, V]: latent @ w_uv -> v
            o = sd["o_proj.weight"].transpose(-2, -1)
            g = sd["g_proj.weight"].transpose(-2, -1) if use_gate else None
            q_a_ln = sd["q_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            kv_a_ln = sd["kv_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            sink = (
                ttMLA.sink_for_sparse_sdpa(sd["attn_sink.weight"], ttMLA.softmax_scale(config))[..., order]
                if use_sink
                else None
            )
        else:
            e = torch.empty
            q_a, q_b, kv_a = e(hidden, Q), e(Q, H * (N + R)), e(hidden, L + R)
            w_uk, w_uv, o = e(H, N, L), e(H, L, V), e(H * V, hidden)
            g = e(hidden, H * V) if use_gate else None
            q_a_ln = e(1, 1, Q // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            kv_a_ln = e(1, 1, L // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            sink = torch.empty(1, 1, 1, H, dtype=torch.bfloat16) if use_sink else None
        out = {
            "q_a_proj": (q_a, 1, 0),
            "q_b_proj": (q_b, 1, 0),
            # K over both axes: hidden index t * (hidden / tp) + s * (hidden / (tp * sp)) + k.
            "kv_a_proj_with_mqa": (kv_a.reshape(tp, sp, hidden // (sp * tp), L + R), 1, 0),
            # Checkpoint head order: dim 0 is the SP block (q_b's / the gate's), dim 1 its TP chunks (2 heads each).
            "wkv_b1": (w_uk.reshape(sp, H // sp, N, L), 0, 1),
            "wkv_b2": (w_uv.reshape(sp, H // sp, L, V), 0, 1),
            "o_proj": (o, 0, 1),
            "q_a_layernorm": (q_a_ln, None, None),
            "kv_a_layernorm": (kv_a_ln, None, None),
        }
        if use_gate:
            out["g_proj"] = (g, 1, 0)
        if use_sink:
            out["attn_sink"] = (sink, None, None)
        return out

    @staticmethod
    def chip_shard(t, sp_dim, tp_dim, sp: int, tp: int, s: int, c: int) -> torch.Tensor:
        """What ``ShardTensor2dMesh`` puts on chip (s, c) for a ``host_weights_2d`` entry (CPU tests)."""
        if sp_dim is not None:
            t = t.chunk(sp, dim=sp_dim)[s]
        if tp_dim is not None:
            t = t.chunk(tp, dim=tp_dim)[c]
        return t

    # ------------------------------------------------------------------------------------------- caches
    VECTOR_WEIGHTS = ("q_a_layernorm", "kv_a_layernorm", "attn_sink")

    @staticmethod
    def cache_name(mesh_shape, sp_axis: int, tp_axis: int) -> str:
        """``mla2d_{rows}x{cols}_sp{n}ax{a}_tp{n}ax{a}``: the shards' file prefix after ``layer_{i}.``."""
        return f"{ttMLA2DTP.CACHE_PREFIX}_{layout_tag(mesh_shape, {'sp': sp_axis, 'tp': tp_axis})}"

    @staticmethod
    def cache_guard(cache_path: Path | None, layer_idx: int, mesh_shape, sp_axis: int, tp_axis: int):
        """The layout stamp of layer ``layer_idx``'s 2D shards (``utils/sharded_cache.py``)."""
        name = f"layer_{layer_idx}.{ttMLA2DTP.cache_name(mesh_shape, sp_axis, tp_axis)}"
        dtypes = {
            w: "BFLOAT16" if w in ttMLA2DTP.VECTOR_WEIGHTS else "BFLOAT8_B" for w in ttMLA.weight_names(True, True)
        }
        descriptor = layout_descriptor(
            "ttMLA2DTP", ttMLA2DTP.CACHE_FORMAT, mesh_shape, {"sp": sp_axis, "tp": tp_axis}, dtypes
        )
        return LayoutCacheGuard(cache_path, name, f"{name}.*.tensorbin", descriptor)

    @staticmethod
    def check_cache_complete(
        cache_path: Path,
        cache_name_prefix: str,
        has_indexer: bool = False,
        has_output_gate: bool = False,
        has_sink: bool = False,
        *,
        mesh_shape=None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
    ) -> bool:
        """``cache_name_prefix`` is ttMLA's ``.../layer_{i}.mla``; the 2D shards live under
        ``layer_{i}.{cache_name(...)}``, so the mesh layout is required. Complete means every shard plus a matching
        layout stamp."""
        from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        if mesh_shape is None or sp_axis is None or tp_axis is None:
            raise ValueError(
                "ttMLA2DTP.check_cache_complete needs mesh_shape, sp_axis and tp_axis: they name the shards"
            )
        prefix = str(cache_name_prefix)
        if not prefix.endswith(".mla"):
            raise ValueError(f"expected ttMLA's '.../layer_{{i}}.mla' prefix, got {prefix!r}")
        layer_idx = int(Path(prefix).name[len("layer_") : -len(".mla")])
        prefix_2d = prefix[: -len(".mla")] + "." + ttMLA2DTP.cache_name(mesh_shape, sp_axis, tp_axis)
        for name in ttMLA.weight_names(has_output_gate, has_sink):
            if not pattern_exists(f"{prefix_2d}.{name}*.tensorbin", "MLA"):
                return False
        if not ttMLA2DTP.cache_guard(cache_path, layer_idx, mesh_shape, sp_axis, tp_axis).stamped():
            return False
        return not has_indexer or TtIndexer.check_cache_complete(cache_path, cache_name_prefix)

    @staticmethod
    def _convert_and_cache_weights(
        state_dict: dict | None,
        mesh_device: ttnn.MeshDevice,
        config,
        layer_idx: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        cache_path: Path | None = None,
        device: ttnn.MeshDevice | None = None,
        kv_only: bool = False,
        replicate_wkv_b1: bool = False,  # ttMLA's early-Q replica; the 2D layout shards wkv_b1 instead
        tp_shard_q_a_norm: bool = False,
    ) -> dict | None:
        assert not tp_shard_q_a_norm, "2D TP normalizes the full q_a row after its gathers"
        sp, tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
        ttMLA2DTP.check_dims(config, sp, tp)
        host = ttMLA2DTP.host_weights_2d(state_dict, config, sp, tp)
        mem = ttnn.DRAM_MEMORY_CONFIG if device else None
        kv_names = ("kv_a_proj_with_mqa", "kv_a_layernorm")
        mesh_shape = tuple(mesh_device.shape)
        guard = ttMLA2DTP.cache_guard(cache_path, layer_idx, mesh_shape, sp_axis, tp_axis)
        guard.check()  # shards from another layout (or unstamped) refuse to load
        prefix = f"layer_{layer_idx}.{ttMLA2DTP.cache_name(mesh_shape, sp_axis, tp_axis)}"
        result = {}
        for name, (t, sp_dim, tp_dim) in host.items():
            if kv_only and name not in kv_names:
                continue
            if sp_dim is None and tp_dim is None:
                mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            else:
                dims = [None, None]
                dims[sp_axis], dims[tp_axis] = sp_dim, tp_dim
                mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
            vector = name in ttMLA2DTP.VECTOR_WEIGHTS
            result[name] = ttnn.as_tensor(
                t,
                device=device,
                dtype=ttnn.bfloat16 if vector else ttnn.bfloat8_b,
                layout=ttnn.ROW_MAJOR_LAYOUT if vector else ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
                cache_file_name=str(cache_path / f"{prefix}.{name}") if cache_path else None,
                cache_dump_mode=weight_cache_dump_mode(),
            )
        guard.write()
        if device is None:
            return None
        return result

    @staticmethod
    def build_ttnn_cache(
        state_dict: dict,
        cache_path: Path,
        mesh_device: ttnn.MeshDevice,
        config,
        layer_idx: int,
        seq_len: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        kv_only: bool = False,
        has_indexer: bool | None = None,
        use_fused_rmsnorm: bool = False,
    ):
        """The 2D shards only: the DSA indexer has no 2D layout yet (#65887)."""
        ttMLA2DTP._convert_and_cache_weights(
            state_dict, mesh_device, config, layer_idx, sp_axis, tp_axis, cache_path, device=None, kv_only=kv_only
        )

    # ------------------------------------------------------------------------------------------- device
    def __init__(self, config, state_dict, mesh_device, *args, **kwargs):
        super().__init__(config, state_dict, mesh_device, *args, **kwargs)
        if not self._has_indexer:
            raise NotImplementedError("2D TP MLA runs the sparse (DSA) core only")
        if not isinstance(self._indexer, type(None)) and type(self._indexer).__name__ == "TtIndexer":
            raise NotImplementedError(
                f"layer {self.layer_idx}: the DSA indexer has no 2D-TP layout yet (#65887); build the layer as a "
                "shared layer and pass its top-k as indexer_indices"
            )
        # The K-over-both-axes shard loads as [1, 1, K, 576]; matmul_reduce_scatter_prefill takes a rank-2 weight.
        kv_a = self.kv_a_proj_with_mqa_weight
        self.kv_a_proj_with_mqa_weight = ttnn.reshape(kv_a, (kv_a.shape[-2], kv_a.shape[-1]))
        self.heads_per_chip = self.num_heads // (self.tp_factor * self.sp_factor)
        # All-gather outputs: persistent, shared by every layer through TT_CCL, and gathered with ttMLA's shared
        # high-bandwidth semaphores (plain ttnn.all_gather allocates its own in L1_SMALL, which the prefill's 1536 B
        # do not hold). Gather outputs alias these buffers, so the forward never deallocates them.
        S = self.active_seq_len
        self._gather_out = {
            name: self.tt_ccl.get_mla_high_bw_all_gather_buffer(
                name=f"mla2d_{name}", shape=[1, 1, S, width], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            for name, width in (
                ("x", self.hidden_size // self.tp_factor),
                ("q_a_col", self.q_lora_rank // self.sp_factor),
                ("q_a", self.q_lora_rank),
                ("o_in", self.num_heads // self.sp_factor * self.v_head_dim),
            )
        }
        ready, data_valid = self.tt_ccl.get_mla_high_bw_gather_semaphores()
        self._gather_semaphores = dict(ready_semaphore=ready, data_valid_semaphore=data_valid)
        logger.info(
            f"MLA layer {self.layer_idx}: 2D TP, residual [S/{self.sp_factor}, hidden/{self.tp_factor}] (SP axis "
            f"{self.sp_axis}, TP axis {self.tp_axis}), {self.heads_per_chip} heads per chip in the per-head ops"
        )

    def _axis_topology(self, axis: int):
        return self.sp_ccl_topology if axis == self.sp_axis else self.tp_ccl_topology

    def _mm_rs(self, x: ttnn.Tensor, w: ttnn.Tensor, axis: int, scatter_dim: int) -> ttnn.Tensor:
        """x @ w with the partials summed (fp32) over ``axis`` and scattered over ``scatter_dim``."""
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            x,
            w,
            cluster_axis=axis,
            scatter_dim=scatter_dim,
            topology=self._axis_topology(axis),
            num_links=self.ccl_num_links,
        )

    def _all_gather(self, x: ttnn.Tensor, axis: int, name: str, dim: int = 3) -> ttnn.Tensor:
        """All-gather ``dim`` over ``axis`` into the persistent buffer ``name`` (returned; not to be freed)."""
        return ttnn.experimental.high_bw_all_gather(
            x,
            dim=dim,
            output_tensor=self._gather_out[name],
            cluster_axis=axis,
            num_links=self.ccl_num_links,
            **self._gather_semaphores,
        )

    def _all_to_all(self, x: ttnn.Tensor, in_dim: int, out_dim: int, axis: int) -> ttnn.Tensor:
        """Gather ``in_dim`` and split ``out_dim`` over ``axis``."""
        return ttnn.experimental.all_to_all_async_generic(
            x,
            in_dim=in_dim,
            out_dim=out_dim,
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=axis,
        )

    def _q_a_latent_2d(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """q_a_layernorm(x @ Wqa), the whole 2048-wide row on every chip: [1, 1, S, q_lora]."""
        part = self._mm_rs(x, self.q_a_proj_weight, self.tp_axis, -1)  # [S, q_lora / (sp * tp)], reduced
        col = self._all_gather(part, self.tp_axis, "q_a_col")  # [S, q_lora / sp]
        ttnn.deallocate(part)
        qr = self._all_gather(col, self.sp_axis, "q_a")  # [S, q_lora]
        return self._q_a_norm(qr, ttnn.DRAM_MEMORY_CONFIG)

    def _q_stem_2d(self, qr: ttnn.Tensor, rope_tensors, kv_actual_isl, seq_len_local: int, metadata) -> ttnn.Tensor:
        """q_b (K over TP) -> this chip's heads -> wkv_b1 -> heads to sequence -> RoPE: [1, H, S / (sp * tp), 576]
        row-major, the q ttMLA's sparse core takes (as _q_stem_seq_sharded returns it)."""
        k_slice = ttnn.mesh_partition(qr, dim=3, cluster_axis=self.tp_axis)  # [S, q_lora / tp]
        ttnn.deallocate(qr)
        q = self._mm_rs(k_slice, self.q_b_proj_weight, self.tp_axis, -1)  # [S, heads_per_chip * qk_head_dim]
        ttnn.deallocate(k_slice)
        q_nope, q_rope = ttnn.experimental.nlp_create_q_heads_split(
            q,
            num_heads=self.heads_per_chip,
            split_head_dim=self.qk_nope_head_dim,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        chunk = q_nope.shape[2]
        q_lat = ttnn.linear(q_nope, self.wkv_b1_weight, **self._make_batched_mm_kwargs("wkv_b1", chunk))
        ttnn.deallocate(q_nope)
        moved = []
        for t in (q_lat, q_rope):
            rows = self._all_to_all(t, 1, 2, self.sp_axis)  # [heads / tp, S / sp]
            ttnn.deallocate(t)
            moved.append(self._all_to_all(rows, 1, 2, self.tp_axis))  # [heads, S / (sp * tp)], core order
            ttnn.deallocate(rows)
        prefix, rope_seq = moved
        tt_q = self._apply_rope_padded(
            rope_seq,
            rope_tensors,
            kv_actual_isl,
            metadata=metadata,
            concat_prefix=prefix,
            seq_subshard_axis=self.tp_axis,
            row_major_output=True,
        )
        ttnn.deallocate(prefix)
        ttnn.deallocate(rope_seq)
        return tt_q

    def _kv_a_proj(self, hidden_states: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """x[:, K_ts] @ Wkva reduced over SP and sequence-scattered over it: [1, 1, S / sp, 576], still partial over
        TP, which ttMLA's KV stem then gathers and reduces in fp32. ``hidden_states`` is the gathered [S, hidden / tp].
        """
        x_rc = ttnn.mesh_partition(hidden_states, dim=3, cluster_axis=self.sp_axis)
        kv = self._mm_rs(x_rc, self.kv_a_proj_with_mqa_weight, self.sp_axis, -2)
        ttnn.deallocate(x_rc)
        return kv

    def _apply_wkv_b2(self, t: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """[1, heads / tp, S / sp, 512] -> sequence to heads over SP -> wkv_b2 on this chip's heads:
        [1, heads_per_chip, S, v_head_dim]."""
        mine = self._all_to_all(t, 2, 1, self.sp_axis)
        # bf16 out (ttMLA's default is bf8): the gated v is all-gathered into a bf16 buffer before o_proj.
        kwargs = {**self._make_batched_mm_kwargs("wkv_b2", mine.shape[2]), "dtype": ttnn.bfloat16}
        v = ttnn.linear(mine, self.wkv_b2_weight, **kwargs)
        ttnn.deallocate(mine)
        return v

    def _output_gate(self, hidden_states: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """sigmoid(x @ Wg) for this chip's heads: the reduce over the hidden (TP) finishes before the sigmoid."""
        g = self._mm_rs(hidden_states, self.g_proj_weight, self.tp_axis, -1)
        out = ttnn.sigmoid(g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(g)
        return out

    def _o_proj_epilogue(
        self, attn_out: ttnn.Tensor, seq_len_local: int, hidden_states: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        v = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [S, hpc_chip * V]
        if self._use_gate:
            g = self._output_gate(hidden_states, seq_len_local)
            gated = ttnn.multiply(v, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(g)
            ttnn.deallocate(v)
            v = gated
        block = self._all_gather(v, self.tp_axis, "o_in")  # [S, heads / sp * V]: this chip's SP head block
        ttnn.deallocate(v)
        return self._mm_rs(block, self.o_proj_weight, self.sp_axis, -2)  # [S / sp, hidden / tp], the residual

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: MlaKvCache,
        cache_layer_idx: int = 0,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        return_kv_intermediates: bool = False,
        index_kv_cache: Optional[ttnn.Tensor] = None,
        indexer_indices: Optional[ttnn.Tensor] = None,
        return_indexer_indices: bool = False,
        metadata: Optional[ttnn.Tensor] = None,
        force_kv_only: bool = False,
    ):
        """``hidden_states``: the residual layout, ``[1, 1, S / sp, hidden / tp]`` (the prefill block's, ttMLA's and
        TtMlp2D's). Returns the same layout. ``indexer_indices``: this chunk's top-k as ttMLA takes it,
        ``[1, 1, S / (sp * tp), k]`` per chip."""
        if kvpe_cache.geometry != self.kv_cache_geometry:
            raise ValueError(f"MLA configured for KV geometry {self.kv_cache_geometry}, got {kvpe_cache.geometry}")
        assert kvpe_cache.format == self.sparse_kv_cache_format
        seq_len_local = hidden_states.shape[2]
        assert (
            seq_len_local == self.active_seq_len_local
        ), f"2D TP MLA was built for {self.active_seq_len_local} local rows, got {seq_len_local}"
        kv_actual_isl = actual_start or 0
        ttnn.tracy_message("`TT_SIGNPOST: MLA_START`")
        # Every hidden-side matmul takes the whole chunk: gather the sequence over SP once (persistent buffer).
        hidden_states = self._all_gather(hidden_states, self.sp_axis, "x", dim=2)  # [S, hidden / tp]

        if self.kv_only or force_kv_only:
            tt_kvpe, tt_kv_nope, _ = self._kv_stem(
                hidden_states, rope_tensors, kv_actual_isl, seq_len_local, False, kvpe_cache, metadata=metadata
            )
            if tt_kv_nope is not None:
                ttnn.deallocate(tt_kv_nope)
            self._update_kv_cache(
                kvpe_cache,
                tt_kvpe,
                cache_user_id=cache_user_id,
                cache_layer_idx=cache_layer_idx,
                kv_actual_isl=kv_actual_isl,
                actual_end=actual_end,
                metadata=metadata,
                tp_axis=self.tp_shard_kv_axis,
            )
            for value in tt_kvpe if isinstance(tt_kvpe, tuple) else (tt_kvpe,):
                ttnn.deallocate(value)
            return None

        if indexer_indices is None:
            raise NotImplementedError("2D TP MLA takes its top-k as indexer_indices (indexer layout: #65887)")
        if self.tt_ccl.sparse_kv_push_planner is not None and self.layer_idx == self._first_layer_idx:
            self.tt_ccl.sparse_kv_push_planner.invalidate()

        qr = self._q_a_latent_2d(hidden_states)
        tt_q = self._q_stem_2d(qr, rope_tensors, kv_actual_isl, seq_len_local, metadata)
        tt_kvpe, tt_kv_nope, kv_intermediates = self._kv_stem(
            hidden_states,
            rope_tensors,
            kv_actual_isl,
            seq_len_local,
            return_kv_intermediates,
            kvpe_cache,
            metadata=metadata,
        )
        attn_out = self._attention(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            tt_kv_nope=tt_kv_nope,
            indices=indexer_indices,
            kvpe_cache=kvpe_cache,
            cache_layer_idx=cache_layer_idx,
            cache_user_id=cache_user_id,
            seq_len_local=seq_len_local,
            kv_actual_isl=kv_actual_isl,
            actual_end=actual_end,
            metadata=metadata,
            injected=True,
        )
        out = self._o_proj_epilogue(attn_out, seq_len_local, hidden_states=hidden_states)
        ttnn.deallocate(attn_out)
        ttnn.tracy_message("`TT_SIGNPOST: MLA_END`")
        result = (out,)
        if return_kv_intermediates:
            result += (kv_intermediates,)
        if return_indexer_indices:
            result += (indexer_indices,)
        return result[0] if len(result) == 1 else result
