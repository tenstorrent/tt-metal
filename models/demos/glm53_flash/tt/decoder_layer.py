# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.3-Flash decoder layer on a single device (HF ``Glm5NextTextDecoderLayer``).

    residual = R                                  # 4 mHC streams, packed [1,1,T,4*4096] fp32
    post, comb, h = attn_hc(R)                    # fused mHC parametrization (fp32)
    y = attn(input_layernorm(h))                  # KDA or DSA, bf16
    R = post * y + comb^T R
    post, comb, h = ffn_hc(R)
    y = mlp(post_attention_layernorm(h))          # dense SwiGLU or MoE
    R = post * y + comb^T R

Prefill runs one user at a time (``[1,1,T_pad,16384]``), decode runs B users (``[1,1,B,16384]``).
Autoregressive state is device-owned: the KDA ``KdaState`` per layer (carried through the
forward calls) and the DSA paged latent cache (``TtLatentCache``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import TtDSA, TtLatentCache, sparse_decode_enabled
from models.demos.glm53_flash.tt.kda import TtKDA
from models.demos.glm53_flash.tt.mesh import GLMMesh, single_chip
from models.demos.glm53_flash.tt.mhc import build_mhc
from models.demos.glm53_flash.tt.ttkda.kda import KdaState

_DUMPED: set = set()


@dataclass(frozen=True)
class WaveInfo:
    """A wave's per-row chunk geometry for one layer call (``doc/mesh/row_ownership.md`` mechanism (a)): every mesh
    row runs its own chunk of ``T`` padded rows; ``starts[r]`` / ``valids[r]`` are row r's absolute chunk start and
    valid row count (0 on an idle row: pad tokens, results discarded); ``start_t`` / ``end_t`` the row-sharded KDA
    chronology scalars of ``(start, start + valid)`` (``TtKDA.scalar_rows``)."""

    T: int
    starts: tuple[int, ...]
    valids: tuple[int, ...]
    start_t: ttnn.Tensor
    end_t: ttnn.Tensor

    @property
    def rows(self) -> int:
        return len(self.starts)

    @property
    def member_rows(self) -> tuple[int, ...]:
        return tuple(r for r, v in enumerate(self.valids) if v > 0)


def sp_attention_mode() -> str:
    """How a DSA layer attends in the sequence-parallel (multi-row) prefill: ``slab`` (default: every row
    attends its own slab's queries only, ``TtDSA._prefill_slab``) or ``v1`` (every row gathers the chunk's
    hidden, attends over all of it and keeps its slab through a reduce-scatter; the A/B reference).
    Knob ``GLM53_SP_ATTN``."""
    mode = os.environ.get("GLM53_SP_ATTN", "slab")
    if mode not in ("slab", "v1"):
        raise ValueError(f"GLM53_SP_ATTN={mode!r}: expected 'slab' or 'v1'")
    return mode


def _dump(dump_dir: str, name: str, t) -> None:
    """Save device 0's shard of ``t`` as fp32 (prefill A/B dumps; see DecoderLayer._hc_call)."""
    import os as _os

    import torch as _torch

    _os.makedirs(dump_dir, exist_ok=True)
    shard = ttnn.get_device_tensors(t)[0]
    _torch.save(ttnn.to_torch(shard).float(), _os.path.join(dump_dir, name + ".pt"))


from models.demos.glm53_flash.tt.mlp import TtClampedSwiGLU
from models.demos.glm53_flash.tt.moe import TtMoE, pad_mask_enabled, pad_row_mask_host, wave_row_mask_host


def _strip(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


@dataclass
class LayerState:
    """Per-layer autoregressive state for one forward call."""

    kda: KdaState | None = None  # KDA layers: recurrent [B,H,K,V] fp32 + conv history [B,3,W] bf16
    cache: TtLatentCache | None = None  # DSA layers: paged latent cache + page table


class TtDecoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        layer_idx: int,
        state_dict: dict[str, torch.Tensor],
        *,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
    ):
        """``state_dict``: raw checkpoint tensors of ``model.language_model.layers.{i}.`` with the
        prefix stripped and the fp8 groups dequantized (``reference/checkpoint.load_layer_raw``).
        ``weight_dtype`` covers the big non-expert matmuls (attention projections, dense MLP, shared
        expert); ``expert_dtype`` / ``expert_down_dtype`` the routed experts' gate/up and down weights.
        ``weight_cache_path``: per-layer tensorbin directory for the KDA prefill weights and the routed
        experts (``TtMoE.expert_cache_complete``: with a complete expert cache the state dict may omit
        the ``mlp.experts.N.*`` tensors).
        ``mesh`` (``tt/mesh.py``): the multi-chip placement; the hyper-connections and norms are
        replicated, the sublayers are tensor-parallel over the columns."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_kda = cfg.is_kda(layer_idx)
        self.is_moe = cfg.is_moe(layer_idx)
        self.hidden = cfg.hidden_size
        self.n_streams = cfg.hc_mult
        sd = state_dict
        self.attn_hc = build_mhc(device, cfg, sd["hc_attn_fn"], sd["hc_attn_base"], sd["hc_attn_scale"])
        self.ffn_hc = build_mhc(device, cfg, sd["hc_ffn_fn"], sd["hc_ffn_base"], sd["hc_ffn_scale"])
        self.input_norm_w = ttnn.from_torch(
            sd["input_layernorm.weight"].float().reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.post_norm_w = ttnn.from_torch(
            sd["post_attention_layernorm.weight"].float().reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        attn_sd = _strip(sd, "self_attn.")
        if self.is_kda:
            self.attn = TtKDA(
                device,
                cfg,
                attn_sd,
                layer_idx=layer_idx,
                weight_cache_path=weight_cache_path,
                weight_dtype=weight_dtype,
                mesh=self.mesh,
            )
        else:
            self.attn = TtDSA(device, cfg, attn_sd, layer_idx=layer_idx, weight_dtype=weight_dtype, mesh=self.mesh)
        mlp_sd = _strip(sd, "mlp.")
        if self.is_moe:
            self.mlp = TtMoE(
                device,
                cfg,
                mlp_sd,
                weight_dtype=weight_dtype,
                expert_dtype=expert_dtype,
                expert_down_dtype=expert_down_dtype,
                mesh=self.mesh,
                weight_cache_path=weight_cache_path,
            )
            self.mlp.experts.call_site = f"layer {layer_idx} MoE"  # the eager dispatch guard's label
        else:
            self.mlp = TtClampedSwiGLU(
                device,
                mlp_sd["gate_proj.weight"],
                mlp_sd["up_proj.weight"],
                mlp_sd["down_proj.weight"],
                cfg.swiglu_limit,
                weight_dtype=weight_dtype,
                mesh=self.mesh,
            )
        self.norm_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ------------------------------------------------------------------ helpers
    def _norm(self, h: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(h, epsilon=self.cfg.rms_norm_eps, weight=w, compute_kernel_config=self.norm_compute)

    def _hc_call(self, hc, streams, sublayer, norm_w, collapse: bool = False):
        """Run a hyper-connection wrap around ``sublayer``. ``TtMHCFast`` applies the sublayer's input
        RMSNorm inside its pre-block (``fuses_norm``: the collapsed hidden comes out normalised, bf16,
        multicore); the composite ``TtMHCWrap`` hands over the raw bf16 hidden and the closure norms it.
        ``collapse`` (the last layer's FFN wrap feeding the LM head): return the mean over the new
        streams ``[1,1,B,hidden]`` fp32 instead of the streams (``supports_collapse`` wraps only).
        ``GLM53_MHC_DUMP=<dir>``: the first prefill call of every wrap dumps its input streams, the
        sublayer's input and output and the output streams (device 0's shard) for an A/B between
        implementations (``tests/mhc_dump_diff.py``)."""
        dump_dir = os.environ.get("GLM53_MHC_DUMP")
        key = f"L{self.layer_idx:02d}_{'attn' if hc is self.attn_hc else 'ffn'}"
        if dump_dir and streams.shape[-2] > 32 and key not in _DUMPED:
            _DUMPED.add(key)
            _dump(dump_dir, key + "_in", streams)
            inner = sublayer

            def sublayer(h, _inner=inner):  # noqa: F811
                _dump(dump_dir, key + "_h", h)
                y = _inner(h)
                _dump(dump_dir, key + "_y", y)
                return y

            out = self._hc_call_impl(hc, streams, sublayer, norm_w, collapse)
            _dump(dump_dir, key + "_out", out)
            return out
        return self._hc_call_impl(hc, streams, sublayer, norm_w, collapse)

    def _hc_call_impl(self, hc, streams, sublayer, norm_w, collapse: bool = False):
        if hasattr(hc, "select"):  # TtMHCSplit: the implementation depends on the row count
            hc = hc.select(streams)
        if getattr(hc, "fuses_norm", False):
            kw = dict(collapse=True) if collapse else {}
            # prefill-scope calls (one-tile slabs routed to the fast path) take the one-core norm: TtMHCFast.forward
            return hc.forward(
                streams,
                sublayer,
                sublayer_dtype=ttnn.bfloat16,
                norm=(norm_w, self.cfg.rms_norm_eps),
                prefill=self.mesh.in_prefill,
                **kw,
            )
        if collapse:
            raise NotImplementedError("collapse needs a supports_collapse hyper-connection (TtMHCFast)")

        def normed(h):
            return sublayer(self._norm(h, norm_w))

        return hc.forward(streams, normed, sublayer_dtype=ttnn.bfloat16)

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        streams: ttnn.Tensor,
        state: LayerState,
        *,
        user: int,
        seq_len: int,
        start: int = 0,
        actual_start: ttnn.Tensor | None = None,
        actual_end: ttnn.Tensor | None = None,
        slot=None,
        ring_sel: ttnn.Tensor | None = None,
        row_mask: ttnn.Tensor | None = None,
        wave: WaveInfo | None = None,
    ) -> tuple[ttnn.Tensor, LayerState]:
        """``_prefill_forward_impl`` inside ``mesh.prefill_scope()`` (the composite TP all-reduce for every prefill
        shape, see ``GLMMesh.prefill_scope``). ``row_mask``: the MoE router's pad-row mask of this row's slab
        (``tt/moe.py`` ``pad_row_mask_host``; the traced prefill's per-length input); built here for an eager chunk
        with rows past ``seq_len`` when not given."""
        with self.mesh.prefill_scope():
            return self._prefill_forward_impl(
                streams,
                state,
                user=user,
                seq_len=seq_len,
                start=start,
                actual_start=actual_start,
                actual_end=actual_end,
                slot=slot,
                ring_sel=ring_sel,
                row_mask=row_mask,
                wave=wave,
            )

    def _prefill_forward_impl(
        self,
        streams: ttnn.Tensor,
        state: LayerState,
        *,
        user: int,
        seq_len: int,
        start: int = 0,
        actual_start: ttnn.Tensor | None = None,
        actual_end: ttnn.Tensor | None = None,
        slot=None,
        ring_sel: ttnn.Tensor | None = None,
        row_mask: ttnn.Tensor | None = None,
        wave: WaveInfo | None = None,
    ) -> tuple[ttnn.Tensor, LayerState]:
        """streams [1,1,T_pad,4*hidden] fp32 for one user: row i is the token at absolute position
        ``start + i`` (``start % 32 == 0``), T_pad % 32 == 0 and rows past ``seq_len`` are padding.
        KDA: ``state.kda`` is the carry after position ``start`` (zeros for a new prompt); DSA: the
        user's cache holds [0, start). Returns the new streams and the layer state after the chunk.
        ``actual_start`` / ``actual_end`` override the KDA device scalars (chunk-relative: 0 and the valid rows) derived from
        ``seq_len`` (global positions, as ttKDA defines them). ``slot`` / ``ring_sel``: the DSA path's
        device-side user slot and ring-row selector (the traced prefill, ``TtDSA.prefill_forward``).
        ``wave`` (``WaveInfo``): every mesh row runs ITS OWN chunk of ``T = streams.shape[-2]`` rows (no slabs):
        KDA scans row-locally from the row's carry with the row-sharded chronology scalars, the MoE router sees the
        row's own pad mask; ``user`` / ``seq_len`` / ``start`` are then the uniform slot index, ``T`` and 0."""
        wave_mode = wave is not None
        T = streams.shape[-2] * (1 if wave_mode else self.mesh.rows)  # the chunk's rows (a slab per chip under SP)
        new_state = LayerState(kda=state.kda, cache=state.cache)

        def attention(h):  # h: [1,1,T/rows,hidden] bf16, input_layernorm applied
            if self.is_kda:
                if wave_mode:
                    start_t, end = wave.start_t, wave.end_t
                else:
                    # CHUNK-RELATIVE bounds (start 0, end = the valid rows), never the absolute positions: ttKDA
                    # places a sequence-parallel chunk block-cyclically by the ABSOLUTE start (first rank =
                    # start / local_rows mod ranks, a split segment at start mod local_rows), which matches the
                    # model's contiguous slab upload (rank r = chunk rows [r T/rows, (r+1) T/rows)) only when
                    # start % T == 0. A chunk continued at 2176 in a 256-row chunk (or at 36736 in a 16384-row chunk:
                    # vLLM's budget split) had its ranks' slabs scanned in the wrong order, the carries taken at
                    # end % T (the 2026-10-29 APC state diff: KDA conv/recurrent wrong, every DSA row right). KDA
                    # has no positional encoding: the state carries the history, the position enters nothing else.
                    start_t = actual_start if actual_start is not None else self.attn.chrono_scalar(0, "start")
                    end = actual_end
                # the exclusive valid end, ALWAYS carried (a full first chunk included: end == T clips nothing): the
                # conv kernel is a different program with and without the bound (has_actual_end is a compile-time
                # define), and a chunk shape first seen without it after the captures compiled at request time under
                # live traces (the 2026-10-28 served APC seal hit)
                if end is None:
                    end = self.attn.chrono_scalar(seq_len, "end")
                T_loc = h.shape[-2]
                y, new_state.kda = self.attn.prefill_forward(
                    ttnn.reshape(h, [1, T_loc, self.hidden]), state.kda, start_t, end, local=wave_mode
                )
                return ttnn.reshape(y, [1, 1, T_loc, self.hidden])
            if wave_mode:  # every row its own chunk: per-row fills / index / attention (``TtDSA._prefill_wave``)
                return self.attn.prefill_forward(
                    h, state.cache, user, T, start=0, wave=wave, slot=slot, ring_sel=ring_sel
                )
            dsa_kw = dict(start=start, slot=slot, ring_sel=ring_sel)
            if self.mesh.rows == 1:
                return self.attn.prefill_forward(h, state.cache, user, seq_len, **dsa_kw)
            if sp_attention_mode() == "slab" and slot is None:
                # sequence-parallel rows, slab-local: the row attends its own slab's queries against the
                # full prefix; only the chunk's latent and index stems are gathered over the rows
                # (``TtDSA._prefill_slab``), the output is the slab's directly. The traced prefill (``slot``)
                # keeps v1 below: its chunks are all-dense and <= 2048 rows, and the slab path's per-geometry
                # constants (the Q offset tensor, window bounds) are uploaded on first use, which a trace
                # capture without an eager warm-up of the same shape would not allow (not validated yet).
                return self.attn.prefill_forward(h, state.cache, user, seq_len, slab=True, **dsa_kw)
            # sequence-parallel rows (v1): every row attends over the whole chunk (gathered slabs, in
            # row order), fills the replicated cache/index identically, then keeps its own slab of the
            # output: the reduce-scatter over the rows of ``rows`` identical copies is exactly
            # rows * slab (a power-of-two sum of equal bf16 values), scaled back by 1/rows.
            rows = self.mesh.rows
            h_full = self.mesh.all_gather_rows(h, dim=2)  # [1,1,T,hidden]
            y_full = self.attn.prefill_forward(h_full, state.cache, user, seq_len, **dsa_kw)
            ttnn.deallocate(h_full)
            y = self.mesh.reduce_scatter_rows(y_full, dim=2)  # [1,1,T/rows,hidden] = rows * slab
            ttnn.deallocate(y_full)
            return ttnn.multiply(y, 1.0 / rows, output_tensor=y)

        streams = self._hc_call(self.attn_hc, streams, attention, self.input_norm_w)
        mask, own = row_mask, False
        if self.is_moe:
            if mask is None and wave_mode and pad_mask_enabled() and any(v < T for v in wave.valids):
                # a wave: every row its own valid count (an idle row all pad)
                mask, own = self.mesh.upload(wave_row_mask_host(T, wave.valids), ttnn.bfloat16, rows=2), True
            elif mask is None and not wave_mode and seq_len < T and pad_mask_enabled():
                # an eager chunk with rows past seq_len (the ragged last chunk): the attention leaves them
                # unspecified and the MoE router must not see them (GLM53_ROUTER_PAD_MASK; 2026-10-27 block hang)
                mask, own = self.mesh.upload(pad_row_mask_host(T, seq_len), ttnn.bfloat16, rows=2), True
            ffn = lambda t, m=mask: self.mlp(t, prefill=True, row_mask=m)  # the router's composite path on slabs
        else:
            ffn = self.mlp
        streams = self._hc_call(self.ffn_hc, streams, ffn, self.post_norm_w)
        if own:
            ttnn.deallocate(mask)
        return streams, new_state

    # ------------------------------------------------------------------- decode
    def decode_forward(
        self, streams: ttnn.Tensor, state: LayerState, *, positions, collapse: bool = False
    ) -> tuple[ttnn.Tensor, LayerState]:
        """streams [1,1,B,4*hidden] fp32 (one token per user); ``positions`` carries the persistent
        [B] int32 device tensors ``positions`` / ``pool_ids`` / ``ring_slots`` and the host values
        (``FunctionalDecoder.DecodePositions``). KDA layers ignore it. ``collapse``: return the LM
        head's mean over the new streams ``[1,1,B,hidden]`` fp32 instead (the last layer)."""
        new_state = LayerState(kda=state.kda, cache=state.cache)

        def attention(h):  # h: [1,1,B,hidden] bf16, input_layernorm applied
            if self.is_kda:
                y, new_state.kda = self.attn.decode_forward(h, state.kda, inplace=True)
                return y
            return self.attn.decode_forward(
                h,
                state.cache,
                positions.positions,
                positions.pool_ids,
                positions.ring_slots,
                positions.host,
                dense_positions=getattr(positions, "dense_positions", None),
                sparse_meta=getattr(positions, "sparse", None) if sparse_decode_enabled() else None,
            )

        streams = self._hc_call(self.attn_hc, streams, attention, self.input_norm_w)
        streams = self._hc_call(self.ffn_hc, streams, self.mlp, self.post_norm_w, collapse=collapse)
        return streams, new_state
