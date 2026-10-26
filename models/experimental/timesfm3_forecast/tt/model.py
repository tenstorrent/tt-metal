# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN TimesFM-3 (``google/timesfm-3.0-pytorch``) on one chip, stock ttnn ops.

reference: google-research/timesfm@e51928e, ``src/timesfm3/torch/{model,transformer}.py``; device
decomposition: ``common/decomposition.py``, checked in fp32/fp64 by ``tests/golden/check_dataflow.py``.

    host   prepare_inputs   ``common/host.py``: format, detrend, causal RevIN, token features (fp64)
    device embed            tokens (R*S, 192) -> W_out relu(W_hid t) + W_res t
           layer x 20       h1 = x + post(SeqAttn(norm(x)))      RoPE, QK-norm (head 96), causal + pad mask
                            h2 = h1 + post(VarAttn(norm(h1)))    V = 1: W_ov; else SDPA over the V variates
                            y  = h2 + post(ff1 relu ff0(norm(h2)))
           head             read-out tokens only -> (R*F, 576), normalized space
    host   postprocess      CPM refine, de-normalization, stitching, trend, sort (fp64)

Activations are flat ``(1, 1, R * S, 1280)`` with ``S`` the token count padded to whole tiles.
"""

from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch

from models.experimental.timesfm3_forecast.common import host as host_lib
from models.experimental.timesfm3_forecast.common.decomposition import (
    HEAD_DIM_PAD,
    NUM_HEADS,
    NUM_LAYERS,
    QK_EPS_PAD,
    SDPA_SCALE,
    Weights,
    load_weights,
    pair_rotation,
    rope_tables,
)
from models.experimental.timesfm3_forecast.tt.common import (
    TILE,
    TtPrecision,
    linear,
    pad_to_tile,
    rms_norm,
    take_slots,
    to_flat,
    to_rows,
    upload,
    upload_linear,
    upload_vector,
)

HF_REPO = "google/timesfm-3.0-pytorch"
HF_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
EPS = torch.finfo(torch.float32).eps  # nn.RMSNorm(eps=None) on fp32 weights
QK_EPS = EPS * QK_EPS_PAD
MASK_FILL = -1e9  # finite: a fully masked query row (leading padding) gets uniform weights, never NaN
_MAX_SDPA_CHUNK = 256


# R2-4 (iter_01/r2_4/micro.json): the fused kernel wins at 65536 rows (1.70 vs 2.76 ms) and loses at 2048-3008
# (one tile row per core); below this the 3-op chain runs.
FUSED_NORM_MIN_ROWS = 16384
ROPE_VIEW_GROUPS = 8  # heads per RoPE call: 16 / 8 = 2


def checkpoint_path() -> Path:
    """``model.safetensors`` at the pinned revision in the HF cache (offline)."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    path = hub / f"models--{HF_REPO.replace('/', '--')}" / "snapshots" / HF_REVISION / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"{path} (download it with ttq --kind download)")
    return path


def load_checkpoint_weights(num_layers: int = NUM_LAYERS, layers=None) -> Weights:
    """Device-decomposition weights (fp32 torch); ``layers`` picks a subset (reduced-layer profiles)."""
    from safetensors.torch import load_file

    w = load_weights(load_file(str(checkpoint_path())), torch.float32, num_layers=NUM_LAYERS)
    if layers is not None:
        w.layers = [w.layers[i] for i in layers]
    elif num_layers < NUM_LAYERS:
        w.layers = w.layers[:num_layers]
    return w


def sdpa_chunk(padded_len: int) -> int:
    """Largest tile multiple up to 256 that divides the padded sequence."""
    return max(c for c in range(TILE, min(padded_len, _MAX_SDPA_CHUNK) + 1, TILE) if padded_len % c == 0)


@dataclass
class TtInputs:
    """Device inputs of one forward."""

    tokens: object  # (1, 1, R * S, 192)
    cos: object  # (1, 1, S, 96)
    sin: object
    seq_mask: object | None  # (R, 1, S, S) additive: causal and leading padding; None = causal only
    var_mask: object | None  # (S * B, 1, Vp, Vp) additive per-variate key padding; None = all variates visible
    rows: int
    variates: int
    num_patches: int
    padded_seq: int
    readout_start: int
    readout_count: int
    # Last-layer cut (R1-5): the masks of the read-out query slots only
    readout_seq_mask: object = None  # (R, 1, F, S) additive: causal and leading padding
    readout_var_mask: object = None  # (F * B, 1, Vp, Vp), or None as ``var_mask``
    # R3-1: ``var_mask`` folded into the packed path's block-diagonal mask, (S * R / 32, 1, 32, 32); None when
    # there is no ``var_mask`` or the variates do not pack
    packed_var_mask: object = None
    readout_packed_var_mask: object = None

    def tensors(self):
        return [
            t
            for t in (
                self.tokens,
                self.cos,
                self.sin,
                self.seq_mask,
                self.var_mask,
                self.readout_seq_mask,
                self.readout_var_mask,
                self.packed_var_mask,
                self.readout_packed_var_mask,
            )
            if t is not None
        ]


def seq_mask_tensor(num_front_masked: torch.Tensor, padded_seq: int) -> torch.Tensor:
    """(R, 1, S, S) additive: query q may read key k when k <= q and k is not leading padding."""
    idx = torch.arange(padded_seq)
    allowed = (idx[:, None] >= idx[None, :])[None] & (idx[None, None, :] >= num_front_masked[:, None, None])
    return torch.where(allowed, 0.0, MASK_FILL)[:, None]


def var_mask_tensor(num_front_masked: torch.Tensor, variates: int, padded_seq: int) -> torch.Tensor:
    """(S * B, 1, Vp, Vp) additive, slot-major (s, item), Vp = V padded to whole tiles: key variate u is masked
    at slot s < its padding, and the pad variates u >= V are always masked.

    The variate axis is padded explicitly because SDPA with a user mask does not mask the tile padding of a
    non-tile-aligned key axis (it does without a mask): ``tests/pcc/test_modules.py::test_sdpa_padded_keys``.
    """
    rows = num_front_masked.shape[0]
    items = rows // variates
    vp = pad_to_tile(variates)
    idx = torch.arange(padded_seq)
    key_valid = idx[:, None] >= num_front_masked[None, :]  # (S, R)
    key_valid = torch.nn.functional.pad(key_valid.reshape(padded_seq * items, variates), (0, vp - variates))
    return torch.where(key_valid, 0.0, MASK_FILL)[:, None, None, :].expand(-1, 1, vp, -1).contiguous()


def packed_var_mask_tensor(num_front_masked: torch.Tensor, variates: int, slots: torch.Tensor) -> torch.Tensor:
    """R3-1: (len(slots) * R / 32, 1, 32, 32) additive block-diagonal mask over slot-major (s, row) packed rows:
    query row reads key row when both are the same item at the same slot and the key's slot is not padding."""
    rows = num_front_masked.shape[0]
    key_valid = (slots[:, None] >= num_front_masked[None, :]).reshape(-1, TILE)  # (tiles, 32)
    group = torch.arange(TILE) // variates
    same = group[:, None] == group[None, :]
    allowed = same[None] & key_valid[:, None, :]
    assert (rows * len(slots)) % TILE == 0
    return torch.where(allowed, 0.0, MASK_FILL)[:, None]


def needs_var_mask(num_front_masked: torch.Tensor, variates: int) -> bool:
    """A variate-attention mask matters only when the variates of an item differ in leading padding. With
    shared padding (every forecaster batch: one time mask per item) a patch has all keys valid or none, and a
    fully masked patch's output is never read by a valid token."""
    if variates == 1:
        return False
    per_item = num_front_masked.reshape(-1, variates)
    return bool((per_item != per_item[:, :1]).any())


class TtTimesFM3:
    def __init__(self, device, weights: Weights, precision: TtPrecision | None = None):
        self.device = device
        self.precision = TtPrecision.accuracy() if precision is None else precision
        # R1-5: the last layer runs only the read-out query slots after K / V. Off until it is bundled (alone -1.5 %,
        # under the 2 % bar: candidates.md); ``test_last_layer_cut`` covers the path.
        self.cut_last_layer = True
        self.pack_variates = True  # R2-2
        self.fuse_rope_norm = True  # R3-2
        self._mask_cache = {}
        self.fuse_post_norm = True  # R2-4
        self._pending_norm = None
        p = self.precision
        up = lambda t, role: upload_linear(device, t, p.weight_dtype(role))  # noqa: E731
        self.tok_hidden, self.tok_out, self.tok_res = (
            up(t, "tokenizer") for t in (weights.tok_hidden, weights.tok_out, weights.tok_res)
        )
        self.head_w, self.head_b = up(weights.head_w, "head"), upload_vector(device, weights.head_b)
        self.rope_trans = upload(device, pair_rotation(TILE).reshape(1, 1, TILE, TILE))
        self.layers = []
        for lw in weights.layers:
            layer = {}
            for kind, aw in (("seq", lw.seq), ("var", lw.var)):
                layer[kind] = {
                    "wqkv": up(aw.wqkv, f"{kind}_qkv"),
                    "q_gain": upload_vector(device, aw.q_gain),
                    "k_gain": upload_vector(device, aw.k_gain),
                    "wo": up(aw.wo, f"{kind}_wo"),
                    "post": upload_vector(device, aw.post_gain),
                }
            layer["var"]["w_ov"] = up(lw.var.w_ov, "w_ov")
            layer["ff0"], layer["ff1"] = up(lw.ff0, "ffn"), up(lw.ff1, "ffn")
            layer["post_ff"] = upload_vector(device, lw.post_ff_gain)
            self.layers.append(layer)

    @classmethod
    def from_checkpoint(cls, device, precision: TtPrecision | None = None, *, num_layers=NUM_LAYERS, layers=None):
        return cls(device, load_checkpoint_weights(num_layers, layers), precision)

    # ------------------------------------------------------------------------------------------ inputs
    @staticmethod
    def prepare_inputs(contexts, horizon: int, **kw) -> host_lib.HostInputs:
        return host_lib.prepare(contexts, horizon, **kw)

    def upload_inputs(self, prepared: host_lib.HostInputs) -> TtInputs:
        rows, n = prepared.rows, prepared.num_kept_patches
        s = pad_to_tile(n)
        tokens = torch.nn.functional.pad(prepared.tokens[:, :n], (0, 0, 0, s - n)).reshape(1, 1, rows * s, -1)
        cos, sin = rope_tables(s, torch.float32)
        nfm = prepared.num_front_masked
        padded = bool((nfm > 0).any())
        start, count = prepared.readout_start, prepared.num_forecast_patches
        var_mask = var_mask_tensor(nfm, prepared.variates, s) if needs_var_mask(nfm, prepared.variates) else None
        items = rows // prepared.variates
        v = prepared.variates
        pack = self.pack_variates and var_mask is not None and TILE % v == 0

        def packed(slots):
            if not pack or (rows * len(slots)) % TILE:
                return None
            return upload(self.device, packed_var_mask_tensor(nfm, v, slots))

        return TtInputs(
            tokens=upload(self.device, tokens),
            cos=upload(self.device, cos.reshape(1, 1, s, HEAD_DIM_PAD)),
            sin=upload(self.device, sin.reshape(1, 1, s, HEAD_DIM_PAD)),
            seq_mask=upload(self.device, seq_mask_tensor(nfm, s)) if padded else None,
            var_mask=None if var_mask is None else upload(self.device, var_mask),
            rows=rows,
            variates=prepared.variates,
            num_patches=n,
            padded_seq=s,
            readout_start=start,
            readout_count=count,
            readout_seq_mask=upload(self.device, seq_mask_tensor(nfm, s)[:, :, start : start + count].contiguous()),
            readout_var_mask=(
                None
                if var_mask is None
                else upload(self.device, var_mask[start * items : (start + count) * items].contiguous())
            ),
            packed_var_mask=packed(torch.arange(s)),
            readout_packed_var_mask=packed(torch.arange(start, start + count)),
        )

    @staticmethod
    def deallocate_inputs(inputs: TtInputs):
        import ttnn

        for t in inputs.tensors():
            ttnn.deallocate(t)

    # ----------------------------------------------------------------------------------------- modules
    def embed(self, inputs: TtInputs):
        import ttnn

        p = self.precision
        cfg = p.embed_config()
        hidden = linear(inputs.tokens, self.tok_hidden, p, activation="relu", compute_kernel_config=cfg)
        out = linear(hidden, self.tok_out, p, compute_kernel_config=cfg)
        ttnn.deallocate(hidden)
        res = linear(inputs.tokens, self.tok_res, p, compute_kernel_config=cfg)
        x = ttnn.add(out, res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(res)
        return x

    def _sdpa(self, q, k, v, mask, causal: bool):
        import ttnn

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=sdpa_chunk(q.padded_shape[-2]),
            k_chunk_size=sdpa_chunk(k.padded_shape[-2]),
            exp_approx_mode=False,
        )
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=causal and mask is None,
            scale=SDPA_SCALE,
            program_config=program_config,
            compute_kernel_config=self.precision.sdpa_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_concat_heads=True,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        b, _, n, width = ctx.shape  # (B, 1, n, heads * 96): the heads are already side by side
        return ttnn.reshape(ctx, (b, n, width))

    def _qk_norm(self, q, k, w):
        import ttnn

        p = self.precision
        qn = rms_norm(q, QK_EPS, p, w["q_gain"])
        kn = rms_norm(k, QK_EPS, p, w["k_gain"])
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        return qn, kn

    def _post(self, o, w_post, residual):
        """``residual + RMSNorm(o) * gain``; consumes ``o``, borrows ``residual``. At M >= FUSED_NORM_MIN_ROWS
        (R2-4) one kernel also writes the next sublayer's pre-norm, which ``_pre`` picks up for this output."""
        import ttnn

        from models.experimental.timesfm3_forecast.tt.ops.post_add_norm.op import post_add_norm

        self._drop_pending_norm()
        if self.fuse_post_norm and o.padded_shape[-2] >= FUSED_NORM_MIN_ROWS and len(o.padded_shape) == 4:
            h, xn = post_add_norm(o, residual, w_post, epsilon=EPS)
            ttnn.deallocate(o)
            self._pending_norm = (h, xn)
            return h
        y = rms_norm(o, EPS, self.precision, w_post)
        ttnn.deallocate(o)
        out = ttnn.add(residual, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        return out

    def _drop_pending_norm(self):
        import ttnn

        if self._pending_norm is not None:
            ttnn.deallocate(self._pending_norm[1])
            self._pending_norm = None

    def _pre(self, x):
        """Gain-free pre-norm of the residual stream (the gain is folded into the next projection)."""
        if self._pending_norm is not None and self._pending_norm[0] is x:
            xn = self._pending_norm[1]
            self._pending_norm = None
            return xn
        return rms_norm(x, EPS, self.precision)

    def seq_attention(self, i: int, x, inputs: TtInputs, cut: bool = False):
        """h1 = x + post_ln(SeqAttn(pre_ln(x))); ``x`` is borrowed. With ``cut`` (R1-5), only the read-out slots'
        queries run: the result is (1, 1, R * F, D) at those slots."""
        import ttnn

        p, w = self.precision, self.layers[i]["seq"]
        r, s = inputs.rows, inputs.padded_seq
        xn = self._pre(x)
        qkv = linear(xn, w["wqkv"], p)
        ttnn.deallocate(xn)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            to_rows(qkv, r, s), num_heads=NUM_HEADS, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        if self.fuse_rope_norm:
            # R3-2: RoPE and the QK norm in one pass over q and k (iter_02/r3_2: 4.94 -> 2.11 ms per layer).
            from models.experimental.timesfm3_forecast.tt.ops.rope_qk_norm.op import rope_qk_norm

            qn, kn = rope_qk_norm(
                q, k, inputs.cos, inputs.sin, self.rope_trans, w["q_gain"], w["k_gain"], epsilon=QK_EPS
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            return self._seq_attention_rest(qn, kn, v, x, w, inputs, cut)
        # R2-1: RoPE on a free (R * 8, 2, S, 96) leading-dim view (iter_00/r1_6: 2.56 vs 3.78 ms/call, bitwise equal).
        g = ROPE_VIEW_GROUPS
        rope = lambda t: ttnn.reshape(  # noqa: E731
            ttnn.experimental.rotary_embedding_llama(
                ttnn.reshape(t, (r * g, NUM_HEADS // g, s, t.shape[-1])),
                inputs.cos,
                inputs.sin,
                self.rope_trans,
                is_decode_mode=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=p.norm_config(),
            ),
            (r, NUM_HEADS, s, t.shape[-1]),
        )
        qr, kr = rope(q), rope(k)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        q, k = self._qk_norm(qr, kr, w)
        return self._seq_attention_rest(q, k, v, x, w, inputs, cut)

    def _seq_attention_rest(self, q, k, v, x, w, inputs: TtInputs, cut: bool):
        """SDPA, wo and the post-norm of the time attention from normed, rotated q and k."""
        import ttnn

        p, r, s = self.precision, inputs.rows, inputs.padded_seq
        if not cut:
            merged = self._sdpa(q, k, v, inputs.seq_mask, causal=True)
            o = linear(to_flat(merged), w["wo"], p)
            ttnn.deallocate(merged)
            return self._post(o, w["post"], x)
        # Q for every token, then sliced: projecting Q at the read-out slots only needs head permutes that cost
        # more than they save (iter_00/r1_5b: 651.7 vs 598.1 ms).
        start, count = inputs.readout_start, inputs.readout_count
        qs = ttnn.slice(q, (0, 0, start, 0), (r, NUM_HEADS, start + count, q.shape[-1]))
        ttnn.deallocate(q)
        merged = self._sdpa(qs, k, v, inputs.readout_seq_mask, causal=True)
        o = linear(to_flat(merged), w["wo"], p)
        ttnn.deallocate(merged)
        residual = take_slots(x, r, s, start, count)
        out = self._post(o, w["post"], residual)
        ttnn.deallocate(residual)
        return out

    def var_attention(self, i: int, h1, inputs: TtInputs):
        """h2 = h1 + post_ln(VarAttn(pre_ln(h1))); ``h1`` is borrowed."""
        import ttnn

        p, w = self.precision, self.layers[i]["var"]
        xn = self._pre(h1)
        if inputs.variates == 1:
            # Softmax over one key is 1: the sublayer is W_o W_v. A leading-pad token (reference output 0)
            # gets a finite value instead; no valid token reads it.
            o = linear(xn, w["w_ov"], p)
            ttnn.deallocate(xn)
            return self._post(o, w["post"], h1)
        r, s, v = inputs.rows, inputs.padded_seq, inputs.variates
        if self.pack_variates and TILE % v == 0 and (r * s) % TILE == 0:
            if inputs.var_mask is None:
                return self._var_attention_packed(w, xn, h1, inputs, self._block_mask(v, s * r // TILE))
            if inputs.packed_var_mask is not None:
                return self._var_attention_packed(w, xn, h1, inputs, inputs.packed_var_mask)
        qkv = linear(xn, w["wqkv"], p)
        ttnn.deallocate(xn)
        slot_major = ttnn.permute(to_rows(qkv, r, s), (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(qkv)
        blocks = ttnn.reshape(slot_major, (s * r // v, v, slot_major.shape[-1]))
        if v % TILE:
            # The reshape leaves stale rows in the tile padding, which SDPA reads as keys. (The result can
            # alias ``blocks``; the old handle is dropped, not deallocated.)
            blocks = ttnn.fill_implicit_tile_padding(blocks, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vp = pad_to_tile(v)
        if inputs.var_mask is not None and vp != v:
            # Per-variate padding: pad the variate axis to whole tiles so the mask covers the pad keys. The
            # tile padding is already zero, so the result can alias ``blocks``: drop the handle, never free it.
            blocks = ttnn.pad(blocks, [(0, 0), (0, vp - v), (0, 0)], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, val = ttnn.transformer.split_query_key_value_and_split_heads(
            blocks, num_heads=NUM_HEADS, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(blocks)
        q, k = self._qk_norm(q, k, w)
        merged = self._sdpa(q, k, val, inputs.var_mask, causal=False)  # (S * B, Vq, 1536), Vq = V or Vp
        vq, width = merged.shape[1], merged.shape[-1]
        merged = ttnn.reshape(merged, (s, r // v * vq, width))
        row_major = ttnn.permute(merged, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # (B * Vq, S, 1536)
        ttnn.deallocate(merged)
        if vq != v:
            # Drop the pad variates on an outer axis (the slice can alias its input: drop, never free it).
            row_major = ttnn.slice(ttnn.reshape(row_major, (r // v, vq, s, width)), (0, 0, 0, 0), (r // v, v, s, width))
        o = linear(to_flat(row_major), w["wo"], p)
        ttnn.deallocate(row_major)
        return self._post(o, w["post"], h1)

    def _var_attention_packed(self, w, xn, h1, inputs: TtInputs, mask):
        """R2-2: 32 / V items per 32-row tile, block-diagonal mask. The permute runs on the 1280-wide xn and o
        instead of the 4608-wide qkv and the 1536-wide context; consumes ``xn``, borrows ``h1``."""
        import ttnn

        p, r, s, v = self.precision, inputs.rows, inputs.padded_seq, inputs.variates
        slot_major = ttnn.permute(to_rows(xn, r, s), (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # (S, R, D)
        ttnn.deallocate(xn)
        qkv = linear(to_flat(slot_major), w["wqkv"], p)
        ttnn.deallocate(slot_major)
        tiles = s * r // TILE
        q, k, val = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.reshape(qkv, (tiles, TILE, qkv.shape[-1])),
            num_heads=NUM_HEADS,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q, k = self._qk_norm(q, k, w)
        merged = self._sdpa(q, k, val, mask, causal=False)  # (tiles, 32, 1536)
        o = linear(to_flat(merged), w["wo"], p)  # slot-major (S * R, D)
        ttnn.deallocate(merged)
        o_rows = ttnn.permute(ttnn.reshape(o, (s, r, o.shape[-1])), (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o)
        return self._post(to_flat(o_rows), w["post"], h1)

    def _block_mask(self, v: int, tiles: int):
        """(tiles, 1, 32, 32) additive block-diagonal mask (V x V blocks), cached per shape and kept alive."""
        key = (v, tiles)
        if key not in self._mask_cache:
            group = torch.arange(TILE) // v
            m = torch.where(group[:, None] == group[None, :], 0.0, MASK_FILL)
            self._mask_cache[key] = upload(self.device, m.expand(tiles, 1, TILE, TILE).contiguous())
        return self._mask_cache[key]

    def ffn(self, i: int, h2):
        """h2 + post_ln(ff1(relu(ff0(pre_ln(h2))))); ``h2`` is borrowed."""
        import ttnn

        p, layer = self.precision, self.layers[i]
        xn = self._pre(h2)
        cfg = p.matmul_config("ffn")
        hidden = linear(xn, layer["ff0"], p, activation="relu", compute_kernel_config=cfg)
        ttnn.deallocate(xn)
        o = linear(hidden, layer["ff1"], p, compute_kernel_config=cfg)
        ttnn.deallocate(hidden)
        return self._post(o, layer["post_ff"], h2)

    def layer(self, i: int, x, inputs: TtInputs, capture: dict | None = None, cut: bool = False):
        """One MixingTransformer layer; consumes ``x``. With ``cut``, the output holds only the read-out slots
        (R1-5): everything after the time attention's keys and values is per slot."""
        import ttnn

        h1 = self.seq_attention(i, x, inputs, cut=cut)
        if cut:
            inputs = dataclasses.replace(
                inputs,
                padded_seq=inputs.readout_count,
                var_mask=inputs.readout_var_mask,
                packed_var_mask=inputs.readout_packed_var_mask,
                readout_start=0,
            )
        ttnn.deallocate(x)
        h2 = self.var_attention(i, h1, inputs)
        if capture is not None:
            capture[f"layer{i}.h1"] = self.rows_to_torch(h1, inputs)
            capture[f"layer{i}.h2"] = self.rows_to_torch(h2, inputs)
        ttnn.deallocate(h1)
        y = self.ffn(i, h2)
        ttnn.deallocate(h2)
        return y

    def head(self, x, inputs: TtInputs):
        """Read-out tokens -> (1, 1, R * F, 576), normalized space; borrows ``x``."""
        import ttnn

        picked = take_slots(x, inputs.rows, inputs.padded_seq, inputs.readout_start, inputs.readout_count)
        out = linear(
            picked,
            self.head_w,
            self.precision,
            bias=self.head_b,
            compute_kernel_config=self.precision.matmul_config("head"),
        )
        ttnn.deallocate(picked)
        return out

    def forward_device(self, inputs: TtInputs, capture: dict | None = None):
        """Device forward: (1, 1, R * F, 576) head output at the read-out tokens."""
        import ttnn

        x = self.embed(inputs)
        if capture is not None:
            capture["transformer_input"] = self.rows_to_torch(x, inputs)
        last = len(self.layers) - 1
        for i in range(len(self.layers)):
            cut = i == last and self.cut_last_layer and capture is None
            x = self.layer(i, x, inputs, capture, cut=cut)
            if capture is not None:
                capture[f"layer{i}"] = self.rows_to_torch(x, inputs)
        if cut:
            inputs = dataclasses.replace(inputs, padded_seq=inputs.readout_count, readout_start=0)
        self._drop_pending_norm()
        out = self.head(x, inputs)
        ttnn.deallocate(x)
        return out

    # ------------------------------------------------------------------------------------------- host
    @staticmethod
    def rows_to_torch(x, inputs: TtInputs) -> torch.Tensor:
        """Flat device activations -> (R, n, D) fp32 (token padding cropped)."""
        import ttnn

        t = ttnn.to_torch(x).float().reshape(inputs.rows, inputs.padded_seq, -1)
        return t[:, : inputs.num_patches]

    @staticmethod
    def head_to_torch(out, inputs: TtInputs) -> torch.Tensor:
        import ttnn

        return ttnn.to_torch(out).float().reshape(inputs.rows, inputs.readout_count, -1)

    def postprocess(self, out, prepared: host_lib.HostInputs, inputs: TtInputs | None = None):
        """Device head output -> (head (R, F, 576) fp32, forecaster quantiles (numpy fp64))."""
        import ttnn

        raw = ttnn.to_torch(out).float().reshape(prepared.rows, prepared.num_forecast_patches, -1)
        return raw, host_lib.postprocess(raw, prepared)

    def __call__(self, contexts, horizon: int, **kw):
        """End to end: quantiles as ``TimesFM3Forecaster.predict_batch`` returns them (one batch)."""
        import ttnn

        prepared = self.prepare_inputs(contexts, horizon, **kw)
        inputs = self.upload_inputs(prepared)
        out = self.forward_device(inputs)
        _, quantiles = self.postprocess(out, prepared)
        ttnn.deallocate(out)
        self.deallocate_inputs(inputs)
        return quantiles
