# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TiRex-2 forecaster on one Blackhole chip.

Host/device split (the bench measures each part):
    prepare_inputs   host: pad to 9120 steps, scaler, patch tokens, row packing (32-row blocks)
    upload_inputs    host -> device: tokens [rows, 1, 320, 64] bf16, variate masks
    forward_device   embed, 12 blocks, final norm, head; returns the head output on a tile-aligned token window
    postprocess      host: read-out tokens, inverse scaling, band calibration, target rows

Pad skip (R1-1): when every row starts with at least 64 all-NaN pad tokens (context <= 6144 steps), the states
after the first ``cut`` pad tokens (a multiple of 64) are the same for every row and input (``pad_check.json``).
They are computed once per ``cut`` on the host by the fp32 dataflow and kept as device constants, and the device
runs only tokens ``cut .. 284`` (padded to a multiple of 64). Variate (grouped) batches keep the full path.

Rows: the device batch is padded to a multiple of 32 with all-NaN dummy rows; for multivariate input every
group (<= 32 rows) sits inside one 32-row block, so variate attention runs on 32 x 32 blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tt import layers
from models.experimental.tirex2_forecast.tt import weights as tw
from models.experimental.tirex2_forecast.tt.common import TILE, Precision, pad_to, upload

SEQ_CHUNK = 64
SEQ = pad_to(df.NUM_TOKENS, SEQ_CHUNK)  # 320: five mLSTM chunks
WINDOW = (224, 288)  # tile-aligned token window holding every read-out token (255 .. 283)
ROW_BLOCK = 32
PAD_CUT_MIN = 64  # below this the full path runs
FUTURE_TOKENS = df.FUTURE_LEN // df.PATCH  # 29 all-NaN future tokens at the end of every row


def pad_cut(tokens: torch.Tensor) -> int:
    """Leading tokens that are all-NaN pad (``[0 x 32 | 0 x 32]``) in every row, rounded down to a multiple of
    ``SEQ_CHUNK``; 0 when that is below ``PAD_CUT_MIN``."""
    # the pad states are input-independent only while nothing reads ahead: no known-future covariates
    assert not bool(tokens[:, df.NUM_TOKENS - FUTURE_TOKENS : df.NUM_TOKENS].any()), "future tokens must be pad"
    used = tokens[:, : df.NUM_TOKENS].ne(0).any(-1).any(0)
    first = int(used.nonzero()[0]) if bool(used.any()) else df.NUM_TOKENS - FUTURE_TOKENS
    cut = first // SEQ_CHUNK * SEQ_CHUNK
    return cut if cut >= PAD_CUT_MIN else 0


def seq_len(cut: int) -> int:
    """Device sequence length of a forward that skips ``cut`` pad tokens."""
    return pad_to(df.NUM_TOKENS - cut, SEQ_CHUNK)


@dataclass
class Prepared:
    tokens: torch.Tensor  # [B, SEQ, 64] device row order
    loc: torch.Tensor  # [rows, 1] source order
    scale: torch.Tensor
    order: torch.Tensor  # [B] source row of each device row, -1 for dummy rows
    rows: int
    horizon: int
    target_mask: torch.Tensor  # [rows] bool, source order
    vmask: torch.Tensor | None  # [1, nb * 4, 32, 32] additive mask, or None when no group has > 1 row
    keep: torch.Tensor | None  # [1, nb, 32, 1]
    cut: int = 0  # leading pad tokens skipped on device (0: full path)

    @property
    def device_rows(self) -> int:
        return self.order.numel()


@dataclass
class DeviceInputs:
    tokens: object
    vmask: object | None
    keep: object | None
    cut: int = 0


def pack_rows(group_ids: torch.Tensor, block: int = ROW_BLOCK) -> torch.Tensor:
    """Device row order (source index or -1): groups kept whole inside ``block``-row blocks, total a multiple
    of ``block``. ``group_ids`` must list each group's rows consecutively.

    Limit: a group (target rows plus covariates) may have at most 32 rows, because variate attention runs on
    32 x 32 row blocks; larger groups raise ``ValueError``."""
    rows = group_ids.numel()
    _, counts = torch.unique_consecutive(group_ids, return_counts=True)
    if bool((counts == 1).all()):
        return torch.cat([torch.arange(rows), torch.full((pad_to(rows, block) - rows,), -1)])
    if int(counts.max()) > block:
        raise ValueError(f"groups of more than {block} rows are not supported (largest: {int(counts.max())})")
    order, start = [], 0
    for c in counts.tolist():
        if len(order) % block + c > block:
            order += [-1] * (block - len(order) % block)
        order += list(range(start, start + c))
        start += c
    order += [-1] * ((-len(order)) % block)
    return torch.tensor(order)


def variate_masks(order: torch.Tensor, group_ids: torch.Tensor, target_mask: torch.Tensor):
    """``(vmask [1, nb*4, 32, 32], keep [1, nb, 32, 1])`` for the device row order, or ``(None, None)``."""
    B = order.numel()
    valid = order >= 0
    gid = torch.full((B,), -1, dtype=torch.long)
    gid[valid] = group_ids[order[valid]].long()
    gid[~valid] = -(torch.arange(B)[~valid] + 2)  # every dummy row is its own group
    tm = torch.zeros(B, dtype=torch.bool)
    tm[valid] = target_mask[order[valid]]
    keep, allowed = df.group_layout(gid, tm)
    if not bool(keep.any()):
        return None, None
    nb = B // ROW_BLOCK
    blocks = torch.stack(
        [allowed[i * ROW_BLOCK : (i + 1) * ROW_BLOCK, i * ROW_BLOCK : (i + 1) * ROW_BLOCK] for i in range(nb)]
    )
    # a group never straddles blocks, so the off-block entries of `allowed` are all False
    vmask = torch.where(blocks, 0.0, layers.MASK_NEG)  # [nb, 32, 32]
    vmask = vmask[:, None].expand(nb, df.HEADS, ROW_BLOCK, ROW_BLOCK).reshape(1, nb * df.HEADS, ROW_BLOCK, ROW_BLOCK)
    return vmask.contiguous(), keep.float().reshape(1, nb, ROW_BLOCK, 1)


class TtTiRex2:
    def __init__(
        self,
        device,
        host_weights: df.TiRex2W,
        precision: Precision | None = None,
        variate: bool = True,
        pad_skip: bool = True,
    ):
        self.device = device
        self.precision = precision or Precision.accuracy()
        self.host = host_weights
        self.weights = tw.build(device, host_weights, self.precision, variate=variate)
        self.consts = layers.DeviceConsts(device, SEQ_CHUNK)
        self.step_hook = None  # profiling: called after every sLSTM step
        self.pad_skip = pad_skip
        self._pad_states: dict = {}  # cut -> per-block host states (fp32 dataflow on one all-pad row)

    @classmethod
    def from_checkpoint(cls, device, precision: Precision | None = None, path=None, variate: bool = True):
        sd = df.load_state_dict(path or df.default_checkpoint())
        return cls(device, df.build_weights(sd), precision, variate=variate)

    # ------------------------------------------------------------------------------------- host side

    def prepare_inputs(self, context: torch.Tensor, horizon: int, group_ids=None, target_mask=None) -> Prepared:
        """``context [rows, T]`` (targets and past covariates, each group's rows consecutive)."""
        rows = context.shape[0]
        group_ids = torch.arange(rows) if group_ids is None else torch.as_tensor(group_ids)
        target_mask = torch.ones(rows, dtype=torch.bool) if target_mask is None else torch.as_tensor(target_mask)
        p = df.prepare(context, horizon, group_ids, target_mask)
        order = pack_rows(group_ids)
        tokens = torch.zeros(order.numel(), SEQ, 2 * df.PATCH)
        valid = order >= 0
        tokens[valid, : df.NUM_TOKENS] = p.tokens[order[valid]]
        vmask, keep = variate_masks(order, group_ids, target_mask)
        cut = pad_cut(tokens) if self.pad_skip and vmask is None else 0
        return Prepared(tokens, p.loc, p.scale, order, rows, horizon, target_mask, vmask, keep, cut)

    def upload_inputs(self, prep: Prepared, host: bool = False) -> DeviceInputs:
        """Device inputs (``host=True``: the same tensors on host, to copy into a traced forward's inputs)."""
        import ttnn

        device = None if host else self.device
        if prep.cut:
            seq = seq_len(prep.cut)
            tokens = upload(device, prep.tokens[:, prep.cut : prep.cut + seq].reshape(prep.device_rows, 1, seq, -1))
            return DeviceInputs(tokens, None, None, prep.cut)
        tokens = upload(device, prep.tokens.reshape(prep.device_rows, 1, SEQ, -1))
        if prep.vmask is None:
            return DeviceInputs(tokens, None, None)
        return DeviceInputs(tokens, upload(device, prep.vmask, dtype=ttnn.float32), upload(device, prep.keep))

    def postprocess(self, out, prep: Prepared) -> torch.Tensor:
        """Head window ``[B, 1, 64, 288]`` (device) -> single-pass forecast ``[targets, 9, H]``."""
        import ttnn

        win = ttnn.to_torch(out).float().reshape(prep.device_rows, WINDOW[1] - WINDOW[0], -1)
        readout = [t - WINDOW[0] for t in df.readout_tokens(prep.horizon)]
        head = torch.empty(prep.rows, len(readout), win.shape[-1])
        valid = prep.order >= 0
        head[prep.order[valid]] = win[valid][:, readout]
        q = df.band_calibration(df.head_to_quantiles(head, prep.loc, prep.scale, prep.horizon))
        return q[prep.target_mask]

    @staticmethod
    def deallocate_inputs(inputs: DeviceInputs):
        layers.free(inputs.tokens, inputs.vmask, inputs.keep)

    def host_pad_states(self, cut: int) -> list:
        """Per block, the state after ``cut`` all-pad tokens, from the fp32 dataflow on one all-pad row:
        mLSTM ``(C [H, 64, 128], n [H, 64], m [H])``, sLSTM ``(y, c, n, m)`` each ``[512]``."""
        if cut not in self._pad_states:
            assert cut % SEQ_CHUNK == 0
            W = self.host
            x = df.residual_block(torch.zeros(1, cut, 2 * df.PATCH), W.embed)
            states = []
            for blk in W.blocks:
                xn = df.rms_norm(x, blk.w_norm_lstm)
                if df.is_mlstm(blk.index):
                    cap = {}
                    y = df.mlstm_cell(xn, blk.cell, chunk=SEQ_CHUNK, capture=cap)
                    _, st = df.mlstm_chunkwise(
                        cap["q"], cap["k"], cap["v"], cap["i"], cap["f"], chunk=SEQ_CHUNK, return_states=True
                    )
                    states.append(tuple(t[0] for t in st[-1]))
                else:
                    Wx = df.slstm_input(xn, blk.cell)
                    yy, sts = df.slstm_recurrence(Wx, blk.cell.r)
                    y = df.mh_layer_norm(yy, blk.cell.w_gnorm)
                    states.append(tuple(t[0] for t in sts[-1]))
                x = x + y
                x = x + df.ffn(df.rms_norm(x, blk.w_norm_ffn), blk.ffn)
            self._pad_states[cut] = states
        return self._pad_states[cut]

    def pad_state(self, cut: int, B: int, block: tw.TtBlock):
        """Device starting state of ``block`` for ``B`` rows after ``cut`` pad tokens (cached constants)."""
        st = self.host_pad_states(cut)[block.index]
        key = (cut, B, block.index)
        if df.is_mlstm(block.index):
            C, n, m = st
            H = C.shape[0]
            rep = lambda t, shape: t.unsqueeze(0).expand(B, *t.shape).reshape(B * H, *shape).contiguous()
            return (
                self.consts.tensor((*key, "C"), rep(C, (1, *C.shape[1:]))),
                self.consts.tensor((*key, "n"), rep(n, (1, n.shape[1], 1))),
                self.consts.tensor((*key, "m"), rep(m, (1, 1, 1))),
            )
        return tuple(
            self.consts.tensor((*key, nm), t.reshape(1, 1, 1, -1).expand(1, 1, B, t.numel()).contiguous())
            for nm, t in zip("ycnm", st)
        )

    # ------------------------------------------------------------------------------------ device side

    def release(self):
        """Free the per-device constants (call before closing the device if the process keeps running)."""
        self.consts.release()

    def embed(self, inputs: DeviceInputs):
        return layers.residual_block(inputs.tokens, self.weights.embed, self.precision, "embed")

    def run_block(self, block: tw.TtBlock, x, inputs: DeviceInputs):
        init = self.pad_state(inputs.cut, x.shape[0], block) if inputs.cut else None
        if df.is_mlstm(block.index):
            y = layers.mlstm_layer(self.device, x, block.cell, self.precision, self.consts, init=init)
        else:
            y = layers.slstm_layer(
                self.device,
                x,
                block.cell,
                self.precision,
                self.weights.head_mean,
                df.NUM_TOKENS - inputs.cut,
                self.consts,
                step_hook=self.step_hook,
                init=init,
            )
        out = layers.ffn_residual(y, block.ffn, self.precision)
        layers.free(y)
        if inputs.vmask is not None:
            mixed = layers.variate_block(self.device, out, block.variate, self.precision, inputs.vmask, inputs.keep)
            layers.free(out)
            out = mixed
        return out

    def head(self, x, cut: int = 0):
        import ttnn

        xn = layers.rms_norm(x, df.OUT_NORM_EPS)
        h = layers.residual_block(xn, self.weights.head, self.precision, "head")
        layers.free(xn)
        B = h.shape[0]
        out = ttnn.slice(h, (0, 0, WINDOW[0] - cut, 0), (B, 1, WINDOW[1] - cut, h.shape[-1]))
        layers.free(h)
        return out

    def forward_device(self, inputs: DeviceInputs, blocks=None):
        x = self.embed(inputs)
        for block in self.weights.blocks if blocks is None else blocks:
            y = self.run_block(block, x, inputs)
            layers.free(x)
            x = y
        out = self.head(x, inputs.cut)
        layers.free(x)
        return out

    # ---------------------------------------------------------------------------------------- end to end

    def predict(self, context: torch.Tensor, horizon: int, group_ids=None, target_mask=None) -> torch.Tensor:
        prep = self.prepare_inputs(context, horizon, group_ids, target_mask)
        inputs = self.upload_inputs(prep)
        out = self.forward_device(inputs)
        self.deallocate_inputs(inputs)
        q = self.postprocess(out, prep)
        layers.free(out)
        return q

    def forecast(self, context: torch.Tensor, horizon: int, group_ids=None, target_mask=None, sign_flip=True):
        """Forecast ``[targets, 9, H]``; with ``sign_flip`` the ``x`` and ``-x`` passes run as one batch."""
        if not sign_flip:
            return self.predict(context, horizon, group_ids, target_mask)
        rows = context.shape[0]
        gid = torch.arange(rows) if group_ids is None else torch.as_tensor(group_ids)
        tm = torch.ones(rows, dtype=torch.bool) if target_mask is None else torch.as_tensor(target_mask)
        both = torch.cat([context, -context])
        q = self.predict(both, horizon, torch.cat([gid, gid + int(gid.max()) + 1]), torch.cat([tm, tm]))
        n = int(tm.sum())
        return df.sign_flip_average(q[:n], q[n:])


__all__ = [
    "TtTiRex2",
    "Prepared",
    "DeviceInputs",
    "pack_rows",
    "variate_masks",
    "pad_cut",
    "seq_len",
    "SEQ",
    "WINDOW",
    "TILE",
]
