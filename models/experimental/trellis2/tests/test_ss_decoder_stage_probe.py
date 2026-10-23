# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: the TT structure decoder stage by stage against `ss_decoder_host`, each stage fed the TT stream so
an error shows up at the stage that makes it (not where it accumulates). Logs PCC and scale <t, g> / <g, g> per
stage; asserts nothing.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.trellis2.reference import ss_decoder_host as host
from models.experimental.trellis2.tests import goldens
from models.experimental.trellis2.tt import ss_decoder as tt

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def _cmp(name: str, t: torch.Tensor, g: torch.Tensor) -> None:
    t, g = t.float().flatten(), g.float().flatten()
    logger.info(
        f"stage {name}: pcc {goldens.pcc(t, g):.6f}  scale {(t @ g / (g @ g)).item():+.6f}  "
        f"max_abs {(t - g).abs().max().item():.3e}  |g| {g.abs().max().item():.3e}"
    )


def _host(x: ttnn.Tensor, c: int | None = None) -> torch.Tensor:
    y = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
    return y[..., :c] if c else y


def _up(x: torch.Tensor, device, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ss_decoder_stage_probe(mesh_device):
    state = host.load_state_dict(f"{goldens.weights_dir()}/{host.CKPT}")
    s = {k: v.float() for k, v in state.items()}
    dec = tt.TtSSDecoder(state, mesh_device=mesh_device)
    z, gold, _ = goldens.ss_decoder_call()

    x_host = z.float().permute(0, 2, 3, 4, 1)
    x = _up(tt._pad_channels(x_host, 4), mesh_device, dec.dtypes[tt.STAGES[0]])
    _cmp("input_upload", _host(x, host.LATENT_CHANNELS), x_host)

    g = host._conv(x_host, s["input_layer.weight"], s["input_layer.bias"])
    y = dec.input_layer(x)
    _cmp("input_layer(conv 8->512)", _host(y), g)
    dhw = tuple(y.shape)[1:4]
    h = tt._tokens(y)
    _cmp("tokens/grid round trip", _host(tt._grid(h, dhw)), _host(y))

    def res_probe(name, block, p, h, dhw):
        hin = _host(tt._grid(h, dhw))
        n1 = block.norm1(h)
        _cmp(f"{name}.norm1", _host(tt._grid(n1, dhw)), host._norm_silu(hin, s, p + "norm1"))
        c1 = block.conv1(tt._grid(n1, dhw))
        _cmp(
            f"{name}.conv1", _host(c1), host._conv(_host(tt._grid(n1, dhw)), s[p + "conv1.weight"], s[p + "conv1.bias"])
        )
        out = block(h, dhw)
        _cmp(f"{name}.out", _host(tt._grid(out, dhw)), host._res(hin, s, p))
        return out

    for i, block in enumerate(dec.middle):
        h = res_probe(f"middle.{i}", block, f"middle_block.{i}.", h, dhw)
    for i, ((kind, block), (_, _, _)) in enumerate(zip(dec.blocks, host.block_layout())):
        if kind == "res":
            h = res_probe(f"blocks.{i}", block, f"blocks.{i}.", h, dhw)
        else:
            hin = _host(tt._grid(h, dhw))
            w, b = host.shuffle_weight_order(s[f"blocks.{i}.conv.weight"], s[f"blocks.{i}.conv.bias"])
            c = block(tt._grid(h, dhw))
            _cmp(f"blocks.{i}.upconv", _host(c), host._conv(hin, w, b))
            sh = tt._shuffle(c)
            _cmp(f"blocks.{i}.shuffle", _host(sh), host.shuffle_ndhwc(_host(c)))
            dhw = tuple(sh.shape)[1:4]
            h = tt._tokens(sh)
            if h.dtype != dec.dtypes[dhw[0]]:
                h = ttnn.typecast(h, dec.dtypes[dhw[0]])
    hin = _host(tt._grid(h, dhw))
    n = dec.out_norm(h)
    _cmp("out_norm", _host(tt._grid(n, dhw)), host._norm_silu(hin, s, "out_layer.0"))
    o = dec.out_conv(tt._grid(n, dhw))
    ref = host._conv(_host(tt._grid(n, dhw)), s["out_layer.2.weight"], s["out_layer.2.bias"])
    _cmp("out_conv(32->1)", _host(o, host.OUT_CHANNELS), ref)
    logger.info(f"out_conv tt shape {tuple(o.shape)} ref {tuple(ref.shape)}")
    _cmp("final vs golden", _host(o, host.OUT_CHANNELS).permute(0, 4, 1, 2, 3), gold)
    _cmp("mirror vs golden", host.functional_forward(state, z), gold)
