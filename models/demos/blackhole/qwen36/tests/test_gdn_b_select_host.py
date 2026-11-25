# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the GDN prefill's one-hot selection of b (tt/gdn/tp.py TPGatedDeltaNet._build_b_select,
_select_b, _project_qkvzab, forward_prefill; QWEN36_PREFILL_GDN_GLUE).

* The one-hot of _build_b_select returns b = ab[..., Nv:2*Nv] bit for bit under fp32 accumulation; only a -0.0 may come
  back +0.0, which the sigmoid maps to the same value. An [a | b] block wider than one tile gets no one-hot.
* _project_qkvzab(select_b=True) takes b with _select_b; select_b=False (decode, verify, batched prefill) and a missing
  one-hot keep the slice. forward_prefill asks for select_b.
* QWEN36_PREFILL_GDN_GLUE defaults to 0.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

QZ, AZ = 1280, 2048  # Qwen3.8-27B at TP=8: q|k|v and q|k|v|z widths per device


def _one_hot(monkeypatch, Nv):
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    gdn = TPGatedDeltaNet.__new__(TPGatedDeltaNet)
    gdn.Nv, gdn.mesh = Nv, None
    return gdn._build_b_select()


@pytest.mark.parametrize("Nv", [6, 12, 16])
def test_the_one_hot_returns_b_bit_for_bit(monkeypatch, Nv):
    sel = _one_hot(monkeypatch, Nv)
    assert sel.shape == (1, 2 * Nv, Nv) and sel.dtype == torch.bfloat16
    torch.manual_seed(Nv)
    ab = (torch.randn(1, 4096, 2 * Nv) * torch.logspace(-30, 30, 4096).reshape(1, -1, 1)).to(torch.bfloat16)
    ab[0, :8] = float("-0.0")
    got = (ab.float() @ sel.float()).to(torch.bfloat16)
    want = ab[..., Nv:]
    nz = want != 0
    assert torch.equal(got[nz], want[nz])
    assert bool((got[~nz] == 0).all())
    assert torch.equal(torch.sigmoid(got.float()), torch.sigmoid(want.float()))


def test_an_ab_block_wider_than_a_tile_gets_no_one_hot(monkeypatch):
    assert _one_hot(monkeypatch, 17) is None


class FT:
    def __init__(self, shape, name):
        self.shape, self.name = tuple(shape), name


@pytest.fixture
def projection(monkeypatch):
    """A TPGatedDeltaNet on the fused prefill projection path; ttnn.slice and ttnn.matmul return named stand-ins."""
    calls = []
    monkeypatch.setattr(tpc, "use_small_m", lambda args, S: False)
    monkeypatch.setattr(tpc, "agmm_config_override", lambda args, S, name: None)
    monkeypatch.setattr(
        tpc, "all_gather_matmul_prefill", lambda x, w, *a, **kw: FT((1, 1, x.shape[-2], AZ + 12), "qkvzab")
    )
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: FT(shape, t.name))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)

    def slice_(t, start, end, memory_config=None):
        calls.append(("slice", t.name, start[-1], end[-1], memory_config))
        return FT(end[:-1] + (end[-1] - start[-1],), f"{t.name}[{start[-1]}:{end[-1]}]")

    def matmul(a, b, **kw):
        calls.append(("matmul", a.name, b, kw))
        return FT(a.shape[:-1] + (6,), "b-by-one-hot")

    monkeypatch.setattr(ttnn, "slice", slice_)
    monkeypatch.setattr(ttnn, "matmul", matmul)
    gdn = TPGatedDeltaNet.__new__(TPGatedDeltaNet)
    gdn.Nv, gdn.qkv_dim_tp, gdn.qkvz_dim_tp = 6, QZ, AZ
    gdn._fuse_ab = gdn._fuse_agmm = True
    gdn.args = SimpleNamespace(prefill_agmm_bf8=False, ccl_topology=lambda: None)
    gdn.tw, gdn.tt_ccl, gdn.cfg = {"qkvz": "w_qkvz"}, None, "hifi2"
    gdn._cfg_onehot = "onehot"
    gdn._b_select = "b_select"
    return gdn, calls


def test_select_b_takes_b_with_the_one_hot(projection):
    gdn, calls = projection
    qkv, z, a, b = gdn._project_qkvzab(FT((1, 1, 2048, 640), "x"), 2048, out_mc="L1", select_b=True)
    assert b.name == "b-by-one-hot"
    assert a.name == f"qkvzab[{AZ}:{AZ + 12}][0:6]"
    assert calls[-1] == (
        "matmul",
        f"qkvzab[{AZ}:{AZ + 12}]",
        "b_select",
        dict(compute_kernel_config="onehot", memory_config="L1", dtype=ttnn.bfloat16),
    )
    assert not any(c[0] == "slice" and c[2] == 6 for c in calls)


@pytest.mark.parametrize("select_b, one_hot", [(False, "b_select"), (True, None)], ids=["not-asked", "no-one-hot"])
def test_the_projection_slices_b_otherwise(projection, select_b, one_hot):
    gdn, calls = projection
    gdn._b_select = one_hot
    qkv, z, a, b = gdn._project_qkvzab(FT((1, 1, 2048, 640), "x"), 2048, out_mc="L1", select_b=select_b)
    assert b.name == f"qkvzab[{AZ}:{AZ + 12}][6:12]"
    assert not any(c[0] == "matmul" for c in calls)


class _Projected(Exception):
    pass


def test_forward_prefill_asks_for_select_b(monkeypatch, expect_error):
    gdn = TPGatedDeltaNet.__new__(TPGatedDeltaNet)
    gdn.tw, gdn.Nk, gdn.Nv, gdn.Dk, gdn.Dv = {}, 2, 6, 128, 128
    gdn.conv_carry = "carry"
    monkeypatch.setattr(tpc, "prefill_act_mem", lambda T, l1: "L1")

    def project(x, S, **kw):
        raise _Projected(S, kw)

    gdn._project_qkvzab = project
    with expect_error(_Projected, "select_b") as e:
        gdn.forward_prefill(FT((1, 2048, 640), "x"))
    assert e.value.args == (2048, dict(out_mc="L1", select_b=True))


def test_the_one_hot_is_off_by_default(monkeypatch):
    monkeypatch.delenv(tpc.PREFILL_GDN_GLUE_ENV, raising=False)
    assert not tpc.prefill_gdn_b_select_enabled()
    monkeypatch.setenv(tpc.PREFILL_GDN_GLUE_ENV, "1")
    assert tpc.prefill_gdn_b_select_enabled()
