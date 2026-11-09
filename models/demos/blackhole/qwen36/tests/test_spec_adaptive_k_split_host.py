# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_SPEC_ADAPTIVE_K on the serving path (r12h2): the split fused step (decode_forward_begin /
decode_forward_finish, the device grammar mask path of supports_spec_device_mask) and a wide step's spec_sampling.

Bugs these pin (first device run of r12h: server ready, then every request failed):
  * decode_forward_finish ran the step's second half in whichever family was live after the begin returned (the base one)
    and with the runner block's mask width [m, 1 + K_api, W]; it now runs in the begin's family with the mask cut to it;
  * Qwen36MTPContractForCausalLM._check_mask required [m, 1 + K, W] (K = the engine's base K) although the runner block is
    [B, 1 + K_api] under the flag;
  * _ak_run_wide gathered every spec_sampling value as a [B] tensor: the penalties dict and a short (m < B) mask raised.
No device.
"""

import types

import torch

from models.demos.blackhole.qwen36.tt import spec_adaptive as sa

B, K_BASE, K_WIDE, b, W = 8, 3, 5, 4, 3


def _mask(m, cols=1 + K_WIDE):
    x = torch.arange(m * cols * W, dtype=torch.int32).reshape(m, cols, W) + 1  # no all-zero row
    return x


def test_mask_base_cuts_columns_keeps_rows():
    m = _mask(5)
    out = sa.ak_mask_for_step(m, False, K_BASE, b, B)
    assert tuple(out.shape) == (5, 1 + K_BASE, W)
    assert torch.equal(out, m[:, : 1 + K_BASE])


def test_mask_base_flag_width_passthrough():
    m = _mask(8, 1 + K_BASE)
    assert sa.ak_mask_for_step(m, False, K_BASE, b, B) is m


def test_mask_wide_short_mask_padded_unconstrained():
    m = _mask(2)  # the plugin sent 2 users only
    out = sa.ak_mask_for_step(m, True, K_WIDE, b, B)
    assert tuple(out.shape) == (b, 1 + K_WIDE, W)
    assert torch.equal(out[:2], m)
    assert bool((out[2:] == -1).all())  # rows the plugin did not send: no constraint


def test_mask_wide_gathers_through_slot_map():
    m = _mask(B)
    phys = [3, 0, 7, 1, 2, 4, 5, 6]  # runner slot -> engine row
    inv = [0] * B
    for s, r in enumerate(phys):
        inv[r] = s
    out = sa.ak_mask_for_step(m, True, K_WIDE, b, B, perm=(phys, inv))
    for r in range(b):
        assert torch.equal(out[r], m[inv[r]])


def _g_identity(x):
    return torch.as_tensor(x)[:b].contiguous()


def test_wide_sampling_penalties_and_rebuild():
    ss = {
        "temperature": torch.ones(B),
        "uniforms": torch.rand(B, K_WIDE + 2),
        "greedy": torch.zeros(B, dtype=torch.bool),
        "penalties": {
            "repetition": torch.ones(B),
            "frequency": torch.zeros(B),
            "presence": torch.zeros(B),
            "rebuild": {1: ("p", "o"), 6: ("p6", "o6")},
        },
        "mask": _mask(3),
    }
    phys = list(range(B))
    out = sa.ak_wide_sampling(ss, _g_identity, phys, K_WIDE, b, B)
    assert tuple(out["temperature"].shape) == (b,)
    assert tuple(out["uniforms"].shape) == (b, K_WIDE + 2)
    assert tuple(out["penalties"]["repetition"].shape) == (b,)
    assert out["penalties"]["rebuild"] == {1: ("p", "o")}  # slot 6 is past the wide width
    assert tuple(out["mask"].shape) == (b, 1 + K_WIDE, W)


def test_wide_sampling_none_and_scalars_pass():
    ss = {"temperature": torch.ones(B), "penalties": None, "mask": None, "flag": 3}
    out = sa.ak_wide_sampling(ss, _g_identity, list(range(B)), K_WIDE, b, B)
    assert out["penalties"] is None and out["mask"] is None and out["flag"] == 3


class _Fam(sa.KFamily):
    pass


def test_finish_routes_through_handle_family():
    """decode_forward_finish with the flag on runs in the handle's family with the cut mask (stubbed engine)."""
    base, wide = _Fam("base", B, K_BASE), _Fam("wide", b, K_WIDE)
    seen = {}

    class Eng(sa.AdaptiveKMixin):
        def __init__(self):
            self._ak = sa.AdaptiveKState(base, wide)
            self.B = B

        import contextlib

        @contextlib.contextmanager
        def _ak_handle_ctx(self, h):
            seen["ctx"] = h.ak_fam
            self._ak.depth += 1
            try:
                yield
            finally:
                self._ak.depth -= 1

        def decode_forward_finish(self, handle, mask=None):
            if not self._ak_inside():
                return self._ak_decode_forward_finish(handle, mask)
            seen["mask"] = mask
            return "ok"

    e = Eng()
    h = types.SimpleNamespace(ak_fam=base, ak_api=dict(perm=None, B=B, W=1 + K_WIDE))
    assert e.decode_forward_finish(h, _mask(B)) == "ok"
    assert seen["ctx"] is base and tuple(seen["mask"].shape) == (B, 1 + K_BASE, W)
    h = types.SimpleNamespace(ak_fam=wide, ak_api=dict(perm=None, B=B, W=1 + K_WIDE))
    assert e.decode_forward_finish(h, _mask(2)) == "ok"
    assert seen["ctx"] is wide and tuple(seen["mask"].shape) == (b, 1 + K_WIDE, W)
    assert e.decode_forward_finish(h, None) == "ok" and seen["mask"] is None
