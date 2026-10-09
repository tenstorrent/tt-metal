# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder


def test_host_loop_runs_untraced(monkeypatch, expect_error):
    monkeypatch.setenv("GEMMA4_PLI", "host")
    decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
    decoder._use_trace = True
    decoder.target_has_pli = True
    decoder._pli_dev_host = False
    decoder._route = "host-loop"
    seen = []

    def seed(*args):
        seen.append(decoder._use_trace)
        raise RuntimeError("seed reached")

    decoder.seed = seed
    with expect_error(RuntimeError, "seed reached"):
        decoder.generate(1, 0, 1)
    assert seen == [False]
    assert decoder._use_trace


def test_fused_host_pli_rejected_before_device_work(expect_error):
    decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
    decoder.target_has_pli = True
    decoder._fused_pli_device = False
    decoder.seed = lambda *args: (_ for _ in ()).throw(AssertionError("device work started"))
    with expect_error(ValueError, "GEMMA4_PLI=device"):
        decoder.generate_fused(1, 0, 1)


def test_packed_fused_body_rejects_reseed(expect_error):
    decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
    decoder.target_has_pli = True
    decoder._fused_pli_device = True
    decoder._use_trace = True
    decoder._fused_reseed = True
    with expect_error(ValueError, "GEMMA4_SPEC_FUSED_RESEED"):
        decoder.generate_fused(1, 0, 1, packed=True)


def test_serving_rejects_pli_targets(expect_error):
    decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
    decoder.target_has_pli = True
    with expect_error(NotImplementedError, "per-layer-input"):
        decoder.serving_setup(1, 0, 1)
