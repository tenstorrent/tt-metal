# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: device time of one KDA layer (layer 0 weights, random input, chunk 5120 at start 5120) with ttKDA's prep
bf16 output mask 0x26 (default) vs 0x6 (final_decay fp32), interleaved in one process. Times (synced, wall) the whole
KDA attention, the recurrence (chunk preparation + scans + prefix) and chunk preparation alone.

    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_kda_op_time.py -s
"""

import time

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_kda_op_time(mesh_device):
    import ttnn
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v3_d_p.tt.kda import recurrence
    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig
    from models.demos.glm53_flash_d_p.reference.weights import WeightLoader
    from models.demos.glm53_flash_d_p.tt.common import split_from_host
    from models.demos.glm53_flash_d_p.tt.kda_attention import build_kda_attention

    S.hooks().apply_device_settings(S)
    path = hf_path(S)
    cfg = GlmConfig.from_json(f"{path}/config.json")
    attn = build_kda_attention(mesh_device, WeightLoader(path), cfg, 0, 56320)
    s = 5120
    torch.manual_seed(0)
    x = split_from_host(mesh_device, torch.randn(1, 1, s, cfg.hidden_size).to(torch.bfloat16))
    t_rec, t_prep = [], []
    orig_call = recurrence.KDARecurrence.__call__
    orig_prep = recurrence._prepare_chunk_terms

    def timed_call(self, **kw):
        ttnn.synchronize_device(mesh_device)
        t = time.perf_counter()
        r = orig_call(self, **kw)
        ttnn.synchronize_device(mesh_device)
        t_rec.append(time.perf_counter() - t)
        return r

    def timed_prep(*a, **kw):
        ttnn.synchronize_device(mesh_device)
        t = time.perf_counter()
        r = orig_prep(*a, **kw)
        ttnn.synchronize_device(mesh_device)
        t_prep.append(time.perf_counter() - t)
        return r

    recurrence.KDARecurrence.__call__ = timed_call
    recurrence._prepare_chunk_terms = timed_prep
    res = {}
    for rep in range(3):
        for mask in (0x26, 0x6):
            recurrence.KDA_PREP_OUTPUT_BF16_MASK = mask
            tot = []
            t_rec.clear()
            t_prep.clear()
            for it in range(12):
                ttnn.synchronize_device(mesh_device)
                t = time.perf_counter()
                y = attn(x, s, split=True)
                ttnn.synchronize_device(mesh_device)
                tot.append(time.perf_counter() - t)
                ttnn.deallocate(y)
            med = lambda v: sorted(v[2:])[len(v[2:]) // 2] * 1e3  # noqa
            res.setdefault(mask, []).append((med(tot), med(t_rec), med(t_prep)))
            print(
                f"[kt] rep {rep} mask {mask:#x}: KDA attention {med(tot):.3f} ms, recurrence {med(t_rec):.3f} ms, "
                f"prep {med(t_prep):.3f} ms (median of 10, synced)",
                flush=True,
            )
    recurrence.KDARecurrence.__call__ = orig_call
    recurrence._prepare_chunk_terms = orig_prep
