# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""After ``warm_serving`` + ``warm_requests`` (APC on) the first real requests must compile NOTHING: the device program cache may not grow and misses are disallowed while they run (the approach of
qwen36's ``test_prefill_warmup_no_recompile``: ``set_program_cache_misses_allowed(False)`` raises, op-named, instead of compiling over a live trace). The requests cover the paths the first
users of a fresh server take: a lone prompt that is not a multiple of the chunk, prompts of several sizes with snapshots at every power-of-two boundary up to 32768, 4 users with different chunk
starts, a prefix hit (restore + continuation), the lazy key hand-off and traced decode steps. With TT_METAL_CACHE pointing at a cold directory the number of NEW kernel-cache directories is logged too
(``find $TT_METAL_CACHE -newer``): zero is expected for everything but the warm-up itself.

    DSV41_PF_SINGLE=1 DSV41_PF_PAGED=read DSV41_POOL_DTYPE=bf16 DSV41_APC=1 PROBE_LAYERS=0,1,2,3,14,20,21,24 pytest tests/test_warm_no_miss.py -s
(layer subset: warmed programs are layer independent; the layers include the Engram, indexer and kv-source layers.)
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "0,1,2,3,14,20,21,24").split(",")]
WMAX = int(os.environ.get("PROBE_WMAX", "65536"))
CHUNK = 512
B = 4


def _kernel_dirs():
    root = os.environ.get("TT_METAL_CACHE")
    if not root or not os.path.isdir(root):
        return None
    n = 0
    for dp, dn, _ in os.walk(root):
        n += len(dn)
    return n


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "650000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_warm_no_miss(mesh_device):
    md = mesh_device
    log = lambda m: print(m, flush=True)
    pp = default_page_params(WMAX, 1)
    args, m, _, _ = create_tt_model(
        md,
        B,
        WMAX,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=LAYERS,
        log=log,
    )
    assert m.Up == 1 and m.U == 1
    m.enable_apc(int(4 * 2**30))
    t0 = time.perf_counter()
    m.warm_serving(CHUNK, WMAX, [], [])
    t_warm = time.perf_counter() - t0
    strict = (
        os.environ.get("WARM_STRICT", "1") == "1"
    )  # 0: the "before" runs (warm-up steps switched off with DSV41_WARM_APC=0 DSV41_WARM_REQUESTS=0): time only, misses allowed
    dt_req, dn_req = (0.0, 0)
    if os.environ.get("DSV41_WARM_REQUESTS", "1") == "1":
        dt_req, dn_req = m.warm_requests(CHUNK)
    log(
        f"WARM serving {t_warm:.1f} s (incl. APC pre-compile), synthetic request {dt_req:.1f} s ({dn_req} new programs)"
    )
    assert (
        not strict or dn_req == 0
    ), f"the synthetic warm-up request compiled {dn_req} program(s) after the traces were captured"

    g = torch.Generator().manual_seed(11)
    rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g, dtype=torch.int64)  # noqa: E731
    n_prog = md.num_program_cache_entries()
    dirs0 = _kernel_dirs()
    md.set_program_cache_misses_allowed(not strict)
    lone = []
    try:
        # (1) a lone prompt that is not a multiple of the chunk (the 2k probe of the first-request penalty), twice (different text)
        for _ in range(2):
            t_l = time.perf_counter()
            m.prefill_interleaved([(0, rnd(2047), 0, 2047)], CHUNK, s_pad_max=WMAX, want_logits=True)
            lone.append(time.perf_counter() - t_l)
        log(f"WARM lone 2047-token prompt: first {lone[0]:.2f} s, second {lone[1]:.2f} s")
        # (2) one long prompt: snapshots at 512 .. 32768 (every size the pow2 policy takes) and a restore of each size below
        long_t = rnd(33000)
        m.prefill_interleaved([(1, long_t, 0, 33000)], CHUNK, s_pad_max=WMAX, want_logits=False)
        for hit_end in (700, 1300, 2600, 5000, 9000, 17000, 33000):  # prefix hits at 512, 1024, ..., 32768
            m.release_user(2)
            m.prefill_interleaved([(2, long_t, 0, hit_end)], CHUNK, s_pad_max=WMAX, want_logits=True)
        # (3) 4 users, different chunk starts in one call, then the final pieces (ragged ends)
        ts = [rnd(6000 + 700 * u) for u in range(4)]
        m.prefill_interleaved(
            [(u, ts[u], 0, 1024 * (u + 1)) for u in range(4)], CHUNK, s_pad_max=WMAX, want_logits=False
        )
        m.prefill_interleaved(
            [(u, ts[u], 1024 * (u + 1), ts[u].numel()) for u in range(4)], CHUNK, s_pad_max=WMAX, want_logits=True
        )
        # (4) traced decode steps of the 4 users (the lazy key hand-off runs before the first one)
        tok = torch.arange(B, dtype=torch.long) + 100
        pos = torch.tensor([t.numel() for t in ts], dtype=torch.long)
        for k in range(3):
            out = m.decode_forward(tok, pos + k, enable_trace=True, reload_inputs=True)
            tok = out.clone().long()
    finally:
        md.set_program_cache_misses_allowed(True)
    dn = md.num_program_cache_entries() - n_prog
    dirs1 = _kernel_dirs()
    log(
        f"WARM first real requests: program cache {n_prog} -> {n_prog + dn}"
        + ("" if dirs0 is None else f", kernel-cache directories {dirs0} -> {dirs1}")
    )
    assert not strict or dn == 0, f"{dn} program(s) compiled by the first real requests"
    if dirs0 is not None and strict:
        assert dirs1 == dirs0, f"{dirs1 - dirs0} new kernel-cache directories during the first real requests"
    log("WARM NO MISS PASSED")
