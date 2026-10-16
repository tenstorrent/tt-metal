# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""APC_CHUNKED_PREFILL_PLAN.md M2/M3 integration (``Model.enable_apc``, tt/prefix_cache.PrefixCache), model level: 16 prompts that share a 700-token prefix (the few-shot header of the GSM8K benchmark) are
prefilled in ONE ``prefill_interleaved`` call (waves of one user per mesh row), once with prefix caching off and once on:

  * the first-token logits and tokens are BIT-IDENTICAL on / off for every prompt,
  * same-step donors: the first wave computes both windows and snapshots the boundary at 512, the other three waves restore it and compute one window (replays 5 instead of 8),
  * a second call with the same prompts restores every user,
  * an image-placeholder token anywhere in the prefix, ``DSV41_APC_FILE`` = 0 and ``reset_prefix_cache`` all bypass / empty the cache.

    DSV41_LAYERS=0-3 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_m3_integration.py -s
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

U = int(os.environ.get("DSV41_M3_U", "4"))
CHUNK = 512
LAYERS = os.environ.get("DSV41_LAYERS", "0-3")
PREFIX = int(os.environ.get("DSV41_M3_PREFIX", "700"))
TAIL = int(os.environ.get("DSV41_M3_TAIL", "200"))


def bits_equal(x, y):
    x, y = x.float().reshape(-1).contiguous(), y.float().reshape(-1).contiguous()
    return bool(torch.equal(x.view(torch.int32), y.view(torch.int32)))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_IL_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_shared_prefix_prompts_with_and_without_prefix_caching(mesh_device):
    os.environ.update({"DSV41_PF_UMASK": "1", "DSV41_PREFILL_UP": "1", "DSV41_MO_OVERLAP": "0"})
    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    e = PREFIX + TAIL
    S_pad = 2048  # (max_ctx > 1024 turns the indexer / sparse prefill on, the path with the kv tables)
    max_ctx = S_pad + 256
    pp = default_page_params(max_ctx, U)
    args, m, pool, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    apc = m.enable_apc(2 << 30, "pow2")
    g = torch.Generator().manual_seed(2024)
    shared = torch.randint(1000, 100000, (PREFIX,), generator=g, dtype=torch.int64)
    prompts = [
        torch.cat([shared, torch.randint(1000, 100000, (TAIL,), generator=g, dtype=torch.int64)]) for _ in range(B)
    ]

    def call(tag, users=range(B)):
        items = [(u, prompts[u], 0, e) for u in users]
        t0 = time.perf_counter()
        res = m.prefill_interleaved(items, CHUNK, s_pad_max=S_pad, want_logits=True)
        dt = time.perf_counter() - t0
        log(
            f"{tag}: {dt:.2f} s, replays {m.timing.get('replays')}, apc snapshot s {m.timing.get('apc_snapshot_s', 0):.2f}, restores so far {apc.stats['restores']}"
        )
        return {u: (int(res[u][0]), res[u][1].clone()) for u in users}, m.timing.get("replays")

    def decode(first, steps=6):
        """greedy decode of every user from its prefill state (the decode reads the pool latents the restore wrote): -> {user: [tokens]}"""
        cur = {u: first[u][0] for u in range(B)}
        out = {u: [cur[u]] for u in range(B)}
        for k in range(steps):
            tok = torch.tensor([cur[u] for u in range(B)], dtype=torch.long)
            pos = torch.full((B,), e + k, dtype=torch.long)
            nxt = m.decode_forward(tok, pos, enable_trace=True, reload_inputs=True).clone()
            for u in range(B):
                cur[u] = int(nxt[u])
                out[u].append(cur[u])
        return out

    def release():
        for u in range(B):
            m.release_user(u)

    apc.enabled = False
    call("warm-up (apc off, compiles and captures)")
    release()
    off, rep_off = call("APC OFF")
    dec_off = decode(off)
    release()
    apc.enabled = True
    snaps0, rest0 = apc.stats["snapshots"], apc.stats["restores"]
    on1, rep_on1 = call("APC ON, first call (same-step donors)")
    assert apc.stats["snapshots"] - snaps0 >= 1 and apc.stats["restores"] - rest0 == B - 4, apc.stats
    dec_on1 = decode(on1)
    assert dec_on1 == dec_off, "greedy decode after a restored prefill differs from the one after a full prefill"
    log(f"decode of {B} users x 6 steps after the restored prefill: identical tokens to APC off")
    release()
    on1b, _ = call("APC ON, store warm, 3rd call")
    assert decode(on1b) == dec_off
    log("decode after a fully restored prefill: identical")
    release()
    on2, rep_on2 = call("APC ON, second call (store warm)")
    assert apc.stats["restores"] - rest0 == (B - 4) + 2 * B, apc.stats
    for u in range(B):
        for tag, on in (("first", on1), ("second", on2)):
            assert on[u][0] == off[u][0], f"user {u} ({tag} call): token {on[u][0]} vs {off[u][0]}"
            assert bits_equal(on[u][1], off[u][1]), f"user {u} ({tag} call): first-token logits differ"
    log(
        f"all {B} users bit-identical on vs off; replays off {rep_off}, on first call {rep_on1}, on second call {rep_on2}"
    )
    assert rep_on1 < rep_off and rep_on2 < rep_on1

    # image placeholder in the prefix: bypass
    release()
    apc.image_token_id = int(prompts[0][10])
    r0 = apc.stats["restores"]
    call("APC ON with the image guard hitting every prompt", users=[0, 5])
    assert apc.stats["restores"] == r0 and apc.stats["skipped_vision"] > 0
    apc.image_token_id = None
    # live off switch
    release()
    open("/tmp/apc_off_switch", "w").write("0")
    os.environ["DSV41_APC_FILE"] = "/tmp/apc_off_switch"
    call("APC switched off by file", users=[0, 5])
    assert apc.stats["restores"] == r0
    os.environ.pop("DSV41_APC_FILE")
    # reset
    m.reset_prefix_cache()
    assert not apc.store.snaps and apc.store.bytes == 0
    release()
    call("APC ON after the reset (empty store, recomputes and snapshots again)", users=[0, 5])
    assert apc.stats["restores"] == r0 and len(apc.store.snaps) >= 1
    log(f"store: {len(apc.store.snaps)} snapshots, {apc.store.bytes / 2**20:.1f} MiB; stats {apc.stats}")
