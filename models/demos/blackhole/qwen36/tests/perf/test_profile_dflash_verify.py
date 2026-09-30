# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy target for the traced 16-row DFlash verify replay (no drafter in the capture).

Builds the 27B, prefills a short prompt, captures the verify trace (so the capture's named ops are
in the profile) and replays it between ``start`` and ``stop`` signposts. The op names of the
replayed ops come from the capture, which is why the capture is inside the profiled run.

Run (T3K)::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
    DFLASH_RUN_TARGET=1 python -m tracy -p --op-support-count 100000 -r -v -m \\
      pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_dflash_verify.py"
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.demo.dflash_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.dflash.config import paged_blocks_for

PROMPT_LEN = 64


def _signpost(name: str) -> None:
    try:
        from tracy import signpost

        signpost(name)
    except ImportError:
        pass


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_profile_dflash_verify(mesh_device, device_params, reset_seeds, ensure_gc):
    """Eager verify over a truncated stack (3 GDN + 1 attention layers): every op is named.

    The 64-layer stack repeats this 3:1 pattern 16 times, so one attention layer's ops times 16 plus
    one GDN layer's times 48 (plus the head) is the full traced verify's device work.
    """
    del device_params
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    num_blocks = paged_blocks_for(PROMPT_LEN + 64)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=num_blocks * 64, n_layers=4)
    model.allocate_kv_caches(
        [num_blocks, model.args.n_local_kv_heads, 64, model.args.head_dim], ttnn.bfloat16, batch_size=1
    )
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    gen = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, model.args.vocab_size, (1, PROMPT_LEN), generator=gen)
    block = torch.randint(0, model.args.vocab_size, (16,), generator=gen).tolist()

    model._reset_gdn_state_for_new_sequence()
    model.prefill_for_spec(prompt.to(torch.int32), page_table, PROMPT_LEN, lambda *a: None)
    model.prepare_verify(page_table, 16, decode_cfg=True)
    for _ in range(2):  # compile
        model.verify_traced(block, PROMPT_LEN, page_table=page_table)
    ttnn.synchronize_device(mesh_device)

    _signpost("start")
    for _ in range(2):
        model.verify_traced(block, PROMPT_LEN, page_table=page_table)
        ttnn.synchronize_device(mesh_device)
    _signpost("stop")
