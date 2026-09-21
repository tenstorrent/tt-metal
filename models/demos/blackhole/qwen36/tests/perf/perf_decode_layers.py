# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer-type DECODE cost at a given TP, via the existing per-layer tracy signposts.

model.py:680 already emits `decode L{idx} {gdn|attn}` under QWEN36_SIGNPOSTS=1, and
decode_tp's per-layer loop always runs eagerly, so a device-profiler capture can be
bucketed straight into GDN vs full-attention layers.

    export HF_MODEL=Qwen/Qwen3.5-2B MESH_DEVICE=P150x8 QWEN36_SIGNPOSTS=1
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
      python -m tracy -p -r -v -m pytest \
      models/demos/blackhole/qwen36/tests/perf/perf_decode_layers.py -sv
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

try:
    from tracy import signpost
except ImportError:

    def signpost(*_a, **_k):
        pass


BLOCK = 64


@torch.no_grad()
@parametrize_mesh_tp()
def test_perf_decode_layers(mesh_device, reset_seeds, ensure_gc):
    T = int(os.environ.get("QWEN_DEC_PREFILL", "512"))
    steps = int(os.environ.get("QWEN_DEC_STEPS", "4"))
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=T + 256, hf_model=model_path())
    nb = (T // BLOCK) + 8
    page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    model.allocate_kv_caches((nb, model.args.n_local_kv_heads, BLOCK, model.args.head_dim), ttnn.bfloat16, batch_size=1)

    tokens = torch.randint(0, model.args.vocab_size, (1, T), dtype=torch.long)
    model.reset_tp()
    model.prefill_tp(tokens, valid_len=T)

    tok, pos = 1234, T
    model.decode_tp(tok, pos)  # warmup / compile, outside the signposts
    ttnn.synchronize_device(mesh_device)

    logger.info(f"measured decode: {steps} steps, {mesh_device.get_num_devices()} devices")
    signpost("start")
    for i in range(steps):
        model.decode_tp(tok, pos + 1 + i)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
