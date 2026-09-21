# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer-type TP prefill + decode cost, via the eager paths' tracy signposts."""
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
def test_perf_tp_layers(mesh_device, reset_seeds, ensure_gc):
    T = int(os.environ.get("QWEN_TP_ISL", "2048"))
    steps = int(os.environ.get("QWEN_TP_DEC_STEPS", "4"))
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=T + 256, hf_model=model_path())
    nb = (T // BLOCK) + 8
    page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    model.allocate_kv_caches((nb, model.args.n_local_kv_heads, BLOCK, model.args.head_dim), ttnn.bfloat16, batch_size=1)
    tokens = torch.randint(0, model.args.vocab_size, (1, T), dtype=torch.long)

    model.reset_tp()
    model.prefill_tp(tokens, valid_len=T)  # warmup, outside signposts
    ttnn.synchronize_device(mesh_device)

    model.reset_tp()
    logger.info(f"measured eager TP prefill: ISL={T} dies={mesh_device.get_num_devices()}")
    signpost("pf_start")
    model.prefill_tp(tokens, valid_len=T)
    ttnn.synchronize_device(mesh_device)
    signpost("pf_stop")

    tok, pos = 1234, T
    model.decode_tp(tok, pos)
    ttnn.synchronize_device(mesh_device)
    signpost("dec_start")
    for i in range(steps):
        model.decode_tp(tok, pos + 1 + i)
    ttnn.synchronize_device(mesh_device)
    signpost("dec_stop")
