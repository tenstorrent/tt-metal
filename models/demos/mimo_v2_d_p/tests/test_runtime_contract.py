# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The common/prefill (tt-d-gen) runtime contract, driven exactly like prefill_runner does:
adapter.build_runtime -> adapter.allocate_kv_cache -> compile -> prefill_chunk(device-order tokens) per chunk,
one ack per layer; then the device KV cache (what decode migrates) vs the HF K/V; then the KV chunk table."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tests.golden import golden
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tt.model import PAD_TOKEN_ID, block_cyclic_index
from models.demos.mimo_v2_d_p.tt.rope import rope_perm


class CountingChannel:
    def __init__(self):
        self.n = 0

    def inject(self, k):
        self.n += k


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("seq,chunk,real_len", [(8192, 4096, 8192 - 1000)], ids=["2x4k-padtail"])
def test_runtime_contract(mesh_device, device_params, n_layers, seq, chunk, real_len, tmp_path):
    g = golden(n_layers, seq)
    cfg = MiMoTextConfig.from_json()
    sp, tp = tuple(mesh_device.shape)
    adapter = get_adapter("mimo_v2_d_p")
    params = PrefillRunParams(mesh_shape=(sp, tp), num_layers=n_layers, first_layer_idx=0, is_first_rank=True, is_last_rank=True,
                              max_seq_len=seq, chunk_size=chunk, num_users=2, capacity_factor=0, num_links=1, gate_mode_name="",
                              kv_only_last_layer=False, weight_cache_path=None)
    hf_cfg = adapter.load_hf_config()
    rt = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_cfg, params=params)
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_cfg, params=params)
    rt.compile(kv)
    ch = CountingChannel()
    rt.set_layer_ack_channel(ch)

    slot = 1  # compile() used slot 0: this also checks slot isolation
    ids = g["ids"].clone()
    ids[real_len:] = PAD_TOKEN_ID
    for c in range(seq // chunk):
        s0 = c * chunk
        dev_order = ids[s0 : s0 + chunk][block_cyclic_index(s0, sp, chunk // sp) - s0]
        rt.prefill_chunk(rt.make_chunk_input(dev_order), kv, slot_id=slot, actual_start=s0, actual_end=min(s0 + chunk, real_len), request_id=c)
    ttnn.synchronize_device(mesh_device)
    assert ch.n == n_layers * (seq // chunk), f"acks {ch.n} != {n_layers} layers x {seq // chunk} chunks"

    worst = 1.0
    for layer in range(n_layers):
        spec = cfg.layer_attn(layer)
        k_ref, v_ref = (t.float()[:, :, :real_len] for t in g["kv"][layer])
        k_dev, v_dev = rt.read_layer_kv(kv, slot, layer, real_len)
        pk = comp_pcc(k_ref[..., rope_perm(spec.head_dim, spec.rope_dim)], k_dev)[1]
        pv = comp_pcc(v_ref, v_dev)[1]
        worst = min(worst, pk, pv)
        logger.info(f"L{layer} ({spec.kind[:4]}): K PCC {pk:.5f}  V PCC {pv:.5f}")

    table = rt.build_kv_chunk_table(kv, str(tmp_path / "mimo_table.pb"))
    assert os.path.getsize(table) > 0
    logger.info(f"mesh={mesh_id(mesh_device)} runtime contract OK: acks {ch.n}, worst KV PCC {worst:.5f}, table {table}")
    assert worst > 0.99, worst
