# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT_OSS_DETERMINISTIC_PREFILL_ALLREDUCE=1 makes prefill bit-reproducible.

The prefill attention all-reduce (attention/operations.apply_allreduce) normally runs as reduce_scatter_minimal_async +
all_gather_async. Inside the full model on a Blackhole 1x4 ring that pair leaves device 0's copy of the reduced output
stale in a few tiles, so the TP replicas of the residual stream differ and repeated prefills of one prompt give
different logits. With the switch the all-reduce runs as ttnn.all_reduce.

A 2-layer model with random weights prefills the same prompt three times with the switch on and requires bit-identical
logits and, after every attention all-reduce, identical replicas on all devices. One prefill with the switch off must
agree with them closely (same math, different reduction order). The replica mismatches of the default path are printed,
not asserted: whether they show up depends on the device memory layout of the whole model.

Run on a QuietBox 2 (Blackhole 1x4):
    HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_prefill_determinism.py
"""

import os

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ..test_factory import TestFactory, parametrize_mesh_with_fabric

NUM_LAYERS = 2
PAGE_BLOCK = 64
PAGE_BLOCKS = 32  # 2048-token KV cache
PROMPT = 1000  # padded to 1024
RUNS = 3


def _prefill(model, ids, page_table):
    """Eager prefill of ids [1, n] (padded like the generator) -> the last position's logits on the host."""
    n = ids.shape[-1]
    padded = torch.zeros(1, get_padded_prefill_len(n), dtype=torch.long)
    padded[:, :n] = ids
    x, rot_global, rot_local, tt_page_table, _, _ = model.prepare_inputs_prefill(padded, page_table=page_table)
    last = n - 1
    tt_out = model.ttnn_prefill_forward(
        x,
        rot_mats_global=rot_global,
        rot_mats_local=rot_local,
        page_table=tt_page_table,
        get_last_token=(last // 32) * 32,
    )
    return model.process_output_prefill(tt_out, last % 32).float()


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Needs a QuietBox 2 for about a minute; the CI gpt-oss-20b unit job has a 5-minute budget.",
)
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_deterministic_allreduce(mesh_device, device_params, reset_seeds, monkeypatch):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    import models.demos.gpt_oss.tt.attention.prefill as attention_prefill
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.attention.operations import ENV_DETERMINISTIC_PREFILL_ALLREDUCE
    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    config.num_hidden_layers = NUM_LAYERS
    state_dict = convert_hf_qkv_to_meta_format(GptOssForCausalLM(config).eval().state_dict(), config.head_dim)

    mesh_shape = tuple(mesh_device.shape)
    page_table = torch.arange(PAGE_BLOCKS, dtype=torch.int32).reshape(1, PAGE_BLOCKS)
    model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=get_default_num_links(mesh_device)),
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        paged_attention_config=PagedAttentionConfig(block_size=PAGE_BLOCK, max_num_blocks=PAGE_BLOCKS),
        mesh_config=MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0])),
        create_kv_cache=True,
        max_local_batch_size=1,
    )

    # Count the tiles in which a device's replica of each attention all-reduce output differs from device 1's.
    allreduce = attention_prefill.apply_allreduce
    stale_tiles = []

    def checked_allreduce(tensor, mesh_config, ccl_manager, hidden_size):
        out = allreduce(tensor, mesh_config, ccl_manager, hidden_size)
        replicas = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
        ref = replicas[1]
        tiles = 0
        for r in replicas:
            diff = (r != ref).reshape(-1, ref.shape[-2], ref.shape[-1]).any(dim=0)
            rows, cols = ref.shape[-2] // 32, -(-ref.shape[-1] // 32)
            padded = torch.zeros(rows * 32, cols * 32, dtype=torch.bool)
            padded[: diff.shape[0], : diff.shape[1]] = diff
            tiles += int(padded.reshape(rows, 32, cols, 32).any(dim=3).any(dim=1).sum())
        stale_tiles.append(tiles)
        return out

    monkeypatch.setattr(attention_prefill, "apply_allreduce", checked_allreduce)
    ids = torch.randint(0, config.vocab_size, (1, PROMPT))

    monkeypatch.setenv(ENV_DETERMINISTIC_PREFILL_ALLREDUCE, "1")
    runs = [_prefill(model, ids, page_table) for _ in range(RUNS)]
    deterministic_stale = list(stale_tiles)
    stale_tiles.clear()

    monkeypatch.setenv(ENV_DETERMINISTIC_PREFILL_ALLREDUCE, "0")
    default = _prefill(model, ids, page_table)
    print(f"default all-reduce: tiles differing between device replicas per call {stale_tiles}")

    assert torch.isfinite(runs[0]).all()
    for i, run in enumerate(runs[1:], start=1):
        assert torch.equal(
            runs[0], run
        ), f"prefill {i} differs from prefill 0 (max abs diff {(runs[0] - run).abs().max().item()})"
    assert len(deterministic_stale) == RUNS * NUM_LAYERS
    assert not any(deterministic_stale), f"device replicas differ after the all-reduce: {deterministic_stale} tiles"
    pcc = _pcc(runs[0], default)
    assert pcc > 0.999, f"deterministic and default all-reduce disagree (logits PCC {pcc})"
