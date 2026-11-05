# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.tests.test_tt_sampling import extract_tokens, make_sharded_logits, refresh_sharded_logits
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.demos.llama3_70b_galaxy.tt.model_config import TtModelArgs
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1345000,
            "trace_region_size": 64000000,
        }
    ],
    indirect=True,
)
def test_sampling_trace_logprobs_match_reference(mesh_device, reset_seeds, monkeypatch, tmp_path):
    """Exercise the real Galaxy CCL and worker sub-device without model weights."""
    if "wormhole" not in str(mesh_device.arch()).lower():
        pytest.skip("Requires the Wormhole Galaxy worker layout")
    monkeypatch.setenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    args = TtModelArgs(mesh_device, instruct=True, dummy_weights=True, max_batch_size=32, max_seq_len=512)
    args.sampling_all_gather_axis = 0
    setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1)
    ccl = TT_CCL(mesh_device, args, setup.worker_sub_device_id, mode="decode")
    values = torch.randn(1, 1, 32, args.vocab_size, dtype=torch.bfloat16)
    padded_values = torch.nn.functional.pad(values, (0, args.padded_vocab_size - args.vocab_size), value=0)
    sampler = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=ccl)
    logits = make_sharded_logits(padded_values, mesh_device, args)
    try:
        for temperature in (0.0, 0.5, 1.0, 1.5):
            params = format_sampling_params(
                SamplingParams(
                    temperature=[temperature] * 32,
                    top_k=[32 if temperature else 1] * 32,
                    top_p=[1.0] * 32,
                    enable_log_probs=[True] * 32,
                    seed=None,  # Explicit seeds use the eager path instead of replaying the trace.
                ),
                32,
            )
            sampler.reset_sampling_params(params)
            sampler.precompile(logits)
            sampler.capture_trace(logits, skip_precompile=True)
            for sign in (1, -1):
                replay_values = values * sign
                padded = torch.nn.functional.pad(replay_values, (0, args.padded_vocab_size - args.vocab_size), value=0)
                refresh_sharded_logits(padded, logits, mesh_device, args)
                sampler.seed_manager.reset_seed(None, list(range(32)))
                sampler.seed_manager.get_new_values()
                tokens, logprobs = sampler.sample(logits, enable_trace=True, skip_precompile=True)
                ttnn.synchronize_device(mesh_device)
                ids = torch.tensor(extract_tokens(tokens, 32), dtype=torch.long)
                actual = ttnn.to_torch(ttnn.get_device_tensors(logprobs)[0]).reshape(-1)[:32].float()
                reference = torch.log_softmax(replay_values.float().squeeze(), dim=-1)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, reference[torch.arange(32), ids], atol=0.15, rtol=0.02)
            sampler.reset_trace()
    finally:
        sampler.reset_trace()
        ccl.close()
