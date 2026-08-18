"""The chunked prefill path is lower precision than the direct path.

The chunked full-attention branch deallocates the bf16 K/V and re-reads K/V from
the paged cache, which is ``bfloat8_b``, so attention runs on 8-bit quantised
K/V. That is the right trade only where it is the sole option (very long
context). Routing ordinary prompts through it cost 20 points of GPQA
(37.5 -> 17.5) while MMLU moved 78.37 -> 78.47, because a ~0.9997 per-layer PCC
compounds over 60 layers.

This test pins the deviation so a future change that silently reroutes short
prompts onto the chunked path is visible rather than only showing up as an eval
regression.
"""
import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_31b.tests.test_multichip_decoder import (  # noqa: F401
    _decoder,
    _rope_device,
    _tt_input,
    hf_config,
    mesh_device,
    optimized_baseline,
)
from models.autoports.google_gemma_4_31b.tt import multichip_decoder as MD


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _prefill_out(hf_config, mesh_device, layer_idx, seq_len, threshold):
    saved = MD.PREFILL_SDPA_MAX_SEQ
    MD.PREFILL_SDPA_MAX_SEQ = threshold
    try:
        decoder = _decoder(hf_config, mesh_device, layer_idx)
        torch.manual_seed(4242)
        prompt = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
        cache, page_table = decoder.init_paged_kv_cache(max_context=seq_len + 1)
        out = decoder.prefill_forward(
            _tt_input(prompt, mesh_device),
            rope_mats=_rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False),
            page_table=page_table,
            kv_cache=cache,
            valid_seq_len=seq_len,
        )
        host = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
        out.deallocate(True)
        return host
    finally:
        MD.PREFILL_SDPA_MAX_SEQ = saved


@pytest.mark.parametrize("layer_idx,kind", [(0, "sliding"), (5, "full")])
@pytest.mark.parametrize("seq_len", [1500])
def test_chunked_vs_direct_prefill_output(hf_config, mesh_device, layer_idx, kind, seq_len):  # noqa: F811
    direct = _prefill_out(hf_config, mesh_device, layer_idx, seq_len, 32768)
    chunked = _prefill_out(hf_config, mesh_device, layer_idx, seq_len, 1024)
    p = _pcc(direct, chunked)
    md = (direct - chunked).abs().max().item()
    print(f"CHUNKED_VS_DIRECT kind={kind} seq={seq_len} pcc={p:.6f} max_abs_diff={md:.6f}", flush=True)
    # The paths are close but NOT equivalent; assert both facts so either a
    # regression in the chunked path or an unexpected convergence is noticed.
    assert p > 0.999, f"chunked path diverged further than expected: pcc={p}"
    assert md > 0.05, (
        "chunked and direct prefill now agree closely; if the chunked path stopped "
        "reading bfloat8_b cache K/V, update this test and the spec note"
    )
