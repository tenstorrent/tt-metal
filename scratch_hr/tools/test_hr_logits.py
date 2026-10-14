# #59149 stack round, scratch: LLM logits untilize. LMHead1D's matmul writes bfp8_b logits (lm_head_dtype default) and
# the model untilizes them with ttnn.untilize(use_multicore=True); the block factory gives each core rows of many tiles.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", ["bfp8_b", "bf16"])
@pytest.mark.parametrize("vocab", [128256, 16032, 151936])
def test_logits_untilize(device, vocab, dtype):
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, 1, 32, vocab).bfloat16(), dtype=ttnn.bfloat8_b if dtype == "bfp8_b" else ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = ttnn.untilize(x, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(y), ttnn.to_torch(x))
