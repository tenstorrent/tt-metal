import ttnn
import torch
import pytest


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seq_len", [128, 352, 1024, 2048, 5632, 9472, 11264, 44520])
@pytest.mark.parametrize("num_heads", [4, 8, 10, 20])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_nlp_create_heads_repro(device, batch, seq_len, num_heads, head_dim):
    torch_input = torch.randn(batch, 1, seq_len, (num_heads * head_dim)).bfloat16()
    torch_output = torch_input.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def create_heads(inp):
        # Unfortunate hack - we don't have a split_heads operation that takes unfused qkv
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            inp,
            ttnn.concat([inp, inp], dim=-1),
            num_heads=num_heads,
            num_kv_heads=num_heads,
            transpose_k_heads=False,
        )
        return out

    tt_output = create_heads(tt_input)
    tt_output = ttnn.to_torch(tt_output)
    assert torch.allclose(torch_output, tt_output, atol=1e-4)
