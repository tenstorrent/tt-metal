# models/demos/blackhole/qwen3_5_9b/tests/unit/test_generator_contract.py

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]
FIXTURE = "models/demos/blackhole/qwen3_5_9b/tests/fixtures/generator_baseline.pt"


def _build(device, n_layers=4):
    return Qwen35Model.from_pretrained(device, max_batch_size=1, max_seq_len=2048, n_layers=n_layers)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_capture_baseline(device):
    device.enable_program_cache()
    model = _build(device)
    BLOCK, NBLK = 64, 32
    model.allocate_kv_caches([NBLK, model.args.n_kv_heads, BLOCK, model.args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NBLK, dtype=torch.int32).unsqueeze(0)

    prompt = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)  # T=16, short
    pf_logits = ttnn.to_torch(model.prefill_paged(prompt, page_table)).squeeze().float()
    next_tok = int(pf_logits.argmax())

    dec = []
    tok = torch.tensor([[next_tok]], dtype=torch.long)
    for pos in range(16, 20):
        dl = ttnn.to_torch(model.decode_paged(tok, current_pos=pos, page_table=page_table)).squeeze().float()
        dec.append(dl)
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)

    torch.save({"prefill_logits": pf_logits, "decode_logits": torch.stack(dec)}, FIXTURE)
    assert pf_logits.shape[-1] == model.args.vocab_size


from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import pack_rope_host, unpack_rope


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rope_pack_roundtrip(device):
    # Mirrors the decode flow: pack on host, copy to device, unpack on device.
    cos = torch.randn(1, 1, 64)
    sin = torch.randn(1, 1, 64)
    cos_host = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_host = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    packed = ttnn.to_device(pack_rope_host(cos_host, sin_host), device)
    c2, s2 = unpack_rope(packed)
    assert tuple(c2.shape) == (1, 1, 64)
    assert tuple(s2.shape) == (1, 1, 64)


# append to tests/unit/test_generator_contract.py
from models.tt_transformers.tt.generator import Generator


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generator_decode_matches_baseline(device):
    device.enable_program_cache()
    base = torch.load(FIXTURE)
    model = _build(device)
    args = model.args
    BLOCK, NBLK = 64, 32
    model.allocate_kv_caches([NBLK, args.n_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NBLK, dtype=torch.int32).unsqueeze(0)
    gen = Generator([model], [args], device)

    # Prefill is model-owned (identical path to the baseline) — establishes KV + GDN state.
    prompt = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)
    pf = ttnn.to_torch(model.prefill_paged(prompt, page_table)).squeeze().float()
    assert torch.allclose(pf, base["prefill_logits"], atol=1e-2, rtol=1e-2), "prefill drifted from baseline"
    next_tok = int(pf.argmax())

    # Decode is Generator-driven — the new contract path under test.
    tok = torch.tensor([[next_tok]], dtype=torch.long)
    for i, pos in enumerate(range(16, 20)):
        out = gen.decode_forward(
            tok, torch.tensor([pos]), page_table=page_table, kv_cache=None, enable_trace=False, read_from_device=True
        )
        dl = out[0].squeeze().float() if isinstance(out, tuple) else out.squeeze().float()
        assert torch.allclose(dl, base["decode_logits"][i], atol=1e-2, rtol=1e-2), f"decode step {i} drifted"
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_generator_decode_traced_matches_baseline(device):
    """Generator-driven TRACED decode must match the baseline (decode_paged) token-for-token.
    Guards against GDN recurrent-state corruption during Generator's trace capture (the old
    bespoke capture wrapped its compile+capture runs in save/restore; this proves the stock
    Generator capture path does not over-advance the in-place GDN state). Coverage for the
    traced path, which test_generator_decode_matches_baseline (enable_trace=False) does not exercise.
    """
    device.enable_program_cache()
    base = torch.load(FIXTURE)
    model = _build(device)
    args = model.args
    BLOCK, NBLK = 64, 32
    model.allocate_kv_caches([NBLK, args.n_kv_heads, BLOCK, args.head_dim], ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NBLK, dtype=torch.int32).unsqueeze(0)
    gen = Generator([model], [args], device)

    prompt = torch.arange(1, 17, dtype=torch.long).unsqueeze(0)
    pf = ttnn.to_torch(model.prefill_paged(prompt, page_table)).squeeze().float()
    next_tok = int(pf.argmax())

    from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prime_decode_trace

    tok = torch.tensor([[next_tok]], dtype=torch.long)
    # Capture the decode trace with GDN-state save/restore so the loop replays from the
    # correct post-prefill state (Generator's capture would otherwise double-advance it).
    prime_decode_trace(gen, model, tok, torch.tensor([16]), page_table)
    for i, pos in enumerate(range(16, 20)):
        out = gen.decode_forward(
            tok, torch.tensor([pos]), page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=True
        )  # TRACED (trace already captured)
        dl = out[0].squeeze().float() if isinstance(out, tuple) else out.squeeze().float()
        assert torch.allclose(dl, base["decode_logits"][i], atol=1e-2, rtol=1e-2), f"traced decode step {i} drifted"
        tok = torch.tensor([[int(dl.argmax())]], dtype=torch.long)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_chunk_seq_flag_selects_chunk_outer(device):
    """The demo's chunk-outer trace selection must be driven by the GDN weights'
    use_chunk_seq_prefill (always True now that chunk-seq is the only prefill path). Guards the
    demo gate _should_use_chunked_trace against regressing to the slow whole-sequence trace.
    """
    from models.demos.blackhole.qwen3_5_9b.demo.text_demo import _should_use_chunked_trace

    model = _build(device)  # n_layers=4
    gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
    assert gdn, "expected at least one GDN (linear-attention) layer"
    # Chunk-seq prefill is always on now; the demo's gate must select chunk-outer.
    assert all(a.weights.use_chunk_seq_prefill for a in gdn)
    assert _should_use_chunked_trace(model) is True
