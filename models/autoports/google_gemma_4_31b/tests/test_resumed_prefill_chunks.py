"""Resumed prefill (scheduler-driven chunked prefill) matches whole-prompt prefill.

vLLM splits one prompt across engine steps: every step after the first calls
prefill with ``chunk_start > 0`` and expects the layer to attend the prefix it
filled earlier. Sliding layers attend a carried BF16 window; full-attention
layers read the paged BFP8 prefix, so their tolerance matches the pinned
chunked-path deviation (see test_zz_chunked_prefill_pcc.py).

Weights are random (seeded): equivalence between the two schedules does not
need the checkpoint, and the base-model cache this box does not carry is what
the shared ``_decoder`` fixture would demand.

Run on qb2 with:
    TT_METAL_HOME=$PWD python_env/bin/python -m pytest -svv \
        models/autoports/google_gemma_4_31b/tests/test_resumed_prefill_chunks.py
"""
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

import models.autoports.google_gemma_4_31b.tests.test_multichip_decoder as TMD
import ttnn
from models.autoports.google_gemma_4_31b.tt import multichip_decoder as MD

CFG_DIR = Path(os.environ.get("TT_METAL_HOME", ".")) / "models/demos/gemma4/configs/gemma-4-31B-it"


def _random_layer_state(layer_idx: int, text_config):
    torch.manual_seed(layer_idx)
    text_config._attn_implementation = "eager"
    layer = Gemma4TextDecoderLayer(text_config, layer_idx=layer_idx)
    with torch.no_grad():
        for p in layer.parameters():
            p.data = p.data.to(torch.bfloat16)
            if p.dim() > 1:
                p.data.normal_(0, 0.02)
    prefix = f"model.language_model.layers.{layer_idx}."
    return {prefix + k: v for k, v in layer.state_dict().items()}


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(str(CFG_DIR), trust_remote_code=True, local_files_only=True).text_config


@pytest.fixture(scope="module", autouse=True)
def _random_weights(hf_config):
    saved = TMD._layer_state
    TMD._layer_state = lambda layer_idx: _random_layer_state(layer_idx, hf_config)
    yield
    TMD._layer_state = saved


@pytest.fixture(scope="module")
def mesh_device():
    if ttnn.get_num_devices() < 4:
        pytest.skip("Gemma 4 multichip decoder requires four local devices")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=96 << 20)
    yield mesh
    ttnn.close_mesh_device(mesh)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rope_slice(hf_config, layer_idx, start, end, mesh_device):
    cos, sin = TMD._rope_host(hf_config, layer_idx, torch.arange(start, end))
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return tuple(
        ttnn.from_torch(
            table.unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )
        for table in (cos, sin)
    )


def _out_host(out):
    host = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    out.deallocate(True)
    return host


CHUNK = 1024  # the sliding window: the smallest legal scheduler budget


def _whole_prompt(hf_config, mesh_device, layer_idx, prompt, *, sdpa_threshold):
    """One whole-prompt prefill, optionally forced through the chunked SDPA path."""
    saved = MD.PREFILL_SDPA_MAX_SEQ
    MD.PREFILL_SDPA_MAX_SEQ = sdpa_threshold
    try:
        decoder = TMD._decoder(hf_config, mesh_device, layer_idx)
        seq_len = prompt.shape[1]
        cache, table = decoder.init_paged_kv_cache(max_context=seq_len + CHUNK)
        out = _out_host(
            decoder.prefill_forward(
                TMD._tt_input(prompt, mesh_device),
                rope_mats=_rope_slice(hf_config, layer_idx, 0, seq_len, mesh_device),
                page_table=table,
                kv_cache=cache,
                valid_seq_len=seq_len,
            )
        )
        return out, cache
    finally:
        MD.PREFILL_SDPA_MAX_SEQ = saved


@pytest.mark.parametrize("layer_idx,kind", [(0, "sliding"), (5, "full")])
@pytest.mark.parametrize("seq_len", [3072, 2600])  # aligned total and a ragged tail
def test_resumed_chunks_match_whole_prompt(hf_config, mesh_device, layer_idx, kind, seq_len):
    torch.manual_seed(4242)
    prompt = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02

    # Reference 1: the ordinary whole-prompt prefill (plain BF16 SDPA).
    reference, ref_cache = _whole_prompt(hf_config, mesh_device, layer_idx, prompt, sdpa_threshold=32768)
    # Reference 2 (full attention only): the same prompt forced through the
    # existing in-call chunked SDPA path, which reads the BFP8 paged cache the
    # same way a resumed chunk must. Rows past the first chunk should match it
    # bit-for-bit; their deviation from the plain reference is the pinned
    # chunked-path quantisation (test_zz_chunked_prefill_pcc.py), not ours.
    if kind == "full":
        forced_chunked, _ = _whole_prompt(hf_config, mesh_device, layer_idx, prompt, sdpa_threshold=CHUNK)

    # Same prompt, split into scheduler-style chunks on a fresh decoder/cache.
    chunked_decoder = TMD._decoder(hf_config, mesh_device, layer_idx)
    cache, table = chunked_decoder.init_paged_kv_cache(max_context=seq_len + CHUNK)
    pieces = []
    for start in range(0, seq_len, CHUNK):
        end = min(start + CHUNK, seq_len)
        pieces.append(
            _out_host(
                chunked_decoder.prefill_forward(
                    TMD._tt_input(prompt[:, start:end], mesh_device),
                    rope_mats=_rope_slice(hf_config, layer_idx, start, end, mesh_device),
                    page_table=table,
                    kv_cache=cache,
                    valid_seq_len=end - start,
                    chunk_start=start,
                )
            )
        )
    resumed = torch.cat(pieces, dim=-2)

    head = slice(0, CHUNK)
    rest = slice(CHUNK, seq_len)
    head_pcc = _pcc(resumed[..., head, :], reference[..., head, :])
    if kind == "full":
        rest_pcc = _pcc(resumed[..., rest, :], forced_chunked[..., rest, :])
        plain_pcc = _pcc(resumed[..., rest, :], reference[..., rest, :])
    else:
        rest_pcc = _pcc(resumed[..., rest, :], reference[..., rest, :])
        plain_pcc = rest_pcc
    print(
        f"RESUMED kind={kind} seq={seq_len} head_pcc={head_pcc:.6f} "
        f"rest_pcc={rest_pcc:.6f} rest_vs_plain_pcc={plain_pcc:.6f}",
        flush=True,
    )
    # Chunk 0 takes the untouched whole-prompt path: bit-exact.
    assert head_pcc > 0.99999, f"chunk 0 diverged from whole-prompt prefill: pcc={head_pcc}"
    # Window-aligned resumed chunks are bit-exact against the matching in-call
    # path (measured 1.000000). A ragged final chunk lands its rows on
    # different matmul program geometry (per-shape block factors at the odd
    # tail M), the same benign per-length numeric variation whole prompts of
    # different lengths already show — caches stay bit-identical.
    threshold = 0.9999 if seq_len % CHUNK == 0 else 0.999
    assert rest_pcc > threshold, f"resumed chunks diverged from the equivalent whole-prompt path: pcc={rest_pcc}"

    # The caches must agree too: decode reads them long after prefill.
    for name, ref_buf, res_buf in (
        ("k", ref_cache[0], cache[0]),
        ("v", ref_cache[1], cache[1]),
    ):
        ref_host = ttnn.to_torch(ttnn.get_device_tensors(ref_buf)[0]).float()
        res_host = ttnn.to_torch(ttnn.get_device_tensors(res_buf)[0]).float()
        cache_pcc = _pcc(ref_host, res_host)
        assert cache_pcc > 0.9999, f"{kind} {name}-cache diverged after resumed prefill: pcc={cache_pcc}"


@pytest.mark.parametrize("layer_idx", [0])
def test_out_of_order_resume_raises(hf_config, mesh_device, layer_idx, expect_error):
    torch.manual_seed(4242)
    decoder = TMD._decoder(hf_config, mesh_device, layer_idx)
    seq_len = 2 * CHUNK
    prompt = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    cache, table = decoder.init_paged_kv_cache(max_context=seq_len + CHUNK)
    # A resume with no chunk 0 first: the sliding carry cannot exist yet.
    with expect_error(RuntimeError, "no carried window"):
        decoder.prefill_forward(
            TMD._tt_input(prompt[:, CHUNK:], mesh_device),
            rope_mats=_rope_slice(hf_config, layer_idx, CHUNK, seq_len, mesh_device),
            page_table=table,
            kv_cache=cache,
            valid_seq_len=CHUNK,
            chunk_start=CHUNK,
        )


@pytest.mark.parametrize("layer_idx", [0])
def test_unaligned_resume_raises(hf_config, mesh_device, layer_idx, expect_error):
    torch.manual_seed(4242)
    decoder = TMD._decoder(hf_config, mesh_device, layer_idx)
    prompt = torch.randn(1, 128, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    cache, table = decoder.init_paged_kv_cache(max_context=4 * CHUNK)
    with expect_error(ValueError, "aligned"):
        decoder.prefill_forward(
            TMD._tt_input(prompt, mesh_device),
            rope_mats=_rope_slice(hf_config, layer_idx, 100, 228, mesh_device),
            page_table=table,
            kv_cache=cache,
            valid_seq_len=128,
            chunk_start=100,
        )
