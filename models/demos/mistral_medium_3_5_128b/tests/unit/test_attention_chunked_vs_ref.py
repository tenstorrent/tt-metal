# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder suite row 9: a 2-chunk sequence through the **same** ``Attention`` module, two ways.

Recipe row ``test_attention_chunked_vs_ref.py``. The same module instance and the same weights run
the sequence once as a single chunk and once as two, and the second chunk's output has to match.
"Same module" is the substance of the row: it rules out a chunked path that happens to agree with
a differently-configured one-shot path.

Both halves are also PCC'd against the torch reference, so a failure separates into three cases:

* both fail vs the reference, chunked matches one-shot -> the block is wrong, not the chunking.
* one-shot passes, chunked fails -> the position offset, the cache write, or the ring read.
* both pass vs the reference but not each other -> impossible without one of them being marginal;
  read the logged PCCs.

This is the first test where the two SDPA cores meet: the one-shot call takes the gathered path,
the second chunk takes the cache-backed ring path (see ``tt/attention/prefill.py``). It is
therefore also the D2/D3 stand-in for what P2 has to demonstrate at full depth.

**Why chunk 1 logs ~0.982 while chunk 0 logs ~0.998.** Not the chunking, and not the bfloat8_b
cache: the one-shot path reads live bf16 K/V and lands on the same 0.982, and the two paths agree
with each other to 0.9998. It is a property of the *random* weights this test uses. Untrained
projections give a near-uniform softmax — measured mean top-1 probability 0.0074 for chunk 0 and
0.0007 for chunk 1, with an effective support of 512 and 1536 keys — so the output is close to the
mean of that many random V vectors and cancels down to ``|v|/11.7`` in chunk 0 and ``|v|/37.9`` in
chunk 1. The device's error per summand is fixed by the dataformats, so the *relative* error grows
with the cancellation: dividing the two, both chunks imply the same ~0.5% per-summand error, which
is bfloat8_b's 2**-8 mantissa. Real trained attention is concentrated, has no such cancellation,
and is what P1/P2 measure against the golden trace — so this number is a property of the fixture,
not a defect, and it is left as a warning rather than tightened away.
"""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.modeling import REF_DTYPE, causal_mask
from models.demos.mistral_medium_3_5_128b.tests.device_utils import assert_pcc, from_mesh_sp, to_mesh
from models.demos.mistral_medium_3_5_128b.tests.unit.test_attention_vs_ref import (
    attention_config,
    reference_attention,
    reference_cos_sin,
)
from models.demos.mistral_medium_3_5_128b.tt.attention import Attention, ProgramConfig
from models.demos.mistral_medium_3_5_128b.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mistral_medium_3_5_128b.tt.rope import build_rope_mats

SEQ, CHUNK = 2048, 1024
SP_AXIS = 0


def _reference_chunks(cfg, ref, x):
    """The torch reference run chunk-by-chunk, returning each chunk's output."""
    outs, past_k, past_v = [], None, None
    for start in range(0, SEQ, CHUNK):
        chunk = x[:, start : start + CHUNK]
        cos, sin = reference_cos_sin(cfg, start, start + CHUNK)
        mask = causal_mask(CHUNK, start + CHUNK, dtype=REF_DTYPE)
        with torch.no_grad():
            out, k, v = ref(chunk, cos, sin, mask, past_k, past_v)
        past_k = k if past_k is None else torch.cat([past_k, k], dim=2)
        past_v = v if past_v is None else torch.cat([past_v, v], dim=2)
        outs.append(out)
    return outs


@pytest.mark.parametrize("chunk", [CHUNK], ids=[f"c{CHUNK}"])
def test_attention_chunked_vs_ref(galaxy, mesh_config, ccl, cfg, chunk):
    """One-shot and 2-chunk runs of one ``Attention`` instance must agree on chunk 1."""
    ref, state_dict = reference_attention(cfg)
    torch.manual_seed(4)
    x = (torch.randn(1, SEQ, cfg.hidden_size) * 0.1).to(REF_DTYPE)
    ref_chunks = _reference_chunks(cfg, ref, x)

    attn = Attention(
        galaxy,
        attention_config(cfg, max_seq_len=SEQ, sequence_parallel=True),
        state_dict,
        ccl,
        mesh_config,
        ProgramConfig(),
        layer_idx=0,
    )

    def new_cache():
        return allocate_kv_cache(
            galaxy,
            num_layers=1,
            max_seq_len=SEQ,
            chunk_size=chunk,
            sp_axis=SP_AXIS,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
        )

    # One shot: the whole sequence in a single call, its own cache.
    oneshot_out = attn(
        to_mesh(galaxy, x.unsqueeze(0), dims=[-2, None]),
        build_rope_mats(galaxy, cfg, 0, SEQ, mesh_config=mesh_config),
        kv_cache=new_cache(),
    )
    oneshot = from_mesh_sp(galaxy, oneshot_out)

    # Chunked: same module, fresh cache, two calls with the running offset.
    kv = new_cache()
    chunked = []
    for start in range(0, SEQ, chunk):
        out = attn(
            to_mesh(galaxy, x[:, start : start + chunk].unsqueeze(0), dims=[-2, None]),
            build_rope_mats(galaxy, cfg, start, start + chunk, mesh_config=mesh_config),
            kv_cache=kv,
            cached_len=start,
        )
        chunked.append(from_mesh_sp(galaxy, out))

    for i, ref_chunk in enumerate(ref_chunks):
        lo, hi = i * chunk, (i + 1) * chunk
        assert_pcc(f"attention_oneshot[chunk{i}]", ref_chunk.unsqueeze(0), oneshot[:, :, lo:hi])
        assert_pcc(f"attention_chunked[chunk{i}]", ref_chunk.unsqueeze(0), chunked[i])

    # The row's own assertion: the second chunk matches between the two ways of getting there.
    assert_pcc("attention_chunked_vs_oneshot[chunk1]", oneshot[:, :, CHUNK:], chunked[1])
