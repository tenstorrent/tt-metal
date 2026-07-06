# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""#47466 — DiffusionGemma bounded-memory chunked long-context prefill.

Acceptance gates for ``tt/chunked_prefill.py``: prefilling a prompt in chunks
must reproduce a single full-length prefill (last-token logits PCC >= 0.999),
proving the fixes the shared gemma4 backbone lacks:

* correct per-chunk RoPE offset (``chunk_start_idx``),
* cross-chunk attention for full-attention layers through the paged KV cache
  (``chunk_page_table`` fill + full-``page_table``
  ``chunked_scaled_dot_product_attention``), and
* correct sliding-window attention for **prompts longer than the sliding
  window** via a bounded rolling in-memory K/V window buffer (the paged chunked
  SDPA op is causal-only, so it over-attends past the window).

Two gates:

1. ``test_chunked_prefill_matches_single`` — 512-token prompt as **2×256**
   chunks vs single **1×512**, sliding_window=1024 (> prompt, so sliding ==
   causal within the window). Exercises the RoPE offset + full-attention paged
   cross-chunk attention.
2. ``test_chunked_prefill_sliding_past_window`` — 2048-token prompt as **8×256**
   chunks vs single **1×2048**, sliding_window=1024 (< prompt, so sliding layers
   EXCEED the window). This is the sliding-window gate: the sliding layers must
   apply the window mask exactly like the single full-length prefill. With the
   old code this path raised ``NotImplementedError`` (sliding-window chunked
   prefill past the window was unimplemented — it would over-attend); with the
   rolling-window buffer it matches the single prefill at PCC >= 0.999.

The vehicle is a tiny random-weight 2-layer model (one sliding + one full
attention layer) so the check isolates the chunked-prefill logic from the 26B
MoE fidelity ceiling. MoE is downstream of and identical between the two paths,
so it does not affect chunked-vs-single equivalence; a tiny model with MoE off
fully exercises the RoPE + KV-fill + SDPA changes.

Run on QB2 (4x Blackhole):

    source /home/zni/venvs/tt-diffusion-gemma/bin/activate
    export TT_METAL_HOME=/home/zni/tt-metal PYTHONPATH=/home/zni/tt-metal
    DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 pytest \
      models/experimental/diffusion_gemma/tests/test_device_chunked_prefill.py -v -s
"""

import os

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.experimental.diffusion_gemma.tt import chunked_prefill as cp
from models.experimental.diffusion_gemma.tt.model import DiffusionGemma4Model
from models.tt_transformers.tt.common import PagedAttentionConfig
from tests.ttnn.utils_for_testing import assert_with_pcc

CHUNK_SIZE = 256
BLOCK_SIZE = 64
HIDDEN = 128
HEAD_DIM = 32
VOCAB = 256

pytestmark = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run chunked-prefill correctness on a Tenstorrent device",
)


def _tiny_config(sliding_window):
    """Two-layer Gemma4 text config: one sliding + one full attention layer, MoE off."""
    layer_types = ["sliding_attention", "full_attention"]
    config = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=HEAD_DIM,
        global_head_dim=HEAD_DIM,
        layer_types=layer_types,
        sliding_window=sliding_window,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_k_eq_v=False,
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=0.0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
    )
    config._attn_implementation = "eager"
    return config


def _to_tt_state(config):
    """Random but self-consistent tiny backbone weights, remapped to gemma4 keys."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextScaledWordEmbedding,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = Gemma4TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
                embed_scale=config.hidden_size**0.5,
            )
            self.layers = torch.nn.ModuleList(
                [Gemma4TextDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
            )
            self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    m = _Tiny().eval()
    return {f"model.{k}": v for k, v in m.state_dict().items()}


def _alloc_caches(mesh_device, model, prompt_len, *, paged):
    """One [k, v] pair per layer: paged (block pool) or contiguous."""
    pac = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=prompt_len // BLOCK_SIZE) if paged else None
    caches = []
    for layer in model.layers:
        caches.append(
            init_kv_cache(
                mesh_device=mesh_device,
                config=layer.self_attn.config,
                max_batch_size=1,
                max_seq_len=prompt_len,
                paged_attention_config=pac,
            )
        )
    return caches


def _last_token_logits(tt_logits, row):
    t = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    if t.dim() == 4:
        t = t.squeeze(0).squeeze(0)  # [seq, vocab]
    elif t.dim() == 3:
        t = t.squeeze(0)
    return t[row, :VOCAB]


def _chunked_vs_single_pcc(device, prompt_len, sliding_window):
    """Build a tiny model, run 1×prompt_len vs (prompt_len/CHUNK_SIZE)×CHUNK_SIZE, return PCC."""
    torch.manual_seed(47466)
    tp = device.shape[1] if hasattr(device, "shape") else 1

    config = _tiny_config(sliding_window)
    model_args = Gemma4ModelArgs.from_hf_config(config)
    model_args._hf_text_config = config
    mesh_config = MeshConfig(device.shape, decode=ModeConfig(tp=tp)) if hasattr(device, "shape") else None

    state = _to_tt_state(config)
    model = DiffusionGemma4Model(
        mesh_device=device,
        hf_config=model_args,
        state_dict=state,
        ccl_manager=CCLManager(device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=prompt_len,
        max_local_batch_size=1,
        num_layers=config.num_hidden_layers,
        create_kv_cache=False,
    )

    # Prompt embeddings [1, 1, prompt_len, HIDDEN], tile-laid.
    input_ids = torch.randint(0, VOCAB, (1, prompt_len), dtype=torch.int64)
    replicate = ttnn.ReplicateTensorToMesh(device) if hasattr(device, "shape") else None
    tokens_tt = ttnn.from_torch(
        input_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate
    )
    embeds = model.embed_tokens(tokens_tt)
    embeds = ttnn.reshape(embeds, (1, 1, prompt_len, HIDDEN))
    embeds_single = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

    # ── baseline: single 1×prompt_len prefill (contiguous cache, stock gemma4 SDPA) ──
    baseline_cache = _alloc_caches(device, model, prompt_len, paged=False)
    logits_single = model(
        embeds_single,
        is_decode=False,
        page_table=None,
        kv_caches=baseline_cache,
        input_ids_torch=input_ids,
        get_last_token=-1,
        batch_size=1,
    )
    single_last = _last_token_logits(logits_single, prompt_len - 1)
    logits_single.deallocate(True)

    # ── chunked: N×CHUNK_SIZE over a paged cache via the DG-local fixed prefill ──────
    paged_cache = _alloc_caches(device, model, prompt_len, paged=True)
    page_table_torch = cp.make_reference_page_table(prompt_len // BLOCK_SIZE, mesh_device=device)
    # embeds_single was consumed (lm_head deallocs its input chain); re-embed for the chunked run.
    embeds2 = model.embed_tokens(tokens_tt)
    embeds2 = ttnn.reshape(embeds2, (1, 1, prompt_len, HIDDEN))
    embeds_chunked = ttnn.to_layout(embeds2, ttnn.TILE_LAYOUT)
    logits_chunked = cp.chunked_prefill(
        model,
        embeds_chunked,
        input_ids_torch=input_ids,
        embeds_torch=None,
        kv_cache=paged_cache,
        page_table_torch=page_table_torch,
        block_size=BLOCK_SIZE,
        chunk_size=CHUNK_SIZE,
    )
    chunked_last = _last_token_logits(logits_chunked, CHUNK_SIZE - 1)
    logits_chunked.deallocate(True)
    embeds_chunked.deallocate(True)

    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(single_last, chunked_last, pcc=0.999)
    return single_last, chunked_last, pcc


@pytest.mark.use_module_device
def test_chunked_prefill_matches_single(device):
    """512 tokens as 2×256 vs 1×512, window 1024 > prompt (sliding == causal)."""
    prompt_len, sliding_window = 512, 1024
    single_last, chunked_last, pcc = _chunked_vs_single_pcc(device, prompt_len, sliding_window)
    print(f"[chunked-prefill] last-token logits PCC (2x{CHUNK_SIZE} vs 1x{prompt_len}, window {sliding_window}): {pcc}")
    assert_with_pcc(single_last, chunked_last, 0.999)


@pytest.mark.use_module_device
def test_chunked_prefill_sliding_past_window(device):
    """2048 tokens as 8×256 vs 1×2048, window 1024 < prompt — sliding layers EXCEED the window.

    This is the sliding-window bounded-buffer gate. The last-token (pos 2047)
    sliding query attends only (1023, 2047]; a causal-only path (the old paged
    chunked SDPA, which raised NotImplementedError here) would over-attend the
    full prefix. Matching the single 1×2048 prefill (which applies
    sliding_window_size=1024) at PCC >= 0.999 proves the rolling K/V window
    buffer applies the exact same window mask across chunk boundaries.
    """
    prompt_len, sliding_window = 2048, 1024
    assert prompt_len > sliding_window, "gate must exceed the sliding window"
    single_last, chunked_last, pcc = _chunked_vs_single_pcc(device, prompt_len, sliding_window)
    n_chunks = prompt_len // CHUNK_SIZE
    print(
        f"[chunked-prefill] SLIDING last-token logits PCC "
        f"({n_chunks}x{CHUNK_SIZE} vs 1x{prompt_len}, window {sliding_window}): {pcc}"
    )
    assert_with_pcc(single_last, chunked_last, 0.999)
