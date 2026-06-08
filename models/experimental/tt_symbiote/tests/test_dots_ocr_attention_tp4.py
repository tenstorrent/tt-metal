# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP=4 (DP1TP4) PCC test for the dots.ocr text-decoder attention (T3K, mesh (1, 4)).

Exercises ``TTNNDotsOCRAttention`` in **decode** (seq=1) with a paged KV cache over a
realistic generation context: ``prior_tokens`` tokens are decoded one-at-a-time to
populate the KV cache (mirroring real autoregressive generation, e.g. the ~180-token
OCR demo run), then the final decode token's output -- which attends over the full
``prior_tokens + 1`` context via paged SDPA -- is compared to the unsharded HF
reference attention's last-token output over the same causal sequence.

This catches distortion that a ``cache_position=0`` (empty-cache, attend-to-self)
check cannot: wrong paged-cache addressing, rotary at non-zero positions, and the
SDPA reduction across a populated K/V cache.

Two QKV tensor-parallel schemes are exercised (both gather the same N-sharded
``o_proj`` output for the PCC check; no head splitting -- dots.ocr has only 2 KV
heads, which would not divide TP=4 anyway):

* ``k_parallel`` (default ``qkv_proj`` = ``IColShardedWAllReduced``): the post-norm
  hidden arrives hidden-K-sharded (``dim=-1``, ``hidden/TP`` per device); the QKV
  matmul (``[M, hidden/TP, N]``) produces a per-device partial sum that
  ``reduce_scatter`` + ``all_gather`` re-assemble into the **full** QKV replicated
  on every device.
* ``n_parallel`` (``qkv_proj`` = ``_TTNNDotsOCRQKVColParallel``): the hidden arrives
  **replicated** full-hidden; the QKV weight is N-sharded so the matmul is
  ``[M, hidden, N/TP]`` (no reduction), and a single ``all_gather(dim=-1)`` rebuilds
  the full QKV. Same per-device matmul FLOPs, trades reduce_scatter for all_gather.

In both schemes ``o_proj`` is ``IReplicatedWColSharded``: replicated full-hidden
attention output in -> weight-N-sharded -> hidden-N-sharded out (``dim=-1``), no
CCL. The test gathers the per-device N-slices back to full hidden before PCC.

Run on T3K (uses 4 of 8 devices; ``MESH_DEVICE`` gates the collective-linear arch)::

    MESH_DEVICE=T3K pytest models/experimental/tt_symbiote/tests/test_dots_ocr_attention_tp4.py -s

Single-decode-attention-layer device perf report
-------------------------------------------------
The paged KV cache is sized to the real text-decoder depth (28 layers, override
with ``DOTS_OCR_ATTN_NUM_LAYERS``) and filled with ``prior_tokens`` (180) tokens
to reproduce a production decode context. The cache-fill loop runs *unprofiled*;
only the final single decode step is bracketed with the Tracy signpost
``dots_ocr.attn_decode_layer``. To capture device perf for just that one decode
attention layer (paged SDPA decode reads only its own layer slice, so per-layer
timing is independent of the 28-layer depth)::

    MESH_DEVICE=T3K python -m tracy -r -p -v -m \
        "pytest models/experimental/tt_symbiote/tests/test_dots_ocr_attention_tp4.py -s"

then filter the generated op-perf CSV to the signposted region::

    python models/tt_transformers/scripts/op_perf_results.py \
        <generated_ops_perf_results.csv> --signpost dots_ocr.attn_decode_layer
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.models.dots_ocr import _create_paged_kv_cache
from models.experimental.tt_symbiote.modules.dots_ocr_attention import TTNNDotsOCRAttention
from models.experimental.tt_symbiote.modules.linear import _tp_requires_ccl
from models.experimental.tt_symbiote.utils.device_management import set_device

TP = 4

# Number of tokens already in the KV cache before the measured decode step. Matches
# the ~180-token OCR demo generation length; override with DOTS_OCR_ATTN_PRIOR_TOKENS.
PRIOR_TOKENS = int(os.environ.get("DOTS_OCR_ATTN_PRIOR_TOKENS", "180"))

# BF16 activations x BFP8 (QKV) / BFP4 (o_proj) weights, HiFi2 SDPA, BF16 KV cache.
# Each cache entry is a one-shot QKV of an exact per-position hidden, so per-entry
# quantization error does NOT compound across the prior_tokens steps; the floor
# tracks the single-token decode bar (~0.99). It guards against gross distortion --
# a wrong shard ordering, dropped collective, or mis-addressed cache slot drops PCC
# toward 0 -- while staying above the model's low-precision quantization noise.
PCC_THRESHOLD = 0.98

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"


def _resolve_model_path() -> str:
    """Resolve dots.ocr model path: HF cache via snapshot_download > model ID."""
    import os

    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


def _raw_ttnn(t):
    """Unwrap the module's TorchTTNNTensor wrapper to the underlying ttnn.Tensor."""
    inner = getattr(t, "ttnn_tensor", None)
    return inner if inner is not None else t


def _signpost(header: str) -> None:
    """Emit a Tracy ``TT_SIGNPOST`` marker so a device perf report can isolate
    the bracketed region (see ``op_perf_results.py --signpost``). No-op when the
    Tracy tooling isn't importable (e.g. plain ``pytest`` without ``-m tracy``)."""
    try:
        from tools.tracy import signpost
    except ImportError:
        return
    signpost(header)


def _decode_step(tt_attn, paged_cache, mesh_device, hidden_1tok, pos, in_mapper):
    """Run one decode step at ``pos``: writes K/V into the cache and returns the
    (still N-sharded) attention output for this token."""
    x_tt = ttnn.from_torch(
        hidden_1tok,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=in_mapper,
    )
    cache_position = ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_out, _ = tt_attn(
        hidden_states=x_tt,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=paged_cache,
        cache_position=cache_position,
    )
    return attn_out


@pytest.mark.parametrize("prior_tokens", [PRIOR_TOKENS], ids=[f"ctx{PRIOR_TOKENS}"])
@pytest.mark.parametrize("scheme", ["k_parallel", "n_parallel"])
@pytest.mark.parametrize("mesh_device", [(1, TP)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_dots_ocr_attention_tp4(mesh_device, prior_tokens, scheme):
    """Text-decoder attention at TP=4 (decode over a populated KV cache), no distortion vs HF.

    Decodes ``prior_tokens`` tokens one-at-a-time to fill the paged KV cache, then
    PCC-checks the final decode token (which attends over all ``prior_tokens + 1``
    keys) against the HF reference's last-token output over the same causal sequence.

    ``scheme`` selects the QKV tensor-parallel layout: ``k_parallel`` (K-sharded in +
    reduce_scatter/all_gather) or ``n_parallel`` (column-parallel, replicated
    full-hidden in + all_gather).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    assert mesh_device.get_num_devices() == TP, f"Expected TP={TP}, got {mesh_device.get_num_devices()}"
    assert _tp_requires_ccl(mesh_device), "(1,4) mesh must engage the TP-CCL path"

    torch.manual_seed(0xA77E)
    torch.set_grad_enabled(False)

    model_path = _resolve_model_path()
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Size the paged KV cache to the REAL decoder depth (dots.ocr text decoder
    # = 28 layers) so the cache allocation / page addressing match a production
    # run, while the HF reference is built with a single layer (cheap + exact).
    # The measured attention is still one layer (layer_idx 0): paged SDPA decode
    # reads only its own layer slice, so the single-layer device perf is
    # independent of num_layers -- the depth only affects cache memory.
    cache_num_layers = int(os.environ.get("DOTS_OCR_ATTN_NUM_LAYERS", model_config.num_hidden_layers))
    model_config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(torch.bfloat16).eval()
    model_config = hf_model.config
    hidden_size = model_config.hidden_size
    assert hidden_size % TP == 0, f"hidden {hidden_size} must divide TP={TP}"

    hf_attn = hf_model.model.layers[0].self_attn
    hf_rotary_emb = hf_model.model.rotary_emb

    # ------------------------------------------------------------------
    # HF reference: full causal sequence of prior_tokens + 1 positions; the
    # measured token is the last one (attends over the whole context).
    # Computed BEFORE building the TTNN module so the from_torch /
    # preprocess_weights lifecycle cannot perturb the reference weights.
    # ------------------------------------------------------------------
    seq = prior_tokens + 1
    hidden_seq = torch.randn(1, seq, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    cos, sin = hf_rotary_emb(hidden_seq, position_ids)
    # Additive causal mask [1, 1, seq, seq]: position i cannot see j > i.
    causal_mask = torch.triu(torch.full((seq, seq), float("-inf"), dtype=torch.bfloat16), diagonal=1).view(
        1, 1, seq, seq
    )
    ref_out = hf_attn(
        hidden_states=hidden_seq,
        position_embeddings=(cos, sin),
        attention_mask=causal_mask,
        past_key_value=None,
        cache_position=position_ids[0],
    )
    ref_full = ref_out[0] if isinstance(ref_out, (tuple, list)) else ref_out
    ref = ref_full[:, -1:, :].to(torch.float32)  # last (measured) token

    # ------------------------------------------------------------------
    # TTNN attention at TP=4.
    # ------------------------------------------------------------------
    n_parallel = scheme == "n_parallel"
    tt_attn = TTNNDotsOCRAttention.from_torch(hf_attn, qkv_n_parallel=n_parallel)
    set_device(tt_attn, mesh_device, register_forward_hook=False, dump_visualization=False)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    # Reflect the real (28-layer) decoder depth in the cache allocation; the
    # measured attention still writes/reads layer_idx 0 only.
    model_config.num_hidden_layers = cache_num_layers
    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)

    if n_parallel:
        # Column-parallel QKV takes a REPLICATED full-hidden activation on every
        # device (matmul is [M, hidden, N/TP]).
        in_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        expected_in_dim = hidden_size
    else:
        # K-parallel QKV takes a hidden-K-sharded activation (the post-attention-
        # norm contract this module consumes under TP).
        in_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        expected_in_dim = hidden_size // TP

    # Cache-fill (NOT profiled): positions 0..prior_tokens-1 only populate the
    # paged KV cache one token at a time; their outputs are discarded. This
    # mirrors a real generation context but is deliberately excluded from the
    # perf report below.
    for pos in range(seq - 1):
        fill_out = _decode_step(tt_attn, paged_cache, mesh_device, hidden_seq[:, pos : pos + 1, :], pos, in_mapper)
        ttnn.deallocate(_raw_ttnn(fill_out))
    ttnn.synchronize_device(mesh_device)

    # Measured step: the SINGLE decode attention layer at the final position,
    # attending over the full prior_tokens + 1 context. Bracketed by Tracy
    # signposts so a device perf report isolates exactly one decode attention
    # layer's ops (QKV matmul, rotary, paged_update, paged_sdpa_decode, o_proj,
    # CCLs) -- excluding the prior_tokens cache-fill steps above. See the module
    # docstring for the tracy + op_perf_results.py --signpost workflow.
    _signpost(f"dots_ocr.attn_decode_layer.{scheme}")
    attn_out = _decode_step(tt_attn, paged_cache, mesh_device, hidden_seq[:, seq - 1 : seq, :], seq - 1, in_mapper)
    ttnn.synchronize_device(mesh_device)
    _signpost(f"dots_ocr.attn_decode_layer.{scheme}.end")

    # o_proj output is hidden-N-sharded; gather the per-device slices back to
    # full hidden along dim=-1.
    out = ttnn.to_torch(
        _raw_ttnn(attn_out),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
    ).to(torch.float32)
    out = out.reshape(ref.shape)

    passed, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    logger.info(
        f"[attention prior_tokens={prior_tokens} scheme={scheme}] TP={TP} "
        f"pcc={float(pcc):.6f} (threshold {PCC_THRESHOLD})"
    )
    assert passed, (
        f"Attention TP={TP} ({scheme}) distorted at prior_tokens={prior_tokens}: "
        f"pcc={float(pcc):.6f} < {PCC_THRESHOLD}"
    )
