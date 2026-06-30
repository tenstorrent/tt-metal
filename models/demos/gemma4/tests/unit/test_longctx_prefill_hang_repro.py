# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal-side reproduction for GH #48289.

[Gemma4] Large single-shot eager prefill hangs the device
(system_memory_manager.cpp:702 fetch-queue timeout) above ~100K ISL.

The vLLM-side symptom is a device dispatch stall while enqueuing a *full-sequence*
op during the un-chunked eager prefill — the issue pins it on the per-head V
RMS-norm in ``models/demos/gemma4/tt/attention/prefill.py::_prefill_forward_single``:

    tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)
        -> ttnn.rms_norm(...)        # LayerNormDeviceOperation, full sequence
        -> system_memory_manager.cpp:702  TIMEOUT in fetch queue wait

SDPA is chunked for ``seq_len > 32768``, but the QKV projection, the per-head
Q/K/V norms and RoPE all run on the whole sequence in one shot. This test drives
a single Gemma4 *sliding* attention layer through the real ``_prefill_forward_single``
path at a sweep of input lengths, isolating the failing full-sequence op without
needing the full 48-layer model or real weights (the stall is a property of the
op size at a given ISL, not of weight values).

The hang is a device-side property, so it reproduces with random weights on the
same 1x4 / TP=4 Blackhole mesh as the issue. To observe the reported *throw*
(rather than a merely-slow op), run with a finite per-op timeout, e.g.::

    TT_METAL_OPERATION_TIMEOUT_SECONDS=5 \
    HF_MODEL=$PWD/models/demos/gemma4/configs/gemma-4-12B-it \
    pytest -svq models/demos/gemma4/tests/unit/test_longctx_prefill_hang_repro.py -k 1x4

Expected envelope (matching the issue): short/medium ISL complete; ~120K stalls
and the engine throws ``TT_THROW ... system_memory_manager.cpp:702 ... TIMEOUT:
device timeout in fetch queue wait`` from the per-head V-norm enqueue.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager

from ...tests.test_factory import TestFactory, find_layer_idx, parametrize_mesh_with_fabric

# ISL sweep bracketing the reported ~100K pass / ~120K hang wall. Override with a
# comma-separated env list, e.g. GEMMA4_REPRO_ISLS="98304,114688,122880,131072".
_DEFAULT_ISLS = [32768, 65536, 98304, 122880]
_ISLS = [int(x) for x in os.environ.get("GEMMA4_REPRO_ISLS", "").split(",") if x.strip()] or _DEFAULT_ISLS


def _to_device(tensor, mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
@pytest.mark.parametrize("seq_len", _ISLS, ids=[f"isl{n}" for n in _ISLS])
def test_longctx_eager_prefill_hang(seq_len, mesh_device, reset_seeds):
    """Single sliding-layer eager prefill at large ISL — reproduces GH #48289.

    Runs the real ``_prefill_forward_single`` (QKV proj -> per-head Q/K/V norms ->
    RoPE -> chunked sliding SDPA). The per-head V-norm runs full-sequence; above the
    ~100K wall its enqueue stalls the device dispatch and the op throws the
    fetch-queue timeout (when TT_METAL_OPERATION_TIMEOUT_SECONDS is finite).
    """
    hf_text_config = TestFactory.create_hf_text_config()
    layer_idx = find_layer_idx(hf_text_config, "sliding_attention")

    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    hf_attn = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx).self_attn
    state_dict = {k: v.clone() for k, v in hf_attn.state_dict().items() if not k.startswith("v_norm")}

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
        create_kv_cache=False,
        max_batch_size=1,
        max_seq_len=seq_len,
    )

    local_kv_heads = max(1, config.num_key_value_heads // tp)
    logger.info(
        "GH#48289 repro: layer_idx={} (sliding) tp={} seq_len={} | full-sequence V-norm op "
        "shape ~ [1, 1, {}*{}={}, {}]  (TT_METAL_OPERATION_TIMEOUT_SECONDS={})",
        layer_idx,
        tp,
        seq_len,
        local_kv_heads,
        seq_len,
        local_kv_heads * seq_len,
        config.head_dim,
        os.environ.get("TT_METAL_OPERATION_TIMEOUT_SECONDS", "<unset>"),
    )

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(mesh_device, hf_text_config, seq_len, layer_idx)
    x_torch = torch.randn(1, 1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    x_tt = _to_device(x_torch, mesh_device)

    t0 = time.time()
    tt_out = tt_attn(x_tt, rope_mats=(cos_tt, sin_tt), is_decode=False, page_table=None, kv_cache=None)
    ttnn.synchronize_device(mesh_device)
    dt = time.time() - t0

    out_shape = tuple(tt_out.shape)
    tt_out.deallocate(True)
    logger.info("GH#48289 repro: seq_len={} eager prefill COMPLETED in {:.2f}s out_shape={}", seq_len, dt, out_shape)


# ── text_demo-driven repro (the actual demo entrypoint, real weights) ─────────
#
# The single-layer test above does NOT reproduce the hang (a lone full-sequence
# V-norm dispatches fine even at 122880). The reported stall is an emergent
# property of the *whole* eager prefill with hybrid KV-cache groups disabled, so
# we reproduce it through the real demo entrypoint: ``text_demo.run_generation``,
# the same single-user prefill path (``run_generation`` ->
# ``model.ttnn_prefill_forward`` -> ``_prefill_forward_single``) that vLLM serving
# drives. Uses the real google/gemma-4-12b-it checkpoint (HF cache).
#
#   HF_MODEL=google/gemma-4-12b-it \
#   TT_METAL_OPERATION_TIMEOUT_SECONDS=5 \
#   pytest -svq models/demos/gemma4/tests/unit/test_longctx_prefill_hang_repro.py \
#       -k "text_demo and 1x4" --max-prefill 200000

# Real-weights model builds are expensive (one full load per ISL), so the demo
# repro defaults to a minimal pass/hang bracket. Override with GEMMA4_DEMO_ISLS.
_DEMO_ISLS = [int(x) for x in os.environ.get("GEMMA4_DEMO_ISLS", "").split(",") if x.strip()] or [98304, 122880]


@pytest.fixture(scope="module")
def _longctx_prompt():
    """Load a real long-context prompt once (128k bucket), reused for every ISL.

    ``run_generation`` truncates it to each target ISL, so the prefill runs on
    real tokens rather than zero-padding.
    """
    from models.demos.gemma4.demo import text_demo

    return text_demo.load_demo_prompt(131072, instruct=True)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
@pytest.mark.parametrize("seq_len", _DEMO_ISLS, ids=[f"isl{n}" for n in _DEMO_ISLS])
def test_text_demo_longctx_hang(seq_len, mesh_device, reset_seeds, _longctx_prompt):
    """Real ``text_demo.run_generation`` single-user prefill at large ISL.

    Reproduces GH #48289 through the demo entrypoint with the real
    google/gemma-4-12b-it checkpoint, eager prefill, hybrid KVC off. Above the
    ~100K wall the measured prefill stalls and the engine throws the fetch-queue
    timeout (system_memory_manager.cpp:702) from the full-sequence per-head V-norm.
    """
    from models.demos.gemma4.demo import text_demo

    model_path = os.environ.get("HF_MODEL") or os.environ.get("GEMMA4_MODEL_PATH")
    assert model_path, "set HF_MODEL to google/gemma-4-12b-it"

    block_size = 64
    # +256 headroom so the single decode step (needed for run_generation's metrics)
    # has KV room past the prompt; the hang we're after is in the *prefill* before it.
    max_seq_len = ((seq_len + 256 + block_size - 1) // block_size) * block_size
    page_params = {"page_block_size": block_size, "page_max_num_blocks": max_seq_len // block_size}

    logger.info(
        "GH#48289 text_demo repro: ISL={} eager prefill via run_generation, hybrid KVC OFF "
        "(TT_METAL_OPERATION_TIMEOUT_SECONDS={})...",
        seq_len,
        os.environ.get("TT_METAL_OPERATION_TIMEOUT_SECONDS", "<unset>"),
    )
    results = text_demo.run_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        prompts=[_longctx_prompt],
        max_new_tokens=2,  # prefill + one decode step (run_generation metrics need a decode timer)
        max_seq_len=max_seq_len,
        page_params=page_params,
        enable_decode_trace=False,
        target_prefill_len=seq_len,
    )
    logger.info("GH#48289 text_demo repro: ISL={} run_generation COMPLETED, {} result(s)", seq_len, len(results))


# ── vLLM-generator-driven repro (the exact issue stack) ───────────────────────
#
# The text_demo repro above calls ``model.ttnn_prefill_forward`` directly, which
# runs the prefill kernel at ``target_prefill_len`` *exactly* (e.g. 122880). The
# real vLLM serving path is different: it goes through the bridge
#
#   Gemma4ForCausalLM.prefill_forward            (generator_vllm.py)
#     -> Generator.prefill_forward_text          (tt_transformers/generator.py)
#       -> prefill_forward_single_user_text
#         -> Gemma4Model.ttnn_prefill_forward
#
# which is precisely the stack in GH #48289. The key behavioural difference:
# ``prefill_forward_single_user_text`` pads the prompt to
# ``get_padded_prefill_len(n)`` — the next power of two — before dispatching the
# (single-chunk, since ``max_prefill_chunk_size == max_seq_len``) eager kernel.
# So any prompt in (65536, 131072] runs a **131072-token** prefill kernel, which
# the direct text_demo path (122880) never exercised. This is the larger op that
# the issue blames for the fetch-queue stall, so we drive it here.
#
#   HF_MODEL=google/gemma-4-12b-it \
#   TT_METAL_OPERATION_TIMEOUT_SECONDS=5 \
#   pytest -svq models/demos/gemma4/tests/unit/test_longctx_prefill_hang_repro.py \
#       -k "generator_vllm and 1x4" --max-prefill 200000

# ISLs chosen so the *padded* kernel brackets the 131072 wall: 65536 -> 65536
# kernel (expected pass), 122880 -> 131072 kernel (the suspect). Override with
# GEMMA4_VLLM_ISLS.
_VLLM_ISLS = [int(x) for x in os.environ.get("GEMMA4_VLLM_ISLS", "").split(",") if x.strip()] or [65536, 122880]


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
@pytest.mark.parametrize("seq_len", _VLLM_ISLS, ids=[f"isl{n}" for n in _VLLM_ISLS])
def test_generator_vllm_longctx_hang(seq_len, mesh_device, reset_seeds, _longctx_prompt):
    """Drive eager prefill through the vLLM bridge (``Gemma4ForCausalLM``).

    Mirrors the GH #48289 stack exactly: ``Gemma4ForCausalLM.prefill_forward`` ->
    ``Generator.prefill_forward_text`` -> ``prefill_forward_single_user_text`` ->
    ``Gemma4Model.ttnn_prefill_forward``. Hybrid kv-cache groups are off, so
    prefill runs eager (no device trace). Unlike the direct text_demo path, the
    kernel is padded to ``get_padded_prefill_len`` (the next power of 2), so ISLs
    in (65536, 131072] dispatch a full 131072-token prefill — the larger op the
    issue blames for the fetch-queue timeout (system_memory_manager.cpp:702).
    """
    from transformers import AutoConfig, AutoTokenizer

    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM
    from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len

    from .test_prefill_trace_parity import _allocate_fresh_kv_cache, _create_page_table

    model_path = os.environ.get("HF_MODEL") or os.environ.get("GEMMA4_MODEL_PATH")
    assert model_path, "set HF_MODEL to google/gemma-4-12b-it"

    kernel_len = get_padded_prefill_len(seq_len)
    block_size = 64
    # KV pool / page table must cover the padded kernel length. The full pool
    # backs the single request (hybrid groups off -> one UniformType group).
    max_seq_len = kernel_len
    max_num_blocks = max_seq_len // block_size
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

    logger.info(
        "GH#48289 vLLM-generator repro: ISL={} -> padded kernel={} (single-shot eager, hybrid KVC OFF) "
        "via Gemma4ForCausalLM.prefill_forward (TT_METAL_OPERATION_TIMEOUT_SECONDS={})...",
        seq_len,
        kernel_len,
        os.environ.get("TT_METAL_OPERATION_TIMEOUT_SECONDS", "<unset>"),
    )

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hf_config._name_or_path = model_path  # initialize_vllm_model loads weights from here
    generator = Gemma4ForCausalLM.initialize_vllm_model(
        hf_config=hf_config,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )

    # Tokenize the real long prompt; tile it to the target ISL if the prompt is
    # shorter (the longest demo prompt is ~100K tokens, below the 122880 target),
    # then zero-pad to the padded kernel length (matches what the Generator hands
    # ttnn_prefill_forward). The hang is a device dispatch / op-size property, not
    # a function of token values, so tiling real tokens is faithful for the repro.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_ids = tokenizer.encode(_longctx_prompt)
    if len(prompt_ids) < seq_len:
        reps = (seq_len + len(prompt_ids) - 1) // len(prompt_ids)
        prompt_ids = (prompt_ids * reps)[:seq_len]
    else:
        prompt_ids = prompt_ids[:seq_len]
    tokens = torch.zeros(1, kernel_len, dtype=torch.int32)
    tokens[0, :seq_len] = torch.tensor(prompt_ids, dtype=torch.int32)
    prompt_lens = torch.tensor([seq_len], dtype=torch.long)

    kv_cache = [
        _allocate_fresh_kv_cache(
            generator.model[0],
            max_batch_size=1,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_cfg,
        )
    ]
    page_table = _create_page_table(1, paged_cfg)

    t0 = time.time()
    out = generator.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=False,
        warmup_prefill=False,
    )
    dt = time.time() - t0
    out_shape = tuple(out.shape) if hasattr(out, "shape") else None
    logger.info(
        "GH#48289 vLLM-generator repro: ISL={} kernel={} prefill_forward COMPLETED in {:.2f}s out_shape={}",
        seq_len,
        kernel_len,
        dt,
        out_shape,
    )
