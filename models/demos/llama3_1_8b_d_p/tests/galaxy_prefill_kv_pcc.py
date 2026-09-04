# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 / P2: per-layer on-device K/V after a REAL-weights prefill vs the CPU golden trace.

Ported from ``minimax_m3/tests/galaxy_prefill_kv_pcc.py``. This is the first test where real weights,
the full layer count, the target parallelism and the whole runtime interact — everything before it
runs on random weights.

Two modes, selected by ``PREFILL_CHUNKED``:

  * ``PREFILL_CHUNKED=0`` (**P1**) — one-shot: the whole prompt in a single ``prefill_chunk``.
  * ``PREFILL_CHUNKED=1`` (**P2**) — multi-chunk: the same prompt pushed ``seq_len / chunk_size``
    chunks at a time, each attending the prefix the earlier chunks left in the cache. It is compared
    against the SAME golden, so passing means chunked prefill produces the same KV as processing the
    whole sequence at once.

Knobs (all optional):

===========================  =========================================================
``HF_MODEL``                 checkpoint dir (default: the vendored config dir, which
                             has no weights — set this for a real run)
``PREFILL_KV_PCC_SEQ_LEN``   prompt length (default 2048)
``PREFILL_CHUNK_SIZE``       tokens per chunk in chunked mode (default 512)
``PREFILL_KV_PCC_LAYERS``    layer count; default is the model's 32. Lower it for a
                             quick smoke run.
``PREFILL_KV_PCC``           per-layer PCC gate (default 0.99, the spec's per_layer_kv)
===========================  =========================================================

The CPU golden is expensive (a full 8B forward), so it is cached by
``reference/golden.py::ReferenceCacheKey`` keyed on every field that changes it.
"""

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt.model_config import ModelArgs
from models.demos.llama3_1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

from .test_factory import make_mesh_config, parametrize_mesh_with_fabric
from .unit.test_kv_cache_write_vs_ref import gather_kv_cache

DEFAULT_PCC = float(os.getenv("PREFILL_KV_PCC", "0.99"))


def _checkpoint_dir():
    path = os.getenv("HF_MODEL")
    if not path or not os.path.isdir(path):
        pytest.skip("set HF_MODEL to a downloaded Llama-3.1-8B-Instruct checkpoint to run the real-weights KV PCC")
    if not any(f.endswith(".safetensors") for f in os.listdir(path)):
        pytest.skip(f"HF_MODEL={path} has no *.safetensors (config-only dir)")
    return path


def build_reference_from_checkpoint(cfg, state_dict, num_layers):
    """The torch reference loaded with the REAL checkpoint, trimmed to ``num_layers``.

    Takes the state dict in HF (un-permuted) layout — the reference uses HF half-split RoPE, so it
    must NOT get the Meta-swizzled q/k the device gets.
    """
    from dataclasses import replace

    model = ref.LlamaModel(replace(cfg, num_hidden_layers=num_layers))
    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            key = k[len("model.") :]
        elif k == "lm_head.weight":
            key = k
        else:
            continue
        if key.startswith("layers."):
            layer_idx = int(key.split(".")[1])
            if layer_idx >= num_layers:
                continue
        mapped[key] = v.to(torch.float32)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert not missing, f"reference is missing weights: {missing[:5]}"
    return model.eval()


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.timeout(0)
def test_galaxy_prefill_kv_pcc(mesh_device, device_params, reset_seeds):
    """Real weights on the target mesh; per-layer K/V vs the CPU golden."""
    checkpoint = _checkpoint_dir()
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    seq_len = int(os.getenv("PREFILL_KV_PCC_SEQ_LEN", "2048"))
    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", "512")) if chunked else seq_len

    mesh_config = make_mesh_config(mesh_device)
    sp, tp = mesh_config.sp, mesh_config.tp

    model_args = ModelArgs(mesh_device=mesh_device, max_seq_len=seq_len)
    cfg = model_args.hf_config
    num_layers = int(os.getenv("PREFILL_KV_PCC_LAYERS", str(cfg.num_hidden_layers)))
    n_kv_local = cfg.num_key_value_heads // tp

    assert seq_len % chunk_size == 0, f"seq_len {seq_len} must be a multiple of chunk_size {chunk_size}"
    assert chunk_size % (ttnn.TILE_SIZE * sp) == 0, f"chunk_size {chunk_size} must be a multiple of {32 * sp}"

    logger.info(
        f"KV PCC: checkpoint={checkpoint} mode={'chunked' if chunked else 'one-shot'} "
        f"seq_len={seq_len} chunk_size={chunk_size} layers={num_layers} mesh={sp}x{tp}"
    )

    # --- weights: HF layout for the reference, Meta-swizzled for the device ---
    logger.info("Loading checkpoint (HF layout, for the CPU golden)...")
    hf_state = ModelArgs.load_state_dict(checkpoint, convert_to_meta_format=False)
    logger.info("Converting q/k to Meta format for the device...")
    from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

    device_state = convert_hf_qkv_to_meta_format(dict(hf_state), cfg.head_dim)

    # --- CPU golden ---
    logger.info(f"Running the CPU reference ({num_layers} layers, {seq_len} tokens) — this is the slow part...")
    ref_model = build_reference_from_checkpoint(cfg, hf_state, num_layers)
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), generator=g)
    with torch.no_grad():
        _, ref_kvs, _ = ref_model(input_ids)
    # Free the fp32 reference and the HF-layout copy before the device build allocates: at 32 layers
    # the three live copies of an 8B checkpoint are the peak host footprint of this test.
    del ref_model, hf_state
    gc.collect()

    # --- device run ---
    runtime = TtPrefillRuntime(
        mesh_device=mesh_device,
        hf_config=cfg,
        state_dict=device_state,
        config=TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=seq_len,
            mesh_shape=(sp, tp),
            default_chunk_size=chunk_size,
            num_users=1,
            sp_axis=mesh_config.sp_axis,
            tp_axis=mesh_config.tp_axis,
            topology=ttnn.Topology.Linear,
            attn_weight_dtype=ttnn.bfloat16,
            mlp_weight_dtype=ttnn.bfloat16,
            owns_kv_cache=True,
        ),
    )
    del device_state

    tokens = input_ids[0].tolist()
    for c in range(seq_len // chunk_size):
        start = c * chunk_size
        runtime.prefill_chunk(
            runtime.make_chunk_input(tokens[start : start + chunk_size], chunk_size),
            slot_id=0,
            actual_start=start,
            actual_end=start + chunk_size,
            chunk_size=chunk_size,
        )
    ttnn.synchronize_device(mesh_device)

    # --- compare ---
    # The cache holds Meta-swizzled post-RoPE K (the device weights were reverse_permuted); V is raw.
    # Pass chunk_local so a multi-chunk cache is un-permuted out of block-cyclic order first.
    chunk_local = chunk_size // sp
    results = []
    for i in range(num_layers):
        want_k = ref.hf_to_meta_head_perm(ref_kvs[i][0], cfg.head_dim)
        want_v = ref_kvs[i][1]
        got_k = gather_kv_cache(mesh_device, runtime.kv_cache.k, n_kv_local, slot_row=i, chunk_local=chunk_local)
        got_v = gather_kv_cache(mesh_device, runtime.kv_cache.v, n_kv_local, slot_row=i, chunk_local=chunk_local)
        ok_k, pcc_k = comp_pcc(want_k, got_k, DEFAULT_PCC)
        ok_v, pcc_v = comp_pcc(want_v, got_v, DEFAULT_PCC)
        results.append((i, ok_k, pcc_k, ok_v, pcc_v))
        logger.info(f"layer {i:>2}: K={pcc_k} {'OK' if ok_k else 'FAIL'}   V={pcc_v} {'OK' if ok_v else 'FAIL'}")

    failures = [(i, pk, pv) for i, ok_k, pk, ok_v, pv in results if not (ok_k and ok_v)]
    logger.info(
        f"KV PCC summary ({'chunked' if chunked else 'one-shot'}): "
        f"{len(results) - len(failures)}/{len(results)} layers >= {DEFAULT_PCC}"
    )
    assert not failures, f"layers below PCC {DEFAULT_PCC}: {failures}"
