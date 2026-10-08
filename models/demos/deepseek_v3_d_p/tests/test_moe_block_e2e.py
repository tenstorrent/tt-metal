# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The all-gather MoE block (TtMoe moe_block="all_gather") end to end on the LoudBox: the production prefill runtime,
traced, with seeded random weights.

The model is built by its own serving adapter (adapter.build_runtime), so the runtime config is exactly what serving
runs; only the weights are injected (random, from the HF reference model) and the MoE block is forced to
"all_gather" ($TT_DS_PREFILL_MOE_BLOCK; every MoE layer is asserted to have taken it, none to have fallen back).
Then, per model:

  * compile + capture_trace, then N_CHUNKS chunks of chunked prefill through prefill_chunk (traced) into one KV slot
    and the same chunks eager (model.forward) into another;
  * every layer's KV must agree traced vs eager. The KV of layer l is computed from the residual stream after layers
    < l, so the layers past the first MoE layer carry the MoE block's output through the trace;
  * the per-chunk wall time of both paths is logged (MOE-BLOCK-E2E lines): the LoudBox end-to-end number.

Few layers (the dense prefix, one MoE layer, and the KV-only last layer) and 1280-token chunks (640 per chip: the
Galaxy's per-chip shape).
GLM-5.3 skips until its adapter wires a torch reference model (random weights come from it). Kimi-K3 is not here:
896 routed experts are 112 per LoudBox chip, past the flat expert's 64, so the all-gather block cannot run it on this
mesh. The Galaxy legs (glm_moe_ag_*, k3_moe_ag_transformer) cover both against their checkpoints.
"""

import gc
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.moe.moe_block import MOE_BLOCK_ENV
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_tp0, unrotate_cache_layer
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import create_hf_model, extract_tt_state_dict

CHUNK = 1280
N_CHUNKS = 3
TOTAL_LEN = CHUNK * N_CHUNKS
SLOT_TRACED, SLOT_EAGER = 1, 2  # slot 0 takes compile() / capture_trace()'s warm forwards
NUM_USERS = 4
TRACED_VS_EAGER_KV_PCC = 0.999  # the same ops on the same device: an agreement bar, not an accuracy bar

# model id -> (variant, model config, layers: the dense prefix + one MoE layer + the KV-only last layer, l1_small_size)
MODELS = {
    "k2_7": ("kimi_k2_7", KimiK27Config, KimiK27Config.NUM_DENSE_LAYERS + 2, 768),
    "glm_5_3": ("glm_5_3", GLM53Config, GLM53Config.NUM_DENSE_LAYERS + 2, 1216),
}


def _metadata_msg(mesh_device, slot_id, actual_start, actual_end):
    """The packed [1, 1, 1, 3] uint32 (slot_id, actual_start, actual_end) the traced path reads on device."""
    return ttnn.from_torch(
        torch.tensor([slot_id, actual_start, actual_end], dtype=torch.int64).reshape(1, 1, 1, 3),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _token_ids(vocab_size):
    g = torch.Generator().manual_seed(2026)
    ids = torch.randint(0, vocab_size, (TOTAL_LEN,), generator=g, dtype=torch.int64)
    return [ids[c * CHUNK : (c + 1) * CHUNK].tolist() for c in range(N_CHUNKS)]


def _run_traced(runtime, kv_caches, mesh_device, token_ids, slot):
    times = []
    for c, ids in enumerate(token_ids):
        msg = _metadata_msg(mesh_device, slot, c * CHUNK, (c + 1) * CHUNK)
        inp = runtime.make_chunk_input(ids)
        t0 = time.perf_counter()
        runtime.prefill_chunk(
            inp,
            kv_caches,
            slot_id=slot,
            actual_start=c * CHUNK,
            actual_end=(c + 1) * CHUNK,
            request_id=c,
            metadata_msg=msg,
        )
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
        ttnn.deallocate(msg)
    return times


def _run_eager(runtime, kv_caches, mesh_device, token_ids, slot):
    """prefill_chunk dispatches on config.use_trace, so the eager pass calls the model the way its eager branch does."""
    times = []
    for c, ids in enumerate(token_ids):
        inp = runtime.make_chunk_input(ids)
        t0 = time.perf_counter()
        runtime.model.forward(
            inp,
            kv_caches.kvpe,
            actual_isl=CHUNK,
            actual_start=c * CHUNK,
            actual_end=(c + 1) * CHUNK,
            cache_user_id=slot,
            index_kv_cache=getattr(kv_caches, "index", None),
        )
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
        ttnn.deallocate(inp)
    return times


@pytest.mark.skipif(not is_blackhole(), reason="the all-gather MoE block needs the flat routed expert (Blackhole)")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(
                fabric_payload_size=KimiK27Config.FABRIC_PAYLOAD_SIZE,
                l1_small_size=1216,  # the larger of the models' pools (GLM); K2.7's 768 fits inside
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="loudbox-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model", list(MODELS))
def test_moe_block_all_gather_e2e_traced(model, mesh_device, device_params, num_links, monkeypatch):
    from models.demos.deepseek_v3_d_p.tests.conftest import TEST_VARIANTS, _resolve_config_only
    from models.demos.deepseek_v3_d_p.tt import tt_prefill_runtime

    variant_name, model_cfg, num_layers, _ = MODELS[model]
    variant = TEST_VARIANTS[variant_name]
    adapter = variant  # the variant objects are the serving adapters (models/demos/common/prefill/adapter.py)
    monkeypatch.setenv(MOE_BLOCK_ENV, "all_gather")

    hf_config = _resolve_config_only(variant.name)
    hf_config.max_seq_len = TOTAL_LEN
    sp, tp = tuple(mesh_device.shape)

    try:
        variant.reference_model_cls
    except NotImplementedError as e:
        # GLM-5.3's adapter has no torch reference (its DSA sparse attention is not vendored), so there are no random
        # weights to build from; its all-gather legs run on the Galaxy against the checkpoint (glm_moe_ag_*).
        pytest.skip(f"no random weights for {model}: {e}")
    logger.info(f"{model}: random HF weights for {num_layers} layers")
    hf_model = create_hf_model(variant, hf_config, num_layers)
    state_dict = extract_tt_state_dict(variant, hf_model)
    del hf_model
    gc.collect()

    # The adapter builds TtPrefillRuntime(state_dict={}) for serving (weights from the TTNN cache); hand it the random
    # weights instead, leaving every other piece of the construction to the adapter.
    real_runtime = tt_prefill_runtime.TtPrefillRuntime
    monkeypatch.setattr(
        tt_prefill_runtime,
        "TtPrefillRuntime",
        lambda **kw: real_runtime(**{**kw, "state_dict": state_dict}),
    )
    params = PrefillRunParams(
        mesh_shape=(sp, tp),
        num_layers=num_layers,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=TOTAL_LEN,
        chunk_size=CHUNK,
        num_users=NUM_USERS,
        capacity_factor=8,
        num_links=num_links,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,  # a traced build: no norm + lm_head readback inside the capture
        weight_cache_path=None,
        use_trace=True,
    )
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    del state_dict
    gc.collect()
    kv_caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)

    # kv_only_last_layer: the last layer only writes its KV (no FFN). The MoE layers are the ones in between, and the
    # last layer's KV is computed from their output.
    moe_layers = [l.ffn for l in runtime.model.layers if getattr(getattr(l, "ffn", None), "moe_block", None)]
    assert len(moe_layers) == num_layers - model_cfg.NUM_DENSE_LAYERS - 1, f"{len(moe_layers)} MoE layers"
    blocks = {m.moe_block for m in moe_layers}
    assert blocks == {"all_gather"}, f"MoE layers run {blocks}: the all-gather block fell back"

    token_ids = _token_ids(hf_config.vocab_size)
    try:
        t0 = time.perf_counter()
        runtime.compile(kv_caches)
        runtime.capture_trace(kv_caches)
        logger.info(f"{model}: compile + capture {time.perf_counter() - t0:.1f} s")
        traced_s = _run_traced(runtime, kv_caches, mesh_device, token_ids, SLOT_TRACED)
        eager_s = _run_eager(runtime, kv_caches, mesh_device, token_ids, SLOT_EAGER)
        cache_full = gather_cache_tp0(kv_caches.kvpe.storage, mesh_device)
        pos = blockcyclic_positions(sp, CHUNK, TOTAL_LEN)
        traced, eager = (
            [unrotate_cache_layer(cache_full[slot * num_layers + i], pos, TOTAL_LEN) for i in range(num_layers)]
            for slot in (SLOT_TRACED, SLOT_EAGER)
        )
    finally:
        runtime.release_trace()  # before the fixture closes the device (the traces live in sub-device managers)
        del runtime
        gc.collect()

    logger.info(
        f"MOE-BLOCK-E2E {model} all_gather loudbox-2x4 L{num_layers} chunk {CHUNK}: "
        f"traced per chunk {[round(t * 1e3, 2) for t in traced_s]} ms, "
        f"eager per chunk {[round(t * 1e3, 2) for t in eager_s]} ms"
    )
    pccs = {}
    for i, (a, b) in enumerate(zip(traced, eager)):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"layer {i}: non-finite KV"
        pccs[i] = float(comp_pcc(a, b)[1])
        logger.info(
            f"  KV layer {i}{' (after a MoE layer)' if i > model_cfg.NUM_DENSE_LAYERS else ''}: "
            f"traced vs eager PCC {pccs[i]:.6f}"
        )
    bad = {i: p for i, p in pccs.items() if p < TRACED_VS_EAGER_KV_PCC}
    assert not bad, f"traced KV diverges from eager on layers {bad} (bar {TRACED_VS_EAGER_KV_PCC})"
