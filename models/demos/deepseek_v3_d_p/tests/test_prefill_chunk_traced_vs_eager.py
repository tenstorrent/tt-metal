# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Traced-vs-eager numerical equivalence for ``TtPrefillRuntime.prefill_chunk`` (#55126).

Nothing else drives prefill_chunk, so the traced serving path had no numerical coverage -- and the
llama4 query temperature is specifically invisible to the coverage that does exist: an unrefreshed
buffer applies temperature 1.0 and moves the chunked PCC gate by ~0.002 against a 0.98 threshold.

So the test measures its own sensitivity. Three passes on one traced runtime, into disjoint KV slots:
traced via prefill_chunk, eager via model.forward, and eager with the temperature suppressed. The
control is asserted FIRST -- if it cannot see a wrong temperature, the agreement numbers mean nothing.

Six chunks (30,720 tokens), because at chunk 5120 the scale bands do not line up: chunk 0 is exactly
1.0 and proves nothing, chunk 2 is the first uniform non-unity value, chunk 5 is a second different
one. Two distinct values is what separates "refreshed per chunk" from "built once".

The comparison is the KV cache, because on a single rank prefill_chunk returns no activation. Layer 0
is temperature-independent, so it is the control for "do the two paths agree at all"; layers >= 1
carry the signal.
"""
import gc
import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_tp0, unrotate_cache_layer
from tests.ttnn.utils_for_testing import comp_pcc

CHUNK = PREFILL_CHUNK_TOKENS  # 5120

# Six chunks = 30,720 tokens. See the module docstring for why fewer proves nothing.
N_CHUNKS = 6
TOTAL_LEN = N_CHUNKS * CHUNK

# Disjoint user slots: compile() and capture_trace() both write real KV into slot 0.
SLOT_WARMUP = 0  # written by compile() / capture_trace()'s warm forwards; never compared
SLOT_TRACED = 1
SLOT_EAGER = 2
SLOT_CONTROL = 3  # eager with the temperature suppressed; the sensitivity control
NUM_USERS = 4

# An agreement bar, not an accuracy bar: traced and eager run the same ops on the same device and
# measure exactly 1.000000. 0.999 leaves room for a bf8 tie, not for a bug.
TRACED_VS_EAGER_KV_PCC = 0.999


def _packed_metadata_msg(mesh_device, slot_id: int, actual_start: int, actual_end: int) -> ttnn.Tensor:
    """The packed [1,1,1,3] uint32 record the traced path reads on-device.

    In serving this comes off the H2D socket; prefill_chunk only copies it into its own persistent
    buffer, so any tensor of this shape works and the test needs no socket. Words are
    (slot_id, actual_start, actual_end).
    """
    return ttnn.from_torch(
        torch.tensor([slot_id, actual_start, actual_end], dtype=torch.int64).reshape(1, 1, 1, 3),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _chunk_token_ids(num_chunks: int, vocab_size: int) -> list[list[int]]:
    """Deterministic in-vocab ids. No golden is needed or wanted: every comparison here is device
    against device, so the ids only have to be identical across passes and legal for the embedding."""
    g = torch.Generator().manual_seed(55126)
    ids = torch.randint(0, vocab_size, (num_chunks * CHUNK,), generator=g, dtype=torch.int64)
    return [ids[c * CHUNK : (c + 1) * CHUNK].tolist() for c in range(num_chunks)]


def _read_slot(kvpe_cache, mesh_device, sp: int, slot: int, num_layers: int, seq_len_cache: int) -> list[torch.Tensor]:
    """One user slot's per-layer KV, un-rotated out of the block-cyclic device layout and trimmed to
    the tokens actually written. Slot layout is user-major (``slot * num_layers + layer``), the same
    linearisation ``update_padded_kv_cache`` computes on device."""
    cache_full = gather_cache_tp0(kvpe_cache.storage, mesh_device)
    p = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    return [unrotate_cache_layer(cache_full[slot * num_layers + i], p, TOTAL_LEN) for i in range(num_layers)]


def _per_layer_pcc(a: list[torch.Tensor], b: list[torch.Tensor]) -> dict[int, float]:
    out = {}
    for i, (x, y) in enumerate(zip(a, b)):
        _, pcc = comp_pcc(x, y)
        out[i] = float(pcc)
    return out


def _expected_scale(pos: int) -> float:
    beta = MistralSmall4Config.LLAMA4_SCALING_BETA
    orig_max = MistralSmall4Config.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS
    return 1.0 + beta * math.log(1.0 + pos // orig_max)


def _build(mesh_device, hf_config, weight_cache_path, num_layers, num_links, use_trace):
    """Build a runtime through the production adapter rather than assembling TtPrefillRuntimeConfig
    here, so this test exercises the same construction serving does and does not drift from it."""
    adapter = get_adapter("mistral_small_4")
    params = PrefillRunParams(
        mesh_shape=tuple(mesh_device.shape),
        num_layers=num_layers,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=TOTAL_LEN,
        chunk_size=CHUNK,
        num_users=NUM_USERS,
        capacity_factor=8,  # PREFILL_CAPACITY_FACTOR default; 1 undersizes the MoE dispatch buffers
        num_links=num_links,
        gate_mode_name=adapter.default_gate_mode,
        # Matches PREFILL_KV_ONLY_LAST_LAYER's default. Load-bearing for a traced build: False runs
        # the forward into norm + lm_head, and logit_to_host is a readback, which records an event --
        # illegal inside a capture. True returns once the last layer's KV is written, which is all a
        # prefill runtime wants anyway.
        kv_only_last_layer=True,
        weight_cache_path=weight_cache_path,
        use_trace=use_trace,
    )
    runtime = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    kv_caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    return runtime, kv_caches


def _run_traced(runtime, kv_caches, mesh_device, token_ids, slot):
    for c, ids in enumerate(token_ids):
        msg = _packed_metadata_msg(mesh_device, slot, c * CHUNK, (c + 1) * CHUNK)
        # Real ints AND the packed message, exactly as prefill_runner._compute_and_send calls it.
        # The traced path reads the three SCALARS on-device from the message; the host ints are still
        # passed and are what selects this chunk's pre-built query-scale buffer.
        runtime.prefill_chunk(
            runtime.make_chunk_input(ids),
            kv_caches,
            slot_id=slot,
            actual_start=c * CHUNK,
            actual_end=(c + 1) * CHUNK,
            request_id=c,
            metadata_msg=msg,
        )
        ttnn.deallocate(msg)
    ttnn.synchronize_device(mesh_device)


def _run_eager_reference(runtime, kv_caches, mesh_device, token_ids, slot):
    """Eager reference on a traced runtime, calling the model directly.

    prefill_chunk dispatches on config.use_trace, so a traced runtime cannot be its own eager
    reference through it. Mirrors prefill_chunk's eager branch, metadata unset so ttMLA takes its
    host-scalar path.
    """
    for c, ids in enumerate(token_ids):
        inp = runtime.make_chunk_input(ids)
        runtime.model.forward(
            inp,
            kv_caches.kvpe,
            actual_isl=CHUNK,
            actual_start=c * CHUNK,
            actual_end=(c + 1) * CHUNK,
            cache_user_id=slot,
            index_kv_cache=kv_caches.index,
        )
        ttnn.deallocate(inp)
    ttnn.synchronize_device(mesh_device)


# 8 layers. The temperature reaches K/V only through the previous layer's attention, so its effect
# compounds then saturates. Measured control PCC by layer: 0.99976 0.99926 0.99742 0.99714 0.99417
# 0.98969 0.99500 -- at 4 layers the worst is only 0.0026 off, which is ABOVE the bar, i.e. the
# comparison could not see a wrong temperature. 8 puts it 10x clear; deeper buys nothing.
@pytest.mark.parametrize("num_layers", [8], ids=["L8"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE,
                l1_small_size=768,
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral4"])
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 prefill requires Blackhole")
@pytest.mark.timeout(0)
def test_prefill_chunk_traced_matches_eager(
    variant, config_only, mesh_device, device_params, weight_cache_path, num_layers, num_links
):
    """Traced prefill_chunk must write the same KV as eager across 6 chunks (30,720 tokens).

    Three passes on one runtime so the leg pays one weight load: traced -> slot 1, eager -> slot 2,
    eager with _llama4_beta suppressed -> slot 3. Same weights and device state, so the only
    difference between slots 2 and 3 is the temperature.
    """
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    assert num_layers >= 2, (
        "layer 0's KV is computed from the embeddings and is identical under any query temperature, "
        "so a 1-layer run cannot see the scale at all; layers >= 1 carry the signal"
    )

    hf_config = config_only
    hf_config.max_seq_len = TOTAL_LEN
    sp, tp = tuple(mesh_device.shape)
    # The pytest fixture returns the cache ROOT; the per-mesh files live one level down under
    # "{sp}x{tp}". run_chunked_transformer_updated does the same join (effective_cache_path), and the
    # adapter's own weight_cache_path() builds the same shape for serving. Passing the root makes
    # TtPrefillRuntime's completeness check look in a directory holding only the 8x4 subdir, which it
    # correctly reports as an incomplete cache rather than loading placeholder weights.
    weight_cache_path = weight_cache_path / f"{sp}x{tp}"
    token_ids = _chunk_token_ids(N_CHUNKS, hf_config.vocab_size)

    for c in range(N_CHUNKS):
        lo, hi = c * CHUNK, (c + 1) * CHUNK - 1
        s_lo, s_hi = _expected_scale(lo), _expected_scale(hi)
        logger.info(
            f"chunk {c}: positions {lo}..{hi}  llama4 scale "
            f"{s_lo:.5f}..{s_hi:.5f}{'  (uniform)' if s_lo == s_hi else '  (mixed)'}"
        )

    runtime, kv_caches = _build(mesh_device, hf_config, weight_cache_path, num_layers, num_links, use_trace=True)
    try:
        runtime.compile(kv_caches)
        runtime.capture_trace(kv_caches)

        _run_traced(runtime, kv_caches, mesh_device, token_ids, SLOT_TRACED)
        _run_eager_reference(runtime, kv_caches, mesh_device, token_ids, SLOT_EAGER)

        betas = [layer.mla._llama4_beta for layer in runtime.model.layers]
        assert all(b is not None for b in betas), "llama_4_scaling_beta is not reaching ttMLA; the control is inert"
        for layer in runtime.model.layers:
            layer.mla._llama4_beta = None
        try:
            _run_eager_reference(runtime, kv_caches, mesh_device, token_ids, SLOT_CONTROL)
        finally:
            for layer, b in zip(runtime.model.layers, betas):
                layer.mla._llama4_beta = b

        read = lambda slot: _read_slot(
            kvpe_cache=kv_caches.kvpe,
            mesh_device=mesh_device,
            sp=sp,
            slot=slot,
            num_layers=num_layers,
            seq_len_cache=TOTAL_LEN,
        )
        traced, eager, control = read(SLOT_TRACED), read(SLOT_EAGER), read(SLOT_CONTROL)
    finally:
        del runtime
        gc.collect()

    agree = _per_layer_pcc(traced, eager)
    sens = _per_layer_pcc(eager, control)
    for i in sorted(agree):
        logger.info(f"  KV layer {i}: traced-vs-eager {agree[i]:.6f}   eager-vs-temperature-1.0 {sens[i]:.6f}")

    # Sensitivity FIRST: if the comparison cannot see a wrong temperature, the agreement numbers
    # below mean nothing, and reporting them as a pass would be the failure mode #55126 is about.
    assert sens[0] >= TRACED_VS_EAGER_KV_PCC, (
        f"KV layer 0 moved ({sens[0]:.6f}) when only the query temperature changed. Layer 0's KV is "
        "computed from the embeddings and must be temperature-independent; if it moved, the control "
        "perturbed something other than the temperature and proves nothing"
    )
    sens_deep = min(v for i, v in sens.items() if i >= 1)
    assert sens_deep < TRACED_VS_EAGER_KV_PCC, (
        f"suppressing the llama4 query temperature left KV layers 1..{num_layers - 1} at PCC "
        f"{sens_deep:.6f}, still above the {TRACED_VS_EAGER_KV_PCC} bar this test asserts for "
        "agreement. The comparison would therefore PASS with a wrong temperature and is not evidence "
        "for #55126. Fix the comparison (more layers, or compare hidden states) before trusting it"
    )
    logger.info(
        f"sensitivity OK: temperature 1.0 drives KV layers 1..{num_layers - 1} to {sens_deep:.6f}, "
        f"under the {TRACED_VS_EAGER_KV_PCC} bar -- the comparison below can see a wrong temperature"
    )

    assert agree[0] >= TRACED_VS_EAGER_KV_PCC, (
        f"KV layer 0 traced-vs-eager PCC {agree[0]:.6f} < {TRACED_VS_EAGER_KV_PCC}. Layer 0's KV comes "
        "from the embeddings and cannot be affected by the query temperature, so this is a general "
        "trace-replay divergence, not the llama4 scale"
    )
    agree_deep = min(v for i, v in agree.items() if i >= 1)
    assert agree_deep >= TRACED_VS_EAGER_KV_PCC, (
        f"KV min PCC over layers 1..{num_layers - 1} is {agree_deep:.6f} < {TRACED_VS_EAGER_KV_PCC} "
        f"while layer 0 agrees ({agree[0]:.6f}). That is the signature of a wrong per-chunk query "
        f"temperature on the traced path: layer 0 is temperature-independent, deeper layers are not. "
        f"For scale, temperature 1.0 drives the same layers to {sens_deep:.6f}"
    )
