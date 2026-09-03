# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Traced-vs-eager numerical equivalence for ``TtPrefillRuntime.prefill_chunk``.

WHY THIS FILE EXISTS. ``prefill_chunk`` is the serving entry point, and until now nothing drove it
from a test: ``prefill_runner.py`` is its only production caller (the second caller,
``tt_prefill_runtime.py``'s ``compile()`` warm-up, is inside the NON-traced branch and never reaches
the traced code), and no workflow runs the runner. So the traced path had no numerical coverage at
all, and Mistral's llama4 query temperature -- the thing
https://github.com/tenstorrent/tt-metal/issues/55126 is about -- is specifically invisible to the
coverage that does exist: ``write_chunk_metadata``'s docstring records that an unrefreshed
(ones-initialised) scale buffer silently applies temperature 1.0 and moves the chunked PCC gate by
~0.002 against a 0.98 threshold. A test that cannot see a wrong temperature is not a test of it, so
this file measures its own sensitivity rather than assuming it (see the control test below).

WHAT IS COMPARED. On a single (last) rank ``prefill_chunk`` returns no activation -- the populated KV
cache IS the output -- so the comparison is the KV cache the two paths wrote, layer by layer. That is
not a weaker check than comparing hidden states: the query temperature multiplies ``tt_q`` only, so it
changes SDPA output -> o_proj -> residual -> the next layer's input -> the next layer's K/V. Layer 0's
KV is computed from the embeddings and is therefore identical under any temperature; it is the
control that says the two paths agree at all, and layers >= 1 carry the signal. Both facts are
asserted separately below.

It also turns the one open KV question into positive evidence for free. A
``chunked_padded[...-traced]`` row failed once with KV PCC 0.001886; the reading is environmental (the
failure moved legs on re-dispatch, the two failures had different modes, near-zero PCC is a fabric
signature rather than numerical drift, and it has been green since) but that reading rests on absence
of failure. Comparing traced KV against eager KV directly is a positive statement instead.

HOW FAR IT HAS TO RUN. The scale is ``1 + beta*ln(1 + floor(pos/orig_max))`` with beta=0.1 and
orig_max=8192, so at chunk 5120 the chunk boundaries and the 8192 bands do not line up:

    chunk 0   pos      0..  5119   floor 0     scale 1.00000   <- uniform, proves NOTHING
    chunk 1   pos   5120.. 10239   floor 0..1  mixed
    chunk 2   pos  10240.. 15359   floor 1     scale 1.06931   <- first uniform non-unity chunk
    chunk 3   pos  15360.. 20479   floor 1..2  mixed
    chunk 4   pos  20480.. 25599   floor 2..3  mixed
    chunk 5   pos  25600.. 30719   floor 3     scale 1.13863   <- a SECOND, DIFFERENT uniform value

A run that stops at chunk 0 (offset 5120, the natural "one chunk" test) passes whether or not the fix
works, because the scale there is exactly 1.0 and a stale ones-buffer is numerically correct. Six
chunks is the minimum that (a) gets past 8192 at all, (b) reaches two DIFFERENT uniform non-unity
values, which is what distinguishes "the buffer is refreshed per chunk" from "a buffer was built once
and reused", and (c) exercises the mixed chunks, where ``rotated_chip_positions`` scatters a window's
rows across chips so the row -> position map is not ``kv_actual_isl + row``.

SLOTS, NOT RE-ZEROING. The KV cache holds ``num_users * num_layers`` user-major slots, and both
``compile()`` and ``capture_trace()`` run warm forwards that write real KV. Rather than re-zero the
cache between passes, each pass owns a disjoint user slot: 0 is the warm-up scratch every setup path
already writes, and the measured passes take 1/2/3. The traced path reads its slot from the packed
metadata record, so this exercises that read rather than working around it.

ORDERING. capture -> traced loop -> eager loop, deliberately: the replay runs with the allocator in
the state the capture saw, and the eager reference (which bakes in no addresses) is the pass that
tolerates running second.

STATUS: this test FAILS TODAY, by design. ``prefill_chunk``'s traced branch raises for any model with
``ChunkMetadata.llama4_scale`` set, which is Mistral only. It is written first so the fix in #55126 is
verifiable; see that issue for the chosen design (one pre-built device buffer per deterministic
``k * chunk_size`` offset, device-to-device copied in per chunk).
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

# Disjoint user slots, so no pass has to re-zero the cache under another's writes.
SLOT_WARMUP = 0  # written by compile() / capture_trace()'s warm forwards; never compared
SLOT_TRACED = 1
SLOT_EAGER = 2
SLOT_CONTROL = 3  # eager with the temperature suppressed; the sensitivity control
NUM_USERS = 4

# Traced and eager run the SAME ops on the SAME device and differ only in how they are dispatched, so
# this is an agreement bar, not an accuracy bar: it should read ~0.999, and anything materially below
# it means the replay is not reproducing the eager forward. Deliberately well above the
# traced-vs-golden bar (TRACE_KV_CACHE_PCC_THRESHOLD = 0.96), which has to absorb real bf8_b
# accumulation drift that this comparison does not.
TRACED_VS_EAGER_KV_PCC = 0.99


def _packed_metadata_msg(mesh_device, slot_id: int, actual_start: int, actual_end: int) -> ttnn.Tensor:
    """The packed [1,1,1,3] uint32 record the traced path consumes on-device.

    In serving this arrives from the H2D socket (``inbound_socket_service_sync``) or a D2D receive;
    ``prefill_chunk`` only ever copies it into its own persistent buffer, so any tensor with this
    shape/dtype/layout is a faithful stand-in and the test needs no socket. Word order is
    (slot_id, actual_start, actual_end), matching ``_meta3_dev`` and ``_decode_metadata``.
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
        kv_only_last_layer=False,
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
    """Eager reference on a TRACED runtime, by calling the model directly.

    prefill_chunk dispatches on ``self.config.use_trace``, so a use_trace=True runtime cannot also
    serve as its own eager reference through that entry point -- it would take the traced branch
    again and compare a replay against itself. Two runtimes would keep both sides on prefill_chunk,
    at the cost of building and loading the model twice and holding both resident while one of them
    holds a capture; one runtime plus a direct forward is the cheaper trade and keeps the side UNDER
    TEST (traced) on the real entry point.

    The argument list mirrors prefill_chunk's eager branch exactly, including the defaults it leaves
    implicit: no d2h_service, no metadata_msg, no layer callbacks, and metadata unset so ttMLA takes
    its host-scalar path and derives the scale from kv_actual_isl.
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


@pytest.mark.parametrize("num_layers", [4], ids=["L4"])
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
    """Traced ``prefill_chunk`` must write the same KV as eager across 6 chunks (30,720 tokens), and
    the comparison that says so must be shown capable of failing.

    THREE PASSES, ONE MODEL. All three run on a single traced runtime, so the leg pays one weight
    load rather than three:

      traced   prefill_chunk, the real serving entry point           -> slot 1   [under test]
      eager    model.forward, prefill_chunk's eager branch inlined   -> slot 2   [reference]
      control  eager with _llama4_beta suppressed                    -> slot 3   [sensitivity]

    The control is the point of the file as much as the comparison is. An unrefreshed scale buffer
    is ones-initialised, which applies temperature 1.0 and moves the chunked PCC gate by ~0.002
    against a 0.98 threshold -- so a PCC assertion that is not itself shown to notice a wrong
    temperature would repeat the exact mistake #55126 exists to fix. Suppressing ``_llama4_beta``
    skips the multiply, which is precisely what a stale ones-buffer computes. Running it in the SAME
    invocation, on the same weights and the same device state, is stronger evidence than a separate
    test would be: the only difference between slots 2 and 3 is the temperature.

    Layer 0 is asserted in both directions. Its KV comes from the embeddings, so it must MATCH under
    traced-vs-eager (a mismatch means the replay is broken generally, not the scale) and must ALSO
    match under eager-vs-control (a mismatch means the control perturbs more than the temperature
    and proves nothing). Layers >= 1 carry the signal in both.

    FAILS TODAY on the traced branch's llama4 guard for any model with ChunkMetadata.llama4_scale
    set, which is Mistral only. That is the bug; see #55126.
    """
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    assert num_layers >= 2, (
        "layer 0's KV is computed from the embeddings and is identical under any query temperature, "
        "so a 1-layer run cannot see the scale at all; layers >= 1 carry the signal"
    )

    hf_config = config_only
    hf_config.max_seq_len = TOTAL_LEN
    sp = tuple(mesh_device.shape)[0]
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
