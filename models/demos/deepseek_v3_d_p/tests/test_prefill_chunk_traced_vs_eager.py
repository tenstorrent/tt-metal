# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Traced-vs-eager numerical equivalence for ``TtPrefillRuntime.prefill_chunk`` (#55126).

The llama4 query temperature is invisible to the existing coverage. Suppressing it moves the traced
reference gate's asserted layers from 0.989719 to 0.981986, against a 0.96 bar, so that test passes
with the bug present; it asserts layers 0..10 and the temperature compounds with depth. So this test
measures its own sensitivity. Three passes on one traced runtime into disjoint KV slots -- traced,
eager, and eager with the temperature suppressed -- and the control is asserted first.

Six chunks (30,720 tokens): chunk 0's scale is exactly 1.0, chunk 2 is the first uniform non-unity
value and chunk 5 a second, and two distinct values separate "refreshed per chunk" from "built once".

The comparison is the KV cache, since on a single rank prefill_chunk returns no activation. Layer 0
is temperature-independent and acts as the control for whether the two paths agree at all.
"""
import copy
import gc
import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_3.block import glm_decoder_layer_reference, rms_norm
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_transformer_state_dict
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, assert_gate_mode_matches_adapter
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural, gather_cache_tp0, unrotate_cache_layer
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


def _read_slots(
    kvpe_cache, mesh_device, sp: int, slots: list[int], num_layers: int, seq_len_cache: int
) -> list[list[torch.Tensor]]:
    """Per-layer KV for each requested slot, un-rotated out of the block-cyclic device layout and
    trimmed to the tokens actually written. Slot layout is user-major (``slot * num_layers + layer``),
    the same linearisation ``update_padded_kv_cache`` computes on device.

    Takes every slot at once because gather_cache_tp0 composes the WHOLE cache off the device: one
    gather for three slots instead of three gathers of the same unchanged cache.
    """
    cache_full = gather_cache_tp0(kvpe_cache.storage, mesh_device)
    pos = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    return [
        [unrotate_cache_layer(cache_full[slot * num_layers + i], pos, TOTAL_LEN) for i in range(num_layers)]
        for slot in slots
    ]


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
        # Load-bearing for a traced build: False runs the forward into norm + lm_head, whose readback
        # records an event, which is illegal inside a capture.
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
        # Real ints and the packed message, as prefill_runner._compute_and_send calls it. The traced
        # path reads the scalars on-device; the host ints select this chunk's query-scale buffer.
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
# compounds then saturates: at 4 layers the control sits above the bar and the comparison cannot see
# a wrong temperature. 8 puts it 10x clear.
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
    # The fixture returns the cache root; the per-mesh files live under "{sp}x{tp}". Passing the root
    # makes TtPrefillRuntime report an incomplete cache.
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

        traced, eager, control = _read_slots(
            kvpe_cache=kv_caches.kvpe,
            mesh_device=mesh_device,
            sp=sp,
            slots=[SLOT_TRACED, SLOT_EAGER, SLOT_CONTROL],
            num_layers=num_layers,
            seq_len_cache=TOTAL_LEN,
        )
    finally:
        # Required before the mesh_device fixture closes the device: the captured MeshTraceBuffers live
        # inside the MoE-overlap SubDeviceManagers, and closing with both registered segfaults.
        runtime.release_trace()
        del runtime
        gc.collect()

    agree = _per_layer_pcc(traced, eager)
    sens = _per_layer_pcc(eager, control)
    for i in sorted(agree):
        logger.info(f"  KV layer {i}: traced-vs-eager {agree[i]:.6f}   eager-vs-temperature-1.0 {sens[i]:.6f}")

    # Sensitivity first: if the comparison cannot see a wrong temperature, the agreement numbers below
    # mean nothing.
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


# ---------------------------------------------------------------------------------------------------------
# GLM-5.3, synthetic weights: a PARTIAL single chunk through the traced path. The chunks above are all full,
# and so are the traced chunks every other GLM test replays, so a partial first chunk (any prompt shorter
# than one chunk) has no traced-vs-eager coverage. Synthetic weights need no checkpoint or ttnn cache.
#
# Two dense layers: layer 0's KV comes from the embeddings and is the control; layer 1's KV is computed
# from layer 0's output, so it shows any difference in layer 0's attention (traced vs eager).
GLM_LAYERS = 2
# actual_end of one chunk starting at 0. 5,120 is the full-chunk control.
GLM_CHUNK_ENDS = [CHUNK, 2048, 896, 256]
GLM_SLOT_WARMUP = 0  # compile() / capture_trace() write here


def _glm_slots(case: int) -> tuple[int, int, int]:
    # (traced, eager, eager through the metadata path): the third runs the traced path's ops without a trace,
    # reading (slot, start, end) from device tensors, to tell trace replay apart from the metadata path.
    return 1 + 3 * case, 2 + 3 * case, 3 + 3 * case


def _glm_natural_slot_rows(rows: torch.Tensor, stripes: int, chunk: int) -> torch.Tensor:
    """[layers, seq, D] in gather_cache_natural's linear chip order (TP-deduped: stripe s*tp + t holds
    chunk_local = chunk / stripes rows of every chunk) -> natural token order."""
    seq = rows.shape[1]
    local, cl = seq // stripes, chunk // stripes
    out = torch.empty_like(rows)
    for i in range(stripes):
        for c in range(local // cl):
            out[:, c * chunk + i * cl : c * chunk + (i + 1) * cl] = rows[
                :, i * local + c * cl : i * local + (c + 1) * cl
            ]
    return out


def _glm_cpu_layer_caches(config, state_dict, token_ids, end):
    """CPU truth for one chunk [0, end): the KVPE and indexer-K that layers 0 and 1 write, each [end, D]."""
    emb = state_dict["embed_weight"][torch.tensor(token_ids[:end])].unsqueeze(0)  # [1, end, hidden]
    l0, l1 = state_dict["layers"][0], state_dict["layers"][1]
    ref0 = SparseMLAReference(config, l0["mla_weights"], seq_len=CHUNK)
    out0, _ = glm_decoder_layer_reference(
        config,
        l0["mla_weights"],
        l0["attn_norm_weight"],
        l0["ffn_norm_weight"],
        emb,
        CHUNK,
        ffn_weights=l0["ffn_weights"],
        mla_ref=ref0,
        actual_start=0,
        actual_end=end,
    )
    ref1 = SparseMLAReference(config, l1["mla_weights"], seq_len=CHUNK)
    ref1.forward(rms_norm(out0, l1["attn_norm_weight"], config.rms_norm_eps), actual_start=0, actual_end=end)
    kvpe = [r.kvpe_cache[0, 0, :end].float() for r in (ref0, ref1)]
    index = [r.index_cache[0, :end].float() for r in (ref0, ref1)]
    return {"kvpe": kvpe, "indexer_k": index}


def _glm_meta_scalar(mesh_device, value: int) -> ttnn.Tensor:
    """One 1-element uint32 replicated DRAM scalar, as the runtime's per-chunk metadata."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rel_err(dev: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-token relative L2 error [tokens]."""
    return (dev - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=GLM53Adapter.l1_small_size,
                trace_region_size=512 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_chunk_traced_matches_eager_partial(variant, config_only, mesh_device, device_params, num_links):
    """Traced prefill_chunk must write the same KV as eager for one chunk [0, actual_end), full or partial.

    One traced runtime; per actual_end, traced -> one slot, eager (model.forward, host scalars) -> another.
    Same ops on the same device, so the real tokens' KV must match bit for bit.
    """
    gate_mode = GateComputeMode.DEVICE_FP32
    assert_gate_mode_matches_adapter(variant, gate_mode)
    sp, tp = tuple(mesh_device.shape)
    config = copy.deepcopy(config_only)
    config.max_seq_len = CHUNK
    num_users = 1 + 3 * len(GLM_CHUNK_ENDS)
    rt_config = TtPrefillRuntimeConfig(
        num_layers=GLM_LAYERS,
        max_seq_len=CHUNK,
        mesh_shape=(sp, tp),
        chunk_size=CHUNK,
        num_users=num_users,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=gate_mode,
        model_cfg=GLM53Config,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        use_trace=True,
        tp_shard_kv=True,
    )
    mesh_device.enable_program_cache()
    state_dict = glm_random_transformer_state_dict(config, GLM53Config, GLM_LAYERS)
    # The CPU reference's view of the weights, taken before the runtime consumes the state dict.
    cpu_weights = {
        "embed_weight": state_dict["embed_weight"],
        "layers": [
            {k: dict(v) if isinstance(v, dict) else v for k, v in layer.items()} for layer in state_dict["layers"]
        ],
    }
    runtime = TtPrefillRuntime(mesh_device, config, state_dict, rt_config)
    del state_dict
    index_layers = full_indexer_rank(config, GLM_LAYERS)
    caches = MlaKvCaches(
        kvpe=init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=CHUNK,
            mesh_shape=[sp, tp],
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=GLM_LAYERS,
            num_users=num_users,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=CHUNK,
            mesh_shape=[sp, tp],
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=index_layers,
            num_users=num_users,
            dtype=ttnn.bfloat8_b,
        ),
    )
    ids = torch.randint(0, config.vocab_size, (CHUNK,), generator=torch.Generator().manual_seed(896)).tolist()
    try:
        runtime.compile(caches)
        runtime.capture_trace(caches)
        for case, end in enumerate(GLM_CHUNK_ENDS):
            traced_slot, eager_slot, meta_slot = _glm_slots(case)
            tokens = ids[:end] + [0] * (CHUNK - end)
            msg = _packed_metadata_msg(mesh_device, traced_slot, 0, end)
            runtime.prefill_chunk(
                runtime.make_chunk_input(tokens),
                caches,
                slot_id=traced_slot,
                actual_start=0,
                actual_end=end,
                metadata_msg=msg,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(msg)
            inp = runtime.make_chunk_input(tokens)
            runtime.model.forward(
                inp,
                caches.kvpe,
                actual_isl=CHUNK,
                actual_start=0,
                actual_end=end,
                cache_user_id=eager_slot,
                index_kv_cache=caches.index,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(inp)
            # The traced forward's exact call (runtime._forward_traced), run eagerly: no capture, no replay.
            meta = ChunkMetadata(*(_glm_meta_scalar(mesh_device, v) for v in (meta_slot, 0, end)))
            inp = runtime.make_chunk_input(tokens)
            runtime.model.forward(
                inp,
                caches.kvpe,
                index_kv_cache=caches.index,
                actual_isl=CHUNK,
                actual_start=None,
                actual_end=None,
                cache_user_id=0,
                metadata=meta,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(inp)
            for scalar in meta.scalars:
                ttnn.deallocate(scalar)
        kvpe_rows, _ = gather_cache_natural(caches.kvpe, mesh_device, tp_shard_kv=True)
        index_rows, _ = gather_cache_natural(caches.index, mesh_device, tp_shard_kv=True)
    finally:
        runtime.release_trace()
        runtime.model.release_sub_device_managers()
        del runtime
        gc.collect()

    # Which side is wrong: wherever traced and eager differ, compare each with the CPU reference. The noise bar is
    # the largest device-vs-CPU error on the tokens where the two paths are identical.
    cpu = {}
    failures = []
    for name, rows, layers in (("kvpe", kvpe_rows, GLM_LAYERS), ("indexer_k", index_rows, index_layers)):
        for case, end in enumerate(GLM_CHUNK_ENDS):
            traced_slot, eager_slot, meta_slot = _glm_slots(case)
            traced = _glm_natural_slot_rows(rows[traced_slot * layers : (traced_slot + 1) * layers], sp * tp, CHUNK)
            eager = _glm_natural_slot_rows(rows[eager_slot * layers : (eager_slot + 1) * layers], sp * tp, CHUNK)
            meta = _glm_natural_slot_rows(rows[meta_slot * layers : (meta_slot + 1) * layers], sp * tp, CHUNK)
            for layer in range(layers):
                t, e, m = traced[layer, :end], eager[layer, :end], meta[layer, :end]
                for label, other in (("eager", e), ("traced", t)):
                    n_diff = int(((m - other).abs().amax(dim=-1) > 0).sum())
                    logger.info(
                        f"[0, {end}) {name}[{layer}] eager-metadata vs {label}: {n_diff} of {end} real tokens differ"
                    )
                differ = torch.nonzero((t - e).abs().amax(dim=-1) > 0).flatten()
                _, pcc = comp_pcc(e, t)
                logger.info(
                    f"[0, {end}) {name}[{layer}]: PCC {float(pcc):.6f}, {len(differ)} of {end} real tokens differ"
                    + (f" (first {differ[0].item()}, last {differ[-1].item()})" if len(differ) else "")
                )
                if len(differ):
                    if end not in cpu:
                        cpu[end] = _glm_cpu_layer_caches(config, cpu_weights, ids, end)
                    ref = cpu[end][name][layer]
                    err_t, err_e = _rel_err(t, ref), _rel_err(e, ref)
                    same = torch.ones(end, dtype=torch.bool)
                    same[differ] = False
                    bar = err_e[same].max().item()
                    logger.info(
                        f"[0, {end}) {name}[{layer}] vs CPU: on the {len(differ)} differing tokens, within device noise "
                        f"(rel err <= {bar:.3e}, the max over the {int(same.sum())} identical tokens): "
                        f"eager {int((err_e[differ] <= bar).sum())}/{len(differ)}, "
                        f"traced {int((err_t[differ] <= bar).sum())}/{len(differ)}; median rel err eager "
                        f"{err_e[differ].median().item():.3e}, traced {err_t[differ].median().item():.3e} "
                        f"(identical tokens {err_e[same].median().item():.3e})"
                    )
                    failures.append(
                        f"[0, {end}) {name} layer {layer}: {len(differ)} real tokens differ, first {differ[0].item()}, "
                        f"PCC {float(pcc):.6f}"
                    )
    assert not failures, "traced vs eager KV mismatch:\n" + "\n".join(failures)
