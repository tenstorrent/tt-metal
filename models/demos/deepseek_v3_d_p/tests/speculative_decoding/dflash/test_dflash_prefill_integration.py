# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration test: full 61-layer Kimi prefill transformer → DFlash drafter context-KV (issue #49586, Phase 1b).

Flow:
  A. Build + run the WHOLE verifier prefill (all 61 Kimi layers) by constructing ``TtPrefillTransformer``
     directly (no ``run_model`` dependency). Weight handling is IDENTICAL to ``test_prefill_transformer``'s
     ``run_model``: pretrained → the memory-bounded ``load_and_compute_layer_by_layer`` + TTNN weight cache;
     random → ``create_hf_model`` + ``extract_tt_state_dict``. We pass an ``on_layer_hidden`` callback into
     the forward so that ONLY the 6 target layers ([1,12,24,35,47,58]) tap their output residual stream, ON
     DEVICE, into the drafter (SP-gathered to full seq, kept in DRAM). The other 55 layers aren't captured.
  B. Feed those SAME 6 target outputs to (i) the device ``TtDFlashDrafter`` (already holding them in DRAM)
     and (ii) the real HF ``DFlashDraftModel``; PCC their per-layer context K/V. The HF reference is a CPU
     torch model, so the 6 taps are read to host ONCE (only those 6) to feed it — the device drafter path
     stays entirely in DRAM.

``use_pretrained`` toggles BOTH the verifier (real TTNN-cached Kimi vs random) and the drafter weights
(real checkpoint vs seeded random, loaded into both the device drafter and the HF ref — via the conftest
fixtures).

MEMORY NOTE: a random verifier at 61 layers is infeasible — ``create_hf_model`` materializes the whole
model in host RAM (Kimi's 384 experts × 60 MoE layers ≈ 2 TB → OOM). So the 61-layer run is PRETRAINED
(``KIMI_K2_6_HF_MODEL`` + ``TT_KIMI_PREFILL_TTNN_CACHE``, memory-bounded layer-by-layer). The random leg
skips at >MAX_RANDOM_LAYERS with guidance.

Requires: a Blackhole galaxy; ``$DFLASH_HF_MODEL`` (drafter ``config.json`` [+ ``model.safetensors`` for the
pretrained axis]); for the pretrained axis also ``KIMI_K2_6_HF_MODEL`` + ``TT_KIMI_PREFILL_TTNN_CACHE``.

    DFLASH_HF_MODEL=/path/to/Kimi-K2.6-DFlash \
    KIMI_K2_6_HF_MODEL=/path/to/Kimi-K2.6 TT_KIMI_PREFILL_TTNN_CACHE=/path/to/kimi_ttnn_cache MESH_DEVICE=8x4 \
    pytest models/demos/deepseek_v3_d_p/tests/speculative_decoding/dflash/test_dflash_prefill_integration.py -svv -k pretrained

NOTE (bring-up): NOT yet run on hardware. Likely iteration points: the on-device SP-gather in the tap, the
concat-mode grouped-shard fc, and the direct verifier construction (kept in lockstep with run_model).
"""

import gc

import pytest
import torch
from loguru import logger

import ttnn
from conftest import is_galaxy
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.dflash.tt_dflash_drafter import TtDFlashDrafter
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    create_hf_model,
    extract_tt_state_dict,
    load_and_compute_layer_by_layer,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.999
SEQ_LEN_5K = 5120  # matches test_prefill_transformer's "5k" config; 5120 / sp_factor(8) = 640 tok/chip
EXPECTED_TARGET_LAYERS = (1, 12, 24, 35, 47, 58)
MAX_RANDOM_LAYERS = 12


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole")
@pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"])
@pytest.mark.parametrize("temperature", [0.0], ids=["greedy"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.parametrize("is_balanced", [False], ids=["non_balanced"])
@pytest.mark.parametrize("seq_parallel", [False, True], ids=["seq_repl", "seq_par"])
@pytest.mark.parametrize("isl_total, dispatch_buffer_capacity_factor", [(SEQ_LEN_5K, 8)], ids=["5k"])
@pytest.mark.parametrize(
    "num_layers",
    [pytest.param(61, marks=pytest.mark.skipif(not is_galaxy(), reason="full 61-layer prefill only on Galaxy"))],
    ids=["61_layers"],
)
@pytest.mark.parametrize("n_routed_experts, gate_fallback_mode", [(384, GateComputeMode.DEVICE)], ids=["e384_device"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.timeout(0)
def test_dflash_prefill_integration(
    variant,
    config_only,
    mesh_device,
    device_params,
    is_balanced,
    seq_parallel,
    isl_total,
    dispatch_buffer_capacity_factor,
    num_layers,
    n_routed_experts,
    gate_fallback_mode,
    num_links,
    topology,
    use_pretrained,
    temperature,
    tokenizer,
    request,
    drafter_cfg,
    drafter_state_dict,
    hf_context_kv,
):
    if not use_pretrained and num_layers > MAX_RANDOM_LAYERS:
        pytest.skip(
            f"random verifier at {num_layers} layers materializes the whole Kimi model in host RAM "
            f"(384 experts × 60 MoE layers ≈ 2 TB → OOM). Use -k pretrained (KIMI_K2_6_HF_MODEL + "
            f"TT_KIMI_PREFILL_TTNN_CACHE, memory-bounded layer-by-layer), or a smaller num_layers."
        )

    # The HF reference drafter, its config (drafter_cfg), and its weights (drafter_state_dict) come from the
    # conftest fixtures — built in setup, so a missing $DFLASH_HF_MODEL skips cleanly. use_pretrained gates
    # both the verifier weights (below) and the drafter weights.
    dcfg = drafter_cfg
    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    sp_factor, tp_factor = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert (
        dcfg.num_key_value_heads % tp_factor == 0
    ), f"kv_heads {dcfg.num_key_value_heads} not divisible by tp {tp_factor}"
    assert num_layers > max(
        dcfg.target_layer_ids
    ), f"verifier num_layers {num_layers} must exceed the last drafter target layer {max(dcfg.target_layer_ids)}"
    # The drafter config must carry the expected Kimi target layers — the ones the hook will tap.
    assert (
        tuple(dcfg.target_layer_ids) == EXPECTED_TARGET_LAYERS
    ), f"drafter target_layer_ids {tuple(dcfg.target_layer_ids)} != expected {EXPECTED_TARGET_LAYERS}"
    H = dcfg.hidden_size
    drafter_sd = drafter_state_dict  # SAME weights the HF reference holds — see conftest hf_drafter/drafter_state_dict

    # Device drafter (concat mode: store the 6 taps, one grouped-shard fc(concat) at write time).
    drafter = TtDFlashDrafter(
        mesh_device,
        dcfg,
        state_dict=drafter_sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        fc_mode="concat",
        seq_parallel=seq_parallel,
    )
    drafter.reset()
    target_ids = set(dcfg.target_layer_ids)
    tapped = []  # records the GLOBAL layer indices that actually fire a tap — asserted == EXPECTED_TARGET_LAYERS

    def on_layer_hidden(global_idx, h):
        # ONLY the 6 target layers are kept, ON DEVICE, in DRAM. h is [1,1,seq/sp,H/tp] (SP-sharded on seq,
        # TP-sharded on hidden). Phase-1 (seq-replicated cache): SP-gather to the full sequence → DRAM, then
        # hand to the drafter. seq_parallel: leave seq SP-sharded — clone (drafter takes ownership + frees
        # its tap; h is the verifier's live residual) and tap this chip's own slice, no gather. No host copy.
        if global_idx not in target_ids:
            return
        tapped.append(global_idx)
        if seq_parallel:
            drafter.tap(ttnn.clone(h), global_idx)  # own a private copy of the live SP-sharded residual slice
        else:
            h_full = ttnn.all_gather(
                h,
                dim=2,
                cluster_axis=sp_axis,
                num_links=num_links,
                topology=topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            drafter.tap(h_full, global_idx)  # drafter takes ownership (DRAM); freed in write_kv_cache

    # ---- Phase A: build + run the full 61-layer verifier DIRECTLY (weight handling identical to
    #      test_prefill_transformer.run_model). on_layer_hidden taps ONLY the 6 target layers on device. ----
    config = config_only
    config.max_seq_len = isl_total
    isl_per_chip = isl_total // sp_factor
    padding_side = tokenizer.padding_side
    torch.manual_seed(42)
    assert not is_balanced, "this test assumes non-balanced (contiguous) SP token sharding"

    # Weights — SAME as run_model:
    #   pretrained → real Kimi via the memory-bounded layer-by-layer loader into a TTNN weight cache;
    #   random     → create_hf_model + extract_tt_state_dict (materializes the whole model → OOM at 61).
    if use_pretrained:
        model_path = request.getfixturevalue("model_path")
        wcp = request.getfixturevalue("weight_cache_path")  # None unless real weights ($KIMI_K2_6_HF_MODEL) present
        if wcp is None:
            pytest.skip(
                "pretrained verifier needs real Kimi weights: set KIMI_K2_6_HF_MODEL (+ TT_KIMI_PREFILL_TTNN_CACHE)"
            )
        rows, cols = mesh_shape
        effective_cache_path = wcp / f"{rows}x{cols}"
        effective_cache_path.mkdir(parents=True, exist_ok=True)
        experts_per_chip = n_routed_experts // (rows * cols)
        if not TtPrefillTransformer.check_cache_complete(
            effective_cache_path, num_layers, experts_per_chip, first_k_dense=variant.model_config.NUM_DENSE_LAYERS
        ):
            logger.info("Building TTNN weight cache from real Kimi weights (layer-by-layer)...")
            load_and_compute_layer_by_layer(
                variant=variant,
                model_path=model_path,
                config=config,
                num_layers=num_layers,
                compute_reference=False,  # our reference is the HF drafter (Phase B), not the verifier
                build_ttnn_cache=True,
                weight_cache_path=effective_cache_path,
                mesh_device=mesh_device,
                seq_len=isl_total,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                gate_fallback_mode=gate_fallback_mode,
            )
        verifier_state_dict = {}  # cache-backed
    else:
        logger.info(f"Building random-weight Kimi verifier: {num_layers} layers, {n_routed_experts} experts")
        hf_model = create_hf_model(variant, config, num_layers, n_routed_experts=n_routed_experts)
        verifier_state_dict = extract_tt_state_dict(variant, hf_model)
        del hf_model
        gc.collect()
        effective_cache_path = None

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict=verifier_state_dict,
        num_layers=num_layers,
        seq_len=isl_total,
        is_balanced=is_balanced,
        padding_side=padding_side,
        dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=effective_cache_path,  # real cache (pretrained) or None (random)
        lm_head_is_column_parallel=True,
    )
    del verifier_state_dict
    gc.collect()

    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )
    token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)
    tt_tokens = ttnn.from_torch(
        token_ids.reshape(sp_factor, 1, isl_per_chip),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )

    logger.info(f"Phase A: running {num_layers}-layer Kimi verifier forward (use_pretrained={use_pretrained})")
    transformer(
        tt_tokens,
        tt_kvpe_cache,
        actual_isl=isl_total,
        return_intermediates=False,
        read_profiler=False,
        temperature=temperature,
        on_layer_hidden=on_layer_hidden,
    )

    # The hook must have tapped EXACTLY the target layers [1,12,24,35,47,58] during the forward — no more,
    # no fewer — and each must have landed in the drafter's DRAM tap slots.
    assert sorted(tapped) == list(EXPECTED_TARGET_LAYERS), (
        f"on_layer_hidden tapped {sorted(tapped)}, expected {list(EXPECTED_TARGET_LAYERS)} "
        f"(check the on_layer_hidden hook + the target-layer guard)"
    )
    assert all(t is not None for t in drafter._taps), "a drafter tap slot is empty after the forward"
    logger.info(f"tap check OK — tapped exactly layers {sorted(tapped)}")

    # ---- Phase B ----
    # The 6 taps live in the drafter's DRAM (drafter._taps, in target order). Read ONLY those 6 to host
    # (they're SP-replicated + TP-sharded on hidden) to feed the CPU HF reference — the device path never
    # left DRAM. This is the sole host touch, and it must happen BEFORE write_kv_cache consumes the taps.
    def _tap_to_host(t):  # -> host [1, seq, H]; TP concatenated on hidden → full H
        if seq_parallel:
            # SP-sharded on seq: concat SP along seq(dim2), TP along hidden(dim3) → full [1,1,seq,H]
            host = ttnn.to_torch(
                t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_shape)
            )
            return host.reshape(1, isl_total, H).float()
        # SP-replicated: concat SP on dim0 (8 identical copies), take one replica; TP on hidden(dim3)
        host = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=mesh_shape)
        )
        return host[0:1].reshape(1, isl_total, H).float()

    target_hiddens = [_tap_to_host(drafter._taps[j]) for j in range(len(dcfg.target_layer_ids))]
    ctx = torch.cat(target_hiddens, dim=-1)  # [1, seq, n*H] — the fc input (concat over target layers)
    assert ctx.shape[-1] == dcfg.target_feature_size, f"ctx feature {ctx.shape[-1]} != {dcfg.target_feature_size}"

    # HF reference: real drafter forward on the SAME 6 target hiddens, per-layer context (k, v)
    real = hf_context_kv(ctx)

    # Device drafter: finalize from the DRAM taps (concat → grouped-shard fc → per-layer k/v/norm/rope).
    drafter.write_kv_cache()
    ttnn.synchronize_device(mesh_device)

    # seq_parallel: cache SP-sharded on seq → concat SP along seq(dim2), TP along kv-head(dim1); the
    # host[:num_layers] slice is then a no-op. Phase-1: SP-replicated → take the first SP replica.
    read_dims = (2, 1) if seq_parallel else (0, 1)

    def _read(cache):
        host = ttnn.to_torch(
            cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=read_dims, mesh_shape=mesh_shape)
        )
        return host[: dcfg.num_hidden_layers][:, :, :isl_total, :].float()  # [n_layers, kv_heads, seq, head_dim]

    dk, dv = _read(drafter.k_cache), _read(drafter.v_cache)

    fails = []
    for i in range(dcfg.num_hidden_layers):
        rk, rv = real[i]
        ok_k, pcc_k = comp_pcc(rk, dk[i], PCC_THRESHOLD)
        ok_v, pcc_v = comp_pcc(rv, dv[i], PCC_THRESHOLD)
        logger.info(f"draft layer {i}: K pcc={pcc_k} (ok={ok_k})  V pcc={pcc_v} (ok={ok_v})")
        if not ok_v:
            fails.append(f"V[{i}]={pcc_v}")
        if not ok_k:
            fails.append(f"K[{i}]={pcc_k}")
    assert not fails, f"drafter context-KV PCC < {PCC_THRESHOLD}: {fails}"
    logger.success(
        f"DFlash prefill integration PASSED: {num_layers}-layer verifier → 6 on-device target taps → drafter, "
        f"all {dcfg.num_hidden_layers} draft layers K≈V≥{PCC_THRESHOLD} "
        f"(weights={'pretrained' if use_pretrained else 'random'})"
    )
