# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillTransformer — verifies composition of embed -> [block x N] -> norm.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model as the reference: creates a model with random or pretrained
weights, extracts those weights into our TT state_dict format, and compares forward passes.

Parametrized over:
- use_pretrained: real pretrained weights from DeepSeek-R1-0528 vs random weights
- input_source: "random", "json_prompts", or InfiniteBench subset (passkey, kv_retrieval, etc.)
- pcc_validation: per-stage PCC check (via return_intermediates) vs shape-only smoke test
- n_routed_experts / capacity_factor / gate_fallback_mode: MoE configurations
"""

import gc
import json
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from conftest import is_galaxy
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import save_norm_output
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ABC_1K_PATH,
    PROMPTS_PATH,
    create_hf_model,
    download_infinitebench_subset,
    extract_tt_state_dict,
    get_4d_causal_mask,
    load_and_compute_layer_by_layer,
    load_reference_cache,
    save_reference_cache,
    tokenize_prompt_to_chat_template,
    tokenize_prompt_to_isl,
)
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.99

# Input sources: "random" = random token IDs, "json_prompts" = test_prompts_1024.json,
# or any InfiniteBench subset name (downloaded on first use via infinitebench_prompt fixture).
INFINITEBENCH_SUBSET_NAMES = {"passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"}


@pytest.skipif(not is_blackhole(), reason="Requires Blackhole.")
@pytest.mark.parametrize("return_kv_cache", [True], ids=["kv_cache"])
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize(
    "input_source",
    ["json_prompts", "abc_1k", "random", "passkey", "kv_retrieval", "longdialogue_qa_eng", "longbook_qa_eng"],
)
@pytest.mark.parametrize("pcc_validation", [True, False], ids=["pcc", "smoke"])
@pytest.mark.parametrize("isl_total", [1024, 6400])
@pytest.mark.parametrize(
    "num_layers",
    [
        12,
        pytest.param(61, marks=pytest.mark.skipif(not is_galaxy(), reason="Testing entire-prefill only on Galaxy")),
    ],
)
@pytest.mark.parametrize(
    "n_routed_experts, capacity_factor, gate_fallback_mode",
    [
        (64, 4, GateComputeMode.HOST_ALL),
        (256, 32, GateComputeMode.HOST_ALL),
        (256, 32, GateComputeMode.DEVICE),
    ],
    ids=["e64_cf4_host", "e256_cf32_host", "e256_cf32_device"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_prefill_transformer(
    config_only,
    mesh_device,
    device_params,
    isl_total,
    num_layers,
    n_routed_experts,
    capacity_factor,
    gate_fallback_mode,
    num_links,
    topology,
    pcc_validation,
    input_source,
    use_pretrained,
    return_kv_cache,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    request,
):
    torch.manual_seed(42)

    # Skip invalid pretrained combinations
    if use_pretrained and n_routed_experts != 256:
        pytest.skip("Pretrained weights only available for 256 experts")

    profiler.clear()
    profiler.start("total_test_time")
    config = config_only
    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    isl_per_chip = isl_total // sp_factor

    weight_type = "pretrained" if use_pretrained else "random"

    # Only enable weight caching for pretrained runs
    if use_pretrained and weight_cache_path is not None:
        rows, cols = mesh_shape
        effective_cache_path = weight_cache_path / f"{rows}x{cols}"
        effective_cache_path.mkdir(parents=True, exist_ok=True)
    else:
        effective_cache_path = None

    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}")
    logger.info(
        f"isl_total={isl_total}, isl_per_chip={isl_per_chip}, "
        f"num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
        f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
        f"input_source={input_source}, pcc_validation={pcc_validation}, "
        f"weights={weight_type}"
    )

    # --- Monkeypatch n_routed_experts ---
    orig_num_routed_experts = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = n_routed_experts

    # --- Cache-aware loading strategy ---
    profiler.start("cache_check")

    from models.demos.deepseek_v3_d_p.utils.cache_utils import check_reference_cache_exists, check_ttnn_cache_complete

    # Check cache states
    experts_per_chip = 256 // (mesh_shape[0] * mesh_shape[1]) if use_pretrained else 8
    ttnn_cache_complete = (
        check_ttnn_cache_complete(effective_cache_path, num_layers, experts_per_chip, tuple(mesh_shape))
        if effective_cache_path
        else False
    )

    cache_key = f"{weight_type}_{input_source}_isl{isl_total}_layers{num_layers}_experts{n_routed_experts}"
    ref_cache_exists = check_reference_cache_exists(cache_key) if pcc_validation else False

    logger.info(f"Cache status: TTNN={ttnn_cache_complete}, Reference={ref_cache_exists}")

    # Determine what we need to load
    need_to_load_weights = not ttnn_cache_complete
    need_to_compute_reference = pcc_validation and not ref_cache_exists
    need_hf_model = need_to_load_weights or need_to_compute_reference

    logger.info(
        f"Loading strategy: need_weights={need_to_load_weights}, "
        f"need_reference={need_to_compute_reference}, "
        f"need_hf_model={need_hf_model}"
    )

    profiler.end("cache_check")

    # --- Create input (needed early for reference computation) ---
    if input_source == "random":
        token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)
    else:
        profiler.start("tokenization")
        tok = request.getfixturevalue("tokenizer")
        if input_source == "json_prompts":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(PROMPTS_PATH))
            prompt_text = prompt_text[0] if isinstance(prompt_text, list) else prompt_text
        elif input_source == "abc_1k":
            from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

            prompt_text = load_prompts_from_json(str(ABC_1K_PATH))
        elif input_source in INFINITEBENCH_SUBSET_NAMES:
            cached_path = download_infinitebench_subset(input_source)
            with open(cached_path) as f:
                prompt_text = json.load(f)["prompt"]
        else:
            raise ValueError(f"Unknown input_source: {input_source}")
        token_ids, attention_mask, tokens = tokenize_prompt_to_isl(tok, max_isl=isl_total, prompt_text=prompt_text)
        profiler.end("tokenization")
        logger.info(
            f"Tokenized {input_source} input shape: {token_ids.shape}, first 10 tokens: {token_ids[0, :10].tolist()}, last 10 tokens: {token_ids[0, -10:].tolist()}"
        )

    # --- Build HF model and/or extract weights based on cache state ---
    profiler.start("weights_creation")

    state_dict = None
    ref_snapshots = None
    ref_kvpe_list = None

    if use_pretrained:
        model_path = request.getfixturevalue("model_path")
        logger.debug(f"{model_path=}")
        if need_hf_model:
            # Use unified loader with flags
            logger.info("Processing layers with unified loader...")
            result = load_and_compute_layer_by_layer(
                model_path=model_path,
                config=config,
                num_layers=num_layers,
                token_ids=token_ids,
                attention_mask=attention_mask,
                compute_reference=need_to_compute_reference,
                build_ttnn_cache=need_to_load_weights,
                weight_cache_path=effective_cache_path,
                mesh_device=mesh_device,
                seq_len=isl_total,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                capacity_factor=capacity_factor,
                gate_fallback_mode=gate_fallback_mode,
            )

            # state_dict is always None (cache built to disk)
            state_dict = {}
            ref_snapshots = result.ref_snapshots
            ref_kvpe_list = result.ref_kvpe_list

            # Save reference cache if computed
            if need_to_compute_reference and ref_snapshots is not None:
                save_reference_cache(cache_key, ref_snapshots, ref_kvpe_list)
                logger.info("Reference cached")
        else:
            # Both caches exist - skip loading entirely
            logger.info("Both caches exist, skipping weight loading")
            state_dict = {}
    else:
        # Random weights - always create HF model
        logger.info("Creating HF model with random weights...")
        hf_model = create_hf_model(config, num_layers, n_routed_experts=n_routed_experts)
        state_dict = extract_tt_state_dict(hf_model)
        del hf_model
        gc.collect()

    profiler.end("weights_creation")

    # --- TT transformer ---
    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        capacity_factor=capacity_factor,
        weight_cache_path=effective_cache_path,
    )
    ttnn.synchronize_device(mesh_device)
    # --- Free memory immediately after transformer creation ---
    del state_dict
    gc.collect()
    logger.info("State dict freed after transformer creation")
    profiler.end("tt_transformer_creation")

    # --- Create external KVPE cache ---
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )

    # --- Shard token_ids to device ---
    # Reshape [1, isl_total] -> [sp_factor, 1, isl_per_chip] for SP sharding
    token_ids_reshaped = token_ids.reshape(sp_factor, 1, isl_per_chip)

    tt_tokens = ttnn.from_torch(
        token_ids_reshaped,
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )

    # --- Forward ---
    profiler.start("tt_forward")
    logger.info("Running TtPrefillTransformer forward...")
    do_return_kv = pcc_validation and return_kv_cache
    result = transformer(tt_tokens, tt_kvpe_cache, return_intermediates=pcc_validation)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    if pcc_validation:
        tt_output, tt_snapshots = result
    else:
        tt_output = result

    # --- Validate output shape ---
    expected_per_device_shape = [1, 1, isl_per_chip, emb_dim // tp_factor]
    output_shape = list(tt_output.shape)
    assert (
        output_shape == expected_per_device_shape
    ), f"Output shape mismatch: got {output_shape}, expected {expected_per_device_shape}"
    logger.info(f"Output shape: {output_shape} (matches expected)")

    # --- Save final norm output ---
    if pcc_validation:
        final_norm_label, final_norm_tensor = tt_snapshots[-1]
        assert final_norm_label == "norm", f"Expected last snapshot to be 'norm', got '{final_norm_label}'"

        save_norm_output(
            norm_tensor=final_norm_tensor,
            test_params={
                "mesh_shape": mesh_shape,
                "isl_total": isl_total,
                "isl_per_chip": isl_per_chip,
                "num_layers": num_layers,
                "n_routed_experts": n_routed_experts,
                "capacity_factor": capacity_factor,
                "gate_fallback_mode": gate_fallback_mode,
                "use_pretrained": use_pretrained,
                "input_source": input_source,
                "topology": str(topology),
                "num_links": num_links,
                "emb_dim": emb_dim,
                "sp_factor": sp_factor,
                "tp_factor": tp_factor,
            },
        )

    # --- PCC check ---
    if pcc_validation:
        profiler.start("pcc_validation")
        if use_pretrained and input_source != "random":
            threshold = 0.97
        elif use_pretrained:
            threshold = 0.95
        elif n_routed_experts < 256:
            threshold = 0.985
        else:
            threshold = PCC_THRESHOLD  # 0.99
        logger.info(f"PCC threshold: {threshold}")

        # Load reference from cache if not already computed
        if ref_snapshots is None:
            logger.info("Loading reference from cache...")
            ref_snapshots, ref_kvpe_list = load_reference_cache(cache_key)
        # else: already computed by load_and_compute_layer_by_layer()

        # Per-stage PCC comparison
        pcc_results = []
        for (label, tt_host), ref_host in zip(tt_snapshots, ref_snapshots):
            try:
                _, pcc = comp_pcc(ref_host.float(), tt_host.float())
                logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
                pcc_results.append((label, pcc))
            except Exception as e:
                logger.error(f"{label:<20s}  PCC comparison failed: {e}")
                pcc_results.append((label, -1.0))

        # Per-layer KVPE PCC comparison — read back from external cache
        if do_return_kv:
            tt_kvpe_all = ttnn.to_torch(
                tt_kvpe_cache,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
            ).to(torch.bfloat16)
            # Shape: [num_layers, tp_factor, seq_total, head_dim] — take first TP replica
            kv_lora_rank = config.kv_lora_rank
            for i, ref_kvpe in enumerate(ref_kvpe_list):
                tt_kvpe_layer = tt_kvpe_all[i : i + 1, :1, :, :]
                label = f"layer_{i}_kvpe"
                try:
                    _, kv_pcc = comp_pcc(
                        ref_kvpe[:, :, :, :kv_lora_rank].float(),
                        tt_kvpe_layer[:, :, :, :kv_lora_rank].float(),
                    )
                    _, pe_pcc = comp_pcc(
                        ref_kvpe[:, :, :, kv_lora_rank:].float(),
                        tt_kvpe_layer[:, :, :, kv_lora_rank:].float(),
                    )
                    logger.info(f"{label:<20s}  KV PCC = {kv_pcc:.6f}, PE PCC = {pe_pcc:.6f}")
                    pcc_results.append((f"{label}_kv", kv_pcc))
                    pcc_results.append((f"{label}_pe", pe_pcc))
                except Exception as e:
                    logger.error(f"{label:<20s}  KVPE PCC comparison failed: {e}")
                    pcc_results.append((f"{label}_kv", -1.0))
                    pcc_results.append((f"{label}_pe", -1.0))

        profiler.end("pcc_validation")

        # Summary table
        logger.info(f"\n{'='*50}")
        logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Status':>8s}")
        logger.info(f"{'-'*50}")
        failures = []
        for label, pcc in pcc_results:
            status = "PASS" if pcc > threshold else ("FAIL" if pcc >= 0 else "ERROR")
            logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
            if pcc <= threshold:
                failures.append((label, pcc))
        logger.info(f"{'='*50}")

        # Store failures for deferred check (after timing report)
        has_pcc_failures = len(failures) > 0

        if not has_pcc_failures:
            logger.success(
                f"TtPrefillTransformer PCC test passed "
                f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
                f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
                f"weights={weight_type})"
            )
        else:
            pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
            logger.error(
                f"TtPrefillTransformer PCC test has failures " f"(num_layers={num_layers}, failures={len(failures)})"
            )
    else:
        logger.success(
            f"TtPrefillTransformer smoke test passed "
            f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
            f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
            f"weights={weight_type})"
        )

    profiler.end("total_test_time")

    # --- Timing report ---
    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    # Restore original config
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = orig_num_routed_experts

    # Deferred PCC failure check (after timing report)
    if pcc_validation and has_pcc_failures:
        pytest.fail(f"PCC below {threshold} at: {pcc_failure_msg}")


def test_tokenize_prompt_to_isl(tokenizer):
    max_isl = 10
    input_ids, attention_mask, tokens = tokenize_prompt_to_isl(
        tokenizer, max_isl=max_isl, prompt_text="This is a test prompt.", debug=True
    )

    logger.debug(f"Input IDs: {input_ids}")
    logger.debug(f"Attention Mask: {attention_mask}")
    logger.debug(f"Tokens: {tokens}")

    assert input_ids.shape == (1, max_isl), f"Expected input_ids shape (1, {max_isl}), got {input_ids.shape}"

    torch.set_printoptions(threshold=float("inf"), edgeitems=3, precision=2, linewidth=200)
    logger.debug(f"4D Causal Attention Mask shape:\n{get_4d_causal_mask(attention_mask, ignore_padding=True)}")
    logger.debug(f"4D Causal Attention Mask Paddshape:\n{get_4d_causal_mask(attention_mask, ignore_padding=False)}")


def test_tokenize_prompt_to_chat_template(tokenizer):
    max_isl = 64
    input_ids, tokens = tokenize_prompt_to_chat_template(
        tokenizer,
        max_isl=max_isl,
        user_prompt="What is the capital of Serbia?",
        system_prompt="You are a helpful assistant.",
        debug=True,
    )
    logger.debug(f"Input IDs: {input_ids}")
    logger.debug(f"Tokens: {tokens}")

    assert input_ids.shape == (1, max_isl), f"Expected input_ids shape (1, {max_isl}), got {input_ids.shape}"


@pytest.mark.parametrize(
    "input_path",
    [
        Path("/tmp/r1/pretrained_abc_1k_isl1024_layers61_experts256.pt"),
        Path(
            "/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Reference-prefill/pretrained_abc_1k_isl1024_layers61_experts256.pt"
        ),
        Path(
            "/workspace/ds_output_all_sources_new_/norm_20260415_114721_mesh8x4_isl1024_L61_e256_cf32_gatehost_all_pretrained_abc_1k.pt"
        ),
    ],
)
def test_first_token_from_reference(input_path, model_path, config_only, tokenizer):
    # Use weights_only=False since this is a trusted local file with custom objects
    logger.info(f"{input_path=}")
    data = torch.load(input_path, weights_only=False)

    if "norm_output" in data:  # this is ttnn output; norm only
        norm_output = data["norm_output"]
        logger.info(f"{norm_output.shape=}")
    elif "ref_snapshots" in data:  # this is torch output; all emb + layers + norm
        logger.info(f"Number of reference snapshots: {len(data['ref_snapshots'])}")
        assert len(data["ref_snapshots"]) == DeepSeekV3Config.NUM_LAYERS + 2  # token embedding + final RMS norm
        norm_output = data["ref_snapshots"][-1]  # Last snapshot's tensor
        logger.warning("Loaded data does not have 'norm_output' key, assuming last snapshot is final rms norm output")
    else:
        logger.warning(data)
        raise ValueError("Loaded data format is unexpected and does not contain 'norm_output' or expected snapshots.")

    #  Remove batch dimension if present
    if norm_output.shape[0] == 1:
        norm_output = norm_output.squeeze(0)
    logger.success("Data loaded successfully.")

    # LM HEAD loading
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

    lazy_sd = LazyStateDict(Path(model_path))
    # lm_head_sd = sub_state_dict(lazy_sd, "model.embed_tokens.")
    lm_head_sd = sub_state_dict(lazy_sd, "lm_head.")
    lm_head_dequant = dequantize_state_dict(lm_head_sd, config_only)
    lm_head_dequant_weight = lm_head_dequant.get("weight")
    assert (
        lm_head_dequant_weight.shape[0] == config_only.vocab_size
    ), f"Expected lm_head_dequant_weight shape[0] to be {config_only.vocab_size}, got {lm_head_dequant_weight.shape[0]}"
    logger.success("LM head weight loaded and dequantized successfully.")

    # Apply LM HEAD
    norm_output = norm_output.to(lm_head_dequant_weight.dtype)
    # norm_output = ref_snapshots[-1][0,:,:].to(lm_head_dequant_weight.dtype)
    logger.debug("Computing logits...")
    with torch.no_grad():
        # lm_head is just a linear layer: logits = hidden @ lm_head_weight.T
        logits = torch.matmul(norm_output, lm_head_dequant_weight.T)  # Shape: [1, 1, vocab_size]
    logger.debug(f"Logits shape: {logits.shape}")

    # Apply sampling
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/generate.py
    def sample(logits, temperature: float = 1.0):
        """
        Samples a token from the logits using temperature scaling.

        Args:
            logits (torch.Tensor): The logits tensor for token predictions.
            temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

        Returns:
            torch.Tensor: The sampled token.
        """
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

    index = -1  # Last token
    for temperature in [0.0, 0.5, 1]:
        first_token = sample(logits=logits[index].clone(), temperature=temperature)
        token_text = tokenizer.decode([first_token.item()])
        logger.info(f"{index} First token (temperature={temperature}): {first_token.item()} -> {repr(token_text)}")

    last_logit = logits[-1, :].clone()
    top_k = 5
    top_logits, top_indices = torch.topk(last_logit, top_k, dim=-1)
    probs = torch.softmax(last_logit, dim=-1)

    for i, (token_id, logit_value) in enumerate(zip(top_indices.tolist(), top_logits.tolist())):
        token_text = tokenizer.decode([token_id])
        prob = probs[token_id].item()

        logger.info(
            f"{i+1}. Token ID: {token_id:6d} | Logit: {logit_value:8.4f} | "
            f"Prob: {prob:8.6f} | Text: {repr(token_text)}"
        )
