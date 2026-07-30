# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused AutoFix experiments for functional-decoder precision and QKV reads.

All operation overrides in this module are pytest-local diagnostics. Nothing
in this file changes the runtime decoder policy. Hardware cases are opt-in
where they load real weights, and the device commands must be run serially.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder import (
    ARTIFACT_DIR,
    _as_tt,
    _causal_mask,
    _decode_mask,
    _load_layer_state,
    _load_text_config,
    _to_torch,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_BLOCK_SIZE,
    FULL_HEAD_DIM,
    FULL_NUM_KV_HEADS,
    HIDDEN_SIZE,
    MODEL_ID,
    NUM_Q_HEADS,
    SLIDING_BLOCK_SIZE,
    SLIDING_HEAD_DIM,
    SLIDING_NUM_KV_HEADS,
    FunctionalDecoder,
    _make_decode_height_sharded_memory_config,
)
from models.common.utility_functions import comp_pcc

SOURCE_PATH = Path("models/autoports/google_gemma_4_26b_a4b_it/tt/functional_decoder.py")
THIS_TEST_PATH = Path(__file__)
KERNEL_DRAM_FIX = "7aa26e4b1f274867bcea5ff6ea99295f961d89b1"
PCC_THRESHOLD = 0.995
QKV_PCC_THRESHOLD = 0.9999
TRACE_REPLAYS = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(name: str, payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "decoder_source_sha256": _sha256(SOURCE_PATH),
        "test_source_sha256": _sha256(THIS_TEST_PATH),
        **payload,
    }
    (ARTIFACT_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _pcc(lhs: torch.Tensor, rhs: torch.Tensor, threshold: float) -> tuple[bool, float]:
    passed, value = comp_pcc(lhs, rhs, threshold)
    return bool(passed), float(value)


def _head_split_reference(
    fused: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = fused.shape[-2]
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    q = fused[..., :q_width].reshape(1, batch, num_q_heads, head_dim)
    k = fused[..., q_width : q_width + kv_width].reshape(1, batch, num_kv_heads, head_dim)
    v = fused[..., q_width + kv_width :].reshape(1, batch, num_kv_heads, head_dim)
    return q, k, v


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "layer_type,num_kv_heads,head_dim",
    [
        pytest.param("sliding_attention", SLIDING_NUM_KV_HEADS, SLIDING_HEAD_DIM, id="sliding"),
        pytest.param("full_attention", FULL_NUM_KV_HEADS, FULL_HEAD_DIM, id="full"),
    ],
)
def test_qkv_head_split_dram_exact_shapes(
    mesh_device,
    device_params,
    layer_type,
    num_kv_heads,
    head_dim,
):
    """Prove the fixed BH reader on Gemma4's two exact fused-QKV widths."""
    batch = 32
    qkv_width = (NUM_Q_HEADS + 2 * num_kv_heads) * head_dim
    torch.manual_seed(17000 + head_dim)
    fused = torch.randn(1, 1, batch, qkv_width, dtype=torch.bfloat16)
    reference = _head_split_reference(
        fused,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    dram_input = _as_tt(mesh_device, fused)
    l1_input = ttnn.to_memory_config(dram_input, ttnn.L1_MEMORY_CONFIG)
    output_mem_config = _make_decode_height_sharded_memory_config(mesh_device, batch, head_dim)

    results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    per_memory_pcc: dict[str, dict[str, float]] = {}
    failures = []
    for memory_name, input_tensor in (("dram", dram_input), ("l1", l1_input)):
        loop_pcc = []
        final_outputs = None
        for _ in range(3):
            outputs = ttnn.experimental.nlp_create_qkv_heads_decode(
                input_tensor,
                num_heads=NUM_Q_HEADS,
                num_kv_heads=num_kv_heads,
                memory_config=output_mem_config,
            )
            ttnn.synchronize_device(mesh_device)
            host_outputs = tuple(_to_torch(mesh_device, output).to(torch.bfloat16) for output in outputs)
            values = []
            for expected, actual in zip(reference, host_outputs):
                passed, pcc = _pcc(expected, actual, QKV_PCC_THRESHOLD)
                if not passed:
                    failures.append(f"{memory_name} vs Torch PCC {pcc}")
                values.append(pcc)
            loop_pcc.append(values)
            final_outputs = host_outputs
            for output in outputs:
                output.deallocate(True)
        assert final_outputs is not None
        results[memory_name] = final_outputs
        per_memory_pcc[memory_name] = {
            head: min(run[index] for run in loop_pcc) for index, head in enumerate(("q", "k", "v"))
        }

    cross_pcc = {}
    for head, dram_output, l1_output in zip(("q", "k", "v"), results["dram"], results["l1"]):
        passed, pcc = _pcc(dram_output, l1_output, QKV_PCC_THRESHOLD)
        if not passed:
            failures.append(f"{head} DRAM vs L1 PCC {pcc}")
        cross_pcc[head] = pcc

    _write_artifact(
        f"qkv_head_split_dram_{'sliding' if layer_type == 'sliding_attention' else 'full'}.json",
        {
            "kernel_dram_fix_commit": KERNEL_DRAM_FIX,
            "kernel_dram_fix_verified_ancestor_by_source_review": True,
            "layer_type": layer_type,
            "batch": batch,
            "dtype": "bfloat16",
            "input_shape": list(fused.shape),
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "program_cache_iterations": 3,
            "torch_pcc_threshold": QKV_PCC_THRESHOLD,
            "minimum_torch_pcc": per_memory_pcc,
            "dram_vs_l1_pcc_threshold": QKV_PCC_THRESHOLD,
            "dram_vs_l1_pcc": cross_pcc,
            "dram_input_memory_config": str(dram_input.memory_config()),
            "l1_input_memory_config": str(l1_input.memory_config()),
            "output_memory_config": str(output_mem_config),
            "failures": failures,
        },
    )
    assert not failures, failures


def _build_decode_host_case(
    cfg,
    state: dict[str, torch.Tensor],
    *,
    layer_idx: int,
    batch: int,
) -> dict[str, Any]:
    layer_type = cfg.layer_types[layer_idx]
    current_position = 32
    if layer_type == "full_attention":
        torch.manual_seed(layer_idx)
        prefix_one = torch.randn(1, current_position, HIDDEN_SIZE, dtype=torch.bfloat16)
        decode_one = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
        prefix_hidden = prefix_one.expand(batch, -1, -1).clone()
        hidden = decode_one.expand(batch, -1, -1).clone()
        num_kv_heads, block_size, head_dim = FULL_NUM_KV_HEADS, FULL_BLOCK_SIZE, FULL_HEAD_DIM
    else:
        torch.manual_seed(2000 + layer_idx + batch)
        prefix_hidden = None
        hidden = torch.randn(batch, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
        num_kv_heads, block_size, head_dim = (
            SLIDING_NUM_KV_HEADS,
            SLIDING_BLOCK_SIZE,
            SLIDING_HEAD_DIM,
        )

    positions = torch.full((batch, 1), current_position, dtype=torch.long)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions, layer_type=layer_type)
    prefix_positions = None
    prefix_cos = None
    prefix_sin = None

    prefix = f"model.language_model.layers.{layer_idx}"
    reference_layer = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    reference_layer.load_state_dict({key[len(prefix) + 1 :]: value for key, value in state.items()}, strict=True)
    reference_cache = DynamicCache(config=cfg)
    if layer_type == "full_attention":
        prefix_positions = torch.arange(current_position).view(1, -1).expand(batch, -1)
        prefix_cos, prefix_sin = rotary(prefix_hidden, prefix_positions, layer_type=layer_type)
        with torch.no_grad():
            reference_layer(
                prefix_hidden,
                shared_kv_states={},
                position_embeddings=(prefix_cos, prefix_sin),
                attention_mask=_causal_mask(current_position, sliding_window=None).expand(batch, -1, -1, -1),
                position_ids=prefix_positions,
                past_key_values=reference_cache,
            )
    else:
        reference_cache.update(
            torch.zeros(batch, num_kv_heads, current_position, head_dim, dtype=torch.bfloat16),
            torch.zeros(batch, num_kv_heads, current_position, head_dim, dtype=torch.bfloat16),
            layer_idx=layer_idx,
        )
    with torch.no_grad():
        reference = reference_layer(
            hidden,
            shared_kv_states={},
            position_embeddings=(cos, sin),
            attention_mask=_decode_mask(
                current_position + 1,
                sliding_window=cfg.sliding_window if layer_type == "sliding_attention" else None,
            ).expand(batch, -1, -1, -1),
            position_ids=positions,
            past_key_values=reference_cache,
        )

    return {
        "layer_type": layer_type,
        "current_position": current_position,
        "hidden": hidden,
        "cos": cos,
        "sin": sin,
        "prefix_hidden": prefix_hidden,
        "prefix_cos": prefix_cos,
        "prefix_sin": prefix_sin,
        "reference": reference,
        "num_kv_heads": num_kv_heads,
        "block_size": block_size,
        "head_dim": head_dim,
    }


def _make_decode_device_args(mesh_device, decoder: FunctionalDecoder, case: dict[str, Any]) -> dict[str, Any]:
    batch = case["hidden"].shape[0]
    blocks_per_user = 2 if case["layer_type"] == "full_attention" else 4
    page_table = _as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (
        batch * blocks_per_user,
        case["num_kv_heads"],
        case["block_size"],
        case["head_dim"],
    )
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    if case["layer_type"] == "full_attention":
        for user_id in range(batch):
            decoder.prefill_forward(
                _as_tt(mesh_device, case["prefix_hidden"][user_id : user_id + 1].unsqueeze(1)),
                position_cos=_as_tt(mesh_device, case["prefix_cos"][user_id : user_id + 1].unsqueeze(1)),
                position_sin=_as_tt(mesh_device, case["prefix_sin"][user_id : user_id + 1].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
            )

    if case["layer_type"] == "sliding_attention":
        tt_cos = case["cos"].unsqueeze(0)
        tt_sin = case["sin"].unsqueeze(0)
    else:
        tt_cos = case["cos"].transpose(0, 1).unsqueeze(0)
        tt_sin = case["sin"].transpose(0, 1).unsqueeze(0)
    return {
        "hidden_states": _as_tt(mesh_device, case["hidden"].transpose(0, 1).unsqueeze(0)),
        "position_cos": _as_tt(mesh_device, tt_cos),
        "position_sin": _as_tt(mesh_device, tt_sin),
        "current_pos": _as_tt(
            mesh_device,
            torch.full((batch,), case["current_position"], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }


@contextmanager
def _bypass_fused_qkv_l1_promotion(decoder: FunctionalDecoder):
    """Bypass only the exact whole-QKV DRAM-to-interleaved-L1 promotion."""
    hits = {"count": 0}
    expected_shape = (1, 1, 32, decoder.layer_kind.qkv_width)
    original = ttnn.to_memory_config
    with pytest.MonkeyPatch.context() as patch:

        def wrapped(tensor, *args, **kwargs):
            requested = args[0] if args else kwargs.get("memory_config")
            is_exact_qkv = tuple(tensor.shape) == expected_shape
            is_dram_input = tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            if is_exact_qkv and is_dram_input and requested == ttnn.L1_MEMORY_CONFIG:
                hits["count"] += 1
                return tensor
            return original(tensor, *args, **kwargs)

        patch.setattr(ttnn, "to_memory_config", wrapped)
        yield hits


def _trace_variant(
    mesh_device,
    decoder: FunctionalDecoder,
    decode_args: dict[str, Any],
    *,
    bypass_promotion: bool,
) -> dict[str, Any]:
    patch_context = _bypass_fused_qkv_l1_promotion(decoder) if bypass_promotion else nullcontext({"count": 0})
    trace_id = None
    with patch_context as hits:
        eager_output = decoder.decode_forward(**decode_args)
        ttnn.synchronize_device(mesh_device)
        eager = _to_torch(mesh_device, eager_output).reshape(1, 32, HIDDEN_SIZE).transpose(0, 1).to(torch.bfloat16)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = decoder.decode_forward(**decode_args)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        latencies_ms = []
        replay = None
        repeat = None
        try:
            # Warm replay is deliberately excluded from latency statistics.
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            replay = (
                _to_torch(mesh_device, traced_output).reshape(1, 32, HIDDEN_SIZE).transpose(0, 1).to(torch.bfloat16)
            )
            for replay_index in range(TRACE_REPLAYS):
                start = time.perf_counter()
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                latencies_ms.append((time.perf_counter() - start) * 1000)
                if replay_index == TRACE_REPLAYS - 1:
                    repeat = (
                        _to_torch(mesh_device, traced_output)
                        .reshape(1, 32, HIDDEN_SIZE)
                        .transpose(0, 1)
                        .to(torch.bfloat16)
                    )
        finally:
            if trace_id is not None:
                ttnn.release_trace(mesh_device, trace_id)

    assert replay is not None and repeat is not None
    ordered = sorted(latencies_ms)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "eager": eager,
        "replay": replay,
        "repeat": repeat,
        "promotion_bypass_hits": hits["count"],
        # The production promotion was removed after the original A/B. Keep
        # the probe active as a regression assertion that no exact whole-QKV
        # DRAM-to-L1 copy has returned.
        "expected_promotion_bypass_hits": 0,
        "median_trace_latency_ms": statistics.median(ordered),
        "p95_trace_latency_ms": ordered[p95_index],
        "trace_replays": TRACE_REPLAYS,
    }


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "full"])
def test_qkv_dram_l1_decoder_ab(mesh_device, device_params, layer_idx):
    """Confirm the removed whole-QKV promotion stays absent and trace-safe."""
    cfg = _load_text_config()
    state = _load_layer_state(layer_idx)
    case = _build_decode_host_case(cfg, state, layer_idx=layer_idx, batch=32)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )

    variants = {}
    for name, bypass in (("production_dram_reader", False), ("removed_promotion_probe", True)):
        decode_args = _make_decode_device_args(mesh_device, decoder, case)
        variants[name] = _trace_variant(
            mesh_device,
            decoder,
            decode_args,
            bypass_promotion=bypass,
        )

    metrics = {}
    failures = []
    for name, result in variants.items():
        hf_ok, hf_pcc = _pcc(case["reference"], result["replay"], PCC_THRESHOLD)
        repeat_ok, repeat_pcc = _pcc(result["replay"], result["repeat"], QKV_PCC_THRESHOLD)
        eager_ok, eager_pcc = _pcc(result["eager"], result["replay"], QKV_PCC_THRESHOLD)
        if not hf_ok:
            failures.append(f"{name} HF PCC {hf_pcc}")
        if not repeat_ok:
            failures.append(f"{name} repeat PCC {repeat_pcc}")
        if not eager_ok:
            failures.append(f"{name} eager-vs-replay PCC {eager_pcc}")
        if result["promotion_bypass_hits"] != result["expected_promotion_bypass_hits"]:
            failures.append(
                f"{name} promotion bypass hits {result['promotion_bypass_hits']}, "
                f"expected {result['expected_promotion_bypass_hits']}"
            )
        metrics[name] = {
            "hf_pcc": hf_pcc,
            "repeat_pcc": repeat_pcc,
            "eager_vs_replay_pcc": eager_pcc,
            "promotion_bypass_hits": result["promotion_bypass_hits"],
            "expected_promotion_bypass_hits": result["expected_promotion_bypass_hits"],
            "median_trace_latency_ms": result["median_trace_latency_ms"],
            "p95_trace_latency_ms": result["p95_trace_latency_ms"],
            "trace_replays": result["trace_replays"],
        }

    cross_ok, cross_pcc = _pcc(
        variants["production_dram_reader"]["replay"],
        variants["removed_promotion_probe"]["replay"],
        QKV_PCC_THRESHOLD,
    )
    if not cross_ok:
        failures.append(f"cross-variant PCC {cross_pcc}")
    qkv_bytes = 32 * decoder.layer_kind.qkv_width * 2
    short_name = "sliding" if case["layer_type"] == "sliding_attention" else "full"
    _write_artifact(
        f"qkv_dram_l1_decoder_ab_{short_name}.json",
        {
            "kernel_dram_fix_commit": KERNEL_DRAM_FIX,
            "kernel_dram_fix_verified_ancestor_by_source_review": True,
            "layer_idx": layer_idx,
            "layer_type": case["layer_type"],
            "batch": 32,
            "current_position": case["current_position"],
            "qkv_width": decoder.layer_kind.qkv_width,
            "whole_qkv_l1_bytes_removed": qkv_bytes,
            "hf_pcc_threshold": PCC_THRESHOLD,
            "cross_variant_pcc_threshold": QKV_PCC_THRESHOLD,
            "cross_variant_pcc": cross_pcc,
            "variants": metrics,
            "failures": failures,
        },
    )
    assert not failures, failures


def _replace_compute_config(kwargs: dict[str, Any], config: Any | None) -> dict[str, Any]:
    updated = dict(kwargs)
    if config is None:
        updated.pop("compute_kernel_config", None)
    else:
        updated["compute_kernel_config"] = config
    return updated


@contextmanager
def _test_compute_policy(
    decoder: FunctionalDecoder,
    mesh_device,
    policies: dict[str, Any | None],
    snapshots: dict[str, torch.Tensor],
):
    """Inject per-group compute configs entirely at Python test boundaries."""
    hits = {group: 0 for group in ("norm", "sdpa", "dense_o", "router", "experts")}
    dense_weights = (
        decoder.weights.o_proj,
        decoder.weights.mlp_gate,
        decoder.weights.mlp_up,
        decoder.weights.mlp_down,
    )
    originals = {
        "rms_norm": ttnn.rms_norm,
        "linear": ttnn.linear,
        "sparse_matmul": ttnn.sparse_matmul,
        "softmax": ttnn.softmax,
        "prefill_sdpa": ttnn.transformer.scaled_dot_product_attention,
        "decode_sdpa": ttnn.transformer.paged_scaled_dot_product_attention_decode,
        "chunked_sdpa": ttnn.transformer.chunked_scaled_dot_product_attention,
    }

    with pytest.MonkeyPatch.context() as patch:

        def rms_norm(*args, **kwargs):
            hits["norm"] += 1
            return originals["rms_norm"](*args, **_replace_compute_config(kwargs, policies["norm"]))

        def linear(*args, **kwargs):
            weight = args[1] if len(args) > 1 else kwargs.get("weight_tensor")
            group = None
            if any(weight is candidate for candidate in dense_weights):
                group = "dense_o"
            elif weight is decoder.weights.router_proj:
                group = "router"
            if group is None:
                return originals["linear"](*args, **kwargs)
            hits[group] += 1
            return originals["linear"](*args, **_replace_compute_config(kwargs, policies[group]))

        def sparse_matmul(*args, **kwargs):
            hits["experts"] += 1
            return originals["sparse_matmul"](
                *args,
                **_replace_compute_config(kwargs, policies["experts"]),
            )

        def softmax(*args, **kwargs):
            hits["router"] += 1
            return originals["softmax"](*args, **_replace_compute_config(kwargs, policies["router"]))

        def prefill_sdpa(*args, **kwargs):
            hits["sdpa"] += 1
            return originals["prefill_sdpa"](*args, **_replace_compute_config(kwargs, policies["sdpa"]))

        def decode_sdpa(*args, **kwargs):
            hits["sdpa"] += 1
            return originals["decode_sdpa"](*args, **_replace_compute_config(kwargs, policies["sdpa"]))

        def chunked_sdpa(*args, **kwargs):
            hits["sdpa"] += 1
            return originals["chunked_sdpa"](*args, **_replace_compute_config(kwargs, policies["sdpa"]))

        patch.setattr(ttnn, "rms_norm", rms_norm)
        patch.setattr(ttnn, "linear", linear)
        patch.setattr(ttnn, "sparse_matmul", sparse_matmul)
        patch.setattr(ttnn, "softmax", softmax)
        patch.setattr(ttnn.transformer, "scaled_dot_product_attention", prefill_sdpa)
        patch.setattr(ttnn.transformer, "paged_scaled_dot_product_attention_decode", decode_sdpa)
        patch.setattr(ttnn.transformer, "chunked_scaled_dot_product_attention", chunked_sdpa)

        for method_name in ("_attention_decode", "_dense_mlp", "_router_weights", "_moe_decode"):
            original_method = getattr(decoder, method_name)

            def capture(*args, _name=method_name, _original=original_method, **kwargs):
                output = _original(*args, **kwargs)
                snapshots[_name] = _to_torch(mesh_device, output).float()
                return output

            patch.setattr(decoder, method_name, capture)
        yield hits


def _precision_policy_ledger(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {"kind": "framework_default", "compute_kernel_config_present": False}
    return {
        "kind": "explicit",
        "compute_kernel_config_present": True,
        "math_fidelity": str(config.math_fidelity),
        "math_approx_mode": bool(config.math_approx_mode),
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "packer_l1_acc": bool(config.packer_l1_acc),
    }


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_precision_policy_isolation(mesh_device, device_params):
    """Localize the marginal real-weight layer-0 decode precision requirement."""
    if os.getenv("GEMMA4_PRECISION_ISOLATION") != "1":
        pytest.skip("set GEMMA4_PRECISION_ISOLATION=1 for the real-weight adaptive precision A/B")

    cfg = _load_text_config()
    layer_idx = 0
    state = _load_layer_state(layer_idx)
    case = _build_decode_host_case(cfg, state, layer_idx=layer_idx, batch=1)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    high = decoder.correctness_compute_config
    default = None
    groups = ("norm", "sdpa", "dense_o", "router", "experts")
    matrix = {
        "D_default": {group: default for group in groups},
        "H_blanket": {group: high for group in groups},
        "A_norm_sdpa": {
            "norm": high,
            "sdpa": high,
            "dense_o": default,
            "router": default,
            "experts": default,
        },
        "B_post_attention": {
            "norm": default,
            "sdpa": default,
            "dense_o": high,
            "router": high,
            "experts": high,
        },
    }
    for selected in groups:
        matrix[f"leaf_{selected}"] = {group: high if group == selected else default for group in groups}

    outputs: dict[str, torch.Tensor] = {}
    snapshots_by_policy: dict[str, dict[str, torch.Tensor]] = {}
    records = {}
    failures = {}
    for policy_name, policies in matrix.items():
        snapshots: dict[str, torch.Tensor] = {}
        try:
            decode_args = _make_decode_device_args(mesh_device, decoder, case)
            with _test_compute_policy(decoder, mesh_device, policies, snapshots) as hits:
                output = decoder.decode_forward(**decode_args)
                ttnn.synchronize_device(mesh_device)
                host_output = _to_torch(mesh_device, output).reshape(1, 1, HIDDEN_SIZE).to(torch.bfloat16)
            for group in groups:
                assert hits[group] > 0, (policy_name, group, hits)
            passed, pcc = _pcc(case["reference"], host_output, PCC_THRESHOLD)
            outputs[policy_name] = host_output
            snapshots_by_policy[policy_name] = snapshots
            records[policy_name] = {
                "hf_pcc": pcc,
                "passes_0_995": passed,
                "hit_counts": hits,
                "policies": {group: _precision_policy_ledger(policies[group]) for group in groups},
            }
        except Exception as error:
            failures[policy_name] = f"{type(error).__name__}: {error}"

    if "D_default" in outputs:
        baseline = outputs["D_default"]
        baseline_snapshots = snapshots_by_policy["D_default"]
        for policy_name, output in outputs.items():
            _, final_pcc = _pcc(baseline, output, 0.0)
            component_pcc = {}
            for component, baseline_component in baseline_snapshots.items():
                if component in snapshots_by_policy[policy_name]:
                    _, value = _pcc(
                        baseline_component,
                        snapshots_by_policy[policy_name][component],
                        0.0,
                    )
                    component_pcc[component] = value
            records[policy_name]["vs_default_final_pcc"] = final_pcc
            records[policy_name]["vs_default_component_pcc"] = component_pcc

    _write_artifact(
        "precision_policy_isolation.json",
        {
            "layer_idx": layer_idx,
            "layer_type": case["layer_type"],
            "batch": 1,
            "current_position": case["current_position"],
            "real_weights": True,
            "hf_pcc_threshold": PCC_THRESHOLD,
            "material_pcc_delta": 0.0002,
            "records": records,
            "failures": failures,
            "diagnostic_host_component_reads": True,
            "eligible_for_runtime_fallback_or_latency_evidence": False,
        },
    )
    assert not failures, failures
    assert records["H_blanket"]["passes_0_995"], records["H_blanket"]["hf_pcc"]
