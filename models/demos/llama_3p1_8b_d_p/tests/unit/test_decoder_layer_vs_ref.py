# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent raw-HF oracle and all-chip gates for one chunked prefill decoder."""

import math
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoConfig, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.demos.llama_3p1_8b_d_p.tests.device_utils import addresses as _addresses
from models.demos.llama_3p1_8b_d_p.tests.device_utils import assert_unchanged as _assert_unchanged
from models.demos.llama_3p1_8b_d_p.tests.device_utils import snapshot as _snapshot
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics as _metrics
from models.demos.llama_3p1_8b_d_p.tests.utils import read_raw_weights
from models.demos.llama_3p1_8b_d_p.tt.attention import FullCausalAttention
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.decoder import DecoderLayer
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache
from models.demos.llama_3p1_8b_d_p.tt.rope import build_indexed_rope, build_transformation_mat

HF_MODEL = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))
WEIGHT_NAMES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _load_weights(layer_idx):
    names = {name: f"model.layers.{layer_idx}.{name}" for name in WEIGHT_NAMES}
    return {name: value.bfloat16() for name, value in read_raw_weights(HF_MODEL, names).items()}


def _stream(slot, length=3072):
    generator = torch.Generator().manual_seed(17007 + slot * 971)
    return (torch.randn(length, 4096, generator=generator) * 0.2).bfloat16()


def _owned_positions(start):
    # Independent encounter-order packing: absolute 256-token blocks rotate over the four SP rows.
    return [[p for p in range(start, start + 1024) if (p // 256) % 4 == sp] for sp in range(4)]


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


@torch.no_grad()
def _reference_prefix(weights, values, config):
    # All mathematics is float32, from raw BF16 checkpoint weights and BF16-rounded inputs. No
    # production norm, positions, Q/K permutation, attention, cache quantization or MLP is reused.
    w = {name: tensor.bfloat16().float() for name, tensor in weights.items()}
    x = values[:2048].bfloat16().float()
    normalized = _rms(x, w["input_layernorm.weight"], config.rms_norm_eps)
    q = F.linear(normalized, w["self_attn.q_proj.weight"]).reshape(-1, 32, 128).transpose(0, 1)
    k = F.linear(normalized, w["self_attn.k_proj.weight"]).reshape(-1, 8, 128).transpose(0, 1)
    v = F.linear(normalized, w["self_attn.v_proj.weight"]).reshape(-1, 8, 128).transpose(0, 1)
    rotary = LlamaRotaryEmbedding(config=config)
    cos, sin = rotary(x.unsqueeze(0), torch.arange(len(x)).unsqueeze(0))
    cos, sin = cos[0].unsqueeze(0), sin[0].unsqueeze(0)
    q = q * cos + torch.cat((-q[..., 64:], q[..., :64]), dim=-1) * sin
    k = k * cos + torch.cat((-k[..., 64:], k[..., :64]), dim=-1) * sin
    # Only final reference K is converted for comparison with the adjacent-pair cache contract.
    adjacent = torch.tensor([coordinate for pair in zip(range(64), range(64, 128)) for coordinate in pair])
    result = {"x": x, "q": q, "k": k, "v": v, "cache_k": k[..., adjacent], "weights": w}
    for name in ("x", "q", "k", "v", "cache_k"):
        assert torch.isfinite(result[name]).all(), name
    return result


@torch.no_grad()
def _reference_chunk(reference, start, end, eps):
    w = reference["weights"]
    positions = torch.arange(start, end)
    keys = torch.arange(end)
    heads = []
    for head in range(32):
        scores = reference["q"][head, start:end] @ reference["k"][head // 4, :end].T / math.sqrt(128)
        probabilities = scores.masked_fill(keys[None, :] > positions[:, None], -torch.inf).softmax(dim=-1)
        heads.append(probabilities @ reference["v"][head // 4, :end])
    attended = torch.stack(heads, dim=1).reshape(end - start, 4096)
    attention_delta = F.linear(attended, w["self_attn.o_proj.weight"])
    residual = reference["x"][start:end] + attention_delta
    normalized = _rms(residual, w["post_attention_layernorm.weight"], eps)
    mlp_delta = F.linear(
        F.silu(F.linear(normalized, w["mlp.gate_proj.weight"])) * F.linear(normalized, w["mlp.up_proj.weight"]),
        w["mlp.down_proj.weight"],
    )
    output = residual + mlp_delta
    assert torch.isfinite(output).all()
    return output


def _check_metrics(expected, actual, limits, label):
    pcc, nl2 = _metrics(expected, actual)
    logger.info(f"{label}: PCC={pcc:.8f}, NL2={nl2:.8f}")
    assert pcc >= limits[0] and nl2 <= limits[1], f"{label}: PCC={pcc}, NL2={nl2}, limits={limits}"
    return pcc, nl2


def _upload(mesh_device, values, start, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    indices = [p for shard in _owned_positions(start) for p in shard]
    return ttnn.from_torch(
        values[indices].reshape(1, 1, 1024, -1),
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(4, 8), dims=(2, None)),
    )


def _check_cache(cache, before, reference, slot, layer, start, end, dtype):
    plane = slot * 32 + layer
    limits = (0.9999, 0.01) if dtype == ttnn.bfloat16 else (0.999, 0.02)
    results = []
    for name, tensor, snapshots in zip(("cache_k", "v"), (cache.k, cache.v), before):
        for chip, (shard, snapshot) in enumerate(zip(ttnn.get_device_tensors(tensor), snapshots)):
            actual = ttnn.to_torch(shard)
            assert torch.isfinite(actual).all()
            sp, head = divmod(chip, 8)
            global_positions = torch.tensor([p for p in range(2048) if (p // 256) % 4 == sp])
            written = (global_positions >= start) & (global_positions < end)
            padding = (global_positions >= end) & (global_positions < (end + 31) // 32 * 32)
            preserve = ~(written | padding)
            assert torch.equal(actual[:plane], snapshot[:plane]), f"{name}, chip={chip}: preceding planes"
            assert torch.equal(actual[plane + 1 :], snapshot[plane + 1 :]), f"{name}, chip={chip}: following planes"
            assert torch.equal(actual[plane, 0, preserve], snapshot[plane, 0, preserve])
            assert torch.count_nonzero(actual[plane, 0, padding]) == 0
            if written.any():
                expected = reference[name][head, global_positions[written]]
                got = actual[plane, 0, written]
                if torch.count_nonzero(expected) == 0:
                    assert torch.count_nonzero(got) == 0
                else:
                    results.append(_check_metrics(expected, got, limits, f"KV {name} s={slot} l={layer} chip={chip}"))
    if results:
        logger.info(f"KV worst chip/head: min_PCC={min(p for p, _ in results)}, max_NL2={max(n for _, n in results)}")


def _resource_identity(attention):
    # These are the accepted gather/mask candidate's resources, shared sequentially across layers.
    return tuple(
        (id(tensor), _addresses(tensor))
        for tensor in (
            attention.gathered_k,
            attention.gathered_v,
            attention.query_position_table,
            attention.key_positions,
        )
    )


def _run(mesh_device, layer, cache, values, reference, config, *, slot, start, end, branch=None):
    x = _upload(mesh_device, values, start)
    before = _snapshot(cache)
    expected = _reference_chunk(reference, start, end, config.rms_norm_eps)
    # Sequential RMSNorm needs 1,146,880 B/core; the selected attention configuration may
    # require more (the stock K512 candidate has a 1,273,856 B conservative bound).
    required_l1 = max(1_146_880, layer.attention._SDPA_L1_BYTES)
    l1 = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    assert l1.total_bytes_free_per_bank >= required_l1
    assert l1.largest_contiguous_bytes_free_per_bank >= required_l1
    logger.info(
        f"decoder preflight: required_L1={required_l1}; free_L1={l1.total_bytes_free_per_bank}; "
        f"largest_L1={l1.largest_contiguous_bytes_free_per_bank}"
    )
    ttnn.synchronize_device(mesh_device)
    started = time.perf_counter()
    output = layer(x, cache, slot_idx=slot, actual_start=start, actual_end=end)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - started
    logger.info(
        f"layer={layer.layer_idx} slot={slot} range=[{start},{end}) synchronized eager wall time={elapsed:.6f}s"
    )
    assert tuple(output.shape) == (1, 1, 256, 4096)
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert all(a != b for a, b in zip(_addresses(x), _addresses(output)))
    output_limits = (0.999, 0.025) if cache.k.dtype == ttnn.bfloat16 else (0.999, 0.05)
    delta_limits = (0.995, 0.05) if cache.k.dtype == ttnn.bfloat16 else (0.99, 0.07)
    metrics = []
    for chip, (in_shard, out_shard) in enumerate(zip(ttnn.get_device_tensors(x), ttnn.get_device_tensors(output))):
        positions = torch.tensor(_owned_positions(start)[chip // 8])
        host_input = values[positions]
        assert torch.equal(ttnn.to_torch(in_shard)[0, 0], host_input)
        valid = positions < end
        if not valid.any():
            continue
        actual = ttnn.to_torch(out_shard)[0, 0, valid].float()
        wanted = expected[positions[valid] - start]
        assert torch.isfinite(actual).all() and torch.isfinite(wanted).all()
        if branch == "identity":
            assert torch.equal(actual, host_input[valid].float())
        else:
            metrics.append(_check_metrics(wanted, actual, output_limits, f"output chip={chip}"))
            if branch is not None:
                # Effective delta includes the final BF16 residual rounding, which matters when
                # subtracting a much larger residual. A vanished branch is never a passing fixture.
                delta = wanted.bfloat16().float() - host_input[valid].float()
                assert torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(host_input[valid].float()) > 0.01
                _check_metrics(delta, actual - host_input[valid].float(), delta_limits, f"{branch} delta chip={chip}")
    if metrics:
        logger.info(f"output worst chip: min_PCC={min(p for p, _ in metrics)}, max_NL2={max(n for _, n in metrics)}")
    _check_cache(cache, before, reference, slot, layer.layer_idx, start, end, cache.k.dtype)
    return x, output


def _make_resources(mesh_device, dtype):
    config = MeshConfig((4, 8), 8)
    attention = FullCausalAttention(mesh_device, config, cache_dtype=dtype)
    rope = build_indexed_rope(mesh_device, max_seq_len=2048, chunk_size=1024)
    transform = build_transformation_mat(mesh_device)
    cache = allocate_kv_cache(mesh_device, config, cache_dtype=dtype)
    # Nonzero untouched planes and suffixes expose unintended clearing as well as unintended writes.
    for name, sign in (("k", 1), ("v", -1)):
        previous = getattr(cache, name)
        sentinel = (torch.arange(64).reshape(64, 1, 1, 1) % 7 + 1).expand(64, 1, 512, 128) * (sign / 8)
        setattr(
            cache,
            name,
            ttnn.from_torch(
                sentinel.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=previous.memory_config(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        )
        previous.deallocate(True)
    return config, attention, rope, transform, cache


def _free_layer(layer):
    for tensor in (
        layer.input_norm.weight,
        layer.post_attention_norm.weight,
        layer.qkv.qkv_weight,
        layer.output_projection.o_weight,
        layer.mlp.gate_weight,
        layer.mlp.up_weight,
        layer.mlp.down_weight,
    ):
        tensor.deallocate(True)


def _free_resources(attention, rope, transform, cache):
    for tensor in (
        attention.gathered_k,
        attention.gathered_v,
        attention.query_position_table,
        attention.key_positions,
        *rope,
        transform,
        cache.k,
        cache.v,
    ):
        tensor.deallocate(True)


# A single real layer with a full first chunk provides the ordered smoke gate after residual-add
# acceptance, and a bounded, all-chip covering selector for the separate Watcher run.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_decoder_one_real_layer_smoke(mesh_device):
    mesh_config, attention, rope, transform, cache = _make_resources(mesh_device, ttnn.bfloat16)
    weights = _load_weights(0)
    config, values = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True), _stream(0)
    reference = _reference_prefix(weights, values, config)
    layer = DecoderLayer(
        mesh_device,
        mesh_config,
        weights,
        layer_idx=0,
        attention=attention,
        rope_tables=rope,
        transformation_mat=transform,
    )
    try:
        for tensor in _run(mesh_device, layer, cache, values, reference, config, slot=0, start=0, end=1024):
            tensor.deallocate(True)
    finally:
        _free_layer(layer)
        _free_resources(attention, rope, transform, cache)


# Real layers 0/13 share attention while interleaving distinct slots and complete prefixes. All
# valid rows, TP replicas and KV shards are checked, including overlap, partial tiles and terminal tail.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
def test_decoder_real_layers_interleaved(mesh_device, cache_dtype):
    mesh_config, attention, rope, transform, cache = _make_resources(mesh_device, cache_dtype)
    config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    layers, references = {}, {}
    streams = {slot: _stream(slot) for slot in (0, 1)}
    for index in (0, 13):
        weights = _load_weights(index)
        layers[index] = DecoderLayer(
            mesh_device,
            mesh_config,
            weights,
            layer_idx=index,
            attention=attention,
            rope_tables=rope,
            transformation_mat=transform,
        )
        for slot in (0, 1):
            references[index, slot] = _reference_prefix(weights, streams[slot], config)
    identity = _resource_identity(attention)
    rotary_before = [[ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(t)] for t in (*rope, transform)]
    assert layers[0].attention is layers[13].attention is attention
    mesh_device.enable_program_cache()
    guards = []
    first_addresses = None
    # Each plane is filled from zero before a continuation. Short scenarios are new requests on
    # their own planes, while long scenarios explicitly fill 1536..2016 before the terminal call.
    schedule = [
        (0, 0, 0, 1024),
        (13, 1, 0, 33),
        (13, 0, 0, 1024),
        (0, 1, 0, 33),
        (0, 0, 1024, 1537),
        (13, 1, 32, 65),
        (13, 0, 1024, 1537),
        (0, 1, 32, 65),
        (0, 0, 1536, 2016),
        (13, 0, 1536, 2016),
        (0, 0, 2016, 2048),
        (13, 0, 2016, 2048),
    ]
    try:
        for step, (index, slot, start, end) in enumerate(schedule):
            tensors = _run(
                mesh_device,
                layers[index],
                cache,
                streams[slot],
                references[index, slot],
                config,
                slot=slot,
                start=start,
                end=end,
            )
            addresses = tuple(_addresses(tensor) for tensor in tensors)
            if step == 0:
                first_addresses = addresses
                guards.extend(tensors)
            else:
                for chip in range(32):
                    assert {a[chip] for a in addresses}.isdisjoint(a[chip] for a in first_addresses)
                for tensor in tensors:
                    tensor.deallocate(True)
            assert _resource_identity(attention) == identity
        # Warm every identical shape/start/end combination before checking program count and live
        # DRAM. Persistent JIT/fabric allocations are included in the warmed baseline, not leaks.
        replay = ((13, 1, 32, 65), (0, 1, 32, 65))
        for index, slot, start, end in replay:
            for tensor in _run(
                mesh_device,
                layers[index],
                cache,
                streams[slot],
                references[index, slot],
                config,
                slot=slot,
                start=start,
                end=end,
            ):
                tensor.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        programs = mesh_device.num_program_cache_entries()
        free_dram = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_free_per_bank
        for repeat in range(3):
            for index, slot, start, end in replay:
                for tensor in _run(
                    mesh_device,
                    layers[index],
                    cache,
                    streams[slot],
                    references[index, slot],
                    config,
                    slot=slot,
                    start=start,
                    end=end,
                ):
                    tensor.deallocate(True)
                ttnn.synchronize_device(mesh_device)
                assert mesh_device.num_program_cache_entries() == programs
                assert ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_free_per_bank == free_dram
                assert _resource_identity(attention) == identity
            logger.info(f"warmed replay={repeat}: programs={programs}; free_DRAM_per_bank={free_dram}")
        for tensor, snapshots in zip((*rope, transform), rotary_before):
            for shard, snapshot in zip(ttnn.get_device_tensors(tensor), snapshots):
                assert torch.equal(ttnn.to_torch(shard), snapshot)
    finally:
        for tensor in guards:
            tensor.deallocate(True)
        for layer in layers.values():
            _free_layer(layer)
        _free_resources(attention, rope, transform, cache)


# Invalid inputs, metadata and cache geometry must fail before either K or V changes; a subsequent
# valid layer call proves rejection did not poison shared attention state or cache ownership.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_decoder_rejects_before_mutating_cache(mesh_device, expect_error):
    mesh_config, attention, rope, transform, cache = _make_resources(mesh_device, ttnn.bfloat8_b)
    weights = _load_weights(0)
    config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    layer = DecoderLayer(
        mesh_device,
        mesh_config,
        weights,
        layer_idx=0,
        attention=attention,
        rope_tables=rope,
        transformation_mat=transform,
    )
    values = _stream(0)
    x = _upload(mesh_device, values, 0)
    before = _snapshot(cache)
    valid = {"slot_idx": 0, "actual_start": 0, "actual_end": 33}
    try:
        for changes, error, message in (
            ({"slot_idx": -1}, ValueError, "slot_idx"),
            ({"slot_idx": 2}, ValueError, "slot_idx"),
            ({"slot_idx": True}, TypeError, "eager Python int"),
            ({"actual_start": -32}, ValueError, "tile-aligned"),
            ({"actual_start": 1}, ValueError, "tile-aligned"),
            ({"actual_start": 0.0}, TypeError, "eager Python int"),
            ({"actual_end": 0}, ValueError, "start < end"),
            ({"actual_start": 64}, ValueError, "start < end"),
            ({"actual_end": 2049}, ValueError, "start < end"),
            ({"actual_end": 1025}, ValueError, "at most"),
            ({"actual_end": None}, TypeError, "eager Python int"),
        ):
            with expect_error(error, message):
                layer(x, cache, **(valid | changes))
            _assert_unchanged(cache, before)
        for kwargs, message in (
            ({"dtype": ttnn.float32}, "(?i)bfloat16"),
            ({"layout": ttnn.ROW_MAJOR_LAYOUT}, "TILE_LAYOUT"),
        ):
            bad = _upload(mesh_device, values, 0, **kwargs)
            with expect_error(ValueError, message):
                layer(bad, cache, **valid)
            bad.deallocate(True)
            _assert_unchanged(cache, before)
        bad = _upload(mesh_device, values[:, :2048], 0)
        with expect_error(ValueError, "local shape"):
            layer(bad, cache, **valid)
        bad.deallocate(True)
        _assert_unchanged(cache, before)
        bad = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        with expect_error(ValueError, "interleaved DRAM"):
            layer(bad, cache, **valid)
        bad.deallocate(True)
        _assert_unchanged(cache, before)
        for malformed in (replace(cache, num_users=1), replace(cache, max_seq_len=1024), replace(cache, k=x)):
            malformed_before = _snapshot(malformed)
            with expect_error(ValueError, "cache"):
                layer(x, malformed, **valid)
            _assert_unchanged(malformed, malformed_before)
            _assert_unchanged(cache, before)
        mismatch = ttnn.from_torch(
            torch.zeros(64, 1, 512, 128),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=cache.v.memory_config(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        malformed = replace(cache, v=mismatch)
        malformed_before = _snapshot(malformed)
        with expect_error(ValueError, "match constructor"):
            layer(x, malformed, **valid)
        _assert_unchanged(malformed, malformed_before)
        mismatch.deallocate(True)
        _assert_unchanged(cache, before)
        # BF8 requires tiled storage, so the ROW_MAJOR fixture uses supported BF16 and
        # relies on layout rejection before the attention constructor-dtype check. The tiled
        # fixture matches cache dtype so interleaved memory is its only invalid property.
        for layout, dtype, message in (
            (ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, "TILE_LAYOUT"),
            (ttnn.TILE_LAYOUT, cache.v.dtype, "NdShard DRAM"),
        ):
            mismatch = ttnn.from_torch(
                torch.ones(64, 1, 512, 128),
                device=mesh_device,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            malformed = replace(cache, v=mismatch)
            malformed_before = _snapshot(malformed)
            with expect_error(ValueError, message):
                layer(x, malformed, **valid)
            _assert_unchanged(malformed, malformed_before)
            _assert_unchanged(cache, before)
            mismatch.deallocate(True)
        for index in (-1, 32, True):
            with expect_error((ValueError, TypeError), "layer_idx"):
                DecoderLayer(
                    mesh_device,
                    mesh_config,
                    weights,
                    layer_idx=index,
                    attention=attention,
                    rope_tables=rope,
                    transformation_mat=transform,
                )
        reference = _reference_prefix(weights, values, config)
        for tensor in _run(mesh_device, layer, cache, values, reference, config, slot=0, start=0, end=33):
            tensor.deallocate(True)
    finally:
        x.deallocate(True)
        _free_layer(layer)
        _free_resources(attention, rope, transform, cache)


def _ablation_weights(branch):
    # Sparse, signed fixtures give well-conditioned, measurable deltas after BF16 rounding. Every
    # matrix retains the real Llama shape; no model dimension, production flag or operation changes.
    weights = {
        "input_layernorm.weight": torch.ones(4096, dtype=torch.bfloat16),
        "post_attention_layernorm.weight": torch.ones(4096, dtype=torch.bfloat16),
    }
    for name, rows, cols in (
        ("self_attn.q_proj.weight", 4096, 4096),
        ("self_attn.k_proj.weight", 1024, 4096),
        ("self_attn.v_proj.weight", 1024, 4096),
        ("self_attn.o_proj.weight", 4096, 4096),
        ("mlp.gate_proj.weight", 14336, 4096),
        ("mlp.up_proj.weight", 14336, 4096),
        ("mlp.down_proj.weight", 4096, 14336),
    ):
        weights[name] = torch.zeros(rows, cols, dtype=torch.bfloat16)
    indices = torch.arange(4096)
    weights["self_attn.v_proj.weight"][torch.arange(1024), torch.arange(1024) * 3] = 0.25
    # Zero Q/K means a true full-causal prefix average; the output still requires all TP partitions.
    if branch == "attention":
        weights["self_attn.o_proj.weight"][indices, indices] = 0.5
    hidden = torch.arange(14336)
    weights["mlp.gate_proj.weight"][hidden, hidden % 4096] = 0.5
    weights["mlp.up_proj.weight"][hidden, (hidden * 13 + 7) % 4096] = 0.5
    if branch == "mlp":
        weights["mlp.down_proj.weight"][indices, indices] = 0.25
    return weights


def _fixture_config():
    return LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        max_position_embeddings=131072,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )


# Host sensitivity is recorded before device acceptance: both live branches must survive BF16
# residual rounding, and the both-zero fixture must be exactly identity under the raw-HF oracle.
def test_decoder_oracle_ablation_sensitivity():
    config = _fixture_config()
    values = _stream(0, length=4)
    for branch in ("attention", "mlp", "identity"):
        reference = _reference_prefix(_ablation_weights(branch), values, config)
        result = _reference_chunk(reference, 0, 4, config.rms_norm_eps)
        delta = result.bfloat16().float() - values.float()
        if branch == "identity":
            assert torch.equal(result, values.float())
        else:
            ratio = (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(values.float())).item()
            assert math.isfinite(ratio) and ratio > 0.01
            assert delta.std() > 0
            logger.info(f"host ablation={branch}: rounded delta/input norm={ratio:.8f}")


# Real device ablations isolate each branch with weight fixtures, checking complete outputs and
# rounded effective deltas. Both zero projections must preserve input exactly on all 32 chips.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
def test_decoder_branch_ablations(mesh_device, cache_dtype):
    mesh_config, attention, rope, transform, cache = _make_resources(mesh_device, cache_dtype)
    config, values = _fixture_config(), _stream(1)
    identity = _resource_identity(attention)
    try:
        for branch in ("attention", "mlp", "identity"):
            weights = _ablation_weights(branch)
            layer = DecoderLayer(
                mesh_device,
                mesh_config,
                weights,
                layer_idx=13,
                attention=attention,
                rope_tables=rope,
                transformation_mat=transform,
            )
            reference = _reference_prefix(weights, values, config)
            try:
                for tensor in _run(
                    mesh_device, layer, cache, values, reference, config, slot=1, start=0, end=1024, branch=branch
                ):
                    tensor.deallocate(True)
                assert _resource_identity(attention) == identity
            finally:
                _free_layer(layer)
    finally:
        _free_resources(attention, rope, transform, cache)
