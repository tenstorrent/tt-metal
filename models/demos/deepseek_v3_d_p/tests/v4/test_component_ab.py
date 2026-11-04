# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One-component A/B measurements of the DSV4 precision changes, on real V4-Flash weights and inputs.

Each test isolates one component of one layer and runs the device op in the OLD and NEW configuration
against the same golden, so the number it reports is that change's own effect, not the whole model's.

Inputs are exact, not synthetic: ``capture_layer.py`` teacher-forces layer L with the vLLM trace's layer
input (chunk 0, 5120 tokens) through the host reference block on the real checkpoint weights and saves the
attention input and the FFN input. Set ``V4_AB_CAPTURE_DIR`` to its output directory and ``V4_AB_LAYER`` to L.

These tests MEASURE: they log an ``AB`` line per variant and only assert that outputs are finite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_CLAMPED_SILU_GLU, TorchExpert
from models.demos.deepseek_v3_d_p.tests.v4 import golden
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.v4.weights import load_checkpoint_tensors
from tests.ttnn.utils_for_testing import comp_pcc
from tt_py_test_utils_common.utility_functions import is_blackhole

CHUNK = 5120
LAYER = int(os.getenv("V4_AB_LAYER", "3"))
CAPTURE_DIR = Path(os.getenv("V4_AB_CAPTURE_DIR", "generated/v4_ab_capture"))


def _capture(layer: int = LAYER) -> dict:
    path = CAPTURE_DIR / f"layer{layer}.pt"
    if not path.is_file():
        pytest.skip(f"no capture at {path}; run capture_layer.py {layer} {CAPTURE_DIR}")
    return torch.load(path)


def _rel_rmse(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double(), got.double()
    return float((ref - got).pow(2).mean().sqrt() / (ref.pow(2).mean().sqrt() + 1e-30))


# --------------------------------------------------------------------------------------------------
# Tile-transposed routed-expert weights: one real expert, the tokens the trace routed to it.
# --------------------------------------------------------------------------------------------------
pytestmark_expert = pytest.mark.use_module_device


@pytest.mark.use_module_device
@pytest.mark.skipif(not is_blackhole(), reason="clamped SiLU-GLU routed expert is Blackhole-only")
@pytest.mark.parametrize("expert_rank", [0, 1], ids=["busiest", "second"])
@pytest.mark.parametrize(
    "weights_dtype, tile_transposed",
    [
        pytest.param(ttnn.bfloat4_b, False, id="bf4-plain"),
        pytest.param(ttnn.bfloat4_b, True, id="bf4-tileT"),
        pytest.param(ttnn.bfloat8_b, False, id="bf8-plain"),
    ],
)
def test_routed_expert_ab(device, expert_rank, weights_dtype, tile_transposed):
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig as C

    cap = _capture()
    trace, ckpt = golden.resolve_trace(golden.V4_FLASH), golden.resolve_checkpoint(golden.V4_FLASH)
    emb, hidden = C.EMB_SIZE, C.MOE_INTERMEDIATE_SIZE

    # The experts the reference router actually picked for this chunk, busiest first.
    ids = trace.expert_ids(LAYER, 0, CHUNK)
    counts = torch.bincount(ids.flatten(), minlength=C.NUM_ROUTED_EXPERTS)
    expert = int(counts.argsort(descending=True)[expert_rank])
    rows = (ids == expert).any(dim=1).nonzero().flatten()
    n = rows.numel()

    # Real weights (dequantized MXFP4 -> bf16 in the checkpoint) and the real FFN input rows.
    names = [f"layers.{LAYER}.ffn.experts.{expert}.w{m}.weight" for m in (1, 2, 3)]
    w = load_checkpoint_tensors(ckpt, names)
    weights = {
        "gate_proj": w[names[0]].float(),
        "down_proj": w[names[1]].float(),
        "up_proj": w[names[2]].float(),
    }
    x_active = cap["ffn_in"].reshape(CHUNK, emb)[rows].float()  # bf16-rounded: the device sees the same values

    with torch.no_grad():
        want = TorchExpert(emb, hidden, weights, activation=ACTIVATION_CLAMPED_SILU_GLU)(x_active)

    x = torch.zeros(CHUNK, emb)
    x[:n] = x_active

    def _idx(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
        )

    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=_idx([0]),
        emb_dim=emb,
        hidden_dim=hidden,
        max_tokens=CHUNK,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        weights_dram_nd_sharded=False,
        weights_tile_transposed=tile_transposed,
    )
    out = tt_expert(tt_x, _idx([n]), _idx([0]))
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:n].float()

    assert torch.isfinite(got).all()
    _, pcc = comp_pcc(want, got)
    logger.info(
        f"AB expert L{LAYER} e{expert} ({n} tokens) {weights_dtype.name}{'-tileT' if tile_transposed else ''}: "
        f"PCC {pcc:.6f}  rel-RMSE {_rel_rmse(want, got):.5f}"
    )


# --------------------------------------------------------------------------------------------------
# fp32 router logits: the MoE gate on real router weights, graded against the vLLM trace's own routing.
# --------------------------------------------------------------------------------------------------
def _galaxy_mesh():
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig as C
    from models.demos.deepseek_v3_d_p.tests.v4.helpers import mesh_params

    return [p for p in mesh_params(C.FABRIC_PAYLOAD_SIZE) if p.id == "torus-xy-8x4"]


@pytest.mark.skipif(not is_blackhole(), reason="V4 gate is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _galaxy_mesh(), indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize(
    "router_logits_dtype",
    [
        pytest.param(None, id="legacy-bf16"),
        pytest.param("bfloat16", id="fp32-matmul-round-once"),
        pytest.param("float32", id="fp32"),
    ],
)
def test_gate_ab(mesh_device, device_params, num_links, router_logits_dtype):
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig as C
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_sp_mesh_composer
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import (
        GateComputeMode,
        TtMoEGateConfig,
        TtMoEGatePrefill,
    )
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
    from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
    from models.demos.deepseek_v3_d_p.utils.test_utils import adjust_shapes_for_testing

    cap = _capture()
    trace, ckpt = golden.resolve_trace(golden.V4_FLASH), golden.resolve_checkpoint(golden.V4_FLASH)
    emb = C.EMB_SIZE

    config = TtMoEGateConfig.from_model_cfg(C, sp_dim=PREFILL_CHUNK_TOKENS_PER_CHIP)
    config.router_logits_dtype = router_logits_dtype
    config.ccl_config["NUM_LINKS"] = int(os.getenv("V4_AB_GATE_LINKS", num_links))
    config.ccl_config["TOPOLOGY"] = per_axis_topology(device_params["fabric_config"])[config.ccl_config["TP_AXIS"]]
    adjust_shapes_for_testing(config, mesh_device)
    assert mesh_device.shape[0] * config.sp_dim == CHUNK, (tuple(mesh_device.shape), config.sp_dim)

    names = [f"layers.{LAYER}.ffn.gate.weight", f"layers.{LAYER}.ffn.gate.bias"]
    w = load_checkpoint_tensors(ckpt, names)
    weight, bias = w[names[0]].to(torch.bfloat16), w[names[1]].float()

    x = cap["ffn_in"].reshape(CHUNK, emb)  # bf16
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
    )
    model = TtMoEGatePrefill(config, mesh_device, weight=weight, bias=bias, fallback_mode=GateComputeMode.DEVICE_FP32)
    tt_scores, tt_idx, tt_logits = model(tt_x)

    comp = get_sp_mesh_composer(mesh_device)
    got_idx = ttnn.to_torch(tt_idx, mesh_composer=comp).reshape(CHUNK, -1).to(torch.int64)
    got_scores = ttnn.to_torch(tt_scores, mesh_composer=comp).reshape(CHUNK, -1).float()
    got_logits = ttnn.to_torch(tt_logits, mesh_composer=comp).reshape(CHUNK, -1).float()

    want_logits = trace.router_logits(LAYER, 0, CHUNK)
    want_idx = trace.expert_ids(LAYER, 0, CHUNK)
    want_scores = trace.expert_weights(LAYER, 0, CHUNK)
    host_logits = x.float() @ weight.float().T

    k = want_idx.shape[1]
    hits = (got_idx.unsqueeze(2) == want_idx.unsqueeze(1)).any(dim=2).float().sum(dim=1)  # per-token overlap
    recall = float(hits.mean() / k)
    exact = float((hits == k).float().mean())
    _, pcc_trace = comp_pcc(want_logits, got_logits)
    _, pcc_host = comp_pcc(host_logits, got_logits)
    _, pcc_scores = comp_pcc(
        want_scores.sort(dim=-1, descending=True).values, got_scores.sort(dim=-1, descending=True).values
    )
    logger.info(
        f"AB gate L{LAYER} router_logits_dtype={router_logits_dtype}: logits PCC vs trace {pcc_trace:.6f}, "
        f"vs host-fp32 {pcc_host:.6f} | top-{k} recall vs trace {recall:.5f}, tokens routed identically {exact:.4f} "
        f"| selected-weights PCC {pcc_scores:.6f}"
    )
    assert torch.isfinite(got_logits).all()


# --------------------------------------------------------------------------------------------------
# Compressors and the CSA indexer. Inputs: the layer's exact attention input (capture_layer.py's
# ``attn_in``). Goldens: the host reference module loaded with the real checkpoint weights, and, for the
# compressors, the vLLM trace's own ``compressed_entries``.
# --------------------------------------------------------------------------------------------------
import contextlib  # noqa: E402

HCA_LAYER = int(os.getenv("V4_AB_HCA_LAYER", "3"))
CSA_LAYER = int(os.getenv("V4_AB_CSA_LAYER", "4"))
OUT_DIR = Path(os.getenv("V4_AB_OUT_DIR", "/data/ddjekic/v4_ab/out"))
_REPO = Path(__file__).resolve().parents[5]
_CSA_KERNEL = (
    _REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/csa_compressor/device/kernels"
    "/compute_csa_compressor.cpp"
)


def _v4_galaxy():
    from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import V4_MESH_CONFIGS

    return [p for p in V4_MESH_CONFIGS if p.id == "torus-xy-8x4"]


def _flash_config(layer: int):
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig as C

    return v4_hf_config(C, layer + 1, max_seq_len=56320)


def _load_into(layer: int, mapping: dict) -> None:
    """Copy ``layers.<layer>.<suffix>`` checkpoint tensors into the module parameters in ``mapping``."""
    ckpt = golden.resolve_checkpoint(golden.V4_FLASH)
    loaded = load_checkpoint_tensors(ckpt, [f"layers.{layer}.{suffix}" for suffix in mapping])
    with torch.no_grad():
        for suffix, param in mapping.items():
            value = loaded[f"layers.{layer}.{suffix}"]
            assert tuple(value.shape) == tuple(param.shape), (suffix, tuple(value.shape), tuple(param.shape))
            param.copy_(value)


def _compressor_map(comp, prefix="attn.compressor"):
    return {
        f"{prefix}.wkv.weight": comp.kv_proj.weight,
        f"{prefix}.wgate.weight": comp.gate_proj.weight,
        f"{prefix}.ape": comp.position_bias,
        f"{prefix}.norm.weight": comp.kv_norm.weight,
    }


def _indexer_map(idx):
    return {
        "attn.indexer.compressor.wkv.weight": idx.kv_proj.weight,
        "attn.indexer.compressor.wgate.weight": idx.gate_proj.weight,
        "attn.indexer.compressor.ape": idx.position_bias,
        "attn.indexer.compressor.norm.weight": idx.kv_norm.weight,
        "attn.indexer.wq_b.weight": idx.q_b_proj.weight,
        "attn.indexer.weights_proj.weight": idx.scorer.weights_proj.weight,
    }


def _record(tag: str, tensor: torch.Tensor) -> None:
    """Save a variant's output so the peer variant can say whether the two are bit-identical (a stale JIT
    cache would make an A/B compare a kernel to itself)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), OUT_DIR / f"{tag}.pt")


def _peer_note(tag: str, peer_tag: str, tensor: torch.Tensor) -> str:
    peer = OUT_DIR / f"{peer_tag}.pt"
    if not peer.is_file():
        return ""
    return f" | bit-identical to {peer_tag}: {bool(torch.equal(torch.load(peer), tensor.cpu()))}"


def _vs_trace(trace, layer: int, got: torch.Tensor, entries: int) -> str:
    try:
        want = trace.compressed_entries(layer, count=entries)
    except Exception as exc:  # the stream may be absent for a layer
        return f"trace n/a ({type(exc).__name__})"
    got2 = got.reshape(entries, -1).float()
    if want.shape != got2.shape:
        return f"trace shape {tuple(want.shape)} != {tuple(got2.shape)}"
    _, pcc = comp_pcc(want, got2)
    return f"PCC vs trace {pcc:.6f} rel-RMSE {_rel_rmse(want, got2):.5f}"


@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, topology", _v4_galaxy(), indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("softmax_config", ["old-default", "new-hifi4-fp32"])
def test_hca_compressor_ab(mesh_device, device_params, topology, softmax_config):
    """HCA compressor: the gate softmax over 128 tokens, default kernel config vs HiFi4 / fp32 / exact."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACompressor
    from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtHCACompressor

    layer, cfg = HCA_LAYER, _flash_config(HCA_LAYER)
    if cfg.layer_types[layer] != "heavily_compressed_attention":
        pytest.skip(f"layer {layer} is {cfg.layer_types[layer]}, not HCA")
    cap, trace = _capture(layer), golden.resolve_trace(golden.V4_FLASH)
    rate = cfg.compress_rates["heavily_compressed_attention"]

    ref = DeepseekV4HCACompressor(cfg).eval()
    _load_into(layer, _compressor_map(ref))
    hidden = cap["attn_in"].reshape(1, CHUNK, cfg.hidden_size).float()
    with torch.no_grad():
        want, _ = ref(
            hidden,
            torch.zeros(1, CHUNK, cfg.q_lora_rank),
            torch.arange(CHUNK).unsqueeze(0),
            past_key_values=None,
            layer_idx=0,
        )

    tt = TtHCACompressor.from_reference(mesh_device, ref, cfg, sp_axis=0, tp_axis=1, topology=topology)
    tt.ccl_num_links = int(os.getenv("V4_AB_GATE_LINKS", "1"))  # hard-coded 2 on Blackhole; this host has 1 link
    if softmax_config == "old-default":
        tt._softmax_compute_kernel_config = None  # what the code passed before the branch (op default)
    hidden_padded, seq_len_actual = TtHCACompressor.prepare_input(hidden, mesh_device.shape[0], rate)
    tt_in = ttnn.from_torch(
        hidden_padded.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    )
    tt.alloc_tables(hidden_padded.shape[1], hidden_padded.shape[1], hidden_padded.shape[1] // rate)
    kv_tt, _mask = tt(tt_in, seq_len_actual=seq_len_actual)
    got = ttnn.to_torch(
        kv_tt,
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
    )[:, :, : seq_len_actual // rate].float()

    assert torch.isfinite(got).all()
    _, pcc = comp_pcc(want.float(), got.reshape(want.shape))
    _record(f"hca_L{layer}-{softmax_config}", got)
    peer = "new-hifi4-fp32" if softmax_config == "old-default" else "old-default"
    logger.info(
        f"AB hca-compressor L{layer} softmax={softmax_config}: PCC vs host-ref {pcc:.6f} rel-RMSE "
        f"{_rel_rmse(want.float(), got.reshape(want.shape)):.5f} | {_vs_trace(trace, layer, got, seq_len_actual // rate)}"
        + _peer_note(softmax_config, f"hca_L{layer}-{peer}", got)
    )


@contextlib.contextmanager
def _csa_exp_variant(approx: bool):
    """Temporarily put the CSA compressor kernel back to the approximate exp (the pre-branch state).

    The kernel source is JIT-compiled on first use, so the swap must wrap the whole device run. The original
    bytes are restored in ``finally``; the check up front refuses to run if the file is not in the branch's
    exact-exp state."""
    src = _CSA_KERNEL.read_text()
    assert "exp_tile_init<false>();" in src and "exp_tile<false>(kWeight);" in src, "kernel not in exact-exp state"
    if not approx:
        yield
        return
    _CSA_KERNEL.write_text(
        src.replace("exp_tile_init<false>();", "exp_tile_init<true>();").replace(
            "exp_tile<false>(kWeight);", "exp_tile<true>(kWeight);"
        )
    )
    try:
        yield
    finally:
        _CSA_KERNEL.write_text(src)


@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, topology", _v4_galaxy(), indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("exp_variant", ["old-approx-exp", "new-exact-exp"])
def test_csa_compressor_ab(mesh_device, device_params, topology, exp_variant):
    """CSA compressor: the pool softmax's exp, approximate (old kernel) vs exact (branch kernel)."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACompressor
    from models.demos.deepseek_v3_d_p.tt.mla.compressor import CSA_STATE_ROWS, TtCSACompressor

    layer, cfg = CSA_LAYER, _flash_config(CSA_LAYER)
    if cfg.layer_types[layer] != "compressed_sparse_attention":
        pytest.skip(f"layer {layer} is {cfg.layer_types[layer]}, not CSA")
    cap, trace = _capture(layer), golden.resolve_trace(golden.V4_FLASH)
    rate = cfg.compress_rates["compressed_sparse_attention"]

    ref = DeepseekV4CSACompressor(cfg).eval()
    _load_into(layer, {**_compressor_map(ref), **_indexer_map(ref.indexer)})
    hidden = cap["attn_in"].reshape(1, CHUNK, cfg.hidden_size).float()
    with torch.no_grad():
        want, _ = ref(
            hidden,
            torch.zeros(1, CHUNK, cfg.q_lora_rank),
            torch.arange(CHUNK).unsqueeze(0),
            past_key_values=None,
            layer_idx=0,
        )

    hidden_padded, seq_len_actual = TtCSACompressor.prepare_input(hidden, mesh_device.shape[0], rate)
    sp = mesh_device.shape[0]
    init_kv = torch.zeros(1, 1, CSA_STATE_ROWS, cfg.head_dim, dtype=torch.bfloat16)
    init_score = torch.full_like(init_kv, float("-inf"))
    state_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None))

    with _csa_exp_variant(approx=exp_variant.startswith("old")):
        tt = TtCSACompressor.from_reference(mesh_device, ref, cfg, sp_axis=0, tp_axis=1, topology=topology)
        tt.ccl_num_links = int(os.getenv("V4_AB_GATE_LINKS", "1"))
        tt_in = ttnn.from_torch(
            hidden_padded.unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
        )
        tt_kv0 = ttnn.from_torch(
            init_kv.repeat(1, 1, sp, 1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=state_mapper,
        )
        tt_sc0 = ttnn.from_torch(
            init_score.repeat(1, 1, sp, 1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=state_mapper,
        )
        tt.alloc_tables(hidden_padded.shape[1], hidden_padded.shape[1])
        kv_tt, _kv_state, _score_state = tt(tt_in, tt_kv0, tt_sc0, seq_len_actual=seq_len_actual)
        got = ttnn.to_torch(
            kv_tt,
            mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(1, 1))),
        )[:, :, : seq_len_actual // rate].float()

    assert torch.isfinite(got).all()
    _, pcc = comp_pcc(want.float(), got.reshape(want.shape))
    _record(f"csa_L{layer}-{exp_variant}", got)
    peer = "new-exact-exp" if exp_variant == "old-approx-exp" else "old-approx-exp"
    logger.info(
        f"AB csa-compressor L{layer} {exp_variant}: PCC vs host-ref {pcc:.6f} rel-RMSE "
        f"{_rel_rmse(want.float(), got.reshape(want.shape)):.5f} | {_vs_trace(trace, layer, got, seq_len_actual // rate)}"
        + _peer_note(exp_variant, f"csa_L{layer}-{peer}", got)
    )


def _topk_overlap(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    """(mean overlap, share of queries with overlap >= 0.99, share identical) over queries with >=1 valid key."""
    overlaps = []
    for row_a, row_e in zip(actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, expected.shape[-1])):
        expected_set = set(row_e[row_e >= 0].tolist())
        if not expected_set:
            continue
        actual_set = set(row_a[row_a >= 0].tolist())
        overlaps.append(len(expected_set & actual_set) / len(expected_set))
    t = torch.tensor(overlaps)
    return float(t.mean()), float((t >= 0.99).float().mean()), float((t == 1.0).float().mean())


@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params, topology", _v4_galaxy(), indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("config_variant", ["old-default-cfgs", "new-branch-cfgs"])
def test_csa_indexer_ab(mesh_device, device_params, topology, config_variant, tmp_path):
    """CSA indexer: q_b_proj / Hadamard / weights_proj / score-op kernel configs, pre-branch vs branch.

    The pre-branch state is emulated on the built indexer: q_b, the Hadamard and weights_proj matmuls take the
    default (HiFi2, bf16-dest) config, and the score op takes no config (so it defaults to LoFi for bf8 q + keys)."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Indexer,
        DeepseekV4RMSNorm,
    )
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtCsaIndexer
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

    layer, cfg = CSA_LAYER, _flash_config(CSA_LAYER)
    if cfg.layer_types[layer] != "compressed_sparse_attention":
        pytest.skip(f"layer {layer} is {cfg.layer_types[layer]}, not CSA")
    cap = _capture(layer)
    hidden = cap["attn_in"].reshape(1, CHUNK, cfg.hidden_size)  # bf16, what the device is fed

    # qr = q_a_norm(q_a_proj(attn_in)): the attention's own low-rank query, on the real weights.
    q_norm = DeepseekV4RMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps).eval()
    wq_a = torch.empty(cfg.q_lora_rank, cfg.hidden_size)
    _load_into(layer, {"attn.wq_a.weight": wq_a, "attn.q_norm.weight": q_norm.weight})
    with torch.no_grad():
        qr = q_norm(hidden.float() @ wq_a.T)

    ref = DeepseekV4Indexer(cfg).eval()
    _load_into(layer, _indexer_map(ref))
    with torch.no_grad():
        want = ref(hidden.float(), qr, torch.arange(CHUNK).unsqueeze(0), past_key_values=None, layer_idx=0)

    sp = mesh_device.shape[0]
    sp_topo, tp_topo = per_axis_topology(device_params["fabric_config"])
    default_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    hifi4_cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    indexer = TtCsaIndexer.from_reference(
        ref,
        config=cfg,
        mesh_device=mesh_device,
        sp_axis=0,
        tp_axis=1,
        default_compute_kernel_config=default_cfg,
        hifi4_fp32_compute_kernel_config=hifi4_cfg,
        weight_cache_path=tmp_path,
        layer_idx=0,
        tt_ccl=get_tt_ccl(mesh_device),
        ccl_num_links=int(os.getenv("V4_AB_GATE_LINKS", "1")),
        sp_ccl_topology=sp_topo,
        tp_ccl_topology=tp_topo,
        seq_len=CHUNK,
        active_seq_len=CHUNK,
        slot_num=1,
        layer_num=1,
    )
    indexer._compressor.ccl_num_links = int(os.getenv("V4_AB_GATE_LINKS", "1"))  # inner compressor hard-codes 2
    if config_variant == "old-default-cfgs":
        indexer.q_b_compute_kernel_config = default_cfg
        indexer.hifi4_fp32_compute_kernel_config = default_cfg
        indexer.score_compute_kernel_config = None

    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=cfg.index_head_dim,
        mesh_device=mesh_device,
        seq_len=CHUNK // 4,
        mesh_shape=list(mesh_device.shape),
        sp_axis=0,
        num_kvpe_cache_layers=1,
        num_users=1,
    )
    dev_hidden = ttnn.from_torch(
        hidden.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    )
    dev_qr = ttnn.from_torch(
        qr.bfloat16().unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None)),
    )
    out = indexer.forward(
        dev_hidden,
        dev_qr,
        seq_len=CHUNK // sp,
        start_pos=0,
        cache_user_id=0,
        cache_layer_idx=0,
        index_kv_cache=index_cache,
    )
    composed = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 1))
    ).to(torch.int64)
    got = torch.where(composed[:, 0] == 0xFFFFFFFF, -1, composed[:, 0])  # TP rank 0
    assert got.shape == want.shape, (tuple(got.shape), tuple(want.shape))

    mean_ov, ge99, identical = _topk_overlap(got, want)
    _record(f"idx_L{layer}-{config_variant}", got)
    peer = "new-branch-cfgs" if config_variant == "old-default-cfgs" else "old-default-cfgs"
    logger.info(
        f"AB csa-indexer L{layer} {config_variant}: top-{want.shape[-1]} overlap vs host-ref mean {mean_ov:.5f}, "
        f"queries >=0.99: {ge99:.4f}, identical sets: {identical:.4f}"
        + _peer_note(config_variant, f"idx_L{layer}-{peer}", got)
    )
