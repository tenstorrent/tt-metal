# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3.2 MLA/indexer/KV-cache vs the OFFICIAL reference values.

Unlike test_mla.py (whose truth is the in-repo MLACPU model), the truth here is
the recorded output of the official DeepSeek-V3.2 CUDA stack, captured from a
vLLM run (a single 5120-token prefill) and stored as safetensors streams. See
context/reference-gpu-results.md. Reference layers available: 0, 30, 60.

Streams (per layer L), loaded by load_stream:
  module_io/mla_input_layer_L     [5120, 7168] bf16   (== indexer_input)
  module_io/mla_output_layer_L    [5120, 7168] bf16
  dsa/indexer_logits_layer_L      [5120, 5120] fp32   (pre-causal-mask index_score)
  dsa/dsa_topk_indices_layer_L    [5120, 2048] int32  (-1 = unfilled / causal pad)
  kv_cache/layer_L                [5120, 576]  bf16   (latent kv 512 ++ k_pe 64)

The reference dir defaults to the sibling bit_sculpt checkout; override with
$DEEPSEEK_V32_REF_DIR. Pretrained layer weights are pulled from HF on demand
(cached) by build_cpu_reference(layer=L) — multi-GB shards, downloaded once.

FUNCTIONAL-PARITY caveat (status.md §A.6): the reference ran the real FP8 (ue8m0)
+ Hadamard indexer path and an fp8 KV cache; our port runs the bf16 functional
path (use_fp8_path=False, simulate_fp8=False). Hadamard is orthogonal (q·k
unchanged) but fp8 adds quantization noise, so PCC is high-but-not-exact BY
DESIGN. Thresholds below absorb that gap; the tests log actual numbers.

KNOWN k_pe ROPE-FRAME ARTIFACT: the KV-cache k_pe slice (last 64) compares poorly
element-wise (~0.43 PCC) because the reference stores it in a different RoPE
frame (rotate_half vs our interleaved). This is harmless — RoPE preserves the
per-row L2 norm (matches to ~0.4%) and q·k is frame-invariant, so the MLA output
matches at ~0.9998. The KV tests therefore assert the latent slice + a
frame-invariant L2 check on k_pe, and log the raw k_pe PCC as a diagnostic only.
For cross-stack interop (a vLLM-written cache consumed by our kernel, or vice versa)
the layout DOES matter: pass --ds-kpe-layout vllm to reindex our k_pe to vLLM's
half-split layout (interleaved_to_halfsplit_perm in tt/mla/mla.py) and assert a hard
element-wise k_pe PCC (~0.99997) instead of the frame-invariant L2.

Runtime (Blackhole 1x4, layer shard already downloaded — measured layer 0):
  host_*  (CPU only)            ~25 s for all 3 (shared module fixture: weight load +
                                one 5120 forward; the 3 asserts are ~0 s each)
  device indexer                ~45 s  (10 s mesh setup + 30 s device stems/score/topk)
  device kv                     ~65 s  (forward + sparse_mla host fallback, ~24 GB RAM)
  device mla                    ~50 s  (forward + sparse_mla host fallback, ~24 GB RAM)
First run adds one-time JIT kernel compile (~1-2 min) and, if uncached, a multi-GB
HF shard download. All are correctness gates → marked `gate`; @timeout(0).
"""

import glob
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.tests.mesh_utils import parametrize_mesh_device
from models.demos.deepseek_v32.tests.test_mla import assert_config_matches, build_cpu_reference
from models.demos.deepseek_v32.tt import ops
from models.demos.deepseek_v32.tt.mla import interleaved_to_halfsplit_perm, ttMLA

pytestmark = pytest.mark.gate

SEQ_LEN = 5120  # the reference prefill length (fixed by the capture)
REF_LAYERS = [0, 30, 60]

# Thresholds — functional-parity (fp8 ref vs bf16 port). Observed layer 0:
# logits 0.96, topk mean 0.985 (min row 0.91), latent 0.998, output 0.9998.
LOGITS_PCC = 0.95
TOPK_OVERLAP_MEAN = 0.95
TOPK_OVERLAP_ROW_MIN = 0.85
KV_LATENT_PCC = 0.99
KV_PE_L2_RELERR_MAX = 0.05  # frame-invariant: per-row ||k_pe|| must match
KV_PE_VLLM_PCC = 0.999  # element-wise, once our k_pe is reindexed to vLLM's half-split layout
OUTPUT_PCC = 0.98


# ----------------------------------------------------------------------------
# Reference loading
# ----------------------------------------------------------------------------
def _ref_dir() -> Path:
    env = os.environ.get("DEEPSEEK_V32_REF_DIR")
    if env:
        return Path(env)
    # sibling checkout: <...>/tt-metal and <...>/bit_sculpt
    return Path(__file__).resolve().parents[5] / "bit_sculpt" / "results" / "deepseek-v32"


def load_stream(stream_dir: Path) -> torch.Tensor:
    """One (stream, layer) → full tensor; single-file or chunked rows_*.safetensors."""
    from safetensors.torch import load_file

    files = sorted(glob.glob(f"{stream_dir}/rows_*.safetensors"))
    parts = []
    for f in files:
        d = load_file(f)
        (key,) = d.keys()
        parts.append(d[key])
    return torch.cat(parts, dim=0)


def load_reference(layer: int) -> dict:
    """Load every reference stream for `layer`, or skip if the data is absent."""
    root = _ref_dir()
    need = {
        "mla_in": root / "module_io" / f"mla_input_layer_{layer}",
        "mla_out": root / "module_io" / f"mla_output_layer_{layer}",
        "logits": root / "dsa" / f"indexer_logits_layer_{layer}",
        "topk": root / "dsa" / f"dsa_topk_indices_layer_{layer}",
        "kv": root / "kv_cache" / f"layer_{layer}",
    }
    missing = [str(p) for p in need.values() if not glob.glob(f"{p}/rows_*.safetensors")]
    if missing:
        pytest.skip(f"reference streams not found (set $DEEPSEEK_V32_REF_DIR): missing {missing}")
    return {k: load_stream(v) for k, v in need.items()}


def _topk_overlap(ref_topk: torch.Tensor, got_topk: torch.Tensor, rows) -> dict:
    """Per-row Jaccard-style overlap |ref ∩ got| / |ref| over the causal valid set."""
    out = {}
    for r in rows:
        n = min(r + 1, ref_topk.shape[1])
        want = set(ref_topk[r][ref_topk[r] >= 0].tolist())
        got = got_topk[r]
        got = set(got[(got >= 0) & (got < SEQ_LEN) & (got <= r)].tolist())
        out[r] = len(want & got) / max(1, len(want))
    return out


# ----------------------------------------------------------------------------
# Host ceiling: MLACPU (the in-repo golden) vs the official reference.
# Establishes the achievable PCC — any device gap above this is a port issue,
# not a model-formulation gap. CPU only, no device needed.
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _host(request):
    """Run MLACPU once per layer: returns (logits, topk, kvpe, output) + reference."""
    layer = request.param
    ref = load_reference(layer)
    args, mla_cpu, _, _ = build_cpu_reference(SEQ_LEN, layer=layer)  # functional path set inside
    x = ref["mla_in"].unsqueeze(0)  # [1, S, hidden]
    freqs = precompute_freqs_cis(args)[:SEQ_LEN]
    mask = torch.full((SEQ_LEN, SEQ_LEN), float("-inf")).triu_(1)
    with torch.no_grad():
        qr = mla_cpu.q_norm(mla_cpu.wq_a(x))
        topk, logits = mla_cpu.indexer(x, qr, 0, freqs, mask)  # logits [1,S,S], topk [1,S,k]
        out = mla_cpu.forward(x, 0, freqs, mask)[0]  # also fills kv_cache/pe_cache
    kvpe = torch.cat([mla_cpu.kv_cache[0, :SEQ_LEN], mla_cpu.pe_cache[0, :SEQ_LEN]], dim=-1)  # [S,576]
    return dict(ref=ref, logits=logits[0], topk=topk[0], kvpe=kvpe, out=out)


@pytest.mark.parametrize("_host", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_indexer_host_vs_reference(_host):
    """Host indexer logits + top-k vs the official capture. ~1-2 min."""
    ref, logits = _host["ref"], _host["logits"]
    tril = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
    _, pcc = comp_pcc(ref["logits"][tril].float(), logits[tril].float(), 0)
    logger.info(f"[host] indexer logits PCC (causal region): {pcc}")
    rows = [0, 1, 100, 1000, 2047, 2048, 2049, 3000, 4096, SEQ_LEN - 1]
    ov = _topk_overlap(ref["topk"], _host["topk"], rows)
    mean = sum(ov.values()) / len(ov)
    logger.info(f"[host] topk overlap mean={mean:.4f} per-row={ {r: round(v,4) for r,v in ov.items()} }")
    assert pcc >= LOGITS_PCC, f"indexer logits PCC {pcc} < {LOGITS_PCC}"
    assert mean >= TOPK_OVERLAP_MEAN, f"topk overlap mean {mean} < {TOPK_OVERLAP_MEAN}"
    assert min(ov.values()) >= TOPK_OVERLAP_ROW_MIN, f"topk overlap row min {min(ov.values())} < {TOPK_OVERLAP_ROW_MIN}"


@pytest.mark.parametrize("_host", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_kv_cache_host_vs_reference(_host, ds_kpe_layout):
    """Host KV cache vs the official capture: latent PCC + k_pe (--ds-kpe-layout). ~1-2 min."""
    ref_kv, kvpe = _host["ref"]["kv"], _host["kvpe"]
    _assert_kv(ref_kv, kvpe, tag="host", kpe_layout=ds_kpe_layout)


@pytest.mark.parametrize("_host", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_mla_output_host_vs_reference(_host):
    """Host MLA output vs the official capture. ~1-2 min."""
    _, pcc = comp_pcc(_host["ref"]["mla_out"].float(), _host["out"].float(), 0)
    logger.info(f"[host] MLA output PCC: {pcc}")
    assert pcc >= OUTPUT_PCC, f"MLA output PCC {pcc} < {OUTPUT_PCC}"


def _assert_kv(ref_kv: torch.Tensor, kvpe: torch.Tensor, tag: str, kpe_layout: str = "interleaved"):
    """Shared KV-cache assertion. Always asserts the latent-512 PCC. For the k_pe
    rope half (last 64) the check depends on kpe_layout (--ds-kpe-layout):

      "interleaved" (default): our/official layout differs from vLLM's only by a
        RoPE-frame permutation, so a raw element-wise PCC is meaningless. Assert a
        frame-INVARIANT per-row L2 match; log raw k_pe PCC as a diagnostic.
      "vllm": reindex our k_pe to vLLM's half-split layout via
        interleaved_to_halfsplit_perm() (see mla.py) and assert element-wise PCC —
        the hard check for cross-stack KV-cache interop."""
    kv = 512
    _, lat = comp_pcc(ref_kv[:, :kv].float(), kvpe[:, :kv].float(), 0)
    assert lat >= KV_LATENT_PCC, f"KV latent PCC {lat} < {KV_LATENT_PCC}"
    pe_ref, pe_got = ref_kv[:, kv:].float(), kvpe[:, kv:].float()
    if kpe_layout == "vllm":
        pe_got = pe_got[:, interleaved_to_halfsplit_perm(pe_got.shape[1])]
        _, pe_pcc = comp_pcc(pe_ref, pe_got, 0)
        logger.info(f"[{tag}] KV latent PCC={lat}  k_pe PCC (vLLM half-split layout)={pe_pcc}")
        assert pe_pcc >= KV_PE_VLLM_PCC, f"k_pe (vLLM-layout) PCC {pe_pcc} < {KV_PE_VLLM_PCC}"
    else:
        _, pe_pcc = comp_pcc(pe_ref, pe_got, 0)
        pe_l2_relerr = ((pe_got.norm(dim=1) - pe_ref.norm(dim=1)).abs() / (pe_ref.norm(dim=1) + 1e-3)).mean().item()
        logger.info(f"[{tag}] KV latent PCC={lat}  k_pe PCC={pe_pcc} (frame diag)  k_pe L2 rel-err={pe_l2_relerr:.4f}")
        assert pe_l2_relerr <= KV_PE_L2_RELERR_MAX, f"k_pe L2 rel-err {pe_l2_relerr} > {KV_PE_L2_RELERR_MAX}"


# ----------------------------------------------------------------------------
# Device: the TT implementation vs the official reference (Blackhole 1x4).
# ----------------------------------------------------------------------------
_DEVICE_PARAMS = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    }
]


def _make_mla(config, layer, mesh_device, is_chunked=False):
    config.max_seq_len = SEQ_LEN  # rope-table length (same hack as v3 run_model / test_mla)
    args, _, weights, _ = build_cpu_reference(SEQ_LEN, layer=layer)
    assert_config_matches(config, args)
    return ttMLA(
        config, weights, mesh_device, layer_idx=0, seq_len=SEQ_LEN, sp_axis=0, tp_axis=1, is_chunked=is_chunked
    )


def _shard_tp(t, mesh_device):
    """Input [1,1,S,hidden] sharded on hidden across TP (indexer-stem layout)."""
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, -1)),
    )


@parametrize_mesh_device()
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.parametrize("layer", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS])
@pytest.mark.timeout(0)
def test_indexer_device_vs_reference(mesh_device, layer, device_params, variant, config_only, monkeypatch):
    """Device indexer (stems + indexer_score + top-k) vs the official capture. ~2-3 min."""
    ref = load_reference(layer)
    mla = _make_mla(config_only, layer, mesh_device)
    x = ref["mla_in"].reshape(1, 1, SEQ_LEN, -1)  # [1,1,S,hidden]

    captured = {}
    orig = ops.indexer_logits
    monkeypatch.setattr(ops, "indexer_logits", lambda *a, **k: captured.setdefault("logits", orig(*a, **k)))
    idx = mla._indexer_topk(_shard_tp(x, mesh_device), SEQ_LEN)

    logits = ops._to_host(captured["logits"]).float()[0, 0]  # [S, S]; future cols -inf
    got_topk = ops._to_host(idx).long()[0, 0]  # [S, k]
    # PCC only over the causal region (device masks future to -inf; ref is pre-mask).
    tril = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
    _, pcc = comp_pcc(ref["logits"][tril].float(), logits[tril], 0)
    logger.info(f"[device] indexer logits PCC (causal region): {pcc}")
    rows = [0, 1, 100, 1000, 2047, 2048, 2049, 3000, 4096, SEQ_LEN - 1]
    ov = _topk_overlap(ref["topk"], got_topk, rows)
    mean = sum(ov.values()) / len(ov)
    logger.info(f"[device] topk overlap mean={mean:.4f} per-row={ {r: round(v,4) for r,v in ov.items()} }")
    ttnn.synchronize_device(mesh_device)
    assert pcc >= LOGITS_PCC, f"indexer logits PCC {pcc} < {LOGITS_PCC}"
    assert mean >= TOPK_OVERLAP_MEAN, f"topk overlap mean {mean} < {TOPK_OVERLAP_MEAN}"
    assert min(ov.values()) >= TOPK_OVERLAP_ROW_MIN, f"topk overlap row min {min(ov.values())} < {TOPK_OVERLAP_ROW_MIN}"


def _run_device_forward(config, layer, mesh_device):
    """Single-shot ttMLA forward over the reference input; returns (output[1,S,hidden], kvpe[S,576])."""
    ref = load_reference(layer)
    mla = _make_mla(config, layer, mesh_device)
    sp_axis, tp_axis = 0, 1
    kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_LEN,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors(SEQ_LEN)

    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    tt_x = ttnn.from_torch(
        ref["mla_in"].reshape(1, 1, SEQ_LEN, -1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    out = mla.forward(tt_x, rope_tensors, kvpe_cache)
    out_t = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)[
        0
    ]  # [1, S, hidden]
    # KVPE: replicated across TP — concat TP replicas on the unused dim 1, keep first.
    kvpe_t = ttnn.to_torch(
        kvpe_cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)[
        0, 0, :SEQ_LEN
    ]  # [S, 576]
    return ref, out_t, kvpe_t


@parametrize_mesh_device()
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.parametrize("layer", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS])
@pytest.mark.timeout(0)
def test_kv_cache_device_vs_reference(mesh_device, layer, device_params, variant, config_only, ds_kpe_layout):
    """Device KV cache vs the official capture: latent PCC + k_pe (--ds-kpe-layout). ~4-7 min."""
    ref, _, kvpe = _run_device_forward(config_only, layer, mesh_device)
    _assert_kv(ref["kv"], kvpe, tag="device", kpe_layout=ds_kpe_layout)
    ttnn.synchronize_device(mesh_device)


@parametrize_mesh_device()
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.parametrize("layer", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS])
@pytest.mark.timeout(0)
def test_mla_output_device_vs_reference(mesh_device, layer, device_params, variant, config_only):
    """Device MLA output vs the official capture. ~4-7 min (sparse_mla host fallback)."""
    ref, out, _ = _run_device_forward(config_only, layer, mesh_device)
    tt = out.unsqueeze(0)  # [1,1,S,hidden] -> compare as [1,S,hidden]
    ref_out = ref["mla_out"].reshape(1, SEQ_LEN, -1)
    for nm, sl in [("rows<2048", slice(0, 2048)), ("rows>=2048", slice(2048, SEQ_LEN))]:
        _, m = comp_pcc(ref_out[:, sl].float(), out[:, sl].float(), 0)
        logger.info(f"[device] band {nm}: {m}")
    _, pcc = comp_pcc(ref_out.float(), out.float(), 0)
    logger.info(f"[device] MLA output PCC: {pcc}")
    ttnn.synchronize_device(mesh_device)
    assert pcc >= OUTPUT_PCC, f"MLA output PCC {pcc} < {OUTPUT_PCC}"
