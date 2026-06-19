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

────────────────────────────────────────────────────────────────────────────────
GLM-5.1 (model id `glm_5_1`) — same harness, different model
────────────────────────────────────────────────────────────────────────────────
GLM-5.1 (`zai-org/GLM-5.1`, HF `glm_moe_dsa`) is a DeepSeek-V3.2-family model (MLA +
DSA indexer + sparse attn); its vLLM trace mirrors the V3.2 layout, so every test
body below is shared. The host/device tests are parametrized over `model` ∈
{deepseek_v32, glm_5_1} (filter with `-k glm_5_1` / `-k deepseek_v32`). GLM differs
only in: hidden 6144 (vs 7168); 64 q-heads (vs 128) → sparse_sdpa needs per-chip
H = 64/tp ≥ 32, so **tp ≤ 2** (tp>2 meshes are skipped for GLM); indexer RoPE is
INTERLEAVED (DS's is not); and NO YaRN (scale = qk_head_dim**-0.5). The config is
hand-built (transformers can't load glm_moe_dsa) and weights load from
zai-org/GLM-5.1 via the identical HF→MLACPU map. Trace dir: $GLM51_REF_DIR (default
<repo>/bit_sculpt/results/glm-51); layers 0/30/60/77. The GLM-only Phase-1 block at
the bottom validates the trace bundle (no weights/device). Full GLM run guide +
weight/trace download: context/GLM_5_1_TRACE.md.
"""

import glob
import os
import types  # GLM: build the HF-attribute config by hand (transformers can't load glm_moe_dsa)
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config  # GLM dims
from models.demos.deepseek_v3_d_p.tests.conftest import _resolve_config_only  # == the config_only fixture
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.demos.deepseek_v32.reference_cpu.model import MLACPU, ModelArgs  # GLM CPU reference
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.reference_cpu.weights import initialize_weights  # GLM per-layer weights
from models.demos.deepseek_v32.tests.mesh_utils import parametrize_mesh_device, skip_if_seq_too_small_for_sp
from models.demos.deepseek_v32.tests.test_mla import WEIGHT_NAME_MAP, assert_config_matches, build_cpu_reference
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
# GLM-5.1 specifics. The DeepSeek-V3.2 code below is unchanged; GLM reuses every
# test body and differs only here: its config/weights builders, its layer set, and
# the tp ≤ 2 mesh cap. (model, layer) cases drive the shared host/device tests.
# ----------------------------------------------------------------------------
GLM_REF_LAYERS = [0, 30, 60, 77]  # indexer_logits captured for these (topk/kv exist for all 78)
GLM_REPO = os.environ.get("GLM51_REPO", "zai-org/GLM-5.1")  # bf16 master; "-FP8" also works (loader dequants)
# (model, layer) cases — DS and GLM share every host/device body.
_CASES = [("deepseek_v32", l) for l in REF_LAYERS] + [("glm_5_1", l) for l in GLM_REF_LAYERS]
_CASE_IDS = [f"{m}-L{l}" for m, l in _CASES]


def _glm_model_args() -> ModelArgs:
    """GLM reference_cpu ModelArgs. max_seq_len == original_seq_len disables BOTH the YaRN
    freq scaling and the mscale² softmax correction → plain rope, scale = qk_head_dim**-0.5.
    index_rope_interleave=True flips IndexerCPU to interleaved RoPE (DS's is non-interleaved)."""
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=8192,
        original_seq_len=8192,
        dim=GLM51Config.EMB_SIZE,
        n_heads=GLM51Config.NUM_ATTENTION_HEADS,
        q_lora_rank=GLM51Config.Q_LORA_RANK,
        kv_lora_rank=GLM51Config.KV_LORA_RANK,
        qk_nope_head_dim=GLM51Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=GLM51Config.QK_ROPE_HEAD_DIM,
        v_head_dim=GLM51Config.V_HEAD_DIM,
        rope_theta=GLM51Config.ROPE_THETA,
        rope_factor=1.0,
        mscale=1.0,
        index_n_heads=GLM51Config.INDEX_N_HEADS,
        index_head_dim=GLM51Config.INDEX_HEAD_DIM,
        index_topk=GLM51Config.INDEX_TOPK,
        index_rope_interleave=True,
    )


def _build_glm_cpu_reference(layer: int):
    """GLM MLACPU with layer-`layer` pretrained weights from GLM_REPO (per-layer shards only;
    bf16 passes through, fp8 is dequantized). Functional path (use_fp8_path=False,
    simulate_fp8=False) — Hadamard + fp8 dropped, KV stored bf16."""
    args = _glm_model_args()
    mla = MLACPU(args, simulate_fp8=False).eval()
    mla.indexer.use_fp8_path = False
    initialize_weights(mla, layer=layer, repo=GLM_REPO)
    return args, mla


def _glm_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the v32 ttMLA reads (GLM dims, no YaRN). Built directly —
    AutoConfig can't load glm_moe_dsa. rope_scaling.factor=1.0 → no mscale (scale=256**-0.5)
    and DeepseekV3YarnRotaryEmbedding(factor=1) → plain RoPE at θ=1e6."""
    return types.SimpleNamespace(
        hidden_size=GLM51Config.EMB_SIZE,
        num_attention_heads=GLM51Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=GLM51Config.NUM_ATTENTION_HEADS,
        kv_lora_rank=GLM51Config.KV_LORA_RANK,
        q_lora_rank=GLM51Config.Q_LORA_RANK,
        qk_nope_head_dim=GLM51Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=GLM51Config.QK_ROPE_HEAD_DIM,
        v_head_dim=GLM51Config.V_HEAD_DIM,
        rms_norm_eps=GLM51Config.RMS_NORM_EPS,
        max_seq_len=max_seq,
        rope_theta=float(GLM51Config.ROPE_THETA),
        attention_bias=False,
        rope_scaling={
            "factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": max_seq,
        },
    )


# ----------------------------------------------------------------------------
# Reference loading
# ----------------------------------------------------------------------------
def _ref_dir(model: str) -> Path:
    """Trace bundle dir for `model`: $GLM51_REF_DIR / $DEEPSEEK_V32_REF_DIR, else the
    in-tree (GLM) / sibling (DS) bit_sculpt checkout (this file → parents[4] == repo)."""
    if model == "glm_5_1":
        env = os.environ.get("GLM51_REF_DIR")
        return Path(env) if env else Path(__file__).resolve().parents[4] / "bit_sculpt" / "results" / "glm-51"
    env = os.environ.get("DEEPSEEK_V32_REF_DIR")
    # sibling checkout: <...>/tt-metal and <...>/bit_sculpt
    return Path(env) if env else Path(__file__).resolve().parents[5] / "bit_sculpt" / "results" / "deepseek-v32"


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


def load_reference(model: str, layer: int) -> dict:
    """Load every reference stream for (`model`, `layer`), or skip if the data is absent."""
    root = _ref_dir(model)
    need = {
        "mla_in": root / "module_io" / f"mla_input_layer_{layer}",
        "mla_out": root / "module_io" / f"mla_output_layer_{layer}",
        "logits": root / "dsa" / f"indexer_logits_layer_{layer}",
        "topk": root / "dsa" / f"dsa_topk_indices_layer_{layer}",
        "kv": root / "kv_cache" / f"layer_{layer}",
    }
    if model == "glm_5_1":
        need["idx_in"] = root / "module_io" / f"indexer_input_layer_{layer}"  # for the Phase-1 trace checks
    missing = [str(p) for p in need.values() if not glob.glob(f"{p}/rows_*.safetensors")]
    if missing:
        env = "GLM51_REF_DIR" if model == "glm_5_1" else "DEEPSEEK_V32_REF_DIR"
        pytest.skip(f"{model} reference streams not found (set ${env}): missing {missing}")
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
    """Run MLACPU once per (model, layer): returns (logits, topk, kvpe, output) + reference.
    NOTE: the full forward at seq5120 is slow on CPU (minutes) — prefer the device tests."""
    model, layer = request.param
    ref = load_reference(model, layer)
    if model == "glm_5_1":
        args, mla_cpu = _build_glm_cpu_reference(layer)
    else:
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


@pytest.mark.parametrize("_host", _CASES, ids=_CASE_IDS, indirect=True)
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


@pytest.mark.parametrize("_host", _CASES, ids=_CASE_IDS, indirect=True)
def test_kv_cache_host_vs_reference(_host, ds_kpe_layout):
    """Host KV cache vs the official capture: latent PCC + k_pe (--ds-kpe-layout). ~1-2 min."""
    ref_kv, kvpe = _host["ref"]["kv"], _host["kvpe"]
    _assert_kv(ref_kv, kvpe, tag="host", kpe_layout=ds_kpe_layout)


@pytest.mark.parametrize("_host", _CASES, ids=_CASE_IDS, indirect=True)
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


def _config_for(model: str):
    """ttMLA HF-style config: DS via the v3 resolver (identical to the config_only fixture);
    GLM hand-built (transformers can't load glm_moe_dsa)."""
    return _glm_hf_config() if model == "glm_5_1" else _resolve_config_only("deepseek_v3_d_p")


def _skip_unsupported(model: str, mesh_device) -> None:
    """Shared device guards: too-few-tokens-per-SP-chip (both); and GLM's tp ≤ 2 (64 q-heads,
    sparse_sdpa needs per-chip H = 64/tp ≥ 32 → tp>2 meshes are skipped for GLM)."""
    skip_if_seq_too_small_for_sp(SEQ_LEN, mesh_device)
    if model == "glm_5_1" and mesh_device.shape[1] > 2:
        pytest.skip(f"GLM sparse_sdpa needs per-chip H=64/tp≥32 → tp≤2 (mesh tp={mesh_device.shape[1]})")


def _make_mla(model, config, layer, mesh_device, is_chunked=False):
    if model == "glm_5_1":
        # GLM weights: load the CPU reference, remap its state_dict to ttMLA's [out,in] names.
        _, mla_cpu = _build_glm_cpu_reference(layer)
        sd = mla_cpu.state_dict()
        weights = {v3: sd[cpu].clone() for cpu, v3 in WEIGHT_NAME_MAP.items()}
        return ttMLA(
            config,
            weights,
            mesh_device,
            layer_idx=0,
            seq_len=SEQ_LEN,
            sp_axis=0,
            tp_axis=1,
            is_chunked=is_chunked,
            index_args=_glm_model_args(),
        )
    config.max_seq_len = SEQ_LEN  # rope-table length (same hack as v3 run_model / test_mla)
    args, _, weights, _ = build_cpu_reference(SEQ_LEN, layer=layer)
    assert_config_matches(config, args)
    return ttMLA(
        config, weights, mesh_device, layer_idx=0, seq_len=SEQ_LEN, sp_axis=0, tp_axis=1, is_chunked=is_chunked
    )


def _shard_idx_input(t, mesh_device):
    """Indexer input [1,1,S,hidden]: SP-shard the sequence (dim -2, mesh axis 0) and TP-shard hidden
    (dim -1, axis 1) — the same SP×TP layout the MLA forward uses. At SP=1 the SP shard is a no-op;
    at SP>1 each chip holds S/sp tokens, so _indexer_topk's SP all-gather rebuilds the global seq."""
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
    )


@parametrize_mesh_device()
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("model, layer", _CASES, ids=_CASE_IDS)
@pytest.mark.timeout(0)
def test_indexer_device_vs_reference(mesh_device, model, layer, device_params, monkeypatch):
    """Device indexer (stems + indexer_score + top-k) vs the official capture. ~2-3 min."""
    _skip_unsupported(model, mesh_device)
    ref = load_reference(model, layer)
    mla = _make_mla(model, _config_for(model), layer, mesh_device)
    x = ref["mla_in"].reshape(1, 1, SEQ_LEN, -1)  # [1,1,S,hidden]

    captured = {}
    # Capture the FULL head-summed logits that feed top-k. Under TP head-sharding (change B) the
    # head-sum is split across tp: ops.indexer_logits returns this chip's PARTIAL (H_idx/tp heads) and
    # _indexer_topk all-reduces the partials before top-k. So capture top-k's input, not indexer_logits'.
    orig_topk = ops.topk_indices

    def _capture_topk(logits, k):
        captured["logits"] = logits
        return orig_topk(logits, k)

    monkeypatch.setattr(ops, "topk_indices", _capture_topk)
    # _indexer_topk takes the per-chip (SP-local) sequence; it all-gathers back to the global glob.
    idx = mla._indexer_topk(_shard_idx_input(x, mesh_device), SEQ_LEN // mesh_device.shape[0])

    logits = ops._to_host(captured["logits"]).float()[0, 0]  # [S, S]; future cols -inf
    got_topk = ops._to_host(idx).long()[0, 0]  # [S, k]
    # PCC only over the causal region (device masks future to -inf; ref is pre-mask).
    tril = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
    _, pcc = comp_pcc(ref["logits"][tril].float(), logits[tril], 0)
    logger.info(f"[device {model} L{layer}] indexer logits PCC (causal region): {pcc}")
    rows = [0, 1, 100, 1000, 2047, 2048, 2049, 3000, 4096, SEQ_LEN - 1]
    ov = _topk_overlap(ref["topk"], got_topk, rows)
    mean = sum(ov.values()) / len(ov)
    logger.info(
        f"[device {model} L{layer}] topk overlap mean={mean:.4f} per-row={ {r: round(v,4) for r,v in ov.items()} }"
    )
    ttnn.synchronize_device(mesh_device)
    assert pcc >= LOGITS_PCC, f"indexer logits PCC {pcc} < {LOGITS_PCC}"
    assert mean >= TOPK_OVERLAP_MEAN, f"topk overlap mean {mean} < {TOPK_OVERLAP_MEAN}"
    assert min(ov.values()) >= TOPK_OVERLAP_ROW_MIN, f"topk overlap row min {min(ov.values())} < {TOPK_OVERLAP_ROW_MIN}"


def _run_device_forward(model, config, layer, mesh_device):
    """Single-shot ttMLA forward over the reference input; returns (ref, output[1,S,hidden], kvpe[S,576])."""
    ref = load_reference(model, layer)
    mla = _make_mla(model, config, layer, mesh_device)
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
@pytest.mark.parametrize("model, layer", _CASES, ids=_CASE_IDS)
@pytest.mark.timeout(0)
def test_kv_cache_device_vs_reference(mesh_device, model, layer, device_params, ds_kpe_layout):
    """Device KV cache vs the official capture: latent PCC + k_pe (--ds-kpe-layout). ~4-7 min."""
    _skip_unsupported(model, mesh_device)
    ref, _, kvpe = _run_device_forward(model, _config_for(model), layer, mesh_device)
    _assert_kv(ref["kv"], kvpe, tag=f"device {model} L{layer}", kpe_layout=ds_kpe_layout)
    ttnn.synchronize_device(mesh_device)


@parametrize_mesh_device()
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("model, layer", _CASES, ids=_CASE_IDS)
@pytest.mark.timeout(0)
def test_mla_output_device_vs_reference(mesh_device, model, layer, device_params):
    """Device MLA output vs the official capture. ~4-7 min (sparse_mla host fallback)."""
    _skip_unsupported(model, mesh_device)
    ref, out, _ = _run_device_forward(model, _config_for(model), layer, mesh_device)
    tt = out.unsqueeze(0)  # [1,1,S,hidden] -> compare as [1,S,hidden]
    ref_out = ref["mla_out"].reshape(1, SEQ_LEN, -1)
    for nm, sl in [("rows<2048", slice(0, 2048)), ("rows>=2048", slice(2048, SEQ_LEN))]:
        _, m = comp_pcc(ref_out[:, sl].float(), out[:, sl].float(), 0)
        logger.info(f"[device {model} L{layer}] band {nm}: {m}")
    _, pcc = comp_pcc(ref_out.float(), out.float(), 0)
    logger.info(f"[device {model} L{layer}] MLA output PCC: {pcc}")
    ttnn.synchronize_device(mesh_device)
    assert pcc >= OUTPUT_PCC, f"MLA output PCC {pcc} < {OUTPUT_PCC}"


# ════════════════════════════════════════════════════════════════════════════════
# GLM-5.1 Phase-1 — trace-internal consistency (no weights, no device). Runs today.
# Validates that the captured GLM bundle is well-formed and self-consistent (and that
# the loader is correct), independent of any model. GLM-only: the shape asserts use
# GLM dims. Filter: -k "glm_5_1 and not host and not device".
# ════════════════════════════════════════════════════════════════════════════════
_GLM_D = GLM51Config.EMB_SIZE  # 6144
_GLM_KV_LORA = GLM51Config.KV_LORA_RANK  # 512  (latent kv; also the V width)
_GLM_ROPE = GLM51Config.QK_ROPE_HEAD_DIM  # 64   (k_pe)
_GLM_KVPE = _GLM_KV_LORA + _GLM_ROPE  # 576
_GLM_INDEX_TOPK = GLM51Config.INDEX_TOPK  # 2048
# Boundary-probing rows around the index_topk=2048 cutover.
_PROBE_ROWS = [0, 1, 100, 1000, 2047, 2048, 2049, 3000, 4096, SEQ_LEN - 1]
_SELFTEST_OVERLAP_MIN = 0.99  # the GPU's own logits must reproduce its own top-k


@pytest.fixture(scope="module")
def trace(request):
    """One GLM layer's full reference bundle, loaded once per parametrized layer."""
    return load_reference("glm_5_1", request.param)


@pytest.mark.parametrize("trace", GLM_REF_LAYERS, ids=[f"glm_5_1-L{l}" for l in GLM_REF_LAYERS], indirect=True)
def test_trace_shapes_and_dtypes(trace):
    """Every stream has the GLM-config shape/dtype and contains no NaN/Inf."""
    assert trace["mla_in"].shape == (SEQ_LEN, _GLM_D), trace["mla_in"].shape
    assert trace["mla_out"].shape == (SEQ_LEN, _GLM_D), trace["mla_out"].shape
    assert trace["idx_in"].shape == (SEQ_LEN, _GLM_D), trace["idx_in"].shape
    assert trace["logits"].shape == (SEQ_LEN, SEQ_LEN), trace["logits"].shape
    assert trace["logits"].dtype == torch.float32, trace["logits"].dtype
    assert trace["topk"].shape == (SEQ_LEN, _GLM_INDEX_TOPK), trace["topk"].shape
    assert trace["topk"].dtype == torch.int32, trace["topk"].dtype
    assert trace["kv"].shape == (SEQ_LEN, _GLM_KVPE), trace["kv"].shape
    for k in ("mla_in", "mla_out", "idx_in", "kv"):
        assert torch.isfinite(trace[k].float()).all(), f"{k} has non-finite values"
    assert torch.isfinite(trace["logits"]).all(), "logits has non-finite values"


@pytest.mark.parametrize("trace", GLM_REF_LAYERS, ids=[f"glm_5_1-L{l}" for l in GLM_REF_LAYERS], indirect=True)
def test_indexer_input_equals_mla_input(trace):
    """The same post-input_layernorm hidden feeds both MLA and the indexer."""
    assert torch.equal(trace["idx_in"], trace["mla_in"])


@pytest.mark.parametrize("trace", GLM_REF_LAYERS, ids=[f"glm_5_1-L{l}" for l in GLM_REF_LAYERS], indirect=True)
def test_topk_causal_and_sentinel(trace):
    """dsa_topk_indices is already causal; -1 pads rows with < index_topk valid keys.
    For row r: exactly min(r+1, k) distinct non-neg indices, all in [0, r]; and for
    r < k the selected set is the entire causal prefix {0..r} (nothing to choose)."""
    tk = trace["topk"]
    for r in _PROBE_ROWS:
        nonneg = tk[r][tk[r] >= 0]
        assert nonneg.numel() == min(r + 1, _GLM_INDEX_TOPK), (r, nonneg.numel())
        assert int(nonneg.max()) <= r and int(nonneg.min()) >= 0, (r, int(nonneg.max()))
        assert nonneg.unique().numel() == nonneg.numel(), f"row {r} has duplicate indices"
        if r + 1 <= _GLM_INDEX_TOPK:
            assert set(nonneg.tolist()) == set(range(r + 1)), f"row {r} not the full causal prefix"
    rows = torch.arange(SEQ_LEN, dtype=tk.dtype).unsqueeze(1)
    valid = tk >= 0
    assert bool((tk[valid] < SEQ_LEN).all()), "a selected index is >= SEQ_LEN"
    assert bool(((tk <= rows) | ~valid).all()), "a selected index is in the future (> its row)"


@pytest.mark.parametrize("trace", GLM_REF_LAYERS, ids=[f"glm_5_1-L{l}" for l in GLM_REF_LAYERS], indirect=True)
def test_logits_reproduce_topk(trace):
    """Headline self-consistency: argtop_k(logits[t, :t+1]) == dsa_topk_indices[t].
    Proves the captured logits and top-k are a matched pair (and the loader is
    correct) — independent of any model weights. Set-overlap absorbs fp32 ties."""
    lg, tk = trace["logits"], trace["topk"]
    ov = {}
    for r in _PROBE_ROWS:
        k = min(r + 1, _GLM_INDEX_TOPK)
        got = set(torch.topk(lg[r, : r + 1].float(), k).indices.tolist())
        want = set(tk[r][tk[r] >= 0].tolist())
        ov[r] = len(want & got) / max(1, len(want))
    mean = sum(ov.values()) / len(ov)
    logger.info(
        f"[glm_5_1 trace] logits→topk self-overlap mean={mean:.5f} per-row={ {r: round(v, 4) for r, v in ov.items()} }"
    )
    assert min(ov.values()) >= _SELFTEST_OVERLAP_MIN, f"logits do not reproduce topk: {ov}"


@pytest.mark.parametrize("trace", GLM_REF_LAYERS, ids=[f"glm_5_1-L{l}" for l in GLM_REF_LAYERS], indirect=True)
def test_kv_split_sanity(trace):
    """kv_cache is [latent kv_lora=512 ‖ k_pe=64]; both halves finite. Logs the
    per-half magnitude (latent is small/compressed; k_pe carries the rope key)."""
    kv = trace["kv"]
    latent, kpe = kv[:, :_GLM_KV_LORA].float(), kv[:, _GLM_KV_LORA:].float()
    assert kpe.shape[1] == _GLM_ROPE, kpe.shape
    assert torch.isfinite(latent).all() and torch.isfinite(kpe).all()
    logger.info(f"[glm_5_1 trace] kv latent |mean|={latent.abs().mean():.4f}  k_pe |mean|={kpe.abs().mean():.4f}")
