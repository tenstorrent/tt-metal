# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GLM-5.1 (glm_moe_dsa) MLA / DSA-indexer / KV-cache vs the OFFICIAL GPU reference.

GLM-5.1 is a DeepSeek-V3.2-family model: HF ``glm_moe_dsa`` runs through vLLM's
``deepseek_v2.py`` ``is_v32`` DSA path, so its trace mirrors the DeepSeek-V3.2
layout and this harness mirrors ``test_vs_gpu_ref.py``. The truth is the recorded
output of the GPU stack, captured from a vLLM 5120-token prefill as safetensors
streams (``bit_sculpt/results/glm-51``; see that bundle's ``TRACE_MANUAL.md``).

────────────────────────────────────────────────────────────────────────────────
TENSOR MAP — vLLM/HF GLM trace  ↔  our deepseek_v32 implementation
────────────────────────────────────────────────────────────────────────────────
Activation streams (per layer L; each is a dir of ``rows_*.safetensors`` holding a
single tensor — the key inside need NOT match the folder name, so we read it blind):

  trace stream                       shape         our tensor
  module_io/mla_input_layer_L        (5120, 6144)  x into ttMLA.forward
                                                    (post input_layernorm; == indexer input)
  module_io/indexer_input_layer_L    (5120, 6144)  same tensor (verified == mla_input)
  module_io/mla_output_layer_L       (5120, 6144)  ttMLA.forward output
                                                    (post o_proj, BEFORE the residual add)
  dsa/indexer_logits_layer_L         (5120, 5120)  indexer index_score, PRE causal mask
                                      fp32          (upper triangle garbage → use logits[t,:t+1])
                                                    [captured for L ∈ {0,30,60,77} only]
  dsa/dsa_topk_indices_layer_L       (5120, 2048)  indexer topk_indices
                                      int32         (already causal; -1 = unfilled pad, t<2048)
  kv_cache/layer_L                   (5120, 576)    kvpe latent = [kv_lora 512 ‖ k_pe 64]
                                      bf16          (file key = kv_post_transform_layer_L)

Weight names — GLM-5.1 HF == DeepSeek-V3.2 HF (verified from the GLM-5.1 weight
index), so ``reference_cpu/weights.py`` loads them unchanged with
``repo="zai-org/GLM-5.1"`` (bf16 repo → no fp8 dequant). HF→MLACPU map (_HF_TO_MLA):
  self_attn.q_a_proj / q_a_layernorm / q_b_proj, kv_a_proj_with_mqa / kv_a_layernorm /
  kv_b_proj, o_proj, indexer.{wq_b, wk, k_norm(.bias), weights_proj}.
Per-layer shards are fetched on demand (resolve_layer_shards) — never the full repo.

GLM-vs-DS config deltas that matter for a recompute (Phase 2/3, NOT the trace checks):
  dim 6144 (DS 7168), n_heads 64 (128), q_lora 2048 (1536), qk_nope 192 (128),
  v_head 256 (128), index_n_heads 32 (64); kv_lora 512 & qk_rope 64 UNCHANGED
  → kvpe still 576, the sparse_sdpa latent contract is unchanged.
  • Indexer RoPE is INTERLEAVED for GLM (config indexer_rope_interleave=true);
    DS IndexerCPU hardcodes interleaved=False — must be parametrized.
  • NO YaRN: softmax scale = qk_head_dim**-0.5 = 256**-0.5 (DS folds an mscale²).
  • sparse_sdpa needs per-chip H a multiple of 32 → 64 heads ⇒ tp ≤ 2.

────────────────────────────────────────────────────────────────────────────────
PHASES
  1 (THIS FILE — runs today, no weights, no device): trace-internal consistency +
    the reusable PCC/overlap/kv utilities. Validates the trace and the loader.
  2 (gated on a GLM CPU reference): recompute logits/topk/kv/output from mla_input
    and PCC vs the trace (host ceiling). Per-layer weights only.
  3 (gated on a GLM ttMLA): device-vs-trace on Blackhole.
────────────────────────────────────────────────────────────────────────────────
"""

import glob
import os

# Device-path deps (Phase 3). Import-safe now that transformers is pinned to 4.53.0.
import types
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config as _G
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.demos.deepseek_v32.reference_cpu.model import MLACPU, ModelArgs
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.reference_cpu.weights import initialize_weights
from models.demos.deepseek_v32.tt import ops
from models.demos.deepseek_v32.tt.mla import ttMLA

# Phase-1 (trace consistency) is fast and needs no weights/device; it carries both
# the `dev` (per-edit) and `gate` (CI) markers so either selector runs it. The
# Phase-2/3 tests below skip until their GLM reference / ttMLA exist.
pytestmark = [pytest.mark.gate, pytest.mark.dev]

# ── Dimensions (single source of truth: GLM51Config) ────────────────────────────
SEQ_LEN = 5120  # the reference prefill length, fixed by the capture
D = _G.EMB_SIZE  # 6144
KV_LORA = _G.KV_LORA_RANK  # 512  (latent kv; also the V width)
ROPE = _G.QK_ROPE_HEAD_DIM  # 64   (k_pe)
KVPE = KV_LORA + ROPE  # 576
INDEX_TOPK = _G.INDEX_TOPK  # 2048
QK_HEAD_DIM = _G.QK_NOPE_HEAD_DIM + _G.QK_ROPE_HEAD_DIM  # 256
# indexer_logits are captured only for these layers; topk/kv exist for all 0..77.
REF_LAYERS = [0, 30, 60, 77]
# Boundary-probing rows (around the index_topk=2048 cutover), as in test_vs_gpu_ref.py.
PROBE_ROWS = [0, 1, 100, 1000, 2047, 2048, 2049, 3000, 4096, SEQ_LEN - 1]

# Thresholds — used by Phase 2/3 (fp8/bf16 GPU ref vs our bf16 recompute). Mirrors
# test_vs_gpu_ref.py; tune during host-ceiling bring-up.
LOGITS_PCC = 0.95
TOPK_OVERLAP_MEAN = 0.95
TOPK_OVERLAP_ROW_MIN = 0.85
KV_LATENT_PCC = 0.99
KV_PE_L2_RELERR_MAX = 0.05  # frame-invariant: per-row ||k_pe|| must match
KV_PE_VLLM_PCC = 0.999  # element-wise, once our k_pe is reindexed to vLLM's half-split
OUTPUT_PCC = 0.98
# Phase-1 self-consistency: the GPU's own logits must reproduce its own top-k.
SELFTEST_OVERLAP_MIN = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over all elements — local, ttnn-free (so the host path
    runs standalone regardless of the ttnn/transformers conftest state)."""
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return max(-1.0, min(1.0, float((a @ b) / denom))) if float(denom) > 0 else 1.0


# ── Reference loading ───────────────────────────────────────────────────────────
def _ref_dir() -> Path:
    env = os.environ.get("GLM51_REF_DIR")
    if env:
        return Path(env)
    # In-tree bundle: <repo>/bit_sculpt/results/glm-51 (this file is at
    # <repo>/models/demos/deepseek_v32/tests/, so parents[4] == <repo>).
    return Path(__file__).resolve().parents[4] / "bit_sculpt" / "results" / "glm-51"


def load_stream(stream_dir: Path) -> torch.Tensor:
    """One (stream, layer) → full tensor; single-file or chunked rows_*.safetensors.
    The single key inside each chunk is read blind (it is not the folder name)."""
    from safetensors.torch import load_file

    files = sorted(glob.glob(f"{stream_dir}/rows_*.safetensors"))
    parts = []
    for f in files:
        d = load_file(f)
        (key,) = d.keys()
        parts.append(d[key])
    return torch.cat(parts, dim=0)


def load_reference(layer: int, need_logits: bool = True) -> dict:
    """Load every reference stream for `layer`, or skip if the data is absent."""
    root = _ref_dir()
    need = {
        "mla_in": root / "module_io" / f"mla_input_layer_{layer}",
        "mla_out": root / "module_io" / f"mla_output_layer_{layer}",
        "idx_in": root / "module_io" / f"indexer_input_layer_{layer}",
        "topk": root / "dsa" / f"dsa_topk_indices_layer_{layer}",
        "kv": root / "kv_cache" / f"layer_{layer}",
    }
    if need_logits:
        need["logits"] = root / "dsa" / f"indexer_logits_layer_{layer}"
    missing = [str(p) for p in need.values() if not glob.glob(f"{p}/rows_*.safetensors")]
    if missing:
        pytest.skip(f"GLM trace streams not found (set $GLM51_REF_DIR): missing {missing}")
    out = {k: load_stream(v) for k, v in need.items()}
    out["layer"] = layer
    return out


@pytest.fixture(scope="module")
def trace(request):
    """One layer's full reference bundle, loaded once per parametrized layer."""
    return load_reference(request.param)


# ── Shared comparison utilities (Phase 1 uses the overlap; Phase 2/3 reuse all) ──
def _topk_overlap(ref_topk: torch.Tensor, got_topk: torch.Tensor, rows) -> dict:
    """Per-row Jaccard-style overlap |ref ∩ got| / |ref| over the causal valid set.
    Both inputs use -1 (or any <0 / out-of-range / future) as 'no selection'."""
    out = {}
    for r in rows:
        want = set(ref_topk[r][ref_topk[r] >= 0].tolist())
        got = got_topk[r]
        got = set(got[(got >= 0) & (got < SEQ_LEN) & (got <= r)].tolist())
        out[r] = len(want & got) / max(1, len(want))
    return out


def _assert_logits_pcc(ref_logits: torch.Tensor, got_logits: torch.Tensor, tag: str) -> float:
    """PCC over the causal (tril) region only — the device/ref masks future cols,
    the trace's logits are pre-mask, so the upper triangle is not comparable."""
    tril = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
    pcc = _pcc(ref_logits[tril], got_logits[tril])
    logger.info(f"[{tag}] indexer logits PCC (causal region): {pcc:.5f}")
    return pcc


def _assert_kv(ref_kv: torch.Tensor, kvpe: torch.Tensor, tag: str, kpe_layout: str = "interleaved"):
    """Shared KV-cache assertion. Always asserts the latent-512 PCC. For the k_pe
    rope half (last 64) the check depends on kpe_layout: 'interleaved' (default)
    asserts a frame-invariant per-row L2 match (our interleaved frame vs vLLM's
    half-split differ element-wise but preserve ||k_pe|| and q·k); 'vllm' reindexes
    our k_pe to vLLM's half-split layout and asserts element-wise PCC."""
    lat = _pcc(ref_kv[:, :KV_LORA], kvpe[:, :KV_LORA])
    assert lat >= KV_LATENT_PCC, f"KV latent PCC {lat} < {KV_LATENT_PCC}"
    pe_ref, pe_got = ref_kv[:, KV_LORA:].float(), kvpe[:, KV_LORA:].float()
    if kpe_layout == "vllm":
        from models.demos.deepseek_v32.tt.mla import interleaved_to_halfsplit_perm

        pe_got = pe_got[:, interleaved_to_halfsplit_perm(pe_got.shape[1])]
        pe_pcc = _pcc(pe_ref, pe_got)
        logger.info(f"[{tag}] KV latent PCC={lat:.5f}  k_pe PCC (vLLM half-split)={pe_pcc:.5f}")
        assert pe_pcc >= KV_PE_VLLM_PCC, f"k_pe (vLLM-layout) PCC {pe_pcc} < {KV_PE_VLLM_PCC}"
    else:
        pe_pcc = _pcc(pe_ref, pe_got)
        l2 = ((pe_got.norm(dim=1) - pe_ref.norm(dim=1)).abs() / (pe_ref.norm(dim=1) + 1e-3)).mean().item()
        logger.info(f"[{tag}] KV latent PCC={lat}  k_pe PCC={pe_pcc} (frame diag)  k_pe L2 rel-err={l2:.4f}")
        assert l2 <= KV_PE_L2_RELERR_MAX, f"k_pe L2 rel-err {l2} > {KV_PE_L2_RELERR_MAX}"


# ════════════════════════════════════════════════════════════════════════════════
# Phase 1 — trace-internal consistency (no weights, no device). Runs today.
# These validate that the captured bundle is well-formed and self-consistent, and
# exercise the loader + the comparison utilities that Phases 2/3 build on.
# ════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("trace", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_trace_shapes_and_dtypes(trace):
    """Every stream has the GLM-config shape/dtype and contains no NaN/Inf."""
    assert trace["mla_in"].shape == (SEQ_LEN, D), trace["mla_in"].shape
    assert trace["mla_out"].shape == (SEQ_LEN, D), trace["mla_out"].shape
    assert trace["idx_in"].shape == (SEQ_LEN, D), trace["idx_in"].shape
    assert trace["logits"].shape == (SEQ_LEN, SEQ_LEN), trace["logits"].shape
    assert trace["logits"].dtype == torch.float32, trace["logits"].dtype
    assert trace["topk"].shape == (SEQ_LEN, INDEX_TOPK), trace["topk"].shape
    assert trace["topk"].dtype == torch.int32, trace["topk"].dtype
    assert trace["kv"].shape == (SEQ_LEN, KVPE), trace["kv"].shape
    for k in ("mla_in", "mla_out", "idx_in", "kv"):
        assert torch.isfinite(trace[k].float()).all(), f"{k} has non-finite values"
    assert torch.isfinite(trace["logits"]).all(), "logits has non-finite values"


@pytest.mark.parametrize("trace", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_indexer_input_equals_mla_input(trace):
    """The same post-input_layernorm hidden feeds both MLA and the indexer."""
    assert torch.equal(trace["idx_in"], trace["mla_in"])


@pytest.mark.parametrize("trace", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_topk_causal_and_sentinel(trace):
    """dsa_topk_indices is already causal; -1 pads rows with < index_topk valid keys.
    For row r: exactly min(r+1, k) distinct non-neg indices, all in [0, r]; and for
    r < k the selected set is the entire causal prefix {0..r} (nothing to choose)."""
    tk = trace["topk"]
    for r in PROBE_ROWS:
        nonneg = tk[r][tk[r] >= 0]
        assert nonneg.numel() == min(r + 1, INDEX_TOPK), (r, nonneg.numel())
        assert int(nonneg.max()) <= r and int(nonneg.min()) >= 0, (r, int(nonneg.max()))
        assert nonneg.unique().numel() == nonneg.numel(), f"row {r} has duplicate indices"
        if r + 1 <= INDEX_TOPK:
            assert set(nonneg.tolist()) == set(range(r + 1)), f"row {r} not the full causal prefix"
    # Global: no selected index exceeds its own row or the sequence length.
    rows = torch.arange(SEQ_LEN, dtype=tk.dtype).unsqueeze(1)
    valid = tk >= 0
    assert bool((tk[valid] < SEQ_LEN).all()), "a selected index is >= SEQ_LEN"
    assert bool(((tk <= rows) | ~valid).all()), "a selected index is in the future (> its row)"


@pytest.mark.parametrize("trace", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_logits_reproduce_topk(trace):
    """Headline self-consistency: argtop_k(logits[t, :t+1]) == dsa_topk_indices[t].
    Proves the captured logits and top-k are a matched pair (and our loader is
    correct) — independent of any model weights. Set-overlap absorbs fp32 ties at
    the selection boundary."""
    lg, tk = trace["logits"], trace["topk"]
    ov = {}
    for r in PROBE_ROWS:
        k = min(r + 1, INDEX_TOPK)
        got = set(torch.topk(lg[r, : r + 1].float(), k).indices.tolist())
        want = set(tk[r][tk[r] >= 0].tolist())
        ov[r] = len(want & got) / max(1, len(want))
    mean = sum(ov.values()) / len(ov)
    logger.info(
        f"[L{trace['layer']}] logits→topk self-overlap mean={mean:.5f} "
        f"per-row={ {r: round(v, 4) for r, v in ov.items()} }"
    )
    assert min(ov.values()) >= SELFTEST_OVERLAP_MIN, f"logits do not reproduce topk: {ov}"


@pytest.mark.parametrize("trace", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_kv_split_sanity(trace):
    """kv_cache is [latent kv_lora=512 ‖ k_pe=64]; both halves finite. Logs the
    per-half magnitude (latent is small/compressed; k_pe carries the rope key)."""
    kv = trace["kv"]
    latent, kpe = kv[:, :KV_LORA].float(), kv[:, KV_LORA:].float()
    assert kpe.shape[1] == ROPE, kpe.shape
    assert torch.isfinite(latent).all() and torch.isfinite(kpe).all()
    logger.info(f"[L{trace['layer']}] kv latent |mean|={latent.abs().mean():.4f}  k_pe |mean|={kpe.abs().mean():.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# Phase 2 — host ceiling (recompute from per-layer GLM weights vs the trace).
# Gated on a GLM CPU reference. Recipe (reuses reference_cpu wholesale):
#   • GLM ModelArgs from GLM51Config: dim 6144, n_heads 64, q_lora_rank 2048,
#     qk_nope_head_dim 192, qk_rope_head_dim 64, v_head_dim 256, kv_lora_rank 512,
#     index_n_heads 32, index_head_dim 128, index_topk 2048.
#   • reference_cpu MLACPU/IndexerCPU parametrized for GLM:
#       - IndexerCPU RoPE interleaved=True  (GLM) — currently hardcoded False (model.py:235/246).
#       - MLACPU softmax_scale = qk_head_dim**-0.5  (GLM has NO YaRN → drop the mscale² branch).
#   • Weights: reference_cpu.weights.initialize_weights(mla, layer=L,
#     repo="zai-org/GLM-5.1") — GLM HF names == DS, bf16 (no fp8 dequant); only that
#     layer's shard(s) are downloaded (resolve_layer_shards), never the full repo.
# Then: feed trace["mla_in"], compare indexer logits/topk (_topk_overlap, _assert_logits_pcc),
# kv (_assert_kv), and mla_out (OUTPUT_PCC) — exactly the assertions Phase 1 already wired.
# ════════════════════════════════════════════════════════════════════════════════
GLM_REPO = os.environ.get("GLM51_REPO", "zai-org/GLM-5.1")  # bf16 master; "-FP8" also works (loader dequants)


def glm_model_args() -> ModelArgs:
    """GLM-5.1 ModelArgs. max_seq_len == original_seq_len disables BOTH the YaRN
    freq scaling and the mscale² softmax correction → plain rope, scale=qk_head_dim**-0.5.
    index_rope_interleave=True: GLM's indexer RoPE is interleaved (DS's is not)."""
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=8192,
        original_seq_len=8192,
        dim=D,
        n_heads=_G.NUM_ATTENTION_HEADS,
        q_lora_rank=_G.Q_LORA_RANK,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=_G.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=ROPE,
        v_head_dim=_G.V_HEAD_DIM,
        rope_theta=_G.ROPE_THETA,
        rope_factor=1.0,
        mscale=1.0,
        index_n_heads=_G.INDEX_N_HEADS,
        index_head_dim=_G.INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        index_rope_interleave=True,
    )


def build_glm_cpu_reference(layer: int, repo: str = None):
    """GLM MLACPU with layer-`layer` pretrained weights from `repo` (per-layer shards
    only; bf16 passes through, fp8 is dequantized). Functional path: use_fp8_path=False
    / simulate_fp8=False (Hadamard + fp8 dropped, KV stored bf16)."""
    args = glm_model_args()
    mla = MLACPU(args, simulate_fp8=False).eval()
    mla.indexer.use_fp8_path = False
    initialize_weights(mla, layer=layer, repo=repo or GLM_REPO)
    return args, mla


@pytest.fixture(scope="module")
def host(request):
    """Run GLM MLACPU once per layer over the trace's mla_input; returns the indexer
    logits/top-k, the kvpe cache and the MLA output alongside the reference streams.
    NOTE: the full mla.forward at seq5120 is slow on CPU (minutes) — prefer the device
    tests, or select the host tests explicitly when you specifically want the host ceiling."""
    layer = request.param
    ref = load_reference(layer)
    args, mla = build_glm_cpu_reference(layer)
    x = ref["mla_in"].unsqueeze(0)  # [1, S, D] bf16
    freqs = precompute_freqs_cis(args)[:SEQ_LEN]
    mask = torch.full((SEQ_LEN, SEQ_LEN), float("-inf")).triu_(1)
    with torch.no_grad():
        qr = mla.q_norm(mla.wq_a(x))
        topk, logits = mla.indexer(x, qr, 0, freqs, mask)
        out = mla.forward(x, 0, freqs, mask)
    kvpe = torch.cat([mla.kv_cache[0, :SEQ_LEN], mla.pe_cache[0, :SEQ_LEN]], dim=-1)  # [S, 576]
    return dict(ref=ref, logits=logits[0], topk=topk[0], kvpe=kvpe, out=out[0], layer=layer)


@pytest.mark.parametrize("host", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_indexer_host_vs_reference(host):
    """Host indexer logits (PCC over the causal region) + top-k overlap vs the GLM capture."""
    pcc = _assert_logits_pcc(host["ref"]["logits"], host["logits"], f"host L{host['layer']}")
    ov = _topk_overlap(host["ref"]["topk"], host["topk"], PROBE_ROWS)
    mean = sum(ov.values()) / len(ov)
    logger.info(
        f"[host L{host['layer']}] topk overlap mean={mean:.4f} per-row={ {r: round(v,4) for r,v in ov.items()} }"
    )
    assert pcc >= LOGITS_PCC, f"indexer logits PCC {pcc} < {LOGITS_PCC}"
    assert mean >= TOPK_OVERLAP_MEAN, f"topk overlap mean {mean} < {TOPK_OVERLAP_MEAN}"
    assert min(ov.values()) >= TOPK_OVERLAP_ROW_MIN, f"topk overlap row min {min(ov.values())} < {TOPK_OVERLAP_ROW_MIN}"


@pytest.mark.parametrize("host", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_kv_cache_host_vs_reference(host, ds_kpe_layout):
    """Host KV cache vs the GLM capture: latent PCC + k_pe (--ds-kpe-layout)."""
    _assert_kv(host["ref"]["kv"], host["kvpe"], tag=f"host L{host['layer']}", kpe_layout=ds_kpe_layout)


@pytest.mark.parametrize("host", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS], indirect=True)
def test_mla_output_host_vs_reference(host):
    """Host MLA output vs the GLM capture."""
    pcc = _pcc(host["ref"]["mla_out"], host["out"])
    logger.info(f"[host L{host['layer']}] MLA output PCC: {pcc:.5f}")
    assert pcc >= OUTPUT_PCC, f"MLA output PCC {pcc} < {OUTPUT_PCC}"


# ════════════════════════════════════════════════════════════════════════════════
# Phase 3 — device (GLM ttMLA on Blackhole vs the trace). Gated on a GLM ttMLA.
# NOTE: sparse_sdpa requires per-chip H a multiple of 32; GLM has 64 heads, so the
# mesh must use tp ∈ {1, 2} (tp=4 → 16 heads/chip is invalid) until the op relaxes it.
# ════════════════════════════════════════════════════════════════════════════════
# MLACPU param name -> v3 ttMLA weights-dict name (same [out,in] layout; mirrors test_mla).
GLM_WEIGHT_NAME_MAP = {
    "wq_a.weight": "q_a_proj.weight",
    "q_norm.weight": "q_a_layernorm.weight",
    "wq_b.weight": "q_b_proj.weight",
    "wkv_a.weight": "kv_a_proj_with_mqa.weight",
    "kv_norm.weight": "kv_a_layernorm.weight",
    "wkv_b.weight": "kv_b_proj.weight",
    "wo.weight": "o_proj.weight",
    "indexer.wq_b.weight": "indexer.wq_b.weight",
    "indexer.wk.weight": "indexer.wk.weight",
    "indexer.k_norm.weight": "indexer.k_norm.weight",
    "indexer.k_norm.bias": "indexer.k_norm_bias.weight",
    "indexer.weights_proj.weight": "indexer.weights_proj.weight",
}

# tp=2 always (64 heads / 2 = 32 = sparse_sdpa per-chip minimum); sp ∈ {1, 2, 4}.
# (sp, tp): 1x2=2 chips, 2x2=4 chips, 4x2=8 chips (loudbox). SEQ_LEN must divide by sp (5120/4=1280, tile-aligned).
_GLM_MESHES = [(1, 2), (2, 2), (4, 2)]
_GLM_MESH_IDS = [f"{sp}x{tp}" for sp, tp in _GLM_MESHES]
_DEVICE_PARAMS = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    }
]


def glm_hf_config(max_seq: int = 8192):
    """HF-attribute-style config the v32 ttMLA reads (GLM dims, no YaRN). Built directly —
    AutoConfig can't load `glm_moe_dsa`. rope_scaling.factor=1.0 → no mscale (scale=256**-0.5)
    and DeepseekV3YarnRotaryEmbedding(factor=1) → plain RoPE at θ=1e6."""
    return types.SimpleNamespace(
        hidden_size=D,
        num_attention_heads=_G.NUM_ATTENTION_HEADS,
        num_key_value_heads=_G.NUM_ATTENTION_HEADS,
        kv_lora_rank=KV_LORA,
        q_lora_rank=_G.Q_LORA_RANK,
        qk_nope_head_dim=_G.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=ROPE,
        v_head_dim=_G.V_HEAD_DIM,
        rms_norm_eps=_G.RMS_NORM_EPS,
        max_seq_len=max_seq,
        rope_theta=float(_G.ROPE_THETA),
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


def _glm_device_mla(mesh_device, layer):
    """v32 ttMLA on `mesh_device` with GLM config + layer-`layer` GLM weights (interleaved
    indexer RoPE, 32 indexer heads via index_args=glm_model_args())."""
    _, mla_cpu = build_glm_cpu_reference(layer)
    sd = mla_cpu.state_dict()
    weights = {v3: sd[cpu].clone() for cpu, v3 in GLM_WEIGHT_NAME_MAP.items()}
    config = glm_hf_config()
    mla = ttMLA(
        config, weights, mesh_device, layer_idx=0, seq_len=SEQ_LEN, sp_axis=0, tp_axis=1, index_args=glm_model_args()
    )
    return mla, config


def _shard_sp_tp(t, mesh_device):
    """Input [1,1,S,hidden] sharded SP on seq (dim2) + TP on hidden (dim3) — matches ttMLA.forward.
    The indexer/forward SP-all-gather the seq internally, so callers pass the per-SP-shard seq
    length (SEQ_LEN // sp_factor), NOT the global seq."""
    shard_dims = [None, None]
    shard_dims[1], shard_dims[0] = -1, -2  # tp_axis=1 → hidden(-1), sp_axis=0 → seq(-2)
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )


def _glm_device_forward(mesh_device, layer):
    """Single-shot ttMLA forward over the reference input; returns (ref, output[S,hidden], kvpe[S,576])."""
    ref = load_reference(layer)
    mla, config = _glm_device_mla(mesh_device, layer)
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
    ).to(torch.bfloat16)[0]
    kvpe_t = ttnn.to_torch(
        kvpe_cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)[0, 0, :SEQ_LEN]
    return ref, out_t, kvpe_t


@pytest.mark.parametrize("mesh_device", _GLM_MESHES, ids=_GLM_MESH_IDS, indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("layer", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS])
@pytest.mark.timeout(0)
def test_indexer_device_vs_reference(mesh_device, layer, device_params, monkeypatch):
    """Device indexer (stems + interleaved RoPE + indexer_score + top-k) vs the GLM capture."""
    ref = load_reference(layer)
    mla, _ = _glm_device_mla(mesh_device, layer)
    captured = {}
    orig = ops.indexer_logits
    monkeypatch.setattr(ops, "indexer_logits", lambda *a, **k: captured.setdefault("logits", orig(*a, **k)))
    sp = mesh_device.shape[0]  # sp_axis = 0
    xt = _shard_sp_tp(ref["mla_in"].reshape(1, 1, SEQ_LEN, -1), mesh_device)
    idx = mla._indexer_topk(xt, SEQ_LEN // sp)  # per-SP-shard seq; _indexer_topk all-gathers to full
    logits = ops._to_host(captured["logits"]).float()[0, 0]  # [S, S]; future cols -inf
    got_topk = ops._to_host(idx).long()[0, 0]  # [S, k]
    pcc = _assert_logits_pcc(ref["logits"], logits, f"device L{layer}")
    ov = _topk_overlap(ref["topk"], got_topk, PROBE_ROWS)
    mean = sum(ov.values()) / len(ov)
    logger.info(f"[device L{layer}] topk overlap mean={mean:.4f} per-row={ {r: round(v,4) for r,v in ov.items()} }")
    ttnn.synchronize_device(mesh_device)
    assert pcc >= LOGITS_PCC, f"indexer logits PCC {pcc} < {LOGITS_PCC}"
    assert mean >= TOPK_OVERLAP_MEAN, f"topk overlap mean {mean} < {TOPK_OVERLAP_MEAN}"
    assert min(ov.values()) >= TOPK_OVERLAP_ROW_MIN, f"topk overlap row min {min(ov.values())} < {TOPK_OVERLAP_ROW_MIN}"


@pytest.mark.parametrize("mesh_device", _GLM_MESHES, ids=_GLM_MESH_IDS, indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("layer", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS])
@pytest.mark.timeout(0)
def test_kv_cache_device_vs_reference(mesh_device, layer, device_params, ds_kpe_layout):
    """Device KV cache vs the GLM capture: latent PCC + k_pe (--ds-kpe-layout)."""
    ref, _, kvpe = _glm_device_forward(mesh_device, layer)
    _assert_kv(ref["kv"], kvpe, tag=f"device L{layer}", kpe_layout=ds_kpe_layout)
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("mesh_device", _GLM_MESHES, ids=_GLM_MESH_IDS, indirect=True)
@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, ids=["line"], indirect=True)
@pytest.mark.parametrize("layer", REF_LAYERS, ids=[f"L{l}" for l in REF_LAYERS])
@pytest.mark.timeout(0)
def test_mla_output_device_vs_reference(mesh_device, layer, device_params):
    """Device MLA output vs the GLM capture (banded dense<2048 / sparse>=2048 diagnostic)."""
    ref, out, _ = _glm_device_forward(mesh_device, layer)
    ref_out = ref["mla_out"].reshape(1, SEQ_LEN, -1)
    out = out.reshape(1, SEQ_LEN, -1)
    for nm, sl in [("rows<2048", slice(0, 2048)), ("rows>=2048", slice(2048, SEQ_LEN))]:
        logger.info(f"[device L{layer}] band {nm}: {_pcc(ref_out[:, sl], out[:, sl]):.4f}")
    pcc = _pcc(ref_out, out)
    logger.info(f"[device L{layer}] MLA output PCC: {pcc:.4f}")
    ttnn.synchronize_device(mesh_device)
    assert pcc >= OUTPUT_PCC, f"MLA output PCC {pcc} < {OUTPUT_PCC}"


# ════════════════════════════════════════════════════════════════════════════════
# Standalone host-ceiling runner — bypasses pytest/conftest (which currently fails
# to import due to a transformers version skew). Compares the GLM CPU reference vs
# the trace for the given layers; needs per-layer weights (downloaded on first use).
#   python models/demos/deepseek_v32/tests/test_vs_gpu_ref_glm.py --layers 0
#   python models/demos/deepseek_v32/tests/test_vs_gpu_ref_glm.py --layers 0,30,60,77 --full
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(map(str, REF_LAYERS)))
    ap.add_argument("--full", action="store_true", help="also run MLA forward (kv + output; ~7GB RAM/layer)")
    ap.add_argument("--repo", default=GLM_REPO)
    cli = ap.parse_args()
    for L in [int(x) for x in cli.layers.split(",")]:
        ref = load_reference(L)
        args, mla = build_glm_cpu_reference(L, repo=cli.repo)
        x = ref["mla_in"].unsqueeze(0)
        freqs = precompute_freqs_cis(args)[:SEQ_LEN]
        mask = torch.full((SEQ_LEN, SEQ_LEN), float("-inf")).triu_(1)
        with torch.no_grad():
            qr = mla.q_norm(mla.wq_a(x))
            topk, logits = mla.indexer(x, qr, 0, freqs, mask)
        pcc = _assert_logits_pcc(ref["logits"], logits[0], f"L{L}")
        ov = _topk_overlap(ref["topk"], topk[0], PROBE_ROWS)
        line = f"L{L}: logits PCC(causal)={pcc:.4f}  topk overlap mean={sum(ov.values())/len(ov):.4f} min={min(ov.values()):.4f}"
        if cli.full:
            with torch.no_grad():
                out = mla.forward(x, 0, freqs, mask)
            kvpe = torch.cat([mla.kv_cache[0, :SEQ_LEN], mla.pe_cache[0, :SEQ_LEN]], dim=-1)
            line += (
                f"  | MLA out PCC={_pcc(ref['mla_out'], out[0]):.4f}"
                f"  kv latent PCC={_pcc(ref['kv'][:, :KV_LORA], kvpe[:, :KV_LORA]):.4f}"
            )
        print(line, flush=True)
