# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config, weight remap, checkpoint acquisition and memory budgets (#47461, #47487).

The config reconciliation reads the gemma4 ``configs/*/config.json`` shipped in the repo
(always present, no checkpoint/HW needed) and asserts ``config.py`` stays in sync — the
guard the #47461 weight-mapping pass relies on to catch renamed/missing keys. The weight
remap has two tiers: index-only (needs only the two ~100 KB ``model.safetensors.index.json``
files) and checkpoint (needs the gated 51 GB DiffusionGemma weights); both discover files
from the HF cache and skip cleanly when absent.
"""

import glob
import json
import math
import os
import time

import pytest
import torch
from loguru import logger

from models.experimental.diffusion_gemma import checkpoint as ckpt
from models.experimental.diffusion_gemma.config import DiffusionConfig, DiffusionGemmaConfig, TextConfig
from models.experimental.diffusion_gemma.memory_budget import estimate_canvas_kv_scratch_bytes
from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning
from models.experimental.diffusion_gemma.weight_mapping import (
    SELF_CONDITIONING_PREFIX,
    classify_keys,
    expected_self_conditioning_shapes,
    gemma4_key_for,
    remap_state_dict,
)

# Repo root: honor TT_METAL_HOME, else derive from this file's location so the
# config-drift guard below actually runs wherever the repo is checked out (a
# personal-home fallback silently skips the guard when TT_METAL_HOME is unset).
REPO = os.environ.get("TT_METAL_HOME") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
CFG_26B = os.path.join(REPO, "models/demos/gemma4/configs/gemma-4-26B-A4B-it/config.json")
CFG_12B = os.path.join(REPO, "models/demos/gemma4/configs/gemma-4-12B-it/config.json")

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
DG_DIR = "models--google--diffusiongemma-26B-A4B-it"
G4_DIR = "models--google--gemma-4-26B-A4B-it"


# --- config defaults + validation -------------------------------------------
def test_max_context_is_256k():
    assert DiffusionGemmaConfig().max_context == 262144  # 256 canvas * 1024 blocks


def test_full_attention_layer_count_matches_pattern():
    tc = TextConfig()
    full_layers = [i for i in range(tc.num_hidden_layers) if (i + 1) % tc.sliding_window_pattern == 0]
    assert full_layers == [5, 11, 17, 23, 29]  # matches 26B-A4B layer_types


@pytest.mark.parametrize(
    "build,match",
    [
        pytest.param(
            lambda: DiffusionGemmaConfig(
                text=TextConfig(canvas_length=128), diffusion=DiffusionConfig(canvas_length=256)
            ),
            "text.canvas_length .* must match diffusion.canvas_length",
            id="canvas_length_mismatch",
        ),
        pytest.param(
            lambda: DiffusionConfig(max_denoise_steps=0),
            "max_denoise_steps must be positive",
            id="max_denoise_steps_zero",
        ),
        pytest.param(
            lambda: DiffusionConfig(max_denoise_steps=-1),
            "max_denoise_steps must be positive",
            id="max_denoise_steps_negative",
        ),
    ],
)
def test_config_rejects_invalid_values(build, match, expect_error):
    with expect_error(ValueError, match=match):
        build()


# --- reconciliation against the in-repo gemma-4 configs ---------------------
@pytest.mark.skipif(not os.path.exists(CFG_26B), reason="in-repo 26B-A4B config not found")
def test_from_hf_config_matches_target_26b():
    hf = json.load(open(CFG_26B))
    tc = TextConfig.from_hf_config(hf)

    # parsed from the real target config.json
    assert tc.num_hidden_layers == 30
    assert tc.hidden_size == 2816
    assert (tc.num_attention_heads, tc.num_key_value_heads) == (16, 8)
    assert tc.head_dim == 256
    assert tc.vocab_size == 262144
    assert tc.intermediate_size == 2112
    assert tc.sliding_window == 1024
    assert tc.sliding_window_pattern == 6  # derived from layer_types
    assert tc.rms_norm_eps == 1e-6
    assert tc.final_logit_softcapping == 30.0
    assert tc.num_experts == 128
    assert tc.num_experts_per_tok == 8
    assert tc.num_shared_experts == 1
    assert tc.moe_intermediate_size == 704
    assert (tc.num_global_key_value_heads, tc.global_head_dim) == (2, 512)
    # NB: this reads the gemma-4-26B-A4B *base* config (the backbone we reuse), which
    # carries attention_k_eq_v=True. The DiffusionGemma config itself omits the key
    # (DG derives K=V tying from layer geometry); we validate the backbone value here.
    assert tc.attention_k_eq_v is True
    assert tc.hidden_activation == "gelu_pytorch_tanh"


@pytest.mark.skipif(not os.path.exists(CFG_26B), reason="in-repo 26B-A4B config not found")
def test_hand_written_defaults_stay_in_sync_with_config_json():
    # config.py defaults must equal what we parse from the shipped config.json,
    # so the two never silently diverge.
    hf = json.load(open(CFG_26B))
    parsed = TextConfig.from_hf_config(hf)
    defaults = TextConfig()
    for field in [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "intermediate_size",
        "sliding_window",
        "sliding_window_pattern",
        "rms_norm_eps",
        "final_logit_softcapping",
        "num_experts",
        "num_experts_per_tok",
        "num_shared_experts",
        "moe_intermediate_size",
        "num_global_key_value_heads",
        "global_head_dim",
        "attention_k_eq_v",
        "hidden_activation",
    ]:
        assert getattr(parsed, field) == getattr(defaults, field), f"{field} drift vs config.json"


@pytest.mark.skipif(not os.path.exists(CFG_12B), reason="in-repo 12B config not found")
def test_from_hf_config_preserves_dense_moe_nulls():
    hf = json.load(open(CFG_12B))
    tc = TextConfig.from_hf_config(hf)

    assert tc.num_experts is None
    assert tc.num_experts_per_tok is None
    assert tc.num_shared_experts is None
    assert tc.moe_intermediate_size is None


# --- weight remap: unit (no files) ------------------------------------------
def test_gemma4_key_for_prefix_swap_and_self_cond_is_none():
    assert gemma4_key_for("model.decoder.layers.5.self_attn.q_proj.weight") == (
        "model.language_model.layers.5.self_attn.q_proj.weight"
    )
    assert gemma4_key_for("model.decoder.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert gemma4_key_for("model.decoder.norm.weight") == "model.language_model.norm.weight"
    assert gemma4_key_for("model.decoder.self_conditioning.gate_proj.weight") is None
    assert gemma4_key_for("model.encoder.vision_tower.std_bias") is None


def test_expected_self_conditioning_shapes_use_intermediate_not_moe():
    tc = TextConfig()
    shapes = expected_self_conditioning_shapes(tc.hidden_size, tc.intermediate_size)
    assert shapes["gate_proj.weight"] == (2112, 2816)  # intermediate_size, NOT moe_intermediate_size(704)
    assert shapes["down_proj.weight"] == (2816, 2112)
    assert shapes["pre_norm.weight"] == (2816,)


def test_unknown_keys_are_not_folded_into_ignored():
    res = classify_keys(
        [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.encoder.language_model.layers.0.layer_scalar",
            "model.unexpected.weight",
        ]
    )

    assert list(res.backbone) == ["model.decoder.layers.0.self_attn.q_proj.weight"]
    assert res.ignored == ["model.encoder.language_model.layers.0.layer_scalar"]
    assert res.unknown == ["model.unexpected.weight"]


def test_tied_lm_head_key_is_ignored():
    res = classify_keys(["lm_head.weight"])

    assert not res.backbone
    assert res.ignored == ["lm_head.weight"]
    assert not res.unknown


def test_remap_state_dict_rejects_unknown_keys(expect_error):
    with expect_error(ValueError, match="unknown DiffusionGemma checkpoint keys"):
        remap_state_dict({"model.unexpected.weight": object()})


# --- weight remap: real checkpoint index ------------------------------------
def _index_path(repo_dirname: str):
    hits = glob.glob(os.path.join(HF_CACHE, repo_dirname, "snapshots", "*", "model.safetensors.index.json"))
    return hits[0] if hits else None


def _weight_map(repo_dirname: str):
    p = _index_path(repo_dirname)
    if p is None:
        return None
    return json.load(open(p))["weight_map"]


@pytest.mark.skipif(_weight_map(DG_DIR) is None, reason="DiffusionGemma index.json not in HF cache")
def test_self_conditioning_keys_are_exactly_the_four():
    dg = _weight_map(DG_DIR)
    sc = sorted(k for k in dg if k.startswith(SELF_CONDITIONING_PREFIX))
    assert sc == sorted(
        SELF_CONDITIONING_PREFIX + w + ".weight" for w in ("down_proj", "gate_proj", "pre_norm", "up_proj")
    )


@pytest.mark.skipif(_weight_map(DG_DIR) is None, reason="DiffusionGemma index.json not in HF cache")
def test_classification_covers_all_keys_no_leftovers():
    dg = _weight_map(DG_DIR)
    res = classify_keys(dg.keys())
    # every key is accounted for exactly once
    assert res.num_backbone + len(res.self_conditioning) + len(res.ignored) + len(res.unknown) == len(dg)
    assert not res.unknown
    assert len(res.self_conditioning) == 4
    # the text backbone is the bulk (30 layers; MoE experts are PACKED into one
    # gate_up_proj + one down_proj tensor per layer, so the count is ~657, not ~1000).
    # Exact set-equality vs the gemma4 keyset is asserted in the test below.
    assert res.num_backbone > 600
    # ignored is encoder/vision/multimodal only
    assert all(k.startswith(("model.encoder.", "model.vision_tower.", "model.embed_vision.")) for k in res.ignored)


@pytest.mark.skipif(
    _weight_map(DG_DIR) is None or _weight_map(G4_DIR) is None,
    reason="need BOTH DiffusionGemma and gemma-4-26B-A4B-it index.json in HF cache",
)
def test_remapped_backbone_matches_gemma4_language_model_keyset():
    """The remapped DiffusionGemma backbone keys must EXACTLY equal the gemma4
    backbone keys the in-repo loader expects (model.language_model.*) — no missing,
    no renamed, no extra. This is the #47461 stage-2 weight-mapping gate."""
    dg = _weight_map(DG_DIR)
    g4 = _weight_map(G4_DIR)
    remapped = set(classify_keys(dg.keys()).backbone.values())
    g4_lm = {k for k in g4 if k.startswith("model.language_model.")}
    missing = g4_lm - remapped  # gemma4 expects but DiffusionGemma (remapped) lacks
    extra = remapped - g4_lm  # DiffusionGemma (remapped) has but gemma4 lacks
    assert not missing, f"DiffusionGemma backbone missing gemma4 keys: {sorted(missing)[:10]}"
    assert not extra, f"DiffusionGemma backbone has non-gemma4 keys after remap: {sorted(extra)[:10]}"


# --- weight remap: real self-conditioning tensors ---------------------------
def _open_self_cond_tensors():
    """Load the 4 self-conditioning tensors from the DiffusionGemma safetensors."""
    idx_path = _index_path(DG_DIR)
    if idx_path is None:
        return None
    snap = os.path.dirname(idx_path)
    wmap = json.load(open(idx_path))["weight_map"]
    sc_keys = [k for k in wmap if k.startswith(SELF_CONDITIONING_PREFIX)]
    shards = {wmap[k] for k in sc_keys}
    if not all(os.path.exists(os.path.join(snap, s)) for s in shards):
        return None  # index present but tensors not downloaded
    from safetensors import safe_open

    out = {}
    handles = {s: safe_open(os.path.join(snap, s), framework="pt") for s in shards}
    for k in sc_keys:
        out[k] = handles[wmap[k]].get_tensor(k)
    return out


@pytest.mark.skipif(_open_self_cond_tensors() is None, reason="DiffusionGemma checkpoint tensors not downloaded")
def test_real_self_conditioning_loads_and_matches_config_shapes():
    sc_state = _open_self_cond_tensors()
    _, sc_short, _ = remap_state_dict(sc_state)  # strips the prefix
    tc = TextConfig()
    expected = expected_self_conditioning_shapes(tc.hidden_size, tc.intermediate_size)
    for name, shape in expected.items():
        assert tuple(sc_short[name].shape) == shape, f"{name}: {tuple(sc_short[name].shape)} != {shape}"

    # load into the reference module and run a forward
    mod = SelfConditioning(tc.hidden_size, intermediate_size=tc.intermediate_size).to(torch.float32)
    mod.load_from_state_dict({k: v.float() for k, v in sc_short.items()})
    emb = torch.randn(1, 4, tc.hidden_size)
    signal = torch.randn(1, 4, tc.hidden_size)
    out = mod(emb, signal)
    assert out.shape == (1, 4, tc.hidden_size) and torch.isfinite(out).all()


# --- checkpoint download retry ----------------------------------------------
# ``resolve_checkpoint_dir`` is the first thing that runs when a server starts on a host
# with a cold weight cache, and it raises straight out of ``initialize_vllm_model`` ->
# ``load_model`` -> ``_init_executor``, where any exception is fatal to the vLLM
# EngineCore. On 2026-07-28 one ``HTTP 500`` from HuggingFace's CDN partway through a
# ~50 GB fetch killed a CI eval before any DiffusionGemma code ran, and the harness then
# held a QB2 runner for the rest of its 3600 s health-check timeout.
def test_transient_download_error_is_retried(monkeypatch, tmp_path):
    """A 5xx-style failure is retried, and the eventual success is returned."""
    calls = []

    def flaky(repo_id, **kwargs):
        calls.append(repo_id)
        if len(calls) < 3:
            raise ConnectionError("Network error: HTTP status server error (500 Internal Server Error)")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", flaky)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    out = ckpt._snapshot_download_with_retry("google/diffusiongemma-26B-A4B-it", attempts=5)

    assert out == str(tmp_path)
    assert len(calls) == 3, "should have retried twice before succeeding"


def test_exhausted_retries_raise_with_the_last_error(monkeypatch, expect_error):
    """After the last attempt it fails loudly, naming what actually went wrong."""

    def always_500(repo_id, **kwargs):
        raise ConnectionError("HTTP status server error (500 Internal Server Error)")

    monkeypatch.setattr("huggingface_hub.snapshot_download", always_500)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with expect_error(RuntimeError, match="failed after 3 attempts"):
        ckpt._snapshot_download_with_retry("google/diffusiongemma-26B-A4B-it", attempts=3)


def test_missing_repo_fails_immediately(monkeypatch, expect_error):
    """A wrong or gated repo id is not transient -- retrying it just wastes the same hour slowly."""
    from huggingface_hub.utils import RepositoryNotFoundError

    calls = []
    # Built with __new__: HfHubHTTPError requires a `response` kwarg on this huggingface_hub, and a
    # TypeError from constructing the exception would itself look transient and be retried -- which
    # is exactly what happened the first time this test was written.
    missing_error = RepositoryNotFoundError.__new__(RepositoryNotFoundError)

    def missing(repo_id, **kwargs):
        calls.append(repo_id)
        raise missing_error

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with expect_error(RepositoryNotFoundError, match=""):
        ckpt._snapshot_download_with_retry("google/does-not-exist", attempts=5)
    assert len(calls) == 1, "a missing repo must not be retried"


# --- canvas KV scratch budget -----------------------------------------------
def test_qb2_canvas_kv_scratch_estimate_matches_gemma4_tp4_shapes():
    est = estimate_canvas_kv_scratch_bytes(tp=4, batch_size=1, bytes_per_elem=2)

    assert est.sliding_bytes == int(12.5 * 2**20)
    assert est.full_attention_bytes == int(2.5 * 2**20)
    assert est.total_bytes == 15 * 2**20


# --- QB2 per-chip DRAM budget (device) --------------------------------------
# #47487 — measures per-chip DRAM (weights, and weights+paged-KV up to 256K) via
# ``ttnn.get_memory_view``, to document the QB2 memory budget + batch ceiling. The paged
# KV cache is allocated eagerly at build, so the weights+KV budget is captured WITHOUT a
# prefill. Env-parameterized (one build per pytest process):
#
#     PROBE_KV=0|1   PROBE_CTX=<max_seq_len>   PROBE_BATCH=<batch>
#
#     DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 HF_MODEL=<gemma-4-26B-A4B-it checkpoint> \
#       PROBE_KV=1 PROBE_CTX=262144 PROBE_BATCH=1 \
#       pytest models/experimental/diffusion_gemma/tests/test_config.py -k 1x4 -q -s
PROBE_KV = os.getenv("PROBE_KV", "1") == "1"
PROBE_CTX = int(os.getenv("PROBE_CTX", "262144"))
PROBE_BATCH = int(os.getenv("PROBE_BATCH", "1"))
# No personal-path default: require HF_MODEL (a 26B-A4B checkpoint dir); skip when
# unset so the harness is portable instead of pointing at a developer's home dir.
MODEL_PATH = os.getenv("HF_MODEL")
PROBE_PREFILL = os.getenv("PROBE_PREFILL", "0") == "1"
PREFILL_LEN = int(os.getenv("PROBE_PREFILL_LEN", str(PROBE_CTX)))
BLOCK = 64
G = 2**30


def _mesh_parametrize():
    """Apply the shared mesh parametrization without importing ttnn at module scope.

    ``parametrize_mesh_with_fabric()`` calls ``ttnn.get_num_devices()`` while the decorator
    is being applied, and it re-raises any enumeration failure whose message it does not
    recognize as an unhealthy runner. At module scope that turns a device fault into a
    collection error for the whole file, taking the pure-host tests above down with it.
    """
    try:
        from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

        return parametrize_mesh_with_fabric()
    except Exception as e:
        return pytest.mark.skip(reason=f"mesh parametrization unavailable (no usable device): {e}")


def _dram(mesh_device, label):
    import ttnn

    ttnn.synchronize_device(mesh_device)
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    used = v.num_banks * v.total_bytes_allocated_per_bank
    total = v.num_banks * v.total_bytes_per_bank
    free = v.num_banks * v.total_bytes_free_per_bank
    logger.info(
        f"[{label}] per-chip DRAM: used={used/G:.3f} GiB  free={free/G:.3f} GiB  "
        f"usable_total={total/G:.3f} GiB  banks={v.num_banks}"
    )
    return used / G, total / G


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (QB2, MESH_DEVICE=P150x4)",
)
@_mesh_parametrize()
def test_qb2_dram_budget(mesh_device, reset_seeds, request):
    import ttnn

    from models.demos.gemma4.tt.common import create_tt_model
    from models.tt_transformers.tt.common import PagedAttentionConfig

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp < 2:
        pytest.skip("26B-A4B backbone needs TP>=2 (use -k 1x4 on QB2)")
    if MODEL_PATH is None:
        pytest.skip("set HF_MODEL to a 26B-A4B checkpoint dir (no personal-path default)")

    base_used, total = _dram(mesh_device, "baseline (empty)")
    pac = (
        PagedAttentionConfig(block_size=BLOCK, max_num_blocks=PROBE_BATCH * math.ceil(PROBE_CTX / BLOCK))
        if PROBE_KV
        else None
    )
    logger.info(f"[cfg] KV={PROBE_KV} ctx={PROBE_CTX} batch={PROBE_BATCH} model={MODEL_PATH}")

    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        max_batch_size=PROBE_BATCH,
        max_seq_len=PROBE_CTX,
        paged_attention_config=pac,
        create_kv_cache=PROBE_KV,
        bounded_sliding_kv_cache=(PROBE_CTX > 16384),
        model_path=MODEL_PATH,
    )

    used, total = _dram(mesh_device, f"built KV={int(PROBE_KV)} ctx={PROBE_CTX} batch={PROBE_BATCH}")
    logger.info(
        f"[BUDGET RESULT] KV={int(PROBE_KV)} ctx={PROBE_CTX} batch={PROBE_BATCH}  "
        f"footprint_over_baseline={used-base_used:.3f} GiB/chip  usable={total:.3f} GiB/chip  "
        f"headroom={total-used:.3f} GiB/chip"
    )

    if PROBE_PREFILL:
        import torch.nn.functional as F

        # Non-traced single-chunk prefill of PREFILL_LEN tokens (pad to pow2, like the
        # demo). Materializes the full [1, L, hidden] activation -> stresses the
        # prefill-activation memory regime (the real long-context ceiling, distinct
        # from the static weights+KV budget above). Completion = fits; OOM = ceiling.
        padded = 1 << max((PREFILL_LEN - 1).bit_length(), 11)
        logger.info(f"[prefill] L={PREFILL_LEN} padded={padded}")
        ids = torch.randint(0, model_args.vocab_size, (1, padded), dtype=torch.long)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        tt_tokens = ttnn.from_torch(
            ids.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = model.embed_tokens(tt_tokens)
        embeds = ttnn.reshape(embeds, (1, 1, padded, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
        embed_w = state_dict.get(
            "model.language_model.embed_tokens.weight", state_dict.get("model.embed_tokens.weight")
        )
        embeds_torch = (F.embedding(ids.long(), embed_w) * model.embed_scale).float()
        out = model.ttnn_prefill_forward(
            embeds,
            page_table=None,
            kv_cache=tt_kv_cache,
            input_ids_torch=ids,
            embeds_torch=embeds_torch,
        )
        pk_used, pk_total = _dram(mesh_device, f"after prefill L={PREFILL_LEN}")
        logger.info(f"[PREFILL OK] L={PREFILL_LEN} padded={padded} completed; post-prefill used={pk_used:.3f} GiB/chip")
        out.deallocate(True)
