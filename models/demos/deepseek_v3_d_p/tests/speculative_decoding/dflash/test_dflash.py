# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Tenstorrent DFlash drafter (``TtDFlashDrafterKV``) vs the REAL z-lab HF ``DFlashDraftModel``.

This is the sign-off validation of the *validator* — it compares the device's context-KV build to the
ground truth produced by the actual HF drafter's forward (NOT our torch golden). It jointly validates
the compute graph, the weights, and the RoPE (the device's deepseek-yarn vs the trained model's own
rope), against the production model rather than a re-derivation.

How the ground truth is obtained: the real ``Qwen3DFlashAttention`` builds K/V as
``[k_proj(target_hidden) | k_proj(noise)]`` (context ++ noise), applies k_norm + RoPE to the whole
thing, and stores it in ``past_key_values``. The CONTEXT part is the first ``ctx_len`` positions — and
because k_norm/RoPE are per-position, that slice equals exactly what prefill builds. So we run the real
forward, slice ``key_cache[:, :, :ctx_len, :]`` per layer, and PCC the device K/V against it.

K vs V isolates the failure mode: V is matmul-only (v_proj, no norm/rope) so it should be ~1.0; K adds
k_norm + RoPE, so if V passes but K drops, the culprit is the rope (device deepseek-yarn ≠ trained rope)
or k_norm — not the weights/matmul.

Requires (host): torch + transformers (with the drafter's rope_type). The reference modeling is
VENDORED in-repo (``reference/speculative_decoding/dflash/dflash.py``), so ``$DFLASH_HF_MODEL`` only needs a dir with
``config.json`` (+ ``model.safetensors`` for the pretrained axis) — no ``dflash.py`` in the checkout.
``DFLASH_HF_MODELING=/path/to/dflash.py`` can override the vendored modeling. Also needs a Blackhole mesh
for the device side. Skips cleanly if the model can't be built.

    DFLASH_HF_MODEL=/path/to/Kimi-K2.x-DFlash MESH_DEVICE=8x4 \
    pytest models/demos/deepseek_v3_d_p/tests/speculative_decoding/dflash/test_dflash.py -svv
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.dflash.dflash_drafter_config import DFlashDrafterConfig
from models.demos.deepseek_v3_d_p.tt.dflash.tt_dflash_drafter_kv import TtDFlashDrafterKV
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from tests.ttnn.utils_for_testing import comp_pcc

HF_ENV = "DFLASH_HF_MODEL"
PCC_THRESHOLD = 0.98

_FABRIC_1D = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
}


def _is_drafter(m) -> bool:
    return all(hasattr(m, a) for a in ("fc", "hidden_norm", "layers", "target_layer_ids"))


def _normalize_rope_config(config):
    """K2.6 uses the new ``rope_parameters`` schema with ``rope_type: deepseek_yarn``. Older transformers
    ignore it → ``rope_type='default'`` + wrong ``theta`` (10000) → plain RoPE, no yarn. Translate it into
    the ``rope_scaling`` schema transformers understands (mapping deepseek_yarn→standard yarn, which is
    numerically equivalent here since mscale==mscale_all_dim==1), and set rope_theta, so the reference
    actually applies yarn. No-op if the config already carries a usable ``rope_scaling``."""
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and not getattr(config, "rope_scaling", None):
        if rp.get("rope_theta") is not None:
            config.rope_theta = float(rp["rope_theta"])
        rtype = rp.get("rope_type") or rp.get("type") or "yarn"
        if rtype == "deepseek_yarn":
            rtype = "yarn"
        rs = {"rope_type": rtype}
        for k in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
            if k in rp:
                rs[k] = rp[k]
        config.rope_scaling = rs
    return config


def _import_drafter_class():
    """The ``DFlashDraftModel`` class. Defaults to the VENDORED reference modeling
    (``reference/speculative_decoding/dflash/dflash.py``) so the test always references in-repo code rather than a
    modeling file inside the (re-downloadable) checkout. ``DFLASH_HF_MODELING=/path/to/dflash.py``
    overrides with an external modeling file if you want a different one."""
    override = os.environ.get("DFLASH_HF_MODELING")
    if override:
        if not os.path.exists(override):
            pytest.skip(f"DFLASH_HF_MODELING={override} not found")
        import importlib.util

        spec = importlib.util.spec_from_file_location("dflash_hf_modeling", override)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.DFlashDraftModel
    from models.demos.deepseek_v3_d_p.reference.speculative_decoding.dflash.dflash import DFlashDraftModel

    return DFlashDraftModel


def _load_hf_drafter(load_weights: bool = True):
    """Build the REAL z-lab DFlashDraftModel (fp32, eager) from the VENDORED reference modeling code
    (``reference/speculative_decoding/dflash``) + the checkout's config (+ safetensors when pretrained). The model *code* is
    always the in-repo reference; only config/weights come from ``$DFLASH_HF_MODEL``. With
    load_weights=False (random mode) no safetensors is loaded — the caller supplies random weights."""
    path = os.environ.get(HF_ENV)
    if not path or not os.path.exists(path):
        pytest.skip(f"set {HF_ENV}=/path/to/Kimi-K2.x-DFlash (dir with config.json [+ model.safetensors])")
    try:
        from transformers import AutoConfig

        draft_cls = _import_drafter_class()  # vendored reference (or DFLASH_HF_MODELING override)
        config = _normalize_rope_config(AutoConfig.from_pretrained(path, trust_remote_code=True))
        model = draft_cls(config).float().eval()
        if load_weights:
            from safetensors.torch import load_file

            sd = load_file(os.path.join(path, "model.safetensors"))
            missing, _ = model.load_state_dict(sd, strict=False)
            required = ["fc.weight", "hidden_norm.weight"] + [
                f"layers.{i}.self_attn.{p}.weight"
                for i in range(config.num_hidden_layers)
                for p in ("k_proj", "v_proj", "k_norm")
            ]
            absent = [k for k in required if k in missing]
            if absent:
                pytest.skip(f"checkpoint missing required drafter tensors, e.g. {absent[:3]}")
    except Exception as e:  # transformers missing / qwen3 or deepseek_yarn unsupported / build error
        pytest.skip(
            f"could not build DFlashDraftModel (reference/speculative_decoding/dflash): {type(e).__name__}: {e}"
        )

    if not _is_drafter(model):
        pytest.skip("built model is not a DFlashDraftModel (missing fc/hidden_norm/target_layer_ids)")
    model.config._attn_implementation = "eager"  # force eager so the synthetic forward runs on CPU
    return model


def _drafter_cfg_from_hf(c) -> DFlashDrafterConfig:
    """Build the device config from the HF model's config so dims + rope params match the checkpoint."""
    rs = dict(getattr(c, "rope_scaling", None) or getattr(c, "rope_parameters", None) or {})
    dfc = dict(getattr(c, "dflash_config", None) or {})
    d = DFlashDrafterConfig()  # defaults fill anything the config omits
    return DFlashDrafterConfig(
        hidden_size=c.hidden_size,
        head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        num_hidden_layers=c.num_hidden_layers,
        rms_norm_eps=c.rms_norm_eps,
        target_layer_ids=tuple(dfc.get("target_layer_ids", d.target_layer_ids)),
        rope_theta=float(rs.get("rope_theta") or getattr(c, "rope_theta", None) or d.rope_theta),
        rope_factor=float(rs.get("factor", d.rope_factor)),
        rope_beta_fast=float(rs.get("beta_fast", d.rope_beta_fast)),
        rope_beta_slow=float(rs.get("beta_slow", d.rope_beta_slow)),
        rope_orig_max_pos=int(rs.get("original_max_position_embeddings", d.rope_orig_max_pos)),
        rope_mscale=float(rs.get("mscale", d.rope_mscale)),
        rope_mscale_all_dim=float(rs.get("mscale_all_dim", d.rope_mscale_all_dim)),
    )


def _random_state_dict(cfg: DFlashDrafterConfig, seed: int = 42) -> dict:
    """Seeded random weights for the 20-tensor prefill subset: proj ~ N(0, initializer_range), norm gains
    = ones (the same seeded-random convention as the deepseek prefill tests). Self-contained; fed
    identically to the HF model and the device."""
    g = torch.Generator().manual_seed(seed)
    H, kv, D, std = cfg.hidden_size, cfg.kv_dim, cfg.head_dim, cfg.initializer_range

    def _lin(out_dim: int, in_dim: int) -> torch.Tensor:
        return (torch.randn(out_dim, in_dim, generator=g) * std).to(torch.bfloat16)

    sd: dict = {
        "fc.weight": _lin(H, cfg.target_feature_size),
        "hidden_norm.weight": torch.ones(H, dtype=torch.bfloat16),
    }
    for i in range(cfg.num_hidden_layers):
        sd[f"layers.{i}.self_attn.k_proj.weight"] = _lin(kv, H)
        sd[f"layers.{i}.self_attn.v_proj.weight"] = _lin(kv, H)
        sd[f"layers.{i}.self_attn.k_norm.weight"] = torch.ones(D, dtype=torch.bfloat16)
    return sd


def _cache_kv(pkv, i):
    """Pull layer i's (key, value) from a DynamicCache across transformers API variants."""
    if hasattr(pkv, "key_cache") and len(pkv.key_cache) > i:
        return pkv.key_cache[i], pkv.value_cache[i]
    if hasattr(pkv, "layers"):
        return pkv.layers[i].keys, pkv.layers[i].values
    kv = pkv[i]
    return kv[0], kv[1]


@torch.inference_mode()
def _hf_context_kv(model, cfg: DFlashDrafterConfig, ctx: torch.Tensor, q_len: int):
    """Run the REAL drafter forward and return per-layer (k_ctx, v_ctx) as [kv_heads, ctx_len, head_dim] fp32.

    The context K/V depend only on ``target_hidden`` (shared across layers), so the noise block content
    is irrelevant — zeros suffice — and the forward's noise/attention path need not be numerically
    meaningful for the captured context slice to be correct.
    """
    from transformers import DynamicCache

    ctx_len = ctx.shape[1]
    total = ctx_len + q_len
    noise = torch.zeros(1, q_len, cfg.hidden_size, dtype=ctx.dtype)
    position_ids = torch.arange(total).unsqueeze(0)
    pkv = DynamicCache()
    try:
        model(
            target_hidden=ctx,
            noise_embedding=noise,
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=pkv,
            use_cache=True,
            cache_position=torch.arange(total),
        )
    except Exception as e:
        pytest.skip(f"HF drafter forward failed (likely a transformers version detail): {type(e).__name__}: {e}")

    out = {}
    for i in range(cfg.num_hidden_layers):
        k, v = _cache_kv(pkv, i)  # [1, kv_heads, total, head_dim]
        out[i] = (k[0, :, :ctx_len, :].float(), v[0, :, :ctx_len, :].float())
    return out


def test_rope_parity_hf_vs_device():
    """Diagnostic (host-only, no mesh): compare the device's RoPE tables (rope.get_cos_sin_matrix, what
    TtDFlashDrafterKV builds) against the HF drafter's own Qwen3RotaryEmbedding, to localize a K-vs-V
    PCC gap to the rope. Prints, so we can SEE the nature of any mismatch:
      * which `interleave` (half-split vs Meta) matches HF -> convention,
      * cos/sin PCC -> frequency (yarn) match,
      * HF inv_freq vs a PLAIN (no-yarn) inv_freq -> whether HF actually applied yarn or fell back to
        plain rope (if HF == plain, the HF reference is the wrong one and the device's yarn is correct).
    """
    from transformers import AutoConfig

    path = os.environ.get(HF_ENV)
    if not path or not os.path.exists(path):
        pytest.skip(f"set {HF_ENV}=/path/to/Kimi-K2.x-DFlash")
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        config = _normalize_rope_config(AutoConfig.from_pretrained(path, trust_remote_code=True))
        hf_rope = Qwen3RotaryEmbedding(config).float().eval()
    except Exception as e:
        pytest.skip(f"cannot build Qwen3RotaryEmbedding from {path}: {type(e).__name__}: {e}")

    from models.demos.deepseek_v3_d_p.tt.dflash.dflash_drafter_config import build_drafter_rope_hf_config
    from models.demos.deepseek_v3_d_p.tt.mla.rope import get_cos_sin_matrix

    cfg = _drafter_cfg_from_hf(config)
    S = 512
    pos = torch.arange(S).unsqueeze(0)
    cos_hf, sin_hf = hf_rope(torch.zeros(1, S, cfg.head_dim), pos)
    cos_hf, sin_hf = cos_hf[0].float(), sin_hf[0].float()  # [S, head_dim]

    hf_cfg = build_drafter_rope_hf_config(cfg, S)
    for interleave in (False, True):
        c, s = get_cos_sin_matrix(hf_cfg, interleave=interleave)  # [1,1,S,head_dim]
        c, s = c[0, 0, :S].float(), s[0, 0, :S].float()
        _, pcc_c = comp_pcc(cos_hf, c, 0.99)
        _, pcc_s = comp_pcc(sin_hf, s, 0.99)
        logger.info(
            f"[rope] interleave={interleave}: cos_pcc={pcc_c} sin_pcc={pcc_s} "
            f"max|dcos|={(cos_hf - c).abs().max().item():.4f} ratio(cos_hf/cos_dev)@[1,0]="
            f"{(cos_hf[1, 0] / c[1, 0]).item():.4f}"
        )

    d, base = cfg.head_dim, cfg.rope_theta
    plain = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    logger.info(f"[rope] cfg: theta={base} factor={cfg.rope_factor} orig={cfg.rope_orig_max_pos} head_dim={d}")
    logger.info(f"[rope] plain(no-yarn) inv_freq[:6]={plain[:6].tolist()}")
    if hasattr(hf_rope, "inv_freq"):
        logger.info(f"[rope] HF inv_freq[:6]={hf_rope.inv_freq[:6].float().tolist()}")
    if hasattr(hf_rope, "attention_scaling"):
        logger.info(f"[rope] HF attention_scaling={float(hf_rope.attention_scaling)}")
    logger.info(f"[rope] HF rope_type={getattr(hf_rope, 'rope_type', '?')}")


@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("ctx_len", [512], ids=["ctx512"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            _FABRIC_1D,
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dflash_device_vs_hf_pcc(mesh_device, device_params, num_links, topology, ctx_len, use_pretrained):
    # Same weights feed BOTH the HF reference and the device (required for a meaningful PCC):
    #   pretrained -> real checkpoint (safetensors loaded into the HF model),
    #   random     -> seeded random_state_dict loaded into both (no pretrained safetensors needed; still
    #                 needs config + dflash.py + transformers to build the HF model).
    model = _load_hf_drafter(load_weights=use_pretrained)
    cfg = _drafter_cfg_from_hf(model.config)
    if use_pretrained:
        sd = model.state_dict()  # the device picks its 20-tensor subset (fc/hidden_norm/k/v_proj/k_norm)
    else:
        sd = _random_state_dict(cfg)
        model.load_state_dict(sd, strict=False)  # give the HF model the SAME random weights

    mesh_shape = tuple(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    tp = mesh_shape[tp_axis]
    assert cfg.num_key_value_heads % tp == 0, f"num_kv_heads {cfg.num_key_value_heads} not divisible by tp {tp}"
    H = cfg.hidden_size

    # One synthetic context feature, fed identically to both sides.
    gen = torch.Generator().manual_seed(0)
    ctx = torch.randn(1, ctx_len, cfg.target_feature_size, generator=gen, dtype=torch.float32)

    # ---- ground truth: the REAL HF drafter forward (context slice of its KV cache) ----
    q_len = int(getattr(model.config, "block_size", cfg.block_size))
    real = _hf_context_kv(model, cfg, ctx, q_len)

    # ---- device ----
    drafter = TtDFlashDrafterKV(
        mesh_device,
        cfg,
        state_dict=sd,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        max_seq_len=ctx_len,
        num_links=num_links,
        topology=topology,
    )
    hidden_shard = [None, None]
    hidden_shard[tp_axis] = 3  # tap hidden TP-sharded on the hidden dim, SP-replicated
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=hidden_shard)

    drafter.reset()
    for j, tid in enumerate(cfg.target_layer_ids):
        h_j = ctx[:, :, j * H : (j + 1) * H].to(torch.bfloat16).reshape(1, 1, ctx_len, H)
        h_tt = ttnn.from_torch(
            h_j,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        drafter.tap(h_tt, tid)
    drafter.write_kv_cache()
    ttnn.synchronize_device(mesh_device)

    def _read(cache):
        host = ttnn.to_torch(
            cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_shape)
        )
        return host[: cfg.num_hidden_layers][:, :, :ctx_len, :].float()  # [num_layers, kv_heads, ctx_len, head_dim]

    dk = _read(drafter.k_cache)
    dv = _read(drafter.v_cache)

    for i in range(cfg.num_hidden_layers):
        rk, rv = real[i]
        ok_k, pcc_k = comp_pcc(rk, dk[i], PCC_THRESHOLD)
        ok_v, pcc_v = comp_pcc(rv, dv[i], PCC_THRESHOLD)
        logger.info(f"layer {i}: K pcc={pcc_k} (ok={ok_k})  V pcc={pcc_v} (ok={ok_v})")
        # V (matmul-only) should be ~1.0; if V passes but K fails, suspect the RoPE (deepseek-yarn vs the
        # trained model's rope) or k_norm, not the weights.
        assert ok_v, f"V layer {i}: device vs HF PCC {pcc_v} < {PCC_THRESHOLD} (matmul/weights mismatch)"
        assert ok_k, f"K layer {i}: device vs HF PCC {pcc_k} < {PCC_THRESHOLD} (norm/rope mismatch if V passed)"
