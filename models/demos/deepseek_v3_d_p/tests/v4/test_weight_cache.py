# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The V4 weight cache, one module at a time, on real DeepSeek-V4-Flash weights.

For each module that caches its weights (sliding-window, HCA and CSA attention with their compressors and the CSA
indexer, and the mHC wrap and head) the same four device builds must hold identical weights:

  1. no cache,
  2. a cache-writing build (a miss converts and writes),
  3. a cache-only build from the files that wrote (no torch module at all),
  4. a cache-only build from a cache the HOST-ONLY ``build_ttnn_cache`` wrote, whose files must also be
     byte-identical to the ones the constructor wrote.

``check_cache_complete`` must be true for a written cache and false for an empty directory, and a cache-only build
over an empty directory must raise instead of converting (and writing out) a placeholder.

Needs an 8x4 Blackhole mesh and the Flash checkpoint; skips without the checkpoint.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.block import v4_block_modules
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig as C
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tests.v4 import golden
from models.demos.deepseek_v3_d_p.tests.v4.helpers import mesh_params
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCHead, TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.mla.compressed_sparse_attention import TtCSA
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCA
from models.demos.deepseek_v3_d_p.tt.mla.sliding_window_attention import TtSWA
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4.weights import load_checkpoint_tensors
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from tt_py_test_utils_common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")

_MESH = [p for p in mesh_params(C.FABRIC_PAYLOAD_SIZE) if p.id == "torus-xy-8x4"]
_ATTN = ("sinks_sdpa", "wq_a", "wq_b", "q_a_norm_weight", "wkv", "kv_norm_weight", "wo_a", "wo_b")
_COMP = ("wkv", "wgate", "position_bias", "kv_norm_weight")
_IDX = ("_idx_wq_b", "_idx_wproj", "_idx_kv_proj", "_idx_gate_proj", "_idx_position_bias", "_idx_knorm")
# The Flash layer of each attention kind: 0 is sliding, 3 heavily compressed, 4 compressed sparse.
_KINDS = {"swa": (0, TtSWA), "hca": (3, TtHCA), "csa": (4, TtCSA)}


def _checkpoint():
    ckpt = golden.resolve_checkpoint(golden.V4_FLASH)
    if ckpt is None:
        pytest.skip(f"no V4-Flash checkpoint at {golden.V4_FLASH.checkpoint}")
    return ckpt


def _fingerprint(tensor, mesh) -> str:
    host = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().contiguous()
    return hashlib.sha256(host.numpy().tobytes()).hexdigest()[:16]


def _fingerprints(obj, names, mesh, prefix="") -> dict:
    return {prefix + n: _fingerprint(getattr(obj, n), mesh) for n in names}


def _differ(a: dict, b: dict) -> list:
    return sorted(k for k in a if a[k] != b[k])


def _byte_identical(a_dir, b_dir) -> tuple:
    names_a = sorted(f.name for f in a_dir.glob("*.tensorbin"))
    names_b = sorted(f.name for f in b_dir.glob("*.tensorbin"))
    same = names_a == names_b and all((a_dir / n).read_bytes() == (b_dir / n).read_bytes() for n in names_a)
    return same, len(names_a)


def _reference_attention(layer: int, ckpt):
    """The layer's reference attention module with the real checkpoint weights loaded into it."""
    cfg = v4_hf_config(C, layer + 1, max_seq_len=56320)
    attn = v4_block_modules(cfg, layer)["attn"]
    mapping = {
        "attn.attn_sink": attn.sinks,
        "attn.wq_a.weight": attn.q_a_proj.weight,
        "attn.q_norm.weight": attn.q_a_norm.weight,
        "attn.wq_b.weight": attn.q_b_proj.weight,
        "attn.wkv.weight": attn.kv_proj.weight,
        "attn.kv_norm.weight": attn.kv_norm.weight,
        "attn.wo_a.weight": attn.o_a_proj.weight,
        "attn.wo_b.weight": attn.o_b_proj.weight,
    }
    comp = attn.compressor
    if comp is not None:
        mapping.update(
            {
                "attn.compressor.wkv.weight": comp.kv_proj.weight,
                "attn.compressor.wgate.weight": comp.gate_proj.weight,
                "attn.compressor.ape": comp.position_bias,
                "attn.compressor.norm.weight": comp.kv_norm.weight,
            }
        )
        idx = getattr(comp, "indexer", None)
        if idx is not None:
            mapping.update(
                {
                    "attn.indexer.compressor.wkv.weight": idx.kv_proj.weight,
                    "attn.indexer.compressor.wgate.weight": idx.gate_proj.weight,
                    "attn.indexer.compressor.ape": idx.position_bias,
                    "attn.indexer.compressor.norm.weight": idx.kv_norm.weight,
                    "attn.indexer.wq_b.weight": idx.q_b_proj.weight,
                    "attn.indexer.weights_proj.weight": idx.scorer.weights_proj.weight,
                }
            )
    loaded = load_checkpoint_tensors(ckpt, [f"layers.{layer}.{k}" for k in mapping])
    with torch.no_grad():
        for key, param in mapping.items():
            param.copy_(loaded[f"layers.{layer}.{key}"])
    return cfg, attn


@pytest.mark.parametrize("mesh_device, device_params, num_links", _MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("kind", list(_KINDS))
def test_attention_weight_cache(mesh_device, device_params, num_links, kind, tmp_path, expect_error):
    layer, cls = _KINDS[kind]
    cfg, ref = _reference_attention(layer, _checkpoint())
    csa = kind == "csa"
    prefix = f"layer_{layer}.attn"
    written, built, empty = tmp_path / "written", tmp_path / "host_built", tmp_path / "empty"
    for d in (written, built, empty):
        d.mkdir()
    common = dict(sp_axis=0, tp_axis=1, topology=per_axis_topology(device_params["fabric_config"]))
    extra = dict(layer_idx=layer) if csa else {}  # names the indexer's files, which are keyed by the GLOBAL layer

    def fingerprints(attn):
        out = _fingerprints(attn, _ATTN, mesh_device)
        if kind != "swa":
            out.update(_fingerprints(attn.compressor, _COMP, mesh_device, "comp."))
        if csa:
            attn.build_tables(10240, chunk_tokens=5120)  # the indexer is built there
            out.update(_fingerprints(attn._indexer, _IDX, mesh_device, "idx."))
        return out

    def from_cache(path):
        return cls.from_cache(mesh_device, cfg, weight_cache_path=path, cache_name_prefix=prefix, **common, **extra)

    def complete(path):
        init_checker(path)
        return cls.check_cache_complete(path, prefix, layer) if csa else cls.check_cache_complete(path, prefix)

    plain = fingerprints(cls.from_reference(mesh_device, ref, cfg, **common, **extra))
    cold = fingerprints(
        cls.from_reference(
            mesh_device, ref, cfg, weight_cache_path=written, cache_name_prefix=prefix, **common, **extra
        )
    )
    warm = fingerprints(from_cache(written))
    cls.build_ttnn_cache(ref, cfg, mesh_device, built, prefix, sp_axis=0, tp_axis=1, **extra)
    same_bytes, n_files = _byte_identical(written, built)
    warm_from_host_built = fingerprints(from_cache(built))

    assert _differ(plain, cold) == [], "converting through the cache changed a weight"
    assert _differ(cold, warm) == [], "loading the cache gave different weights from the build that wrote it"
    assert _differ(cold, warm_from_host_built) == [], "the host-only build wrote a different cache"
    assert same_bytes and n_files > 0, "the host-only build's files differ from the constructor's"
    assert complete(written) and complete(built) and not complete(empty)
    with expect_error(FileNotFoundError, "weight cache miss with no torch weight"):
        from_cache(empty)


@pytest.mark.parametrize("mesh_device, device_params, num_links", _MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("site", ["wrap", "head"])
def test_mhc_weight_cache(mesh_device, device_params, num_links, site, tmp_path):
    ckpt = _checkpoint()
    cfg = v4_hf_config(C, 4, max_seq_len=56320)
    mcfg = MHCConfig(
        dim=cfg.hidden_size,
        n=cfg.hc_mult,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )
    if site == "wrap":
        cls, names, prefix, key = TtMHCWrap, ("fn_T", "consts"), "layer_3.mhc_attn", "layers.3.hc_attn_"
    else:
        cls, names, prefix, key = TtMHCHead, ("fn_T", "base"), "hc_head", "hc_head_"
    loaded = load_checkpoint_tensors(ckpt, [key + k for k in ("fn", "base", "scale")])
    args = tuple(loaded[key + k].float() for k in ("fn", "base", "scale"))
    written, built, empty = tmp_path / "written", tmp_path / "host_built", tmp_path / "empty"
    for d in (written, built, empty):
        d.mkdir()
    kw = dict(tp_axis=1, num_links=1, topology=per_axis_topology(device_params["fabric_config"])[1])

    def fingerprints(module):
        out = _fingerprints(module, names, mesh_device)
        if site == "head":
            out["a"] = repr(module.a)  # the host scalar scale[0], which the cache keeps too
        return out

    def complete(path):
        init_checker(path)
        return cls.check_cache_complete(path, prefix)

    plain = fingerprints(cls(mesh_device, mcfg, *args, **kw))
    cold = fingerprints(cls(mesh_device, mcfg, *args, weight_cache_path=written, cache_name_prefix=prefix, **kw))
    warm = fingerprints(cls.from_cache(mesh_device, mcfg, weight_cache_path=written, cache_name_prefix=prefix, **kw))
    cls.build_ttnn_cache(*args, mcfg, mesh_device, built, prefix, tp_axis=1)
    same_bytes, n_files = _byte_identical(written, built)
    warm_from_host_built = fingerprints(
        cls.from_cache(mesh_device, mcfg, weight_cache_path=built, cache_name_prefix=prefix, **kw)
    )

    assert _differ(plain, cold) == [] and _differ(cold, warm) == [] and _differ(cold, warm_from_host_built) == []
    assert same_bytes and n_files > 0
    assert complete(written) and complete(built) and not complete(empty)
