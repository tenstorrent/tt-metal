# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview prefill device smoke: layer-0 KV and index-key caches vs the HF hy_v4 golden.

Drives ``TtPrefillTransformer`` over a Hy4 weight cache through the chunked-transformer test driver
(``run_chunked_transformer_updated``): one 4096-token chunk = the whole 4K golden prompt (#65919's
``hy4_golden_v1`` trace, read with the driver's own trace loaders). Then it scores the un-rotated device
caches against the golden:

  * KVPE cache, both halves: ``nope`` (the 512-wide latent) and ``pe`` (k_pe; the golden and the device both pair
    RoPE interleaved, the checkpoint's pairing, so it compares directly: ``Hy4Adapter.kv_pe_interleave = False``);
  * index-key cache (both halves; the golden stores it in the device frame ``[rope 64 interleaved | nope 64]``).

Layer 0 is asserted >= 0.99 on all three. It sees the real embeddings, so it isolates the attention weights
and their RoPE frames (``hy4_checkpoint.indexer_to_device_frame``; the MLA rope rows need no reorder). Deeper layers are recorded only: the learnable sink (#65921) and
iHC residual (#65891) are not implemented, so their inputs already differ from HF.

It runs the DEV / NUMERICS layout (``glm_prefill_sp8_tp4``, the 8x4 mesh): the Hy4 layout is TP32 / DP1 (#65884),
which the shared prefill stack cannot run yet. ``PREFILL_HY4_LAYOUT`` defaults to the dev layout here; naming
``tp32`` fails with its blockers (``tt/hy4/layout.py``).

Needs (each missing one skips): the Hy4 checkpoint, the golden trace (``PREFILL_TRACE_DIR``), a Hy4 weight
cache built at the current ``HY4_MLA_CACHE_FORMAT`` (build it with the prefill runner, which writes the
missing / stale categories), an 8x4 Blackhole Galaxy, and ``PREFILL_CHUNK_SIZE=4096``. A cache whose MLA
layers predate the current RoPE frames FAILS rather than skips: it would load silently and score ~0.

Env: ``HY4_HF_MODEL`` / ``PREFILL_HF_MODEL`` (checkpoint), ``TT_HY4_PREFILL_TTNN_CACHE`` / ``PREFILL_TTNN_CACHE``
(cache root, as the runner takes it), ``HY4_SMOKE_LAYERS`` (layers to run, default 1),
``HY4_SMOKE_REPORT`` (optional JSON path for the per-layer table).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from loguru import logger

import models.demos.deepseek_v3_d_p.utils.chunk_config as chunk_config
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.runners.adapters.hy4_preview import Hy4Adapter

GOLDEN_TOKENS = 4096  # the 4K golden = one chunk
CHUNK_SIZE = GOLDEN_TOKENS
PCC_FLOOR = 0.99
NUM_LAYERS = int(os.environ.get("HY4_SMOKE_LAYERS", "1"))
MESH = (8, 4)

# The chunked driver module asserts its 55k split tables (chunks up to 5120) against its CHUNK at import. This
# test runs the whole prompt as one 4096-token chunk, so import it with the production width and pass
# chunk_size=4096 explicitly; every other module keeps the PREFILL_CHUNK_SIZE it was imported with.
_chunk_env = chunk_config.PREFILL_CHUNK_TOKENS
chunk_config.PREFILL_CHUNK_TOKENS = max(_chunk_env, chunk_config.DEFAULT_PREFILL_CHUNK_TOKENS)
try:
    import models.demos.deepseek_v3_d_p.tests.test_prefill_transformer_chunked as chunked  # noqa: E402
finally:
    chunk_config.PREFILL_CHUNK_TOKENS = _chunk_env


def _checkpoint_dir() -> Path:
    return Path(os.environ.get("HY4_HF_MODEL") or os.environ.get("PREFILL_HF_MODEL") or Hy4Adapter.hf_model_default)


def _cache_root() -> Path:
    return Path(
        os.environ.get(Hy4Adapter.ttnn_cache_env)
        or os.environ.get("PREFILL_TTNN_CACHE")
        or Hy4Adapter.ttnn_cache_default
    )


def _trace_dir() -> Path:
    return chunked._resolve_trace_dir(Hy4Adapter)


def _golden_ok(trace: Path) -> bool:
    try:
        meta = json.loads((trace / "metadata.json").read_text())
    except (OSError, ValueError):
        return False
    return (
        meta.get("model") == "hy4_preview"
        and int(meta.get("n_prompt_tokens", 0)) >= GOLDEN_TOKENS
        and (trace / "kv_cache" / "layer_0").is_dir()
        and (trace / "dsa" / "indexer_k_layer_0").is_dir()
    )


pytestmark = [
    pytest.mark.skipif(
        not (_checkpoint_dir() / "model.safetensors.index.json").is_file(),
        reason=f"Hy4-preview checkpoint not found at {_checkpoint_dir()} (HY4_HF_MODEL / PREFILL_HF_MODEL)",
    ),
    pytest.mark.skipif(
        not _golden_ok(_trace_dir()), reason=f"no Hy4 4K golden trace at {_trace_dir()} (PREFILL_TRACE_DIR)"
    ),
    pytest.mark.skipif(
        _chunk_env != CHUNK_SIZE,
        reason=f"run with PREFILL_CHUNK_SIZE={CHUNK_SIZE}: the 4K golden is one chunk (got {_chunk_env})",
    ),
]


def _mesh_cache_dir(num_devices: int) -> Path:
    """The weight-cache dir the runner and the pytest fixture both use: ``<root>/<variant>_<arch>_<n>dev/<rows>x<cols>``."""
    from models.common.utility_functions import is_blackhole
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import mesh_cache_key

    arch = "bh" if is_blackhole() else "wh"
    return _cache_root() / f"{Hy4Adapter.name}_{arch}_{num_devices}dev" / mesh_cache_key(MESH, 0)


class _PccRecorder:
    """Captures the per-half PCCs the driver's cache recorders compute (``cache_half_pccs``), per layer."""

    def __init__(self, monkeypatch):
        self.kv: dict[int, dict] = {}
        self.index_k: dict[int, dict] = {}
        self._sink = None
        orig_half = chunked.cache_half_pccs
        orig_kv, orig_ix = chunked._record_kv_cache_pcc, chunked._record_indexer_k_cache_pcc

        def half(*a, **kw):
            first, second = orig_half(*a, **kw)
            if self._sink is not None:
                self._sink(first, second)
            return first, second

        def kv(*a, **kw):
            layers = iter(range(NUM_LAYERS))
            self._sink = lambda nope, pe: self.kv.__setitem__(next(layers), {"nope": nope, "pe": pe})
            kw["assert_threshold"] = None  # this test asserts layer 0 itself; deeper layers are recorded only
            try:
                return orig_kv(*a, **kw)
            finally:
                self._sink = None

        def ix(trace_dir, *a, **kw):
            have = [i for i in range(NUM_LAYERS) if (Path(trace_dir) / "dsa" / f"indexer_k_layer_{i}").exists()]
            layers = iter(have)
            self._sink = lambda rope, nope: self.index_k.__setitem__(next(layers), {"rope": rope, "nope": nope})
            try:
                return orig_ix(trace_dir, *a, **kw)
            except AssertionError as e:  # its 0.95 floor spans every captured layer; layer 0 is asserted below
                logger.warning(f"[hy4 smoke] indexer-K recorder assert (recorded, not fatal): {e}")
            finally:
                self._sink = None

        monkeypatch.setattr(chunked, "cache_half_pccs", half)
        monkeypatch.setattr(chunked, "_record_kv_cache_pcc", kv)
        monkeypatch.setattr(chunked, "_record_indexer_k_cache_pcc", ix)

    def table(self) -> list[dict]:
        rows = []
        for layer in sorted(self.kv):
            row = {"layer": layer, "kvpe_nope": self.kv[layer]["nope"], "kvpe_pe": self.kv[layer]["pe"]}
            if layer in self.index_k:
                row["index_k_rope"] = self.index_k[layer]["rope"]
                row["index_k_nope"] = self.index_k[layer]["nope"]
            rows.append(row)
        return rows


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, sp_axis",
    [
        pytest.param(
            MESH,
            torus_xy_device_params(
                fabric_payload_size=chunked.moe_fabric_payload(GLM53Config),
                l1_small_size=chunked.GLM_L1_SMALL_SIZE,
                trace_region_size=chunked.GLM_TRACE_REGION_SIZE,
            ),
            2,
            0,
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["hy4_preview"], indirect=True, ids=["hy4"])
@pytest.mark.timeout(0)
def test_hy4_prefill_layer0_kv_and_index_k_pcc(
    variant, config_only, mesh_device, device_params, num_links, sp_axis, monkeypatch
):
    from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config
    from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    # The adapter's layout gate, which this driver (TtPrefillTransformer directly) does not pass through.
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, os.environ.get(hy4_layout.LAYOUT_ENV) or hy4_layout.DEV_LAYOUT)
    hy4_layout.resolve_layout(MESH[sp_axis], MESH[1 - sp_axis], sp_axis, Hy4Config.NUM_ATTENTION_HEADS)

    num_devices = mesh_device.get_num_devices()
    cache = _mesh_cache_dir(num_devices)
    if not cache.is_dir():
        pytest.skip(f"no Hy4 weight cache at {cache}: build it with the prefill runner (PREFILL_MODEL=hy4_preview)")
    stale = hc.stale_mla_cache_layers(cache, range(NUM_LAYERS))
    assert not stale, (
        f"Hy4 MLA/indexer cache at {cache} for layers {stale} predates format {hc.HY4_MLA_CACHE_FORMAT} (RoPE pairing "
        f"{hc.HY4_ROPE_PAIRING}); rebuild it with the prefill runner (it deletes and rewrites stale layers)"
    )
    missing = [i for i in range(NUM_LAYERS) if not hc.mla_cache_stamp(cache, i).exists()]
    if missing:
        pytest.skip(f"Hy4 weight cache at {cache} has no MLA cache for layers {missing}: build them with the runner")

    rec = _PccRecorder(monkeypatch)
    chunked.run_chunked_transformer_updated(
        variant,
        config_only,
        mesh_device,
        cache.parent,  # the driver appends the mesh key itself
        NUM_LAYERS,
        1,  # one 4096-token chunk = the whole golden prompt
        GateComputeMode.DEVICE_FP32,
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        1,
        routing_use_l1_small_for_semaphores=True,
        preload_isl=0,
        check_pcc=True,
        check_layer_pcc=False,  # the Hy4 golden carries no decoder outputs
        use_trace=False,
        seq_cache=2 * CHUNK_SIZE,
        kv_cache_format=chunked.GLM_KV_CACHE_FORMAT,
        sp_axis=sp_axis,
        chunk_size=CHUNK_SIZE,
    )

    rows = rec.table()
    logger.info("Hy4 prefill cache PCC vs golden (layers > 0 recorded only: no sink / iHC yet):")
    logger.info("layer  kvpe_nope  kvpe_pe   index_k_rope  index_k_nope")
    for r in rows:
        ik = f"{r['index_k_rope']:12.6f}  {r['index_k_nope']:12.6f}" if "index_k_rope" in r else f"{'-':>12}  {'-':>12}"
        logger.info(f"{r['layer']:5d}  {r['kvpe_nope']:9.6f}  {r['kvpe_pe']:8.6f}  {ik}")
    if os.environ.get("HY4_SMOKE_REPORT"):
        Path(os.environ["HY4_SMOKE_REPORT"]).write_text(json.dumps({"threshold": PCC_FLOOR, "layers": rows}, indent=1))

    assert rows and rows[0]["layer"] == 0, f"no layer-0 KV PCC recorded: {rows}"
    l0 = rows[0]
    assert "index_k_rope" in l0, "no layer-0 index-key PCC recorded (layer 0 is a full indexer layer)"
    for key in ("kvpe_nope", "kvpe_pe", "index_k_rope", "index_k_nope"):
        assert l0[key] >= PCC_FLOOR, f"layer 0 {key} PCC {l0[key]:.6f} < {PCC_FLOOR}"
