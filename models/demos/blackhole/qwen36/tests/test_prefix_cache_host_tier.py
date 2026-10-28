# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE tests of the GDN prefix-snapshot host spill tier (QWEN36_PREFIX_HOST_SLOTS, tt/prefix_cache.py).

test_host_tier_roundtrip_bytes (no weights): GdnPrefixStateCache on stub GDN layers with the real per-device state shapes
(48 layers, rec_state [1, 48/TP, 128, 128] fp32, conv_carry [1, 3, 10240/TP] bf16), DIFFERENT random bytes on every
device. A snapshot evicted to host and restored must equal the saved state bit for bit on every device (torch.equal),
like a device-slot restore. Logs the measured spill / restore cost (the numbers that decide QWEN36_PREFIX_HOST_SPILL_MIN_POS
and QWEN36_PREFIX_HOST_MIN_GAIN): "HOST_TIER spill_ms=... restore_ms=...".

test_spec_prefix_host_tier_e2e (needs HF_MODEL): the spec class with ONE device snapshot slot and 4 host slots. Request
A saves its 4096 snapshot, an unrelated request U evicts it to host, request B (A's first 4352 tokens + new text,
start_pos 4352) must resume from the host copy and match a full non-cached prefill of B (test_spec_prefix_cache.Harness:
logits PCC + top-1, MTP hidden, GDN state, main + drafter KV).

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_prefix_cache_host_tier.py::test_host_tier_roundtrip_bytes -svq
    MESH_DEVICE=P150x8 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_prefix_cache_host_tier.py -svq
"""

import os
import statistics
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.prefix_cache import GdnPrefixStateCache, HostSnap

N_LAYERS = int(os.environ.get("QWEN36_HOST_TIER_TEST_LAYERS", "48"))
NV, DK, DV, CONV = 48, 128, 128, 10240  # Qwen3.6/3.8-27B GDN: value heads, head dims, conv channels (q+k+v)


def _alloc(mesh, shape, dt):
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    return ttnn.from_torch(
        torch.zeros(*shape, dtype=tdt),
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _fill(mesh, t, g):
    """Distinct random bytes on every device, written in place (same buffer). -> per-device host copies."""
    tdt = torch.float32 if t.dtype == ttnn.float32 else torch.bfloat16
    per = [torch.randn(*[int(d) for d in t.shape], generator=g).to(tdt) for _ in range(mesh.get_num_devices())]
    host = ttnn.from_host_shards([ttnn.from_torch(p, dtype=t.dtype, layout=ttnn.TILE_LAYOUT) for p in per], mesh.shape)
    ttnn.copy_host_to_device_tensor(host, t)
    return per


def _read(t):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


class _StubModel:
    """Just what GdnPrefixStateCache needs: the B=1 GDN scratch tuples (dn, rec, conv_states, carry, ...)."""

    def __init__(self, mesh, layers):
        tp = mesh.get_num_devices()
        self._gdn_prefill_scratch = []
        for _ in range(layers):
            rec = _alloc(mesh, (1, max(NV // tp, 1), DK, DV), ttnn.float32)
            carry = _alloc(mesh, (1, 3, max(CONV // tp, 32)), ttnn.bfloat16)
            dn = SimpleNamespace(rec_state=rec, conv_carry=carry, mesh=mesh)
            self._gdn_prefill_scratch.append((dn, rec, None, carry, None, None))

    def _ensure_gdn_prefill_scratch(self):
        pass


def _bind_random(mesh, model, g):
    return [(_fill(mesh, dn.rec_state, g), _fill(mesh, dn.conv_carry, g)) for dn, *_ in model._gdn_prefill_scratch]


def _check(mesh, model, want, tag):
    ttnn.synchronize_device(mesh)
    for li, ((dn, *_), (rec, carry)) in enumerate(zip(model._gdn_prefill_scratch, want)):
        for name, got, exp in (("rec", _read(dn.rec_state), rec), ("carry", _read(dn.conv_carry), carry)):
            for d, (a, b) in enumerate(zip(got, exp)):
                assert torch.equal(a.reshape(b.shape), b), f"{tag}: layer {li} {name} differs on device {d}"


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_host_tier_roundtrip_bytes(mesh_device, monkeypatch):
    for k in ("QWEN36_PREFIX_HOST_SPILL_MIN_POS", "QWEN36_PREFIX_HOST_MIN_GAIN", "QWEN36_PREFIX_HOST_SPILL_LOW"):
        monkeypatch.delenv(k, raising=False)
    mesh = mesh_device
    g = torch.Generator().manual_seed(0)
    model = _StubModel(mesh, N_LAYERS)
    cache = GdnPrefixStateCache(model, num_slots=1, host_slots=2, name="host-tier test")
    tokens = torch.arange(40000, dtype=torch.int32) % 991

    st1 = _bind_random(mesh, model, g)
    assert cache.save(tokens, 4096)
    st2 = _bind_random(mesh, model, g)
    assert cache.save(tokens, 8192)  # evicts 4096 -> host
    assert cache.stats["spills"] == 1

    _bind_random(mesh, model, g)  # junk in the bound state
    pos, h = cache.lookup(tokens, 8000)
    assert pos == 4096 and isinstance(h, HostSnap)
    cache.restore(h)
    _check(mesh, model, st1, "host restore")

    _bind_random(mesh, model, g)
    pos, h = cache.lookup(tokens, 8192)
    assert pos == 8192 and not isinstance(h, HostSnap)
    cache.restore(h)
    _check(mesh, model, st2, "device restore")

    # cost: spill (blocking read of one snapshot) and host restore (PCIe write + staging copy, synchronized)
    spill_ms, restore_ms = [], []
    for i in range(5):
        _bind_random(mesh, model, g)
        ttnn.synchronize_device(mesh)
        n0, ms0 = cache.stats["spills"], cache.stats["spill_ms"]
        cache.save(tokens, 12288 + 4096 * i)  # each save evicts the previous one -> one spill
        assert cache.stats["spills"] == n0 + 1
        spill_ms.append(cache.stats["spill_ms"] - ms0)
        pos, h = cache.lookup(tokens, 12288 + 4096 * i - 1)
        assert isinstance(h, HostSnap)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        cache.restore(h)
        ttnn.synchronize_device(mesh)
        restore_ms.append((time.perf_counter() - t0) * 1e3)
    logger.info(
        f"HOST_TIER layers={N_LAYERS} devices={mesh.get_num_devices()} snapshot~{cache._snapshot_mb():.0f}MB "
        f"spill_ms={statistics.median(spill_ms):.1f} restore_ms={statistics.median(restore_ms):.1f} "
        f"(medians of {len(spill_ms)}; {cache.summary()})"
    )


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_prefix_host_tier_e2e(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4 / P150x8")
    if not os.environ.get("HF_MODEL"):
        pytest.skip("needs HF_MODEL")
    from transformers import AutoTokenizer

    import models.demos.blackhole.qwen36.tests.test_spec_prefix_cache as spc
    from models.demos.blackhole.qwen36.tests.spec_helpers import _prompt_pool
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM

    monkeypatch.setenv("QWEN36_PREFIX_CACHE_SLOTS", "1")
    monkeypatch.setenv("QWEN36_PREFIX_HOST_SLOTS", "4")
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_DEBUG", "1")
    for k in ("QWEN36_PREFIX_HOST_SPILL_MIN_POS", "QWEN36_PREFIX_HOST_MIN_GAIN", "QWEN36_PREFIX_SNAPSHOT_MIN_LEN"):
        monkeypatch.delenv(k, raising=False)

    mesh_device.enable_program_cache()
    cls = Qwen36MTPContractForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=os.environ.get("HF_MODEL")),
        mesh_device,
        max_batch_size=spc.WIDTH,
        max_seq_len=spc.MAX_SEQ,
        tt_data_parallel=1,
        optimizations=None,
    )
    model = cls.model[0]
    tc = model.args.hf_config.get_text_config()
    n_kv = tc.num_key_value_heads
    kv_shape = (spc.NUM_BLOCKS, n_kv // min(model.num_devices, n_kv), spc.BLOCK, model.args.head_dim)
    kv = cls.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model._attention_layer_indices))
    for trace in (False, True):
        cls.warmup_model_prefill(enable_trace=trace, kv_cache=kv, can_sample_on_device=False)
        cls.warmup_model_decode(
            enable_trace=trace, kv_cache=kv, max_batch_size=spc.WIDTH, num_blocks=spc.BPU, can_sample_on_device=False
        )
    cache = model._prefix_cache
    assert cache is not None and cache.num_slots == 1 and cache.host_slots == 4

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    h = spc.Harness(cls, kv, _prompt_pool(tok, 16384))
    failures = []

    def reference(toks):
        model._prefix_cache = None  # plain full prefill, fresh blocks
        try:
            return h.prefill(toks, h.blocks_for(len(toks)), 0, 1)
        finally:
            model._prefix_cache = cache

    shared = h.text(4352)
    A = shared + h.text(700)
    a_blocks = h.blocks_for(len(A))
    h.prefill(A, a_blocks, 0, 0)  # saves A's last full-chunk snapshot (4096) in the only device slot
    sp0 = cache.stats.get("spills", 0)
    U = h.text(4800)
    h.prefill(U, h.blocks_for(len(U)), 0, 0)  # saves U@4096: evicts A@4096 to host
    assert cache.stats.get("spills", 0) > sp0, "A's snapshot was not spilled"
    hh0, rh0 = cache.stats.get("host_hits", 0), cache.stats.get("resume_host", 0)
    B = shared + h.text(600)
    hit = h.prefill(B, h.blocks_for(len(B), a_blocks[:68]), 4352, 0)
    assert cache.stats.get("host_hits", 0) == hh0 + 1, "B did not resume from the host tier"
    assert cache.stats.get("resume_host", 0) == rh0 + 1
    h.compare("host-hit", hit, reference(B), failures)
    logger.info(f"host tier e2e: {cache.summary()}")
    assert not failures, "; ".join(failures)
