# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ring joint SDPA with the K split (program_config.ring_k_split; the op merges the partitions with
ttnn.transformer.sdpa_k_split_merge), MiMo GA shapes on the 2x2: 64 Q / 4 KV heads (per chip 32 / 2), qk 192, v 128,
HiFi2, a random bf8 KV cache and a chunk at ``kv_actual``.

* test_sdpa_ksplit_accuracy: every split vs the unsplit op (PCC) and vs an fp32 host reference for chip (0, 0)
  (PCC, relative error, norm ratio), over chunk sizes, contexts, split counts and Q / K chunk sizes.
* test_sdpa_ksplit_determinism: repeated calls bitwise identical, compared on device.
* test_sdpa_ksplit_perf: real-time profiler device time of the ring op and of the merge, against baselines.

    scripts/run_safe_pytest.sh models/demos/mimo_v2_d_p/tests/perf/test_sdpa_ksplit.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import MiMoKVCache, _cache_mem
from models.demos.mimo_v2_d_p.tt.attention.sdpa import ring_attention, ring_program_config
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler

NQ, NKV, DK, DV = 64, 4, 192, 128
SCALE = DK**-0.5


class _Setup:
    """Q chunk + random bf8 KV cache (same on every chip, so every split reads the same data)."""

    def __init__(self, mesh_device, device_params, chunk_local, ctx, seed=0):
        torch.manual_seed(seed)
        self.mesh = mesh_device
        self.sp, self.tp = tuple(mesh_device.shape)
        self.chunk_local, self.chunk = chunk_local, chunk_local * self.sp
        sp_topo, _ = per_axis_topology(device_params["fabric_config"])
        self.ccl = CCLManager(mesh_device, num_links=MiMoRuntimeOptions().num_links, topology=sp_topo)
        max_seq = (ctx + self.chunk - 1) // self.chunk * self.chunk
        self.kv_actual = max_seq - self.chunk
        self.nkv_l = NKV // self.tp
        cache = lambda d: ttnn.from_torch(
            torch.randn(1, self.nkv_l, max_seq // self.sp, d),
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_cache_mem(mesh_device, d),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.kv = MiMoKVCache(cache(DK), cache(DV), 1, 1, max_seq, self.sp, self.nkv_l, DK, DV)
        self.q_host = torch.randn(1, NQ, self.chunk, DK)
        self.q = ttnn.from_torch(
            self.q_host,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(self.sp, self.tp), dims=(2, 1)),
        )

    def run(self, k_split, q_chunk=None, k_chunk=None, two_level=False, two_level_fold=0):
        pc = ring_program_config(
            self.mesh,
            q_chunk=q_chunk,
            k_chunk=k_chunk,
            q_local=self.chunk_local,
            kv_local=self.kv.max_seq_len // self.sp,
            k_split=k_split,
            two_level=two_level,
            two_level_fold=two_level_fold,
        )
        return ring_attention(
            self.q,
            self.kv,
            kv_actual=self.kv_actual,
            logical_n=self.kv_actual + self.chunk,
            window=None,
            sink=None,
            layer_slot=0,
            mesh_device=self.mesh,
            ccl_manager=self.ccl,
            sp_axis=0,
            scale=SCALE,
            program_config=pc,
        )

    def reference_chip0(self):
        """fp32 attention for chip (0, 0): q heads 0 .. NQ/tp - 1, q rows 0 .. chunk_local - 1 of the chunk at global
        positions kv_actual + i; key at global position p = local cache row (p // chunk) * chunk_local + p %
        chunk_local (block-cyclic, every chip holds the same cache); causal keys p <= query position."""
        k_all = ttnn.to_torch(ttnn.get_device_tensors(self.kv.k)[0]).float()[0]  # bf8 values
        v_all = ttnn.to_torch(ttnn.get_device_tensors(self.kv.v)[0]).float()[0]
        pos = torch.arange(self.kv_actual + self.chunk)
        rows = (pos // self.chunk) * self.chunk_local + pos % self.chunk_local
        nq_l = NQ // self.tp
        qh = self.q_host[0, :nq_l, : self.chunk_local].bfloat16().double()
        mask = pos[None, :] <= (self.kv_actual + torch.arange(self.chunk_local))[:, None]
        g = nq_l // self.nkv_l
        ref = torch.empty(nq_l, self.chunk_local, DV, dtype=torch.float64)
        for h in range(nq_l):
            sc = (qh[h] @ k_all[h // g, rows].double().T) * SCALE
            ref[h] = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1) @ v_all[h // g, rows].double()
        return ref


def _stats(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (
        torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
        ((a - b).norm() / b.norm()).item(),
        (a.norm() / b.norm()).item(),
    )


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize(
    "chunk_local, ctx, splits, q_chunk, k_chunk",
    [
        (2048, 32768, (2, 3), None, None),  # QB production (split 2 at 2048 tok/chip)
        (640, 32768, (3,), None, None),  # Galaxy-sized slab (split 3)
        (2048, 65536, (2, 4), None, None),
        (2048, 32768, (2, 5), 64, 512),  # other chunk sizes; 5 partitions: the merge's row-sum path
        (1024, 16384, (2, 6), 256, 512),
    ],
    ids=["C2048-32K", "C640-32K", "C2048-64K", "C2048-32K-q64k512", "C1024-16K-q256k512"],
)
def test_sdpa_ksplit_accuracy(mesh_device, device_params, chunk_local, ctx, splits, q_chunk, k_chunk):
    st = _Setup(mesh_device, device_params, chunk_local, ctx)
    to_host = lambda t: torch.cat([ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)])
    base = to_host(st.run(1, q_chunk, k_chunk))
    ref = st.reference_chip0()
    p0, rel0, nr0 = _stats(base[0], ref)
    logger.info(f"split 1 vs fp32 (chip 0): PCC {p0:.6f} rel {rel0:.2e} norm ratio {nr0:.5f}")
    for s in splits:
        out = to_host(st.run(s, q_chunk, k_chunk))
        p, rel, nr = _stats(out[0], ref)
        pu, relu, _ = _stats(out, base)
        logger.info(
            f"split {s} vs fp32 (chip 0): PCC {p:.6f} rel {rel:.2e} norm ratio {nr:.5f}; vs split 1: PCC {pu:.6f} "
            f"rel {relu:.2e}"
        )
        assert pu > 0.999, (s, pu)
        assert rel <= rel0 * 1.1 + 1e-3, (s, rel, rel0)  # the split must not be less accurate than the unsplit op
        assert abs(nr - 1) < 5e-3, (s, nr)


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("fold", [0, 4], ids=["per-iteration", "fold4"])
def test_sdpa_two_level_accuracy(mesh_device, device_params, fold):
    """program_config.ring_two_level (unsplit op) runs and stays close to the one-level op vs fp32. Random scores do
    not show the long-context bf16 row-sum drift two levels exist for (on them it is ~5% worse in relative error)."""
    st = _Setup(mesh_device, device_params, 2048, 65536)
    ref = st.reference_chip0()
    chip0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()[0]
    _, rel1, _ = _stats(chip0(st.run(1)), ref)
    p, rel, nr = _stats(chip0(st.run(1, two_level=True, two_level_fold=fold)), ref)
    logger.info(
        f"two-level (fold {fold}) vs fp32: PCC {p:.6f} rel {rel:.2e} norm ratio {nr:.5f}; one-level rel {rel1:.2e}"
    )
    assert p > 0.999 and rel <= rel1 * 1.15 and abs(nr - 1) < 5e-3, (p, rel, rel1, nr)


def _mismatch_marker(reference, actual):
    """On-device exact compare (as tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py): one element per chip,
    non-zero iff any element differs; outputs never leave the device."""
    return ttnn.max(ttnn.ne(reference, actual, dtype=ttnn.bfloat16))


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("chunk_local, split", [(2048, 2), (640, 3)], ids=["C2048-s2", "C640-s3"])
def test_sdpa_ksplit_determinism(mesh_device, device_params, chunk_local, split):
    st = _Setup(mesh_device, device_params, chunk_local, 32768)
    reference = st.run(split)
    marker = None
    for _ in range(19):
        out = st.run(split)
        m = _mismatch_marker(reference, out)
        marker = m if marker is None else ttnn.maximum(marker, m)
        out.deallocate(True)
    host = ttnn.from_device(marker)
    bad = [i for i, t in enumerate(ttnn.get_device_tensors(host)) if float(ttnn.to_torch(t).item()) != 0.0]
    assert not bad, f"ring SDPA k_split={split} not deterministic on chips {bad}"


# Device ns (max over chips) of the ring op and of the merge, median of 3; BH QuietBox 2x2.
# (chunk_local, ctx, split) -> (ring ns, merge ns). Recalibrate from the "RT-CAL" lines.
_PERF_EXPECTED_NS = {  # 2026-09-29
    (2048, 32768, 1): (8_115_081, None),
    (2048, 32768, 2): (7_443_270, 181_670),
    (2048, 131072, 1): (31_928_165, None),
    (2048, 131072, 2): (29_293_972, 184_332),
    (640, 32768, 1): (2_893_136, None),
    (640, 32768, 3): (2_591_908, 88_367),
}
_PERF_MARGIN = 0.03


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize(
    "chunk_local, ctx, split",
    [(2048, 32768, 1), (2048, 32768, 2), (2048, 131072, 1), (2048, 131072, 2), (640, 32768, 1), (640, 32768, 3)],
    ids=lambda v: str(v),
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_sdpa_ksplit_perf(mesh_device, device_params, chunk_local, ctx, split):
    require_realtime_profiler("ring SDPA k split perf checks")
    st = _Setup(mesh_device, device_params, chunk_local, ctx)
    key = (chunk_local, ctx, split)
    expected = _PERF_EXPECTED_NS.get(key)
    run = lambda: st.run(split).deallocate(True)
    parts = [("ring", "compute/ring_joint_sdpa.cpp")] + ([("merge", "compute/ksplit_merge.cpp")] if split > 1 else [])
    for i, (name, kernel) in enumerate(parts):
        exp = expected[i] if expected else None  # (ring ns, merge ns)
        assert_op_duration_merged(
            mesh_device,
            run,
            kernel,
            expected_ns=exp or 1,
            margin=_PERF_MARGIN if exp else float("inf"),
            label=f"{key} {name}",
            iters=3,
        )
    if expected is None:
        pytest.skip(f"no baseline for {key}; add it to _PERF_EXPECTED_NS")
