# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""moe_compute ComputeOnly and FullLocal through the expert rows programs on one card.

On Blackhole, moe_compute runs the expert-row program plus a second program: metadata and placement
(ComputeOnly), or metadata, the double-buffer feed and selective_reduce_combine's own kernels (FullLocal);
TT_METAL_MOE_COMPUTE_KERNEL=ring|expert_rows forces a path and TT_METAL_MOE_COMPUTE_ROW_TILES=M the expert rows
path's row tiles per job (M > 1: weight-stationary jobs). These tests check, for both paths on the same inputs:
  - outputs 0-2 (counts, activation rows, e_t) against the 6U goldens, and equal between the two paths;
  - output 4: every row of the last two local experts (what moe_compute leaves in its double buffer) against an FP32
    golden on the BFP4-rounded weights;
  - FullLocal output 5 (the combine output): bitwise equal between the paths, and against the FP32 golden;
  - a repeated call and a traced call bitwise equal to the eager call, in every DEST mode and with the defaults;
  - rows (output 4, and every row through the FullLocal combine) of jobs of M row tiles bitwise equal to one row
    tile per job, in every DEST mode.
"""

import os
import random
from contextlib import contextmanager

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.ccl import MoEActivationFunction
from ttnn.experimental.moe_compute_utils import (
    auto_output_width_shard_dim,
    effective_matmul_ring_size,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
)
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
    compute_e_t_golden,
    compute_expert_activation_golden,
    compute_matmul_golden,
    compute_selective_tilize_golden,
    create_sharded_memory_config,
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    gen_expert_mapping,
    gen_sparse_buffer_and_indices,
    prepare_output_tensor_from_combine_writer,
    validate_activation,
    validate_e_t,
    validate_per_expert_tokens,
)
from ttnn.experimental.moe_compute_utils import (
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w0_w1_tensor_with_bias,
    prepare_w2_tensor_for_moe_compute,
    prepare_w2_tensor_with_bias,
)

DEVICE_PARAMS = [
    {"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 8 << 20}
]
KERNEL_ENV = "TT_METAL_MOE_COMPUTE_KERNEL"
ROW_TILES_ENV = "TT_METAL_MOE_COMPUTE_ROW_TILES"


@contextmanager
def _env(key, value):
    old = os.environ.get(key)
    os.environ[key] = str(value)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _kernel(name):
    return _env(KERNEL_ENV, name)


def _skip_unless_blackhole(mesh_device):
    """Blackhole runs the expert rows path by default; Wormhole only under TT_METAL_MOE_COMPUTE_KERNEL=expert_rows,
    which every test here sets for its expert rows calls."""
    if mesh_device.arch() not in (ttnn.device.Arch.BLACKHOLE, ttnn.device.Arch.WORMHOLE_B0):
        pytest.skip("the expert rows path runs on Blackhole and Wormhole")


def _bfp4_round(w):
    return ttnn.to_torch(ttnn.from_torch(w.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def _routing(kind, T, E, K, seed=2003):
    """Token-major top-k lists: random (uniform, distinct ids), skewed (expert 0 in every token, the upper half never
    chosen), empty (only the even experts), dup (one id twice per token, at random k), all_one (every k of every token
    names expert 0: K T entries for one expert, more than its e_t page holds), hot_last (the last two experts in every
    token: T rows each), cold_last (the last two experts in the first 20 tokens only, the others about
    T (K - 2) / (E - 2) rows each)."""
    rng = random.Random(seed)
    out = []
    for _ in range(T):
        if kind == "random":
            ids = rng.sample(range(E), K)
        elif kind == "skewed":
            ids = [0] + rng.sample(range(1, max(2, E // 2)), K - 1)
        elif kind == "empty":
            ids = rng.sample(range(0, E, 2), K)
        elif kind == "dup":
            ids = rng.sample(range(E), K - 1)
            ids.insert(rng.randrange(K), ids[rng.randrange(K - 1)])
        elif kind == "all_one":
            ids = [0] * K
        elif kind == "hot_last":
            ids = [E - 1, E - 2] + rng.sample(range(E - 2), K - 2)
            rng.shuffle(ids)
        elif kind == "cold_last":
            ids = ([E - 1, E - 2] + rng.sample(range(E - 2), K - 2)) if len(out) < 20 else rng.sample(range(E - 2), K)
            rng.shuffle(ids)
        else:
            raise ValueError(kind)
        out.append(ids)
    return out


class _Case:
    """One single-card ComputeOnly case: inputs on device, goldens, prepared weights."""

    def __init__(
        self,
        mesh_device,
        E,
        T,
        K,
        N,
        H,
        act=MoEActivationFunction.SILU,
        limit=None,
        has_bias=False,
        kind="random",
        ntp=4,
        layers=1,
        goldens=True,
    ):
        torch.manual_seed(2003)
        random.seed(2003)
        self.mesh_device, self.E, self.T, self.K, self.N, self.H = mesh_device, E, T, K, N, H
        self.act, self.limit, self.has_bias, self.ntp, self.L = act, limit, has_bias, ntp, layers
        mesh_shape = (1, 1)
        self.dp = auto_output_width_shard_dim(H)
        drain = ttnn.experimental.get_moe_tilize_drain_core(mesh_device, ntp, self.dp, H)
        drain_set = ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)})
        mapping = gen_expert_mapping(1, 1, None, E, E, E)
        if kind == "random":
            sparse, indices, scores, _ = gen_sparse_buffer_and_indices(T, H, E, K, mesh_shape, None)
        else:
            sparse = (torch.rand(1, T, H) - 0.5).bfloat16()
            indices = torch.tensor(_routing(kind, T, E, K), dtype=torch.int32).reshape(1, T, K).to(torch.uint16)
            raw = torch.rand(1, T, K) + 1e-3
            scores = (raw / raw.sum(-1, keepdim=True)).bfloat16()
        self.token_experts = indices.reshape(T, K).tolist()
        self.counts = self.golden = None
        if goldens:  # the 6U goldens hold at most T rows per expert
            tilize, self.counts = compute_selective_tilize_golden(sparse, indices, scores, mapping, mesh_shape, None)
            self.golden_act, _ = compute_expert_activation_golden(indices, scores, mapping, mesh_shape, None)
            self.golden_e_t, _ = compute_e_t_golden(indices, mapping, mesh_shape, None)

        w01_map, w2_map, dram_set = get_weight_core_shard_maps(mesh_device, H, N)
        L = layers
        w0, w1, w2 = create_torch_w0(L, E, H, N), create_torch_w1(L, E, H, N), create_torch_w2(L, E, N, H)
        b0 = b1 = b2 = None
        if has_bias:
            b0 = (torch.randn(L, E, N) * 0.12).to(torch.bfloat16)
            b1 = (torch.randn(L, E, N) * 0.12).to(torch.bfloat16)
            b2 = (torch.randn(L, E, H) * 0.12).to(torch.bfloat16)
        # FP32 golden on the BFP4-rounded weights (bias tiles are BFP4 too)
        rw = lambda w: torch.stack([torch.stack([_bfp4_round(w[l, e]) for e in range(E)]) for l in range(L)])  # noqa
        r0, r1, r2 = rw(w0), rw(w1), rw(w2)
        rb = lambda b: None if b is None else torch.stack([_bfp4_round(b[l].float()) for l in range(L)])  # noqa
        self.golden = (
            None
            if not goldens
            else compute_matmul_golden(
                tilize.unsqueeze(0).repeat([L] + [1] * tilize.dim()),
                r0,
                r1,
                r2,
                L,
                E,
                1,
                T,
                H,
                torch_b0=rb(b0),
                torch_b1=rb(b1),
                torch_b2=rb(b2),
                activation_type=act,
                activation_limit=limit,
            ).float()
        )
        m01, m2, _, _ = get_weight_mem_configs(L, E, H, N, w01_map, w2_map, dram_set, has_bias=has_bias)
        if has_bias:
            p01 = prepare_w0_w1_tensor_with_bias(w0, w1, b0, b1, L, E, H, N, w01_map)
            p2 = prepare_w2_tensor_with_bias(w2, b2, L, E, N, H, w2_map, w01_map)
        else:
            p01 = prepare_w0_w1_tensor_for_moe_compute(w0, w1, L, E, H, N, w01_map)
            p2 = prepare_w2_tensor_for_moe_compute(w2, L, E, N, H, w2_map, w01_map)
        # host BFP4 rounding, the same the golden's weights get
        upload = lambda t, mc: ttnn.from_torch(  # noqa: E731
            t, dtype=ttnn.bfloat4_b, device=mesh_device, layout=ttnn.TILE_LAYOUT, memory_config=mc
        )
        self.w01, self.w2 = upload(p01, m01), upload(p2, m2)
        self.x = ttnn.from_torch(
            sparse,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.idx = ttnn.from_torch(
            indices.reshape(1, T, K),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=create_sharded_memory_config(drain_set, [T, K], ttnn.uint16),
        )
        self.sc = ttnn.from_torch(
            scores.reshape(1, T, K),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=create_sharded_memory_config(drain_set, [T, K], ttnn.bfloat16),
        )
        self.map = ttnn.from_torch(
            mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.all_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.out_cores = ttnn.experimental.get_moe_combine_cores(mesh_device, ntp, self.dp, H)
        self.bbox = ttnn.experimental.get_moe_worker_mcast_bounding_box(mesh_device, ntp, self.dp, H)

    def call(self, compute_only=True, out=None, layer_id=0, **precision):
        return ttnn.experimental.moe_compute(
            self.x,
            self.idx,
            self.sc,
            self.map,
            self.w01,
            self.w2,
            layer_id=layer_id,
            output_height_shard_dim=self.ntp,
            intermediate_size=self.N,
            has_bias=self.has_bias,
            activation_type=self.act,
            activation_limit=self.limit,
            compute_only=compute_only,
            optional_output_tensor=out,
            **precision,
        )

    def new_combine_output(self):
        return ttnn.from_torch(
            torch.zeros([self.K, self.T, self.H], dtype=torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    def combine_golden(self):
        """[k, T, H]: page (k, t) holds token t's row of its k-th expert; pages of no routed row stay zero."""
        out = torch.zeros(self.K, self.T, self.H)
        for e in range(self.E):
            # one row per (token, k) naming e, in token order; a repeated id lands on its last k
            tokens = [t for t in range(self.T) for _ in range(self.token_experts[t].count(e))]
            for i, t in enumerate(tokens):
                k_last = len(self.token_experts[t]) - 1 - self.token_experts[t][::-1].index(e)
                out[k_last, t] = self.golden[0, 0, e, i]
        return out

    def combine_rel_l2(self, got):
        ref = self.combine_golden()
        return float((got.float() - ref).norm() / ref.norm())

    def read(self, outs, free=True, check=False):
        """Host copies: counts, activation, e_t (u32), and the last two experts' rows [2, rows, H] (bf16). check: the
        6U validators on outputs 0-2."""
        assert len(outs) == 5, f"compute_only must return 5 tensors, got {len(outs)}"
        dram = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (outs[0], outs[1], outs[2], outs[4])]
        if check:
            ok_counts = validate_per_expert_tokens(self.mesh_device, self.E, 1, dram[0], self.counts, self.bbox)
            ok_act = validate_activation(self.mesh_device, self.E, 1, dram[1], self.golden_act)
            ok_e_t = validate_e_t(self.mesh_device, self.T, self.E, 1, dram[2], self.golden_e_t)
            assert ok_counts and ok_act and ok_e_t, f"metadata: counts {ok_counts} activation {ok_act} e_t {ok_e_t}"
        host = [ttnn.to_torch(t) for t in dram]
        rows = None
        if self.counts is not None:
            bufc = torch.zeros(2, dtype=self.counts.dtype)
            for e in range(max(0, self.E - 2), self.E):
                bufc[e % 2] = self.counts[0, e]
            rows = prepare_output_tensor_from_combine_writer(
                host[3], bufc, 0, self.all_cores, self.out_cores, self.ntp, self.dp, 2, self.H
            )
        for t in dram:
            ttnn.deallocate(t)
        if free:
            for t in (outs[0], outs[1], outs[2], outs[4]):
                ttnn.deallocate(t)
        # activation rows: [token, k per expert, score bits per expert]; the row's alignment words are not defined
        words = 2 * self.E + 1
        act = host[1].reshape(self.T, -1)[:, :words]
        return dict(counts=host[0], act=act, e_t=host[2], rows=rows)

    def rows_rel_l2(self, got, layer=0):
        num = den = 0.0
        for e in range(max(0, self.E - 2), self.E):
            n = int(self.counts[0, e])
            ref = self.golden[layer, 0, e, :n].float()
            num += float((got["rows"][e % 2, :n].float() - ref).norm() ** 2)
            den += float(ref.norm() ** 2)
        return (num / den) ** 0.5 if den else 0.0

    def defined_equal(self, a, b):
        """Bitwise equality of everything the contract defines."""
        if not torch.equal(a["counts"][:, : self.E], b["counts"][:, : self.E]):
            return "counts"
        if not torch.equal(a["act"], b["act"]):
            return "activation"
        for e, lst in enumerate(self.golden_e_t[0].values()):
            n = len(lst) + 1
            if not torch.equal(a["e_t"][e, : 4 * n : 4], b["e_t"][e, : 4 * n : 4]):
                return f"e_t {e}"
        for e in range(max(0, self.E - 2), self.E):
            n = int(self.counts[0, e])
            if not torch.equal(a["rows"][e % 2, :n].view(torch.int16), b["rows"][e % 2, :n].view(torch.int16)):
                return f"rows {e}"
        return None


def _programs_added(mesh_device, fn):
    before = mesh_device.num_program_cache_entries()
    out = fn()
    return out, mesh_device.num_program_cache_entries() - before


def _expert_rows_and_ring(case, **precision):
    """One call on each path (fresh program cache): the expert rows path compiles its two programs, the ring path one;
    outputs 0-2 match the goldens on both; the last two experts' rows are bitwise equal between the paths (both
    accumulate every K row in the same order on the same weights)."""
    with _kernel("expert_rows"):
        outs, added = _programs_added(case.mesh_device, lambda: case.call(**precision))
        assert added == 2, f"expert rows call compiled {added} programs, expected 2 (expert rows + metadata)"
        js = case.read(outs, check=True)
    with _kernel("ring"):
        outs, added = _programs_added(case.mesh_device, lambda: case.call(**precision))
        assert added == 1, f"ring call compiled {added} programs, expected 1"
        ring = case.read(outs, check=True)
    assert torch.equal(js["counts"][:, : case.E], ring["counts"][:, : case.E]), "counts differ between the paths"
    for e in range(max(0, case.E - 2), case.E):
        n = int(case.counts[0, e])
        assert torch.equal(
            js["rows"][e % 2, :n].view(torch.int16), ring["rows"][e % 2, :n].view(torch.int16)
        ), f"expert {e}: rows of the expert rows path differ from the ring path's"
    return js, ring


def _full_local(case, out_before=None, **precision):
    """One FullLocal call on each path (fresh program cache): outputs 0-2 match the goldens on both; the combine output
    is bitwise equal between the paths (the combine copies rows that are bitwise equal). Returns the combine outputs."""
    res = {}
    for name, programs in (("expert_rows", 2), ("ring", 1)):
        with _kernel(name):
            out = case.new_combine_output()
            outs, added = _programs_added(case.mesh_device, lambda: case.call(compute_only=False, out=out, **precision))
            assert len(outs) == 6, f"FullLocal must return 6 tensors, got {len(outs)}"
            assert added == programs, f"{name} FullLocal call compiled {added} programs, expected {programs}"
            res[name] = ttnn.to_torch(outs[5]).reshape(case.K, case.T, case.H)
            res[name + "_outs"] = case.read(outs[:5], check=True)
            ttnn.deallocate(outs[5])
    assert torch.equal(res["expert_rows"].view(torch.int16), res["ring"].view(torch.int16)), "combine outputs differ"
    js, ring = res["expert_rows_outs"], res["ring_outs"]
    for e in range(max(0, case.E - 2), case.E):
        n = int(case.counts[0, e])
        assert torch.equal(
            js["rows"][e % 2, :n].view(torch.int16), ring["rows"][e % 2, :n].view(torch.int16)
        ), f"expert {e}: double-buffer rows differ between the paths"
    return res


# moe_compute's single-card shapes and the model shapes of the A/B (E local experts, T tokens, k, N, H, activation)
SHAPES = {
    "deepseek_e16_t32": (16, 32, 8, 2048, 7168, MoEActivationFunction.SILU, None),
    "qwen36_e32_t32": (32, 32, 8, 512, 2048, MoEActivationFunction.SILU, None),
    "qwen36_e32_t128": (32, 128, 8, 512, 2048, MoEActivationFunction.SILU, None),
    "flash_next_e128_t1": (128, 1, 8, 640, 2560, MoEActivationFunction.SILU, None),
    "glm53_e18_t32_clamped": (18, 32, 8, 2048, 4096, MoEActivationFunction.CLAMPED_SILU, 10.0),
    "gpt_oss_e16_t32_swiglu": (16, 32, 4, 2880, 2880, MoEActivationFunction.SWIGLU, None),
    "gemma_e8_t32_gelu": (8, 32, 8, 704, 2816, MoEActivationFunction.GELU, None),
}


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_shapes(mesh_device, mesh_shape, shape):
    _skip_unless_blackhole(mesh_device)
    E, T, K, N, H, act, limit = SHAPES[shape]
    case = _Case(mesh_device, E, T, K, N, H, act=act, limit=limit)
    js, ring = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel_js, rel_ring = case.rows_rel_l2(js), case.rows_rel_l2(ring)
    logger.info(f"{shape}: rows rel L2 expert rows {rel_js:.5f} ring {rel_ring:.5f} (HiFi4, FP32 DEST)")
    assert rel_js <= 0.01, f"expert rows rel L2 {rel_js}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("act", [MoEActivationFunction.SILU, MoEActivationFunction.SWIGLU], ids=["silu", "swiglu"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_bias(mesh_device, mesh_shape, act):
    _skip_unless_blackhole(mesh_device)
    H = 2880 if act == MoEActivationFunction.SWIGLU else 2048
    N = 2880 if act == MoEActivationFunction.SWIGLU else 1024
    case = _Case(mesh_device, 8, 32, 4, N, H, act=act, has_bias=True)
    js, ring = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel_js, rel_ring = case.rows_rel_l2(js), case.rows_rel_l2(ring)
    logger.info(f"bias {act}: rows rel L2 expert rows {rel_js:.5f} ring {rel_ring:.5f}")
    assert rel_js <= 0.01, f"expert rows rel L2 {rel_js}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("tokens", [1, 2, 3, 6, 16, 32, 48, 63, 64, 128])
@pytest.mark.parametrize("cfg", [(4, 4, 256, 512), (8, 4, 256, 320)], ids=["c4_h512", "c2_h320"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_tokens(mesh_device, mesh_shape, cfg, tokens):
    _skip_unless_blackhole(mesh_device)
    E, K, N, H = cfg
    N = max(N, 32 * effective_matmul_ring_size(mesh_device))
    case = _Case(mesh_device, E, tokens, K, N, H)
    js, ring = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    assert case.rows_rel_l2(js) <= 0.01


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("kind", ["skewed", "empty"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_routing(mesh_device, mesh_shape, kind):
    """Uneven experts (one expert in every token: jobs of 32 rows chained) and experts with no token."""
    _skip_unless_blackhole(mesh_device)
    case = _Case(mesh_device, 16, 128, 8, 512, 2048, kind=kind)
    js, _ = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    assert case.rows_rel_l2(js) <= 0.01


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_many_jobs_per_group(mesh_device, mesh_shape):
    """36 local experts, 128 tokens of top-8: more jobs than expert groups, every a2 slot reused."""
    _skip_unless_blackhole(mesh_device)
    case = _Case(mesh_device, 36, 128, 8, 2048, 4096)
    js, _ = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    assert case.rows_rel_l2(js) <= 0.01


PRECISION_MODES = {
    "default": {},
    "lofi_bf16": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False),
    "lofi_fp32": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
    "hifi2_bf16": dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
    "hifi4_fp32": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
}


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mode", sorted(PRECISION_MODES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_deterministic(mesh_device, mesh_shape, mode):
    """Eager, repeated eager and a trace replay give bitwise-equal outputs in every DEST mode and with the defaults."""
    _skip_unless_blackhole(mesh_device)
    precision = PRECISION_MODES[mode]
    case = _Case(mesh_device, 32, 128, 8, 512, 2048)
    with _kernel("expert_rows"):
        first = case.read(case.call(**precision), check=True)
        second = case.read(case.call(**precision))
        assert (diff := case.defined_equal(first, second)) is None, f"repeated eager call differs: {diff}"
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            outs = case.call(**precision)
        finally:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        traced = case.read(outs, free=False)
        ttnn.release_trace(mesh_device, tid)
    assert (diff := case.defined_equal(first, traced)) is None, f"traced call differs: {diff}"


FULL_LOCAL_SHAPES = [
    "deepseek_e16_t32",
    "flash_next_e128_t1",
    "gemma_e8_t32_gelu",
    "glm53_e18_t32_clamped",
    "gpt_oss_e16_t32_swiglu",
    "qwen36_e32_t128",
]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("shape", FULL_LOCAL_SHAPES)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_full_local(mesh_device, mesh_shape, shape):
    """FullLocal (compute_only=False on one card): the combine output of the expert rows path is bitwise equal to the
    ring path's and within 1 % rel L2 of the FP32 golden (HiFi4, FP32 DEST)."""
    _skip_unless_blackhole(mesh_device)
    E, T, K, N, H, act, limit = SHAPES[shape]
    case = _Case(mesh_device, E, T, K, N, H, act=act, limit=limit)
    res = _full_local(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel = case.combine_rel_l2(res["expert_rows"])
    logger.info(f"{shape}: FullLocal combine rel L2 {rel:.5f} (HiFi4, FP32 DEST)")
    assert rel <= 0.01, f"combine output rel L2 {rel}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("tokens", [1, 3, 32, 64])
@pytest.mark.parametrize("kind", ["random", "skewed", "empty"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_full_local_routing(mesh_device, mesh_shape, kind, tokens):
    """FullLocal over token counts and uneven experts (experts with no token, one expert in every token)."""
    _skip_unless_blackhole(mesh_device)
    case = _Case(mesh_device, 16, tokens, 8, 512, 2048, kind=kind)
    res = _full_local(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    assert case.combine_rel_l2(res["expert_rows"]) <= 0.01


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mode", sorted(PRECISION_MODES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_full_local_deterministic(mesh_device, mesh_shape, mode):
    """FullLocal: eager, repeated eager and a trace replay give bitwise-equal combine outputs in every DEST mode and
    with the defaults."""
    _skip_unless_blackhole(mesh_device)
    precision = PRECISION_MODES[mode]
    case = _Case(mesh_device, 32, 128, 8, 512, 2048)

    def run():
        out = case.new_combine_output()
        outs = case.call(compute_only=False, out=out, **precision)
        return outs, out

    def combine(outs):
        got = ttnn.to_torch(outs[5]).view(torch.int16)
        case.read(outs[:5])
        ttnn.deallocate(outs[5])
        return got

    with _kernel("expert_rows"):
        first = combine(run()[0])
        second = combine(run()[0])
        assert torch.equal(first, second), "repeated eager FullLocal call differs"
        out = case.new_combine_output()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            outs = case.call(compute_only=False, out=out, **precision)
        finally:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        traced = ttnn.to_torch(outs[5]).view(torch.int16)
        ttnn.release_trace(mesh_device, tid)
    assert torch.equal(first, traced), "traced FullLocal call differs"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_layer_id(mesh_device, mesh_shape):
    """Three layers of different weights: each layer_id's rows match that layer's golden, and the two expert rows
    programs compiled for the first layer serve the others (layer_id is a runtime offset, not a cache key)."""
    _skip_unless_blackhole(mesh_device)
    case = _Case(mesh_device, 8, 32, 8, 512, 2048, layers=3)
    precision = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    entries = []
    with _kernel("expert_rows"):
        for layer in (0, 1, 2, 1):
            got = case.read(case.call(layer_id=layer, **precision), check=True)
            entries.append(mesh_device.num_program_cache_entries())
            rel = case.rows_rel_l2(got, layer)
            other = case.rows_rel_l2(got, (layer + 1) % 3)
            logger.info(f"layer {layer}: rows rel L2 {rel:.5f} (against layer {(layer + 1) % 3}: {other:.3f})")
            assert rel <= 0.01, f"layer {layer}: rows rel L2 {rel}"
            assert other > 0.5, f"layer {layer}: the rows also match layer {(layer + 1) % 3}"
    assert entries[1:] == [entries[0]] * 3, f"program cache entries per layer: {entries}"


def _not_routed_as_k1(act, K):
    """moe_compute writes K + 1 from one RISC and K from the other for an expert a token does not route to."""
    out = act.to(torch.int64)
    E = (out.shape[1] - 1) // 2
    ks = out[:, 1 : 1 + E]
    ks[ks >= K] = K + 1
    return out


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("compute_only", [True, False], ids=["compute_only", "full_local"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_duplicate_ids(mesh_device, mesh_shape, compute_only):
    """A token naming one expert at two k is listed once per k, as moe_compute lists it: counts and the e_t page hold
    every (token, k), the activation row holds the last k, each listing has its row, and the combine writes page
    (last k, t). Checked against the routing itself (the ring path's e_t pages put a repeated token out of order),
    counts against the ring path's."""
    _skip_unless_blackhole(mesh_device)
    case = _Case(mesh_device, 16, 32, 8, 512, 2048, kind="dup")
    precision = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    with _kernel("expert_rows"):
        out = None if compute_only else case.new_combine_output()
        outs = case.call(compute_only=compute_only, out=out, **precision)
        combine = None
        if not compute_only:
            combine = ttnn.to_torch(outs[5]).reshape(case.K, case.T, case.H)
            ttnn.deallocate(outs[5])
        js = case.read(outs[:5])
    with _kernel("ring"):
        ring = case.read(case.call(**precision))
    lists = [[t for t in range(case.T) for _ in range(case.token_experts[t].count(e))] for e in range(case.E)]
    counts = [len(lst) for lst in lists]
    assert js["counts"][0, : case.E].tolist() == counts, "counts: one per (token, k)"
    assert torch.equal(js["counts"][:, : case.E], ring["counts"][:, : case.E]), "counts differ from the ring path's"
    for e in range(case.E):
        n = counts[e]
        assert js["e_t"][e, : 4 * n : 4].tolist() == lists[e], f"e_t page {e}"
    act = js["act"].to(torch.int64)
    rows = [t for t in range(case.T) if any(e < case.E for e in case.token_experts[t])]
    for r, t in enumerate(rows):
        assert int(act[r, 0]) == t, f"activation row {r} is token {int(act[r, 0])}, expected {t}"
        for e in range(case.E):
            ks = [k for k, x in enumerate(case.token_experts[t]) if x == e]
            got = int(act[r, 1 + e])
            assert (got == ks[-1]) if ks else (got >= case.K), f"token {t} expert {e}: k {got}, listed at {ks}"
    assert case.rows_rel_l2(js) <= 0.01
    if combine is not None:
        assert case.combine_rel_l2(combine) <= 0.01, "combine output"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("compute_only", [True, False], ids=["compute_only", "full_local"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_listing_over_tokens(mesh_device, mesh_shape, compute_only):
    """Every k of every token names expert 0 (K T entries). moe_compute's ring overruns expert 0's e_t page (and its
    FullLocal does not finish); the expert rows path keeps the first T entries in token order (tokens 0 .. T / K - 1,
    K times each), reports that count, and finishes."""
    _skip_unless_blackhole(mesh_device)
    T, K = 32, 8
    case = _Case(mesh_device, 8, T, K, 512, 2048, kind="all_one", goldens=False)
    with _kernel("expert_rows"):
        out = None if compute_only else case.new_combine_output()
        outs = case.call(compute_only=compute_only, out=out)
        if not compute_only:
            ttnn.deallocate(outs[5])
        got = case.read(outs[:5])
    assert int(got["counts"][0, 0]) == T, f"expert 0 keeps {int(got['counts'][0, 0])} rows, expected {T}"
    assert all(int(got["counts"][0, e]) == 0 for e in range(1, case.E))
    expected = [t for t in range(T // K) for _ in range(K)]
    assert got["e_t"][0, : 4 * T : 4].tolist() == expected, "expert 0's page: the first T entries in token order"
    assert int(got["e_t"][0, 4 * T]) & 0xFFFFFFFF == 0xFFFFFFFF, "page terminator"


# weight-stationary jobs: (E, T, K, N, H, activation, limit, bias); hot_last routing gives the two experts output 4
# keeps T rows (jobs of several row tiles), cold_last gives them 20 (one-row-tile jobs among jobs of several)
ROW_TILE_SHAPES = {
    "deepseek_e16_t128": (16, 128, 8, 2048, 7168, MoEActivationFunction.SILU, None, False),
    "glm53_e18_t96_clamped": (18, 96, 8, 2048, 4096, MoEActivationFunction.CLAMPED_SILU, 10.0, False),
    "qwen36_e32_t128": (32, 128, 8, 512, 2048, MoEActivationFunction.SILU, None, False),
    "gpt_oss_e8_t128_bias": (8, 128, 4, 2880, 2880, MoEActivationFunction.SWIGLU, None, True),
}


def _rows_with(case, row_tiles, **precision):
    """One ComputeOnly expert rows call with jobs of `row_tiles` row tiles (fresh program cache), read back and
    checked."""
    case.mesh_device.disable_and_clear_program_cache()
    case.mesh_device.enable_program_cache()
    with _kernel("expert_rows"), _env(ROW_TILES_ENV, row_tiles):
        outs, added = _programs_added(case.mesh_device, lambda: case.call(**precision))
        assert added == 2, f"row tiles {row_tiles}: compiled {added} programs, expected the 2 expert rows programs"
        return case.read(outs, check=True)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("kind", ["hot_last", "cold_last"])
@pytest.mark.parametrize("row_tiles", [2, 4])
@pytest.mark.parametrize("shape", sorted(ROW_TILE_SHAPES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_row_tiles(mesh_device, mesh_shape, shape, row_tiles, kind):
    """Jobs of several row tiles (each expert's weights read once for up to 32 M rows), and jobs of one row tile in the
    same program, give the rows of one row tile per job bitwise, and those equal the ring's."""
    _skip_unless_blackhole(mesh_device)
    E, T, K, N, H, act, limit, bias = ROW_TILE_SHAPES[shape]
    case = _Case(mesh_device, E, T, K, N, H, act=act, limit=limit, has_bias=bias, kind=kind)
    precision = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    one = _rows_with(case, 1, **precision)
    many = _rows_with(case, row_tiles, **precision)
    assert (diff := case.defined_equal(one, many)) is None, f"{row_tiles} row tiles per job differ: {diff}"
    with _kernel("ring"):
        ring = case.read(case.call(**precision), check=True)
    # against the ring: counts and rows, as _expert_rows_and_ring (outputs 1 / 2 are checked against the goldens above)
    assert torch.equal(ring["counts"][:, :E], many["counts"][:, :E]), "counts differ from the ring's"
    for e in range(E - 2, E):
        n = int(case.counts[0, e])
        assert torch.equal(
            ring["rows"][e % 2, :n].view(torch.int16), many["rows"][e % 2, :n].view(torch.int16)
        ), f"expert {e}: {row_tiles} row tiles per job differ from the ring's rows"
    rel = case.rows_rel_l2(many)
    logger.info(f"{shape} {kind} M {row_tiles}: rows rel L2 {rel:.5f} (HiFi4, FP32 DEST)")
    assert rel <= 0.01, f"rows rel L2 {rel}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("kind", ["hot_last", "cold_last"])
@pytest.mark.parametrize("mode", sorted(PRECISION_MODES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_row_tiles_deterministic(mesh_device, mesh_shape, mode, kind):
    """In every DEST mode: jobs of 4 row tiles (and of one row tile in the same program) give one row tile per job's
    rows bitwise; eager, repeated eager and two trace replays are bitwise equal."""
    _skip_unless_blackhole(mesh_device)
    precision = PRECISION_MODES[mode]
    case = _Case(mesh_device, 16, 128, 8, 2048, 7168, kind=kind)
    one = _rows_with(case, 1, **precision)
    first = _rows_with(case, 4, **precision)
    assert (diff := case.defined_equal(one, first)) is None, f"4 row tiles per job differ from 1: {diff}"
    with _kernel("expert_rows"), _env(ROW_TILES_ENV, 4):
        second = case.read(case.call(**precision))
        assert (diff := case.defined_equal(first, second)) is None, f"repeated eager call differs: {diff}"
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            outs = case.call(**precision)
        finally:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        traced = case.read(outs, free=False)
        ttnn.release_trace(mesh_device, tid)
    assert (diff := case.defined_equal(first, traced)) is None, f"traced call differs: {diff}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mode", ["hifi4_fp32", "lofi_bf16"])
@pytest.mark.parametrize("kind", ["random", "hot_last", "cold_last"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_row_tiles_full_local(mesh_device, mesh_shape, kind, mode):
    """FullLocal: the combine output (every routed row of every expert) with jobs of 4 row tiles is bitwise equal to
    one row tile per job and to the ring path's."""
    _skip_unless_blackhole(mesh_device)
    precision = PRECISION_MODES[mode]
    case = _Case(mesh_device, 16, 128, 8, 2048, 7168, kind=kind)
    res = _full_local(case, **precision)
    case.mesh_device.disable_and_clear_program_cache()
    case.mesh_device.enable_program_cache()
    with _kernel("expert_rows"), _env(ROW_TILES_ENV, 4):
        out = case.new_combine_output()
        outs, added = _programs_added(case.mesh_device, lambda: case.call(compute_only=False, out=out, **precision))
        assert added == 2, f"FullLocal call with 4 row tiles per job compiled {added} programs, expected 2"
        many = ttnn.to_torch(outs[5]).reshape(case.K, case.T, case.H)
        case.read(outs[:5], check=True)
        ttnn.deallocate(outs[5])
    assert torch.equal(
        many.view(torch.int16), res["expert_rows"].view(torch.int16)
    ), "combine output differs from M = 1"
