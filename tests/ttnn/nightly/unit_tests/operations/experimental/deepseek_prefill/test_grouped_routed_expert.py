# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the GROUPED unified_routed_expert_moe program factory (num_row_groups > 0).

Several experts on ONE chip with different token counts, laid out production-style (32-row aligned
regions back to back with random slack between them). The grid is split into row groups that run
experts concurrently; every expert's output must land in its own region and match a torch
reference, for both x layouts, both weight dtypes, and every (rows, groups) geometry the op
accepts. Zero-token experts, more experts than groups, a single non-empty expert, an all-empty
call, a count exceeding the region (clamp) and repeated dispatch with different counts
(program-cache hit path) are covered.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc

TILE = 32
MAX_TOKENS = 5120  # production m_tiles (5120 / 32 = 160)

MODELS = {
    # emb, hidden, activation
    "kimi": (7168, 2048, ttnn.RoutedExpertActivation.Silu),
    "m3": (6144, 3072, ttnn.RoutedExpertActivation.SwiGluOai),
}

DISTS = {
    "kimi_u12": ("kimi", [107] * 12),
    "kimi_zipf": ("kimi", [640, 320, 224, 160, 128, 96, 64, 64, 32, 32, 0, 0]),
    "kimi_zeros": ("kimi", [0, 320, 0, 160, 96, 0, 0, 64, 0, 0, 0, 32]),
    "kimi_e3": ("kimi", [100, 200, 300]),
    "kimi_single": ("kimi", [0, 0, 160, 0, 0, 0, 0, 0]),
    "kimi_giant": ("kimi", [2048] + [64] * 11),
    "m3_u4": ("m3", [160] * 4),
    "m3_skew8": ("m3", [800, 400, 200, 100, 50, 25, 0, 5]),
    "m3_u16": ("m3", [160] * 16),
}


def _torch_ref(x, w, activation):
    gate = F.linear(x, w["gate_proj"])
    up = F.linear(x, w["up_proj"])
    if activation == ttnn.RoutedExpertActivation.Silu:
        act = F.silu(gate) * up
    else:  # SwiGluOai (alpha=1.702, limit=7.0 baked into the kernel)
        gate_c = gate.clamp(max=7.0)
        up_c = up.clamp(min=-7.0, max=7.0)
        act = (up_c + 1.0) * (gate_c * torch.sigmoid(1.702 * gate_c))
    return F.linear(act, w["down_proj"])


class GroupedCase:
    def __init__(self, device, counts, model, weights_dtype, x_row_major, seed=123, **ffn_kwargs):
        torch.manual_seed(seed)
        self.emb, self.hidden, self.activation = MODELS[model]
        self.counts = list(counts)
        self.weights_dtype = weights_dtype
        offs, cur = [], 0
        for c in self.counts:
            offs.append(cur)
            cur += (c + TILE - 1) // TILE * TILE + TILE  # region + one tile-row of slack
        self.offsets = offs
        rows = max(cur, MAX_TOKENS)
        self.weights = [
            {
                "gate_proj": torch.randn(self.hidden, self.emb) * 0.02,
                "up_proj": torch.randn(self.hidden, self.emb) * 0.02,
                "down_proj": torch.randn(self.emb, self.hidden) * 0.02,
            }
            for _ in self.counts
        ]
        buf = torch.randn(rows, self.emb)  # slack must never be read
        self.inputs = []
        for e, c in enumerate(self.counts):
            x = torch.randn(c, self.emb)
            self.inputs.append(x)
            if c > 0:
                buf[offs[e] : offs[e] + c] = x
        self.buf = buf
        if x_row_major:
            self.tt_buf = ttnn.from_torch(
                buf,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                dtype=ttnn.bfloat16,
            )
        else:
            self.tt_buf = ttnn.from_torch(
                buf,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=ttnn.bfloat8_b,
            )
        self.device = device
        self.expert = TtRoutedExpert(
            mesh_device=device,
            experts_per_chip=len(self.counts),
            global_expert_idx_table=self.idx(list(range(len(self.counts)))),
            emb_dim=self.emb,
            hidden_dim=self.hidden,
            max_tokens=MAX_TOKENS,
            torch_weights=self.weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            activation=self.activation,
            **ffn_kwargs,
        )

    def idx(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )

    def run(self, counts=None):
        counts = self.counts if counts is None else counts
        return self.expert(self.tt_buf, self.idx(counts), self.idx(self.offsets))

    def check(self, out, counts=None, pcc_threshold=0.97):
        counts = self.counts if counts is None else counts
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        for e, c in enumerate(counts):
            if c <= 0:
                continue
            c_eff = min(c, self.counts[e])  # reference rows available
            ref = _torch_ref(self.inputs[e][:c_eff], self.weights[e], self.activation)
            got = out_t[self.offsets[e] : self.offsets[e] + c_eff].float()
            passing, pcc = comp_pcc(ref, got, pcc_threshold)
            logger.info(f"expert {e} count={c} pcc={pcc:.5f}")
            assert not torch.isnan(got).any() and not torch.isinf(got).any(), f"expert {e}: NaN/Inf"
            assert passing, f"expert {e} count={c}: PCC {pcc} < {pcc_threshold}"


GEOMETRIES = [
    # (num_row_groups, grid_rows)
    pytest.param(10, 10, id="G10r10"),
    pytest.param(5, 10, id="G5r10"),
    pytest.param(8, 8, id="G8r8"),
    pytest.param(4, 8, id="G4r8"),
    pytest.param(2, 8, id="G2r8"),
]


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("dist", list(DISTS))
@pytest.mark.parametrize("num_row_groups, grid_rows", GEOMETRIES)
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b], ids=["bf4", "bf8"])
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_grouped_routed_expert(device, dist, num_row_groups, grid_rows, weights_dtype, x_row_major):
    if grid_rows > device.compute_with_storage_grid_size().y:
        pytest.skip("grid too small")
    model, counts = DISTS[dist]
    case = GroupedCase(
        device, counts, model, weights_dtype, x_row_major, ffn_num_row_groups=num_row_groups, ffn_grid_rows=grid_rows
    )
    out = case.run()
    case.check(out)


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "num_row_groups, grid_rows", [pytest.param(10, 10, id="G10r10"), pytest.param(4, 8, id="G4r8")]
)
def test_grouped_routed_expert_cache_hit_varying_counts(device, num_row_groups, grid_rows):
    """Same program, different device-side counts each dispatch: the LPT assignment and chunk
    geometry are recomputed on device, so the program cache must be hit and results stay right."""
    if grid_rows > device.compute_with_storage_grid_size().y:
        pytest.skip("grid too small")
    model, counts = DISTS["kimi_zipf"]
    case = GroupedCase(
        device, counts, model, ttnn.bfloat4_b, True, ffn_num_row_groups=num_row_groups, ffn_grid_rows=grid_rows
    )
    out = case.run()
    case.check(out)
    entries = device.num_program_cache_entries()
    for new_counts in ([107] * 12, [0] * 12, [32, 0, 640, 0, 160, 0, 96, 0, 64, 0, 0, 0], counts):
        out = case.run(new_counts)
        if any(c > 0 for c in new_counts):
            case.check(out, new_counts)
        assert device.num_program_cache_entries() == entries, "program cache miss on a counts-only change"


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
def test_grouped_routed_expert_all_empty(device):
    """Every expert empty: every row group has nothing to do and the program must still terminate."""
    if device.compute_with_storage_grid_size().y < 10:
        pytest.skip("grid too small")
    case = GroupedCase(device, [0] * 12, "kimi", ttnn.bfloat4_b, True, ffn_num_row_groups=10, ffn_grid_rows=10)
    case.run()
    ttnn.synchronize_device(device)


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
def test_grouped_routed_expert_count_clamp(device):
    """A device count larger than the region is clamped to the region (m_tiles): the first 5120 rows
    of the expert are computed correctly and the excess is dropped, on every core consistently."""
    counts = [5120, 107, 107, 107]
    case = GroupedCase(device, counts, "kimi", ttnn.bfloat4_b, True, ffn_num_row_groups=4, ffn_grid_rows=8)
    out = case.run([6000, 107, 107, 107])
    case.check(out, [5120, 107, 107, 107])


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
def test_legacy_path_unchanged(device):
    """num_row_groups=0 (default) selects the legacy factory and still passes on the same inputs."""
    model, counts = DISTS["kimi_zeros"]
    case = GroupedCase(device, counts, model, ttnn.bfloat4_b, True)
    case.check(case.run())
