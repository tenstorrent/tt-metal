# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the ``sparse_ep`` MoE path — the production multi-device path.

The routed-expert list is partitioned across an "ep" mesh axis
(``SparseMoEEP``), each chip running ``E / D_ep`` experts and all-reducing the
partial outputs. One set of host weights is loaded into

  * a replicated dense ``MoE`` reference holding all E experts, and
  * an EP-sharded ``SparseMoEEP`` (expert ``e`` lives on shard ``e // e_local``),

both are run on the same replicated input, and the (replicated) forward output
and gate gradient must match within PCC bounds.

Blackhole only: every bundled MGD and every measured EP configuration is
Blackhole, so the module skips on other archs rather than exercising an untuned
path.

Runs on a full 32-chip Blackhole galaxy as an ``8 x 4`` mesh (DP=8, EP=4). The
module-scoped fixture skips if that mesh cannot be opened, so it is inert on
smaller boards. The bundled ``bh_galaxy_8_4_torus_x`` descriptor is used unless
``TT_MESH_GRAPH_DESC_PATH`` is already set.

Multi-device only: this module opens a mesh, so it must not share a process with
tests that open a default single-device context first — reopening the device as
a mesh afterwards leaves the fabric routers half-initialized.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tests.ttnn.utils_for_testing import assert_with_pcc  # noqa: E402

import ttnn
import ttml
from ttml.models.deepseek.moe import MoE
from ttml.models.deepseek.moe_sparse_ep import SparseMoEEP

SEED = 2026

# Fixed 8x4 mesh: the full 32-chip Blackhole galaxy, DP=8 x EP=4.
# EP=4 splits the 8 routed experts 2-per-shard, and DP=8 means EP sharding is
# exercised alongside a real data-parallel axis.
DP_AXIS_SIZE = 8
EP_AXIS_SIZE = 4
MESH_SHAPE = (DP_AXIS_SIZE, EP_AXIS_SIZE)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Blackhole galaxy only — see the module docstring. The galaxy fabric is a torus
# in X; a LINE/LINE descriptor faults with SIGBUS, so this must be the torus_x one.
_MGD_FOR_SHAPE = {
    MESH_SHAPE: os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_8_4_torus_x.textproto"),
}


class _Cfg:
    """Minimal MoE config (only the fields the MoE paths read)."""

    def __init__(self, **kw):
        self.dim = kw.get("dim", 64)
        self.moe_inter_dim = kw.get("moe_inter_dim", 64)
        self.n_routed_experts = kw.get("n_routed_experts", 8)
        self.n_activated_experts = kw.get("n_activated_experts", 2)
        self.n_shared_experts = kw.get("n_shared_experts", 0)
        self.n_expert_groups = kw.get("n_expert_groups", 1)
        self.n_limited_groups = kw.get("n_limited_groups", 1)
        self.score_func = kw.get("score_func", "sigmoid")
        self.route_scale = kw.get("route_scale", 1.0)
        # Read by MoE.__init__ to decide EP sharding across moe_axis_name.
        self.moe_type = kw.get("moe_type", "sparse_ep")
        self.moe_axis_name = kw.get("moe_axis_name", "ep")


# ---------------------------------------------------------------------------
# Multi-device mesh fixture
# ---------------------------------------------------------------------------


def _is_blackhole() -> bool:
    try:
        return "blackhole" in ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return False


# Blackhole-only: the bundled MGDs and the measured EP configurations are all
# Blackhole, so on any other arch this module skips rather than silently
# exercising an untuned path.
pytestmark = [
    pytest.mark.requires_device,
    pytest.mark.skipif(
        not _is_blackhole(),
        reason="sparse_ep MoE is supported/validated on Blackhole only",
    ),
]


def _close_device_quietly() -> None:
    try:
        ttml.autograd.AutoContext.get_instance().close_device()
    except Exception:  # noqa: BLE001
        pass


def _ensure_mgd_path(shape: tuple[int, ...]) -> Optional[str]:
    """Point TT_MESH_GRAPH_DESC_PATH at a bundled MGD if unset. Returns the old value."""
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if previous:
        return previous
    candidate = _MGD_FOR_SHAPE.get(shape)
    if candidate and os.path.isfile(candidate):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = candidate
    return previous


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


@pytest.fixture(scope="module")
def ep_mesh():
    """Open the ``[8, 4]`` galaxy mesh with axes ``("dp", "ep")``."""
    shape = MESH_SHAPE
    previous_mgd = _ensure_mgd_path(shape)

    _close_device_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(shape, ("dp", "ep")))
    except Exception as e:  # noqa: BLE001
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"sparse_ep tests need a {shape[0]}x{shape[1]} mesh ('ep' axis = {EP_AXIS_SIZE}): {e}")

    yield ttml.mesh()

    _close_device_quietly()
    try:
        import ttml._mesh as _mesh_mod  # type: ignore[import-not-found]

        _mesh_mod._mesh = None
    except Exception:  # noqa: BLE001
        pass
    _restore_mgd_path(previous_mgd)


# ---------------------------------------------------------------------------
# Multi-device numerics: SparseMoEEP vs replicated dense MoE
# ---------------------------------------------------------------------------


def _host_weights(cfg: _Cfg, seed: int = SEED):
    """Deterministic host weights: gate [E, dim] and per-expert w1/w3 [I,H], w2 [H,I]."""
    g = torch.Generator().manual_seed(seed)
    E, H, I = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
    gate = torch.randn(E, H, generator=g) * (H**-0.5)
    w1 = [torch.randn(I, H, generator=g) * (H**-0.5) for _ in range(E)]
    w3 = [torch.randn(I, H, generator=g) * (H**-0.5) for _ in range(E)]
    w2 = [torch.randn(H, I, generator=g) * (I**-0.5) for _ in range(E)]
    return gate, w1, w3, w2


def _4d(arr: np.ndarray) -> np.ndarray:
    """[out, in] -> [1, 1, out, in], the native LinearLayer weight shape.

    ``set_value`` adopts whatever shape it is handed, so feeding a 2-D array
    silently reshapes the Parameter and backward then fails with a grad-shape
    mismatch. Always hand replicated weights the 4-D form.
    """
    return arr[None, None]


def _set(param, np_arr, mapper, device):
    """Set a Parameter's underlying tensor from a host array via the given mesh mapper."""
    t = ttnn.from_torch(
        torch.from_numpy(np_arr).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=mapper,
    )
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    param.tensor.set_value(t)


@pytest.fixture(scope="module")
def ep_vs_reference(ep_mesh):
    """Build a replicated dense MoE reference and an EP-sharded SparseMoEEP.

    Runs forward and backward once for both and returns the gathered host
    tensors, so the forward and backward assertions don't pay for it twice.
    """
    ep_size = EP_AXIS_SIZE
    # Keep E a multiple of ep_size so every shard holds e_local = E / ep_size experts.
    n_experts = 8 if 8 % ep_size == 0 else ep_size * 2
    cfg = _Cfg(n_routed_experts=n_experts)
    E = cfg.n_routed_experts
    e_local = E // ep_size

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.reset_graph()
    ctx.set_seed(SEED)
    torch.manual_seed(SEED)
    device = ctx.get_device()

    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)
    ep_shard = ttml.mesh().axis_mapper("ep", tdim=0)  # shard dim0 across ep
    gather = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    gate_w, w1, w3, w2 = _host_weights(cfg)

    # Dense MoE reference: keeps all E experts replicated on every chip (an EP
    # module would shard them, which is the thing under test).
    ref = MoE(_Cfg(n_routed_experts=cfg.n_routed_experts, moe_type="dense", moe_axis_name=None))
    ep = SparseMoEEP(cfg, axis_name="ep")
    assert ep.e_local == e_local, f"e_local {ep.e_local} != {e_local}"

    # Gate: replicated on both (LinearLayer weight is [1, 1, E, H]).
    _set(ref.gate.weight, _4d(gate_w.numpy()), replicate, device)
    _set(ep.gate.weight, _4d(gate_w.numpy()), replicate, device)

    # Experts: reference holds all E (replicated); EP shard r holds global experts
    # [r*e_local, (r+1)*e_local). For each local index i, build [ep_size,1,I,H]
    # with slice r = expert (r*e_local + i) and shard dim0.
    for e in range(E):
        _set(ref.experts[e].w1.weight, _4d(w1[e].numpy()), replicate, device)
        _set(ref.experts[e].w3.weight, _4d(w3[e].numpy()), replicate, device)
        _set(ref.experts[e].w2.weight, _4d(w2[e].numpy()), replicate, device)
    for i in range(e_local):
        gate_i = np.stack([w1[r * e_local + i].numpy() for r in range(ep_size)], axis=0)[:, None]
        up_i = np.stack([w3[r * e_local + i].numpy() for r in range(ep_size)], axis=0)[:, None]
        down_i = np.stack([w2[r * e_local + i].numpy() for r in range(ep_size)], axis=0)[:, None]
        _set(ep.w_gate[i], gate_i, ep_shard, device)
        _set(ep.w_up[i], up_i, ep_shard, device)
        _set(ep.w_down[i], down_i, ep_shard, device)

    # Same input, replicated across the mesh.
    B, S = 2, 32
    g = torch.Generator().manual_seed(SEED + 7)
    x_np = torch.randn(B, 1, S, cfg.dim, generator=g).numpy().astype(np.float32)

    def make_x():
        t = ttnn.from_torch(
            torch.from_numpy(x_np).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=replicate,
        )
        return ttml.autograd.create_tensor(t)

    x_ref, x_ep = make_x(), make_x()
    ttnn.synchronize_device(device)
    out_ref = ref(x_ref)
    out_ep = ep(x_ep)
    ttnn.synchronize_device(device)

    # Both outputs are replicated across ep (ref: replicated compute; ep: all_reduced).
    # Gather concatenates the ep replicas along dim0; compare the first B rows.
    fwd_ref = ttnn.to_torch(out_ref.get_value(), mesh_composer=gather).float()[:B]
    fwd_ep = ttnn.to_torch(out_ep.get_value(), mesh_composer=gather).float()[:B]

    # Backward: same random upstream into both; compare the replicated gate grad.
    gg = torch.Generator().manual_seed(SEED + 11)
    up_np = torch.randn(B, 1, S, cfg.dim, generator=gg).numpy().astype(np.float32)

    def make_up():
        return ttnn.from_torch(
            torch.from_numpy(up_np).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=replicate,
        )

    out_ref.set_grad(make_up())
    out_ep.set_grad(make_up())
    out_ref.backward(False)
    out_ep.backward(False)
    ttnn.synchronize_device(device)

    # gate.weight is [1, 1, E, H] and replicated; the gather concatenates the ep
    # replicas along dim0, so slice replica 0 out of both.
    grad_ref = ttnn.to_torch(ref.gate.weight.tensor.get_grad(), mesh_composer=gather).float()[:1]
    grad_ep = ttnn.to_torch(ep.gate.weight.tensor.get_grad(), mesh_composer=gather).float()[:1]

    return {
        "ep_size": ep_size,
        "e_local": e_local,
        "n_experts": E,
        "fwd_ref": fwd_ref,
        "fwd_ep": fwd_ep,
        "grad_ref": grad_ref,
        "grad_ep": grad_ep,
    }


def test_ep_shards_experts(ep_vs_reference):
    """Each shard owns exactly E / EP experts — i.e. sharding actually happened."""
    r = ep_vs_reference
    assert r["e_local"] * r["ep_size"] == r["n_experts"]
    assert r["e_local"] < r["n_experts"], "EP axis > 1 must give each shard a strict subset of experts"


def _assert_close(label: str, ref: torch.Tensor, got: torch.Tensor, *, pcc: float, atol: float, rtol: float) -> None:
    """PCC plus an absolute/relative bound.

    PCC alone is scale- and offset-invariant: a result that is uniformly scaled,
    or off by a constant, still correlates perfectly. The magnitude bound is what
    catches a mis-scaled all_reduce (e.g. summing instead of averaging, or
    double-counting a shard), which is the realistic EP failure mode.
    """
    diff = (ref - got).abs()
    max_abs = float(diff.max())
    scale = float(ref.abs().max())
    tol = atol + rtol * scale
    assert_with_pcc(ref, got, pcc=pcc)
    assert max_abs <= tol, (
        f"{label}: max_abs_diff={max_abs:.5f} exceeds tol={tol:.5f} "
        f"(atol={atol} + rtol={rtol} * ref_absmax={scale:.5f}); "
        f"mean_abs_diff={float(diff.mean()):.5f}"
    )


def test_forward_parity_ep_vs_replicated_sparse(ep_vs_reference):
    r = ep_vs_reference
    # bf16 accumulation over E experts; tolerance scales with the output range.
    _assert_close("forward", r["fwd_ref"], r["fwd_ep"], pcc=0.99, atol=0.05, rtol=0.05)


def test_gate_grad_parity_ep_vs_replicated_sparse(ep_vs_reference):
    r = ep_vs_reference
    # Gradients are noisier than activations: routing puts a different subset of
    # tokens through each expert, so per-element spread is wider than forward.
    _assert_close("gate.weight grad", r["grad_ref"], r["grad_ep"], pcc=0.95, atol=0.1, rtol=0.1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
