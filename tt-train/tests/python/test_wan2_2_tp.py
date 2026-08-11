# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP collectives for the Wan port: all_gather/scatter semantics and the sharded q/k norm.

Separate file and separate pytest process on purpose -- opening a 4x8 mesh requires that
nothing has already opened a device in this process, and every test in test_wan2_2.py does.

    TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt-train/configs/mgd/bh_galaxy_4_8_line_line.textproto \
    TT_LOGGER_LEVEL=FATAL python_env/bin/python -m pytest \
        tt-train/tests/python/test_wan2_2_tp.py --tb=native -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import ttnn
import ttml

torch = pytest.importorskip("torch")

# Opening a 4x8 mesh -- fabric init plus JIT kernels for every collective -- can take many
# minutes on a cold ~/.cache/tt-metal-cache, and because the mesh fixture is module-scoped
# that whole cost is billed to whichever test runs first. Well above pytest-timeout's 300s.
pytestmark = pytest.mark.timeout(1800)

DP, TP = 4, 8
DIM, SEQ, EPS = 512, 64, 1e-6
LOCAL = DIM // TP


@pytest.fixture(scope="module")
def mesh():
    ttml.open_device_mesh(ttml.Mesh((DP, TP), ("dp", "tp")))
    try:
        yield ttml.mesh()
    finally:
        ttml.close_device_mesh()


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def shard_over_tp(arr: np.ndarray, tp_axis: int):
    """Upload (1,1,S,DIM) so chip i holds columns [i*LOCAL, (i+1)*LOCAL)."""
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(_device(), 3, tp_axis)
    t = ttml.autograd.Tensor.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32), ttnn.Layout.TILE, ttnn.bfloat16, mapper
    )
    t.set_requires_grad(True)
    return t


def concat_all_devices(tensor) -> np.ndarray:
    """Concatenate dim 3 across all DP*TP devices, in device order."""
    value = tensor.get_value() if hasattr(tensor, "get_value") else tensor
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(_device(), 3)
    return np.asarray(ttnn.to_torch(value, mesh_composer=composer).float().numpy())


def gather_dim(tensor, dim: int) -> np.ndarray:
    """Reassemble a tensor sharded on `dim` across the tp axis, dropping the dp replicas.

    Devices are ordered dp-major, so concatenating all DP*TP of them puts the dp=0 tp row
    first; everything after it is a replica.
    """
    value = tensor.get_value() if hasattr(tensor, "get_value") else tensor
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(_device(), dim)
    arr = np.asarray(ttnn.to_torch(value, mesh_composer=composer).float().numpy())
    return np.take(arr, range(arr.shape[dim] // DP), axis=dim)


def replicated_value(tensor) -> np.ndarray:
    """Read a tensor that is replicated on every device (e.g. a RowParallelLinear output)."""
    arr = concat_all_devices(tensor)
    return np.take(arr, range(arr.shape[3] // (DP * TP)), axis=3)


def first_tp_row(tensor) -> np.ndarray:
    """The dp=0 replica's TP row, i.e. the logical full-width tensor."""
    return gather_dim(tensor, 3)


def as_bf16(x: np.ndarray) -> np.ndarray:
    """Round through bf16 so the torch reference evaluates at the same point ttml does."""
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def torch_rmsnorm(x: np.ndarray, gamma: np.ndarray):
    t = torch.tensor(as_bf16(x), dtype=torch.float32, requires_grad=True)
    out = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * torch.tensor(gamma)
    return t, out


def pcc(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# collective semantics -- what all_gather and scatter actually do
# ---------------------------------------------------------------------------


def test_all_gather_gives_every_chip_the_full_width(mesh):
    """Each chip starts with LOCAL columns and must end up holding all DIM of them."""
    tp_axis = mesh.axis_index("tp")
    # Random, not arange: bf16 spacing at 32767 is 128, so large integers cannot round-trip.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, SEQ, DIM)).astype(np.float32)

    full = ttml.ops.distributed.all_gather(shard_over_tp(x, tp_axis), 3, tp_axis)
    assert tuple(full.shape())[-1] == DIM, "all_gather did not restore full width"

    # Every one of the 32 devices should now hold an identical copy of x.
    arr = concat_all_devices(full)
    assert arr.shape[3] == DIM * DP * TP
    blocks = np.split(arr, DP * TP, axis=3)
    np.testing.assert_allclose(blocks[0], x, rtol=0.02, atol=0.02)
    for i, b in enumerate(blocks[1:], start=1):
        np.testing.assert_allclose(b, blocks[0], rtol=0, atol=0.0, err_msg=f"device {i} differs")

    # Guard against a vacuous pass: a shard-order error must not survive the comparison above.
    assert not np.allclose(blocks[0], np.roll(x, LOCAL, axis=3), rtol=0.02, atol=0.02)


def test_scatter_takes_this_chips_slice(mesh):
    """scatter must be the inverse of all_gather, not a scatter-from-root."""
    tp_axis = mesh.axis_index("tp")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 1, SEQ, DIM)).astype(np.float32)

    xs = shard_over_tp(x, tp_axis)
    roundtrip = ttml.ops.distributed.scatter(ttml.ops.distributed.all_gather(xs, 3, tp_axis), 3, tp_axis)
    assert tuple(roundtrip.shape())[-1] == LOCAL, "scatter did not re-shard to LOCAL width"
    np.testing.assert_allclose(first_tp_row(roundtrip), x, rtol=0.02, atol=0.02)


def test_gather_scatter_roundtrip_preserves_gradient(mesh):
    """The SHARDED default double-counts by TP; REPLICATED is the pairing the model uses.

    Both are asserted so that dropping the enum argument as a "simplification" fails here
    rather than silently scaling every q/k gradient by the TP size.
    """
    tp_axis = mesh.axis_index("tp")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, SEQ, DIM)).astype(np.float32)
    # unary.mean reduces the LOCAL tensor (SEQ*LOCAL), not the logical one.
    baseline = 1.0 / (SEQ * LOCAL)

    def roundtrip_grad(grad_output_type):
        xs = shard_over_tp(x, tp_axis)
        gathered = ttml.ops.distributed.all_gather(xs, 3, tp_axis, grad_output_type)
        out = ttml.ops.distributed.scatter(gathered, 3, tp_axis)
        np.testing.assert_allclose(first_tp_row(out), x, rtol=0.02, atol=0.02)
        ttml.ops.unary.mean(out).backward(False)
        assert xs.is_grad_initialized(), "no gradient reached the sharded input"
        factor = float(np.mean(first_tp_row(xs.get_grad())) / baseline)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        return factor

    sharded = roundtrip_grad(ttml.ops.distributed.GradOutputType.SHARDED)
    replicated = roundtrip_grad(ttml.ops.distributed.GradOutputType.REPLICATED)
    print(
        f"\ngrad factor: SHARDED={sharded:.3f} (expect {float(TP):.1f}), " f"REPLICATED={replicated:.3f} (expect 1.0)"
    )

    assert abs(sharded - TP) < 0.05 * TP, f"SHARDED default no longer inflates by TP: {sharded:.3f}"
    assert abs(replicated - 1.0) < 0.05, f"REPLICATED gradient is wrong: {replicated:.3f}"


# ---------------------------------------------------------------------------
# the thing the collectives exist for
# ---------------------------------------------------------------------------


def test_rmsnorm_tp_matches_torch_forward(mesh):
    """Sharded RMSNorm must equal a full-width RMSNorm, not a per-shard one."""
    from ttml.models.wan2_2.attention import _RMSNorm

    tp_axis = mesh.axis_index("tp")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, SEQ, DIM)).astype(np.float32)
    gamma = np.ones((1, 1, 1, DIM), dtype=np.float32)

    out = _RMSNorm(DIM, EPS, tp_axis)(shard_over_tp(x, tp_axis))
    _, reference = torch_rmsnorm(x, gamma)
    reference_np = reference.detach().numpy()

    got = first_tp_row(out)
    assert pcc(got, reference_np) > 0.999

    # A per-shard norm would also be finite and correctly shaped -- prove we are not that.
    per_shard = np.concatenate(
        [s * (1.0 / np.sqrt(np.mean(s**2, axis=-1, keepdims=True) + EPS)) for s in np.split(x, TP, axis=3)],
        axis=3,
    )
    assert pcc(got, per_shard) < pcc(got, reference_np)


def test_rmsnorm_tp_matches_torch_backward(mesh):
    from ttml.models.wan2_2.attention import _RMSNorm

    tp_axis = mesh.axis_index("tp")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 1, SEQ, DIM)).astype(np.float32)
    gamma = np.ones((1, 1, 1, DIM), dtype=np.float32)

    xs = shard_over_tp(x, tp_axis)
    ttml.ops.unary.mean(_RMSNorm(DIM, EPS, tp_axis)(xs)).backward(False)

    ref_in, ref_out = torch_rmsnorm(x, gamma)
    ref_out.mean().backward()

    assert xs.is_grad_initialized(), "no gradient reached the sharded input"
    got = first_tp_row(xs.get_grad())
    reference = ref_in.grad.numpy()

    # Magnitude first: PCC is scale-invariant and scores ~1.0 on a uniformly inflated
    # gradient, so it cannot see the TP-scaling bug. torch means over SEQ*DIM while
    # ttml.ops.unary.mean means over SEQ*LOCAL, so the only legitimate ratio is exactly
    # TP; double-counting would show TP*TP.
    ratio = float(np.linalg.norm(got) / np.linalg.norm(reference))
    correlation = pcc(got, reference)
    print(f"\ngrad norm ratio = {ratio:.4f} (expect {float(TP):.1f})  pcc = {correlation:.6f}")
    assert abs(ratio - TP) < 0.02 * TP, f"gradient magnitude off: {ratio:.3f}, expected {TP}"

    # Shape agreement. The floor here is bf16 through gather -> rmsnorm -> scatter and
    # back; the normalisation gradient amplifies input rounding more than the elementwise
    # ops elsewhere in the suite, so this sits below their 0.999.
    assert correlation > 0.998, f"gradient shape disagrees: pcc {correlation:.6f}"


# ---------------------------------------------------------------------------
# step 3 -- the shard plan used when loading a checkpoint
# ---------------------------------------------------------------------------


def _tp_attention():
    from ttml.models.wan2_2 import WanConfig
    from ttml.models.wan2_2.attention import WanAttention

    cfg = WanConfig(dim=DIM, num_heads=8, eps=EPS, use_tp=True, init_weights=True)
    return WanAttention(cfg, is_self=True)


def test_shard_plan_marks_exactly_the_parallel_parameters(mesh):
    """Column shards weight on 2 and bias on 3; row shards weight on 3 and replicates bias."""
    from ttml.models.wan2_2.weights import _shard_plan

    plan = {name: tdim for name, (_, tdim, _) in _shard_plan(_tp_attention()).items()}

    for proj in ("to_q", "to_k", "to_v"):
        assert plan[f"{proj}.weight"] == 2, f"{proj}.weight must shard on dim 2"
        assert plan[f"{proj}.bias"] == 3, f"{proj}.bias must shard on dim 3"
    assert plan["to_out.weight"] == 3, "row-parallel weight must shard on dim 3"

    # The traps: these sit inside a TP'd block but must stay whole.
    for replicated in ("to_out.bias", "norm_q.weight", "norm_k.weight"):
        assert replicated not in plan, f"{replicated} must stay replicated"


def test_sharded_load_roundtrips_the_full_checkpoint_tensor(mesh):
    """A full-size checkpoint array must reassemble exactly after a sharded upload."""
    from ttml.models.wan2_2.weights import _global_shape, _shard_plan

    attn = _tp_attention()
    params = dict(attn.named_parameters())
    plan = _shard_plan(attn)
    assert plan, "no sharded parameters -- is use_tp wired through?"

    rng = np.random.default_rng(0)
    for name, (mapper, tdim, tp_size) in plan.items():
        target = params[name]
        full_shape = _global_shape(target.shape(), tdim, tp_size)
        assert full_shape[tdim] == tuple(target.shape())[tdim] * tp_size

        full = rng.standard_normal(full_shape).astype(np.float32)
        uploaded = ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(full), ttnn.Layout.TILE, ttnn.bfloat16, mapper)
        target.set_value(uploaded.get_value())

        back = gather_dim(target, tdim)
        assert back.shape == full_shape, f"{name}: got {back.shape}, want {full_shape}"
        np.testing.assert_allclose(back, full, rtol=0.02, atol=0.02, err_msg=name)


def test_shard_plan_is_empty_without_tp(mesh):
    """A non-parallel model must produce no plan even while a mesh is open."""
    from ttml.models.wan2_2 import WanConfig
    from ttml.models.wan2_2.attention import WanAttention
    from ttml.models.wan2_2.weights import _shard_plan

    cfg = WanConfig(dim=DIM, num_heads=8, eps=EPS, use_tp=False, init_weights=True)
    assert _shard_plan(WanAttention(cfg, is_self=True)) == {}


def test_tp_feedforward_matches_torch(mesh):
    """ff1 column-parallel -> tanh gelu -> ff2 row-parallel must equal the unsharded result.

    Exercises the gather_output=False / input_is_parallel=True pairing: the intermediate
    never leaves the shard, and RowParallelLinear all-reduces once at the end.
    """
    from ttml.models.wan2_2 import WanConfig
    from ttml.models.wan2_2.transformer import WanFeedForward
    from ttml.models.wan2_2.weights import _global_shape, _shard_plan

    FFN = 1024  # 1024/8 = 128 per chip, tile-aligned
    cfg = WanConfig(dim=DIM, ffn_dim=FFN, num_heads=8, eps=EPS, use_tp=True, init_weights=True)
    ff = WanFeedForward(cfg)

    plan = _shard_plan(ff)
    assert set(plan) == {
        "ff1.weight",
        "ff1.bias",
        "ff2.weight",
    }, f"unexpected FFN shard plan: {sorted(plan)} (ff2.bias must stay replicated)"

    rng = np.random.default_rng(0)
    ref = {}
    for name, target in ff.named_parameters():
        mapper, tdim, tp_size = plan.get(name, (None, None, 1))
        shape = _global_shape(target.shape(), tdim, tp_size)
        w = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        ref[name] = as_bf16(w)
        target.set_value(
            ttml.autograd.Tensor.from_numpy(
                np.ascontiguousarray(w), ttnn.Layout.TILE, ttnn.bfloat16, mapper
            ).get_value()
        )

    x = (rng.standard_normal((1, 1, SEQ, DIM)) * 0.5).astype(np.float32)
    xt = ttml.autograd.Tensor.from_numpy(
        as_bf16(x),
        ttnn.Layout.TILE,
        ttnn.bfloat16,
        ttml.core.distributed.replicate_tensor_to_mesh_mapper(_device()),
    )
    got = replicated_value(ff(xt))

    tx = torch.tensor(as_bf16(x))
    h = torch.nn.functional.gelu(
        tx @ torch.tensor(ref["ff1.weight"][0, 0]).T + torch.tensor(ref["ff1.bias"][0, 0, 0]),
        approximate="tanh",
    )
    expected = (h @ torch.tensor(ref["ff2.weight"][0, 0]).T + torch.tensor(ref["ff2.bias"][0, 0, 0])).numpy()

    assert got.shape == expected.shape, f"got {got.shape}, want {expected.shape}"
    assert pcc(got, expected) > 0.99


# ---------------------------------------------------------------------------
# LoRA under TP -- injection, and the adapter save/load round trip
# ---------------------------------------------------------------------------

RANK, FFN_DIM = 8, 1024


def _lora_export():
    """Import the example's lora_export lazily.

    Its directory also holds train.py, pipeline.py and utils/, so putting it on sys.path at
    module scope would shadow those names for the whole pytest session.
    """
    path = str(Path(__file__).resolve().parents[2] / "sources/examples/lora_wan2_2")
    if path not in sys.path:
        sys.path.insert(0, path)
    import utils.lora_export as mod

    return mod


def _tp_block():
    from ttml.models.wan2_2 import WanConfig, WanTransformerBlock

    cfg = WanConfig(dim=DIM, ffn_dim=FFN_DIM, num_heads=8, eps=EPS, use_tp=True, init_weights=True)
    return WanTransformerBlock(cfg)


def _lora_block():
    lora_cfg = ttml.modules.LoraConfig(
        rank=RANK,
        alpha=float(RANK),
        target_modules=[r"attn[12]\.to_q", r"attn[12]\.to_out", r"ffn\.ff[12]"],
        lora_dropout=0.0,
        use_rslora=False,
        verbose=False,
    )
    return ttml.modules.LoraModel(_tp_block(), lora_cfg)


def test_lora_wraps_parallel_layers_with_parallel_variants(mesh):
    """LoraModel must pick the parallel wrappers, not plain LoraLinear."""
    seen = {}
    for name, module in _lora_block().named_modules():
        cls = type(module).__name__
        if cls.startswith("Lora") and cls != "LoraModel":
            seen[name] = cls

    assert seen, "LoRA injected nothing"
    assert "LoraLinear" not in set(seen.values()), f"a plain LoraLinear was injected into a TP model: {seen}"
    kinds = set(seen.values())
    assert kinds <= {"LoraColumnParallelLinear", "LoraRowParallelLinear"}, kinds


def test_lora_adapter_roundtrips_at_tp8(mesh, tmp_path):
    """Save/load must reassemble the global adapter, not a per-device shard.

    Writes a known adapter, scatters it into the sharded model, gathers it back out, and
    compares -- so _scatter and _gather are checked against each other and against the
    diffusers key/shape contract that makes the file portable.
    """
    from safetensors.numpy import load_file, save_file

    export = _lora_export()

    # Global 2-D shapes: lora_A is (rank, in_features), lora_B is (out_features, rank).
    expected = {
        "transformer.attn1.to_q.lora_A.weight": (RANK, DIM),
        "transformer.attn1.to_q.lora_B.weight": (DIM, RANK),
        "transformer.attn1.to_out.0.lora_A.weight": (RANK, DIM),
        "transformer.attn1.to_out.0.lora_B.weight": (DIM, RANK),
        "transformer.attn2.to_q.lora_A.weight": (RANK, DIM),
        "transformer.attn2.to_q.lora_B.weight": (DIM, RANK),
        "transformer.attn2.to_out.0.lora_A.weight": (RANK, DIM),
        "transformer.attn2.to_out.0.lora_B.weight": (DIM, RANK),
        "transformer.ffn.net.0.proj.lora_A.weight": (RANK, DIM),
        "transformer.ffn.net.0.proj.lora_B.weight": (FFN_DIM, RANK),
        "transformer.ffn.net.2.lora_A.weight": (RANK, FFN_DIM),
        "transformer.ffn.net.2.lora_B.weight": (DIM, RANK),
    }

    model = _lora_block()
    mesh_shape = (DP, TP)

    # What the model actually exposes, before trusting any of it.
    produced = export.lora_state_dict(model, mesh_shape)
    assert set(produced) == set(expected), (
        f"adapter keys differ\n  missing: {sorted(set(expected) - set(produced))}"
        f"\n  extra:   {sorted(set(produced) - set(expected))}"
    )
    for key, shape in expected.items():
        assert produced[key].shape == shape, (
            f"{key}: got {produced[key].shape}, want global {shape} "
            f"(a per-device shard would be {shape[0] // TP} or {shape[1] // TP} on one axis)"
        )

    # Round trip: known adapter -> scatter into the mesh -> gather back out.
    rng = np.random.default_rng(0)
    reference = {k: (rng.standard_normal(v) * 0.05).astype(np.float32) for k, v in expected.items()}
    src = tmp_path / "adapter.safetensors"
    save_file(reference, str(src))

    restored_count = export.load_lora_expert(model, str(src), mesh_shape)
    assert restored_count == len(expected), f"restored {restored_count}/{len(expected)}"

    dst = tmp_path / "roundtrip.safetensors"
    export.save_lora_expert(model, str(dst), mesh_shape)
    got = load_file(str(dst))

    for key, want in reference.items():
        assert got[key].shape == want.shape, f"{key}: {got[key].shape} != {want.shape}"
        np.testing.assert_allclose(got[key], want, rtol=0.05, atol=0.01, err_msg=key)


def test_replicated_lora_gradients_agree_across_tp(mesh):
    """lora_A on a column-parallel projection is replicated, but everything downstream of
    it is sharded -- each TP rank only sees its own slice of the output.

    Its gradient is therefore correct only if the low-rank path inserts the same broadcast
    that ColumnParallelLinear.forward does for x, so the partials get summed. Checked
    without a reference: a correctly reduced replicated grad is *identical* on every rank,
    a partial one is not. ttml.sync_gradients deliberately does not reduce over TP, so
    nothing downstream would repair this.
    """
    from ttml.models.wan2_2 import build_rope_params

    model = _lora_block()
    head_dim = DIM // 8
    rope = build_rope_params(
        head_dim=head_dim,
        patch_size=(1, 2, 2),
        latent_shape=(1, 4, 1, 16, 16),
        max_seq_len=128,
    )

    rng = np.random.default_rng(0)
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(_device())

    def rep(shape):
        return ttml.autograd.Tensor.from_numpy(
            as_bf16((rng.standard_normal(shape) * 0.5).astype(np.float32)),
            ttnn.Layout.TILE,
            ttnn.bfloat16,
            replicate,
        )

    out = model(rep((1, 1, SEQ, DIM)), None, rep((1, 1, SEQ, DIM)), rep((1, 1, 6, DIM)), rope)
    ttml.ops.unary.mean(out).backward(False)

    params = dict(model.named_parameters())
    checked = 0
    for name, param in params.items():
        if not name.endswith("lora_A") or "to_q" not in name:
            continue
        assert param.is_grad_initialized(), f"{name} received no gradient"
        blocks = np.split(concat_all_devices(param.get_grad()), DP * TP, axis=3)
        spread = max(float(np.abs(b - blocks[0]).max()) for b in blocks[1:])
        print(f"\n{name}: max cross-rank gradient spread = {spread:.3e}")
        assert spread < 1e-3, (
            f"{name} differs across TP ranks by {spread:.3e}; the replicated lora_A holds a "
            f"per-rank partial sum, so adapter gradients are wrong by up to a factor of {TP}"
        )
        checked += 1

    assert checked, "no column-parallel lora_A parameters were found to check"
