# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does it matter that MiniMax-H3 merges a Turbo LoRA AFTER quantizing the weight?

On a small mesh the DiT's linears are built as ``bfloat8_b`` at construction time, and
``h3_adapter_loader.load_h3_adapter_into`` then merges ``scale * B@A`` into the already-quantized
device weight -- ``LoRAMixin._apply_delta`` does ``ttnn.add(weight.data, delta,
output_tensor=weight.data)`` where ``weight.data`` is ``bfloat8_b``. So the shipped arithmetic is

    requantize(quantize(W) + delta)

and not

    quantize(W + delta)

which is what a host-side fuse before quantization would give. The two are not the same function:
bfloat8_b shares one 8-bit exponent across a 16-element tile row, so adding a delta can move the
row's maximum and re-round every other element in it. Whether that difference is material is a
measurement, not an opinion, and this is the measurement -- on the real checkpoint weights and the
real published adapter, not on random tensors.

Both orderings are scored against the same fp32 ``W + delta``, so the comparison says how much
accuracy each ordering gives up rather than only how far apart they are. The rows are the two
targets whose ``B`` needs no layout transform (``attn.to_out``, ``ff.net.2``), so nothing here
depends on the rope permutation, the head interleave or the swiglu pack being right; those have
their own coverage. Orientation is the torch one -- the question is invariant under the transpose
``_prepare_torch_state`` applies.

The bind/unbind round trip is measured for the same reason: ``_apply_delta``'s docstring calls the
pair "an exact negation", which is true of the delta but not of a weight that is re-rounded twice.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn

from .common import H3_MESH_PARALLEL

TURBO_FILE_ENV = "MINIMAX_H3_TURBO_FILE"
MODEL_PATH_ENV = "MINIMAX_H3_MODEL_PATH"

# The two targets the adapter registers with no layout transform, as the adapter and the checkpoint
# respectively spell them.
TARGETS = [
    ("transformer_blocks.{i}.attn.to_out.0", "attn.to_out"),
    ("transformer_blocks.{i}.ff.net.2", "ff.net.2"),
]
# Two blocks, one at each end of the stack: block 0 sees the rawest activations and block 25 sits
# where the residual has grown, and their weight scales differ enough to be worth both.
BLOCKS = [0, 25]


def _pcc(reference: torch.Tensor, test: torch.Tensor) -> float:
    a = reference.flatten().to(torch.float64)
    b = test.flatten().to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((a @ b) / denom) if denom > 0 else float("nan")


def _stats(reference: torch.Tensor, test: torch.Tensor) -> dict:
    err = (test.to(torch.float64) - reference.to(torch.float64)).abs()
    sigma = reference.to(torch.float64).std()
    return {
        "pcc": _pcc(reference, test),
        "max_abs_err": float(err.max()),
        "rmse_over_sigma": float(err.pow(2).mean().sqrt() / sigma),
    }


def _checkpoint_weight(directory: Path, key: str) -> torch.Tensor:
    import json

    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
    shard = index["weight_map"][key]
    with safe_open(directory / shard, framework="pt") as handle:
        return handle.get_tensor(key)


def _adapter_pair(path: str, base: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``(A, B, scale)`` for one target; scale is the file's ``alpha / rank``, never 1."""
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
        a_key = next(k for k in (f"{base}.lora_A.default.weight", f"{base}.lora_A.weight") if k in keys)
        b_key = next(k for k in (f"{base}.lora_B.default.weight", f"{base}.lora_B.weight") if k in keys)
        a, b = handle.get_tensor(a_key), handle.get_tensor(b_key)
    alpha = metadata.get("alpha") or metadata.get("lora_alpha")
    assert alpha is not None, f"{path} carries no file-level alpha; the scale would silently be 1"
    rank = a.shape[0]
    return a, b, float(alpha) / rank


@H3_MESH_PARALLEL
def test_quantize_then_merge_matches_merge_then_quantize(
    mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int, num_links: int, is_fsdp: bool, topology, reset_seeds
) -> None:
    turbo = os.environ.get(TURBO_FILE_ENV)
    model_root = os.environ.get(MODEL_PATH_ENV)
    if not turbo or not os.path.exists(turbo):
        pytest.skip(f"set {TURBO_FILE_ENV} to a lightx2v MiniMax-H3 Turbo safetensors file")
    if not model_root:
        pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot")
    directory = Path(model_root) / "transformer"

    rows = []
    for index in BLOCKS:
        for adapter_base_tmpl, leaf in TARGETS:
            adapter_base = adapter_base_tmpl.format(i=index)
            ckpt_key = f"{adapter_base}.weight"
            weight = _checkpoint_weight(directory, ckpt_key).to(torch.float32)
            a, b, scale = _adapter_pair(turbo, adapter_base)
            # register_lora uploads A and B as bfloat16, so the delta the device actually adds is
            # the bf16 product -- not an fp32 one. Reproduce that, or this measures the wrong thing.
            delta = (b.to(torch.bfloat16).to(torch.float32) @ a.to(torch.bfloat16).to(torch.float32)) * scale
            fused_fp32 = weight + delta

            def up(tensor: torch.Tensor, dtype) -> ttnn.Tensor:
                return ttnn.from_torch(
                    tensor.unsqueeze(0).unsqueeze(0), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
                )

            def down(tensor: ttnn.Tensor) -> torch.Tensor:
                out = ttnn.to_torch(
                    tensor,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape)
                    ),
                )
                return out.reshape(-1, *out.shape[2:])[0].to(torch.float32)

            # The briefed ordering: fuse on the host in fp32, quantize once.
            briefed = down(up(fused_fp32, ttnn.bfloat8_b))
            # The shipped ordering: quantize the base weight, then add the bf16 delta into the
            # bfloat8_b weight in place, which re-quantizes every tile row the add touches.
            shipped_w = up(weight, ttnn.bfloat8_b)
            delta_dev = up(delta, ttnn.bfloat16)
            ttnn.add(shipped_w, delta_dev, output_tensor=shipped_w)
            shipped = down(shipped_w)
            # And back out again, which is what unbinding an adapter does.
            ttnn.subtract(shipped_w, delta_dev, output_tensor=shipped_w)
            round_trip = down(shipped_w)
            base_quantized = down(up(weight, ttnn.bfloat8_b))
            # Did the delta survive, in EACH ordering separately? The briefed one quantizes a weight
            # that already contains the delta, so it almost has to move. The shipped one adds a
            # bf16 delta into a bfloat8_b tensor, where the tile row's shared exponent gives a
            # quantization step of roughly 2^-3 of the row maximum -- and a rank-128 adapter delta
            # is ~1e-4 to 3e-3 of the weight. If that is below the step, `ttnn.add` returns the
            # weight unchanged and the adapter silently does NOTHING on a quantized model, which is
            # the one failure this whole test exists to exclude.
            moved = _stats(base_quantized, briefed)
            moved_shipped = _stats(base_quantized, shipped)

            row = {
                "block": index,
                "target": leaf,
                "shape": tuple(weight.shape),
                "rank": int(a.shape[0]),
                "scale": scale,
                "delta_over_w": float(delta.abs().max() / weight.abs().max()),
                # Does the delta survive quantization at all? A rank-128 update's peak is ~1e-4 of
                # the weight's peak, so a ratio of maxima says nothing; what matters is whether the
                # fused weight differs from the quantized base one. If it did not, every comparison
                # above would pass by measuring nothing.
                "fused_vs_base_quantized": moved,
                "shipped_vs_base_quantized": moved_shipped,
                "briefed_vs_fp32": _stats(fused_fp32, briefed),
                "shipped_vs_fp32": _stats(fused_fp32, shipped),
                "shipped_vs_briefed": _stats(briefed, shipped),
                "unbind_vs_quantized_base": _stats(base_quantized, round_trip),
            }
            rows.append(row)
            logger.info(
                f"block {index} {leaf} {tuple(weight.shape)} scale={scale:.4f} "
                f"|delta|/|W|={row['delta_over_w']:.4f}\n"
                f"    quantize(W+d)          vs fp32: PCC {row['briefed_vs_fp32']['pcc']:.6f}  "
                f"RMSE/sigma {row['briefed_vs_fp32']['rmse_over_sigma']:.5f}\n"
                f"    requantize(q(W)+d)     vs fp32: PCC {row['shipped_vs_fp32']['pcc']:.6f}  "
                f"RMSE/sigma {row['shipped_vs_fp32']['rmse_over_sigma']:.5f}\n"
                f"    shipped vs briefed            : PCC {row['shipped_vs_briefed']['pcc']:.6f}  "
                f"RMSE/sigma {row['shipped_vs_briefed']['rmse_over_sigma']:.5f}\n"
                f"    bind->unbind vs q(W)          : PCC {row['unbind_vs_quantized_base']['pcc']:.6f}  "
                f"max|err| {row['unbind_vs_quantized_base']['max_abs_err']:.3e}\n"
                f"    DID THE DELTA SURVIVE? quantize(W+d) vs q(W): max|err| "
                f"{row['fused_vs_base_quantized']['max_abs_err']:.3e}  RMSE/sigma "
                f"{row['fused_vs_base_quantized']['rmse_over_sigma']:.3e}\n"
                f"                           requantize(q(W)+d) vs q(W): max|err| "
                f"{row['shipped_vs_base_quantized']['max_abs_err']:.3e}  RMSE/sigma "
                f"{row['shipped_vs_base_quantized']['rmse_over_sigma']:.3e}"
            )

    # What the assertions are for. The claim being tested is NOT that the two orderings are equal --
    # they cannot be, bfloat8_b is not additive. It is that the shipped ordering costs no more
    # accuracy than the briefed one against the same fp32 target, which is the only thing that
    # decides whether the host fuse is worth building.
    for row in rows:
        briefed, shipped = row["briefed_vs_fp32"], row["shipped_vs_fp32"]
        tag = f"block {row['block']} {row['target']}"
        assert shipped["rmse_over_sigma"] <= briefed["rmse_over_sigma"] * 1.5 + 1e-6, (
            f"{tag}: merging after quantization is materially worse than merging before "
            f"({shipped['rmse_over_sigma']:.5f} vs {briefed['rmse_over_sigma']:.5f} RMSE/sigma) -- "
            "the host fuse the brief asked for would be worth building"
        )
        assert shipped["pcc"] >= 0.999, f"{tag}: shipped fuse PCC {shipped['pcc']:.6f} vs fp32 W+delta"
        # Each ordering must actually be a quantization of the FUSED weight and not of the base one:
        # a delta that rounded away entirely would pass every check above by doing nothing.
        assert row["fused_vs_base_quantized"]["max_abs_err"] > 0.0, (
            f"{tag}: quantize(W+delta) is identical to quantize(W), so the adapter's delta vanished "
            "in the rounding and this row measures nothing"
        )
        # Whether the SHIPPED ordering's delta survives at all is its own test below, because it
        # does not -- see test_the_adapter_delta_survives_a_quantized_merge.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN DEFECT, measured here. Merging a Turbo LoRA into a bfloat8_b weight is a no-op: "
        "`requantize(quantize(W) + delta)` comes back BIT-IDENTICAL to `quantize(W)` on 3 of the 4 "
        "rows measured (max|err| 0.000e+00), and moves by a single LSB (1.953e-03) on the fourth. "
        "bfloat8_b shares one exponent across a tile row, so its quantization step is ~2^-7 of the "
        "row maximum, and a rank-128 adapter delta is 1.7e-4 (v1.2_768p at alpha/rank) to 3.3e-3 "
        "(v0.1 at the scale-1 fallback) of the weight -- below the step, so `ttnn.add` returns the "
        "weight unchanged. The briefed ordering `quantize(W + delta)` does carry the delta "
        "(max|err| 1.6e-2 to 3.1e-2, RMSE/sigma 7.5e-3). End-to-end consequence on a p150 before "
        "the fix: the same seeded 512x288x56 request returned a BIT-IDENTICAL clip with the "
        "adapter at strength 1.0, at 0.0625, and with no adapter at all. RESOLVED NOT BY FIXING "
        "THIS but by routing around it -- `fuse_h3_adapter_into_state_dict` merges on the host "
        "before construction-time quantization whenever the DiT is quantized (see "
        "test_a_host_fuse_reaches_the_quantized_weight), because the behaviour measured here is a "
        "property of bfloat8_b and not something the merge can fix. It stays strict=True so that "
        "if ttnn ever makes the in-place add carry a sub-step delta, this says so."
    ),
)
@H3_MESH_PARALLEL
def test_the_adapter_delta_survives_a_quantized_merge(
    mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int, num_links: int, is_fsdp: bool, topology, reset_seeds
) -> None:
    """Does binding an adapter to a quantized weight change the weight at all?

    Separate from the ordering comparison above because it asks a different question, and because
    the answer is no: the ordering comparison is a valid passing measurement (both orderings are
    equally accurate *against an fp32 target*), while this one is the defect that measurement was
    not built to see. Two equally accurate orderings can still differ in whether either of them
    applied the adapter.
    """
    turbo = os.environ.get(TURBO_FILE_ENV)
    model_root = os.environ.get(MODEL_PATH_ENV)
    if not turbo or not os.path.exists(turbo):
        pytest.skip(f"set {TURBO_FILE_ENV} to a lightx2v MiniMax-H3 Turbo safetensors file")
    if not model_root:
        pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot")
    directory = Path(model_root) / "transformer"

    unchanged = []
    for index in BLOCKS:
        for adapter_base_tmpl, leaf in TARGETS:
            adapter_base = adapter_base_tmpl.format(i=index)
            weight = _checkpoint_weight(directory, f"{adapter_base}.weight").to(torch.float32)
            a, b, scale = _adapter_pair(turbo, adapter_base)
            delta = (b.to(torch.bfloat16).to(torch.float32) @ a.to(torch.bfloat16).to(torch.float32)) * scale

            def up(tensor: torch.Tensor, dtype) -> ttnn.Tensor:
                return ttnn.from_torch(
                    tensor.unsqueeze(0).unsqueeze(0), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device
                )

            def down(tensor: ttnn.Tensor) -> torch.Tensor:
                out = ttnn.to_torch(
                    tensor,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape)
                    ),
                )
                return out.reshape(-1, *out.shape[2:])[0].to(torch.float32)

            base = down(up(weight, ttnn.bfloat8_b))
            merged_dev = up(weight, ttnn.bfloat8_b)
            ttnn.add(merged_dev, up(delta, ttnn.bfloat16), output_tensor=merged_dev)
            moved = float((down(merged_dev) - base).abs().max())
            logger.info(f"block {index} {leaf}: merging into bfloat8_b moved the weight by {moved:.3e}")
            if moved == 0.0:
                unchanged.append(f"block {index} {leaf}")

    assert not unchanged, (
        f"binding this adapter to a bfloat8_b weight left it BIT-IDENTICAL on {len(unchanged)} of "
        f"{len(BLOCKS) * len(TARGETS)} targets ({', '.join(unchanged)}) -- the delta rounded away "
        "against the tile row's shared exponent, so the adapter does nothing on a quantized mesh"
    )


def test_a_host_fuse_reaches_the_quantized_weight() -> None:
    """The fix for the defect above: fusing before quantization puts the delta in the weight.

    Host only -- ``ttnn.from_torch`` quantizes to ``bfloat8_b`` without a device, and the question
    is about the tensor, not about a mesh.

    Two things are asserted and they are different. That the fused state dict is exactly
    ``(W + scale*B@A)`` in the checkpoint's dtype is arithmetic. That its bfloat8_b quantization
    differs from the base weight's on a real fraction of elements is the part that matters: the
    device merge satisfies the first and fails the second, which is precisely how it generated
    base-model clips under an adapter's name.
    """
    turbo = os.environ.get(TURBO_FILE_ENV)
    model_root = os.environ.get(MODEL_PATH_ENV)
    if not turbo or not os.path.exists(turbo):
        pytest.skip(f"set {TURBO_FILE_ENV} to a lightx2v MiniMax-H3 Turbo safetensors file")
    if not model_root:
        pytest.skip(f"set {MODEL_PATH_ENV} to a MiniMax-H3 diffusers snapshot")
    from ....experimental.lora.h3_adapter_loader import fuse_h3_adapter_into_state_dict

    directory = Path(model_root) / "transformer"
    keys = [f"{tmpl.format(i=index)}.weight" for index in BLOCKS for tmpl, _ in TARGETS]
    base = {key: _checkpoint_weight(directory, key) for key in keys}
    # The fuse raises on a target it cannot place, so a partial state dict has to be the whole
    # adapter's worth of targets or nothing. Give it every key it asks for, base weights only.
    with safe_open(turbo, framework="pt") as handle:
        wanted = sorted({k.rsplit(".lora_", 1)[0] + ".weight" for k in handle.keys() if ".lora_" in k})
    state = {key: _checkpoint_weight(directory, key) for key in wanted}

    scales = fuse_h3_adapter_into_state_dict(state, turbo, name=os.path.basename(turbo))
    assert len(scales) == len(wanted), f"fused {len(scales)} targets against {len(wanted)} adapter targets"
    assert set(scales.values()) == {0.0625}, f"unexpected effective scales {sorted(set(scales.values()))}"

    def quantize(tensor: torch.Tensor) -> torch.Tensor:
        device_tensor = ttnn.from_torch(
            tensor.unsqueeze(0).unsqueeze(0).to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        return ttnn.to_torch(device_tensor).reshape(tensor.shape).to(torch.float32)

    for key in keys:
        adapter_base = key[: -len(".weight")]
        a, b, scale = _adapter_pair(turbo, adapter_base)
        expected = base[key].to(torch.float32)
        expected = expected + (b.to(torch.bfloat16).to(torch.float32) @ a.to(torch.bfloat16).to(torch.float32)) * scale
        fused = state[key]
        assert fused.dtype == base[key].dtype, (
            f"{key}: the fuse returned {fused.dtype} where the checkpoint is {base[key].dtype}. Keeping fp32 models a "
            "DIFFERENT model than the adapter was distilled against -- measured at full depth, it costs audio PCC "
            "0.982 -> 0.801 against a reference that fuses the way diffusers does."
        )
        torch.testing.assert_close(fused.to(torch.float32), expected.to(fused.dtype).to(torch.float32), rtol=0, atol=0)

        # The delta has to reach the quantized weight AT ALL -- which the device merge does not, and
        # is the entire defect. The bar is deliberately "more than nothing" and not a fraction:
        # measured here, a bf16 host fuse moves 0.06 % of elements, an fp32 one 10.4 %, and the
        # device merge 0 %. Fraction is the wrong yardstick because the 0.06 % is COHERENT -- the
        # same rank-128 update in all fifty blocks -- and at full depth it is worth audio PCC
        # 0.9739 -> 0.9817 against a reference carrying the same adapter, where the fp32 fuse's
        # hundredfold larger fraction scores 0.801 by no longer matching how diffusers fuses. That
        # end-to-end number is the real proof; this test only guards the necessary condition.
        moved = (quantize(fused) != quantize(base[key])).to(torch.float32).mean().item()
        assert moved > 0.0, (
            f"{key}: quantizing the host-fused weight is bit-identical to quantizing the base weight, so the delta "
            "did not survive the fuse and the adapter would do nothing on a quantized DiT -- which is exactly the "
            "defect the device-side merge has"
        )
        logger.info(f"{key}: the host fuse moves {moved:.4%} of elements through bfloat8_b")
