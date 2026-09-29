# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 compressor (bead F2) vs the reference ``Compressor``: ratio 2 and 1, synthetic and real
weights, an odd valid length (partial group -> carry), chunk boundaries, determinism. Bar (G1): 0.999."""

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.v41.compressor import TtV41Compressor
from tests.ttnn.utils_for_testing import comp_pcc

PCC = 0.999
SEQ = 2048
SNAPSHOT = Path.home() / (
    ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277"
)


def _weights(source: str, layer: int, ratio: int, gen) -> dict:
    if source == "synthetic":
        w = {"wkv": torch.randn(C.HEAD_DIM, C.EMB_SIZE, generator=gen) * C.EMB_SIZE**-0.5}
        if ratio > 1:
            w["wgate"] = torch.randn(C.HEAD_DIM, C.EMB_SIZE, generator=gen) * C.EMB_SIZE**-0.5
        w["norm"] = 1 + 0.1 * torch.randn(C.HEAD_DIM, generator=gen)
        return {k: v.to(torch.bfloat16) for k, v in w.items()}
    from safetensors import safe_open

    if not (SNAPSHOT / "model.safetensors.index.json").is_file():
        pytest.skip("V4.1 checkpoint shards not downloaded")
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    keys = {"wkv": "wkv.weight", "norm": "norm.weight"} | ({"wgate": "wgate.weight"} if ratio > 1 else {})
    out = {}
    for k, suffix in keys.items():
        name = f"layers.{layer}.attn.compressor.{suffix}"
        with safe_open(SNAPSHOT / index[name], "pt") as f:
            out[k] = f.get_tensor(name)
    return out


def _reference(ratio, weights, x):
    """Reference Compressor over x [1, L, dim] bf16 -> (rows [L//r, d], kv_state row of an odd tail or None)."""
    args = v41.ModelArgs(
        max_batch_size=1,
        max_seq_len=SEQ,
        dim=C.EMB_SIZE,
        head_dim=C.HEAD_DIM,
        compress_ratios=(ratio,),
        norm_eps=C.RMS_NORM_EPS,
    )
    with v41.set_dtype(torch.bfloat16):
        comp = v41.Compressor(args, 0)
        with torch.no_grad():
            comp.wkv.weight.copy_(weights["wkv"].to(comp.wkv.weight.dtype))
            if ratio > 1:
                comp.wgate.weight.copy_(weights["wgate"].to(comp.wgate.weight.dtype))
            comp.norm.weight.copy_(weights["norm"].float())
            rows = comp(x, 0)
    carry = comp.kv_state[0, 0].clone() if ratio > 1 and x.shape[1] % ratio else None
    return rows[0], carry


@pytest.mark.parametrize("weights_source", ["synthetic", "real"])
@pytest.mark.parametrize("layer", [2, 20], ids=["ratio2-L2", "ratio1-L20"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_compressor(mesh_device, device_params, layer, weights_source):
    ratio = C.compress_ratio(layer)
    gen = torch.Generator().manual_seed(layer)
    weights = _weights(weights_source, layer, ratio, gen)
    shape, tp = tuple(mesh_device.shape), mesh_device.shape[1]
    comp = TtV41Compressor(mesh_device, C, layer, weights)
    x = torch.randn(1, SEQ, C.EMB_SIZE, generator=gen).to(torch.bfloat16)

    def run(tokens):
        tt = ttnn.from_torch(
            tokens[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
        )
        latent, carry = comp(tt)
        down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))
        rows = down(latent)[0, 0, :, : C.HEAD_DIM]  # replicated across TP
        kv = None if carry is None else down(carry[0])[0, 0, :, : C.HEAD_DIM]
        return rows, kv

    rows, kv = run(x[0])
    rows2, _ = run(x[0])
    assert torch.equal(rows, rows2), "compressor is not bit-identical across repeats"

    results = {}
    ref_rows, _ = _reference(ratio, weights, x)
    results["full"] = comp_pcc(ref_rows.float(), rows.float(), 0.0)[1]

    # odd valid length: rows of complete groups only, trailing token's projection is the carry
    valid = SEQ - 3
    ref_odd, ref_carry = _reference(ratio, weights, x[:, :valid])
    results["odd_rows"] = comp_pcc(ref_odd.float(), rows[: valid // ratio].float(), 0.0)[1]
    if ratio > 1:
        results["odd_carry"] = comp_pcc(ref_carry.float(), kv[valid - 1].float(), 0.0)[1]

    # chunk boundary: two chunks of half the sequence give the single-shot rows
    half = SEQ // 2
    c0, _ = run(x[0, :half])
    c1, _ = run(x[0, half:])
    results["chunked"] = comp_pcc(ref_rows.float(), torch.cat([c0, c1]).float(), 0.0)[1]
    print(f"compressor L{layer} {weights_source}: {results}")
    for key, value in results.items():
        assert value >= PCC, (key, results)
