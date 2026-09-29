# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 DSpark prefill seeding (beads 9.1 / 9.2) vs the reference oracle.

The oracle runs the vendored reference single-shot (backbone, then ``forward_spec`` at start_pos 0) and captures
the block input streams of the tap layers, the concatenated taps (``main_hidden``) and the DSpark window rings.
The device gets the oracle's tap-layer streams, per chunk and padded with junk rows beyond the valid length, and
must reproduce: the taps (N6), main_x = main_norm(main_proj(taps)) on the rows it seeds (D1, G1 bar for new
linear ops 0.999), and the final rings (D2, 0.998), bit-identically on repeat. Rings slots a prompt never reaches
must stay zero. Cases: prompts shorter than the window, longer than it, and a chunked prompt whose last
``window`` positions span two chunks.

"small" is the oracle's small schedule (one DSpark layer, taps of layers 4 and 5, window 16); "real" is V4.1
layers 20, 36-39 with the three DSpark layers at real dims (window 128). Weights: synthetic, or ("real_ckpt") the
checkpoint's DSpark units ``mtp.0-2`` on the synthetic backbone of "real" (the backbone shards are not downloaded;
the backbone only produces the taps, which are teacher-forced from the oracle either way).
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as o
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.v41.dspark import TtV41DSpark
from models.demos.deepseek_v3_d_p.tt.v41.weights import dequant_fp8_block
from tests.ttnn.utils_for_testing import comp_pcc

MAIN_X_PCC = 0.999
TAP_PCC = 0.999
RING_PCC = 0.998

REAL_LAYERS = (20, 36, 37, 38, 39)
DSPARK_UNITS = tuple(f"mtp.{k}" for k in range(DeepSeekV41FlashConfig.NUM_DSPARK_LAYERS))
REAL_PROMPTS = [(100, [100]), (300, [300]), (300, [256, 44])]  # window 128: < window, > window, last 128 span two

# (spec, prompt length, chunk lengths); every chunk but the last is a multiple of 2*32*sp (sp = 2)
CASES = {
    "small": (
        lambda: o.small_spec(seq_len=256),
        [(13, [13]), (100, [100]), (135, [128, 7])],  # window 16: < window, > window, last 16 span two chunks
    ),
    "real": (lambda: o.real_spec(REAL_LAYERS, 512, dspark=True), REAL_PROMPTS),
    "real_ckpt": (
        lambda: o.real_spec(REAL_LAYERS, 512, dspark=True, checkpoint=o.HF_SNAPSHOT, checkpoint_units=DSPARK_UNITS),
        REAL_PROMPTS,
    ),
}


def dspark_shards_present() -> bool:
    """Whether the checkpoint shards of the DSpark units are downloaded."""
    import json

    index = o.HF_SNAPSHOT / "model.safetensors.index.json"
    if not index.is_file():
        return False
    weight_map = json.loads(index.read_text())["weight_map"]
    files = {f for name, f in weight_map.items() if name.split(".")[0] == "mtp"}
    return bool(files) and all((o.HF_SNAPSHOT / f).is_file() for f in files)


def _config(args: v41.ModelArgs):
    """The config attributes TtV41DSpark reads, from the reference args (equal to the canonical ones at real
    dims, checked in the test)."""
    return type(
        "DSparkTestConfig",
        (DeepSeekV41FlashConfig,),
        {
            "EMB_SIZE": args.dim,
            "HC_MULT": args.hc_mult,
            "HEAD_DIM": args.head_dim,
            "QK_ROPE_HEAD_DIM": args.rope_head_dim,
            "SLIDING_WINDOW": args.window_size,
            "RMS_NORM_EPS": args.norm_eps,
            "ROPE_THETA": args.rope_theta,
            "NUM_DSPARK_LAYERS": args.n_mtp_layers,
            "DSPARK_TARGET_LAYER_IDS": args.dspark_target_layer_ids,
        },
    )


def _fp8(linear) -> torch.Tensor:
    return dequant_fp8_block(linear.weight.detach(), linear.scale.detach())


def _weights(model: v41.Transformer) -> dict:
    stage0 = model.mtp[0]
    return {
        "main_proj": _fp8(stage0.main_proj),
        "main_norm": stage0.main_norm.weight.detach(),
        "layers": [{"wkv": _fp8(b.attn.wkv), "kv_norm": b.attn.kv_norm.weight.detach()} for b in model.mtp],
    }


def _pack(x: torch.Tensor, tp: int) -> torch.Tensor:
    """[S, hc, D] -> [1, 1, S, hc * D] with chip t's columns = its D/tp slice of every copy (mhc.py layout)."""
    s, n, d = x.shape
    return x.reshape(s, n, tp, d // tp).permute(0, 2, 1, 3).reshape(1, 1, s, n * d)


def _pcc(a, b) -> float:
    return comp_pcc(a.float(), b.float(), 0.0)[1]


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("dims", ["small", "real", "real_ckpt"])
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
def test_v41_dspark_seeding(mesh_device, device_params, dims):
    make_spec, prompts = CASES[dims]
    if dims == "real_ckpt" and not dspark_shards_present():
        pytest.skip(f"DSpark checkpoint shards not in {o.HF_SNAPSHOT}")
    spec = make_spec()
    args = spec.args
    cfg = _config(args)
    if dims != "small":
        for name in ("EMB_SIZE", "HEAD_DIM", "QK_ROPE_HEAD_DIM", "SLIDING_WINDOW", "RMS_NORM_EPS", "ROPE_THETA"):
            assert getattr(cfg, name) == getattr(DeepSeekV41FlashConfig, name), name
    tap_ids = [spec.layer_ids[p] for p in args.dspark_target_layer_ids]
    if dims != "small":
        assert tap_ids == list(DeepSeekV41FlashConfig.DSPARK_TARGET_LAYER_IDS)
    model = o.build_reference(spec)
    shape, (sp, tp) = tuple(mesh_device.shape), tuple(mesh_device.shape)
    pad = 2 * 32 * sp
    dim, window, n_taps = args.dim, args.window_size, len(tap_ids)
    dspark = TtV41DSpark(mesh_device, cfg, _weights(model))
    gen = torch.Generator().manual_seed(3)

    def first_device(t) -> torch.Tensor:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])[0, 0]

    def all_devices_equal(t) -> bool:
        views = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
        return all(torch.equal(views[0], v) for v in views[1:])

    results, exact = {}, {}
    for seq, chunks in prompts:
        tag = f"S{seq}-" + "+".join(map(str, chunks))
        result = o.oracle(spec, o.random_tokens(spec, seq), model)
        main_hidden = result["state"]["main_hidden"]  # [S, n_taps * dim]
        with v41.set_dtype(torch.bfloat16), torch.no_grad():
            ref_main_x = model.mtp[0].main_norm(model.mtp[0].main_proj(main_hidden[None]))[0]

        def chunk_taps(start, length):
            padded = -(-length // pad) * pad
            taps = []
            for lid in tap_ids:
                x = torch.randn(padded, args.hc_mult, dim, generator=gen) * 100  # junk beyond the valid rows
                x[:length] = result["blocks"][lid]["x_in"][start : start + length].float()
                tt = ttnn.from_torch(
                    _pack(x, tp),
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
                )
                taps.append(dspark.tap(tt))
            return taps

        def run(check: bool):
            rings, start, main_xs = dspark.new_rings(), 0, []
            for length in chunks:
                taps = chunk_taps(start, length)
                if check:
                    for i, t in enumerate(taps):
                        dev = ttnn.to_torch(
                            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3))
                        )[0, 0, :length]
                        ref = main_hidden[start : start + length, i * dim : (i + 1) * dim]
                        results[f"{tag}_c{start}_tap{tap_ids[i]}"] = _pcc(ref, dev)
                main_x, a = dspark.project(taps, length)
                end = min(main_x.shape[2], length - a)
                main_xs.append(first_device(main_x)[:end])
                if check:
                    ref = ref_main_x[start + a : start + a + end]
                    results[f"{tag}_c{start}_main_x"] = _pcc(ref, main_xs[-1])
                    exact[f"{tag}_c{start}_main_x_replicated"] = all_devices_equal(main_x)
                dspark.seed(taps, start, length, rings, dspark.seed_rope(start, length))
                start += length
            return [first_device(r) for r in rings], main_xs, rings

        rings, main_xs, tt_rings = run(check=True)
        rings2, main_xs2, _ = run(check=False)
        exact[f"{tag}_deterministic"] = all(torch.equal(a, b) for a, b in zip(rings + main_xs, rings2 + main_xs2))
        for k, ring in enumerate(rings):
            ref = result["state"]["dspark_window"][k]
            results[f"{tag}_ring{k}"] = _pcc(ref, ring)
            exact[f"{tag}_ring{k}_replicated"] = all_devices_equal(tt_rings[k])
            if seq < window:  # slots of positions the prompt never reached stay zero (rule 8: padding writes nothing)
                exact[f"{tag}_ring{k}_unwritten_zero"] = bool((ring[seq:] == 0).all())
            exact_rows = (ring.float() == ref.float()).all(dim=-1).float().mean().item()
            print(f"{tag} ring{k}: rows bit-equal to the reference {exact_rows:.3f}")
    print(f"dspark {dims}: {results}")
    print(f"dspark {dims} exact: {exact}")
    for key, value in results.items():
        bar = RING_PCC if "_ring" in key else MAIN_X_PCC if key.endswith("main_x") else TAP_PCC
        assert value >= bar, (key, value, bar)
    for key, value in exact.items():
        assert value, key
