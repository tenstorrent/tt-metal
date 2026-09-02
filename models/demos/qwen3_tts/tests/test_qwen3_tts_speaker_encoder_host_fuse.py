# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gate for the ECAPA host-fusion change (``QWEN3_TTS_SE_HOST_FUSE``).

The Res2Net branch convs are k=3 dilated with reflect pad, which TTNN cannot do,
so they run on the host. The device path interleaved the cheap glue — slice,
ReLU, add, concat — between them, which meant a D2H/H2D round-trip per branch:
22 device ops and 7 round-trips per SERes2Net block, for ~45 us of device work.
Running the glue on the host beside the convs collapses that to one round-trip.

Two things have to hold:

  * the ops actually disappear (spy on the ttnn calls, not on a config object),
  * the speaker embedding does not drift from the fp32 reference. Fusing removes
    bf16 rounding steps, so the fused path should be at least as close to the
    reference as the device path, never further.

    pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_speaker_encoder_host_fuse.py
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.reference.functional import speaker_encoder_forward
from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder, SpeakerEncoderConfig

# ~4.01 s @ 24 kHz, hop 256 -> ~376 mel frames, padded to one tile (matches the demo).
T = 384
CH = 512
SCALE = 8


def _open_device():
    mesh_shape = {"N150": (1, 1), "N300": (1, 2)}.get(os.environ.get("MESH_DEVICE"))
    if mesh_shape is None:
        return ttnn.open_device(device_id=0, l1_small_size=32768), None
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=32768), mesh_shape


@pytest.fixture(scope="module")
def device():
    d, mesh_shape = _open_device()
    yield d
    if mesh_shape is None:
        ttnn.close_device(d)
    else:
        ttnn.close_mesh_device(d)
        if mesh_shape != (1, 1):
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    ac, bc = a - a.mean(), b - b.mean()
    d = ac.norm() * bc.norm()
    return 0.0 if d < 1e-12 else float((ac * bc).sum() / d)


def _synthetic_sd(seed: int = 0) -> dict:
    """Full ECAPA weights. Scaled down so bf16 activations stay in range."""
    torch.manual_seed(seed)
    part = CH // SCALE
    s = 0.03

    def w(*shape):
        return torch.randn(*shape) * s

    sd = {
        "speaker_encoder.blocks.0.conv.weight": w(CH, 128, 5),
        "speaker_encoder.blocks.0.conv.bias": torch.zeros(CH),
        "speaker_encoder.mfa.conv.weight": w(CH * 3, CH * 3, 1),
        "speaker_encoder.mfa.conv.bias": torch.zeros(CH * 3),
        "speaker_encoder.asp.tdnn.conv.weight": w(128, CH * 3 * 3, 1),
        "speaker_encoder.asp.tdnn.conv.bias": torch.zeros(128),
        "speaker_encoder.asp.conv.weight": w(CH * 3, 128, 1),
        "speaker_encoder.asp.conv.bias": torch.zeros(CH * 3),
        "speaker_encoder.fc.weight": w(2048, CH * 3 * 2, 1),
        "speaker_encoder.fc.bias": torch.zeros(2048),
    }
    for b in (1, 2, 3):
        p = f"speaker_encoder.blocks.{b}."
        sd[p + "tdnn1.conv.weight"] = w(CH, CH, 1)
        sd[p + "tdnn1.conv.bias"] = torch.zeros(CH)
        sd[p + "tdnn2.conv.weight"] = w(CH, CH, 1)
        sd[p + "tdnn2.conv.bias"] = torch.zeros(CH)
        sd[p + "se_block.conv1.weight"] = w(128, CH, 1)
        sd[p + "se_block.conv1.bias"] = torch.zeros(128)
        sd[p + "se_block.conv2.weight"] = w(CH, 128, 1)
        sd[p + "se_block.conv2.bias"] = torch.zeros(CH)
        for i in range(SCALE - 1):
            sd[p + f"res2net_block.blocks.{i}.conv.weight"] = w(part, part, 3)
            sd[p + f"res2net_block.blocks.{i}.conv.bias"] = torch.zeros(part)
    return sd


def _mel(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 128, T)


class _Spy:
    """Counts ttnn calls by name, for the duration of a ``with`` block."""

    NAMES = ("slice", "relu", "tanh", "add", "multiply", "concat")

    def __init__(self, monkeypatch):
        self.mp = monkeypatch
        self.n = {k: 0 for k in self.NAMES}

    def __enter__(self):
        for name in self.NAMES:
            real = getattr(ttnn, name)

            def wrap(*a, _r=real, _n=name, **k):
                self.n[_n] += 1
                return _r(*a, **k)

            self.mp.setattr(ttnn, name, wrap)
        return self

    def __exit__(self, *exc):
        return False


def test_relu_commutes_with_bf16_rounding():
    """The TDNN ReLU fold is only bit-exact if rounding preserves sign and zero."""
    torch.manual_seed(0)
    y = torch.cat([torch.randn(1 << 16) * 10.0, torch.zeros(8), torch.tensor([-0.0, 1e-45, -1e-45])])
    assert torch.equal(torch.relu(y.bfloat16()), torch.relu(y).bfloat16())


@pytest.mark.parametrize("block_idx", [1, 2, 3])
def test_res2net_cascade_removes_device_ops(device, monkeypatch, block_idx):
    """One SERes2Net block: the fused path must issue no slice/relu/concat at all."""
    enc = SpeakerEncoder(device, _synthetic_sd(), config=SpeakerEncoderConfig())
    x = enc._torch_ncl_to_ttnn_nlc(torch.randn(1, CH, T))

    counts = {}
    for fuse in (False, True):
        enc._se_host_fuse = enc._se_device_asp = fuse
        with _Spy(monkeypatch) as spy:
            enc._se_res2net_block(x, block_idx=block_idx, scale=SCALE)
        counts[fuse] = dict(spy.n)
        monkeypatch.undo()

    off, on = counts[False], counts[True]
    print(f"\n[block {block_idx}] device ops OFF={off}\n[block {block_idx}] device ops  ON={on}")

    # Res2Net glue: 8 slices, 7 branch ReLUs, 6 cascade adds, 1 concat.
    assert off["slice"] == SCALE and on["slice"] == 0
    assert off["concat"] == 1 and on["concat"] == 0
    assert off["relu"] == SCALE - 1 and on["relu"] == 0
    # ttnn.add: 6 cascade adds + 1 residual (off) -> residual only (on).
    assert off["add"] == SCALE - 1 and on["add"] == 1
    # The SE scale multiply is real device work and must survive.
    assert off["multiply"] == on["multiply"] == 1


def test_full_encoder_op_count_and_accuracy(device, monkeypatch):
    """Whole ECAPA forward, three configurations, so each change is attributable.

    off  = original device-glue path
    fuse = Res2Net/TDNN/ASP glue moved to the host beside the convs
    +asp = ASP's two k=1 convs moved onto the device as matmuls
    """
    sd = _synthetic_sd()
    mel = _mel()
    ref = speaker_encoder_forward(mel.transpose(1, 2), {k[len("speaker_encoder.") :]: v for k, v in sd.items()})

    CFG = {"off": (False, False), "fuse": (True, False), "+asp": (True, True)}
    out, counts, trips = {}, {}, {}
    for name, (fuse, dev_asp) in CFG.items():
        enc = SpeakerEncoder(device, sd, config=SpeakerEncoderConfig())
        enc._se_host_fuse, enc._se_device_asp = fuse, dev_asp
        n = [0, 0]
        d2h, h2d = enc._ttnn_nlc_to_torch_nlc, enc._torch_nlc_to_ttnn
        enc._ttnn_nlc_to_torch_nlc = lambda *a, _f=d2h, **k: (n.__setitem__(0, n[0] + 1), _f(*a, **k))[1]
        enc._torch_nlc_to_ttnn = lambda *a, _f=h2d, **k: (n.__setitem__(1, n[1] + 1), _f(*a, **k))[1]
        with _Spy(monkeypatch) as spy:
            y = enc.forward(mel)
        counts[name], trips[name] = dict(spy.n), tuple(n)
        monkeypatch.undo()
        out[name] = y.float().reshape(1, -1)

    print()
    for name in CFG:
        c = counts[name]
        print(
            f"{name:>5}  glue ops {sum(c.values()):>3}  {c}  "
            f"round-trips D2H/H2D {trips[name][0]}/{trips[name][1]}  "
            f"PCC vs fp32 ref {_pcc(out[name], ref):.6f}"
        )

    off, fuse, asp = counts["off"], counts["fuse"], counts["+asp"]
    # 3 SERes2Net blocks x (8 slice + 7 relu + 6 add + 1 concat), the entry-TDNN
    # ReLU, and ASP's relu+tanh. The MFA and ASP concats are device work and stay.
    assert fuse["slice"] == 0, "Res2Net channel slices should all be gone"
    assert off["slice"] - fuse["slice"] == 3 * SCALE
    assert off["concat"] - fuse["concat"] == 3, "one Res2Net concat per block"
    assert off["relu"] - fuse["relu"] == 3 * (SCALE - 1) + 2  # +entry TDNN, +ASP
    # 6 cascade adds per block; each block's residual add survives.
    assert off["add"] - fuse["add"] == 3 * (SCALE - 2)
    # ASP on device: its ReLU fuses into the matmul, tanh stays a real device op.
    assert off["relu"] - asp["relu"] == 3 * (SCALE - 1) + 2
    assert off["tanh"] == asp["tanh"] == 1

    # Round-trips are the real cost. Before fusion there is one per host conv:
    # entry TDNN + 3 blocks x 7 Res2Net branches + 2 ASP = 24. Fusion collapses
    # each block's 7 into 1; moving ASP onto the device removes its own.
    assert trips["off"] == (1 + 3 * (SCALE - 1) + 2,) * 2
    assert trips["fuse"] == (1 + 3 + 1,) * 2
    assert trips["+asp"] == (1 + 3,) * 2

    p_off, p_fuse, p_asp = (_pcc(out[n], ref) for n in ("off", "fuse", "+asp"))
    assert _pcc(out["fuse"], out["off"]) > 0.999, "fused output diverged from the device path"
    assert p_fuse >= p_off - 1e-6, f"host fusion moved away from the reference: {p_fuse} < {p_off}"
    # ASP on device swaps an fp32 host conv for a bf16 matmul, so a small loss is
    # expected; it must stay well inside the band the model already tolerates.
    assert p_asp > 0.99, f"speaker embedding PCC too low with device ASP: {p_asp}"


@pytest.mark.parametrize("cin,k,dilation", [(128, 5, 1), (CH // SCALE, 3, 2), (CH // SCALE, 3, 3), (CH // SCALE, 3, 4)])
def test_device_conv_matches_host_conv(device, cin, k, dilation):
    """``_conv1d_device_nlc`` (gather + concat + matmul) vs the host reflect-pad conv.

    Covers every (kernel, dilation) ECAPA uses. The gathers reproduce torch's own
    reflect map exactly, so the only difference left is bf16 vs fp32 accumulation.
    """
    from models.demos.qwen3_tts.tt.mesh_utils import to_torch as mesh_to_torch

    enc = SpeakerEncoder(device, _synthetic_sd(), config=SpeakerEncoderConfig())
    torch.manual_seed(3)
    w, b = torch.randn(cin, cin, k) * 0.03, torch.zeros(cin)
    x = torch.randn(1, T, cin)
    x_tt = ttnn.from_torch(
        x.bfloat16(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    got = mesh_to_torch(
        enc._conv1d_device_nlc(x_tt, w, b, dilation, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)),
        dtype=torch.float32,
    )
    want = torch.relu(enc._conv1d_same_padding_torch_nlc(x, w, b, dilation))
    assert got.shape == want.shape
    assert _pcc(got, want) > 0.999, f"PCC {_pcc(got, want)}"


def test_device_conv_is_off_by_default(device):
    """It is correct but slower — ``ttnn.gather`` costs ~481 us/call. See PERF_NOTES 3.4."""
    enc = SpeakerEncoder(device, _synthetic_sd(), config=SpeakerEncoderConfig())
    assert enc._se_device_conv is False


class _StubDevice:
    """Enough of a device to construct SpeakerEncoder for host-only math checks."""

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(8, 8)

    def arch(self):
        return ttnn._ttnn.device.Arch.WORMHOLE_B0


@pytest.mark.parametrize("dilation", [2, 3, 4])
def test_host_cascade_matches_reference_exactly(dilation):
    """``_res2net_cascade_torch`` must be the reference cascade, op for op.

    Both are fp32 torch doing the same arithmetic in the same order, so this is
    an equality check, not a PCC one — it catches an off-by-one in the branch
    indexing or a wrong add operand, which PCC would happily hide.
    """
    from models.demos.qwen3_tts.reference.functional import res2net_block

    block_idx = dilation - 1
    sd = _synthetic_sd()
    enc = SpeakerEncoder(_StubDevice(), sd, config=SpeakerEncoderConfig())

    torch.manual_seed(7)
    x_ncl = torch.randn(1, CH, T)
    prefix = f"blocks.{block_idx}.res2net_block."

    got = enc._res2net_cascade_torch(x_ncl.permute(0, 2, 1).contiguous(), prefix, SCALE, dilation)
    want = res2net_block(x_ncl, enc.pytorch_weights, prefix, scale=SCALE, dilation=dilation)

    assert got.shape == want.permute(0, 2, 1).shape
    assert torch.equal(
        got, want.permute(0, 2, 1).contiguous()
    ), f"max|diff| = {(got - want.permute(0, 2, 1)).abs().max().item():g}"
