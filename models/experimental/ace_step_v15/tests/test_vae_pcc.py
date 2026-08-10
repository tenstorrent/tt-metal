# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN ACE-Step 1.5 Oobleck VAE decoder (Block 3).

Block boundary: latent ``[1, 64, T]`` (25 Hz) -> waveform ``[1, 2, 1920*T]``
(48 kHz stereo). The TTNN decoder carries the 1-D signal channels-last as
``(B, T, C)`` ROW_MAJOR internally; goldens and oracles are channels-first
``[B, C, T]``, matching the reference.

Per-stage oracles cover **every** one of the 37 convs and 36 Snakes, plus the 15
residual-unit sums and the 5 block outputs — 93 stages. They come from forward
hooks on the fp32 CPU ``diffusers.OobleckDecoder`` loaded from the same folded
weights, so every oracle sees a bit-identical input and a divergence localises to
a single op. All are printed via ``comp_pcc(ref, got, pcc=0.0)``-style output
before the gate.

Goldens (Block 0 owns the dump, ``golden/vae/``, seed 1234, fp32 ``torch.save``):

    latents.pt            [1, 64, T]        decoder input
    decoder_state_dict.pt dict              decoder weights (raw weight_g/_v or folded)
    waveform.pt           [1, 2, 1920*T]    reference decoder output

When ``golden/vae/`` is empty the test self-hosts: it builds the reference decoder
at seed 1234, uses *its* weights and a seeded random latent, and gates against its
own fp32 CPU output. That exercises the full 37-conv / 36-Snake graph and every
TTNN op; it just does not pin the numbers to the deployed checkpoint. The mode is
printed loudly.

Sizing: the per-stage pass runs at ``ACE_STEP_VAE_T`` latent frames (default 32 =
2.56 s, the §5b "unit" duration and the 32-frame floor imposed by the fp32 conv
blocking table's ``T_out_block=32``). 93 fp32 oracles at 32 frames is ~1.1 GB of
host RAM; at a 512-frame chunk a single block.4 intermediate is already 503 MB, so
do not raise this casually. Set ``ACE_STEP_VAE_FULL=1`` to additionally gate the
full-length golden waveform with recording off.

fp32 activations are mandatory (Snake overflows fp16 for alpha > ~11; bf16
accumulation caps waveform PCC at ~0.96 through this conv chain) and the fp32 conv
config tensor needs ``l1_small_size=65536``.
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tt.ttnn_ace_step_vae import (
    AUDIO_CHANNELS,
    CHANNEL_MULTIPLES,
    DECODER_CHANNELS,
    DECODER_INPUT_CHANNELS,
    NUM_CONVS,
    NUM_FOLDED_STATE_TENSORS,
    NUM_PARAMS,
    NUM_SNAKES,
    TOTAL_UPSAMPLE,
    UPSAMPLING_RATIOS,
    OobleckDecoder,
    apply_conv3d_blockings,
    prepare_decoder_state_dict,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "golden", "vae")
SEED = 1234

TARGET_PCC = 0.99  # final waveform gate (block registry §5b)
STAGE_PCC = 0.995  # per-stage gate
CHUNKED_PCC = 0.999  # chunked (overlap-discard) vs single-pass decode
TRACED_PCC = 0.9999  # traced replay vs eager device run

STRICT_STAGES = os.environ.get("ACE_STEP_VAE_STRICT_STAGES", "1") != "0"
DEFAULT_T = int(os.environ.get("ACE_STEP_VAE_T", "32"))
FULL_WAVEFORM = os.environ.get("ACE_STEP_VAE_FULL", "0") != "0"
NUM_STAGES = 93  # 37 convs + 36 Snakes + 15 residual sums + 5 block outputs

_WEIGHT_NORM_RAW = ("weight_g", "weight_v", "parametrizations.weight.original")


# ---------------------------------------------------------------------------
# Reference (fp32 CPU diffusers) — oracle source
# ---------------------------------------------------------------------------


def _is_raw_weight_norm(state_dict) -> bool:
    return any(m in k for k in state_dict for m in _WEIGHT_NORM_RAW)


def _strip_prefix(state_dict):
    """Drop a leading ``decoder.``-style prefix. The *reference* also needs this — it consumes
    the raw weight_norm form directly, so it cannot go through ``prepare_decoder_state_dict``
    (which folds)."""
    for prefix in ("vae.decoder.", "model.decoder.", "decoder."):
        if any(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return dict(state_dict)


def _strip_weight_norm(ref) -> None:
    """Materialise ``w = g*v/||v||`` in place so the reference accepts folded weights."""
    for module in ref.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                pass  # already plain


def _build_reference(*, folded_weights: bool):
    """Reference ``OobleckDecoder`` with the deployed ACE-Step 1.5 geometry.

    ``folded_weights=True`` removes ``weight_norm`` first, so the module's own
    state dict is the 145-tensor folded form our TTNN decoder consumes.
    """
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoder as RefDecoder

    ref = RefDecoder(
        channels=DECODER_CHANNELS,
        input_channels=DECODER_INPUT_CHANNELS,
        audio_channels=AUDIO_CHANNELS,
        upsampling_ratios=list(UPSAMPLING_RATIOS),
        channel_multiples=list(CHANNEL_MULTIPLES),
    ).float()
    n = sum(p.numel() for p in ref.parameters())
    assert n == NUM_PARAMS, f"weight-normed reference has {n} params, expected {NUM_PARAMS}"
    if folded_weights:
        _strip_weight_norm(ref)
    ref.eval()
    return ref


def _load_reference(state_dict):
    """Reference decoder loaded from either the raw or the folded state dict."""
    raw = _is_raw_weight_norm(state_dict)
    ref = _build_reference(folded_weights=not raw)
    ref.load_state_dict(state_dict, strict=True)
    return ref


def _seeded_state_dict():
    """Reproducible folded (145-tensor) weights for the no-goldens path.

    Snake ``alpha``/``beta`` initialise to zeros (``exp(0) = 1``), a degenerate
    activation; give them spread so ``ttnn.snake_beta`` is genuinely exercised.
    Kept modest (``0.5 * randn``) so ``exp(alpha)`` stays well below the
    ``alpha > ~11`` fp16-overflow regime — this test is about graph correctness,
    not overflow.
    """
    torch.manual_seed(SEED)
    ref = _build_reference(folded_weights=False)
    with torch.no_grad():
        for module in ref.modules():
            if type(module).__name__ == "Snake1d":
                module.alpha.copy_(0.5 * torch.randn_like(module.alpha))
                module.beta.copy_(0.5 * torch.randn_like(module.beta))
    _strip_weight_norm(ref)
    sd = ref.state_dict()
    assert len(sd) == NUM_FOLDED_STATE_TENSORS, f"{len(sd)} tensors, expected {NUM_FOLDED_STATE_TENSORS}"
    return sd


def _load_goldens(fallback_t: int):
    """``(state_dict, latents, golden_waveform_or_None, using_goldens)``."""
    sd_path = os.path.join(GOLDEN, "decoder_state_dict.pt")
    lat_path = os.path.join(GOLDEN, "latents.pt")
    wav_path = os.path.join(GOLDEN, "waveform.pt")

    if os.path.exists(sd_path) and os.path.exists(lat_path):
        state_dict = _strip_prefix(torch.load(sd_path, map_location="cpu", weights_only=False))
        latents = torch.load(lat_path, map_location="cpu").float()
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        waveform = torch.load(wav_path, map_location="cpu").float() if os.path.exists(wav_path) else None
        return state_dict, latents, waveform, True

    print(
        f"  !! no goldens in {os.path.normpath(GOLDEN)} — self-hosting the oracle from a seed-{SEED} "
        "reference decoder. The graph is fully exercised; the numbers are NOT pinned to the "
        "deployed checkpoint. Land Block 0's golden/vae dump to pin them."
    )
    torch.manual_seed(SEED + 1)
    latents = torch.randn(1, DECODER_INPUT_CHANNELS, fallback_t)
    return _seeded_state_dict(), latents, None, False


def _reference_oracles(state_dict, latents):
    """Run the fp32 CPU reference with a forward hook on every submodule.

    Returns ``(waveform, {module_path: [B, C, T] fp32})``. The keys are exactly the
    stage names the TTNN decoder records, because both use the reference's module
    paths (``conv1``, ``block.0.res_unit2.conv1``, ``snake1``, ``conv2``, ...).
    """
    ref = _load_reference(state_dict)

    oracles: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            oracles[name] = output.detach().float()

        return hook

    for name, module in ref.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        waveform = ref(latents).float()

    for h in handles:
        h.remove()

    n_conv = sum(1 for _n, m in ref.named_modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)))
    n_snake = sum(1 for _n, m in ref.named_modules() if type(m).__name__ == "Snake1d")
    assert n_conv == NUM_CONVS, f"{n_conv} convs, expected {NUM_CONVS}"
    assert n_snake == NUM_SNAKES, f"{n_snake} Snakes, expected {NUM_SNAKES}"
    return waveform, oracles


def _reference_waveform(state_dict, latents):
    with torch.no_grad():
        return _load_reference(state_dict)(latents).float()


# ---------------------------------------------------------------------------
# Main runners
# ---------------------------------------------------------------------------


def _build_ttnn(device, state_dict):
    model = OobleckDecoder(mesh_device=device, dtype=ttnn.float32)
    patched = apply_conv3d_blockings(model)  # TRAP-3 — must precede the load
    assert len(patched) == NUM_CONVS, f"patched {len(patched)} conv blockings, expected {NUM_CONVS}"
    model.load_torch_state_dict(prepare_decoder_state_dict(state_dict))
    return model


def _stage_frames(total: int, cap: int) -> int:
    """Largest multiple of 32 that is <= min(total, cap); the conv table needs >= 32."""
    n = min(total, max(cap, 32))
    n -= n % 32
    assert n >= 32, f"latent T={total} is below the 32-frame floor of the fp32 conv blocking table"
    return n


def run_vae_pcc(device, verbose=True, t_frames=DEFAULT_T, full_waveform=FULL_WAVEFORM):
    state_dict, latents, gold_wav, using_goldens = _load_goldens(t_frames)
    total_t = latents.shape[-1]
    t_stage = _stage_frames(total_t, t_frames)
    lat_stage = latents[:, :, :t_stage].contiguous()
    print(
        f"  VAE decoder: stage pass on latent [1, {DECODER_INPUT_CHANNELS}, {t_stage}] -> "
        f"[1, {AUDIO_CHANNELS}, {t_stage * TOTAL_UPSAMPLE}]  "
        f"({'goldens' if using_goldens else 'self-hosted oracle'}; golden T={total_t})"
    )

    ref_wav, oracles = _reference_oracles(state_dict, lat_stage)
    model = _build_ttnn(device, state_dict)

    # Stream the comparison: pop each oracle as it is matched so we hold ~one extra
    # stage rather than two full sets.
    results: list[tuple[str, float, bool]] = []
    unmatched: list[str] = []

    def sink(name: str, got: torch.Tensor) -> None:
        ref = oracles.pop(name, None)
        if ref is None:
            unmatched.append(name)
            return
        assert tuple(ref.shape) == tuple(got.shape), f"[{name}] ref {tuple(ref.shape)} vs ttnn {tuple(got.shape)}"
        passed, pcc = comp_pcc(ref, got, pcc=STAGE_PCC)
        results.append((name, pcc, passed))
        if verbose:
            print(f"  {' ' if passed else '!'} [{name:<34s}] {str(tuple(got.shape)):<22s} pcc {pcc}")
        del ref

    wav = model.forward(lat_stage, record=sink)

    assert not unmatched, f"TTNN recorded stages with no reference oracle: {unmatched}"
    assert not oracles, f"reference stages never recorded by TTNN: {sorted(oracles)}"
    assert len(results) == NUM_STAGES, f"{len(results)} stage oracles, expected {NUM_STAGES}"
    if verbose:
        print(f"  compared {len(results)} stages ({NUM_CONVS} convs + {NUM_SNAKES} Snakes + 20 sums)")

    failed = [(n, p) for n, p, ok in results if not ok]
    assert tuple(wav.shape) == tuple(ref_wav.shape), f"waveform {tuple(wav.shape)} vs ref {tuple(ref_wav.shape)}"
    passed, pcc = comp_pcc(ref_wav, wav, pcc=TARGET_PCC)
    print(f"waveform {tuple(wav.shape)} vs fp32 CPU reference  pcc: {pcc}")

    if full_waveform and gold_wav is not None and t_stage != total_t:
        wav_full = model.decode(latents)
        assert tuple(wav_full.shape) == tuple(
            gold_wav.shape
        ), f"full waveform {tuple(wav_full.shape)} vs golden {tuple(gold_wav.shape)}"
        full_passed, full_pcc = comp_pcc(gold_wav, wav_full, pcc=TARGET_PCC)
        print(f"waveform {tuple(wav_full.shape)} vs golden/vae/waveform.pt  pcc: {full_pcc}")
        passed, pcc = (passed and full_passed), min(pcc, full_pcc)

    if failed and STRICT_STAGES:
        worst = sorted(failed, key=lambda x: x[1])[:8]
        return False, f"{len(failed)} stage(s) below {STAGE_PCC}; worst: {worst}"
    if failed:
        print(f"  (non-strict) {len(failed)} stage(s) below {STAGE_PCC}: {sorted(failed, key=lambda x: x[1])[:8]}")
    return passed, pcc


def run_vae_chunked_pcc(device, verbose=True):
    """Gate chunked overlap-discard decode against a single-pass decode.

    Uses a scaled-down chunk (128/32/64 rather than the production 512/64/384) so
    the test is cheap while still crossing four chunk boundaries. The decoder's
    latent-domain receptive field is ~9 frames (conv1's 3, plus block.0's 39 taps
    at 10x the latent rate, plus a fast-shrinking tail), so 32 discarded frames of
    context is ample and the two paths should agree to near fp32 precision.
    """
    chunk, overlap, t_frames = 128, 32, 256
    state_dict, _lat, _wav, using_goldens = _load_goldens(t_frames)
    torch.manual_seed(SEED + 2)
    latents = torch.randn(1, DECODER_INPUT_CHANNELS, t_frames)

    model = _build_ttnn(device, state_dict)
    whole = model.decode(latents, chunked=False)
    parts = model.decode(latents, chunked=True, chunk=chunk, overlap=overlap)

    assert tuple(parts.shape) == tuple(whole.shape), f"chunked {tuple(parts.shape)} vs whole {tuple(whole.shape)}"
    passed, pcc = comp_pcc(whole, parts, pcc=CHUNKED_PCC)
    if verbose:
        print(
            f"chunked(chunk={chunk}, overlap={overlap}, stride={chunk - 2 * overlap}) vs single-pass "
            f"{tuple(parts.shape)}  pcc: {pcc}  ({'goldens' if using_goldens else 'self-hosted'})"
        )
    return passed, pcc


def run_vae_traced_pcc(device, verbose=True, t_frames=DEFAULT_T):
    """Gate the captured/replayed device trace against the eager device graph."""
    state_dict, latents, _wav, _g = _load_goldens(t_frames)
    t_stage = _stage_frames(latents.shape[-1], t_frames)
    latents = latents[:, :, :t_stage].contiguous()

    model = _build_ttnn(device, state_dict)
    eager = model.forward(latents)
    try:
        model.forward_traced(latents)  # capture
        replay = model.forward_traced(latents)  # replay
    finally:
        model.release_trace()
    passed, pcc = comp_pcc(eager, replay, pcc=TRACED_PCC)
    if verbose:
        print(f"traced replay vs eager {tuple(replay.shape)}  pcc: {pcc}")
    return passed, pcc


# ---------------------------------------------------------------------------
# pytest wrappers
# ---------------------------------------------------------------------------

# ttnn.conv sliding-window/halo config lives in L1_SMALL, and the fp32 conv config
# tensor needs more than bf16's 32768 — use 65536 (same as the XTTS HiFi-GAN and
# the LTX vocoder).
_DEVICE_PARAMS = [{"l1_small_size": 65536}]
_TRACE_DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 200_000_000}]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_vae_pcc(device):
    passed, msg = run_vae_pcc(device)
    assert passed, f"VAE decoder PCC below {TARGET_PCC}: {msg}"


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
def test_vae_chunked_pcc(device):
    passed, msg = run_vae_chunked_pcc(device)
    assert passed, f"chunked decode PCC below {CHUNKED_PCC}: {msg}"


@pytest.mark.parametrize("device_params", _TRACE_DEVICE_PARAMS, indirect=True)
def test_vae_traced_pcc(device):
    passed, msg = run_vae_traced_pcc(device)
    assert passed, f"traced decode PCC below {TRACED_PCC}: {msg}"


if __name__ == "__main__":
    import sys
    import time

    which = sys.argv[1] if len(sys.argv) > 1 else "pcc"
    runners = {"pcc": run_vae_pcc, "chunked": run_vae_chunked_pcc, "traced": run_vae_traced_pcc}
    if which not in runners:
        print(f"usage: {sys.argv[0]} [{'|'.join(runners)}]")
        sys.exit(2)

    kwargs = {"l1_small_size": 65536}
    if which == "traced":
        kwargs["trace_region_size"] = 200_000_000

    dev = None
    for attempt in range(20):
        try:
            dev = ttnn.open_device(device_id=0, **kwargs)
            break
        except Exception as e:  # device momentarily busy (shared with other agents)
            print(f"open_device attempt {attempt} failed ({e}); retrying in 45s")
            time.sleep(45)
    if dev is None:
        print("FAILED could not open device")
        sys.exit(1)
    try:
        dev.enable_program_cache()
        ok, msg = runners[which](dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
