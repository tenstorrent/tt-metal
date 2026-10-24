# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only, bit-exact checks of the port's pure-torch stages against the diffusers ``Flux2KleinPipeline`` code
itself (no device, no weights except the VAE BatchNorm statistics when the 4B snapshot is in the HF cache).

Covered, per canvas (1024x1024 T2I, 1024x1024 + reference(s), 2048x2048 T2I):

* ``compute_empirical_mu`` and the whole ``FlowMatchEulerDiscreteScheduler`` schedule (sigmas, timesteps, ``dt``)
  against ``packing``/``scheduler``;
* ``prepare_latents`` (bf16 ``randn_tensor`` + ``_pack_latents``) and ``_prepare_latent_ids`` against
  ``make_noise`` / ``latent_ids``;
* ``_prepare_text_ids`` against ``text_ids``;
* ``prepare_image_latents`` (``_patchify_latents`` -> BatchNorm -> ``_prepare_image_ids`` -> ``_pack_latents`` ->
  cat) against ``patchify_latents`` / ``bn_normalize`` / ``reference_ids``, through a fake VAE so the reference's own
  method bodies run;
* the per-step timestep chain the DiT embeds and ``scheduler.step`` (fp32 upcast, bf16 ``dt * v``, cast back) over
  the full schedule, with random bf16 velocities, against ``Flux2KleinScheduler.update``;
* the base model's CFG combine against ``cfg_velocity``;
* ``_unpack_latents_with_ids`` -> inverse BatchNorm -> ``_unpatchify_latents`` against ``bn_denormalize`` /
  ``unpatchify_tokens`` / ``latent_grid_from_ids``.

Every comparison is ``torch.equal`` (plus dtype): these are pure torch on both sides and must agree exactly.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ....pipelines.flux2_klein import packing
from ....pipelines.flux2_klein import scheduler as port_sched

diffusers = pytest.importorskip("diffusers")
from diffusers.pipelines.flux2.pipeline_flux2_klein import (  # noqa: E402
    Flux2KleinPipeline,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402

# scheduler/scheduler_config.json of black-forest-labs/FLUX.2-klein-4B (used when the snapshot is not cached).
_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
_VAE_SCALE_FACTOR = 8  # 2 ** (len(block_out_channels) - 1) with 4 blocks
_BN_EPS = 1e-4  # vae/config.json batch_norm_eps
_IN_CHANNELS = 128

# (name, canvas (height, width), reference canvases)
CASES = [
    ("t2i_1024", (1024, 1024), []),
    ("ref_1024_plus_1", (1024, 1024), [(1024, 1024)]),
    ("ref_1024_plus_2_mixed", (1024, 1024), [(768, 1024), (1024, 576)]),
    ("t2i_2048", (2048, 2048), []),
]
STEPS = [4, 50]


@functools.lru_cache(maxsize=None)
def _snapshot() -> Path | None:
    from .common import variant_checkpoint_dir

    try:
        return variant_checkpoint_dir("distilled", "4b")
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=None)
def _scheduler_config() -> dict:
    """The checkpoint's scheduler_config.json when cached (read once: the shares are slow), else the copy above."""
    snapshot = _snapshot()
    if snapshot is not None and (snapshot / "scheduler" / "scheduler_config.json").exists():
        raw = json.loads((snapshot / "scheduler" / "scheduler_config.json").read_text())
        config = {k: v for k, v in raw.items() if not k.startswith("_")}
        assert config == _SCHEDULER_CONFIG, config
        return config
    return dict(_SCHEDULER_CONFIG)


def _reference_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler(**_scheduler_config())


@functools.lru_cache(maxsize=None)
def _bn_stats() -> list[tuple[str, torch.Tensor, torch.Tensor, float]]:
    """``(label, running_mean, running_var, eps)``: the checkpoint's bf16 statistics when cached (the pipeline runs
    the VAE in bf16, so its buffers are bf16; read once), plus random fp32 / bf16 statistics."""
    out = []
    snapshot = _snapshot()
    if snapshot is not None and (snapshot / "vae" / "diffusion_pytorch_model.safetensors").exists():
        from safetensors import safe_open

        with safe_open(str(snapshot / "vae" / "diffusion_pytorch_model.safetensors"), "pt") as f:
            mean = f.get_tensor("bn.running_mean")
            var = f.get_tensor("bn.running_var")
        eps = json.loads((snapshot / "vae" / "config.json").read_text())["batch_norm_eps"]
        out.append(("checkpoint_bf16", mean.to(torch.bfloat16), var.to(torch.bfloat16), float(eps)))
    g = torch.Generator().manual_seed(7)
    mean = torch.randn(_IN_CHANNELS, generator=g, dtype=torch.float32) * 0.1
    var = torch.rand(_IN_CHANNELS, generator=g, dtype=torch.float32) * 4 + 0.5
    out.append(("random_fp32", mean, var, _BN_EPS))
    out.append(("random_bf16", mean.to(torch.bfloat16), var.to(torch.bfloat16), _BN_EPS))
    return out


class _FakeLatentDist:
    def __init__(self, z: torch.Tensor) -> None:
        self._z = z

    def mode(self) -> torch.Tensor:
        return self._z

    def sample(self, generator=None) -> torch.Tensor:  # noqa: ARG002
        raise AssertionError("the pipeline must take the posterior mode (sample_mode='argmax')")


class _FakeVAE:
    """Stands in for ``AutoencoderKLFlux2`` in ``prepare_image_latents``: ``encode`` hands back the pre-drawn
    ``(1, 32, H/8, W/8)`` latents of the image it is given (keyed by pixel shape), ``bn`` / ``config`` carry the
    BatchNorm statistics."""

    def __init__(self, latents_by_shape: dict, mean: torch.Tensor, var: torch.Tensor, eps: float) -> None:
        self._latents = latents_by_shape
        self.bn = SimpleNamespace(running_mean=mean, running_var=var)
        self.config = SimpleNamespace(batch_norm_eps=eps)
        self.dtype = torch.bfloat16

    def encode(self, image: torch.Tensor):
        return SimpleNamespace(latent_dist=_FakeLatentDist(self._latents[tuple(image.shape)]))


def _fake_pipeline_self(vae: _FakeVAE | None = None) -> SimpleNamespace:
    """A ``self`` for the pipeline's instance methods carrying only what they read."""
    host = SimpleNamespace(
        vae_scale_factor=_VAE_SCALE_FACTOR,
        vae=vae,
        _prepare_latent_ids=Flux2KleinPipeline._prepare_latent_ids,
        _pack_latents=Flux2KleinPipeline._pack_latents,
        _patchify_latents=Flux2KleinPipeline._patchify_latents,
        _unpatchify_latents=Flux2KleinPipeline._unpatchify_latents,
        _prepare_image_ids=Flux2KleinPipeline._prepare_image_ids,
        _unpack_latents_with_ids=Flux2KleinPipeline._unpack_latents_with_ids,
    )
    host._encode_vae_image = lambda image, generator: Flux2KleinPipeline._encode_vae_image(host, image, generator)
    return host


def _grid(canvas: tuple[int, int]) -> tuple[int, int]:
    return canvas[0] // 16, canvas[1] // 16


@pytest.fixture(autouse=True, scope="module")
def _few_torch_threads():
    """The tt_dit conftest sets ``torch.set_num_threads(os.cpu_count())``; on a shared login box that makes these
    tiny elementwise loops ~100x slower (thread spin under the per-user thread limit). 8 is plenty here."""
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    yield
    torch.set_num_threads(previous)


# --- schedule ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("name,canvas,refs", CASES, ids=[c[0] for c in CASES])
def test_schedule_matches_diffusers(name: str, canvas: tuple[int, int], refs: list, steps: int) -> None:
    h, w = _grid(canvas)
    image_seq_len = h * w  # references excluded (P:817)
    assert port_sched.compute_empirical_mu(image_seq_len, steps) == compute_empirical_mu(image_seq_len, steps)

    reference = _reference_scheduler()
    sigmas_in = np.linspace(1.0, 1 / steps, steps)
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    timesteps, n = retrieve_timesteps(reference, steps, None, sigmas=sigmas_in, mu=mu)
    assert n == steps

    port = port_sched.Flux2KleinScheduler(steps, image_seq_len)
    assert port.mu == mu
    assert len(port) == steps
    assert port.sigmas.dtype == reference.sigmas.dtype == torch.float32
    assert torch.equal(port.sigmas, reference.sigmas), (port.sigmas, reference.sigmas)
    assert torch.equal(port.timesteps, timesteps), (port.timesteps, timesteps)
    assert torch.equal(port_sched.sigmas(steps, image_seq_len), reference.sigmas)
    for i, step in enumerate(port.steps()):
        assert step.index == i
        assert step.sigma == reference.sigmas[i].item()
        assert step.sigma_next == reference.sigmas[i + 1].item()
        assert step.dt == (reference.sigmas[i + 1] - reference.sigmas[i]).item()
        assert step.timestep == timesteps[i].item()
    # The last sigma is the appended terminal 0 (S:386); the first is the shifted 1.0.
    assert reference.sigmas[-1].item() == 0.0 and reference.sigmas[0].item() == 1.0


# --- noise, ids -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,canvas,refs", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("seed", [0, 1234567])
def test_noise_and_ids_match_diffusers(name: str, canvas: tuple[int, int], refs: list, seed: int) -> None:
    h, w = _grid(canvas)
    host = _fake_pipeline_self()
    generator = torch.Generator().manual_seed(seed)
    latents, latent_ids = Flux2KleinPipeline.prepare_latents(
        host,
        batch_size=1,
        num_latents_channels=_IN_CHANNELS // 4,
        height=canvas[0],
        width=canvas[1],
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        generator=generator,
    )
    assert latents.dtype == torch.bfloat16 and latents.shape == (1, h * w, _IN_CHANNELS)
    assert packing.latent_shape(*canvas) == (h, w)

    noise = packing.make_noise(seed, h, w)
    assert noise.dtype == torch.float32 and noise.shape == (h * w, _IN_CHANNELS)
    assert torch.equal(noise, latents[0].float())
    assert torch.equal(noise.to(torch.bfloat16), latents[0])

    ids = packing.latent_ids(h, w)
    assert ids.dtype == latent_ids.dtype == torch.int64
    assert torch.equal(ids, latent_ids[0])
    assert packing.latent_grid_from_ids(ids) == (h, w)

    text = torch.zeros(1, packing.FLUX2_KLEIN_TEXT_LENGTH, 16, dtype=torch.bfloat16)
    ref_text_ids = Flux2KleinPipeline._prepare_text_ids(text)
    port_text_ids = packing.text_ids(packing.FLUX2_KLEIN_TEXT_LENGTH)
    assert port_text_ids.dtype == ref_text_ids.dtype == torch.int64
    assert torch.equal(port_text_ids, ref_text_ids[0])


# --- references: patchify, BatchNorm, ids, pack -------------------------------------------------------------------------


@pytest.mark.parametrize("name,canvas,refs", [c for c in CASES if c[2]], ids=[c[0] for c in CASES if c[2]])
@pytest.mark.parametrize("stats", _bn_stats(), ids=[s[0] for s in _bn_stats()])
def test_reference_tokens_match_diffusers(name: str, canvas: tuple[int, int], refs: list, stats) -> None:
    label, mean, var, eps = stats
    g = torch.Generator().manual_seed(42)
    images = []
    latents_by_shape = {}
    for height, width in refs:
        image = torch.rand((1, 3, height, width), generator=g, dtype=torch.float32) * 2 - 1
        z = torch.randn((1, 32, height // 8, width // 8), generator=g, dtype=torch.float32) * 2
        images.append(image.to(torch.bfloat16))
        latents_by_shape[tuple(image.shape)] = z.to(torch.bfloat16)

    vae = _FakeVAE(latents_by_shape, mean, var, eps)
    host = _fake_pipeline_self(vae)
    ref_tokens, ref_ids = Flux2KleinPipeline.prepare_image_latents(
        host, images, batch_size=1, generator=None, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert ref_tokens.dtype == torch.bfloat16 and ref_ids.dtype == torch.int64

    port_tokens, port_ids = [], []
    for index, image in enumerate(images):
        z = latents_by_shape[tuple(image.shape)][0]
        hi, wi = packing.latent_shape(image.shape[2], image.shape[3])
        tokens = packing.patchify_latents(z)
        assert tokens.shape == (hi * wi, _IN_CHANNELS)
        port_tokens.append(packing.bn_normalize(tokens, mean, var, eps))
        port_ids.append(packing.reference_ids(hi, wi, index))
    port_tokens = torch.cat(port_tokens)
    port_ids = torch.cat(port_ids)
    assert port_tokens.dtype == torch.bfloat16
    assert torch.equal(port_tokens, ref_tokens[0]), (port_tokens - ref_tokens[0].float()).abs().max()
    assert torch.equal(port_ids, ref_ids[0])
    # t of reference i is 10 + 10 i (P:322 scale=10)
    for index in range(len(refs)):
        hi, wi = _grid(refs[index])
        start = sum(_grid(r)[0] * _grid(r)[1] for r in refs[:index])
        assert port_ids[start, 0].item() == 10 + 10 * index
        assert port_ids[start + hi * wi - 1].tolist() == [10 + 10 * index, hi - 1, wi - 1, 0]

    # The joint image stream the DiT sees: [latents | references] with ids concatenated in the same order (P:846-848).
    h, w = _grid(canvas)
    joint_ref = torch.cat([Flux2KleinPipeline._prepare_latent_ids(torch.zeros(1, _IN_CHANNELS, h, w)), ref_ids], dim=1)
    joint_port = torch.cat([packing.latent_ids(h, w), port_ids])
    assert torch.equal(joint_port, joint_ref[0])

    # patchify / unpatchify round trip against the reference's own helpers
    for image in images:
        z = latents_by_shape[tuple(image.shape)]
        hi, wi = packing.latent_shape(image.shape[2], image.shape[3])
        ref_patched = Flux2KleinPipeline._patchify_latents(z)
        assert torch.equal(packing.patchify_latents(z[0]), Flux2KleinPipeline._pack_latents(ref_patched)[0])
        assert torch.equal(packing.unpatchify_tokens(packing.patchify_latents(z[0]), hi, wi), z[0])
        assert torch.equal(Flux2KleinPipeline._unpatchify_latents(ref_patched), z)


# --- the Euler loop and the final unpack -------------------------------------------------------------------------------


@pytest.mark.parametrize("steps,guidance", [(4, None), (50, 4.0)], ids=["distilled_4", "base_50_cfg4"])
@pytest.mark.parametrize("name,canvas,refs", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("stats", _bn_stats(), ids=[s[0] for s in _bn_stats()])
def test_euler_loop_and_unpack_match_diffusers(
    name: str, canvas: tuple[int, int], refs: list, steps: int, guidance: float | None, stats
) -> None:
    label, mean, var, eps = stats
    h, w = _grid(canvas)
    image_seq_len = h * w
    cond_rows = sum(_grid(r)[0] * _grid(r)[1] for r in refs)
    seed = 3
    host = _fake_pipeline_self()
    generator = torch.Generator().manual_seed(seed)
    latents, latent_ids = Flux2KleinPipeline.prepare_latents(
        host, 1, _IN_CHANNELS // 4, canvas[0], canvas[1], torch.bfloat16, torch.device("cpu"), generator
    )

    reference = _reference_scheduler()
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    timesteps, _ = retrieve_timesteps(reference, steps, None, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    reference.set_begin_index(0)
    port = port_sched.Flux2KleinScheduler(steps, image_seq_len)

    vg = torch.Generator().manual_seed(seed + 1)
    x = packing.make_noise(seed, h, w)  # fp32 holding bf16 values, the device state slot
    assert torch.equal(x.to(torch.bfloat16), latents[0])
    for i, t in enumerate(timesteps):
        step = port.step(i)
        # P:841 / P:853 / transformer_flux2.py:1231: the value the DiT embeds
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        dit_in = timestep / 1000
        assert dit_in.dtype == torch.bfloat16
        embedded = dit_in.to(torch.bfloat16) * 1000
        assert step.embedding_t == embedded[0].item(), (i, step, embedded)
        assert step.timestep == t.item()

        # The DiT output over [latents | references], sliced to the latents (P:866); random bf16 velocities.
        pred = torch.randn((1, image_seq_len + cond_rows, _IN_CHANNELS), generator=vg, dtype=torch.float32).to(
            torch.bfloat16
        )
        noise_pred = pred[:, : latents.size(1) :]
        port_v = pred[0, :image_seq_len]
        if guidance is not None:
            neg = torch.randn_like(pred, dtype=torch.float32).to(torch.bfloat16)
            neg_noise_pred = neg[:, : latents.size(1) :]
            noise_pred = neg_noise_pred + guidance * (noise_pred - neg_noise_pred)
            port_v = port_sched.Flux2KleinScheduler.cfg_velocity(neg[0, :image_seq_len], port_v, guidance)
            assert port_v.dtype == torch.bfloat16
            assert torch.equal(port_v, noise_pred[0])

        latents = reference.step(noise_pred, t, latents, return_dict=False)[0]
        assert latents.dtype == torch.bfloat16
        x_new = port_sched.Flux2KleinScheduler.update(x, port_v, step)
        assert x_new.dtype == torch.bfloat16
        assert torch.equal(x_new, latents[0]), (i, (x_new.float() - latents[0].float()).abs().max())
        x = x_new.float()

    # P:898-910: unpack by ids, inverse BatchNorm, unpatchify
    ref_out = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids, h, w)
    assert torch.equal(ref_out, Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids))
    bn_mean = mean.view(1, -1, 1, 1).to(ref_out.dtype)
    bn_std = torch.sqrt(var.view(1, -1, 1, 1) + eps).to(ref_out.dtype)
    ref_out = ref_out * bn_std + bn_mean
    ref_out = Flux2KleinPipeline._unpatchify_latents(ref_out)
    assert ref_out.shape == (1, 32, 2 * h, 2 * w) and ref_out.dtype == torch.bfloat16

    port_out = packing.unpatchify_tokens(packing.bn_denormalize(x.to(torch.bfloat16), mean, var, eps), h, w)
    assert port_out.dtype == torch.bfloat16
    assert torch.equal(port_out, ref_out[0]), (port_out.float() - ref_out[0].float()).abs().max()

    # and the encode-side normalisation inverts it exactly where the reference's does
    ref_norm = (Flux2KleinPipeline._patchify_latents(ref_out) - bn_mean) / bn_std
    assert torch.equal(
        packing.bn_normalize(packing.patchify_latents(ref_out[0]), mean, var, eps),
        Flux2KleinPipeline._pack_latents(ref_norm)[0],
    )
