# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A fuse-mode LoRA must survive the weight page-out/page-in cycle a small mesh runs on every request.

MiniMax-H3's `(1, 1)` and `(1, 4)` presets are `coresident: False`: the DiT, the text encoder and the
video VAE are registered as `Module` coresident exclusions, so loading any one of them deallocates
the others. `_prepare_transformer` therefore re-loads the DiT -- from the pristine cached base
weights -- before every denoise, while the adapter is bound exactly once.

Two defects made that silently serve the BASE model under an adapter's name on a mesh where the DiT
is unquantized (so the adapter is merged on device rather than into the checkpoint):

  * `experimental/lora/promote.py` builds the promoted class base-first, which puts `Module` ahead of
    `LoRAMixin` in the MRO and shadows `LoRAMixin.deallocate_weights`. `_delta_applied` then stays
    true across a page-out, and `reapply_after_load` takes its "already applied" early return.
  * Nothing re-applied after a load in the first place; only the LTX and Wan-runtime pipelines called
    `reapply_after_load` by hand.

Both are fixed in the shared layer (hooks on `LoRAMixin`, wired explicitly by `promote`), so the
tests here are written against the lifecycle rather than against either pipeline.
"""

import json
import os
import shutil
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ....experimental.lora.h3_adapter_loader import fuse_h3_adapter_into_state_dict, load_h3_adapter_into
from ....experimental.lora.promote import promote_to_lora
from ....layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ....layers.lora import LoRAMixin
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....utils import tensor as tensor_utils
from .common import SMALL_LINE_PARALLEL, skip_if_unsupported_num_links
from .test_transformer_minimax_h3 import (
    _CALLER_OWNED_CONFIG_KEYS,
    TURBO_FILE_ENV,
    _checkpoint_dir,
    _load_reference_state_dict,
    _modality_metadata,
    _prepare_tt_inputs,
)

RANK = 128
SCALE = 0.0625


def _read_weight(param) -> torch.Tensor:
    return tensor_utils.to_torch(param.data, mesh_axes=param.mesh_axes).float()


def _host_delta(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """The delta `register_lora`'s A ([rank,in]) and B ([out,rank]) describe, in the device weight's
    [in, out] layout, rounded the way the device path rounds it."""
    return scale * (A.transpose(0, 1).to(torch.bfloat16).float() @ B.transpose(0, 1).to(torch.bfloat16).float())


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return (got - want).norm().item() / max(want.norm().item(), 1e-12)


# ---------------------------------------------------------------- what promotion still shadows


def test_promoted_class_shadows_forward_and_refuses_runtime_mode() -> None:
    """Lock in the part of the base-first MRO that is NOT fixed, only guarded.

    ``promote`` wires the weight lifecycle explicitly, but ``LoRAMixin.forward`` and
    ``forward_fused_addcmul`` are still shadowed by the base on a promoted class. In fuse mode that
    is harmless — the mixin's versions only do anything in runtime mode — so the containment is
    ``promote_to_lora`` refusing runtime mode. If someone makes promotion mixin-first, or
    reintroduces runtime promotion, this test is the thing that notices.
    """
    from ....experimental.lora.promote import _promoted_class

    for base in (Linear, ColParallelLinear, RowParallelLinear):
        promoted = _promoted_class(base)
        assert promoted.forward is base.forward, f"{promoted.__name__}.forward is no longer the base's"
        assert promoted.__mro__.index(base) < promoted.__mro__.index(LoRAMixin)
        # ... while the lifecycle IS wired, by the explicit wrappers rather than by the MRO.
        for name in ("deallocate_weights", "load", "_mark_loaded"):
            assert name in promoted.__dict__, f"{promoted.__name__} lost its {name} wrapper"

    with pytest.raises(ValueError, match="runtime"):  # allow-pytest.raises: the guard IS the contract
        promote_to_lora(object(), mode="runtime")


# ---------------------------------------------------------------- the lifecycle, on bare Linears


@pytest.mark.timeout(900)
@SMALL_LINE_PARALLEL
def test_lora_bind_survives_weight_reload(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    tmp_path: Path,
    reset_seeds,
) -> None:
    """Page a LoRA-bound Linear out and back in, both reload routes, and the delta must still be there.

    Shapes are MiniMax-H3's own (`to_qkv`-width column parallel and an FFN-width row parallel) and
    the mesh axes are the `(1, 4)` preset's, so the sharding this exercises is the served one.
    """
    skip_if_unsupported_num_links(mesh_device, num_links)
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    cases = [
        ("col_parallel", ColParallelLinear(5376, 7168, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis)),
        ("row_parallel", RowParallelLinear(14336, 5376, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis)),
    ]
    for name, layer in cases:
        in_f, out_f = layer.in_features, layer.out_features
        assert in_f % tp_factor == 0 and out_f % tp_factor == 0

        base_w = torch.randn(out_f, in_f) * 0.02
        layer.load_torch_state_dict({"weight": base_w})
        w_base = _read_weight(layer.weight)

        # A cache directory written BEFORE the adapter exists -- exactly what `cache.load_model`
        # reads back on a later request. It is read through a COPY so the load is not served out of
        # the pages this process just wrote: that is what a later request's load really is, and
        # reading the just-written inode back is not reliably cheap (on this bring-up's host it
        # livelocks the loader outright).
        written = tmp_path / f"{name}.written"
        written.mkdir()
        layer.save(written)
        cache_dir = tmp_path / f"{name}.cache"
        shutil.copytree(written, cache_dir)

        promote_to_lora(layer)
        A = (torch.randn(RANK, in_f) * 0.05).to(torch.bfloat16).float()
        B = (torch.randn(out_f, RANK) * 0.05).to(torch.bfloat16).float()
        want = w_base + _host_delta(A, B, SCALE)
        layer.bind_active(layer.register_lora(A, B, scale=SCALE, name="probe"))

        bound_err = _rel_err(_read_weight(layer.weight), want)
        logger.info(f"{name}: bind rel-err vs host delta {bound_err:.2e}")
        assert bound_err < 0.02, f"{name}: bind_active did not land the delta (rel-err {bound_err:.3g})"

        for route, reload_fn in (
            ("tensorbin cache", lambda: layer.load(cache_dir)),
            ("torch state dict", lambda: layer.load_torch_state_dict({"weight": base_w})),
        ):
            layer.deallocate_weights()
            applied_after_unload = layer._delta_applied
            reload_fn()
            err = _rel_err(_read_weight(layer.weight), want)
            logger.info(f"{name}: after a {route} reload, rel-err vs host delta {err:.2e}")
            assert err < 0.02, (
                f"{name}: the adapter did not survive a {route} reload (rel-err {err:.3g}); the layer "
                "is serving BASE weights under the adapter's name"
            )
            # The bookkeeping behind it, named separately so a regression points straight at the cause.
            assert not applied_after_unload, (
                f"{name}/{route}: the merged delta died with the weight, but `_delta_applied` stayed "
                "true -- LoRAMixin.deallocate_weights is being shadowed on this class"
            )

        layer.deallocate_lora()
        layer.deallocate_weights()


# ---------------------------------------------------------------- the real DiT, at a served shape


@pytest.mark.timeout(7200)
@SMALL_LINE_PARALLEL
@pytest.mark.parametrize(
    ("num_text", "num_audio", "num_video", "grid"),
    [pytest.param(512, 414, 37296, (24, 42), id="prod_768p_5s")],
)
def test_minimax_h3_lora_survives_dit_reload(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    num_text: int,
    num_audio: int,
    num_video: int,
    grid: tuple[int, int],
    reset_seeds,
) -> None:
    """Full-depth, real checkpoint, real Turbo adapter, unquantized weights, at the 1344x768 x 124
    packed length the p300x2 profile serves.

    Three velocities off one model instance:
      base    no adapter
      bound   adapter bound on device (what the first request after a bind sees)
      reload  the page-out + page-in a `coresident: False` preset performs before every denoise

    `reload` must differ from `base` -- that is the regression -- and must agree with a host fuse of
    the same adapter, which is the path the quantized p150 profile already ships.
    """
    if tuple(mesh_device.shape) != (1, 4):
        pytest.skip("the unquantized device-merge path this covers is the (1, 4) profile's")
    skip_if_unsupported_num_links(mesh_device, num_links)

    turbo_file = os.environ.get(TURBO_FILE_ENV)
    if not (turbo_file and os.path.exists(turbo_file)):
        pytest.skip(f"set {TURBO_FILE_ENV} to a lightx2v MiniMax-H3 Turbo safetensors file")
    directory = _checkpoint_dir()

    config = {k: v for k, v in json.loads((directory / "config.json").read_text()).items() if not k.startswith("_")}
    model_kwargs = {k: v for k, v in config.items() if k not in _CALLER_OWNED_CONFIG_KEYS}
    model_kwargs["patch_size"] = tuple(model_kwargs["patch_size"])

    per_modality = _modality_metadata(num_text, num_audio, num_video, grid, ())
    inputs = _prepare_tt_inputs(
        mesh_device,
        sp_axis,
        tp_axis,
        num_links,
        topology,
        per_modality,
        text_dim=model_kwargs["text_dim"],
        video_patch_dim=model_kwargs["in_channels"] * int(torch.tensor(model_kwargs["patch_size"]).prod()),
        audio_channels=model_kwargs["audio_in_channels"],
        head_dim=model_kwargs["attention_head_dim"],
        rope_freq_dim=config["rope_freq_dim"],
        rope_theta=config["rope_theta"],
    )

    tt_model = MiniMaxH3Transformer3DModel(
        **model_kwargs,
        mesh_device=mesh_device,
        ccl_manager=inputs.ccl_manager,
        parallel_config=inputs.parallel_config,
        is_fsdp=is_fsdp,
    )

    def load(state: dict[str, torch.Tensor]) -> None:
        start = time.time()
        tt_model.load_torch_state_dict(state)
        logger.info(f"loaded {model_kwargs['num_layers']} layers onto the mesh in {time.time() - start:.1f}s")

    def velocity() -> tuple[torch.Tensor, torch.Tensor]:
        # The refiner runs on model weights, so it is re-run after every reload, as the pipeline does.
        tt_model.prepare_static_sources(**inputs.tt_static)
        video, audio = tt_model(**inputs.tt)
        ttnn.synchronize_device(mesh_device)

        def compose(t: ttnn.Tensor, rows: int) -> torch.Tensor:
            out = ttnn.to_torch(
                t,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=tuple(mesh_device.shape)),
            )
            return out.reshape(-1, *out.shape[2:])[0].float()[:rows]

        return compose(video, num_video), compose(audio, num_audio)

    load(_load_reference_state_dict(directory))
    v_base, a_base = velocity()

    handle = load_h3_adapter_into(tt_model, turbo_file, name=os.path.basename(turbo_file))
    assert len(handle) > 0, "the adapter bound zero targets"
    logger.info(f"bound {len(handle)} adapter targets on device")
    v_bound, a_bound = velocity()
    assert not torch.equal(v_bound, v_base), "bind_active changed nothing even before a reload"

    # The page-out + page-in `_prepare_transformer` performs before every denoise.
    tt_model.deallocate_weights()
    load(_load_reference_state_dict(directory))
    v_reload, a_reload = velocity()

    assert not torch.equal(v_reload, v_base), (
        "the DiT velocity after a weight reload is BIT-IDENTICAL to the adapter-free one: the bound "
        "adapter did not survive the page-in, so this profile renders base-model output"
    )
    assert not torch.equal(
        a_reload, a_base
    ), "the audio velocity after a reload is bit-identical to the adapter-free one"
    reload_vs_bound = comp_pcc(v_bound, v_reload, 0.999)
    logger.info(f"reload vs bound: {reload_vs_bound[1]}")
    assert reload_vs_bound[0], f"the re-applied delta does not match the originally bound one: {reload_vs_bound[1]}"

    # Host-fuse reference: drop the device adapter first, so the reload does not re-merge on top.
    tt_model.deallocate_weights()
    _drop_lora(tt_model)
    fused_state = _load_reference_state_dict(directory)
    scales = fuse_h3_adapter_into_state_dict(fused_state, turbo_file, name=os.path.basename(turbo_file))
    logger.info(f"host-fused {len(scales)} adapter targets into the checkpoint")
    load(fused_state)
    v_host, a_host = velocity()

    video_pcc = comp_pcc(v_host, v_reload, 0.99)
    audio_pcc = comp_pcc(a_host, a_reload, 0.99)
    logger.info(f"device-bound vs host-fused — video {video_pcc[1]}, audio {audio_pcc[1]}")
    assert video_pcc[0], f"video velocity disagrees with the host fuse: {video_pcc[1]}"
    assert audio_pcc[0], f"audio velocity disagrees with the host fuse: {audio_pcc[1]}"


def _drop_lora(root) -> int:
    dropped = 0
    stack = [root]
    while stack:
        module = stack.pop()
        stack.extend(child for _, child in module.named_children())
        if isinstance(module, LoRAMixin):
            module.deallocate_lora()
            dropped += 1
    return dropped
