# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from loguru import logger


class TtTextEncoderLoRAWeightsManager:
    """LoRA state for the device text encoders.

    A sibling of ``TtLoRAWeightsManager`` rather than a part of it. The UNet manager
    fuses deltas onto weights it allocated itself, so it can write them in place; the
    encoders are owned by tt_dit ``CLIPEncoder`` instances and are updated by handing
    merged torch weights back through a reload hook. Those are different mechanisms on
    different weights, so neither manager drives the other's.

    Ordering across the two is the caller's job (see ``TtSDXLPipeline``): the base
    snapshot has to be taken before any adapter attaches, and the UNet fuse has to run
    before the text-encoder fuse.
    """

    def __init__(self, torch_pipeline):
        self._torch_pipeline = torch_pipeline

        # `_base_state` is a lazily-captured clean snapshot of the torch text-encoder
        # weights (taken before any adapter is applied) used to revert; `_fused` tracks
        # whether the on-device encoders currently hold merged LoRA weights. `_reload`
        # is the caller-supplied hook that pushes torch weights onto the device
        # encoders — see register_reload().
        self._base_state = None
        self._fused = False
        self._reload = None
        self._device_encoders = ()
        self._components = []

    def register_reload(self, reload_fn, components):
        """Register the hook that pushes torch text-encoder weights onto the device.

        ``reload_fn()`` takes no arguments and reloads the device encoders from the
        current torch state dicts. ``components`` is the subset of
        ``("text_encoder", "text_encoder_2")`` that ``reload_fn`` actually reloads.

        Not registering — or registering no components — disables the text-encoder LoRA
        path entirely, which is how the caller signals that the encoders run on host.
        """
        self._reload = reload_fn
        self._device_encoders = tuple(components)

    def _reloadable(self):
        return self._reload is not None and bool(self._device_encoders)

    @property
    def is_fused(self):
        return self._fused

    def state(self):
        """``fused`` is whether the device encoders hold merged weights; ``components``
        lists the text encoders the loaded adapter trains."""
        return {
            "fused": self._fused,
            "components": list(self._components),
        }

    def ensure_base_snapshot(self):
        """Capture a clean copy of the torch text-encoder weights, once.

        Must run before anything can attach an adapter to the torch text encoders, so
        the snapshot is of clean weights.
        """
        if self._base_state is not None or not self._reloadable():
            return
        state = {}
        for name in self._device_encoders:
            text_encoder = getattr(self._torch_pipeline, name, None)
            if text_encoder is not None:
                state[name] = {k: v.detach().cpu().clone() for k, v in text_encoder.state_dict().items()}
        self._base_state = state

    def refresh_components(self):
        """Record which text encoders the currently-attached adapter trains.

        Read from the torch pipeline rather than passed in, so a rejected adapter (the
        UNet manager unloads it on DoRA or unsupported ops) correctly leaves this empty.
        """
        adapters = self._torch_pipeline.get_list_adapters()
        self._components = [c for c in ("text_encoder", "text_encoder_2") if adapters.get(c)]

    def fuse(self, lora_scale):
        # Idempotency guard, mirroring the UNet path (_fuse_unet_lora early-returns on
        # self._is_fused). Without this, a second fuse before an unload would merge the
        # TE delta on top of already-merged torch weights, double-applying the adapter.
        if self._fused:
            logger.info("Text-encoder LoRA already fused; skipping re-fuse (idempotent).")
            return
        if not self._components:
            return
        # scale=0.0 means "do not apply to CLIP" — skip the host fuse + device reload
        # entirely rather than fusing a zero delta (saves a full TE reload). _fused
        # stays False, so state() reports fused: false.
        if lora_scale == 0.0:
            logger.info("CLIP LoRA scale is 0.0 — skipping text-encoder fusion.")
            return
        if not self._reloadable():
            logger.warning("Text-encoder LoRA present but encoders run on host; TE LoRA not applied.")
            return
        logger.info(f"Fusing text-encoder LoRA into {self._components} and reloading on device...")
        # Merge the TE LoRA into the torch encoders, then strip all adapters. The merged
        # weights stay in place with clean state-dict keys, which the reload hook pushes
        # onto the device encoders. UNet deltas are already applied on device by the
        # time this runs, so dropping the torch UNet adapter here is harmless.
        self._torch_pipeline.fuse_lora(components=self._components, lora_scale=lora_scale)
        self._torch_pipeline.unload_lora_weights()
        self._reload()
        self._fused = True

    def unload(self):
        self._components = []
        if not self._fused:
            return
        logger.info("Restoring base text-encoder weights on device...")
        # ensure_base_snapshot only stores components the reload hook covers, so the
        # presence of a key is itself the "this encoder exists on device" check.
        base = self._base_state or {}
        for name in ("text_encoder", "text_encoder_2"):
            if base.get(name) is not None:
                getattr(self._torch_pipeline, name).load_state_dict(base[name])
        self._reload()
        self._fused = False
        # Base snapshot is deliberately retained: it is a one-shot capture of clean
        # weights, valid for every later load/fuse/unload cycle.
