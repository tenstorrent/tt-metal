# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import re

from loguru import logger

import ttnn

# Ops a CLIP text-encoder LoRA may touch, in torch naming. Anything outside this set
# means the adapter would only be partially applied, so the whole load is rejected
# rather than silently producing weights that no reference describes.
SUPPORTED_TE_OPS = (".q_proj", ".k_proj", ".v_proj", ".out_proj", ".fc1", ".fc2")


def _torch_path_to_tt(path):
    """Map a torch CLIP module path onto the matching tt_dit module path.

    These are the same four renames tt_dit's own ``_prepare_torch_state`` methods apply
    when loading torch weights (see models/tt_dit/encoders/clip/model_clip.py); the LoRA
    deltas have to land on the same modules, so the mapping is shared rather than guessed.
    transformers>=5 flattened CLIPTextModel, so the ``text_model.`` prefix may be absent
    already and the first rule is then a no-op.
    """
    path = re.sub(r"^text_model\.", "", path)
    path = path.replace("self_attn.out_proj", "self_attn.o_proj")
    path = path.replace("mlp.fc1", "mlp.ff1").replace("mlp.fc2", "mlp.ff2")
    return path


def _lora_capable_modules(module, prefix=""):
    """Dotted path -> module, for every module in the tree that can hold a LoRA.

    Recurses ``named_children`` because tt_dit's ``named_parameters`` is deliberately
    non-recursive (it yields only the module's own parameters).
    """
    found = {}
    if hasattr(module, "register_lora"):
        found[prefix] = module
    for name, child in module.named_children():
        found.update(_lora_capable_modules(child, f"{prefix}.{name}" if prefix else name))
    return found


class TtTextEncoderLoRAWeightsManager:
    """LoRA state for the device text encoders.

    A sibling of ``TtLoRAWeightsManager`` rather than a part of it: each owns the weights
    it fuses into, and neither drives the other's. Both now fuse the same way, in place on
    device, so a swap costs a few matmuls and an add rather than a host merge and a full
    encoder reload.

    The delta arithmetic itself is tt_dit's (``models/tt_dit/layers/lora.py``): register an
    adapter to bank it, ``bind_active`` to add it into the weight at its existing address,
    ``unbind_active`` to subtract the same cached factors back out. This class only decides
    which module each adapter tensor belongs to.
    """

    def __init__(self, torch_pipeline):
        self._torch_pipeline = torch_pipeline

        # component -> {tt module path: LoRA bank index}, populated on load.
        self._registered = {}
        # component -> {tt module path: the adapter's own scaling (alpha/rank)}. Held
        # separately because bind_active's scale argument *replaces* the registered scale
        # rather than multiplying it, so the two have to be combined by the caller.
        self._scalings = {}
        # component -> {tt module path: host copy of the pristine weight}. bind/unbind is
        # arithmetic on a bf16 weight, so (W + d) - d does not round-trip; restoring from
        # these makes rollback bit-exact, the same way the UNet manager keeps a host copy
        # of every weight it allocated.
        self._base_weights_host = {}
        # component -> tt_dit CLIPEncoder, supplied by the caller once the encoders exist.
        self._encoders = {}
        self._fused = False
        self._components = []

    def register_encoders(self, encoders):
        """Hand over the device encoders, keyed by ``text_encoder`` / ``text_encoder_2``.

        Registering nothing disables the text-encoder LoRA path, which is how the caller
        signals that the encoders run on host.
        """
        self._encoders = {name: enc for name, enc in encoders.items() if enc is not None}

    def _active(self):
        return bool(self._encoders)

    @property
    def is_fused(self):
        return self._fused

    def state(self):
        """``fused`` is whether the device encoders currently hold merged weights;
        ``components`` lists the text encoders the loaded adapter trains."""
        return {
            "fused": self._fused,
            "components": list(self._components),
        }

    def affects_unsupported_ops(self):
        """Whether the attached adapter touches a CLIP op we cannot fuse."""
        for name in self._components:
            for path, _, _, _ in self._iter_torch_lora(name):
                if not any(path.endswith(op) for op in SUPPORTED_TE_OPS):
                    return True
        return False

    def _iter_torch_lora(self, component):
        """Yield ``(module_path, lora_a, lora_b, scaling)`` for the attached adapter.

        Mirrors ``TtLoRAWeightsManager._get_lora_params``: PEFT wraps each targeted module
        and hangs ``lora_A``/``lora_B`` off it, so the live torch tree is the source of
        truth rather than the adapter file's own key spelling.
        """
        text_encoder = getattr(self._torch_pipeline, component, None)
        if text_encoder is None:
            return
        for name, module in text_encoder.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            adapters = list(module.lora_A.keys())
            if not adapters:
                continue
            adapter = adapters[0]  # TODO: handle multiple adapters
            clean = re.sub(r"^base_model\.model\.", "", name)
            scaling = module.scaling.get(adapter, 1.0) if hasattr(module, "scaling") else 1.0
            yield clean, module.lora_A[adapter].weight.data, module.lora_B[adapter].weight.data, scaling

    def refresh_components(self):
        """Record which text encoders the attached adapter trains.

        Read from the torch pipeline rather than passed in, so a rejected adapter (the UNet
        manager unloads it again) correctly leaves this empty.
        """
        adapters = self._torch_pipeline.get_list_adapters()
        self._components = [c for c in ("text_encoder", "text_encoder_2") if adapters.get(c)]

    def register_adapter(self):
        """Bank the attached adapter's deltas on the matching device modules.

        Banking is separate from fusing: this uploads the A/B factors once, and a later
        bind or unbind is then just arithmetic on weights already in place.
        """
        self._registered = {}
        self._scalings = {}
        self._base_weights_host = {}
        if not (self._components and self._active()):
            return
        for name in self._components:
            encoder = self._encoders.get(name)
            if encoder is None:
                continue
            targets = _lora_capable_modules(encoder)
            banked, scalings, snapshots, missing = {}, {}, {}, []
            for path, lora_a, lora_b, scaling in self._iter_torch_lora(name):
                tt_path = _torch_path_to_tt(path)
                module = targets.get(tt_path)
                if module is None:
                    missing.append(tt_path)
                    continue
                # tt_dit validates A as [rank, in] and B as [out, rank], which is exactly
                # how PEFT stores them, so they go across untransposed.
                banked[tt_path] = module.register_lora(lora_a, lora_b, scale=scaling)
                scalings[tt_path] = scaling
                # Straight off the device, so the copy needs no knowledge of layout or
                # sharding and restores into the same allocation later.
                snapshots[tt_path] = ttnn.from_device(module.weight.data)
            if missing:
                logger.debug(f"{name}: {len(missing)} LoRA targets had no device module, e.g. {missing[:3]}")
            self._registered[name] = banked
            self._scalings[name] = scalings
            self._base_weights_host[name] = snapshots
            logger.info(f"{name}: banked {len(banked)} LoRA targets on device")

    def fuse(self, lora_scale):
        # Idempotency guard mirroring the UNet path. tt_dit's bind_active is itself
        # idempotent per module, but returning early keeps the reported state honest.
        if self._fused:
            logger.info("Text-encoder LoRA already fused; skipping re-fuse (idempotent).")
            return
        if not self._components:
            return
        # A scale of 0.0 means "leave CLIP alone": skip the bind entirely rather than
        # adding a zero delta, so the weights stay bit-for-bit identical and state()
        # keeps reporting fused: false.
        if lora_scale == 0.0:
            logger.info("CLIP LoRA scale is 0.0, skipping text-encoder fusion.")
            return
        if not self._active():
            logger.warning("Text-encoder LoRA present but encoders run on host; TE LoRA not applied.")
            return
        if not any(self._registered.values()):
            logger.warning("No text-encoder LoRA targets were banked; nothing to fuse.")
            return
        logger.info(f"Fusing text-encoder LoRA into {self._components} on device (scale={lora_scale})...")
        for name, banked in self._registered.items():
            targets = _lora_capable_modules(self._encoders[name])
            scalings = self._scalings.get(name, {})
            for tt_path, idx in banked.items():
                # bind_active's scale replaces the registered one rather than multiplying,
                # so the adapter's own scaling has to be folded in here. Dropping it would
                # silently mis-scale every delta by alpha/rank.
                targets[tt_path].bind_active(idx, scale=scalings.get(tt_path, 1.0) * lora_scale)
        self._fused = True

    def unload(self):
        components, self._components = self._components, []
        if not self._fused:
            self._registered = {}
            self._scalings = {}
            self._base_weights_host = {}
            return
        logger.info("Restoring base text-encoder weights on device...")
        for name, banked in self._registered.items():
            targets = _lora_capable_modules(self._encoders[name])
            snapshots = self._base_weights_host.get(name, {})
            for tt_path in banked:
                module = targets[tt_path]
                # unbind_active clears the mixin's own bookkeeping so a later bind starts
                # clean. Its subtraction leaves about a bf16 ulp behind, which a pcc=1.0
                # image comparison would catch, so overwrite with the pristine copy.
                module.unbind_active()
                host = snapshots.get(tt_path)
                if host is not None:
                    ttnn.copy_host_to_device_tensor(host, module.weight.data)
        self._registered = {}
        self._scalings = {}
        self._base_weights_host = {}
        self._fused = False
        logger.debug(f"Rolled back text-encoder LoRA for {components}")
